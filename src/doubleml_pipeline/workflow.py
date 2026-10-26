# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module acting as the Orchestrator for the Causal Inference workflow."""

import contextlib
import copy
import difflib
import glob
import itertools
import json
import os
import re
import shutil
import threading
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from . import optimization
from .agent import generator as agent_generator
# Phase 0 lives in `evaluation.alpha_selection`; it is aliased so that the
# `alpha_selection` constructor argument (the config key) cannot shadow it.
from .evaluation import alpha_selection as phase0
from .evaluation import hparam_search
from .evaluation import metrics
from .evaluation import runner
from .evaluation import saturation_evaluation
from .evaluation import selection
from .evaluation import sensitivity
from .models import saturation as saturation_lib
from .preprocessing import scalers
from .utils import io
from .visualization import html_extraction
from .visualization import plots

# The `hparam_policy` values.
HPARAM_POLICIES = runner.HPARAM_POLICIES
DEFAULT_HPARAM_POLICY = runner.HPARAM_POLICY_GLOBAL


def resolve_n_cores(n_jobs: Optional[int]) -> int:
  """``n_cores`` of the parallel plan ``(W, h)``.

  ``os.cpu_count()`` when ``n_jobs`` is ``-1`` (or None); a positive ``n_jobs``
  is used in place of ``n_cores``; joblib's ``n_jobs < -1`` ("all but
  ``-n_jobs - 1``") is honoured as well.

  Args:
    n_jobs: The orchestrator's ``n_jobs``.

  Returns:
    The number of cores available to the parallel sections (>= 1).
  """
  cpu = os.cpu_count() or 1
  if n_jobs is None or n_jobs == -1 or n_jobs == 0:
    return int(cpu)
  if n_jobs > 0:
    return int(n_jobs)
  return max(1, int(cpu) + 1 + int(n_jobs))


def compute_workers_threads(n_tasks: int, n_cores: int) -> tuple[int, int]:
  """``W = min(N_task, n_cores)``, ``h = max(1, n_cores // W)``.

  Args:
    n_tasks: ``N_task`` of the parallel section (treated as 1 when 0).
    n_cores: ``n_cores`` (see ``resolve_n_cores``).

  Returns:
    ``(W, h)`` -- workers (loky processes) and OpenMP / learner threads per
    worker.
  """
  n_cores = max(1, int(n_cores))
  n_tasks = max(1, int(n_tasks))
  workers = min(n_tasks, n_cores)
  threads = max(1, n_cores // workers)
  return workers, threads


# The joblib backend of the parallel sections.  `loky` (joblib's default
# process backend) is the "worker" of the parallel plan `(W, h)` and the only
# backend joblib combines with `inner_max_num_threads`.  The
# environment variable is an escape hatch for environments in which loky cannot
# spawn its workers (e.g. hermetic Python, where `sys.executable` is None):
# `DOUBLEML_JOBLIB_BACKEND=multiprocessing` selects the fork-based pool; the
# thread cap `h` is then applied through the same environment variables loky
# would set in its workers (`OMP_NUM_THREADS`, ...), which forked children
# inherit, in addition to the explicit `n_jobs=h` of FLAML / the fixed fits.
PARALLEL_BACKEND_ENV = "DOUBLEML_JOBLIB_BACKEND"
DEFAULT_PARALLEL_BACKEND = "loky"
_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def _env_thread_limited_config(backend: str, workers: int, threads: int):
  """``parallel_config(backend, n_jobs=W)``; ``h`` set via the environment."""
  previous = {var: os.environ.get(var) for var in _THREAD_LIMIT_ENV_VARS}
  for var in _THREAD_LIMIT_ENV_VARS:
    os.environ[var] = str(int(threads))
  try:
    with joblib.parallel_config(backend=backend, n_jobs=int(workers)):
      yield
  finally:
    for var, value in previous.items():
      if value is None:
        os.environ.pop(var, None)
      else:
        os.environ[var] = value


def parallel_section(workers: int, threads: int):
  """``joblib.parallel_config(n_jobs=W, inner_max_num_threads=h)``.

  joblib requires an explicit backend together with ``inner_max_num_threads``;
  ``loky`` is its default process backend (the "workers" ``W``).  With
  ``W = 1`` joblib runs the tasks sequentially in the parent process.  See
  ``PARALLEL_BACKEND_ENV`` for the (non-default) fork-based alternative.

  Args:
    workers: ``W``.
    threads: ``h``.

  Returns:
    The context manager.
  """
  backend = os.environ.get(PARALLEL_BACKEND_ENV, DEFAULT_PARALLEL_BACKEND)
  if backend == DEFAULT_PARALLEL_BACKEND:
    return joblib.parallel_config(
        backend=backend, n_jobs=int(workers), inner_max_num_threads=int(threads)
    )
  return _env_thread_limited_config(backend, workers, threads)


# The keys of a run result that every repetition must carry (the
# aggregation of workflow Phase 3 reads nothing else from repetitions >= 1).
REP_RESULT_KEYS = ("model_name", "model_group", "metrics_1c", "cate_coef_dict")


def rep_df_columns(d_col: str, gt_col: Optional[str]) -> List[str]:
  """The ``df_1a`` columns kept for repetitions >= 1."""
  cols = [
      f"estimated_incremental_KPI_{d_col}",
      f"estimated_incremental_KPI_{d_col}_2.5%",
      f"estimated_incremental_KPI_{d_col}_97.5%",
      d_col,
  ]
  if gt_col:
    cols.append(gt_col)
  return cols


def reduce_rep_result(
    res: Dict[str, Any], d_col: str, gt_col: Optional[str]
) -> Dict[str, Any]:
  """Keeps only what Phase 3 consumes from a repetition >= 1.

  ``model_obj`` and the other ``df_1a`` columns are dropped before the result
  is sent back to the parent process; the KPI point estimate, its lower /
  upper bounds, the treatment column and the ground-truth column (when
  present) are kept with the original index, so the median aggregation
  (positional) and the raw-treatment restoration are unchanged.

  Args:
    res: The full ``runner.DoubleMLPipeline.run`` result (+ ``model_group``).
    d_col: Treatment column.
    gt_col: Ground-truth effect column, or None / empty.

  Returns:
    The reduced result dictionary.
  """
  frame = res["df_1a"]
  keep = [c for c in rep_df_columns(d_col, gt_col) if c in frame.columns]
  reduced = {key: res[key] for key in REP_RESULT_KEYS if key in res}
  reduced["df_1a"] = frame[keep].copy()
  return reduced


def regroup_phase3_results(
    candidate_names: List[str],
    rep0_results: Dict[str, Dict[str, Any]],
    jobs: List[tuple[str, int]],
    flat_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
  """Rebuilds the per-candidate result list of Phase 3.

  The flattened ``(candidate, rep)`` job list is regrouped by candidate name,
  in shortlist order, with repetition 0 (= the Phase 1 result) first and the
  repetitions ``1 .. R-1`` in ascending order -- the very shape the
  sequential loop produced before.

  Args:
    candidate_names: Shortlisted ``model`` names in shortlist order.
    rep0_results: ``{model_name: Phase 1 result}`` (``model_group`` set).
    jobs: ``(model_name, rep)`` of every flattened job, in submission order.
    flat_results: The worker results, aligned with ``jobs``.

  Returns:
    The regrouped list.

  Raises:
    ValueError: If a candidate has no Phase 1 result or the job list and the
      results are misaligned.
  """
  if len(jobs) != len(flat_results):
    raise ValueError(
        f"{len(jobs)} Phase 3 jobs but {len(flat_results)} results."
    )
  by_candidate: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
  for (name, rep), res in zip(jobs, flat_results):
    by_candidate.setdefault(name, []).append((rep, res))
  regrouped: List[Dict[str, Any]] = []
  for name in candidate_names:
    if name not in rep0_results:
      raise ValueError(
          f"Shortlisted candidate '{name}' has no Phase 1 result to reuse as"
          " repetition 0."
      )
    regrouped.append(rep0_results[name])
    regrouped.extend(
        res for _, res in sorted(by_candidate.get(name, []), key=lambda t: t[0])
    )
  return regrouped


def _model_group_label(m_data: Dict[str, Any]) -> str:
  """Builds the ``<ml_y>_<ml_t>_<n_folds>`` display label for a run.

  This used to be `"_".join(m_data["model"].split("_")[:3])`, which assumed
  each of the first three fields of ``model`` was underscore-free. FLAML
  learner names like ``extra_tree`` and ``xgb_limitdepth`` break that
  assumption: the learner names alone consume all three slots and ``n_folds``
  silently falls out of the label, so runs differing only in fold count
  collapse into one group and get averaged together.

  Args:
    m_data: A ``metrics_1c`` dict as produced by `evaluation.runner`.

  Returns:
    The group label. For underscore-free learner names this is byte-identical
    to the previous implementation.
  """
  # Must be "n_folds" (the configured value, which is what `model_name` itself
  # embeds) and NOT "number_of_folds", which is DoubleML's effective count
  # `n_folds ** n_cluster_vars` -- 9 rather than 3 under 2-way clustering.
  return "_".join([
      str(m_data.get("ml_y_name", "")),
      str(m_data.get("ml_t_name", "")),
      str(m_data.get("n_folds", "")),
  ])


# ---------------------------------------------------------------------------
# The `saturation_config` schema.
#
# `saturation_config` is the SINGLE source of truth for every
# saturation-related setting.
# Only `saturation.family` is required; everything else falls back to the
# defaults recorded below, so switching saturation on is a one-key change.
#
# The key sets are enumerated so that a typo fails loudly.  Without this a
# misspelled block (e.g. "saturaton") would simply be ignored, `family` would
# stay None, and Phase 0 would be silently skipped -- the user would get a
# normal linear run and no indication that their saturation_config did
# nothing.
# ---------------------------------------------------------------------------
_CONFIG_BLOCKS = frozenset(
    {"saturation", "alpha_selection", "final_fit", "optimization"}
)
_CONFIG_KEYS = {
    "saturation": frozenset({
        "family",
        "grid",
        "basis_dim_T",
        "ridge",
        "basis_tol",
        "basis_dim_range",
        # Quantile of T > 0 defining T_ref (default 0.95).
        "t_ref_quantile",
    }),
    "alpha_selection": frozenset({
        "sel_fraction",
        "n_seeds",
        "k_folds",
        "refine",
        "time_budget",
        "linear_test_k",
        # Use the Student-t critical value t_{G-1}^{-1}(Phi(k)) instead of
        # k itself in the 1-SE decision (default False).
        "linear_test_t_critical",
        # Fixed half-saturation point (units of T) used
        # downstream when Diag1 / Diag2 fires.
        "fallback_half_saturation_T",
        # Diag2 ratio threshold and Diag6 tolerance.
        "flatness_ratio",
        "t_ref_tolerance",
        "output_dir",
        "learners",
    }),
    "final_fit": frozenset({"data"}),
    # budget_mode / budget_reallocate join budget_increment; cap_quantile
    # (support cap quantile, default 1.0) and reallocate_within
    # ("all" | "date") are the allocation-scope settings.
    "optimization": frozenset({
        "budget_increment",
        "budget_mode",
        "budget_reallocate",
        "cap_quantile",
        "reallocate_within",
    }),
}
# `mode`, `c_min`, `c_max` and `max_extensions` configure the automatic grid;
# `min` / `max` / `scale` are the manual (legacy) override.
_GRID_KEYS = frozenset(
    {"min", "max", "num", "scale", "mode", "c_min", "c_max", "max_extensions"}
)
# Keys of `alpha_selection.learners`.
_LEARNER_KEYS = frozenset({
    "learner_selection",
    "y_model",
    "t_model",
    "small_sample_threshold",
    "small_sample_model",
})
# Default ROI floor r = 1.0 across every optimizer path.
DEFAULT_THRESHOLD_ROI = 1.0


def _validate_config(saturation_config: Dict[str, Any]) -> None:
  """Rejects unknown keys in the `saturation_config` dict.

  Args:
    saturation_config: The user-supplied `saturation_config` mapping.

  Raises:
    TypeError: If `saturation_config` or any of its blocks is not a mapping.
    ValueError: If any block or key is not recognised.
  """
  if not isinstance(saturation_config, dict):
    raise TypeError(
        "saturation_config must be a dict (got"
        f" {type(saturation_config).__name__})."
    )

  unknown = sorted(set(saturation_config) - _CONFIG_BLOCKS)
  if unknown:
    raise ValueError(
        f"Unknown saturation_config block(s): {unknown}. "
        f"Valid blocks are {sorted(_CONFIG_BLOCKS)}."
    )

  for block, valid in _CONFIG_KEYS.items():
    values = saturation_config.get(block)
    if values is None:
      continue
    if not isinstance(values, dict):
      raise TypeError(
          f"saturation_config['{block}'] must be a dict (got"
          f" {type(values).__name__})."
      )
    bad = sorted(set(values) - valid)
    if bad:
      raise ValueError(
          f"Unknown key(s) in saturation_config['{block}']: {bad}. "
          f"Valid keys are {sorted(valid)}."
      )

  grid = (saturation_config.get("saturation") or {}).get("grid")
  if grid is not None:
    if not isinstance(grid, dict):
      raise TypeError(
          "saturation_config['saturation']['grid'] must be a dict (got"
          f" {type(grid).__name__})."
      )
    bad = sorted(set(grid) - _GRID_KEYS)
    if bad:
      raise ValueError(
          f"Unknown key(s) in saturation_config['saturation']['grid']: {bad}. "
          f"Valid keys are {sorted(_GRID_KEYS)}."
      )
    # Validates `mode` (legacy min/max without mode -> manual).
    saturation_lib.resolve_grid_mode(grid)

  learners = (saturation_config.get("alpha_selection") or {}).get("learners")
  if learners is not None:
    if not isinstance(learners, dict):
      raise TypeError(
          "saturation_config['alpha_selection']['learners'] must be a dict"
          f" (got {type(learners).__name__})."
      )
    bad = sorted(set(learners) - _LEARNER_KEYS)
    if bad:
      raise ValueError(
          "Unknown key(s) in saturation_config['alpha_selection']['learners']:"
          f" {bad}. Valid keys are {sorted(_LEARNER_KEYS)}."
      )

  # Numeric sanity of the alpha_selection keys.
  # Named `alpha_sel_cfg` (not `selection`) so it cannot shadow the
  # `evaluation.selection` module imported above.
  alpha_sel_cfg = saturation_config.get("alpha_selection") or {}
  for key, lower in (("flatness_ratio", 1.0), ("t_ref_tolerance", 0.0)):
    value = alpha_sel_cfg.get(key)
    if value is None:
      continue
    if not isinstance(value, (int, float)) or isinstance(value, bool):
      raise ValueError(
          f"alpha_selection.{key} must be numeric (got {value!r})."
      )
    if float(value) <= lower:
      raise ValueError(
          f"alpha_selection.{key} must be > {lower:g} (got {value})."
      )
  fallback = alpha_sel_cfg.get("fallback_half_saturation_T")
  if fallback is not None:
    if not isinstance(fallback, (int, float)) or isinstance(fallback, bool):
      raise ValueError(
          "alpha_selection.fallback_half_saturation_T must be a number or"
          f" null (got {fallback!r})."
      )
    if float(fallback) <= 0.0:
      raise ValueError(
          "alpha_selection.fallback_half_saturation_T must be > 0 (a"
          f" half-saturation point in units of T; got {fallback})."
      )
  t_critical = alpha_sel_cfg.get("linear_test_t_critical")
  if t_critical is not None and not isinstance(t_critical, bool):
    raise ValueError(
        "alpha_selection.linear_test_t_critical must be a boolean (got"
        f" {t_critical!r})."
    )

  # The optimization block.
  optimization_cfg = saturation_config.get("optimization") or {}
  mode = optimization_cfg.get("budget_mode")
  if mode is not None and str(mode).strip().lower() not in (
      "fixed",
      "roi_target",
  ):
    raise ValueError(
        "optimization.budget_mode must be 'fixed' or 'roi_target' (got"
        f" '{mode}')."
    )
  reallocate = optimization_cfg.get("budget_reallocate")
  if reallocate is not None and not isinstance(reallocate, bool):
    raise ValueError(
        "optimization.budget_reallocate must be a boolean (got"
        f" {reallocate!r})."
    )
  cap_quantile = optimization_cfg.get("cap_quantile")
  if cap_quantile is not None:
    if not isinstance(cap_quantile, (int, float)) or isinstance(
        cap_quantile, bool
    ):
      raise ValueError(
          f"optimization.cap_quantile must be numeric (got {cap_quantile!r})."
      )
    if not 0.0 < float(cap_quantile) <= 1.0:
      raise ValueError(
          "optimization.cap_quantile must be in (0, 1] (got"
          f" {cap_quantile}); 1.0 = the cell maximum."
      )
  within = optimization_cfg.get("reallocate_within")
  if within is not None and str(within).strip().lower() not in (
      optimization.REALLOCATE_WITHIN
  ):
    raise ValueError(
        "optimization.reallocate_within must be one of"
        f" {list(optimization.REALLOCATE_WITHIN)} (got '{within}')."
    )


class TRefMismatchError(RuntimeError):
  """Raised when the Phase 1 T_ref does not match the Phase 0 T_ref.

  Corresponds to a failure of Diag6 (unit consistency of T_ref): the T_ref
  recomputed at the start of Phase 1 differs from the T_ref saved by Phase 0,
  which indicates a unit or handover bug.
  """


class CausalWorkflowOrchestrator:
  """Main orchestrator handling causal inference modeling.

  Handles execution, parallelization, selection, and consolidation.
  """

  def __init__(
      self,
      y_col: str,
      d_cols: List[str],
      x_cols_list: List[List[str]],
      ml_models_y: List[str],
      ml_models_t: List[str],
      n_folds_list: List[int],
      output_dir: str,
      n_jobs: int = -1,
      covariates_for_cate_list: Optional[List[List[str]]] = None,
      cate_structure_list: Optional[List[Any]] = None,
      time_budget: int = 15,
      date_col: Optional[str] = None,
      geo_col: Optional[str] = None,
      item_col: Optional[str] = None,
      geo_name_col: Optional[str] = None,
      item_name_col: Optional[str] = None,
      run_sensitivity: bool = False,
      sensitivity_scope: Optional[List[str]] = None,
      ground_truth_existence: bool = False,
      ground_truth_effect_column: Optional[List[str]] = None,
      treatment_types: Optional[Dict[str, str]] = None,
      treatment_spend_cols: Optional[Dict[str, str]] = None,
      sales_col: Optional[str] = None,
      peak_months: Optional[List[int]] = None,
      optimize: bool = False,
      # `None` means "derive it": False when a saturation family is set,
      # True otherwise. An explicit True/False from the caller always wins.
      simple_optimization: Optional[bool] = None,
      opt_periods: Optional[tuple[Any, Any]] = None,
      opt_threshold_roi: float = DEFAULT_THRESHOLD_ROI,
      opt_other_cost_variables: Optional[List[str]] = None,
      generate_html: bool = True,
      # Geos / items whose rows are allocation units of the
      # saturation-aware optimization. `None` = all (when the column is set).
      opt_geos: Optional[List[Any]] = None,
      opt_items: Optional[List[Any]] = None,
      # ---- Saturation-aware DML extension --------------------------------
      # Every saturation setting lives in this ONE dict. Omit it entirely and
      # the existing behaviour is reproduced exactly (family=None -> Phase 0
      # skipped). The only required key is `saturation.family`.
      saturation_config: Optional[Dict[str, Any]] = None,
      # ---- Hyperparameter policy ----------------------------------------
      # "global" (default): ONE FLAML search per (treatment, learner) on the
      # full Phases 1-3 sample at the start of Phase 1; every fold of every
      # Phase 1 / Phase 3 run then fits that fixed theta*.  "per_fold": the
      # previous behaviour (a FLAML search inside every fold).
      hparam_policy: str = DEFAULT_HPARAM_POLICY,
      generate_agent: bool = True,
  ):
    """Initializes the central parameters for orchestration tasks."""
    # Validate parameters
    if not isinstance(d_cols, list):
      raise TypeError("d_cols must be a list of strings.")

    if not isinstance(hparam_policy, str) or (
        hparam_policy not in HPARAM_POLICIES
    ):
      raise ValueError(
          f"hparam_policy must be one of {list(HPARAM_POLICIES)} (got"
          f" {hparam_policy!r})."
      )
    self.hparam_policy = hparam_policy

    if not isinstance(x_cols_list, list):
      raise TypeError("x_cols_list must be a list of lists.")
    if not all(isinstance(x, list) for x in x_cols_list):
      raise TypeError("Each element in x_cols_list must be a list.")
    if len(x_cols_list) != len(d_cols):
      raise ValueError(
          f"Length of x_cols_list ({len(x_cols_list)}) must match length of"
          f" d_cols ({len(d_cols)})."
      )

    if covariates_for_cate_list is not None:
      if not isinstance(covariates_for_cate_list, list):
        raise TypeError("covariates_for_cate_list must be a list of lists.")
      if not all(isinstance(x, list) for x in covariates_for_cate_list):
        raise TypeError(
            "Each element in covariates_for_cate_list must be a list."
        )
      if len(covariates_for_cate_list) != len(d_cols):
        raise ValueError(
            "Length of covariates_for_cate_list "
            f"({len(covariates_for_cate_list)}) must match length of "
            f"d_cols ({len(d_cols)})."
        )

    self.y_col = y_col
    self.d_cols = d_cols
    self.x_cols_list = x_cols_list
    self.ml_models_y = ml_models_y
    self.ml_models_t = ml_models_t
    self.n_folds_list = n_folds_list
    self.output_dir = output_dir
    self.n_jobs = n_jobs
    self.covariates_for_cate_list = covariates_for_cate_list or [
        [] for _ in d_cols
    ]
    self.cate_structure_list = cate_structure_list or [
        {
            "type": "auto_additive",
            "df": 3,
            "degree": 2,
            "include_intercept": False,
        }
        for _ in d_cols
    ]
    self.time_budget = time_budget
    self.date_col = date_col
    self.geo_col = geo_col
    self.item_col = item_col
    self.geo_name_col = geo_name_col
    self.item_name_col = item_name_col
    self.run_sensitivity = run_sensitivity
    self.sensitivity_scope = sensitivity_scope or ["total"]
    if "geo" in self.sensitivity_scope and self.geo_col is None:
      raise ValueError(
          "geo_col must be specified when sensitivity_scope includes 'geo'."
      )
    if "item" in self.sensitivity_scope and self.item_col is None:
      raise ValueError(
          "item_col must be specified when sensitivity_scope includes 'item'."
      )
    self.ground_truth_existence = ground_truth_existence
    self.ground_truth_effect_column = ground_truth_effect_column or []
    self.treatment_types = treatment_types or {}
    for d_col in self.d_cols:
      if d_col not in self.treatment_types:
        self.treatment_types[d_col] = "spend"

    self.treatment_spend_cols = treatment_spend_cols or {}
    for d_col in self.d_cols:
      if self.treatment_types[d_col] == "spend":
        if d_col not in self.treatment_spend_cols:
          self.treatment_spend_cols[d_col] = d_col
    self.sales_col = sales_col
    self.optimize = optimize

    # ------------------------------------------------------------------
    # Every saturation setting is read from the SINGLE `saturation_config`
    # dict. Only `saturation.family` is required; all other keys
    # fall back to their documented defaults, so enabling saturation is a
    # one-key change. If the `saturation` block is absent, family=None is set
    # automatically, Phase 0 is skipped and Phases 1-4 run unchanged.
    # ------------------------------------------------------------------
    saturation_config = dict(saturation_config or {})
    _validate_config(saturation_config)

    self.saturation_cfg = dict(saturation_config.get("saturation") or {})
    # Block omitted -> None (Phase 0 skipped); block
    # present but `family` unspecified -> the default family `tanh`.  An
    # explicit `family: None` inside the block keeps the linear behaviour.
    # `saturation_cfg` itself is left exactly as passed.
    saturation_block_present = saturation_config.get("saturation") is not None
    if saturation_block_present and "family" not in self.saturation_cfg:
      family_value = saturation_lib.DEFAULT_FAMILY
    else:
      family_value = self.saturation_cfg.get("family")
    self.sat_family = saturation_lib.validate_family(family_value)
    self.alpha_selection_cfg = dict(
        saturation_config.get("alpha_selection") or {}
    )
    final_fit_cfg = dict(saturation_config.get("final_fit") or {})
    self.final_fit_cfg = final_fit_cfg
    self.final_fit = str(final_fit_cfg.get("data", "d_inf")).strip().lower()
    if self.final_fit not in ("d_inf", "full_sample"):
      raise ValueError(
          "final_fit.data must be 'd_inf' or 'full_sample' (got"
          f" '{self.final_fit}')."
      )

    # A saturation run is only meaningful with the saturation-aware optimizer,
    # so derive it rather than making the caller remember a second flag. An
    # explicit True/False from the caller always wins ...
    if simple_optimization is None:
      simple_optimization = self.sat_family is None

    # ... except for non-spend treatments.  The saturation-aware optimizer
    # budgets the treatment column directly, which presupposes `g(T) = T` --
    # the treatment IS the money.  For "percentage" / "impressions" / "price"
    # a cost model `spend = g(treatment)` would be required and is not yet
    # specified, so those types ALWAYS run the simple optimizer:
    # `simple_optimization` is forced to True (and reported as such in
    # metadata.json / the HTML header).
    non_spend_treatments = [
        d
        for d in self.d_cols
        if self.treatment_types.get(d, "spend") != "spend"
    ]
    forced_simple_optimization = bool(non_spend_treatments) and (
        not simple_optimization
    )
    if forced_simple_optimization:
      described = ", ".join(
          f"'{d}' ({self.treatment_types[d]})" for d in non_spend_treatments
      )
      warnings.warn(
          "simple_optimization is forced to True because the saturation-aware"
          " optimizer is only implemented for treatment_type='spend';"
          f" treatment(s) {described} will use the simple optimizer.",
          UserWarning,
      )
      simple_optimization = True

    optimization_cfg = dict(saturation_config.get("optimization") or {})
    budget_increment = optimization_cfg.get("budget_increment", 0.0)
    if not isinstance(budget_increment, (int, float)):
      raise ValueError("budget_increment must be a numeric value.")
    # budget_increment >= -1 (a budget cannot shrink by more than 100 %).
    if float(budget_increment) < -1.0:
      raise ValueError(
          f"budget_increment must be >= -1 (got {budget_increment})."
      )
    self.budget_increment = float(budget_increment)
    # budget_mode (fixed | roi_target) and
    # budget_reallocate (fixed only, default True).
    self.budget_mode = (
        str(optimization_cfg.get("budget_mode", "fixed") or "fixed")
        .strip()
        .lower()
    )
    reallocate_value = optimization_cfg.get("budget_reallocate")
    self._budget_reallocate_given = reallocate_value is not None
    self.budget_reallocate = (
        True if reallocate_value is None else bool(reallocate_value)
    )
    # cap_quantile (support cap T_bar_t = max(Q_q(cell), T_actual); 1.0 =
    # cell maximum) and reallocate_within ("all" | "date"); both validated
    # by _validate_config above.
    self.cap_quantile = float(
        optimization_cfg.get("cap_quantile", optimization.DEFAULT_CAP_QUANTILE)
    )
    self.reallocate_within = (
        str(optimization_cfg.get("reallocate_within", "all") or "all")
        .strip()
        .lower()
    )

    # This validation runs at Orchestrator initialization, i.e. strictly
    # BEFORE the Phase 0 / Stage A computation.
    if optimize and not simple_optimization and self.sat_family is None:
      raise ValueError(
          "optimize=True with simple_optimization=False requires a saturation"
          " function: the budget allocation problem is a linear program under"
          " the linear PLR and would put the whole budget on a single time"
          " point (corner solution). Please set a saturation family, e.g."
          " saturation_config={'saturation': {'family': 'tanh'}} ('tanh' |"
          " 'exp' | 'log' |"
          " 'power'), or keep simple_optimization=True."
      )
    # The reverse combination is not an error -- an explicit choice wins -- but
    # it is almost never what the caller meant: Phase 0 would pay for the full
    # alpha search and the simple optimizer would then ignore the estimated
    # curve entirely. Only reachable by passing simple_optimization=True
    # explicitly, since the derived value would have been False (non-spend
    # treatments always run the simple optimizer, see above, so no warning).
    if (
        optimize
        and simple_optimization
        and self.sat_family is not None
        and not non_spend_treatments
    ):
      warnings.warn(
          "simple_optimization=True was passed explicitly together with"
          f" saturation family '{self.sat_family}'. Phase 0 will still"
          " estimate alpha, but the simple optimizer ignores the saturation"
          " curve, so that work is discarded. Drop simple_optimization to let"
          " it default to False, or remove the saturation block.",
          UserWarning,
      )
    self.simple_optimization = simple_optimization

    # Phase 0 state (populated by `_run_phase0`; stays empty when family=None).
    self._cluster_plan: Optional[phase0.ClusterKeyPlan] = None
    self._top_level_split: Optional[phase0.TopLevelSplit] = None
    self._alpha_results: Dict[str, phase0.AlphaSelectionResult] = {}
    self._alpha_hat: Dict[str, float] = {}
    # Phase 0 output per treatment (family or None).
    self._selected_family: Dict[str, Optional[str]] = {}
    # step0_saturation/ directory per treatment.
    self._phase0_out_dirs: Dict[str, str] = {}
    # Phase 4 optimization path actually taken per treatment.
    self._phase4_optimization: Dict[str, Dict[str, Any]] = {}
    # Treatment normalization bridge: the scaler divisor `c` per treatment,
    # fitted ONCE on the full input frame and reused by Stage B, by the
    # Phases 1-3 transform and by Phase 4, so all three see the same scale.
    self._treatment_divisor: Dict[str, pd.Series] = {}
    self._treatment_normalization: Dict[str, str] = {}
    # The same divisor with its user-facing formula / description, so logs,
    # the Phase 0 panel and summary_alpha_selection.json can say what `c` is.
    self._treatment_divisor_info: Dict[str, phase0.TreatmentDivisor] = {}
    # T_ref = q95(T | T > 0) per treatment, on the Phase 0
    # scale T / c, computed ONCE on the full frame before the split (always
    # on).  `_t_ref_for` prefers the value written to
    # summary_alpha_selection.json.
    self._t_ref: Dict[str, float] = {}
    # T_max = q99.9(T | T > 0) per treatment (the right end of the T-basis
    # domain, `saturation.BASIS_T_MAX_QUANTILE`), on the same scale and
    # computed at the same time as T_ref.
    self._t_max: Dict[str, float] = {}
    self._phase0_diagnostics: Dict[str, Any] = {}
    self._pending_cluster_log: Optional[str] = None
    # Phase 0 `d_inf` split: the D_sel rows (treatment transformed, cluster
    # helper columns attached) that every Phase 1 / Phase 3 repetition scores
    # with the fitted CATE so the per-row outputs cover the whole sample.
    self._score_frame: Optional[pd.DataFrame] = None
    # Phase 3 result per treatment (`model_name`, `model_obj`, `df_1a`,
    # `df_1b`, `metrics_1c`, `cate_coef_dict`), populated by
    # `run_full_pipeline`.  `model_obj` is the fitted estimator that is also
    # exported as `best_dml_object_<model>.pkl`; keeping it here lets callers
    # use it in-process instead of unpickling that file (unpickling can
    # execute arbitrary code).
    self.best_model_data: Dict[str, Dict[str, Any]] = {}

    self.opt_periods = opt_periods

    if not isinstance(opt_threshold_roi, (int, float)) or isinstance(
        opt_threshold_roi, bool
    ):
      raise ValueError("opt_threshold_roi must be a numeric value.")
    # The ROI threshold is a constraint ROI_t >= r
    # with r >= 0; a negative threshold is rejected at initialization.
    if float(opt_threshold_roi) < 0.0:
      raise ValueError(
          f"opt_threshold_roi must be >= 0 (got {opt_threshold_roi})."
      )
    self.opt_threshold_roi = opt_threshold_roi
    # roi_target needs r > 0 (with r = 0 every unit
    # would merely sit at its historical maximum).
    if self.budget_mode == "roi_target" and float(opt_threshold_roi) == 0.0:
      raise ValueError(
          "saturation_config['optimization']['budget_mode'] = 'roi_target'"
          " requires opt_threshold_roi > 0 (got 0)."
      )
    # opt_geos / opt_items restrict the allocation units.
    for name, value, column in (
        ("opt_geos", opt_geos, geo_col),
        ("opt_items", opt_items, item_col),
    ):
      if value is None:
        continue
      if isinstance(value, (str, bytes)) or not isinstance(
          value, (list, tuple, set)
      ):
        raise TypeError(f"{name} must be a list of values or None.")
      if column is None:
        raise ValueError(
            f"{name} was given but the corresponding column"
            f" ({'geo_col' if name == 'opt_geos' else 'item_col'}) is not"
            " set."
        )
    self.opt_geos = list(opt_geos) if opt_geos is not None else None
    self.opt_items = list(opt_items) if opt_items is not None else None

    self.opt_other_cost_variables = opt_other_cost_variables or []
    self.peak_months = (
        peak_months if peak_months is not None else [3, 7, 11, 12]
    )
    self.generate_html = generate_html
    self.generate_agent = generate_agent

  def _validate_columns(self, df: pd.DataFrame) -> None:
    """Checks that every user-specified column exists in ``df``.

    Runs once at the start of ``run_full_pipeline`` so that a misspelled
    column name fails immediately with ONE message naming every offending
    parameter.  Without this check the typo would surface late as a raw
    ``KeyError`` inside a worker (``y_col`` / ``d_cols`` / ``x_cols_list`` /
    ``covariates_for_cate_list``) or, worse, be silently dropped by a
    downstream ``if col in df.columns`` guard (``date_col`` / ``geo_col`` /
    ``item_col`` / ...), changing the cluster DML keys, the charts and the
    summaries without any notice.

    Optional parameters are only checked when they were set;
    ``ground_truth_effect_column`` only when ``ground_truth_existence`` is
    True.  ``treatment_spend_cols`` is left to ``_validate_treatments``,
    which already reports it.

    Args:
      df: The input DataFrame of ``run_full_pipeline``.

    Raises:
      ValueError: If any specified column is not in ``df.columns``.
    """
    # Order is preserved (dict.fromkeys de-duplicates without sorting).
    required: Dict[str, List[Any]] = {
        "y_col": [self.y_col],
        "d_cols": list(dict.fromkeys(self.d_cols)),
        "x_cols_list": list(
            dict.fromkeys(itertools.chain.from_iterable(self.x_cols_list))
        ),
        "covariates_for_cate_list": list(
            dict.fromkeys(
                itertools.chain.from_iterable(self.covariates_for_cate_list)
            )
        ),
    }
    for name in (
        "date_col",
        "geo_col",
        "item_col",
        "geo_name_col",
        "item_name_col",
        "sales_col",
    ):
      value = getattr(self, name)
      if value:
        required[name] = [value]
    if self.opt_other_cost_variables:
      required["opt_other_cost_variables"] = list(
          dict.fromkeys(self.opt_other_cost_variables)
      )
    if self.ground_truth_existence:
      required["ground_truth_effect_column"] = list(
          dict.fromkeys(self.ground_truth_effect_column)
      )

    available = [str(c) for c in df.columns]
    problems: List[str] = []
    for param, cols in required.items():
      missing = [c for c in cols if c not in df.columns]
      if not missing:
        continue
      described = []
      for col in missing:
        hint = difflib.get_close_matches(str(col), available, n=1)
        described.append(
            f"'{col}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")
        )
      problems.append(f"{param}: {', '.join(described)}")
    if not problems:
      return

    max_listed = 50
    shown = ", ".join(repr(c) for c in available[:max_listed])
    if len(available) > max_listed:
      shown += f", ... ({len(available) - max_listed} more)"
    raise ValueError(
        "The following user-specified column(s) were not found in the input"
        " DataFrame:\n  - "
        + "\n  - ".join(problems)
        + f"\nAvailable columns ({len(available)}): [{shown}]"
    )

  def _validate_treatments(self, df: pd.DataFrame) -> None:
    """Validates treatment types, percentage bounds, and required columns.

    Specifically, validates treatment types, percentage bounds, sales_col
    presence, and target spend columns.

    Args:
      df: The input dataframe containing the treatment and sales data.
    """
    for d_col in self.d_cols:
      ttype = self.treatment_types.get(d_col, "spend")
      if ttype not in ["spend", "impressions", "percentage", "price"]:
        raise ValueError(
            f"Invalid treatment_type '{ttype}' for column '{d_col}'. "
            "treatment_type must be one of: 'spend', 'impressions', "
            "'percentage', or 'price'."
        )
      if ttype == "percentage":
        if d_col in df.columns:
          vals = df[d_col].dropna().values
          if len(vals) > 0 and (np.any(vals < 0.0) or np.any(vals > 1.0)):
            raise ValueError(
                f"Treatment column '{d_col}' is marked as 'percentage', but"
                " contains values outside [0.0, 1.0]."
            )
        if not self.sales_col or self.sales_col not in df.columns:
          raise ValueError(
              f"Treatment column '{d_col}' is marked as 'percentage', but valid"
              f" 'sales_col' ('{self.sales_col}') was not provided in"
              " dataframe."
          )
      elif ttype in ["spend", "impressions", "price"]:
        if d_col in df.columns:
          vals = df[d_col].dropna().values
          if len(vals) > 0 and np.all((vals >= 0.0) & (vals <= 1.0)):
            lower_d_col = d_col.lower()
            if any(sub in lower_d_col for sub in ["discount", "pct", "rate"]):
              warnings.warn(
                  f"Treatment column '{d_col}' has values in [0.0, 1.0] and"
                  " name containing discount/pct/rate, but is set to"
                  f" '{ttype}'. Consider setting treatment_types['{d_col}'] ="
                  " 'percentage'.",
                  UserWarning,
              )

      if d_col in self.treatment_spend_cols:
        target_spend = self.treatment_spend_cols[d_col]
        if target_spend not in df.columns:
          raise ValueError(
              f"Specified spend column '{target_spend}' for treatment"
              f" '{d_col}' not found in dataframe."
          )

    if self.optimize:
      treatment_spend_vals = list(self.treatment_spend_cols.values())

      if self.opt_periods and self.date_col in df.columns:
        try:
          opt_start = pd.to_datetime(self.opt_periods[0])
          opt_end = pd.to_datetime(self.opt_periods[1])
          df_min_date = pd.to_datetime(df[self.date_col].min())
          df_max_date = pd.to_datetime(df[self.date_col].max())
          if opt_start > opt_end:
            warnings.warn(
                f"opt_periods start ({opt_start}) is after end ({opt_end})."
                f" Falling back to default date range: {df_min_date} to"
                f" {df_max_date}.",
                UserWarning,
            )
            self.opt_periods = None
          elif opt_start < df_min_date or opt_end > df_max_date:
            warnings.warn(
                f"opt_periods ({opt_start} to {opt_end}) is outside the"
                f" available data range ({df_min_date} to {df_max_date})."
                " Falling back to default date range.",
                UserWarning,
            )
            self.opt_periods = None
        except (ValueError, TypeError):
          pass

      if self.opt_other_cost_variables:
        for col in self.opt_other_cost_variables:
          if col in self.d_cols:
            raise ValueError(
                f"Variable '{col}' in opt_other_cost_variables is also a"
                " treatment variable (d_cols)."
            )
          if col in treatment_spend_vals:
            raise ValueError(
                f"Variable '{col}' in opt_other_cost_variables is also in"
                " treatment_spend_cols."
            )
          if col in df.columns:
            col_lower = col.lower()
            if any(
                sub in col_lower
                for sub in ["impression", "imps", "pct", "percentage"]
            ):
              warnings.warn(
                  f"Variable '{col}' in opt_other_cost_variables has a name"
                  " suggesting it might not be a cost variable."
              )
            else:
              vals = df[col].dropna().values
              if len(vals) > 0:
                if np.all((vals >= 0.0) & (vals <= 1.0)):
                  warnings.warn(
                      f"Variable '{col}' in opt_other_cost_variables has values"
                      " between 0 and 1, suggesting it might not be a cost"
                      " variable."
                  )
                elif np.any(vals < 0.0):
                  warnings.warn(
                      f"Variable '{col}' in opt_other_cost_variables contains"
                      " negative values, suggesting it might not be a cost"
                      " variable."
                  )

      for col in df.columns:
        col_lower = col.lower()
        if any(sub in col_lower for sub in ["cost", "spend"]):
          if (
              col not in treatment_spend_vals
              and col not in self.opt_other_cost_variables
          ):
            warnings.warn(
                f"Column '{col}' in the dataframe contains 'cost' or 'spend' in"
                " its name, but is not included in treatment_spend_cols or"
                " opt_other_cost_variables."
            )

  def _monitor_phase0_progress(
      self,
      target_dir: str,
      total_tasks: int,
      stop_event: threading.Event,
      interval: int = 60,
  ) -> None:
    """The periodic progress line of the Phase 0 progress display.

    One line every ``interval`` seconds while the Stage B ML trainings run:
    elapsed time, completed / total trainings and the estimated remaining
    time.  Nothing else is printed (no final summary line; a run shorter
    than one interval prints no progress line at all).

    The ``.done`` markers are written to per-treatment sub-directories of
    ``target_dir`` (``<target_dir>/<treatment>/seed_<s>_<k>_<y|t>.done``), so
    they are counted recursively.

    Args:
      target_dir: Directory where Stage B writes its `.done` markers.
      total_tasks: Total number of ML trainings scheduled by Stage B.
      stop_event: Event used to stop the polling thread.
      interval: Reporting interval in seconds.
    """
    started = time.monotonic()

    def _hms(seconds: float) -> str:
      seconds = max(0, int(round(seconds)))
      hours, remainder = divmod(seconds, 3600)
      minutes, secs = divmod(remainder, 60)
      return f"{hours:d}:{minutes:02d}:{secs:02d}"

    while not stop_event.is_set():
      for _ in range(interval * 10):
        if stop_event.is_set():
          break
        time.sleep(0.1)
      if stop_event.is_set():
        break
      completed = 0
      if os.path.exists(target_dir):
        completed = len(
            glob.glob(os.path.join(target_dir, "**", "*.done"), recursive=True)
        )
      elapsed = time.monotonic() - started
      if completed > 0 and total_tasks > 0:
        eta = _hms(elapsed * (total_tasks - completed) / completed)
      else:
        eta = "n/a"
      print(
          f"[Phase 0 Progress] elapsed {_hms(elapsed)} |"
          f" {completed}/{total_tasks} ML trainings completed | ETA {eta}",
          flush=True,
      )

  def _monitor_phase1_progress(
      self,
      target_dir: str,
      total_tasks: int,
      stop_event: threading.Event,
      interval: int = 60,
  ) -> None:
    """Polls directory to update progress messages for Phase 1."""
    while not stop_event.is_set():
      for _ in range(interval * 10):
        if stop_event.is_set():
          break
        time.sleep(0.1)
      if stop_event.is_set():
        break
      if os.path.exists(target_dir):
        metrics_files = sorted(
            glob.glob(os.path.join(target_dir, "1c_metrics_*.csv")),
            key=os.path.getmtime,
        )
        completed = len(metrics_files)
        pct = (completed / total_tasks) * 100
        if completed > 0:
          latest_model = (
              os.path.basename(metrics_files[-1])
              .replace("1c_metrics_", "")
              .replace(".csv", "")
          )
          print(
              f"[Phase 1 Progress] {completed}/{total_tasks} ({pct:.1f}%)"
              f" completed. Latest estimated model: {latest_model}",
              flush=True,
          )
        else:
          print(
              f"[Phase 1 Progress] 0/{total_tasks} (0.0%) completed...",
              flush=True,
          )
    print(
        f"[Phase 1 Progress] {total_tasks}/{total_tasks} (100.0%) completed.",
        flush=True,
    )

  def _monitor_phase3_progress(
      self,
      target_dir: str,
      total_reps: int,
      model_name: Optional[str],
      stop_event: threading.Event,
      interval: int = 60,
  ) -> None:
    """Polls directory to update progress messages for Phase 3.

    Phase 3 is ONE ``(candidate, rep)`` job
    list, so the monitor counts the ``.done`` markers of all candidates
    against the total ``M (R - 1)``; ``model_name`` is None in that case and
    the label is omitted.

    Args:
      target_dir: Directory polled for ``*.done`` completion markers.
      total_reps: Total number of jobs expected, used as the denominator of
        the progress percentage.
      model_name: Candidate model label included in the progress tag, or
        None when the monitor spans all candidates (the label is omitted).
      stop_event: Event set by the caller to stop polling and print the
        final 100% message.
      interval: Seconds to wait between successive directory polls.
    """
    tag = (
        "[Phase 3 Progress]"
        if model_name is None
        else f"[Phase 3 Progress - {model_name}]"
    )
    while not stop_event.is_set():
      for _ in range(interval * 10):
        if stop_event.is_set():
          break
        time.sleep(0.1)
      if stop_event.is_set():
        break
      if os.path.exists(target_dir):
        completed = len(glob.glob(os.path.join(target_dir, "*.done")))
        pct = (completed / total_reps) * 100 if total_reps else 100.0
        print(
            f"{tag} {completed}/{total_reps} ({pct:.1f}%) completed.",
            flush=True,
        )
    print(
        f"{tag} {total_reps}/{total_reps} (100.0%) completed.",
        flush=True,
    )

  def _insert_metadata_columns(
      self, df_1b: pd.DataFrame, meta_dict: Dict[str, Any]
  ) -> pd.DataFrame:
    """Injects tracking metadata cleanly into aggregated dataframes."""
    try:
      model_idx = list(df_1b.columns).index("model")
      for i, (k_val, v_val) in enumerate(meta_dict.items()):
        if k_val not in df_1b.columns:
          df_1b.insert(model_idx + 1 + i, k_val, v_val)
    except ValueError:
      for k_val, v_val in meta_dict.items():
        if k_val not in df_1b.columns:
          df_1b[k_val] = v_val
    return df_1b

  def _resolve_cluster_cols(self, df: pd.DataFrame) -> List[str]:
    """Returns the cluster DML keys to use for cross-fitting and CIs.

    The keys come from the cluster-key determination
    rule (the two candidate keys with the largest cluster counts). The rule is
    applied to saturation runs *and* to normal runs (family=None), so this
    helper is the single source of truth for both. If Phase 0 has not run
    (e.g. a direct unit-test call), the historical geo+item behaviour is kept.

    Args:
      df: The frame the pipeline is about to be fitted on.

    Returns:
      The list of cluster columns present in `df`.
    """
    plan = getattr(self, "_cluster_plan", None)
    if plan is not None:
      return [k for k in plan.keys if k in df.columns]

    cluster_cols = []
    if self.geo_col and self.geo_col in df.columns:
      cluster_cols.append(self.geo_col)
    if self.item_col and self.item_col in df.columns:
      cluster_cols.append(self.item_col)
    return cluster_cols

  def _strip_helper_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
    """Drops Phase 0 helper columns so existing outputs stay unchanged.

    The derived contiguous-date-block key is an internal column;
    it must not leak into the Phase 1 / Phase 3 result CSVs.

    Args:
      frame: A result frame produced by the runner.

    Returns:
      The frame without the internal helper columns.
    """
    drop_cols = [c for c in phase0.INTERNAL_HELPER_COLS if c in frame.columns]
    return frame.drop(columns=drop_cols) if drop_cols else frame

  def _cate_basis_builder(
      self,
      d_col: str,
      x_cols: List[str],
      cate_cols: List[str],
      cate_structure: Any,
  ):
    """Returns the existing pipeline's CATE basis builder `p(X)`.

    Reusing `DoubleMLPipeline._build_cate_basis` keeps a single source of truth
    for the basis: Phase 0 must profile `beta` on exactly the same `p(X)` that
    Phases 1-3 will use.  Both paths end in the shared
    module-level `runner.build_cate_basis` (the cate_structure resolution of
    `DoubleMLPipeline.__init__` is reused through the helper instance).

    Args:
      d_col: Treatment column (only used to instantiate the helper).
      x_cols: Nuisance covariates.
      cate_cols: `covariates_for_cate`.
      cate_structure: The CATE structure dict of this channel.

    Returns:
      A callable `(df_norm, df_orig) -> DataFrame`.
    """
    helper = runner.DoubleMLPipeline(
        self.y_col,
        d_col,
        x_cols,
        self.ml_models_y[0],
        self.ml_models_t[0],
        n_folds=int(self.n_folds_list[0]) if self.n_folds_list else 5,
        time_budget=self.time_budget,
        cate_structure=cate_structure,
        covariates_for_cate=cate_cols,
    )
    return helper._build_cate_basis  # pylint: disable=protected-access

  def _family_for(self, d_col: str) -> Optional[str]:
    """Returns the saturation family in force for `d_col` after Phase 0.

    Phase 0 returns `selected_family`, either the
    configured family (saturation adopted) or `None` (linear adopted); all
    later phases refer to it.  Before Phase 0 has run (or with family=None)
    the configured family is returned.

    Args:
      d_col: Treatment column.

    Returns:
      The family name, or `None` for the linear treatment.
    """
    if self.sat_family is None:
      return None
    return self._selected_family.get(d_col, self.sat_family)

  def _record_linear_fallback(self, d_col: str) -> None:
    """Warns about and records the Phase 4 linear fallback.

    The decision basis (`d_bar`, `se`, `k` and the 1-SE linear test's
    `decision_reason`) is written to
    `step0_saturation/summary_alpha_selection.json` and kept for the HTML
    descriptive block through `metadata.json`.

    Args:
      d_col: Treatment column.
    """
    result = self._alpha_results.get(d_col)
    message = (
        "saturation was not identified; falling back to linear optimization"
    )
    detail = ""
    if result is not None:
      detail = (
          f" (decision_reason = {result.decision_reason}, d_bar ="
          f" {result.d_bar:.4g}, se = {result.se:.4g} ({result.se_method}),"
          f" crit = {result.critical_value:.3g} (k = {result.linear_test_k:g});"
          f" treatment '{d_col}')"
      )
    warnings.warn(message + detail, UserWarning)
    print(f"  ├─ [Warning] {message}{detail}", flush=True)
    self._phase4_optimization[d_col] = {
        "phase4_optimization": "linear_fallback",
        "phase4_fallback_to_linear": True,
        "phase4_fallback_reason": message,
        # "1se_fail", "grid_edge_upper" or "basis_ill_conditioned".
        "phase4_fallback_decision_reason": (
            None if result is None else result.decision_reason
        ),
        "phase4_fallback_d_bar": None if result is None else result.d_bar,
        "phase4_fallback_se": None if result is None else result.se,
        "phase4_fallback_se_method": (
            None if result is None else result.se_method
        ),
        "phase4_fallback_linear_test_k": (
            None if result is None else result.linear_test_k
        ),
        "phase4_fallback_critical_value": (
            None if result is None else result.critical_value
        ),
        "simple_optimization_config": self.simple_optimization,
    }
    self._update_phase0_summary(d_col)

  def _update_phase0_summary(self, d_col: str) -> None:
    """Adds the Phase 4 path to the treatment's summary JSON (best effort)."""
    out_dir = self._phase0_out_dirs.get(d_col)
    if out_dir and d_col in self._phase4_optimization:
      saturation_evaluation.update_summary(
          out_dir, self._phase4_optimization[d_col]
      )

  def _alpha_raw_for(
      self, d_col: str, index: pd.Index
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(alpha_i, gain_i)` in RAW treatment units for Phase 4.

    Implements the normalization bridge on the Phase 4 side.  Stage B
    estimates a single alpha against the normalized treatment `T / c`; the
    optimizer budgets in raw units, so the equivalent raw-scale parameters are

        f(T/c; alpha) = gain_i * f(T; alpha_i)

    (see `saturation.rescale_alpha_to_raw`).  With no normalization `c = 1`,
    which returns `alpha_i = alpha` and `gain_i = 1`, i.e. the previous
    behaviour exactly.

    With the normalized `fbar` the gain is not needed --
    `fbar(T/c; alpha, T_ref) = fbar(T; alpha_i, c T_ref)` exactly -- so Phase
    4 uses `alpha_i` together with `_t_ref_raw_for` and ignores `gain_i`.

    Args:
      d_col: Treatment column.
      index: The index the arrays must align with.

    Returns:
      An `(alpha_raw, gain)` pair of arrays of length `len(index)`.
    """
    alpha = float(self._alpha_hat.get(d_col, 1.0))
    divisor = self._treatment_divisor.get(d_col)
    if divisor is None:
      values = np.ones(len(index), dtype=float)
    else:
      values = divisor.reindex(index).fillna(1.0).to_numpy(dtype=float)
    return saturation_lib.rescale_alpha_to_raw(self.sat_family, alpha, values)

  def _normalized_treatment(
      self, d_col: str, df_source: pd.DataFrame
  ) -> np.ndarray:
    """Returns `T / c` for `d_col` (the treatment normalization bridge)."""
    raw = (
        pd.to_numeric(df_source[d_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    divisor = self._treatment_divisor.get(d_col)
    if divisor is None:
      return raw
    values = divisor.reindex(df_source.index).fillna(1.0).to_numpy(dtype=float)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 1.0)
    return raw / values

  def _cate_response_specs(
      self, df_source: pd.DataFrame
  ) -> Dict[str, Dict[str, Any]]:
    """Describes the response curve of every saturated treatment for Phase 4.

    Feeds `plots.plot_cate_response_curves`.  For a treatment whose Phase 0
    decision adopted a saturation family the per-row response is
    `fbar(T_i / c_i; alpha_hat, T_ref)` -- exactly the regressor Phases 1-3
    used -- and the curve drawn in RAW treatment units is
    `fbar(T / c_g; alpha_hat, T_ref)` with the group's divisor `c_g` (the
    normalization bridge).  Treatments on the linear path (family=None
    or `selected_family=None`) are omitted: the chart draws a straight line
    for them.

    Args:
      df_source: The frame the chart is drawn from (raw treatment units).

    Returns:
      `{treatment: {"f_row", "divisor_row", "curve_fn", "label"}}`.
    """
    specs: Dict[str, Dict[str, Any]] = {}
    for d_col in self.d_cols:
      family = self._family_for(d_col)
      if family is None:
        continue
      try:
        t_ref = self._t_ref_for(d_col)
      except RuntimeError:
        continue
      alpha = float(self._alpha_hat.get(d_col, 1.0))
      divisor = self._treatment_divisor.get(d_col)
      if divisor is None:
        divisor_row = np.ones(len(df_source), dtype=float)
      else:
        divisor_row = (
            divisor.reindex(df_source.index).fillna(1.0).to_numpy(dtype=float)
        )

      def _curve(t_norm, fam=family, a=alpha, ref=t_ref):
        return saturation_lib.normalized_saturation(
            np.asarray(t_norm, dtype=float), fam, a, ref
        )

      specs[d_col] = {
          "f_row": _curve(self._normalized_treatment(d_col, df_source)),
          "divisor_row": divisor_row,
          "curve_fn": _curve,
          "label": (
              f"{family} saturation (alpha_hat = {alpha:.4g}, T_ref ="
              f" {t_ref:.4g} on T/c)"
          ),
      }
    return specs

  def _t_ref_for(self, d_col: str) -> float:
    """Returns `T_ref` of `d_col` on the Phase 0 scale `T / c`.

    Phases 1-3 and Phase 4 read `T_ref` from
    `summary_alpha_selection.json` (the single source of truth of the run);
    the in-memory value computed by `_run_phase0` is the fallback when the
    file is unavailable (e.g. a failed write).

    Args:
      d_col: Treatment column.

    Returns:
      `T_ref` (strictly positive).

    Raises:
      RuntimeError: If no `T_ref` is known for `d_col`.
    """
    out_dir = self._phase0_out_dirs.get(d_col)
    if out_dir:
      path = os.path.join(out_dir, saturation_evaluation.SUMMARY_JSON)
      try:
        with open(path) as handle:
          value = json.load(handle).get("T_ref")
        if value is not None and float(value) > 0.0:
          return float(value)
      except (OSError, ValueError, TypeError):
        pass
    value = self._t_ref.get(d_col)
    if value is None or not np.isfinite(value) or value <= 0.0:
      raise RuntimeError(
          f"T_ref is unknown for treatment '{d_col}'; Phase 0 must run"
          " before the saturation transform is applied."
      )
    return float(value)

  def _sensitivity_treatment_info(self, d_col: str) -> Dict[str, Any]:
    """Describes the PLR's treatment variable `D` for the sensitivity outputs.

    The sensitivity analysis runs on the Phase 1-3
    `DoubleMLPLR` as fitted -- with `D = fbar(T; alpha_hat)` when Phase 0
    adopted a family (`selected_family != None`) and `D = T` otherwise (no
    `saturation` block, or `selected_family = None`).  The columns
    `treatment_variable` ("f_bar(T)" / "T"), `selected_family`, `alpha_hat`,
    `half_saturation_T` and `T_ref` of `sensitivity_metrics.csv` come from
    here; the Phase 0 values are read from `summary_alpha_selection.json`
    (the run's single source of truth), with the in-memory result as the
    fallback when the file is unavailable.

    Args:
      d_col: Treatment column.

    Returns:
      `{"treatment_variable", "selected_family", "alpha_hat",
      "half_saturation_T", "T_ref"}`; the Phase 0 entries are None on the
      family=None path.
    """
    info: Dict[str, Any] = {
        "treatment_variable": sensitivity.TREATMENT_VARIABLE_T,
        "selected_family": None,
        "alpha_hat": None,
        "half_saturation_T": None,
        "T_ref": None,
    }
    result = self._alpha_results.get(d_col)
    if self.sat_family is None or result is None:
      return info
    summary: Dict[str, Any] = {}
    out_dir = self._phase0_out_dirs.get(d_col)
    if out_dir:
      path = os.path.join(out_dir, saturation_evaluation.SUMMARY_JSON)
      try:
        with open(path) as handle:
          loaded = json.load(handle)
        if isinstance(loaded, dict):
          summary = loaded
      except (OSError, ValueError, TypeError):
        summary = {}
    fallback = {
        "selected_family": result.selected_family,
        "alpha_hat": result.alpha_hat,
        "half_saturation_T": result.half_saturation_t,
        "T_ref": result.t_ref,
    }
    for key, default in fallback.items():
      info[key] = summary.get(key, default)
    if info["selected_family"] is not None:
      info["treatment_variable"] = sensitivity.TREATMENT_VARIABLE_FBAR
    return info

  def _t_ref_raw_for(self, d_col: str, index: pd.Index) -> np.ndarray:
    """Returns `T_ref` in RAW treatment units per row (`c_i * T_ref`).

    Normalization bridge: `fbar(T / c_i; alpha, T_ref) =
    fbar(T; alpha_i, c_i T_ref)`, so Phase 4 (which budgets in raw units)
    evaluates `fbar` with `alpha_i` from `_alpha_raw_for` and this reference
    point.

    Args:
      d_col: Treatment column.
      index: The index the array must align with.

    Returns:
      An array of length `len(index)`.
    """
    t_ref = self._t_ref_for(d_col)
    divisor = self._treatment_divisor.get(d_col)
    if divisor is None:
      values = np.ones(len(index), dtype=float)
    else:
      values = divisor.reindex(index).fillna(1.0).to_numpy(dtype=float)
    return saturation_lib.raw_scale_reference(t_ref, values)

  def _compute_theta_for_optimization(
      self,
      d_col: str,
      idx: int,
      df_est: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      best_model_data: Dict[str, Any],
  ) -> np.ndarray:
    """Recovers `theta_hat_t` in KPI units for the saturation-aware optimizer.

    The optimizer maximises `sum_t theta_t * fbar(T_t; alpha_hat)`, so `theta`
    must be expressed in KPI units. It is primarily recovered as
    `KPI_t / fbar(T_t; alpha_hat)`; rows with `T_t = 0` are filled from the
    pooled CATE coefficients evaluated on the very same basis `p(X)` that
    Phases 1-3 used.

    Args:
      d_col: Treatment column.
      idx: Index of `d_col` within `self.d_cols`.
      df_est: The raw-treatment estimation frame.
      scaler: The pipeline scaler (deep-copied before use so the caller's fitted
        state is never mutated).
      best_model_data: The `best_model_data_dict` entry of this channel.

    Returns:
      An array of `theta_hat_t` aligned with `df_est`'s rows.
    """
    frame = best_model_data.get("df_1a")
    kpi_col = f"estimated_incremental_KPI_{d_col}"
    if frame is None or kpi_col not in frame.columns:
      return np.zeros(len(df_est), dtype=float)

    kpi_values = (
        pd.to_numeric(frame[kpi_col], errors="coerce").fillna(0.0).to_numpy()
    )
    transformed_col = f"{d_col}_transformed"
    if transformed_col in frame.columns:
      f_values = (
          pd.to_numeric(frame[transformed_col], errors="coerce")
          .fillna(0.0)
          .to_numpy()
      )
    else:
      # Normalization bridge: Phases 1-3 saw
      # fbar(T/c; alpha_hat, T_ref), so the fallback must reproduce the same
      # quantity, not f(T; alpha).
      f_values = saturation_lib.normalized_saturation(
          self._normalized_treatment(d_col, df_est),
          self.sat_family,
          float(self._alpha_hat.get(d_col, 1.0)),
          self._t_ref_for(d_col),
      )

    # Reconstruct p(x)'beta on the same basis the pipeline fitted, so the
    # T = 0 rows get a sensible theta instead of 0.
    basis_theta = None
    try:
      scaler_local = copy.deepcopy(scaler)
      df_norm = scaler_local.fit_transform(df_est)
      basis_builder = self._cate_basis_builder(
          d_col,
          self.x_cols_list[idx],
          self.covariates_for_cate_list[idx],
          self.cate_structure_list[idx],
      )
      basis_df = basis_builder(df_norm, df_est)
      basis_theta = optimization.compute_basis_theta(
          basis_df, best_model_data.get("cate_coef_dict", {}) or {}
      )
    except Exception as exc:  # pylint: disable=broad-except
      warnings.warn(
          f"Could not rebuild the CATE basis for '{d_col}' ({exc}); falling"
          " back to the KPI/f ratio only."
      )

    group_keys = None
    for key_col in (self.item_col, self.geo_col):
      if key_col and key_col in df_est.columns:
        group_keys = df_est[key_col]
        break

    return optimization.estimate_theta_kpi_units(
        kpi_values, f_values, basis_theta, group_keys
    )

  def _run_phase0(
      self, df: pd.DataFrame, scaler: scalers.CausalDataScaler
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Phase 0 of the saturation-aware extension.

    Stage A performs the top-level split `D = D_sel + D_inf` on the
    intersection cells of the two adopted cluster keys;
    Stage B selects `alpha_hat` by profiled R-loss
    minimization on `D_sel`. When `family=None` the whole phase is skipped and
    only the cluster-key determination rule is evaluated (it also
    governs normal runs).

    The input `df` is never modified: every mutation happens on copies.

    Args:
      df: The raw input dataframe.
      scaler: The pipeline scaler (deep-copied before any use).

    Returns:
      A `(df_fit, df_est)` tuple. `df_fit` is what Phases 1-3 are fitted on
      (treatment replaced by `f(T; alpha_hat)` when saturation is active, plus
      the derived cluster key); `df_est` holds the very same rows with the
      original raw values and is what Phase 4 consolidates.
    """
    # The cluster-key determination rule runs for BOTH saturation
    # runs and normal runs (family=None).
    self._score_frame = None
    plan = phase0.determine_cluster_keys(
        df,
        item_col=self.item_col,
        geo_col=self.geo_col,
        date_col=self.date_col,
    )
    self._cluster_plan = plan
    df_fit = plan.attach(df)
    self._phase0_diagnostics = {
        "cluster_keys": list(plan.keys),
        "cluster_counts": dict(plan.cluster_counts),
        "candidate_cluster_counts": dict(plan.candidate_counts),
        "date_block_granularity": plan.date_block_freq,
        "messages": list(plan.messages),
    }

    if self.sat_family is None:
      # "skipped" is shown in the progress display.
      print(
          "\n======================================================================"
      )
      print(
          "=== Phase 0: Saturation-Aware Setup === skipped (no 'saturation'"
          " block in saturation_config; family=None)"
      )
      print(
          "======================================================================",
          flush=True,
      )
      # In a normal run the selected keys are logged at the start
      # of Phase 1 instead of in the Phase 0 progress display.  The line names
      # the adopted keys AND the rejected candidates so it is visible that the
      # 2-way key selection ran although no saturation_config was given.
      self._pending_cluster_log = (
          "  ├─ Cluster DML keys (determination rule applied automatically;"
          " no saturation_config needed):"
          f" {plan.describe_selection()}\n"
          "  │   date_col / item_col / geo_col keep their visualization role;"
          " the keys above only drive the cross-fitting folds and the"
          " cluster-robust standard errors."
      )
      return df_fit, df

    print(
        "\n======================================================================"
    )
    print(f"=== Phase 0: Saturation-Aware Setup (family={self.sat_family}) ===")
    print(
        "======================================================================",
        flush=True,
    )

    # `alpha_selection.output_dir` (default
    # step0_saturation) directly under the output root.
    phase0_dir = os.path.join(
        self.output_dir,
        str(
            self.alpha_selection_cfg.get("output_dir", "step0_saturation")
            or "step0_saturation"
        ),
    )
    os.makedirs(phase0_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage A: top-level split.
    # ------------------------------------------------------------------
    sel_fraction = float(self.alpha_selection_cfg.get("sel_fraction", 0.20))
    k_folds = int(self.alpha_selection_cfg.get("k_folds", 5))
    split = phase0.make_top_level_split(
        df_fit,
        plan,
        sel_fraction=sel_fraction,
        k_folds=k_folds,
        geo_col=self.geo_col,
        random_state=0,
    )
    self._top_level_split = split
    if split.forced_full_sample and self.final_fit != "full_sample":
      # Stage A final fallback: post-selection independence is lost.
      self.final_fit = "full_sample"

    # The header names the split unit actually used.
    unit_label = {
        "cell": "cell-level",
        "single_key": "cluster-level, single key",
        "observation": "observation-level fallback",
    }.get(split.unit, split.level)
    print(f"  ├─ Stage A: Top-Level Split ({unit_label})")
    print(
        "  │   Cluster DML keys (determination rule):"
        f" {plan.describe_selection()}",
        flush=True,
    )
    if plan.date_block_freq:
      print(f"  │   Date blocks: {plan.date_block_freq}")
    print(
        f"  │   Split level: {split.level} ({split.tier});"
        f" split key: {split.split_key or 'n/a'}"
    )
    if split.unit == "cell" and split.coverage is not None:
      # Coverage guarantee: every level of every key on both sides.
      n_cov = int(split.coverage["covered"].sum())
      print(
          f"  │   Split unit: cell ({split.n_clusters_sel} of"
          f" {split.n_clusters_total} cells in D_sel); coverage guarantee:"
          f" {'OK' if split.coverage_ok else 'NOT MET'}"
          f" ({n_cov}/{len(split.coverage)} key levels on both sides)"
      )
    print(
        f"  │   n_sel = {len(split.idx_sel)} rows"
        f" ({len(split.idx_sel) / max(len(df_fit), 1):.1%}),"
        f" n_inf = {len(split.idx_inf)} rows;"
        f" final_fit = {self.final_fit}",
        flush=True,
    )

    # Three-tier rule for cluster-identifier dummies.
    dummy_messages: List[str] = []
    if self.final_fit == "d_inf":
      check_cols = sorted(
          set(itertools.chain.from_iterable(self.x_cols_list))
          | set(itertools.chain.from_iterable(self.covariates_for_cate_list))
      )
      dummy_messages = phase0.check_identifier_dummies(
          df_fit,
          split.idx_inf,
          check_cols,
          split.split_key,
          self.geo_col,
      )
      for message in dummy_messages:
        print(f"  │   [Warning] {message}")

    self._phase0_diagnostics.update({
        "split_level": split.level,
        "split_tier": split.tier,
        "split_key": split.split_key,
        "sel_fraction": sel_fraction,
        "n_sel_rows": int(len(split.idx_sel)),
        "n_inf_rows": int(len(split.idx_inf)),
        "n_clusters_total": int(split.n_clusters_total),
        "n_clusters_sel": int(split.n_clusters_sel),
        "final_fit": self.final_fit,
        "forced_full_sample": bool(split.forced_full_sample),
        "three_tier_rule_messages": dummy_messages,
        "stage_a_messages": list(split.messages),
        "split_unit": split.unit,
        "split_keys": list(split.split_keys),
        "coverage_ok": bool(split.coverage_ok),
    })

    # ------------------------------------------------------------------
    # Stage B: alpha selection.
    # The Stage B display is a fixed sequence of lines and nothing else (no
    # internal step logs).  The ML-training count line opens the stage; the
    # progress monitor follows; the per-treatment lines (learners, grid,
    # Diag1, alpha estimate, Diag2 and the 1-SE decision) are printed once
    # Stage B is done; the completion message closes the stage.
    # ------------------------------------------------------------------
    df_sel = df_fit.loc[split.idx_sel]
    # With a cell-level split the Stage B folds are built on
    # the same intersection cells (round-robin rule with M = K).
    cluster_labels, fold_key_values, fold_key_names = (
        phase0.stage_b_cluster_inputs(df_sel, split, plan)
    )
    # se_obs (of the 1-SE linear test) is clustered on the selected key with
    # the FEWER clusters (conservative).  Under the observation-level
    # fallback (no usable key) the iid formula is used.
    se_key = None
    if split.level == "cluster":
      usable = [k for k in plan.keys if k in df_sel.columns]
      if usable:
        se_key = min(usable, key=lambda k: plan.cluster_counts.get(k, 0))
    se_labels = (
        df_sel[se_key].astype(str).to_numpy() if se_key is not None else None
    )
    # cluster_id = the intersection cell of the two
    # adopted keys (or the combination of the selected keys otherwise).
    if split.cell_ids is not None:
      combined_ids = split.cell_ids.astype(str)
    else:
      id_keys = [k for k in plan.keys if k in df_fit.columns]
      if id_keys:
        combined_ids = df_fit[id_keys].astype(str).agg("|".join, axis=1)
      else:
        combined_ids = pd.Series(df_fit.index.astype(str), index=df_fit.index)
    if split.unit == "cell":
      key1, key2 = split.split_keys
      sel_key_values = (
          df_sel[key1].astype(str).to_numpy(),
          df_sel[key2].astype(str).to_numpy(),
      )
      df_inf_keys = df_fit.loc[split.idx_inf]
      inf_key_values = (
          df_inf_keys[key1].astype(str).to_numpy(),
          df_inf_keys[key2].astype(str).to_numpy(),
      )
    else:
      sel_key_values = inf_key_values = None

    n_seeds = int(self.alpha_selection_cfg.get("n_seeds", 10))
    # Compute accounting: [N_y + N_t (only with learners.learner_selection:
    # true)] + 2 * S * K refits.
    learner_selection = bool(
        (self.alpha_selection_cfg.get("learners") or {}).get(
            "learner_selection", False
        )
    )
    # The ML-training count line of the progress display.
    count_line, total_tasks = phase0.describe_ml_training_count(
        len(self.d_cols),
        n_seeds,
        k_folds,
        n_models_y=len(self.ml_models_y),
        n_models_t=len(self.ml_models_t),
        learner_selection=learner_selection,
    )
    print(f"  ├─ Stage B: {count_line}", flush=True)

    # Normalization bridge. The divisor `c` is resolved ONCE, on the
    # full input frame, so Stage B, the Phases 1-3 substitution and Phase 4 all
    # refer to exactly the same scale. This also raises early (before any ML
    # work) if a treatment is configured for a shifting normalization.
    # T_ref = q95(T | T > 0) and the T-basis domain right end T_max =
    # q99.9(T | T > 0) are computed on the SAME full frame, on the same scale
    # T / c, before the split (no Y is used).  The divisor formula, T_ref and
    # T_max go to summary_alpha_selection.json and the HTML Phase 0 block;
    # the display shows T_ref in the grid line and T_max in the Diag1 line.
    t_ref_quantile = float(
        self.saturation_cfg.get(
            "t_ref_quantile", saturation_lib.DEFAULT_T_REF_QUANTILE
        )
    )
    for d_col in self.d_cols:
      divisor_info = phase0.resolve_treatment_divisor(scaler, d_col, df_fit)
      self._treatment_divisor_info[d_col] = divisor_info
      self._treatment_divisor[d_col] = divisor_info.divisor
      self._treatment_normalization[d_col] = divisor_info.mode
      normalized_t = self._normalized_treatment(d_col, df_fit)
      self._t_ref[d_col] = saturation_lib.reference_point(
          normalized_t, t_ref_quantile
      )
      self._t_max[d_col] = saturation_lib.basis_domain_max(normalized_t)

    progress_dir = os.path.join(self.output_dir, "temp_p0_alpha")
    if os.path.exists(progress_dir):
      shutil.rmtree(progress_dir, ignore_errors=True)
    os.makedirs(progress_dir, exist_ok=True)

    stop_event_p0 = threading.Event()
    monitor_thread_p0 = threading.Thread(
        target=self._monitor_phase0_progress,
        args=(progress_dir, total_tasks, stop_event_p0, 60),
    )
    monitor_thread_p0.start()

    alpha_rows = []
    try:
      for idx, d_col in enumerate(self.d_cols):
        result = phase0.select_alpha(
            df_sel=df_sel,
            y_col=self.y_col,
            d_col=d_col,
            x_cols=self.x_cols_list[idx],
            basis_builder=self._cate_basis_builder(
                d_col,
                self.x_cols_list[idx],
                self.covariates_for_cate_list[idx],
                self.cate_structure_list[idx],
            ),
            scaler=scaler,
            family=self.sat_family,
            saturation_cfg=self.saturation_cfg,
            selection_cfg=self.alpha_selection_cfg,
            cluster_labels=cluster_labels,
            ml_models_y=self.ml_models_y,
            ml_models_t=self.ml_models_t,
            time_budget=self.time_budget,
            n_jobs=self.n_jobs,
            progress_dir=os.path.join(progress_dir, d_col),
            random_state=idx,
            divisor=self._treatment_divisor[d_col]
            .reindex(df_sel.index)
            .fillna(1.0)
            .to_numpy(dtype=float),
            normalization=self._treatment_normalization[d_col],
            n_total=len(df_fit),
            se_cluster_labels=se_labels,
            se_cluster_key=se_key,
            item_col=self.item_col,
            geo_col=self.geo_col,
            date_col=self.date_col,
            fold_key_values=fold_key_values,
            fold_key_names=fold_key_names,
            t_ref=self._t_ref[d_col],
            t_max=self._t_max[d_col],
        )
        self._alpha_results[d_col] = result
        self._alpha_hat[d_col] = float(result.alpha_hat)
        self._selected_family[d_col] = result.selected_family
        alpha_rows.append({
            "treatment": d_col,
            "family": self.sat_family,
            "selected_family": result.selected_family,
            "decision": result.decision,
            "decision_reason": result.decision_reason,
            "grid_edge_hit": result.grid_edge_hit,
            "T_ref": result.t_ref,
            "alpha_hat": result.alpha_hat,
            "alpha_hat_grid": result.grid_alpha_hat,
            "brent_refined": result.refined,
            "fallback_half_saturation_T_used": (
                result.fallback_half_saturation_used
            ),
            "ml_model_Y": result.ml_model_y,
            "ml_model_T": result.ml_model_t,
            "diag1_fired": result.diag1_fired,
            "basis_cond": result.basis_cond,
            "T_max": result.t_max,
            "diag2_fired": result.diag2_fired,
            "diag2_ratio": result.diag2_ratio,
            "r_T": result.r_t,
            "saturation_at_T_ref": result.saturation_at_t_ref,
            "normalization": result.normalization,
            "n_ml_fits": result.n_ml_fits,
            "multioutput_native": result.multioutput_native,
        })
    finally:
      stop_event_p0.set()
      monitor_thread_p0.join()
      shutil.rmtree(progress_dir, ignore_errors=True)

    alpha_row_by_col = {row["treatment"]: row for row in alpha_rows}
    for d_col, result in self._alpha_results.items():
      # The per-treatment lines of the progress display.
      for line in self._phase0_display_lines(d_col, result):
        print(f"  │   {line}", flush=True)
      # step0_saturation/ CSV / JSON / figures.
      out_dir_d = saturation_evaluation.treatment_output_dir(
          phase0_dir, d_col, len(self.d_cols)
      )
      self._phase0_out_dirs[d_col] = out_dir_d
      divisor_info = self._treatment_divisor_info[d_col]
      saturation_evaluation.write_phase0_outputs(
          result,
          out_dir_d,
          treatment=d_col,
          cluster_keys=list(plan.keys),
          cluster_fallback_tier=split.tier,
          n_inf=int(len(split.idx_inf)),
          sel_cluster_ids=combined_ids.reindex(df_sel.index).to_numpy(),
          inf_cluster_ids=combined_ids.reindex(split.idx_inf).to_numpy(),
          sel_key_values=sel_key_values,
          inf_key_values=inf_key_values,
          coverage=split.coverage,
          summary_extras={
              "split_unit": split.unit,
              "split_keys": list(split.split_keys),
              "coverage_ok": bool(split.coverage_ok),
              "n_cells_total": int(split.n_clusters_total),
              "n_cells_sel": int(split.n_clusters_sel),
              # What the "normalized treatment T/c" of alpha_hat / T_ref /
              # the charts concretely is (formula of c and its meaning).
              **divisor_info.as_dict(),
          },
      )
      # The diagnostic plots are ALWAYS emitted.  The former
      # alpha_profile_<treatment>.csv is gone: its columns (alpha,
      # mean_r_loss, in_1se_set) are part of summary_alpha_grid.csv.
      #
      # The stats panel mirrors this treatment's row of
      # `summary_alpha_selection.csv` so the png is readable on its own, plus
      # the compute accounting of the run.  Every row that belongs
      # to one of the identification diagnostics is tagged "diag1" ... "diag6"
      # (shown in the table's `diagnostics` column); the other rows are the
      # result / process metrics.
      summary_row = alpha_row_by_col.get(d_col, {})

      def _fmt3(value: float) -> str:
        return f"{value:.3f}" if np.isfinite(value) else "n/a"

      phi_at_hat = " / ".join(
          _fmt3(result.fbar_quantile(p)) for p in phase0.DIAG4_PERCENTILES
      )
      linear_refs = " / ".join(
          _fmt3(result.linear_ref_quantile(p)) for p in phase0.DIAG4_PERCENTILES
      )
      if result.family in saturation_lib.BOUNDED_FAMILIES:
        kappa_value: Any = result.saturation_at_t_ref
      else:
        kappa_value = "n/a (log / power: no ceiling)"
      if result.se_method == "two-way" and result.se_cluster_keys:
        se_label = (
            f"se (cluster-robust, two-way {result.se_cluster_keys[0]} x"
            f" {result.se_cluster_keys[1]})"
        )
      elif result.se_method == "one-way":
        se_label = f"se (cluster-robust, one-way {result.se_cluster_key})"
      else:
        se_label = "se (iid)"
      crit_label = (
          f"critical value crit (k = {result.linear_test_k:g}, "
          + (
              f"t with df = {result.se_df})"
              if result.linear_test_t_critical
              else "normal)"
          )
      )
      se_rows = [(se_label, result.se, "")]
      if result.se_method == "two-way":
        se_rows.append((
            f"se (one-way {result.se_cluster_key}; comparison only)",
            result.se_one_way,
            "",
        ))
      se_rows.append((crit_label, result.critical_value, ""))
      stats_rows = [
          ("family", summary_row.get("family", self.sat_family), ""),
          ("alpha_hat", result.alpha_hat, ""),
          ("alpha_hat (grid argmin)", result.grid_alpha_hat, ""),
          ("Brent refined", result.refined, ""),
          (
              "fallback_half_saturation_T used",
              result.fallback_half_saturation_used,
              "",
          ),
          ("T_ref (fbar reference point)", result.t_ref, ""),
          ("decision", f"{result.decision} ({result.decision_reason})", ""),
          ("d_bar = L_bar(alpha_hat) - L_bar_T", result.d_bar, ""),
          *se_rows,
          (
              "Diag1 fired (eps(J) >= basis_tol or cond(B) >= 1e8)",
              result.diag1_fired,
              "diag1",
          ),
          ("J (T-basis size)", result.n_basis, "diag1"),
          (
              "eps(J) (max over the final grid)",
              result.basis_approx_error,
              "diag1",
          ),
          ("cond(B) (T-basis conditioning)", result.basis_cond, "diag1"),
          ("Diag2 fired (1-SE flatness)", result.diag2_fired, "diag2"),
          (
              "Diag2 ratio (T_1/2 max/min over A_1SE)",
              result.diag2_ratio,
              "diag2",
          ),
          ("r_T = sd(T~)/sd(T/T_ref)", result.r_t, "diag3"),
          ("phi_p(alpha_hat), p = 25/50/75/90", phi_at_hat, "diag4"),
          ("linear ref q_p/T_ref, p = 25/50/75/90", linear_refs, "diag4"),
          ("kappa(alpha_hat) = f(T_ref; alpha_hat)", kappa_value, "diag5"),
          (
              "T_ref unit consistency (diag6 csv)",
              "checked at Phase 1 start",
              "diag6",
          ),
          ("alpha grid points", int(np.size(result.alpha_grid)), ""),
          ("seeds (S)", int(np.size(result.per_seed_alpha)), ""),
          ("k_folds (K)", k_folds, ""),
          ("normalization of T", result.normalization, ""),
          ("divisor c of T/c", divisor_info.short_formula, ""),
          ("ML model Y", result.ml_model_y, ""),
          ("ML model T", result.ml_model_t, ""),
          ("ML fits performed", result.n_ml_fits, ""),
          ("multi-output native", result.multioutput_native, ""),
          ("rows in D_sel", int(len(df_sel)), ""),
      ]
      stats_dict = {name: value for name, value, _ in stats_rows}
      stats_tags = {name: tag for name, _, tag in stats_rows if tag}
      rel_dir = os.path.relpath(out_dir_d, phase0_dir)
      folder = "this folder" if rel_dir == "." else f"the sub-folder {rel_dir}/"
      plots.plot_alpha_profile(
          result.alpha_grid,
          result.mean_loss,
          result.per_seed_loss,
          result.per_seed_alpha,
          result.alpha_hat,
          result.one_se_set,
          os.path.join(phase0_dir, io.alpha_profile_png(d_col)),
          f"[{d_col}] Phase 0 alpha selection ({self.sat_family})",
          family=self.sat_family,
          t_normalized=self._normalized_treatment(d_col, df_sel),
          stats_dict=stats_dict,
          t_ref=result.t_ref,
          t_axis_label=divisor_info.axis_label,
          stats_tags=stats_tags,
          grid_result=result,
          footnote_lines=[
              f"• Files of this treatment ({folder}): "
              + "; ".join(
                  f"{name.format(treatment=d_col)} [{category}] ="
                  f" {description}"
                  for name, category, description in io.FILE_INDEX
              )
              + ". Run-level files: summary_alpha_selection.csv (one row per"
              " treatment) and summary_phase0_diagnostics.json (Stage A split"
              " and every Phase 0 warning)."
          ],
      )

    if alpha_rows:
      pd.DataFrame(alpha_rows).to_csv(
          os.path.join(phase0_dir, io.SUMMARY_CSV), index=False
      )
    self._phase0_diagnostics["alpha_selection"] = alpha_rows
    with open(
        os.path.join(phase0_dir, io.PHASE0_DIAGNOSTICS_JSON), "w"
    ) as fobj:
      json.dump(self._phase0_diagnostics, fobj, indent=4, default=str)

    # ------------------------------------------------------------------
    # `final_fit`: the DML model is FITTED on D_inf (default) or on
    # the full sample.  Either way `df_est` is the FULL raw frame: with
    # `d_inf` the held-out D_sel rows are kept in `_score_frame` and every
    # Phase 1 / Phase 3 repetition scores them with the fitted CATE
    # (`runner.DoubleMLPipeline.run(df_score=...)`), so step1 / step3 / step4
    # tables, charts and the optimizer cover all data points while the
    # estimation itself stays post-selection independent.
    #
    # The treatment columns are still RAW here.  Diag6
    # (unit consistency of T_ref) and the replacement of the treatment by
    # fbar(T/c; alpha_hat, T_ref) happen at the start of Phase 1
    # (`_apply_saturation_transform`).
    # ------------------------------------------------------------------
    if self.final_fit == "d_inf":
      self._score_frame = df_fit.loc[split.idx_sel]
      df_fit = df_fit.loc[split.idx_inf]
    else:
      self._score_frame = None
    df_est = df
    n_scored = 0 if self._score_frame is None else len(self._score_frame)
    if n_scored:
      scored_note = (
          f"; the {n_scored} D_sel rows are scored with the fitted CATE so"
          f" every per-row output covers all {len(df_est)} rows"
          " (column 'estimation_sample' = D_inf / D_sel)."
      )
    else:
      scored_note = "."
    # The completion message of the progress display.
    print(
        "  └─ Phase 0 complete. Phases 1-3 will be estimated on"
        f" {len(df_fit)} rows ({self.final_fit}){scored_note}",
        flush=True,
    )
    return df_fit, df_est

  def _phase0_display_lines(
      self, d_col: str, result: phase0.AlphaSelectionResult
  ) -> List[str]:
    """The per-treatment lines of the Phase 0 progress display.

    Args:
      d_col: Treatment column.
      result: Its Stage B result.

    Returns:
      The display lines, in order: learners, grid, Diag1, alpha estimate,
      Diag2 (+ Diag3 / Diag4 when it fired, Diag5 for an adopted bounded
      family) and the 1-SE decision line.
    """
    lines: List[str] = []
    tag = f"[{d_col}]"

    # Learners.
    def _learner_name(kind: Optional[str], description: Dict[str, Any]) -> str:
      if kind == "xgboost":
        return "XGBoost"
      if kind == "ridge_cv":
        return "RidgeCV"
      cls = str((description or {}).get("class", kind or "unknown"))
      return cls.rsplit(".", 1)[-1]

    y_name = _learner_name(result.ml_model_y, result.learners.get("y_model"))
    t_name = _learner_name(result.ml_model_t, result.learners.get("t_model"))
    relation = "<" if result.n_sel < result.small_sample_threshold else ">="
    lines.append(
        f"{tag} Learners: Y-model = {y_name}, T-model = {t_name}"
        f" (n_sel = {result.n_sel} {relation} small_sample_threshold ="
        f" {result.small_sample_threshold})"
    )

    # Grid settings.
    grid = result.grid
    unit = result.grid_variable or "alpha"
    if grid is not None:
      t_half = np.asarray(grid.t_half, dtype=float)
      if t_half.size and np.all(np.isfinite(t_half)) and np.all(t_half > 0):
        lo, hi = float(t_half.min()), float(t_half.max())
      else:
        alphas = np.asarray(grid.alphas, dtype=float)
        lo, hi = float(alphas.min()), float(alphas.max())
      n_points = int(np.size(grid.alphas))
      mode = grid.mode
    else:
      lo = hi = float("nan")
      n_points = int(np.size(result.alpha_grid))
      mode = "n/a"
    lines.append(
        f"{tag} Grid: mode = {mode}, {unit} range = [{lo:.4g}, {hi:.4g}]"
        f" ({n_points} points), T_ref = {result.t_ref:.4g},"
        f" extensions applied = {result.grid_extensions},"
        f" grid_edge_hit = {result.grid_edge_hit}"
    )

    # Diag1 (one line): eps(J) against basis_tol, cond(B) against
    # MAX_BASIS_CONDITION, and the D_sel rows clipped at T_max.
    eps_cmp = ">=" if result.basis_approx_error >= result.basis_tol else "<"
    cond_cmp = ">=" if result.basis_ill_conditioned else "<"
    lines.append(
        f"{tag} Diag1{' WARNING' if result.diag1_fired else ''}:"
        f" J = {result.n_basis}, eps(J) = {result.basis_approx_error:.2e}"
        f" {eps_cmp} basis_tol = {result.basis_tol:g},"
        f" cond(B) = {result.basis_cond:.1e} {cond_cmp}"
        f" {saturation_lib.MAX_BASIS_CONDITION:.0e},"
        f" T_max = {result.t_max:.4g}"
        f" (q{100 * result.t_max_quantile:g} of T>0;"
        f" {result.n_above_t_max} D_sel rows clipped)"
    )

    # alpha estimate.
    def _grid_value(alpha: float) -> str:
      if result.family == "power":
        return ""
      value = saturation_lib.half_saturation_point(result.family, alpha)
      return f" ({unit} = {float(value):.4g})"

    refine_txt = (
        "Brent-refined alpha_hat"
        if result.refined
        else "alpha_hat (no Brent refinement)"
    )
    lines.append(
        f"{tag} alpha: grid-argmin alpha = {result.grid_alpha_hat:.6g}"
        f"{_grid_value(result.grid_alpha_hat)}, {refine_txt} ="
        f" {result.alpha_hat_estimated:.6g}"
        f"{_grid_value(result.alpha_hat_estimated)},"
        f" L_bar(alpha_hat) = {result.rloss_min:.6g}"
    )

    # Diag2 (always one line), Diag3 / Diag4 (only when Diag2 fired),
    # Diag5 (adopted tanh / exp), then the 1-SE decision line.
    if result.family == "power":
      threshold_txt = f"alpha range threshold = {phase0.POWER_FLATNESS_RANGE:g}"
      stat_txt = f"alpha^max - alpha^min = {result.diag2_ratio:.3g}"
    else:
      threshold_txt = f"flatness_ratio = {result.flatness_ratio:g}"
      stat_txt = f"{unit}^max / {unit}^min = {result.diag2_ratio:.3g}"
    n_in_set = (
        int(np.count_nonzero(result.one_se_set))
        if result.one_se_set is not None
        else 0
    )
    if result.diag2_fired:
      level = "WARNING" if result.selected_family is not None else "INFO"
      lines.append(
          f"{tag} Diag2 {level}: 1-SE flatness fired, {stat_txt} >"
          f" {threshold_txt} (A_1SE: {n_in_set} of"
          f" {int(np.size(result.alpha_grid))} grid points)"
      )
      lines.append(
          f"{tag} Diag3: r_T = sd(T~)/sd(T/T_ref) = {result.r_t:.3f}"
          " (residual treatment variation given X; the cause of a flat"
          " profile)"
      )
      phi = " ".join(
          f"fbar(q{p}) = {result.fbar_quantile(p):.3f}"
          f" (linear ref {result.linear_ref_quantile(p):.3f});"
          for p in phase0.DIAG4_PERCENTILES
      )
      lines.append(f"{tag} Diag4: shape quantiles at alpha_hat: {phi}")
    else:
      lines.append(
          f"{tag} Diag2: 1-SE flatness not fired, {stat_txt} <="
          f" {threshold_txt} (A_1SE: {n_in_set} of"
          f" {int(np.size(result.alpha_grid))} grid points)"
      )
    if (
        result.selected_family is not None
        and result.family in saturation_lib.BOUNDED_FAMILIES
        and result.saturation_at_t_ref is not None
    ):
      lines.append(
          f"{tag} Diag5: saturation level at T_ref kappa(alpha_hat) ="
          f" f(T_ref; alpha_hat) = {result.saturation_at_t_ref:.3f}"
          " (share of the ceiling reached at the q95 spend)"
      )
    if result.se_method == "two-way" and result.se_cluster_keys:
      se_txt = (
          f"se = {result.se:.4g} (2-way {result.se_cluster_keys[0]} x"
          f" {result.se_cluster_keys[1]}; 1-way"
          f" {result.se_cluster_key or 'key'} = {result.se_one_way:.4g})"
      )
    elif result.se_method == "one-way":
      se_txt = (
          f"se = {result.se:.4g} (1-way {result.se_cluster_key or 'key'})"
      )
    else:
      se_txt = f"se = {result.se:.4g} (iid)"
    if result.linear_test_t_critical:
      crit_ref = f"t with df = {result.se_df}"
    else:
      crit_ref = "normal"
    crit_txt = (
        f"crit = {result.critical_value:.3g} (k = {result.linear_test_k:g},"
        f" {crit_ref})"
    )
    decision_line = (
        f"{tag} 1-SE decision: decision = {result.decision}"
        f" (decision_reason = {result.decision_reason}),"
        f" d_bar = {result.d_bar:.4g}, {se_txt}, {crit_txt}"
    )
    # The pure 1-SE outcome with the se that was used (before the grid-edge /
    # conditioning overrides), to compare with the one-way counterfactual.
    one_se_decision = "linear"
    if phase0.linear_test_decision(
        result.d_bar, result.se, result.critical_value
    ):
      one_se_decision = "saturation"
    if (
        result.se_method == "two-way"
        and result.decision_under_one_way_se != one_se_decision
    ):
      decision_line += (
          f" [1-way se would decide: {result.decision_under_one_way_se}]"
      )
    if result.selected_family is None:
      decision_line += (
          " -> selected_family = None: FALLBACK TO LINEAR DML (Phases 1-3"
          " keep the untransformed treatment; Phase 4 uses the existing"
          " optimization)"
      )
    else:
      decision_line += (
          f" -> selected_family = {result.selected_family}, alpha_hat ="
          f" {result.alpha_hat:.6g}"
      )
      if result.fallback_half_saturation_used:
        decision_line += (
            " (fixed by alpha_selection.fallback_half_saturation_T because"
            " Diag1/Diag2 fired)"
        )
    lines.append(decision_line)
    return lines

  def _apply_saturation_transform(self, df_fit: pd.DataFrame) -> pd.DataFrame:
    """Diag6 and the treatment replacement at the start of Phase 1.

    Diag6 (unit consistency of ``T_ref``): per treatment with
    ``selected_family != None``, ``T_ref^(1)`` is recomputed with the Phase 0
    definition ``q95(T | T > 0)`` from the treatment column Phase 1 actually
    receives (the fitted rows plus the scored ``D_sel`` rows, i.e. the full
    data on the Phase 0 scale ``T / c``) and compared with ``T_ref^(0)`` of
    ``summary_alpha_selection.json``.  A relative error ``>=
    t_ref_tolerance`` is an implementation / data-handover bug: both values
    and the column name are printed (marked ERROR) and the run stops.
    Nothing is displayed when the check holds.  Either way the outcome is
    recorded in ``diag6_t_ref_consistency.csv`` (``pass`` / ``fail``;
    ``skipped`` for a treatment with ``selected_family = None``) and in the
    summary JSON (``diag6_status``, ``T_ref_phase1``, ``diag6_rel_error``).

    The treatment is then replaced by the NORMALIZED
    ``fbar(T/c; alpha_hat, T_ref)`` evaluated directly (no basis); alpha_hat
    and T_ref are constants from here on.  No replacement when Phase 0
    selected linear.

    Args:
      df_fit: The frame Phases 1-3 are fitted on (raw treatment).

    Returns:
      ``df_fit`` with the treatment columns transformed; ``self._score_frame``
      is transformed the same way.

    Raises:
      TRefMismatchError: When Diag6 fails for a treatment.
    """
    if self.sat_family is None or not self._alpha_results:
      return df_fit
    tolerance = float(
        self.alpha_selection_cfg.get(
            "t_ref_tolerance", phase0.DEFAULT_T_REF_TOLERANCE
        )
    )
    quantile = float(
        self.saturation_cfg.get(
            "t_ref_quantile", saturation_lib.DEFAULT_T_REF_QUANTILE
        )
    )
    frames = [df_fit]
    if self._score_frame is not None:
      frames.append(self._score_frame)
    received = pd.concat(frames, axis=0)
    df_fit = df_fit.copy()
    score_frame = (
        None if self._score_frame is None else self._score_frame.copy()
    )

    def _record_diag6(d_col: str, row: Dict[str, Any]) -> None:
      """Writes the Diag6 record of ``d_col`` (CSV + summary JSON keys)."""
      out_dir = self._phase0_out_dirs.get(d_col)
      if not out_dir:
        return
      saturation_evaluation.write_diag6_output(out_dir, [row])
      saturation_evaluation.update_summary(
          out_dir,
          {
              "diag6_status": row["status"],
              "T_ref_phase1": row["T_ref_phase1"],
              "diag6_rel_error": row["rel_error"],
              "t_ref_tolerance": row["t_ref_tolerance"],
          },
      )

    for d_col, result in self._alpha_results.items():
      # No replacement (and no Diag6) when linear was selected.
      if result.selected_family is None:
        _record_diag6(
            d_col,
            {
                "treatment": d_col,
                "status": "skipped",
                "T_ref_phase0": result.t_ref,
                "T_ref_phase1": None,
                "rel_error": None,
                "t_ref_tolerance": tolerance,
                "t_ref_quantile": quantile,
                "n_rows_phase1": int(len(received)),
                "note": (
                    "selected_family = None: the treatment is not transformed,"
                    " so the unit check is not needed"
                ),
            },
        )
        continue
      t_ref_0 = self._t_ref_for(d_col)
      holds, t_ref_1, rel_err = phase0.t_ref_consistency(
          t_ref_0,
          self._normalized_treatment(d_col, received),
          quantile=quantile,
          tolerance=tolerance,
      )
      if holds:
        note = (
            "T_ref recomputed from the column Phase 1 received matches Phase 0"
        )
      else:
        note = "MISMATCH: unit / data-handover bug; the run was stopped"
      _record_diag6(
          d_col,
          {
              "treatment": d_col,
              "status": "pass" if holds else "fail",
              "T_ref_phase0": t_ref_0,
              "T_ref_phase1": t_ref_1,
              "rel_error": rel_err,
              "t_ref_tolerance": tolerance,
              "t_ref_quantile": quantile,
              "n_rows_phase1": int(len(received)),
              "note": note,
          },
      )
      if not holds:
        message = (
            "Diag6 (unit consistency of T_ref) FAILED for treatment"
            f" '{d_col}': T_ref^(0) = {t_ref_0:.10g} (Phase 0,"
            f" {saturation_evaluation.SUMMARY_JSON}) vs T_ref^(1) ="
            f" {t_ref_1:.10g} (recomputed as q"
            f"{int(round(100 * quantile))}(T | T > 0) from column '{d_col}'"
            f" as received by Phase 1); relative error {rel_err:.3e} >="
            f" t_ref_tolerance = {tolerance:g}. This is a data-handover /"
            " unit mismatch, not statistical uncertainty: the alpha_hat"
            " selected in Phase 0 would be applied as a different curve."
            " Stopping."
        )
        print(f"  ├─ [ERROR] {message}", flush=True)
        raise TRefMismatchError(message)
      for frame in (df_fit, score_frame):
        if frame is None:
          continue
        frame[d_col] = saturation_lib.normalized_saturation(
            self._normalized_treatment(d_col, frame),
            result.selected_family,
            result.alpha_hat,
            t_ref_0,
        )
    self._score_frame = score_frame
    return df_fit

  def _parallel_plan(self, n_tasks: int) -> tuple[int, int]:
    """The parallel plan ``(W, h)`` for ``n_tasks`` tasks."""
    return compute_workers_threads(n_tasks, resolve_n_cores(self.n_jobs))

  def _hparam_context(self, d_col: str, n_rows: int) -> Dict[str, Any]:
    """Traceability fields of ``hparams_<d_col>_<learner>.json``."""
    result = self._alpha_results.get(d_col)
    selected_family = (
        self._selected_family.get(d_col)
        if self.sat_family is not None
        else None
    )
    t_ref: Optional[float] = None
    if selected_family is not None:
      try:
        t_ref = self._t_ref_for(d_col)
      except RuntimeError:
        t_ref = None
    return {
        "selected_family": selected_family,
        "alpha_hat": (
            self._alpha_hat.get(d_col) if selected_family is not None else None
        ),
        "T_ref": t_ref,
        "decision": None if result is None else result.decision,
        "final_fit": self.final_fit if self.sat_family is not None else None,
        "treatment_variable": (
            sensitivity.TREATMENT_VARIABLE_FBAR
            if selected_family is not None
            else sensitivity.TREATMENT_VARIABLE_T
        ),
        "n_rows": int(n_rows),
        "time_budget": self.time_budget,
        "eval_method_requested": hparam_search.SEARCH_EVAL_METHOD,
        "early_stop": hparam_search.SEARCH_EARLY_STOP,
    }

  def _run_global_hparam_search(
      self,
      d_col: str,
      x_cols: List[str],
      df_norm: pd.DataFrame,
      out_dir: str,
  ) -> Dict[tuple[str, str], hparam_search.HparamSearchResult]:
    """One FLAML search per (treatment, side, learner) (global policy).

    Runs ``|Y| + |T|`` searches in parallel on the scaled Phases 1-3 sample
    ``df_norm`` with target ``Y`` for the Y side and the ``D``
    column of this run (``fbar(T; alpha_hat)`` or ``T``) for the
    T side, persists ``hparams_<d_col>_<learner>.json`` and prints one log
    line per (treatment, learner).

    Args:
      d_col: Treatment column.
      x_cols: Nuisance covariates ``X``.
      df_norm: The scaled fitting frame of this treatment .
      out_dir: ``step1_exploration_<d_col>`` (JSON destination).

    Returns:
      ``{(side, learner): HparamSearchResult}``.
    """
    specs = [(hparam_search.SIDE_Y, m) for m in self.ml_models_y] + [
        (hparam_search.SIDE_T, m) for m in self.ml_models_t
    ]
    x_values = df_norm[x_cols].to_numpy(dtype=float)
    targets = {
        hparam_search.SIDE_Y: df_norm[self.y_col].to_numpy(dtype=float),
        hparam_search.SIDE_T: df_norm[d_col].to_numpy(dtype=float),
    }
    target_cols = {
        hparam_search.SIDE_Y: self.y_col,
        hparam_search.SIDE_T: d_col,
    }
    workers, threads = self._parallel_plan(len(specs))
    print(
        f"  ├─ Global hyperparameter search [{d_col}]: {len(specs)} FLAML"
        f" searches (|Y| = {len(self.ml_models_y)}, |T| ="
        f" {len(self.ml_models_t)}) on {len(df_norm)} rows, time_budget ="
        f" {self.time_budget}s, eval_method ="
        f" {hparam_search.SEARCH_EVAL_METHOD}, early_stop ="
        f" {hparam_search.SEARCH_EARLY_STOP}; workers = {workers},"
        f" threads/worker = {threads}",
        flush=True,
    )
    with parallel_section(workers, threads):
      results = joblib.Parallel(n_jobs=workers, verbose=0)(
          joblib.delayed(hparam_search.run_search)(
              d_col=d_col,
              side=side,
              learner=learner,
              target_col=target_cols[side],
              x=x_values,
              y=targets[side],
              time_budget=self.time_budget,
              n_jobs=threads,
              seed=hparam_search.SEARCH_SEED,
          )
          for side, learner in specs
      )

    context = self._hparam_context(d_col, len(df_norm))
    paths: Dict[str, str] = {}
    for learner, items in hparam_search.group_by_learner(results).items():
      path = os.path.join(
          out_dir, hparam_search.hparams_json_name(d_col, learner)
      )
      hparam_search.write_hparams_json(path, items, context)
      paths[learner] = path
    for result in results:
      # One log line per (treatment, learner).
      print(
          f"  │   [{d_col}] {result.side}-side learner '{result.learner}'"
          f" (target = {result.target_col}): search stopped after"
          f" {result.elapsed_seconds:.1f}s, trials = {result.n_trials}"
          f" (iterations = {result.n_iterations}), eval_method ="
          f" {result.eval_method}, best_loss = {result.best_loss:.6g},"
          f" estimator = {result.estimator_cls.__name__}, saved ->"
          f" {paths[result.learner]}",
          flush=True,
      )
    return {(r.side, r.learner): r for r in results}

  def _run_single_exploration(
      self,
      ml_y: str,
      ml_t: str,
      n_folds: int,
      d_col: str,
      x_cols: List[str],
      cate_cols: List[str],
      cate_structure: Any,
      gt_col: str,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      process_log: str = "lightweight",
      df_score: Optional[pd.DataFrame] = None,
      row_order: Optional[pd.Index] = None,
      prepared: Optional[runner.PreparedInputs] = None,
      hparams_y: Optional[hparam_search.HparamSearchResult] = None,
      hparams_t: Optional[hparam_search.HparamSearchResult] = None,
      learner_n_jobs: Optional[int] = None,
  ) -> Dict[str, Any]:
    """Executes a single estimation run across cross-fitting folds.

    Args:
      ml_y: FLAML learner name for the outcome nuisance.
      ml_t: FLAML learner name for the treatment nuisance.
      n_folds: Number of cross-fitting folds.
      d_col: Treatment column.
      x_cols: Nuisance covariates.
      cate_cols: CATE covariates.
      cate_structure: CATE structure of this channel.
      gt_col: Ground-truth effect column (may be empty).
      df: The rows the model is fitted on.
      scaler: The pipeline scaler.
      process_log: Output verbosity of `io.save_step1_outputs`.
      df_score: Held-out rows (Phase 0 D_sel) scored with the fitted CATE and
        appended to `df_1a`; None when there is no split.
      row_order: Row order of the full sample for the returned `df_1a`.
      prepared: the scaled frames / CATE basis of this treatment.
      hparams_y: global search result of ``ml_y`` (None = per_fold).
      hparams_t: global search result of ``ml_t`` (None = per_fold).
      learner_n_jobs: ``h`` threads per learner fit.

    Returns:
      The FULL run result (``model_name``, ``df_1a``, ``metrics_1c``,
      ``model_obj``, ``cate_coef_dict``); Phase 3 reuses it as
      repetition 0 of the candidate (same seed ``random_state = 1`` and the
      same splits as that repetition).
    """
    # FIX: Extract valid cluster columns for cluster robust inference
    # The keys follow the cluster-key determination rule.
    cluster_cols = self._resolve_cluster_cols(df)

    pipeline = runner.DoubleMLPipeline(
        self.y_col,
        d_col,
        x_cols,
        ml_y,
        ml_t,
        n_folds=n_folds,
        time_budget=self.time_budget,
        cate_structure=cate_structure,
        covariates_for_cate=cate_cols,
        ground_truth_col=gt_col,
        cluster_cols=cluster_cols,
        random_state=1,
        n_jobs=learner_n_jobs,
        hparams_y=hparams_y,
        hparams_t=hparams_t,
    )
    res = pipeline.run(
        df,
        scaler,
        rep_id=1,
        df_score=df_score,
        row_order=row_order,
        prepared=prepared,
    )
    # The derived date-block key is internal; keep existing outputs
    # byte-compatible by dropping it from the result frame.
    res["df_1a"] = self._strip_helper_columns(res["df_1a"])
    df_1b = metrics.aggregate_geo_item_metrics(
        res["df_1a"],
        res["model_name"],
        d_col,
        self.y_col,
        geo_col=self.geo_col,
        item_col=self.item_col,
        geo_name_col=self.geo_name_col,
        item_name_col=self.item_name_col,
        gt_col=gt_col,
    )

    run_metrics = res["metrics_1c"]
    meta_dict = {
        "d_col": d_col,
        "x_col": str(x_cols),
        "y_col": self.y_col,
        "covariates_for_cate": str(cate_cols),
        "cate_structure": run_metrics.get("cate_structure", ""),
        "cate_df": run_metrics.get("cate_df", ""),
        "cate_degree": run_metrics.get("cate_degree", ""),
        "cate_intercept": run_metrics.get("cate_intercept", ""),
    }
    df_1b = self._insert_metadata_columns(df_1b, meta_dict)

    out_dir = os.path.join(self.output_dir, f"step1_exploration_{d_col}")
    io.save_step1_outputs(
        res["df_1a"],
        df_1b,
        pd.DataFrame([res["metrics_1c"]]),
        res["model_name"],
        out_dir,
        process_log,
    )
    return res

  def _run_single_iteration(
      self,
      row: pd.Series,
      d_col: str,
      x_cols: List[str],
      cate_cols: List[str],
      cate_structure: Any,
      gt_col: str,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      rep: int,
      temp_dir: str,
      df_score: Optional[pd.DataFrame] = None,
      row_order: Optional[pd.Index] = None,
      prepared: Optional[runner.PreparedInputs] = None,
      hparams_y: Optional[hparam_search.HparamSearchResult] = None,
      hparams_t: Optional[hparam_search.HparamSearchResult] = None,
      learner_n_jobs: Optional[int] = None,
      reduced: bool = True,
      done_name: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Spins up iteration tasks inside Phase 3's robust ensemble structure.

    Args:
      row: The shortlisted model row (Phase 2).
      d_col: Treatment column.
      x_cols: Nuisance covariates.
      cate_cols: CATE covariates.
      cate_structure: CATE structure of this channel.
      gt_col: Ground-truth effect column (may be empty).
      df: The rows the model is fitted on.
      scaler: The pipeline scaler.
      rep: Repetition index.
      temp_dir: Directory of the `.done` progress markers.
      df_score: Held-out rows (Phase 0 D_sel) scored with the fitted CATE and
        appended to `df_1a`; None when there is no split.
      row_order: Row order of the full sample for the returned `df_1a` (all
        repetitions must share it: the median-of-repetitions aggregation is
        positional).
      prepared: the scaled frames / CATE basis of this treatment.
      hparams_y: global search result of the Y learner (None =
        per_fold).
      hparams_t: global search result of the T learner.
      learner_n_jobs: ``h`` threads per learner fit.
      reduced: Return only ``model_name``, ``model_group``,
        ``metrics_1c``, ``cate_coef_dict`` and the KPI / treatment /
        ground-truth columns of ``df_1a`` (repetitions >= 1).  False keeps
        the full objects.
      done_name: Name of the ``.done`` marker (``rep_<rep>.done`` when None).

    Returns:
      The (reduced) runner result dictionary plus `model_group`.
    """
    # FIX: Extract valid cluster columns for cluster robust inference
    # The keys follow the cluster-key determination rule.
    cluster_cols = self._resolve_cluster_cols(df)

    pipeline = runner.DoubleMLPipeline(
        self.y_col,
        d_col,
        x_cols,
        row["ML_model_Y"],
        row["ML_model_T"],
        n_folds=row["n_folds"],
        time_budget=self.time_budget,
        cate_structure=cate_structure,
        covariates_for_cate=cate_cols,
        ground_truth_col=gt_col,
        cluster_cols=cluster_cols,
        random_state=rep + 1,
        n_jobs=learner_n_jobs,
        hparams_y=hparams_y,
        hparams_t=hparams_t,
    )
    res = pipeline.run(
        df,
        scaler,
        rep_id=rep + 1,
        df_score=df_score,
        row_order=row_order,
        prepared=prepared,
    )
    # The derived date-block key is internal (see Phase 1).
    res["df_1a"] = self._strip_helper_columns(res["df_1a"])
    res["model_group"] = row["model"]
    if reduced:
      res = reduce_rep_result(res, d_col, gt_col)
    os.makedirs(temp_dir, exist_ok=True)
    marker = done_name or f"rep_{rep}.done"
    with open(os.path.join(temp_dir, marker), "w") as f_obj:
      f_obj.write("")
    return res

  def format_cate_equation(self, coef_dict: Dict[str, float]) -> str:
    """Turns a CATE coefficient dict into a readable equation string."""
    eq_parts = []
    for name, val in coef_dict.items():
      if not str(name).endswith("_coef"):
        continue
      if abs(val) < 1e-10:
        continue

      clean_name = str(name)[:-5]
      clean_name = re.sub(
          r"bs\(([^,]+)[^\)]*\)(?:\[(\d+)\])?", r"Spline_\2(\1)", clean_name
      )
      clean_name = (
          clean_name.replace("[T.", "_").replace("]", "").replace(":", " * ")
      )

      if abs(val) < 0.0001:
        val_str = f"{val:+.2e}"
      else:
        val_str = f"{val:+.4f}"

      if clean_name == "Intercept":
        eq_parts.append(val_str)
      else:
        eq_parts.append(f"{val_str} * {clean_name}")

    if not eq_parts:
      return "Constant Effect"
    res = " ".join(eq_parts).strip()
    if res.startswith("+"):
      res = res[1:].strip()
    return res

  def extract_priors(
      self,
      df_1a_consolidated: pd.DataFrame,
      treatment_base_name: str,
      prior_type: str = "roi",
      distribution_type: str = "Normal",
  ) -> Dict[str, float]:
    """Generates prior distribution parameters (mu / sigma or alpha / beta)."""
    if self.date_col and self.date_col in df_1a_consolidated.columns:
      ts_df = df_1a_consolidated.groupby(self.date_col, observed=False).sum(
          numeric_only=True
      )
    else:
      ts_df = df_1a_consolidated

    group_cols = [
        c
        for c in self.d_cols
        if re.sub(r"_l\d+$", "", c) == treatment_base_name
    ]
    if not group_cols:
      return {}

    kpi_cols = [f"estimated_incremental_KPI_{c}" for c in group_cols]
    total_inc_kpi = ts_df[kpi_cols].sum(axis=1)
    input_col = (
        ts_df[treatment_base_name]
        if treatment_base_name in ts_df.columns
        else pd.Series(np.zeros(len(ts_df)), index=ts_df.index)
    )

    if prior_type == "roi":
      mask = input_col > 0
      data = (total_inc_kpi.loc[mask] / input_col.loc[mask]).values
    else:
      mask = (input_col > 0) & (ts_df[self.y_col] > 0)
      data = (total_inc_kpi.loc[mask] / ts_df.loc[mask, self.y_col]).values

    data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values
    if len(data) == 0:
      return {}

    if distribution_type == "Normal":
      mu_val, std = stats.norm.fit(data)
      return {"mu": mu_val, "sigma": std}
    elif distribution_type == "LogNormal":
      valid_data = data[data > 0]
      if len(valid_data) == 0:
        return {}
      shape, _, scale = stats.lognorm.fit(valid_data, floc=0)
      return {"mu": np.log(scale), "sigma": shape}
    elif distribution_type == "Beta":
      valid_data = data[(data > 0) & (data < 1)]
      if len(valid_data) == 0:
        return {}
      a_val, b_val, _, _ = stats.beta.fit(valid_data, floc=0, fscale=1)
      return {"alpha": a_val, "beta": b_val}
    return {}

  def _generate_dim_summary(
      self,
      dim_col: str,
      df_1a_consolidated: pd.DataFrame,
      best_model_data_dict: Dict[str, Any],
      df_3b_concat: pd.DataFrame,
  ) -> Optional[pd.DataFrame]:
    """Consolidates metric outputs per dimension group (geo, item)."""
    if not dim_col or dim_col not in df_1a_consolidated.columns:
      return None
    rows = []
    for dim_val in df_1a_consolidated[dim_col].dropna().unique():
      dim_df = df_1a_consolidated[df_1a_consolidated[dim_col] == dim_val]
      sum_input_total = 0
      sum_kpi_total = 0

      y_true_ts_tot = np.zeros(len(dim_df))
      y_pred_ts_tot = np.zeros(len(dim_df))

      for i, d_col in enumerate(self.d_cols):
        m_data = best_model_data_dict[d_col]["metrics_1c"]
        kpi_col = f"estimated_incremental_KPI_{d_col}"

        t_input = dim_df[d_col].sum()
        t_kpi = dim_df[kpi_col].fillna(0).sum()
        m_group = _model_group_label(m_data)

        channel_summary = df_3b_concat[df_3b_concat["Channel"] == d_col]
        if not channel_summary.empty:
          c_err_calc_med = channel_summary[
              "combined_error_calculated_by_median_rmse"
          ].iloc[0]
          c_err_mean = channel_summary["combined_error_mean"].iloc[0]
          c_err_sd = channel_summary["combined_error_sd"].iloc[0]
          norm_y_med = channel_summary["norm_median_Y_rmse"].iloc[0]
          norm_y_mean = channel_summary["norm_Y_rmse_mean"].iloc[0]
          norm_y_sd = channel_summary["norm_Y_rmse_sd"].iloc[0]
          norm_t_med = channel_summary["norm_median_T_rmse"].iloc[0]
          norm_t_mean = channel_summary["norm_T_rmse_mean"].iloc[0]
          norm_t_sd = channel_summary["norm_T_rmse_sd"].iloc[0]
        else:
          c_err_calc_med = c_err_mean = c_err_sd = np.nan
          norm_y_med = norm_y_mean = norm_y_sd = np.nan
          norm_t_med = norm_t_mean = norm_t_sd = np.nan

        row = {
            dim_col: dim_val,
            "Channel": d_col,
            "Model Group": m_group,
            "x_col": m_data.get("x_col"),
            "y_col": m_data.get("y_col"),
            "covariates_for_cate": m_data.get("covariates_for_cate"),
            "cate_structure": m_data.get("cate_structure"),
            "cate_df": m_data.get("cate_df"),
            "cate_degree": m_data.get("cate_degree"),
            "cate_intercept": m_data.get("cate_intercept"),
            "total_input": t_input,
            "total_estimated_incremental_KPI": t_kpi,
            "total_estimated_roi": t_kpi / t_input if t_input > 0 else 0,
            "combined_error_calculated_by_median_rmse": c_err_calc_med,
            "combined_error_mean": c_err_mean,
            "combined_error_sd": c_err_sd,
            "norm_median_Y_rmse": norm_y_med,
            "norm_Y_rmse_mean": norm_y_mean,
            "norm_Y_rmse_sd": norm_y_sd,
            "norm_median_T_rmse": norm_t_med,
            "norm_T_rmse_mean": norm_t_mean,
            "norm_T_rmse_sd": norm_t_sd,
            "best_model": 1,
        }

        if self.ground_truth_existence:
          gt_col = self.ground_truth_effect_column[i]
          y_true_ts = (
              dim_df[gt_col].fillna(0).values
              if gt_col in dim_df.columns
              else np.zeros(len(dim_df))
          )
          y_pred_ts = dim_df[kpi_col].fillna(0).values
          row.update(
              metrics.calculate_ground_truth_metrics(
                  pd.Series(y_true_ts), pd.Series(y_pred_ts), t_input
              )
          )
          y_true_ts_tot += y_true_ts
          y_pred_ts_tot += y_pred_ts
        else:
          row.update({
              "true_total_roi": np.nan,
              "SMAPE": np.nan,
              "R_squared": np.nan,
              "MAE": np.nan,
              "RMSE": np.nan,
          })

        rows.append(row)
        if not re.search(r"_l\d+$", d_col):
          sum_input_total += t_input
        sum_kpi_total += t_kpi

      tot_row = {
          dim_col: dim_val,
          "Channel": "Total",
          "Model Group": "-",
          "x_col": "-",
          "y_col": self.y_col,
          "covariates_for_cate": "-",
          "cate_structure": "-",
          "cate_df": np.nan,
          "cate_degree": np.nan,
          "cate_intercept": np.nan,
          "total_input": sum_input_total,
          "total_estimated_incremental_KPI": sum_kpi_total,
          "total_estimated_roi": (
              sum_kpi_total / sum_input_total if sum_input_total > 0 else 0
          ),
          "combined_error_calculated_by_median_rmse": np.nan,
          "combined_error_mean": np.nan,
          "combined_error_sd": np.nan,
          "norm_median_Y_rmse": np.nan,
          "norm_Y_rmse_mean": np.nan,
          "norm_Y_rmse_sd": np.nan,
          "norm_median_T_rmse": np.nan,
          "norm_T_rmse_mean": np.nan,
          "norm_T_rmse_sd": np.nan,
          "best_model": 1,
      }
      if self.ground_truth_existence:
        tot_row.update(
            metrics.calculate_ground_truth_metrics(
                pd.Series(y_true_ts_tot),
                pd.Series(y_pred_ts_tot),
                sum_input_total,
            )
        )
      else:
        tot_row.update({
            "true_total_roi": np.nan,
            "SMAPE": np.nan,
            "R_squared": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        })

      rows.append(tot_row)

    # ---------------------------------------------------------
    # Overall "Total" Dimension Block (e.g., Geo = "Total")
    # ---------------------------------------------------------
    dim_val_total = "Total"
    sum_input_total_all = 0
    sum_kpi_total_all = 0

    use_date_agg = (
        self.ground_truth_existence
        and self.date_col
        and self.date_col in df_1a_consolidated.columns
    )

    df_date_agg = None
    y_true_ts_tot_all = None
    y_pred_ts_tot_all = None

    if use_date_agg:
      df_date_agg = df_1a_consolidated.groupby(
          self.date_col, as_index=False, observed=False
      ).sum()
      y_true_ts_tot_all = np.zeros(len(df_date_agg))
      y_pred_ts_tot_all = np.zeros(len(df_date_agg))
    elif self.ground_truth_existence:
      y_true_ts_tot_all = np.zeros(len(df_1a_consolidated))
      y_pred_ts_tot_all = np.zeros(len(df_1a_consolidated))

    for i, d_col in enumerate(self.d_cols):
      m_data = best_model_data_dict[d_col]["metrics_1c"]
      kpi_col = f"estimated_incremental_KPI_{d_col}"

      t_input = df_1a_consolidated[d_col].sum()
      t_kpi = df_1a_consolidated[kpi_col].fillna(0).sum()
      m_group = _model_group_label(m_data)

      channel_summary = df_3b_concat[df_3b_concat["Channel"] == d_col]
      if not channel_summary.empty:
        c_err_calc_med = channel_summary[
            "combined_error_calculated_by_median_rmse"
        ].iloc[0]
        c_err_mean = channel_summary["combined_error_mean"].iloc[0]
        c_err_sd = channel_summary["combined_error_sd"].iloc[0]
        norm_y_med = channel_summary["norm_median_Y_rmse"].iloc[0]
        norm_y_mean = channel_summary["norm_Y_rmse_mean"].iloc[0]
        norm_y_sd = channel_summary["norm_Y_rmse_sd"].iloc[0]
        norm_t_med = channel_summary["norm_median_T_rmse"].iloc[0]
        norm_t_mean = channel_summary["norm_T_rmse_mean"].iloc[0]
        norm_t_sd = channel_summary["norm_T_rmse_sd"].iloc[0]
      else:
        c_err_calc_med = c_err_mean = c_err_sd = np.nan
        norm_y_med = norm_y_mean = norm_y_sd = np.nan
        norm_t_med = norm_t_mean = norm_t_sd = np.nan

      row = {
          dim_col: dim_val_total,
          "Channel": d_col,
          "Model Group": m_group,
          "x_col": m_data.get("x_col"),
          "y_col": m_data.get("y_col"),
          "covariates_for_cate": m_data.get("covariates_for_cate"),
          "cate_structure": m_data.get("cate_structure"),
          "cate_df": m_data.get("cate_df"),
          "cate_degree": m_data.get("cate_degree"),
          "cate_intercept": m_data.get("cate_intercept"),
          "total_input": t_input,
          "total_estimated_incremental_KPI": t_kpi,
          "total_estimated_roi": t_kpi / t_input if t_input > 0 else 0,
          "combined_error_calculated_by_median_rmse": c_err_calc_med,
          "combined_error_mean": c_err_mean,
          "combined_error_sd": c_err_sd,
          "norm_median_Y_rmse": norm_y_med,
          "norm_Y_rmse_mean": norm_y_mean,
          "norm_Y_rmse_sd": norm_y_sd,
          "norm_median_T_rmse": norm_t_med,
          "norm_T_rmse_mean": norm_t_mean,
          "norm_T_rmse_sd": norm_t_sd,
          "best_model": 1,
      }

      if self.ground_truth_existence:
        gt_col = self.ground_truth_effect_column[i]
        if use_date_agg and df_date_agg is not None:
          y_true_ts = (
              df_date_agg[gt_col].fillna(0).values
              if gt_col in df_date_agg.columns
              else np.zeros(len(df_date_agg))
          )
          y_pred_ts = (
              df_date_agg[kpi_col].fillna(0).values
              if kpi_col in df_date_agg.columns
              else np.zeros(len(df_date_agg))
          )
        else:
          y_true_ts = (
              df_1a_consolidated[gt_col].fillna(0).values
              if gt_col in df_1a_consolidated.columns
              else np.zeros(len(df_1a_consolidated))
          )
          y_pred_ts = df_1a_consolidated[kpi_col].fillna(0).values

        row.update(
            metrics.calculate_ground_truth_metrics(
                pd.Series(y_true_ts), pd.Series(y_pred_ts), t_input
            )
        )
        y_true_ts_tot_all += y_true_ts
        y_pred_ts_tot_all += y_pred_ts
      else:
        row.update({
            "true_total_roi": np.nan,
            "SMAPE": np.nan,
            "R_squared": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        })

      rows.append(row)
      if not re.search(r"_l\d+$", d_col):
        sum_input_total_all += t_input
      sum_kpi_total_all += t_kpi

    tot_row_all = {
        dim_col: dim_val_total,
        "Channel": "Total",
        "Model Group": "-",
        "x_col": "-",
        "y_col": self.y_col,
        "covariates_for_cate": "-",
        "cate_structure": "-",
        "cate_df": np.nan,
        "cate_degree": np.nan,
        "cate_intercept": np.nan,
        "total_input": sum_input_total_all,
        "total_estimated_incremental_KPI": sum_kpi_total_all,
        "total_estimated_roi": (
            sum_kpi_total_all / sum_input_total_all
            if sum_input_total_all > 0
            else 0
        ),
        "combined_error_calculated_by_median_rmse": np.nan,
        "combined_error_mean": np.nan,
        "combined_error_sd": np.nan,
        "norm_median_Y_rmse": np.nan,
        "norm_Y_rmse_mean": np.nan,
        "norm_Y_rmse_sd": np.nan,
        "norm_median_T_rmse": np.nan,
        "norm_T_rmse_mean": np.nan,
        "norm_T_rmse_sd": np.nan,
        "best_model": 1,
    }

    if self.ground_truth_existence:
      tot_row_all.update(
          metrics.calculate_ground_truth_metrics(
              pd.Series(y_true_ts_tot_all),
              pd.Series(y_pred_ts_tot_all),
              sum_input_total_all,
          )
      )
    else:
      tot_row_all.update({
          "true_total_roi": np.nan,
          "SMAPE": np.nan,
          "R_squared": np.nan,
          "MAE": np.nan,
          "RMSE": np.nan,
      })

    rows.append(tot_row_all)
    return pd.DataFrame(rows)

  def treatment_lag_covariates(self) -> Dict[str, List[str]]:
    """Lags of a treatment used as covariates, per treatment.

    ``preprocessing.features.create_lagged_features`` names the lag ``p`` of
    column ``c`` ``f"{c}_l{p}"``; a column of that form built from a
    treatment column is a function of PAST treatment.

    Returns:
      ``{d_col: [lag columns found in x_cols or covariates_for_cate]}``,
      only for treatments with at least one such column.
    """
    found: Dict[str, List[str]] = {}
    for idx, d_col in enumerate(self.d_cols):
      pattern = re.compile(rf"^{re.escape(str(d_col))}_l\d+$")
      candidates = list(self.x_cols_list[idx]) + list(
          self.covariates_for_cate_list[idx] or []
      )
      lags = sorted({c for c in candidates if pattern.match(str(c))})
      if lags:
        found[d_col] = lags
    return found

  def _warn_treatment_lag_covariates(self) -> None:
    """Warns, at the start of Phase 4, about treatment lags among X.

    The saturation-aware optimizer treats every geo x item x date row as an
    allocation unit whose response depends on that row's spend only (no
    carryover).  When a lag of the treatment is a covariate, ``X_t`` depends
    on past ``T`` and the row-wise separability behind the KKT allocation no
    longer holds: moving spend on day ``t`` also changes the covariates --
    hence the CATE -- of later days, which the optimizer ignores.  Only the
    saturation-aware path is affected (the simple optimizer scales rows
    independently as well, but makes no claim of optimality).
    """
    if not (
        self.optimize
        and not self.simple_optimization
        and self.sat_family is not None
    ):
      return
    found = self.treatment_lag_covariates()
    if not found:
      return
    detail = "; ".join(f"{d}: {cols}" for d, cols in found.items())
    message = (
        "[Phase 4] The saturation-aware optimizer assumes no carryover (the"
        " response of a row depends on that row's spend only), but lagged"
        f" treatment columns are used as covariates ({detail}). X_t then"
        " depends on past T, so reallocating spend on one day also changes"
        " the covariates of later days, which the per-row KKT allocation"
        " ignores. Interpret the optimized allocation at the geo x item x"
        " period level, consider optimization.reallocate_within = 'date', or"
        " drop the treatment lags from x_cols / covariates_for_cate."
    )
    print(f"  ├─ [Warning] {message}", flush=True)
    warnings.warn(message, UserWarning, stacklevel=2)

  def run_full_pipeline(
      self,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      top_n: int = 5,
      n_reps: int = 20,
      process_log: str = "lightweight",
  ) -> None:
    """Orchestrates and executes the complete DoubleML modeling sequence."""
    # Rows are tracked by index label throughout (D_sel / D_inf selection,
    # divisor and cell-id alignment, Phase 4 raw-treatment restoration), so
    # the labels must be unique. Any order or dtype is fine.
    if not df.index.is_unique:
      dup_labels = df.index[df.index.duplicated()].unique()
      raise ValueError(
          f"The input DataFrame index has {len(dup_labels)} duplicated"
          f" label(s) (e.g. {list(dup_labels[:5])}). Rows are matched by index"
          " label throughout the pipeline, so the index must be unique. Call"
          " `df = df.reset_index(drop=True)` (or set a unique index) before"
          " run_full_pipeline."
      )
    self._validate_columns(df)
    self._validate_treatments(df)
    actual_cores = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
    print(
        "\n======================================================================"
    )
    print(
        "Model Naming Convention"
        " Note:\n[Y-model]_[T-model]_[number_of_folds]_[number_of_repetitions]_[time_budget]_[number_of_covariates_for_cate]."
    )
    print(
        "======================================================================\n",
        flush=True,
    )
    print(
        f"★★★ Starting DML Pipeline Execution on {actual_cores} Parallel"
        " Cores ★★★"
    )

    # ------------------------------------------------------------------
    # Phase 0. Skipped entirely when family=None; only the
    # cluster-key determination rule is evaluated in that case.
    # `df_fit` is what Phases 1-3 are FITTED on; `df_est` is the FULL raw
    # frame that drives Phase 4.  With a Phase 0 `d_inf` split the held-out
    # D_sel rows (`self._score_frame`) are scored with the fitted CATE in
    # every run so all per-row outputs cover `df_est`.  The caller's `df` is
    # never modified.
    # ------------------------------------------------------------------
    df_fit, df_est = self._run_phase0(df, scaler)
    # Diag6 (unit consistency of T_ref) per treatment at the
    # start of Phase 1 -- error stop on a mismatch, silent otherwise -- then
    # the treatment replacement by fbar(T/c; alpha_hat, T_ref).
    df_fit = self._apply_saturation_transform(df_fit)
    df_score = self._score_frame
    row_order = df_est.index

    best_model_data_dict: Dict[str, Dict[str, Any]] = {}
    # Exposed as `self.best_model_data` (reset on every run).
    self.best_model_data = best_model_data_dict
    all_best_models_3b = []
    all_3a_details = []
    all_cate_pool_rows = []

    for idx, d_col in enumerate(self.d_cols):
      x_cols = self.x_cols_list[idx]
      cate_cols = self.covariates_for_cate_list[idx]
      cate_structure = self.cate_structure_list[idx]
      gt_col = (
          self.ground_truth_effect_column[idx]
          if self.ground_truth_existence
          else None
      )

      kpi_col = f"estimated_incremental_KPI_{d_col}"
      kpi_lower = f"estimated_incremental_KPI_{d_col}_2.5%"
      kpi_upper = f"estimated_incremental_KPI_{d_col}_97.5%"

      total_p1_tasks = (
          len(self.ml_models_y) * len(self.ml_models_t) * len(self.n_folds_list)
      )
      print(f"\n>>> Executing Channel Workflow: [{d_col}] <<<")
      print(
          f"=== Phase 1: Grid Search Exploration ({total_p1_tasks} total"
          " combinations of models) ===",
          flush=True,
      )
      # When Phase 0 is skipped (family=None) the selected
      # cluster keys are logged here instead.
      if self._pending_cluster_log:
        print(self._pending_cluster_log, flush=True)
        self._pending_cluster_log = None

      out_dir_p1 = os.path.join(self.output_dir, f"step1_exploration_{d_col}")
      if os.path.exists(out_dir_p1):
        shutil.rmtree(out_dir_p1, ignore_errors=True)
      os.makedirs(out_dir_p1, exist_ok=True)

      # The scaler fit / transform and the
      # CATE basis p(X) of this treatment are computed ONCE here and handed
      # to every Phase 1 task and Phase 3 repetition.
      prepared = runner.prepare_run_inputs(
          df_fit, scaler, df_score, cate_cols, cate_structure
      )
      # The parallel plan (W, h) of Phase 1 and its log line.
      workers_p1, threads_p1 = self._parallel_plan(total_p1_tasks)
      print(f"  ├─ hparam_policy = {self.hparam_policy}", flush=True)
      # Under "global" ONE FLAML search per
      # (treatment, learner) on the Phases 1-3 sample with the
      # D column of this run as the T-side target, BEFORE the
      # Phase 1 task parallelism.  Diag6 / the treatment replacement already
      # ran above (`_apply_saturation_transform`), as before.
      hparams: Dict[tuple[str, str], hparam_search.HparamSearchResult] = {}
      if self.hparam_policy == runner.HPARAM_POLICY_GLOBAL:
        hparams = self._run_global_hparam_search(
            d_col, x_cols, prepared.df_norm, out_dir_p1
        )

      print(
          f"  ├─ Phase 1: workers = {workers_p1}, threads/worker ="
          f" {threads_p1}",
          flush=True,
      )
      tasks_p1 = []
      for y_val in self.ml_models_y:
        for t_val in self.ml_models_t:
          for f_val in self.n_folds_list:
            tasks_p1.append({"ml_y": y_val, "ml_t": t_val, "n_folds": f_val})
      stop_event = threading.Event()
      monitor_thread = threading.Thread(
          target=self._monitor_phase1_progress,
          args=(out_dir_p1, total_p1_tasks, stop_event, 60),
      )
      monitor_thread.start()

      try:
        with parallel_section(workers_p1, threads_p1):
          phase1_runs = joblib.Parallel(n_jobs=workers_p1, verbose=0)(
              joblib.delayed(self._run_single_exploration)(
                  t_dict["ml_y"],
                  t_dict["ml_t"],
                  t_dict["n_folds"],
                  d_col,
                  x_cols,
                  cate_cols,
                  cate_structure,
                  gt_col,
                  df_fit,
                  scaler,
                  process_log,
                  df_score=df_score,
                  row_order=row_order,
                  prepared=prepared,
                  hparams_y=hparams.get((hparam_search.SIDE_Y, t_dict["ml_y"])),
                  hparams_t=hparams.get((hparam_search.SIDE_T, t_dict["ml_t"])),
                  learner_n_jobs=threads_p1,
              )
              for t_dict in tasks_p1
          )
      finally:
        stop_event.set()
        monitor_thread.join()

      # The full Phase 1 result of every candidate is kept
      # (keyed by model name) and becomes Phase 3 repetition 0.
      phase1_results: Dict[str, Dict[str, Any]] = {
          res["model_name"]: res for res in phase1_runs
      }
      exploration_metrics = [res["metrics_1c"] for res in phase1_runs]

      df_1c_all = pd.DataFrame(exploration_metrics)
      print(f"\n=== Phase 2: Shortlisting Top {top_n} Performing Models ===")
      df_shortlisted = selection.shortlist_top_models(df_1c_all, top_n=top_n)
      # `model` is a '_'-joined string whose first two fields are FLAML learner
      # names, and those names may themselves contain '_' ('extra_tree',
      # 'xgb_limitdepth'). Splitting it positionally shifts every field left
      # and makes int() fail on a learner name, so read the structured columns
      # that runner.py now carries on metrics_1c instead.  Note this must be
      # 'n_folds' (the configured value) and NOT 'number_of_folds', which is
      # DoubleML's effective count and equals n_folds ** n_cluster_vars.
      df_shortlisted["ML_model_Y"] = df_shortlisted["ml_y_name"]
      df_shortlisted["ML_model_T"] = df_shortlisted["ml_t_name"]
      df_shortlisted["n_folds"] = df_shortlisted["n_folds"].astype(int)
      io.save_step2_outputs(
          df_shortlisted,
          os.path.join(self.output_dir, f"step2_shortlisted_{d_col}"),
      )

      print(
          f"\n=== Phase 3: Robust Ensemble Estimation ({n_reps} repetitions per"
          " model) ===",
          flush=True,
      )
      shortlisted_rows = df_shortlisted[df_shortlisted["shortlisted"] == 1]
      candidate_names = [str(name) for name in shortlisted_rows["model"]]
      # The Phase 1 results of candidates that were NOT shortlisted are
      # never used again: release their fitted models and frames now.
      for name in list(phase1_results):
        if name not in candidate_names:
          del phase1_results[name]
      for name in candidate_names:
        phase1_results[name]["model_group"] = name

      # ONE job list {(m, r): m in shortlist, r = 1..R-1}
      # (repetition 0 is the Phase 1 result); the total
      # number of repetitions per candidate stays R.
      jobs_p3: List[tuple[str, int]] = []
      rows_by_name: Dict[str, pd.Series] = {}
      for model_idx, (_, row) in enumerate(shortlisted_rows.iterrows(), 1):
        name = str(row["model"])
        rows_by_name[name] = row
        print(
            f"\n>>> Phase 3 Candidate {model_idx}/{len(shortlisted_rows)}:"
            f" Evaluating Model '{name}' (repetition 0 = Phase 1 result,"
            f" repetitions 1..{n_reps - 1} scheduled) <<<",
            flush=True,
        )
        for rep in range(1, n_reps):
          jobs_p3.append((name, rep))
      workers_p3, threads_p3 = self._parallel_plan(len(jobs_p3))
      print(
          f"  ├─ Phase 3: workers = {workers_p3}, threads/worker ="
          f" {threads_p3} ({len(jobs_p3)} = {len(shortlisted_rows)} x"
          f" {max(n_reps - 1, 0)} new repetitions)",
          flush=True,
      )

      temp_p3_dir = os.path.join(self.output_dir, f"temp_p3_{d_col}")
      if os.path.exists(temp_p3_dir):
        shutil.rmtree(temp_p3_dir, ignore_errors=True)
      os.makedirs(temp_p3_dir, exist_ok=True)

      flat_results: List[Dict[str, Any]] = []
      if jobs_p3:
        stop_event_p3 = threading.Event()
        monitor_thread_p3 = threading.Thread(
            target=self._monitor_phase3_progress,
            args=(temp_p3_dir, len(jobs_p3), None, stop_event_p3, 60),
        )
        monitor_thread_p3.start()
        try:
          with parallel_section(workers_p3, threads_p3):
            flat_results = joblib.Parallel(n_jobs=workers_p3, verbose=0)(
                joblib.delayed(self._run_single_iteration)(
                    rows_by_name[name],
                    d_col,
                    x_cols,
                    cate_cols,
                    cate_structure,
                    gt_col,
                    df_fit,
                    scaler,
                    rep,
                    temp_p3_dir,
                    df_score=df_score,
                    row_order=row_order,
                    prepared=prepared,
                    hparams_y=hparams.get(
                        (hparam_search.SIDE_Y, rows_by_name[name]["ML_model_Y"])
                    ),
                    hparams_t=hparams.get(
                        (hparam_search.SIDE_T, rows_by_name[name]["ML_model_T"])
                    ),
                    learner_n_jobs=threads_p3,
                    reduced=True,
                    done_name=f"job_{job_idx}.done",
                )
                for job_idx, (name, rep) in enumerate(jobs_p3)
            )
        finally:
          stop_event_p3.set()
          monitor_thread_p3.join()
      shutil.rmtree(temp_p3_dir, ignore_errors=True)

      # Results in the shape of the former sequential loop: per candidate in
      # shortlist order, repetition 0 first, then 1..R-1.
      all_iter_results_for_channel = regroup_phase3_results(
          candidate_names, phase1_results, jobs_p3, flat_results
      )

      # Phases 1-3 are estimated on the transformed treatment
      # f(T; alpha_hat), but every Phase 4 aggregation (total input, ROI,
      # charts, optimization) must be expressed in RAW treatment units. The
      # transformed column is preserved under `<d_col>_transformed` so the
      # fitted quantity remains auditable. No-op when family=None and
      # when Phase 0 selected linear.
      if self._family_for(d_col) is not None:
        for res_item in all_iter_results_for_channel:
          frame = res_item.get("df_1a")
          if frame is None or d_col not in frame.columns:
            continue
          frame[f"{d_col}_transformed"] = frame[d_col].values
          frame[d_col] = df_est[d_col].reindex(frame.index).values

      df_3a_detail = pd.DataFrame(
          [r["metrics_1c"] for r in all_iter_results_for_channel]
      )
      df_3a_detail["Model Group"] = [
          r["model_group"] for r in all_iter_results_for_channel
      ]

      y_col_rmse = "nuisance_models_rmse_Y_model_evaluated_by_AutoML"
      t_col_rmse = "nuisance_models_rmse_T_model_evaluated_by_AutoML"
      eps = 1e-9

      df_3a_detail["raw_Y_rmse"] = df_3a_detail[y_col_rmse]
      df_3a_detail["median_raw_Y_rmse"] = df_3a_detail.groupby("Model Group")[
          "raw_Y_rmse"
      ].transform("median")

      min_y = df_3a_detail["raw_Y_rmse"].min()
      max_y = df_3a_detail["raw_Y_rmse"].max()
      df_3a_detail["norm_Y_rmse"] = (
          df_3a_detail["raw_Y_rmse"] - min_y + eps
      ) / (max_y - min_y + eps)

      df_3a_detail["norm_median_Y_rmse"] = (
          df_3a_detail["median_raw_Y_rmse"] - min_y + eps
      ) / (max_y - min_y + eps)

      df_3a_detail["raw_T_rmse"] = df_3a_detail[t_col_rmse]
      df_3a_detail["median_raw_T_rmse"] = df_3a_detail.groupby("Model Group")[
          "raw_T_rmse"
      ].transform("median")

      min_t = df_3a_detail["raw_T_rmse"].min()
      max_t = df_3a_detail["raw_T_rmse"].max()
      df_3a_detail["norm_T_rmse"] = (
          df_3a_detail["raw_T_rmse"] - min_t + eps
      ) / (max_t - min_t + eps)

      df_3a_detail["norm_median_T_rmse"] = (
          df_3a_detail["median_raw_T_rmse"] - min_t + eps
      ) / (max_t - min_t + eps)

      df_3a_detail["combined_error"] = df_3a_detail["norm_T_rmse"] * (
          df_3a_detail["norm_T_rmse"] + df_3a_detail["norm_Y_rmse"]
      )
      df_3a_detail["combined_error_calculated_by_median_rmse"] = df_3a_detail[
          "norm_median_T_rmse"
      ] * (
          df_3a_detail["norm_median_T_rmse"]
          + df_3a_detail["norm_median_Y_rmse"]
      )

      min_combined = df_3a_detail[
          "combined_error_calculated_by_median_rmse"
      ].min()
      best_model_group = df_3a_detail.loc[
          df_3a_detail["combined_error_calculated_by_median_rmse"]
          == min_combined,
          "Model Group",
      ].iloc[0]
      df_3a_detail["best_model"] = (
          df_3a_detail["Model Group"] == best_model_group
      ).astype(int)

      # The `model_obj` and the full `df_1a` of the
      # repetition-0 results of the OTHER candidates are discarded now; their
      # KPI columns / metrics stay for the per-group summaries below.
      for name in candidate_names:
        if name != best_model_group:
          rep0 = phase1_results[name]
          rep0.pop("model_obj", None)
          rep0["df_1a"] = reduce_rep_result(rep0, d_col, gt_col)["df_1a"]

      iteration_summaries = []
      for m_group, df_group in df_3a_detail.groupby(
          "Model Group", observed=False
      ):
        group_results = [
            r
            for r in all_iter_results_for_channel
            if r["model_group"] == m_group
        ]

        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          median_est_sales = np.nanmedian(
              np.array([r["df_1a"][kpi_col].values for r in group_results]),
              axis=0,
          )

        total_input = np.nansum(group_results[0]["df_1a"][d_col].values)
        total_kpi = np.nansum(median_est_sales)

        sum_res = {
            "Channel": d_col,
            "Model Group": m_group,
            "x_col": str(x_cols),
            "y_col": self.y_col,
            "covariates_for_cate": str(cate_cols),
            "cate_structure": df_group["cate_structure"].iloc[0],
            "cate_df": df_group["cate_df"].iloc[0],
            "cate_degree": df_group["cate_degree"].iloc[0],
            "cate_intercept": df_group["cate_intercept"].iloc[0],
            "total_input": total_input,
            "total_estimated_incremental_KPI": total_kpi,
            "total_estimated_roi": (
                total_kpi / total_input if total_input > 0 else 0
            ),
        }
        if gt_col and gt_col in group_results[0]["df_1a"].columns:
          sum_res.update(
              metrics.calculate_ground_truth_metrics(
                  group_results[0]["df_1a"][gt_col],
                  pd.Series(median_est_sales),
                  total_input,
              )
          )
        else:
          sum_res.update({
              "true_total_roi": np.nan,
              "SMAPE": np.nan,
              "R_squared": np.nan,
              "MAE": np.nan,
              "RMSE": np.nan,
          })

        sum_res.update({
            "norm_median_Y_rmse": df_group["norm_median_Y_rmse"].iloc[0],
            "norm_Y_rmse_mean": df_group["norm_Y_rmse"].mean(),
            "norm_Y_rmse_sd": df_group["norm_Y_rmse"].std(),
            "norm_median_T_rmse": df_group["norm_median_T_rmse"].iloc[0],
            "norm_T_rmse_mean": df_group["norm_T_rmse"].mean(),
            "norm_T_rmse_sd": df_group["norm_T_rmse"].std(),
            "combined_error_mean": df_group["combined_error"].mean(),
            "combined_error_sd": df_group["combined_error"].std(),
            "combined_error_calculated_by_median_rmse": (
                df_group["combined_error_calculated_by_median_rmse"].iloc[0]
            ),
            "best_model": 1 if m_group == best_model_group else 0,
        })
        iteration_summaries.append(sum_res)

      df_3b_summary = pd.DataFrame(iteration_summaries)

      best_group_results = [
          r
          for r in all_iter_results_for_channel
          if r["model_group"] == best_model_group
      ]
      best_model_data = best_group_results[0].copy()

      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        point_estimates = np.array(
            [r["df_1a"][kpi_col].values for r in best_group_results]
        )
        lower_bounds = np.array(
            [r["df_1a"][kpi_lower].values for r in best_group_results]
        )
        upper_bounds = np.array(
            [r["df_1a"][kpi_upper].values for r in best_group_results]
        )

        # Chernozhukov et al. (2018), Section 3.4:
        # Median Method for Repeated Cross-Fitting
        tilde_theta = np.nanmedian(point_estimates, axis=0)

        # Back-calculate SE for each repetition b:
        # SE_b = (Upper_b - Lower_b) / (2 * 1.96)
        se_b = (upper_bounds - lower_bounds) / (2.0 * 1.96)
        var_b = se_b**2

        # Penalty for deviation from median point estimate:
        # (theta_b - tilde_theta)^2
        deviation_penalty = (point_estimates - tilde_theta) ** 2

        # Combined variance for each repetition
        combined_var_b = var_b + deviation_penalty

        # Aggregated variance across repetitions: Median_b(combined_var_b)
        tilde_var = np.nanmedian(combined_var_b, axis=0)
        tilde_sigma = np.sqrt(np.maximum(0.0, tilde_var))

        best_model_data["df_1a"][kpi_col] = tilde_theta
        best_model_data["df_1a"][kpi_lower] = tilde_theta - 1.96 * tilde_sigma
        best_model_data["df_1a"][kpi_upper] = tilde_theta + 1.96 * tilde_sigma

      df_1b_base = metrics.aggregate_geo_item_metrics(
          best_model_data["df_1a"],
          best_model_data["model_name"],
          d_col,
          self.y_col,
          geo_col=self.geo_col,
          item_col=self.item_col,
          geo_name_col=self.geo_name_col,
          item_name_col=self.item_name_col,
          gt_col=gt_col,
      )
      bm_metrics = best_model_data["metrics_1c"]
      meta_dict = {
          "d_col": d_col,
          "x_col": str(x_cols),
          "y_col": self.y_col,
          "covariates_for_cate": str(cate_cols),
          "cate_structure": bm_metrics.get("cate_structure", ""),
          "cate_df": bm_metrics.get("cate_df", ""),
          "cate_degree": bm_metrics.get("cate_degree", ""),
          "cate_intercept": bm_metrics.get("cate_intercept", ""),
      }
      df_1b_base = self._insert_metadata_columns(df_1b_base, meta_dict)
      best_model_data["df_1b"] = df_1b_base

      all_best_models_3b.append(
          df_3b_summary[df_3b_summary["best_model"] == 1].copy()
      )
      all_3a_details.append(
          df_3a_detail[df_3a_detail["best_model"] == 1].copy()
      )
      best_model_data_dict[d_col] = best_model_data

      for misleading_key in ["Y_pred", "T_pred", "Y_res", "T_res"]:
        best_model_data.pop(misleading_key, None)

      if "cate_coef_dict" in best_group_results[0]:
        pooled_cate_coef_dict = {}
        all_keys = set()
        for r in best_group_results:
          if "cate_coef_dict" in r:
            all_keys.update(r["cate_coef_dict"].keys())

        for cov in [str(k)[:-5] for k in all_keys if str(k).endswith("_coef")]:
          coef_key = f"{cov}_coef"
          stderr_key = f"{cov}_stderr"
          pvalue_key = f"{cov}_pvalue"

          coef_vals = []
          stderr_vals = []
          for r in best_group_results:
            if "cate_coef_dict" in r and coef_key in r["cate_coef_dict"]:
              coef_vals.append(r["cate_coef_dict"][coef_key])
              stderr_vals.append(r["cate_coef_dict"].get(stderr_key, np.nan))

          if coef_vals:
            coef_array = np.array(coef_vals, dtype=float)

            # 1. Pool the Coefficients
            pooled_coef = float(np.nanmedian(coef_array))
            pooled_cate_coef_dict[coef_key] = pooled_coef

            # 2. Calculate the Robust Pooled Standard Error
            if not np.all(np.isnan(stderr_vals)):
              stderr_array = np.array(stderr_vals, dtype=float)
              variance_array = stderr_array**2 + (coef_array - pooled_coef) ** 2
              pooled_stderr = float(np.sqrt(np.nanmedian(variance_array)))
              pooled_cate_coef_dict[stderr_key] = pooled_stderr

              # 3. Recalculate the Pooled P-value
              if pooled_stderr > 0 and not np.isnan(pooled_stderr):
                z_score = pooled_coef / pooled_stderr
                pooled_pvalue = float(stats.norm.sf(np.abs(z_score)) * 2)
              elif pooled_stderr == 0.0:
                pooled_pvalue = 0.0
              else:
                pooled_pvalue = np.nan
              pooled_cate_coef_dict[pvalue_key] = pooled_pvalue

        # Handle any remaining keys that were not processed (if any)
        for key in all_keys - set(pooled_cate_coef_dict.keys()):
          vals = [
              r["cate_coef_dict"][key]
              for r in best_group_results
              if "cate_coef_dict" in r and key in r["cate_coef_dict"]
          ]
          if vals:
            pooled_cate_coef_dict[key] = float(np.nanmedian(vals))

        best_model_data["cate_coef_dict"] = pooled_cate_coef_dict

        keys_to_keep = [
            "model",
            "d_col",
            "x_col",
            "y_col",
            "covariates_for_cate",
            "cate_structure",
            "cate_df",
            "cate_degree",
            "cate_intercept",
        ]
        for r in best_group_results:
          row = {k: r["metrics_1c"].get(k) for k in keys_to_keep}
          c_dict = r.get("cate_coef_dict", {})
          row["cate_coef_dict"] = json.dumps(
              {k: v for k, v in c_dict.items() if str(k).endswith("_coef")}
          )
          row["cate_pvalue_dict"] = json.dumps(
              {k: v for k, v in c_dict.items() if str(k).endswith("_pvalue")}
          )
          row["cate_stderr_dict"] = json.dumps(
              {k: v for k, v in c_dict.items() if str(k).endswith("_stderr")}
          )
          all_cate_pool_rows.append(row)

        median_row = {
            k: best_group_results[0]["metrics_1c"].get(k) for k in keys_to_keep
        }
        median_row["model"] = "median"
        median_row["cate_coef_dict"] = json.dumps({
            k: v
            for k, v in pooled_cate_coef_dict.items()
            if str(k).endswith("_coef")
        })
        median_row["cate_pvalue_dict"] = json.dumps({
            k: v
            for k, v in pooled_cate_coef_dict.items()
            if str(k).endswith("_pvalue")
        })
        median_row["cate_stderr_dict"] = json.dumps({
            k: v
            for k, v in pooled_cate_coef_dict.items()
            if str(k).endswith("_stderr")
        })
        all_cate_pool_rows.append(median_row)

      if "model_obj" in best_model_data:
        # Inject contextual metadata for the historical results.
        best_model_data["model_obj"].pipeline_metadata_ = {
            "historical_results_path": io.FULL_DF_FILE,
            "message": (
                "Valid, pooled historical estimation results are stored in the "
                "above CSV. This pickle acts as an inference engine for future "
                "out-of-sample data, not for historical diagnostic lookup."
            ),
        }

        # Ensure the fitted models are saved for out-of-sample inference.
        # Enforce store_models=True so that the exported .pkl explicitly retains
        # the fitted T-models, Y-models, and causal effect functions (theta).
        best_model_data["model_obj"].store_models = True

        if "cate_coef_dict" in best_model_data:
          best_model_data["model_obj"].cate_coef_dict_ = best_model_data[
              "cate_coef_dict"
          ]

      io.save_step3_outputs(
          df_3a_detail,
          df_3b_summary,
          best_model_data,
          os.path.join(self.output_dir, f"step3_final_{d_col}"),
      )

    # Phase 4
    print(
        "\n======================================================================"
    )
    print("=== Phase 4: Consolidating Results & Visualizing Impact ===")
    print(
        "======================================================================"
    )
    self._warn_treatment_lag_covariates()

    phase4_dir = os.path.join(self.output_dir, io.PHASE4_DIR)
    graph_dir = os.path.join(phase4_dir, "graph")
    total_graph_dir = os.path.join(graph_dir, "001_total")
    geo_graph_dir = os.path.join(graph_dir, "002_geo")
    item_graph_dir = os.path.join(graph_dir, "003_item")
    cate_graph_dir = os.path.join(graph_dir, "004_cate")
    prior_graph_dir = os.path.join(graph_dir, "005_prior")
    sens_graph_dir = os.path.join(graph_dir, "006_sensitivity")

    for dir_path in [
        phase4_dir,
        total_graph_dir,
        geo_graph_dir,
        item_graph_dir,
        sens_graph_dir,
        cate_graph_dir,
        prior_graph_dir,
    ]:
      os.makedirs(dir_path, exist_ok=True)

    pd.concat(all_3a_details, ignore_index=True).to_csv(
        os.path.join(self.output_dir, io.REPETITIONS_FILE), index=False
    )
    if all_cate_pool_rows:
      pd.DataFrame(all_cate_pool_rows).to_csv(
          os.path.join(self.output_dir, io.CATE_COEF_FILE),
          index=False,
      )
    df_3b_concat = pd.concat(all_best_models_3b, ignore_index=True)
    is_lag = df_3b_concat["Channel"].str.contains(r"_l\d+$", regex=True)
    sum_input = df_3b_concat.loc[~is_lag, "total_input"].sum()
    sum_kpi = df_3b_concat["total_estimated_incremental_KPI"].sum()

    total_row_dict = {
        "Channel": "Total",
        "Model Group": "-",
        "x_col": "-",
        "y_col": self.y_col,
        "covariates_for_cate": "-",
        "cate_structure": "-",
        "cate_df": np.nan,
        "cate_degree": np.nan,
        "cate_intercept": np.nan,
        "total_input": sum_input,
        "total_estimated_incremental_KPI": sum_kpi,
        "total_estimated_roi": sum_kpi / sum_input if sum_input > 0 else 0,
        "combined_error_calculated_by_median_rmse": np.nan,
        "combined_error_mean": np.nan,
        "combined_error_sd": np.nan,
        "norm_median_Y_rmse": np.nan,
        "norm_Y_rmse_mean": np.nan,
        "norm_Y_rmse_sd": np.nan,
        "norm_median_T_rmse": np.nan,
        "norm_T_rmse_mean": np.nan,
        "norm_T_rmse_sd": np.nan,
        "best_model": 1,
    }

    if self.ground_truth_existence:
      # Phase 4 always operates on `df_est` (raw treatment).
      # Identical to `df` when family=None.
      y_true_ts = np.zeros(len(df_est))
      y_pred_ts = np.zeros(len(df_est))
      for i, c in enumerate(self.d_cols):
        b_data = best_model_data_dict[c]["df_1a"]
        kpi_col = f"estimated_incremental_KPI_{c}"
        gt_col = self.ground_truth_effect_column[i]
        y_pred_ts += b_data[kpi_col].fillna(0).values
        if gt_col in b_data.columns:
          y_true_ts += b_data[gt_col].fillna(0).values
      total_row_dict.update(
          metrics.calculate_ground_truth_metrics(
              pd.Series(y_true_ts), pd.Series(y_pred_ts), sum_input
          )
      )
    else:
      total_row_dict.update({
          "true_total_roi": np.nan,
          "SMAPE": np.nan,
          "R_squared": np.nan,
          "MAE": np.nan,
          "RMSE": np.nan,
      })

    pd.concat(
        [df_3b_concat, pd.DataFrame([total_row_dict])], ignore_index=True
    ).to_csv(
        os.path.join(self.output_dir, io.SUMMARY_FILE), index=False
    )

    # The consolidated frame carries the RAW treatment values.
    # `df_est` is row-identical to `df` whenever family=None.
    df_1a_consolidated = df_est.copy()
    total_est_sales, total_lower_ci, total_upper_ci = (
        np.zeros(len(df_est)),
        np.zeros(len(df_est)),
        np.zeros(len(df_est)),
    )
    total_input_series = np.zeros(len(df_est))
    total_gt_sales = (
        np.zeros(len(df_est)) if self.ground_truth_existence else None
    )
    individual_kpi_cols = []
    cate_eq_dict = {}
    cate_stats_dict = {}

    for idx, (d_col, b_data) in enumerate(best_model_data_dict.items()):
      c_kpi, c_low, c_upp = (
          f"estimated_incremental_KPI_{d_col}",
          f"estimated_incremental_KPI_{d_col}_2.5%",
          f"estimated_incremental_KPI_{d_col}_97.5%",
      )
      df_1a_consolidated[c_kpi] = b_data["df_1a"][c_kpi].values
      df_1a_consolidated[c_low] = b_data["df_1a"][c_low].values
      df_1a_consolidated[c_upp] = b_data["df_1a"][c_upp].values
      individual_kpi_cols.append(c_kpi)

      total_est_sales += b_data["df_1a"][c_kpi].values
      total_lower_ci += b_data["df_1a"][c_low].values
      total_upper_ci += b_data["df_1a"][c_upp].values
      if not re.search(r"_l\d+$", d_col, flags=re.IGNORECASE):
        total_input_series += b_data["df_1a"][d_col].values
      if self.ground_truth_existence:
        total_gt_sales += (
            b_data["df_1a"][self.ground_truth_effect_column[idx]]
            .fillna(0)
            .values
        )
      if "cate_coef_dict" in b_data:
        cate_eq_dict[d_col] = self.format_cate_equation(
            b_data["cate_coef_dict"]
        )
        c_dict = b_data["cate_coef_dict"]
        cate_stats_dict[d_col] = {
            "coef": {
                str(k)[:-5]: v
                for k, v in c_dict.items()
                if str(k).endswith("_coef")
            },
            "pvalue": {
                str(k)[:-7]: v
                for k, v in c_dict.items()
                if str(k).endswith("_pvalue")
            },
            "stderr": {
                str(k)[:-7]: v
                for k, v in c_dict.items()
                if str(k).endswith("_stderr")
            },
        }

    df_1a_consolidated["total_estimated_incremental_KPI"] = total_est_sales
    df_1a_consolidated["total_estimated_incremental_KPI_2.5%"] = total_lower_ci
    df_1a_consolidated["total_estimated_incremental_KPI_97.5%"] = total_upper_ci
    df_1a_consolidated["total_input_aggregate"] = total_input_series
    consolidated_gt_col = (
        "total_ground_truth_KPI" if self.ground_truth_existence else None
    )
    if consolidated_gt_col:
      df_1a_consolidated[consolidated_gt_col] = total_gt_sales
    # Phase 0 `d_inf` split: flag which rows the model was fitted on (D_inf)
    # and which were scored with the fitted CATE (D_sel).  The column exists
    # only when such scoring happened, so family=None outputs are unchanged.
    sample_flags = [
        b["df_1a"][runner.ESTIMATION_SAMPLE_COL].values
        for b in best_model_data_dict.values()
        if runner.ESTIMATION_SAMPLE_COL in b["df_1a"].columns
    ]
    if sample_flags:
      df_1a_consolidated[runner.ESTIMATION_SAMPLE_COL] = sample_flags[0]
      n_scored = int(np.sum(sample_flags[0] == runner.SCORED_SAMPLE_LABEL))
      print(
          f"  ├─ Consolidated frame covers all {len(df_1a_consolidated)} rows:"
          f" {len(df_1a_consolidated) - n_scored} fitted (D_inf) +"
          f" {n_scored} scored with the fitted CATE (D_sel).",
          flush=True,
      )

    if self.optimize:
      if len(self.d_cols) > 1:
        # Several treatments -> the saturation-aware
        # optimizer is not run; the existing optimization path (which itself
        # does not support several treatments) is taken instead.
        if self.sat_family is not None and not self.simple_optimization:
          warnings.warn(
              "Saturation-aware optimization supports a single treatment of"
              f" type 'spend'; {len(self.d_cols)} treatments were given."
              " Falling back to the existing optimization (saturation-aware"
              " optimization for several treatments is future work).",
              UserWarning,
          )
        warnings.warn(
            "Optimization with multiple treatments is not yet supported."
        )
      # `optimize=True` with `simple_optimization=False` and no saturation
      # family is rejected by the constructor, so no other case remains.

    for d_col in self.d_cols:
      ttype = self.treatment_types.get(d_col, "spend")

      if ttype in ["percentage", "impressions", "price"]:
        financial_spend_col = self.treatment_spend_cols.get(d_col)
        if not financial_spend_col or not self.sales_col:
          warnings.warn(
              f"Skipping ROI output and optimization for treatment '{d_col}'. "
              f"To conduct these processes when treatment_type is '{ttype}', "
              "please provide both 'treatment_spend_cols' and 'sales_col'."
          )
          continue

      if ttype in ["percentage", "impressions", "spend", "price"]:
        financial_spend_col = self.treatment_spend_cols.get(d_col)

        if (
            financial_spend_col
            and financial_spend_col in df_1a_consolidated.columns
        ):
          kpi_col = f"estimated_incremental_KPI_{d_col}"
          spend_vals = df_1a_consolidated[financial_spend_col].values
          kpi_vals = df_1a_consolidated[kpi_col].values
          df_1a_consolidated[f"roi_amount_based_{d_col}"] = np.divide(
              kpi_vals,
              spend_vals,
              out=np.zeros_like(kpi_vals, dtype=float),
              where=spend_vals != 0,
          )

          roi_dir = os.path.join(phase4_dir, "roi_output", d_col)
          opt_out_dir = graph_dir if ttype == "spend" else roi_dir
          if ttype != "spend":
            print("  ├─ Generating ROI Charts...")
            plots.generate_roi_charts(
                df_1a_consolidated,
                self.date_col,
                self.geo_col,
                self.sales_col,
                financial_spend_col,
                kpi_col,
                roi_dir,
                peak_months=self.peak_months,
                item_col=getattr(self, "item_col", None),
            )

          # The saturation-aware optimizer budgets the treatment column
          # directly, which is only meaningful when the treatment IS the money
          # being spent (`g(T) = T`).  For "percentage" / "impressions" /
          # "price" the constructor already forced `simple_optimization=True`
          # (no cost model `spend = g(treatment)` is specified yet); the
          # `ttype == "spend"` guard below is kept as a defensive invariant.
          is_spend_treatment = ttype == "spend"

          # Phase 0 returned selected_family=None (the data
          # did not support saturation) -> automatic fallback to the existing
          # optimization, with a warning.  The config value of
          # `simple_optimization` itself is NOT rewritten.
          linear_fallback_opt = (
              self.optimize
              and not self.simple_optimization
              and is_spend_treatment
              and self.sat_family is not None
              and self._family_for(d_col) is None
              and len(self.d_cols) <= 1
          )
          if linear_fallback_opt:
            self._record_linear_fallback(d_col)

          run_simple_opt = (
              self.optimize
              and (
                  self.simple_optimization
                  or not is_spend_treatment
                  or linear_fallback_opt
              )
              and len(self.d_cols) <= 1
          )
          # The saturation-aware detailed optimizer. Mutually
          # exclusive with the simple optimizer above; unreachable when
          # family=None because the constructor forbids that combination.
          run_saturation_opt = (
              self.optimize
              and not self.simple_optimization
              and is_spend_treatment
              and self._family_for(d_col) is not None
              and len(self.d_cols) <= 1
          )
          if self.sat_family is not None and d_col not in (
              self._phase4_optimization
          ):
            if run_saturation_opt:
              phase4_opt_mode = "saturation"
            elif run_simple_opt:
              phase4_opt_mode = "simple"
            else:
              phase4_opt_mode = "none"
            self._phase4_optimization[d_col] = {
                "phase4_optimization": phase4_opt_mode,
                "phase4_fallback_to_linear": False,
            }
            self._update_phase0_summary(d_col)

          # Bound unconditionally so the optimizer call sites below always
          # see a defined value, and so no value leaks in from a previous
          # d_col iteration.
          opt_periods: Optional[tuple[Any, Any]] = None
          if run_simple_opt or run_saturation_opt:
            if self.date_col and self.date_col in df_1a_consolidated.columns:
              min_d = df_1a_consolidated[self.date_col].min()
              max_d = df_1a_consolidated[self.date_col].max()
            else:
              min_d, max_d = None, None

            opt_periods = (
                self.opt_periods
                if self.opt_periods is not None
                else (min_d, max_d)
            )

          if run_simple_opt:
            print("  ├─ Generating Optimization Charts...")
            optimization.optimize(
                df_1a_consolidated=df_1a_consolidated,
                treatment=d_col,
                treatment_amt=financial_spend_col,
                periods=opt_periods,
                threshold_roi=self.opt_threshold_roi,
                other_cost_variables=self.opt_other_cost_variables,
                target_geo="all",
                target_item="all",
                outcome_col=self.y_col,
                date_col=self.date_col,
                geo_col=self.geo_col,
                item_col=getattr(self, "item_col", None),
                sales_col=self.sales_col,
                out_dir=opt_out_dir,
            )
          elif run_saturation_opt:
            # theta must be in KPI units for the closed-form
            # allocator to be comparable across rows.
            d_idx = self.d_cols.index(d_col)
            theta_hat = self._compute_theta_for_optimization(
                d_col,
                d_idx,
                df_est,
                scaler,
                best_model_data_dict[d_col],
            )
            # Normalization bridge: alpha_hat lives on the
            # normalized treatment, but the Phase 4 budget is in raw units.
            # `fbar(T/c; alpha, T_ref) = fbar(T; alpha_i,
            # c_i T_ref)`, so the raw-unit objective is `theta * fbar` with
            # `alpha_i` and `T_ref_raw_i`; `gain` is only kept for the legacy
            # un-normalized path inside `optimize_saturation`.
            alpha_raw, gain = self._alpha_raw_for(d_col, df_est.index)
            t_ref_raw = self._t_ref_raw_for(d_col, df_est.index)
            print(
                "  ├─ Generating Saturation Optimization Charts"
                f" (family={self._family_for(d_col)},"
                f" alpha={self._alpha_hat.get(d_col, float('nan')):.6g}"
                f" on {self._treatment_normalization.get(d_col, 'none')}"
                f" scale, T_ref={self._t_ref_for(d_col):.6g},"
                f" budget_mode={self.budget_mode},"
                f" threshold_roi={float(self.opt_threshold_roi):g},"
                f" cap_quantile={self.cap_quantile:g},"
                f" reallocate_within={self.reallocate_within}; no carryover"
                " is modelled: spend moves across the allocation units"
                " (geo x item x date rows) within that scope)..."
            )
            # Under roi_target the budget settings are
            # irrelevant (INFO, not an error).
            if self.budget_mode == "roi_target" and (
                self.budget_increment != 0.0 or self._budget_reallocate_given
            ):
              print(
                  "  │   [INFO] budget_mode = roi_target: budget_increment ="
                  f" {self.budget_increment:g} and budget_reallocate ="
                  f" {self.budget_reallocate} are ignored (no budget; every"
                  " unit is allocated min(T_bar_t, T_t^r)).",
                  flush=True,
              )
            self._phase4_optimization.setdefault(d_col, {}).update({
                "budget_mode": self.budget_mode,
                "budget_reallocate": self.budget_reallocate,
                "cap_quantile": self.cap_quantile,
                "reallocate_within": self.reallocate_within,
                "assumptions": "no_carryover",
                "opt_geos": self.opt_geos,
                "opt_items": self.opt_items,
            })
            # Rebound so the optimizer's output columns are picked up by the
            # consolidated CSV written further below.
            df_1a_consolidated = optimization.optimize_saturation(
                df_1a_consolidated=df_1a_consolidated,
                treatment=d_col,
                treatment_amt=financial_spend_col,
                periods=opt_periods,
                family=self.sat_family,
                alpha=alpha_raw,
                theta=theta_hat,
                gain=gain,
                t_ref=t_ref_raw,
                budget_increment=float(self.budget_increment or 0.0),
                threshold_roi=self.opt_threshold_roi,
                other_cost_variables=self.opt_other_cost_variables,
                target_geo="all",
                target_item="all",
                outcome_col=self.y_col,
                budget_mode=self.budget_mode,
                budget_reallocate=self.budget_reallocate,
                opt_geos=self.opt_geos,
                opt_items=self.opt_items,
                cap_quantile=self.cap_quantile,
                reallocate_within=self.reallocate_within,
                date_col=self.date_col,
                geo_col=self.geo_col,
                item_col=getattr(self, "item_col", None),
                sales_col=self.sales_col,
                out_dir=opt_out_dir,
            )

          # Updated Prior for roi spend
          if ttype == "spend":
            continue

          if self.date_col and self.date_col in df_1a_consolidated.columns:
            ts_promo = df_1a_consolidated.groupby(
                self.date_col, observed=False
            ).sum(numeric_only=True)
          else:
            ts_promo = df_1a_consolidated

          if financial_spend_col in ts_promo.columns:
            promo_spend_series = ts_promo[financial_spend_col]
            promo_kpi_series = ts_promo[kpi_col]
            roi_mask_p = promo_spend_series > 0
            roi_data_p = (
                (
                    promo_kpi_series.loc[roi_mask_p]
                    / promo_spend_series.loc[roi_mask_p]
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            contrib_mask_p = (promo_spend_series > 0) & (
                ts_promo[self.y_col] > 0
            )
            contrib_data_p = (
                (
                    promo_kpi_series.loc[contrib_mask_p]
                    / ts_promo.loc[contrib_mask_p, self.y_col]
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            updated_prior_dir = os.path.join(roi_dir, "006_updated_prior")
            os.makedirs(updated_prior_dir, exist_ok=True)
            plots.plot_prior_distributions(
                roi_data_p,
                contrib_data_p,
                os.path.join(
                    updated_prior_dir, f"prior_distributions_{d_col}.png"
                ),
                d_col,
            )
            p_recs = []
            for is_contrib, p_t, p_d in [
                (True, "contribution", contrib_data_p),
                (False, "roi", roi_data_p),
            ]:
              if len(p_d) > 0:
                mu_v, std_v = stats.norm.fit(p_d)
                p_recs.append({
                    "treatment": d_col,
                    "type": p_t,
                    "distribution": "Normal",
                    "mu": mu_v,
                    "sigma": std_v,
                    "alpha": np.nan,
                    "beta": np.nan,
                })

                p_log = p_d[p_d > 0]
                if len(p_log) > 0:
                  shape, _, scale = stats.lognorm.fit(p_log, floc=0)
                  p_recs.append({
                      "treatment": d_col,
                      "type": p_t,
                      "distribution": "LogNormal",
                      "mu": np.log(scale),
                      "sigma": shape,
                      "alpha": np.nan,
                      "beta": np.nan,
                  })

                if is_contrib:
                  p_beta = p_d[(p_d > 0) & (p_d < 1)]
                  if len(p_beta) > 10:
                    a_val, b_val, _, _ = stats.beta.fit(
                        p_beta, floc=0, fscale=1
                    )
                    p_recs.append({
                        "treatment": d_col,
                        "type": p_t,
                        "distribution": "Beta",
                        "mu": np.nan,
                        "sigma": np.nan,
                        "alpha": a_val,
                        "beta": b_val,
                    })

            if p_recs:
              pd.DataFrame(p_recs).to_csv(
                  os.path.join(updated_prior_dir, "prior_parameters.csv"),
                  index=False,
              )

    df_1a_consolidated.to_csv(
        os.path.join(self.output_dir, io.FULL_DF_FILE), index=False
    )

    geo_summary_df = self._generate_dim_summary(
        self.geo_col, df_1a_consolidated, best_model_data_dict, df_3b_concat
    )
    if geo_summary_df is not None:
      geo_summary_df.to_csv(
          os.path.join(self.output_dir, io.GEO_SUMMARY_FILE),
          index=False,
      )

    item_summary_df = self._generate_dim_summary(
        self.item_col, df_1a_consolidated, best_model_data_dict, df_3b_concat
    )
    if item_summary_df is not None:
      item_summary_df.to_csv(
          os.path.join(self.output_dir, io.ITEM_SUMMARY_FILE),
          index=False,
      )

    if self.date_col and self.date_col in df_1a_consolidated.columns:
      print("  ├─ Generating timeseries charts with CIs...", flush=True)
      plots.plot_incremental_kpi_line(
          df_1a_consolidated,
          self.date_col,
          "total_estimated_incremental_KPI",
          "total_estimated_incremental_KPI_2.5%",
          "total_estimated_incremental_KPI_97.5%",
          os.path.join(
              total_graph_dir, "total_incremental_KPI_timeseries_line.png"
          ),
          "Total Incremental KPI Timeseries (Line)",
          consolidated_gt_col,
          actual_y_col=self.y_col,
      )
      plots.plot_incremental_kpi_bar_stacked(
          df_1a_consolidated,
          self.date_col,
          individual_kpi_cols,
          os.path.join(
              total_graph_dir, "total_incremental_KPI_timeseries_bar.png"
          ),
          "Total Incremental KPI Component Stacked (Bar)",
          consolidated_gt_col,
          actual_y_col=self.y_col,
      )
      plots.plot_roi_timeseries_bar(
          df_1a_consolidated,
          self.date_col,
          "total_input_aggregate",
          "total_estimated_incremental_KPI",
          "total_estimated_incremental_KPI_2.5%",
          "total_estimated_incremental_KPI_97.5%",
          os.path.join(total_graph_dir, "total_ROI_timeseries_bar.png"),
          "Total Input, KPI, and ROI Timeseries",
          consolidated_gt_col,
      )

      for e_col, e_dir in [
          (self.geo_col, geo_graph_dir),
          (self.item_col, item_graph_dir),
      ]:
        if e_col and e_col in df_1a_consolidated.columns:
          for i, e_val in enumerate(
              df_1a_consolidated[e_col].dropna().unique(), 1
          ):
            e_df = df_1a_consolidated[df_1a_consolidated[e_col] == e_val]
            plots.plot_incremental_kpi_line(
                e_df,
                self.date_col,
                "total_estimated_incremental_KPI",
                "total_estimated_incremental_KPI_2.5%",
                "total_estimated_incremental_KPI_97.5%",
                os.path.join(
                    e_dir,
                    f"{i:03d}_{e_val}_incremental_KPI_timeseries_line.png",
                ),
                f"[{e_val}] Total Incremental KPI",
                consolidated_gt_col,
                actual_y_col=self.y_col,
            )
            plots.plot_incremental_kpi_bar_stacked(
                e_df,
                self.date_col,
                individual_kpi_cols,
                os.path.join(
                    e_dir, f"{i:03d}_{e_val}_incremental_KPI_timeseries_bar.png"
                ),
                f"[{e_val}] Stacked Component KPI",
                consolidated_gt_col,
                actual_y_col=self.y_col,
            )
            plots.plot_roi_timeseries_bar(
                e_df,
                self.date_col,
                "total_input_aggregate",
                "total_estimated_incremental_KPI",
                "total_estimated_incremental_KPI_2.5%",
                "total_estimated_incremental_KPI_97.5%",
                os.path.join(e_dir, f"{i:03d}_{e_val}_ROI_timeseries_bar.png"),
                f"[{e_val}] Input, KPI & ROI",
                consolidated_gt_col,
            )
          plots.plot_entity_roi_comparison(
              df_1a_consolidated,
              e_col,
              "total_input_aggregate",
              "total_estimated_incremental_KPI",
              "total_estimated_incremental_KPI_2.5%",
              "total_estimated_incremental_KPI_97.5%",
              os.path.join(e_dir, f"000_{e_col}_ROI_comparison.png"),
              f"Cross-{e_col} ROI & Impact Comparison",
              consolidated_gt_col,
          )

    all_cate_covs = list(set().union(*self.covariates_for_cate_list))
    # Covariate typing uses the raw-value frame (`df_est`).
    cont_cols = [c for c in all_cate_covs if df_est[c].nunique() > 2]
    bin_cols = [c for c in all_cate_covs if df_est[c].nunique() <= 2]
    print("  ├─ Generating CATE Scatter Matrix...", flush=True)
    cate_axis_limits = plots.plot_cate_scatter_matrix(
        df_1a_consolidated,
        self.d_cols,
        cont_cols,
        bin_cols,
        cate_eq_dict,
        os.path.join(cate_graph_dir, "cate_scatter_matrix.png"),
        cate_stats_dict,
    )
    # The companion chart: the response curve behind the dots, per binary
    # covariate group, on the same axes.  Under a saturation family the
    # curve is theta * fbar(T / c; alpha_hat, T_ref) in RAW treatment units
    # (normalization bridge: c is the row's scaler divisor); otherwise a line.
    print("  ├─ Generating CATE Response Curves...", flush=True)
    plots.plot_cate_response_curves(
        df_1a_consolidated,
        self.d_cols,
        cont_cols,
        bin_cols,
        cate_eq_dict,
        os.path.join(cate_graph_dir, "cate_response_curves.png"),
        cate_stats_dict,
        response_specs=self._cate_response_specs(df_1a_consolidated),
        axis_limits=cate_axis_limits,
    )

    print("  ├─ Generating Prior Distributions & CSV...", flush=True)
    prior_records = []

    if self.date_col and self.date_col in df_1a_consolidated.columns:
      ts_df = df_1a_consolidated.groupby(self.date_col, observed=False).sum(
          numeric_only=True
      )
    else:
      ts_df = df_1a_consolidated

    base_channels = {}
    for d_col in self.d_cols:
      base_name = re.sub(r"_l\d+$", "", d_col)
      if base_name not in base_channels:
        base_channels[base_name] = []
      base_channels[base_name].append(d_col)

    for base_name, group_cols in base_channels.items():
      kpi_cols = [f"estimated_incremental_KPI_{c}" for c in group_cols]
      total_inc_kpi = ts_df[kpi_cols].sum(axis=1)

      input_col = (
          ts_df[base_name]
          if base_name in ts_df.columns
          else pd.Series(np.zeros(len(ts_df)), index=ts_df.index)
      )

      roi_mask = input_col > 0
      roi_data = (
          (total_inc_kpi.loc[roi_mask] / input_col.loc[roi_mask])
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
          .values
      )

      contrib_mask = (input_col > 0) & (ts_df[self.y_col] > 0)
      contrib_data = (
          (
              total_inc_kpi.loc[contrib_mask]
              / ts_df.loc[contrib_mask, self.y_col]
          )
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
          .values
      )

      plots.plot_prior_distributions(
          roi_data,
          contrib_data,
          os.path.join(prior_graph_dir, f"prior_distributions_{base_name}.png"),
          base_name,
      )

      for is_contrib, p_type, p_data in [
          (True, "contribution", contrib_data),
          (False, "roi", roi_data),
      ]:
        if len(p_data) == 0:
          continue
        mu_val, std = stats.norm.fit(p_data)
        prior_records.append({
            "treatment": base_name,
            "type": p_type,
            "distribution": "Normal",
            "mu": mu_val,
            "sigma": std,
            "alpha": np.nan,
            "beta": np.nan,
        })

        p_log = p_data[p_data > 0]
        if len(p_log) > 0:
          shape, _, scale = stats.lognorm.fit(p_log, floc=0)
          prior_records.append({
              "treatment": base_name,
              "type": p_type,
              "distribution": "LogNormal",
              "mu": np.log(scale),
              "sigma": shape,
              "alpha": np.nan,
              "beta": np.nan,
          })

        if is_contrib:
          p_beta = p_data[(p_data > 0) & (p_data < 1)]
          if len(p_beta) > 10:
            a_val, b_val, _, _ = stats.beta.fit(p_beta, floc=0, fscale=1)
            prior_records.append({
                "treatment": base_name,
                "type": p_type,
                "distribution": "Beta",
                "mu": np.nan,
                "sigma": np.nan,
                "alpha": a_val,
                "beta": b_val,
            })

    if prior_records:
      pd.DataFrame(prior_records).to_csv(
          os.path.join(self.output_dir, io.PRIOR_FILE), index=False
      )

    if self.run_sensitivity:
      print("  ├─ Running Sensitivity Analysis Contours...", flush=True)
      sensitivity_records = []
      # The sensitivity / GATE objects belong to the FITTED model, whose rows
      # are `df_fit` (D_inf under a Phase 0 split); `df_est` is the full raw
      # frame, so the fitted rows are selected explicitly here.
      df_fitted_raw = df_est.loc[df_fit.index]
      for idx, d_col in enumerate(self.d_cols):
        # The existing OVB logic is applied unchanged to
        # the Phase 1-3 `DoubleMLPLR` as fitted -- D = fbar(T; alpha_hat)
        # when Phase 0 adopted a family, D = T otherwise.  No separate linear
        # PLR is fitted for the sensitivity analysis; the only extra fit is
        # the ONE short model of the group benchmark below.
        dml_model = best_model_data_dict[d_col]["model_obj"]
        sens_info = self._sensitivity_treatment_info(d_col)
        saturated = (
            sens_info["treatment_variable"]
            == sensitivity.TREATMENT_VARIABLE_FBAR
        )
        benchmarks = (
            self.x_cols_list[idx] if idx < len(self.x_cols_list) else []
        )

        top_bench = None
        if benchmarks:
          num_bench = [
              b
              for b in benchmarks
              if pd.api.types.is_numeric_dtype(df_fitted_raw[b]) and b != d_col
          ]
          if num_bench:
            # The benchmark covariate is the observed
            # covariate most correlated with the PLR's treatment variable:
            # the fbar(T_i; alpha_hat) column Phases 1-3 regressed on
            # (`df_fit[d_col]`) in the saturation version, the raw treatment
            # column in the linear version (unchanged).
            if saturated:
              corr_target = df_fit.loc[df_fitted_raw.index, d_col]
            else:
              corr_target = df_fitted_raw[d_col]
            top_bench = [
                df_fitted_raw[num_bench].corrwith(corr_target).abs().idxmax()
            ]

        contour_kwargs = {
            "treatment_variable": sens_info["treatment_variable"],
            "t_ref": sens_info["T_ref"] if saturated else None,
        }

        # Group-specific sensitivity benchmark: the short model (without the
        # benchmark covariate Z) is fitted ONCE per treatment with the same
        # learners, folds and seed as the long model; the benchmark point of
        # every scope is then computed from the long / short out-of-fold
        # residuals restricted to that scope.  2 S(K) fits instead of
        # 2 S(K) (1 + n_geo + n_item).
        dml_short = None
        residuals_long = residuals_short = None
        if top_bench:
          try:
            dml_short = sensitivity.fit_short_model(dml_model, top_bench)
            residuals_long = sensitivity.residuals(dml_model)
            residuals_short = sensitivity.residuals(dml_short)
          except (
              ValueError,
              KeyError,
              AttributeError,
              TypeError,
              RuntimeError,
          ) as e:
            print(
                f"    [Warning] Benchmark (short model) unavailable for"
                f" {d_col}: {e}"
            )
            dml_short = None
            residuals_long = residuals_short = None

        # `res_long` / `res_short` / `treatment` are defaults on purpose: they
        # bind this iteration's values at definition time (the closure lives in
        # the `d_col` loop) and are never passed by callers.
        def _benchmark_kwargs(
            theta_long: float,
            theta_short: Optional[float],
            mask: Optional[np.ndarray],
            res_long=residuals_long,
            res_short=residuals_short,
            treatment=d_col,
        ) -> Dict[str, Any]:
          """The group sensitivity benchmark for one scope (empty if none)."""
          if res_long is None or res_short is None or theta_short is None:
            return {}
          try:
            bench = sensitivity.group_benchmark(
                res_long[0],
                res_long[1],
                res_short[0],
                res_short[1],
                theta_long,
                theta_short,
                mask=mask,
            )
          except ValueError as e:
            print(f"    [Warning] Group benchmark skipped for {treatment}: {e}")
            return {}
          return {
              "benchmark_cf_d": bench.cf_d,
              "benchmark_cf_y": bench.cf_y,
              "benchmark_rho": bench.rho,
              "benchmark_scope": bench.scope,
          }

        if "total" in self.sensitivity_scope:
          theta_long_total = float(np.asarray(dml_model.coef).reshape(-1)[0])
          theta_short_total = (
              float(np.asarray(dml_short.coef).reshape(-1)[0])
              if dml_short is not None
              else None
          )
          sens_res = plots.plot_sensitivity_contour(
              dml_model,
              os.path.join(
                  sens_graph_dir, f"{d_col}_total_sensitivity_contour.png"
              ),
              f"{d_col} (Total)",
              top_bench,
              **contour_kwargs,
              **_benchmark_kwargs(theta_long_total, theta_short_total, None),
          )
          if sens_res:
            sens_res.update({
                "treatment": d_col,
                "scope": "total",
                "group": "Total",
                "benchmark_covariate": top_bench[0] if top_bench else None,
                **sens_info,
            })
            sensitivity_records.append(sens_res)

        for scope, col in [("geo", self.geo_col), ("item", self.item_col)]:
          if scope in self.sensitivity_scope and col in df_fitted_raw.columns:
            try:
              groups = pd.get_dummies(df_fitted_raw[col])
              gate_res = dml_model.gate(groups=groups)
              # theta^s_g: `gate` applied to the short model.
              gate_res_short = (
                  dml_short.gate(groups=groups)
                  if dml_short is not None
                  else None
              )
              for group_idx, group_name in enumerate(groups.columns):
                safe_name = str(group_name).replace("/", "_").replace("\\", "_")
                gate_theta = gate_res.coef[group_idx]
                gate_se = gate_res.se[group_idx]
                gate_theta_short = (
                    float(gate_res_short.coef[group_idx])
                    if gate_res_short is not None
                    else None
                )
                # GATE rows use theta_g,
                # se_g, n_g and the group's own residual moments sigma2_g,
                # nu2_g over the group's rows (never se_g sqrt(n_g)).
                gate_mask = groups[group_name].to_numpy(dtype=bool)
                gate_n_obs = int(gate_mask.sum())

                sens_res = plots.plot_sensitivity_contour(
                    dml_model,
                    os.path.join(
                        sens_graph_dir,
                        f"{d_col}_{safe_name}_sensitivity_contour.png",
                    ),
                    f"{d_col} ({scope.capitalize()}: {group_name})",
                    top_bench,
                    override_theta=gate_theta,
                    override_se=gate_se,
                    override_n_obs=gate_n_obs,
                    group_mask=gate_mask,
                    **contour_kwargs,
                    **_benchmark_kwargs(
                        float(gate_theta), gate_theta_short, gate_mask
                    ),
                )
                if sens_res:
                  sens_res.update({
                      "treatment": d_col,
                      "scope": scope,
                      "group": group_name,
                      "benchmark_covariate": (
                          top_bench[0] if top_bench else None
                      ),
                      **sens_info,
                  })
                  sensitivity_records.append(sens_res)
            except (
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
                RuntimeError,
            ) as e:
              print(f"    [Warning] Failed GATE sensitivity for {scope}: {e}")

      if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            os.path.join(self.output_dir, io.SENSITIVITY_FILE), index=False
        )

    metadata_path = os.path.join(self.output_dir, io.METADATA_FILE)

    date_col_min = (
        str(df_1a_consolidated[self.date_col].min())
        if self.date_col and self.date_col in df_1a_consolidated.columns
        else "N/A"
    )
    date_col_max = (
        str(df_1a_consolidated[self.date_col].max())
        if self.date_col and self.date_col in df_1a_consolidated.columns
        else "N/A"
    )
    geos = (
        df_1a_consolidated[self.geo_col].dropna().unique().tolist()
        if self.geo_col and self.geo_col in df_1a_consolidated.columns
        else []
    )
    items = (
        df_1a_consolidated[self.item_col].dropna().unique().tolist()
        if self.item_col and self.item_col in df_1a_consolidated.columns
        else []
    )
    months = []
    if self.date_col and self.date_col in df_1a_consolidated.columns:
      months = (
          pd.to_datetime(df_1a_consolidated[self.date_col])
          .dt.strftime("%Y-%m")
          .dropna()
          .unique()
          .tolist()
      )

    gt_metrics_dict = None
    if self.ground_truth_existence:
      gt_metrics_dict = {
          k: float(v) if pd.notnull(v) else None
          for k, v in total_row_dict.items()
          if k in ["SMAPE", "R_squared", "MAE", "RMSE"]
      }

    treatment_spend_totals = {}
    if getattr(self, "treatment_spend_cols", None):
      for t, spend_col in self.treatment_spend_cols.items():
        if spend_col in df_1a_consolidated.columns:
          treatment_spend_totals[t] = float(df_1a_consolidated[spend_col].sum())

    metadata = {
        "treatment_cols": self.d_cols,
        "outcome_col": self.y_col,
        "date_col_min": date_col_min,
        "date_col_max": date_col_max,
        "x_cols_list": self.x_cols_list,
        "treatment_types": self.treatment_types,
        "treatment_spend_cols": getattr(self, "treatment_spend_cols", {}),
        "treatment_spend_totals": treatment_spend_totals,
        "geo_col": self.geo_col,
        "item_col": self.item_col,
        "geo_name_col": self.geo_name_col,
        "item_name_col": self.item_name_col,
        "covariates_for_cate_list": self.covariates_for_cate_list,
        "geos": geos,
        "items": items,
        "months": months,
        "ground_truth_existence": self.ground_truth_existence,
        "ground_truth_metrics": gt_metrics_dict,
        "peak_month": self.peak_months,
        "date_col": self.date_col,
        "population_col": getattr(scaler, "population_col", None),
        "opt_threshold_roi": getattr(self, "opt_threshold_roi", None),
        "optimization": self.optimize,
        "sensitivity_scope": getattr(self, "sensitivity_scope", ["total"]),
        # Consumed by the HTML report. Always emitted; the
        # saturation entries are null/empty on the family=None path.
        "saturation_family": self.sat_family,
        "saturation_alpha": dict(self._alpha_hat),
        "simple_optimization": self.simple_optimization,
        "budget_increment": self.budget_increment,
        "budget_mode": self.budget_mode,
        "budget_reallocate": self.budget_reallocate,
        "opt_geos": self.opt_geos,
        "opt_items": self.opt_items,
    }
    # The Cluster DML keys are selected automatically for EVERY run
    # (with or without saturation_config), so they are always recorded here;
    # `keys_used` are the columns the runner actually clustered on.
    plan = getattr(self, "_cluster_plan", None)
    if plan is not None:
      cluster_meta = plan.to_metadata(
          item_col=self.item_col, geo_col=self.geo_col, date_col=self.date_col
      )
      cluster_meta["keys_used"] = self._resolve_cluster_cols(df_fit)
      metadata["cluster_dml"] = saturation_evaluation.to_jsonable(cluster_meta)
    if self.sat_family is not None:
      # Phase 0 decision and the Phase 4 path
      # (linear fallback) for the HTML Phase 0 tab and descriptive block.
      # Only added when a family is configured: family=None metadata is
      # unchanged.
      metadata["saturation_selected_family"] = dict(self._selected_family)
      metadata["saturation_phase0"] = {
          d: saturation_evaluation.to_jsonable({
              "decision": r.decision,
              # Why the family was (not) adopted
              # and the normalization reference point of fbar.
              "decision_reason": r.decision_reason,
              "selected_family": r.selected_family,
              "alpha_hat": r.alpha_hat,
              "half_saturation_T": r.half_saturation_t,
              "T_ref": r.t_ref,
              "grid_edge_hit": r.grid_edge_hit,
              "d_bar": r.d_bar,
              "se": r.se,
              "linear_test_k": r.linear_test_k,
              "r_T": r.r_t,
              "output_dir": os.path.relpath(
                  self._phase0_out_dirs.get(d, self.output_dir),
                  self.output_dir,
              ),
          })
          for d, r in self._alpha_results.items()
      }
      metadata["phase4_optimization"] = saturation_evaluation.to_jsonable(
          dict(self._phase4_optimization)
      )
    with open(metadata_path, "w") as f:
      json.dump(metadata, f, indent=4)

    report_path = None
    if self.generate_html:
      print("  ├─ Generating HTML Report...", flush=True)
      report_path = html_extraction.generate_html_report(self.output_dir)
      if report_path:
        print(f"  │    Saved: {report_path}", flush=True)

    agent_dir = None
    if self.generate_agent:
      print("  ├─ Generating ADK Analysis Agent...", flush=True)
      agent_dir = agent_generator.generate_agent(self.output_dir)
      print(f"  │    Saved: {agent_dir}", flush=True)

    print(
        "======================================================================\n★★★"
        " Full Pipeline Execution Finished ★★★",
        flush=True,
    )
    if report_path or agent_dir:
      print("\nNext Steps (Interactive AI Analysis):", flush=True)
      if report_path:
        print(
            f"  - Browser (no server needed): Open {report_path} ->"
            " 'AI Assistant (Gemini)' tab and enter a Gemini API key.",
            flush=True,
        )
      if agent_dir:
        print(
            "  - ADK Web UI (optional):      To launch later, run"
            f" `adk web {agent_dir.parent}` in your terminal.",
            flush=True,
        )
    print("", flush=True)
