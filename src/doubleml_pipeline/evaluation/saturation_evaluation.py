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

"""Phase 0 output files in ``step0_saturation/``.

Every Phase 0 run writes, regardless of warnings, the summary
JSON, the CSV tables and the figures listed in
:mod:`doubleml_pipeline.utils.io`.  Every file name carries a
prefix that states what it belongs to: ``diag1_`` ... ``diag6_`` (the
identification diagnostics), ``summary_`` (the result of the alpha selection)
or ``others_`` (supporting material without a diagnostic rule).

With a single treatment the files live directly in ``step0_saturation/``.
With several treatments each gets its own ``step0_saturation/<treatment>/``
sub-directory (documented assumption: the file set is written per treatment
and ``selected_family`` is per treatment).
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils import io as pipeline_io
from ..visualization import plots

# Diag4 percentiles (shared with alpha_selection).
DIAG4_PERCENTILES = (25, 50, 75, 90)

SUMMARY_JSON = pipeline_io.SUMMARY_JSON
CSV_FILES = (
    pipeline_io.ALPHA_GRID_CSV,
    pipeline_io.RLOSS_CURVES_CSV,
    pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV,
    pipeline_io.FOLD_ASSIGNMENTS_CSV,
    pipeline_io.DIAG3_IDENTIFICATION_CSV,
    pipeline_io.COVERAGE_CHECK_CSV,
)
# Columns of summary_alpha_grid.csv.  The core columns
# (grid_index, half_saturation_T, alpha, fbar_q*, basis_approx_error_alpha,
# in_1se_set) plus the profiled R-loss `mean_r_loss` = L_bar(alpha) and the
# Diag2 `se_alpha` = se(alpha), so the table is the complete data behind the
# R-loss profile figures (the former alpha_profile_<treatment>.csv).
ALPHA_GRID_COLUMNS = (
    "grid_index", "half_saturation_T", "alpha", "mean_r_loss", "se_alpha",
    "in_1se_set", "fbar_q25", "fbar_q50", "fbar_q75", "fbar_q90",
    "basis_approx_error_alpha",
)
# Columns of fold_assignments / coverage_check.
FOLD_ASSIGNMENT_COLUMNS = (
    "seed", "cluster_id", "key1_value", "key2_value", "fold", "split",
)
COVERAGE_COLUMNS = (
    "key", "level", "n_cells_sel", "n_cells_inf", "covered",
    "n_folds_covered_min", "fold_covered",
)
# Diag6: columns of diag6_t_ref_consistency.csv.
DIAG6_COLUMNS = pipeline_io.DIAG6_COLUMNS
# Figures always written by Phase 0 (diag5_* only for tanh / exp).  The grid
# construction and the seed-wise alpha distribution are panels of the alpha
# profile chart (written by the orchestrator), not stand-alone files.
PNG_FILES = (
    pipeline_io.DIAG1_BASIS_APPROX_ERROR_PNG,
    pipeline_io.DIAG2_RLOSS_PROFILE_PNG,
    pipeline_io.DIAG4_F_TRANSFORM_QUANTILES_PNG,
    pipeline_io.SPLIT_COVERAGE_PNG,
)
BOUNDED_FAMILY_PNG_FILES = (pipeline_io.DIAG5_SATURATION_AT_T_REF_PNG,)
# The mandatory keys of summary_alpha_selection.json (in order).  Phases 1-3 /
# Phase 4 read `T_ref` from this file; `se` is the cluster-robust se_obs only.
SUMMARY_KEYS = (
    "family",
    "selected_family",
    "alpha_hat",
    "half_saturation_T",
    "alpha_grid_argmin",
    "rloss_min",
    "refine_method",
    "d_bar",
    "se",
    "linear_test_k",
    "decision",
    "decision_reason",
    "diag1_fired",
    "J",
    "basis_approx_error",
    "diag2_fired",
    "diag2_ratio",
    "r_T",
    "fbar_q25",
    "fbar_q50",
    "fbar_q75",
    "fbar_q90",
    "linear_ref_q25",
    "linear_ref_q50",
    "linear_ref_q75",
    "linear_ref_q90",
    "saturation_at_T_ref",
    "grid_mode",
    "grid_extensions",
    "grid_edge_hit",
    "T_ref",
    "T_max",
    "cluster_keys",
    "cluster_fallback_tier",
    "n_sel",
    "n_inf",
    "n_seeds",
    "k_folds",
    "learners",
)


def treatment_output_dir(root: str, d_col: str, n_treatments: int) -> str:
  """Directory of one treatment's Phase 0 outputs (see module docstring)."""
  return root if n_treatments <= 1 else os.path.join(root, str(d_col))


def to_jsonable(value: Any) -> Any:
  """Recursively converts numpy / NaN values; NaN and inf become ``null``."""
  if isinstance(value, dict):
    return {str(k): to_jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [to_jsonable(v) for v in value]
  if isinstance(value, np.ndarray):
    return [to_jsonable(v) for v in value.tolist()]
  if isinstance(value, (np.bool_, bool)):
    return bool(value)
  if isinstance(value, (np.integer,)):
    return int(value)
  if isinstance(value, (float, np.floating)):
    val = float(value)
    return val if math.isfinite(val) else None
  if value is None or isinstance(value, (int, str)):
    return value
  return str(value)


def build_summary(
    result: Any,
    treatment: str,
    cluster_keys: Sequence[str],
    cluster_fallback_tier: str,
    n_inf: int,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Builds the ``summary_alpha_selection.json`` payload.

  Args:
    result: The ``AlphaSelectionResult``.
    treatment: Treatment column name.
    cluster_keys: Cluster DML keys selected by the cluster-key determination
      rule.
    cluster_fallback_tier: Tier of the cluster-key rule that was applied.
    n_inf: Rows in ``D_inf``.
    extras: Additional documented keys appended at the end (e.g. the Stage A
      split unit and coverage status).

  Returns:
    An ordered dict: the mandatory keys first, then documented extras.
  """
  summary: Dict[str, Any] = {
      "family": result.family,
      "selected_family": result.selected_family,
      "alpha_hat": result.alpha_hat,
      # For the log family the grid variable is T_knee = alpha_hat.
      "half_saturation_T": result.half_saturation_t,
      # Grid argmin before Brent refinement, and L_bar(alpha_hat).
      "alpha_grid_argmin": result.grid_alpha_hat,
      "rloss_min": result.rloss_min,
      "refine_method": result.refine_method,
      "d_bar": result.d_bar,
      # 1-SE linear test: the cluster-robust se_obs is the only se.
      "se": result.se,
      "linear_test_k": result.linear_test_k,
      "decision": result.decision,
      # Linear-test outcome: "1se_pass" | "1se_fail" | "grid_edge_upper" |
      # "basis_ill_conditioned".
      "decision_reason": result.decision_reason,
      # Diag1 (eps(J) on the final grid), Diag2, Diag3,
      # Diag4 (phi_p(alpha_hat) and the linear references q_p / T_ref) and
      # Diag5 (kappa(alpha_hat); None for log / power).
      "diag1_fired": result.diag1_fired,
      "J": result.n_basis,
      "basis_approx_error": result.basis_approx_error,
      "diag2_fired": result.diag2_fired,
      "diag2_ratio": result.diag2_ratio,
      "r_T": result.r_t,
      "fbar_q25": result.fbar_quantile(25),
      "fbar_q50": result.fbar_quantile(50),
      "fbar_q75": result.fbar_quantile(75),
      "fbar_q90": result.fbar_quantile(90),
      "linear_ref_q25": result.linear_ref_quantile(25),
      "linear_ref_q50": result.linear_ref_quantile(50),
      "linear_ref_q75": result.linear_ref_quantile(75),
      "linear_ref_q90": result.linear_ref_quantile(90),
      "saturation_at_T_ref": result.saturation_at_t_ref,
      "grid_mode": result.grid.mode if result.grid is not None else None,
      "grid_extensions": result.grid_extensions,
      # Alpha-grid edge hit: "none" | "upper" (linear side) | "lower".
      "grid_edge_hit": result.grid_edge_hit,
      # The reference point of fbar on the scale Phase 0 works on (T / c),
      # and the T-basis domain upper end T_max (the BASIS_T_MAX_QUANTILE
      # quantile of T > 0 of the full data) on the same scale.
      "T_ref": result.t_ref,
      "T_max": result.t_max,
      "cluster_keys": list(cluster_keys),
      "cluster_fallback_tier": cluster_fallback_tier,
      "n_sel": result.n_sel,
      "n_inf": int(n_inf),
      "n_seeds": int(np.size(result.per_seed_alpha)),
      "k_folds": result.k_folds,
      "learners": result.learners,
      # ---- documented extras -------------------------------------------
      "treatment": treatment,
      "se_cluster_key": result.se_cluster_key,
      # The se of the 1-SE decision: `se_method` is "two-way" (the two
      # adopted keys `se_cluster_keys`, Cameron-Gelbach-Miller), "one-way"
      # (`se_cluster_key`, the key with the fewer clusters) or "iid".  Both
      # se values are reported (`se_two_way` is null without two keys);
      # `critical_value` multiplies `se` in the decision (= linear_test_k, or
      # the t value with `se_df` degrees of freedom when
      # `linear_test_t_critical` is on); `decision_under_one_way_se` is what
      # the one-way se would have decided with the same critical value.
      "se_method": result.se_method,
      "se_cluster_keys": (
          list(result.se_cluster_keys) if result.se_cluster_keys else None
      ),
      "se_one_way": result.se_one_way,
      "se_two_way": result.se_two_way,
      "se_df": result.se_df,
      "linear_test_t_critical": result.linear_test_t_critical,
      "critical_value": result.critical_value,
      "decision_under_one_way_se": result.decision_under_one_way_se,
      "alpha_hat_estimated": result.alpha_hat_estimated,
      # The Diag2 threshold in force, the
      # Diag1 tolerance and whether the fixed fallback_half_saturation_T
      # replaced the estimate.
      "flatness_ratio": result.flatness_ratio,
      "basis_tol": result.basis_tol,
      "fallback_half_saturation_T_used": result.fallback_half_saturation_used,
      "grid_edge_unresolved": result.grid_edge_unresolved,
      "basis_mode": result.basis_mode,
      # Diag1 conditioning: cond(B) of the adopted T-basis and whether it
      # reached MAX_BASIS_CONDITION (then decision_reason =
      # "basis_ill_conditioned").  T_max bookkeeping: the quantile T_max was
      # taken at, max_{D_sel} T and the D_sel rows the basis clips to T_max.
      "basis_cond": result.basis_cond,
      "basis_ill_conditioned": result.basis_ill_conditioned,
      "T_max_quantile": result.t_max_quantile,
      "T_max_observed_sel": result.t_max_observed,
      "n_rows_above_T_max": result.n_above_t_max,
      "normalization": result.normalization,
      "n_ml_fits": result.n_ml_fits,
      # The quantile T_ref was taken at, T_ref in RAW treatment units
      # (c * T_ref; None when c varies by row) and the name of the grid
      # variable that `half_saturation_T` holds.
      "t_ref_quantile": result.t_ref_quantile,
      "T_ref_raw": result.t_ref_raw,
      "grid_variable": result.grid_variable,
      "messages": list(result.messages),
      # The split unit and the keys of the Stage B folds.
      "fold_unit": result.fold_unit,
      "fold_keys": (
          list(result.fold_key_names) if result.fold_key_names else None
      ),
  }
  summary.update(extras or {})
  return summary


def _key_lookup(
    ids: Sequence[Any], key_values: Optional[Any]
) -> Dict[str, Any]:
  """Maps ``cluster_id -> (key1_value, key2_value)`` (first occurrence)."""
  if key_values is None:
    return {}
  key1, key2 = key_values
  frame = pd.DataFrame({
      "cluster_id": np.asarray(ids).astype(str),
      "key1_value": np.asarray(key1).astype(str),
      "key2_value": np.asarray(key2).astype(str),
  }).drop_duplicates("cluster_id")
  return {
      row.cluster_id: (row.key1_value, row.key2_value)
      for row in frame.itertuples(index=False)
  }


def fold_assignments_frame(
    result: Any,
    sel_cluster_ids: Sequence[Any],
    inf_cluster_ids: Sequence[Any],
    sel_key_values: Optional[Any] = None,
    inf_key_values: Optional[Any] = None,
) -> pd.DataFrame:
  """``fold_assignments.csv``.

  One row per (seed, split unit).  ``cluster_id`` is the id
  of the intersection cell of the two adopted keys and ``key1_value`` /
  ``key2_value`` give its two key values (empty when the split is not
  cell-level).  ``fold`` is the Stage B fold of a ``D_sel`` cell and ``-1``
  for a ``D_inf`` cell; ``split`` is ``sel`` / ``inf`` (``mixed`` only under
  the observation-level final fallback of the cluster-key rule).

  Args:
    result: The ``AlphaSelectionResult``.
    sel_cluster_ids: Cell / cluster id per ``D_sel`` row (row order of
      ``result.seed_partitions``).
    inf_cluster_ids: Cell / cluster id per ``D_inf`` row.
    sel_key_values: ``(key1, key2)`` values per ``D_sel`` row (defaults to
      ``result.fold_key_values``).
    inf_key_values: ``(key1, key2)`` values per ``D_inf`` row.

  Returns:
    The frame with columns :data:`FOLD_ASSIGNMENT_COLUMNS`.
  """
  if sel_key_values is None:
    sel_key_values = getattr(result, "fold_key_values", None)
  lookup = _key_lookup(sel_cluster_ids, sel_key_values)
  lookup.update(
      {k: v for k, v in _key_lookup(inf_cluster_ids, inf_key_values).items()
       if k not in lookup}
  )
  sel_ids = pd.Series(np.asarray(sel_cluster_ids).astype(str))
  inf_ids = pd.Series(np.asarray(inf_cluster_ids).astype(str))
  sel_set = set(sel_ids.unique())
  inf_set = set(inf_ids.unique())
  mixed = sel_set & inf_set
  inf_only = sorted(inf_set - sel_set)
  frames: List[pd.DataFrame] = []
  for seed, folds in enumerate(result.seed_partitions or []):
    per_cluster = (
        pd.DataFrame({"cluster_id": sel_ids, "fold": np.asarray(folds)})
        .groupby("cluster_id", sort=True)["fold"]
        .agg(lambda v: int(pd.Series(v).mode().iloc[0]))
        .reset_index()
    )
    per_cluster["split"] = np.where(
        per_cluster["cluster_id"].isin(mixed), "mixed", "sel"
    )
    inf_frame = pd.DataFrame({
        "cluster_id": inf_only,
        "fold": -1,
        "split": "inf",
    })
    block = pd.concat([per_cluster, inf_frame], ignore_index=True)
    block.insert(0, "seed", seed)
    frames.append(block)
  if not frames:
    return pd.DataFrame(columns=list(FOLD_ASSIGNMENT_COLUMNS))
  out = pd.concat(frames, ignore_index=True)
  pairs = out["cluster_id"].map(lookup)
  out["key1_value"] = pairs.map(
      lambda p: p[0] if isinstance(p, tuple) else None
  )
  out["key2_value"] = pairs.map(
      lambda p: p[1] if isinstance(p, tuple) else None
  )
  return out[list(FOLD_ASSIGNMENT_COLUMNS)]


def coverage_check_frame(
    result: Any, coverage: Optional[pd.DataFrame]
) -> pd.DataFrame:
  """``coverage_check.csv``.

  For every level of every adopted key: the number of cells on the ``sel`` /
  ``inf`` side (``covered`` is True when both are >= 1, which verifies the
  Stage A coverage guarantee) plus, for the Stage B folds, the minimum
  over seeds of the number of folds the level appears in
  (``n_folds_covered_min``; ``fold_covered`` is True when it equals K).

  Args:
    result: The ``AlphaSelectionResult``.
    coverage: The top-level coverage table of the Stage A split (``None``
      under the observation-level fallback).

  Returns:
    The frame with columns :data:`COVERAGE_COLUMNS` (empty when ``coverage``
    is ``None``).
  """
  if coverage is None or coverage.empty:
    return pd.DataFrame(columns=list(COVERAGE_COLUMNS))
  out = coverage.copy()
  out["level"] = out["level"].astype(str)
  out["n_folds_covered_min"] = np.nan
  names = getattr(result, "fold_key_names", None)
  values = getattr(result, "fold_key_values", None)
  partitions = getattr(result, "seed_partitions", None) or []
  if names and values is not None and partitions:
    for name, key_values in zip(names, values):
      key_values = np.asarray(key_values).astype(str)
      per_seed = [
          pd.DataFrame({"level": key_values, "fold": np.asarray(folds)})
          .groupby("level")["fold"].nunique()
          for folds in partitions
      ]
      min_folds = pd.concat(per_seed, axis=1).min(axis=1)
      mask = out["key"] == name
      out.loc[mask, "n_folds_covered_min"] = (
          out.loc[mask, "level"].map(min_folds).fillna(0).to_numpy()
      )
  k_folds = int(getattr(result, "k_folds", 0) or 0)
  out["fold_covered"] = out["n_folds_covered_min"] >= max(k_folds, 1)
  return out[list(COVERAGE_COLUMNS)]


def write_phase0_outputs(
    result: Any,
    out_dir: str,
    treatment: str,
    cluster_keys: Sequence[str],
    cluster_fallback_tier: str,
    n_inf: int,
    sel_cluster_ids: Sequence[Any],
    inf_cluster_ids: Sequence[Any],
    sel_key_values: Optional[Any] = None,
    inf_key_values: Optional[Any] = None,
    coverage: Optional[pd.DataFrame] = None,
    summary_extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
  """Writes every Phase 0 CSV / JSON / PNG for one treatment.

  Args:
    result: The ``AlphaSelectionResult``.
    out_dir: Destination (``step0_saturation`` or its per-treatment subdir).
    treatment: Treatment column name.
    cluster_keys: Cluster DML keys selected by the cluster-key determination
      rule.
    cluster_fallback_tier: Tier of the cluster-key rule that was applied.
    n_inf: Rows in ``D_inf``.
    sel_cluster_ids: Cell / cluster id per ``D_sel`` row.
    inf_cluster_ids: Cell / cluster id per ``D_inf`` row.
    sel_key_values: ``(key1, key2)`` values per ``D_sel``
      row (defaults to ``result.fold_key_values``).
    inf_key_values: ``(key1, key2)`` values per ``D_inf`` row.
    coverage: The Stage A coverage table (``TopLevelSplit.coverage``).
    summary_extras: Extra keys appended to the summary JSON.

  Returns:
    Mapping ``file name -> path`` of everything written.
  """
  os.makedirs(out_dir, exist_ok=True)
  written: Dict[str, Optional[str]] = {}
  grid = result.grid
  n_grid = int(np.size(result.alpha_grid))

  summary = build_summary(
      result, treatment, cluster_keys, cluster_fallback_tier, n_inf,
      extras=summary_extras,
  )
  path = os.path.join(out_dir, SUMMARY_JSON)
  with open(path, "w") as handle:
    json.dump(to_jsonable(summary), handle, indent=2)
  written[SUMMARY_JSON] = path

  t_half = (
      np.asarray(grid.t_half, float) if grid is not None
      else np.full(n_grid, np.nan)
  )
  # `fbar_q25..q90` = phi_p(alpha) = fbar(q_p(T>0); alpha)
  # per grid point and `in_1se_set` = membership in A_1SE of Diag2;
  # `mean_r_loss` = L_bar(alpha) and `se_alpha` = se(alpha) of Diag2 complete
  # the per-grid-point table (the former alpha_profile_<treatment>.csv).
  phi_grid = result.fbar_quantiles_grid
  if phi_grid is None:
    phi_grid = np.full((len(DIAG4_PERCENTILES), n_grid), np.nan)
  one_se = result.one_se_set
  if one_se is None:
    one_se = np.zeros(n_grid, dtype=bool)
  se_alpha = result.se_alpha
  if se_alpha is None:
    se_alpha = np.full(n_grid, np.nan)
  grid_frame = pd.DataFrame({
      "grid_index": np.arange(n_grid),
      "half_saturation_T": t_half,
      "alpha": result.alpha_grid,
      "mean_r_loss": np.asarray(result.mean_loss, dtype=float),
      "se_alpha": np.asarray(se_alpha, dtype=float),
      "in_1se_set": np.asarray(one_se, dtype=bool),
      "fbar_q25": np.asarray(phi_grid[0], dtype=float),
      "fbar_q50": np.asarray(phi_grid[1], dtype=float),
      "fbar_q75": np.asarray(phi_grid[2], dtype=float),
      "fbar_q90": np.asarray(phi_grid[3], dtype=float),
      "basis_approx_error_alpha": result.basis_approx_error_alpha,
  })
  frames = {pipeline_io.ALPHA_GRID_CSV: grid_frame[list(ALPHA_GRID_COLUMNS)]}
  n_seeds = int(result.per_seed_loss.shape[0])
  # `rloss_identity` = L_hat_{s,T} of the identity
  # candidate (identical across the rows of a seed).
  frames[pipeline_io.RLOSS_CURVES_CSV] = pd.DataFrame({
      "seed": np.repeat(np.arange(n_seeds), n_grid),
      "grid_index": np.tile(np.arange(n_grid), n_seeds),
      "alpha": np.tile(result.alpha_grid, n_seeds),
      "rloss_cate": result.per_seed_loss.reshape(-1),
      "rloss_ate": result.per_seed_loss_ate.reshape(-1),
      "rloss_identity": np.repeat(result.per_seed_loss_identity, n_grid),
  })
  frames[pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV] = pd.DataFrame({
      "seed": np.arange(n_seeds),
      "alpha_hat_s": result.per_seed_alpha,
      "rloss_at_alpha_hat_s": result.per_seed_min_loss,
      "rloss_identity_s": result.per_seed_loss_identity,
      "d_s": result.d_s,
  })
  frames[pipeline_io.FOLD_ASSIGNMENTS_CSV] = fold_assignments_frame(
      result, sel_cluster_ids, inf_cluster_ids,
      sel_key_values=sel_key_values, inf_key_values=inf_key_values,
  )
  coverage_frame = coverage_check_frame(result, coverage)
  frames[pipeline_io.COVERAGE_CHECK_CSV] = coverage_frame
  # Diag3: r_T and its parts, plus the treatment-support statistics
  # the grid was built from (T_max is the basis-domain right end, the
  # BASIS_T_MAX_QUANTILE quantile of T > 0; T_max_observed is max_{D_sel} T).
  frames[pipeline_io.DIAG3_IDENTIFICATION_CSV] = pd.DataFrame([{
      "r_T": result.r_t,
      "sd_T": result.sd_t,
      "sd_T_resid": result.sd_t_resid,
      "share_T_zero": result.share_t_zero,
      "q50_T_pos": result.q50_t_pos,
      "q95_T_pos": result.q95_t_pos,
      "T_max": result.t_max,
      "T_max_observed": result.t_max_observed,
      # The normalization reference point (same scale).
      "T_ref": result.t_ref,
  }])
  for name, frame in frames.items():
    path = os.path.join(out_dir, name)
    frame.to_csv(path, index=False)
    written[name] = path

  written.update(
      plots.write_phase0_figures(result, out_dir, coverage=coverage_frame)
  )
  return written


def write_diag6_output(
    out_dir: str, rows: Sequence[Dict[str, Any]]
) -> Optional[str]:
  """Writes ``diag6_t_ref_consistency.csv`` (Diag6).

  Diag6 runs at the start of Phase 1, so this file is written by the
  orchestrator after Phase 0 has finished -- one row per treatment with
  ``status`` = ``pass`` / ``fail`` / ``skipped`` (``skipped`` when
  ``selected_family = None``: no replacement, hence no check).

  Args:
    out_dir: The treatment's Phase 0 directory.
    rows: Dicts with the :data:`DIAG6_COLUMNS` keys.

  Returns:
    The CSV path, or ``None`` when ``out_dir`` does not exist.
  """
  if not out_dir or not os.path.isdir(out_dir):
    return None
  path = os.path.join(out_dir, pipeline_io.DIAG6_T_REF_CONSISTENCY_CSV)
  frame = pd.DataFrame(list(rows), columns=list(DIAG6_COLUMNS))
  frame.to_csv(path, index=False)
  return path


def update_summary(out_dir: str, updates: Dict[str, Any]) -> Optional[str]:
  """Merges ``updates`` into an existing summary JSON (e.g. the Phase 4 path).

  Args:
    out_dir: Directory holding ``summary_alpha_selection.json``.
    updates: Keys to add / overwrite.

  Returns:
    The JSON path, or ``None`` when no summary exists.
  """
  path = os.path.join(out_dir, SUMMARY_JSON)
  if not os.path.exists(path):
    return None
  with open(path) as handle:
    payload = json.load(handle)
  payload.update(to_jsonable(updates))
  with open(path, "w") as handle:
    json.dump(payload, handle, indent=2)
  return path
