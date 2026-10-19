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

"""Phase 0 of the saturation-aware DML extension.

This module is an *additive* extension of the pipeline.  Nothing in here is
executed when the ``saturation`` config block is omitted, with the single
documented exception of :func:`determine_cluster_keys` (the cluster-key
determination rule also applies to normal runs with ``family=None``).

* Phase 0 / Stage A: the cluster-key determination rule (cluster-count
  driven, top-2 selection), the top-level split ``D = D_sel + D_inf`` on the
  intersection cells of the two adopted keys with its coverage guarantee
  (a single key is used only when a single key is adopted), the warning
  criteria, the three-tier rule for identifier dummies and the final
  fallback.
* Phase 0 / Stage B: alpha selection by profiled R-loss minimization with
  fixed default learners (optional legacy learner selection), the T-basis
  (``basis_dim_T: auto``), the seed loop with cross-fitting (every nuisance
  prediction is out-of-fold; the inner CATE coefficients ``beta^(-k)`` are
  fitted on the out-of-fold residuals of the other folds), the batched
  alpha-dependent algebra on the half-saturation grid with automatic
  extension, aggregation / Brent refinement ("grid + Brent only"), the
  intercept-only diagnostic curve and the 1-SE comparison with the identity
  candidate.
* The identification diagnostics Diag1-Diag6.
* The treatment normalization bridge: every Stage B candidate is the
  normalized ``fbar(T; alpha) = f / f(T_ref; alpha)`` with
  ``T_ref = q95(T | T > 0)`` (grid curves, the identity candidate
  ``fbar_T = T / T_ref``, the intercept curve); the basis coefficients are
  ``c(alpha) = P fbar_alpha`` with the SVD least-squares projector on a fixed
  grid over ``[0, T_max]`` (equally spaced points, quantiles of ``T > 0`` and
  the Greville abscissae), ``T_max = q99.9(T | T > 0)`` of the full data;
  the ``log`` grid variable is ``T_knee = alpha``; an ``alpha_hat`` on the
  linear-side grid edge adopts the linear model
  (``decision_reason = "grid_edge_upper"``), and so does a basis whose
  condition number ``cond(B)`` is at or above
  ``saturation.MAX_BASIS_CONDITION``
  (``decision_reason = "basis_ill_conditioned"``); ``T_ref``, ``T_max``,
  ``cond(B)``, ``grid_edge_hit`` and ``decision_reason`` are reported.

Details of the decision and the diagnostics:

* The treatment without a saturation transform is the *identity
  candidate* ``fbar_T(T) = T / T_ref`` with nuisance ``m_hat_T`` and loss
  ``L_hat_{s,T}`` (``rloss_identity`` in the outputs); the fixed downstream
  value is ``fallback_half_saturation_T`` (a half-saturation point in units
  of ``T``; ``T_knee`` for ``log``, alpha itself for ``power``).
* The standard error of the 1-SE decision and of Diag2 is the
  cluster-robust ``se_obs`` of the per-observation paired differences
  (:func:`cluster_robust_se`): two-way on the two adopted keys
  (Cameron-Gelbach-Miller) when both are known, one-way on the key with the
  fewer clusters otherwise; there is no across-seed se and no se-type
  selector.  The multiplier of ``se`` is ``linear_test_k`` (normal
  reference) or, with ``alpha_selection.linear_test_t_critical: true``, the
  Student-t value with the same tail probability
  (:func:`linear_test_critical_value`).
* The diagnostics are Diag1 (basis approximation error ``eps(J)`` and the
  basis condition number ``cond(B)``), Diag2 (1-SE flatness ratio ``T_1/2^max
  / T_1/2^min`` over ``A_1SE``), Diag3 (``r_T``), Diag4 (shape quantiles
  ``phi_p``), Diag5 (``kappa = f(T_ref; alpha_hat)``) and Diag6 (unit
  consistency of ``T_ref`` at the start of Phase 1,
  :func:`t_ref_consistency`).  Diag1 / Diag2 warn and continue; the
  ``cond(B)`` trigger of Diag1 additionally adopts the linear model
  (``decision_reason = "basis_ill_conditioned"``) because the Stage B
  statistics cannot be trusted; only Diag6 stops.  The intercept curve is
  visualization-only.
* :func:`describe_ml_training_count` renders the ML-training count line of
  the Phase 0 progress display; the messages recorded in ``messages`` carry
  no internal markers (they are written to ``summary_alpha_selection.json``).
"""

import copy
import dataclasses
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import optimize as scipy_optimize
from scipy import stats as scipy_stats
from sklearn import base as sk_base

from ..models import flaml_dml
from ..models import saturation

# Optional GPU/TPU acceleration.  The alpha-dependent part of Stage B is pure
# batched linear algebra, so it is expressed with an array API
# that is shared by numpy and jax.numpy: when jax is importable the very same
# code runs on GPU/TPU and the inner ridge solves are additionally `vmap`-ed
# over the (fold x alpha) batch.
try:  # pragma: no cover - depends on the execution environment.
  # pylint: disable=g-import-not-at-top
  import jax as _JAX
  from jax import numpy as _JNP
  # pylint: enable=g-import-not-at-top
except ImportError:  # pragma: no cover
  _JAX = None
  _JNP = None


# Warning threshold of the cluster-key rule: reliability guideline for
# cluster-robust CIs (Cameron & Miller 2015).
MIN_CLUSTERS_WARNING = 30

# Name of the derived (contiguous date block) cluster key.  The cluster-key
# rule forbids clustering on raw individual dates.
DATE_BLOCK_COL = "cluster_date_block"

# Separator of the intersection-cell id "<key1>|<key2>".
CELL_ID_SEP = "|"

# Helper columns created by Phase 0.  They are dropped from the per-run result
# frames so that the existing Phase 1-4 outputs stay byte-identical.
INTERNAL_HELPER_COLS = (DATE_BLOCK_COL,)

# Diag2 default: fires when T_1/2^max / T_1/2^min over the 1-SE
# set exceeds `flatness_ratio` (T_knee for log).  `power` is scale free, so
# its rule is alpha^max - alpha^min > POWER_FLATNESS_RANGE instead.
DEFAULT_FLATNESS_RATIO = 10.0
POWER_FLATNESS_RANGE = 0.4
# Diag6 default: tolerated relative error between the T_ref
# recomputed at the start of Phase 1 and the T_ref of Phase 0.
DEFAULT_T_REF_TOLERANCE = 1e-9
# Diag4: the percentiles p of the shape quantiles
# phi_p(alpha) = fbar(q_p(T > 0); alpha).
DIAG4_PERCENTILES = (25, 50, 75, 90)


@dataclasses.dataclass
class ClusterKeyPlan:
  """Outcome of the cluster-key determination rule."""

  keys: List[str] = dataclasses.field(default_factory=list)
  primary_key: Optional[str] = None
  cluster_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
  candidate_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
  derived_columns: Dict[str, pd.Series] = dataclasses.field(
      default_factory=dict
  )
  date_block_freq: Optional[str] = None
  messages: List[str] = dataclasses.field(default_factory=list)

  def describe(self) -> str:
    """Returns the one-line description used by the progress display."""
    if not self.keys:
      return "none (no cluster key available -> observation-level inference)"
    parts = [
        f"{k} ({self.cluster_counts.get(k, 0)} clusters)" for k in self.keys
    ]
    way = f"{len(self.keys)}-way"
    return f"{way}: " + ", ".join(parts)

  @property
  def not_adopted(self) -> Dict[str, int]:
    """Candidate keys (with their cluster counts) that were NOT adopted."""
    return {
        k: int(v)
        for k, v in self.candidate_counts.items()
        if k not in self.keys
    }

  def describe_selection(self) -> str:
    """Returns the full log line: adopted keys, rejected candidates, blocks.

    Example::

        2-way: cluster_date_block (18 clusters), geo (6 clusters);
        not adopted: item (2 clusters); date block = W
    """
    text = self.describe()
    rest = self.not_adopted
    if rest:
      text += "; not adopted: " + ", ".join(
          f"{k} ({n} clusters)" for k, n in rest.items()
      )
    if self.date_block_freq:
      text += f"; date block = {self.date_block_freq}"
    return text

  def to_metadata(
      self,
      item_col: Optional[str] = None,
      geo_col: Optional[str] = None,
      date_col: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Returns the ``metadata.json`` payload describing the Cluster DML keys.

    Emitted for EVERY run (with or without a ``saturation`` config block) so
    users can see which keys the 2-way Cluster DML actually clustered on,
    which candidates were rejected and why, and that ``date_col`` /
    ``item_col`` / ``geo_col`` keep their visualization role regardless.

    Args:
      item_col: The item column of the run (visualization dimension).
      geo_col: The geo column of the run (visualization dimension).
      date_col: The date column of the run (visualization dimension; the
        derived ``cluster_date_block`` is built from it).

    Returns:
      A JSON-serializable dictionary.
    """
    return {
        "applies_to": (
            "every run: the cluster-key determination rule selects the Cluster"
            " DML keys whether or not saturation_config is given"
        ),
        "rule": (
            "candidates = item_col and geo_col (each needs >= 2 distinct"
            " values) plus a derived contiguous date block built from"
            " date_col (never raw dates); the two candidates with the MOST"
            " clusters are adopted (2-way Cluster DML), one if only one is"
            " usable, none -> observation-level inference"
        ),
        "keys": list(self.keys),
        "n_way": int(len(self.keys)),
        "primary_key": self.primary_key,
        "cluster_counts": {k: int(v) for k, v in self.cluster_counts.items()},
        "candidate_counts": {
            k: int(v) for k, v in self.candidate_counts.items()
        },
        "not_adopted": self.not_adopted,
        "date_block_granularity": self.date_block_freq,
        "derived_columns": list(self.derived_columns),
        "visualization_cols": {
            "date_col": date_col,
            "item_col": item_col,
            "geo_col": geo_col,
            "note": (
                "date_col / item_col / geo_col are the dimensions of every"
                " chart and summary CSV; the Cluster DML keys above only"
                " affect the cross-fitting folds and the standard errors"
            ),
        },
        "messages": list(self.messages),
    }

  def attach(self, df: pd.DataFrame) -> pd.DataFrame:
    """Returns a *copy* of ``df`` with the derived key columns attached.

    Args:
      df: Any frame sharing the index of the frame the plan was built from.

    Returns:
      A copy of ``df`` with the derived cluster columns added.  The input frame
      is never modified in place.
    """
    if not self.derived_columns:
      return df
    out = df.copy()
    for name, series in self.derived_columns.items():
      if name not in out.columns:
        out[name] = series.reindex(out.index)
    return out


@dataclasses.dataclass
class TopLevelSplit:
  """Outcome of the Stage A top-level split.

  With two adopted keys the split unit is the intersection
  cell ``key1 x key2`` (``unit == "cell"``); ``split_key`` is then the
  display label ``"key1 × key2"`` and ``split_keys`` holds both names.
  ``n_clusters_*`` count split units (cells, single-key levels, or 0 under
  the observation-level fallback).
  """

  idx_sel: pd.Index
  idx_inf: pd.Index
  split_key: Optional[str] = None
  level: str = "cluster"  # "cluster" | "observation"
  tier: str = ""
  forced_full_sample: bool = False
  n_clusters_total: int = 0
  n_clusters_sel: int = 0
  messages: List[str] = dataclasses.field(default_factory=list)
  # Cell-level split fields (defaulted so existing call sites keep working).
  unit: str = "cell"  # "cell" | "single_key" | "observation"
  split_keys: List[str] = dataclasses.field(default_factory=list)
  # Cell id ("<key1 value>|<key2 value>") per row of the split frame.
  cell_ids: Optional[pd.Series] = None
  # One row per level of every key: key, level, n_cells_sel, n_cells_inf,
  # covered (both sides >= 1).  Written as coverage_check.csv.
  coverage: Optional[pd.DataFrame] = None
  coverage_ok: bool = False


@dataclasses.dataclass
class AlphaSelectionResult:
  """Outcome of the Stage B profiled R-loss minimization.

  Field names map to the mathematical notation as follows: ``alpha_hat``,
  ``T_ref`` (``t_ref``), ``T_max`` (``t_max``), ``L_bar`` (``mean_loss``),
  ``L_hat_{s,T}`` (``per_seed_loss_identity``), ``d_s`` / ``d_bar`` / ``d_i``
  (``d_obs``), ``se`` (= ``se_obs``), ``A_1SE`` (``one_se_set``), ``eps(J)``
  (``basis_approx_error``), ``r_T`` (``r_t``), ``phi_p`` (``fbar_q*``),
  ``kappa`` (``saturation_at_t_ref``).
  """

  alpha_hat: float
  alpha_grid: np.ndarray
  mean_loss: np.ndarray
  per_seed_loss: np.ndarray
  per_seed_alpha: np.ndarray
  grid_alpha_hat: float
  refined: bool = False
  ml_model_y: Optional[str] = None
  ml_model_t: Optional[str] = None
  # Normalization bridge: the scale `alpha_hat` was estimated on.
  # "none" means raw treatment units; otherwise alpha refers to `T / c` and
  # `saturation.rescale_alpha_to_raw` maps it back for Phase 4.
  normalization: str = "none"
  # Compute accounting: the realised number of ML fits.
  n_ml_fits: int = 0
  # False when the treatment learner needed the per-output fallback.
  multioutput_native: bool = True
  # Warnings / notes of the run (written to summary_alpha_selection.json;
  # not part of the progress display).
  messages: List[str] = dataclasses.field(default_factory=list)
  # ------------------------------------------------------------------
  # Phase 0 returns (selected_family, alpha_hat).
  # ------------------------------------------------------------------
  family: str = ""
  selected_family: Optional[str] = None
  decision: str = "saturation"  # "saturation" | "linear"
  # argmin + Brent BEFORE a fallback_half_saturation_T override.
  alpha_hat_estimated: float = float("nan")
  refine_method: str = "none"  # "brent" | "none"
  # The final grid (after automatic extensions).
  grid: Optional[saturation.AlphaGrid] = None
  grid_extensions: int = 0
  grid_edge_unresolved: bool = False
  # Intercept-only curve L_hat^ATE_s(alpha).
  # Visualization only: no decision, no argmin marker.
  per_seed_loss_ate: Optional[np.ndarray] = None
  mean_loss_ate: Optional[np.ndarray] = None
  # Identity candidate fbar_T = T / T_ref
  # (L_hat_{s,T} per seed, L_bar_T) and the 1-SE comparison.
  per_seed_loss_identity: Optional[np.ndarray] = None
  mean_loss_identity: float = float("nan")
  per_seed_loss_at_alpha_hat: Optional[np.ndarray] = None
  per_seed_min_loss: Optional[np.ndarray] = None
  d_s: Optional[np.ndarray] = None
  d_bar: float = float("nan")
  # 1-SE linear test: per-observation paired differences d_i (n_sel,) and
  # the cluster-robust se_obs computed from them.  `se` is the se the decision
  # used: the TWO-WAY cluster-robust se on the two adopted keys
  # (`se_cluster_keys`, Cameron-Gelbach-Miller) when both are known, otherwise
  # the one-way se on `se_cluster_key` (the key with the fewer clusters), or
  # the iid se.  `se_one_way` is always reported (the one-way se on
  # `se_cluster_key`, iid without a key); `se_two_way` is NaN when not
  # available.  The decision is d_bar < -critical_value * se, where
  # critical_value = linear_test_k (normal reference, default) or the t value
  # t_{se_df}^{-1}(Phi(k)) when `alpha_selection.linear_test_t_critical` is
  # on (`se_df` = number of clusters - 1; min over the two keys for two-way).
  # `decision_under_one_way_se` is the decision the one-way se would have
  # given with the same critical value (reported for comparison only).
  d_obs: Optional[np.ndarray] = None
  se: float = float("nan")
  linear_test_k: float = 1.0
  se_cluster_key: Optional[str] = None
  se_method: str = "iid"  # "two-way" | "one-way" | "iid"
  se_cluster_keys: Optional[Tuple[str, str]] = None
  se_one_way: float = float("nan")
  se_two_way: float = float("nan")
  se_df: int = 0
  linear_test_t_critical: bool = False
  critical_value: float = 1.0
  decision_under_one_way_se: str = "linear"
  # Diag2: se(alpha) per grid point (paired against
  # alpha_hat, same formula), the 1-SE set A_1SE, the ratio and the flag.
  se_alpha: Optional[np.ndarray] = None
  one_se_set: Optional[np.ndarray] = None
  flatness_ratio: float = DEFAULT_FLATNESS_RATIO  # the configured threshold
  diag2_ratio: float = float("nan")
  diag2_fired: bool = False
  # T-basis / Diag1: adopted J (fixed after the
  # initial grid), eps(J) recomputed on the FINAL grid, basis_tol, the flag.
  n_basis: int = 0
  basis_approx_error: float = float("nan")
  basis_approx_error_alpha: Optional[np.ndarray] = None
  basis_mode: str = "fixed"
  basis_satisfied: bool = True  # eps(J) < basis_tol on the INITIAL grid
  basis_tol: float = 1e-3
  diag1_fired: bool = False
  # cond(B) of the adopted basis (largest / smallest singular value of the
  # projection-grid basis matrix) and whether it reached
  # `saturation.MAX_BASIS_CONDITION` (then the linear model is adopted,
  # decision_reason = "basis_ill_conditioned").
  basis_cond: float = float("nan")
  basis_ill_conditioned: bool = False
  # Diag3: r_T = sd(T~) / sd(T / T_ref) and its parts
  # (sd_t / sd_t_resid on the Phase 0 scale T / c; the ratio is scale free).
  r_t: float = float("nan")
  sd_t: float = float("nan")
  sd_t_resid: float = float("nan")
  share_t_zero: float = float("nan")
  q50_t_pos: float = float("nan")
  q95_t_pos: float = float("nan")
  # The right end T_max of the T-basis domain (`T_max` JSON key): the
  # `saturation.BASIS_T_MAX_QUANTILE` quantile of T > 0 of the FULL data, on
  # the Phase 0 scale (T / c); `t_max_observed` is max_{D_sel} T on the same
  # scale and `n_above_t_max` the D_sel rows the basis clips to T_max.
  t_max: float = float("nan")
  t_max_quantile: float = saturation.BASIS_T_MAX_QUANTILE
  t_max_observed: float = float("nan")
  n_above_t_max: int = 0
  # Diag4: shape quantiles phi_p(alpha) = fbar(q_p(T>0);
  # alpha) for p in DIAG4_PERCENTILES -- per grid point (len(p), G), at
  # alpha_hat (len(p),), and the linear references q_p / T_ref (len(p),).
  fbar_quantiles_grid: Optional[np.ndarray] = None
  fbar_quantiles: Optional[np.ndarray] = None
  linear_ref_quantiles: Optional[np.ndarray] = None
  # Diag5: kappa(alpha_hat) = f(T_ref; alpha_hat) for
  # tanh / exp (share of the ceiling reached at T_ref); None for log / power.
  saturation_at_t_ref: Optional[float] = None
  # True when `fallback_half_saturation_T` replaced
  # the estimated alpha downstream (Diag1 or Diag2 fired and it was set).
  fallback_half_saturation_used: bool = False
  # Learner description (`learners` JSON key) and the
  # small-sample threshold the switch was judged against (display item 3).
  learner_selection: bool = True
  learners: Dict[str, Any] = dataclasses.field(default_factory=dict)
  small_sample_threshold: int = 2000
  # Raw material of fold_assignments.csv / the figures.
  seed_partitions: Optional[List[np.ndarray]] = None
  cluster_labels: Optional[np.ndarray] = None
  t_values: Optional[np.ndarray] = None
  n_sel: int = 0
  k_folds: int = 0
  # "cell" when the folds were built on the
  # intersection cells of the two adopted keys, "cluster" otherwise.  The
  # per-row key values (D_sel row order) feed fold_assignments.csv and the
  # per-fold columns of coverage_check.csv.
  fold_unit: str = "cluster"
  fold_key_names: Optional[Tuple[str, str]] = None
  fold_key_values: Optional[Tuple[np.ndarray, np.ndarray]] = None
  # ------------------------------------------------------------------
  # Normalization bridge and grid-edge reporting.
  # ------------------------------------------------------------------
  # The reference point T_ref = q95(T | T > 0) of
  # fbar = f / f(T_ref; alpha), on the scale Phase 0 works on (T / c).  It is
  # computed once on the FULL data before the split (the caller passes it in);
  # Phases 1-3 and Phase 4 read it back from summary_alpha_selection.json
  # (`T_ref` key).  `t_ref_raw` is c * T_ref when the divisor
  # is a single constant (None otherwise), for the human reader.
  t_ref: float = float("nan")
  t_ref_quantile: float = saturation.DEFAULT_T_REF_QUANTILE
  t_ref_raw: Optional[float] = None
  # Which grid edge alpha_hat sits on after the
  # extensions, in the T_1/2 orientation ("upper" = linear side), and why the
  # final decision was taken.
  grid_edge_hit: str = "none"  # "none" | "upper" | "lower"
  # "1se_pass" | "1se_fail" | "grid_edge_upper" | "basis_ill_conditioned"
  decision_reason: str = "1se_pass"

  @property
  def half_saturation_t(self) -> float:
    """Grid variable of ``alpha_hat``.

    ``T_1/2`` for ``tanh`` / ``exp``, ``T_knee = alpha_hat`` for ``log`` and
    NaN for ``power``; this is what the ``half_saturation_T`` JSON key holds.
    """
    if not self.family or not np.isfinite(self.alpha_hat):
      return float("nan")
    return float(
        saturation.half_saturation_point(self.family, float(self.alpha_hat))
    )

  @property
  def grid_variable(self) -> str:
    """Name of the grid variable of ``family``."""
    if not self.family:
      return ""
    return saturation.grid_variable(self.family)

  @property
  def rloss_min(self) -> float:
    """``L_bar(alpha_hat)``: the seed-averaged loss at the selected alpha."""
    if self.per_seed_loss_at_alpha_hat is None:
      return float("nan")
    return float(np.mean(self.per_seed_loss_at_alpha_hat))

  def fbar_quantile(self, percentile: int) -> float:
    """``phi_p(alpha_hat)`` for ``p`` in :data:`DIAG4_PERCENTILES`."""
    if self.fbar_quantiles is None:
      return float("nan")
    return float(self.fbar_quantiles[DIAG4_PERCENTILES.index(percentile)])

  def linear_ref_quantile(self, percentile: int) -> float:
    """Linear reference ``q_p(T > 0) / T_ref`` of Diag4."""
    if self.linear_ref_quantiles is None:
      return float("nan")
    return float(
        self.linear_ref_quantiles[DIAG4_PERCENTILES.index(percentile)]
    )


def _infer_date_block(
    dates: pd.Series,
) -> Tuple[Optional[pd.Series], Optional[str]]:
  """Builds contiguous date blocks one calendar unit coarser than the data.

  The rule is: weekly blocks for daily data, monthly blocks for
  weekly data (and, by the same rule, quarterly for monthly data).  Clustering
  on raw individual dates is forbidden because adjacent dates are strongly
  autocorrelated (Bergmeir & Benitez 2012).

  Args:
    dates: The raw date column.

  Returns:
    A ``(block_series, frequency_label)`` tuple, or ``(None, None)`` when the
    column cannot be parsed as dates.
  """
  parsed = pd.to_datetime(dates, errors="coerce")
  if parsed.isna().all():
    return None, None
  uniq = np.sort(parsed.dropna().unique())
  if uniq.size < 2:
    return None, None
  diffs = np.diff(uniq).astype("timedelta64[s]").astype(float) / 86400.0
  step_days = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

  if step_days <= 1.5:
    freq, label = "W", "weekly blocks (daily data)"
  elif step_days <= 10.0:
    freq, label = "M", "monthly blocks (weekly data)"
  elif step_days <= 45.0:
    freq, label = "Q", "quarterly blocks (monthly data)"
  else:
    freq, label = "Y", "yearly blocks"
  blocks = parsed.dt.to_period(freq).astype(str)
  blocks.name = DATE_BLOCK_COL
  return blocks, label


def determine_cluster_keys(
    df: pd.DataFrame,
    item_col: Optional[str] = None,
    geo_col: Optional[str] = None,
    date_col: Optional[str] = None,
    min_clusters: int = MIN_CLUSTERS_WARNING,
    emit_warnings: bool = True,
) -> ClusterKeyPlan:
  """Applies the cluster-key determination rule.

  Candidate keys are ``item_col``, ``geo_col`` and *contiguous blocks* of
  ``date_col``; the two keys with the largest cluster counts are adopted for
  cluster DML (2-way, Chiang et al. 2021).  If a single candidate exists the
  run is 1-way.  This rule is applied to saturation runs *and* to normal runs
  (``family=None``).

  The use of ``item_col`` / ``geo_col`` / ``date_col`` for visualisation is
  completely independent of this selection (separation from visualisation).

  Args:
    df: The input dataframe (never modified).
    item_col: Item identifier column, if any.
    geo_col: Geo identifier column, if any.
    date_col: Date column, if any.
    min_clusters: Warning threshold on the number of clusters.
    emit_warnings: Whether to raise ``UserWarning``s for the warning criteria.

  Returns:
    The :class:`ClusterKeyPlan` describing the adopted keys.
  """
  plan = ClusterKeyPlan()
  candidates: List[Tuple[str, int]] = []

  for col in (item_col, geo_col):
    if col and col in df.columns:
      count = int(df[col].nunique(dropna=True))
      if count >= 2:
        candidates.append((col, count))
        plan.candidate_counts[col] = count

  if date_col and date_col in df.columns:
    blocks, label = _infer_date_block(df[date_col])
    if blocks is not None:
      count = int(blocks.nunique(dropna=True))
      if count >= 2:
        candidates.append((DATE_BLOCK_COL, count))
        plan.candidate_counts[DATE_BLOCK_COL] = count
        plan.derived_columns[DATE_BLOCK_COL] = blocks
        plan.date_block_freq = label

  if not candidates:
    plan.messages.append(
        "No cluster key candidate available (item_col / geo_col / date_col are"
        " unset or constant); falling back to observation-level inference."
    )
    if emit_warnings:
      warnings.warn(plan.messages[-1], UserWarning)
    return plan

  # Adopt the two keys with the largest cluster counts.
  candidates.sort(key=lambda kv: (-kv[1], kv[0]))
  selected = candidates[:2]
  plan.keys = [name for name, _ in selected]
  plan.cluster_counts = {name: count for name, count in selected}
  plan.primary_key = plan.keys[0]

  # Drop derived columns that were not adopted (nothing to attach then).
  plan.derived_columns = {
      k: v for k, v in plan.derived_columns.items() if k in plan.keys
  }

  # Warning criteria.
  for name, count in selected:
    if count < min_clusters:
      msg = (
          f"Cluster key '{name}' has only {count} clusters (< {min_clusters})."
          " Cluster-robust confidence intervals may be unreliable"
          " (Cameron & Miller 2015). Continuing."
      )
      plan.messages.append(msg)
      if emit_warnings:
        warnings.warn(msg, UserWarning)
  return plan


def _cluster_strata(
    cluster_values: pd.Series, strata_values: Optional[pd.Series]
) -> Dict[Any, Any]:
  """Maps every cluster to a single stratum (its modal stratum value)."""
  if strata_values is None:
    return {}
  frame = pd.DataFrame(
      {"cluster": cluster_values.values, "stratum": strata_values.values}
  )
  modes = frame.groupby("cluster", observed=False)["stratum"].agg(
      lambda s: s.iloc[0] if s.empty else s.mode(dropna=False).iloc[0]
  )
  return modes.to_dict()


def _assign_clusters(
    clusters: Sequence[Any],
    sel_fraction: float,
    strata_map: Dict[Any, Any],
    is_time_key: bool,
    rng: np.random.Generator,
) -> List[Any]:
  """Chooses the clusters that go to ``D_sel``.

  Implements the Stage A stratification: geo-stratified assignment
  for identifier keys, and *interleaving along the time axis* for date-block
  keys so that both sides cover all seasons.

  Args:
    clusters: The list of cluster labels.
    sel_fraction: Target share of ``D_sel``.
    strata_map: Mapping cluster -> stratum (may be empty).
    is_time_key: Whether the key is a contiguous date-block key.
    rng: Random generator used for the (stratified) shuffling.

  Returns:
    The list of cluster labels assigned to ``D_sel``.
  """
  clusters = list(clusters)
  if is_time_key:
    ordered = sorted(clusters, key=str)
    step = max(2, int(round(1.0 / max(sel_fraction, 1e-6))))
    # Alternating assignment: every `step`-th block goes to D_sel.
    return [c for i, c in enumerate(ordered) if i % step == 0]

  if strata_map:
    grouped: Dict[Any, List[Any]] = {}
    for cluster in clusters:
      grouped.setdefault(strata_map.get(cluster, "__na__"), []).append(cluster)
  else:
    grouped = {"__all__": clusters}

  selected: List[Any] = []
  for _, members in sorted(grouped.items(), key=lambda kv: str(kv[0])):
    members = sorted(members, key=str)
    rng.shuffle(members)
    n_members = len(members)
    if n_members < 2:
      continue
    n_take = int(round(sel_fraction * n_members))
    n_take = max(1, min(n_members - 1, n_take))
    selected.extend(members[:n_take])
  return selected


def _cell_frame(
    key1_values: Any, key2_values: Any
) -> Tuple[pd.DataFrame, np.ndarray]:
  """Builds the intersection cells of two cluster keys.

  Args:
    key1_values: Value of ``k1`` per row.
    key2_values: Value of ``k2`` per row.

  Returns:
    ``(cells, row_cell)``: ``cells`` has one row per distinct ``(k1, k2)``
    pair with columns ``cell_id`` (``"<k1>|<k2>"``), ``k1``, ``k2``;
    ``row_cell[i]`` is the position in ``cells`` of row ``i``.
  """
  k1 = pd.Series(np.asarray(key1_values)).astype(str).to_numpy()
  k2 = pd.Series(np.asarray(key2_values)).astype(str).to_numpy()
  ids = pd.Series(k1, dtype=object) + CELL_ID_SEP + pd.Series(k2, dtype=object)
  row_cell, uniques = pd.factorize(ids, sort=False)
  _, first = np.unique(row_cell, return_index=True)
  cells = pd.DataFrame({
      "cell_id": np.asarray(uniques, dtype=object),
      "k1": k1[first],
      "k2": k2[first],
  })
  return cells, np.asarray(row_cell, dtype=int)


def _round_robin_groups(
    cells: pd.DataFrame, n_groups: int, rng: np.random.Generator
) -> np.ndarray:
  """Interleaved round-robin group of every cell.

  ``group(a, b) = (rank_b(a) + rank_a(b)) mod n_groups`` where ``rank_a(b)`` is
  the position of ``b`` among the ``k2`` levels present in ``a`` (sorted
  order; date blocks are period strings, so this is time order) and
  ``rank_b(a)`` the position of ``a`` among the ``k1`` levels present in ``b``
  under a random permutation ``pi`` of the ``k1`` levels.  On a complete
  (balanced) panel ``rank_a(b) = idx(b)`` and ``rank_b(a) = pi(a)``, i.e. this
  is exactly the split rule ``(idx(b) + pi(a)) mod M``.  Applied to the cells
  of ``D_sel`` (which no longer form a complete grid) the ranks are taken
  within the sub-grid, so every level still cycles through all groups.

  Args:
    cells: Output of :func:`_cell_frame`.
    n_groups: ``M`` (top-level split) or ``K`` (Stage B folds).
    rng: Random generator drawing ``pi``.

  Returns:
    An integer array (one entry per cell) in ``[0, n_groups)``.
  """
  k1_levels = sorted(cells["k1"].unique())
  k2_levels = sorted(cells["k2"].unique())
  pi = dict(zip(k1_levels, rng.permutation(len(k1_levels))))
  idx2 = {level: i for i, level in enumerate(k2_levels)}
  frame = cells.assign(_pi=cells["k1"].map(pi), _idx=cells["k2"].map(idx2))
  rank_in_k1 = frame.sort_values(["k1", "_idx"]).groupby("k1").cumcount()
  rank_in_k2 = frame.sort_values(["k2", "_pi"]).groupby("k2").cumcount()
  groups = (rank_in_k1.reindex(frame.index) + rank_in_k2.reindex(frame.index))
  return (groups.to_numpy(dtype=int) % int(n_groups)).astype(int)


def _greedy_balanced_sel(
    cells: pd.DataFrame,
    sel_fraction: float,
    n_cycle: int,
    rng: np.random.Generator,
) -> np.ndarray:
  """Greedy balanced selection when ``sel_fraction != 1/M``.

  Every ``k1`` level ``a`` (visited in random order) puts
  ``ceil(sel_fraction * n_a)`` of its ``n_a`` cells into ``D_sel``, choosing
  the ``k2`` levels with the fewest ``D_sel`` cells so far (balance on the
  ``k2`` side) and, among those, the ones closest to evenly spaced target
  positions along the ``k2`` order (interleaving along the time axis).

  Args:
    cells: Output of :func:`_cell_frame`.
    sel_fraction: Target share ``r``.
    n_cycle: ``M = round(1 / r)``, used for the per-level phase offset.
    rng: Random generator.

  Returns:
    A boolean array (one entry per cell), ``True`` for ``D_sel``.
  """
  k1_levels = sorted(cells["k1"].unique())
  pi = dict(zip(k1_levels, rng.permutation(len(k1_levels))))
  k2_levels = sorted(cells["k2"].unique())
  idx2 = {level: i for i, level in enumerate(k2_levels)}
  is_sel = np.zeros(len(cells), dtype=bool)
  sel_count: Dict[str, int] = {level: 0 for level in k2_levels}
  members = cells.groupby("k1").indices
  for a in sorted(k1_levels, key=lambda lvl: pi[lvl]):
    rows = np.asarray(members[a])
    rows = rows[np.argsort([idx2[b] for b in cells["k2"].to_numpy()[rows]])]
    n_a = len(rows)
    if n_a < 2:
      continue
    n_take = int(math.ceil(sel_fraction * n_a - 1e-9))
    n_take = max(1, min(n_a - 1, n_take))
    offset = (int(pi[a]) % max(n_cycle, 1)) / float(max(n_cycle, 1))
    targets = (np.arange(n_take) + offset) * n_a / n_take
    k2_of = cells["k2"].to_numpy()[rows]
    available = set(range(n_a))
    for target in targets:
      # Arg-min over (k2 balance, distance to target, position); the position
      # ``j`` is unique, so the smallest tuple identifies exactly one cell.
      _, _, best = min(
          (sel_count[k2_of[j]], abs(j - target), j) for j in available
      )
      available.discard(best)
      sel_count[k2_of[best]] += 1
      is_sel[rows[best]] = True
  return is_sel


def _repair_coverage(
    cells: pd.DataFrame, groups: np.ndarray, n_groups: int, max_passes: int = 20
) -> np.ndarray:
  """Moves single cells so that every level reaches every group.

  The round-robin rule already covers every group on a complete panel; on an
  unbalanced panel (missing cells) a level can miss a group.  For every such
  ``(level, group)`` hole one cell of that level is moved into the group,
  taken from a group where BOTH its ``k1`` level and its ``k2`` level keep at
  least one other cell, so a repair never opens a new hole.  When no such
  single move exists, the cell is instead *swapped* with a cell of the same
  other-key level that sits in the target group (whose own level keeps at
  least one other cell there); a swap leaves the other key's per-group counts
  unchanged.  Levels with fewer cells than groups cannot be covered and are
  left alone (they are reported by the coverage check).

  Args:
    cells: Output of :func:`_cell_frame`.
    groups: Group per cell (modified copy is returned).
    n_groups: Number of groups.
    max_passes: Upper bound on full sweeps.

  Returns:
    The repaired group array.
  """
  groups = np.asarray(groups, dtype=int).copy()
  keys = {"k1": cells["k1"].to_numpy(), "k2": cells["k2"].to_numpy()}
  members = {name: cells.groupby(name).indices for name in keys}
  for _ in range(max_passes):
    counts = {
        name: pd.crosstab(values, groups).reindex(
            columns=range(n_groups), fill_value=0
        )
        for name, values in keys.items()
    }
    moved = False
    for name, other in (("k1", "k2"), ("k2", "k1")):
      table = counts[name]
      holes = table.index[(table.to_numpy() == 0).any(axis=1)]
      for level in holes:
        rows = members[name][level]
        if len(rows) < n_groups:
          continue  # infeasible: fewer cells than groups.
        for target in [g for g in range(n_groups) if table.at[level, g] == 0]:
          best, best_score = None, None
          for row in rows:
            src = groups[row]
            if src == target or table.at[level, src] < 2:
              continue
            other_level = keys[other][row]
            if counts[other].at[other_level, src] < 2:
              continue
            score = (table.at[level, src], counts[other].at[other_level, src])
            if best_score is None or score > best_score:
              best, best_score = row, score
          if best is not None:
            src = groups[best]
            other_level = keys[other][best]
            groups[best] = target
            table.at[level, src] -= 1
            table.at[level, target] += 1
            counts[other].at[other_level, src] -= 1
            counts[other].at[other_level, target] += 1
            moved = True
            continue
          # Swap fallback: x (this level, in src) <-> y (same other-key
          # level, in target).  The other key's counts are unchanged; y's own
          # level must keep >= 1 cell in target.
          swap, swap_score = None, None
          for row in rows:
            src = groups[row]
            if src == target or table.at[level, src] < 2:
              continue
            other_level = keys[other][row]
            for peer in members[other][other_level]:
              if groups[peer] != target:
                continue
              peer_level = keys[name][peer]
              if peer_level == level or table.at[peer_level, target] < 2:
                continue
              score = (table.at[level, src], table.at[peer_level, target])
              if swap_score is None or score > swap_score:
                swap, swap_score = (row, peer), score
          if swap is None:
            continue
          row, peer = swap
          src = groups[row]
          peer_level = keys[name][peer]
          groups[row] = target
          groups[peer] = src
          table.at[level, src] -= 1
          table.at[level, target] += 1
          table.at[peer_level, target] -= 1
          table.at[peer_level, src] += 1
          moved = True
    if not moved:
      break
  return groups


def _coverage_table(
    cells: pd.DataFrame,
    is_sel: np.ndarray,
    key_names: Tuple[str, str],
) -> pd.DataFrame:
  """``coverage_check.csv`` rows for the top-level split."""
  frames = []
  for column, name in zip(("k1", "k2"), key_names):
    table = (
        pd.DataFrame({"level": cells[column].to_numpy(), "sel": is_sel})
        .groupby("level", sort=True)["sel"]
        .agg(n_cells_sel="sum", n_cells_total="size")
        .reset_index()
    )
    table["n_cells_inf"] = table["n_cells_total"] - table["n_cells_sel"]
    table.insert(0, "key", name)
    frames.append(table[["key", "level", "n_cells_sel", "n_cells_inf"]])
  out = pd.concat(frames, ignore_index=True)
  out["n_cells_sel"] = out["n_cells_sel"].astype(int)
  out["n_cells_inf"] = out["n_cells_inf"].astype(int)
  out["covered"] = (out["n_cells_sel"] >= 1) & (out["n_cells_inf"] >= 1)
  return out


def _order_cell_keys(
    plan: ClusterKeyPlan, keys: Sequence[str]
) -> Tuple[str, str]:
  """Returns ``(k1, k2)``: ``k2`` is indexed in order, ``k1`` is permuted.

  Date blocks are indexed in time order, so the date-block key
  (when adopted) is ``k2``.  Otherwise ``k1`` is the key with more clusters.

  Args:
    plan: The cluster-key plan.
    keys: The two adopted keys present in the frame.

  Returns:
    The ``(k1, k2)`` pair.
  """
  first, second = keys[0], keys[1]
  if first == DATE_BLOCK_COL:
    return second, first
  if second == DATE_BLOCK_COL:
    return first, second
  if plan.cluster_counts.get(second, 0) > plan.cluster_counts.get(first, 0):
    return second, first
  return first, second


def cell_kfold(
    key1_values: Any,
    key2_values: Any,
    k_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
  """Stage B cell-level K-fold partition.

  Applies the round-robin rule with ``M = K`` to the intersection cells of
  the two adopted keys inside ``D_sel`` (one fresh permutation ``pi`` per
  call, i.e. per seed), then repairs holes so that every level of every key
  is distributed over all folds whenever it has at least ``K`` cells.

  Args:
    key1_values: ``k1`` value per row of ``D_sel``.
    key2_values: ``k2`` value per row of ``D_sel`` (date blocks: time order).
    k_folds: ``K``.
    rng: Random generator (one per seed).

  Returns:
    The fold id per row, in ``[0, K)``; a cell never straddles folds.
  """
  cells, row_cell = _cell_frame(key1_values, key2_values)
  groups = _round_robin_groups(cells, int(k_folds), rng)
  groups = _repair_coverage(cells, groups, int(k_folds))
  return groups[row_cell]


def stage_b_cluster_inputs(
    df_sel: pd.DataFrame, split: TopLevelSplit, plan: ClusterKeyPlan
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]],
           Optional[Tuple[str, str]]]:
  """Cluster labels and fold keys for Stage B.

  Args:
    df_sel: ``D_sel`` (rows of the split frame).
    split: The Stage A split.
    plan: The cluster-key plan.

  Returns:
    ``(cluster_labels, fold_key_values, fold_key_names)``.  For a cell-level
    split the labels are the cell ids and the fold keys are the two key
    columns (``select_alpha`` then builds cell-level folds with
    :func:`cell_kfold`); otherwise the labels are the single key (or one
    label per row) and the fold keys are ``None``.
  """
  if split.unit == "cell" and len(split.split_keys) == 2:
    key1, key2 = split.split_keys
    if split.cell_ids is not None:
      labels = split.cell_ids.reindex(df_sel.index).astype(str).to_numpy()
    else:
      labels, _ = _cell_frame(df_sel[key1], df_sel[key2])
      labels = labels["cell_id"].to_numpy()
    fold_values = (
        df_sel[key1].astype(str).to_numpy(),
        df_sel[key2].astype(str).to_numpy(),
    )
    return labels, fold_values, (key1, key2)
  key = split.split_keys[0] if split.split_keys else plan.primary_key
  if key and key in df_sel.columns:
    return df_sel[key].astype(str).to_numpy(), None, None
  return np.arange(len(df_sel)).astype(str), None, None


def _cell_level_split(
    df: pd.DataFrame,
    plan: ClusterKeyPlan,
    keys: Sequence[str],
    sel_fraction: float,
    k_folds: int,
    rng: np.random.Generator,
    messages: List[str],
) -> Optional[TopLevelSplit]:
  """Stage A split on the intersection cells of two keys.

  Args:
    df: The split frame (carrying the derived cluster columns).
    plan: The cluster-key plan.
    keys: The two adopted keys present in ``df``.
    sel_fraction: Target ``D_sel`` share ``r``.
    k_folds: ``K`` (Stage B requirement and fold-coverage feasibility).
    rng: Random generator drawing the permutation ``pi``.
    messages: Message sink (warnings are emitted by the caller).

  Returns:
    The split, or ``None`` when the preconditions / Stage A-B requirements
    fail (the caller then takes the final fallback).
  """
  key1, key2 = _order_cell_keys(plan, keys)
  n_cycle = max(2, int(round(1.0 / max(sel_fraction, 1e-6))))
  c1 = int(df[key1].astype(str).nunique())
  c2 = int(df[key2].astype(str).nunique())
  if c1 < n_cycle or c2 < n_cycle:
    messages.append(
        f"Cell-level split needs C1 >= M and C2 >= M (M = round(1/sel_fraction)"
        f" = {n_cycle}); got '{key1}' = {c1} and '{key2}' = {c2} levels, so the"
        " coverage guarantee cannot hold."
    )
    return None

  cells, row_cell = _cell_frame(df[key1], df[key2])
  n_cells = len(cells)
  if abs(sel_fraction - 1.0 / n_cycle) <= 1e-9:
    is_sel = _round_robin_groups(cells, n_cycle, rng) == 0
  else:
    is_sel = _greedy_balanced_sel(cells, sel_fraction, n_cycle, rng)
  sides = _repair_coverage(cells, np.where(is_sel, 0, 1), 2)
  is_sel = sides == 0
  n_sel_cells = int(is_sel.sum())
  if n_sel_cells < max(k_folds, 2) or n_sel_cells >= n_cells:
    messages.append(
        f"Cell-level split yields {n_sel_cells} of {n_cells} cells in D_sel,"
        f" which is below the Stage B requirement (>= k_folds = {k_folds})."
    )
    return None

  coverage = _coverage_table(cells, is_sel, (key1, key2))
  uncovered = coverage.loc[~coverage["covered"]]
  if not uncovered.empty:
    detail = ", ".join(
        f"{key}={level}" for key, level in
        uncovered[["key", "level"]].head(10).itertuples(index=False)
    )
    messages.append(
        f"Coverage guarantee not met for {len(uncovered)} level(s) (e.g."
        f" {detail}): they have no cell on one side of D_sel / D_inf"
        " (unbalanced panel). Continuing."
    )
  # Stage B folds can only cover a level with >= K cells inside D_sel.
  thin = coverage.loc[coverage["n_cells_sel"] < int(k_folds)]
  for key, group in thin.groupby("key", sort=False):
    n_levels = int((coverage["key"] == key).sum())
    messages.append(
        f"{len(group)} of {n_levels} level(s) of '{key}' have fewer than"
        f" k_folds = {k_folds} cells in D_sel, so they cannot appear in every"
        " Stage B fold (needs the other key to have"
        f" >= M * K = {n_cycle * int(k_folds)} levels). Continuing."
    )

  mask_sel = is_sel[row_cell]
  cell_ids = pd.Series(
      cells["cell_id"].to_numpy()[row_cell], index=df.index, name="cell_id"
  )
  return TopLevelSplit(
      idx_sel=df.index[mask_sel],
      idx_inf=df.index[~mask_sel],
      split_key=f"{key1} × {key2}",
      level="cluster",
      tier=(
          "tier-1 (cell-level split on the intersection of the two adopted"
          " keys)"
      ),
      forced_full_sample=False,
      n_clusters_total=int(n_cells),
      n_clusters_sel=n_sel_cells,
      messages=messages,
      unit="cell",
      split_keys=[key1, key2],
      cell_ids=cell_ids,
      coverage=coverage,
      coverage_ok=bool(uncovered.empty),
  )


def _single_key_split(
    df: pd.DataFrame,
    key: str,
    sel_fraction: float,
    k_folds: int,
    geo_col: Optional[str],
    rng: np.random.Generator,
    messages: List[str],
) -> Optional[TopLevelSplit]:
  """Stage A split on the levels of the only adopted key.

  Args:
    df: The split frame.
    key: The single adopted key.
    sel_fraction: Target ``D_sel`` share.
    k_folds: ``K`` (Stage B requirement).
    geo_col: Stratification column.
    rng: Random generator.
    messages: Message sink.

  Returns:
    The split, or ``None`` when the Stage A / B requirements fail.
  """
  required_clusters = int(math.ceil(1.0 / max(sel_fraction, 1e-6)))
  cluster_values = df[key]
  clusters = list(pd.unique(cluster_values.dropna()))
  n_clusters = len(clusters)
  is_time_key = key == DATE_BLOCK_COL
  if n_clusters < max(required_clusters, 2):
    messages.append(
        f"Cluster key '{key}' has {n_clusters} clusters, which is below the"
        f" Stage A requirement C >= ceil(1/sel_fraction) = {required_clusters}."
    )
    return None
  strata_series = df[geo_col] if geo_col and geo_col in df.columns else None
  strata_map = (
      {}
      if (is_time_key or strata_series is None or key == geo_col)
      else _cluster_strata(cluster_values, strata_series)
  )
  sel_clusters = _assign_clusters(
      clusters, sel_fraction, strata_map, is_time_key, rng
  )
  if len(sel_clusters) < max(k_folds, 2) or len(sel_clusters) >= n_clusters:
    messages.append(
        f"Cluster key '{key}' yields {len(sel_clusters)} clusters in D_sel,"
        f" which is below the Stage B requirement (>= k_folds = {k_folds})."
    )
    return None
  mask_sel = cluster_values.isin(sel_clusters).fillna(False).to_numpy()
  if not mask_sel.any() or mask_sel.all():
    messages.append(f"Cluster key '{key}' produced an empty side.")
    return None
  levels = cluster_values.astype(str)
  sel_levels = set(levels[mask_sel])
  coverage = pd.DataFrame({"key": key, "level": sorted(levels.unique())})
  coverage["n_cells_sel"] = coverage["level"].isin(sel_levels).astype(int)
  coverage["n_cells_inf"] = 1 - coverage["n_cells_sel"]
  coverage["covered"] = False
  messages.append(
      "Only one cluster key is adopted, so the split unit is that key's level"
      " and the coverage guarantee cannot hold; tier 1 of the three-tier rule"
      " applies."
  )
  return TopLevelSplit(
      idx_sel=df.index[mask_sel],
      idx_inf=df.index[~mask_sel],
      split_key=key,
      level="cluster",
      tier="tier-2 (single-key split: only one cluster key adopted)",
      forced_full_sample=False,
      n_clusters_total=n_clusters,
      n_clusters_sel=len(sel_clusters),
      messages=messages,
      unit="single_key",
      split_keys=[key],
      cell_ids=levels.rename("cell_id"),
      coverage=coverage,
      coverage_ok=False,
  )


def make_top_level_split(
    df: pd.DataFrame,
    plan: ClusterKeyPlan,
    sel_fraction: float = 0.20,
    k_folds: int = 5,
    geo_col: Optional[str] = None,
    random_state: int = 0,
    emit_warnings: bool = True,
) -> TopLevelSplit:
  """Performs the Stage A top-level split ``D = D_sel + D_inf``.

  With two adopted keys the split unit is the intersection
  cell of the two keys.  A cell never straddles ``D_sel`` and ``D_inf``, and
  every level of every key appears on both sides (coverage guarantee).  Cells
  are assigned by the interleaved round-robin rule ``cell(a, b) in D_sel <=>
  (idx(b) + pi(a)) mod M == 0`` (``M = round(1/sel_fraction)``, date blocks
  indexed in time order), or by the greedy balanced selection when
  ``sel_fraction != 1/M``.  Splitting on a single key is used only when a
  single key is adopted.  If the preconditions (``C1, C2 >= M``) or the Stage
  A/B requirements fail, the *final fallback* (observation-level split plus a
  forced switch to ``full_sample``) is taken.

  Args:
    df: The frame the split is computed on (already carrying the derived
      cluster columns).
    plan: The cluster-key plan from :func:`determine_cluster_keys`.
    sel_fraction: ``n_sel : n_inf`` ratio (default 20% : 80%).
    k_folds: ``K`` of the alpha-selection stage (Stage B requirement).
    geo_col: Stratification column of the single-key split.
    random_state: Seed of the permutation ``pi`` / the cluster shuffling.
    emit_warnings: Whether to raise ``UserWarning``s.

  Returns:
    The :class:`TopLevelSplit`.
  """
  rng = np.random.default_rng(random_state)
  messages: List[str] = []
  usable = [k for k in plan.keys if k in df.columns]

  split: Optional[TopLevelSplit] = None
  if len(usable) >= 2:
    split = _cell_level_split(
        df, plan, usable[:2], sel_fraction, k_folds, rng, messages
    )
  elif len(usable) == 1:
    split = _single_key_split(
        df, usable[0], sel_fraction, k_folds, geo_col, rng, messages
    )
  if split is not None:
    if emit_warnings:
      for msg in messages:
        warnings.warn(msg, UserWarning)
    return split

  # Final fallback.
  fallback_msg = (
      "Stage A/B requirements could not be met by the cluster keys"
      f" ({plan.describe()}). Falling back to an OBSERVATION-LEVEL top-level"
      " split; post-selection independence is lost, therefore final_fit is"
      " automatically switched to 'full_sample'."
  )
  messages.append(fallback_msg)
  if emit_warnings:
    for msg in messages:
      warnings.warn(msg, UserWarning)

  n_rows = len(df)
  shuffled = np.array(df.index)
  rng.shuffle(shuffled)
  n_sel = max(1, min(n_rows - 1, int(round(sel_fraction * n_rows))))
  idx_sel = pd.Index(shuffled[:n_sel])
  idx_inf = pd.Index(shuffled[n_sel:])
  return TopLevelSplit(
      idx_sel=idx_sel,
      idx_inf=idx_inf,
      split_key=None,
      level="observation",
      tier="tier-3 (final fallback: observation-level split + full_sample)",
      forced_full_sample=True,
      n_clusters_total=0,
      n_clusters_sel=0,
      messages=messages,
      unit="observation",
      split_keys=[],
      cell_ids=None,
      coverage=None,
      coverage_ok=False,
  )


def check_identifier_dummies(
    df: pd.DataFrame,
    idx_inf: pd.Index,
    columns: Sequence[str],
    split_key: Optional[str],
    geo_col: Optional[str],
    emit_warnings: bool = True,
) -> List[str]:
  """Verifies the three-tier rule for identifier dummies.

  Under ``final_fit: d_inf`` a dummy at the same granularity as the split unit
  becomes an all-zero column inside ``D_inf`` (unidentified coefficient).
  Dummies at a coarser granularity are allowed provided the stratified
  assignment gives every level clusters on both sides.

  Args:
    df: The full frame.
    idx_inf: Index of ``D_inf``.
    columns: Columns entering ``p(X)`` / ``X`` that must be checked.
    split_key: The cluster key used for the top-level split.
    geo_col: Coarser identifier column whose two-sided coverage is verified.
    emit_warnings: Whether to raise ``UserWarning``s.

  Returns:
    The list of diagnostic messages (empty when everything is fine).
  """
  messages: List[str] = []
  if len(idx_inf) == 0:
    return messages
  df_inf = df.loc[idx_inf]
  for col in columns:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
      continue
    if df[col].nunique(dropna=True) > 2:
      continue
    if (
        df[col].nunique(dropna=True) > 1
        and df_inf[col].nunique(dropna=True) <= 1
    ):
      messages.append(
          f"Identifier-like dummy '{col}' is constant inside D_inf and cannot"
          " be identified under final_fit='d_inf' (tier-1 of the three-tier"
          " rule). Consider final_fit='full_sample' or a coarser"
          " dummy."
      )

  if geo_col and geo_col in df.columns and split_key and split_key != geo_col:
    inf_levels = set(df_inf[geo_col].dropna().unique())
    all_levels = set(df[geo_col].dropna().unique())
    missing = sorted(map(str, all_levels - inf_levels))
    if missing:
      messages.append(
          f"Two-sided coverage check failed for '{geo_col}': level(s)"
          f" {missing} have no cluster inside D_inf (tier-2 of the three-tier"
          " rule)."
      )

  if emit_warnings:
    for msg in messages:
      warnings.warn(msg, UserWarning)
  return messages


def _cluster_kfold(
    cluster_labels: np.ndarray, k_folds: int, rng: np.random.Generator
) -> np.ndarray:
  """Assigns every observation to a fold at cluster level.

  Args:
    cluster_labels: Cluster label per observation.
    k_folds: Number of folds.
    rng: Random generator.

  Returns:
    An integer array of fold ids in ``[0, k_folds)``.
  """
  clusters = list(pd.unique(pd.Series(cluster_labels)))
  rng.shuffle(clusters)
  mapping = {c: i % k_folds for i, c in enumerate(clusters)}
  return np.array(
      [mapping.get(c, 0) for c in cluster_labels], dtype=int
  )


# Normalization bridge.  Only these scaler strategies are admissible
# for a saturated treatment: both divide by a strictly positive quantity, so
# they preserve `T >= 0` and `f(0) = 0`, and both pin `median(T_tilde) = 1`,
# which is what makes the single fixed default grid meaningful.
_DIVISIVE_TREATMENT_SCALINGS = (
    "median_normalize",
    "population_median_normalize",
)
# These SHIFT the treatment, so roughly half the rows go negative and would be
# silently clipped to 0 by `apply_saturation`; `f(0) = 0` (on which the Phase 4
# theta recovery depends) would also be lost.
_SHIFTING_TREATMENT_SCALINGS = (
    "standardize",
    "population_standardize",
    "min_max",
)

# Floor for the scaled-down Phase 0 FLAML budget: below a few seconds FLAML
# cannot complete even a single trial.
MIN_PHASE0_TIME_BUDGET = 5


@dataclasses.dataclass(frozen=True)
class TreatmentDivisor:
  """The scaler divisor ``c`` of one treatment, spelled out for users.

  Everything Phase 0 reports (``alpha_hat``, ``T_ref``, the fitted curve) and
  every downstream chart / output column that says "normalized treatment" is
  expressed on the scale ``T / c``.  This object carries ``c`` itself together
  with a formula and a one-sentence explanation so that logs, the Phase 0
  panel and ``summary_alpha_selection.json`` can state concretely what ``c``
  is instead of the bare symbol.

  Attributes:
    divisor: Series ``c`` aligned with the frame it was resolved on.
    mode: ``"median_normalize"``, ``"population_median_normalize"`` or
      ``"none"``.
    constant: The scalar in the formula: ``median(T | T != 0)`` for
      ``median_normalize``; ``median(T / population | T / population > 0)``
      for ``population_median_normalize``; ``1`` otherwise.
    population_col: Name of the population column (population mode only).
    formula: Short formula of ``c``, e.g. ``c = median(T | T != 0) = 12.3``.
    description: One-sentence explanation of what ``T / c`` means.
  """

  divisor: pd.Series
  mode: str
  constant: float
  population_col: Optional[str]
  formula: str
  description: str

  @property
  def axis_label(self) -> str:
    """Returns the x-axis label of charts drawn on the ``T / c`` scale."""
    if self.mode == "median_normalize":
      return (
          "normalized treatment T/c"
          f"  (c = median of non-zero T = {self.constant:.4g})"
      )
    if self.mode == "population_median_normalize":
      return (
          "normalized treatment T/c"
          f"  (c = {self.population_col} x {self.constant:.4g})"
      )
    return "normalized treatment T/c  (c = 1: raw treatment units)"

  @property
  def short_formula(self) -> str:
    """Returns a compact formula of ``c`` for narrow table cells."""
    if self.mode == "median_normalize":
      return f"c = median(T | T != 0) = {self.constant:.4g}"
    if self.mode == "population_median_normalize":
      return f"c_i = {self.population_col}_i x {self.constant:.4g}"
    return "c = 1 (raw units)"

  def as_dict(self) -> Dict[str, Any]:
    """Returns the JSON-serializable summary of the divisor."""
    return {
        "normalization": self.mode,
        "divisor_constant": float(self.constant),
        "divisor_population_col": self.population_col,
        "divisor_formula": self.formula,
        "divisor_description": self.description,
    }


def _describe_divisor(
    d_col: str, mode: str, constant: float, population_col: Optional[str]
) -> Tuple[str, str]:
  """Returns ``(formula, description)`` for :class:`TreatmentDivisor`."""
  if mode == "median_normalize":
    formula = f"c = median({d_col} | {d_col} != 0) = {constant:.6g}"
    description = (
        f"T/c = {d_col} / {constant:.6g}: the treatment divided by the median"
        f" of its non-zero values (median_normalize; one constant for all"
        " rows, so T/c = 1 is the typical non-zero row)."
    )
  elif mode == "population_median_normalize":
    formula = (
        f"c_i = {population_col}_i x median({d_col}/{population_col} |"
        f" {d_col}/{population_col} > 0) = {population_col}_i x"
        f" {constant:.6g}"
    )
    description = (
        f"T/c = ({d_col} / {population_col}) / {constant:.6g}: the treatment"
        " per head of population, divided by the median of the positive"
        " per-head values (population_median_normalize; c varies by row with"
        f" the population, so T/c = 1 is the typical per-head level)."
    )
  else:
    formula = "c = 1"
    description = (
        f"T/c = {d_col}: no divisive normalization is configured for the"
        " treatment, so the normalized and the raw treatment coincide."
    )
  return formula, description


def resolve_treatment_divisor(
    scaler: Any, d_col: str, df: pd.DataFrame
) -> TreatmentDivisor:
  """Resolves the scaler divisor ``c`` of a treatment column.

  The saturation transform is applied to the NORMALIZED treatment
  ``T_tilde = T / c``:

  * ``median_normalize``            -> ``c = median(T | T != 0)``, a global
    constant.
  * ``population_median_normalize`` -> ``c_i = p_i * median(T / p | > 0)``,
    which varies by row, and therefore induces a row-varying raw-scale
    ``alpha_i``.
  * anything else                   -> ``c = 1`` (alpha in raw units).

  The scaler is deep-copied and fitted on ``df`` so the caller's fitted state
  is never touched and the same divisor is obtained no matter when this is
  called.

  Args:
    scaler: A ``CausalDataScaler`` (or ``None``).
    d_col: The treatment column.
    df: The frame the divisor is wanted for.

  Returns:
    A :class:`TreatmentDivisor` whose ``divisor`` is aligned with
    ``df.index``.

  Raises:
    ValueError: If ``d_col`` is configured with a SHIFTING normalization,
      which is incompatible with a saturation family.
  """
  ones = pd.Series(1.0, index=df.index)

  def _none() -> TreatmentDivisor:
    formula, description = _describe_divisor(d_col, "none", 1.0, None)
    return TreatmentDivisor(ones, "none", 1.0, None, formula, description)

  if scaler is None:
    return _none()

  def _is_listed(strategy: str) -> bool:
    return d_col in (getattr(scaler, f"{strategy}_cols", None) or [])

  for strategy in _SHIFTING_TREATMENT_SCALINGS:
    if _is_listed(strategy):
      raise ValueError(
          f"Treatment '{d_col}' is configured for '{strategy}' normalization,"
          " which is incompatible with a saturation family: it shifts the"
          " treatment, so a large share of the rows become negative and are"
          " clipped to 0 by f(T;alpha), and f(0)=0 (which the Phase 4 theta"
          " recovery relies on) no longer lines up with T=0. Move"
          f" '{d_col}' to median_normalize_cols or"
          " population_median_normalize_cols, or disable saturation"
          " (saturation.family = None)."
      )

  if not any(_is_listed(s) for s in _DIVISIVE_TREATMENT_SCALINGS):
    return _none()

  fitted = copy.deepcopy(scaler)
  fitted.fit(df)

  if _is_listed("median_normalize"):
    params = (getattr(fitted, "median_params_", None) or {}).get(d_col)
    if not params:
      return _none()
    constant = float(params["median_val"])
    formula, description = _describe_divisor(
        d_col, "median_normalize", constant, None
    )
    return TreatmentDivisor(
        pd.Series(constant, index=df.index),
        "median_normalize",
        constant,
        None,
        formula,
        description,
    )

  params = (getattr(fitted, "pop_median_params_", None) or {}).get(d_col)
  if not params:
    return _none()
  median_per_pop = float(params["median_val_per_pop"])
  pop_col = str(fitted.population_col)
  pop = pd.to_numeric(df[pop_col], errors="coerce")
  pop = pop.where(pop > 0)
  fallback = pop.median()
  if not np.isfinite(fallback) or fallback <= 0:
    fallback = 1.0
  pop = pop.fillna(float(fallback))
  formula, description = _describe_divisor(
      d_col, "population_median_normalize", median_per_pop, pop_col
  )
  return TreatmentDivisor(
      pop * median_per_pop,
      "population_median_normalize",
      median_per_pop,
      pop_col,
      formula,
      description,
  )


def treatment_divisor(
    scaler: Any, d_col: str, df: pd.DataFrame
) -> Tuple[pd.Series, str]:
  """Returns the scaler divisor ``c`` of a treatment column.

  Thin wrapper around :func:`resolve_treatment_divisor` kept for the callers
  that only need the ``(divisor, mode)`` pair.

  Args:
    scaler: A ``CausalDataScaler`` (or ``None``).
    d_col: The treatment column.
    df: The frame the divisor is wanted for.

  Returns:
    A ``(divisor, mode)`` pair. ``divisor`` is a Series aligned with
    ``df.index``; ``mode`` names the normalization that produced it.

  Raises:
    ValueError: If ``d_col`` is configured with a SHIFTING normalization,
      which is incompatible with a saturation family.
  """
  info = resolve_treatment_divisor(scaler, d_col, df)
  return info.divisor, info.mode


def scale_time_budget(
    time_budget: int,
    n_sel: int,
    n_total: int,
    minimum: int = MIN_PHASE0_TIME_BUDGET,
) -> int:
  """Scales the FLAML time budget to the size of ``D_sel``.

  Phase 0 only ever sees ``D_sel`` (20% of the rows by default).  A FLAML
  search, however, consumes its WALL-CLOCK ``time_budget`` almost exactly
  regardless of ``n``, so the smaller sample buys nothing unless the budget is
  reduced with it.  The budget is scaled by ``n_sel / n_total`` and floored at
  ``minimum`` so FLAML can still complete a trial.

  Args:
    time_budget: The pipeline-wide budget in seconds.
    n_sel: Rows in ``D_sel``.
    n_total: Rows in the full sample.
    minimum: Lower bound for the scaled budget.

  Returns:
    The Phase 0 budget in seconds, never above ``time_budget``.
  """
  budget = float(time_budget)
  if n_total <= 0 or n_sel <= 0 or n_sel >= n_total:
    return int(max(minimum, round(budget)))
  scaled = budget * (float(n_sel) / float(n_total))
  return int(min(round(budget), max(minimum, round(scaled))))


def _touch(done_path: Optional[str]) -> None:
  """Writes a progress marker file for the Phase 0 display (best effort).

  The marker directory is created on demand: ``select_alpha`` receives a
  per-treatment sub-directory of the Phase 0 progress directory, and the
  ``[Phase 0 Progress]`` monitor counts the markers recursively.

  Args:
    done_path: Path of the marker file to create. ``None`` or empty disables
      the marker and the call is a no-op.
  """
  if not done_path:
    return
  try:
    parent = os.path.dirname(done_path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(done_path, "w") as handle:
      handle.write("")
  except OSError:
    pass


def _tune_scalar(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    time_budget: int,
    random_state: int,
    done_path: Optional[str] = None,
) -> Tuple[np.ndarray, Any]:
  """ONE AutoML search for a scalar nuisance (learner selection).

  Predictions on the training rows are required as well, because the inner
  ridge solve of the profiled R-loss uses ``i not in I_k``.

  Args:
    model_name: FLAML estimator name (e.g. ``"rf"``).
    x_train: Training design matrix.
    y_train: Training target.
    x_all: Design matrix of all rows to predict.
    time_budget: FLAML time budget in seconds.
    random_state: Seed.
    done_path: Optional marker file written on completion.

  Returns:
    A ``(predictions, template)`` pair. ``template`` is the tuned estimator,
    which the seed loop REFITS (never re-searches).
  """
  model = flaml_dml.FlamlRegressorDoubleML(
      time=time_budget,
      estimator_list=[model_name],
      metric="rmse",
      random_state=random_state,
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(x_train, y_train)
    preds = np.asarray(model.predict(x_all), dtype=float).reshape(-1)
  _touch(done_path)
  return preds, model.tuned_model_


def _tune_multioutput(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    time_budget: int,
    random_state: int,
    done_path: Optional[str] = None,
) -> Tuple[np.ndarray, Any, bool]:
  """ONE AutoML search for the whole J-output treatment basis.

  Args:
    model_name: FLAML estimator name.
    x_train: Training design matrix.
    y_train: Training targets, shape ``(n_train, J)``.
    x_all: Design matrix of all rows to predict.
    time_budget: FLAML time budget in seconds.
    random_state: Seed.
    done_path: Optional marker file written on completion.

  Returns:
    A ``(predictions, template, native_multioutput)`` triple, where
    ``predictions`` has shape ``(len(x_all), J)``.
  """
  model = flaml_dml.MultiOutputFlamlRegressor(
      time=time_budget,
      estimator_list=[model_name],
      metric="rmse",
      random_state=random_state,
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(x_train, y_train)
    preds = np.asarray(model.predict(x_all), dtype=float)
  _touch(done_path)
  return preds, model.template_, bool(model.native_multioutput_)


def _refit_predict(
    template: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    done_path: Optional[str] = None,
) -> np.ndarray:
  """Refits the SELECTED learner - no new hyperparameter search.

  The seed loop *refits* the two chosen learners once per (fold, seed).
  Running a fresh AutoML search here would cost a full time budget per fit and
  would leak learner-selection noise into the alpha loss curve.

  Args:
    template: The tuned estimator returned by ``_tune_scalar`` /
      ``_tune_multioutput``. It is cloned, never mutated.
    x_train: Training design matrix (the ``K-1`` in-fold parts).
    y_train: Training target, shape ``(n_train,)`` or ``(n_train, J)``.
    x_predict: Design matrix of the rows to predict (the held-out fold).
    done_path: Optional marker file written on completion.

  Returns:
    Predictions for every row of ``x_predict``: ``(n_predict,)`` for a scalar
    target, ``(n_predict, J)`` for the multi-output treatment basis.
  """
  targets = np.asarray(y_train, dtype=float)
  if targets.ndim > 1 and targets.shape[1] > 1:
    estimator, _ = flaml_dml.fit_multioutput(template, x_train, targets)
    preds = np.asarray(estimator.predict(x_predict), dtype=float)
  else:
    estimator = sk_base.clone(template)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      estimator.fit(x_train, targets.reshape(-1))
    preds = np.asarray(estimator.predict(x_predict), dtype=float).reshape(-1)
  _touch(done_path)
  return preds


def _cross_fit_jobs(
    seed_partitions: Sequence[np.ndarray], k_folds: int
) -> List[Tuple[int, int, str, np.ndarray, np.ndarray]]:
  """Enumerates the cross-fitting jobs ``(s, k, kind, train, held_out)``.

  One job per (seed, fold) and per nuisance (``"y"`` for ``l_hat``, ``"t"``
  for ``mu_hat``).  The model of fold ``k`` is trained on ``train = fold_ids
  != k`` and predicts ``held_out = fold_ids == k`` only.  Folds without a
  held-out row (fewer clusters than ``K``) need no model and are skipped;
  so is the degenerate case without a training row.

  Args:
    seed_partitions: Per-seed fold ids, each ``(n,)`` with values in
      ``0..K-1``.
    k_folds: ``K``.

  Returns:
    The job list, in (seed, fold, kind) order.
  """
  jobs = []
  for s, fold_ids in enumerate(seed_partitions):
    fold_ids = np.asarray(fold_ids)
    for k in range(int(k_folds)):
      held_out = fold_ids == k
      train = ~held_out
      if not np.any(train) or not np.any(held_out):
        continue
      jobs.append((s, k, "y", train, held_out))
      jobs.append((s, k, "t", train, held_out))
  return jobs


def _scatter_oof_predictions(
    fit_jobs: Sequence[Tuple[int, int, str, np.ndarray, np.ndarray]],
    fit_out: Sequence[np.ndarray],
    l_oof: np.ndarray,
    mu_oof: np.ndarray,
) -> None:
  """Writes the held-out predictions of every job into the OOF arrays.

  Args:
    fit_jobs: The jobs of :func:`_cross_fit_jobs`.
    fit_out: The prediction of each job, aligned with ``fit_jobs``:
      ``(n_held_out,)`` for ``"y"``, ``(n_held_out, J)`` for ``"t"``.
    l_oof: ``(S, n)`` target array of ``l_hat^(-k(i))(X_i)`` (in place).
    mu_oof: ``(S, n, J)`` target array of ``mu_hat^(-k(i))(X_i)`` (in place).
  """
  n_basis = mu_oof.shape[-1]
  for (s, _, kind, _, held_out), preds in zip(fit_jobs, fit_out):
    n_held_out = int(np.count_nonzero(held_out))
    if kind == "y":
      l_oof[s, held_out] = np.asarray(preds, dtype=float).reshape(n_held_out)
    else:
      mu_oof[s, held_out] = np.asarray(preds, dtype=float).reshape(
          n_held_out, n_basis
      )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Root mean squared error."""
  diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
  return float(np.sqrt(np.mean(np.square(diff))))


def _profile_losses(
    l_pred: np.ndarray,
    mu_pred: np.ndarray,
    y_values: np.ndarray,
    f_values: np.ndarray,
    coefs: np.ndarray,
    p_basis: np.ndarray,
    fold_ids: np.ndarray,
    ridge: float,
    return_obs: bool = False,
) -> Any:
  """Computes the out-of-fold profiled R-loss for a whole alpha grid.

  Implements the profiled R-loss exactly, with every nuisance prediction
  out-of-fold: row ``i`` carries ``l_hat^(-k(i))(X_i)`` and
  ``mu_hat_j^(-k(i))(X_i)`` from the models that were trained without its
  own fold ``k(i)`` (cross-fitting), so the residuals of the training rows
  are never in-sample residuals of the model being used:

  * ``m_hat(X_i;alpha) = sum_j c_j(alpha) mu_hat_j^(-k(i))(X_i)``
  * ``S_tilde_i(alpha) = f(T_i;alpha) - m_hat(X_i;alpha)``,
    ``Y_tilde_i = Y_i - l_hat^(-k(i))(X_i)``
  * ``beta^(-k)(alpha) = (sum_{i not in I_k} S_tilde_i^2 p_i p_i' + rho I)^-1
    sum_{i not in I_k} S_tilde_i p_i Y_tilde_i`` -- fitted on the out-of-fold
    residuals of the OTHER folds (honest: the evaluation fold is excluded)
  * ``L(alpha) = (1/n) sum_k sum_{i in I_k} [Y_tilde_i - S_tilde_i p_i'
    beta^(-k)(alpha)]^2``

  The normal equations of fold ``k`` are the totals over all rows minus the
  contribution of fold ``k`` itself, so the alpha-dependent algebra touches
  every row once (``(n, G)`` residual matrices instead of ``(K, n, G)``); the
  ``(K x G)`` ridge solves are batched (``jax.vmap`` on GPU/TPU when jax is
  available, batched ``numpy.linalg.solve`` otherwise).

  Args:
    l_pred: ``(n,)`` out-of-fold predictions of ``l_hat``.
    mu_pred: ``(n, J)`` out-of-fold predictions of ``mu_hat_j``.
    y_values: ``(n,)`` outcome.
    f_values: ``(n, G)`` exact ``f(T_i; alpha_g)``.
    coefs: ``(J, G)`` basis coefficients ``c_j(alpha_g)``.
    p_basis: ``(n, P)`` CATE basis ``p(X_i)``.
    fold_ids: ``(n,)`` fold assignment ``k(i)`` in ``0..K-1``.
    ridge: Ridge term ``rho`` of the inner solve.
    return_obs: Also return the per-observation
      out-of-fold squared residuals (needed for ``se_obs``).

  Returns:
    A ``(G,)`` array with the profiled R-loss of every alpha; with
    ``return_obs=True`` a ``(losses (G,), obs_sq_resid (n, G))`` pair, where
    ``losses == obs_sq_resid.mean(axis=0)``.
  """
  # Bound to locals so that the ``is not None`` checks below actually narrow the
  # type (a module-level global could in principle be rebound at any time, so a
  # static checker cannot narrow ``_JAX`` / ``_JNP`` themselves).
  jax_lib = _JAX
  jnp_lib = _JNP
  xnp = jnp_lib if jnp_lib is not None else np

  fold_arr = np.asarray(fold_ids).astype(int).reshape(-1)
  n_folds = int(fold_arr.max()) + 1 if fold_arr.size else 1
  l_pred = xnp.asarray(l_pred).reshape(-1)
  mu_pred = xnp.asarray(mu_pred)
  y_values = xnp.asarray(y_values).reshape(-1)
  f_values = xnp.asarray(f_values)
  coefs = xnp.asarray(coefs)
  p_basis = xnp.asarray(p_basis)

  n_obs, n_grid = f_values.shape
  n_p = p_basis.shape[1]

  # Out-of-fold nuisances -> one residual matrix per quantity.
  m_hat = mu_pred @ coefs  # (n, G)
  s_res = f_values - m_hat  # (n, G)
  y_tilde = y_values - l_pred  # (n,)
  s_sq = s_res * s_res  # (n, G)
  s_y = s_res * y_tilde[:, None]  # (n, G)

  # Normal equations: totals over all rows, minus each fold's own rows.
  pp_outer = (p_basis[:, :, None] * p_basis[:, None, :]).reshape(
      n_obs, n_p * n_p
  )
  a_total = s_sq.T @ pp_outer  # (G, P^2)
  u_total = s_y.T @ p_basis  # (G, P)
  a_parts = []
  u_parts = []
  for k in range(n_folds):
    idx = np.flatnonzero(fold_arr == k)
    if idx.size:
      a_parts.append(a_total - s_sq[idx].T @ pp_outer[idx])
      u_parts.append(u_total - s_y[idx].T @ p_basis[idx])
    else:
      # An empty fold trains on every row; no row is evaluated with it.
      a_parts.append(a_total)
      u_parts.append(u_total)
  a_mat = xnp.stack(a_parts).reshape(n_folds, n_grid, n_p, n_p)
  a_mat = a_mat + ridge * xnp.eye(n_p)[None, None, :, :]
  # The RHS is kept as an explicit (P, 1) column rather than a (P,) vector.
  # NumPy >= 2.0 only treats `b` as a vector when it is EXACTLY 1-D; a 3-D `b`
  # against a 4-D `a` would otherwise be read as a stack of (G, P) matrices and
  # raise a core-dimension mismatch.
  b_vec = xnp.stack(u_parts).reshape(n_folds, n_grid, n_p, 1)

  if jax_lib is not None and jnp_lib is not None:
    solve_batched = jax_lib.vmap(jax_lib.vmap(jnp_lib.linalg.solve))
    beta = solve_batched(a_mat, b_vec)
  else:
    beta = np.linalg.solve(a_mat, b_vec)
  beta = beta.reshape(n_folds, n_grid, n_p)

  # theta_hat(X_i; alpha) = p_i' beta^(-k(i))(alpha): every row is evaluated
  # with the beta of the fold that held it out  -> (n, G), one small GEMM
  # per fold (no (n, G, P) tensor).
  theta_hat = np.zeros((int(n_obs), int(n_grid)), dtype=float)
  for k in range(n_folds):
    idx = np.flatnonzero(fold_arr == k)
    if idx.size:
      theta_hat[idx] = np.asarray(p_basis[idx] @ beta[k].T, dtype=float)
  resid = np.asarray(y_tilde, dtype=float)[:, None] - (
      np.asarray(s_res, dtype=float) * theta_hat
  )  # (n, G)
  obs = resid * resid
  loss = obs.mean(axis=0)
  if return_obs:
    return loss, obs
  return loss


def _cluster_codes(clusters: Optional[np.ndarray]) -> Optional[np.ndarray]:
  """Integer codes ``0..G-1`` of the cluster labels, or ``None``.

  Missing labels form their own cluster. ``None`` is returned when there are
  fewer than two distinct clusters.

  Args:
    clusters: ``(n,)`` cluster labels, or ``None``.

  Returns:
    ``(n,)`` integer array of cluster codes, or ``None`` when ``clusters`` is
    ``None`` or has fewer than two distinct labels.
  """
  if clusters is None:
    return None
  codes, uniques = pd.factorize(pd.Series(np.asarray(clusters)))
  if uniques.size < 2:
    return None
  return np.where(codes < 0, uniques.size, codes).astype(int)


def count_clusters(clusters: Optional[np.ndarray]) -> int:
  """Number of distinct cluster labels (missing labels count as one)."""
  codes = _cluster_codes(clusters)
  if codes is None:
    return 1 if clusters is not None and len(clusters) > 0 else 0
  return int(np.count_nonzero(np.bincount(codes)))


def _cluster_variance(centered: np.ndarray, codes: np.ndarray) -> np.ndarray:
  """``G / (G - 1) * sum_g (sum_{i in g} centered_i)^2`` per column.

  The Liang-Zeger (CR1) variance of the column sums for one clustering.

  Args:
    centered: ``(n, k)`` per-observation values, already centered on their
      column means.
    codes: ``(n,)`` integer cluster codes ``0..G-1`` (see ``_cluster_codes``).

  Returns:
    A ``(k,)`` array with the CR1 variance of each column's sum.
  """
  n_groups = int(codes.max()) + 1
  sums = np.zeros((n_groups, centered.shape[1]), dtype=float)
  np.add.at(sums, codes, centered)
  n_groups = int(np.count_nonzero(np.bincount(codes)))
  factor = n_groups / max(n_groups - 1, 1)
  return factor * np.sum(sums * sums, axis=0)


def cluster_robust_se(
    values: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    clusters2: Optional[np.ndarray] = None,
) -> Any:
  """Cluster-robust ``se_obs`` of the column means (one-way or two-way).

  The standard error of the 1-SE decision is the cluster-robust ``se_obs`` of
  the per-observation paired differences ``d_i`` (``n_g`` = cluster size,
  ``D_g = sum_{i in g} d_i``, ``d_bar_obs = (1/n_sel) sum_i d_i``).  With one
  clustering (``clusters``; ``g = 1..G``):

      se_obs = (1 / n_sel) * sqrt( G / (G - 1) * sum_g (D_g - n_g d_bar_obs)^2 )

  (square root of the Liang-Zeger cluster-robust variance of the intercept-only
  regression ``d_i = mu + eps_i``; Diebold & Mariano 1995).  With two
  clusterings (``clusters`` and ``clusters2``, the two adopted keys, e.g. geo
  and date block) the variance is the two-way cluster-robust variance of
  Cameron, Gelbach & Miller (2011):

      V = V_key1 + V_key2 - V_cell

  where ``V_cell`` clusters on the intersection cells key1 x key2 and every
  term carries its own ``G / (G - 1)`` factor; it accounts for dependence
  along BOTH keys (common date-block shocks across geos AND persistent geo
  effects across dates), which the one-way se on either key alone ignores.
  When ``V <= 0`` (possible in small samples) the larger of the two one-way
  variances is used instead.  Under independence (no cluster labels, or fewer
  than two clusters) the se reduces to ``sd_i(d_i) / sqrt(n_sel)``.  The same
  function gives ``se(alpha)`` of Diag2 column by column.

  Args:
    values: ``(n,)`` or ``(n, G)`` per-observation values ``d_i``.
    clusters: ``(n,)`` cluster labels of the first key, or ``None``.
    clusters2: ``(n,)`` cluster labels of the second key (two-way), or
      ``None`` (one-way on ``clusters``).

  Returns:
    A float for 1-D input, a ``(G,)`` array for 2-D input (NaN if ``n < 2``).
  """
  arr = np.asarray(values, dtype=float)
  one_d = arr.ndim == 1
  mat = arr.reshape(arr.shape[0], -1)
  n_obs = mat.shape[0]
  if n_obs < 2:
    out = np.full(mat.shape[1], np.nan)
    return float(out[0]) if one_d else out
  # D_g - n_g d_bar_obs == sum_{i in g} (d_i - d_bar_obs).
  centered = mat - mat.mean(axis=0, keepdims=True)
  codes = [c for c in (_cluster_codes(clusters), _cluster_codes(clusters2))
           if c is not None]
  if not codes:
    out = mat.std(axis=0, ddof=1) / math.sqrt(n_obs)
  elif len(codes) == 1:
    out = np.sqrt(_cluster_variance(centered, codes[0])) / float(n_obs)
  else:
    v_1 = _cluster_variance(centered, codes[0])
    v_2 = _cluster_variance(centered, codes[1])
    cell_codes, _ = pd.factorize(
        pd.Series(codes[0] * (int(codes[1].max()) + 1) + codes[1])
    )
    v_cell = _cluster_variance(centered, cell_codes.astype(int))
    var = v_1 + v_2 - v_cell
    var = np.where(var > 0.0, var, np.maximum(v_1, v_2))
    out = np.sqrt(var) / float(n_obs)
  return float(out[0]) if one_d else out


def linear_test_critical_value(
    k: float, df: Optional[int], use_t: bool
) -> float:
  """The multiplier ``crit`` of ``se`` in the 1-SE decision.

  ``crit = k`` (normal reference, the default) or, with
  ``alpha_selection.linear_test_t_critical: true``, the Student-t value with
  the same one-sided tail probability as ``k`` under the normal,
  ``crit = t_df^{-1}(Phi(k))``, ``df = G - 1`` with ``G`` the number of
  clusters (``min(G_key1, G_key2)`` for the two-way se).  Few clusters make
  the se noisy; the t value compensates (``k = 1``: ``G = 30 -> 1.02``,
  ``10 -> 1.06``, ``4 -> 1.20``).

  Args:
    k: ``linear_test_k``.
    df: Degrees of freedom; ``None`` or ``< 1`` -> ``k``.
    use_t: Whether the t critical value is requested.

  Returns:
    ``crit``.
  """
  k = float(k)
  if not use_t or df is None or int(df) < 1 or not np.isfinite(k):
    return k
  return float(scipy_stats.t.ppf(scipy_stats.norm.cdf(k), int(df)))


def linear_test_decision(d_bar: float, se: float, k: float) -> bool:
  """The 1-SE linear test: adopt saturation iff ``d_bar < -k * se``.

  A NaN ``d_bar`` or ``se`` never adopts saturation (parsimony: linear).

  Args:
    d_bar: Mean paired difference ``L_bar(alpha_hat) - L_bar_T`` (identity
      candidate).
    se: The cluster-robust ``se_obs`` of the paired differences.
    k: The multiplier of ``se`` (``linear_test_k``, or the critical value of
      :func:`linear_test_critical_value`).

  Returns:
    True when saturation is adopted.
  """
  if not (np.isfinite(d_bar) and np.isfinite(se)):
    return False
  return bool(d_bar < -float(k) * float(se))


def diag2_flatness(
    family: str,
    alpha_grid: np.ndarray,
    one_se_set: np.ndarray,
    flatness_ratio: float = DEFAULT_FLATNESS_RATIO,
) -> Tuple[float, bool, float]:
  """Diag2: 1-SE flatness over ``A_1SE``.

  For ``tanh`` / ``exp`` (``T_1/2``) and ``log`` (``T_knee``) the statistic
  is ``T_1/2^max / T_1/2^min`` over the grid points in the 1-SE set and the
  rule is ``ratio > flatness_ratio``; for the scale-free ``power`` family it
  is ``alpha^max - alpha^min`` with the fixed rule ``> 0.4``
  (:data:`POWER_FLATNESS_RANGE`).  Contiguity of the set is not required.

  Args:
    family: The saturation family.
    alpha_grid: ``(G,)`` alpha values of the final grid.
    one_se_set: ``(G,)`` membership in ``A_1SE``.
    flatness_ratio: The configured threshold (ignored for ``power``).

  Returns:
    ``(diag2_ratio, diag2_fired, threshold_applied)``; the ratio is NaN
    when the set is empty.
  """
  alphas = np.asarray(alpha_grid, dtype=float)
  inside = np.asarray(one_se_set, dtype=bool)
  if alphas.size == 0 or not np.any(inside):
    return float("nan"), False, float(flatness_ratio)
  if family == "power":
    picked = alphas[inside]
    spread = float(picked.max() - picked.min())
    return spread, bool(spread > POWER_FLATNESS_RANGE), POWER_FLATNESS_RANGE
  t_half = np.asarray(
      saturation.half_saturation_point(family, alphas[inside]), dtype=float
  )
  t_half = t_half[np.isfinite(t_half) & (t_half > 0)]
  if t_half.size == 0:
    return float("nan"), False, float(flatness_ratio)
  ratio = float(t_half.max() / t_half.min())
  return ratio, bool(ratio > float(flatness_ratio)), float(flatness_ratio)


def t_ref_consistency(
    t_ref_phase0: float,
    t_values: np.ndarray,
    quantile: float = saturation.DEFAULT_T_REF_QUANTILE,
    tolerance: float = DEFAULT_T_REF_TOLERANCE,
) -> Tuple[bool, float, float]:
  """Diag6: unit consistency of ``T_ref``.

  ``T_ref^(1)`` is recomputed with the Phase 0 definition
  ``q95(T | T > 0)`` from the treatment column Phase 1 actually receives and
  compared with ``T_ref^(0)`` of ``summary_alpha_selection.json``::

      |T_ref^(1) - T_ref^(0)| / T_ref^(0) < t_ref_tolerance

  Args:
    t_ref_phase0: ``T_ref^(0)`` saved by Phase 0.
    t_values: The treatment column received by Phase 1 (Phase 0 scale).
    quantile: ``saturation.t_ref_quantile`` used by Phase 0.
    tolerance: ``alpha_selection.t_ref_tolerance`` (relative error).

  Returns:
    ``(holds, t_ref_phase1, relative_error)``.
  """
  t_ref_0 = float(t_ref_phase0)
  t_ref_1 = float(saturation.reference_point(t_values, quantile))
  if not (np.isfinite(t_ref_0) and t_ref_0 > 0.0):
    return False, t_ref_1, float("inf")
  rel_err = abs(t_ref_1 - t_ref_0) / t_ref_0
  return bool(rel_err < float(tolerance)), t_ref_1, float(rel_err)


def _warn(messages: List[str], msg: str) -> None:
  """Records a Phase 0 warning (message list + ``UserWarning``)."""
  messages.append(msg)
  warnings.warn(msg, UserWarning)


def select_alpha(
    df_sel: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: Sequence[str],
    basis_builder: Any,
    scaler: Any,
    family: str,
    saturation_cfg: Dict[str, Any],
    selection_cfg: Dict[str, Any],
    cluster_labels: np.ndarray,
    ml_models_y: Sequence[str],
    ml_models_t: Sequence[str],
    time_budget: int,
    n_jobs: int = -1,
    progress_dir: Optional[str] = None,
    random_state: int = 0,
    divisor: Optional[np.ndarray] = None,
    normalization: str = "none",
    n_total: Optional[int] = None,
    se_cluster_labels: Optional[np.ndarray] = None,
    se_cluster_key: Optional[str] = None,
    item_col: Optional[str] = None,
    geo_col: Optional[str] = None,
    date_col: Optional[str] = None,
    fold_key_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    fold_key_names: Optional[Tuple[str, str]] = None,
    t_ref: Optional[float] = None,
    t_max: Optional[float] = None,
) -> AlphaSelectionResult:
  """Runs Phase 0 / Stage B: alpha selection by profiled R-loss minimization.

  Implements Stage B end to end: the nuisance learners (fixed defaults,
  optional legacy learner selection) and the T-basis (``basis_dim_T:
  auto``), the seed loop (all alphas AND the identity candidate share the
  partition and nuisances of a seed), the batched alpha-dependent algebra
  on the half-saturation grid (with automatic extension), the
  intercept-only diagnostic curve and the identity candidate, "average
  the curves, then take the argmin" plus Brent (grid + Brent only, no
  per-alpha retraining) and the 1-SE comparison with the identity
  candidate (cluster-robust ``se_obs`` only), and the diagnostics
  Diag1-Diag5 (Diag1 / Diag2 warn and continue; Diag6 belongs to the
  start of Phase 1, see :func:`t_ref_consistency`).

  Every candidate is the NORMALIZED
  ``fbar(T; alpha) = f(T; alpha) / f(T_ref; alpha)`` --
  the grid curves, the identity candidate ``fbar_T = T / T_ref`` and the
  intercept curve -- so all candidates pass through ``(T_ref, 1)`` and the
  absolutely-regularized inner beta solve (``ridge``) treats them on an
  equal footing.  The basis coefficients are ``c(alpha) = P fbar_alpha`` on a
  fixed grid over ``[0, T_max]`` (``T_max`` the
  ``saturation.BASIS_T_MAX_QUANTILE`` quantile of ``T > 0`` of the full data).
  After the grid extensions an ``alpha_hat`` on the linear-side edge adopts
  the linear model regardless of the 1-SE result, and so does a T-basis with
  ``cond(B) >= saturation.MAX_BASIS_CONDITION`` (its Stage B statistics are
  computed and reported but not trusted).

  ML fits: ``2 * S * K`` refits (plus ``N_y + N_t`` searches when
  ``learners.learner_selection: true``).  Neither the grid size, the grid
  extensions, the identity candidate nor the intercept curve adds a fit.

  Args:
    df_sel: ``D_sel`` (the 20% selection sample).
    y_col: Outcome column.
    d_col: Raw treatment column.
    x_cols: Nuisance covariates.
    basis_builder: Callable ``(df_norm, df_orig) -> DataFrame`` returning the
      CATE basis ``p(X)`` (the same builder Phases 1-3 use).
    scaler: The pipeline scaler (deep-copied; never mutated).
    family: The configured saturation family.
    saturation_cfg: The ``saturation`` config block.
    selection_cfg: The ``alpha_selection`` config block.
    cluster_labels: Cluster label per row of ``df_sel`` (cluster-level folds).
    ml_models_y: Candidate learners for ``l_hat`` (learner_selection only).
    ml_models_t: Candidate learners for ``mu_hat`` (learner_selection only).
    time_budget: The pipeline-wide FLAML time budget (learner_selection
      only; scaled to ``D_sel`` unless ``alpha_selection.time_budget`` is set).
    n_jobs: joblib parallelism.
    progress_dir: Directory where ``.done`` markers are written.
    random_state: Base seed.
    divisor: Normalization bridge. The scaler divisor ``c`` per row
      of ``df_sel``; alpha is estimated against ``T / c``.
    normalization: Name of the normalization the divisor came from.
    n_total: Row count of the FULL sample, used to scale the time budget.
    se_cluster_labels: Labels of the selected key
      with the FEWER clusters.  They give the ONE-WAY cluster-robust se
      (``se_one_way``), which is the decision se only when the two adopted
      keys are not available (``fold_key_values is None``); ``None`` -> iid.
    se_cluster_key: Name of that key (reported only).
    item_col: Item column (RidgeCV small-sample features).
    geo_col: Geo column (RidgeCV small-sample features).
    date_col: Date column (RidgeCV seasonal Fourier terms).
    fold_key_values: The ``(k1, k2)`` values per row of
      ``df_sel``.  When given, the per-seed folds are cell-level
      (:func:`cell_kfold`, the round-robin rule with ``M = K``) AND the se of
      the 1-SE decision and of Diag2 is the TWO-WAY cluster-robust se on
      these two keys (:func:`cluster_robust_se`); when ``None`` the folds are
      built on ``cluster_labels`` (single key / fallback) and the se is
      one-way on ``se_cluster_labels``.
    fold_key_names: Names of the two keys (reported only).
    t_ref: The reference point ``T_ref = q95(T | T > 0)`` on
      the scale ``T / c`` Phase 0 works on, computed ONCE on the FULL data
      before the split by the caller.  ``None`` (direct calls) computes it
      from ``D_sel`` with ``saturation.t_ref_quantile`` and says so.
    t_max: The right end ``T_max`` of the T-basis domain
      (:func:`saturation.basis_domain_max`, the 99.9th percentile of
      ``T > 0``) on the same scale, computed ONCE on the FULL data before the
      split by the caller.  ``None`` (direct calls) computes it from ``D_sel``
      and says so.

  Returns:
    The :class:`AlphaSelectionResult`.
  """
  messages: List[str] = []
  n_seeds = int(selection_cfg.get("n_seeds", 10))
  k_folds = int(selection_cfg.get("k_folds", 5))
  refine = str(selection_cfg.get("refine", "brent")).strip().lower()
  linear_test_k = float(selection_cfg.get("linear_test_k", 1.0))
  linear_test_t_critical = bool(
      selection_cfg.get("linear_test_t_critical", False)
  )
  flatness_ratio = float(
      selection_cfg.get("flatness_ratio", DEFAULT_FLATNESS_RATIO)
  )
  fallback_half_saturation = selection_cfg.get(
      "fallback_half_saturation_T", None
  )
  learner_cfg = saturation.resolve_learner_config(
      selection_cfg.get("learners")
  )
  basis_tol = float(saturation_cfg.get("basis_tol", 1e-3))

  ridge = float(saturation_cfg.get("ridge", 1e-4))
  grid_cfg = dict(saturation_cfg.get("grid", {}) or {})
  max_extensions = int(grid_cfg.get("max_extensions", 2) or 0)

  # The scaler is deep-copied so that Phase 0 can never disturb the state the
  # existing Phases 1-3 rely on.
  scaler_local = copy.deepcopy(scaler)
  df_norm = scaler_local.fit_transform(df_sel)

  y_values = (
      pd.to_numeric(df_norm[y_col], errors="coerce")
      .fillna(0.0)
      .to_numpy(dtype=float)
  )
  x_frame = df_norm[list(x_cols)].apply(pd.to_numeric, errors="coerce")
  x_values = x_frame.fillna(0.0).to_numpy(dtype=float)
  # Normalization bridge: alpha is estimated against the NORMALIZED
  # treatment `T_tilde = T / c`; the raw-scale parameters are recovered
  # downstream by `saturation.rescale_alpha_to_raw`.
  t_raw = pd.to_numeric(df_sel[d_col], errors="coerce").fillna(0.0).to_numpy(
      dtype=float
  )
  if np.any(t_raw < 0.0):
    _warn(
        messages,
        f"[Phase 0 / Stage B] Treatment '{d_col}' contains"
        f" {int(np.count_nonzero(t_raw < 0.0))} negative value(s). All"
        " saturation families are defined on T >= 0, so those rows are"
        " clipped to 0 by `apply_saturation`. Use a normalization that"
        " preserves non-negativity (median_normalize /"
        " population_median_normalize) and check the raw data.",
    )
  if divisor is None:
    div_values = np.ones_like(t_raw)
  else:
    div_values = np.asarray(divisor, dtype=float).reshape(-1)
    div_values = np.where(
        np.isfinite(div_values) & (div_values > 0.0), div_values, 1.0
    )
  t_eff = t_raw / div_values

  # ---------------------------------------------------------------
  # The reference point T_ref of fbar = f / f(T_ref; alpha).
  # Always on.  Normally computed once on the FULL data by the caller
  # (before the split, on the scale T / c); a direct call without it falls
  # back to D_sel.
  # ---------------------------------------------------------------
  t_ref_quantile = float(
      saturation_cfg.get("t_ref_quantile", saturation.DEFAULT_T_REF_QUANTILE)
  )
  if t_ref is None:
    t_ref_value = saturation.reference_point(t_eff, t_ref_quantile)
    messages.append(
        "[Phase 0 / Stage B] T_ref was not supplied; computed as"
        f" q{int(round(100 * t_ref_quantile))}(T | T > 0) = {t_ref_value:.6g}"
        " on D_sel (the orchestrator computes it on the full data before the"
        " split and passes it in)."
    )
  else:
    t_ref_value = float(t_ref)
    if not (np.isfinite(t_ref_value) and t_ref_value > 0.0):
      raise ValueError(
          f"t_ref must be finite and strictly positive (got {t_ref});"
          " T_ref is defined as q95(T | T > 0)."
      )
  # Raw-scale reference point c * T_ref for the human reader (only well
  # defined when the divisor is a single constant on D_sel).
  t_ref_raw: Optional[float] = None
  if div_values.size and np.allclose(div_values, div_values[0]):
    t_ref_raw = float(
        saturation.raw_scale_reference(t_ref_value, float(div_values[0]))
    )

  # ---------------------------------------------------------------
  # The dimensionless half-saturation grid (the
  # log family's grid variable is T_knee = alpha).
  # ---------------------------------------------------------------
  grid = saturation.build_half_saturation_grid(t_eff, family, grid_cfg)

  # ---------------------------------------------------------------
  # T-basis: J = smallest basis size meeting eps(J) < basis_tol,
  # chosen ONCE on the initial grid and kept fixed (Diag1: after a grid
  # extension only eps(J) is recomputed, no retraining).
  # The basis lives on [0, T_max] with T_max the BASIS_T_MAX_QUANTILE
  # quantile of T > 0 (full data, passed in like T_ref), interior knots at
  # quantiles of T > 0, and carries the SVD least-squares projector P on the
  # fixed grid (equally spaced points, quantiles of T > 0 and the Greville
  # abscissae), so c(alpha) = P fbar_alpha.  Bases with
  # cond(B) >= MAX_BASIS_CONDITION are skipped by the automatic choice of J
  # and, if none remains (or J is pinned), flagged `ill_conditioned`.
  # ---------------------------------------------------------------
  if t_max is None:
    t_max_value = saturation.basis_domain_max(t_eff)
    messages.append(
        "[Phase 0 / Stage B] T_max was not supplied; computed as"
        f" q{100 * saturation.BASIS_T_MAX_QUANTILE:g}(T | T > 0) ="
        f" {t_max_value:.6g} on D_sel (the orchestrator computes it on the"
        " full data before the split and passes it in)."
    )
  else:
    t_max_value = float(t_max)
    if not (np.isfinite(t_max_value) and t_max_value > 0.0):
      raise ValueError(
          f"t_max must be finite and strictly positive (got {t_max});"
          " T_max is defined as q99.9(T | T > 0)."
      )
  dim_range = saturation_cfg.get("basis_dim_range", (6, 20)) or (6, 20)
  basis_sel = saturation.select_basis_dim(
      t_eff,
      family,
      grid.alphas,
      basis_dim=saturation_cfg.get("basis_dim_T", "auto"),
      tol=basis_tol,
      dim_range=tuple(int(v) for v in dim_range),
      t_ref=t_ref_value,
      t_max=t_max_value,
  )
  treatment_basis = basis_sel.basis
  if basis_sel.ill_conditioned:
    _warn(
        messages,
        "[Phase 0 / Stage B] Diag1 WARNING: the T-basis matrix is"
        f" ill-conditioned (J = {treatment_basis.n_basis}, cond(B) ="
        f" {treatment_basis.basis_cond:.2e} >="
        f" {saturation.MAX_BASIS_CONDITION:g}; basis_dim_T ="
        f" {saturation_cfg.get('basis_dim_T', 'auto')}). The basis"
        " coefficients c(alpha), m_hat(X; alpha) and every Stage B statistic"
        " derived from them cannot be trusted, so the linear model is adopted"
        " (decision_reason = basis_ill_conditioned); the statistics are still"
        " computed and reported.",
    )
  b_targets = treatment_basis.transform(t_eff)  # (n, J)
  n_basis = b_targets.shape[1]
  greville = treatment_basis.greville_abscissae()  # (J,)
  # The identity candidate fbar_T = T / T_ref
  # has the exact basis representation (xi_j / T_ref) (Greville abscissae), so
  # m_hat_T = (1 / T_ref) sum_j xi_j mu_hat_j without any extra training.
  coef_identity = greville / t_ref_value  # (J,)
  fbar_identity = t_eff / t_ref_value  # (n,)
  # Diag4: the positive-spend quantiles q_p(T > 0) and the
  # linear references q_p / T_ref.
  t_pos = t_eff[t_eff > 0]
  if t_pos.size:
    q_pos = np.quantile(t_pos, [p / 100.0 for p in DIAG4_PERCENTILES])
  else:
    q_pos = np.full(len(DIAG4_PERCENTILES), np.nan)
  linear_ref_quantiles = q_pos / t_ref_value

  p_basis = np.asarray(basis_builder(df_norm, df_sel), dtype=float)
  p_ate = np.ones((p_basis.shape[0], 1), dtype=float)

  n_obs = len(df_sel)
  rng = np.random.default_rng(random_state)
  # Cell-level folds (a fresh permutation pi per seed) when
  # the two adopted keys are known; single-key / row-level folds otherwise.
  fold_key_arrays: Optional[Tuple[np.ndarray, np.ndarray]]
  if fold_key_values is not None:
    key1_values = np.asarray(fold_key_values[0])
    key2_values = np.asarray(fold_key_values[1])
    fold_key_arrays = (key1_values, key2_values)
    seed_partitions = [
        cell_kfold(
            key1_values, key2_values, k_folds,
            np.random.default_rng(random_state + 100 + s),
        )
        for s in range(n_seeds)
    ]
  else:
    fold_key_arrays = None
    seed_partitions = [
        _cluster_kfold(
            cluster_labels, k_folds,
            np.random.default_rng(random_state + 100 + s),
        )
        for s in range(n_seeds)
    ]

  # ---------------------------------------------------------------
  # Nuisance fits, one model per (seed, fold).  The model of fold k is
  # trained on the other folds and predicts ONLY the rows of fold k, so every
  # row carries exactly one out-of-fold prediction per seed (cross-fitting):
  # l_oof[s, i] = l_hat^(-k(i))(X_i), mu_oof[s, i, :] = mu_hat^(-k(i))(X_i).
  # No in-sample prediction is kept anywhere.
  # ---------------------------------------------------------------
  l_oof = np.zeros((n_seeds, n_obs), dtype=float)
  mu_oof = np.zeros((n_seeds, n_obs, n_basis), dtype=float)
  multioutput_native = True
  if learner_cfg["learner_selection"]:
    (l_oof, mu_oof, n_ml_fits, best_y, best_t, multioutput_native,
     learner_desc) = _v3_learner_selection_fits(
         x_values, y_values, b_targets, cluster_labels, seed_partitions,
         ml_models_y, ml_models_t, time_budget, selection_cfg, n_total,
         n_jobs, progress_dir, random_state, rng, messages)
  else:
    if n_obs < int(learner_cfg["small_sample_threshold"]) and str(
        learner_cfg["small_sample_model"]
    ).lower() in ("ridge_cv", "ridgecv", "ridge"):
      x_fit, spline_cols, _ = saturation.build_small_sample_features(
          df_norm, list(x_cols), item_col, geo_col, date_col
      )
    else:
      x_fit, spline_cols = x_values, []
    learners = saturation.resolve_stage_b_learners(
        learner_cfg, n_obs, spline_cols, x_fit.shape[1]
    )
    messages.extend(learners.messages)
    groups = np.asarray(cluster_labels)
    fit_jobs = _cross_fit_jobs(seed_partitions, k_folds)

    def _fit_one(s, k, kind, train, held_out):
      learner = learners.y_learner if kind == "y" else learners.t_learner
      target = y_values if kind == "y" else b_targets
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = learner.fit_predict(
            x_fit[train], target[train], groups[train], x_fit[held_out],
            seed=int(random_state * 10007 + 1000 * s + k),
        )
      _touch(
          os.path.join(progress_dir, f"seed_{s}_{k}_{kind}.done")
          if progress_dir
          else None
      )
      return preds

    fit_out = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(_fit_one)(*job) for job in fit_jobs
    )
    _scatter_oof_predictions(fit_jobs, fit_out, l_oof, mu_oof)
    n_ml_fits = len(fit_jobs)
    best_y = learners.y_learner.kind
    best_t = learners.t_learner.kind
    learner_desc = {"learner_selection": False, **learners.describe()}

  # ---------------------------------------------------------------
  # Alpha-dependent algebra (matrix algebra only).
  # Every candidate is fbar(T; alpha) =
  # f / f(T_ref; alpha); the basis coefficients are c(alpha) = P fbar_alpha.
  # ---------------------------------------------------------------
  def _evaluate(alphas: np.ndarray) -> Dict[str, np.ndarray]:
    """CATE / ATE losses, per-obs losses and shape quantiles for ``alphas``."""
    alphas = np.asarray(alphas, dtype=float)
    coefs = treatment_basis.project_family(family, alphas)  # (J, g)
    # (n, g)
    fbar_vals = treatment_basis.normalized_values(t_eff, family, alphas)
    cate = np.zeros((n_seeds, alphas.size))
    ate = np.zeros((n_seeds, alphas.size))
    obs = np.zeros((n_obs, alphas.size))
    for s in range(n_seeds):
      loss_s, obs_s = _profile_losses(
          l_oof[s], mu_oof[s], y_values, fbar_vals, coefs, p_basis,
          seed_partitions[s], ridge, return_obs=True,
      )
      cate[s] = loss_s
      obs += obs_s / float(n_seeds)
      # The SAME Y~, S~(alpha) with p(X) = 1
      # (intercept curve; visualization only).
      ate[s] = _profile_losses(
          l_oof[s], mu_oof[s], y_values, fbar_vals, coefs, p_ate,
          seed_partitions[s], ridge,
      )
    # Diag4: phi_p(alpha) = fbar(q_p(T > 0); alpha).
    if np.all(np.isfinite(q_pos)):
      phi = saturation.normalized_saturation(
          q_pos[:, None], family, alphas[None, :], t_ref_value
      )  # (len(p), g)
    else:
      phi = np.full((len(DIAG4_PERCENTILES), alphas.size), np.nan)
    return {
        "cate": cate,
        "ate": ate,
        "obs": obs,
        "phi": phi,
        "approx": treatment_basis.approximation_error(family, alphas),
    }

  evals = _evaluate(grid.alphas)

  # Automatic extension at a grid edge (x4, <= 2 times).
  while True:
    mean_curve = evals["cate"].mean(axis=0)
    arg = int(np.argmin(mean_curve))
    at_edge = grid.coord.size >= 2 and arg in (0, grid.coord.size - 1)
    if not (at_edge and grid.extensible and grid.extensions < max_extensions):
      break
    side = "lower" if arg == 0 else "upper"
    new_grid = saturation.extend_alpha_grid(grid, side, factor=4.0)
    n_added = new_grid.coord.size - grid.coord.size
    added_alphas = (
        new_grid.alphas[:n_added] if side == "lower"
        else new_grid.alphas[-n_added:]
    )
    added = _evaluate(added_alphas)
    for key, val in evals.items():
      axis = 1 if val.ndim == 2 else 0
      pieces = [added[key], val] if side == "lower" else [val, added[key]]
      evals[key] = np.concatenate(pieces, axis=axis)
    messages.append(
        f"Stage B: argmin at the {side} grid edge -> grid extended x4 on that"
        f" side (extension {new_grid.extensions}/{max_extensions}; matrix"
        " algebra only, no ML retraining)."
    )
    grid = new_grid

  alpha_grid = np.asarray(grid.alphas, dtype=float)
  per_seed_loss = evals["cate"]  # (S, G)
  per_seed_loss_ate = evals["ate"]
  obs_loss = evals["obs"]  # (n, G)

  # Average the CURVES, then take the argmin (never the average of argmins).
  mean_loss = np.mean(per_seed_loss, axis=0)
  grid_argmin = int(np.argmin(mean_loss))
  grid_alpha_hat = float(alpha_grid[grid_argmin])
  per_seed_argmin = np.argmin(per_seed_loss, axis=1)
  per_seed_alpha = alpha_grid[per_seed_argmin]
  per_seed_min_loss = per_seed_loss[np.arange(n_seeds), per_seed_argmin]

  # ---------------------------------------------------------------
  # Diag1: eps(J) recomputed over the FINAL grid with the
  # SAME J (no retraining); fires when eps(J) >= basis_tol (approximation
  # too coarse: warn and continue) or when the basis is ill-conditioned
  # (cond(B) >= MAX_BASIS_CONDITION: warned above, linear model adopted
  # below).
  # ---------------------------------------------------------------
  basis_approx_error = float(np.max(evals["approx"]))
  diag1_eps_fired = bool(basis_approx_error >= basis_tol)
  diag1_fired = bool(diag1_eps_fired or basis_sel.ill_conditioned)
  if diag1_eps_fired:
    _warn(
        messages,
        f"[Phase 0 / Stage B] Diag1 WARNING: basis approximation error eps(J="
        f"{n_basis}) = {basis_approx_error:.2e} >= basis_tol = {basis_tol:g}"
        f" on the final grid ({basis_sel.mode} J; initial-grid tolerance"
        f" {'met' if basis_sel.satisfied else 'not met'}; cond(B) ="
        f" {treatment_basis.basis_cond:.2e}). The treatment-side"
        " nuisance carries an alpha-dependent approximation bias; continuing.",
    )

  # ---------------------------------------------------------------
  # Edge of the (extended) grid, reported in the
  # T_1/2 orientation: "upper" = the linear (slow-saturation) side, "lower"
  # = the fast-saturation side.  `grid.linear_side` maps the index end to the
  # orientation for alpha-coordinate grids (power / manual).  Reached here
  # after the extension loop, so the edge is final: either the two
  # extensions are used up or the grid is not extensible (manual / power).
  # ---------------------------------------------------------------
  grid_edge_hit = "none"
  if alpha_grid.size >= 2 and grid_argmin in (0, alpha_grid.size - 1):
    index_end = "upper" if grid_argmin == alpha_grid.size - 1 else "lower"
    grid_edge_hit = "upper" if index_end == grid.linear_side else "lower"
  grid_edge_unresolved = grid_edge_hit != "none"
  if grid_edge_hit == "upper":
    _warn(
        messages,
        "[Phase 0 / Stage B] alpha reached the linear-side grid edge (upper"
        f" T_1/2) after {grid.extensions} extension(s) (grid mode="
        f"{grid.mode}); the data do not identify a saturation point inside"
        " the grid, so the linear model is adopted regardless of the 1-SE"
        " result (decision_reason = grid_edge_upper).",
    )
  elif grid_edge_hit == "lower":
    _warn(
        messages,
        "[Phase 0 / Stage B] The R-loss argmin is at the fast-saturation"
        f" (lower T_1/2) grid edge after {grid.extensions} extension(s) (grid"
        f" mode={grid.mode}); warning only -- the 1-SE decision and Diag2"
        " decide.",
    )

  # ---------------------------------------------------------------
  # Grid + Brent only (coordinate = grid coordinate).
  # ---------------------------------------------------------------
  alpha_est = grid_alpha_hat
  refined = False
  if refine == "brent" and alpha_grid.size >= 3:
    lo = float(grid.coord[max(0, grid_argmin - 1)])
    hi = float(grid.coord[min(alpha_grid.size - 1, grid_argmin + 1)])
    if hi > lo:

      def _scalar_loss(coord_value: float) -> float:
        alpha_value = grid.alpha_at(float(coord_value))
        alpha_one = np.array([alpha_value])
        coefs_one = treatment_basis.project_family(family, alpha_one)
        fbar_one = treatment_basis.normalized_values(t_eff, family, alpha_one)
        losses = [
            _profile_losses(
                l_oof[s], mu_oof[s], y_values, fbar_one, coefs_one,
                p_basis, seed_partitions[s], ridge,
            )[0]
            for s in range(n_seeds)
        ]
        return float(np.mean(losses))

      try:
        opt_res = scipy_optimize.minimize_scalar(
            _scalar_loss, bounds=(lo, hi), method="bounded"
        )
        if opt_res.success and float(opt_res.fun) <= float(
            mean_loss[grid_argmin]
        ):
          alpha_est = float(grid.alpha_at(float(opt_res.x)))
          refined = True
      except (ValueError, RuntimeError) as err:  # pragma: no cover
        messages.append(f"Brent refinement skipped ({err}).")

  # Per-seed / per-observation losses and shape quantiles at the (refined)
  # alpha_hat.
  if refined:
    at_hat = _evaluate(np.array([alpha_est]))
    loss_at_hat = at_hat["cate"][:, 0]
    obs_at_hat = at_hat["obs"][:, 0]
    phi_at_hat = at_hat["phi"][:, 0]
  else:
    loss_at_hat = per_seed_loss[:, grid_argmin]
    obs_at_hat = obs_loss[:, grid_argmin]
    phi_at_hat = evals["phi"][:, grid_argmin]
  rloss_min = float(np.mean(loss_at_hat))  # L_bar(alpha_hat)

  # ---------------------------------------------------------------
  # Identity candidate fbar_T(T) = T / T_ref
  # with the exact nuisance m_hat_T = (1 / T_ref) sum_j xi_j mu_hat_j, on the
  # same folds, l_hat, p(X) and ridge.  L_hat_{s,T} per seed.
  # ---------------------------------------------------------------
  identity_losses = np.zeros(n_seeds)
  identity_obs = np.zeros(n_obs)
  r_t_seeds = np.zeros(n_seeds)
  sd_resid_seeds = np.zeros(n_seeds)
  sd_t = float(np.std(t_eff))
  for s in range(n_seeds):
    loss_s, obs_s = _profile_losses(
        l_oof[s], mu_oof[s], y_values, fbar_identity.reshape(-1, 1),
        coef_identity.reshape(-1, 1), p_basis, seed_partitions[s], ridge,
        return_obs=True,
    )
    identity_losses[s] = float(loss_s[0])
    identity_obs += obs_s[:, 0] / float(n_seeds)
    # Diag3: T~_i = T_i / T_ref - m_hat_T(X_i) out of
    # fold; r_T = sd(T~) / sd(T / T_ref) is scale free, so it is evaluated on
    # the Phase 0 scale (T / c) and sd_t / sd_t_resid are reported on it.
    m_identity_oof = mu_oof[s] @ greville  # (n,)
    t_resid = t_eff - m_identity_oof
    sd_resid_seeds[s] = float(np.std(t_resid))
    r_t_seeds[s] = sd_resid_seeds[s] / sd_t if sd_t > 0 else float("nan")
  mean_loss_identity = float(np.mean(identity_losses))

  # ---------------------------------------------------------------
  # 1-SE comparison with the identity candidate.
  #   d_s = L_hat_s(alpha_hat) - L_hat_{s,T},  d_bar = mean_s d_s,
  #   d_i = (1/S) sum_s (e_{i,s}(alpha_hat)^2 - (e^T_{i,s})^2),
  #   se = se_obs = cluster-robust se of d_bar_obs: TWO-WAY on the two
  #   adopted keys when they are known (fold_key_values), otherwise one-way
  #   on se_cluster_labels (fewer-cluster key) or iid.  The one-way se is
  #   always computed and reported next to the two-way one.
  #   crit = linear_test_k, or t_{df}^{-1}(Phi(k)) with df = G_min - 1 when
  #   alpha_selection.linear_test_t_critical is on; adopt iff d_bar < -crit se.
  # An alpha_hat on the linear-side grid edge adopts the
  # linear model regardless of the 1-SE result (statistics still reported).
  # ---------------------------------------------------------------
  d_s = loss_at_hat - identity_losses
  d_bar = float(np.mean(d_s))
  d_obs = obs_at_hat - identity_obs
  se_one_way = float(cluster_robust_se(d_obs, se_cluster_labels))
  se_two_way = float("nan")
  se_keys: Optional[Tuple[str, str]] = None
  if fold_key_arrays is not None and all(
      count_clusters(k) >= 2 for k in fold_key_arrays
  ):
    se_two_way = float(
        cluster_robust_se(d_obs, fold_key_arrays[0], fold_key_arrays[1])
    )
    se_keys = (
        (str(fold_key_names[0]), str(fold_key_names[1]))
        if fold_key_names is not None else ("key1", "key2")
    )
    se_method = "two-way"
    se = se_two_way
    se_clusters_min = min(count_clusters(k) for k in fold_key_arrays)
  elif count_clusters(se_cluster_labels) >= 2:
    se_method = "one-way"
    se = se_one_way
    se_clusters_min = count_clusters(se_cluster_labels)
  else:
    se_method = "iid"
    se = se_one_way
    se_clusters_min = int(n_obs)
  se_df = max(int(se_clusters_min) - 1, 0)
  critical_value = linear_test_critical_value(
      linear_test_k, se_df, linear_test_t_critical
  )
  one_se_pass = linear_test_decision(d_bar, se, critical_value)
  one_way_pass = linear_test_decision(d_bar, se_one_way, critical_value)
  decision_under_one_way_se = "saturation" if one_way_pass else "linear"
  one_way_key = se_cluster_key or "key"
  if se_method == "two-way":
    se_scope = (
        f"two-way {se_keys[0]} x {se_keys[1]}; one-way {one_way_key} ="
        f" {se_one_way:.4g}"
    )
  elif se_method == "one-way":
    se_scope = f"one-way {one_way_key}"
  else:
    se_scope = "iid"
  crit_ref = f"t with df = {se_df}" if linear_test_t_critical else "normal"
  se_desc = (
      f"se = {se:.4g} ({se_scope}), crit = {critical_value:.3g}"
      f" (k = {linear_test_k:g}, {crit_ref})"
  )
  if basis_sel.ill_conditioned:
    # The coefficients c(alpha) -- hence m_hat, the losses and the 1-SE
    # statistics -- cannot be trusted: adopt the linear model (parsimony),
    # keep reporting the statistics.  Checked before the grid edge because
    # the argmin itself is meaningless here.
    adopt = False
    decision_reason = "basis_ill_conditioned"
  elif grid_edge_hit == "upper":
    adopt = False
    decision_reason = "grid_edge_upper"
  else:
    adopt = one_se_pass
    decision_reason = "1se_pass" if one_se_pass else "1se_fail"
  decision = "saturation" if adopt else "linear"
  selected_family: Optional[str] = family if adopt else None
  if decision_reason == "basis_ill_conditioned":
    messages.append(
        "[Phase 0 / Stage B] decision = linear (decision_reason ="
        " basis_ill_conditioned; the untrusted 1-SE test would have said"
        f" {'saturation' if one_se_pass else 'linear'}: d_bar = {d_bar:.4g},"
        f" {se_desc}). selected_family = None."
    )
  elif decision_reason == "grid_edge_upper":
    messages.append(
        "[Phase 0 / Stage B] decision = linear (decision_reason ="
        " grid_edge_upper; the 1-SE test alone would have said"
        f" {'saturation' if one_se_pass else 'linear'}: d_bar = {d_bar:.4g},"
        f" {se_desc}). selected_family = None."
    )
  elif not adopt:
    _warn(
        messages,
        "[Phase 0 / Stage B] saturation did not improve on the identity"
        f" candidate by crit = {critical_value:.3g} SE (d_bar = {d_bar:.4g},"
        f" {se_desc}); adopting the linear model (selected_family = None,"
        " decision_reason = 1se_fail).",
    )
  if se_method == "two-way" and one_way_pass != one_se_pass:
    messages.append(
        "[Phase 0 / Stage B] The one-way se on the fewer-cluster key"
        f" ({one_way_key}: {se_one_way:.4g}) would have decided"
        f" '{decision_under_one_way_se}' instead of"
        f" '{'saturation' if one_se_pass else 'linear'}' (two-way se"
        f" {se_keys[0]} x {se_keys[1]}: {se_two_way:.4g}); the two-way se is"
        " the one used because it accounts for dependence along both keys."
    )

  # ---------------------------------------------------------------
  # Diag2: 1-SE flatness.  se(alpha) is the same
  # cluster-robust se (two-way when available; no critical value), paired
  # against alpha_hat:
  #   d_i(alpha) = (1/S) sum_s (e_{i,s}(alpha)^2 - e_{i,s}(alpha_hat)^2),
  #   A_1SE = {alpha : L_bar(alpha) - L_bar(alpha_hat) <= se(alpha)}.
  # ---------------------------------------------------------------
  if se_method == "two-way" and fold_key_arrays is not None:
    se_alpha = np.asarray(
        cluster_robust_se(
            obs_loss - obs_at_hat[:, None], fold_key_arrays[0],
            fold_key_arrays[1],
        ),
        dtype=float,
    )
  else:
    se_alpha = np.asarray(
        cluster_robust_se(obs_loss - obs_at_hat[:, None], se_cluster_labels),
        dtype=float,
    )
  gap = mean_loss - rloss_min
  one_se_set = gap <= np.where(np.isfinite(se_alpha), se_alpha, 0.0)
  diag2_ratio, diag2_fired, diag2_threshold = diag2_flatness(
      family, alpha_grid, one_se_set, flatness_ratio
  )
  mean_loss_ate = per_seed_loss_ate.mean(axis=0)

  # ---------------------------------------------------------------
  # Diag3 / Diag4 (reported values, no rule).
  # ---------------------------------------------------------------
  r_t = float(np.mean(r_t_seeds))

  # ---------------------------------------------------------------
  # Alpha used downstream.  When Diag1 or
  # Diag2 fires the computation continues; the fixed
  # `fallback_half_saturation_T` (a half-saturation point in units of T on
  # the Phase 0 scale; T_knee for log, alpha for power) replaces the
  # estimate if it is set, otherwise the 1-SE decision stands.
  # ---------------------------------------------------------------
  alpha_hat = alpha_est
  fallback_used = False
  if fallback_half_saturation is not None and (diag1_fired or diag2_fired):
    if family == "power":
      alpha_fixed = float(fallback_half_saturation)
    else:
      alpha_fixed = float(
          saturation.alpha_from_half_saturation(
              family, float(fallback_half_saturation)
          )
      )
    alpha_hat = float(saturation.clip_alpha(family, alpha_fixed))
    fallback_used = True
    selected_family = family
    messages.append(
        "[Phase 0 / Stage B] Diag1/Diag2 fired and"
        " alpha_selection.fallback_half_saturation_T ="
        f" {float(fallback_half_saturation):g} is set: using the fixed alpha"
        f" = {alpha_hat:g} downstream instead of the estimated {alpha_est:g}"
        f" (decision of the 1-SE test: {decision}, decision_reason:"
        f" {decision_reason}); selected_family = {family}."
    )

  # Diag2 message: WARNING only when selected_family != None, INFO otherwise.
  if diag2_fired:
    unit = saturation.grid_variable(family)
    if family == "power":
      rule = (
          f"{unit}^max - {unit}^min = {diag2_ratio:.3g} > {diag2_threshold:g}"
      )
    else:
      rule = (
          f"{unit}^max / {unit}^min = {diag2_ratio:.3g} >"
          f" flatness_ratio = {diag2_threshold:g}"
      )
    text = (
        f"[Phase 0 / Stage B] Diag2 1-SE flatness: {rule} over the 1-SE set"
        f" ({int(np.count_nonzero(one_se_set))} of {alpha_grid.size} grid"
        " points); the half-saturation point is not pinned down within one"
        " order of magnitude."
    )
    if selected_family is not None:
      _warn(messages, text + " Continuing.")
    else:
      messages.append(text + " (INFO: selected_family = None.)")

  # Diag5: kappa(alpha_hat) = f(T_ref; alpha_hat) for the
  # bounded families (share of the ceiling reached at T_ref); None otherwise.
  saturation_at_t_ref: Optional[float] = None
  if family in saturation.BOUNDED_FAMILIES and np.isfinite(alpha_hat):
    saturation_at_t_ref = float(
        saturation.apply_saturation(
            np.array([t_ref_value]), family, float(alpha_hat)
        )[0]
    )

  return AlphaSelectionResult(
      alpha_hat=float(alpha_hat),
      alpha_grid=alpha_grid,
      mean_loss=mean_loss,
      per_seed_loss=per_seed_loss,
      per_seed_alpha=per_seed_alpha,
      grid_alpha_hat=grid_alpha_hat,
      refined=refined,
      ml_model_y=best_y,
      ml_model_t=best_t,
      normalization=normalization,
      n_ml_fits=int(n_ml_fits),
      multioutput_native=bool(multioutput_native),
      messages=messages,
      family=family,
      selected_family=selected_family,
      decision=decision,
      alpha_hat_estimated=float(alpha_est),
      refine_method="brent" if refined else "none",
      grid=grid,
      grid_extensions=int(grid.extensions),
      grid_edge_unresolved=grid_edge_unresolved,
      per_seed_loss_ate=per_seed_loss_ate,
      mean_loss_ate=mean_loss_ate,
      per_seed_loss_identity=identity_losses,
      mean_loss_identity=mean_loss_identity,
      per_seed_loss_at_alpha_hat=np.asarray(loss_at_hat, dtype=float),
      per_seed_min_loss=per_seed_min_loss,
      d_s=d_s,
      d_bar=d_bar,
      d_obs=np.asarray(d_obs, dtype=float),
      se=se,
      linear_test_k=linear_test_k,
      se_cluster_key=se_cluster_key,
      se_method=se_method,
      se_cluster_keys=se_keys,
      se_one_way=float(se_one_way),
      se_two_way=float(se_two_way),
      se_df=int(se_df),
      linear_test_t_critical=bool(linear_test_t_critical),
      critical_value=float(critical_value),
      decision_under_one_way_se=decision_under_one_way_se,
      se_alpha=se_alpha,
      one_se_set=one_se_set,
      flatness_ratio=float(flatness_ratio),
      diag2_ratio=float(diag2_ratio),
      diag2_fired=bool(diag2_fired),
      n_basis=int(n_basis),
      basis_approx_error=basis_approx_error,
      basis_approx_error_alpha=evals["approx"],
      basis_mode=basis_sel.mode,
      basis_satisfied=bool(basis_sel.satisfied),
      basis_tol=float(basis_tol),
      diag1_fired=diag1_fired,
      basis_cond=float(treatment_basis.basis_cond),
      basis_ill_conditioned=bool(basis_sel.ill_conditioned),
      r_t=r_t,
      sd_t=sd_t,
      sd_t_resid=float(np.mean(sd_resid_seeds)),
      share_t_zero=float(np.mean(t_eff <= 0.0)),
      q50_t_pos=float(np.quantile(t_pos, 0.5)) if t_pos.size else float("nan"),
      q95_t_pos=float(np.quantile(t_pos, 0.95)) if t_pos.size else float("nan"),
      t_max=float(treatment_basis.t_max),
      t_max_quantile=float(saturation.BASIS_T_MAX_QUANTILE),
      t_max_observed=float(np.max(t_eff)) if t_eff.size else float("nan"),
      n_above_t_max=int(treatment_basis.n_above_t_max),
      fbar_quantiles_grid=np.asarray(evals["phi"], dtype=float),
      fbar_quantiles=np.asarray(phi_at_hat, dtype=float),
      linear_ref_quantiles=np.asarray(linear_ref_quantiles, dtype=float),
      saturation_at_t_ref=saturation_at_t_ref,
      fallback_half_saturation_used=fallback_used,
      learner_selection=bool(learner_cfg["learner_selection"]),
      learners=learner_desc,
      small_sample_threshold=int(learner_cfg["small_sample_threshold"]),
      seed_partitions=seed_partitions,
      cluster_labels=np.asarray(cluster_labels),
      t_values=t_eff,
      n_sel=int(n_obs),
      k_folds=int(k_folds),
      fold_unit="cell" if fold_key_values is not None else "cluster",
      fold_key_names=(
          tuple(fold_key_names) if fold_key_names is not None else None
      ),
      fold_key_values=fold_key_arrays,
      t_ref=float(t_ref_value),
      t_ref_quantile=float(t_ref_quantile),
      t_ref_raw=t_ref_raw,
      grid_edge_hit=grid_edge_hit,
      decision_reason=decision_reason,
  )


def _v3_learner_selection_fits(
    x_values: np.ndarray,
    y_values: np.ndarray,
    b_targets: np.ndarray,
    cluster_labels: np.ndarray,
    seed_partitions: List[np.ndarray],
    ml_models_y: Sequence[str],
    ml_models_t: Sequence[str],
    time_budget: int,
    selection_cfg: Dict[str, Any],
    n_total: Optional[int],
    n_jobs: int,
    progress_dir: Optional[str],
    random_state: int,
    rng: np.random.Generator,
    messages: List[str],
) -> Tuple[np.ndarray, np.ndarray, int, str, str, bool, Dict[str, Any]]:
  """The legacy learner-selection path (``learners.learner_selection``).

  One AutoML search per candidate learner (independently per regression, by
  out-of-fold RMSE; no Cartesian search), then refits - not searches - of the
  two winners, one per (seed, fold).  Cost ``N_y + N_t + 2 S K`` fits.

  Args:
    x_values: Nuisance design matrix.
    y_values: Outcome.
    b_targets: ``(n, J)`` treatment-basis targets.
    cluster_labels: Cluster labels (cluster-level holdout).
    seed_partitions: Per-seed fold ids.
    ml_models_y: Candidate learners for ``l_hat``.
    ml_models_t: Candidate learners for ``mu_hat``.
    time_budget: Pipeline-wide FLAML budget (seconds).
    selection_cfg: The ``alpha_selection`` block.
    n_total: Full-sample rows (time-budget scaling).
    n_jobs: joblib parallelism.
    progress_dir: Progress marker directory.
    random_state: Base seed.
    rng: Random generator (holdout split).
    messages: Message sink.

  Returns:
    ``(l_oof, mu_oof, n_ml_fits, best_y, best_t, native, description)`` with
    the out-of-fold predictions ``l_oof`` of shape ``(S, n)`` and ``mu_oof``
    of shape ``(S, n, J)``.
  """
  n_obs = len(y_values)
  n_basis = b_targets.shape[1]
  n_seeds = len(seed_partitions)
  k_folds = int(selection_cfg.get("k_folds", 5))
  if selection_cfg.get("time_budget", None) is not None:
    stage_b_budget = int(selection_cfg["time_budget"])
  else:
    stage_b_budget = scale_time_budget(
        time_budget, n_obs, int(n_total) if n_total else n_obs
    )
    if stage_b_budget != int(time_budget):
      messages.append(
          f"Stage B: FLAML time_budget scaled to the size of D_sel"
          f" ({n_obs}/{int(n_total) if n_total else n_obs} rows):"
          f" {int(time_budget)}s -> {stage_b_budget}s per fit. Set"
          " alpha_selection.time_budget to override."
      )

  holdout_folds = _cluster_kfold(cluster_labels, k_folds, rng)
  val_mask = holdout_folds == 0
  trn_mask = ~val_mask
  if not np.any(val_mask) or not np.any(trn_mask):
    val_mask = np.zeros(n_obs, dtype=bool)
    val_mask[: max(1, n_obs // k_folds)] = True
    trn_mask = ~val_mask

  sel_y_out = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
      joblib.delayed(_tune_scalar)(
          name, x_values[trn_mask], y_values[trn_mask], x_values,
          stage_b_budget, random_state + 7,
          os.path.join(progress_dir, f"sel_y_{name}.done")
          if progress_dir else None,
      )
      for name in ml_models_y
  )
  sel_t_out = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
      joblib.delayed(_tune_multioutput)(
          name, x_values[trn_mask], b_targets[trn_mask], x_values,
          stage_b_budget, random_state + 7,
          os.path.join(progress_dir, f"sel_t_{name}.done")
          if progress_dir else None,
      )
      for name in ml_models_t
  )

  y_scores: Dict[str, float] = {}
  y_templates: Dict[str, Any] = {}
  for name, (preds, template) in zip(ml_models_y, sel_y_out):
    y_scores[name] = _rmse(y_values[val_mask], preds[val_mask])
    y_templates[name] = template
  t_scores: Dict[str, float] = {}
  t_templates: Dict[str, Any] = {}
  t_native: Dict[str, bool] = {}
  for name, (preds, template, native) in zip(ml_models_t, sel_t_out):
    t_scores[name] = float(
        np.mean([
            _rmse(b_targets[val_mask, j], preds[val_mask, j])
            for j in range(n_basis)
        ])
    )
    t_templates[name] = template
    t_native[name] = bool(native)

  best_y = min(y_scores, key=y_scores.get) if y_scores else None
  best_t = min(t_scores, key=t_scores.get) if t_scores else None
  if best_y is None or best_t is None:
    raise ValueError(
        "Phase 0 / Stage B needs at least one candidate learner in BOTH"
        " ml_models_y and ml_models_t; got"
        f" ml_models_y={list(ml_models_y)}, ml_models_t={list(ml_models_t)}."
    )
  template_y = y_templates[best_y]
  template_t = t_templates[best_t]
  messages.append(
      f"Stage B learner selection: l_hat -> '{best_y}' (out-of-fold RMSE"
      f" {y_scores[best_y]:.5f}), mu_hat -> '{best_t}'"
      f" (mean out-of-fold RMSE over J={n_basis} basis outputs"
      f" {t_scores[best_t]:.5f})."
  )
  if not t_native[best_t]:
    messages.append(
        f"Stage B: learner '{best_t}' has no native multi-output support, so"
        " the treatment basis is fitted through a per-output wrapper."
    )

  fit_jobs = _cross_fit_jobs(seed_partitions, k_folds)
  fit_out = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
      joblib.delayed(_refit_predict)(
          template_y if kind == "y" else template_t,
          x_values[train],
          (y_values if kind == "y" else b_targets)[train],
          x_values[held_out],
          os.path.join(progress_dir, f"seed_{s}_{k}_{kind}.done")
          if progress_dir else None,
      )
      for s, k, kind, train, held_out in fit_jobs
  )
  l_oof = np.zeros((n_seeds, n_obs), dtype=float)
  mu_oof = np.zeros((n_seeds, n_obs, n_basis), dtype=float)
  _scatter_oof_predictions(fit_jobs, fit_out, l_oof, mu_oof)
  n_ml_fits = len(ml_models_y) + len(ml_models_t) + len(fit_jobs)
  description = {
      "learner_selection": True,
      "y_model": {"class": f"flaml:{best_y}",
                  "params": _safe_params(template_y)},
      "t_model": {"class": f"flaml:{best_t}",
                  "params": _safe_params(template_t)},
      "small_sample_switch": False,
  }
  return (l_oof, mu_oof, n_ml_fits, best_y, best_t,
          bool(t_native[best_t]), description)


def _safe_params(estimator: Any) -> Dict[str, Any]:
  """JSON-friendly ``get_params`` of a tuned estimator (best effort)."""
  try:
    params = estimator.get_params(deep=False)
  except Exception:  # pylint: disable=broad-except
    return {}
  return {
      k: v for k, v in params.items()
      if isinstance(v, (int, float, str, bool)) or v is None
  }


def count_stage_b_tasks(
    n_models_y: int,
    n_models_t: int,
    n_seeds: int,
    k_folds: int,
    learner_selection: bool = True,
) -> int:
  """Returns the number of ML fits of Stage B (used by the progress display).

  Implements the compute accounting:

      selection : N_y + N_t          (only with ``learner_selection: true``;
                                      0 with the fixed
                                      default learners)
      seed loop : 2 * S * K          (the two learners, refit once per
                                      (seed, fold) - cross-fitting needs one
                                      model per fold because every observation
                                      needs an out-of-fold prediction)

  Neither the alpha grid size, the grid extensions, the identity candidate nor
  the basis dimension J multiplies ML training.

  Args:
    n_models_y: Number of candidate learners for ``l_hat``.
    n_models_t: Number of candidate learners for ``mu_hat``.
    n_seeds: ``S``.
    k_folds: ``K``.
    learner_selection: Whether the legacy learner selection runs.  Defaults to
      ``True`` so that legacy call sites keep their accounting.

  Returns:
    The total number of ML fits.
  """
  selection = (n_models_y + n_models_t) if learner_selection else 0
  seed_loop = 2 * n_seeds * k_folds
  return int(selection + seed_loop)


def describe_ml_training_count(
    n_treatments: int,
    n_seeds: int,
    k_folds: int,
    n_models_y: int = 0,
    n_models_t: int = 0,
    learner_selection: bool = False,
) -> Tuple[str, int]:
  """The ML-training count line of the Phase 0 progress display.

  Renders ``Number of ML trainings = <treatments> x 2 nuisance models x <S> x
  <K>`` (``2S`` expanded to the fold level: one ML training =
  one ``fit`` of a nuisance learner per (seed, fold)), followed by the total.
  With ``learners.learner_selection: true`` the ``N_y + N_t`` selection
  searches per treatment are appended as a separate term.

  Args:
    n_treatments: Number of treatment columns.
    n_seeds: ``S``.
    k_folds: ``K``.
    n_models_y: ``N_y`` (learner selection only).
    n_models_t: ``N_t`` (learner selection only).
    learner_selection: Whether the legacy learner selection runs.

  Returns:
    ``(line, total)``: the display line and the total number of trainings
    (identical to ``n_treatments * count_stage_b_tasks(...)``).
  """
  per_treatment = count_stage_b_tasks(
      n_models_y, n_models_t, n_seeds, k_folds,
      learner_selection=learner_selection,
  )
  total = int(n_treatments) * int(per_treatment)
  line = (
      f"Number of ML trainings = {int(n_treatments)} x 2 nuisance models x"
      f" {int(n_seeds)} x {int(k_folds)}"
  )
  if learner_selection:
    line += (
        f" + {int(n_treatments)} x ({int(n_models_y)} + {int(n_models_t)})"
        " learner-selection searches"
    )
  return f"{line} = {total}", total
