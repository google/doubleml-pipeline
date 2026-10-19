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

"""Saturation function families for saturation-aware DML.

This module is an *additive* extension of the linear DML pipeline.  Its
mathematical part (families, normalization, T-basis, alpha grid, KKT
allocation) carries no I/O and no pipeline state, so it can be unit tested in
isolation and reused by Phase 0 (alpha selection) and Phase 4 (saturation
aware budget optimization); the last section holds the fixed default nuisance
learners of Phase 0 / Stage B (the only part that uses scikit-learn /
XGBoost).

Contents:

* The supported saturation families ``exp`` / ``log`` / ``power`` and their
  derivatives.
* The manual ``grid`` config block: construction of the compact alpha grid
  on a log or linear scale.
* The B-spline basis expansion of ``f(.;alpha)`` used to synthesise
  ``m_hat(X;alpha)`` for every alpha from a single multi-output regression.
* The closed-form (water-filling type) budget allocation for ``tanh`` /
  ``exp`` / ``log`` / ``power`` with per-unit caps and the ROI-threshold cap.

Half-saturation grid and T-basis:

* The ``tanh`` family ``f(T) = tanh(lambda T / 2)`` (the default family when
  a ``saturation`` block is present) and its semi-closed-form KKT allocation.
* The dimensionless alpha grid laid out in the half-saturation point
  ``T_1/2`` (:func:`build_half_saturation_grid`) and its edge auto-extension
  (:func:`extend_alpha_grid`).
* :class:`TreatmentBasis` with explicit quantile knots on ``T > 0`` (boundary
  0 included), Greville abscissae (exact linear precision, used by the
  identity candidate of the alpha grid) and the automatic choice of ``J``
  (:func:`select_basis_dim`).

Treatment normalization (``fbar``, ``T_ref``, ``kappa``):

* The NORMALIZED saturation function
  ``fbar(T; alpha) = f(T; alpha) / f(T_ref; alpha)`` with the reference point
  ``T_ref = q95(T | T > 0)`` (:func:`reference_point`,
  :func:`normalized_saturation`).  Every candidate curve passes through
  ``(T_ref, 1)``; ``fbar`` is what Stage B, Phases 1-3 and Phase 4 use.
  :func:`apply_saturation` keeps returning the raw ``f`` (the building block).
* The basis coefficients ``c(alpha) = P fbar_alpha`` with the least-squares
  projector ``P = B^+`` (pseudo-inverse through the SVD of ``B``, never
  through ``B'B``) on a fixed projection grid over ``[0, T_max]`` -- the
  union of 500 equally spaced points, 201 empirical quantiles of ``T > 0``
  and the Greville abscissae, so that every basis function is seen by the
  grid and ``B`` has full column rank -- precomputed once per
  :class:`TreatmentBasis`.  The condition number ``cond(B)`` is kept and
  compared with :data:`MAX_BASIS_CONDITION` (Diag1).  The basis domain is
  ``[0, T_max]`` with ``T_max`` the :data:`BASIS_T_MAX_QUANTILE` quantile of
  ``T > 0`` (computed once on the full data, like ``T_ref``); observations
  above ``T_max`` are clipped when the basis is evaluated.
* The grid variable of the ``log`` family is the marginal-effect-halving
  point ``T_knee = alpha`` (:data:`GRID_VARIABLE`).
* The KKT inversions carry the constant ``kappa = f(T_ref; alpha)``
  (``tanh(lambda T_ref / 2)`` for ``tanh``); the allocation itself is
  unchanged by the normalization.
* The ROI threshold ``r`` is a per-unit cap ``T_t^r``
  (``ROI_t(T) = theta_t fbar(T) / T >= r  <=>  T <= T_t^r``,
  :func:`roi_cap`), the KKT inversion ``T_t*(mu) = clip((fbar')^-1(mu /
  theta_t), 0, cap_t)`` takes per-unit caps (:func:`kkt_allocation`), and
  the budget problem is solved by ONE bisection on ``mu``
  (:func:`solve_budget_allocation`, which also returns ``mu*``).  The former
  greedy "exclude below the threshold -> re-solve -> iterate" loop no longer
  exists.

Stage B nuisance learners (``alpha_selection.learners``):

* The purpose of Stage B is a *stable ranking* of alpha, so no AutoML is run
  by default.  The defaults are: Y-model ``l_hat`` = ``xgboost.XGBRegressor``
  with early stopping (patience 30) on a cluster-level 10% internal
  validation split inside the training folds; T-model ``mu_hat_j`` = ONE
  ``xgboost.XGBRegressor(multi_strategy="multi_output_tree")`` with ``J``
  outputs (XGBoost >= 2.0; wrapping in ``MultiOutputRegressor`` is not
  allowed); small-sample switch: when ``n_sel < small_sample_threshold``
  (default 2000) both models become ``RidgeCV`` on one-hot(item, geo) +
  annual Fourier terms (order 3) + numeric covariates and their cubic
  B-splines (5 df) (:func:`resolve_learner_config`,
  :func:`resolve_stage_b_learners`, :func:`build_small_sample_features`).
* XGBoost is imported lazily.  When XGBoost >= 2.0 is not importable (e.g.
  an environment that only ships xgboost 1.5.1), a warning is emitted and
  scikit-learn stand-ins with matching hyperparameters are used:
  ``HistGradientBoostingRegressor`` for Y (same cluster-level early stopping)
  and a single native multi-output ``RandomForestRegressor`` for T (still ONE
  model with J outputs).  No new heavy dependency is introduced.
"""

import copy
import dataclasses
import importlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate as scipy_interpolate
from sklearn import compose
from sklearn import ensemble
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing


# `hill` is intentionally not supported.  `tanh` is the default family.
SUPPORTED_FAMILIES = ("tanh", "exp", "log", "power")

# Default family when the `saturation` block exists but `family` is
# unspecified (the block being omitted still means `None`).
DEFAULT_FAMILY = "tanh"

# Families whose alpha grid is laid out in the half-saturation point.
HALF_SATURATION_FAMILIES = ("tanh", "exp", "log")

# `power` is dimensionless; its grid is laid out directly in alpha.
POWER_GRID_BOUNDS = (0.1, 0.95)

# Families with a ceiling (sup f = 1).  Diag5 reports the saturation level
# at T_ref, kappa(alpha_hat) = f(T_ref; alpha_hat), for these families only
# ("the share of the ceiling reached at the q95 spend"); log / power have no
# ceiling and report null.
BOUNDED_FAMILIES = ("tanh", "exp")

# `saturation.t_ref_quantile` default -- T_ref is this quantile of T > 0,
# computed once on the full data before the split.
DEFAULT_T_REF_QUANTILE = 0.95

# Number of equally spaced points over [0, T_max] in the projection grid on
# which the basis coefficients c(alpha) = P fbar_alpha are defined.  The
# projection grid is the union of these points, BASIS_QUANTILE_GRID empirical
# quantiles of T > 0 and the Greville abscissae of the basis.
BASIS_PROJECTION_GRID = 500

# Number of empirical quantiles of T > 0 (equally spaced probabilities from 0
# to 1) added to the projection grid, so that the least-squares fit of
# fbar(.; alpha) weights the region where the observations actually lie.
BASIS_QUANTILE_GRID = 201

# Quantile of T > 0 that defines the right end T_max of the basis domain
# [0, T_max].  Like T_ref it is computed ONCE on the full data before the
# split (`basis_domain_max`).  Observations above T_max (at most 0.1 % of the
# positive rows) are clipped to T_max when the basis is evaluated; using the
# sample maximum instead lets a single extreme day stretch the last spline
# piece over a huge range and degrade the fit on the ordinary rows.
BASIS_T_MAX_QUANTILE = 0.999

# Condition number of the projection-grid basis matrix B (largest / smallest
# singular value) at or above which the least-squares coefficients c(alpha)
# are not trusted: Diag1 fires and Stage B adopts the linear model
# (`decision_reason = "basis_ill_conditioned"`).  A healthy basis has cond(B)
# of order 10; coefficients start to blow up (cancellation) around 1e4, so
# the threshold is lenient and only guards against pathological knot
# placements (e.g. heavily tied treatment values).
MAX_BASIS_CONDITION = 1e8

# Relative cut-off of the singular values in the pseudo-inverse of B.
_BASIS_PINV_RCOND = 1e-10

# The variable each family's grid is laid out in.  `log` has no
# half-saturation point (f(alpha; alpha) = ln 2 != 1/2), so its grid variable
# is the marginal-effect-halving point T_knee = alpha.  `half_saturation_T`
# outputs hold T_knee for the log family.
GRID_VARIABLE = {
    "tanh": "T_1/2",
    "exp": "T_1/2",
    "log": "T_knee",
    "power": "alpha",
}

# Numerical guard used when alpha or the treatment approaches a boundary.
_EPS = 1e-12


def validate_family(family: Optional[str]) -> Optional[str]:
  """Normalizes and validates the saturation family name.

  Args:
    family: Raw family value coming from the config. ``None``, ``"none"`` and
      ``""`` all mean "no saturation" (existing linear pipeline).

  Returns:
    The normalized family name, or ``None`` when saturation is disabled.

  Raises:
    ValueError: If the family is not one of the supported families.
  """
  if family is None:
    return None
  family_str = str(family).strip().lower()
  if family_str in ("", "none", "null", "linear"):
    return None
  if family_str not in SUPPORTED_FAMILIES:
    raise ValueError(
        f"Unsupported saturation family '{family}'. Supported families are"
        f" {list(SUPPORTED_FAMILIES)} or None (existing linear pipeline)."
        " Note: 'hill' is intentionally not supported."
    )
  return family_str


def default_alpha_bounds(family: str) -> Tuple[float, float]:
  """Returns the default `[alpha_min, alpha_max]` support for a family.

  The grid must stay bounded away from the boundaries of the parameter
  space.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.

  Returns:
    A ``(alpha_min, alpha_max)`` tuple.
  """
  if family == "power":
    # alpha in (0, 1): keep an epsilon away from 1 to preserve concavity.
    return (0.05, 0.95)
  return (0.05, 5.0)


def clip_alpha(
    family: str, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
  """Clips alpha into the admissible open interval of the family.

  A scalar in gives a ``float`` out, so every pre-existing scalar call site is
  bit-for-bit unchanged.  An array in gives an array out, which is what the
  treatment normalization bridge needs for the row-varying ``alpha_i`` of
  ``population_median_normalize``.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: Saturation parameter(s) to clip.

  Returns:
    The clipped alpha: a ``float`` if ``alpha`` is a scalar, otherwise an array
    of the same shape as ``alpha``.
  """
  arr = np.asarray(alpha, dtype=float)
  if family == "power":
    out = np.clip(arr, 1e-3, 1.0 - 1e-3)
  else:
    out = np.maximum(arr, 1e-6)
  return float(out) if out.ndim == 0 else out


def _broadcast_alpha(
    family: str, alpha: Union[float, np.ndarray], like: np.ndarray
) -> np.ndarray:
  """Clips ``alpha`` and broadcasts it to the shape of ``like``."""
  arr = np.asarray(clip_alpha(family, alpha), dtype=float)
  return np.broadcast_to(arr, np.shape(like)).astype(float, copy=False)


def rescale_alpha_to_raw(
    family: str, alpha: float, divisor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Maps alpha from NORMALIZED treatment units back to RAW treatment units.

  Implements the treatment normalization bridge.  Phase 0 selects ``alpha``
  against the normalized treatment ``T_tilde = T / c``, where ``c`` is the
  scaler divisor: a global constant for ``median_normalize``, a row-varying
  ``c_i = p_i * m`` for ``population_median_normalize``.  Because median-type
  normalization forces ``median(T_tilde) = 1`` by construction, the single
  fixed default grid ``[0.05, 5.0]`` is meaningful for every dataset.

  Phase 4, in contrast, budgets in RAW treatment units.  Writing the raw-scale
  response as ``F_i(T) = f(T / c_i; alpha)`` and re-expressing it inside the
  same family gives, exactly and for all three families,

      F_i(T) = gain_i * f(T; alpha_i)

  with

      exp   : alpha_i = alpha / c_i ,  gain_i = 1
      log   : alpha_i = alpha * c_i ,  gain_i = 1
      power : alpha_i = alpha       ,  gain_i = c_i ** (-alpha)

  ``power`` is scale-equivariant in its *argument* rather than in alpha, so its
  rescaling collapses entirely into a multiplicative gain, which is absorbed by
  ``theta``.

  Only ONE parameter is ever estimated: the ``alpha_i`` are derived from the
  single normalized-scale ``alpha``, never searched over.

  With the normalized ``fbar`` the bridge needs no gain at all.  For every
  family, exactly,

      fbar(T / c_i; alpha, T_ref) = fbar(T; alpha_i, c_i * T_ref)

  (the same ``alpha_i`` as above; the ``power`` gain ``c_i ** (-alpha)``
  cancels between numerator and denominator), so Phase 4 evaluates ``fbar`` in
  raw units with ``alpha_i`` and the raw-scale reference point
  ``T_ref_raw_i = c_i * T_ref`` (:func:`raw_scale_reference`) and leaves
  ``theta`` untouched.  The ``gain`` returned here is only needed by callers
  that still work with the un-normalized ``f``.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The scalar alpha estimated on the normalized scale.
    divisor: The scaler divisor ``c`` (scalar, or one value per row).

  Returns:
    An ``(alpha_raw, gain)`` pair, both shaped like ``divisor``.
  """
  div = np.asarray(divisor, dtype=float)
  # A non-positive or non-finite divisor means "no normalization for this row".
  div = np.where(np.isfinite(div) & (div > 0.0), div, 1.0)
  alpha_value = float(clip_alpha(family, float(alpha)))
  ones = np.ones_like(div)
  # tanh(lambda (T/c) / 2) = tanh((lambda/c) T / 2), i.e. the same
  # rescaling as `exp` (lambda_i = lambda / c_i, gain_i = 1).
  if family in ("exp", "tanh"):
    return np.asarray(clip_alpha(family, alpha_value / div), dtype=float), ones
  if family == "log":
    return np.asarray(clip_alpha(family, alpha_value * div), dtype=float), ones
  if family == "power":
    return np.full_like(div, alpha_value), np.power(div, -alpha_value)
  raise ValueError(f"Unsupported saturation family '{family}'.")


def raw_scale_reference(
    t_ref: float, divisor: Union[float, np.ndarray]
) -> np.ndarray:
  """Maps ``T_ref`` from the normalized scale ``T / c`` to RAW units.

  Treatment normalization bridge: Phase 0 computes ``T_ref`` on the scale it
  works on (``T / c``).  Since
  ``fbar(T / c; alpha, T_ref) = fbar(T; alpha_raw, c * T_ref)`` for every
  family (see :func:`rescale_alpha_to_raw`), the raw-scale reference point is
  simply ``c * T_ref`` (row-varying when ``c`` is).

  Args:
    t_ref: ``T_ref`` on the normalized scale.
    divisor: The scaler divisor ``c`` (scalar, or one value per row).

  Returns:
    ``T_ref_raw``, shaped like ``divisor``.
  """
  div = np.asarray(divisor, dtype=float)
  div = np.where(np.isfinite(div) & (div > 0.0), div, 1.0)
  return float(t_ref) * div


def apply_saturation(
    t_values: np.ndarray, family: str, alpha: Union[float, np.ndarray]
) -> np.ndarray:
  """Evaluates the RAW (un-normalized) ``f(T; alpha)``.

  The supported families:
    * ``tanh``  : ``tanh(alpha * T / 2)``  (alpha = lambda)
    * ``exp``   : ``1 - exp(-alpha * T)``
    * ``log``   : ``log(1 + T / alpha)``
    * ``power`` : ``T ** alpha``

  Estimation, comparison and optimization all use the normalized
  :func:`normalized_saturation` (``fbar = f / f(T_ref; alpha)``); this
  function is the building block (numerator and denominator of ``fbar``) and
  the only place the family formulas live.

  Args:
    t_values: Treatment values (negative values are clipped to 0, since all
      families are defined on ``T >= 0``).
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter: a scalar, or an array broadcastable to
      ``t_values`` (treatment normalization bridge).

  Returns:
    ``f(T; alpha)`` broadcast over ``t_values`` and ``alpha``.
  """
  t_arr = np.maximum(np.asarray(t_values, dtype=float), 0.0)
  alpha = clip_alpha(family, alpha)
  if family == "tanh":
    return np.tanh(0.5 * alpha * t_arr)
  if family == "exp":
    return 1.0 - np.exp(-alpha * t_arr)
  if family == "log":
    return np.log1p(t_arr / alpha)
  if family == "power":
    return np.power(t_arr, alpha)
  raise ValueError(f"Unsupported saturation family '{family}'.")


def reference_point(
    t_values: np.ndarray, quantile: float = DEFAULT_T_REF_QUANTILE
) -> float:
  """Returns the normalization reference point ``T_ref = q(T | T > 0)``.

  ``T_ref := q95(T | T > 0)`` (the ``saturation.t_ref_quantile`` setting),
  computed ONCE on the full data before the top-level split -- no
  leakage, since ``Y`` is not used.  The quantile is taken over the strictly
  positive treatment values so that the mass at zero of sporadic spend does
  not pull the reference point towards 0.

  Args:
    t_values: Treatment values on the scale Phase 0 works on (``T / c``).
    quantile: The quantile of ``T > 0`` (default 0.95).

  Returns:
    ``T_ref`` (strictly positive).

  Raises:
    ValueError: If ``quantile`` is outside ``(0, 1]`` or no strictly positive
      treatment value exists.
  """
  q = float(quantile)
  if not (0.0 < q <= 1.0):
    raise ValueError(
        f"saturation.t_ref_quantile must lie in (0, 1] (got {quantile})."
    )
  t_arr = np.asarray(t_values, dtype=float).reshape(-1)
  t_pos = t_arr[np.isfinite(t_arr) & (t_arr > 0.0)]
  if t_pos.size == 0:
    raise ValueError(
        "The normalization reference point T_ref = q95(T | T > 0) needs at"
        " least one strictly positive treatment value."
    )
  return float(np.quantile(t_pos, q))


def basis_domain_max(
    t_values: np.ndarray, quantile: float = BASIS_T_MAX_QUANTILE
) -> float:
  """Returns the right end ``T_max`` of the T-basis domain ``[0, T_max]``.

  ``T_max := q(T | T > 0)`` at :data:`BASIS_T_MAX_QUANTILE` (0.999), the
  counterpart of :func:`reference_point` for the domain: computed ONCE on the
  full data before the top-level split and passed to Stage B by the caller.
  Observations above ``T_max`` are clipped to ``T_max`` when the basis is
  evaluated (:meth:`TreatmentBasis.transform`), so a single extreme day no
  longer stretches the last spline piece over a huge range.  When no
  strictly positive value exists the (finite) maximum is returned.

  Args:
    t_values: Treatment values on the scale Phase 0 works on (``T / c``).
    quantile: The quantile of ``T > 0`` (default 0.999).

  Returns:
    ``T_max`` (``> 0`` whenever a positive value exists).

  Raises:
    ValueError: If ``quantile`` is outside ``(0, 1]`` or no finite value
      exists.
  """
  q = float(quantile)
  if not (0.0 < q <= 1.0):
    raise ValueError(f"The T_max quantile must lie in (0, 1] (got {quantile}).")
  t_arr = np.asarray(t_values, dtype=float).reshape(-1)
  t_arr = t_arr[np.isfinite(t_arr)]
  if t_arr.size == 0:
    raise ValueError("T_max needs at least one finite treatment value.")
  t_pos = t_arr[t_arr > 0.0]
  if t_pos.size == 0:
    return float(np.max(t_arr))
  return float(np.quantile(t_pos, q))


def normalized_saturation(
    t_values: np.ndarray,
    family: str,
    alpha: Union[float, np.ndarray],
    t_ref: Union[float, np.ndarray],
) -> np.ndarray:
  """Evaluates the normalized saturation ``fbar = f / f(T_ref; alpha)``.

  The normalization is always on and is applied uniformly to every alpha of
  the Stage B grid, to the identity candidate
  (``fbar_T = T / T_ref``), to the intercept curve, to the Phase 1-3
  treatment variable and to the Phase 4 objective.  Properties: for every
  family ``fbar(0; alpha) = 0`` and ``fbar(T_ref; alpha) = 1`` -- all
  candidate curves pass through ``(T_ref, 1)`` and differ only in shape; in
  the slow-saturation limit ``fbar -> T / T_ref``; ``fbar`` is dimensionless
  and invariant to unit changes of ``T``.  The unit of ``theta`` becomes "the
  increment of ``Y`` when the treatment moves from ``0`` to ``T_ref``".

  Args:
    t_values: Treatment values (negative values are clipped to 0).
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter: a scalar, or an array broadcastable to
      ``t_values`` (treatment normalization bridge).
    t_ref: The reference point ``T_ref > 0`` (scalar, or an array
      broadcastable to ``t_values`` -- the raw-scale ``c_i * T_ref`` of
      :func:`raw_scale_reference` is row-varying under
      ``population_median_normalize``).

  Returns:
    ``fbar(T; alpha)`` broadcast over ``t_values``, ``alpha`` and ``t_ref``.

  Raises:
    ValueError: If any ``t_ref`` is not finite and strictly positive.
  """
  t_ref_arr = np.asarray(t_ref, dtype=float)
  if np.any(~np.isfinite(t_ref_arr)) or np.any(t_ref_arr <= 0.0):
    raise ValueError(
        f"T_ref must be finite and strictly positive (got {t_ref}); it is"
        " defined as q95(T | T > 0)."
    )
  denominator = apply_saturation(t_ref_arr, family, alpha)
  return apply_saturation(t_values, family, alpha) / denominator


def grid_variable(family: str) -> str:
  """Returns the name of the grid variable of ``family``.

  ``T_1/2`` for ``tanh`` / ``exp``, ``T_knee`` for ``log`` and ``alpha`` for
  ``power``.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.

  Returns:
    The grid-variable label.
  """
  try:
    return GRID_VARIABLE[family]
  except KeyError as err:
    raise ValueError(f"Unsupported saturation family '{family}'.") from err


def saturation_derivative(
    t_values: np.ndarray, family: str, alpha: Union[float, np.ndarray]
) -> np.ndarray:
  """Evaluates ``f_T(T; alpha)`` (the marginal effect multiplier).

  Args:
    t_values: Treatment values.
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter: a scalar, or an array broadcastable to
      ``t_values``.

  Returns:
    ``d f(T; alpha) / dT`` broadcast over ``t_values`` and ``alpha``.
  """
  t_arr = np.maximum(np.asarray(t_values, dtype=float), 0.0)
  alpha = clip_alpha(family, alpha)
  if family == "tanh":
    # f'(T) = (lambda / 2) * (1 - f(T)^2).
    f_val = np.tanh(0.5 * alpha * t_arr)
    return 0.5 * alpha * (1.0 - f_val * f_val)
  if family == "exp":
    return alpha * np.exp(-alpha * t_arr)
  if family == "log":
    return 1.0 / (alpha + t_arr)
  if family == "power":
    return alpha * np.power(np.maximum(t_arr, _EPS), alpha - 1.0)
  raise ValueError(f"Unsupported saturation family '{family}'.")


def half_saturation_point(
    family: str, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
  """Returns the grid variable of ``f(.; alpha)``.

  Per family:

  * ``tanh``: half-saturation point ``f(T_1/2) = 1/2`` ->
    ``T_1/2 = ln 3 / lambda``
  * ``exp`` : half-saturation point ``f(T_1/2) = 1/2`` ->
    ``T_1/2 = ln 2 / alpha``
  * ``log`` : NOT a half-saturation point (``f(alpha; alpha) = ln 2 !=
    1/2``).  The grid variable is the marginal-effect-halving point
    ``T_knee = alpha`` (``f'(alpha) = f'(0) / 2``), and this is what the
    ``half_saturation_T`` output key / column holds for the log family.
  * ``power``: scale free  ->  ``nan``

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter(s).

  Returns:
    ``T_1/2`` (``T_knee`` for ``log``) with the same shape as ``alpha`` (a
    float for a scalar).
  """
  arr = np.asarray(clip_alpha(family, alpha), dtype=float)
  if family == "tanh":
    out = math.log(3.0) / arr
  elif family == "exp":
    out = math.log(2.0) / arr
  elif family == "log":
    out = arr.copy()  # T_knee = alpha.
  elif family == "power":
    out = np.full_like(arr, np.nan)
  else:
    raise ValueError(f"Unsupported saturation family '{family}'.")
  return float(out) if np.ndim(out) == 0 else out


def alpha_from_half_saturation(
    family: str, t_half: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
  """Inverse of :func:`half_saturation_point`.

  ``tanh``: ``lambda = ln 3 / T_1/2``; ``exp``: ``alpha = ln 2 / T_1/2``;
  ``log``: ``alpha = T_knee`` (the argument is the marginal-effect-halving
  point).

  Args:
    family: One of ``HALF_SATURATION_FAMILIES``.
    t_half: Strictly positive grid-variable value(s) (``T_1/2``, or
      ``T_knee`` for ``log``).

  Returns:
    The saturation parameter(s) (a float for a scalar input).

  Raises:
    ValueError: For ``power`` (scale free) or a non-positive ``t_half``.
  """
  arr = np.asarray(t_half, dtype=float)
  if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
    raise ValueError(f"t_half must be strictly positive (got {t_half}).")
  if family == "tanh":
    out = math.log(3.0) / arr
  elif family == "exp":
    out = math.log(2.0) / arr
  elif family == "log":
    out = arr.copy()  # alpha = T_knee.
  else:
    raise ValueError(
        f"Family '{family}' has no half-saturation parametrisation"
        " ('power' is scale free)."
    )
  return float(out) if np.ndim(out) == 0 else out


def build_alpha_grid(grid_cfg: Dict[str, Any], family: str) -> np.ndarray:
  """Builds the compact alpha grid `G`.

  Reads the ``saturation.grid`` block: ``min`` / ``max`` / ``num`` /
  ``scale`` (``log`` or ``linear``).

  Args:
    grid_cfg: The ``grid`` sub-dictionary of the ``saturation`` config block.
    family: One of ``SUPPORTED_FAMILIES`` (used for the default bounds).

  Returns:
    A strictly increasing 1-D array of admissible alpha values.

  Raises:
    ValueError: If the configured grid is degenerate.
  """
  grid_cfg = grid_cfg or {}
  lo_default, hi_default = default_alpha_bounds(family)
  lo = float(grid_cfg.get("min", lo_default))
  hi = float(grid_cfg.get("max", hi_default))
  num = int(grid_cfg.get("num", 25))
  scale = str(grid_cfg.get("scale", "log")).strip().lower()

  if num < 2:
    raise ValueError("saturation.grid.num must be >= 2.")
  if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    raise ValueError(
        f"Invalid saturation grid bounds: min={lo}, max={hi} (need min < max)."
    )
  if lo <= 0:
    raise ValueError(
        "saturation.grid.min must be strictly positive (the grid must stay"
        " bounded away from the parameter space boundary)."
    )
  if family == "power" and hi >= 1.0:
    raise ValueError(
        "For family='power' the grid must satisfy max < 1 (alpha in (0, 1));"
        f" got max={hi}."
    )

  if scale == "log":
    grid = np.logspace(np.log10(lo), np.log10(hi), num=num)
  elif scale == "linear":
    grid = np.linspace(lo, hi, num=num)
  else:
    raise ValueError(
        f"Unsupported saturation.grid.scale '{scale}' (use 'log' or 'linear')."
    )
  return np.array([clip_alpha(family, a) for a in grid], dtype=float)


class TreatmentBasis:
  """Fixed B-spline basis ``{b_j(T)}`` on ``[0, T_max]``.

  The basis is used *only* to approximate ``fbar(.;alpha)`` so that the
  multi-output regression ``E[b_j(T)|X]`` is trained once and reused for
  every alpha of the grid.  It is unrelated to ``covariates_for_cate``.

  Domain and knots: the basis lives on ``[0, T_max]`` with ``T_max`` the
  :data:`BASIS_T_MAX_QUANTILE` quantile of ``T > 0`` (normally computed once
  on the FULL data before the split and passed in by the caller, exactly like
  ``T_ref``; see :func:`basis_domain_max`).  The boundary knots are ``0`` and
  ``T_max`` (repeated ``degree + 1`` times, clamped B-splines) and the
  ``J - 4`` interior knots sit at equally spaced quantiles of ``T > 0`` in
  ``D_sel`` (quantiles including ``T = 0`` are not used, to avoid piling knots
  onto the mass at zero).  Observations above ``T_max`` are clipped to
  ``T_max`` by :meth:`transform` (``n_above_t_max`` counts them).  The basis
  is used only in Stage B; Phases 1-3 evaluate ``fbar(T; alpha_hat)``
  directly, so ``T > T_max`` in ``D_inf`` causes no extrapolation.

  Definition of ``c(alpha)``: on the fixed projection grid ``{t_g}`` -- the
  union of :data:`BASIS_PROJECTION_GRID` equally spaced points over
  ``[0, T_max]``, :data:`BASIS_QUANTILE_GRID` empirical quantiles of ``T > 0``
  and the Greville abscissae of the basis -- the basis-value matrix ``B``
  (``G x J``) and the least-squares projector ``P = B^+`` (``J x G``, the
  pseudo-inverse through the SVD of ``B``) are constants -- independent of
  alpha and of ``Y, X`` -- computed ONCE here and kept.  ``c(alpha) = P
  fbar_alpha`` with ``fbar_alpha = (fbar(t_g; alpha))_g`` is the least-squares
  projection on the grid: NOT a regression on the observed ``T_i``.  The
  Greville abscissae guarantee that every basis function is seen by the grid
  (Schoenberg-Whitney), i.e. that ``B`` has full column rank; the quantile
  points weight the fit towards the region where the observations lie.  The
  condition number ``cond(B)`` (``basis_cond``) is kept and
  ``ill_conditioned`` flags ``cond(B) >= MAX_BASIS_CONDITION``.

  All ``J`` basis functions are kept (partition of unity),
  so the basis has exact *linear precision*: ``T = sum_j xi_j b_j(T)`` with
  the Greville abscissae ``xi_j`` (de Boor 2001), which the identity candidate
  of the alpha grid relies on.  The approximation error ``eps(J)`` is still
  measured over the observed ``T_i``.

  The basis holds the reference point ``T_ref`` and projects
  the NORMALIZED ``fbar = f / f(T_ref; alpha)``.
  """

  def __init__(
      self,
      t_values: np.ndarray,
      basis_dim: int = 15,
      degree: int = 3,
      t_ref: Optional[float] = None,
      n_grid: int = BASIS_PROJECTION_GRID,
      t_max: Optional[float] = None,
  ):
    """Places the knots, builds ``B`` and the projector ``P``.

    Args:
      t_values: Observed treatment values of ``D_sel`` (knots, quantile grid
        points and ``eps(J)`` support).
      basis_dim: ``J``, the number of basis functions (``basis_dim_T``).  It
        is reduced automatically when ties in ``T`` make some inner knots
        coincide (the realised ``J`` is ``n_basis``).
      degree: Spline degree (cubic by default).
      t_ref: The reference point ``T_ref`` of ``fbar``
        (normally ``q95(T | T > 0)`` of the FULL data, passed in by the
        caller).  ``None`` falls back to the 95th percentile of ``T > 0`` of
        ``t_values``; the normalization is always on.
      n_grid: Number of equally spaced points of the projection grid (500 by
        default); the quantile points and the Greville abscissae are added on
        top, so the realised grid size is ``n_grid`` (attribute).
      t_max: The right end ``T_max`` of the domain (normally
        :func:`basis_domain_max` of the FULL data, passed in by the caller).
        ``None`` computes it from ``t_values`` with the same rule.
    """
    t_arr = np.asarray(t_values, dtype=float)
    t_arr = t_arr[np.isfinite(t_arr)]
    if t_arr.size == 0:
      raise ValueError("Cannot build a treatment basis from an empty array.")
    self.degree = int(degree)
    order = self.degree + 1
    # The domain is [0, T_max]; T >= 0 for every family (negative values
    # are clipped to 0 by `apply_saturation`).
    self.t_min = 0.0
    positive = t_arr[t_arr > 0.0]
    if t_max is None:
      t_max = basis_domain_max(t_arr)
    self.t_max = float(t_max)
    if not np.isfinite(self.t_max) or self.t_max <= self.t_min:
      # Degenerate treatment (constant / all zero): widen artificially so the
      # basis is still defined; every alpha then gives a constant f and the
      # R-loss cannot discriminate, which the diagnostics report.
      self.t_max = self.t_min + 1.0
    # Rows of `t_values` that `transform` clips to T_max (reported by Diag1).
    self.n_above_t_max = int(np.count_nonzero(t_arr > self.t_max))
    # T_ref (the normalization is always on).
    if t_ref is None:
      t_ref = (
          float(np.quantile(positive, DEFAULT_T_REF_QUANTILE))
          if positive.size else self.t_max
      )
    self.t_ref = float(t_ref)
    if not (np.isfinite(self.t_ref) and self.t_ref > 0.0):
      raise ValueError(f"T_ref must be finite and > 0 (got {t_ref}).")
    requested = max(int(basis_dim), order)
    n_inner = requested - order
    # Interior knots at equally spaced quantiles of T > 0 (inside the domain).
    if n_inner > 0 and positive.size > 0:
      probs = np.linspace(0.0, 1.0, num=n_inner + 2)[1:-1]
      inner = np.quantile(positive, probs)
      inner = np.unique(inner[(inner > self.t_min) & (inner < self.t_max)])
    else:
      inner = np.zeros(0, dtype=float)
    self.inner_knots = inner
    self.knots = np.concatenate([
        np.full(order, self.t_min),
        inner,
        np.full(order, self.t_max),
    ])
    self.n_basis = int(self.knots.size - order)
    self.basis_dim = self.n_basis
    self._spline = scipy_interpolate.BSpline(
        self.knots, np.eye(self.n_basis), self.degree, extrapolate=True
    )
    # Projection grid {t_g}: equally spaced points, empirical quantiles of
    # T > 0 (clipped to the domain) and the Greville abscissae.  With the
    # Greville abscissae every basis function has a grid point inside its
    # support where it is strictly positive (Schoenberg-Whitney), so B has
    # full column rank no matter how skewed T is; the quantile points make the
    # least-squares fit weight the observed region.  Computed once and kept.
    uniform = np.linspace(
        self.t_min, self.t_max, num=int(max(n_grid, self.n_basis + 1))
    )
    if positive.size:
      quantile_points = np.quantile(
          np.minimum(positive, self.t_max),
          np.linspace(0.0, 1.0, num=BASIS_QUANTILE_GRID),
      )
    else:
      quantile_points = np.zeros(0, dtype=float)
    self.t_grid = np.unique(
        np.concatenate([uniform, quantile_points, self.greville_abscissae()])
    )
    self.n_grid = int(self.t_grid.size)
    self.b_grid = self.transform(self.t_grid)
    # Least-squares projector P = B^+ (J x G) through the SVD of B.  The
    # normal equations B'B are never formed: cond(B'B) = cond(B)^2 would lose
    # half of the available precision, and `np.linalg.solve` does not raise on
    # a numerically singular (as opposed to exactly singular) Gram matrix.
    singular_values = np.linalg.svd(self.b_grid, compute_uv=False)
    s_max = float(singular_values[0]) if singular_values.size else 0.0
    s_min = float(singular_values[-1]) if singular_values.size else 0.0
    self.basis_cond = float(s_max / s_min) if s_min > 0.0 else float("inf")
    self.ill_conditioned = bool(self.basis_cond >= MAX_BASIS_CONDITION)
    self.projector = np.linalg.pinv(self.b_grid, rcond=_BASIS_PINV_RCOND)
    # eps(J) is measured over the observed T_i of D_sel (quantile-compressed
    # for very large samples; repeated values keep their empirical mass).
    # Used ONLY for eps(J), never for c(alpha).
    support = np.sort(np.clip(t_arr, self.t_min, self.t_max))
    if support.size > 20000:
      support = np.quantile(support, np.linspace(0.0, 1.0, num=20000))
    self._t_support = support
    self._b_support = self.transform(support)

  def transform(self, t_values: np.ndarray) -> np.ndarray:
    """Evaluates the basis at arbitrary treatment values.

    Args:
      t_values: Treatment values (clipped to the domain ``[0, T_max]`` to avoid
        spline extrapolation).

    Returns:
      An ``(n, J)`` design matrix.
    """
    t_arr = np.clip(
        np.asarray(t_values, dtype=float).reshape(-1), self.t_min, self.t_max
    )
    mat = np.asarray(self._spline(t_arr), dtype=float)
    return mat.reshape(t_arr.size, self.n_basis)

  def greville_abscissae(self) -> np.ndarray:
    """Returns the Greville abscissae ``xi_j``.

    ``xi_j = (t_{j+1} + ... + t_{j+degree}) / degree`` for the clamped knot
    vector; then ``sum_j xi_j b_j(T) = T`` exactly on ``[0, T_max]``.  Hence
    the identity candidate ``fbar_T = T / T_ref`` has the exact nuisance
    ``m_T = (1 / T_ref) sum_j xi_j mu_j``.

    Returns:
      A ``(J,)`` array.
    """
    deg = max(self.degree, 1)
    return np.array(
        [
            float(np.mean(self.knots[j + 1 : j + 1 + deg]))
            for j in range(self.n_basis)
        ],
        dtype=float,
    )

  def normalized_values(
      self, t_values: np.ndarray, family: str, alphas: np.ndarray
  ) -> np.ndarray:
    """``fbar(T; alpha)`` for every alpha, column by column.

    Args:
      t_values: Treatment values (``n``).
      family: One of ``SUPPORTED_FAMILIES``.
      alphas: 1-D array of alpha values (length ``G_alpha``).

    Returns:
      An ``(n, G_alpha)`` matrix of ``fbar(T_i; alpha_g)``.
    """
    alphas = np.ravel(np.asarray(alphas, dtype=float))
    return np.stack(
        [
            normalized_saturation(t_values, family, float(a), self.t_ref)
            for a in alphas
        ],
        axis=1,
    )

  def project_values(self, f_grid: np.ndarray) -> np.ndarray:
    """``c = P f`` for arbitrary function values on the fixed grid.

    Args:
      f_grid: ``(G, G_alpha)`` function values at ``self.t_grid``.

    Returns:
      A ``(J, G_alpha)`` coefficient matrix.
    """
    f_arr = np.asarray(f_grid, dtype=float)
    if f_arr.shape[0] != self.n_grid:
      raise ValueError(
          f"Expected function values on the fixed {self.n_grid}-point grid"
          f" (got {f_arr.shape[0]} rows); c(alpha) is defined on that grid,"
          " not on the observed T_i."
      )
    return self.projector @ f_arr

  def project_family(
      self, family: str, alphas: np.ndarray, n_grid: Optional[int] = None
  ) -> np.ndarray:
    """``c(alpha) = P fbar_alpha`` for every alpha at once.

    The basis projection ``fbar(T;alpha) ~= sum_j c_j(alpha) b_j(T)``: the
    only alpha-dependent work is evaluating ``fbar`` on the ``G`` grid points
    and one matrix product with the precomputed projector ``P``; ``c(alpha)``
    is not a regression coefficient on the observed data.

    Args:
      family: One of ``SUPPORTED_FAMILIES``.
      alphas: 1-D array of alpha values (length ``G_alpha``).
      n_grid: Unused; kept for compatibility with earlier call sites (the
        grid size is fixed at construction).

    Returns:
      A ``(J, G_alpha)`` matrix of basis coefficients ``c_j(alpha)``.
    """
    del n_grid
    return self.project_values(
        self.normalized_values(self.t_grid, family, alphas)
    )

  def approximation_error(
      self,
      family: str,
      alphas: np.ndarray,
      t_eval: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Relative RMS approximation error per alpha.

    ``eps(alpha) = RMS_i(fbar(T_i;alpha) - sum_j c_j(alpha) b_j(T_i))
    / sd_i(fbar(T_i;alpha))`` over the observed ``T_i`` of ``D_sel`` (the
    normalization cancels between numerator and denominator, so ``eps`` is
    the same quantity as for the un-normalized ``f``).

    Args:
      family: One of ``SUPPORTED_FAMILIES``.
      alphas: 1-D array of alpha values.
      t_eval: Treatment values the error is measured on (defaults to the
        observed support the basis was built from).

    Returns:
      A ``(G_alpha,)`` array of relative errors (0 where ``sd(fbar) == 0``).
    """
    alphas = np.ravel(np.asarray(alphas, dtype=float))
    coefs = self.project_family(family, alphas)
    if t_eval is None:
      t_arr, design = self._t_support, self._b_support
    else:
      t_arr = np.asarray(t_eval, dtype=float).reshape(-1)
      design = self.transform(t_arr)
    f_true = self.normalized_values(t_arr, family, alphas)
    resid = f_true - design @ coefs
    rms = np.sqrt(np.mean(resid * resid, axis=0))
    sd = np.std(f_true, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
      rel = np.where(sd > 0.0, rms / np.where(sd > 0.0, sd, 1.0), 0.0)
    return rel


@dataclasses.dataclass
class BasisSelection:
  """Outcome of the automatic choice of J.

  ``ill_conditioned`` is True when the adopted basis (fixed ``J``) or every
  scanned basis (``auto``) has ``cond(B) >= MAX_BASIS_CONDITION``; the caller
  (Stage B) then adopts the linear model (``decision_reason =
  "basis_ill_conditioned"``) instead of trusting the coefficients.
  """

  basis: TreatmentBasis
  n_basis: int
  approx_error: float
  satisfied: bool
  mode: str  # "auto" | "fixed"
  scanned: Dict[int, float] = dataclasses.field(default_factory=dict)
  # cond(B) of the adopted basis and of every scanned J.
  basis_cond: float = float("nan")
  scanned_cond: Dict[int, float] = dataclasses.field(default_factory=dict)
  ill_conditioned: bool = False


def select_basis_dim(
    t_values: np.ndarray,
    family: str,
    alphas: np.ndarray,
    basis_dim: Union[str, int, None] = "auto",
    tol: float = 1e-3,
    dim_range: Sequence[int] = (6, 20),
    t_ref: Optional[float] = None,
    t_max: Optional[float] = None,
) -> BasisSelection:
  """Chooses the number ``J`` of T-basis functions.

  With ``basis_dim='auto'``, ``J`` is scanned over ``dim_range`` in increasing
  order; bases with ``cond(B) >= MAX_BASIS_CONDITION`` are skipped and the
  smallest admissible ``J`` with ``max_alpha eps(J) < tol`` is adopted.  If no
  admissible ``J`` meets the tolerance, the admissible ``J`` with the smallest
  ``eps(J)`` is used and ``satisfied=False`` is returned (the caller warns).
  If no ``J`` is admissible at all, the last scanned basis is returned with
  ``ill_conditioned=True`` (the caller falls back to the linear model).  An
  integer ``basis_dim`` fixes ``J``; it is never rejected here, but
  ``ill_conditioned`` reports whether its basis can be trusted.  Knot
  placement and ``c(alpha)`` are described in :class:`TreatmentBasis`.

  Args:
    t_values: Observed treatment values of ``D_sel`` (on the scale Phase 0
      works on).
    family: The saturation family.
    alphas: The alpha grid ``G``.
    basis_dim: ``"auto"`` or an integer.
    tol: ``basis_tol``.
    dim_range: ``basis_dim_range`` ``[J_min, J_max]``.
    t_ref: The reference point ``T_ref`` of ``fbar``.
    t_max: The right end ``T_max`` of the basis domain (normally
      :func:`basis_domain_max` of the full data); ``None`` derives it from
      ``t_values``.

  Returns:
    The :class:`BasisSelection`.
  """
  if basis_dim is not None and str(basis_dim).strip().lower() != "auto":
    basis = TreatmentBasis(
        t_values, basis_dim=int(basis_dim), t_ref=t_ref, t_max=t_max
    )
    err = float(np.max(basis.approximation_error(family, alphas)))
    return BasisSelection(
        basis=basis,
        n_basis=basis.n_basis,
        approx_error=err,
        satisfied=bool(err < float(tol) and not basis.ill_conditioned),
        mode="fixed",
        scanned={basis.n_basis: err},
        basis_cond=basis.basis_cond,
        scanned_cond={basis.n_basis: basis.basis_cond},
        ill_conditioned=basis.ill_conditioned,
    )

  lo, hi = int(dim_range[0]), int(dim_range[1])
  if hi < lo:
    lo, hi = hi, lo
  scanned: Dict[int, float] = {}
  scanned_cond: Dict[int, float] = {}
  best: Optional[Tuple[TreatmentBasis, float]] = None
  last: Optional[Tuple[TreatmentBasis, float]] = None
  for j_dim in range(lo, hi + 1):
    basis = TreatmentBasis(t_values, basis_dim=j_dim, t_ref=t_ref, t_max=t_max)
    err = float(np.max(basis.approximation_error(family, alphas)))
    scanned[j_dim] = err
    scanned_cond[j_dim] = basis.basis_cond
    last = (basis, err)
    if basis.ill_conditioned:
      continue
    if err < float(tol):
      return BasisSelection(
          basis=basis,
          n_basis=basis.n_basis,
          approx_error=err,
          satisfied=True,
          mode="auto",
          scanned=scanned,
          basis_cond=basis.basis_cond,
          scanned_cond=scanned_cond,
          ill_conditioned=False,
      )
    if best is None or err < best[1]:
      best = (basis, err)
  assert last is not None
  basis, err = best if best is not None else last
  return BasisSelection(
      basis=basis,
      n_basis=basis.n_basis,
      approx_error=err,
      satisfied=False,
      mode="auto",
      scanned=scanned,
      basis_cond=basis.basis_cond,
      scanned_cond=scanned_cond,
      ill_conditioned=best is None,
  )


@dataclasses.dataclass
class AlphaGrid:
  """The alpha grid ``G`` of Stage B.

  ``coord`` is the coordinate the grid is equally spaced in: ``log T_1/2``
  for the automatic half-saturation grid (tanh / exp / log), ``alpha`` for
  ``power`` and for manual linear grids, ``log alpha`` for manual log grids.
  ``grid_index`` is the position in these arrays (increasing ``coord``).
  """

  family: str
  mode: str  # "auto" | "manual"
  coord_kind: str  # "log_t_half" | "alpha" | "log_alpha"
  coord: np.ndarray
  alphas: np.ndarray
  t_half: np.ndarray
  q50: float = float("nan")
  q95: float = float("nan")
  c_min: float = float("nan")
  c_max: float = float("nan")
  extensions: int = 0
  extensible: bool = False

  @property
  def width(self) -> float:
    """Width of the grid in its native (equally spaced) coordinate."""
    if self.coord.size < 2:
      return 0.0
    return float(self.coord[-1] - self.coord[0])

  def alpha_at(self, coord_value: float) -> float:
    """Maps a coordinate value back to alpha."""
    if self.coord_kind == "log_t_half":
      return float(
          alpha_from_half_saturation(self.family, math.exp(coord_value))
      )
    if self.coord_kind == "log_alpha":
      return float(clip_alpha(self.family, math.exp(coord_value)))
    return float(clip_alpha(self.family, coord_value))

  def coord_of(self, alpha: float) -> float:
    """Maps alpha to the grid coordinate."""
    if self.coord_kind == "log_t_half":
      return float(math.log(half_saturation_point(self.family, alpha)))
    if self.coord_kind == "log_alpha":
      return float(math.log(alpha))
    return float(alpha)

  @property
  def linear_side(self) -> str:
    """Which index end of the grid is the linear side.

    The edge rule is stated in the ``T_1/2`` orientation: the
    "upper edge" is the large-``T_1/2`` (slow-saturation, close-to-linear)
    end.  For the automatic ``log_t_half`` grid that is the last index.  For
    grids laid out in ``alpha`` (``power``, manual grids) the orientation
    depends on the family: ``tanh`` / ``exp`` approach the linear model as
    ``alpha -> 0`` (first index), ``log`` as ``T_knee = alpha -> inf`` and
    ``power`` as ``alpha -> 1`` (last index).

    Returns:
      ``"upper"`` if the linear side is the last grid index, ``"lower"`` if it
      is the first.
    """
    if self.coord_kind == "log_t_half":
      return "upper"
    return "lower" if self.family in ("tanh", "exp") else "upper"


def _grid_from_coord(template: AlphaGrid, coord: np.ndarray) -> AlphaGrid:
  """Rebuilds alphas / t_half for a new coordinate vector."""
  alphas = np.array([template.alpha_at(c) for c in coord], dtype=float)
  t_half = np.asarray(
      half_saturation_point(template.family, alphas), dtype=float
  ).reshape(-1)
  return dataclasses.replace(
      template, coord=np.asarray(coord, float), alphas=alphas, t_half=t_half
  )


def resolve_grid_mode(grid_cfg: Optional[Dict[str, Any]]) -> str:
  """Returns ``auto`` or ``manual`` for a ``saturation.grid`` block.

  ``mode`` defaults to ``auto``.  For backward compatibility a legacy block
  that sets ``min``/``max`` but no ``mode`` is treated as
  ``manual`` (the explicit override is honoured rather than silently ignored).

  Args:
    grid_cfg: The ``grid`` block (may be ``None``).

  Returns:
    ``"auto"`` or ``"manual"``.
  """
  grid_cfg = grid_cfg or {}
  mode = grid_cfg.get("mode")
  if mode is None:
    has_bounds = (grid_cfg.get("min") is not None) or (
        grid_cfg.get("max") is not None
    )
    return "manual" if has_bounds else "auto"
  mode = str(mode).strip().lower()
  if mode not in ("auto", "manual"):
    raise ValueError(
        f"saturation.grid.mode must be 'auto' or 'manual' (got '{mode}')."
    )
  return mode


def build_half_saturation_grid(
    t_values: np.ndarray,
    family: str,
    grid_cfg: Optional[Dict[str, Any]] = None,
) -> AlphaGrid:
  """Builds the dimensionless alpha grid.

  ``G = logspace(c_min * q50(T>0), c_max * q95(T>0), num)`` in the grid
  variable, converted per family (tanh: ``alpha = ln3 / T_1/2``; exp:
  ``alpha = ln2 / T_1/2``; log: ``alpha = T_knee`` -- the log family has no
  half-saturation point, its grid variable is the knee point ``T_knee``
  where the marginal effect has halved).  ``power`` uses ``num`` equally
  spaced alphas on ``[0.1, 0.95]``.  With ``mode: manual`` the legacy
  ``min`` / ``max`` / ``scale`` grid (alpha units) is used as an override.

  Why log spacing: the R-loss profile is a function of the *ratio*
  ``T_1/2 / T``; a curve with ``T_1/2 = 2 q95`` differs from one with
  ``4 q95`` about as much as ``0.5 q95`` differs from ``q95``, so equal
  steps in ``log T_1/2`` give equal resolution in curve shape over the whole
  range (and the x4 edge extensions are plain translations of the grid).

  Args:
    t_values: Observed treatment values (the scale Phase 0 works on).
    family: The saturation family.
    grid_cfg: The ``saturation.grid`` block.

  Returns:
    The :class:`AlphaGrid`.
  """
  grid_cfg = dict(grid_cfg or {})
  mode = resolve_grid_mode(grid_cfg)
  num = int(grid_cfg.get("num", 25) or 25)
  if num < 2:
    raise ValueError("saturation.grid.num must be >= 2.")

  if mode == "manual":
    manual_cfg = {
        k: v
        for k, v in grid_cfg.items()
        if k in ("min", "max", "num", "scale") and v is not None
    }
    alphas = build_alpha_grid(manual_cfg, family)
    scale = str(grid_cfg.get("scale", "log") or "log").strip().lower()
    if family == "power":
      scale = "linear"  # power grids are always linear.
    coord_kind = "log_alpha" if scale == "log" else "alpha"
    coord = np.log(alphas) if coord_kind == "log_alpha" else alphas.copy()
    t_half = np.asarray(half_saturation_point(family, alphas), float)
    return AlphaGrid(
        family=family,
        mode="manual",
        coord_kind=coord_kind,
        coord=np.asarray(coord, float),
        alphas=alphas,
        t_half=t_half.reshape(-1),
        extensible=False,
    )

  if family == "power":
    lo, hi = POWER_GRID_BOUNDS
    alphas = np.linspace(lo, hi, num=num)
    return AlphaGrid(
        family=family,
        mode="auto",
        coord_kind="alpha",
        coord=alphas.copy(),
        alphas=alphas,
        t_half=np.full_like(alphas, np.nan),
        extensible=False,
    )

  c_min = float(grid_cfg.get("c_min", 0.25) or 0.25)
  c_max = float(grid_cfg.get("c_max", 4.0) or 4.0)
  t_arr = np.asarray(t_values, dtype=float)
  t_pos = t_arr[np.isfinite(t_arr) & (t_arr > 0.0)]
  if t_pos.size == 0:
    raise ValueError(
        "The automatic half-saturation grid needs at least one"
        " strictly positive treatment value."
    )
  q50 = float(np.quantile(t_pos, 0.50))
  q95 = float(np.quantile(t_pos, 0.95))
  lo = c_min * q50
  hi = c_max * q95
  if not hi > lo:
    hi = lo * 16.0
  coord = np.linspace(math.log(lo), math.log(hi), num=num)
  template = AlphaGrid(
      family=family,
      mode="auto",
      coord_kind="log_t_half",
      coord=coord,
      alphas=np.zeros(0),
      t_half=np.zeros(0),
      q50=q50,
      q95=q95,
      c_min=c_min,
      c_max=c_max,
      extensible=True,
  )
  return _grid_from_coord(template, coord)


def extend_alpha_grid(
    grid: AlphaGrid, side: str, factor: float = 4.0
) -> AlphaGrid:
  """Extends the grid on one side by ``factor``.

  The added points keep (approximately) the existing log spacing and span a
  factor of exactly ``factor`` in ``T_1/2``.

  Args:
    grid: The current grid (must be ``extensible``).
    side: ``"lower"`` (smaller ``T_1/2``: faster saturation) or ``"upper"``
      (larger ``T_1/2``: closer to linear).
    factor: Multiplicative extension of the range (4 by default).

  Returns:
    The extended grid (``extensions`` incremented).
  """
  if not grid.extensible:
    raise ValueError("This grid cannot be extended (manual or power grid).")
  step = grid.width / max(grid.coord.size - 1, 1)
  span = math.log(float(factor))
  n_add = max(1, int(round(span / step))) if step > 0 else 1
  if side == "lower":
    added = np.linspace(grid.coord[0] - span, grid.coord[0], num=n_add + 1)[:-1]
    coord = np.concatenate([added, grid.coord])
  elif side == "upper":
    added = np.linspace(grid.coord[-1], grid.coord[-1] + span, num=n_add + 1)[
        1:
    ]
    coord = np.concatenate([grid.coord, added])
  else:
    raise ValueError(f"side must be 'lower' or 'upper' (got '{side}').")
  out = _grid_from_coord(grid, coord)
  out.extensions = grid.extensions + 1
  return out


def _normalization_constant(
    family: str,
    alpha: np.ndarray,
    t_ref: Optional[Union[float, np.ndarray]],
    like: np.ndarray,
) -> np.ndarray:
  """``kappa = f(T_ref; alpha)`` broadcast to ``like``.

  ``kappa`` is the normalization constant of ``fbar = f / kappa``; with
  ``t_ref=None`` the un-normalized ``f`` is meant and ``kappa = 1``.  For the
  ``tanh`` family ``kappa = tanh(lambda T_ref / 2)``.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The (already broadcast) saturation parameter.
    t_ref: ``T_ref`` (scalar, or one value per row), or ``None``.
    like: Array whose shape ``kappa`` is broadcast to.

  Returns:
    ``kappa`` shaped like ``like``.
  """
  if t_ref is None:
    return np.ones(np.shape(like), dtype=float)
  t_ref_arr = np.asarray(t_ref, dtype=float)
  if np.any(~np.isfinite(t_ref_arr)) or np.any(t_ref_arr <= 0.0):
    raise ValueError(
        f"T_ref must be finite and strictly positive (got {t_ref}); it is"
        " defined as q95(T | T > 0)."
    )
  t_ref_b = np.broadcast_to(t_ref_arr, np.shape(like)).astype(float)
  return np.asarray(apply_saturation(t_ref_b, family, alpha), dtype=float)


def _response(
    t_values: np.ndarray,
    family: str,
    alpha: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]],
) -> np.ndarray:
  """``fbar(T; alpha)`` with ``t_ref``, the raw ``f(T; alpha)`` without."""
  if t_ref is None:
    return apply_saturation(t_values, family, alpha)
  return normalized_saturation(t_values, family, alpha, t_ref)


def normalized_derivative(
    t_values: np.ndarray,
    family: str,
    alpha: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
  """``fbar'(T; alpha) = f'(T; alpha) / kappa``.

  The marginal ROI of a unit is ``mROI_t(T) = theta_t fbar'(T)``.
  With ``t_ref=None`` the raw ``f'`` is returned (``kappa = 1``).

  Args:
    t_values: Treatment values.
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter (scalar or broadcastable array).
    t_ref: ``T_ref`` (scalar, or one value per row), or ``None``.

  Returns:
    ``fbar'(T)`` broadcast over the inputs.
  """
  t_arr = np.asarray(t_values, dtype=float)
  alpha_b = _broadcast_alpha(family, alpha, t_arr)
  kappa = _normalization_constant(family, alpha_b, t_ref, t_arr)
  return saturation_derivative(t_arr, family, alpha_b) / kappa


def marginal_effect_at_zero(
    family: str,
    alpha: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
  """``fbar'(0; alpha)``, the marginal effect at the origin (``T = 0``).

  Per family this is ``(lambda/2)/kappa`` (tanh), ``alpha/kappa`` (exp),
  ``1/(alpha kappa)`` (log) and ``+inf`` (power).

  Used by the ROI cap (:func:`roi_cap`, ``T_t^r = 0`` when
  ``theta_t fbar'(0) <= r``) and by the KKT inversion (``T_t*(mu) = 0`` when
  ``theta_t fbar'(0) <= mu``).  Note that ``log`` has a FINITE slope at the
  origin (``f'(0) = 1 / alpha``), so the zero branch is reachable for
  ``tanh`` / ``exp`` / ``log``; only ``power`` always allocates a positive
  amount to every ``theta_t > 0`` unit.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter (scalar or array).
    t_ref: ``T_ref`` (scalar, or one value per ``alpha``), or ``None``.

  Returns:
    ``fbar'(0)`` shaped like ``alpha``.
  """
  alpha_arr = np.asarray(clip_alpha(family, alpha), dtype=float)
  kappa = _normalization_constant(family, alpha_arr, t_ref, alpha_arr)
  if family == "tanh":
    slope = 0.5 * alpha_arr
  elif family == "exp":
    slope = alpha_arr
  elif family == "log":
    slope = 1.0 / alpha_arr
  elif family == "power":
    slope = np.full_like(alpha_arr, np.inf)
  else:
    raise ValueError(f"Unsupported saturation family '{family}'.")
  return np.asarray(slope / kappa, dtype=float)


def roi_cap(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    threshold_roi: float,
    t_ref: Optional[Union[float, np.ndarray]] = None,
    t_upper: Optional[Union[float, np.ndarray]] = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
  """The ROI cap ``T_t^r`` of the saturation-aware optimizer.

  ``ROI_t(T) = theta_t fbar(T) / T`` is strictly decreasing in ``T`` (``fbar``
  concave with ``fbar(0) = 0``), so ``ROI_t(T) >= r`` is equivalent to the
  upper bound ``T <= T_t^r`` with

  * ``T_t^r = inf`` when ``r = 0`` (no constraint),
  * ``T_t^r = 0`` when ``theta_t fbar'(0) <= r`` (even the first unit of
    spend is below the threshold),
  * otherwise ``T_t^r`` = the unique root of ``theta_t fbar(T) / T = r``,
    found by (vectorized) bisection.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter (scalar or one value per unit).
    theta: ``theta_hat_t`` per unit (units of ``fbar``); units with
      ``theta_t <= 0`` get ``T_t^r = 0``.
    threshold_roi: ``r = opt_threshold_roi >= 0``.
    t_ref: ``T_ref`` (scalar or per unit), or ``None`` for the raw ``f``.
    t_upper: Optional search bound per unit (normally the cap ``T_bar_t``).
      When ``ROI_t(t_upper) >= r`` the cap does not bind below ``t_upper``
      and ``+inf`` is returned for that unit instead of the exact root (the
      effective cap ``min(T_bar_t, T_t^r)`` is unaffected).
    tol: Relative tolerance of the bisection on ``T``.
    max_iter: Maximum number of bisection steps.

  Returns:
    ``T_t^r`` per unit (``+inf`` where the threshold never binds).

  Raises:
    ValueError: If ``threshold_roi < 0``.
  """
  r = float(threshold_roi)
  if r < 0.0:
    raise ValueError(f"opt_threshold_roi must be >= 0 (got {threshold_roi}).")
  theta = np.asarray(theta, dtype=float)
  out = np.full(theta.shape, np.inf, dtype=float)
  if theta.size == 0:
    return out
  if r == 0.0:
    out[theta <= 0.0] = 0.0
    return out
  alpha_b = _broadcast_alpha(family, alpha, theta)
  t_ref_b = (
      None if t_ref is None
      else np.broadcast_to(np.asarray(t_ref, dtype=float), theta.shape)
  )
  slope0 = marginal_effect_at_zero(family, alpha_b, t_ref_b)
  zero = (theta <= 0.0) | (theta * slope0 <= r)
  out[zero] = 0.0
  active = ~zero
  if not np.any(active):
    return out

  def _roi(t_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t_vals = np.asarray(t_vals, dtype=float)
    resp = _response(
        t_vals, family, alpha_b[mask],
        None if t_ref_b is None else t_ref_b[mask],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.where(t_vals > 0.0, theta[mask] * resp / t_vals, np.inf)

  # Upper bracket: the search bound when given and binding, otherwise an
  # expanding geometric bracket starting at T_ref (or 1).
  idx = np.flatnonzero(active)
  if t_upper is not None:
    upper = np.broadcast_to(np.asarray(t_upper, dtype=float), theta.shape)
    upper_a = np.maximum(upper[idx], 0.0)
    has_upper = np.isfinite(upper_a) & (upper_a > 0.0)
    not_binding = np.zeros(idx.size, dtype=bool)
    if np.any(has_upper):
      roi_up = _roi(np.where(has_upper, upper_a, 1.0), idx)
      not_binding = has_upper & (roi_up >= r)
    out[idx[not_binding]] = np.inf
    keep = ~not_binding
    idx = idx[keep]
    if idx.size == 0:
      return out
    hi = np.where(has_upper[keep], upper_a[keep], np.nan)
  else:
    hi = np.full(idx.size, np.nan)
  start = (
      np.ones(idx.size) if t_ref_b is None
      else np.maximum(np.asarray(t_ref_b[idx], dtype=float), 1e-12)
  )
  hi = np.where(np.isfinite(hi), hi, start)
  for _ in range(int(max_iter)):
    below = _roi(hi, idx) >= r
    if not np.any(below):
      break
    hi = np.where(below, hi * 2.0, hi)
  lo = np.zeros(idx.size)
  for _ in range(int(max_iter)):
    mid = 0.5 * (lo + hi)
    above = _roi(mid, idx) >= r
    lo = np.where(above, mid, lo)
    hi = np.where(above, hi, mid)
    if np.all(hi - lo <= tol * np.maximum(hi, 1.0)):
      break
  out[idx] = 0.5 * (lo + hi)
  return out


def kkt_allocation(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    mu: float,
    caps: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
  """``T_t*(mu) = clip((fbar')^-1(mu / theta_t), 0, cap_t)``.

  Implements the family-specific inversions of the KKT condition
  (each is ``0`` when ``theta_t fbar'(0) <= mu``; ``kappa =
  f(T_ref; alpha)`` is the normalization constant, so ``mu`` enters as
  ``mu kappa``):

  * tanh : ``(2/lambda) artanh sqrt(1 - 2 mu kappa / (lambda theta_t))``
  * exp  : ``(1/alpha) ln(alpha theta_t / (mu kappa))``
  * log  : ``theta_t / (mu kappa) - alpha``
  * power: ``(alpha theta_t / (mu kappa)) ** (1 / (1 - alpha))``

  Units with ``theta_t <= 0`` get ``0``.  ``mu <= 0`` puts every positive unit
  at its cap.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter (scalar, or one value per unit).
    theta: ``theta_hat_t`` per unit.
    mu: The Lagrange multiplier of the budget constraint (``>= 0``).
    caps: The per-unit upper bound (scalar ``T_bar`` or one value per unit;
      e.g. ``min(T_bar_t, T_t^r)``).
    t_ref: ``T_ref`` (scalar or per unit), or ``None`` for the raw ``f``.

  Returns:
    The box-clipped allocation implied by ``mu``.
  """
  theta = np.asarray(theta, dtype=float)
  caps_b = np.broadcast_to(np.asarray(caps, dtype=float), theta.shape)
  caps_b = np.where(np.isfinite(caps_b), np.maximum(caps_b, 0.0), np.inf)
  positive = theta > 0
  out = np.zeros_like(theta)
  if not np.any(positive):
    return out
  if mu <= 0:
    out[positive] = caps_b[positive]
    return out
  theta_p = theta[positive]
  # Treatment normalization bridge: every closed form below is elementwise in
  # (theta, alpha, kappa), so row-varying `alpha_i` / `T_ref_i` need nothing
  # but broadcasting.
  alpha_all = _broadcast_alpha(family, alpha, theta)
  alpha_p = alpha_all[positive]
  # Normalization: mu -> mu * kappa.
  kappa_all = _normalization_constant(family, alpha_all, t_ref, theta)
  mu_p = float(mu) * kappa_all[positive]
  with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    if family == "tanh":
      ratio = 2.0 * mu_p / (alpha_p * theta_p)
      inside = np.clip(1.0 - ratio, 0.0, 1.0)
      vals = np.where(
          ratio < 1.0, (2.0 / alpha_p) * np.arctanh(np.sqrt(inside)), 0.0
      )
    elif family == "exp":
      vals = np.log(np.maximum(alpha_p * theta_p / mu_p, _EPS)) / alpha_p
    elif family == "log":
      vals = theta_p / mu_p - alpha_p
    elif family == "power":
      vals = np.power(
          np.maximum(alpha_p * theta_p / mu_p, _EPS), 1.0 / (1.0 - alpha_p)
      )
    else:
      raise ValueError(f"Unsupported saturation family '{family}'.")
  cap_p = caps_b[positive]
  vals = np.where(np.isnan(vals), 0.0, vals)
  vals = np.where(np.isposinf(vals), cap_p, vals)
  out[positive] = np.clip(vals, 0.0, cap_p)
  return out


def _allocation_from_lambda(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    lam: float,
    t_max: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
  """Backward-compatible name of :func:`kkt_allocation` (``lam`` = ``mu``)."""
  return kkt_allocation(family, alpha, theta, lam, t_max, t_ref)


@dataclasses.dataclass
class BudgetSolution:
  """Outcome of :func:`solve_budget_allocation`."""

  allocation: np.ndarray
  # mu*: the common marginal ROI at the optimum (0 when every unit sits at
  # its cap, i.e. g(0) = sum_t cap_t <= K').
  mu: float
  budget: float
  spent: float
  all_at_cap: bool
  n_iter: int


def solve_budget_allocation(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    budget: float,
    caps: Union[float, np.ndarray],
    t_ref: Optional[Union[float, np.ndarray]] = None,
    tol: float = 1e-10,
    max_iter: int = 400,
) -> BudgetSolution:
  """Solves the saturation-aware budget problem in ONE KKT pass.

  ``max sum_{t not in Z_0} theta_t fbar(T_t)`` s.t. ``sum_t T_t <= K'``,
  ``0 <= T_t <= cap_t`` (``T_t = 0`` for ``theta_t <= 0``).  The objective is
  concave and the feasible set convex, so the KKT point is the global
  maximizer: with ``g(mu) = sum_t T_t*(mu)`` (:func:`kkt_allocation`)
  nonincreasing and continuous, ``mu* = 0`` if ``g(0) = sum_t cap_t <= K'``
  (every unit at its cap), otherwise the unique ``mu* > 0`` with
  ``g(mu*) = K'`` is found by bisection (geometric bracket, then bisection in
  log ``mu``).  No greedy iteration.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The saturation parameter (scalar, or one value per unit).
    theta: ``theta_hat_t`` per unit.
    budget: ``K'`` (``K`` or ``K_+`` depending on ``budget_reallocate``).
    caps: Per-unit caps (``T_bar_t^eff`` or ``T_bar_t``; scalar allowed).
    t_ref: ``T_ref`` (scalar or per unit), or ``None`` for the raw ``f``.
    tol: Relative tolerance on ``|g(mu) - K'|``.
    max_iter: Maximum number of bisection steps.

  Returns:
    The :class:`BudgetSolution`.
  """
  theta = np.asarray(theta, dtype=float)
  budget = float(budget)
  caps_b = np.broadcast_to(np.asarray(caps, dtype=float), theta.shape)
  caps_b = np.where(np.isfinite(caps_b), np.maximum(caps_b, 0.0), np.inf)
  positive = theta > 0
  zeros = np.zeros_like(theta)
  if budget <= 0.0 or not np.any(positive):
    return BudgetSolution(zeros, float("nan") if budget <= 0 else 0.0,
                          budget, 0.0, False, 0)
  total_caps = float(np.sum(caps_b[positive]))
  if total_caps <= budget:
    alloc = kkt_allocation(family, alpha, theta, 0.0, caps_b, t_ref)
    return BudgetSolution(alloc, 0.0, budget, float(alloc.sum()), True, 0)

  def _g(mu: float) -> float:
    return float(np.sum(kkt_allocation(family, alpha, theta, mu, caps_b,
                                       t_ref)))

  # Geometric bracket [mu_lo, mu_hi] with g(mu_lo) >= K' >= g(mu_hi).
  mu_hi = 1.0
  n_iter = 0
  for _ in range(200):
    n_iter += 1
    if _g(mu_hi) <= budget:
      break
    mu_hi *= 2.0
  mu_lo = mu_hi
  for _ in range(200):
    n_iter += 1
    if _g(mu_lo) >= budget:
      break
    mu_lo /= 2.0
  mu_star = mu_hi
  alloc = kkt_allocation(family, alpha, theta, mu_star, caps_b, t_ref)
  for _ in range(int(max_iter)):
    n_iter += 1
    mu_mid = math.sqrt(mu_lo * mu_hi)
    alloc_mid = kkt_allocation(family, alpha, theta, mu_mid, caps_b, t_ref)
    total = float(np.sum(alloc_mid))
    mu_star, alloc = mu_mid, alloc_mid
    if abs(total - budget) <= tol * max(budget, 1.0):
      break
    if total > budget:
      mu_lo = mu_mid
    else:
      mu_hi = mu_mid
    if mu_hi / mu_lo - 1.0 <= 1e-15:
      break
  return BudgetSolution(alloc, float(mu_star), budget, float(alloc.sum()),
                        False, n_iter)


def optimal_allocation(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    budget: float,
    t_max: Union[float, np.ndarray],
    tol: float = 1e-9,
    max_iter: int = 200,
    t_ref: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
  """Allocation of :func:`solve_budget_allocation`.

  ``max_T sum_t fbar(T_t; alpha) * theta_t`` subject to ``sum_t T_t <= K``
  and ``0 <= T_t <= T_bar`` (``t_max`` may also be a per-unit cap vector).
  ``fbar`` is concave and increasing for all supported families, therefore
  the problem is a separable concave program whose global optimum is
  characterised by a single shadow price ``mu``.

  ``t_ref`` selects the normalized objective
  (``fbar = f / f(T_ref; alpha)``).  Because Phases 1-3 estimate ``theta`` on
  ``fbar``, ``theta fbar = theta_raw f`` and the allocation is unchanged by
  the normalization; ``t_ref=None`` keeps the pre-normalization solver.

  Args:
    family: One of ``SUPPORTED_FAMILIES``.
    alpha: The (fixed) saturation parameter ``alpha_hat``. Either a scalar or
      a per-row array broadcastable to ``theta``.
    theta: Per-period effect amplitudes; non-positive entries receive ``0``.
    budget: The budget upper bound ``K``.
    t_max: The cap ``T_bar`` (scalar, or one value per unit).
    tol: Relative tolerance on the budget constraint.
    max_iter: Maximum number of bisection iterations.
    t_ref: ``T_ref`` on the scale of ``t_max`` (scalar or per-row), or
      ``None`` for the raw ``f``.

  Returns:
    The optimal allocation ``T*`` (same shape as ``theta``).
  """
  theta = np.asarray(theta, dtype=float)
  caps = np.broadcast_to(np.asarray(t_max, dtype=float), theta.shape)
  if float(budget) <= 0 or not np.any(caps > 0) or not np.any(theta > 0):
    return np.zeros_like(theta)
  return solve_budget_allocation(
      family, alpha, theta, float(budget), caps, t_ref=t_ref, tol=tol,
      max_iter=max(int(max_iter), 400),
  ).allocation


# ---------------------------------------------------------------------------
# Stage B nuisance learners (fixed defaults; spec §4.1 [v4-H]).
#
# Everything below is the ML side of Phase 0 / Stage B: the learners that
# estimate E[Y | X] (l_hat) and E[b_j(T) | X] (mu_hat_j) for the T-basis of
# :class:`TreatmentBasis`.  They do not depend on the saturation function
# itself; they live here because Stage B is their only consumer.
# ---------------------------------------------------------------------------

# Default hyperparameters shared by the Y- and T-model.
DEFAULT_TREE_PARAMS: Dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
}
DEFAULT_EARLY_STOPPING_ROUNDS = 30
# Cluster-level 10% internal validation split.
INTERNAL_VALIDATION_FRACTION = 0.10
DEFAULT_SMALL_SAMPLE_THRESHOLD = 2000
# RidgeCV feature construction.
FOURIER_ORDER = 3
SPLINE_DF = 5
SPLINE_DEGREE = 3
MIN_XGBOOST_VERSION = (2, 0)


def default_learner_config() -> Dict[str, Any]:
  """Returns the ``alpha_selection.learners`` defaults."""
  return {
      "learner_selection": False,
      "y_model": {
          "class": "xgboost.XGBRegressor",
          "params": dict(DEFAULT_TREE_PARAMS),
          "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
      },
      "t_model": {
          "class": "xgboost.XGBRegressor",
          "params": {
              "multi_strategy": "multi_output_tree",
              **DEFAULT_TREE_PARAMS,
          },
          "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
      },
      "small_sample_threshold": DEFAULT_SMALL_SAMPLE_THRESHOLD,
      "small_sample_model": "ridge_cv",
  }


def resolve_learner_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  """Merges a user ``learners`` block onto the defaults.

  Args:
    cfg: The ``alpha_selection.learners`` block (may be ``None``).

  Returns:
    The fully populated learner configuration.
  """
  merged = default_learner_config()
  cfg = dict(cfg or {})
  for key in ("learner_selection", "small_sample_threshold",
              "small_sample_model"):
    if key in cfg and cfg[key] is not None:
      merged[key] = cfg[key]
  for side in ("y_model", "t_model"):
    user = dict(cfg.get(side) or {})
    if not user:
      continue
    base = merged[side]
    if "class" in user and user["class"]:
      base["class"] = str(user["class"])
      if base["class"] != "xgboost.XGBRegressor":
        # A different class does not inherit XGBoost-specific parameters.
        base["params"] = {}
    if "params" in user and user["params"] is not None:
      base["params"] = {**base["params"], **dict(user["params"])}
    if "early_stopping_rounds" in user:
      base["early_stopping_rounds"] = user["early_stopping_rounds"]
  merged["learner_selection"] = bool(merged["learner_selection"])
  merged["small_sample_threshold"] = int(merged["small_sample_threshold"])
  return merged


def xgboost_status() -> Tuple[Optional[Any], str]:
  """Lazily imports XGBoost and checks the >= 2.0 requirement.

  Returns:
    ``(module_or_None, reason)``; ``reason`` is empty when usable.
  """
  try:
    xgb = importlib.import_module("xgboost")
  except ImportError as exc:  # pragma: no cover - environment dependent.
    return None, f"xgboost is not importable ({exc})"
  version = str(getattr(xgb, "__version__", "0"))
  parts = []
  for token in version.split(".")[:2]:
    digits = "".join(ch for ch in token if ch.isdigit())
    parts.append(int(digits) if digits else 0)
  while len(parts) < 2:
    parts.append(0)
  if tuple(parts) < MIN_XGBOOST_VERSION:
    return None, (
        f"xgboost {version} < 2.0 (multi_strategy='multi_output_tree'"
        " unavailable)"
    )
  return xgb, ""


def cluster_validation_split(
    groups: Optional[np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
    fraction: float = INTERNAL_VALIDATION_FRACTION,
) -> np.ndarray:
  """Boolean validation mask for early stopping.

  Whole clusters are held out (cluster-level 10%); falls back to an
  observation-level split when fewer than two clusters are available.

  Args:
    groups: Cluster label per training row (or ``None``).
    n_rows: Number of training rows.
    rng: Random generator.
    fraction: Validation share.

  Returns:
    A boolean mask of length ``n_rows`` (True = validation).
  """
  mask = np.zeros(n_rows, dtype=bool)
  if n_rows < 4:
    return mask
  if groups is not None:
    labels = pd.unique(pd.Series(np.asarray(groups)))
    if labels.size >= 2:
      n_val = int(
          min(labels.size - 1, max(1, math.ceil(fraction * labels.size)))
      )
      chosen = set(rng.choice(labels, size=n_val, replace=False).tolist())
      mask = np.isin(np.asarray(groups), list(chosen))
      if 0 < mask.sum() < n_rows:
        return mask
  n_val = max(1, int(round(fraction * n_rows)))
  mask[rng.choice(n_rows, size=n_val, replace=False)] = True
  return mask


class StageBLearner:
  """Interface of a Stage B nuisance learner."""

  kind: str = "base"

  def fit_predict(
      self,
      x_train: np.ndarray,
      y_train: np.ndarray,
      groups_train: Optional[np.ndarray],
      x_all: np.ndarray,
      seed: int,
  ) -> np.ndarray:
    """Fits on the training rows and predicts every row of ``x_all``."""
    raise NotImplementedError

  def describe(self) -> Dict[str, Any]:
    """Class name and main hyperparameters (``learners`` JSON key)."""
    raise NotImplementedError


class XGBoostLearner(StageBLearner):
  """XGBoost >= 2.0 learner; multi-output via ``multi_output_tree``."""

  kind = "xgboost"

  def __init__(
      self,
      xgb_module: Any,
      params: Dict[str, Any],
      early_stopping_rounds: Optional[int],
      multi_output: bool,
  ):
    self._xgb = xgb_module
    self.params = dict(params)
    if not multi_output:
      self.params.pop("multi_strategy", None)
    else:
      self.params.setdefault("multi_strategy", "multi_output_tree")
      self.params.setdefault("tree_method", "hist")
    self.early_stopping_rounds = early_stopping_rounds
    self.multi_output = multi_output

  def fit_predict(self, x_train, y_train, groups_train, x_all, seed):
    rng = np.random.default_rng(seed)
    params = dict(self.params)
    params.setdefault("random_state", int(seed))
    use_es = bool(self.early_stopping_rounds)
    val = (
        cluster_validation_split(groups_train, len(x_train), rng)
        if use_es
        else np.zeros(len(x_train), dtype=bool)
    )
    if use_es and val.any():
      params["early_stopping_rounds"] = int(self.early_stopping_rounds)
      model = self._xgb.XGBRegressor(**params)
      model.fit(
          x_train[~val],
          y_train[~val],
          eval_set=[(x_train[val], y_train[val])],
          verbose=False,
      )
    else:
      model = self._xgb.XGBRegressor(**params)
      model.fit(x_train, y_train, verbose=False)
    return np.asarray(model.predict(x_all), dtype=float)

  def describe(self):
    return {
        "class": "xgboost.XGBRegressor",
        "params": dict(self.params),
        "early_stopping_rounds": self.early_stopping_rounds,
    }


class SklearnBoostingFallback(StageBLearner):
  """Y-model stand-in when XGBoost >= 2.0 is unavailable.

  ``HistGradientBoostingRegressor`` with the default hyperparameters mapped
  one-to-one (depth, learning rate, rounds, leaf size, L2) and the SAME
  cluster-level 10% early stopping (patience applied on staged predictions).
  """

  kind = "sklearn_hgb"

  def __init__(self, params: Dict[str, Any], early_stopping_rounds, reason):
    self.params = {
        "max_depth": int(params.get("max_depth", 5)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "max_iter": int(params.get("n_estimators", 500)),
        "min_samples_leaf": int(params.get("min_child_weight", 10)),
        "l2_regularization": float(params.get("reg_lambda", 5)),
    }
    self.early_stopping_rounds = early_stopping_rounds
    self.reason = reason

  def fit_predict(self, x_train, y_train, groups_train, x_all, seed):
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    val = (
        cluster_validation_split(groups_train, len(x_train), rng)
        if self.early_stopping_rounds
        else np.zeros(len(x_train), dtype=bool)
    )
    model = ensemble.HistGradientBoostingRegressor(
        early_stopping=False, random_state=int(seed), **self.params
    )
    if not val.any():
      model.fit(x_train, y_train)
      return np.asarray(model.predict(x_all), dtype=float)
    model.fit(x_train[~val], y_train[~val])
    patience = int(self.early_stopping_rounds)
    best_iter, best_loss, since = 0, float("inf"), 0
    for it, pred in enumerate(model.staged_predict(x_train[val])):
      loss = float(np.mean((pred - y_train[val]) ** 2))
      if loss < best_loss - 1e-12:
        best_iter, best_loss, since = it, loss, 0
      else:
        since += 1
        if since >= patience:
          break
    preds = None
    for it, pred in enumerate(model.staged_predict(x_all)):
      preds = pred
      if it >= best_iter:
        break
    return np.asarray(preds, dtype=float)

  def describe(self):
    return {
        "class": "sklearn.ensemble.HistGradientBoostingRegressor",
        "params": dict(self.params),
        "early_stopping_rounds": self.early_stopping_rounds,
        "fallback_reason": self.reason,
    }


class SklearnMultiOutputForestFallback(StageBLearner):
  """T-model stand-in: ONE native multi-output tree ensemble (J outputs)."""

  kind = "sklearn_rf"

  def __init__(self, params: Dict[str, Any], reason: str):
    self.params = {
        "n_estimators": 200,
        "max_depth": int(params.get("max_depth", 5)) * 2,
        "min_samples_leaf": int(params.get("min_child_weight", 10)),
        "max_features": float(params.get("colsample_bytree", 0.8)),
        "max_samples": float(params.get("subsample", 0.8)),
    }
    self.reason = reason

  def fit_predict(self, x_train, y_train, groups_train, x_all, seed):
    del groups_train
    model = ensemble.RandomForestRegressor(
        random_state=int(seed), n_jobs=1, **self.params
    )
    model.fit(x_train, np.asarray(y_train, dtype=float))
    preds = np.asarray(model.predict(x_all), dtype=float)
    return preds.reshape(len(x_all), -1) if preds.ndim == 1 else preds

  def describe(self):
    return {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": dict(self.params),
        "fallback_reason": self.reason,
    }


class GenericSklearnLearner(StageBLearner):
  """User-configured ``module.Class`` estimator (config override)."""

  kind = "generic"

  def __init__(self, class_path: str, params: Dict[str, Any]):
    module_name, _, class_name = class_path.rpartition(".")
    self._cls = getattr(importlib.import_module(module_name), class_name)
    self.class_path = class_path
    self.params = dict(params)

  def fit_predict(self, x_train, y_train, groups_train, x_all, seed):
    del groups_train, seed
    model = self._cls(**self.params)
    model.fit(x_train, np.asarray(y_train, dtype=float))
    return np.asarray(model.predict(x_all), dtype=float)

  def describe(self):
    return {"class": self.class_path, "params": dict(self.params)}


class RidgeCVLearner(StageBLearner):
  """Small-sample switch: standardised RidgeCV.

  ``x`` must be the matrix of :func:`build_small_sample_features`; cubic
  B-splines (5 df) are fitted *inside* the training fold on the continuous
  numeric covariates (column indices ``spline_cols``).  ``RidgeCV`` handles
  the J-output target natively (one model).
  """

  kind = "ridge_cv"

  def __init__(self, spline_cols: Sequence[int], n_features: int):
    self.spline_cols = [int(c) for c in spline_cols]
    self.n_features = int(n_features)
    self.alphas = np.logspace(-3, 3, 13)

  def _make(self):
    """Builds a fresh, unfitted spline -> scaler -> RidgeCV pipeline.

    B-splines are added for ``spline_cols`` next to a passthrough of all raw
    columns; building it per fold keeps the knots inside the training data.

    Returns:
      An unfitted ``sklearn.pipeline.Pipeline``.
    """
    transformers = []
    if self.spline_cols:
      transformers.append((
          "spline",
          preprocessing.SplineTransformer(
              n_knots=SPLINE_DF - SPLINE_DEGREE + 1,
              degree=SPLINE_DEGREE,
              knots="quantile",
              extrapolation="linear",
          ),
          self.spline_cols,
      ))
    transformers.append(("raw", "passthrough", list(range(self.n_features))))
    return pipeline.make_pipeline(
        compose.ColumnTransformer(transformers),
        preprocessing.StandardScaler(),
        linear_model.RidgeCV(alphas=self.alphas),
    )

  def fit_predict(self, x_train, y_train, groups_train, x_all, seed):
    del groups_train, seed
    model = self._make()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      model.fit(x_train, np.asarray(y_train, dtype=float))
    return np.asarray(model.predict(x_all), dtype=float)

  def describe(self):
    return {
        "class": "sklearn.linear_model.RidgeCV",
        "params": {
            "alphas": [float(a) for a in self.alphas],
            "features": (
                "one-hot(item_col, geo_col) + annual Fourier (order"
                f" {FOURIER_ORDER}) + numeric covariates + cubic B-splines"
                f" ({SPLINE_DF} df)"
            ),
        },
    }


def build_small_sample_features(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    item_col: Optional[str] = None,
    geo_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
  """Feature matrix of the RidgeCV small-sample switch.

  Columns: numeric covariates (raw), one-hot(``item_col``, ``geo_col``) and
  annual Fourier terms ``sin/cos(2 pi k doy / 365.25)``, ``k = 1..3``.  The
  cubic B-splines (5 df) of the continuous covariates are added inside the
  learner so that their knots are fitted within the training fold.

  Args:
    df: The frame (D_sel), raw or normalised.
    x_cols: Nuisance covariates.
    item_col: Item identifier (optional).
    geo_col: Geo identifier (optional).
    date_col: Date column (optional).

  Returns:
    ``(matrix, spline_col_indices, column_names)``.
  """
  blocks: List[np.ndarray] = []
  names: List[str] = []
  spline_cols: List[int] = []
  numeric = df[list(x_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
  for j, col in enumerate(x_cols):
    if int(numeric[col].nunique()) > SPLINE_DF + 1:
      spline_cols.append(j)
  blocks.append(numeric.to_numpy(dtype=float))
  names.extend(list(x_cols))
  for col in (item_col, geo_col):
    if col and col in df.columns:
      dummies = pd.get_dummies(df[col].astype(str), prefix=col, dtype=float)
      blocks.append(dummies.to_numpy(dtype=float))
      names.extend(list(dummies.columns))
  if date_col and date_col in df.columns:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    doy = dates.dt.dayofyear.fillna(1).to_numpy(dtype=float)
    phase = 2.0 * np.pi * doy / 365.25
    for k in range(1, FOURIER_ORDER + 1):
      blocks.append(np.column_stack([np.sin(k * phase), np.cos(k * phase)]))
      names.extend([f"fourier_sin_{k}", f"fourier_cos_{k}"])
  matrix = np.column_stack(blocks) if blocks else np.zeros((len(df), 0))
  return matrix, spline_cols, names


@dataclasses.dataclass
class StageBLearners:
  """The resolved pair of Stage B learners."""

  y_learner: StageBLearner
  t_learner: StageBLearner
  small_sample: bool
  uses_small_sample_features: bool
  messages: List[str] = dataclasses.field(default_factory=list)

  def describe(self) -> Dict[str, Any]:
    return {
        "y_model": self.y_learner.describe(),
        "t_model": self.t_learner.describe(),
        "small_sample_switch": self.small_sample,
    }


def _tree_learner(
    side_cfg: Dict[str, Any], multi_output: bool
) -> Tuple[StageBLearner, Optional[str]]:
  """Builds the configured tree learner, falling back when needed."""
  class_path = str(side_cfg.get("class", "xgboost.XGBRegressor"))
  params = copy.deepcopy(dict(side_cfg.get("params") or {}))
  esr = side_cfg.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
  if class_path == "xgboost.XGBRegressor":
    xgb, reason = xgboost_status()
    if xgb is not None:
      return XGBoostLearner(xgb, params, esr, multi_output), None
    if multi_output:
      return SklearnMultiOutputForestFallback(params, reason), reason
    return SklearnBoostingFallback(params, esr, reason), reason
  params.pop("multi_strategy", None)
  return GenericSklearnLearner(class_path, params), None


def resolve_stage_b_learners(
    learner_cfg: Dict[str, Any],
    n_sel: int,
    small_sample_spline_cols: Sequence[int] = (),
    small_sample_n_features: int = 0,
) -> StageBLearners:
  """Resolves the fixed default learners.

  Args:
    learner_cfg: Output of :func:`resolve_learner_config`.
    n_sel: Rows in ``D_sel``.
    small_sample_spline_cols: Spline column indices of the RidgeCV features.
    small_sample_n_features: Width of the RidgeCV feature matrix.

  Returns:
    The :class:`StageBLearners`.
  """
  threshold = int(learner_cfg.get("small_sample_threshold", 2000))
  model = str(learner_cfg.get("small_sample_model", "ridge_cv")).lower()
  if n_sel < threshold and model in ("ridge_cv", "ridgecv", "ridge"):
    ridge = RidgeCVLearner(small_sample_spline_cols, small_sample_n_features)
    return StageBLearners(
        y_learner=ridge,
        t_learner=RidgeCVLearner(
            small_sample_spline_cols, small_sample_n_features
        ),
        small_sample=True,
        uses_small_sample_features=True,
        messages=[
            f"Stage B learners: n_sel={n_sel} < small_sample_threshold="
            f"{threshold} -> RidgeCV for both nuisances."
        ],
    )
  messages: List[str] = []
  y_learner, y_reason = _tree_learner(learner_cfg["y_model"], False)
  t_learner, t_reason = _tree_learner(learner_cfg["t_model"], True)
  for side, reason in (("Y", y_reason), ("T", t_reason)):
    if reason:
      msg = (
          f"[Phase 0 / Stage B] XGBoost >= 2.0 unavailable for the {side}-model"
          f" ({reason}); using the scikit-learn stand-in"
          f" '{(y_learner if side == 'Y' else t_learner).describe()['class']}'"
          " with matching hyperparameters."
      )
      messages.append(msg)
      warnings.warn(msg, UserWarning)
  return StageBLearners(
      y_learner=y_learner,
      t_learner=t_learner,
      small_sample=False,
      uses_small_sample_features=False,
      messages=messages,
  )
