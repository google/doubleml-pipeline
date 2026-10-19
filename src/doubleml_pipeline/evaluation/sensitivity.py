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

"""Utilities for the sensitivity analysis of the estimated treatment effects.

The OVB bias bound and the
robustness value shown by ``plots.plot_sensitivity_contour`` and written to
``sensitivity_metrics.csv`` are computed here, with the following names:

* ``D``: treatment variable of the PLR (``fbar(T; alpha_hat)`` in the
  saturation version, ``T`` in the linear version).
* ``n``: number of observations used in the estimation (``n_obs``).
* ``t_star_alpha``: two-sided critical value (1.96 at alpha = 0.05).
* ``u = Y - l_hat``, ``v = D - m_hat``: the out-of-fold nuisance residuals.
* ``sigma2 = E[(u - theta v)^2]``, ``nu2 = 1 / E[v^2]`` -- DoubleML's
  ``sensitivity_elements`` for the ATE; for a GATE group ``g`` the same two
  moments over the group's observations with ``theta_g``.
* ``scale = sqrt(sigma2 nu2)``: the residual-based scale of the bias bound (a
  population quantity about the point estimate).  The cluster-robust ``se``
  enters the sampling-error term only; ``se sqrt(n)`` is NOT a substitute
  for ``scale`` (it carries the cluster inflation).
* ``cf_y``, ``cf_d``: partial R^2 of the unobserved confounder U with the Y
  residual / the D residual; ``rho_U``: their correlation.

    B(cf_y, cf_d) = rho_U scale sqrt(cf_y cf_d / (1 - cf_d))
    contour value  = |theta_hat| - B(cf_y, cf_d) - t_star_alpha se_hat
    m   = |theta_hat| - t_star_alpha se_hat      (m <= 0  =>  RV = 0)
    f_q = m / scale
    RV  = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2

so that the contour value at ``cf_y = cf_d = RV`` is exactly 0 (the RV lies
on the zero contour by construction).  The substitution of ``scale`` by
``|theta_hat|`` or by ``se sqrt(n)``, and the arguments ``k = (|t| - t*)/|t|``
or ``f = (|t| - t*)/sqrt(n)``, are prohibited.
"""

import copy
import dataclasses
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import doubleml as dml
import numpy as np
import pandas as pd

# t*_alpha, the two-sided critical value at alpha = 0.05.
T_STAR_ALPHA = 1.96
# The `treatment_variable` column of
# sensitivity_metrics.csv -- D = fbar(T; alpha_hat) or D = T.
TREATMENT_VARIABLE_FBAR = 'f_bar(T)'
TREATMENT_VARIABLE_T = 'T'
# Where sigma2 / nu2 (hence `scale`) came from.
SCALE_SOURCE_ELEMENTS = 'sensitivity_elements'
SCALE_SOURCE_RESIDUALS = 'residuals'
SCALE_SOURCE_RESIDUALS_GROUP = 'residuals_group'
# The `benchmark_scope` column of
# sensitivity_metrics.csv -- the reference population of the benchmark point.
BENCHMARK_SCOPE_GROUP = 'group'
BENCHMARK_SCOPE_WHOLE_SAMPLE = 'whole_sample'


@dataclasses.dataclass(frozen=True)
class BenchmarkStats:
  """The group-specific sensitivity benchmark.

  Attributes:
    cf_d: ``cf_{d,g}`` -- benchmark partial R^2 on the D side.
    cf_y: ``cf_{y,g}`` -- benchmark partial R^2 on the Y side.
    rho: ``rho_{U,g}`` -- signed benchmark correlation, clipped to [-1, 1].
    delta_theta: ``theta_short_g - theta_long_g``.
    sigma2_long: ``sigma2_g`` of the long model (with Z).
    nu2_long: ``nu2_g`` of the long model.
    sigma2_short: ``sigma2^s_g`` of the short model (without Z).
    nu2_short: ``nu2^s_g`` of the short model.
    n_obs: ``n_g``.
    scope: ``"group"`` or ``"whole_sample"``.
  """

  cf_d: float
  cf_y: float
  rho: float
  delta_theta: float
  sigma2_long: float
  nu2_long: float
  sigma2_short: float
  nu2_short: float
  n_obs: int
  scope: str

  def as_record(self) -> Dict[str, Any]:
    """The ``sensitivity_metrics.csv`` benchmark fields."""
    return {
        'benchmark_cf_d': self.cf_d,
        'benchmark_cf_y': self.cf_y,
        'benchmark_rho': self.rho,
        'benchmark_scope': self.scope,
    }


def fit_short_model(dml_model: Any, benchmark_covariates: List[str]) -> Any:
  """Fits the SHORT model (without the benchmark covariate ``Z``) once.

  The short model uses the same learners (the same
  ``theta*`` under ``hparam_policy = "global"``), the same folds and the same
  seed as the long model -- the fitted long ``DoubleMLPLR`` is deep-copied
  (which keeps its sample splitting ``smpls``) and refitted with ``Z`` removed
  from ``x_cols``.  This is exactly what ``DoubleML.sensitivity_benchmark``
  does internally, done ONCE per treatment instead of once per contour.

  Args:
    dml_model: The fitted long ``DoubleMLPLR`` (Phase 3 ``model_obj``).
    benchmark_covariates: ``[Z]`` -- must be a subset of the long ``x_cols``.

  Returns:
    The fitted short ``DoubleMLPLR`` (``store_predictions=True``).

  Raises:
    ValueError: If ``Z`` is not a long covariate or nothing would remain.
  """
  x_long = list(dml_model._dml_data.x_cols)  # pylint: disable=protected-access
  missing = [z for z in benchmark_covariates if z not in x_long]
  if missing:
    raise ValueError(
        f'benchmark covariates {missing} are not among the model covariates'
        f' {x_long}.'
    )
  x_short = [x for x in x_long if x not in benchmark_covariates]
  if not x_short:
    raise ValueError(
        'The short model needs at least one remaining covariate; removing'
        f' {list(benchmark_covariates)} leaves none.'
    )
  dml_short = copy.deepcopy(dml_model)
  dml_short._dml_data.x_cols = x_short  # pylint: disable=protected-access
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dml_short.fit()
  return dml_short


def benchmark_from_moments(
    sigma2_long: float,
    nu2_long: float,
    sigma2_short: float,
    nu2_short: float,
    delta_theta: float,
) -> Tuple[float, float, float]:
  """``(cf_d, cf_y, rho)`` from long / short moments.

  The algebra is DoubleML's ``utils.gain_statistics`` (the function behind
  ``sensitivity_benchmark``) applied to the given moments, so that for the
  whole sample the values coincide with ``sensitivity_benchmark`` (acceptance
  check: relative error <= 1e-6):

  * ``cf_y = clip((R2_long - R2_short) / (1 - R2_long), 0, 1)`` with
    ``R2 = 1 - sigma2 / var(Y)``, i.e. ``(sigma2_short - sigma2_long) /
    sigma2_long`` (``var(Y)`` cancels);
  * ``cf_d = clip((1 - R2_riesz) / R2_riesz, 0, 1)`` with ``R2_riesz =
    nu2_short / nu2_long``;
  * ``rho = sign(delta_theta) clip(|delta_theta| / sqrt((sigma2_short -
    sigma2_long)(nu2_long - nu2_short)), 0, 1)``; DoubleML's convention
    ``rho = sign(delta_theta)`` is kept when that denominator is 0.

  Args:
    sigma2_long: ``sigma2_g`` of the long model.
    nu2_long: ``nu2_g`` of the long model.
    sigma2_short: ``sigma2^s_g`` of the short model.
    nu2_short: ``nu2^s_g`` of the short model.
    delta_theta: ``theta^s_g - theta_g``.

  Returns:
    ``(cf_d, cf_y, rho)``.
  """
  sigma2_long = float(sigma2_long)
  nu2_long = float(nu2_long)
  sigma2_short = float(sigma2_short)
  nu2_short = float(nu2_short)
  delta_theta = float(delta_theta)

  cf_y = float(np.clip((sigma2_short - sigma2_long) / sigma2_long, 0.0, 1.0))
  r2_riesz = nu2_short / nu2_long
  cf_d = float(np.clip((1.0 - r2_riesz) / r2_riesz, 0.0, 1.0))

  var_g = sigma2_short - sigma2_long
  var_riesz = nu2_long - nu2_short
  denom = (
      math.sqrt(var_g * var_riesz) if (var_g > 0.0 and var_riesz > 0.0) else 0.0
  )
  rho_abs = abs(delta_theta) / denom if denom != 0.0 else 1.0
  rho = float(np.sign(delta_theta)) * float(np.clip(rho_abs, 0.0, 1.0))
  return cf_d, cf_y, rho


def group_benchmark(
    u_long: np.ndarray,
    v_long: np.ndarray,
    u_short: np.ndarray,
    v_short: np.ndarray,
    theta_long: float,
    theta_short: float,
    mask: Optional[np.ndarray] = None,
) -> BenchmarkStats:
  """The group benchmark ``(cf_{d,g}, cf_{y,g}, rho_{U,g})``.

  Group moments over the rows in ``mask`` (all rows when None) --
  ``sigma2_g = mean_g((u - theta_g v)^2)``, ``nu2_g = 1 / mean_g(v^2)`` for the
  long residuals with ``theta_g``, likewise for the short residuals with
  ``theta^s_g`` -- then :func:`benchmark_from_moments`.

  Args:
    u_long: ``Y - l_hat`` of the long model, ``(n_obs, n_rep)``.
    v_long: ``D - m_hat`` of the long model.
    u_short: ``Y - l_hat^s`` of the short model.
    v_short: ``D - m_hat^s`` of the short model.
    theta_long: ``theta_g`` (GATE of the long model; the ATE for the total).
    theta_short: ``theta^s_g`` (GATE of the short model).
    mask: Boolean row selector of the group ``g``; None = whole sample.

  Returns:
    The :class:`BenchmarkStats` (``scope`` reflects whether a proper subset
    was selected).
  """
  _, sigma2_l, nu2_l = residual_scale(u_long, v_long, theta_long, mask=mask)
  _, sigma2_s, nu2_s = residual_scale(u_short, v_short, theta_short, mask=mask)
  delta = float(theta_short) - float(theta_long)
  cf_d, cf_y, rho = benchmark_from_moments(
      sigma2_l, nu2_l, sigma2_s, nu2_s, delta
  )
  n_total = int(np.asarray(u_long).shape[0])
  if mask is None:
    n_obs = n_total
  else:
    n_obs = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
  scope = (
      BENCHMARK_SCOPE_WHOLE_SAMPLE
      if n_obs == n_total
      else BENCHMARK_SCOPE_GROUP
  )
  return BenchmarkStats(
      cf_d=cf_d,
      cf_y=cf_y,
      rho=rho,
      delta_theta=delta,
      sigma2_long=float(sigma2_l),
      nu2_long=float(nu2_l),
      sigma2_short=float(sigma2_s),
      nu2_short=float(nu2_s),
      n_obs=n_obs,
      scope=scope,
  )


@dataclasses.dataclass(frozen=True)
class SensitivityStats:
  """The quantities behind one sensitivity contour.

  Attributes:
    theta: ``theta_hat`` (ATE, or ``theta_g`` for a GATE group).
    se: ``se_hat`` (cluster-robust; ``se_g`` for a GATE group).
    n_obs: ``n`` (``n_g`` for a GATE group).
    t_stat: ``theta_hat / se_hat``.
    f_q: ``m / scale`` with ``m = |theta_hat| - t_star_alpha se_hat``; 0 when
      ``m <= 0``.
    robustness_value: ``RV`` (q = 1, alpha = 0.05).
    sigma2: ``E[(u - theta v)^2]`` (group moment for a GATE row).
    nu2: ``1 / E[v^2]`` (group moment for a GATE row).
    scale: ``sqrt(sigma2 nu2)``.
    scale_source: ``"sensitivity_elements"``, ``"residuals"`` or
      ``"residuals_group"``.
  """

  theta: float
  se: float
  n_obs: int
  t_stat: float
  f_q: float
  robustness_value: float
  sigma2: float
  nu2: float
  scale: float
  scale_source: str

  def as_record(self) -> Dict[str, Any]:
    """The ``sensitivity_metrics.csv`` fields of this row."""
    return {
        'theta': self.theta,
        'se': self.se,
        'robustness_value': self.robustness_value,
        'n_obs': self.n_obs,
        'f_q': self.f_q,
        'sigma2': self.sigma2,
        'nu2': self.nu2,
        'scale': self.scale,
    }


def robustness_value(
    theta: float, se: float, scale: float, t_star: float = T_STAR_ALPHA
) -> Tuple[float, float]:
  """Returns the robustness value ``RV`` and ``f_q``.

  ``m = |theta_hat| - t_star_alpha se_hat``; ``RV = f_q = 0`` when ``m <= 0``;
  otherwise ``f_q = m / scale`` and ``RV = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) /
  2``, i.e. the solution of ``scale sqrt(RV^2 / (1 - RV)) = m`` -- the point
  ``cf_y = cf_d = RV`` lies on the zero contour of ``|theta_hat| - B -
  t_star_alpha se_hat``.

  Args:
    theta: ``theta_hat``.
    se: ``se_hat`` (cluster-robust; strictly positive).
    scale: ``sqrt(sigma2 nu2)`` (strictly positive).
    t_star: ``t_star_alpha`` (1.96).

  Returns:
    The pair ``(RV, f_q)``.

  Raises:
    ValueError: If ``se`` or ``scale`` is not strictly positive / finite.
  """
  se = float(se)
  theta = float(theta)
  scale = float(scale)
  if not math.isfinite(se) or se <= 0.0:
    raise ValueError(f'se must be a positive finite number, got {se!r}.')
  if not math.isfinite(scale) or scale <= 0.0:
    raise ValueError(
        f'scale = sqrt(sigma2 nu2) must be a positive finite number, got'
        f' {scale!r}.'
    )
  m = abs(theta) - t_star * se
  if not m > 0.0:
    return 0.0, 0.0
  f_q = m / scale
  rv = 0.5 * (math.sqrt(f_q**4 + 4.0 * f_q**2) - f_q**2)
  return float(rv), float(f_q)


def bias_bound(cf_y, cf_d, scale: float, rho: float = 1.0):
  """``B(cf_y, cf_d) = rho_U scale sqrt(cf_y cf_d / (1 - cf_d))``.

  Args:
    cf_y: Partial R^2 of U with the Y residual (scalar or array).
    cf_d: Partial R^2 of U with the D residual (scalar or array, ``< 1``).
    scale: ``sqrt(sigma2 nu2)``.
    rho: ``rho_U`` (the adversarial default is 1).

  Returns:
    The bound, broadcast over the inputs.
  """
  cf_y = np.asarray(cf_y, dtype=float)
  cf_d = np.asarray(cf_d, dtype=float)
  denominator = np.clip(1.0 - cf_d, 1e-12, None)
  return float(rho) * float(scale) * np.sqrt(cf_y * cf_d / denominator)


def sigma2_nu2(
    dml_model: Any, i_treat: int = 0
) -> Tuple[Optional[float], Optional[float]]:
  """Reads ``sigma2`` / ``nu2`` from ``dml_model.sensitivity_elements``.

  DoubleML stores them with shape ``(1, n_rep, n_coefs)``; the median over
  the repetitions is taken (DoubleML aggregates ``theta`` / ``se`` the same
  way).

  Args:
    dml_model: A fitted ``DoubleMLPLR`` (or anything exposing
      ``sensitivity_elements``).
    i_treat: Treatment index.

  Returns:
    ``(sigma2, nu2)``; a component is None when it is unavailable or not a
    positive finite number.
  """
  elements = getattr(dml_model, 'sensitivity_elements', None)
  if not isinstance(elements, dict):
    return None, None

  def _read(key: str) -> Optional[float]:
    try:
      values = np.asarray(elements.get(key), dtype=float)
    except (TypeError, ValueError):
      return None
    if values.size == 0:
      return None
    if values.ndim == 3:
      values = values[0, :, i_treat]
    elif values.ndim == 2:
      values = values[:, i_treat]
    value = float(np.median(values.reshape(-1)))
    if not math.isfinite(value) or value <= 0.0:
      return None
    return value

  return _read('sigma2'), _read('nu2')


def _nuisance_array(dml_model: Any, store: str, learner: str,
                    i_treat: int) -> Optional[np.ndarray]:
  """Returns ``(n_obs, n_rep)`` slice of a prediction / target store."""
  container = getattr(dml_model, store, None)
  if not isinstance(container, dict) or container.get(learner) is None:
    return None
  values = np.asarray(container[learner], dtype=float)
  if values.ndim == 3:
    values = values[:, :, i_treat]
  elif values.ndim == 1:
    values = values[:, None]
  return values


def residuals(
    dml_model: Any, i_treat: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
  """Out-of-fold residuals ``u = Y - l_hat`` and ``v = D - m_hat``.

  Both are taken from ``dml_model.predictions`` (``ml_l``, ``ml_m``) with the
  nuisance targets ``Y`` / ``D`` from ``dml_model.nuisance_targets`` (the
  arrays DoubleML stores next to the predictions, in the model's row order).

  Args:
    dml_model: A fitted ``DoubleMLPLR`` (``fit(store_predictions=True)``,
      the default).
    i_treat: Treatment index.

  Returns:
    ``(u, v)``, each of shape ``(n_obs, n_rep)``.

  Raises:
    ValueError: If the predictions or the targets are unavailable.
  """
  l_hat = _nuisance_array(dml_model, 'predictions', 'ml_l', i_treat)
  m_hat = _nuisance_array(dml_model, 'predictions', 'ml_m', i_treat)
  y = _nuisance_array(dml_model, 'nuisance_targets', 'ml_l', i_treat)
  d = _nuisance_array(dml_model, 'nuisance_targets', 'ml_m', i_treat)
  if l_hat is None or m_hat is None:
    raise ValueError(
        'dml_model.predictions has no ml_l / ml_m entries; the PLR must be'
        ' fitted with store_predictions=True (partialling-out score).'
    )
  if y is None or d is None or not np.all(np.isfinite(y)) or not np.all(
      np.isfinite(d)
  ):
    # Older DoubleML versions do not keep the targets: fall back to the data
    # object, broadcast over the repetitions.
    data = getattr(dml_model, '_dml_data', None)
    if data is None:
      raise ValueError('The nuisance targets Y / D are unavailable.')
    y = np.broadcast_to(
        np.asarray(data.y, dtype=float)[:, None], l_hat.shape
    )
    d_all = np.asarray(data.d, dtype=float)
    if d_all.ndim == 2:
      d_all = d_all[:, i_treat]
    d = np.broadcast_to(d_all[:, None], m_hat.shape)
  return y - l_hat, d - m_hat


def residual_scale(
    u: np.ndarray,
    v: np.ndarray,
    theta: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
  """``sigma2 = mean((u - theta v)^2)``, ``nu2 = 1 / mean(v^2)``, ``scale``.

  The means run over the rows selected by ``mask`` (all rows when None); with
  several repetitions the per-repetition values are combined by the median,
  as ``sigma2_nu2`` does for ``sensitivity_elements``.

  Args:
    u: ``Y - l_hat``, shape ``(n_obs, n_rep)`` (or ``(n_obs,)``).
    v: ``D - m_hat``, same shape.
    theta: ``theta_hat`` (or ``theta_g`` for a group).
    mask: Boolean row selector of length ``n_obs`` (a GATE group).

  Returns:
    ``(scale, sigma2, nu2)``.

  Raises:
    ValueError: If no row is selected or the moments are not positive.
  """
  u = np.asarray(u, dtype=float)
  v = np.asarray(v, dtype=float)
  if u.ndim == 1:
    u = u[:, None]
  if v.ndim == 1:
    v = v[:, None]
  if mask is not None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != u.shape[0]:
      raise ValueError(
          f'mask has {mask.shape[0]} rows, the residuals {u.shape[0]}.'
      )
    u = u[mask]
    v = v[mask]
  if u.shape[0] == 0:
    raise ValueError('No observation selected for sigma2 / nu2.')
  sigma2 = float(np.median(np.mean((u - float(theta) * v) ** 2, axis=0)))
  mean_v2 = float(np.median(np.mean(v**2, axis=0)))
  if not (math.isfinite(sigma2) and sigma2 > 0.0) or not (
      math.isfinite(mean_v2) and mean_v2 > 0.0
  ):
    raise ValueError(
        f'Degenerate residual moments: sigma2 = {sigma2!r}, mean(v^2) ='
        f' {mean_v2!r}.'
    )
  nu2 = 1.0 / mean_v2
  return math.sqrt(sigma2 * nu2), sigma2, nu2


def sensitivity_stats(
    dml_model: Any,
    override_theta: Optional[float] = None,
    override_se: Optional[float] = None,
    override_n_obs: Optional[int] = None,
    group_mask: Optional[np.ndarray] = None,
) -> SensitivityStats:
  """Collects the sensitivity quantities (``theta_hat`` ... ``RV``) of a row.

  For the ATE row ``theta_hat``, ``se_hat``
  and ``n`` come from the fitted model and ``sigma2`` / ``nu2`` from
  ``dml_model.sensitivity_elements``; when the elements are unavailable they
  are recomputed from the out-of-fold residuals ``u = Y - l_hat``, ``v = D -
  m_hat``.  For a GATE row (geo / item) the caller passes the group's
  ``theta_g``, ``se_g`` and the row mask; ``sigma2_g = mean_g((u - theta_g
  v)^2)`` and ``nu2_g = 1 / mean_g(v^2)`` are the group's own moments.  The
  scale is always ``sqrt(sigma2 nu2)`` -- never ``se sqrt(n)``.

  Args:
    dml_model: The Phase 1-3 ``DoubleMLPLR`` (fitted with ``D = fbar(T;
      alpha_hat)`` or ``D = T``).
    override_theta: ``theta_g`` of a GATE group.
    override_se: ``se_g`` of a GATE group.
    override_n_obs: ``n_g`` of a GATE group (defaults to ``group_mask.sum()``).
    group_mask: Boolean mask over the fitted rows selecting the GATE group.

  Returns:
    A ``SensitivityStats``.

  Raises:
    ValueError: If a GATE override lacks ``group_mask`` or the moments cannot
      be computed.
  """
  is_gate = override_theta is not None or override_se is not None
  theta = (
      float(override_theta)
      if override_theta is not None
      else float(np.asarray(dml_model.coef).reshape(-1)[0])
  )
  se = (
      float(override_se)
      if override_se is not None
      else float(np.asarray(dml_model.se).reshape(-1)[0])
  )

  if is_gate or group_mask is not None:
    if group_mask is None:
      raise ValueError(
          'A GATE row needs group_mask: sigma2_g / nu2_g are the moments of'
          ' the group (se_g sqrt(n_g) is not a valid scale).'
      )
    mask = np.asarray(group_mask, dtype=bool)
    n_obs = int(
        override_n_obs if override_n_obs is not None else int(mask.sum())
    )
    u, v = residuals(dml_model)
    scale, sigma2, nu2 = residual_scale(u, v, theta, mask=mask)
    source = SCALE_SOURCE_RESIDUALS_GROUP
  else:
    n_obs = int(
        override_n_obs if override_n_obs is not None else dml_model.n_obs
    )
    sigma2, nu2 = sigma2_nu2(dml_model)
    if sigma2 is not None and nu2 is not None:
      scale = math.sqrt(sigma2 * nu2)
      source = SCALE_SOURCE_ELEMENTS
    else:
      u, v = residuals(dml_model)
      scale, sigma2, nu2 = residual_scale(u, v, theta)
      source = SCALE_SOURCE_RESIDUALS

  rv, f_q = robustness_value(theta, se, scale)
  return SensitivityStats(
      theta=theta,
      se=se,
      n_obs=n_obs,
      t_stat=theta / se,
      f_q=f_q,
      robustness_value=rv,
      sigma2=float(sigma2),
      nu2=float(nu2),
      scale=float(scale),
      scale_source=source,
  )


def perform_gate_and_sensitivity_analysis(
    dml_model: dml.DoubleMLPLR,
    df_orig: pd.DataFrame,
    gate_cols: List[str],
    benchmark_covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Performs GATE (group average treatment effect) and sensitivity checks."""
  benchmark_covariates = benchmark_covariates or []

  if len(gate_cols) == 1:
    group_series = df_orig[gate_cols[0]].astype(str)
  else:
    group_series = df_orig[gate_cols[0]].astype(str)
    for col in gate_cols[1:]:
      group_series = group_series + '_' + df_orig[col].astype(str)

  groups_df = pd.get_dummies(group_series)
  gate_est = dml_model.gate(groups=groups_df)

  dml_model.sensitivity_analysis(
      cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0
  )

  benchmarks_dict = {}
  for cov in benchmark_covariates:
    try:
      benchmarks = dml_model.sensitivity_benchmark(benchmarking_set=[cov])
      if isinstance(benchmarks, pd.DataFrame):
        bench_x = benchmarks['cf_d'].iloc[0]
        bench_y = benchmarks['cf_y'].iloc[0]
      else:
        bench_x = None
        bench_y = None
      benchmarks_dict[cov] = {'x': bench_x, 'y': bench_y}
    except (
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        RuntimeError,
    ) as err:
      benchmarks_dict[cov] = {'x': None, 'y': None, 'error': str(err)}

  return {
      'gate_summary': gate_est.summary,
      'sensitivity_elements': dml_model.sensitivity_elements,
      'benchmarks': benchmarks_dict,
      'groups_df': groups_df,
  }
