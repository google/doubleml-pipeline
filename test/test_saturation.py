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

"""Tests for the saturation core (``models.saturation``).

The response families and the alpha grids:

* The response families (``exp``, ``log``, ``power``, ``tanh``) and their
  derivatives; the round trip T_1/2 <-> alpha per family; the log grid
  variable ``T_knee = alpha``.
* The normalized saturation ``fbar`` and the reference point
  ``T_ref = q95(T | T > 0)``: ``fbar(0; alpha) = 0`` and ``fbar(T_ref; alpha)
  = 1`` for every family and every alpha of the automatic grid; ``fbar`` is
  invariant to unit changes of ``T``.
* Alpha grids: the manual grid builder, the documented defaults of
  docs/api_reference.md, the automatic half-saturation grid (25 log-spaced
  T_1/2 in [0.25 q50, 4 q95]), its x4 extension, the manual override and
  ``AlphaGrid.linear_side``.

The T-basis:

* ``c(alpha) = P fbar_alpha`` on the fixed projection grid (equally spaced
  points, quantiles of ``T > 0`` and the Greville abscissae) with the SVD
  projector precomputed once; ``project_values`` rejects observed-``T`` input;
  the projection error under ``P_T`` on the grid Phase 0 uses.
* Basis domain ``[0, T_max]`` with ``T_max = q99.9(T > 0)`` and interior knots
  at quantiles of ``T > 0``; Greville linear precision; heavy-tailed treatments
  keep ``cond(B)`` small and the coefficients bounded; the ill-conditioning
  guard.
* The automatic choice of J: the smallest J in ``[6, 20]`` with ``eps(J) <
  basis_tol``, the smallest-``eps`` J when the tolerance is unreachable.

The allocators and the normalization bridge:

* The closed-form / water-filling allocators, the semi-closed-form tanh
  allocator (KKT + brute force); the kappa-carrying KKT solver reproduces the
  pre-normalization allocation exactly (``theta_bar = kappa theta_raw``), also
  with a row-varying ``T_ref_i`` and under the ROI-threshold rule; the
  allocator accepts one alpha per row.
* The bridge ``f(T/c; alpha) == gain_i * f(T; alpha_i)`` holds to machine
  precision for every family, for a global constant ``c``
  (``median_normalize``) and for a row-varying ``c_i``
  (``population_median_normalize``).  If this identity ever drifts, the KPI
  reported by Phase 4 stops matching the KPI Phases 1-3 estimated.
* The profiled R-loss recovers a known alpha.

The budget bound, the theta recovery and the optimizer charts live in
test_optimization.py; ``treatment_divisor``, the Phase 0 fit accounting and
the FLAML time budget in test_alpha_selection.py; the default family
resolution of the orchestrator in test_workflow.py.
"""

import itertools
import math
import unittest
from unittest import mock

from doubleml_pipeline.models import saturation
import numpy as np


def _treatment(seed: int = 0, n: int = 3000, zero_share: float = 0.2):
  rng = np.random.default_rng(seed)
  t = rng.gamma(2.0, 1.5, size=n)
  t[rng.uniform(size=n) < zero_share] = 0.0
  return t


def _sporadic_treatment(seed: int = 0, n: int = 2000, zero_share: float = 0.6):
  rng = np.random.default_rng(seed)
  t = rng.gamma(2.0, 10.0, size=n)
  t[rng.uniform(size=n) < zero_share] = 0.0
  return t


def _heavy_tailed_treatment(
    seed: int = 0,
    n: int = 4000,
    sigma: float = 1.5,
    zero_share: float = 0.6,
    outlier_factor: float = 0.0,
):
  """60 % zeros + log-normal spend; optionally one day at outlier_factor x q99.

  With ``sigma = 1.5`` the ratio ``max(T) / median(T > 0)`` is about 100, the
  regime in which the former equally spaced projection grid left whole knot
  intervals without a grid point and ``B'B`` became numerically singular.
  """
  rng = np.random.default_rng(seed)
  t = np.where(
      rng.random(n) < zero_share, 0.0, rng.lognormal(3.0, sigma, size=n)
  )
  if outlier_factor > 0.0:
    t[np.argmax(t)] = float(np.quantile(t[t > 0], 0.99)) * outlier_factor
  return t


_FAMILY_ALPHAS = (("tanh", 0.05), ("exp", 0.03), ("log", 20.0), ("power", 0.5))


class SaturationFamilyTest(unittest.TestCase):
  """The response families and their derivatives."""

  def test_f_of_zero_is_zero(self):
    for family, alpha in [("exp", 0.05), ("log", 0.3), ("power", 0.6)]:
      value = saturation.apply_saturation(np.array([0.0]), family, alpha)
      self.assertAlmostEqual(float(value[0]), 0.0, places=12, msg=family)

  def test_monotone_and_concave(self):
    treatment = np.array([1.0, 5.0, 20.0, 60.0, 150.0])
    for family, alpha in [("exp", 0.05), ("log", 0.3), ("power", 0.6)]:
      values = saturation.apply_saturation(treatment, family, alpha)
      self.assertTrue(bool(np.all(np.diff(values) > 0)), family)
      slopes = saturation.saturation_derivative(treatment, family, alpha)
      self.assertTrue(bool(np.all(slopes > 0)), family)
      self.assertTrue(bool(np.all(np.diff(slopes) < 1e-9)), family)

  def test_unsupported_family_raises(self):
    with self.assertRaises(ValueError):
      saturation.validate_family("hill")


class HalfSaturationRoundTripTest(unittest.TestCase):
  """The round trip T_1/2 <-> alpha per family."""

  def test_tanh_half_saturation(self):
    for t_half in (0.05, 1.0, 7.3, 250.0):
      lam = saturation.alpha_from_half_saturation("tanh", t_half)
      self.assertAlmostEqual(lam, math.log(3.0) / t_half)
      self.assertAlmostEqual(
          float(saturation.apply_saturation(np.array([t_half]), "tanh", lam)[0]),
          0.5,
          places=12,
      )
      self.assertAlmostEqual(
          saturation.half_saturation_point("tanh", lam), t_half, places=10
      )

  def test_exp_and_log_round_trips(self):
    for t_half in (0.3, 2.0, 40.0):
      alpha = saturation.alpha_from_half_saturation("exp", t_half)
      self.assertAlmostEqual(
          float(saturation.apply_saturation(np.array([t_half]), "exp", alpha)[0]),
          0.5,
          places=12,
      )
      alpha_log = saturation.alpha_from_half_saturation("log", t_half)
      self.assertAlmostEqual(alpha_log, t_half)
      d0 = saturation.saturation_derivative(np.array([0.0]), "log", alpha_log)
      dh = saturation.saturation_derivative(
          np.array([t_half]), "log", alpha_log
      )
      self.assertAlmostEqual(float(dh[0] / d0[0]), 0.5, places=10)

  def test_power_has_no_half_saturation(self):
    self.assertTrue(math.isnan(saturation.half_saturation_point("power", 0.5)))
    with self.assertRaises(ValueError):
      saturation.alpha_from_half_saturation("power", 1.0)

  def test_tanh_shape(self):
    lam = 0.7
    t = np.linspace(0.0, 10.0, 2001)
    f = saturation.apply_saturation(t, "tanh", lam)
    np.testing.assert_allclose(f, np.tanh(lam * t / 2.0), atol=1e-14)
    deriv = saturation.saturation_derivative(t, "tanh", lam)
    np.testing.assert_allclose(deriv, np.gradient(f, t), atol=1e-4)
    self.assertTrue(np.all(np.diff(deriv) <= 1e-12))  # concave on T >= 0
    self.assertAlmostEqual(float(deriv[0]), lam / 2.0)  # linear at origin


class LogFamilyGridVariableTest(unittest.TestCase):
  """The log grid variable is T_knee = alpha."""

  def test_round_trip_is_identity(self):
    for t_knee in (0.3, 2.0, 40.0, 1e3):
      alpha = saturation.alpha_from_half_saturation("log", t_knee)
      self.assertAlmostEqual(alpha, t_knee)
      self.assertAlmostEqual(
          saturation.half_saturation_point("log", alpha), t_knee
      )
      # f'(T_knee) = f'(0) / 2 -- the marginal effect halves at T_knee.
      d0 = saturation.saturation_derivative(np.array([0.0]), "log", alpha)[0]
      dk = saturation.saturation_derivative(np.array([t_knee]), "log", alpha)[0]
      self.assertAlmostEqual(dk / d0, 0.5, places=12)
      # ... and f(T_knee) = ln 2 != 1/2: it is NOT a half-saturation point.
      self.assertAlmostEqual(
          float(
              saturation.apply_saturation(np.array([t_knee]), "log", alpha)[0]
          ),
          math.log(2.0),
          places=12,
      )

  def test_log_grid_is_laid_out_in_t_knee(self):
    t = _sporadic_treatment(3)
    grid = saturation.build_half_saturation_grid(t, "log")
    np.testing.assert_allclose(grid.t_half, grid.alphas)
    q50 = np.quantile(t[t > 0], 0.5)
    q95 = np.quantile(t[t > 0], 0.95)
    self.assertAlmostEqual(grid.t_half[0], 0.25 * q50)
    self.assertAlmostEqual(grid.t_half[-1], 4.0 * q95)


class ReferencePointTest(unittest.TestCase):
  """The reference point T_ref = q95(T | T > 0)."""

  def test_reference_point_ignores_zero_mass(self):
    t = _sporadic_treatment()
    t_ref = saturation.reference_point(t)
    self.assertAlmostEqual(t_ref, float(np.quantile(t[t > 0], 0.95)))
    self.assertAlmostEqual(
        saturation.reference_point(t, 0.5), float(np.median(t[t > 0]))
    )
    self.assertEqual(saturation.DEFAULT_T_REF_QUANTILE, 0.95)

  def test_reference_point_rejects_degenerate_input(self):
    with self.assertRaises(ValueError):
      saturation.reference_point(np.zeros(10))
    with self.assertRaises(ValueError):
      saturation.reference_point(np.ones(10), quantile=0.0)
    with self.assertRaises(ValueError):
      saturation.normalized_saturation(np.ones(3), "tanh", 0.5, t_ref=0.0)


class NormalizedSaturationTest(unittest.TestCase):
  """The normalized fbar satisfies fbar(0) = 0 and fbar(T_ref) = 1."""

  def test_passes_through_origin_and_reference_point(self):
    t = _sporadic_treatment()
    t_ref = saturation.reference_point(t)
    for family in saturation.SUPPORTED_FAMILIES:
      grid = saturation.build_half_saturation_grid(t, family)
      for alpha in grid.alphas:
        with self.subTest(family=family, alpha=alpha):
          fbar = saturation.normalized_saturation(
              np.array([0.0, t_ref]), family, float(alpha), t_ref
          )
          self.assertAlmostEqual(float(fbar[0]), 0.0, places=12)
          self.assertAlmostEqual(float(fbar[1]), 1.0, places=12)
          # fbar is the raw f divided by f(T_ref; alpha).
          raw = saturation.apply_saturation(t, family, float(alpha))
          kappa = saturation.apply_saturation(
              np.array([t_ref]), family, float(alpha)
          )[0]
          np.testing.assert_allclose(
              saturation.normalized_saturation(t, family, float(alpha), t_ref),
              raw / kappa,
              rtol=1e-12,
          )

  def test_slow_saturation_limit_is_linear(self):
    t = np.linspace(0.0, 50.0, 11)
    t_ref = 40.0
    for family, alpha in (("tanh", 1e-5), ("exp", 1e-5), ("log", 1e6)):
      with self.subTest(family=family):
        np.testing.assert_allclose(
            saturation.normalized_saturation(t, family, alpha, t_ref),
            t / t_ref,
            rtol=1e-3,
        )

  def test_fbar_is_unit_free(self):
    """fbar(100 T; alpha', 100 T_ref) == fbar(T; alpha, T_ref)."""
    t = _sporadic_treatment(1)
    t_ref = saturation.reference_point(t)
    for family, alpha in _FAMILY_ALPHAS:
      with self.subTest(family=family):
        # T' = 100 T is T' / c with the divisor c = 100 (normalization bridge).
        alpha_scaled, _ = saturation.rescale_alpha_to_raw(family, alpha, 100.0)
        np.testing.assert_allclose(
            saturation.normalized_saturation(
                100.0 * t, family, alpha_scaled, 100.0 * t_ref
            ),
            saturation.normalized_saturation(t, family, alpha, t_ref),
            rtol=1e-9,
        )

  def test_bridge_identity_has_unit_gain(self):
    """Bridge identity: fbar(T/c; a, T_ref) = fbar(T; a_i, c T_ref)."""
    t = _sporadic_treatment(2)
    t_ref = saturation.reference_point(t / 7.5)
    divisor = np.where(np.arange(t.size) % 2 == 0, 7.5, 3.0)
    for family, alpha in _FAMILY_ALPHAS:
      with self.subTest(family=family):
        alpha_raw, _ = saturation.rescale_alpha_to_raw(family, alpha, divisor)
        t_ref_raw = saturation.raw_scale_reference(t_ref, divisor)
        np.testing.assert_allclose(t_ref_raw, t_ref * divisor)
        np.testing.assert_allclose(
            saturation.normalized_saturation(t, family, alpha_raw, t_ref_raw),
            saturation.normalized_saturation(t / divisor, family, alpha, t_ref),
            rtol=1e-9,
        )

  def test_grid_variable_names(self):
    self.assertEqual(saturation.grid_variable("tanh"), "T_1/2")
    self.assertEqual(saturation.grid_variable("exp"), "T_1/2")
    self.assertEqual(saturation.grid_variable("log"), "T_knee")
    self.assertEqual(saturation.grid_variable("power"), "alpha")


class AlphaGridTest(unittest.TestCase):
  """The manual alpha grid builder."""

  def test_log_grid_is_sorted_and_sized(self):
    grid = saturation.build_alpha_grid(
        {"min": 0.01, "max": 0.9, "num": 11, "scale": "log"}, "power"
    )
    self.assertEqual(len(grid), 11)
    self.assertTrue(bool(np.all(np.diff(grid) > 0)))
    self.assertLess(float(grid.max()), 1.0)

  def test_power_alpha_at_or_above_one_rejected(self):
    with self.assertRaises(ValueError):
      saturation.build_alpha_grid(
          {"min": 0.1, "max": 2.0, "num": 5, "scale": "log"}, "power"
      )

  def test_non_positive_min_rejected(self):
    with self.assertRaises(ValueError):
      saturation.build_alpha_grid(
          {"min": 0.0, "max": 0.5, "num": 5, "scale": "log"}, "exp"
      )


class ApiReferenceExamplesTest(unittest.TestCase):
  """The config examples in docs/api_reference.md actually work.

  Documentation examples rot silently. These tests execute the grid
  configuration shown in the api_reference `saturation` examples and assert
  it produces a usable grid, so a future change to validation or defaults
  breaks the test rather than the docs.
  """

  def test_documented_defaults_match_the_code(self):
    """The default table in api_reference.md must stay true."""
    self.assertEqual(saturation.default_alpha_bounds("power"), (0.05, 0.95))
    self.assertEqual(saturation.default_alpha_bounds("exp"), (0.05, 5.0))
    self.assertEqual(saturation.default_alpha_bounds("log"), (0.05, 5.0))
    # Documented grid defaults: num=25, scale="log".
    grid = saturation.build_alpha_grid({}, "exp")
    self.assertEqual(len(grid), 25)
    self.assertAlmostEqual(grid[0], 0.05)
    self.assertAlmostEqual(grid[-1], 5.0)

  def test_out_of_default_bounds_are_accepted_not_clipped(self):
    """`build_alpha_grid` validates bounds but never silently clips them.

    api_reference documents that the default bounds may be overridden when the
    treatment is not normalized at all, in which case alpha carries raw units.
    If `build_alpha_grid` ever started clipping to the family defaults, that
    escape hatch would silently stop working, so pin the behaviour.
    """
    grid = saturation.build_alpha_grid(
        {"min": 1e-5, "max": 1e-2, "num": 25, "scale": "log"}, "exp"
    )
    self.assertEqual(len(grid), 25)
    self.assertAlmostEqual(grid[0], 1e-5)
    self.assertAlmostEqual(grid[-1], 1e-2)
    self.assertTrue(np.all(np.diff(grid) > 0))

  def test_default_grid_needs_a_divisive_normalization(self):
    """Why api_reference requires a divisive normalization for a saturated T.

    On a raw, large-magnitude treatment every alpha in the default grid maps
    every observation to f = 1, so the R-loss profile is flat and alpha is
    unidentified. Dividing by the median first (the normalization bridge
    ``T / c``) makes the SAME default grid discriminate, which is what makes
    `[0.05, 5.0]` scale-free.
    """
    spend = np.array([5000.0, 10000.0, 20000.0])
    default_grid = saturation.build_alpha_grid({}, "exp")

    for alpha in default_grid:
      f_raw = saturation.apply_saturation(spend, "exp", float(alpha))
      np.testing.assert_allclose(f_raw, 1.0)

    # Normalization bridge: alpha is estimated against T / c, where c is the
    # median of the non-zero treatment, so median(T / c) == 1 by construction.
    normalized = spend / np.median(spend)
    values = np.array([
        saturation.apply_saturation(normalized, "exp", float(a))
        for a in default_grid
    ])
    self.assertGreater(values.std(), 0.1)

  def test_power_grid_rejects_max_at_or_above_one(self):
    with self.assertRaises(ValueError):
      saturation.build_alpha_grid({"min": 0.1, "max": 1.0}, "power")

  def test_non_positive_min_is_rejected(self):
    with self.assertRaises(ValueError):
      saturation.build_alpha_grid({"min": 0.0, "max": 1e-2}, "exp")


class HalfSaturationGridTest(unittest.TestCase):
  """The automatic half-saturation grid, its extension and manual override."""

  def test_auto_grid_bounds_and_conversion(self):
    t = _treatment(2)
    t_pos = t[t > 0]
    q50, q95 = np.quantile(t_pos, [0.5, 0.95])
    for family, conv in (
        ("tanh", lambda th: math.log(3.0) / th),
        ("exp", lambda th: math.log(2.0) / th),
        ("log", lambda th: th),
    ):
      with self.subTest(family=family):
        grid = saturation.build_half_saturation_grid(t, family, {})
        self.assertEqual(grid.mode, "auto")
        self.assertEqual(grid.t_half.size, 25)
        self.assertAlmostEqual(grid.t_half[0], 0.25 * q50, places=9)
        self.assertAlmostEqual(grid.t_half[-1], 4.0 * q95, places=9)
        ratios = np.diff(np.log(grid.t_half))
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
        np.testing.assert_allclose(
            grid.alphas, [conv(th) for th in grid.t_half], rtol=1e-12
        )

  def test_power_grid(self):
    grid = saturation.build_half_saturation_grid(_treatment(), "power", {})
    np.testing.assert_allclose(grid.alphas, np.linspace(0.1, 0.95, 25))
    self.assertFalse(grid.extensible)

  def test_custom_c_and_num(self):
    t = _treatment(3)
    grid = saturation.build_half_saturation_grid(
        t, "tanh", {"c_min": 0.5, "c_max": 2.0, "num": 11}
    )
    t_pos = t[t > 0]
    self.assertEqual(grid.coord.size, 11)
    self.assertAlmostEqual(grid.t_half[0], 0.5 * np.quantile(t_pos, 0.5))
    self.assertAlmostEqual(grid.t_half[-1], 2.0 * np.quantile(t_pos, 0.95))

  def test_extension_by_factor_four(self):
    grid = saturation.build_half_saturation_grid(_treatment(), "tanh", {})
    lower = saturation.extend_alpha_grid(grid, "lower")
    self.assertAlmostEqual(lower.t_half[0], grid.t_half[0] / 4.0, places=9)
    self.assertAlmostEqual(lower.t_half[-1], grid.t_half[-1], places=9)
    self.assertEqual(lower.extensions, 1)
    upper = saturation.extend_alpha_grid(lower, "upper")
    self.assertAlmostEqual(upper.t_half[-1], grid.t_half[-1] * 4.0, places=9)
    self.assertEqual(upper.extensions, 2)
    self.assertTrue(np.all(np.diff(upper.coord) > 0))

  def test_manual_override_and_v3_compat(self):
    manual = saturation.build_half_saturation_grid(
        _treatment(), "exp", {"mode": "manual", "min": 0.1, "max": 2.0,
                              "num": 5}
    )
    self.assertEqual(manual.mode, "manual")
    np.testing.assert_allclose(manual.alphas, np.logspace(-1, np.log10(2), 5))
    self.assertFalse(manual.extensible)
    # Legacy block with min/max but no mode -> manual (documented assumption).
    self.assertEqual(saturation.resolve_grid_mode({"min": 0.1, "max": 1}),
                     "manual")
    self.assertEqual(saturation.resolve_grid_mode({}), "auto")
    with self.assertRaises(ValueError):
      saturation.resolve_grid_mode({"mode": "optuna"})


class LinearSideTest(unittest.TestCase):
  """The linear side of the grid per family / layout."""

  def test_auto_grids(self):
    t = _sporadic_treatment(8)
    for family in ("tanh", "exp", "log"):
      self.assertEqual(
          saturation.build_half_saturation_grid(t, family).linear_side, "upper"
      )
    self.assertEqual(
        saturation.build_half_saturation_grid(t, "power").linear_side, "upper"
    )

  def test_manual_alpha_grids(self):
    t = _sporadic_treatment(9)
    cfg = {"mode": "manual", "min": 0.05, "max": 5.0, "num": 5}
    self.assertEqual(
        saturation.build_half_saturation_grid(t, "tanh", cfg).linear_side,
        "lower",
    )
    self.assertEqual(
        saturation.build_half_saturation_grid(t, "exp", cfg).linear_side,
        "lower",
    )
    self.assertEqual(
        saturation.build_half_saturation_grid(t, "log", cfg).linear_side,
        "upper",
    )


class TreatmentBasisTest(unittest.TestCase):
  """The B-spline approximation of fbar(T; alpha)."""

  def _treatment_scenarios(self):
    rng = np.random.default_rng(0)
    gamma = np.abs(rng.gamma(2.0, 30.0, size=1000))
    return {
        # Unit scale: the easy case.
        "uniform_unit": np.linspace(0.0, 1.0, 1000),
        # Large scale: alpha and T are on very different scales.
        "uniform_large": np.linspace(0.0, 200.0, 1000),
        # Realistic skewed spend.
        "gamma_spend": gamma,
        # Realistic spend with a large point mass at zero: the fixed-grid
        # projection puts no extra weight there (see below).
        "spend_with_zeros": np.where(rng.random(1000) < 0.3, 0.0, gamma),
    }

  def test_projection_error_is_small_on_the_observed_distribution(self):
    """The projection must be accurate under P_T on the grid Phase 0 uses.

    ``c(alpha) = P fbar_alpha`` is a least-squares projection on a fixed,
    equally spaced 500-point grid over ``[0, T_max]`` (NOT a regression on
    the observed ``T_i``).  The R-loss nevertheless only evaluates the
    approximation at the observed ``T_i``, so the relevant error is the
    relative RMSE under ``P_T`` -- measured here on the alpha grid Phase 0
    actually uses (``T_1/2 in [0.25 q50, 4 q95]`` for tanh / exp / log,
    ``alpha in [0.1, 0.95]`` for power).  The projected target is the
    normalized ``fbar``.

    Known limitation of the fixed-grid projection:
    ``power`` has an infinite slope at 0, and with a point mass at ``T = 0``
    the equally spaced grid does not pin the spline to ``fbar(0) = 0`` there,
    so the error under ``P_T`` can exceed the tolerance.  That case must then
    be *reported* through ``eps(J)`` / ``select_basis_dim(satisfied=False)``
    rather than pass silently.
    """
    tol = 0.05
    known_k3_limitation = {("spend_with_zeros", "power")}
    for name, treatment in self._treatment_scenarios().items():
      basis = saturation.TreatmentBasis(treatment, basis_dim=15)
      design = basis.transform(treatment)
      for family in saturation.SUPPORTED_FAMILIES:
        alphas = saturation.build_half_saturation_grid(
            treatment, family, {}
        ).alphas
        coefs = basis.project_family(family, alphas)
        truths = basis.normalized_values(treatment, family, alphas)
        worst_alpha, worst = None, 0.0
        for grid_idx in range(len(alphas)):
          truth = truths[:, grid_idx]
          approx = design @ coefs[:, grid_idx]
          scale = max(float(np.sqrt((truth**2).mean())), 1e-9)
          rel_rmse = float(np.sqrt(((approx - truth) ** 2).mean())) / scale
          if rel_rmse > worst:
            worst_alpha, worst = float(alphas[grid_idx]), rel_rmse
        if worst < tol:
          continue
        # Only the documented limitation may exceed the tolerance ...
        self.assertIn(
            (name, family),
            known_k3_limitation,
            msg=f"{name}/{family}/alpha={worst_alpha:.4g}: {worst}",
        )
        # ... and it must be visible to the caller through eps(J).
        eps = basis.approximation_error(family, alphas)
        self.assertGreater(float(eps.max()), tol, msg=f"{name}/{family}")
        selection = saturation.select_basis_dim(
            treatment, family, alphas, "auto"
        )
        self.assertFalse(selection.satisfied, msg=f"{name}/{family}")

  def test_more_bases_do_not_degrade_the_fit(self):
    treatment = np.abs(np.random.default_rng(1).gamma(2.0, 30.0, size=1000))
    errors = {}
    for basis_dim in (10, 25):
      basis = saturation.TreatmentBasis(treatment, basis_dim=basis_dim)
      design = basis.transform(treatment)
      coefs = basis.project_family("power", np.array([0.5]))
      # The projected target is fbar, not f.
      truth = basis.normalized_values(treatment, "power", np.array([0.5]))[:, 0]
      errors[basis_dim] = float(
          np.sqrt(((design @ coefs[:, 0] - truth) ** 2).mean())
      )
    self.assertLessEqual(errors[25], errors[10] * 1.05)


class GrevilleLinearPrecisionTest(unittest.TestCase):
  """Greville linear precision: T = sum_j xi_j b_j(T) exactly."""

  def test_linear_precision_for_all_j(self):
    for seed, j_dim in itertools.product(range(3), (4, 6, 9, 15, 20)):
      with self.subTest(seed=seed, J=j_dim):
        t = _treatment(seed)
        basis = saturation.TreatmentBasis(t, basis_dim=j_dim)
        xi = basis.greville_abscissae()
        self.assertEqual(xi.shape, (basis.n_basis,))
        t_eval = np.concatenate([t, np.linspace(0.0, t.max(), 101)])
        # Exact on the domain [0, T_max]; `transform` clips values above the
        # domain right end T_max = q99.9(T > 0).
        np.testing.assert_allclose(
            basis.transform(t_eval) @ xi,
            np.minimum(t_eval, basis.t_max),
            atol=1e-10,
            rtol=1e-10,
        )

  def test_partition_of_unity_and_zero_knot(self):
    t = _treatment(1)
    basis = saturation.TreatmentBasis(t, basis_dim=8)
    self.assertEqual(basis.t_min, 0.0)  # boundary 0 is a knot.
    np.testing.assert_allclose(basis.transform(t).sum(axis=1), 1.0, atol=1e-12)


class BasisProjectorTest(unittest.TestCase):
  """c(alpha) = P fbar_alpha on the fixed projection grid."""

  def test_domain_knots_and_projector(self):
    t = _sporadic_treatment(4)
    t_ref = saturation.reference_point(t)
    basis = saturation.TreatmentBasis(t, basis_dim=12, t_ref=t_ref)
    self.assertEqual(basis.t_min, 0.0)
    # The domain right end is the BASIS_T_MAX_QUANTILE quantile of T > 0, not
    # the sample maximum.
    self.assertEqual(basis.t_max, saturation.basis_domain_max(t))
    self.assertLess(basis.t_max, float(t.max()))
    self.assertEqual(basis.t_ref, t_ref)
    # Grid = equally spaced points + quantiles of T > 0 + Greville abscissae.
    self.assertGreaterEqual(basis.n_grid, saturation.BASIS_PROJECTION_GRID)
    self.assertEqual(basis.t_grid.size, basis.n_grid)
    self.assertEqual(basis.t_grid[0], 0.0)
    self.assertEqual(basis.t_grid[-1], basis.t_max)
    for xi in basis.greville_abscissae():
      self.assertTrue(np.any(np.isclose(basis.t_grid, xi)))
    self.assertEqual(basis.b_grid.shape, (basis.n_grid, basis.n_basis))
    self.assertEqual(basis.projector.shape, (basis.n_basis, basis.n_grid))
    # P = B^+  <=>  P B = I.
    np.testing.assert_allclose(
        basis.projector @ basis.b_grid, np.eye(basis.n_basis), atol=1e-8
    )
    self.assertLess(basis.basis_cond, 1e3)
    self.assertFalse(basis.ill_conditioned)
    # Interior knots are quantiles of T > 0 (never 0, never T_max).
    self.assertTrue(np.all(basis.inner_knots > 0.0))
    self.assertTrue(np.all(basis.inner_knots < basis.t_max))
    positive = t[t > 0]
    for knot in basis.inner_knots:
      self.assertGreaterEqual(knot, positive.min())

  def test_projection_is_least_squares_on_the_grid(self):
    t = _sporadic_treatment(5)
    basis = saturation.TreatmentBasis(t, basis_dim=10)
    grid = saturation.build_half_saturation_grid(t, "tanh")
    coefs = basis.project_family("tanh", grid.alphas)
    self.assertEqual(coefs.shape, (basis.n_basis, grid.alphas.size))
    fbar_grid = basis.normalized_values(basis.t_grid, "tanh", grid.alphas)
    expected, *_ = np.linalg.lstsq(basis.b_grid, fbar_grid, rcond=None)
    np.testing.assert_allclose(coefs, expected, atol=1e-8)
    # The coefficients are NOT a regression on the observed T_i.
    with self.assertRaises(ValueError):
      basis.project_values(basis.normalized_values(t, "tanh", grid.alphas))

  def test_identity_candidate_through_greville(self):
    """Identity candidate: fbar_T = T / T_ref = sum_j (xi_j / T_ref) b_j(T)."""
    t = _sporadic_treatment(6)
    t_ref = saturation.reference_point(t)
    basis = saturation.TreatmentBasis(t, basis_dim=9, t_ref=t_ref)
    coef_lin = basis.greville_abscissae() / t_ref
    # Exact linear precision on the domain; rows above T_max are clipped.
    np.testing.assert_allclose(
        basis.transform(t) @ coef_lin,
        np.minimum(t, basis.t_max) / t_ref,
        atol=1e-10,
    )

  def test_select_basis_dim_threads_t_ref(self):
    t = _sporadic_treatment(7)
    grid = saturation.build_half_saturation_grid(t, "exp")
    sel = saturation.select_basis_dim(t, "exp", grid.alphas, t_ref=12.5)
    self.assertEqual(sel.basis.t_ref, 12.5)
    self.assertTrue(sel.satisfied)
    self.assertLess(sel.approx_error, 1e-3)
    self.assertFalse(sel.ill_conditioned)
    self.assertLess(sel.basis_cond, 1e3)
    self.assertEqual(set(sel.scanned), set(sel.scanned_cond))

  def test_basis_domain_max(self):
    t = _heavy_tailed_treatment(0)
    self.assertEqual(
        saturation.basis_domain_max(t),
        float(np.quantile(t[t > 0], saturation.BASIS_T_MAX_QUANTILE)),
    )
    self.assertEqual(saturation.basis_domain_max(t, 1.0), float(t.max()))
    # No positive value: the finite maximum.
    self.assertEqual(saturation.basis_domain_max(np.zeros(5)), 0.0)
    with self.assertRaises(ValueError):
      saturation.basis_domain_max(t, 0.0)
    with self.assertRaises(ValueError):
      saturation.basis_domain_max(np.array([np.nan]))

  def test_caller_supplied_t_max_clips_rows_above_it(self):
    t = _heavy_tailed_treatment(1)
    t_max = float(np.quantile(t[t > 0], 0.9))
    basis = saturation.TreatmentBasis(t, basis_dim=12, t_max=t_max)
    self.assertEqual(basis.t_max, t_max)
    self.assertEqual(basis.n_above_t_max, int(np.sum(t > t_max)))
    self.assertGreater(basis.n_above_t_max, 0)
    self.assertTrue(np.all(basis.inner_knots < t_max))
    # transform clips to the domain, so the partition of unity still holds.
    design = basis.transform(t)
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        design @ basis.greville_abscissae(), np.minimum(t, t_max), atol=1e-9
    )

  def test_heavy_tail_stays_well_conditioned(self):
    """Skewed spend (with and without an outlier day) keeps B well-conditioned.

    With the former equally spaced grid these cases had cond(B) between 1e4
    and infinity and coefficients up to 1e15; the augmented grid keeps cond(B)
    of order 10 and the coefficients within the range of fbar.
    """
    for outlier_factor in (0.0, 30.0):
      t = _heavy_tailed_treatment(2, outlier_factor=outlier_factor)
      t_ref = saturation.reference_point(t)
      t_max = saturation.basis_domain_max(t)
      grid = saturation.build_half_saturation_grid(t, "tanh")
      for j_dim in (12, 15, 20):
        with self.subTest(outlier_factor=outlier_factor, j_dim=j_dim):
          basis = saturation.TreatmentBasis(
              t, basis_dim=j_dim, t_ref=t_ref, t_max=t_max
          )
          self.assertLess(basis.basis_cond, 1e3)
          self.assertFalse(basis.ill_conditioned)
          np.testing.assert_allclose(
              basis.projector @ basis.b_grid,
              np.eye(basis.n_basis),
              atol=1e-6,
          )
          coefs = basis.project_family("tanh", grid.alphas)
          self.assertLess(float(np.max(np.abs(coefs))), 20.0)
          # Bounded approximation error at every J (it was 1e1 to 1e14 with
          # the former grid); small J is legitimately coarse for the
          # fast-saturating candidates, accuracy is the job of the auto rule.
          eps = basis.approximation_error("tanh", grid.alphas)
          self.assertLess(float(np.max(eps)), 0.5)
      # The automatic choice of J is well-conditioned; on this heavy tail the
      # tolerance 1e-3 is not reachable within J <= 20 (eps(20) is between
      # 2e-2 and 1e-1 depending on the seed -- the long last knot interval,
      # not conditioning), so the rule returns the smallest-eps J and Diag1
      # warns.
      sel = saturation.select_basis_dim(
          t, "tanh", grid.alphas, t_ref=t_ref, t_max=t_max
      )
      self.assertFalse(sel.ill_conditioned)
      self.assertLess(sel.basis_cond, 1e3)
      self.assertLess(sel.approx_error, 0.2)
      self.assertEqual(sel.approx_error, min(sel.scanned.values()))

  def test_ill_conditioning_guard(self):
    """cond(B) >= MAX_BASIS_CONDITION flags the basis instead of raising."""
    t = _heavy_tailed_treatment(3)
    grid = saturation.build_half_saturation_grid(t, "tanh")
    # The new grid never trips the real threshold on data, so lower it below
    # the healthy cond(B) of order 10 to exercise the guard.
    with mock.patch.object(saturation, "MAX_BASIS_CONDITION", 1.0):
      fixed = saturation.select_basis_dim(t, "tanh", grid.alphas, basis_dim=12)
      self.assertEqual(fixed.mode, "fixed")
      self.assertTrue(fixed.ill_conditioned)
      self.assertFalse(fixed.satisfied)
      auto = saturation.select_basis_dim(t, "tanh", grid.alphas)
      self.assertEqual(auto.mode, "auto")
      self.assertTrue(auto.ill_conditioned)
      self.assertFalse(auto.satisfied)
      self.assertEqual(len(auto.scanned), 15)
    # With the real threshold the same data is admissible.
    self.assertFalse(
        saturation.select_basis_dim(t, "tanh", grid.alphas).ill_conditioned
    )


class BasisDimAutoTest(unittest.TestCase):
  """Automatic J: the smallest J in [6, 20] with eps(J) < basis_tol.

  When no J meets the tolerance the J with the smallest eps(J) is adopted
  (``satisfied=False``; Diag1 warns downstream).
  """

  def test_smallest_j_is_selected(self):
    t = _treatment(4)
    grid = saturation.build_half_saturation_grid(t, "tanh", {})
    sel = saturation.select_basis_dim(t, "tanh", grid.alphas, "auto", 1e-3)
    self.assertEqual(sel.mode, "auto")
    self.assertGreaterEqual(sel.n_basis, 6)
    self.assertLessEqual(sel.n_basis, 20)
    if sel.satisfied:
      self.assertLess(sel.approx_error, 1e-3)
      smaller = [j for j in sel.scanned if j < sel.n_basis]
      for j in smaller:
        self.assertGreaterEqual(sel.scanned[j], 1e-3)
    # Direct check of the definition for the adopted J.
    eps = sel.basis.approximation_error("tanh", grid.alphas)
    self.assertAlmostEqual(float(np.max(eps)), sel.approx_error, places=12)

  def test_unsatisfiable_returns_the_smallest_error_j(self):
    """No J meets the tolerance -> the smallest-eps J, satisfied=False."""
    t = _treatment(5)
    grid = saturation.build_half_saturation_grid(t, "exp", {})
    sel = saturation.select_basis_dim(
        t, "exp", grid.alphas, "auto", tol=1e-14, dim_range=(6, 8)
    )
    self.assertFalse(sel.satisfied)
    self.assertFalse(sel.ill_conditioned)
    self.assertEqual(set(sel.scanned), {6, 7, 8})
    self.assertEqual(sel.n_basis, min(sel.scanned, key=sel.scanned.get))
    self.assertEqual(sel.approx_error, min(sel.scanned.values()))
    # The same rule on a heavy tail, where even J = 20 misses 1e-12.
    t = _heavy_tailed_treatment(4)
    grid = saturation.build_half_saturation_grid(t, "tanh")
    sel = saturation.select_basis_dim(t, "tanh", grid.alphas, tol=1e-12)
    self.assertFalse(sel.satisfied)
    self.assertFalse(sel.ill_conditioned)
    self.assertEqual(sel.approx_error, min(sel.scanned.values()))

  def test_fixed_integer(self):
    t = _treatment(6)
    sel = saturation.select_basis_dim(t, "log", np.array([1.0]), 9)
    self.assertEqual(sel.mode, "fixed")
    self.assertEqual(sel.n_basis, 9)


class AllocationTest(unittest.TestCase):
  """The closed-form / water-filling allocators."""

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.theta = np.abs(rng.normal(2.0, 0.7, size=40))
    self.actual = np.abs(rng.normal(50.0, 20.0, size=40))
    self.budget = float(self.actual.sum())
    self.t_bar = float(self.actual.max())

  def test_allocation_respects_budget_and_box(self):
    for family, alpha in [("exp", 0.02), ("log", 0.05), ("power", 0.5)]:
      alloc = saturation.optimal_allocation(
          family=family,
          alpha=alpha,
          theta=self.theta,
          budget=self.budget,
          t_max=self.t_bar,
      )
      self.assertAlmostEqual(
          float(alloc.sum()) / self.budget, 1.0, places=4, msg=family
      )
      self.assertTrue(bool(np.all(alloc >= -1e-9)), family)
      self.assertTrue(bool(np.all(alloc <= self.t_bar + 1e-6)), family)

  def test_allocation_improves_the_objective(self):
    for family, alpha in [("exp", 0.02), ("log", 0.05), ("power", 0.5)]:
      alloc = saturation.optimal_allocation(
          family=family,
          alpha=alpha,
          theta=self.theta,
          budget=self.budget,
          t_max=self.t_bar,
      )
      obj_opt = float(
          (self.theta * saturation.apply_saturation(alloc, family, alpha)).sum()
      )
      obj_act = float(
          (
              self.theta
              * saturation.apply_saturation(self.actual, family, alpha)
          ).sum()
      )
      self.assertGreaterEqual(obj_opt, obj_act - 1e-6, msg=family)

  def test_allocation_is_monotone_in_theta(self):
    alloc = saturation.optimal_allocation(
        family="power",
        alpha=0.5,
        theta=self.theta,
        budget=self.budget,
        t_max=self.t_bar,
    )
    interior = (alloc > 1e-6) & (alloc < self.t_bar - 1e-6)
    if int(interior.sum()) > 2:
      order = np.argsort(self.theta[interior])
      self.assertTrue(bool(np.all(np.diff(alloc[interior][order]) > -1e-6)))


class TanhAllocatorTest(unittest.TestCase):
  """The semi-closed-form tanh allocator (bisection on mu)."""

  def _objective(self, t, theta, lam):
    return float(np.sum(theta * np.tanh(lam * t / 2.0)))

  def test_kkt_conditions(self):
    rng = np.random.default_rng(0)
    for trial in range(5):
      with self.subTest(trial=trial):
        theta = rng.uniform(0.2, 3.0, size=12)
        theta[rng.uniform(size=12) < 0.2] = 0.05  # some tiny effects
        lam, t_max = 0.8, 6.0
        budget = float(rng.uniform(3.0, 25.0))
        alloc = saturation.optimal_allocation(
            "tanh", lam, theta, budget, t_max
        )
        self.assertAlmostEqual(float(alloc.sum()), budget, places=5)
        self.assertTrue(np.all(alloc >= -1e-12))
        self.assertTrue(np.all(alloc <= t_max + 1e-9))
        marginal = theta * (lam / 2.0) * (1.0 - np.tanh(lam * alloc / 2) ** 2)
        interior = (alloc > 1e-6) & (alloc < t_max - 1e-6)
        if np.any(interior):
          mu = float(np.median(marginal[interior]))
          np.testing.assert_allclose(marginal[interior], mu, rtol=1e-4)
          # zero allocations: marginal at 0 must not exceed mu
          zeros = alloc <= 1e-6
          self.assertTrue(np.all(theta[zeros] * lam / 2.0 <= mu * (1 + 1e-4)))
          # capped allocations: marginal at T_bar at least mu
          capped = alloc >= t_max - 1e-6
          self.assertTrue(np.all(marginal[capped] >= mu * (1 - 1e-4)))

  def test_matches_brute_force(self):
    theta = np.array([2.0, 1.0, 0.4])
    lam, t_max, budget = 1.1, 4.0, 5.0
    alloc = saturation.optimal_allocation("tanh", lam, theta, budget, t_max)
    step = 0.01
    best = -np.inf
    grid = np.arange(0.0, t_max + 1e-9, step)
    for t0 in grid:
      for t1 in grid:
        t2 = budget - t0 - t1
        if t2 < -1e-9 or t2 > t_max + 1e-9:
          continue
        val = self._objective(np.array([t0, t1, max(t2, 0.0)]), theta, lam)
        best = max(best, val)
    got = self._objective(alloc, theta, lam)
    self.assertGreaterEqual(got, best - 1e-6)
    self.assertLess(best - got, 1e-3 + 1e-9)


class NormalizationBridgeTest(unittest.TestCase):
  """`f(T/c; alpha) == gain * f(T; alpha_raw)` for every family."""

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.treatment = rng.gamma(2.0, 50.0, size=500)
    # Include exact zeros: f(0) = 0 is what makes the Phase 4 theta recovery
    # exact, so the bridge must preserve it.
    self.treatment[:20] = 0.0

  def test_global_constant_divisor(self):
    """The median-normalize case: a single `c` shared by every row."""
    divisor = 137.5
    for family, alpha in (("exp", 0.8), ("log", 1.7), ("power", 0.4)):
      with self.subTest(family=family):
        direct = saturation.apply_saturation(
            self.treatment / divisor, family, alpha
        )
        alpha_raw, gain = saturation.rescale_alpha_to_raw(
            family, alpha, np.full_like(self.treatment, divisor)
        )
        bridged = gain * saturation.apply_saturation(
            self.treatment, family, alpha_raw
        )
        np.testing.assert_allclose(direct, bridged, rtol=1e-12, atol=1e-12)

  def test_row_varying_divisor(self):
    """The population-median case: one alpha, many derived alpha_i."""
    rng = np.random.default_rng(1)
    divisor = rng.uniform(0.5, 20.0, size=self.treatment.shape)
    for family, alpha in (("exp", 0.8), ("log", 1.7), ("power", 0.4)):
      with self.subTest(family=family):
        direct = saturation.apply_saturation(
            self.treatment / divisor, family, alpha
        )
        alpha_raw, gain = saturation.rescale_alpha_to_raw(
            family, alpha, divisor
        )
        bridged = gain * saturation.apply_saturation(
            self.treatment, family, alpha_raw
        )
        np.testing.assert_allclose(direct, bridged, rtol=1e-12, atol=1e-12)

  def test_alpha_i_direction_matches_the_documented_mapping(self):
    """The derived alpha_i moves the way ``rescale_alpha_to_raw`` documents.

    A round-trip check cannot catch a sign or inversion error that cancels out,
    so the exact per-family mapping is pinned here: ``exp`` divides alpha by
    ``c_i``, ``log`` multiplies it, and ``power`` leaves alpha alone and pushes
    the whole rescaling into ``gain_i = c_i ** -alpha``.
    """
    divisor = np.array([2.0, 4.0])
    exp_alpha, exp_gain = saturation.rescale_alpha_to_raw("exp", 1.0, divisor)
    np.testing.assert_allclose(exp_alpha, [0.5, 0.25])
    np.testing.assert_allclose(exp_gain, [1.0, 1.0])

    log_alpha, log_gain = saturation.rescale_alpha_to_raw("log", 1.0, divisor)
    np.testing.assert_allclose(log_alpha, [2.0, 4.0])
    np.testing.assert_allclose(log_gain, [1.0, 1.0])

    pow_alpha, pow_gain = saturation.rescale_alpha_to_raw(
        "power", 0.5, divisor
    )
    np.testing.assert_allclose(pow_alpha, [0.5, 0.5])
    np.testing.assert_allclose(pow_gain, divisor**-0.5)

  def test_zero_treatment_stays_zero(self):
    """`f(0) == 0` survives the rescale; Phase 4's theta recovery needs it."""
    for family in saturation.SUPPORTED_FAMILIES:
      with self.subTest(family=family):
        alpha_raw, gain = saturation.rescale_alpha_to_raw(
            family, 0.7, np.array([3.0])
        )
        value = gain * saturation.apply_saturation(
            np.array([0.0]), family, alpha_raw
        )
        self.assertAlmostEqual(float(value[0]), 0.0)

  def test_scalar_alpha_still_returns_a_float(self):
    """Backward compatibility: existing scalar call sites are unchanged."""
    for family in saturation.SUPPORTED_FAMILIES:
      with self.subTest(family=family):
        self.assertIsInstance(saturation.clip_alpha(family, 0.5), float)
    self.assertIsInstance(
        saturation.clip_alpha("exp", np.array([0.5, 1.0])), np.ndarray
    )


class NormalizedAllocatorTest(unittest.TestCase):
  """The allocation is unchanged by the normalization."""

  def setUp(self):
    super().setUp()
    self.theta_raw = np.array([3.0, 1.0, 0.0, 2.5, 0.7, -1.0])
    self.t_max = 80.0
    self.t_ref = 31.0

  def test_reproduces_pre_normalization_allocation(self):
    for family, alpha in _FAMILY_ALPHAS:
      kappa = float(
          saturation.apply_saturation(np.array([self.t_ref]), family, alpha)[0]
      )
      theta_bar = self.theta_raw * kappa
      for budget in (5.0, 60.0, 250.0, 1e4):
        with self.subTest(family=family, budget=budget):
          before = saturation.optimal_allocation(
              family, alpha, self.theta_raw, budget, self.t_max
          )
          after = saturation.optimal_allocation(
              family, alpha, theta_bar, budget, self.t_max, t_ref=self.t_ref
          )
          np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)
          # The ROI cap T_t^r is invariant as well
          # (theta_bar fbar = theta_raw f), hence so is the capped solve.
          before_cap = saturation.roi_cap(
              family, alpha, self.theta_raw, 0.02, t_upper=self.t_max
          )
          after_cap = saturation.roi_cap(
              family, alpha, theta_bar, 0.02, t_ref=self.t_ref,
              t_upper=self.t_max,
          )
          np.testing.assert_allclose(
              after_cap, before_cap, rtol=1e-6, atol=1e-7
          )
          before_roi = saturation.solve_budget_allocation(
              family, alpha, self.theta_raw, budget,
              np.minimum(self.t_max, before_cap),
          ).allocation
          after_roi = saturation.solve_budget_allocation(
              family, alpha, theta_bar, budget,
              np.minimum(self.t_max, after_cap), t_ref=self.t_ref,
          ).allocation
          np.testing.assert_allclose(
              after_roi, before_roi, rtol=1e-6, atol=1e-7
          )

  def test_tanh_kkt_formula_with_kappa(self):
    """T*(mu) = (2/lambda) artanh sqrt(1 - 2 mu kappa / (lambda theta))."""
    lam, mu = 0.05, 0.01
    kappa = math.tanh(lam * self.t_ref / 2.0)
    theta = np.array([3.0, 1.0, 0.3])
    # pylint: disable-next=protected-access
    alloc = saturation._allocation_from_lambda(
        "tanh", lam, theta, mu, 1e6, t_ref=self.t_ref
    )
    for theta_t, got in zip(theta, alloc):
      if theta_t > 2.0 * mu * kappa / lam:
        expected = (2.0 / lam) * math.atanh(
            math.sqrt(1.0 - 2.0 * mu * kappa / (lam * theta_t))
        )
      else:
        expected = 0.0
      self.assertAlmostEqual(float(got), expected, places=9)

  def test_row_varying_reference_point(self):
    t_ref_rows = self.t_ref * np.array([1.0, 2.0, 0.5, 1.5, 1.0, 1.0])
    for family, alpha in _FAMILY_ALPHAS:
      with self.subTest(family=family):
        kappa = saturation.apply_saturation(t_ref_rows, family, alpha)
        before = saturation.optimal_allocation(
            family, alpha, self.theta_raw, 60.0, self.t_max
        )
        after = saturation.optimal_allocation(
            family, alpha, self.theta_raw * kappa, 60.0, self.t_max,
            t_ref=t_ref_rows,
        )
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)


class VectorAlphaAllocationTest(unittest.TestCase):
  """The allocator must accept one alpha per row."""

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(2)
    self.theta = rng.uniform(1.0, 10.0, size=60)
    self.budget = 400.0
    self.t_max = 30.0

  def test_constant_vector_matches_the_scalar_path(self):
    """A constant alpha vector must allocate exactly like a scalar alpha."""
    for family, alpha in (("exp", 0.3), ("log", 2.0), ("power", 0.5)):
      with self.subTest(family=family):
        scalar = saturation.optimal_allocation(
            family, alpha, self.theta, self.budget, self.t_max
        )
        vector = saturation.optimal_allocation(
            family,
            np.full_like(self.theta, alpha),
            self.theta,
            self.budget,
            self.t_max,
        )
        np.testing.assert_allclose(scalar, vector, rtol=1e-9, atol=1e-9)

  def test_heterogeneous_alpha_is_feasible_and_uses_the_budget(self):
    """Per-row alphas must still respect the box bounds and spend the budget."""
    rng = np.random.default_rng(3)
    for family, lo, hi in (("exp", 0.1, 1.5), ("log", 0.5, 4.0)):
      with self.subTest(family=family):
        alpha = rng.uniform(lo, hi, size=self.theta.shape)
        allocation = saturation.optimal_allocation(
            family, alpha, self.theta, self.budget, self.t_max
        )
        self.assertTrue(np.all(allocation >= -1e-9))
        self.assertTrue(np.all(allocation <= self.t_max + 1e-9))
        # The budget binds: t_max * n far exceeds it, so the optimum spends it.
        self.assertAlmostEqual(
            float(np.sum(allocation)), self.budget, delta=1e-4
        )

  def test_heterogeneous_alpha_beats_a_misspecified_common_alpha(self):
    """Sanity check that the per-row alpha is actually being used."""
    rng = np.random.default_rng(4)
    alpha = rng.uniform(0.1, 1.5, size=self.theta.shape)
    correct = saturation.optimal_allocation(
        "exp", alpha, self.theta, self.budget, self.t_max
    )
    wrong = saturation.optimal_allocation(
        "exp", float(np.mean(alpha)), self.theta, self.budget, self.t_max
    )
    objective = lambda t: float(
        np.sum(self.theta * saturation.apply_saturation(t, "exp", alpha))
    )
    self.assertGreater(objective(correct), objective(wrong))


class RLossProfileTest(unittest.TestCase):
  """The profiled R-loss must recover a known alpha."""

  def test_profile_argmin_recovers_alpha(self):
    rng = np.random.default_rng(7)
    n = 1500
    family, alpha0 = "power", 0.4

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treatment = np.abs(rng.gamma(2.0, 20.0, size=n))
    f0 = saturation.apply_saturation(treatment, family, alpha0)
    confounding = 0.5 * x1 + 0.3 * x2
    outcome = (
        f0 + confounding + rng.normal(0, 0.05 * float(f0.std()), size=n)
    )

    grid = saturation.build_alpha_grid(
        {"min": 0.05, "max": 0.95, "num": 19, "scale": "linear"}, family
    )
    # The nuisance is supplied exactly so the test isolates the profiling math.
    y_res = outcome - confounding
    basis = saturation.TreatmentBasis(treatment, basis_dim=12)
    design = basis.transform(treatment)
    design = design - design.mean(axis=0, keepdims=True)
    coefs = basis.project_family(family, grid)

    losses = []
    for grid_idx in range(len(grid)):
      f_hat = design @ coefs[:, grid_idx]
      denom = float((f_hat**2).sum())
      if denom <= 0:
        losses.append(np.inf)
        continue
      theta_hat = float((f_hat * y_res).sum() / denom)
      losses.append(float(((y_res - theta_hat * f_hat) ** 2).mean()))

    alpha_star = float(grid[int(np.argmin(losses))])
    self.assertLess(
        abs(alpha_star - alpha0), float(np.diff(grid).max()) * 1.5
    )


if __name__ == "__main__":
  unittest.main()
