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

"""Tests for ``evaluation.sensitivity`` and the sensitivity contour plot.

The sensitivity analysis computes the bias bound ``B(cf_y,
cf_d) = rho_U scale sqrt(cf_y cf_d / (1 - cf_d))`` with the residual-based
``scale = sqrt(sigma2 nu2)`` (never ``se sqrt(n)``), and the robustness value
``RV = (sqrt(f^4 + 4 f^2) - f^2) / 2`` with ``f = (|theta| - 1.96 se) /
scale`` so that the contour value at ``cf_y = cf_d = RV`` is exactly 0.
Required checks: ``theta = 3, se = 1, scale = 31.62  =>  RV = 0.0324 +-
1e-4`` and ``theta = 1.5, se = 1  =>  RV = 0``.
"""

import inspect
import math
import os
import re
import types
import unittest
from unittest import mock

import doubleml as dml
import doubleml_pipeline
from doubleml_pipeline.evaluation import sensitivity
from doubleml_pipeline.visualization import plots
import numpy as np
import pandas as pd
from sklearn import linear_model


def _fake_model(theta=3.0, se=1.0, n_obs=1000, sigma2=None, nu2=None,
                n_rep=1, residual_seed=None):
  """A stand-in for a fitted ``DoubleMLPLR`` with the attributes used.

  With ``residual_seed`` the model also carries out-of-fold predictions and
  nuisance targets (shape ``(n_obs, n_rep, 1)``) so the residual-based
  ``sigma2`` / ``nu2`` can be computed.
  """
  model = types.SimpleNamespace(
      coef=np.array([theta]),
      se=np.array([se]),
      n_obs=n_obs,
      sensitivity_analysis=lambda **kwargs: None,
      sensitivity_benchmark=lambda benchmarking_set: pd.DataFrame(
          {"cf_y": [0.04], "cf_d": [0.02]}
      ),
  )
  if sigma2 is not None and nu2 is not None:
    # DoubleML shape: (1, n_rep, n_coefs).
    model.sensitivity_elements = {
        "sigma2": np.full((1, n_rep, 1), sigma2),
        "nu2": np.full((1, n_rep, 1), nu2),
    }
  if residual_seed is not None:
    rng = np.random.default_rng(residual_seed)
    v = rng.normal(0.0, 0.5, size=(n_obs, n_rep, 1))
    u = theta * v + rng.normal(0.0, 2.0, size=(n_obs, n_rep, 1))
    l_hat = rng.normal(size=(n_obs, n_rep, 1))
    m_hat = rng.normal(size=(n_obs, n_rep, 1))
    model.predictions = {"ml_l": l_hat, "ml_m": m_hat}
    model.nuisance_targets = {"ml_l": l_hat + u, "ml_m": m_hat + v}
  return model


class RobustnessValueTest(unittest.TestCase):
  """The closed-form robustness value ``RV`` and ``f_q``."""

  def test_required_example(self):
    """``theta = 3, se = 1, scale = 31.62`` gives ``RV = 0.0324 +- 1e-4``."""
    rv, f_q = sensitivity.robustness_value(3.0, 1.0, 31.62)
    self.assertAlmostEqual(f_q, (3.0 - 1.96) / 31.62, places=12)
    self.assertAlmostEqual(rv, 0.0324, delta=1e-4)
    # Neither abolished argument reproduces it: k = (|t| - t*)/|t| gave
    # 0.292 and f = (|t| - t*)/sqrt(n) depends on n, not on the scale.
    k = (3.0 - 1.96) / 3.0
    abolished = 0.5 * (math.sqrt(k**4 + 4 * k**2) - k**2)
    self.assertAlmostEqual(abolished, 0.292, delta=1e-3)
    self.assertNotAlmostEqual(rv, abolished, delta=0.1)

  def test_insignificant_effect_gives_zero(self):
    """``theta = 1.5, se = 1``: ``m = |theta| - t* se <= 0`` so ``RV = 0``."""
    rv, f_q = sensitivity.robustness_value(1.5, 1.0, 31.62)
    self.assertEqual(rv, 0.0)
    self.assertEqual(f_q, 0.0)
    rv, _ = sensitivity.robustness_value(-1.5, 1.0, 31.62)
    self.assertEqual(rv, 0.0)

  def test_rv_lies_on_the_zero_contour(self):
    """``|theta| - B(RV, RV) - 1.96 se = 0`` for every row."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(500):
      theta = float(rng.uniform(-5.0, 5.0))
      se = float(rng.uniform(0.05, 2.0))
      scale = float(rng.uniform(0.1, 50.0))
      rv, f_q = sensitivity.robustness_value(theta, se, scale)
      m = abs(theta) - sensitivity.T_STAR_ALPHA * se
      if m <= 0:
        self.assertEqual(rv, 0.0)
        continue
      checked += 1
      self.assertAlmostEqual(f_q, m / scale, places=12)
      contour = (
          abs(theta)
          - float(sensitivity.bias_bound(rv, rv, scale))
          - sensitivity.T_STAR_ALPHA * se
      )
      self.assertAlmostEqual(contour, 0.0, delta=1e-8)
      self.assertGreater(rv, 0.0)
      self.assertLess(rv, 1.0)
    self.assertGreater(checked, 100)

  def test_depends_on_the_scale_not_on_n(self):
    """Same ``theta``, ``se``, ``scale`` => same RV whatever ``n`` is."""
    rv_a, _ = sensitivity.robustness_value(3.0, 1.0, 10.0)
    rv_b, _ = sensitivity.robustness_value(300.0, 100.0, 1000.0)
    self.assertAlmostEqual(rv_a, rv_b, places=12)
    rv_small_scale, _ = sensitivity.robustness_value(3.0, 1.0, 1.0)
    self.assertGreater(rv_small_scale, rv_a)

  def test_invalid_inputs(self):
    """Non-positive ``se`` / ``scale`` are rejected."""
    with self.assertRaises(ValueError):
      sensitivity.robustness_value(3.0, 0.0, 10.0)
    with self.assertRaises(ValueError):
      sensitivity.robustness_value(3.0, 1.0, 0.0)
    with self.assertRaises(ValueError):
      sensitivity.robustness_value(3.0, 1.0, float("nan"))


class BiasBoundTest(unittest.TestCase):
  """The bias bound ``B = rho scale sqrt(cf_y cf_d / (1 - cf_d))``."""

  def test_formula(self):
    """The bound equals the closed form and is linear in ``rho``."""
    scale = 7.5
    value = sensitivity.bias_bound(0.09, 0.19, scale, rho=1.0)
    self.assertAlmostEqual(
        float(value), scale * math.sqrt(0.09 * 0.19 / (1 - 0.19)), places=12
    )
    half = sensitivity.bias_bound(0.09, 0.19, scale, rho=0.5)
    self.assertAlmostEqual(float(half), 0.5 * float(value), places=12)

  def test_broadcasts_over_grids(self):
    """Array inputs broadcast; the bound grows with ``cf_d``."""
    cf_d, cf_y = np.meshgrid(np.linspace(0, 0.5, 4), np.linspace(0, 0.5, 3))
    bound = sensitivity.bias_bound(cf_y, cf_d, 2.0)
    self.assertEqual(bound.shape, (3, 4))
    self.assertEqual(bound[0, 0], 0.0)
    self.assertTrue(np.all(np.diff(bound, axis=1)[1:] >= 0))


class ResidualMomentsTest(unittest.TestCase):
  """``sigma2 = mean((u - theta v)^2)``, ``nu2 = 1/mean(v^2)``."""

  def test_matches_direct_computation(self):
    """``residual_scale`` equals the direct moment computation."""
    model = _fake_model(theta=3.0, residual_seed=1, n_obs=400)
    u, v = sensitivity.residuals(model)
    self.assertEqual(u.shape, (400, 1))
    scale, sigma2, nu2 = sensitivity.residual_scale(u, v, 3.0)
    self.assertAlmostEqual(sigma2, float(np.mean((u - 3.0 * v) ** 2)))
    self.assertAlmostEqual(nu2, 1.0 / float(np.mean(v**2)))
    self.assertAlmostEqual(scale, math.sqrt(sigma2 * nu2))

  def test_group_moments_use_only_the_group(self):
    """A mask restricts the moments to the group; bad masks raise."""
    model = _fake_model(theta=3.0, residual_seed=2, n_obs=400)
    u, v = sensitivity.residuals(model)
    mask = np.zeros(400, dtype=bool)
    mask[:50] = True
    _, sigma2_g, nu2_g = sensitivity.residual_scale(u, v, 2.5, mask=mask)
    self.assertAlmostEqual(
        sigma2_g, float(np.mean((u[:50] - 2.5 * v[:50]) ** 2))
    )
    self.assertAlmostEqual(nu2_g, 1.0 / float(np.mean(v[:50] ** 2)))
    with self.assertRaises(ValueError):
      sensitivity.residual_scale(u, v, 2.5, mask=np.zeros(400, dtype=bool))
    with self.assertRaises(ValueError):
      sensitivity.residual_scale(u, v, 2.5, mask=np.ones(10, dtype=bool))

  def test_missing_predictions_raise(self):
    """Without stored predictions the residuals are unavailable."""
    model = _fake_model(theta=3.0)
    with self.assertRaises(ValueError):
      sensitivity.residuals(model)


class SensitivityStatsTest(unittest.TestCase):
  """The scale is always ``sqrt(sigma2 nu2)``."""

  def test_scale_from_sensitivity_elements(self):
    """The ATE row takes ``sigma2`` / ``nu2`` from the elements."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=1000, sigma2=4.0, nu2=0.25)
    stats = sensitivity.sensitivity_stats(model)
    self.assertEqual(stats.theta, 3.0)
    self.assertEqual(stats.se, 1.0)
    self.assertEqual(stats.n_obs, 1000)
    self.assertEqual(stats.sigma2, 4.0)
    self.assertEqual(stats.nu2, 0.25)
    self.assertAlmostEqual(stats.scale, math.sqrt(4.0 * 0.25))
    self.assertEqual(stats.scale_source, sensitivity.SCALE_SOURCE_ELEMENTS)
    expected_rv, _ = sensitivity.robustness_value(3.0, 1.0, 1.0)
    self.assertEqual(stats.robustness_value, expected_rv)
    record = stats.as_record()
    self.assertEqual(
        set(record),
        {"theta", "se", "robustness_value", "n_obs", "f_q", "sigma2", "nu2",
         "scale"},
    )

  def test_median_over_repetitions(self):
    """``sigma2`` / ``nu2`` are the medians over the repetitions."""
    model = _fake_model(sigma2=4.0, nu2=0.25, n_rep=3)
    model.sensitivity_elements["sigma2"][0, :, 0] = [1.0, 4.0, 100.0]
    sigma2, nu2 = sensitivity.sigma2_nu2(model)
    self.assertEqual(sigma2, 4.0)
    self.assertEqual(nu2, 0.25)

  def test_no_elements_recomputes_from_residuals_not_se_sqrt_n(self):
    """Without ``sensitivity_elements`` the moments come from ``u``, ``v``."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=400, residual_seed=3)
    stats = sensitivity.sensitivity_stats(model)
    u, v = sensitivity.residuals(model)
    expected_scale, expected_sigma2, expected_nu2 = (
        sensitivity.residual_scale(u, v, 3.0)
    )
    self.assertAlmostEqual(stats.scale, expected_scale)
    self.assertAlmostEqual(stats.sigma2, expected_sigma2)
    self.assertAlmostEqual(stats.nu2, expected_nu2)
    self.assertEqual(stats.scale_source, sensitivity.SCALE_SOURCE_RESIDUALS)
    self.assertNotAlmostEqual(stats.scale, 1.0 * math.sqrt(400), delta=1.0)

  def test_no_elements_and_no_predictions_raise(self):
    """No elements and no predictions: the row cannot be computed."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=400)
    with self.assertRaises(ValueError):
      sensitivity.sensitivity_stats(model)

  def test_gate_override_uses_the_group_moments(self):
    """GATE rows: ``theta_g``, ``se_g``, ``n_g`` and ``sigma2_g``, ``nu2_g``."""
    model = _fake_model(
        theta=3.0, se=1.0, n_obs=400, sigma2=4.0, nu2=0.25, residual_seed=4
    )
    mask = np.zeros(400, dtype=bool)
    mask[100:160] = True
    stats = sensitivity.sensitivity_stats(
        model, override_theta=2.5, override_se=0.5, group_mask=mask
    )
    self.assertEqual(stats.theta, 2.5)
    self.assertEqual(stats.se, 0.5)
    self.assertEqual(stats.n_obs, 60)
    u, v = sensitivity.residuals(model)
    expected_scale, expected_sigma2, expected_nu2 = (
        sensitivity.residual_scale(u, v, 2.5, mask=mask)
    )
    self.assertAlmostEqual(stats.scale, expected_scale)
    self.assertAlmostEqual(stats.sigma2, expected_sigma2)
    self.assertAlmostEqual(stats.nu2, expected_nu2)
    self.assertEqual(
        stats.scale_source, sensitivity.SCALE_SOURCE_RESIDUALS_GROUP
    )
    # The full-sample elements (scale 1.0) are NOT used for the group.
    self.assertNotAlmostEqual(stats.scale, 1.0, delta=1e-6)
    expected_rv, _ = sensitivity.robustness_value(2.5, 0.5, expected_scale)
    self.assertEqual(stats.robustness_value, expected_rv)

  def test_gate_override_requires_a_mask(self):
    """A GATE override without ``group_mask`` is rejected."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=400, residual_seed=5)
    with self.assertRaises(ValueError):
      sensitivity.sensitivity_stats(
          model, override_theta=2.5, override_se=0.5, override_n_obs=60
      )


class ContourPlotTest(unittest.TestCase):
  """``plots.plot_sensitivity_contour`` uses the residual-based quantities."""

  def test_returns_the_k19_record_and_writes_the_png(self):
    """The record carries the sensitivity fields and the PNG exists."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=1000, sigma2=4.0, nu2=0.25)
    # The benchmark point is passed in; the plot must NOT refit a short model
    # through sensitivity_benchmark.
    model.sensitivity_benchmark = mock.Mock(
        side_effect=AssertionError("sensitivity_benchmark must not be called")
    )
    out = os.path.join(
        os.environ.get("TEST_TMPDIR", "/tmp"), "k19_contour_total.png"
    )
    res = plots.plot_sensitivity_contour(
        model,
        out,
        "spend (Total)",
        ["X1"],
        benchmark_cf_d=0.02,
        benchmark_cf_y=0.04,
        benchmark_rho=0.5,
        benchmark_scope=sensitivity.BENCHMARK_SCOPE_WHOLE_SAMPLE,
    )
    self.assertIsNotNone(res)
    self.assertTrue(os.path.exists(out))
    model.sensitivity_benchmark.assert_not_called()
    expected_rv, expected_f_q = sensitivity.robustness_value(3.0, 1.0, 1.0)
    self.assertEqual(res["robustness_value"], expected_rv)
    self.assertEqual(res["f_q"], expected_f_q)
    self.assertEqual(res["n_obs"], 1000)
    self.assertEqual(res["sigma2"], 4.0)
    self.assertEqual(res["nu2"], 0.25)
    self.assertEqual(res["scale"], 1.0)
    self.assertEqual(res["benchmark_cf_d"], 0.02)
    self.assertEqual(res["benchmark_cf_y"], 0.04)
    self.assertEqual(res["benchmark_rho"], 0.5)
    self.assertEqual(
        res["benchmark_scope"], sensitivity.BENCHMARK_SCOPE_WHOLE_SAMPLE
    )
    # The RV lies on the zero contour.
    contour = (
        abs(res["theta"])
        - float(sensitivity.bias_bound(expected_rv, expected_rv, res["scale"]))
        - 1.96 * res["se"]
    )
    self.assertAlmostEqual(contour, 0.0, delta=1e-8)

  def test_without_benchmark_values_the_point_is_zero(self):
    """No benchmark arguments: (0, 0) is recorded and nothing is refitted."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=1000, sigma2=4.0, nu2=0.25)
    model.sensitivity_benchmark = mock.Mock(
        side_effect=AssertionError("sensitivity_benchmark must not be called")
    )
    out = os.path.join(
        os.environ.get("TEST_TMPDIR", "/tmp"), "k19_contour_nobench.png"
    )
    res = plots.plot_sensitivity_contour(model, out, "spend (Total)", ["X1"])
    self.assertIsNotNone(res)
    self.assertEqual(res["benchmark_cf_d"], 0)
    self.assertEqual(res["benchmark_cf_y"], 0)
    self.assertIsNone(res["benchmark_rho"])
    self.assertIsNone(res["benchmark_scope"])
    model.sensitivity_benchmark.assert_not_called()

  def test_gate_override_and_saturation_labels(self):
    """Labels under ``treatment_variable = "f_bar(T)"``; group scale."""
    model = _fake_model(
        theta=3.0, se=1.0, n_obs=400, sigma2=4.0, nu2=0.25, residual_seed=6
    )
    mask = np.zeros(400, dtype=bool)
    mask[:120] = True
    out = os.path.join(
        os.environ.get("TEST_TMPDIR", "/tmp"), "k19_contour_gate.png"
    )
    with mock.patch.object(
        plots.plt, "xlabel", wraps=plots.plt.xlabel
    ) as xlabel, mock.patch.object(
        plots.plt, "text", wraps=plots.plt.text
    ) as text:
      res = plots.plot_sensitivity_contour(
          model,
          out,
          "spend (Geo: G1)",
          None,
          override_theta=2.5,
          override_se=0.5,
          override_n_obs=120,
          treatment_variable=sensitivity.TREATMENT_VARIABLE_FBAR,
          t_ref=150.0,
          group_mask=mask,
      )
    self.assertIsNotNone(res)
    self.assertEqual(res["n_obs"], 120)
    u, v = sensitivity.residuals(model)
    expected_scale, _, _ = sensitivity.residual_scale(u, v, 2.5, mask=mask)
    self.assertAlmostEqual(res["scale"], expected_scale)
    expected_rv, _ = sensitivity.robustness_value(2.5, 0.5, expected_scale)
    self.assertEqual(res["robustness_value"], expected_rv)
    label = xlabel.call_args[0][0]
    self.assertIn("w.r.t.", label)
    self.assertIn("bar{f}", label)
    box = text.call_args[0][2]
    self.assertIn("increment of Y for moving the treatment from 0 to T_ref",
                  box)
    self.assertIn("150", box)
    self.assertIn("scale = sqrt(sigma2 nu2)", box)

  def test_linear_label_unchanged(self):
    """The linear version keeps the original x label."""
    model = _fake_model(theta=3.0, se=1.0, n_obs=1000, sigma2=4.0, nu2=0.25)
    out = os.path.join(
        os.environ.get("TEST_TMPDIR", "/tmp"), "k19_contour_linear.png"
    )
    with mock.patch.object(
        plots.plt, "xlabel", wraps=plots.plt.xlabel
    ) as xlabel:
      res = plots.plot_sensitivity_contour(model, out, "spend (Total)", None)
    self.assertIsNotNone(res)
    self.assertEqual(xlabel.call_args[0][0], "Treatment Partial R^2 (cf_d)")


class NoAbolishedFormulaTest(unittest.TestCase):
  """No ``|theta|`` or ``se sqrt(n)`` scale, no ``k`` argument."""

  def test_sources(self):
    """Scans the package sources for the abolished expressions."""
    root = os.path.dirname(os.path.abspath(doubleml_pipeline.__file__))
    pattern = re.compile(
        r"m_val\s*/\s*np\.abs\(theta\)"  # k = (|theta| - t* se) / |theta|
        r"|\)\s*\*\s*np\.abs\(theta\)"  # bias scaled by |theta|
        r"|k_val\s*\*\*\s*4"  # the RV evaluated at k instead of f_q
        r"|SCALE_SOURCE_SE_SQRT_N"
    )
    # The sensitivity modules must not fall back to se sqrt(n) as the scale
    # nor use (|t| - t*) / sqrt(n) as the RV argument.  The
    # package-wide scan excludes this pattern because alpha_selection.py
    # legitimately divides a standard deviation by sqrt(n_obs).
    sensitivity_pattern = re.compile(
        r"se\s*\*\s*(?:math|np)\.sqrt\("
        r"|/\s*(?:math|np)\.sqrt\(\s*(?:int\()?n_obs"
    )
    sensitivity_modules = (
        os.path.join(root, "evaluation", "sensitivity.py"),
        os.path.join(root, "visualization", "plots.py"),
    )
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
      dirnames[:] = [d for d in dirnames if d not in ("test", "__pycache__")]
      for name in filenames:
        if not name.endswith(".py"):
          continue
        path = os.path.join(dirpath, name)
        with open(path, encoding="utf-8") as fh:
          text = fh.read()
        checked += 1
        hits = sorted({m.group(0) for m in pattern.finditer(text)})
        if path in sensitivity_modules:
          hits += sorted(
              {m.group(0) for m in sensitivity_pattern.finditer(text)}
          )
        self.assertFalse(hits, f"{os.path.relpath(path, root)}: {hits}")
    self.assertGreaterEqual(checked, 10)
    source = inspect.getsource(plots.plot_sensitivity_contour)
    self.assertIn("sensitivity.sensitivity_stats", source)
    self.assertIn("sensitivity.bias_bound", source)
    self.assertIn("stats_k19.scale", source)


class GateAndSensitivityAnalysisTest(unittest.TestCase):
  """``perform_gate_and_sensitivity_analysis`` on a mocked model."""

  def test_perform_gate_and_sensitivity_analysis(self):
    """Tests sensitivity analysis function."""
    df_orig = pd.DataFrame({"gate1": ["A", "B", "A"], "gate2": ["X", "Y", "X"]})

    mock_model = mock.Mock()
    mock_model.gate.return_value = mock.Mock(summary="gate_summary")
    mock_model.sensitivity_elements = "sensitivity_elements"
    mock_model.sensitivity_benchmark.return_value = pd.DataFrame(
        {"cf_y": [0.1], "cf_d": [0.2]}
    )

    res = sensitivity.perform_gate_and_sensitivity_analysis(
        mock_model,
        df_orig,
        gate_cols=["gate1", "gate2"],
        benchmark_covariates=["cov1"],
    )

    self.assertEqual(res["gate_summary"], "gate_summary")
    self.assertEqual(res["sensitivity_elements"], "sensitivity_elements")
    self.assertIn("cov1", res["benchmarks"])
    self.assertEqual(res["benchmarks"]["cov1"]["x"], 0.2)
    self.assertEqual(res["benchmarks"]["cov1"]["y"], 0.1)


class GroupBenchmarkTest(unittest.TestCase):
  """The group-specific sensitivity benchmark."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    rng = np.random.default_rng(123)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    d = 0.5 * x1 + 0.8 * z + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + x1 - x2 + 1.5 * z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"Y": y, "D": d, "X1": x1, "X2": x2, "Z": z})
    data = dml.DoubleMLData(df, y_col="Y", d_cols="D", x_cols=["X1", "X2", "Z"])
    np.random.seed(0)
    cls.model = dml.DoubleMLPLR(
        data,
        ml_l=linear_model.LinearRegression(),
        ml_m=linear_model.LinearRegression(),
        n_folds=3,
        n_rep=1,
    )
    cls.model.fit()
    cls.expected = cls.model.sensitivity_benchmark(benchmarking_set=["Z"])
    cls.short = sensitivity.fit_short_model(cls.model, ["Z"])
    cls.u_long, cls.v_long = sensitivity.residuals(cls.model)
    cls.u_short, cls.v_short = sensitivity.residuals(cls.short)
    cls.theta_long = float(cls.model.coef[0])
    cls.theta_short = float(cls.short.coef[0])

  def test_short_model_drops_only_the_benchmark_covariate(self):
    self.assertEqual(list(self.short._dml_data.x_cols), ["X1", "X2"])  # pylint: disable=protected-access
    self.assertEqual(list(self.model._dml_data.x_cols), ["X1", "X2", "Z"])  # pylint: disable=protected-access
    # Same folds as the long model (deep copy keeps the sample splitting).
    for (tr_l, te_l), (tr_s, te_s) in zip(
        self.model.smpls[0], self.short.smpls[0]
    ):
      np.testing.assert_array_equal(tr_l, tr_s)
      np.testing.assert_array_equal(te_l, te_s)

  def test_total_row_matches_doubleml_within_1e6(self):
    """Acceptance check: whole-sample values == sensitivity_benchmark."""
    stats = sensitivity.group_benchmark(
        self.u_long, self.v_long, self.u_short, self.v_short,
        self.theta_long, self.theta_short, mask=None,
    )
    self.assertEqual(stats.scope, sensitivity.BENCHMARK_SCOPE_WHOLE_SAMPLE)
    self.assertEqual(stats.n_obs, 300)
    for ours, column in (
        (stats.cf_d, "cf_d"),
        (stats.cf_y, "cf_y"),
        (stats.rho, "rho"),
        (stats.delta_theta, "delta_theta"),
    ):
      want = float(self.expected[column].iloc[0])
      with self.subTest(column=column):
        self.assertGreater(abs(want), 0.0)
        self.assertLessEqual(abs(ours - want) / abs(want), 1e-6)

  def test_degenerate_group_equals_the_total_row(self):
    """``n_g = n`` reduces to the total row."""
    total = sensitivity.group_benchmark(
        self.u_long, self.v_long, self.u_short, self.v_short,
        self.theta_long, self.theta_short,
    )
    all_rows = sensitivity.group_benchmark(
        self.u_long, self.v_long, self.u_short, self.v_short,
        self.theta_long, self.theta_short, mask=np.ones(300, dtype=bool),
    )
    self.assertEqual(all_rows.scope, sensitivity.BENCHMARK_SCOPE_WHOLE_SAMPLE)
    self.assertEqual(all_rows.as_record(), total.as_record())

  def test_subgroup_uses_only_its_rows(self):
    """A proper subgroup has scope "group" and its own moments."""
    mask = np.zeros(300, dtype=bool)
    mask[:120] = True
    stats = sensitivity.group_benchmark(
        self.u_long, self.v_long, self.u_short, self.v_short,
        self.theta_long, self.theta_short, mask=mask,
    )
    self.assertEqual(stats.scope, sensitivity.BENCHMARK_SCOPE_GROUP)
    self.assertEqual(stats.n_obs, 120)
    _, sigma2_g, nu2_g = sensitivity.residual_scale(
        self.u_long, self.v_long, self.theta_long, mask=mask
    )
    self.assertAlmostEqual(stats.sigma2_long, sigma2_g)
    self.assertAlmostEqual(stats.nu2_long, nu2_g)
    self.assertTrue(0.0 <= stats.cf_d <= 1.0)
    self.assertTrue(0.0 <= stats.cf_y <= 1.0)
    self.assertTrue(-1.0 <= stats.rho <= 1.0)

  def test_benchmark_from_moments_is_doubleml_gain_statistics(self):
    """The closed form equals DoubleML's R^2-based formulas."""
    sigma2_l, nu2_l, sigma2_s, nu2_s, delta = 0.8, 2.0, 1.0, 1.5, 0.3
    var_y = 5.0  # cancels
    r2_long = 1.0 - sigma2_l / var_y
    r2_short = 1.0 - sigma2_s / var_y
    want_cf_y = (r2_long - r2_short) / (1.0 - r2_long)
    r2_riesz = nu2_s / nu2_l
    want_cf_d = (1.0 - r2_riesz) / r2_riesz
    want_rho = delta / math.sqrt((sigma2_s - sigma2_l) * (nu2_l - nu2_s))
    cf_d, cf_y, rho = sensitivity.benchmark_from_moments(
        sigma2_l, nu2_l, sigma2_s, nu2_s, delta
    )
    self.assertAlmostEqual(cf_y, want_cf_y, places=12)
    self.assertAlmostEqual(cf_d, want_cf_d, places=12)
    self.assertAlmostEqual(rho, min(1.0, want_rho), places=12)
    # Clipping and the degenerate-denominator convention of DoubleML.
    cf_d0, cf_y0, rho0 = sensitivity.benchmark_from_moments(
        1.0, 2.0, 0.9, 2.5, -0.2
    )
    self.assertEqual((cf_d0, cf_y0), (0.0, 0.0))
    self.assertEqual(rho0, -1.0)

  def test_fit_short_model_validates_the_covariate(self):
    with self.assertRaises(ValueError):
      sensitivity.fit_short_model(self.model, ["not_a_covariate"])
    with self.assertRaises(ValueError):
      sensitivity.fit_short_model(self.model, ["X1", "X2", "Z"])


if __name__ == "__main__":
  unittest.main()
