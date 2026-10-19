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

"""Tests for ``optimization``: the simple and the saturation-aware optimizer.

* ``optimize``, the simple ROI-threshold optimizer of the ``family=None`` and
  non-spend paths, its charts (``<prefix>_timeseries_spend_vs_est.png``, the
  ``<prefix>_<outcome>.png`` outcome chart, the three before / after charts
  with a shared y scale and half-height panels) and the yearly summary.
* The array-level saturation-aware solver
  (:func:`optimization.allocate_saturation_budget`,
  :func:`saturation.solve_budget_allocation`, :func:`saturation.roi_cap`):
  for every family and random ``theta_t`` (some ``<= 0``): ``sum_t T_t* <= K``
  and ``T_t* <= T_bar_t^eff``; with ``r > 0`` every funded unit satisfies
  ``ROI_t >= r``; with ``budget_reallocate = True`` and ``sum_t T_bar_t^eff >
  K`` the budget is spent to within ``1e-8``; the KKT condition
  ``theta_t fbar'(T_t*) = mu*`` holds (rel. err. < 1e-6) for every interior
  unit; ``roi_target`` with ``r -> 0+`` approaches "every unit at T_bar_t";
  the defaults ``threshold_roi = 1.0`` / ``budget_reallocate = True``.
* A worked example (exp, theta = (10, 5), K = 2, r = 4): exact solution
  ``T = (1.540, 0.460)``, total KPI 9.70, versus 8.65 for the abolished greedy
  rule.
* The budget upper bound ``K`` (``compute_budget``) and the theta recovery in
  KPI units (``estimate_theta_kpi_units``).
* The frame-level :func:`optimization.optimize_saturation`: allocation units,
  caps by geo x item cell, ``opt_geos`` / ``opt_items``, the ``optimized_*``
  columns, the statistics columns of ``optimization_metrics.csv`` and the
  ``treatment_types="spend"`` charts shared with the simple optimizer.

The orchestrator's validation of the optimization settings (``opt_threshold_roi``,
``budget_mode``, ``cap_quantile``, ``opt_geos``, ...) is covered by
test_workflow.py.
"""

import os
import tempfile
import unittest
from unittest import mock

from doubleml_pipeline import optimization
from doubleml_pipeline.models import saturation
import numpy as np
import pandas as pd
from scipy import optimize as scipy_optimize


class SimpleOptimizerTest(unittest.TestCase):
  """The simple ROI-threshold optimizer and its charts."""

  @mock.patch("matplotlib.figure.Figure.savefig")
  def test_optimize(self, mock_savefig):
    """Tests optimization."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "Geo": ["A", "A", "A"],
        "Item": ["X", "X", "X"],
        "Sales": [100, 200, 300],
        "Spend": [10, 20, 30],
        "estimated_incremental_KPI_Spend": [15, 25, 45],
    })

    with tempfile.TemporaryDirectory() as out_dir:
      optimization.optimize(
          df_1a_consolidated=df,
          treatment="Spend",
          treatment_amt="Spend",
          periods=("2020-01-01", "2020-01-02"),
          threshold_roi=1.0,
          out_dir=out_dir,
      )
      self.assertTrue(
          os.path.exists(
              os.path.join(out_dir, "007_optimization",
                           "optimization_metrics.csv")
          )
      )
    # Per dimension (total, geo A, item X): the spend_vs_est chart and the
    # three before / after charts (spend, incremental effect, ROI).
    written = sorted(
        os.path.basename(call.args[0]) for call in mock_savefig.call_args_list
    )
    expected = []
    for prefix in ("total", "geo_A", "item_X"):
      expected.extend([
          f"{prefix}_timeseries_spend_vs_est.png",
          f"{prefix}_timeseries_spend_before_after.png",
          f"{prefix}_timeseries_incremental_effect_before_after.png",
          f"{prefix}_timeseries_roi_before_after.png",
      ])
    self.assertEqual(written, sorted(expected))

  @mock.patch("matplotlib.figure.Figure.savefig")
  def test_optimize_writes_outcome_chart_when_outcome_col_given(
      self, mock_savefig
  ):
    """`<prefix>_<outcome>.png`: observed vs projected outcome per date."""
    del mock_savefig
    # Geo A's 2020-01-01 row (spend 10, KPI 5, ROI 0.5) is cut, so its KPI
    # leaves the projected outcome: 2020-01-01 total = 300 + (25 - 30) = 295.
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        "Geo": ["A", "B", "A"],
        "Item": ["X", "X", "X"],
        "Sales": [100.0, 200.0, 150.0],
        "Spend": [10.0, 10.0, 20.0],
        "estimated_incremental_KPI_Spend": [5.0, 25.0, 30.0],
    })
    charts = {}

    def _record(dates, series, out_path, title, start_date, end_date):
      del dates, start_date, end_date
      charts[os.path.basename(out_path)] = (title, series)

    with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
        optimization, "_timeseries_line_chart", _record
    ):
      optimization.optimize(
          df_1a_consolidated=df,
          treatment="Spend",
          treatment_amt="Spend",
          periods=("2020-01-01", "2020-01-02"),
          threshold_roi=1.0,
          outcome_col="Sales",
          out_dir=out_dir,
      )
    # One chart per dimension, named like the saturation optimizer's so the
    # HTML report finds it under `<prefix>_<outcome>.png` for both.
    self.assertEqual(
        sorted(charts),
        [
            "geo_A_Sales.png",
            "geo_B_Sales.png",
            "item_X_Sales.png",
            "total_Sales.png",
        ],
    )
    title, series = charts["total_Sales.png"]
    self.assertEqual(title, "Total Sales (Threshold=1.0)")
    self.assertEqual(
        [(color, label) for _, color, label in series],
        [
            ("red", "Original KPI (Sales)"),
            ("blue", "Estimated total KPI with optimal treatment"),
        ],
    )
    self.assertEqual(list(series[0][0]), [300.0, 150.0])
    self.assertEqual(list(series[1][0]), [295.0, 150.0])

  @mock.patch("matplotlib.figure.Figure.savefig")
  def test_optimize_applies_threshold_per_row(self, mock_savefig):
    """ROI threshold in optimize() is applied per row, not on date sums."""
    # On 2020-01-01, Geo A has ROI = 5 / 10 = 0.5 (< 1.0) and Geo B has
    # ROI = 25 / 10 = 2.5 (>= 1.0). The date-aggregated ROI is
    # (5 + 25) / (10 + 10) = 1.5 (>= 1.0), so a date-grouped rule would keep
    # both rows (reduction = 0), whereas per-row cutting drops Geo A's 10.0.
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        "Geo": ["A", "B", "A"],
        "Item": ["X", "X", "X"],
        "Sales": [100.0, 200.0, 150.0],
        "Spend": [10.0, 10.0, 20.0],
        "estimated_incremental_KPI_Spend": [5.0, 25.0, 30.0],
    })
    with tempfile.TemporaryDirectory() as out_dir:
      optimization.optimize(
          df_1a_consolidated=df,
          treatment="Spend",
          treatment_amt="Spend",
          periods=("2020-01-01", "2020-01-02"),
          threshold_roi=1.0,
          out_dir=out_dir,
      )
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
      yearly = pd.read_csv(
          os.path.join(
              out_dir, "007_optimization", optimization.YEARLY_SUMMARY_CSV
          )
      )
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertAlmostEqual(
        float(total["total_cost_of_treatment_in_periods"]), 40.0
    )
    self.assertAlmostEqual(
        float(total["reduction_of_cost_of_treatment"]), 10.0
    )
    self.assertAlmostEqual(
        float(total["optimized_total_cost_of_treatment_in_periods"]), 30.0
    )
    mock_savefig.assert_called()
    # The yearly before / after summary is written next to the metrics.
    self.assertEqual(
        list(yearly.columns), list(optimization.YEARLY_SUMMARY_COLUMNS)
    )
    total_all = yearly[
        (yearly["dimension"] == "total") & (yearly["year"] == "Total")
    ].iloc[0]
    self.assertAlmostEqual(float(total_all["spend_before"]), 40.0)
    self.assertAlmostEqual(float(total_all["incremental_kpi_before"]), 60.0)
    self.assertAlmostEqual(float(total_all["roi_before"]), 1.5)
    # Geo A's 2020-01-01 row (spend 10, KPI 5) is cut: its KPI goes too.
    self.assertAlmostEqual(float(total_all["spend_after"]), 30.0)
    self.assertAlmostEqual(float(total_all["incremental_kpi_after"]), 55.0)
    self.assertAlmostEqual(float(total_all["roi_after"]), 55.0 / 30.0)
    geo_a = yearly[(yearly["dimension"] == "geo") & (yearly["level"] == "A")]
    self.assertEqual(list(geo_a["year"]), ["2020", "Total"])
    self.assertAlmostEqual(float(geo_a.iloc[0]["spend_after"]), 20.0)

  def test_yearly_summary_frame_by_year_and_dimension(self):
    """One row per (dimension, level, year | Total) with before/after ROI."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(
            ["2022-06-01", "2022-12-01", "2023-03-01", "2023-09-01"]
        ),
        "Geo": ["A", "B", "A", "B"],
        "Item": ["X", "X", "Y", "Y"],
        "Spend": [10.0, 20.0, 30.0, 40.0],
        "opt_spend": [10.0, 0.0, 30.0, 40.0],
        "kpi": [20.0, 10.0, 60.0, 40.0],
        "opt_kpi": [20.0, 0.0, 60.0, 40.0],
    })
    frame = optimization.yearly_summary_frame(
        df,
        treatment="Spend",
        date_col="Date",
        geo_col="Geo",
        item_col="Item",
        spend_col="Spend",
        opt_spend_col="opt_spend",
        kpi_col="kpi",
        opt_kpi_col="opt_kpi",
    )
    self.assertEqual(
        list(frame.columns), list(optimization.YEARLY_SUMMARY_COLUMNS)
    )
    total = frame[frame["dimension"] == "total"].set_index("year")
    self.assertEqual(list(total.index), ["2022", "2023", "Total"])
    self.assertAlmostEqual(total.loc["2022", "spend_before"], 30.0)
    self.assertAlmostEqual(total.loc["2022", "spend_after"], 10.0)
    self.assertAlmostEqual(total.loc["2022", "roi_before"], 1.0)
    self.assertAlmostEqual(total.loc["2022", "roi_after"], 2.0)
    self.assertAlmostEqual(total.loc["2023", "spend_before"], 70.0)
    self.assertAlmostEqual(total.loc["Total", "spend_before"], 100.0)
    self.assertAlmostEqual(total.loc["Total", "incremental_kpi_after"], 120.0)
    self.assertAlmostEqual(total.loc["Total", "roi_after"], 1.5)
    # Geo B only has one row per year; a zero spend yields ROI 0, not NaN.
    geo_b = frame[
        (frame["dimension"] == "geo") & (frame["level"] == "B")
    ].set_index("year")
    self.assertAlmostEqual(geo_b.loc["2022", "spend_after"], 0.0)
    self.assertAlmostEqual(geo_b.loc["2022", "roi_after"], 0.0)
    # Items: X lives in 2022 only, Y in 2023 only.
    item_x = frame[
        (frame["dimension"] == "item") & (frame["level"] == "X")
    ]
    self.assertEqual(list(item_x["year"]), ["2022", "Total"])
    self.assertEqual(
        set(frame["dimension"]), {"total", "geo", "item"}
    )

  @mock.patch("matplotlib.pyplot.close")
  def test_spend_before_after_chart_shares_the_y_scale(self, mock_close):
    """Two stacked charts, actual spend on top, identical y limits."""
    del mock_close
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

    df_ts = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20"]),
        "Spend": [10.0, 50.0, 30.0],
        "opt_spend": [10.0, 0.0, 30.0],
    })
    with mock.patch.object(plt.Figure, "savefig") as mock_savefig:
      optimization.plot_spend_before_after(
          df_ts,
          date_col="Date",
          spend_col="Spend",
          opt_spend_col="opt_spend",
          out_path="unused.png",
          title_prefix="Total",
          spend_label="Spend (Spend)",
          threshold_roi=1.0,
      )
      mock_savefig.assert_called_once()
      fig = plt.gcf()
    axes = list(fig.axes)
    self.assertEqual(len(axes), 2)
    top, bottom = axes
    self.assertEqual(top.get_ylim(), bottom.get_ylim())
    self.assertGreaterEqual(top.get_ylim()[1], 50.0)
    self.assertGreater(top.get_position().y0, bottom.get_position().y0)
    self.assertIn("before optimization", top.get_title())
    self.assertIn("after optimization", bottom.get_title())
    self.assertIn("Threshold=1.0", bottom.get_title())
    # Top line = actual spend, bottom line = optimized spend.
    self.assertEqual(list(top.get_lines()[0].get_ydata()), [10.0, 50.0, 30.0])
    self.assertEqual(
        list(bottom.get_lines()[0].get_ydata()), [10.0, 0.0, 30.0]
    )

  def _before_after_frame(self):
    return pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20"]),
        "Spend": [10.0, 50.0, 30.0],
        "opt_spend": [10.0, 0.0, 30.0],
        "kpi": [20.0, 25.0, 60.0],
        "opt_kpi": [20.0, 0.0, 60.0],
    })

  @mock.patch("matplotlib.pyplot.close")
  def test_incremental_effect_before_after_chart(self, mock_close):
    """Red incremental effect before on top, blue after below, same scale."""
    del mock_close
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

    with mock.patch.object(plt.Figure, "savefig") as mock_savefig:
      optimization.plot_incremental_effect_before_after(
          self._before_after_frame(),
          date_col="Date",
          kpi_col="kpi",
          opt_kpi_col="opt_kpi",
          out_path="unused.png",
          title_prefix="[G0]",
          threshold_roi=1.0,
      )
      mock_savefig.assert_called_once()
      fig = plt.gcf()
    top, bottom = fig.axes
    self.assertEqual(top.get_ylim(), bottom.get_ylim())
    self.assertGreaterEqual(top.get_ylim()[1], 60.0)
    self.assertGreater(top.get_position().y0, bottom.get_position().y0)
    self.assertEqual(
        top.get_title(), "[G0] Incremental Effect before optimization"
    )
    self.assertEqual(
        bottom.get_title(),
        "[G0] Incremental Effect after optimization (Threshold=1.0)",
    )
    self.assertEqual(top.get_lines()[0].get_color(), "red")
    self.assertEqual(bottom.get_lines()[0].get_color(), "blue")
    self.assertEqual(list(top.get_lines()[0].get_ydata()), [20.0, 25.0, 60.0])
    self.assertEqual(
        list(bottom.get_lines()[0].get_ydata()), [20.0, 0.0, 60.0]
    )
    self.assertEqual(top.get_ylabel(), "Incremental KPI")

  @mock.patch("matplotlib.pyplot.close")
  def test_roi_before_after_chart(self, mock_close):
    """Per-date ROI = KPI / spend (0 without spend), threshold as reference."""
    del mock_close
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

    with mock.patch.object(plt.Figure, "savefig") as mock_savefig:
      optimization.plot_roi_before_after(
          self._before_after_frame(),
          date_col="Date",
          spend_col="Spend",
          opt_spend_col="opt_spend",
          kpi_col="kpi",
          opt_kpi_col="opt_kpi",
          out_path="unused.png",
          title_prefix="Total",
          threshold_roi=1.0,
      )
      mock_savefig.assert_called_once()
      fig = plt.gcf()
    top, bottom = fig.axes
    self.assertEqual(top.get_ylim(), bottom.get_ylim())
    self.assertEqual(top.get_title(), "Total ROI before optimization")
    self.assertEqual(
        bottom.get_title(), "Total ROI after optimization (Threshold=1.0)"
    )
    self.assertEqual(top.get_ylabel(), "ROI")
    before_line, after_line = top.get_lines()[0], bottom.get_lines()[0]
    self.assertEqual(before_line.get_color(), "red")
    self.assertEqual(after_line.get_color(), "blue")
    self.assertEqual(list(before_line.get_ydata()), [2.0, 0.5, 2.0])
    # 2020-01-13 was cut entirely (spend 0): ROI after is 0, not NaN.
    self.assertEqual(list(after_line.get_ydata()), [2.0, 0.0, 2.0])
    # The ROI threshold is a labelled horizontal reference on both panels.
    for ax in (top, bottom):
      ref = [
          ln
          for ln in ax.get_lines()
          if ln.get_label().startswith("ROI threshold")
      ]
      self.assertEqual(len(ref), 1)
      self.assertEqual(list(ref[0].get_ydata()), [1.0, 1.0])

  @mock.patch("matplotlib.pyplot.close")
  def test_before_after_panels_share_the_spend_vs_est_geometry(
      self, mock_close
  ):
    """Every stacked panel spans exactly the x extent of a single panel."""
    del mock_close
    # pylint: disable=protected-access
    fig_single, (single,) = optimization._timeseries_figure(1)
    fig_double, (top, bottom) = optimization._timeseries_figure(2)
    fig_half, (half_top, half_bottom) = optimization._timeseries_figure(
        2, optimization._TS_PANEL_HEIGHT_IN / 2.0
    )
    # pylint: enable=protected-access
    single_h = single.get_position().height * fig_single.get_figheight()
    for fig, axes, ratio in (
        (fig_double, (top, bottom), 1.0),
        (fig_half, (half_top, half_bottom), 0.5),
    ):
      self.assertEqual(fig.get_figwidth(), fig_single.get_figwidth())
      for ax in axes:
        self.assertAlmostEqual(ax.get_position().x0, single.get_position().x0)
        self.assertAlmostEqual(ax.get_position().x1, single.get_position().x1)
        # Panel height in inches relative to the single-panel chart: equal by
        # default, exactly half for the stacked before / after charts.
        self.assertAlmostEqual(
            ax.get_position().height * fig.get_figheight(), ratio * single_h
        )

  @mock.patch("matplotlib.pyplot.close")
  def test_before_after_charts_use_half_height_panels(self, mock_close):
    """Every before / after chart is half as tall per panel as spend_vs_est."""
    del mock_close
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

    frame = self._before_after_frame()
    heights = {}
    ylims = {}
    with mock.patch.object(plt.Figure, "savefig"):
      optimization.plot_incremental_effect_before_after(
          frame, "Date", "kpi", "opt_kpi", "unused.png", "Total"
      )
      fig = plt.gcf()
      heights["incremental"] = [
          ax.get_position().height * fig.get_figheight() for ax in fig.axes
      ]
      optimization.plot_roi_before_after(
          frame, "Date", "Spend", "opt_spend", "kpi", "opt_kpi",
          "unused.png", "Total",
      )
      fig = plt.gcf()
      heights["roi"] = [
          ax.get_position().height * fig.get_figheight() for ax in fig.axes
      ]
      optimization.plot_spend_before_after(
          frame, "Date", "Spend", "opt_spend", "unused.png", "Total", "Spend"
      )
      fig = plt.gcf()
      heights["spend"] = [
          ax.get_position().height * fig.get_figheight() for ax in fig.axes
      ]
      ylims["spend"] = [ax.get_ylim() for ax in fig.axes]
    # pylint: disable=protected-access
    full = optimization._TS_PANEL_HEIGHT_IN
    # pylint: enable=protected-access
    for key in ("spend", "incremental", "roi"):
      self.assertEqual(len(heights[key]), 2)
      for h in heights[key]:
        self.assertAlmostEqual(h, full / 2.0, places=6)
    # The spend panels are exactly as tall as the incremental effect panels.
    for h_spend, h_incr in zip(heights["spend"], heights["incremental"]):
      self.assertAlmostEqual(h_spend, h_incr, places=12)
    # Halving the height leaves the vertical scale alone: the y range still
    # spans the data (max spend 50 plus the 8 % head room) from 0.
    for y_min, y_max in ylims["spend"]:
      self.assertEqual(y_min, 0.0)
      self.assertAlmostEqual(y_max, 50.0 * 1.08)


_FAMILIES = (("tanh", 0.7), ("exp", 0.5), ("log", 2.0), ("power", 0.6))


_T_REF = 3.0


def _kpi(family, alpha, theta, t, t_ref):
  return float(
      np.sum(theta * saturation.normalized_saturation(t, family, alpha, t_ref))
  )


class SolverPropertiesTest(unittest.TestCase):
  """Budget, caps, ROI constraint and KKT condition of the solver."""

  def _random_problem(self, rng, n=40):
    theta = rng.normal(2.0, 1.5, size=n)  # some negative units
    theta[: n // 8] = -np.abs(theta[: n // 8])
    t_actual = np.abs(rng.normal(2.0, 1.0, size=n))
    t_bar = t_actual * rng.uniform(1.0, 2.5, size=n)
    return theta, t_actual, t_bar

  def test_fixed_reallocate_true(self):
    """Tests fixed-budget allocation with reallocation across families."""
    rng = np.random.default_rng(0)
    for family, alpha in _FAMILIES:
      for r in (0.0, 0.3, 0.8):
        with self.subTest(family=family, r=r):
          theta, t_actual, t_bar = self._random_problem(rng)
          res = optimization.allocate_saturation_budget(
              family,
              alpha,
              theta,
              t_actual,
              t_bar,
              threshold_roi=r,
              t_ref=_T_REF,
              budget_mode="fixed",
              budget_increment=0.0,
              budget_reallocate=True,
          )
          t_star = res.allocation
          k = res.budget_k
          self.assertAlmostEqual(k, float(t_actual.sum()))
          self.assertLessEqual(float(t_star.sum()), k * (1 + 1e-10))
          self.assertTrue(np.all(t_star >= 0.0))
          self.assertTrue(np.all(t_star <= res.t_bar_eff * (1 + 1e-10) + 1e-12))
          self.assertTrue(np.all(t_star[theta <= 0] == 0.0))
          # ROI constraint on every funded unit.
          funded = t_star > 1e-9
          roi = (
              theta[funded]
              * saturation.normalized_saturation(
                  t_star[funded], family, alpha, _T_REF
              )
              / t_star[funded]
          )
          self.assertTrue(np.all(roi >= r - 1e-7), (family, r, roi.min()))
          # Budget spent when the caps do not absorb it.
          if float(res.t_bar_eff[theta > 0].sum()) > k:
            self.assertLessEqual(abs(float(t_star.sum()) - k), 1e-8 * k)
            self.assertGreater(res.marginal_roi_at_optimum, 0.0)
          else:
            self.assertEqual(res.marginal_roi_at_optimum, 0.0)
          # KKT: theta fbar'(T*) = mu* on interior units.
          mu = res.marginal_roi_at_optimum
          interior = (
              (theta > 0)
              & (t_star > 1e-7)
              & (t_star < res.t_bar_eff - 1e-7 * np.maximum(res.t_bar_eff, 1.0))
          )
          if mu > 0 and np.any(interior):
            mroi = theta[interior] * saturation.normalized_derivative(
                t_star[interior], family, alpha, _T_REF
            )
            self.assertLess(float(np.max(np.abs(mroi - mu) / mu)), 1e-6)
          # Unspent budget bookkeeping.
          self.assertAlmostEqual(res.unspent_budget, k - float(t_star.sum()))
          self.assertEqual(
              res.n_units_excluded_negative_roi, int(np.sum(theta <= 0))
          )

  def test_fixed_reallocate_false_truncates_without_redistribution(self):
    """Fixed budget with budget_reallocate=False truncates at ROI cap."""
    rng = np.random.default_rng(1)
    for family, alpha in _FAMILIES:
      with self.subTest(family=family):
        theta, t_actual, t_bar = self._random_problem(rng)
        r = 0.6
        res = optimization.allocate_saturation_budget(
            family,
            alpha,
            theta,
            t_actual,
            t_bar,
            threshold_roi=r,
            t_ref=_T_REF,
            budget_mode="fixed",
            budget_reallocate=False,
        )
        # K' = K_+ (negative units removed), caps T_bar only, then cut.
        k_plus = float(t_actual[theta > 0].sum())
        self.assertAlmostEqual(res.budget_k_prime, k_plus)
        first = saturation.solve_budget_allocation(
            family, alpha, theta, k_plus, t_bar, t_ref=_T_REF
        )
        want = np.minimum(first.allocation, res.t_roi_cap)
        want[theta <= 0] = 0.0
        np.testing.assert_allclose(res.allocation, want, rtol=1e-9, atol=1e-12)
        self.assertLessEqual(float(res.allocation.sum()), res.budget_k + 1e-9)
        self.assertTrue(np.all(res.allocation <= t_bar + 1e-12))
        funded = res.allocation > 1e-9
        roi = (
            theta[funded]
            * saturation.normalized_saturation(
                res.allocation[funded], family, alpha, _T_REF
            )
            / res.allocation[funded]
        )
        self.assertTrue(np.all(roi >= r - 1e-7))
        # The cut is recorded as unspent, not redistributed.
        self.assertGreaterEqual(res.unspent_budget, -1e-9)
        if np.any(res.capped_by_roi):
          self.assertGreater(res.unspent_budget, 0.0)

  def test_roi_target(self):
    """Tests roi_target allocation, r -> 0+ limit, and invalid inputs."""
    rng = np.random.default_rng(2)
    for family, alpha in _FAMILIES:
      with self.subTest(family=family):
        theta, t_actual, t_bar = self._random_problem(rng)
        res = optimization.allocate_saturation_budget(
            family,
            alpha,
            theta,
            t_actual,
            t_bar,
            threshold_roi=0.5,
            t_ref=_T_REF,
            budget_mode="roi_target",
        )
        want = np.where(theta > 0, np.minimum(t_bar, res.t_roi_cap), 0.0)
        np.testing.assert_allclose(res.allocation, want)
        self.assertAlmostEqual(res.k_out, float(want.sum()))
        self.assertTrue(np.isnan(res.budget_k))
        self.assertTrue(np.isnan(res.unspent_budget))
        self.assertTrue(np.isnan(res.marginal_roi_at_optimum))
        # r -> 0+: every positive unit at its historical maximum.
        tiny = optimization.allocate_saturation_budget(
            family,
            alpha,
            theta,
            t_actual,
            t_bar,
            threshold_roi=1e-9,
            t_ref=_T_REF,
            budget_mode="roi_target",
        )
        np.testing.assert_allclose(
            tiny.allocation, np.where(theta > 0, t_bar, 0.0), rtol=1e-6
        )
    with self.assertRaises(ValueError):
      optimization.allocate_saturation_budget(
          "exp",
          0.5,
          np.ones(3),
          np.ones(3),
          2.0,
          threshold_roi=0.0,
          budget_mode="roi_target",
      )
    with self.assertRaises(ValueError):
      optimization.allocate_saturation_budget(
          "exp", 0.5, np.ones(3), np.ones(3), 2.0, budget_mode="greedy"
      )

  def test_roi_cap_cases(self):
    """Tests roi_cap roots, boundary conditions, and invalid thresholds."""
    # exp, alpha = 1, kappa = 1: theta = 5, r = 4 -> 5 (1 - e^-T) / T = 4,
    # whose root is T^r = 0.46421 (brentq), i.e. 0.46 to two decimals.
    cap = saturation.roi_cap("exp", 1.0, np.array([5.0]), 4.0)
    root = scipy_optimize.brentq(
        lambda t: 5.0 * (1 - np.exp(-t)) - 4.0 * t, 1e-6, 5.0
    )
    self.assertAlmostEqual(float(cap[0]), root, places=8)
    self.assertAlmostEqual(float(cap[0]), 0.46, places=2)
    roi = 5.0 * (1 - np.exp(-cap[0])) / cap[0]
    self.assertAlmostEqual(float(roi), 4.0, places=8)
    # theta fbar'(0) <= r -> 0 (exp slope at 0 is alpha).
    cap = saturation.roi_cap("exp", 1.0, np.array([3.0]), 4.0)
    self.assertEqual(float(cap[0]), 0.0)
    # r = 0 -> inf; theta <= 0 -> 0.
    cap = saturation.roi_cap("exp", 1.0, np.array([3.0, -1.0]), 0.0)
    self.assertTrue(np.isinf(cap[0]))
    self.assertEqual(float(cap[1]), 0.0)
    # Search bound: not binding below t_upper -> inf.
    cap = saturation.roi_cap("exp", 1.0, np.array([5.0]), 4.0, t_upper=0.1)
    self.assertTrue(np.isinf(cap[0]))
    # log has a FINITE slope at the origin: 1 / (alpha kappa).
    slope = saturation.marginal_effect_at_zero("log", 2.0, _T_REF)
    self.assertAlmostEqual(float(slope), 1.0 / (2.0 * np.log1p(_T_REF / 2.0)))
    self.assertTrue(np.isinf(saturation.marginal_effect_at_zero("power", 0.5)))
    cap = saturation.roi_cap("log", 2.0, np.array([0.1]), 10.0, t_ref=_T_REF)
    self.assertEqual(float(cap[0]), 0.0)
    with self.assertRaises(ValueError):
      saturation.roi_cap("exp", 1.0, np.array([1.0]), -0.1)

  def test_kkt_inversion_closed_forms(self):
    """The closed-form (fbar')^-1(mu / theta) for all four families."""
    theta = np.array([4.0, 2.5, 1.0])
    mu = 0.3
    for family, alpha in _FAMILIES:
      with self.subTest(family=family):
        alloc = saturation.kkt_allocation(
            family, alpha, theta, mu, np.inf, t_ref=_T_REF
        )
        kappa = float(
            saturation.apply_saturation(np.array([_T_REF]), family, alpha)[0]
        )
        for th, t in zip(theta, alloc):
          if (
              th
              * float(saturation.marginal_effect_at_zero(family, alpha, _T_REF))
              <= mu
          ):
            self.assertEqual(float(t), 0.0)
            continue
          if family == "tanh":
            want = (2 / alpha) * np.arctanh(
                np.sqrt(1 - 2 * mu * kappa / (alpha * th))
            )
          elif family == "exp":
            want = np.log(alpha * th / (mu * kappa)) / alpha
          elif family == "log":
            want = th / (mu * kappa) - alpha
          else:
            want = (alpha * th / (mu * kappa)) ** (1 / (1 - alpha))
          self.assertAlmostEqual(float(t), float(want), places=9)
          # and theta fbar'(T) = mu there.
          self.assertAlmostEqual(
              float(
                  th
                  * saturation.normalized_derivative(
                      np.array([t]), family, alpha, _T_REF
                  )[0]
              ),
              mu,
              places=9,
          )

  def test_roi_cap_never_exceeds_the_budget(self):
    """The ROI threshold acts as a per-unit cap T_t^r in one KKT pass."""
    rng = np.random.default_rng(0)
    theta = np.abs(rng.normal(2.0, 0.7, size=40))
    actual = np.abs(rng.normal(50.0, 20.0, size=40))
    budget = float(actual.sum())
    t_bar = float(actual.max())
    res = optimization.allocate_saturation_budget(
        family="power",
        alpha=0.5,
        theta=theta,
        t_actual=actual,
        t_bar=t_bar,
        threshold_roi=1.0,
        budget_mode="fixed",
        budget_reallocate=True,
    )
    alloc = res.allocation
    self.assertLessEqual(float(alloc.sum()), budget + 1e-6)
    self.assertTrue(bool(np.all(alloc >= -1e-9)))
    self.assertTrue(bool(np.all(alloc <= res.t_bar_eff + 1e-9)))
    funded = alloc > 1e-9
    roi = theta[funded] * saturation.apply_saturation(
        alloc[funded], "power", 0.5) / alloc[funded]
    self.assertTrue(bool(np.all(roi >= 1.0 - 1e-7)))

  def test_defaults_are_threshold_one_and_reallocate(self):
    """``allocate_saturation_budget`` defaults to r = 1.0, reallocate = True."""
    theta = np.array([10.0, -2.0])
    t_actual = np.array([1.0, 1.0])
    res = optimization.allocate_saturation_budget(
        "exp", 1.0, theta, t_actual, t_bar=10.0
    )
    # With budget_reallocate=True, K' = K = 2.0 (including negative unit's 1.0).
    self.assertAlmostEqual(res.budget_k, 2.0)
    self.assertAlmostEqual(res.budget_k_prime, 2.0)
    self.assertAlmostEqual(res.k_out, 2.0, places=7)


class WorkedExampleTest(unittest.TestCase):
  """Worked example: exp family, theta = (10, 5), K = 2, r = 4."""

  def test_exact_solution_beats_the_abolished_greedy_rule(self):
    """Verifies the exact KKT solution outperforms the abolished greedy rule."""
    theta = np.array([10.0, 5.0])
    t_actual = np.array([1.0, 1.0])  # K = 2 with budget_increment = 0
    res = optimization.allocate_saturation_budget(
        "exp",
        1.0,
        theta,
        t_actual,
        t_bar=10.0,
        threshold_roi=4.0,
        t_ref=None,
        budget_mode="fixed",
        budget_reallocate=True,
    )
    # Exact solution: unit 2 sits on its ROI cap T_2^r (root of
    # 5 (1 - e^-T) / T = 4, i.e. 0.4642), unit 1 takes the rest of K = 2.
    t2 = scipy_optimize.brentq(
        lambda t: 5.0 * (1 - np.exp(-t)) - 4.0 * t, 1e-6, 5.0
    )
    np.testing.assert_allclose(res.allocation, [2.0 - t2, t2], atol=1e-6)
    # The solution rounded to two decimals: (1.54, 0.46).
    np.testing.assert_allclose(np.round(res.allocation, 2), [1.54, 0.46])
    kpi = float(np.sum(theta * (1 - np.exp(-res.allocation))))
    self.assertAlmostEqual(kpi, 9.70, places=2)
    self.assertAlmostEqual(res.k_out, 2.0, places=9)
    self.assertEqual(res.n_units_capped_by_roi, 1)
    self.assertTrue(res.capped_by_roi[1])
    # Every funded unit meets ROI >= 4.
    roi = theta * (1 - np.exp(-res.allocation)) / res.allocation
    self.assertTrue(np.all(roi >= 4.0 - 1e-7), roi)
    # mu* = theta_1 f'(T_1*) on the interior unit.
    self.assertAlmostEqual(
        res.marginal_roi_at_optimum, 10.0 * np.exp(-res.allocation[0]), places=6
    )
    # The abolished greedy rule: unconstrained argmin violates the floor on
    # unit 2 (ROI 3.67 < 4), unit 2 is dropped and the whole budget goes to
    # unit 1 -> KPI 10 (1 - e^-2) = 8.65 < 9.70.
    greedy_kpi = 10.0 * (1 - np.exp(-2.0))
    self.assertAlmostEqual(greedy_kpi, 8.65, places=2)
    self.assertGreater(kpi, greedy_kpi)
    # The same allocation with the normalized fbar and theta in fbar units.
    kappa = 1 - np.exp(-_T_REF)
    res_norm = optimization.allocate_saturation_budget(
        "exp",
        1.0,
        theta * kappa,
        t_actual,
        t_bar=10.0,
        threshold_roi=4.0,
        t_ref=_T_REF,
        budget_mode="fixed",
        budget_reallocate=True,
    )
    np.testing.assert_allclose(res_norm.allocation, res.allocation, rtol=1e-6)

  def test_no_greedy_loop_exists(self):
    self.assertFalse(
        hasattr(saturation, "allocate_with_" + "roi_" + "threshold")
    )


class BudgetAndThetaTest(unittest.TestCase):
  """Budget upper bound K and theta recovery."""

  def test_compute_budget_applies_the_increment(self):
    actual = np.array([10.0, 20.0, 30.0])
    self.assertAlmostEqual(
        optimization.compute_budget(actual, 0.0), 60.0
    )
    self.assertAlmostEqual(
        optimization.compute_budget(actual, 0.1), 66.0
    )

  def test_theta_recovered_in_kpi_units(self):
    rng = np.random.default_rng(3)
    n = 300
    treatment = np.where(
        rng.random(n) < 0.2, 0.0, np.abs(rng.gamma(2.0, 30.0, size=n))
    )
    f_values = saturation.apply_saturation(treatment, "power", 0.5)
    kpi = 2.5 * f_values
    theta_hat = optimization.estimate_theta_kpi_units(
        kpi,
        f_values,
        None,
        pd.Series(np.resize(["a", "b"], n)),
    )
    positive = f_values > 0
    self.assertLess(float(np.abs(theta_hat[positive] - 2.5).max()), 1e-6)
    # Zero-treatment rows must be imputed rather than left at zero.
    self.assertLess(float(np.abs(theta_hat[~positive] - 2.5).max()), 1e-6)


def _frame(n_geo=3, n_items=2, n_weeks=12, seed=0):
  """Builds a synthetic geo x item x week DataFrame and theta vector."""
  rng = np.random.default_rng(seed)
  dates = pd.date_range("2022-01-03", periods=n_weeks, freq="7D")
  rows = []
  for g in range(n_geo):
    for i in range(n_items):
      for d in dates:
        spend = float(rng.gamma(2.0, 10.0))
        rows.append({
            "date": d,
            "geo": f"G{g}",
            "item": f"I{i}",
            "spend": spend,
            "sales": 100.0 + 3.0 * np.sqrt(spend) + rng.normal(),
        })
  df = pd.DataFrame(rows)
  theta = rng.uniform(1.0, 4.0, len(df))
  theta[:5] = -1.0
  df["estimated_incremental_KPI_spend"] = (
      theta
      * saturation.normalized_saturation(
          df["spend"].to_numpy(), "exp", 0.05, 40.0
      )
  )
  return df, theta


class OptimizeSaturationFrameTest(unittest.TestCase):
  """The allocation and its outputs at the frame level."""

  def _run(self, df, theta, out_dir, **kwargs):
    with mock.patch.object(optimization, "_plot_saturation_optimization"):
      return optimization.optimize_saturation(
          df_1a_consolidated=df,
          treatment="spend",
          treatment_amt="spend",
          periods=(df["date"].iloc[0], df["date"].iloc[-1]),
          family="exp",
          alpha=0.05,
          theta=theta,
          t_ref=40.0,
          outcome_col="sales",
          date_col="date",
          geo_col="geo",
          item_col="item",
          sales_col="sales",
          out_dir=out_dir,
          **kwargs,
      )

  def test_units_caps_and_statistics(self):
    """Verifies optimized columns, cell caps, budget limits, and metrics."""
    df, theta = _frame()
    with tempfile.TemporaryDirectory() as out_dir:
      out = self._run(
          df, theta, out_dir, threshold_roi=0.1, budget_reallocate=True
      )
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
    for col in (
        "optimized_spend",
        "optimized_spend_transformed",
        "optimized_spend_spend",
        "optimized_incremental_KPI_spend",
        "optimized_roi_spend",
        "optimized_sales",
    ):
      self.assertIn(col, out.columns)
    # Cap: the maximum actual treatment within the geo x item cell.
    cell_max = df.groupby(["geo", "item"])["spend"].transform("max")
    self.assertTrue((out["optimized_spend"] <= cell_max + 1e-9).all())
    # Budget never exceeded; negative units at 0.
    self.assertLessEqual(out["optimized_spend"].sum(), df["spend"].sum() + 1e-8)
    self.assertTrue((out.loc[theta <= 0, "optimized_spend"] == 0).all())
    for col in (
        "budget_mode",
        "budget_reallocate",
        "K",
        "K_out",
        "unspent_budget",
        "marginal_roi_at_optimum",
        "n_units",
        "n_units_excluded_negative_roi",
        "n_units_capped_by_roi",
    ):
      self.assertIn(col, metrics.columns)
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertEqual(total["budget_mode"], "fixed")
    self.assertAlmostEqual(total["K"], df["spend"].sum())
    self.assertAlmostEqual(total["K_out"], out["optimized_spend"].sum())
    self.assertAlmostEqual(total["unspent_budget"], total["K"] - total["K_out"])
    self.assertEqual(int(total["n_units"]), len(df))
    self.assertEqual(int(total["n_units_excluded_negative_roi"]), 5)
    self.assertGreaterEqual(total["marginal_roi_at_optimum"], 0.0)
    # The incremental-KPI / ROI columns are still adjacent to their totals.
    cols = list(metrics.columns)
    self.assertEqual(
        cols[cols.index("total_cost_of_treatment_in_periods") + 1],
        "incremental_KPI_spend",
    )
    self.assertEqual(
        cols[cols.index("optimized_total_cost_of_treatment_in_periods") + 1],
        "optimized_incremental_KPI_spend",
    )

  def test_opt_geos_and_periods_restrict_the_units(self):
    """Verifies opt_geos, opt_items, and periods restrict target units."""
    df, theta = _frame()
    with tempfile.TemporaryDirectory() as out_dir:
      with mock.patch.object(optimization, "_plot_saturation_optimization"):
        out = optimization.optimize_saturation(
            df_1a_consolidated=df,
            treatment="spend",
            treatment_amt="spend",
            periods=(df["date"].iloc[0], df["date"].unique()[5]),
            family="exp",
            alpha=0.05,
            theta=theta,
            t_ref=40.0,
            outcome_col="sales",
            opt_geos=["G0", "G2"],
            opt_items=["I1"],
            budget_reallocate=True,
            date_col="date",
            geo_col="geo",
            item_col="item",
            sales_col="sales",
            out_dir=out_dir,
        )
        metrics = pd.read_csv(
            os.path.join(
                out_dir, "007_optimization", "optimization_metrics.csv"
            )
        )
    in_scope = (
        (df["date"] <= df["date"].unique()[5])
        & df["geo"].isin(["G0", "G2"])
        & (df["item"] == "I1")
    ).to_numpy()
    # Rows outside the target set keep the pre-optimization values.
    np.testing.assert_allclose(
        out.loc[~in_scope, "optimized_spend"], df.loc[~in_scope, "spend"]
    )
    np.testing.assert_allclose(
        out.loc[~in_scope, "optimized_incremental_KPI_spend"],
        df.loc[~in_scope, "estimated_incremental_KPI_spend"],
    )
    self.assertNotEqual(
        float(out.loc[in_scope, "optimized_spend"].sum()),
        float(df.loc[in_scope, "spend"].sum()) + 1.0,
    )
    # Budget K = actual spend of the target units only.
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertAlmostEqual(total["K"], df.loc[in_scope, "spend"].sum())
    self.assertEqual(int(total["n_units"]), int(in_scope.sum()))
    self.assertEqual(set(metrics["target_geo"]), {"all", "G0", "G2"})
    self.assertEqual(set(metrics["target_item"]), {"all", "I1"})

  def test_roi_target_mode_in_the_frame(self):
    """Verifies frame-level optimization metrics and caps in roi_target mode."""
    df, theta = _frame()
    with tempfile.TemporaryDirectory() as out_dir:
      out = self._run(
          df, theta, out_dir, threshold_roi=0.05, budget_mode="roi_target"
      )
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertEqual(total["budget_mode"], "roi_target")
    self.assertTrue(pd.isna(total["K"]))
    self.assertTrue(pd.isna(total["unspent_budget"]))
    self.assertTrue(pd.isna(total["marginal_roi_at_optimum"]))
    self.assertAlmostEqual(total["K_out"], out["optimized_spend"].sum())
    cell_max = df.groupby(["geo", "item"])["spend"].transform("max")
    self.assertTrue((out["optimized_spend"] <= cell_max + 1e-9).all())

  def test_cap_quantile_default_is_the_cell_maximum(self):
    """cap_quantile = 1.0 reproduces the historical cell-maximum cap."""
    df, theta = _frame()
    keys = df[["geo", "item"]]
    cell_max = df.groupby(["geo", "item"])["spend"].transform("max")
    np.testing.assert_allclose(
        optimization._cell_cap(df["spend"].to_numpy(), keys, 1.0),  # pylint: disable=protected-access
        cell_max.to_numpy(),
    )
    self.assertEqual(optimization.DEFAULT_CAP_QUANTILE, 1.0)
    with tempfile.TemporaryDirectory() as out_dir:
      base = self._run(df, theta, out_dir, threshold_roi=0.1)
    with tempfile.TemporaryDirectory() as out_dir:
      explicit = self._run(df, theta, out_dir, threshold_roi=0.1,
                           cap_quantile=1.0)
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
    np.testing.assert_allclose(
        explicit["optimized_spend"], base["optimized_spend"]
    )
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertEqual(total["cap_quantile"], 1.0)
    self.assertEqual(total["reallocate_within"], "all")
    self.assertEqual(total["assumptions"], "no_carryover")
    for col in ("n_units_at_support_cap", "marginal_roi_at_optimum_min",
                "marginal_roi_at_optimum_max"):
      self.assertIn(col, metrics.columns)
    # One KKT problem: min = median = max of mu*.
    self.assertAlmostEqual(total["marginal_roi_at_optimum_min"],
                           total["marginal_roi_at_optimum"])
    self.assertAlmostEqual(total["marginal_roi_at_optimum_max"],
                           total["marginal_roi_at_optimum"])

  def test_cell_cap_quantile_of_positive_spend_floored_at_actual(self):
    """T_bar_t = max(Q_q(cell, T > 0), T_t^actual); zeros do not shift Q_q."""
    values = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 100.0])
    keys = pd.DataFrame({"geo": ["A"] * 7, "item": ["x"] * 7})
    cap = optimization._cell_cap(values, keys, 0.5)  # pylint: disable=protected-access
    q50 = float(np.quantile([10.0, 20.0, 30.0, 40.0, 100.0], 0.5))  # 30
    np.testing.assert_allclose(cap, np.maximum(q50, values))
    # Rows above the quantile keep their own actual spend as the cap.
    self.assertEqual(cap[-1], 100.0)
    self.assertEqual(cap[5], 40.0)
    # Rows at or below it share the cell quantile (zero-spend rows included).
    self.assertTrue(np.all(cap[:5] == q50))
    # A cell without positive spend has cap 0; no keys -> one cell.
    np.testing.assert_allclose(
        optimization._cell_cap(np.zeros(3), keys.iloc[:3], 0.9), 0.0  # pylint: disable=protected-access
    )
    np.testing.assert_allclose(
        optimization._cell_cap(values, None, 1.0), np.maximum(100.0, values)  # pylint: disable=protected-access
    )
    for bad in (0.0, 1.5, -0.1):
      with self.assertRaises(ValueError):
        optimization._cell_cap(values, keys, bad)  # pylint: disable=protected-access

  def test_cap_quantile_caps_every_unit_except_the_outlier(self):
    """q = 0.95: units are capped at Q_0.95(cell); the outlier day at itself."""
    df, theta = _frame()
    theta = np.abs(theta) + 1.0  # every unit positive -> caps matter
    # One outlier day in cell (G0, I0).
    cell = (df["geo"] == "G0") & (df["item"] == "I0")
    outlier = df.index[cell][3]
    df.loc[outlier, "spend"] = 50.0 * df.loc[cell, "spend"].max()
    q_cell = (
        df.assign(pos=df["spend"].where(df["spend"] > 0))
        .groupby(["geo", "item"])["pos"]
        .transform(lambda s: s.dropna().quantile(0.95))
    )
    with tempfile.TemporaryDirectory() as out_dir:
      # A large budget increment and no ROI cap so that the support cap is
      # the binding constraint for (almost) every unit.
      out = self._run(
          df, theta, out_dir, threshold_roi=0.0, budget_increment=5.0,
          cap_quantile=0.95,
      )
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
    cap = np.maximum(q_cell.to_numpy(), df["spend"].to_numpy())
    self.assertTrue((out["optimized_spend"] <= cap + 1e-9).all())
    # The outlier is never pushed above its own actual spend ...
    self.assertLessEqual(out.loc[outlier, "optimized_spend"],
                         df.loc[outlier, "spend"] + 1e-9)
    # ... while every other unit of its cell stays at or below Q_0.95 < actual
    # outlier spend.
    others = cell & (df.index != outlier)
    self.assertTrue(
        (out.loc[others, "optimized_spend"] <= q_cell[others] + 1e-9).all()
    )
    self.assertLess(float(q_cell[outlier]), float(df.loc[outlier, "spend"]))
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertEqual(total["cap_quantile"], 0.95)
    # With a 6x budget almost every unit sits on its support cap.
    self.assertGreater(int(total["n_units_at_support_cap"]), len(df) // 2)
    self.assertEqual(int(total["n_units_capped_by_roi"]), 0)

  def test_reallocate_within_date_keeps_daily_totals(self):
    """"date": one KKT problem per date; spend never moves across days."""
    df, theta = _frame()
    with tempfile.TemporaryDirectory() as out_dir:
      by_date = self._run(
          df, theta, out_dir, threshold_roi=0.1, reallocate_within="date"
      )
      metrics = pd.read_csv(
          os.path.join(out_dir, "007_optimization", "optimization_metrics.csv")
      )
    with tempfile.TemporaryDirectory() as out_dir:
      pooled = self._run(df, theta, out_dir, threshold_roi=0.1)
    actual_daily = df.groupby("date")["spend"].sum()
    opt_daily = by_date.groupby("date")["optimized_spend"].sum()
    # Per-date budget K_d = sum_{t in d} T_t: never exceeded on any day.
    self.assertTrue((opt_daily <= actual_daily + 1e-8).all())
    # The pooled problem does move spend across days.
    pooled_daily = pooled.groupby("date")["optimized_spend"].sum()
    self.assertTrue((pooled_daily > actual_daily + 1e-6).any())
    # Negative units are still zeroed; caps still hold.
    self.assertTrue((by_date.loc[theta <= 0, "optimized_spend"] == 0).all())
    cell_max = df.groupby(["geo", "item"])["spend"].transform("max")
    self.assertTrue((by_date["optimized_spend"] <= cell_max + 1e-9).all())
    total = metrics[
        (metrics["target_geo"] == "all") & (metrics["target_item"] == "all")
    ].iloc[0]
    self.assertEqual(total["reallocate_within"], "date")
    self.assertAlmostEqual(total["K"], df["spend"].sum())
    self.assertAlmostEqual(total["K_out"], by_date["optimized_spend"].sum())
    # mu* is per date: min <= median <= max, not all equal in general.
    self.assertLessEqual(total["marginal_roi_at_optimum_min"],
                         total["marginal_roi_at_optimum"] + 1e-12)
    self.assertLessEqual(total["marginal_roi_at_optimum"],
                         total["marginal_roi_at_optimum_max"] + 1e-12)
    self.assertEqual(int(total["n_units"]), len(df))
    with self.assertRaises(ValueError):
      with tempfile.TemporaryDirectory() as out_dir:
        self._run(df, theta, out_dir, reallocate_within="week")


class SpendChartsTest(unittest.TestCase):
  """The ``treatment_types="spend"`` optimizer charts."""

  def _frame(self):
    dates = pd.date_range("2022-01-03", periods=8, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "spend": np.linspace(10.0, 80.0, 8),
        "optimized_spend": np.linspace(12.0, 90.0, 8),
        "estimated_incremental_KPI_spend": np.linspace(1.0, 8.0, 8),
        "optimized_incremental_KPI_spend": np.linspace(1.5, 9.0, 8),
        "sales": np.linspace(100.0, 180.0, 8),
        "optimized_sales": np.linspace(105.0, 195.0, 8),
    })

  def _call(self, out_dir):
    df_ts = self._frame()
    # pylint: disable=protected-access
    optimization._plot_spend_charts(
        df_ts=df_ts,
        date_col="date",
        prefix="geo_G0",
        title_prefix="Geo G0",
        treatment="spend",
        opt_treatment_col="optimized_spend",
        outcome_col="sales",
        opt_outcome_col="optimized_sales",
        opt_dir=out_dir,
        threshold_roi=0.0,
        start_date=df_ts["date"].iloc[0],
        end_date=df_ts["date"].iloc[-1],
    )
    # pylint: enable=protected-access

  def test_writes_two_charts(self):
    # The incremental effect is no longer a separate two-line chart: it is the
    # geo_G0_timeseries_incremental_effect_before_after.png chart shared with
    # the simple optimizer (see BeforeAfterChartsTest).
    with tempfile.TemporaryDirectory() as out_dir:
      self._call(out_dir)
      self.assertEqual(
          sorted(os.listdir(out_dir)),
          [
              "geo_G0_original_vs_optimal_treatment.png",
              "geo_G0_sales.png",
          ],
      )

  def test_chart_colors_and_labels(self):
    calls = []

    def _record(dates, series, out_path, title, start_date, end_date):
      del dates, title, start_date, end_date
      calls.append((
          os.path.basename(out_path),
          [(color, label) for _, color, label in series],
      ))

    with mock.patch.object(
        optimization, "_timeseries_line_chart", _record
    ):
      self._call("/does/not/need/to/exist")

    self.assertEqual(
        calls,
        [
            (
                "geo_G0_original_vs_optimal_treatment.png",
                [
                    ("purple", "Original treatment (spend)"),
                    ("green", "Optimal treatment"),
                ],
            ),
            (
                "geo_G0_sales.png",
                [
                    ("red", "Original KPI (sales)"),
                    ("blue", "Estimated total KPI with optimal treatment"),
                ],
            ),
        ],
    )

  def test_missing_columns_skip_charts(self):
    df_ts = self._frame().drop(columns=["optimized_sales"])
    with tempfile.TemporaryDirectory() as out_dir:
      # pylint: disable=protected-access
      optimization._plot_spend_charts(
          df_ts=df_ts,
          date_col="date",
          prefix="total",
          title_prefix="Total",
          treatment="spend",
          opt_treatment_col="optimized_spend",
          outcome_col="sales",
          opt_outcome_col="optimized_sales",
          opt_dir=out_dir,
          threshold_roi=0.0,
          start_date=df_ts["date"].iloc[0],
          end_date=df_ts["date"].iloc[-1],
      )
      # pylint: enable=protected-access
      self.assertEqual(
          sorted(os.listdir(out_dir)),
          ["total_original_vs_optimal_treatment.png"],
      )


class BeforeAfterChartsTest(unittest.TestCase):
  """The before / after charts shared with the simple optimizer."""

  def test_saturation_optimizer_writes_the_before_after_set(self):
    """Every dimension gets the spend / incremental effect / ROI charts."""
    dates = pd.date_range("2022-01-03", periods=4, freq="W-MON")
    df_work = pd.DataFrame({
        "date": list(dates) * 2,
        "geo": ["G0"] * 4 + ["G1"] * 4,
        "spend": np.linspace(10.0, 80.0, 8),
        "optimized_spend_spend": np.linspace(12.0, 90.0, 8),
        "optimized_spend": np.linspace(12.0, 90.0, 8),
        "estimated_incremental_KPI_spend": np.linspace(1.0, 8.0, 8),
        "optimized_incremental_KPI_spend": np.linspace(1.5, 9.0, 8),
    })
    written = []
    with tempfile.TemporaryDirectory() as out_dir, mock.patch(
        "matplotlib.figure.Figure.savefig",
        side_effect=lambda path, *a, **k: written.append(
            os.path.basename(path)
        ),
    ):
      # pylint: disable=protected-access
      optimization._plot_saturation_optimization(
          df_work,
          treatment="spend",
          treatment_amt="spend",
          kpi_col="estimated_incremental_KPI_spend",
          opt_treatment_col="optimized_spend",
          opt_spend_col="optimized_spend_spend",
          opt_kpi_col="optimized_incremental_KPI_spend",
          date_col="date",
          geo_col="geo",
          item_col=None,
          sales_col=None,
          outcome_col=None,
          opt_outcome_col=None,
          threshold_roi=1.0,
          opt_dir=out_dir,
          start_date=dates[0],
          end_date=dates[-1],
      )
      # pylint: enable=protected-access
      self.assertIn(optimization.YEARLY_SUMMARY_CSV, os.listdir(out_dir))
    for prefix in ("total", "geo_G0", "geo_G1"):
      self.assertIn(f"{prefix}_original_vs_optimal_treatment.png", written)
      self.assertIn(f"{prefix}_timeseries_spend_before_after.png", written)
      self.assertIn(
          f"{prefix}_timeseries_incremental_effect_before_after.png", written
      )
      self.assertIn(f"{prefix}_timeseries_roi_before_after.png", written)
    self.assertNotIn("total_incremental_effect.png", written)


if __name__ == "__main__":
  unittest.main()
