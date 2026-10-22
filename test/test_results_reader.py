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

"""Tests for the agent results reader."""

import datetime
import json
import pathlib
import shutil
import tempfile
import unittest

from doubleml_pipeline.agent import results_reader
from doubleml_pipeline.agent.tools import base
from doubleml_pipeline.agent.tools import cate_model
import numpy as np
import pandas as pd

_PHASE4 = "step4_consolidated_results"
_FULL_DF = f"{_PHASE4}/consolidated_1a_full_df.csv"


def _write_results(
    root: pathlib.Path, prefix: str = "", **metadata_overrides
) -> None:
  """Writes a small, hand-computable set of pipeline outputs to `root`.

  T1 (spend): rows (Jan A, Jan B, Feb A, Feb B) spend 10/0/10/20 and KPI
  20/0/10/40 with +-5 bounds. T2 (percentage, spend T2_cost 5/10/5/10): KPI
  5/-10/2/3, which sums to zero.
  """
  t1, t2 = f"{prefix}T1", f"{prefix}T2"
  phase4 = root / _PHASE4
  for sub in ("graph/004_cate", "graph/006_sensitivity"):
    (phase4 / sub).mkdir(parents=True, exist_ok=True)

  full = pd.DataFrame({
      "date": ["2024-01-15", "2024-01-15", "2024-02-15", "2024-02-15"],
      "store": ["A", "B", "A", "B"],
      "store_name": ["Alpha", "Beta", "Alpha", "Beta"],
      "sku": ["X", "X", "Y", "Y"],
      "demand": [1.0, 2.0, 3.0, 4.0],
      "media": [5.0, 1.0, 1.0, 5.0],
      "channel": ["web", "web", "store", "app"],
      "sales": [100.0, 200.0, 300.0, 400.0],
      t1: [10.0, 0.0, 10.0, 20.0],
      t2: [0.1, 0.2, 0.1, 0.2],
      f"{t2}_cost": [5.0, 10.0, 5.0, 10.0],
  })
  for treatment, kpi, margin in (
      (t1, [20.0, 0.0, 10.0, 40.0], [5.0, 5.0, 5.0, 5.0]),
      (t2, [5.0, -10.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0]),
  ):
    col = f"estimated_incremental_KPI_{treatment}"
    full[col] = kpi
    full[f"{col}_2.5%"] = full[col] - margin
    full[f"{col}_97.5%"] = full[col] + margin
  for suffix in ("", "_2.5%", "_97.5%"):
    full[f"total_estimated_incremental_KPI{suffix}"] = (
        full[f"estimated_incremental_KPI_{t1}{suffix}"]
        + full[f"estimated_incremental_KPI_{t2}{suffix}"]
    )
  full.to_csv(root / _FULL_DF, index=False)

  pd.DataFrame({
      "d_col": [t1] * 3 + [t2] * 3,
      "Model Group": ["lgbm_lgbm_3"] * 6,
      "total_estimated_incremental_KPI": [60.0, 70.0, 80.0, -10.0, 5.0, 10.0],
  }).to_csv(phase4 / "consolidated_3a_detail.csv", index=False)

  spline = "bs(demand, df=3, degree=2, include_intercept=False)"
  pd.DataFrame({
      "model": ["rep_1", "median"],
      "d_col": [t1, t1],
      "cate_structure": ["auto_additive"] * 2,
      "cate_df": [3.0, 3.0],
      "cate_degree": [2.0, 2.0],
      "cate_coef_dict": [
          json.dumps({"Intercept_coef": 1.0}),
          json.dumps({
              "Intercept_coef": 1.2,
              f"{spline}[0]_coef": 0.5,
              f"{spline}[1]_coef": -0.1,
          }),
      ],
      "cate_stderr_dict": [
          json.dumps({"Intercept_stderr": 0.5}),
          json.dumps({
              "Intercept_stderr": 0.4,
              f"{spline}[0]_stderr": 0.2,
              f"{spline}[1]_stderr": 0.3,
          }),
      ],
      "cate_pvalue_dict": [
          json.dumps({"Intercept_pvalue": 0.9}),
          json.dumps({
              "Intercept_pvalue": 0.01,
              f"{spline}[0]_pvalue": 0.02,
              f"{spline}[1]_pvalue": 0.5,
          }),
      ],
  }).to_csv(phase4 / "graph/004_cate/cate_coef_dict_pooling.csv", index=False)

  pd.DataFrame({
      "theta": [1.5, 1.2, 2.0],
      "se": [0.2, 0.4, 0.3],
      "robustness_value": [0.3, 0.05, 0.3],
      "benchmark_cf_d": [0.1, 0.1, 0.1],
      "benchmark_cf_y": [0.2, 0.2, 0.2],
      "treatment": [t1, t1, t1],
      "scope": ["total", "geo", "geo"],
      "group": ["Total", "A", "B"],
      "benchmark_covariate": ["demand"] * 3,
  }).to_csv(
      phase4 / "graph/006_sensitivity/sensitivity_metrics.csv", index=False
  )

  metadata = {
      "treatment_cols": [t1, t2],
      "outcome_col": "sales",
      "date_col": "date",
      "date_col_min": "2024-01-15",
      "date_col_max": "2024-02-15",
      "x_cols_list": [["demand"], ["demand"]],
      "covariates_for_cate_list": [["demand"], []],
      "treatment_types": {t1: "spend", t2: "percentage"},
      "treatment_spend_cols": {t1: t1, t2: f"{t2}_cost"},
      "geo_col": "store",
      "item_col": "sku",
      "geo_name_col": "store_name",
      "item_name_col": None,
      "geos": ["A", "B"],
      "items": ["X", "Y"],
      "peak_month": [12],
      "opt_threshold_roi": 1.0,
      "ground_truth_existence": False,
      **metadata_overrides,
  }
  (root / "metadata.json").write_text(json.dumps(metadata))


def _write_daily_results(root: pathlib.Path) -> None:
  """Writes a two-week daily run of one store where only Mondays and

  Fridays respond to the spend.
  """
  (root / _PHASE4).mkdir(parents=True, exist_ok=True)
  dates = pd.date_range("2024-01-01", periods=14, freq="D")  # Starts Monday.
  full = pd.DataFrame({
      "date": dates.strftime("%Y-%m-%d"),
      "store": "A",
      "sales": 100.0,
      "spend": 10.0,
  })
  kpi = "estimated_incremental_KPI_spend"
  full[kpi] = np.where(dates.dayofweek.isin([0, 4]), 30.0, 2.0)
  full[f"{kpi}_2.5%"] = full[kpi] - 1.0
  full[f"{kpi}_97.5%"] = full[kpi] + 1.0
  full.to_csv(root / _FULL_DF, index=False)
  (root / "metadata.json").write_text(
      json.dumps({
          "treatment_cols": ["spend"],
          "outcome_col": "sales",
          "date_col": "date",
          "x_cols_list": [[]],
          "covariates_for_cate_list": [[]],
          "treatment_spend_cols": {"spend": "spend"},
          "geo_col": "store",
          "geos": ["A"],
          "opt_threshold_roi": 1.0,
      })
  )


class CausalResultsReaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)
    _write_results(self.root)
    self.reader = results_reader.CausalResultsReader(self.root)

  def _rewrite(
      self, **metadata_overrides
  ) -> results_reader.CausalResultsReader:
    _write_results(self.root, **metadata_overrides)
    return results_reader.CausalResultsReader(self.root)

  def test_missing_metadata_raises(self):
    with self.assertRaises(FileNotFoundError):
      results_reader.CausalResultsReader(self.root / "missing")

  def test_tools_are_bound_methods_with_docstrings(self):
    tools = self.reader.tools()
    self.assertEqual(
        [tool.__name__ for tool in tools], list(results_reader.TOOL_NAMES)
    )
    for tool in tools:
      self.assertTrue(tool.__name__.startswith("get_"))
      self.assertTrue(tool.__doc__)

  def test_every_tool_returns_strict_json(self):
    calls = {
        "get_analysis_overview": {},
        "get_treatment_effects": {},
        "get_effects_by": {"by": "geo"},
        "get_segment_profile": {"dimension": "geo", "segment": "A"},
        "get_model_diagnostics": {},
    }
    for tool in self.reader.tools():
      with self.subTest(tool=tool.__name__):
        if tool.__name__ == "get_budget_optimization":
          # This two-treatment run was not optimized.
          self.assertEqual(tool("T1")["status"], "unavailable")
          continue
        result = tool(**calls[tool.__name__])
        self.assertEqual(result["status"], "ok", result)
        json.dumps(result, allow_nan=False)

  # --- get_analysis_overview ---

  def test_overview(self):
    (overview,) = self.reader.get_analysis_overview()["runs"]
    self.assertEqual(overview["run"], self.root.resolve().name)
    self.assertEqual(overview["outcome"], "sales")
    self.assertEqual(overview["threshold_roi"], 1.0)
    self.assertEqual(
        overview["treatments"][1],
        {
            "name": "T2",
            "type": "percentage",
            "spend_column": "T2_cost",
            "is_lagged": False,
            "saturation_family": None,
            "adjusted_for": ["demand"],
            "cate_covariates": [],
        },
    )
    data = overview["data"]
    self.assertEqual(data["rows"], 4)
    self.assertEqual(data["frequency"], "monthly")
    self.assertEqual(
        data["segment_dimensions"]["geo"], {"column": "store", "segments": 2}
    )
    keys = overview["effects_by_keys"]
    self.assertEqual(keys["segments"], ["geo", "item"])
    self.assertNotIn("day_of_week", keys["periods"])
    self.assertIn("media", keys["columns"])
    self.assertNotIn("estimated_incremental_KPI_T1", keys["columns"])
    self.assertEqual(
        overview["unavailable_analyses"], ["get_budget_optimization"]
    )

  def test_without_threshold_there_is_no_verdict(self):
    reader = self._rewrite(opt_threshold_roi=None)
    (overview,) = reader.get_analysis_overview()["runs"]
    t1 = reader.get_treatment_effects()["runs"][0]["treatments"][0]
    by_geo = reader.get_effects_by("geo", treatment="T1")
    self.assertIsNone(overview["threshold_roi"])
    self.assertNotIn("roi_vs_threshold", t1)
    self.assertIsNone(by_geo["threshold_summary"])

  # --- get_treatment_effects ---

  def test_treatment_effects(self):
    """Per-treatment and combined totals match the hand-computed fixture."""
    (result,) = self.reader.get_treatment_effects()["runs"]
    t1, t2 = result["treatments"]
    self.assertEqual(result["outcome_total"], 1000.0)
    self.assertEqual(t1["incremental_kpi"], 70.0)
    self.assertEqual(t1["incremental_kpi_low"], 50.0)
    self.assertEqual(t1["incremental_kpi_high"], 90.0)
    self.assertEqual(t1["effect_direction"], "positive")
    self.assertEqual(t1["effect_per_unit"], 1.75)
    self.assertEqual(t1["roi_on_spend"], 1.75)
    self.assertEqual(
        (t1["roi_on_spend_low"], t1["roi_on_spend_high"]), (1.25, 2.25)
    )
    self.assertEqual(t1["share_of_outcome"], 0.07)
    self.assertEqual(t1["treated_observations"], 3)
    self.assertEqual(t1["share_of_total_incremental_kpi"], 1.0)
    self.assertEqual(t2["effect_direction"], "zero")
    self.assertEqual(t2["roi_on_spend"], 0.0)
    self.assertEqual(result["all_treatments"]["incremental_kpi"], 70.0)
    self.assertNotIn("effect_per_unit", result["all_treatments"])

  def test_treatment_effects_reliability_signals(self):
    (result,) = self.reader.get_treatment_effects()["runs"]
    t1, t2 = result["treatments"]
    # T1's ROI interval [1.25, 2.25] is above the threshold of 1.
    self.assertEqual(t1["roi_vs_threshold"], "worth_investing")
    # T2's ROI interval [-0.67, 0.67] is below it.
    self.assertEqual(t2["roi_vs_threshold"], "not_worth_investing")
    self.assertTrue(t1["low_sample"])  # 3 treated rows.
    self.assertEqual(t1["plausibility_flags"], [])
    self.assertNotIn("roi_vs_threshold", result["all_treatments"])

  def test_roi_vs_threshold_undetermined_when_interval_straddles(self):
    reader = self._rewrite(opt_threshold_roi=2.0)
    t1 = reader.get_treatment_effects()["runs"][0]["treatments"][0]
    self.assertEqual(t1["roi_vs_threshold"], "undetermined")

  def test_plausibility_flags_effect_larger_than_outcome(self):
    full = pd.read_csv(self.root / _FULL_DF)
    kpi = "estimated_incremental_KPI_T1"
    full.loc[0, [kpi, f"{kpi}_2.5%", f"{kpi}_97.5%"]] = [150.0, 140.0, 160.0]
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    by_item = reader.get_effects_by("item", treatment="T1")

    flagged = {g["group"]: g["plausibility_flags"] for g in by_item["groups"]}
    # Item X sums to KPI 150 against sales 300, so it is fine as a whole...
    self.assertEqual(flagged, {"X": [], "Y": []})
    # ...but the store A x item X cell has KPI 150 against sales 100.
    by_geo = reader.get_effects_by("geo", treatment="T1", cross_with="item")
    cell = next(
        c
        for c in by_geo["cells"]
        if (c["row_group"], c["col_group"]) == ("A", "X")
    )
    self.assertEqual(
        cell["plausibility_flags"], ["incremental_kpi_exceeds_outcome"]
    )

  def test_kpi_without_spend_only_flags_spend_treatments(self):
    full = pd.read_csv(self.root / _FULL_DF)
    # Store B, January: T1 (a spend treatment) is 0 but gets a KPI; T2 (a
    # percentage treatment) has no cost but a KPI, which is legitimate.
    full.loc[1, "estimated_incremental_KPI_T1"] = 3.0
    full.loc[1, "T2_cost"] = 0.0
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    def january_b(treatment):
      cells = reader.get_effects_by(
          "geo", treatment=treatment, cross_with="month"
      )["cells"]
      return next(
          c["plausibility_flags"]
          for c in cells
          if (c["row_group"], c["col_group"]) == ("B", "2024-01")
      )

    self.assertEqual(january_b("T1"), ["kpi_without_spend"])
    self.assertEqual(january_b("T2"), [])

  def test_mixed_sign_treatment_has_no_per_unit_effect(self):
    full = pd.read_csv(self.root / _FULL_DF)
    full.loc[1, "T1"] = -10.0  # A price-gap-like negative input in store B.
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    by_store = {
        g["group"]: g
        for g in reader.get_effects_by("geo", treatment="T1")["groups"]
    }

    self.assertIsNone(by_store["B"]["effect_per_unit"])
    self.assertEqual(by_store["A"]["effect_per_unit"], 1.5)

  def test_interval_orders_swapped_row_bounds(self):
    """Rows below the treatment's zero point have swapped bounds."""
    full = pd.read_csv(self.root / _FULL_DF)
    kpi = "estimated_incremental_KPI_T1"
    full.loc[3, [f"{kpi}_2.5%", f"{kpi}_97.5%"]] = [45.0, 35.0]
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    t1 = reader.get_treatment_effects()["runs"][0]["treatments"][0]

    self.assertEqual(
        (t1["incremental_kpi_low"], t1["incremental_kpi_high"]), (50.0, 90.0)
    )

  # --- get_effects_by: segments ---

  def test_effects_by_geo_ranks_with_names_and_verdicts(self):
    result = self.reader.get_effects_by(
        "geo", treatment="T1", sort_by="effect_per_unit"
    )
    self.assertEqual(
        result["grouping"], {"key": "geo", "kind": "segment", "column": "store"}
    )
    self.assertEqual(
        result["window"],
        {
            "page": 1,
            "pages": 1,
            "page_size": 50,
            "groups_total": 2,
            "groups_shown": 2,
            "first": "B",
            "last": "A",
        },
    )
    self.assertEqual(
        result["data_period"], {"start": "2024-01-15", "end": "2024-02-15"}
    )
    self.assertNotIn("filter", result)
    high, low = result["groups"]
    self.assertEqual((high["group"], high["name"]), ("B", "Beta"))
    self.assertEqual(high["effect_per_unit"], 2.0)
    self.assertEqual(low["group"], "A")
    self.assertEqual(
        result["spread"],
        {"metric": "roi_on_spend", "min": 1.5, "median": 1.75, "max": 2.0},
    )
    # B: ROI [1.5, 2.5] clears 1; A: [1.0, 2.0] touches it.
    self.assertEqual(
        result["threshold_summary"],
        {
            "worth_investing": ["B"],
            "not_worth_investing": [],
            "undetermined": 1,
        },
    )
    self.assertEqual(
        result["ci_separation"],
        {"metric": "roi_on_spend", "separated_pairs": 0},
    )
    # Lists follow the ranking order; both stores have fewer than 30 rows.
    self.assertEqual(result["low_sample_groups"], ["B", "A"])
    self.assertIsNone(result["heterogeneity_evidence"])

  def test_effects_by_pages_in_presentation_order(self):
    page1 = self.reader.get_effects_by("geo", treatment="T1", page_size=1)
    page2 = self.reader.get_effects_by(
        "geo", treatment="T1", page_size=1, page=2
    )
    self.assertEqual(
        (page1["window"]["pages"], page1["window"]["first"]), (2, "B")
    )
    self.assertEqual([g["group"] for g in page2["groups"]], ["A"])
    # Summaries cover all groups whatever the page.
    self.assertEqual(page2["threshold_summary"]["worth_investing"], ["B"])
    self.assertEqual(
        self.reader.get_effects_by("geo", page=3, page_size=1)["status"],
        "invalid_argument",
    )

  def test_effects_by_date_window(self):
    result = self.reader.get_effects_by(
        "geo", treatment="T1", start="2024-02-01"
    )
    self.assertEqual(result["filter"], {"start": "2024-02-01"})
    self.assertEqual(
        {g["group"]: g["incremental_kpi"] for g in result["groups"]},
        {"A": 10.0, "B": 40.0},
    )
    crossed = self.reader.get_effects_by(
        "geo", treatment="T1", cross_with="item", end="2024-01-31"
    )
    self.assertEqual([c["col_group"] for c in crossed["cells"]], ["X", "X"])
    for kwargs in (
        {"start": "yesterday"},
        {"start": "2025-01-01"},
    ):
      with self.subTest(**kwargs):
        self.assertEqual(
            self.reader.get_effects_by("geo", **kwargs)["status"],
            "invalid_argument",
        )
    no_dates = self._rewrite(date_col=None)
    self.assertEqual(
        no_dates.get_effects_by("geo", start="2024-01-01")["status"],
        "invalid_argument",
    )

  def test_ci_separation_finds_non_overlapping_groups(self):
    full = pd.read_csv(self.root / _FULL_DF)
    kpi = "estimated_incremental_KPI_T1"
    # Store B's effect moves far above store A's interval.
    full.loc[3, [kpi, f"{kpi}_2.5%", f"{kpi}_97.5%"]] = [200.0, 190.0, 210.0]
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    separation = reader.get_effects_by("geo", treatment="T1")["ci_separation"]

    self.assertEqual(separation["separated_pairs"], 1)
    self.assertEqual(
        (
            separation["largest_gap"]["higher_group"],
            separation["largest_gap"]["lower_group"],
        ),
        ("B", "A"),
    )

  def test_effects_by_explains_unmodeled_differences(self):
    drivers = self.reader.get_effects_by("geo", treatment="T1")[
        "effect_drivers"
    ]
    self.assertEqual(drivers["cate_covariates"], {"T1": ["demand"]})
    self.assertFalse(drivers["dimension_modeled"])
    self.assertIn("only through ['demand']", drivers["explanation"])

    constant = self.reader.get_effects_by("geo", treatment="T2")
    self.assertIn("constant", constant["effect_drivers"]["explanation"])

    modeled = self._rewrite(covariates_for_cate_list=[["store"], ["store"]])
    self.assertTrue(
        modeled.get_effects_by("geo")["effect_drivers"]["dimension_modeled"]
    )

  def test_effects_by_all_treatments_has_no_per_unit_metrics(self):
    result = self.reader.get_effects_by("item")
    self.assertEqual(result["treatment"], "all")
    self.assertNotIn("spread", result)
    self.assertIsNone(result["threshold_summary"])
    self.assertIsNone(result["ci_separation"])
    self.assertIsNone(result["low_sample_groups"])
    self.assertEqual(
        self.reader.get_effects_by("item", sort_by="roi_on_spend")["status"],
        "invalid_argument",
    )

  def test_effects_by_validates_arguments(self):
    for kwargs in (
        {"by": "region"},
        {"by": "geo", "treatment": "T3"},
        {"by": "geo", "page_size": 0},
        {"by": "demand", "bins": 1},
        {"by": "geo", "cross_with": "geo"},
        {"by": "demand", "within": "sku"},
        {"by": "channel", "within": "geo"},
    ):
      with self.subTest(**kwargs):
        result = self.reader.get_effects_by(**kwargs)
        self.assertEqual(result["status"], "invalid_argument", result)

  def test_effects_by_unconfigured_dimension(self):
    reader = self._rewrite(item_col=None)
    self.assertEqual(reader.get_effects_by("item")["status"], "unavailable")

  # --- get_effects_by: periods ---

  def test_effects_by_month(self):
    result = self.reader.get_effects_by("month", treatment="T1")
    self.assertEqual(
        [(g["group"], g["incremental_kpi"]) for g in result["groups"]],
        [("2024-01", 20.0), ("2024-02", 50.0)],
    )
    self.assertEqual(result["highest_period"], "2024-02")
    self.assertEqual(result["configured_peak_months"], [12])
    self.assertFalse(result["effect_drivers"]["dimension_modeled"])

  def test_effects_by_month_of_year(self):
    result = self.reader.get_effects_by("month_of_year", treatment="T1")
    self.assertEqual([g["group"] for g in result["groups"]], [1, 2])

  def test_effects_by_period_without_date_is_unavailable(self):
    reader = self._rewrite(date_col=None)
    self.assertEqual(reader.get_effects_by("month")["status"], "unavailable")

  def test_day_of_week_needs_daily_data(self):
    result = self.reader.get_effects_by("day_of_week", treatment="T1")
    self.assertEqual(result["status"], "unavailable")
    self.assertIn("monthly", result["reason"])

  def test_day_of_week_on_daily_data(self):
    _write_daily_results(self.root)
    reader = results_reader.CausalResultsReader(self.root)

    (overview,) = reader.get_analysis_overview()["runs"]
    result = reader.get_effects_by("day_of_week")

    self.assertEqual(overview["data"]["frequency"], "daily")
    self.assertIn("day_of_week", overview["effects_by_keys"]["periods"])
    self.assertEqual(
        [g["group"] for g in result["groups"]],
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    by_day = {g["group"]: g for g in result["groups"]}
    self.assertEqual(by_day["Mon"]["roi_on_spend"], 3.0)
    self.assertEqual(by_day["Wed"]["roi_on_spend"], 0.2)
    self.assertEqual(by_day["Mon"]["roi_vs_threshold"], "worth_investing")
    self.assertEqual(by_day["Wed"]["roi_vs_threshold"], "not_worth_investing")
    self.assertEqual(
        result["threshold_summary"]["not_worth_investing"],
        ["Tue", "Wed", "Thu", "Sat", "Sun"],
    )

  # --- get_effects_by: columns ---

  def test_effects_by_numeric_column_bins_by_quantile(self):
    result = self.reader.get_effects_by("demand", treatment="T1", bins=2)
    self.assertEqual(result["grouping"]["kind"], "bin")
    self.assertEqual(result["grouping"]["binning"], "quantiles")
    self.assertEqual(
        result["grouping"]["cut_points"],
        {"Q1": {"low": 1.0, "high": 2.5}, "Q2": {"low": 2.5, "high": 4.0}},
    )
    q1, q2 = result["groups"]
    self.assertEqual((q1["group"], q1["effect_per_unit"]), ("Q1", 2.0))
    self.assertEqual(q2["group"], "Q2")
    self.assertAlmostEqual(q2["effect_per_unit"], 50 / 30)
    self.assertEqual([g["observations"] for g in result["groups"]], [2, 2])
    self.assertEqual(result["heterogeneity_evidence"], "strong (p=0.04)")
    self.assertEqual(result["column_role"], "cate_covariate")
    self.assertTrue(result["effect_drivers"]["dimension_modeled"])

  def test_effects_by_column_that_is_not_a_cate_covariate(self):
    result = self.reader.get_effects_by("media", treatment="T1", bins=2)
    self.assertIsNone(result["heterogeneity_evidence"])
    self.assertEqual(result["column_role"], "unused")
    self.assertFalse(result["effect_drivers"]["dimension_modeled"])

  def test_effects_by_categorical_column_pools_rare_values(self):
    result = self.reader.get_effects_by("channel", treatment="T1", bins=2)
    self.assertEqual(result["grouping"]["binning"], "categories")
    self.assertEqual(
        {g["group"]: g["observations"] for g in result["groups"]},
        {"web": 2, "(other)": 2},
    )
    self.assertNotIn("cut_points", result["grouping"])

  def test_effects_by_categorical_column_drops_missing_values(self):
    full = pd.read_csv(self.root / _FULL_DF)
    full.loc[3, "channel"] = None
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    result = reader.get_effects_by("channel", treatment="T1", bins=2)

    # Two categories remain, so nothing is pooled and the NaN row is gone.
    self.assertEqual(
        {g["group"]: g["observations"] for g in result["groups"]},
        {"web": 2, "store": 1},
    )

  def test_effects_by_low_cardinality_numeric_keeps_values(self):
    result = self.reader.get_effects_by("demand", treatment="T1", bins=4)
    self.assertEqual(result["grouping"]["binning"], "distinct_values")
    self.assertEqual(
        [g["group"] for g in result["groups"]], [1.0, 2.0, 3.0, 4.0]
    )

  def test_effects_by_constant_column_is_one_bin(self):
    full = pd.read_csv(self.root / _FULL_DF)
    full["media"] = 7.0
    full.to_csv(self.root / _FULL_DF, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    result = reader.get_effects_by("media", treatment="T1", within="geo")

    self.assertEqual([g["group"] for g in result["groups"]], ["Q1"])
    self.assertEqual(
        result["grouping"]["cut_points"]["A"],
        {"Q1": {"low": 7.0, "high": 7.0}},
    )

  def test_effects_by_within_segment_quantiles(self):
    result = self.reader.get_effects_by(
        "media", treatment="T1", bins=2, within="geo"
    )
    grouping = result["grouping"]
    self.assertEqual(grouping["within"], "geo")
    # Each store has media 5 and 1, so Q2 is its own high-media row.
    self.assertEqual(
        grouping["cut_points"]["A"],
        {"Q1": {"low": 1.0, "high": 3.0}, "Q2": {"low": 3.0, "high": 5.0}},
    )
    by_bin = {g["group"]: g["incremental_kpi"] for g in result["groups"]}
    self.assertEqual(by_bin, {"Q1": 10.0, "Q2": 60.0})

  def test_effects_by_cross_with(self):
    result = self.reader.get_effects_by(
        "geo", treatment="T1", cross_with="demand", bins=2
    )
    self.assertEqual(result["row_groups"], ["A", "B"])
    self.assertEqual(result["col_groups"], ["Q1", "Q2"])
    cells = {(c["row_group"], c["col_group"]): c for c in result["cells"]}
    self.assertEqual(
        sorted(cells), [("A", "Q1"), ("A", "Q2"), ("B", "Q1"), ("B", "Q2")]
    )
    self.assertEqual(cells[("B", "Q2")]["incremental_kpi"], 40.0)
    self.assertEqual(cells[("B", "Q2")]["roi_vs_threshold"], "worth_investing")
    # Store A's only January row and store B's February row both clear 1.
    self.assertEqual(
        result["threshold_summary"]["worth_investing"],
        [["A", "Q1"], ["B", "Q2"]],
    )
    self.assertEqual(
        result["column_role"], {"geo": "unused", "demand": "cate_covariate"}
    )
    self.assertEqual(
        result["heterogeneity_evidence"]["demand"], "strong (p=0.04)"
    )
    self.assertFalse(result["effect_drivers"]["geo"]["dimension_modeled"])

  # --- get_model_diagnostics ---

  def test_model_diagnostics_robustness(self):
    """Repetition stability and sensitivity flags per scope."""
    result = self.reader.get_model_diagnostics("T1")
    (stability,) = result["repetition_stability"]
    self.assertEqual(stability["treatment"], "T1")
    self.assertEqual(stability["repetitions"], 3)
    self.assertEqual(stability["share_positive"], 1.0)
    self.assertEqual(stability["median_incremental_kpi"], 70.0)

    sensitivity = result["confounding_sensitivity"]
    self.assertTrue(sensitivity["overall"][0]["robust_to_benchmark_confounder"])
    (geo,) = sensitivity["by_segment"]
    self.assertEqual((geo["segments"], geo["robust_segments"]), (2, 1))
    self.assertEqual(geo["least_robust"][0]["group"], "A")
    self.assertEqual(geo["least_robust"][0]["group_name"], "Alpha")

  def test_model_diagnostics_cate_model(self):
    result = self.reader.get_model_diagnostics()
    (run,) = result["runs"]
    model = run["cate_model"]["T1"]
    self.assertEqual(model["structure"], "auto_additive")
    self.assertEqual(model["covariates"], ["demand"])
    self.assertEqual(model["covariate_evidence"], {"demand": "strong (p=0.04)"})
    self.assertEqual(
        [(t["covariate"], t["coef"], t["pvalue"]) for t in model["terms"]],
        [
            ("Intercept", 1.2, 0.01),
            ("demand", 0.5, 0.02),
            ("demand", -0.1, 0.5),
        ],
    )
    distribution = model["effect_per_unit_distribution"]
    self.assertEqual(distribution["treated_observations"], 3)
    self.assertEqual(distribution["p50"], 2.0)
    self.assertEqual(distribution["share_negative"], 0.0)
    self.assertEqual(run["cate_model"]["T2"]["status"], "unavailable")

  def test_model_diagnostics_without_benchmark_is_unknown(self):
    """Zero benchmark strengths mean no benchmark, not a robust estimate."""
    path = self.root / _PHASE4 / "graph/006_sensitivity/sensitivity_metrics.csv"
    sensitivity = pd.read_csv(path)
    sensitivity["benchmark_covariate"] = None
    sensitivity[["benchmark_cf_d", "benchmark_cf_y"]] = 0.0
    sensitivity.to_csv(path, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    (run,) = reader.get_model_diagnostics()["runs"]
    overall = run["confounding_sensitivity"]["overall"]

    self.assertIsNone(overall[0]["robust_to_benchmark_confounder"])

  def test_model_diagnostics_reports_missing_sections(self):
    (
        self.root / _PHASE4 / "graph/006_sensitivity/sensitivity_metrics.csv"
    ).unlink()
    reader = results_reader.CausalResultsReader(self.root)
    result = reader.get_model_diagnostics()
    self.assertEqual(result["status"], "ok")
    (run,) = result["runs"]
    self.assertEqual(len(run["repetition_stability"]), 2)
    self.assertEqual(run["confounding_sensitivity"]["status"], "unavailable")

  # --- get_segment_profile ---

  def test_segment_profile_by_code_and_name(self):
    by_code = self.reader.get_segment_profile("geo", "B", treatment="T1")
    by_name = self.reader.get_segment_profile("geo", "Beta", treatment="T1")
    self.assertEqual(by_code, by_name)
    self.assertEqual((by_code["segment"], by_code["name"]), ("B", "Beta"))
    self.assertEqual(by_code["observations"], 2)
    self.assertEqual(by_code["outcome"]["share_of_total"], 0.6)
    (treatment,) = by_code["treatments"]
    self.assertEqual(treatment["treated_share"], 0.5)
    self.assertEqual(treatment["mean_when_treated"], 20.0)
    self.assertEqual(treatment["share_of_spend"], 0.5)
    (demand,) = by_code["covariates"]
    self.assertEqual(demand["column"], "demand")
    self.assertEqual(demand["segment_mean"], 3.0)
    self.assertEqual(demand["ratio_to_overall"], 1.2)
    self.assertAlmostEqual(
        demand["gap_in_row_sd"], 0.5 / np.std([1, 2, 3, 4], ddof=1)
    )
    self.assertEqual(by_code["period"]["distinct_dates"], 2)

  def test_segment_profile_extra_and_unknown_columns(self):
    result = self.reader.get_segment_profile(
        "geo", "A", treatment="T1", columns=["media", "channel"]
    )
    self.assertEqual(
        [c["column"] for c in result["covariates"]],
        ["demand", "media", "channel"],
    )
    self.assertEqual(
        result["covariates"][2], {"column": "channel", "note": "not numeric"}
    )
    for kwargs in (
        {"dimension": "geo", "segment": "Z"},
        {
            "dimension": "geo",
            "segment": "A",
            "columns": ["estimated_incremental_KPI_T1"],
        },
        {"dimension": "region", "segment": "A"},
    ):
      with self.subTest(**kwargs):
        self.assertEqual(
            self.reader.get_segment_profile(**kwargs)["status"],
            "invalid_argument",
        )

  def test_missing_consolidated_table_is_unavailable(self):
    (self.root / _FULL_DF).unlink()
    reader = results_reader.CausalResultsReader(self.root)
    self.assertEqual(reader.get_treatment_effects()["status"], "unavailable")


class HelpersTest(unittest.TestCase):
  """Module-level helpers that shape every tool result."""

  def test_to_json_safe(self):
    value = {
        "float_noise": np.float64(0.1) + np.float64(0.2),
        "infinite": float("inf"),
        "nan": float("nan"),
        "na": pd.NA,
        "nat": pd.NaT,
        "integer": np.int64(3),
        "flag": np.bool_(True),
        "timestamp": pd.Timestamp("2024-01-15"),
        "date": datetime.date(2024, 1, 15),
        "period": pd.Period("2024-01", freq="M"),
        "interval": pd.Interval(0.5, 1.5),
        "nested": (1, [np.float32(2.5)]),
    }

    safe = base.to_json_safe(value)

    self.assertEqual(
        safe,
        {
            "float_noise": 0.3,
            "infinite": None,
            "nan": None,
            "na": None,
            "nat": None,
            "integer": 3,
            "flag": True,
            "timestamp": "2024-01-15 00:00:00",
            "date": "2024-01-15",
            "period": "2024-01",
            "interval": "(0.5, 1.5]",
            "nested": [1, [2.5]],
        },
    )
    json.dumps(safe, allow_nan=False)

  def test_covariate_pvalue(self):
    spline = "bs(total_demand, df=3, degree=2, include_intercept=False)"
    term_pvalues = {
        f"{spline}[0]": 0.01,
        f"{spline}[1]": 0.2,
        f"{spline}[2]": float("nan"),
        "holiday[T.True]": 0.3,
        "holiday_flag": 0.001,
        "C(region)[T.West]": 0.04,
        "Intercept": 0.0,
        "weather": None,
    }

    cases = {
        "spline terms, Bonferroni over finite values": ("total demand", 0.02),
        "binary term": ("holiday", 0.3),
        "categorical term": ("region", 0.04),
        "no matching term": ("price", None),
        "only missing p-values": ("weather", None),
    }
    for name, (covariate, expected) in cases.items():
      with self.subTest(name):
        self.assertEqual(
            cate_model.covariate_pvalue(covariate, term_pvalues), expected
        )

  def test_evidence_label(self):
    self.assertEqual(base.evidence_label(0.0004), "strong (p<0.001)")
    self.assertEqual(base.evidence_label(0.03), "strong (p=0.03)")
    self.assertEqual(base.evidence_label(0.07), "weak (p=0.07)")
    self.assertEqual(base.evidence_label(0.12), "none (p=0.12)")
    self.assertIsNone(base.evidence_label(None))
    self.assertIsNone(base.evidence_label(float("nan")))

  def test_infer_data_frequency(self):
    cases = {
        "daily": pd.date_range("2024-01-01", periods=10, freq="D"),
        "weekly": pd.date_range("2024-01-01", periods=10, freq="W-MON"),
        "monthly": pd.date_range("2024-01-01", periods=10, freq="MS"),
        "irregular": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-02-20"]),
    }
    for expected, dates in cases.items():
      with self.subTest(expected):
        self.assertEqual(base.infer_data_frequency(pd.Series(dates)), expected)
    self.assertEqual(
        base.infer_data_frequency(pd.Series(["2024-01-01"])), "irregular"
    )


class MultiRunReaderTest(unittest.TestCase):
  """Results of several pipeline runs, for example one per channel."""

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)
    _write_results(self.root / "tv", prefix="TV_")
    _write_results(self.root / "flyer", prefix="Flyer_")
    self.reader = results_reader.CausalResultsReader(
        {"tv": self.root / "tv", "flyer": self.root / "flyer"}
    )

  def test_effects_are_grouped_by_run_and_never_summed_across_runs(self):
    runs = self.reader.get_treatment_effects()["runs"]

    self.assertEqual([r["run"] for r in runs], ["tv", "flyer"])
    self.assertEqual(
        [t["treatment"] for t in runs[1]["treatments"]],
        ["Flyer_T1", "Flyer_T2"],
    )
    self.assertEqual(runs[1]["all_treatments"]["incremental_kpi"], 70.0)

  def test_matches_single_run_reader(self):
    single = results_reader.CausalResultsReader(self.root / "flyer")

    self.assertEqual(
        self.reader.get_effects_by("geo", treatment="Flyer_T1")["groups"],
        single.get_effects_by("geo", treatment="Flyer_T1")["groups"],
    )

  def test_routes_treatments_to_their_run(self):
    trend = self.reader.get_effects_by("month", treatment="Flyer_T1")
    diagnostics = self.reader.get_model_diagnostics("TV_T1")
    profile = self.reader.get_segment_profile("geo", "A", treatment="TV_T1")

    self.assertEqual((trend["run"], trend["treatment"]), ("flyer", "Flyer_T1"))
    self.assertEqual(diagnostics["run"], "tv")
    self.assertEqual(profile["run"], "tv")
    self.assertEqual(len(self.reader.get_model_diagnostics()["runs"]), 2)

  def test_requires_treatment_across_runs(self):
    for result in (
        self.reader.get_effects_by("geo"),
        self.reader.get_effects_by("month", treatment="Radio_T1"),
        self.reader.get_segment_profile("geo", "A"),
    ):
      with self.subTest(result=result.get("reason")):
        self.assertEqual(result["status"], "invalid_argument")

  def test_rejects_treatment_shared_by_runs(self):
    _write_results(self.root / "tv_again", prefix="TV_")

    with self.assertRaisesRegex(ValueError, "unique across runs"):
      results_reader.CausalResultsReader(
          {"tv": self.root / "tv", "tv_again": self.root / "tv_again"}
      )

  def test_snapshot_precomputes_every_tool_but_the_profile(self):
    snap = self.reader.snapshot()
    self.assertCountEqual(
        snap.keys(),
        set(results_reader.TOOL_NAMES) - {"get_segment_profile"},
    )
    self.assertCountEqual(
        snap["get_effects_by"]["TV_T1"],
        ["geo", "item", "month", "quarter", "year", "month_of_year", "demand"],
    )
    json.dumps(snap, allow_nan=False)


def _write_optimized_run(
    root: pathlib.Path, saturation: bool = False, **metadata_overrides
) -> None:
  """Writes a single-treatment run with Phase 4 optimization outputs.

  Rows (date, store, sku, spend, KPI): 2024-01 A X 10 30 (ROI 3), 2024-01 B X
  10 5 (ROI 0.5), 2024-02 A Y 20 -4 (ROI -0.2), 2024-02 B Y 0 0, and a row in
  2023-12 outside the optimization period.
  """
  phase4 = root / _PHASE4
  opt_dir = phase4 / "graph" / "007_optimization"
  opt_dir.mkdir(parents=True, exist_ok=True)
  full = pd.DataFrame({
      "date": [
          "2023-12-15",
          "2024-01-15",
          "2024-01-15",
          "2024-02-15",
          "2024-02-15",
      ],
      "store": ["A", "A", "B", "A", "B"],
      "store_name": ["Alpha", "Alpha", "Beta", "Alpha", "Beta"],
      "sku": ["X", "X", "X", "Y", "Y"],
      "sales": [100.0, 100.0, 200.0, 300.0, 400.0],
      "spend": [50.0, 10.0, 10.0, 20.0, 0.0],
      "estimated_incremental_KPI_spend": [5.0, 30.0, 5.0, -4.0, 0.0],
  })
  col = "estimated_incremental_KPI_spend"
  full[f"{col}_2.5%"] = full[col] - 1.0
  full[f"{col}_97.5%"] = full[col] + 1.0
  if saturation:
    # Moves spend from the low-ROI rows to store A in January.
    full["optimized_spend_spend"] = [50.0, 34.0, 6.0, 0.0, 0.0]
    full["optimized_incremental_KPI_spend"] = [5.0, 62.0, 3.0, 0.0, 0.0]
    spend_after, kpi_after = 40.0, 65.0
  else:
    spend_after, kpi_after = 10.0, 30.0
  full.to_csv(phase4 / "consolidated_1a_full_df.csv", index=False)

  metrics = {
      "treatment": ["spend", "spend"],
      "target_geo": ["all", "all"],
      "target_item": ["all", "X"],
      "periods_start_date": ["2024-01-01"] * 2,
      "periods_end_date": ["2024-02-29"] * 2,
      "threshold_roi": [1.0, 1.0],
      "total_cost_of_treatment_in_periods": [40.0, 20.0],
      "optimized_total_cost_of_treatment_in_periods": [spend_after, 0.0],
  }
  if saturation:
    metrics.update({
        "saturation_family": ["tanh"] * 2,
        "budget_increment": [0.0] * 2,
        "budget_mode": ["fixed"] * 2,
        "budget_reallocate": [True] * 2,
        "cap_quantile": [1.0] * 2,
        "reallocate_within": ["all"] * 2,
        "K": [40.0, 20.0],
        "K_out": [40.0, 36.0],
        "unspent_budget": [0.0, -16.0],
        "marginal_roi_at_optimum": [1.2] * 2,
        "optimized_incremental_KPI_spend": [kpi_after, 63.0],
    })
  pd.DataFrame(metrics).to_csv(
      opt_dir / "optimization_metrics.csv", index=False
  )

  metadata = {
      "treatment_cols": ["spend"],
      "outcome_col": "sales",
      "date_col": "date",
      "x_cols_list": [["demand"]],
      "covariates_for_cate_list": [[]],
      "treatment_types": {"spend": "spend"},
      "treatment_spend_cols": {"spend": "spend"},
      "geo_col": "store",
      "item_col": "sku",
      "geo_name_col": "store_name",
      "geos": ["A", "B"],
      "items": ["X", "Y"],
      "optimization": True,
      "opt_threshold_roi": 1.0,
      "simple_optimization": not saturation,
      **metadata_overrides,
  }
  if saturation:
    metadata["saturation_family"] = "tanh"
    metadata["phase4_optimization"] = {
        "spend": {"phase4_optimization": "saturation"}
    }
  (root / "metadata.json").write_text(json.dumps(metadata))


class BudgetOptimizationTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)

  def _reader(self, **kwargs) -> results_reader.CausalResultsReader:
    _write_optimized_run(self.root, **kwargs)
    return results_reader.CausalResultsReader(self.root)

  def test_simple_optimizer_is_reproduced_from_row_roi(self):
    result = self._reader().get_budget_optimization()

    self.assertEqual(result["status"], "ok", result)
    self.assertEqual(result["settings"]["optimizer"], "simple")
    overall = result["overall"]
    # The 2023-12 row is outside the period; only the ROI 3 row keeps spend.
    self.assertEqual(overall["allocation_units"], 4)
    self.assertEqual(overall["spend_before"], 40.0)
    self.assertEqual(overall["spend_after"], 10.0)
    self.assertEqual(overall["incremental_kpi_before"], 31.0)
    self.assertEqual(overall["incremental_kpi_after"], 30.0)
    self.assertEqual(overall["units_spend_cut_to_zero"], 2)
    self.assertEqual(overall["units_with_actual_spend"], 3)
    self.assertTrue(result["matches_pipeline_totals"])
    self.assertIn("not reallocated", " ".join(result["assumptions"]))
    self.assertNotIn("optimizer_statistics", result)
    json.dumps(result, allow_nan=False)

  def test_saturation_optimizer_reads_optimized_columns(self):
    result = self._reader(saturation=True).get_budget_optimization()

    self.assertEqual(result["settings"]["optimizer"], "saturation")
    self.assertEqual(result["settings"]["saturation_family"], "tanh")
    overall = result["overall"]
    self.assertEqual(overall["spend_after"], 40.0)
    self.assertEqual(overall["incremental_kpi_after"], 65.0)
    self.assertEqual(overall["units_spend_increased"], 1)
    self.assertEqual(overall["units_spend_reduced"], 1)
    self.assertEqual(overall["units_spend_cut_to_zero"], 1)
    self.assertTrue(result["matches_pipeline_totals"])
    self.assertEqual(result["optimizer_statistics"]["K"], 40.0)
    # 65 is more than twice the 31 estimated for the actual spend.
    self.assertIn("2.1 times", " ".join(result["warnings"]))

  def test_by_geo_ranks_spend_changes_with_names(self):
    result = self._reader(saturation=True).get_budget_optimization(
        dimension="geo"
    )

    self.assertEqual(result["segments"], 2)
    (increase,) = result["largest_increases"]
    (decrease,) = result["largest_decreases"]
    self.assertEqual(
        (increase["segment"], increase["name"], increase["spend_change"]),
        ("A", "Alpha", 4.0),
    )
    self.assertEqual(
        (decrease["segment"], decrease["name"], decrease["spend_change"]),
        ("B", "Beta", -4.0),
    )

  def test_by_date_returns_one_row_per_period(self):
    result = self._reader().get_budget_optimization(
        dimension="date", granularity="month"
    )

    self.assertEqual(
        [
            (p["period"], p["spend_before"], p["spend_after"])
            for p in result["periods"]
        ],
        [("2024-01", 20.0, 10.0), ("2024-02", 20.0, 0.0)],
    )

  def test_by_item(self):
    result = self._reader().get_budget_optimization(dimension="item")

    self.assertEqual(
        [
            (d["segment"], d["spend_change"])
            for d in result["largest_decreases"]
        ],
        [("Y", -20.0), ("X", -10.0)],
    )

  def test_mismatch_with_pipeline_totals_is_flagged(self):
    _write_optimized_run(self.root)
    path = (
        self.root / _PHASE4 / "graph/007_optimization/optimization_metrics.csv"
    )
    metrics = pd.read_csv(path)
    metrics["optimized_total_cost_of_treatment_in_periods"] = 99.0
    metrics.to_csv(path, index=False)

    result = results_reader.CausalResultsReader(
        self.root
    ).get_budget_optimization()

    self.assertFalse(result["matches_pipeline_totals"])
    self.assertIn("differ", " ".join(result["warnings"]))

  def test_unavailable_without_optimize(self):
    result = self._reader(optimization=False).get_budget_optimization()

    self.assertEqual(result["status"], "unavailable")
    self.assertIn("optimize=True", result["reason"])

  def test_overview_and_snapshot(self):
    reader = self._reader(saturation=True)

    (overview,) = reader.get_analysis_overview()["runs"]
    snap = reader.snapshot()["get_budget_optimization"]["spend"]

    self.assertIn("get_budget_optimization", overview["available_analyses"])
    self.assertCountEqual(snap, ["total", "date", "geo", "item"])
    self.assertEqual(snap["geo"]["status"], "ok")

  def test_invalid_arguments(self):
    reader = self._reader()
    for kwargs in (
        {"dimension": "store"},
        {"dimension": "date", "granularity": "day"},
        {"top_n": 0},
    ):
      with self.subTest(**kwargs):
        self.assertEqual(
            reader.get_budget_optimization(**kwargs)["status"],
            "invalid_argument",
        )


if __name__ == "__main__":
  unittest.main()
