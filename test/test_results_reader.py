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

import json
import pathlib
import shutil
import tempfile
import unittest

from doubleml_pipeline.agent import results_reader
from doubleml_pipeline.agent.tools import base
from doubleml_pipeline.agent.tools import effect_heterogeneity
import numpy as np
import pandas as pd

_PHASE4 = "step4_consolidated_results"


def _write_results(
    root: pathlib.Path, prefix: str = "", **metadata_overrides
) -> None:
  """Writes a small, hand-computable set of pipeline outputs to `root`."""
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
  full.to_csv(phase4 / "consolidated_1a_full_df.csv", index=False)

  pd.DataFrame({
      "d_col": [t1] * 3 + [t2] * 3,
      "Model Group": ["lgbm_lgbm_3"] * 6,
      "total_estimated_incremental_KPI": [60.0, 70.0, 80.0, -10.0, 5.0, 10.0],
  }).to_csv(phase4 / "consolidated_3a_detail.csv", index=False)

  spline = "bs(demand, df=3, degree=2, include_intercept=False)"
  pd.DataFrame({
      "model": ["rep_1", "median"],
      "d_col": [t1, t1],
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
      "ground_truth_existence": False,
      **metadata_overrides,
  }
  (root / "metadata.json").write_text(json.dumps(metadata))


class CausalResultsReaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)
    _write_results(self.root)
    self.reader = results_reader.CausalResultsReader(self.root)

  def test_missing_metadata_raises(self):
    with self.assertRaises(FileNotFoundError):
      results_reader.CausalResultsReader(self.root / "missing")

  def test_tools_are_bound_methods_with_docstrings(self):
    tools = self.reader.tools()
    self.assertEqual(len(tools), 6)
    for tool in tools:
      self.assertTrue(tool.__name__.startswith("get_"))
      self.assertTrue(tool.__doc__)

  def test_every_tool_returns_strict_json(self):
    for tool in self.reader.tools():
      with self.subTest(tool=tool.__name__):
        if tool.__name__ == "get_segment_effects":
          result = tool("geo")
        elif tool.__name__ == "get_effect_heterogeneity":
          result = tool("T1")
        else:
          result = tool()
        self.assertEqual(result["status"], "ok")
        json.dumps(result, allow_nan=False)

  def test_overview(self):
    (overview,) = self.reader.get_analysis_overview()["runs"]
    self.assertEqual(overview["run"], self.root.resolve().name)
    self.assertEqual(overview["outcome"], "sales")
    self.assertEqual(
        overview["treatments"][1],
        {
            "name": "T2",
            "type": "percentage",
            "spend_column": "T2_cost",
            "is_lagged": False,
            "adjusted_for": ["demand"],
            "cate_covariates": [],
        },
    )
    self.assertEqual(
        overview["segment_dimensions"]["geo"],
        {"column": "store", "segments": 2},
    )
    self.assertEqual(overview["unavailable_analyses"], [])

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

  def test_interval_orders_swapped_row_bounds(self):
    """Rows below the treatment's zero point have swapped bounds."""
    path = self.root / _PHASE4 / "consolidated_1a_full_df.csv"
    full = pd.read_csv(path)
    kpi = "estimated_incremental_KPI_T1"
    full.loc[3, [f"{kpi}_2.5%", f"{kpi}_97.5%"]] = [45.0, 35.0]
    full.to_csv(path, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    t1 = reader.get_treatment_effects()["runs"][0]["treatments"][0]

    self.assertEqual(
        (t1["incremental_kpi_low"], t1["incremental_kpi_high"]), (50.0, 90.0)
    )

  def test_segment_effects_ranks_with_names(self):
    result = self.reader.get_segment_effects(
        "geo", treatment="T1", sort_by="effect_per_unit", top_n=1
    )
    self.assertEqual(result["segments"], 2)
    self.assertEqual(result["highest"][0]["segment"], "B")
    self.assertEqual(result["highest"][0]["name"], "Beta")
    self.assertEqual(result["highest"][0]["effect_per_unit"], 2.0)
    self.assertEqual(result["lowest"][0]["segment"], "A")
    self.assertEqual(result["spread"], {"min": 1.5, "median": 1.75, "max": 2.0})

  def test_segment_effects_explain_unmodeled_differences(self):
    """Segments that are not CATE covariates differ only through covariates."""
    drivers = self.reader.get_segment_effects("geo", treatment="T1")[
        "effect_drivers"
    ]
    self.assertEqual(drivers["cate_covariates"], {"T1": ["demand"]})
    self.assertFalse(drivers["dimension_modeled"])
    self.assertIn("only through ['demand']", drivers["explanation"])

    constant = self.reader.get_segment_effects("geo", treatment="T2")
    self.assertIn("constant", constant["effect_drivers"]["explanation"])

  def test_segment_effects_modeled_dimension(self):
    _write_results(self.root, covariates_for_cate_list=[["store"], ["store"]])
    reader = results_reader.CausalResultsReader(self.root)

    drivers = reader.get_segment_effects("geo")["effect_drivers"]

    self.assertTrue(drivers["dimension_modeled"])

  def test_segment_effects_returns_each_segment_once(self):
    result = self.reader.get_segment_effects("geo", treatment="T1", top_n=10)
    self.assertEqual(len(result["highest"]), 2)
    self.assertEqual(result["lowest"], [])

  def test_segment_effects_rejects_per_unit_sort_for_all_treatments(self):
    result = self.reader.get_segment_effects("item", sort_by="roi_on_spend")
    self.assertEqual(result["status"], "invalid_argument")

  def test_segment_effects_validates_arguments(self):
    for kwargs in (
        {"dimension": "region"},
        {"dimension": "geo", "treatment": "T3"},
        {"dimension": "geo", "top_n": 0},
    ):
      with self.subTest(**kwargs):
        result = self.reader.get_segment_effects(**kwargs)
        self.assertEqual(result["status"], "invalid_argument")

  def test_segment_effects_unconfigured_dimension(self):
    _write_results(self.root, item_col=None)
    reader = results_reader.CausalResultsReader(self.root)
    self.assertEqual(
        reader.get_segment_effects("item")["status"], "unavailable"
    )

  def test_effect_trend_by_month(self):
    result = self.reader.get_effect_trend("T1")
    self.assertEqual(
        [p["period"] for p in result["periods"]], ["2024-01", "2024-02"]
    )
    self.assertEqual(
        [p["incremental_kpi"] for p in result["periods"]], [20.0, 50.0]
    )
    self.assertEqual(result["highest_period"], "2024-02")
    self.assertEqual(result["configured_peak_months"], [12])
    self.assertFalse(result["effect_drivers"]["dimension_modeled"])

  def test_effect_trend_month_of_year(self):
    result = self.reader.get_effect_trend("T1", granularity="month_of_year")
    self.assertEqual([p["period"] for p in result["periods"]], [1, 2])

  def test_effect_trend_without_date_is_unavailable(self):
    _write_results(self.root, date_col=None)
    reader = results_reader.CausalResultsReader(self.root)
    self.assertEqual(reader.get_effect_trend()["status"], "unavailable")

  def test_effect_heterogeneity(self):
    result = self.reader.get_effect_heterogeneity("T1", bins=2)
    (driver,) = result["covariates"]
    self.assertEqual(driver["covariate"], "demand")
    self.assertAlmostEqual(driver["heterogeneity_pvalue"], 0.04)
    self.assertEqual(driver["bins"][0]["effect_per_unit"], 2.0)
    self.assertAlmostEqual(driver["bins"][1]["effect_per_unit"], 50 / 30)
    self.assertEqual([b["observations"] for b in driver["bins"]], [2, 2])

  def test_effect_heterogeneity_groups_rare_categories(self):
    """Categorical covariates keep the most frequent values, then "other"."""
    _write_results(self.root, covariates_for_cate_list=[["channel"], []])
    reader = results_reader.CausalResultsReader(self.root)

    (driver,) = reader.get_effect_heterogeneity("T1", bins=2)["covariates"]

    self.assertIsNone(driver["heterogeneity_pvalue"])
    self.assertEqual(
        {b["bin"]: b["observations"] for b in driver["bins"]},
        {"web": 2, "other": 2},
    )

  def test_effect_heterogeneity_requires_covariates_and_treatment(self):
    self.assertEqual(
        self.reader.get_effect_heterogeneity()["status"], "invalid_argument"
    )
    self.assertEqual(
        self.reader.get_effect_heterogeneity("T2")["status"], "unavailable"
    )

  def test_robustness_checks(self):
    """Repetition stability and sensitivity flags per scope."""
    result = self.reader.get_robustness_checks("T1")
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

  def test_robustness_without_benchmark_is_unknown(self):
    """Zero benchmark strengths mean no benchmark, not a robust estimate."""
    path = self.root / _PHASE4 / "graph/006_sensitivity/sensitivity_metrics.csv"
    sensitivity = pd.read_csv(path)
    sensitivity["benchmark_covariate"] = None
    sensitivity[["benchmark_cf_d", "benchmark_cf_y"]] = 0.0
    sensitivity.to_csv(path, index=False)
    reader = results_reader.CausalResultsReader(self.root)

    (run,) = reader.get_robustness_checks()["runs"]
    overall = run["confounding_sensitivity"]["overall"]

    self.assertIsNone(overall[0]["robust_to_benchmark_confounder"])

  def test_robustness_checks_reports_missing_sections(self):
    (
        self.root / _PHASE4 / "graph/006_sensitivity/sensitivity_metrics.csv"
    ).unlink()
    reader = results_reader.CausalResultsReader(self.root)
    result = reader.get_robustness_checks()
    self.assertEqual(result["status"], "ok")
    (run,) = result["runs"]
    self.assertEqual(len(run["repetition_stability"]), 2)
    self.assertEqual(run["confounding_sensitivity"]["status"], "unavailable")

  def test_missing_consolidated_table_is_unavailable(self):
    (self.root / _PHASE4 / "consolidated_1a_full_df.csv").unlink()
    reader = results_reader.CausalResultsReader(self.root)
    self.assertEqual(reader.get_treatment_effects()["status"], "unavailable")


class HelpersTest(unittest.TestCase):
  """Module-level helpers that shape every tool result."""

  # pylint: disable=protected-access

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
        "Intercept": 0.0,
    }

    cases = {
        "spline terms, Bonferroni over finite values": ("total demand", 0.02),
        "binary term": ("holiday", 0.3),
        "no matching term": ("price", None),
        "only missing p-values": ("weather", None),
    }
    term_pvalues["weather"] = None
    for name, (covariate, expected) in cases.items():
      with self.subTest(name):
        self.assertEqual(
            effect_heterogeneity.covariate_pvalue(covariate, term_pvalues),
            expected,
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
        self.reader.get_segment_effects("geo", treatment="Flyer_T1")["highest"],
        single.get_segment_effects("geo", treatment="Flyer_T1")["highest"],
    )

  def test_routes_treatments_to_their_run(self):
    trend = self.reader.get_effect_trend("Flyer_T1")
    robustness = self.reader.get_robustness_checks("TV_T1")

    self.assertEqual((trend["run"], trend["treatment"]), ("flyer", "Flyer_T1"))
    self.assertEqual(robustness["run"], "tv")
    self.assertEqual(len(self.reader.get_robustness_checks()["runs"]), 2)

  def test_requires_treatment_across_runs(self):
    for result in (
        self.reader.get_segment_effects("geo"),
        self.reader.get_effect_trend(),
        self.reader.get_effect_heterogeneity(),
        self.reader.get_effect_trend("Radio_T1"),
    ):
      with self.subTest(result=result.get("reason")):
        self.assertEqual(result["status"], "invalid_argument")

  def test_rejects_treatment_shared_by_runs(self):
    _write_results(self.root / "tv_again", prefix="TV_")

    with self.assertRaisesRegex(ValueError, "unique across runs"):
      results_reader.CausalResultsReader(
          {"tv": self.root / "tv", "tv_again": self.root / "tv_again"}
      )

  def test_snapshot_contains_all_tool_outputs(self):
    snap = self.reader.snapshot()
    self.assertCountEqual(snap.keys(), results_reader.TOOL_NAMES)
    json.dumps(snap, allow_nan=False)


if __name__ == "__main__":
  unittest.main()
