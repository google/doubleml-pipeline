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

"""End-to-end guards: full ``run_full_pipeline`` runs of the orchestrator.

* ``family=None`` (no ``saturation`` block): the legacy path is intact (no
  ``<d_col>_transformed`` column, no ``step0_saturation/``) and the
  cluster-key determination rule still records the adopted 2-way keys.
* ``family`` set: Phase 0 (Stage A split + Stage B alpha selection) and the
  Phase 4 saturation-aware optimizer both run and emit their artifacts.
* Non-spend treatments (``percentage`` / ``impressions`` / ``price``) never
  reach the saturation-aware optimizer.
* The Phase 4 linear fallback when Phase 0 decides "linear".
* The ``geo=6, daily, no item_col`` cell-level split case.
* The unique-index guard fails fast before any phase runs.

The other full-pipeline runs of the test suite are the spec acceptance tests
(test_acceptance.py) and the hyperparameter-policy tests
(test_hparam_policy.py).

Environment constraints (do not "fix" these without checking):

* ``rf`` is the only FLAML learner that works here; ``lgbm`` and ``xgboost``
  raise inside FLAML's own code.
* ``n_jobs=1`` is required because joblib/loky cannot fork under the test
  runner (``sys.executable`` is None).
* Every orchestrator passes ``hparam_policy="per_fold"`` explicitly: these
  tests patch (``helpers.deterministic_fit``) or time-box the per-fold FLAML
  search, and the default policy is ``"global"``.

Each full run takes one to three minutes, so the BUILD target is ``large``
and sharded.  The pytest shard plugin assigns the collected test cases
round-robin, so the heavy classes are listed first (one test each) and the
cheap index-guard tests last: no shard runs more than two full pipelines.
"""

import contextlib
import glob
import io
import json
import os
import tempfile
import unittest
from unittest import mock
import warnings

from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import alpha_selection
from doubleml_pipeline.evaluation import saturation_evaluation
from doubleml_pipeline.evaluation import sensitivity
from doubleml_pipeline.models import flaml_dml
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
from doubleml_pipeline.utils import io as pipeline_io
import numpy as np
import pandas as pd


DATE_BLOCK = alpha_selection.DATE_BLOCK_COL


def make_panel(n_geo: int = 6, n_periods: int = 30, seed: int = 0):
  """Deterministic geo x week panel with a concave response to spend.

  Sales follow `base + 2 * sqrt(spend) + 3 * X1 - 1.5 * X2 + noise`, so the
  true dose-response is saturating and the Phase 0 alpha search has a real
  signal to recover.

  Args:
    n_geo: Number of geos; each is assigned to one of two items.
    n_periods: Number of weekly periods starting at 2022-01-03.
    seed: Seed for the random number generator.

  Returns:
    A DataFrame with date, geo, item, spend, X1, X2, sales and population
    columns, holding `n_geo * n_periods` rows.
  """
  rng = np.random.default_rng(seed)
  rows = []
  dates = pd.date_range("2022-01-03", periods=n_periods, freq="W")
  for g in range(n_geo):
    base = rng.uniform(50, 150)
    for d in dates:
      spend = float(max(0.0, rng.gamma(2.0, 20.0)))
      x1 = float(rng.normal())
      x2 = float(rng.normal())
      sales = (
          base + 2.0 * np.sqrt(spend) + 3.0 * x1 - 1.5 * x2 + rng.normal(0, 2.0)
      )
      rows.append({
          "date": d,
          "geo": f"G{g}",
          "item": f"I{g % 2}",
          "spend": spend,
          "X1": x1,
          "X2": x2,
          "sales": sales,
          "population": 1000.0,
      })
  return pd.DataFrame(rows)


class FamilyNoneEndToEndTest(unittest.TestCase):
  """Backward compatibility of the ``family=None`` path, end to end.

  Runs the WHOLE pipeline (Phase 0 skip -> 1 -> 2 -> 3 -> 4) with no
  `saturation` config and asserts that none of the Saturation DML machinery
  leaks into the output: no `<d_col>_transformed` column and no
  `step0_saturation/` directory (the extension is additive; existing
  ``family=None`` behaviour is untouched).  Without a ``saturation_config``
  the cluster-key determination rule still selects the 2-way Cluster DML keys
  automatically and ``metadata.json`` must say which keys were adopted.

  This is the only runtime proof that adding Phase 0 left the legacy path
  intact, so please keep it green rather than deleting it.

  FLAML searches are wall-clock bounded and therefore non-deterministic, so
  the FLAML wrapper is patched with ``helpers.deterministic_fit``.  Everything
  else (DoubleML cross-fitting, cluster keys, CATE basis, consolidation,
  Phase 4) runs exactly as in production.
  """

  def _run(self, out_dir, **extra):
    df = make_panel(n_geo=6, n_periods=30)
    scaler = scalers.CausalDataScaler(
        standardize_cols=["spend", "X1", "X2", "sales"],
        population_col="population",
    )
    orch = workflow.CausalWorkflowOrchestrator(
        y_col="sales",
        d_cols=["spend"],
        x_cols_list=[["X1", "X2"]],
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[3],
        output_dir=out_dir,
        covariates_for_cate_list=[["X1"]],
        time_budget=1,
        n_jobs=1,
        # This test patches the per-fold FLAML search: keep the legacy
        # hyperparameter policy (the default is "global").
        hparam_policy="per_fold",
        date_col="date",
        geo_col="geo",
        item_col="item",
        run_sensitivity=False,
        generate_html=False,
        treatment_types={"spend": "spend"},
        treatment_spend_cols={"spend": "spend"},
        sales_col="sales",
        optimize=True,
        **extra,
    )
    with mock.patch.object(
        flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
    ), warnings.catch_warnings():
      warnings.simplefilter("ignore")
      orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=2)
    return df

  def test_family_none_runs(self):
    with tempfile.TemporaryDirectory() as out_dir:
      df = self._run(out_dir)
      self.assertFalse(
          os.path.isdir(os.path.join(out_dir, "step0_saturation")),
          sorted(os.listdir(out_dir)),
      )
      out = pd.read_csv(helpers.find(out_dir, "consolidated_1a_full_df.csv"))
      print("\n[family=None] rows:", len(out))
      print("[family=None] has transformed col:",
            "spend_transformed" in out.columns)
      self.assertEqual(len(out), len(df))
      self.assertNotIn("spend_transformed", out.columns)
      self.assertIn("estimated_incremental_KPI_spend", out.columns)
      # Phase 4 took the simple optimizer (no saturation-aware columns) and
      # wrote its metrics.
      self.assertNotIn("optimized_spend_transformed", out.columns)
      self.assertTrue(
          os.path.exists(helpers.find(out_dir, "optimization_metrics.csv"))
      )
      summary = pd.read_csv(helpers.find(out_dir, "consolidated_3b_summary.csv"))
      self.assertGreater(len(summary), 0)

      # The cluster-key determination rule: 6 geos x 2 items x 30 weeks ->
      # monthly date blocks and geo beat item, and date/item/geo keep their
      # visualization role.
      with open(helpers.find(out_dir, "metadata.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
      self.assertIn("cluster_dml", metadata)
      cluster_dml = metadata["cluster_dml"]
      self.assertEqual(cluster_dml["keys"], ["cluster_date_block", "geo"])
      self.assertEqual(cluster_dml["n_way"], 2)
      self.assertEqual(cluster_dml["keys_used"], cluster_dml["keys"])
      self.assertEqual(cluster_dml["cluster_counts"]["geo"], 6)
      self.assertGreater(cluster_dml["cluster_counts"]["cluster_date_block"], 2)
      self.assertEqual(cluster_dml["not_adopted"], {"item": 2})
      self.assertEqual(
          cluster_dml["visualization_cols"],
          {
              "date_col": "date",
              "item_col": "item",
              "geo_col": "geo",
              "note": cluster_dml["visualization_cols"]["note"],
          },
      )
      self.assertIn("whether or not saturation_config", cluster_dml["applies_to"])


# Per-treatment Phase 0 files (tanh is bounded: the Diag5 figure exists) plus
# the Diag6 record written at the start of Phase 1 and the run-level files.
# The grid construction and the seed-wise alpha distribution are panels of
# the alpha profile chart (no stand-alone files).
_STEP0_FILES = (
    pipeline_io.SUMMARY_JSON,
    pipeline_io.ALPHA_GRID_CSV,
    pipeline_io.RLOSS_CURVES_CSV,
    pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV,
    pipeline_io.FOLD_ASSIGNMENTS_CSV,
    pipeline_io.DIAG3_IDENTIFICATION_CSV,
    pipeline_io.COVERAGE_CHECK_CSV,
    pipeline_io.DIAG6_T_REF_CONSISTENCY_CSV,
    pipeline_io.DIAG1_BASIS_APPROX_ERROR_PNG,
    pipeline_io.DIAG2_RLOSS_PROFILE_PNG,
    pipeline_io.DIAG4_F_TRANSFORM_QUANTILES_PNG,
    pipeline_io.DIAG5_SATURATION_AT_T_REF_PNG,
    pipeline_io.SPLIT_COVERAGE_PNG,
    pipeline_io.SUMMARY_CSV,
    pipeline_io.PHASE0_DIAGNOSTICS_JSON,
    pipeline_io.alpha_profile_png("spend"),
)


class LinearFallbackEndToEndTest(unittest.TestCase):
  """Phase 4 linear fallback end to end.

  Runs the whole pipeline with ``family: tanh`` and an absurdly conservative
  ``linear_test_k`` so that Phase 0 always decides "linear".  It then checks
  that:

  * the fallback warning is emitted and Phase 4 takes the existing (simple)
    optimization path, while the config value ``simple_optimization`` is NOT
    rewritten;
  * ``step0_saturation/`` holds the complete Phase 0 file set, and
    ``summary_alpha_selection.json`` records ``phase4_optimization``;
  * ``metadata.json`` and the HTML report carry the Phase 0 tab and the
    fallback notice;
  * the Phase 0 / Stage B log is the Phase 0 progress display: it gives the
    number of ML trainings, states the linear fallback explicitly, and
    nothing of the former Stage B logging remains;
  * the held-out ``D_sel`` rows are scored with the fitted CATE so the
    consolidated outputs cover every input row (``estimation_sample``), and
    ``metadata.json`` records the auto-selected Cluster DML keys.

  FLAML is patched with ``helpers.deterministic_fit`` so the test is fast
  and reproducible.
  """

  def test_linear_fallback_pipeline(self):
    """selected_family=None -> warning + simple optimization + outputs."""
    with tempfile.TemporaryDirectory() as out_dir:
      df = make_panel(n_geo=6, n_periods=30)
      scaler = scalers.CausalDataScaler(
          median_normalize_cols=["spend"],
          standardize_cols=["X1", "X2", "sales"],
          population_col="population",
      )
      orch = workflow.CausalWorkflowOrchestrator(
          y_col="sales",
          d_cols=["spend"],
          x_cols_list=[["X1", "X2"]],
          ml_models_y=["rf"],
          ml_models_t=["rf"],
          n_folds_list=[3],
          output_dir=out_dir,
          covariates_for_cate_list=[["X1"]],
          time_budget=1,
          n_jobs=1,
          # This test patches the per-fold FLAML search: keep the per-fold
          # hyperparameter policy (the default is "global").
          hparam_policy="per_fold",
          date_col="date",
          geo_col="geo",
          item_col="item",
          # The sensitivity outputs are checked
          # below (total scope keeps the run fast).
          run_sensitivity=True,
          sensitivity_scope=["total"],
          generate_html=True,
          treatment_types={"spend": "spend"},
          treatment_spend_cols={"spend": "spend"},
          sales_col="sales",
          optimize=True,
          saturation_config={
              "saturation": {"family": "tanh"},
              "alpha_selection": {
                  "n_seeds": 2,
                  "k_folds": 2,
                  # d_bar < -1e9 * se never holds -> decision "linear".
                  "linear_test_k": 1e9,
              },
          },
      )
      self.assertFalse(orch.simple_optimization)
      log_buffer = io.StringIO()
      with mock.patch.object(
          flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
      ), warnings.catch_warnings(record=True) as caught, (
          contextlib.redirect_stdout(log_buffer)
      ):
        warnings.simplefilter("always")
        orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)
      log = log_buffer.getvalue()
      print(log)  # keep the pipeline log visible in the test output

      # Warning + the config value is not rewritten.
      messages = [str(w.message) for w in caught]
      self.assertTrue(
          any(
              "saturation was not identified; falling back to linear"
              " optimization" in m
              for m in messages
          ),
          messages[:20],
      )
      self.assertFalse(orch.simple_optimization)

      # The ML-training count line of the progress display
      # (1 treatment x 2 nuisance models x 2 seeds x 2 folds = 8).
      self.assertIn(
          "Stage B: Number of ML trainings = 1 x 2 nuisance models x 2 x 2"
          " = 8", log,
      )
      # The linear fallback is stated explicitly.
      self.assertIn("FALLBACK TO LINEAR DML", log)
      # The former Stage B logging and the internal markers are gone.
      for banned in ("Stage B schedules", "one unit = one ML training",
                     "Alpha is estimated on the normalized treatment",
                     "[Phase 0 / Stage B]", "[v4-"):
        self.assertNotIn(banned, log)
      # The divisor c of T/c is kept in the JSON / HTML (not in the display)
      # and is checked next to alpha_hat below.
      # The Cluster DML keys and the rejected candidate are named.
      self.assertIn(
          "Cluster DML keys (determination rule): 2-way:"
          " cluster_date_block (7 clusters), geo (6 clusters); not adopted:"
          " item (2 clusters)",
          log,
      )
      # The D_sel rows are scored and the consolidated frame covers
      # every input row.
      self.assertIn("scored with the fitted CATE", log)
      self.assertIn(f"Consolidated frame covers all {len(df)} rows:", log)

      # Phase 0 file set with the diag / summary / others prefixes.
      step0 = os.path.join(out_dir, "step0_saturation")
      self.assertTrue(os.path.isdir(step0), sorted(os.listdir(out_dir)))
      written = sorted(os.listdir(step0))
      print("\n[e2e] step0_saturation/:", written)
      for name in _STEP0_FILES:
        self.assertIn(name, written)
      for name in written:
        self.assertTrue(
            name.startswith(("summary_", "others_", "diag")), name
        )
      # The former alpha_profile_<treatment>.csv is merged into
      # summary_alpha_grid.csv (mean_r_loss / in_1se_set columns).
      self.assertNotIn("alpha_profile_spend.csv", written)
      grid = pd.read_csv(os.path.join(step0, pipeline_io.ALPHA_GRID_CSV))
      for column in ("alpha", "mean_r_loss", "se_alpha", "in_1se_set"):
        self.assertIn(column, grid.columns)
      # Diag6 is skipped (recorded, not run) under the linear
      # fallback: no treatment replacement, hence no unit check.
      diag6 = pd.read_csv(
          os.path.join(step0, pipeline_io.DIAG6_T_REF_CONSISTENCY_CSV)
      )
      self.assertEqual(list(diag6.columns),
                       list(saturation_evaluation.DIAG6_COLUMNS))
      self.assertEqual(diag6.loc[0, "treatment"], "spend")
      self.assertEqual(diag6.loc[0, "status"], "skipped")

      with open(os.path.join(step0, saturation_evaluation.SUMMARY_JSON)) as fh:
        summary = json.load(fh)
      for key in saturation_evaluation.SUMMARY_KEYS:
        self.assertIn(key, summary)
      self.assertEqual(summary["diag6_status"], "skipped")
      self.assertEqual(summary["family"], "tanh")
      self.assertIsNone(summary["selected_family"])
      self.assertEqual(summary["decision"], "linear")
      self.assertEqual(summary["linear_test_k"], 1e9)
      self.assertEqual(summary["phase4_optimization"], "linear_fallback")
      # A linear decision carries its
      # reason, and the Phase 4 fallback record repeats it.
      self.assertIn(
          summary["decision_reason"], ("1se_fail", "grid_edge_upper")
      )
      self.assertIn(summary["grid_edge_hit"], ("none", "upper", "lower"))
      if summary["decision_reason"] == "grid_edge_upper":
        self.assertEqual(summary["grid_edge_hit"], "upper")
      self.assertGreater(summary["T_ref"], 0.0)
      self.assertGreaterEqual(summary["T_max"], summary["T_ref"])
      self.assertEqual(
          summary["phase4_fallback_decision_reason"], summary["decision_reason"]
      )
      # The divisor of T/c is recorded next to alpha_hat.
      self.assertEqual(summary["normalization"], "median_normalize")
      self.assertTrue(
          summary["divisor_formula"].startswith(
              "c = median(spend | spend != 0) = "
          ),
          summary["divisor_formula"],
      )
      self.assertGreater(summary["divisor_constant"], 0.0)
      self.assertIn("T/c = spend / ", summary["divisor_description"])

      # Phase 1-3 ran on the raw (linear) treatment: no *_transformed columns
      # from the saturation optimizer.
      consolidated = pd.read_csv(helpers.find(out_dir, "consolidated_1a_full_df.csv"))
      self.assertNotIn("optimized_spend_transformed", consolidated.columns)

      # The consolidated frame covers EVERY input row.  The D_inf
      # rows were fitted, the D_sel rows were scored with the fitted CATE, and
      # the split sizes match Phase 0's own record.
      self.assertEqual(len(consolidated), len(df))
      self.assertIn("estimation_sample", consolidated.columns)
      sample_counts = consolidated["estimation_sample"].value_counts()
      self.assertEqual(int(sample_counts["D_inf"]), int(summary["n_inf"]))
      self.assertEqual(int(sample_counts["D_sel"]), int(summary["n_sel"]))
      self.assertEqual(int(summary["n_inf"]) + int(summary["n_sel"]), len(df))
      kpi_col = "estimated_incremental_KPI_spend"
      self.assertIn(kpi_col, consolidated.columns)
      scored = consolidated[consolidated["estimation_sample"] == "D_sel"]
      self.assertTrue(np.isfinite(scored[kpi_col]).all())
      # The 1c metrics of the fitted model carry the split sizes as well.
      metrics_files = glob.glob(
          os.path.join(out_dir, "**", "1c_metrics_*.csv"), recursive=True
      )
      self.assertTrue(metrics_files, sorted(os.listdir(out_dir)))
      metrics_1c = pd.read_csv(metrics_files[0])
      self.assertEqual(
          int(metrics_1c["number_of_fitted_rows"].iloc[0]),
          int(summary["n_inf"]),
      )
      self.assertEqual(
          int(metrics_1c["number_of_scored_rows"].iloc[0]),
          int(summary["n_sel"]),
      )

      # Metadata + HTML.
      with open(helpers.find(out_dir, "metadata.json")) as fh:
        metadata = json.load(fh)
      self.assertIn("saturation_phase0", metadata)
      self.assertIn("phase4_optimization", metadata)
      phase0_meta = metadata["saturation_phase0"]["spend"]
      self.assertEqual(
          phase0_meta["decision_reason"], summary["decision_reason"]
      )
      self.assertAlmostEqual(phase0_meta["T_ref"], summary["T_ref"])
      # The auto-selected Cluster DML keys are recorded.  On this
      # panel the date blocks (7) and the geos (6) beat the items (2).
      cluster_dml = metadata["cluster_dml"]
      self.assertEqual(cluster_dml["keys"], ["cluster_date_block", "geo"])
      self.assertEqual(cluster_dml["n_way"], 2)
      self.assertEqual(cluster_dml["keys_used"], cluster_dml["keys"])
      self.assertEqual(cluster_dml["not_adopted"], {"item": 2})
      self.assertEqual(
          cluster_dml["visualization_cols"]["geo_col"], "geo"
      )
      html_files = glob.glob(os.path.join(out_dir, "*.html"))
      self.assertTrue(html_files, sorted(os.listdir(out_dir)))
      with open(html_files[0], encoding="utf-8") as fh:
        html = fh.read()
      self.assertIn("000_saturation", html)
      self.assertNotIn("000_phase0", html)
      self.assertIn("linear_fallback", html)
      self.assertIn("falling back to linear optimization", html)
      self.assertIn("Cluster DML Keys", html)
      self.assertIn("Meaning of T/c", html)
      # Header: the optimizer actually in force (derived False here, the
      # linear fallback does not rewrite it).
      self.assertIn(
          "<strong>Simple Optimization:</strong> False", html
      )
      # 007_optimization (spend): the simple optimizer's chart, the outcome
      # chart (observed vs projected sales, shared with the saturation-aware
      # optimizer), the three before / after charts and the yearly matrix are
      # wired into the tab, in that order.
      self.assertIn("img_007_spend_vs_est", html)
      self.assertIn("img_007_outcome", html)
      self.assertIn("img_007_before_after", html)
      self.assertIn("img_007_incremental_before_after", html)
      self.assertIn("img_007_roi_before_after", html)
      self.assertIn("yearly_matrix_007_main", html)
      self.assertIn("renderYearlyMatrix", html)
      self.assertLess(
          html.index('id="img_007_spend_vs_est"'),
          html.index('id="img_007_outcome"'),
      )
      self.assertLess(
          html.index('id="img_007_outcome"'),
          html.index('id="img_007_before_after"'),
      )
      self.assertLess(
          html.index('id="img_007_before_after"'),
          html.index('id="img_007_incremental_before_after"'),
      )
      self.assertLess(
          html.index('id="img_007_incremental_before_after"'),
          html.index('id="img_007_roi_before_after"'),
      )
      self.assertLess(
          html.index('id="img_007_roi_before_after"'),
          html.index('id="yearly_matrix_007_main"'),
      )
      self.assertIn("_timeseries_incremental_effect_before_after.png", html)
      self.assertIn("_timeseries_roi_before_after.png", html)
      # 004_cate: the response-curve chart sits under the scatter matrix.
      self.assertIn("cate_response_curves.png", html)

      # Phase 4 (simple optimizer, spend): before / after charts, yearly
      # summary and the CATE response curves exist on disk.
      opt_dir = os.path.join(
          out_dir, "step4_consolidated_results", "graph", "007_optimization"
      )
      opt_files = sorted(os.listdir(opt_dir))
      self.assertIn("total_timeseries_spend_vs_est.png", opt_files)
      # The outcome chart is written by the simple optimizer as well.
      self.assertIn("total_sales.png", opt_files)
      for prefix in ("total", "geo_G0", "item_I0"):
        self.assertIn(f"{prefix}_timeseries_spend_before_after.png", opt_files)
        self.assertIn(
            f"{prefix}_timeseries_incremental_effect_before_after.png",
            opt_files,
        )
        self.assertIn(f"{prefix}_timeseries_roi_before_after.png", opt_files)
      self.assertIn("optimization_yearly_summary.csv", opt_files)
      yearly = pd.read_csv(
          os.path.join(opt_dir, "optimization_yearly_summary.csv")
      )
      self.assertEqual(
          list(yearly.columns),
          list(workflow.optimization.YEARLY_SUMMARY_COLUMNS),
      )
      total_rows = yearly[yearly["dimension"] == "total"]
      self.assertEqual(list(total_rows["year"]), ["2022", "Total"])
      self.assertEqual(
          set(yearly["dimension"]), {"total", "geo", "item"}
      )
      self.assertEqual(
          set(yearly.loc[yearly["dimension"] == "geo", "level"]),
          set(df["geo"].unique()),
      )
      total_all = total_rows[total_rows["year"] == "Total"].iloc[0]
      self.assertAlmostEqual(total_all["spend_before"], df["spend"].sum())
      self.assertAlmostEqual(
          total_all["incremental_kpi_before"], consolidated[kpi_col].sum()
      )
      self.assertLessEqual(total_all["spend_after"], total_all["spend_before"])
      cate_dir = os.path.join(
          out_dir, "step4_consolidated_results", "graph", "004_cate"
      )
      self.assertIn("cate_scatter_matrix.png", os.listdir(cate_dir))
      self.assertIn("cate_response_curves.png", os.listdir(cate_dir))

      # sensitivity_metrics.csv on the
      # selected_family=None path -- D = T, the Phase 0 values are still
      # reported from summary_alpha_selection.json, and the RV / f_q / n_obs
      # follow the closed form on the fitted (D_inf) rows.
      sens = pd.read_csv(helpers.find(out_dir, "sensitivity_metrics.csv"))
      for column in (
          "theta", "se", "robustness_value", "n_obs", "f_q", "sigma2", "nu2",
          "scale", "benchmark_cf_d", "benchmark_cf_y", "treatment", "scope",
          "group", "benchmark_covariate", "treatment_variable",
          "selected_family", "alpha_hat", "half_saturation_T", "T_ref",
      ):
        self.assertIn(column, sens.columns)
      total = sens[sens["scope"] == "total"].iloc[0]
      self.assertEqual(total["treatment_variable"], "T")
      self.assertTrue(pd.isna(total["selected_family"]))
      self.assertAlmostEqual(total["alpha_hat"], summary["alpha_hat"])
      self.assertAlmostEqual(total["T_ref"], summary["T_ref"])
      self.assertEqual(int(total["n_obs"]), int(summary["n_inf"]))
      self.assertGreater(total["sigma2"], 0.0)
      self.assertGreater(total["nu2"], 0.0)
      # scale = sqrt(sigma2 nu2) (never se sqrt(n)); the RV is
      # the closed form of (theta, se, scale) and lies on the zero contour.
      self.assertAlmostEqual(
          total["scale"], np.sqrt(total["sigma2"] * total["nu2"])
      )
      for _, row in sens.iterrows():
        expected_rv, expected_f_q = sensitivity.robustness_value(
            float(row["theta"]), float(row["se"]), float(row["scale"])
        )
        self.assertAlmostEqual(row["robustness_value"], expected_rv)
        self.assertAlmostEqual(row["f_q"], expected_f_q)
        if expected_rv > 0:
          contour = (
              abs(float(row["theta"]))
              - float(sensitivity.bias_bound(
                  expected_rv, expected_rv, float(row["scale"])
              ))
              - sensitivity.T_STAR_ALPHA * float(row["se"])
          )
          self.assertAlmostEqual(contour, 0.0, delta=1e-8)
      self.assertIn(total["benchmark_covariate"], ("X1", "X2"))
      self.assertTrue(
          os.path.exists(helpers.find(out_dir, "spend_total_sensitivity_contour.png"))
      )
      # HTML: the sensitivity note hook exists; with selected_family=None the
      # per-treatment map carries null, so the line stays hidden.
      self.assertIn('id="sens_note_006"', html)
      self.assertIn('const sensSelectedFamily = {"spend": null};', html)


def _make_percentage_panel(n_geo=5, n_periods=26, seed=0):
  """Builds a geo/week panel whose treatment is a discount *rate*.

  `discount` lives in [0, 0.4] so it passes the `percentage` validation, and
  `discount_amount` is the monetary cost implied by that rate. Sales respond
  concavely to the rate, giving Phase 0 a real signal.

  Args:
    n_geo: Number of geos; each is assigned to one of two items.
    n_periods: Number of weekly periods starting at 2022-01-03.
    seed: Seed for the random number generator.

  Returns:
    A DataFrame with date, geo, item, discount, discount_amount, X1, X2, sales
    and population columns.
  """
  rng = np.random.default_rng(seed)
  rows = []
  dates = pd.date_range("2022-01-03", periods=n_periods, freq="W")
  for g in range(n_geo):
    base = rng.uniform(50, 150)
    for d in dates:
      discount = float(rng.uniform(0.01, 0.4))
      x1 = float(rng.normal())
      x2 = float(rng.normal())
      sales = (
          base
          + 40.0 * np.sqrt(discount)
          + 3.0 * x1
          - 1.5 * x2
          + rng.normal(0, 2.0)
      )
      rows.append({
          "date": d,
          "geo": f"G{g}",
          "item": f"I{g % 2}",
          "discount": discount,
          # Monetary cost of the promotion for this row.
          "discount_amount": discount * sales,
          "X1": x1,
          "X2": x2,
          "sales": sales,
          "population": 1000.0,
      })
  return pd.DataFrame(rows)


class NonSpendFallsBackToSimpleOptimizerTest(unittest.TestCase):
  """`treatment_types != "spend"` must route to the simple optimizer.

  The saturation-aware allocator budgets the treatment column directly, which
  is only meaningful when the treatment *is* the money (``g(T) = T``).  For
  ``"percentage"``, ``"impressions"`` and ``"price"`` the orchestrator
  therefore forces ``simple_optimization=True`` at construction (with a
  warning) and Phase 4 runs :func:`doubleml_pipeline.optimization.optimize`,
  even though Phases 0-3 still run the saturation model.
  """

  def _make_orchestrator(self, out_dir, **overrides):
    kwargs = dict(
        y_col="sales",
        d_cols=["discount"],
        x_cols_list=[["X1", "X2"]],
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[3],
        output_dir=out_dir,
        covariates_for_cate_list=[["X1"]],
        time_budget=1,
        n_jobs=1,
        # This test patches the per-fold FLAML search: keep the legacy
        # hyperparameter policy (the default is "global").
        hparam_policy="per_fold",
        date_col="date",
        geo_col="geo",
        item_col="item",
        run_sensitivity=False,
        generate_html=False,
        treatment_types={"discount": "percentage"},
        treatment_spend_cols={"discount": "discount_amount"},
        sales_col="sales",
        optimize=True,
        saturation_config={
            "saturation": {
                "family": "power",
                "grid": {
                    "min": 0.2,
                    "max": 0.8,
                    "num": 5,
                    "scale": "linear",
                },
                "basis_dim_T": 4,
            },
            "alpha_selection": {
                "n_seeds": 1,
                "k_folds": 2,
                "refine": "none",
            },
        },
    )
    kwargs.update(overrides)
    return workflow.CausalWorkflowOrchestrator(**kwargs)

  def test_simple_optimization_is_forced_true_for_non_spend(self):
    """A non-spend treatment forces simple_optimization=True at init."""
    with tempfile.TemporaryDirectory() as out_dir:
      # Derived (family set -> would be False) ...
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orch = self._make_orchestrator(out_dir)
      self.assertTrue(orch.simple_optimization)
      messages = [str(w.message) for w in caught]
      self.assertTrue(
          any("simple_optimization is forced to True" in m for m in messages),
          messages,
      )
      # ... and explicit False are both overridden.
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orch_false = self._make_orchestrator(
            out_dir, simple_optimization=False
        )
      self.assertTrue(orch_false.simple_optimization)
      self.assertTrue(
          any(
              "simple_optimization is forced to True" in str(w.message)
              for w in caught
          )
      )
      # Explicit True is simply accepted -- without the "explicit True with a
      # saturation family" warning, which only concerns spend treatments.
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orch_true = self._make_orchestrator(out_dir, simple_optimization=True)
      self.assertTrue(orch_true.simple_optimization)
      self.assertFalse(
          [w for w in caught if "simple_optimization" in str(w.message)],
          [str(w.message) for w in caught],
      )
      # Without a saturation family nothing needs forcing and nothing warns.
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orch_linear = self._make_orchestrator(out_dir, saturation_config=None)
      self.assertTrue(orch_linear.simple_optimization)
      self.assertFalse(
          [w for w in caught if "simple_optimization" in str(w.message)]
      )

  def test_percentage_uses_simple_optimizer(self):
    """A percentage treatment with a saturation family uses the simple path."""
    with tempfile.TemporaryDirectory() as out_dir:
      df = _make_percentage_panel()
      scaler = scalers.CausalDataScaler(
          median_normalize_cols=["discount"],
          standardize_cols=["X1", "X2", "sales"],
          population_col="population",
      )
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        orch = self._make_orchestrator(out_dir)
      self.assertTrue(orch.simple_optimization)

      with mock.patch.object(
          workflow.optimization, "optimize_saturation"
      ) as mock_saturation_opt, mock.patch.object(
          workflow.optimization, "optimize"
      ) as mock_simple_opt:
        with warnings.catch_warnings(record=True):
          warnings.simplefilter("always")
          orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)

      mock_saturation_opt.assert_not_called()
      mock_simple_opt.assert_called_once()

      # The simple optimizer must still be pointed at the roi_output tree
      # and receive the orchestrator's opt_threshold_roi (default 1.0).
      out_dir_kwarg = mock_simple_opt.call_args.kwargs["out_dir"]
      self.assertIn("roi_output", out_dir_kwarg)
      self.assertEqual(
          mock_simple_opt.call_args.kwargs["threshold_roi"],
          orch.opt_threshold_roi,
      )
      self.assertEqual(orch.opt_threshold_roi, workflow.DEFAULT_THRESHOLD_ROI)
      # ... and the outcome column, so the `<prefix>_<outcome>.png` chart
      # (observed vs projected outcome) is drawn on the roi_output 007 tab.
      self.assertEqual(mock_simple_opt.call_args.kwargs["outcome_col"], "sales")
      # metadata.json reports the forced value.
      with open(os.path.join(out_dir, "metadata.json")) as fh:
        metadata = json.load(fh)
      self.assertTrue(metadata["simple_optimization"])


class SaturationEndToEndTest(unittest.TestCase):
  """End-to-end guard for the Saturation DML path (`family` set).

  Runs the WHOLE pipeline with a `saturation` block and asserts that Phase 0
  (Stage A split + Stage B alpha selection) and the Phase 4 saturation-aware
  optimizer both execute and emit their artifacts.

  Note on what is NOT asserted: the consolidated CSV deliberately carries RAW
  treatment units (`df_1a_consolidated = df_est.copy()`), so
  `<d_col>_transformed` is absent by design and lives only in the
  per-repetition `df_1a` frames. Assert on the `optimized_*` block instead.

  The Phase 0 search space is kept deliberately small (basis_dim_T=4, 1 seed,
  2 folds => 2 x 1 x 2 = 4 nuisance fits with the fixed RidgeCV small-sample
  learners) so the test stays well inside the timeout; the saturation path is
  pinned through `alpha_selection.fallback_half_saturation_T` (see the config
  comment) so that the Phase 4 saturation optimizer always runs.
  """

  def test_saturation_runs(self):
    """Runs the pipeline with a saturation block and checks its artifacts."""
    with tempfile.TemporaryDirectory() as out_dir:
      df = make_panel(n_geo=5, n_periods=26)
      # Treatment normalization bridge: a saturated treatment must use a
      # DIVISIVE normalization. `standardize` shifts, so it would send roughly
      # half the rows negative (silently clipped to 0 by f) and would break
      # f(0) = 0; the pipeline now rejects that combination outright.
      # Covariates and the outcome are unconstrained.
      scaler = scalers.CausalDataScaler(
          median_normalize_cols=["spend"],
          standardize_cols=["X1", "X2", "sales"],
          population_col="population",
      )
      orch = workflow.CausalWorkflowOrchestrator(
          y_col="sales",
          d_cols=["spend"],
          x_cols_list=[["X1", "X2"]],
          ml_models_y=["rf"],
          ml_models_t=["rf"],
          n_folds_list=[3],
          output_dir=out_dir,
          covariates_for_cate_list=[["X1"]],
          time_budget=1,
          n_jobs=1,
          # This test patches the per-fold FLAML search: keep the previous
          # hyperparameter policy (the default is "global").
          hparam_policy="per_fold",
          date_col="date",
          geo_col="geo",
          item_col="item",
          run_sensitivity=False,
          generate_html=False,
          treatment_types={"spend": "spend"},
          treatment_spend_cols={"spend": "spend"},
          sales_col="sales",
          optimize=True,
          # simple_optimization is omitted on purpose: it is derived as False
          # because a saturation family is set.
          saturation_config={
              "saturation": {
                  "family": "power",
                  "grid": {
                      "min": 0.2,
                      "max": 0.8,
                      "num": 5,
                      "scale": "linear",
                  },
                  "basis_dim_T": 4,
              },
              "alpha_selection": {
                  "n_seeds": 1,
                  "k_folds": 2,
                  "refine": "none",
                  # This guard exercises the saturation-aware Phase 4 path,
                  # not the power of the 1-SE test: with 26 D_sel rows, one
                  # seed and two folds the honest (out-of-fold) 1-SE decision
                  # sits on a knife edge.  The deliberately tiny T-basis
                  # (J = 4) makes Diag1 fire deterministically (eps(J) ~ 0.15
                  # >= basis_tol), so the fixed fallback below keeps the
                  # family with alpha = 0.5 (the sqrt truth; for `power` the
                  # value is alpha itself) and Phase 4 always runs the
                  # saturation optimizer.
                  "fallback_half_saturation_T": 0.5,
              },
          },
      )
      self.assertFalse(orch.simple_optimization)
      orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)

      path = None
      for root, _, files in os.walk(out_dir):
        for f in files:
          if f == "consolidated_1a_full_df.csv":
            path = os.path.join(root, f)
      self.assertIsNotNone(path, os.listdir(out_dir))
      out = pd.read_csv(path)
      print("\n[saturation] rows:", len(out))
      print("[saturation] cols:", list(out.columns))
      print("[saturation] tree:", sorted(os.listdir(out_dir)))
      # Consolidated CSV is intentionally in RAW treatment units, so the
      # evidence that the saturation path ran is the optimized_* block
      # produced by optimization.optimize_saturation.
      for col in (
          "optimized_spend",
          "optimized_spend_transformed",
          "optimized_incremental_KPI_spend",
          "optimized_roi_spend",
      ):
        self.assertIn(col, out.columns)
      # Raw treatment must be restored (not the power-transformed values).
      self.assertGreater(out["spend"].max(), 10.0)
      self.assertTrue(
          os.path.isdir(os.path.join(out_dir, "step0_saturation")),
          sorted(os.listdir(out_dir)),
      )

      # optimization_metrics.csv reports the alpha range, because
      # population_median_normalize makes alpha_i row-varying. This run uses
      # 'power', for which alpha_i is constant, so all three must coincide.
      metrics_path = None
      for root, _, files in os.walk(out_dir):
        for f in files:
          if f == "optimization_metrics.csv":
            metrics_path = os.path.join(root, f)
      self.assertIsNotNone(metrics_path, sorted(os.listdir(out_dir)))
      metrics = pd.read_csv(metrics_path)
      for col in ("alpha_hat", "alpha_hat_min", "alpha_hat_max"):
        self.assertIn(col, metrics.columns)
      self.assertTrue((metrics["alpha_hat_min"] <= metrics["alpha_hat"]).all())
      self.assertTrue((metrics["alpha_hat"] <= metrics["alpha_hat_max"]).all())
      # 'power' keeps alpha_i constant, so the range must be degenerate.
      self.assertTrue(
          np.allclose(metrics["alpha_hat_min"], metrics["alpha_hat_max"]),
          metrics[["alpha_hat_min", "alpha_hat_max"]],
      )
      self.assertTrue((metrics["threshold_roi"] == 1.0).all())
      self.assertTrue(metrics["budget_reallocate"].astype(bool).all())
      # Allocation-scope defaults and the stated premise.
      self.assertTrue((metrics["cap_quantile"] == 1.0).all())
      self.assertTrue((metrics["reallocate_within"] == "all").all())
      self.assertTrue((metrics["assumptions"] == "no_carryover").all())
      self.assertIn("n_units_at_support_cap", metrics.columns)

      # The family was kept through the configured fallback (Diag1 fires with
      # J = 4), with alpha fixed at 0.5.
      summary_path = None
      for root, _, files in os.walk(out_dir):
        for f in files:
          if f == "summary_alpha_selection.json":
            summary_path = os.path.join(root, f)
      self.assertIsNotNone(summary_path, sorted(os.listdir(out_dir)))
      with open(summary_path) as handle:
        summary = json.load(handle)
      self.assertEqual(summary["selected_family"], "power")
      self.assertTrue(summary["diag1_fired"])
      self.assertTrue(summary["fallback_half_saturation_T_used"])
      self.assertAlmostEqual(summary["alpha_hat"], 0.5)
      self.assertEqual(summary["se_method"], "two-way")


def _daily_geo_panel(n_geo=6, n_days=70, seed=0):
  """geo=6 x daily panel without item_col (the acceptance-test case)."""
  rng = np.random.default_rng(seed)
  rows = []
  dates = pd.date_range("2022-01-03", periods=n_days, freq="D")
  for g in range(n_geo):
    base = rng.uniform(50, 150)
    for d in dates:
      spend = float(max(0.0, rng.gamma(2.0, 20.0)))
      x1 = float(rng.normal())
      x2 = float(rng.normal())
      rows.append({
          "date": d,
          "geo": f"G{g}",
          "spend": spend,
          "X1": x1,
          "X2": x2,
          "sales": base + 2.0 * np.sqrt(spend) + 3.0 * x1 - 1.5 * x2
                   + rng.normal(0, 2.0),
          "population": 1000.0,
      })
  return pd.DataFrame(rows)


class GeoSixDailyEndToEndTest(unittest.TestCase):
  """(iii) geo=6, daily, no item_col: cells = geo x week block, end to end."""

  def test_pipeline_runs_with_cell_level_split(self):
    """The full pipeline runs with the cell-level split and reports it."""
    with tempfile.TemporaryDirectory() as out_dir:
      orch = workflow.CausalWorkflowOrchestrator(
          y_col="sales",
          d_cols=["spend"],
          x_cols_list=[["X1", "X2"]],
          ml_models_y=["rf"],
          ml_models_t=["rf"],
          n_folds_list=[3],
          output_dir=out_dir,
          covariates_for_cate_list=[["X1"]],
          time_budget=1,
          n_jobs=1,
          # This test patches the per-fold FLAML search: keep the previous
          # hyperparameter policy (the default is "global").
          hparam_policy="per_fold",
          date_col="date",
          geo_col="geo",
          run_sensitivity=False,
          generate_html=False,
          treatment_types={"spend": "spend"},
          treatment_spend_cols={"spend": "spend"},
          sales_col="sales",
          saturation_config={
              "saturation": {"family": "tanh"},
              "alpha_selection": {"n_seeds": 2, "k_folds": 5},
          },
      )
      scaler = scalers.CausalDataScaler(
          median_normalize_cols=["spend"],
          standardize_cols=["X1", "X2", "sales"],
          population_col="population",
      )
      with mock.patch.object(
          flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
      ), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orch.run_full_pipeline(
            df=_daily_geo_panel(), scaler=scaler, top_n=1, n_reps=1
        )

      step0 = os.path.join(out_dir, "step0_saturation")
      diagnostics_path = os.path.join(
          step0, pipeline_io.PHASE0_DIAGNOSTICS_JSON
      )
      with open(diagnostics_path) as fh:
        diagnostics = json.load(fh)
      self.assertEqual(diagnostics["cluster_keys"], [DATE_BLOCK, "geo"])
      self.assertEqual(diagnostics["split_unit"], "cell")
      self.assertEqual(diagnostics["split_keys"], ["geo", DATE_BLOCK])
      self.assertTrue(diagnostics["coverage_ok"])
      self.assertEqual(diagnostics["final_fit"], "d_inf")

      with open(os.path.join(step0, saturation_evaluation.SUMMARY_JSON)) as fh:
        summary = json.load(fh)
      self.assertEqual(summary["fold_unit"], "cell")
      self.assertEqual(summary["fold_keys"], ["geo", DATE_BLOCK])
      self.assertIn("cell-level", summary["cluster_fallback_tier"])

      # coverage_check.csv: one row per level of every key.
      coverage = pd.read_csv(
          os.path.join(step0, pipeline_io.COVERAGE_CHECK_CSV)
      )
      self.assertEqual(list(coverage.columns),
                       list(saturation_evaluation.COVERAGE_COLUMNS))
      self.assertEqual(len(coverage), 6 + 10)  # 6 geos + 10 week blocks.
      self.assertTrue(coverage["covered"].all())
      self.assertTrue((coverage["n_cells_sel"] >= 1).all())
      self.assertTrue((coverage["n_cells_inf"] >= 1).all())

      # fold_assignments.csv carries the cell's key values.
      folds = pd.read_csv(
          os.path.join(step0, pipeline_io.FOLD_ASSIGNMENTS_CSV)
      )
      self.assertEqual(list(folds.columns),
                       list(saturation_evaluation.FOLD_ASSIGNMENT_COLUMNS))
      self.assertEqual(set(folds["split"]), {"sel", "inf"})
      self.assertEqual(
          (folds["key1_value"] + "|" + folds["key2_value"]).tolist(),
          folds["cluster_id"].tolist(),
      )
      self.assertFalse(folds.duplicated(["seed", "cluster_id"]).any())
      self.assertEqual(folds.groupby("seed")["cluster_id"].nunique().iloc[0],
                       6 * 10)


def _make_minimal_panel(n_geo: int = 4, n_periods: int = 12, seed: int = 0):
  """Builds a small synthetic geo x period panel with a unique index."""
  rng = np.random.default_rng(seed)
  rows = []
  dates = pd.date_range("2022-01-03", periods=n_periods, freq="W")
  for g in range(n_geo):
    for d in dates:
      spend = float(rng.gamma(2.0, 20.0))
      x1 = float(rng.normal())
      rows.append({
          "date": d,
          "geo": f"G{g}",
          "spend": spend,
          "X1": x1,
          "sales": 100.0 + 2.0 * np.sqrt(spend) + 3.0 * x1,
          "population": 1000.0,
      })
  return pd.DataFrame(rows)


def _orchestrator(out_dir: str, saturation: bool):
  return workflow.CausalWorkflowOrchestrator(
      y_col="sales",
      d_cols=["spend"],
      x_cols_list=[["X1"]],
      ml_models_y=["rf"],
      ml_models_t=["rf"],
      n_folds_list=[3],
      output_dir=out_dir,
      covariates_for_cate_list=[["X1"]],
      time_budget=1,
      n_jobs=1,
      # This test patches the per-fold FLAML search: keep the legacy
      # hyperparameter policy (the default is "global").
      hparam_policy="per_fold",
      date_col="date",
      geo_col="geo",
      run_sensitivity=False,
      generate_html=False,
      treatment_types={"spend": "spend"},
      treatment_spend_cols={"spend": "spend"},
      sales_col="sales",
      saturation_config=(
          {"saturation": {"family": "tanh"}} if saturation else None
      ),
  )


class UniqueIndexGuardTest(unittest.TestCase):
  """The orchestrator rejects input frames whose index is not unique.

  Rows are tracked by index label throughout the pipeline (Phase 0 D_sel /
  D_inf selection, divisor and cell-id alignment, Phase 4 raw-treatment
  restoration).  A duplicated label must therefore fail fast with an
  actionable message, in both the saturation and the non-saturation
  (family=None) scenario, before any phase runs.
  """

  def _assert_rejected(self, df: pd.DataFrame, saturation: bool):
    """Runs the orchestrator on ``df`` and expects the index guard to fire."""
    with tempfile.TemporaryDirectory() as out_dir:
      orch = _orchestrator(out_dir, saturation)
      scaler = scalers.CausalDataScaler(
          median_normalize_cols=["spend"],
          standardize_cols=["X1", "sales"],
          population_col="population",
      )
      with self.assertRaisesRegex(ValueError, "index must be unique"):
        orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)
      # Fails before any phase writes output.
      written = os.listdir(out_dir) if os.path.isdir(out_dir) else []
      self.assertFalse(
          [name for name in written if name.startswith("step")], written
      )

  def test_duplicate_labels_rejected_in_both_scenarios(self):
    df = _make_minimal_panel()
    idx = np.arange(len(df))
    idx[-3:] = [0, 1, 2]
    df.index = idx
    for saturation in (False, True):
      with self.subTest(saturation=saturation):
        self._assert_rejected(df, saturation)

  def test_every_label_twice_rejected(self):
    df = _make_minimal_panel()
    df.index = np.arange(len(df)) % (len(df) // 2)
    self._assert_rejected(df, saturation=True)

  def test_error_message_names_duplicates_and_fix(self):
    df = _make_minimal_panel()
    df.index = [7] * 2 + list(range(100, 100 + len(df) - 2))
    with tempfile.TemporaryDirectory() as out_dir:
      orch = _orchestrator(out_dir, saturation=False)
      with self.assertRaises(ValueError) as ctx:
        orch.run_full_pipeline(df=df, scaler=None, top_n=1, n_reps=1)
    message = str(ctx.exception)
    self.assertIn("1 duplicated label(s)", message)
    self.assertIn("[7]", message)
    self.assertIn("reset_index(drop=True)", message)


if __name__ == "__main__":
  unittest.main()
