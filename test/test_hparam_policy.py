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

"""End-to-end tests of the hyperparameter policy (dml_performance_spec §8.2).

The FLAML hyperparameter search is replaced by the deterministic OLS "search"
``helpers.deterministic_search`` (``LinearRegression`` with no
hyperparameters) so that every run is exactly reproducible; everything else --
the ``FixedHparamRegressor`` learners, the DoubleML cross-fitting, the Phase 1
-> Phase 3 hand-over, the consolidation and the sensitivity analysis -- runs
as in production.

``hparam_policy="global"`` (the default):

* |Y| + |T| searches per treatment, the Y target for the Y side and the D
  column for the T side; the ``hparams_<d_col>_<learner>.json`` files and the
  log lines.
* Phase 3 repetition 0 == the Phase 1 result of the candidate; the best
  group's rep 0 keeps ``df_1a`` / ``df_1b`` / ``model_obj``.
* ``step3_*`` / ``step4_consolidated_results`` CSVs are identical across two
  runs with the same seeds and ``hparams_*.json``, and identical with /
  without the reduced worker returns.
* The total row of ``sensitivity_metrics.csv`` matches DoubleML's
  ``sensitivity_benchmark`` within 1e-6; group rows carry ``benchmark_scope =
  "group"``.

The hand-over and the ``per_fold`` regression:

* The T-side search target is the ``D`` column Phases 1-3 regress on
  (``fbar(T; alpha_hat)`` when ``selected_family != None``, ``T`` otherwise)
  and the search rows are exactly the ``final_fit`` sample.
* The ``per_fold`` path still runs one FLAML search per fold and never touches
  the global search / ``hparams_*.json``.
* ``df_norm`` and the CATE basis built once equal the ones ``run()`` builds
  itself; the run results are identical.
* The helpers ``reduce_rep_result`` / ``regroup_phase3_results``.

Every full run takes one to three minutes on Forge (almost all of it Phase 4
plotting), so the BUILD target is ``large`` and sharded; the classes are
ordered so that the round-robin shard assignment never puts more than three
full runs (including the shared reference run) on one shard.
"""

import atexit
import contextlib
import glob
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import cloudpickle
from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import hparam_search
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.models import flaml_dml
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
from doubleml_pipeline.utils import io as pipeline_io
import numpy as np
import pandas as pd
from sklearn import linear_model


def make_panel(n_geo: int = 6, n_periods: int = 30, seed: int = 0):
  """Geo x week panel with a confounder ``X3`` (drives spend and sales).

  Args:
    n_geo: Number of geos; each is assigned to one of two items.
    n_periods: Number of weekly periods starting at 2022-01-03.
    seed: Seed of the random number generator.

  Returns:
    A DataFrame with date, geo, item, spend, X1, X2, X3, sales and
    population columns.
  """
  rng = np.random.default_rng(seed)
  rows = []
  dates = pd.date_range("2022-01-03", periods=n_periods, freq="W")
  for g in range(n_geo):
    base = rng.uniform(50, 150)
    for d in dates:
      x1 = float(rng.normal())
      x2 = float(rng.normal())
      x3 = float(rng.normal())
      spend = float(max(0.0, rng.gamma(2.0, 20.0) * np.exp(0.4 * x3)))
      sales = (
          base
          + 30.0 * np.tanh(spend / 40.0)
          + 3.0 * x1
          - 1.5 * x2
          + 4.0 * x3
          + rng.normal(0, 2.0)
      )
      rows.append({
          "date": d,
          "geo": f"G{g}",
          "item": f"I{g % 2}",
          "spend": spend,
          "X1": x1,
          "X2": x2,
          "X3": x3,
          "sales": sales,
          "population": 1000.0,
      })
  return pd.DataFrame(rows)


def make_scaler():
  return scalers.CausalDataScaler(
      median_normalize_cols=["spend"],
      standardize_cols=["X1", "X2", "X3", "sales"],
      population_col="population",
  )


def orchestrator(out_dir, saturation_config=None, **extra):
  """The orchestrator of the tests (``hparam_policy`` default = global)."""
  kwargs = dict(
      y_col="sales",
      d_cols=["spend"],
      x_cols_list=[["X1", "X2", "X3"]],
      ml_models_y=["rf", "extra_tree"],
      ml_models_t=["rf"],
      n_folds_list=[3],
      output_dir=out_dir,
      covariates_for_cate_list=[["X1"]],
      time_budget=1,
      n_jobs=1,
      date_col="date",
      geo_col="geo",
      item_col="item",
      run_sensitivity=True,
      sensitivity_scope=["total", "geo"],
      generate_html=False,
      treatment_types={"spend": "spend"},
      treatment_spend_cols={"spend": "spend"},
      sales_col="sales",
      optimize=True,
      simple_optimization=True,
      saturation_config=saturation_config,
  )
  kwargs.update(extra)
  return workflow.CausalWorkflowOrchestrator(**kwargs)


def run_global(out_dir, df, saturation_config=None, top_n=2, n_reps=3,
               **extra):
  """Runs the pipeline under ``global`` with the deterministic search.

  Args:
    out_dir: Output directory.
    df: Input panel.
    saturation_config: Optional Phase 0 configuration.
    top_n: Phase 2 shortlist size.
    n_reps: Phase 3 repetitions.
    **extra: Overrides of the orchestrator arguments.

  Returns:
    ``(search_mock, log_text)``.
  """
  orch = orchestrator(out_dir, saturation_config=saturation_config, **extra)
  search = mock.Mock(side_effect=helpers.deterministic_search)
  log_buffer = io.StringIO()
  with mock.patch.object(
      hparam_search, "search_hyperparameters", search
  ), warnings.catch_warnings(), contextlib.redirect_stdout(log_buffer):
    warnings.simplefilter("ignore")
    orch.run_full_pipeline(
        df=df, scaler=make_scaler(), top_n=top_n, n_reps=n_reps,
        process_log="full",
    )
  return search, log_buffer.getvalue()


def read_csvs(out_dir):
  """All CSVs below ``step3_*`` and ``step4_consolidated_results``."""
  out = {}
  for pattern in ("step3_*/**/*.csv", "step4_consolidated_results/**/*.csv"):
    for path in sorted(glob.glob(os.path.join(out_dir, pattern),
                                 recursive=True)):
      out[os.path.relpath(path, out_dir)] = pd.read_csv(path)
  return out


_REFERENCE = {}


def _reference_run():
  """One shared ``global`` run (seed 0, top_n = 2, n_reps = 3).

  Returns:
    ``(df, out_dir, search_mock, log_text)``; the output directory lives
    until the interpreter exits.
  """
  if not _REFERENCE:
    df = make_panel()
    out_dir = tempfile.mkdtemp(prefix="perf_global_reference_")
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    search, log = run_global(out_dir, df)
    _REFERENCE.update(df=df, out_dir=out_dir, search=search, log=log)
  return (_REFERENCE["df"], _REFERENCE["out_dir"], _REFERENCE["search"],
          _REFERENCE["log"])


class HandOverTest(unittest.TestCase):
  """The search uses the D column and the sample of Phases 1-3."""

  def _check(self, saturation_config, expect_family, expect_rows):
    df = make_panel(seed=2)
    with tempfile.TemporaryDirectory() as out_dir:
      search, _ = run_global(
          out_dir, df, saturation_config=saturation_config, top_n=1,
          n_reps=1, ml_models_y=["rf"], ml_models_t=["rf"],
          run_sensitivity=False, optimize=False,
      )
      with open(helpers.find(out_dir, pipeline_io.SUMMARY_JSON)) as fh:
        summary = json.load(fh)
      self.assertEqual(summary["selected_family"], expect_family)
      pkl = glob.glob(os.path.join(out_dir, "step3_final_spend", "best_model",
                                   "best_dml_object_*.pkl"))[0]
      with open(pkl, "rb") as fh:
        model = cloudpickle.load(fh)
      d_used = np.asarray(model._dml_data.d, dtype=float).ravel()  # pylint: disable=protected-access
      y_used = np.asarray(model._dml_data.y, dtype=float).ravel()  # pylint: disable=protected-access
      self.assertEqual(search.call_count, 2)
      # The T-side target is exactly the D column the PLR regressed on
      # (fbar(T; alpha_hat) under an adopted family, T otherwise) and the Y
      # side exactly Y, both on the final_fit sample.
      targets = [np.asarray(c.kwargs["y"], dtype=float) for c in
                 search.call_args_list]
      matched_d = [t for t in targets if t.shape == d_used.shape
                   and np.allclose(t, d_used)]
      matched_y = [t for t in targets if t.shape == y_used.shape
                   and np.allclose(t, y_used)]
      self.assertEqual(len(matched_d), 1)
      self.assertEqual(len(matched_y), 1)
      self.assertEqual(len(d_used), expect_rows(summary, df))
      for call in search.call_args_list:
        self.assertEqual(call.kwargs["x"].shape[0], len(d_used))
      # Traceability in the JSON.
      with open(helpers.find(out_dir,
                           hparam_search.hparams_json_name("spend", "rf")),
                encoding="utf-8") as fh:
        payload = json.load(fh)
      self.assertEqual(payload["context"]["selected_family"], expect_family)
      self.assertEqual(payload["context"]["n_rows"], len(d_used))
      return summary, d_used, df

  def test_saturation_adopted_uses_fbar_and_d_inf(self):
    cfg = {
        "saturation": {"family": "tanh"},
        "alpha_selection": {"n_seeds": 2, "k_folds": 2,
                            "linear_test_k": -1e12},
    }
    summary, d_used, df = self._check(
        cfg, "tanh", lambda s, _: int(s["n_inf"])
    )
    # D = fbar(T; alpha_hat) differs from the raw treatment column.
    raw = df["spend"].to_numpy(dtype=float)
    self.assertFalse(np.allclose(np.sort(d_used), np.sort(raw[: len(d_used)])))
    self.assertLess(len(d_used), len(df))
    self.assertIsNotNone(summary["alpha_hat"])

  def test_linear_fallback_uses_t_and_d_inf(self):
    cfg = {
        "saturation": {"family": "tanh"},
        "alpha_selection": {"n_seeds": 2, "k_folds": 2,
                            "linear_test_k": 1e9},
    }
    self._check(cfg, None, lambda s, _: int(s["n_inf"]))

  def test_full_sample_final_fit_uses_all_rows(self):
    cfg = {
        "saturation": {"family": "tanh"},
        "alpha_selection": {"n_seeds": 2, "k_folds": 2,
                            "linear_test_k": -1e12},
        "final_fit": {"data": "full_sample"},
    }
    self._check(cfg, "tanh", lambda _, df: len(df))


class PerFoldRegressionTest(unittest.TestCase):
  """``per_fold`` = one FLAML search per fold, as before."""

  def test_per_fold_path_function_calls(self):
    df = make_panel(seed=3)
    n_reps = 2
    fit_calls = []

    def _counting_fit(self, X, y):  # pylint: disable=invalid-name
      fit_calls.append((len(y), self.early_stop, self.n_jobs))
      return helpers.deterministic_fit(self, X, y)

    search = mock.Mock(side_effect=AssertionError("no global search"))
    with tempfile.TemporaryDirectory() as out_dir:
      orch = orchestrator(
          out_dir, hparam_policy="per_fold", ml_models_y=["rf"],
          ml_models_t=["rf"], run_sensitivity=False, optimize=False,
      )
      with mock.patch.object(
          hparam_search, "search_hyperparameters", search
      ), mock.patch.object(
          flaml_dml.FlamlRegressorDoubleML, "fit", _counting_fit
      ), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orch.run_full_pipeline(df=df, scaler=make_scaler(), top_n=1,
                               n_reps=n_reps)
      search.assert_not_called()
      self.assertEqual(
          glob.glob(os.path.join(out_dir, "step1_exploration_spend",
                                 "hparams_*.json")), [],
      )
      metrics = pd.read_csv(helpers.find(out_dir, "3a_iteration_detail.csv"))
      self.assertEqual(metrics["rep_id"].tolist(), list(range(1, n_reps + 1)))
      # 2 learners x S(K) folds per run; runs = 1 (Phase 1 = rep 0) +
      # (n_reps - 1) new repetitions.
      s_k = int(metrics["number_of_folds"].iloc[0])
      self.assertEqual(len(fit_calls), 2 * s_k * n_reps)
      # early_stop and n_jobs reach every per-fold search.
      self.assertEqual({es for _, es, _ in fit_calls}, {True})
      self.assertEqual({nj for _, _, nj in fit_calls}, {1})
      # The RMSE columns still come from FLAML's best_loss (finite).
      self.assertTrue(np.isfinite(
          metrics["nuisance_models_rmse_Y_model_evaluated_by_AutoML"]).all())


class PreparedInputsTest(unittest.TestCase):
  """df_norm / CATE basis built once are identical, results identical."""

  def _hparams(self, side, target):
    return hparam_search.HparamSearchResult(
        d_col="spend", side=side, learner="rf", target_col=target,
        estimator_cls=linear_model.LinearRegression, params={},
        best_loss=0.1, n_trials=1, n_iterations=1, elapsed_seconds=0.0,
        eval_method="cv", n_rows=180, n_features=3, time_budget=1.0,
        seed=1, n_jobs=1,
    )

  def test_run_with_prepared_inputs_equals_run_without(self):
    df = make_panel(seed=4)
    make = lambda: runner.DoubleMLPipeline(  # pylint: disable=g-long-lambda
        "sales", "spend", ["X1", "X2", "X3"], "rf", "rf", n_folds=3,
        covariates_for_cate=["X1"], cluster_cols=["geo"], random_state=1,
        n_jobs=1, hparams_y=self._hparams("Y", "sales"),
        hparams_t=self._hparams("T", "spend"),
    )
    scaler_a, scaler_b = make_scaler(), make_scaler()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      res_a = make().run(df, scaler_a, rep_id=1)
      prepared = runner.prepare_run_inputs(df, scaler_b, None, ["X1"],
                                           "auto_additive")
      res_b = make().run(df, scaler_b, rep_id=1, prepared=prepared)
    # df_norm and the basis built once equal what run() builds itself.
    pd.testing.assert_frame_equal(prepared.df_norm,
                                  make_scaler().fit_transform(df))
    pd.testing.assert_frame_equal(
        prepared.cate_basis,
        runner.build_cate_basis(prepared.df_norm, df, ["X1"]),
    )
    pd.testing.assert_frame_equal(res_a["df_1a"], res_b["df_1a"])
    self.assertEqual(res_a["metrics_1c"], res_b["metrics_1c"])
    self.assertEqual(res_a["cate_coef_dict"], res_b["cate_coef_dict"])
    self.assertEqual(res_a["model_name"], res_b["model_name"])
    # The RMSE columns are the internal validation RMSE.
    fold_models = res_b["model_obj"].models["ml_l"]["spend"][0]
    self.assertTrue(all(hasattr(m, "internal_val_rmse_") for m in fold_models))
    self.assertAlmostEqual(
        res_b["metrics_1c"]["nuisance_models_rmse_Y_model_evaluated_by_AutoML"],
        float(np.mean([m.internal_val_rmse_ for m in fold_models])),
    )

  def test_hand_over_guards(self):
    df = make_panel(seed=4)
    with self.assertRaises(ValueError):
      runner.DoubleMLPipeline(
          "sales", "spend", ["X1"], "rf", "rf",
          hparams_y=self._hparams("Y", "sales"),
          hparams_t=self._hparams("T", "other_column"),
      )
    with self.assertRaises(ValueError):
      runner.DoubleMLPipeline(
          "sales", "spend", ["X1"], "rf", "rf",
          hparams_y=self._hparams("Y", "sales"),
      )
    pipeline = runner.DoubleMLPipeline(
        "sales", "spend", ["X1", "X2", "X3"], "rf", "rf", n_folds=3,
        random_state=1, hparams_y=self._hparams("Y", "sales"),
        hparams_t=self._hparams("T", "spend"),
    )
    scaler = make_scaler()
    prepared = runner.prepare_run_inputs(df.iloc[:100], scaler, None, [],
                                         "auto_additive")
    with self.assertRaises(ValueError):
      pipeline.run(df.iloc[:100], scaler, prepared=prepared)


class ReduceRegroupTest(unittest.TestCase):
  """The ``reduce_rep_result`` / ``regroup_phase3_results`` helpers."""

  def test_reduce_rep_result_keeps_only_section_5_2_items(self):
    frame = pd.DataFrame({
        "estimated_incremental_KPI_spend": [1.0, 2.0],
        "estimated_incremental_KPI_spend_2.5%": [0.5, 1.5],
        "estimated_incremental_KPI_spend_97.5%": [1.5, 2.5],
        "spend": [10.0, 20.0],
        "truth": [1.1, 2.1],
        "Y_pred": [0.0, 0.0],
        "geo": ["a", "b"],
    }, index=[7, 3])
    res = {"model_name": "m", "model_group": "m", "metrics_1c": {"a": 1},
           "cate_coef_dict": {"Intercept_coef": 1.0}, "model_obj": object(),
           "df_1a": frame}
    reduced = workflow.reduce_rep_result(res, "spend", "truth")
    self.assertEqual(
        set(reduced), {"model_name", "model_group", "metrics_1c",
                       "cate_coef_dict", "df_1a"}
    )
    self.assertEqual(
        list(reduced["df_1a"].columns),
        ["estimated_incremental_KPI_spend",
         "estimated_incremental_KPI_spend_2.5%",
         "estimated_incremental_KPI_spend_97.5%", "spend", "truth"],
    )
    self.assertEqual(reduced["df_1a"].index.tolist(), [7, 3])
    without_gt = workflow.reduce_rep_result(res, "spend", None)
    self.assertNotIn("truth", without_gt["df_1a"].columns)

  def test_regroup_orders_candidates_and_reps(self):
    rep0 = {"A": {"model_name": "A", "rep": 0}, "B": {"model_name": "B",
                                                      "rep": 0}}
    jobs = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    flat = [{"model_name": n, "rep": r} for n, r in jobs]
    out = workflow.regroup_phase3_results(["A", "B"], rep0, jobs, flat)
    self.assertEqual([(o["model_name"], o["rep"]) for o in out],
                     [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1),
                      ("B", 2)])
    self.assertIs(out[0], rep0["A"])
    with self.assertRaises(ValueError):
      workflow.regroup_phase3_results(["A", "C"], rep0, jobs, flat)
    with self.assertRaises(ValueError):
      workflow.regroup_phase3_results(["A"], rep0, jobs, flat[:-1])
    # n_reps = 1: no jobs, rep 0 only.
    self.assertEqual(workflow.regroup_phase3_results(["A"], rep0, [], []),
                     [rep0["A"]])


class GlobalPolicyEndToEndTest(unittest.TestCase):
  """The default policy end to end (no saturation block)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df, cls.out_dir, cls.search, cls.log = _reference_run()

  def test_one_search_per_treatment_and_learner(self):
    """|Y| + |T| searches, Y target for the Y side, D for the T side."""
    self.assertEqual(self.search.call_count, 3)
    scaled = make_scaler().fit_transform(self.df)
    calls = {}
    for call in self.search.call_args_list:
      kwargs = call.kwargs
      calls.setdefault(kwargs["learner"], []).append(kwargs)
      self.assertEqual(kwargs["x"].shape, (len(self.df), 3))
      self.assertEqual(kwargs["n_jobs"], 1)
      self.assertEqual(kwargs["time_budget"], 1)
      self.assertEqual(kwargs["seed"], hparam_search.SEARCH_SEED)
    self.assertEqual(len(calls["rf"]), 2)  # Y side and T side
    self.assertEqual(len(calls["extra_tree"]), 1)  # Y side only
    targets = sorted(
        np.round(np.mean(k["y"]), 6) for k in calls["rf"]
    )
    want = sorted([
        round(float(scaled["sales"].mean()), 6),
        round(float(scaled["spend"].mean()), 6),
    ])
    self.assertEqual(targets, want)

  def test_hparams_json_written(self):
    """``hparams_<d_col>_<learner>.json`` is written per learner."""
    step1 = os.path.join(self.out_dir, "step1_exploration_spend")
    for learner, sides in (("rf", {"Y", "T"}), ("extra_tree", {"Y"})):
      path = os.path.join(step1, hparam_search.hparams_json_name("spend",
                                                                 learner))
      self.assertTrue(os.path.exists(path), path)
      with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
      self.assertEqual(payload["hparam_policy"], "global")
      self.assertEqual(set(payload["sides"]), sides)
      for side, data in payload["sides"].items():
        self.assertEqual(data["side"], side)
        self.assertEqual(data["learner"], learner)
        self.assertEqual(
            data["target_col"], "sales" if side == "Y" else "spend"
        )
        self.assertEqual(data["n_rows"], len(self.df))
        for key in ("params", "best_loss", "n_trials", "elapsed_seconds",
                    "eval_method", "estimator_cls"):
          self.assertIn(key, data)
      self.assertIn("selected_family", payload["context"])
      self.assertIsNone(payload["context"]["selected_family"])
      # The JSON round-trips into usable search results.
      loaded = hparam_search.read_hparams_json(path)
      self.assertEqual(set(loaded), sides)
      self.assertIs(loaded["Y"].estimator_cls, linear_model.LinearRegression)

  def test_log_lines_of_section_7(self):
    self.assertIn("hparam_policy = global", self.log)
    self.assertIn("Phase 1: workers = 1, threads/worker = 1", self.log)
    self.assertIn("Phase 3: workers = 1, threads/worker = 1", self.log)
    self.assertIn("Y-side learner 'rf'", self.log)
    self.assertIn("T-side learner 'rf'", self.log)
    self.assertIn("eval_method = cv", self.log)
    self.assertIn("saved ->", self.log)

  def test_phase3_rep0_is_the_phase1_result(self):
    """Metrics and KPI columns of rep 0 == Phase 1."""
    shortlist = pd.read_csv(helpers.find(self.out_dir,
                                       "2_shortlisted_models_metrics.csv"))
    shortlisted = shortlist[shortlist["shortlisted"] == 1]["model"].tolist()
    self.assertEqual(len(shortlisted), 2)
    detail = pd.read_csv(helpers.find(self.out_dir, "3a_iteration_detail.csv"))
    # Regrouped per candidate, repetitions 1..R ascending (R = 3).
    self.assertEqual(len(detail), 2 * 3)
    for name in shortlisted:
      rows = detail[detail["Model Group"] == name]
      self.assertEqual(rows["rep_id"].tolist(), [1, 2, 3])
      rep0 = rows.iloc[0]
      phase1 = pd.read_csv(
          os.path.join(self.out_dir, "step1_exploration_spend",
                       f"1c_metrics_{name}.csv")
      ).iloc[0]
      self.assertEqual(rep0["model"], name)
      for col in ("total_input", "total_estimated_incremental_KPI",
                  "total_estimated_roi", "rep_total_kpi",
                  "nuisance_models_rmse_Y_model_evaluated_by_AutoML",
                  "nuisance_models_rmse_T_model_evaluated_by_AutoML"):
        self.assertAlmostEqual(float(rep0[col]), float(phase1[col]),
                               places=10, msg=col)
    # Per-candidate order: candidate 1 rows first, then candidate 2.
    self.assertEqual(detail["Model Group"].tolist(),
                     [shortlisted[0]] * 3 + [shortlisted[1]] * 3)

  def test_best_model_outputs_are_full(self):
    """The best group's rep 0 keeps df_1a / df_1b / model_obj."""
    best_dir = glob.glob(os.path.join(self.out_dir, "step3_final_spend",
                                      "best_model"))[0]
    files = os.listdir(best_dir)
    self.assertTrue(any(f.startswith("3c_best_1a_df_") for f in files))
    self.assertTrue(any(f.startswith("3c_best_1b_geo_item_") for f in files))
    self.assertTrue(any(f.startswith("best_dml_object_") for f in files))
    best_1a = pd.read_csv(helpers.find(best_dir, [
        f for f in files if f.startswith("3c_best_1a_df_")][0]))
    for col in ("Y_pred", "T_pred", "Y_res", "T_res", "X1", "geo",
                "estimated_incremental_KPI_spend"):
      self.assertIn(col, best_1a.columns)
    consolidated = pd.read_csv(helpers.find(self.out_dir,
                                          "consolidated_1a_full_df.csv"))
    self.assertEqual(len(consolidated), len(self.df))

  def test_sensitivity_benchmark_total_matches_doubleml(self):
    """Total row == DoubleML's sensitivity_benchmark."""
    metrics = pd.read_csv(helpers.find(self.out_dir, "sensitivity_metrics.csv"))
    self.assertIn("benchmark_rho", metrics.columns)
    self.assertIn("benchmark_scope", metrics.columns)
    total = metrics[metrics["scope"] == "total"].iloc[0]
    self.assertEqual(total["benchmark_scope"], "whole_sample")
    self.assertEqual(total["benchmark_covariate"], "X3")
    geo_rows = metrics[metrics["scope"] == "geo"]
    self.assertEqual(len(geo_rows), 6)
    self.assertTrue((geo_rows["benchmark_scope"] == "group").all())
    # Group values are group-specific (not copies of the total row).
    self.assertGreater(geo_rows["benchmark_cf_y"].std(), 0.0)

    pkl = glob.glob(os.path.join(self.out_dir, "step3_final_spend",
                                 "best_model", "best_dml_object_*.pkl"))[0]
    with open(pkl, "rb") as fh:
      model = cloudpickle.load(fh)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      expected = model.sensitivity_benchmark(benchmarking_set=["X3"])
    for column, key in (("cf_d", "benchmark_cf_d"), ("cf_y", "benchmark_cf_y"),
                        ("rho", "benchmark_rho")):
      want = float(expected[column].iloc[0])
      got = float(total[key])
      with self.subTest(column=column):
        self.assertLessEqual(abs(got - want), 1e-6 * max(1.0, abs(want)))


class GlobalPolicyIdentityTest(unittest.TestCase):
  """Identical outputs for identical seeds and theta*."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df, reference_dir, _, _ = _reference_run()
    cls.reference = read_csvs(reference_dir)

  def _assert_same_csvs(self, out_dir):
    csvs = read_csvs(out_dir)
    self.assertTrue(csvs)
    self.assertEqual(set(csvs), set(self.reference))
    for name, frame in self.reference.items():
      with self.subTest(file=name):
        pd.testing.assert_frame_equal(
            frame, csvs[name], check_exact=False, rtol=1e-10
        )

  def test_two_runs_give_identical_step3_and_step4_csvs(self):
    with tempfile.TemporaryDirectory() as out_dir:
      run_global(out_dir, self.df)
      self._assert_same_csvs(out_dir)

  def test_reduced_worker_returns_do_not_change_any_csv(self):
    """Full worker returns (before) vs reduced (after): identical."""
    with tempfile.TemporaryDirectory() as out_dir:
      with mock.patch.object(
          workflow, "reduce_rep_result", lambda res, d_col, gt_col: res
      ):
        run_global(out_dir, self.df)
      self._assert_same_csvs(out_dir)


if __name__ == "__main__":
  unittest.main()
