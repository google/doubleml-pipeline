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

"""Spec acceptance tests of Phase 0: simulation, scale invariance, TV_spend.

Three families of acceptance tests, each a full or near-full run of the
saturation-aware pipeline; they share one ``large`` sharded BUILD target.

1. Simulation (20 data seeds each).  Each replication runs the Phase 0 path
   exactly as the workflow does: the cluster-key determination rule, the Stage
   A top-level split, then Stage B (``select_alpha``) on D_sel with se_obs
   clustered on the selected key with the fewer clusters.  D_sel stays below
   2000 rows, so the small-sample RidgeCV learners are used throughout.

   * linear truth                      -> selected_family=None in >= 16/20
     (linear_test_k = 1.0);
   * tanh truth, residual ratio >= 0.5 -> selected_family=tanh in >= 18/20 and
     the median half_saturation_T within x/÷1.5 of the truth;
   * tanh truth, residual ratio <= 0.1 -> Diag3 reports a small r_T and every
     run completes.  A narrow residual share has no rule and no warning of its
     own (the former identification diagnostic is abolished).

   Panel designs: tests 1-2 use ``T = 2 exp(0.4 X1 + geo_t) x lognormal(sd=1)``
   noise, so the true residual ratio is ~0.95.  Test 1 uses 40 geos x 200
   weeks (D_sel ~1600 rows): at 100 weeks the in-sample optimism of the
   saturated candidate over the (nested) linear one produced 8/20 linear
   decisions; at 200 weeks the rule behaves as designed.  Test 3 uses
   ``T = 3.5 + 1.2 X1 + 1.2 resid noise`` (X1 clipped to +-2.5), so the RidgeCV
   nuisance can represent E[T | X] out of fold and the estimated r_T tracks
   the true ratio ~= resid = 0.05.

2. Scale invariance under ``T -> 100 T``.  Every candidate is ``fbar(T; alpha)
   = f(T; alpha) / f(T_ref; alpha)`` with ``T_ref = q95(T | T > 0)``, so the
   whole of Phase 0 -- the T-basis (domain ``[0, T_max]``, knots at quantiles
   of ``T > 0``), the projection coefficients ``c(alpha) = P fbar_alpha``, the
   profiled R-loss, the identity candidate ``T / T_ref`` and the 1-SE decision
   -- is unit free.  Rescaling the treatment by 100 must leave
   ``selected_family``, ``decision``, the R-loss curves and ``theta_hat *
   fbar`` unchanged while ``half_saturation_T`` (and ``T_ref``, ``T_max``)
   scale by 100.  Two levels are checked:

   * Stage B directly (``alpha_selection.select_alpha`` on a D_sel-like
     panel): R-loss curves within a relative error of 1e-6, decision fields
     equal, ``half_saturation_T`` x 100, ``fbar(T; alpha_hat, T_ref)``
     identical.
   * The full pipeline (FLAML patched by OLS, no treatment normalization so
     that the raw ``T`` enters Phase 0): ``summary_alpha_selection.json``
     fields and the Phases 1-3 effect ``estimated_incremental_KPI = theta_hat
     * fbar`` must coincide, and the Phase 4 allocation must scale by 100 with
     the same optimized KPI (saturation-aware KKT allocation).

3. TV_spend-type regression for the fbar normalization.  Sporadic spend --
   60 % of the rows have ``T = 0`` -- and a nearly linear true response,
   ``tanh`` with half-saturation point ``T_1/2 = 20 q95(T | T > 0)``, with a
   known ``theta``.  Without the normalization the candidate functions
   ``f(T; alpha)`` had an alpha-dependent magnitude: for near-linear alpha the
   values are tiny, the ridge-regularised inner ``beta`` solve shrinks
   ``beta_hat ~ c / (c^2 + rho)`` toward 0, the profiled R-loss is biased
   against the linear side, and whenever such a family was adopted the effect
   ``theta_hat * f`` collapsed to ~0.  With the normalization every candidate
   passes through ``(T_ref, 1)``, so neither the comparison with the identity
   candidate nor the size of the recovered effect depends on the magnitude of
   ``f(.; alpha)``.  Acceptance criterion: the recovered effect -- the
   linear-DML ``theta_hat * T`` when ``selected_family is None``, else
   ``theta_hat * fbar`` -- must be within 10 % RMSE of the truth, and an
   effect collapsed below 10 % of the truth fails.  A second run forces
   adoption of the family (``linear_test_k = -inf``) so the ``theta_hat *
   fbar`` path itself is exercised whenever Phase 0 identifies alpha inside
   the grid.

The full pipeline runs patch FLAML with ``helpers.deterministic_fit`` so they
are fast and reproducible.  The heavy classes are listed first so that the
round-robin shard assignment of the pytest shard plugin spreads the full runs
evenly over the shards.
"""

import functools
import json
import tempfile
import unittest
from unittest import mock
import warnings

from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import alpha_selection
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.models import flaml_dml
from doubleml_pipeline.models import saturation
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
from doubleml_pipeline.utils import io as pipeline_io
import numpy as np
import pandas as pd


# 1. Simulation.
N_DATA_SEEDS = 20

TRUE_T_HALF = 2.0

LINEAR_TEST_WEEKS = 200


# 2. Scale invariance.
_SCALE = 100.0


# 3. TV_spend regression.
_T_HALF_MULT = 20.0  # T_1/2 = 20 q95(T > 0): practically linear on the support.

_ZERO_SHARE = 0.6

_THETA = 5.0

_MAX_REL_RMSE = 0.10

_COLLAPSE_RATIO = 0.10


def simulation_panel(
    seed: int,
    truth: str,
    resid: float,
    n_geo: int = 40,
    n_weeks: int = 100,
    theta: float = 4.0,
) -> pd.DataFrame:
  """Geo x week panel (see the module docstring for the two T designs)."""
  rng = np.random.default_rng(1000 + seed)
  dates = pd.date_range("2020-01-06", periods=n_weeks, freq="7D")
  geo_level = rng.normal(0.0, 0.3, n_geo)
  geo_t = rng.normal(0.0, 0.5, n_geo)
  # Diag3 r_T: with a narrow residual share, T is a linear function of X1
  # (which the RidgeCV learner represents out of fold) plus a small noise.
  # A geo-level T component is deliberately absent: Stage B folds are
  # cluster-level, so a held-out geo's level is (correctly) residual variation.
  narrow = resid < 0.3
  x_slope = 1.2
  frames = []
  for g in range(n_geo):
    x1 = rng.normal(size=n_weeks)
    x2 = rng.normal(size=n_weeks)
    if narrow:
      x1 = np.clip(x1, -2.5, 2.5)
      t = np.maximum(
          3.5 + x_slope * x1 + resid * x_slope * rng.normal(size=n_weeks),
          0.05,
      )
    else:
      base = 2.0 * np.exp(0.4 * x1 + geo_t[g])
      t = base * np.exp(resid * rng.normal(size=n_weeks) - resid**2 / 2.0)
    if truth == "tanh":
      effect = theta * np.tanh(np.log(3.0) / TRUE_T_HALF * t / 2.0)
    else:
      effect = 0.4 * theta * t / TRUE_T_HALF
    y = effect + 1.5 * x1 - x2 + geo_level[g] + rng.normal(size=n_weeks)
    frames.append(pd.DataFrame({
        "date": dates, "geo": f"G{g}", "item": "I0", "spend": t,
        "X1": x1, "X2": x2, "sales": y,
    }))
  return pd.concat(frames, ignore_index=True)


def run_phase0(df: pd.DataFrame, seed: int, linear_test_k: float = 1.0):
  """Stage A + Stage B, wired like `CausalWorkflowOrchestrator._run_phase0`."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plan = alpha_selection.determine_cluster_keys(
        df, item_col="item", geo_col="geo", date_col="date"
    )
    df_fit = plan.attach(df)
    split = alpha_selection.make_top_level_split(
        df_fit, plan, sel_fraction=0.2, k_folds=5, geo_col="geo",
        random_state=seed,
    )
    df_sel = df_fit.loc[split.idx_sel]
    # Cell-level Stage B folds, as in the workflow.
    cluster_labels, fold_key_values, fold_key_names = (
        alpha_selection.stage_b_cluster_inputs(df_sel, split, plan)
    )
    se_key = min(plan.keys, key=lambda k: plan.cluster_counts.get(k, 0))
    result = alpha_selection.select_alpha(
        df_sel=df_sel,
        y_col="sales",
        d_col="spend",
        x_cols=["X1", "X2"],
        basis_builder=functools.partial(
            runner.build_cate_basis, covariates_for_cate=["X1"],
            cate_type="auto_additive", cate_df=3, cate_degree=2,
            cate_intercept=False,
        ),
        scaler=scalers.CausalDataScaler(
            standardize_cols=["X1", "X2", "sales"]
        ),
        family="tanh",
        saturation_cfg={},
        selection_cfg={"n_seeds": 10, "k_folds": 5,
                       "linear_test_k": linear_test_k},
        cluster_labels=cluster_labels,
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        time_budget=1,
        n_jobs=1,
        random_state=seed,
        se_cluster_labels=df_sel[se_key].astype(str).to_numpy(),
        se_cluster_key=se_key,
        item_col="item",
        geo_col="geo",
        date_col="date",
        fold_key_values=fold_key_values,
        fold_key_names=fold_key_names,
    )
  return result, [str(w.message) for w in caught]


def rescaled(df, factor=_SCALE):
  out = df.copy()
  out["spend"] = out["spend"] * factor
  return out


def run_stage_b(df, family="tanh"):
  """Calls Stage B directly on ``df`` (treated as D_sel), T un-normalized."""
  scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
  builder = functools.partial(
      runner.build_cate_basis,
      covariates_for_cate=["X1"],
      cate_type="auto_additive",
      cate_df=3,
      cate_degree=2,
      cate_intercept=False,
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return alpha_selection.select_alpha(
        df_sel=df,
        y_col="sales",
        d_col="spend",
        x_cols=["X1", "X2"],
        basis_builder=builder,
        scaler=scaler,
        family=family,
        saturation_cfg={},
        selection_cfg={"n_seeds": 3, "k_folds": 3},
        cluster_labels=df["geo"].to_numpy(),
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        time_budget=1,
        n_jobs=1,
        se_cluster_labels=df["geo"].to_numpy(),
        se_cluster_key="geo",
        item_col="item",
        geo_col="geo",
        date_col="date",
    )


def run_raw_t_pipeline(df, out_dir):
  """Full pipeline, tanh family, raw T (no treatment normalization)."""
  scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
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
      # A non-zero ROI floor r in units of Y/T is not invariant to T -> 100*T
      # with Y unscaled (since ROI scales by 1/100); pure T-scale invariance of
      # the unconstrained KKT allocation holds at r = 0.
      opt_threshold_roi=0.0,
      saturation_config={
          "saturation": {"family": "tanh"},
          "alpha_selection": {"n_seeds": 3, "k_folds": 3},
      },
  )
  with mock.patch.object(
      flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
  ), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)
  cons = pd.read_csv(helpers.find(out_dir, "consolidated_1a_full_df.csv"))
  cons["date"] = pd.to_datetime(cons["date"])
  cons = cons.sort_values(["geo", "date"]).reset_index(drop=True)
  with open(helpers.find(out_dir, pipeline_io.SUMMARY_JSON)) as handle:
    summary = json.load(handle)
  return cons, summary


def make_tv_spend_panel(n_geo=20, n_weeks=100, seed=0):
  """Sporadic spend, nearly linear tanh truth, known theta.

  Args:
    n_geo: Number of geos in the panel.
    n_weeks: Number of weekly periods per geo.
    seed: Seed of the random number generator.

  Returns:
    ``(df, info)`` where ``df`` carries ``true_effect = theta * fbar_true(T)``
    and ``info`` the raw-scale ``t_ref``, ``t_half`` and ``theta``.
  """
  rng = np.random.default_rng(seed)
  dates = pd.date_range("2021-01-04", periods=n_weeks, freq="7D")
  frames = []
  geo_level = rng.normal(0.0, 0.5, n_geo)
  for g in range(n_geo):
    x1 = rng.normal(size=n_weeks)
    x2 = rng.normal(size=n_weeks)
    on = rng.random(n_weeks) < (1.0 - _ZERO_SHARE)
    spend = np.where(
        on,
        np.maximum(0.0, 20.0 + 5.0 * x1 + 4.0 * rng.normal(size=n_weeks)),
        0.0,
    )
    frames.append(pd.DataFrame({
        "date": dates, "geo": f"G{g}", "item": "I0", "spend": spend,
        "X1": x1, "X2": x2, "geo_level": geo_level[g],
        "noise": rng.normal(size=n_weeks),
    }))
  df = pd.concat(frames, ignore_index=True)
  t = df["spend"].to_numpy(dtype=float)
  t_ref = float(np.quantile(t[t > 0], 0.95))
  t_half = _T_HALF_MULT * t_ref
  lam = np.log(3.0) / t_half
  fbar_true = np.tanh(0.5 * lam * t) / np.tanh(0.5 * lam * t_ref)
  truth = _THETA * fbar_true
  df["sales"] = (
      truth + 1.5 * df["X1"] - df["X2"] + df["geo_level"] + df["noise"]
  )
  df["true_effect"] = truth
  df = df.drop(columns=["geo_level", "noise"])
  return df, {"t_ref": t_ref, "t_half": t_half, "theta": _THETA}


def run_tv_spend_pipeline(df, out_dir, selection_cfg=None):
  """Runs the full pipeline (tanh family) and returns the effect diagnostics."""
  scaler = scalers.CausalDataScaler(
      median_normalize_cols=["spend"], standardize_cols=["X1", "X2", "sales"]
  )
  sat_cfg = {
      "saturation": {"family": "tanh"},
      "alpha_selection": {"n_seeds": 3, "k_folds": 3, **(selection_cfg or {})},
  }
  orch = workflow.CausalWorkflowOrchestrator(
      y_col="sales", d_cols=["spend"], x_cols_list=[["X1", "X2"]],
      ml_models_y=["rf"], ml_models_t=["rf"], n_folds_list=[3],
      output_dir=out_dir, covariates_for_cate_list=[["X1"]],
      time_budget=1, n_jobs=1, date_col="date", geo_col="geo",
      # This test patches the per-fold FLAML search: keep the previous
      # hyperparameter policy (the default is "global").
      hparam_policy="per_fold",
      item_col="item", run_sensitivity=False, generate_html=False,
      treatment_types={"spend": "spend"},
      treatment_spend_cols={"spend": "spend"}, sales_col="sales",
      optimize=False, saturation_config=sat_cfg,
  )
  features = df.drop(columns=["true_effect"])
  with mock.patch.object(
      flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
  ), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    orch.run_full_pipeline(df=features, scaler=scaler, top_n=1, n_reps=1)

  cons = pd.read_csv(helpers.find(out_dir, "consolidated_1a_full_df.csv"))
  cons["date"] = pd.to_datetime(cons["date"])
  key = ["geo", "date"]
  merged = cons[key + ["estimated_incremental_KPI_spend",
                       "estimation_sample"]].merge(
      df[key + ["true_effect"]], on=key, how="left", validate="one_to_one"
  )
  with open(helpers.find(out_dir, pipeline_io.SUMMARY_JSON)) as handle:
    summary = json.load(handle)

  def _effect(frame):
    truth = frame["true_effect"].to_numpy(dtype=float)
    est = frame["estimated_incremental_KPI_spend"].to_numpy(dtype=float)
    return {
        "n_rows": int(len(est)),
        "rel_rmse": float(
            np.sqrt(np.mean((est - truth) ** 2)) / np.sqrt(np.mean(truth**2))
        ),
        "effect_ratio": float(est.sum() / truth.sum()),
    }

  out = _effect(merged)
  # The D_inf rows were fitted, the D_sel rows were scored with the fitted
  # CATE; both must recover the truth (every per-row output covers all rows).
  out["by_sample"] = {
      label: _effect(merged[merged["estimation_sample"] == label])
      for label in ("D_inf", "D_sel")
  }
  out["summary"] = summary
  return out


def _diagnostics(summary):
  keys = ("selected_family", "decision", "decision_reason", "alpha_hat",
          "half_saturation_T", "T_ref", "T_max", "d_bar", "se",
          "grid_extensions", "grid_edge_hit", "r_T", "normalization")
  return {k: summary.get(k) for k in keys}


class SimulationTest(unittest.TestCase):
  """The three simulation acceptance tests."""

  def test_linear_truth_selects_linear(self):
    """Linear truth -> selected_family=None in >= 16/20."""
    selected = []
    for seed in range(N_DATA_SEEDS):
      result, _ = run_phase0(
          simulation_panel(seed, "linear", resid=1.0, n_weeks=LINEAR_TEST_WEEKS),
          seed,
      )
      selected.append(result.selected_family)
      print(f"[linear] seed={seed} decision={result.decision}"
            f" d_bar={result.d_bar:.4g} se={result.se:.4g}"
            f" r_T={result.r_t:.3f} ext={result.grid_extensions}")
    n_linear = sum(s is None for s in selected)
    print(f"[linear] selected_family=None in {n_linear}/{N_DATA_SEEDS}")
    self.assertGreaterEqual(n_linear, 16)

  def test_tanh_truth_selects_tanh_and_recovers_t_half(self):
    """Tanh truth -> tanh in >= 18/20, T_half within x/÷1.5."""
    selected, t_halves, r_ts = [], [], []
    for seed in range(N_DATA_SEEDS):
      result, _ = run_phase0(simulation_panel(seed, "tanh", resid=1.0), seed)
      selected.append(result.selected_family)
      t_halves.append(result.half_saturation_t)
      r_ts.append(result.r_t)
      print(f"[tanh] seed={seed} decision={result.decision}"
            f" T_half={result.half_saturation_t:.3f}"
            f" d_bar={result.d_bar:.4g} se={result.se:.4g}"
            f" r_T={result.r_t:.3f}")
    n_tanh = sum(s == "tanh" for s in selected)
    median_t_half = float(np.median(t_halves))
    print(f"[tanh] selected_family=tanh in {n_tanh}/{N_DATA_SEEDS};"
          f" median T_half={median_t_half:.3f} (truth {TRUE_T_HALF});"
          f" median r_T={np.median(r_ts):.3f}")
    self.assertGreaterEqual(float(np.median(r_ts)), 0.5)
    self.assertGreaterEqual(n_tanh, 18)
    self.assertLessEqual(median_t_half, TRUE_T_HALF * 1.5)
    self.assertGreaterEqual(median_t_half, TRUE_T_HALF / 1.5)

  def test_narrow_residual_support_is_reported_and_completes(self):
    """Residual ratio <= 0.1: Diag3 reports a small r_T, every run completes.

    A narrow residual share has no rule and no warning of its own.
    """
    r_ts, diag2 = [], 0
    for seed in range(N_DATA_SEEDS):
      result, messages = run_phase0(simulation_panel(seed, "tanh", resid=0.05), seed)
      r_ts.append(result.r_t)
      diag2 += int(result.diag2_fired)
      # Former identification diagnostic 1 is abolished: no r_T warning.
      self.assertFalse(any("residual support is narrow" in m
                           for m in messages), messages)
      self.assertIn(result.decision, ("saturation", "linear"))
      print(f"[narrow] seed={seed} r_T={result.r_t:.3f}"
            f" diag2_fired={result.diag2_fired}"
            f" diag2_ratio={result.diag2_ratio:.3g}"
            f" decision={result.decision}")
    print(f"[narrow] median r_T={np.median(r_ts):.3f};"
          f" Diag2 fired in {diag2}/{N_DATA_SEEDS}")
    self.assertLessEqual(float(np.median(r_ts)), 0.1)


class PipelineScaleInvarianceTest(unittest.TestCase):
  """Scale invariance end to end: theta_hat * fbar and Phase 4."""

  def test_pipeline_outputs_are_scale_invariant(self):
    """Summary fields, theta_hat * fbar and Phase 4 agree under T -> 100 T."""
    df = helpers.saturating_panel(n_weeks=60, theta=6.0)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
      cons1, s1 = run_raw_t_pipeline(df, d1)
      cons2, s2 = run_raw_t_pipeline(rescaled(df), d2)

    print(
        "\n[scale x1]  ",
        json.dumps({
            k: s1.get(k)
            for k in (
                "selected_family",
                "decision",
                "decision_reason",
                "alpha_hat",
                "half_saturation_T",
                "T_ref",
                "T_max",
                "d_bar",
                "se",
            )
        }),
    )
    print(
        "[scale x100]",
        json.dumps({
            k: s2.get(k)
            for k in (
                "selected_family",
                "decision",
                "decision_reason",
                "alpha_hat",
                "half_saturation_T",
                "T_ref",
                "T_max",
                "d_bar",
                "se",
            )
        }),
    )

    # The scenario saturates clearly, so the family must be adopted (otherwise
    # the theta_hat * fbar path would not be exercised at all).
    self.assertEqual(s1["selected_family"], "tanh")
    self.assertEqual(s2["selected_family"], "tanh")
    for key in (
        "decision",
        "decision_reason",
        "grid_edge_hit",
        "J",
        "n_sel",
        "n_inf",
    ):
      self.assertEqual(s1[key], s2[key], key)
    np.testing.assert_allclose(
        s2["half_saturation_T"], _SCALE * s1["half_saturation_T"], rtol=1e-5
    )
    np.testing.assert_allclose(s2["T_ref"], _SCALE * s1["T_ref"], rtol=1e-12)
    np.testing.assert_allclose(s2["T_max"], _SCALE * s1["T_max"], rtol=1e-12)
    np.testing.assert_allclose(s2["d_bar"], s1["d_bar"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(s2["se"], s1["se"], rtol=1e-5, atol=1e-12)

    # Phases 1-3: the effect theta_hat * fbar is identical.  The model is
    # fitted on D_inf and the held-out D_sel rows are scored with the fitted
    # CATE, so the consolidated frame covers every input row of the panel.
    self.assertEqual(len(cons1), len(cons2))
    self.assertEqual(len(cons1), s1["n_inf"] + s1["n_sel"])
    self.assertEqual(len(cons1), len(df))
    self.assertEqual(
        cons1["estimation_sample"].tolist(), cons2["estimation_sample"].tolist()
    )
    self.assertEqual(
        int((cons1["estimation_sample"] == "D_inf").sum()), s1["n_inf"]
    )
    self.assertEqual(
        int((cons1["estimation_sample"] == "D_sel").sum()), s1["n_sel"]
    )
    kpi1 = cons1["estimated_incremental_KPI_spend"].to_numpy(dtype=float)
    kpi2 = cons2["estimated_incremental_KPI_spend"].to_numpy(dtype=float)
    self.assertGreater(float(np.abs(kpi1).sum()), 0.0)
    np.testing.assert_allclose(kpi2, kpi1, rtol=1e-5, atol=1e-8)
    # The scored rows carry a finite, non-trivial effect of their own.
    scored = (cons1["estimation_sample"] == "D_sel").to_numpy()
    self.assertTrue(np.isfinite(kpi1[scored]).all())
    self.assertGreater(float(np.abs(kpi1[scored]).sum()), 0.0)
    # Raw treatment is restored in the consolidated CSV (x100 between runs).
    np.testing.assert_allclose(
        cons2["spend"].to_numpy(dtype=float),
        _SCALE * cons1["spend"].to_numpy(dtype=float),
        rtol=1e-10,
    )

    # Phase 4 (KKT allocation): the allocation scales by 100, the
    # transformed allocation fbar(T*) is unit free and the optimized KPI
    # coincides.
    self.assertIn("optimized_spend", cons1.columns)
    np.testing.assert_allclose(
        cons2["optimized_spend"].to_numpy(dtype=float),
        _SCALE * cons1["optimized_spend"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        cons2["optimized_spend_transformed"].to_numpy(dtype=float),
        cons1["optimized_spend_transformed"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cons2["optimized_incremental_KPI_spend"].to_numpy(dtype=float),
        cons1["optimized_incremental_KPI_spend"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-8,
    )


class TvSpendRegressionTest(unittest.TestCase):
  """The recovered TV-spend effect must stay within 10 % of the truth."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df, cls.info = make_tv_spend_panel()

  def _check_effect(self, out, label):
    """Asserts the recovered effect is within tolerance and prints it."""
    print(f"\n[{label}] effect diagnostics:", json.dumps({
        "rel_rmse": out["rel_rmse"], "effect_ratio": out["effect_ratio"],
        "by_sample": out["by_sample"],
        **_diagnostics(out["summary"]),
    }, default=str), flush=True)
    # Phases 1-3 are FITTED on D_inf (Stage A top-level split) and the held-out
    # D_sel rows are SCORED with the fitted CATE, so the consolidated frame
    # covers every input row.
    summary = out["summary"]
    self.assertEqual(out["n_rows"], summary["n_inf"] + summary["n_sel"])
    self.assertEqual(out["n_rows"], len(self.df))
    self.assertEqual(out["by_sample"]["D_inf"]["n_rows"], summary["n_inf"])
    self.assertEqual(out["by_sample"]["D_sel"]["n_rows"], summary["n_sel"])
    self.assertGreater(summary["n_sel"], 0)
    # Collapse guard (the pre-normalization failure mode): the total recovered
    # effect must not fall below 10 % of the truth -- overall and on each
    # sample, so that a wrong scoring of the held-out rows cannot hide behind
    # the fitted ones.
    checks = {"all rows": out, **out["by_sample"]}
    for sample, eff in checks.items():
      self.assertGreater(
          eff["effect_ratio"], _COLLAPSE_RATIO,
          f"{label} / {sample}: effect collapsed to"
          f" {eff['effect_ratio']:.3f} x truth",
      )
      self.assertLess(
          eff["rel_rmse"], _MAX_REL_RMSE,
          f"{label} / {sample}: relative RMSE {eff['rel_rmse']:.4f}"
          " exceeds 10 %",
      )

  def _check_summary(self, summary):
    """Checks the Phase 0 summary (T_ref, divisor, decision bookkeeping)."""
    # T_ref is the q95 of the normalized T > 0 on the full data; with
    # median_normalize c = median(T > 0).
    t = self.df["spend"].to_numpy(dtype=float)
    c = float(np.median(t[t > 0]))
    self.assertAlmostEqual(summary["T_ref"], self.info["t_ref"] / c, places=6)
    self.assertIn(summary["decision_reason"],
                  ("1se_pass", "1se_fail", "grid_edge_upper"))
    self.assertIn(summary["grid_edge_hit"], ("none", "upper", "lower"))
    self.assertEqual(summary["decision"] == "saturation",
                     summary["selected_family"] == "tanh")
    # An upper-edge hit never adopts the family.
    if summary["grid_edge_hit"] == "upper":
      self.assertIsNone(summary["selected_family"])
      self.assertEqual(summary["decision_reason"], "grid_edge_upper")

  def test_default_decision_recovers_the_effect(self):
    """Default 1-SE rule: the recovered effect is within 10 % of the truth.

    Over the observed support (T mostly below T_ref) the candidate shapes
    differ by only a few percent, so the R-loss profile is flat (|d_bar| << se)
    and the 1-SE rule is expected to keep the linear model; the effect then
    comes from the linear DML.  Whatever the decision, the acceptance bounds
    must hold.
    """
    with tempfile.TemporaryDirectory() as out_dir:
      out = run_tv_spend_pipeline(self.df, out_dir)
    self._check_effect(out, "default")
    self._check_summary(out["summary"])

  def test_forced_adoption_recovers_the_effect(self):
    """theta_hat * fbar path: forcing the family must not collapse the effect.

    ``linear_test_k = -inf`` makes the 1-SE rule always adopt the family; only
    the upper-edge rule can still fall back to linear.  Either way the
    recovered effect must be within 10 % of the truth: with every candidate
    normalised through ``(T_ref, 1)`` the magnitude of ``theta_hat * fbar`` no
    longer depends on where alpha_hat lands on a flat profile.
    """
    with tempfile.TemporaryDirectory() as out_dir:
      out = run_tv_spend_pipeline(
          self.df, out_dir, selection_cfg={"linear_test_k": -1e12}
      )
    self._check_effect(out, "forced adoption")
    self._check_summary(out["summary"])
    summary = out["summary"]
    self.assertIn(summary["decision_reason"], ("1se_pass", "grid_edge_upper"))
    if summary["decision_reason"] == "1se_pass":
      self.assertEqual(summary["selected_family"], "tanh")


class StageBScaleInvarianceTest(unittest.TestCase):
  """Scale invariance at the Stage B level."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df = helpers.saturating_panel(n_weeks=60, theta=6.0)
    cls.r1 = run_stage_b(cls.df)
    cls.r2 = run_stage_b(rescaled(cls.df))

  def test_decision_fields_coincide(self):
    r1, r2 = self.r1, self.r2
    self.assertEqual(r1.selected_family, r2.selected_family)
    self.assertEqual(r1.decision, r2.decision)
    self.assertEqual(r1.decision_reason, r2.decision_reason)
    self.assertEqual(r1.grid_edge_hit, r2.grid_edge_hit)
    self.assertEqual(r1.n_basis, r2.n_basis)
    self.assertEqual(r1.alpha_grid.size, r2.alpha_grid.size)

  def test_rloss_curves_coincide(self):
    """R-loss curves (CATE, intercept, identity) within rel. err. 1e-6."""
    r1, r2 = self.r1, self.r2
    np.testing.assert_allclose(r2.per_seed_loss, r1.per_seed_loss, rtol=1e-6)
    np.testing.assert_allclose(
        r2.per_seed_loss_ate, r1.per_seed_loss_ate, rtol=1e-6
    )
    np.testing.assert_allclose(
        r2.per_seed_loss_identity, r1.per_seed_loss_identity, rtol=1e-6
    )
    np.testing.assert_allclose(r2.mean_loss, r1.mean_loss, rtol=1e-6)
    np.testing.assert_allclose(r2.d_bar, r1.d_bar, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(r2.se, r1.se, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(r2.r_t, r1.r_t, rtol=1e-6)
    np.testing.assert_allclose(
        r2.basis_approx_error_alpha,
        r1.basis_approx_error_alpha,
        rtol=1e-6,
        atol=1e-12,
    )

  def test_scale_parameters_scale_by_100(self):
    """half_saturation_T, T_ref and T_max scale x100; alpha_hat (tanh) /100."""
    r1, r2 = self.r1, self.r2
    # Brent refines alpha in the log T_1/2 coordinate (xatol 1e-5), so the
    # refined point is compared at 1e-5 while the grid is exact.
    np.testing.assert_allclose(
        r2.half_saturation_t, _SCALE * r1.half_saturation_t, rtol=1e-5
    )
    np.testing.assert_allclose(r2.alpha_hat, r1.alpha_hat / _SCALE, rtol=1e-5)
    np.testing.assert_allclose(r2.t_ref, _SCALE * r1.t_ref, rtol=1e-12)
    np.testing.assert_allclose(r2.t_max, _SCALE * r1.t_max, rtol=1e-12)
    np.testing.assert_allclose(
        r2.alpha_grid, r1.alpha_grid / _SCALE, rtol=1e-10
    )

  def test_fbar_at_alpha_hat_is_identical(self):
    """theta_hat * fbar coincides because fbar itself is unit free."""
    r1, r2 = self.r1, self.r2
    t = self.df["spend"].to_numpy(dtype=float)
    fbar1 = saturation.normalized_saturation(t, "tanh", r1.alpha_hat, r1.t_ref)
    fbar2 = saturation.normalized_saturation(
        _SCALE * t, "tanh", r2.alpha_hat, r2.t_ref
    )
    np.testing.assert_allclose(fbar2, fbar1, rtol=1e-5, atol=1e-12)
    # Diag4: the shape quantiles phi_p(alpha) are unit free.
    np.testing.assert_allclose(
        r2.fbar_quantiles_grid, r1.fbar_quantiles_grid, rtol=1e-6
    )
    np.testing.assert_allclose(r2.fbar_quantiles, r1.fbar_quantiles, rtol=1e-5)
    np.testing.assert_allclose(
        r2.linear_ref_quantiles, r1.linear_ref_quantiles, rtol=1e-10
    )
    np.testing.assert_allclose(r2.diag2_ratio, r1.diag2_ratio, rtol=1e-6)


if __name__ == "__main__":
  unittest.main()
