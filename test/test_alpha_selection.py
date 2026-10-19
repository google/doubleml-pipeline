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

"""Tests for Phase 0 (``evaluation.alpha_selection``): Stage A and Stage B.

Stage A -- the cell-level split unit and coverage guarantee.  With two adopted
cluster keys the split unit of the top-level split and of the Stage B K-folds
is the intersection cell of the two keys.  The tests check that

(i)   every level of each key has >= 1 cell on each side of D_sel / D_inf and
      in each Stage B fold;
(ii)  no cell straddles the two sides or two folds;
(iii) the ``sel`` share of cells is within +-2 percentage points of
      ``sel_fraction``;

plus the round-robin / interleaving rule itself, the unbalanced-panel repair,
the single-key case and the final fallback.

Stage B -- alpha selection:

* 1-SE comparison with the identity candidate: the cluster-robust se_obs and
  the decision rule on synthetic curves.  The standard error of the 1-SE
  decision is the cluster-robust ``se_obs`` only (no across-seed SE, no
  SE-type switch); the same ``se(alpha)`` (paired against ``alpha_hat``)
  drives Diag2.
* p(X) identity between Phase 0 and Phases 1-3; intercept curve
  (visualization only).
* Phase 0 output contract (file names, columns, JSON keys); the identity
  candidate names (``rloss_identity``, ``rloss_identity_s``,
  ``fallback_half_saturation_T``) and the absence of every abolished name in
  the sources and the outputs.
* Fixed default learners, small-sample switch, fit accounting; grid + Brent
  only: no per-alpha nuisance retraining; out-of-fold nuisances everywhere
  with an honest inner beta fit.
* Diag1 (``eps(J)`` on the final grid), Diag2 (1-SE flatness ratio over
  ``A_1SE``), Diag3 (``r_T``), Diag4 (shape quantiles ``phi_p``), Diag5
  (``kappa(alpha_hat)``) and the Diag6 helper; Diag1 / Diag2 warn and continue
  and the fixed ``fallback_half_saturation_T`` is used downstream when they
  fire.
* The grid-edge rule and the ground-truth recovery of a known exponent.
* The normalization bridge between Phase 0 and Phase 4: ``treatment_divisor``
  mirrors ``CausalDataScaler``'s own arithmetic (global median, row-varying
  population median; shifting normalizations are rejected).
* The compute budget: Phase 0 trains ``2 S K`` nuisance models (plus
  ``N_y + N_t`` selection fits only with ``learner_selection``); neither the
  alpha grid size nor the B-spline dimension J multiplies it, and the FLAML
  time budget shrinks with ``D_sel``.

The config schema of the orchestrator (accepted / retired keys) is covered by
test_workflow.py; the ``geo=6, daily, no item_col`` case is run end to end in
test_workflow_e2e.py.
"""

import functools
import json
import os
import re
import tempfile
import unittest
from unittest import mock
import warnings

import doubleml_pipeline
from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import alpha_selection
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.evaluation import saturation_evaluation
from doubleml_pipeline.models import saturation
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
from doubleml_pipeline.utils import io as pipeline_io
import numpy as np
import pandas as pd


# Names retired from earlier revisions of the package; none may remain in
# the package sources.  The list is assembled from pieces so
# that this test file itself does not contain the literal tokens.
_FORBIDDEN = [
    "se_" + "seed",
    "se_" + "type",
    "crit" + "erion",
    "rloss_" + "linear",
    "fallback_" + "alpha",
    "roi_" + "threshold",
    "flatness_" + "threshold",
    "dispersion_" + "threshold",
    "resid_ratio_" + "threshold",
    "fbar_median_" + "obs",
    "fbar_q90_" + "obs",
    "linear " + "candidate",
    "combine_" + "se",
    "allocate_with_" + "roi",
    "alpha_hat_" + "ate",
]


def run_select_with_messages(
    df, selection_cfg=None, saturation_cfg=None, cate=("X1",), builder=None,
    family="tanh",
):
  """Calls Stage B directly on ``df`` (treated as D_sel).

  Returns:
    ``(result, messages)``: the ``select_alpha`` result and the text of the
    warnings emitted while it ran.
  """
  scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
  cfg = {"n_seeds": 3, "k_folds": 3}
  cfg.update(selection_cfg or {})
  if builder is None:
    builder = functools.partial(
        runner.build_cate_basis,
        covariates_for_cate=list(cate),
        cate_type="auto_additive",
        cate_df=3,
        cate_degree=2,
        cate_intercept=False,
    )
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = alpha_selection.select_alpha(
        df_sel=df,
        y_col="sales",
        d_col="spend",
        x_cols=["X1", "X2"],
        basis_builder=builder,
        scaler=scaler,
        family=family,
        saturation_cfg=saturation_cfg or {},
        selection_cfg=cfg,
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
  return result, [str(w.message) for w in caught]


def run_select(df, **kwargs):
  """``run_select_with_messages`` without the captured warnings."""
  return run_select_with_messages(df, **kwargs)[0]


DATE_BLOCK = alpha_selection.DATE_BLOCK_COL


def _key_panel(n_geo, n_days, drop=0.0, seed=0, with_date=True):
  """Geo x day panel (key columns only); ``drop`` removes whole geo x weeks."""
  rng = np.random.default_rng(seed)
  dates = pd.date_range("2022-01-03", periods=n_days, freq="D")  # a Monday.
  frame = pd.DataFrame({
      "geo": np.repeat([f"G{g:02d}" for g in range(n_geo)], n_days),
      "date": np.tile(dates, n_geo),
  })
  if drop:
    week = (frame["date"] - dates[0]).dt.days // 7
    cell = frame["geo"] + "|" + week.astype(str)
    uniq = cell.unique()
    dropped = set(uniq[rng.random(len(uniq)) < drop])
    frame = frame.loc[~cell.isin(dropped)].reset_index(drop=True)
  if not with_date:
    frame = frame.drop(columns="date")
  return frame


def _split(frame, sel_fraction=0.2, k_folds=5, seed=0):
  """Runs the cluster-key rule and the Stage A split like the workflow does."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plan = alpha_selection.determine_cluster_keys(
        frame, geo_col="geo",
        date_col="date" if "date" in frame.columns else None,
    )
    frame = plan.attach(frame)
    split = alpha_selection.make_top_level_split(
        frame, plan, sel_fraction=sel_fraction, k_folds=k_folds,
        geo_col="geo", random_state=seed,
    )
  return plan, frame, split


def _stage_b_folds(frame, split, k_folds, n_seeds=3):
  """Stage B cell-level folds of D_sel, one partition per seed."""
  df_sel = frame.loc[split.idx_sel]
  key1, key2 = split.split_keys
  folds = [
      alpha_selection.cell_kfold(
          df_sel[key1], df_sel[key2], k_folds,
          np.random.default_rng(100 + s),
      )
      for s in range(n_seeds)
  ]
  return df_sel, folds


class CellLevelSplitTest(unittest.TestCase):
  """50 geos x 420 days -> 50 geos x 60 week blocks = 3000 cells."""

  K = 5

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.plan, cls.frame, cls.split = _split(_key_panel(50, 420), k_folds=cls.K)
    cls.df_sel, cls.folds = _stage_b_folds(cls.frame, cls.split, cls.K)

  def test_split_unit_is_the_intersection_cell(self):
    split = self.split
    self.assertEqual(self.plan.keys, [DATE_BLOCK, "geo"])
    self.assertEqual(split.unit, "cell")
    # k2 = the date-block key (indexed in time order), k1 = geo.
    self.assertEqual(split.split_keys, ["geo", DATE_BLOCK])
    self.assertTrue(split.tier.startswith("tier-1 (cell-level"))
    self.assertEqual(split.n_clusters_total, 50 * 60)
    self.assertFalse(split.forced_full_sample)
    self.assertEqual(
        len(split.idx_sel) + len(split.idx_inf), len(self.frame)
    )

  def test_every_level_on_both_sides(self):
    """(i) top level: every level of every key in D_sel AND in D_inf."""
    sel = self.frame.loc[self.split.idx_sel]
    inf = self.frame.loc[self.split.idx_inf]
    for key in self.split.split_keys:
      with self.subTest(key=key):
        levels = set(self.frame[key].astype(str))
        self.assertEqual(set(sel[key].astype(str)), levels)
        self.assertEqual(set(inf[key].astype(str)), levels)
    self.assertTrue(self.split.coverage_ok)
    self.assertTrue(self.split.coverage["covered"].all())
    self.assertEqual(len(self.split.coverage), 50 + 60)

  def test_every_level_in_every_fold(self):
    """(i) Stage B: every level of every key in each of the K folds."""
    for seed, folds in enumerate(self.folds):
      for key in self.split.split_keys:
        with self.subTest(seed=seed, key=key):
          n_folds = (
              pd.DataFrame({"level": self.df_sel[key].to_numpy(),
                            "fold": folds})
              .groupby("level")["fold"].nunique()
          )
          self.assertEqual(int(n_folds.min()), self.K)

  def test_no_cell_straddles_sides_or_folds(self):
    """(ii) a cell is entirely in one side and in one fold."""
    cell_ids = self.split.cell_ids
    self.assertEqual(
        set(cell_ids.loc[self.split.idx_sel])
        & set(cell_ids.loc[self.split.idx_inf]),
        set(),
    )
    sel_cells = cell_ids.loc[self.df_sel.index].to_numpy()
    for seed, folds in enumerate(self.folds):
      with self.subTest(seed=seed):
        per_cell = (
            pd.DataFrame({"cell": sel_cells, "fold": folds})
            .groupby("cell")["fold"].nunique()
        )
        self.assertEqual(int(per_cell.max()), 1)

  def test_round_robin_rule_and_time_interleaving(self):
    """Round-robin rule: (idx(b) + pi(a)) mod M == 0 on a complete panel."""
    blocks = sorted(self.frame[DATE_BLOCK].unique())
    idx = {b: i for i, b in enumerate(blocks)}
    sel = self.frame.loc[self.split.idx_sel, ["geo", DATE_BLOCK]]
    sel = sel.drop_duplicates()
    residues = set()
    for geo, group in sel.groupby("geo"):
      positions = np.sort(group[DATE_BLOCK].map(idx).to_numpy())
      with self.subTest(geo=geo):
        # every M-th week block of this geo -> interleaved along time.
        self.assertTrue(np.all(np.diff(positions) == 5), positions)
        self.assertEqual(len(positions), 60 // 5)
      residues.add(int(positions[0] % 5))
    # pi is a permutation, so the phase differs across geos.
    self.assertEqual(residues, set(range(5)))

  def test_sel_share_within_two_points(self):
    """(iv) the D_sel share of cells is sel_fraction +- 2pp (incl. greedy)."""
    for sel_fraction in (0.2, 0.25, 0.3, 0.15):
      with self.subTest(sel_fraction=sel_fraction):
        _, _, split = _split(_key_panel(50, 420), sel_fraction=sel_fraction)
        self.assertEqual(split.unit, "cell")
        share = split.n_clusters_sel / split.n_clusters_total
        self.assertLessEqual(abs(share - sel_fraction), 0.02, share)
        self.assertTrue(split.coverage_ok)

  def test_stage_b_inputs_are_cell_level(self):
    labels, fold_values, fold_names = alpha_selection.stage_b_cluster_inputs(
        self.df_sel, self.split, self.plan
    )
    self.assertEqual(fold_names, ("geo", DATE_BLOCK))
    np.testing.assert_array_equal(
        labels, self.split.cell_ids.loc[self.df_sel.index].to_numpy()
    )
    self.assertEqual(len(fold_values[0]), len(self.df_sel))


class CellLevelSplitEdgeCaseTest(unittest.TestCase):
  """Unbalanced panels, the single-key case and the final fallback."""

  def test_unbalanced_panel_keeps_coverage(self):
    """Missing rows: the repair step restores the coverage guarantee."""
    for seed in range(3):
      with self.subTest(seed=seed):
        _, frame, split = _split(_key_panel(30, 400, drop=0.15, seed=seed))
        self.assertEqual(split.unit, "cell")
        self.assertTrue(split.coverage_ok, split.coverage[
            ~split.coverage["covered"]])
        cell_ids = split.cell_ids
        self.assertFalse(
            set(cell_ids.loc[split.idx_sel])
            & set(cell_ids.loc[split.idx_inf])
        )
        share = split.n_clusters_sel / split.n_clusters_total
        self.assertLessEqual(abs(share - 0.2), 0.02, share)
        df_sel, folds = _stage_b_folds(frame, split, 5, n_seeds=2)
        for folds_s in folds:
          for key in split.split_keys:
            per_level = pd.DataFrame(
                {"level": df_sel[key].to_numpy(), "fold": folds_s}
            ).groupby("level")["fold"].agg(["nunique", "size"])
            cells_per_level = (
                pd.DataFrame({"level": df_sel[key].to_numpy(),
                              "cell": cell_ids.loc[df_sel.index].to_numpy()})
                .drop_duplicates().groupby("level").size()
            )
            feasible = cells_per_level[cells_per_level >= 5].index
            self.assertTrue(
                (per_level.loc[feasible, "nunique"] == 5).all(),
                per_level.loc[feasible],
            )

  def test_single_key_split_when_only_one_key(self):
    """One adopted key -> its levels are the split unit."""
    plan, frame, split = _split(_key_panel(40, 10, with_date=False))
    self.assertEqual(plan.keys, ["geo"])
    self.assertEqual(split.unit, "single_key")
    self.assertTrue(split.tier.startswith("tier-2"))
    self.assertFalse(split.coverage_ok)
    _, fold_values, _ = alpha_selection.stage_b_cluster_inputs(
        frame.loc[split.idx_sel], split, plan
    )
    self.assertIsNone(fold_values)

  def test_too_few_levels_take_the_final_fallback(self):
    """C1 < M -> warning + final fallback (full_sample)."""
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      plan = alpha_selection.determine_cluster_keys(
          _key_panel(3, 420), geo_col="geo", date_col="date"
      )
      split = alpha_selection.make_top_level_split(
          plan.attach(_key_panel(3, 420)), plan, sel_fraction=0.2
      )
    self.assertEqual(split.unit, "observation")
    self.assertTrue(split.forced_full_sample)
    self.assertTrue(split.tier.startswith("tier-3"))
    self.assertTrue(
        any("C1 >= M and C2 >= M" in str(w.message) for w in caught)
    )


class SeHelpersTest(unittest.TestCase):
  """The 1-SE linear test: cluster-robust se_obs and the decision rule."""

  def test_cluster_robust_se_matches_cr1(self):
    """se_obs equals the CR1 cluster-robust formula."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    clusters = np.repeat(np.arange(20), 10)
    got = alpha_selection.cluster_robust_se(values, clusters)
    centered = values - values.mean()
    sums = np.array([centered[clusters == g].sum() for g in range(20)])
    want = np.sqrt(20 / 19 * np.sum(sums**2)) / 200
    self.assertAlmostEqual(got, want, places=12)
    iid = alpha_selection.cluster_robust_se(values, None)
    self.assertAlmostEqual(iid, values.std(ddof=1) / np.sqrt(200), places=12)
    mat = np.column_stack([values, 2 * values])
    np.testing.assert_allclose(
        alpha_selection.cluster_robust_se(mat, clusters), [want, 2 * want]
    )

  def test_decision_rule_on_synthetic_curves(self):
    """Saturation is adopted iff d_bar < -k se, on synthetic curves."""
    rng = np.random.default_rng(1)
    n_obs = 500
    for gap, k, expected in (
        (-0.30, 1.0, True),  # clear improvement over the identity candidate
        (-0.30, 50.0, False),  # same curves, very conservative k
        (-0.001, 1.0, False),  # within noise
        (+0.10, 1.0, False),  # saturation worse than the identity candidate
    ):
      with self.subTest(gap=gap, k=k):
        identity_obs = rng.chisquare(1, size=n_obs)
        sat_obs = identity_obs + gap + 0.5 * rng.normal(size=n_obs)
        d_i = sat_obs - identity_obs
        # se is the cluster-robust se_obs of d_i only.
        se = alpha_selection.cluster_robust_se(d_i)
        self.assertGreater(se, 0.01)
        self.assertLess(se, 0.1)
        self.assertEqual(
            alpha_selection.linear_test_decision(float(d_i.mean()), se, k),
            expected,
        )
    self.assertFalse(alpha_selection.linear_test_decision(np.nan, 0.1, 1.0))
    self.assertFalse(alpha_selection.linear_test_decision(-1.0, np.nan, 1.0))

  def test_two_way_se_matches_cameron_gelbach_miller(self):
    """se_two_way^2 = V_key1 + V_key2 - V_cell (each with G / (G - 1))."""
    rng = np.random.default_rng(2)
    n_geo, n_block = 12, 8
    geo = np.repeat(np.arange(n_geo), n_block)
    block = np.tile(np.arange(n_block), n_geo)
    # Shocks along both keys so that both one-way se's are too small.
    values = (
        rng.normal(size=n_geo)[geo] + rng.normal(size=n_block)[block]
        + 0.3 * rng.normal(size=geo.size)
    )
    n_obs = values.size

    def cr1_var(labels):
      centered = values - values.mean()
      sums = np.array([centered[labels == g].sum() for g in np.unique(labels)])
      n_groups = sums.size
      return n_groups / (n_groups - 1) * np.sum(sums**2)

    cell = geo * n_block + block
    want = np.sqrt(cr1_var(geo) + cr1_var(block) - cr1_var(cell)) / n_obs
    got = alpha_selection.cluster_robust_se(values, geo, block)
    self.assertAlmostEqual(got, want, places=12)
    # Symmetric in the two keys; strictly larger than either one-way se here.
    self.assertAlmostEqual(
        alpha_selection.cluster_robust_se(values, block, geo), want, places=12
    )
    one_way_geo = alpha_selection.cluster_robust_se(values, geo)
    one_way_block = alpha_selection.cluster_robust_se(values, block)
    self.assertGreater(got, max(one_way_geo, one_way_block))
    # Column-wise for a matrix input.
    mat = np.column_stack([values, 2.0 * values])
    np.testing.assert_allclose(
        alpha_selection.cluster_robust_se(mat, geo, block), [want, 2 * want]
    )
    # A degenerate second key (one cluster) falls back to one-way.
    self.assertAlmostEqual(
        alpha_selection.cluster_robust_se(values, geo, np.zeros(n_obs)),
        one_way_geo,
    )
    self.assertEqual(alpha_selection.count_clusters(geo), n_geo)
    self.assertEqual(alpha_selection.count_clusters(None), 0)

  def test_two_way_se_non_positive_variance_falls_back(self):
    """V <= 0 -> the larger one-way variance (never NaN)."""
    # Alternating signs: every key1 pair and every key2 group nearly cancels
    # (small V_key1, V_key2) while the singleton cells do not (large V_cell),
    # so V = V_key1 + V_key2 - V_cell < 0 and the fallback max(V_1, V_2)
    # applies.
    values = np.array([1.1, -1.0, -1.0, 0.9, 1.0, -1.1, -0.9, 1.0])
    key1 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    key2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    se_1 = alpha_selection.cluster_robust_se(values, key1)
    se_2 = alpha_selection.cluster_robust_se(values, key2)
    got = alpha_selection.cluster_robust_se(values, key1, key2)
    self.assertTrue(np.isfinite(got))
    self.assertGreater(got, 0.0)
    self.assertGreater(se_2, se_1)
    self.assertAlmostEqual(got, max(se_1, se_2))

  def test_t_critical_value(self):
    """crit = t_df^{-1}(Phi(k)): off -> k; decreasing in df towards k."""
    self.assertEqual(alpha_selection.linear_test_critical_value(1.0, 9, False),
                     1.0)
    self.assertEqual(alpha_selection.linear_test_critical_value(1.0, None, True),
                     1.0)
    self.assertEqual(alpha_selection.linear_test_critical_value(1.0, 0, True),
                     1.0)
    crit = {
        df: alpha_selection.linear_test_critical_value(1.0, df, True)
        for df in (2, 3, 5, 9, 29, 1000)
    }
    self.assertAlmostEqual(crit[29], 1.02, places=2)
    self.assertAlmostEqual(crit[9], 1.06, places=2)
    self.assertAlmostEqual(crit[3], 1.20, places=2)
    self.assertAlmostEqual(crit[1000], 1.0, places=2)
    self.assertTrue(all(crit[a] > crit[b] for a, b in zip(
        (2, 3, 5, 9, 29), (3, 5, 9, 29, 1000))))
    self.assertAlmostEqual(alpha_selection.linear_test_critical_value(
        0.0, 5, True), 0.0)

  def test_stage_b_uses_two_way_se_with_two_keys(self):
    """fold_key_values -> two-way se (both keys), one-way reported alongside."""
    df = helpers.saturating_panel(4)
    block = (df["date"].rank(method="dense").astype(int) - 1) // 10
    keys = (df["geo"].to_numpy(), block.astype(str).to_numpy())
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    builder = functools.partial(
        runner.build_cate_basis, covariates_for_cate=["X1"],
        cate_type="auto_additive", cate_df=3, cate_degree=2,
        cate_intercept=False,
    )
    results = {}
    for use_t in (False, True):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results[use_t] = alpha_selection.select_alpha(
            df_sel=df, y_col="sales", d_col="spend", x_cols=["X1", "X2"],
            basis_builder=builder, scaler=scaler, family="tanh",
            saturation_cfg={},
            selection_cfg={"n_seeds": 2, "k_folds": 3,
                           "linear_test_t_critical": use_t},
            cluster_labels=df["geo"].to_numpy(), ml_models_y=["rf"],
            ml_models_t=["rf"], time_budget=1, n_jobs=1,
            se_cluster_labels=df["geo"].to_numpy(), se_cluster_key="geo",
            item_col="item", geo_col="geo", date_col="date",
            fold_key_values=keys, fold_key_names=("geo", "block"),
        )
    r = results[False]
    self.assertEqual(r.se_method, "two-way")
    self.assertEqual(r.se_cluster_keys, ("geo", "block"))
    self.assertEqual(r.se, r.se_two_way)
    self.assertTrue(np.isfinite(r.se_two_way))
    self.assertTrue(np.isfinite(r.se_one_way))
    np.testing.assert_allclose(
        r.se_two_way,
        alpha_selection.cluster_robust_se(r.d_obs, keys[0], keys[1]),
    )
    np.testing.assert_allclose(
        r.se_one_way, alpha_selection.cluster_robust_se(r.d_obs, keys[0])
    )
    # df = min(G_geo, G_block) - 1 = min(10, 8) - 1.
    self.assertEqual(r.se_df, min(10, 8) - 1)
    self.assertEqual(r.critical_value, r.linear_test_k)
    self.assertFalse(r.linear_test_t_critical)
    self.assertIn(r.decision_under_one_way_se, ("saturation", "linear"))
    # Diag2's se(alpha) is the same two-way se.
    self.assertTrue(np.all(np.isfinite(r.se_alpha)))
    # The t critical value only raises the threshold (same data, same se).
    r_t = results[True]
    self.assertTrue(r_t.linear_test_t_critical)
    self.assertGreater(r_t.critical_value, r_t.linear_test_k)
    self.assertAlmostEqual(
        r_t.critical_value,
        alpha_selection.linear_test_critical_value(1.0, r_t.se_df, True),
    )
    self.assertAlmostEqual(r_t.se, r.se)
    if r.decision_reason == "1se_fail":
      self.assertEqual(r_t.decision_reason, "1se_fail")


class CateBasisIdentityTest(unittest.TestCase):
  """The same p(X) in Phase 0 and Phases 1-3."""

  def test_identical_design_matrix(self):
    """Phase 0 and Phases 1-3 build the same p(X)."""
    df = helpers.saturating_panel(0)
    df["flag"] = (df["X2"] > 0).astype(float)
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    df_norm = scaler.fit_transform(df)
    for cate_cols, structure in (
        (["X1"], {"type": "auto_additive", "df": 3, "degree": 2,
                  "include_intercept": False}),
        (["X1", "flag"], {"type": "auto_additive", "df": 4, "degree": 3,
                          "include_intercept": False}),
        (["X1", "X2"], "X1 + X2 + X1:X2"),
        ([], None),
    ):
      with self.subTest(cate=cate_cols, structure=structure):
        orch = workflow.CausalWorkflowOrchestrator(
            y_col="sales", d_cols=["spend"], x_cols_list=[["X1", "X2"]],
            ml_models_y=["rf"], ml_models_t=["rf"], n_folds_list=[2],
            output_dir="/tmp/unused_v4_basis",
            covariates_for_cate_list=[cate_cols],
            cate_structure_list=[structure] if structure else None,
        )
        phase0_p = orch._cate_basis_builder(  # pylint: disable=protected-access
            "spend", ["X1", "X2"], orch.covariates_for_cate_list[0],
            orch.cate_structure_list[0],
        )(df_norm, df)
        phase13 = runner.DoubleMLPipeline(
            "sales", "spend", ["X1", "X2"], "rf", "rf",
            cate_structure=orch.cate_structure_list[0],
            covariates_for_cate=orch.covariates_for_cate_list[0],
        )
        phase13_p = phase13._build_cate_basis(  # pylint: disable=protected-access
            df_norm, df
        )
        pd.testing.assert_frame_equal(phase0_p, phase13_p)

  def test_stage_b_uses_the_supplied_basis(self):
    """Stage B profiles on the supplied CATE basis."""
    df = helpers.saturating_panel(1)
    seen = {}
    base = functools.partial(
        runner.build_cate_basis, covariates_for_cate=["X1"],
        cate_type="auto_additive", cate_df=3, cate_degree=2,
        cate_intercept=False,
    )

    def spy(df_norm, df_orig):
      out = base(df_norm, df_orig)
      seen["p"] = out
      return out

    run_select(df, builder=spy)
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    expected = base(scaler.fit_transform(df), df)
    pd.testing.assert_frame_equal(seen["p"], expected)


class TreatmentDivisorTest(unittest.TestCase):
  """`treatment_divisor` must mirror `CausalDataScaler`'s own arithmetic."""

  def setUp(self):
    super().setUp()
    self.df = pd.DataFrame({
        "spend": [10.0, 20.0, 30.0, 40.0],
        "population": [100.0, 200.0, 400.0, 800.0],
        "sales": [1.0, 2.0, 3.0, 4.0],
    })

  def test_median_normalize_gives_the_global_median(self):
    """`median_normalize` divides every row by one global median."""
    scaler = scalers.CausalDataScaler(median_normalize_cols=["spend"])
    divisor, mode = alpha_selection.treatment_divisor(
        scaler, "spend", self.df
    )
    self.assertEqual(mode, "median_normalize")
    expected = float(self.df["spend"].median())
    np.testing.assert_allclose(divisor.to_numpy(), expected)
    # The whole point of median normalization: median(T/c) == 1.
    self.assertAlmostEqual(
        float((self.df["spend"] / divisor).median()), 1.0
    )

  def test_population_median_normalize_varies_by_row(self):
    """The divisor is population_i * median(T/population), one per row."""
    scaler = scalers.CausalDataScaler(
        population_median_normalize_cols=["spend"],
        population_col="population",
    )
    divisor, mode = alpha_selection.treatment_divisor(
        scaler, "spend", self.df
    )
    self.assertEqual(mode, "population_median_normalize")
    per_pop = self.df["spend"] / self.df["population"]
    median_per_pop = float(per_pop[per_pop > 0].median())
    np.testing.assert_allclose(
        divisor.to_numpy(), self.df["population"] * median_per_pop
    )
    self.assertAlmostEqual(
        float((self.df["spend"] / divisor).median()), 1.0
    )

  def test_divisor_reproduces_the_scaler_transform(self):
    """T / c must equal what the scaler itself would have produced."""
    for kwargs in (
        {"median_normalize_cols": ["spend"]},
        {
            "population_median_normalize_cols": ["spend"],
            "population_col": "population",
        },
    ):
      with self.subTest(**kwargs):
        scaler = scalers.CausalDataScaler(**kwargs)
        expected = scaler.fit_transform(self.df)["spend"].to_numpy()
        divisor, _ = alpha_selection.treatment_divisor(
            scaler, "spend", self.df
        )
        got = (self.df["spend"] / divisor).to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-12)

  def test_shifting_normalizations_are_rejected(self):
    """Scalers that shift the origin break `f(0) == 0`, so they must raise."""
    for kwargs in (
        {"standardize_cols": ["spend"]},
        {
            "population_standardize_cols": ["spend"],
            "population_col": "population",
        },
        {"min_max_cols": ["spend"]},
    ):
      with self.subTest(**kwargs):
        scaler = scalers.CausalDataScaler(**kwargs)
        with self.assertRaises(ValueError):
          alpha_selection.treatment_divisor(scaler, "spend", self.df)

  def test_unnormalized_treatment_gives_a_unit_divisor(self):
    scaler = scalers.CausalDataScaler(standardize_cols=["sales"])
    divisor, mode = alpha_selection.treatment_divisor(
        scaler, "spend", self.df
    )
    self.assertEqual(mode, "none")
    np.testing.assert_allclose(divisor.to_numpy(), 1.0)

  def test_caller_scaler_is_not_fitted_in_place(self):
    scaler = scalers.CausalDataScaler(median_normalize_cols=["spend"])
    alpha_selection.treatment_divisor(scaler, "spend", self.df)
    self.assertEqual(scaler.median_params_, {})


class InterceptCurveAndOutputsTest(unittest.TestCase):
  """Intercept curve, fit count, identity names and the output contract."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df = helpers.saturating_panel(2)
    cls.result = run_select(cls.df)

  def test_result_fields(self):
    """The result carries every summary field."""
    r = self.result
    n_grid = r.alpha_grid.size
    self.assertEqual(r.per_seed_loss.shape, (3, n_grid))
    self.assertEqual(r.per_seed_loss_ate.shape, (3, n_grid))
    self.assertEqual(r.per_seed_loss_identity.shape, (3,))
    self.assertIn(r.decision, ("saturation", "linear"))
    self.assertIn(r.selected_family, ("tanh", None))
    self.assertEqual(r.decision == "saturation", r.selected_family == "tanh")
    self.assertIn(r.refine_method, ("brent", "none"))
    self.assertGreaterEqual(r.n_basis, 6)
    self.assertLessEqual(r.n_basis, 20)
    # Diag1-Diag5 are always computed.
    self.assertIsInstance(r.diag1_fired, bool)
    self.assertIsInstance(r.diag2_fired, bool)
    self.assertTrue(np.isfinite(r.diag2_ratio))
    self.assertTrue(np.isfinite(r.r_t))
    self.assertEqual(r.fbar_quantiles.shape, (4,))
    self.assertIsNotNone(r.saturation_at_t_ref)
    np.testing.assert_allclose(r.d_s, r.per_seed_loss_at_alpha_hat
                               - r.per_seed_loss_identity)

  def test_identity_candidate_naming(self):
    """Identity-candidate fields carry the current names and definitions."""
    r = self.result
    self.assertEqual(r.per_seed_loss_identity.shape, (3,))
    self.assertAlmostEqual(r.mean_loss_identity,
                           float(np.mean(r.per_seed_loss_identity)))
    np.testing.assert_allclose(
        r.d_s, r.per_seed_loss_at_alpha_hat - r.per_seed_loss_identity
    )
    self.assertFalse(hasattr(r, "per_seed_loss_" + "linear"))
    self.assertFalse(hasattr(r, "fallback_" + "alpha_used"))

  def test_intercept_only_basis_equals_ate_curve(self):
    r = run_select(self.df, cate=())
    np.testing.assert_allclose(r.per_seed_loss, r.per_seed_loss_ate,
                               rtol=1e-10, atol=1e-12)

  def test_default_learners_and_fit_count(self):
    r = self.result
    self.assertFalse(r.learner_selection)
    self.assertEqual(r.n_ml_fits, 2 * 3 * 3)  # 2 S K; no selection fits
    self.assertTrue(r.learners["small_sample_switch"])
    self.assertEqual(r.learners["y_model"]["class"],
                     "sklearn.linear_model.RidgeCV")

  def test_no_per_alpha_retraining(self):
    """Grid size / extensions never multiply ML fits."""
    calls = {"n": 0}
    original = saturation.RidgeCVLearner.fit_predict

    def counting(self_, *args, **kwargs):
      calls["n"] += 1
      return original(self_, *args, **kwargs)

    with mock.patch.object(saturation.RidgeCVLearner, "fit_predict",
                           counting):
      r = run_select(helpers.saturating_panel(3, truth="linear"),
                     saturation_cfg={"grid": {"num": 40}})
    self.assertEqual(calls["n"], 2 * 3 * 3)
    self.assertEqual(r.n_ml_fits, calls["n"])
    self.assertGreaterEqual(r.alpha_grid.size, 40)

  def test_output_contract(self):
    """File names (diag/summary/others prefixes), columns, keys."""
    r = self.result
    with tempfile.TemporaryDirectory() as out_dir:
      written = saturation_evaluation.write_phase0_outputs(
          r, out_dir, treatment="spend", cluster_keys=["geo"],
          cluster_fallback_tier="tier 1", n_inf=1234,
          sel_cluster_ids=self.df["geo"].to_numpy(),
          inf_cluster_ids=np.array(["G98", "G99"]),
      )
      expected_files = {
          pipeline_io.SUMMARY_JSON,
          pipeline_io.ALPHA_GRID_CSV,
          pipeline_io.RLOSS_CURVES_CSV,
          pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV,
          pipeline_io.FOLD_ASSIGNMENTS_CSV,
          pipeline_io.DIAG3_IDENTIFICATION_CSV,
          pipeline_io.COVERAGE_CHECK_CSV,
          pipeline_io.DIAG1_BASIS_APPROX_ERROR_PNG,
          pipeline_io.DIAG2_RLOSS_PROFILE_PNG,
          pipeline_io.DIAG4_F_TRANSFORM_QUANTILES_PNG,
          # tanh is a bounded family, so Diag5 has a figure.
          pipeline_io.DIAG5_SATURATION_AT_T_REF_PNG,
          pipeline_io.SPLIT_COVERAGE_PNG,
      }
      self.assertEqual(set(os.listdir(out_dir)), expected_files)
      self.assertTrue(all(written[f] for f in expected_files))
      # The grid construction and the seed-wise alpha distribution are panels
      # of the alpha profile chart, not stand-alone files any more.
      self.assertNotIn("others_grid_construction.png", written)
      self.assertNotIn("others_alpha_seed_distribution.png", written)
      # The renamed files state what they belong to.
      self.assertEqual(
          pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV,
          "summary_alpha_seed-wise_alpha_distribution.csv",
      )
      self.assertEqual(
          pipeline_io.COVERAGE_CHECK_CSV, "others_split_coverage.csv"
      )
      self.assertEqual(
          pipeline_io.FOLD_ASSIGNMENTS_CSV, "others_split_coverage_detail.csv"
      )
      # Every file name states what it belongs to.
      for name in expected_files:
        self.assertTrue(
            name.startswith(("summary_", "others_"))
            or name[:5] in {f"diag{i}" for i in range(1, 7)},
            name,
        )

      def strict(name):
        raise ValueError(f"non-JSON constant {name}")

      with open(os.path.join(out_dir, pipeline_io.SUMMARY_JSON)) as fh:
        summary = json.load(fh, parse_constant=strict)
      for key in saturation_evaluation.SUMMARY_KEYS:
        self.assertIn(key, summary)
      self.assertEqual(list(summary)[: len(saturation_evaluation.SUMMARY_KEYS)],
                       list(saturation_evaluation.SUMMARY_KEYS))
      self.assertEqual(summary["n_inf"], 1234)
      self.assertEqual(summary["J"], r.n_basis)
      self.assertIn(summary["decision"], ("saturation", "linear"))
      self.assertIn("y_model", summary["learners"])
      # The decision keys carry the documented enums and the
      # normalization reference point (strictly positive).
      self.assertIn(summary["decision_reason"],
                    ("1se_pass", "1se_fail", "grid_edge_upper",
                     "basis_ill_conditioned"))
      self.assertIn(summary["grid_edge_hit"], ("none", "upper", "lower"))
      self.assertGreater(summary["T_ref"], 0.0)
      self.assertGreater(summary["T_max"], 0.0)
      self.assertAlmostEqual(summary["T_ref"], r.t_ref)
      # T_max is the basis-domain right end (q99.9 of T > 0), reported with
      # its quantile, the observed D_sel maximum and the clipped-row count;
      # cond(B) of the adopted basis is reported with its flag.
      self.assertAlmostEqual(summary["T_max"], r.t_max)
      self.assertEqual(summary["T_max_quantile"], 0.999)
      self.assertGreaterEqual(summary["T_max_observed_sel"], summary["T_max"])
      self.assertIsInstance(summary["n_rows_above_T_max"], int)
      self.assertGreater(summary["basis_cond"], 1.0)
      self.assertLess(summary["basis_cond"], 1e3)
      self.assertIs(summary["basis_ill_conditioned"], False)
      # The se of the 1-SE decision: this direct call passes one key only,
      # so the se is one-way; the two-way value is null and the critical
      # value is k itself (t critical value off by default).
      self.assertEqual(summary["se_method"], "one-way")
      self.assertEqual(summary["se_cluster_key"], "geo")
      self.assertIsNone(summary["se_cluster_keys"])
      self.assertAlmostEqual(summary["se_one_way"], summary["se"])
      self.assertIsNone(summary["se_two_way"])
      self.assertEqual(summary["se_df"], 10 - 1)
      self.assertIs(summary["linear_test_t_critical"], False)
      self.assertEqual(summary["critical_value"], summary["linear_test_k"])
      self.assertIn(summary["decision_under_one_way_se"],
                    ("saturation", "linear"))
      # Identity-candidate keys, the grid mode and no abolished name anywhere
      # in the JSON text.
      for key in (
          "alpha_grid_argmin", "rloss_min", "se", "diag1_fired",
          "diag2_fired", "diag2_ratio", "fbar_q25", "fbar_q50", "fbar_q75",
          "fbar_q90", "linear_ref_q25", "linear_ref_q50", "linear_ref_q75",
          "linear_ref_q90", "saturation_at_T_ref", "grid_mode",
      ):
        self.assertIn(key, summary)
      self.assertEqual(summary["alpha_grid_argmin"], r.grid_alpha_hat)
      self.assertAlmostEqual(summary["rloss_min"], r.rloss_min)
      self.assertEqual(summary["grid_mode"], "auto")
      with open(os.path.join(out_dir, pipeline_io.SUMMARY_JSON)) as fh:
        summary_text = fh.read()
      for token in _FORBIDDEN:
        self.assertNotIn(token, summary_text, token)

      columns = {
          # phi_p(alpha) per grid point and A_1SE, plus
          # the profiled R-loss and se(alpha) (the former alpha_profile csv).
          pipeline_io.ALPHA_GRID_CSV: list(
              saturation_evaluation.ALPHA_GRID_COLUMNS
          ),
          # Identity candidate columns.
          pipeline_io.RLOSS_CURVES_CSV: [
              "seed", "grid_index", "alpha", "rloss_cate", "rloss_ate",
              "rloss_identity",
          ],
          pipeline_io.ALPHA_SEED_DISTRIBUTION_CSV: [
              "seed", "alpha_hat_s", "rloss_at_alpha_hat_s",
              "rloss_identity_s", "d_s",
          ],
          # Cell id plus its two key values.
          pipeline_io.FOLD_ASSIGNMENTS_CSV: [
              "seed", "cluster_id", "key1_value", "key2_value", "fold",
              "split",
          ],
          pipeline_io.DIAG3_IDENTIFICATION_CSV: [
              "r_T", "sd_T", "sd_T_resid", "share_T_zero", "q50_T_pos",
              "q95_T_pos", "T_max", "T_max_observed", "T_ref",
          ],
          pipeline_io.COVERAGE_CHECK_CSV: list(
              saturation_evaluation.COVERAGE_COLUMNS
          ),
      }
      for name, cols in columns.items():
        with self.subTest(file=name):
          frame = pd.read_csv(os.path.join(out_dir, name))
          self.assertEqual(list(frame.columns), cols)
          for token in _FORBIDDEN:
            self.assertNotIn(token, ",".join(frame.columns), f"{name}: {token}")
      grid = pd.read_csv(os.path.join(out_dir, pipeline_io.ALPHA_GRID_CSV))
      self.assertEqual(
          [
              "grid_index", "half_saturation_T", "alpha", "mean_r_loss",
              "se_alpha", "in_1se_set", "fbar_q25", "fbar_q50", "fbar_q75",
              "fbar_q90", "basis_approx_error_alpha",
          ],
          list(grid.columns),
      )
      np.testing.assert_allclose(grid["mean_r_loss"], r.mean_loss)
      np.testing.assert_allclose(grid["se_alpha"], r.se_alpha)
      self.assertEqual(int(grid["in_1se_set"].sum()),
                       int(np.count_nonzero(r.one_se_set)))
      curves = pd.read_csv(
          os.path.join(out_dir, pipeline_io.RLOSS_CURVES_CSV)
      )
      self.assertEqual(len(curves), 3 * r.alpha_grid.size)
      self.assertEqual(
          curves.groupby("seed")["rloss_identity"].nunique().max(), 1
      )
      folds = pd.read_csv(
          os.path.join(out_dir, pipeline_io.FOLD_ASSIGNMENTS_CSV)
      )
      self.assertEqual(set(folds["split"]), {"sel", "inf"})
      self.assertTrue((folds.loc[folds["split"] == "inf", "fold"] == -1).all())

  def test_diag6_record(self):
    """Diag6: the Phase 1 check leaves a CSV record per treatment."""
    with tempfile.TemporaryDirectory() as out_dir:
      self.assertIsNone(
          saturation_evaluation.write_diag6_output("/nonexistent/dir", [])
      )
      path = saturation_evaluation.write_diag6_output(out_dir, [{
          "treatment": "spend", "status": "pass", "T_ref_phase0": 2.0,
          "T_ref_phase1": 2.0, "rel_error": 0.0, "t_ref_tolerance": 1e-9,
          "t_ref_quantile": 0.95, "n_rows_phase1": 100, "note": "ok",
      }])
      self.assertEqual(
          os.path.basename(path), pipeline_io.DIAG6_T_REF_CONSISTENCY_CSV
      )
      frame = pd.read_csv(path)
      self.assertEqual(list(frame.columns),
                       list(saturation_evaluation.DIAG6_COLUMNS))
      self.assertEqual(frame.loc[0, "status"], "pass")


class BasisConditioningTest(unittest.TestCase):
  """Diag1 conditioning trigger: linear fallback, statistics still reported."""

  def test_ill_conditioned_basis_falls_back_to_linear(self):
    """cond(B) >= MAX_BASIS_CONDITION adopts linear; statistics reported."""
    df = helpers.saturating_panel(5)
    # The augmented projection grid never trips the real threshold, so lower
    # it below the healthy cond(B) of order 10 to exercise the path.
    with mock.patch.object(saturation, "MAX_BASIS_CONDITION", 1.0):
      r = run_select(df)
    self.assertTrue(r.basis_ill_conditioned)
    self.assertTrue(r.diag1_fired)
    self.assertEqual(r.decision, "linear")
    self.assertEqual(r.decision_reason, "basis_ill_conditioned")
    self.assertIsNone(r.selected_family)
    self.assertFalse(r.fallback_half_saturation_used)
    # The statistics are still computed and reported.
    self.assertTrue(np.isfinite(r.d_bar))
    self.assertTrue(np.isfinite(r.se))
    self.assertTrue(np.all(np.isfinite(r.mean_loss)))
    self.assertTrue(any("ill-conditioned" in m for m in r.messages))
    self.assertTrue(
        any("basis_ill_conditioned" in m for m in r.messages)
    )

  def test_fixed_alpha_override_survives_ill_conditioning(self):
    """A configured fallback_half_saturation_T keeps the family (as for eps)."""
    df = helpers.saturating_panel(6)
    with mock.patch.object(saturation, "MAX_BASIS_CONDITION", 1.0):
      r = run_select(df, selection_cfg={"fallback_half_saturation_T": 2.5})
    self.assertTrue(r.basis_ill_conditioned)
    self.assertEqual(r.decision_reason, "basis_ill_conditioned")
    self.assertTrue(r.fallback_half_saturation_used)
    self.assertEqual(r.selected_family, "tanh")
    self.assertAlmostEqual(r.half_saturation_t, 2.5)

  def test_default_threshold_does_not_trip(self):
    r = run_select(helpers.saturating_panel(7))
    self.assertFalse(r.basis_ill_conditioned)
    self.assertNotEqual(r.decision_reason, "basis_ill_conditioned")
    self.assertLess(r.basis_cond, 1e3)
    self.assertEqual(r.t_max_quantile, saturation.BASIS_T_MAX_QUANTILE)
    self.assertGreaterEqual(r.t_max_observed, r.t_max)

  def test_caller_supplied_t_max_is_used(self):
    """The orchestrator's full-data T_max is the basis domain right end."""
    df = helpers.saturating_panel(8)
    t = df["spend"].to_numpy()
    t_max = float(np.quantile(t[t > 0], 0.9))
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    builder = functools.partial(
        runner.build_cate_basis, covariates_for_cate=["X1"],
        cate_type="auto_additive", cate_df=3, cate_degree=2,
        cate_intercept=False,
    )
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      r = alpha_selection.select_alpha(
          df_sel=df, y_col="sales", d_col="spend", x_cols=["X1", "X2"],
          basis_builder=builder, scaler=scaler, family="tanh",
          saturation_cfg={}, selection_cfg={"n_seeds": 2, "k_folds": 3},
          cluster_labels=df["geo"].to_numpy(), ml_models_y=["rf"],
          ml_models_t=["rf"], time_budget=1, n_jobs=1,
          se_cluster_labels=df["geo"].to_numpy(), se_cluster_key="geo",
          item_col="item", geo_col="geo", date_col="date", t_max=t_max,
      )
    self.assertEqual(r.t_max, t_max)
    self.assertEqual(r.n_above_t_max, int(np.sum(t > t_max)))
    self.assertGreater(r.n_above_t_max, 0)
    self.assertFalse(any("T_max was not supplied" in m for m in r.messages))


def _naive_cross_fit_losses(l_pred, mu_pred, y, f_vals, coefs, p, fold_ids,
                            ridge):
  """Reference profiled R-loss with an explicit per-fold loop.

  ``beta^(-k)`` is the ridge least-squares fit on the out-of-fold residuals
  of the rows NOT in fold ``k``; fold ``k`` is evaluated with it.

  Args:
    l_pred: ``(n,)`` out-of-fold ``l_hat``.
    mu_pred: ``(n, J)`` out-of-fold ``mu_hat``.
    y: ``(n,)`` outcome.
    f_vals: ``(n, G)`` ``f(T_i; alpha_g)``.
    coefs: ``(J, G)`` basis coefficients.
    p: ``(n, P)`` CATE basis.
    fold_ids: ``(n,)`` fold of every row.
    ridge: Ridge term of the inner solve.

  Returns:
    ``(losses (G,), obs_sq_resid (n, G))``.
  """
  n_obs, n_grid = f_vals.shape
  s_res = f_vals - mu_pred @ coefs
  y_tilde = y - l_pred
  obs = np.zeros((n_obs, n_grid))
  for g in range(n_grid):
    for k in np.unique(fold_ids):
      train = fold_ids != k
      test = ~train
      z = s_res[train, g][:, None] * p[train]
      beta = np.linalg.solve(
          z.T @ z + ridge * np.eye(p.shape[1]), z.T @ y_tilde[train]
      )
      fitted = s_res[test, g] * (p[test] @ beta)
      obs[test, g] = (y_tilde[test] - fitted) ** 2
  return obs.mean(axis=0), obs


class OutOfFoldResidualsTest(unittest.TestCase):
  """Cross-fitting: out-of-fold nuisances everywhere, honest inner beta fit."""

  def test_profile_losses_matches_naive_cross_fit(self):
    """The batched algebra equals the explicit per-fold reference."""
    rng = np.random.default_rng(11)
    n_obs, n_basis, n_grid, k_folds = 90, 4, 3, 3
    l_pred = rng.normal(size=n_obs)
    mu_pred = rng.normal(size=(n_obs, n_basis))
    y = rng.normal(size=n_obs)
    f_vals = rng.normal(size=(n_obs, n_grid))
    coefs = rng.normal(size=(n_basis, n_grid))
    p = np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
    fold_ids = rng.integers(0, k_folds, size=n_obs)
    fold_ids[:k_folds] = np.arange(k_folds)  # every fold non-empty
    for ridge in (0.0, 0.3):
      with self.subTest(ridge=ridge):
        loss, obs = alpha_selection._profile_losses(  # pylint: disable=protected-access
            l_pred, mu_pred, y, f_vals, coefs, p, fold_ids, ridge,
            return_obs=True,
        )
        want_loss, want_obs = _naive_cross_fit_losses(
            l_pred, mu_pred, y, f_vals, coefs, p, fold_ids, ridge
        )
        np.testing.assert_allclose(loss, want_loss, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(obs, want_obs, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(obs.mean(axis=0), loss, rtol=1e-12)
        only_loss = alpha_selection._profile_losses(  # pylint: disable=protected-access
            l_pred, mu_pred, y, f_vals, coefs, p, fold_ids, ridge
        )
        np.testing.assert_allclose(only_loss, loss, rtol=1e-12)

  def test_cross_fitted_loss_is_not_an_in_sample_fit(self):
    """The reported loss is honest: it exceeds the pooled in-sample minimum.

    A beta fitted on all rows and evaluated on the same rows (in-sample)
    attains the smallest possible average squared residual for a single
    coefficient vector; the cross-fitted loss, where fold k is evaluated with
    the beta of the other folds, is larger on this fixed-seed data.
    """
    rng = np.random.default_rng(12)
    n_obs, n_basis, n_grid, k_folds = 60, 3, 2, 3
    l_pred = rng.normal(size=n_obs)
    mu_pred = rng.normal(size=(n_obs, n_basis))
    y = rng.normal(size=n_obs)
    f_vals = rng.normal(size=(n_obs, n_grid))
    coefs = rng.normal(size=(n_basis, n_grid))
    p = np.ones((n_obs, 1))
    fold_ids = np.arange(n_obs) % k_folds
    loss = alpha_selection._profile_losses(  # pylint: disable=protected-access
        l_pred, mu_pred, y, f_vals, coefs, p, fold_ids, 0.0
    )
    # A pooled (non-cross-fitted) beta on all rows gives a smaller in-sample
    # loss than the honest cross-fitted evaluation.
    s_res = f_vals - mu_pred @ coefs
    y_tilde = y - l_pred
    pooled = np.array([
        np.mean((y_tilde - s_res[:, g] * (
            (s_res[:, g] @ y_tilde) / (s_res[:, g] @ s_res[:, g])
        )) ** 2)
        for g in range(n_grid)
    ])
    self.assertTrue(np.all(pooled < loss))

  def test_each_learner_predicts_only_its_held_out_fold(self):
    """Every (seed, fold) model predicts n / K rows, never the training rows."""
    df = helpers.saturating_panel(13)
    n_obs = len(df)
    seen = []
    original = saturation.RidgeCVLearner.fit_predict

    def recording(self_, x_train, y_train, groups_train, x_all, seed):
      seen.append((len(x_train), len(x_all)))
      return original(self_, x_train, y_train, groups_train, x_all, seed)

    with mock.patch.object(saturation.RidgeCVLearner, "fit_predict",
                           recording):
      r = run_select(df, selection_cfg={"n_seeds": 2, "k_folds": 3})
    self.assertEqual(len(seen), 2 * 2 * 3)
    for n_train, n_predict in seen:
      self.assertEqual(n_train + n_predict, n_obs)
      self.assertLess(n_predict, n_obs)
    # Per seed and nuisance the held-out folds partition the rows.
    self.assertEqual(sum(n_predict for _, n_predict in seen), 2 * 2 * n_obs)
    self.assertEqual(r.n_ml_fits, len(seen))
    self.assertTrue(np.all(np.isfinite(r.mean_loss)))

  def test_cross_fit_jobs_skip_empty_folds(self):
    """Folds without held-out rows need no model (fewer clusters than K)."""
    fold_ids = np.array([0, 0, 1, 1, 1])
    jobs = alpha_selection._cross_fit_jobs([fold_ids], k_folds=3)  # pylint: disable=protected-access
    self.assertEqual([(s, k, kind) for s, k, kind, _, _ in jobs],
                     [(0, 0, "y"), (0, 0, "t"), (0, 1, "y"), (0, 1, "t")])
    for _, k, _, train, held_out in jobs:
      np.testing.assert_array_equal(held_out, fold_ids == k)
      np.testing.assert_array_equal(train, ~held_out)
    l_oof = np.zeros((1, 5))
    mu_oof = np.zeros((1, 5, 2))
    fit_out = [np.array([1.0, 2.0]), np.ones((2, 2)),
               np.array([3.0, 4.0, 5.0]), 2 * np.ones((3, 2))]
    alpha_selection._scatter_oof_predictions(jobs, fit_out, l_oof, mu_oof)  # pylint: disable=protected-access
    np.testing.assert_array_equal(l_oof[0], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(mu_oof[0, :, 0], [1.0, 1.0, 2.0, 2.0, 2.0])


class LearnerResolutionTest(unittest.TestCase):
  """Default Stage B learners and the small-sample switch."""

  def test_defaults(self):
    cfg = saturation.resolve_learner_config(None)
    self.assertFalse(cfg["learner_selection"])
    self.assertEqual(cfg["small_sample_threshold"], 2000)
    self.assertEqual(cfg["t_model"]["params"]["multi_strategy"],
                     "multi_output_tree")
    self.assertEqual(cfg["y_model"]["params"]["max_depth"], 5)
    self.assertEqual(cfg["y_model"]["early_stopping_rounds"], 30)

  def test_small_sample_switch(self):
    cfg = saturation.resolve_learner_config(None)
    small = saturation.resolve_stage_b_learners(cfg, 1999, [0], 3)
    self.assertTrue(small.small_sample)
    self.assertEqual(small.t_learner.kind, "ridge_cv")
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      big = saturation.resolve_stage_b_learners(cfg, 2000)
    self.assertFalse(big.small_sample)
    self.assertIn(big.y_learner.kind, ("xgboost", "sklearn_hgb"))
    self.assertIn(big.t_learner.kind, ("xgboost", "sklearn_rf"))

  def test_t_model_is_one_multi_output_model(self):
    """The T-model is ONE model with J outputs."""
    cfg = saturation.resolve_learner_config(None)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      learners = saturation.resolve_stage_b_learners(cfg, 5000)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = np.column_stack([x[:, 0], x[:, 1] ** 2, np.sin(x[:, 2]), x[:, 0]])
    preds = learners.t_learner.fit_predict(x, y, None, x, seed=0)
    self.assertEqual(preds.shape, (300, 4))
    y_pred = learners.y_learner.fit_predict(
        x, y[:, 0], np.repeat(np.arange(30), 10), x, seed=0
    )
    self.assertEqual(y_pred.shape, (300,))
    self.assertGreater(np.corrcoef(y_pred, y[:, 0])[0, 1], 0.8)


class ComputeAccountingTest(unittest.TestCase):
  """Fit count 2 S K (+ N_y + N_t with learner selection); no J, no |G|."""

  def test_formula(self):
    self.assertEqual(
        alpha_selection.count_stage_b_tasks(
            n_models_y=2, n_models_t=2, n_seeds=10, k_folds=5
        ),
        2 + 2 + 2 * 10 * 5,
    )

  def test_without_learner_selection(self):
    self.assertEqual(
        alpha_selection.count_stage_b_tasks(
            n_models_y=3, n_models_t=2, n_seeds=10, k_folds=5,
            learner_selection=False,
        ),
        100,
    )

  def test_single_seed_and_folds(self):
    self.assertEqual(
        alpha_selection.count_stage_b_tasks(
            n_models_y=1, n_models_t=1, n_seeds=1, k_folds=5
        ),
        1 + 1 + 10,
    )

  def test_defaults_are_far_below_the_pre_multioutput_cost(self):
    """Regression guard: the old formula gave 832 for these defaults."""
    self.assertEqual(
        alpha_selection.count_stage_b_tasks(
            n_models_y=2, n_models_t=2, n_seeds=10, k_folds=5
        ),
        104,
    )


class TimeBudgetScalingTest(unittest.TestCase):
  """Phase 0 sees only D_sel, so its FLAML budget shrinks with it."""

  def test_scales_by_the_selection_fraction(self):
    self.assertEqual(
        alpha_selection.scale_time_budget(100, n_sel=20, n_total=100), 20
    )

  def test_floored_so_flaml_can_still_run_a_trial(self):
    self.assertEqual(
        alpha_selection.scale_time_budget(10, n_sel=1, n_total=1000),
        alpha_selection.MIN_PHASE0_TIME_BUDGET,
    )

  def test_never_exceeds_the_original_budget(self):
    self.assertEqual(
        alpha_selection.scale_time_budget(30, n_sel=500, n_total=100), 30
    )

  def test_degenerate_totals_fall_back_to_the_original_budget(self):
    self.assertEqual(
        alpha_selection.scale_time_budget(42, n_sel=0, n_total=0), 42
    )


class StandardErrorTest(unittest.TestCase):
  """se = cluster-robust se_obs, nothing else."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df = helpers.saturating_panel(0)
    cls.result, cls.messages = run_select_with_messages(cls.df)

  def test_se_equals_the_cluster_robust_formula(self):
    """se matches the hand-computed cluster-robust (CR1) formula."""
    r = self.result
    labels = self.df["geo"].to_numpy()
    n = len(self.df)
    d_i = np.asarray(r.d_obs, dtype=float)
    self.assertEqual(d_i.shape, (n,))
    # d_bar = mean_s d_s = mean_i d_i (both average the same residuals).
    self.assertAlmostEqual(float(d_i.mean()), r.d_bar, places=10)
    codes, uniques = pd.factorize(pd.Series(labels))
    n_groups = uniques.size
    d_bar_obs = d_i.mean()
    sums = np.array([
        (d_i[codes == g] - d_bar_obs).sum() for g in range(n_groups)
    ])
    want = np.sqrt(n_groups / (n_groups - 1) * np.sum(sums**2)) / n
    self.assertAlmostEqual(r.se, want, places=12)
    self.assertAlmostEqual(
        r.se, alpha_selection.cluster_robust_se(d_i, labels), places=14
    )

  def test_single_cluster_robust_se_only(self):
    """No alternative SE is exposed; the decision uses d_bar and se only."""
    r = self.result
    for name in ("se_" + "seed", "se_" + "type", "se_" + "obs"):
      self.assertFalse(hasattr(r, name), name)
    self.assertFalse(hasattr(alpha_selection, "combine_" + "se"))
    # The decision uses d_bar and se only.
    self.assertEqual(
        r.decision == "saturation" and r.decision_reason == "1se_pass",
        r.decision_reason != "grid_edge_upper"
        and alpha_selection.linear_test_decision(r.d_bar, r.se,
                                                 r.linear_test_k),
    )

  def test_se_alpha_is_paired_against_alpha_hat(self):
    """Diag2's se(alpha) is the same formula applied to d_i(alpha)."""
    r = self.result
    self.assertEqual(r.se_alpha.shape, r.alpha_grid.shape)
    self.assertTrue(np.all(np.isfinite(r.se_alpha)))
    self.assertTrue(np.all(r.se_alpha >= 0.0))
    # A_1SE = {alpha : L_bar(alpha) - L_bar(alpha_hat) <= se(alpha)}.
    gap = r.mean_loss - r.rloss_min
    want = gap <= r.se_alpha
    np.testing.assert_array_equal(np.asarray(r.one_se_set, bool), want)
    self.assertTrue(np.all(gap >= -1e-12))


class DiagnosticsTest(unittest.TestCase):
  """Diag1-Diag5 definitions."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.df = helpers.saturating_panel(2)
    cls.result, cls.messages = run_select_with_messages(cls.df)

  def test_diag1_definition(self):
    """Diag1 (eps(J) >= basis_tol) is a warn-and-continue diagnostic."""
    r = self.result
    self.assertAlmostEqual(
        r.basis_approx_error, float(np.max(r.basis_approx_error_alpha))
    )
    self.assertEqual(r.diag1_fired, bool(r.basis_approx_error >= r.basis_tol))
    self.assertEqual(r.basis_tol, 1e-3)
    # Forcing the tolerance to 0 makes Diag1 fire with a warning while the
    # run completes (warn and continue).
    fired, messages = run_select_with_messages(
        self.df, saturation_cfg={"basis_dim_T": 8, "basis_tol": 1e-12}
    )
    self.assertTrue(fired.diag1_fired)
    self.assertEqual(fired.n_basis, 8)
    self.assertTrue(any("Diag1 WARNING" in m for m in messages), messages)
    self.assertIn(fired.decision, ("saturation", "linear"))

  def test_diag2_definition_and_helper(self):
    """Diag2 ratio is max/min T_half over A_1SE; the helper matches per family.

    ``diag2_flatness`` is checked on synthetic inputs: tanh and log compare the
    knee ratio against ``flatness_ratio``, power compares the alpha range
    against ``POWER_FLATNESS_RANGE``, and an empty 1-SE set gives NaN (not
    fired).
    """
    r = self.result
    inside = np.asarray(r.one_se_set, bool)
    t_half = saturation.half_saturation_point("tanh", r.alpha_grid[inside])
    want_ratio = float(t_half.max() / t_half.min())
    self.assertAlmostEqual(r.diag2_ratio, want_ratio, places=12)
    self.assertEqual(r.diag2_fired, bool(want_ratio > r.flatness_ratio))
    self.assertEqual(r.flatness_ratio, 10.0)
    # The helper on synthetic inputs.
    alphas = np.array([np.log(3.0) / th for th in (1.0, 2.0, 4.0, 40.0)])
    ratio, fired, thr = alpha_selection.diag2_flatness(
        "tanh", alphas, np.array([True, True, False, True]), 10.0
    )
    self.assertAlmostEqual(ratio, 40.0)
    self.assertTrue(fired)
    self.assertEqual(thr, 10.0)
    ratio, fired, _ = alpha_selection.diag2_flatness(
        "tanh", alphas, np.array([True, True, True, False]), 10.0
    )
    self.assertAlmostEqual(ratio, 4.0)
    self.assertFalse(fired)
    # log: T_knee = alpha, same ratio rule.
    ratio, fired, _ = alpha_selection.diag2_flatness(
        "log", np.array([0.5, 10.0]), np.array([True, True]), 10.0
    )
    self.assertAlmostEqual(ratio, 20.0)
    self.assertTrue(fired)
    # power: alpha range > 0.4.
    ratio, fired, thr = alpha_selection.diag2_flatness(
        "power", np.array([0.2, 0.5, 0.7]), np.array([True, False, True]), 10.0
    )
    self.assertAlmostEqual(ratio, 0.5)
    self.assertTrue(fired)
    self.assertEqual(thr, alpha_selection.POWER_FLATNESS_RANGE)
    ratio, fired, _ = alpha_selection.diag2_flatness(
        "power", np.array([0.2, 0.5, 0.7]), np.array([True, True, False]), 10.0
    )
    self.assertAlmostEqual(ratio, 0.3)
    self.assertFalse(fired)
    # Empty set: NaN, not fired.
    ratio, fired, _ = alpha_selection.diag2_flatness(
        "tanh", alphas, np.zeros(4, bool), 10.0
    )
    self.assertTrue(np.isnan(ratio))
    self.assertFalse(fired)

  def test_diag2_warning_only_when_family_adopted(self):
    """Diag2 is a WARNING only when a nonlinear family is adopted.

    Diag2 fires here by construction (ratio threshold 1); the message is a
    WARNING iff selected_family is not None, and INFO otherwise.
    """
    adopted, messages = run_select_with_messages(
        self.df, selection_cfg={"flatness_ratio": 1.0 + 1e-9,
                                "linear_test_k": -1e12},
    )
    if adopted.diag2_fired and adopted.selected_family is not None:
      self.assertTrue(
          any("Diag2 1-SE flatness" in m and m.endswith("Continuing.")
              for m in messages), messages)
    linear, warned = run_select_with_messages(
        self.df, selection_cfg={"flatness_ratio": 1.0 + 1e-9,
                                "linear_test_k": 1e12},
    )
    self.assertIsNone(linear.selected_family)
    if linear.diag2_fired:
      # INFO: recorded in the result messages, NOT emitted as a warning.
      info = [m for m in linear.messages if "Diag2 1-SE flatness" in m]
      self.assertTrue(info, linear.messages)
      self.assertTrue(all("INFO" in m for m in info), info)
      self.assertFalse(any("Diag2 1-SE flatness" in m for m in warned), warned)

  def test_diag3_diag4_diag5_values(self):
    """Diag3 r_T, Diag4 quantiles phi_p and Diag5 kappa match the formulas."""
    r = self.result
    t = self.df["spend"].to_numpy(dtype=float)
    t_pos = t[t > 0]
    # Diag3: r_T = sd(T~) / sd(T / T_ref) (scale free; sd_t on T / c).
    self.assertTrue(np.isfinite(r.r_t))
    self.assertAlmostEqual(r.r_t, r.sd_t_resid / r.sd_t, places=10)
    # Diag4: phi_p(alpha_hat) = fbar(q_p(T>0); alpha_hat), refs q_p / T_ref.
    for p in (25, 50, 75, 90):
      q_p = float(np.quantile(t_pos, p / 100.0))
      want = float(saturation.normalized_saturation(
          np.array([q_p]), "tanh", r.alpha_hat, r.t_ref)[0])
      self.assertAlmostEqual(r.fbar_quantile(p), want, places=10)
      self.assertAlmostEqual(r.linear_ref_quantile(p), q_p / r.t_ref,
                             places=10)
    self.assertEqual(r.fbar_quantiles_grid.shape, (4, r.alpha_grid.size))
    # Monotone in p for a fixed alpha.
    self.assertTrue(np.all(np.diff(r.fbar_quantiles_grid, axis=0) >= -1e-12))
    # Diag5: kappa(alpha_hat) = f(T_ref; alpha_hat) = tanh(alpha T_ref / 2).
    self.assertAlmostEqual(
        r.saturation_at_t_ref, float(np.tanh(0.5 * r.alpha_hat * r.t_ref)),
        places=12,
    )
    self.assertGreater(r.saturation_at_t_ref, 0.0)
    self.assertLess(r.saturation_at_t_ref, 1.0)

  def test_diag5_is_null_for_unbounded_families(self):
    """Diag5 is None for an unbounded family (log); Diag2 still computes."""
    r, _ = run_select_with_messages(self.df, family="log")
    self.assertIsNone(r.saturation_at_t_ref)
    self.assertTrue(np.isfinite(r.diag2_ratio))

  def test_intercept_curve_is_visualization_only(self):
    """The ATE loss curve is kept for plots only and drives no decision."""
    r = self.result
    self.assertEqual(r.mean_loss_ate.shape, r.mean_loss.shape)
    self.assertFalse(hasattr(r, "alpha_hat_" + "ate"))
    self.assertFalse(hasattr(r, "crit" + "erion3_fired"))

  def test_diag6_helper(self):
    """t_ref_consistency holds for T itself and fails for a rescaled T."""
    t = self.df["spend"].to_numpy(dtype=float)
    t_ref = saturation.reference_point(t)
    holds, t1, err = alpha_selection.t_ref_consistency(t_ref, t)
    self.assertTrue(holds)
    self.assertAlmostEqual(t1, t_ref)
    self.assertLess(err, 1e-9)
    holds, t1, err = alpha_selection.t_ref_consistency(t_ref, 100.0 * t)
    self.assertFalse(holds)
    self.assertAlmostEqual(t1, 100.0 * t_ref, places=8)
    self.assertAlmostEqual(err, 99.0, places=6)


class FallbackHalfSaturationTest(unittest.TestCase):
  """The fallback_half_saturation_T override."""

  def test_used_only_when_diag1_or_diag2_fires(self):
    """The fixed half-saturation point is used only when Diag1/Diag2 fire."""
    df = helpers.saturating_panel(4)
    # Diag1 forced (basis_tol = 0) -> the fixed half-saturation point is used
    # downstream and the family is kept.
    r, messages = run_select_with_messages(
        df,
        selection_cfg={"fallback_half_saturation_T": 3.0},
        saturation_cfg={"basis_dim_T": 8, "basis_tol": 1e-12},
    )
    self.assertTrue(r.diag1_fired)
    self.assertTrue(r.fallback_half_saturation_used)
    self.assertEqual(r.selected_family, "tanh")
    self.assertAlmostEqual(r.alpha_hat, np.log(3.0) / 3.0)
    self.assertAlmostEqual(r.half_saturation_t, 3.0)
    self.assertNotEqual(r.alpha_hat, r.alpha_hat_estimated)
    self.assertTrue(any("Diag1 WARNING" in m for m in messages), messages)
    self.assertTrue(
        any("fallback_half_saturation_T" in m for m in r.messages), r.messages
    )
    # Nothing fires -> the 1-SE decision stands and the estimate is kept.
    # (basis_tol is made generous so that Diag1 cannot fire on this small
    # frame, and flatness_ratio is made huge so that Diag2 cannot fire.)
    r2, _ = run_select_with_messages(
        df,
        selection_cfg={"fallback_half_saturation_T": 3.0,
                       "flatness_ratio": 1e12},
        saturation_cfg={"basis_dim_T": 8, "basis_tol": 1.0},
    )
    self.assertFalse(r2.diag1_fired)
    self.assertFalse(r2.diag2_fired)
    self.assertFalse(r2.fallback_half_saturation_used)
    self.assertEqual(r2.alpha_hat, r2.alpha_hat_estimated)

  def test_units_per_family(self):
    """fallback_half_saturation_T is read in each family's own grid unit.

    exp: ``alpha = ln 2 / T_1/2``; log: ``alpha = T_knee``; power: the value
    is ``alpha`` itself.
    """
    df = helpers.saturating_panel(5)
    r_exp, _ = run_select_with_messages(
        df, family="exp",
        selection_cfg={"fallback_half_saturation_T": 2.0},
        saturation_cfg={"basis_dim_T": 8, "basis_tol": 1e-12},
    )
    self.assertAlmostEqual(r_exp.alpha_hat, np.log(2.0) / 2.0)
    r_log, _ = run_select_with_messages(
        df, family="log",
        selection_cfg={"fallback_half_saturation_T": 2.0},
        saturation_cfg={"basis_dim_T": 8, "basis_tol": 1e-12},
    )
    self.assertAlmostEqual(r_log.alpha_hat, 2.0)  # T_knee = alpha
    r_pow, _ = run_select_with_messages(
        df, family="power",
        selection_cfg={"fallback_half_saturation_T": 0.6},
        saturation_cfg={"basis_dim_T": 8, "basis_tol": 1e-12},
    )
    self.assertAlmostEqual(r_pow.alpha_hat, 0.6)  # alpha itself


def make_response_panel(response, n_geo=8, n_weeks=50, seed=0):
  """Geo x week panel; ``response(t)`` is the true (deterministic) effect."""
  rng = np.random.default_rng(seed)
  dates = pd.date_range("2021-01-04", periods=n_weeks, freq="7D")
  frames = []
  for g in range(n_geo):
    x1 = rng.normal(size=n_weeks)
    x2 = rng.normal(size=n_weeks)
    base = 2.0 * np.exp(0.4 * x1 + rng.normal(0.0, 0.3))
    t = base * np.exp(rng.normal(size=n_weeks) - 0.5)
    y = response(t) + 1.5 * x1 - x2 + rng.normal(0.0, 0.3) + rng.normal(
        size=n_weeks)
    frames.append(pd.DataFrame({
        "date": dates, "geo": f"G{g}", "item": "I0", "spend": t,
        "X1": x1, "X2": x2, "sales": y,
    }))
  return pd.concat(frames, ignore_index=True)


def run_manual_grid_select(df, grid):
  """Stage B on ``df`` (as D_sel) with a manual tanh grid; 1-SE forced on."""
  scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
  builder = functools.partial(
      runner.build_cate_basis, covariates_for_cate=["X1"],
      cate_type="auto_additive", cate_df=3, cate_degree=2,
      cate_intercept=False,
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return alpha_selection.select_alpha(
        df_sel=df, y_col="sales", d_col="spend", x_cols=["X1", "X2"],
        basis_builder=builder, scaler=scaler, family="tanh",
        saturation_cfg={"grid": dict(grid, mode="manual", scale="log")},
        # d_bar < 1e12 * se always holds -> the 1-SE rule alone would adopt.
        selection_cfg={"n_seeds": 2, "k_folds": 2, "linear_test_k": -1e12},
        cluster_labels=df["geo"].to_numpy(), ml_models_y=["rf"],
        ml_models_t=["rf"], time_budget=1, n_jobs=1,
        se_cluster_labels=df["geo"].to_numpy(), se_cluster_key="geo",
        item_col="item", geo_col="geo", date_col="date",
    )


class GridEdgeRuleTest(unittest.TestCase):
  """The grid-edge rule: upper edge forces linear, lower edge only warns.

  * An ``alpha_hat`` that is still on the **upper** (linear-side, large
    ``T_1/2``) edge of the final grid forces ``selected_family = None`` with
    ``decision_reason = "grid_edge_upper"`` and ``grid_edge_hit = "upper"``,
    regardless of the 1-SE decision (the 1-SE statistics are still reported).
  * An ``alpha_hat`` on the **lower** (fast-saturation) edge only warns:
    ``grid_edge_hit = "lower"`` and the 1-SE rule decides as usual.

  Both cases use a coarse *manual* grid, which is not extensible, so the edge
  is final without waiting for two x4 extensions (the code applies the rule
  to the final grid whatever the reason it can no longer be extended).  The
  1-SE rule is forced to "pass" with ``linear_test_k = -1e12`` so that the
  grid-edge override is the only thing that can produce a linear decision.
  """

  def test_upper_edge_forces_linear_regardless_of_one_se(self):
    """Linear truth, coarse tanh grid: argmin at the smallest alpha."""
    df = make_response_panel(lambda t: 3.0 * t)
    # tanh approaches the linear model as alpha -> 0, so the first index is
    # the linear (upper T_1/2) side of this alpha-coordinate grid.
    r = run_manual_grid_select(df, {"min": 0.3, "max": 3.0, "num": 4})
    self.assertEqual(r.grid_extensions, 0)
    self.assertEqual(int(np.argmin(r.mean_loss)), 0)
    self.assertEqual(r.grid_edge_hit, "upper")
    self.assertEqual(r.decision_reason, "grid_edge_upper")
    self.assertEqual(r.decision, "linear")
    self.assertIsNone(r.selected_family)
    # The 1-SE statistics are still computed and reported ...
    self.assertTrue(np.isfinite(r.d_bar))
    self.assertTrue(np.isfinite(r.se))
    self.assertEqual(r.linear_test_k, -1e12)
    # ... and on their own they would have adopted saturation.
    self.assertTrue(
        alpha_selection.linear_test_decision(r.d_bar, r.se, r.linear_test_k)
    )
    self.assertTrue(any("grid_edge_upper" in m for m in r.messages))

  def test_lower_edge_only_warns(self):
    """Strongly saturating truth, too-linear grid: argmin at the max alpha."""
    lam = np.log(3.0) / 0.5  # T_1/2 = 0.5 -> alpha ~ 2.2, far above the grid
    df = make_response_panel(lambda t: 6.0 * np.tanh(lam * t / 2.0))
    r = run_manual_grid_select(df, {"min": 0.05, "max": 0.3, "num": 4})
    self.assertEqual(r.grid_extensions, 0)
    self.assertEqual(int(np.argmin(r.mean_loss)), r.alpha_grid.size - 1)
    self.assertEqual(r.grid_edge_hit, "lower")
    # No forced None: the (forced) 1-SE rule decides.
    self.assertEqual(r.decision_reason, "1se_pass")
    self.assertEqual(r.decision, "saturation")
    self.assertEqual(r.selected_family, "tanh")
    self.assertTrue(r.grid_edge_unresolved)
    self.assertTrue(
        any("fast-saturation" in m and "warning only" in m
            for m in r.messages),
        r.messages,
    )


ALPHA_TRUE = 0.5


N_ROWS = 1500


def _simulate(seed=0):
  """Y = g(X) + theta * T**ALPHA_TRUE + eps, with T confounded by X."""
  rng = np.random.default_rng(seed)
  x1 = rng.normal(size=N_ROWS)
  x2 = rng.normal(size=N_ROWS)
  # Treatment is confounded by X (so the R-loss orthogonalization matters).
  log_t = 3.0 + 0.6 * x1 - 0.3 * x2 + rng.normal(0, 0.5, size=N_ROWS)
  t = np.exp(log_t)
  g_x = 10.0 + 3.0 * x1 - 2.0 * x2 + 1.5 * x1 * x2
  y = g_x + 2.0 * t**ALPHA_TRUE + rng.normal(0, 0.5, size=N_ROWS)
  return pd.DataFrame({
      "sales": y,
      "spend": t,
      "X1": x1,
      "X2": x2,
      "population": 1000.0,
  })


class AlphaRecoveryTest(unittest.TestCase):
  """Ground-truth recovery: alpha selection recovers a KNOWN exponent.

  Simulates a partially linear model with a known saturation exponent
  (`Y = g(X) + 2*T**0.5 + eps`, with T confounded by X so that the R-loss
  orthogonalization actually matters) and checks that profiled R-loss
  minimization recovers it.

  This guards the central scientific claim of the feature: a mis-identified
  alpha would silently produce wrong budget allocations, and no other test
  would catch it.

  Tolerance rationale: the assertion band [0.35, 0.65] is one grid step
  either side of the truth. The R-loss profile is intrinsically flat near its
  optimum, so demanding exact recovery would make this test flaky without
  making it more meaningful. What matters is that the minimum is INTERIOR and
  adjacent to the truth, not at a grid boundary.
  """

  def test_recovers_known_power_exponent(self):
    """Alpha selection on simulated data lands within one grid step of 0.5."""
    df = _simulate()
    scaler = scalers.CausalDataScaler(
        standardize_cols=["spend", "X1", "X2", "sales"],
        population_col="population",
    )

    def basis_builder(df_norm, df_orig):
      del df_orig
      return pd.DataFrame({"const": np.ones(len(df_norm))}, index=df_norm.index)

    # 10 clusters so cluster-level K-fold is well defined.
    cluster_labels = np.arange(len(df)) % 10

    result = alpha_selection.select_alpha(
        df_sel=df,
        y_col="sales",
        d_col="spend",
        x_cols=["X1", "X2"],
        basis_builder=basis_builder,
        scaler=scaler,
        family="power",
        saturation_cfg={
            "family": "power",
            "grid": {"min": 0.2, "max": 0.8, "num": 7, "scale": "linear"},
            "basis_dim_T": 6,
        },
        # This test targets the legacy FLAML learner-selection
        # path (the default RidgeCV small-sample learner is linear in X
        # and cannot represent the X1*X2 interaction of g(X)).
        selection_cfg={
            "n_seeds": 1,
            "k_folds": 3,
            "refine": "none",
            "learners": {"learner_selection": True},
        },
        cluster_labels=cluster_labels,
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        time_budget=1,
        n_jobs=1,
        random_state=0,
    )

    print("\n[alpha recovery] true =", ALPHA_TRUE)
    print("[alpha recovery] alpha_hat =", result.alpha_hat)
    print("[alpha recovery] grid_argmin =", result.grid_alpha_hat)
    print("[alpha recovery] grid =", np.round(result.alpha_grid, 3).tolist())
    print("[alpha recovery] mean_loss =",
          np.round(result.mean_loss, 6).tolist())

    print("[alpha recovery] diag1_fired =", result.diag1_fired)
    print("[alpha recovery] diag2_fired =", result.diag2_fired,
          "ratio =", result.diag2_ratio)
    print("[alpha recovery] r_T =", result.r_t)
    for m in result.messages:
      print("[alpha recovery] msg:", m)

    self.assertGreaterEqual(result.alpha_hat, 0.35)
    self.assertLessEqual(result.alpha_hat, 0.65)


class NoAbolishedNamesTest(unittest.TestCase):
  """No abolished name remains in the package sources."""

  def test_sources(self):
    """Scans every non-test ``.py`` file of the package for abolished names.

    ``test/`` is skipped because this file spells the tokens in pieces, and
    the final assertion requires that at least 10 modules were actually
    scanned so that a wrong ``root`` cannot make the test pass vacuously.
    """
    root = os.path.dirname(os.path.abspath(doubleml_pipeline.__file__))
    checked = 0
    pattern = re.compile("|".join(re.escape(t) for t in _FORBIDDEN),
                         re.IGNORECASE)
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
        self.assertFalse(hits, f"{os.path.relpath(path, root)}: {hits}")
    self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
  unittest.main()
