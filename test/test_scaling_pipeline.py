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

"""Scaling checks at every place where the pipeline calls the scaler.

``run_full_pipeline`` scales the data, or restores scaled values to their
original unit, at six places:

1. Phase 0: the divisor ``c`` of the treatment, from the numbers of ALL rows
   (``T_ref`` and Stage B's treatment are on the scale ``T / c``).
2. Phase 0, Stage B: the outcome and the covariates of the selection rows
   ``D_sel``, with the numbers of ``D_sel``.
3. The entrance of Phases 1-3: the scaler is fitted on the rows the model is
   fitted on (``D_inf``; all rows without a ``saturation`` block) and applied
   to them and to the rows the model only scores (``D_sel``).  A treatment
   whose saturation curve was adopted is ``fbar`` and is not scaled again.
4. The exit of Phases 1-3, fitted rows: ``Y_pred``, ``T_pred`` and the
   incremental KPI are restored.
5. The same for the scored rows.
6. Phase 4: the effect handed to the saturation-aware optimizer is restored
   to KPI units (``_compute_theta_for_optimization``).

The criterion is the same at every place: the value the pipeline works with
equals the hand formula of the scaling strategy (test/helpers.py) applied
exactly once with the numbers of the right rows -- not twice, not skipped, not
with the numbers of another column or of other rows.

Places 1-5 are checked in ``run_full_pipeline`` runs that are cut short once
the first Phase 1 model is fitted: every value the checks read exists by
then, Phase 2 does not use the scaler and Phase 3 repeats the Phase 1
calculation on the same scaled frames.  Place 6 is checked by calling the
function.  The complete runs live in test_workflow_e2e.py.

The runs vary

* the scaling of the treatment (outcome: ``population_median_normalize``) and
  the scaling of the outcome (treatment: not scaled);
* the way through Phase 0: no ``saturation`` block, the linear fallback, the
  saturation family adopted;
* the scaler: a new one, and one the caller has already fitted on the data.

Scaler settings the pipeline rejects are checked to stop with their error.

A run with a ``saturation`` block takes a few seconds (Phase 0, even without
its chart, which these runs do not draw), the others well under a second, so
the BUILD target is sharded.  The pytest shard plugin assigns the test cases
round-robin, and the runs with a ``saturation`` block come first: every shard
gets the same number of them.
"""

import contextlib
import dataclasses
import io
import json
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from unittest import mock
import warnings

from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.evaluation import saturation_evaluation
from doubleml_pipeline.models import flaml_dml
from doubleml_pipeline.preprocessing import features
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
import numpy as np
import pandas as pd
import pytest

# pylint: disable=protected-access

NONE = helpers.NO_SCALING
POPULATION = "population"

# The ways through Phase 0.
NO_SATURATION = "no_saturation"  # No `saturation` block: Phase 0 is skipped.
LINEAR = "linear_fallback"  # Phase 0 runs and keeps the linear model.
ADOPTED = "saturation_adopted"  # Phase 0 runs and adopts the family.

# Strategies a treatment may have when a saturation family is configured, and
# the ones that shift the treatment (T = 0 would no longer be 0).
_DIVIDING = (NONE, "median_normalize", "population_median_normalize")
_SHIFTING = ("standardize", "population_standardize", "min_max")

# The exponent of the adopted saturation curve `fbar = (T / T_ref) ** 0.5`
# (the power family), fixed by `_saturation_config`.
_EXPONENT = 0.5


def _panel() -> pd.DataFrame:
  """Geo x week panel: geos of very different size, weeks without spend.

  Spend and sales are proportional to the population of the geo, so the
  per-head strategies and the plain ones give clearly different numbers.
  About one week in ten has no spend (the median strategies skip such rows).
  With this seed the numbers of every scaled column differ between all rows,
  the rows Phase 0 holds out and the remaining rows (see
  `_check_numbers_tell_the_rows_apart`).

  Returns:
    A DataFrame with date, geo, item, spend, X1, X2, sales and population.
  """
  rng = np.random.default_rng(0)
  dates = pd.date_range("2022-01-03", periods=30, freq="7D")
  frames = []
  for g, population in enumerate([100.0, 150.0, 200.0, 500.0, 1000.0, 2000.0]):
    x1 = rng.normal(size=len(dates))
    x2 = rng.normal(size=len(dates))
    spend_per_head = rng.gamma(2.0, 1.5, size=len(dates)) * np.exp(0.3 * x1)
    spend_per_head[rng.random(len(dates)) < 0.1] = 0.0
    sales_per_head = (
        20.0
        + 3.0 * np.sqrt(spend_per_head)
        + 2.0 * x1
        - x2
        + rng.normal(0.0, 0.5, size=len(dates))
    )
    frames.append(
        pd.DataFrame({
            "date": dates,
            "geo": f"G{g}",
            "item": f"I{g % 2}",
            "spend": population * spend_per_head,
            "X1": x1,
            "X2": x2,
            "sales": population * sales_per_head,
            POPULATION: population,
        })
    )
  return pd.concat(frames, ignore_index=True)


def _make_scaler(strategy_of: Dict[str, str]) -> scalers.CausalDataScaler:
  """A scaler that lists every column of `strategy_of` under its strategy."""
  lists = {}
  for column, strategy in strategy_of.items():
    if strategy != NONE:
      lists.setdefault(f"{strategy}_cols", []).append(column)
  return scalers.CausalDataScaler(population_col=POPULATION, **lists)


def _saturation_config(mode: str) -> Optional[Dict[str, Any]]:
  """The `saturation_config` that makes Phase 0 take the way `mode`.

  `LINEAR`: Phase 0 adopts the family only when it improves the loss of the
  linear model by more than `linear_test_k` standard errors, so a huge
  `linear_test_k` always keeps the linear model.

  `ADOPTED`: the T-basis is pinned to 4 functions, far too few for the power
  curves (approximation error about 0.4 against a tolerance of 0.001), so the
  basis check of Phase 0 fires in every run; with
  `fallback_half_saturation_T` set, Phase 0 then adopts the family with that
  fixed exponent whatever the data say.

  Args:
    mode: `NO_SATURATION`, `LINEAR` or `ADOPTED`.

  Returns:
    The config, or None for `NO_SATURATION`.
  """
  if mode == NO_SATURATION:
    return None
  selection = {"n_seeds": 1, "k_folds": 2, "refine": "none"}
  if mode == LINEAR:
    selection["linear_test_k"] = 1e9
  else:
    selection["fallback_half_saturation_T"] = _EXPONENT
  return {
      "saturation": {
          "family": "power",
          "grid": {"min": 0.3, "max": 0.7, "num": 3, "scale": "linear"},
          "basis_dim_T": 4,
      },
      "alpha_selection": selection,
  }


def _make_orchestrator(out_dir, d_col: str, mode: str, **extra):
  """The orchestrator of the runs: one treatment, one candidate model.

  `rf` is the only FLAML learner that works under the test runner, `n_jobs=1`
  keeps joblib in the test process, and `hparam_policy="per_fold"` is the
  path `helpers.deterministic_fit` replaces.

  Args:
    out_dir: The output directory.
    d_col: The treatment column.
    mode: The way through Phase 0 (see `_saturation_config`).
    **extra: Further arguments of the orchestrator.

  Returns:
    The orchestrator.
  """
  return workflow.CausalWorkflowOrchestrator(
      y_col="sales",
      d_cols=[d_col],
      x_cols_list=[["X1", "X2"]],
      ml_models_y=["rf"],
      ml_models_t=["rf"],
      n_folds_list=[2],
      output_dir=str(out_dir),
      covariates_for_cate_list=[["X1"]],
      time_budget=1,
      n_jobs=1,
      hparam_policy="per_fold",
      date_col="date",
      geo_col="geo",
      item_col="item",
      run_sensitivity=False,
      generate_html=False,
      optimize=False,
      saturation_config=_saturation_config(mode),
      **extra,
  )


class _StopAfterFirstModelError(Exception):
  """Raised by the spy that cuts a run short after the first Phase 1 model."""


@dataclasses.dataclass
class _CutShortRun:
  """What a cut-short `run_full_pipeline` worked with.

  Attributes:
    orch: The orchestrator after the run.
    stage_b: The scaled `D_sel` frame Stage B took the outcome and the
      covariates from; None when Phase 0 was skipped.
    fit_rows: Index of the rows the model was fitted on.
    scored_rows: Index of the rows the model only scored (may be empty).
    scaled: The scaled frame the model was given: fitted, then scored rows.
    outputs: What the model returned for those rows, still in scaled units:
      the nuisance predictions `y_hat` / `t_hat` (cross-fitted for the fitted
      rows, the mean of the fold models for the scored rows) and the effect
      with its confidence bounds (`effect`, `2.5 %`, `97.5 %`).
    df_1a: The per-row result frame of the model (original units).
  """

  orch: workflow.CausalWorkflowOrchestrator
  stage_b: Optional[pd.DataFrame]
  fit_rows: pd.Index
  scored_rows: pd.Index
  scaled: pd.DataFrame
  outputs: pd.DataFrame
  df_1a: pd.DataFrame


def _mean_fold_prediction(model, learner: str, d_col: str, x) -> np.ndarray:
  """Mean prediction of the stored fold models of `learner` for `x`."""
  predictions = []
  for repetition in model.models[learner][d_col]:
    for fold_model in repetition:
      predictions.append(np.ravel(fold_model.predict(x)))
  return np.mean(predictions, axis=0)


def _run_until_first_model(
    orch, df: pd.DataFrame, scaler, d_col: str, x_cols: List[str]
) -> _CutShortRun:
  """Runs the pipeline until the first Phase 1 model is fitted.

  Args:
    orch: The orchestrator.
    df: The input frame.
    scaler: The scaler handed to `run_full_pipeline`.
    d_col: The treatment column of the model (the gap of a `price`).
    x_cols: The covariates of the model, in the order the model uses.

  Returns:
    The values the run worked with.
  """
  seen = {}
  prepare = runner.prepare_run_inputs
  basis_builder = orch._cate_basis_builder
  explore = orch._run_single_exploration

  def spy_prepare(df_orig, sc, df_score, cate_cols, cate_structure):
    seen["prepared"] = prepare(df_orig, sc, df_score, cate_cols, cate_structure)
    return seen["prepared"]

  def spy_basis_builder(*args, **kwargs):
    build = basis_builder(*args, **kwargs)

    def spy_build(df_norm, df_orig):
      seen["stage_b"] = df_norm.copy()
      return build(df_norm, df_orig)

    return spy_build

  def explore_then_stop(*args, **kwargs):
    seen["result"] = explore(*args, **kwargs)
    raise _StopAfterFirstModelError()

  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(runner, "prepare_run_inputs", spy_prepare)
    )
    stack.enter_context(
        mock.patch.object(orch, "_cate_basis_builder", spy_basis_builder)
    )
    stack.enter_context(
        mock.patch.object(orch, "_run_single_exploration", explore_then_stop)
    )
    stack.enter_context(
        mock.patch.object(
            flaml_dml.FlamlRegressorDoubleML, "fit", helpers.deterministic_fit
        )
    )
    # The chart of Phase 0 is not read by any check and takes about two
    # thirds of the time of a run with a `saturation` block.
    stack.enter_context(mock.patch.object(workflow.plots, "plot_alpha_profile"))
    stack.enter_context(warnings.catch_warnings())
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    warnings.simplefilter("ignore")
    with pytest.raises(_StopAfterFirstModelError):
      orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)

  prepared = seen["prepared"]
  model = seen["result"]["model_obj"]
  cate = model.cate(basis=prepared.cate_basis)
  scaled = [prepared.df_norm]
  outputs = [
      cate.confint(basis=prepared.cate_basis)
      .set_axis(prepared.df_norm.index, axis="index")
      .assign(
          y_hat=model.predictions["ml_l"].reshape(-1),
          t_hat=model.predictions["ml_m"].reshape(-1),
      )
  ]
  scored_rows = prepared.df_norm.index[:0]
  if prepared.df_norm_score is not None:
    scored_rows = prepared.df_norm_score.index
    x_scored = prepared.df_norm_score[x_cols].to_numpy(dtype=float)
    scaled.append(prepared.df_norm_score)
    outputs.append(
        cate.confint(basis=prepared.cate_basis_score)
        .set_axis(scored_rows, axis="index")
        .assign(
            y_hat=_mean_fold_prediction(model, "ml_l", d_col, x_scored),
            t_hat=_mean_fold_prediction(model, "ml_m", d_col, x_scored),
        )
    )
  return _CutShortRun(
      orch=orch,
      stage_b=seen.get("stage_b"),
      fit_rows=prepared.df_norm.index,
      scored_rows=scored_rows,
      scaled=pd.concat(scaled),
      outputs=pd.concat(outputs),
      df_1a=seen["result"]["df_1a"],
  )


def _assert_close(actual, expected, what: str):
  np.testing.assert_allclose(
      np.asarray(actual, dtype=float),
      np.asarray(expected, dtype=float),
      rtol=1e-9,
      atol=1e-9,
      err_msg=what,
  )


def _scaled_by_hand(strategy: str, frame: pd.DataFrame, column: str, rows):
  """`column` of ALL rows scaled by hand with the numbers of `rows`.

  Args:
    strategy: The scaling strategy of the column.
    frame: The data in original units.
    column: The column to scale.
    rows: Index of the rows the numbers (mean, median, ...) come from.

  Returns:
    `(numbers, scaled values of all rows)`.
  """
  numbers = helpers.scaling_numbers(
      strategy, frame.loc[rows, column], frame.loc[rows, POPULATION]
  )
  values = helpers.scale_by_hand(
      strategy, numbers, frame[column], frame[POPULATION]
  )
  return numbers, values


def _treatment_over_divisor(frame, d_col: str, t_strategy: str) -> pd.Series:
  """Phase 0's `T / c`: the treatment scaled with the numbers of ALL rows."""
  _, t_over_c = _scaled_by_hand(t_strategy, frame, d_col, frame.index)
  return t_over_c


def _reference_point(t_over_c: pd.Series) -> float:
  """`T_ref`: the 95% point of the positive `T / c`."""
  return float(np.quantile(t_over_c[t_over_c > 0], 0.95))


def _check_rows(run: _CutShortRun, frame: pd.DataFrame, mode: str):
  """Every row is fitted or scored; rows are scored only after Phase 0."""
  assert run.fit_rows.intersection(run.scored_rows).empty
  assert run.fit_rows.union(run.scored_rows).sort_values().equals(frame.index)
  assert (len(run.scored_rows) > 0) == (mode != NO_SATURATION)


def _check_numbers_tell_the_rows_apart(run: _CutShortRun, frame, strategy_of):
  """The numbers of every scaled column differ between the sets of rows.

  The checks compare with the numbers of ALL rows (the divisor of Phase 0),
  of the scored rows (Stage B) and of the fitted rows (Phases 1-3).  A median
  or an extreme can be the same row in two of these sets; a mix-up of the
  rows would then go unnoticed for that column, so the panel must not allow
  it.

  Args:
    run: The cut-short run.
    frame: The data in original units.
    strategy_of: The scaling strategy of every column.
  """
  row_sets = {
      "all rows": frame.index,
      "fitted rows": run.fit_rows,
      "scored rows": run.scored_rows,
  }
  numbers = {}
  for column, strategy in strategy_of.items():
    for name, rows in row_sets.items():
      numbers[column, name] = helpers.scaling_numbers(
          strategy, frame.loc[rows, column], frame.loc[rows, POPULATION]
      )
  same = []
  for column, strategy in strategy_of.items():
    for first, second in (
        ("all rows", "fitted rows"),
        ("all rows", "scored rows"),
        ("fitted rows", "scored rows"),
    ):
      if strategy != NONE and numbers[column, first] == numbers[column, second]:
        same.append(f"{column} ({strategy}): {first} = {second}")
  if same:
    raise AssertionError(
        "The panel cannot tell these sets of rows apart: " + "; ".join(same)
    )


def _check_phase0(run: _CutShortRun, frame, d_col: str, x_cols, strategy_of):
  """Places 1 and 2: the scaling inside Phase 0."""
  sel_rows = run.stage_b.index
  assert sel_rows.equals(run.scored_rows)

  # 1. The divisor c of the treatment comes from the numbers of ALL rows.
  t_over_c = _treatment_over_divisor(frame, d_col, strategy_of[d_col])
  summary_path = helpers.find(
      run.orch.output_dir, saturation_evaluation.SUMMARY_JSON
  )
  with open(summary_path) as handle:
    summary = json.load(handle)
  _assert_close(summary["T_ref"], _reference_point(t_over_c), "T_ref")
  _assert_close(
      run.orch._alpha_results[d_col].t_values,
      t_over_c.loc[sel_rows],
      "treatment of Stage B",
  )

  # 2. The outcome and the covariates of Stage B use the numbers of D_sel.
  for column in ["sales"] + x_cols:
    _, expected = _scaled_by_hand(strategy_of[column], frame, column, sel_rows)
    _assert_close(
        run.stage_b[column], expected.loc[sel_rows], f"{column} of Stage B"
    )


def _treatment_of_the_model(run, frame, d_col: str, t_strategy: str, mode):
  """The treatment the model is given, calculated by hand.

  With the saturation family adopted it is `fbar = ((T / c) / T_ref) ** 0.5`,
  which is not scaled again and is therefore restored unchanged.  Otherwise
  it is the treatment scaled with the numbers of the rows the model is fitted
  on.

  Args:
    run: The cut-short run.
    frame: The data in original units.
    d_col: The treatment column.
    t_strategy: The scaling strategy the caller gave the treatment.
    mode: The way through Phase 0.

  Returns:
    `(strategy the model's treatment is restored with, its numbers, the
    treatment of all rows)`.
  """
  if mode == ADOPTED:
    t_over_c = _treatment_over_divisor(frame, d_col, t_strategy)
    fbar = (t_over_c / _reference_point(t_over_c)) ** _EXPONENT
    return NONE, helpers.scaling_numbers(NONE, fbar, frame[POPULATION]), fbar
  numbers, values = _scaled_by_hand(t_strategy, frame, d_col, run.fit_rows)
  return t_strategy, numbers, values


def _check_phases_1_to_3(
    run: _CutShortRun, frame, d_col: str, x_cols, strategy_of, mode: str
):
  """Places 3-5: the entrance and the exit of Phases 1-3."""
  # Every column of the model by hand: (strategy, numbers, scaled values).
  # The numbers are those of the rows the model is fitted on.
  by_hand = {}
  for column in ["sales"] + x_cols:
    strategy = strategy_of[column]
    by_hand[column] = (strategy,) + _scaled_by_hand(
        strategy, frame, column, run.fit_rows
    )
  by_hand[d_col] = _treatment_of_the_model(
      run, frame, d_col, strategy_of[d_col], mode
  )
  y_strategy, y_numbers, _ = by_hand["sales"]
  t_strategy, t_numbers, t_scaled = by_hand[d_col]
  kpi = f"estimated_incremental_KPI_{d_col}"

  for rows, label in (
      (run.fit_rows, "fitted rows"),
      (run.scored_rows, "scored rows"),
  ):
    population = frame.loc[rows, POPULATION]
    outputs = run.outputs.loc[rows]
    results = run.df_1a.loc[rows]

    # 3. Entrance: each column is scaled once.
    for column, (_, _, values) in by_hand.items():
      _assert_close(
          run.scaled.loc[rows, column],
          values.loc[rows],
          f"{column} given to the model, {label}",
      )

    # 4. / 5. Exit: the outputs of the model are restored once.
    y_pred = helpers.unscale_by_hand(
        y_strategy, y_numbers, outputs["y_hat"], population
    )
    _assert_close(results["Y_pred"], y_pred, f"Y_pred, {label}")
    _assert_close(
        results["Y_res"], frame.loc[rows, "sales"] - y_pred, f"Y_res, {label}"
    )
    t_pred = helpers.unscale_by_hand(
        t_strategy, t_numbers, outputs["t_hat"], population
    )
    treatment = helpers.unscale_by_hand(
        t_strategy, t_numbers, t_scaled.loc[rows], population
    )
    _assert_close(results["T_pred"], t_pred, f"T_pred, {label}")
    _assert_close(results["T_res"], treatment - t_pred, f"T_res, {label}")
    # Incremental KPI: (treatment - treatment at T = 0) x effect, which is a
    # difference of two outcomes.
    t_zero = helpers.scale_by_hand(t_strategy, t_numbers, 0.0, population)
    for column, bound in (
        (kpi, "effect"),
        (f"{kpi}_2.5%", "2.5 %"),
        (f"{kpi}_97.5%", "97.5 %"),
    ):
      _assert_close(
          results[column],
          helpers.unscale_by_hand(
              y_strategy,
              y_numbers,
              (t_scaled.loc[rows] - t_zero) * outputs[bound],
              population,
              is_difference=True,
          ),
          f"{column}, {label}",
      )


def _id(t_strategy: str, y_strategy: str, mode: str, fitted: bool) -> str:
  scaler = "fitted_scaler" if fitted else "new_scaler"
  return f"T_{t_strategy}-Y_{y_strategy}-{mode}-{scaler}"


# The scaler settings of the runs as (treatment strategy, outcome strategy):
# the treatment varies under the outcome `population_median_normalize`, the
# outcome varies under an unscaled treatment.
_OUTCOME_FIXED = "population_median_normalize"
_OUTCOME_VARIES = [
    (NONE, y)
    for y in helpers.SCALING_STRATEGIES + (NONE,)
    if y != _OUTCOME_FIXED
]


def _saturation_cases():
  """The runs with a `saturation` block.

  Returns:
    The pytest parameters (treatment strategy, outcome strategy, way through
    Phase 0, whether the caller has already fitted the scaler).
  """
  settings = [(t, _OUTCOME_FIXED) for t in _DIVIDING] + _OUTCOME_VARIES
  cases = []
  for mode in (ADOPTED, LINEAR):
    for t_strategy, y_strategy in settings:
      for fitted in (False, True):
        cases.append(
            pytest.param(
                t_strategy,
                y_strategy,
                mode,
                fitted,
                id=_id(t_strategy, y_strategy, mode, fitted),
            )
        )
  return cases


def _no_saturation_cases():
  """The runs without a `saturation` block.

  Returns:
    The pytest parameters (treatment strategy, outcome strategy, whether the
    caller has already fitted the scaler).
  """
  settings = [
      (t, _OUTCOME_FIXED) for t in _DIVIDING + _SHIFTING
  ] + _OUTCOME_VARIES
  cases = []
  for t_strategy, y_strategy in settings:
    for fitted in (False, True):
      cases.append(
          pytest.param(
              t_strategy,
              y_strategy,
              fitted,
              id=_id(t_strategy, y_strategy, NO_SATURATION, fitted),
          )
      )
  return cases


def _strategies(t_strategy: str, y_strategy: str) -> Dict[str, str]:
  return {
      "spend": t_strategy,
      "sales": y_strategy,
      "X1": "standardize",
      "X2": "standardize",
  }


@pytest.mark.parametrize(
    ("t_strategy", "y_strategy", "mode", "fitted"), _saturation_cases()
)
def test_scaling_with_a_saturation_block(
    t_strategy, y_strategy, mode, fitted, tmp_path
):
  """Places 1-5 on both ways through Phase 0: linear fallback and adopted."""
  df = _panel()
  strategy_of = _strategies(t_strategy, y_strategy)
  scaler = _make_scaler(strategy_of)
  if fitted:
    scaler.fit(df)
  orch = _make_orchestrator(tmp_path, "spend", mode)

  run = _run_until_first_model(orch, df, scaler, "spend", ["X1", "X2"])

  selected_family = {LINEAR: None, ADOPTED: "power"}[mode]
  assert orch._selected_family["spend"] == selected_family
  _check_rows(run, df, mode)
  _check_numbers_tell_the_rows_apart(run, df, strategy_of)
  _check_phase0(run, df, "spend", ["X1", "X2"], strategy_of)
  _check_phases_1_to_3(run, df, "spend", ["X1", "X2"], strategy_of, mode)


@pytest.mark.parametrize(
    ("t_strategy", "y_strategy", "fitted"), _no_saturation_cases()
)
def test_scaling_without_a_saturation_block(
    t_strategy, y_strategy, fitted, tmp_path
):
  """Places 3 and 4: Phase 0 is skipped and the model is fitted on all rows."""
  df = _panel()
  strategy_of = _strategies(t_strategy, y_strategy)
  scaler = _make_scaler(strategy_of)
  if fitted:
    scaler.fit(df)
  orch = _make_orchestrator(tmp_path, "spend", NO_SATURATION)

  run = _run_until_first_model(orch, df, scaler, "spend", ["X1", "X2"])

  _check_rows(run, df, NO_SATURATION)
  _check_phases_1_to_3(
      run, df, "spend", ["X1", "X2"], strategy_of, NO_SATURATION
  )


@pytest.mark.parametrize(
    "fitted", [False, True], ids=["new_scaler", "fitted_scaler"]
)
def test_scaling_of_a_price_treatment(fitted, tmp_path):
  """The gap and the benchmark of a price are scaled like the price."""
  rng = np.random.default_rng(11)
  df = _panel()
  df["price"] = rng.uniform(80.0, 120.0, size=len(df))
  scaler = _make_scaler({
      "price": "standardize",
      "sales": "population_median_normalize",
      "X1": "standardize",
      "X2": "standardize",
  })
  if fitted:
    scaler.fit(df)
  orch = _make_orchestrator(
      tmp_path, "price", NO_SATURATION, treatment_types={"price": "price"}
  )
  model_x_cols = ["X1", "X2", "price_benchmark"]

  run = _run_until_first_model(orch, df, scaler, "price_gap", model_x_cols)

  # The pipeline replaces the price by its gap from the benchmark (the
  # highest price of the item x geo group) and adds the benchmark to the
  # covariates; both join the scaler list of the price.
  benchmark = df.groupby(["item", "geo"])["price"].transform("max")
  frame = df.assign(
      price_benchmark=benchmark, price_gap=benchmark - df["price"]
  )
  strategy_of = {
      "price_gap": "standardize",
      "price_benchmark": "standardize",
      "sales": "population_median_normalize",
      "X1": "standardize",
      "X2": "standardize",
  }
  _check_rows(run, frame, NO_SATURATION)
  _check_phases_1_to_3(
      run, frame, "price_gap", model_x_cols, strategy_of, NO_SATURATION
  )


def test_scaling_of_a_lagged_price_treatment(tmp_path):
  """The gap of a price lag is scaled like the lag, not like the price."""
  rng = np.random.default_rng(11)
  df = _panel()
  df["price"] = rng.uniform(80.0, 120.0, size=len(df))
  df = features.create_lagged_features(
      df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
  ).reset_index(drop=True)
  scaler = _make_scaler({
      "price_l1": "standardize",
      "price": "min_max",
      "sales": "population_median_normalize",
      "X1": "standardize",
      "X2": "standardize",
  })
  orch = _make_orchestrator(
      tmp_path, "price_l1", NO_SATURATION, treatment_types={"price_l1": "price"}
  )
  model_x_cols = ["X1", "X2", "price_benchmark"]

  run = _run_until_first_model(orch, df, scaler, "price_gap_l1", model_x_cols)

  # The pipeline replaces the lag by the gap of the same item x geo group one
  # week earlier (0 in the first week; the weeks of a group are consecutive
  # rows) and adds the benchmark to the covariates.  The gap joins the scaler
  # list of the lag, the benchmark the list of the price.
  groups = [df["item"], df["geo"]]
  benchmark = df.groupby(["item", "geo"])["price"].transform("max")
  gap = benchmark - df["price"]
  frame = df.assign(
      price_benchmark=benchmark,
      price_gap_l1=gap.groupby(groups).shift(1).fillna(0.0),
  )
  strategy_of = {
      "price_gap_l1": "standardize",
      "price_benchmark": "min_max",
      "sales": "population_median_normalize",
      "X1": "standardize",
      "X2": "standardize",
  }
  _check_rows(run, frame, NO_SATURATION)
  _check_phases_1_to_3(
      run, frame, "price_gap_l1", model_x_cols, strategy_of, NO_SATURATION
  )


@pytest.mark.parametrize("mode", [NO_SATURATION, LINEAR])
def test_run_stops_without_the_population_column(mode, tmp_path):
  """With and without Phase 0, the run stops before any model is fitted."""
  df = _panel().drop(columns=POPULATION)
  scaler = _make_scaler(
      _strategies("population_median_normalize", "population_median_normalize")
  )
  orch = _make_orchestrator(tmp_path, "spend", mode)
  message = "Specified population column 'population' not found in dataframe."

  with pytest.raises(ValueError, match=re.escape(message)):
    _run_until_first_model(orch, df, scaler, "spend", ["X1", "X2"])


@pytest.mark.parametrize("t_strategy", _SHIFTING)
def test_run_stops_when_a_shifted_treatment_meets_a_saturation_family(
    t_strategy, tmp_path
):
  """Phase 0 rejects a treatment whose scaling moves T = 0 away from 0."""
  df = _panel()
  scaler = _make_scaler(_strategies(t_strategy, "population_median_normalize"))
  orch = _make_orchestrator(tmp_path, "spend", LINEAR)
  message = (
      f"Treatment 'spend' is configured for '{t_strategy}' normalization,"
      " which is incompatible with a saturation family"
  )

  with pytest.raises(ValueError, match=re.escape(message)):
    _run_until_first_model(orch, df, scaler, "spend", ["X1", "X2"])


@pytest.mark.parametrize("y_strategy", helpers.SCALING_STRATEGIES + (NONE,))
def test_effect_for_the_optimizer_is_restored_to_kpi_units(
    y_strategy, tmp_path
):
  """Place 6: the effect of every row, also of the rows without spend.

  The optimizer takes the effect as KPI / fbar where there is spend and from
  the CATE coefficients -- restored to KPI units -- where there is none.
  """
  df = _panel()
  assert (df["spend"] == 0).any()  # Rows the CATE coefficients have to fill.
  fit_rows = df.index[df.index % 5 != 0]
  strategy_of = _strategies(NONE, y_strategy)
  scaler = _make_scaler(strategy_of).fit(df.loc[fit_rows])
  orch = _make_orchestrator(
      tmp_path, "spend", ADOPTED, cate_structure_list=["X1"]
  )
  t_ref = _reference_point(df["spend"])
  orch._alpha_hat["spend"] = _EXPONENT
  orch._selected_family["spend"] = "power"
  orch._t_ref["spend"] = t_ref
  # The effect of the model in scaled units, linear in the scaled X1, and
  # the same effect in KPI units.
  intercept, slope = 0.2, 0.05
  _, x1_scaled = _scaled_by_hand("standardize", df, "X1", fit_rows)
  y_numbers, _ = _scaled_by_hand(y_strategy, df, "sales", fit_rows)
  effect_kpi = helpers.unscale_by_hand(
      y_strategy,
      y_numbers,
      intercept + slope * x1_scaled,
      df[POPULATION],
      is_difference=True,
  )
  fbar = (df["spend"] / t_ref) ** _EXPONENT
  best_model_data = {
      "df_1a": pd.DataFrame({
          "estimated_incremental_KPI_spend": effect_kpi * fbar,
          "spend_transformed": fbar,
      }),
      "cate_coef_dict": {"Intercept_coef": intercept, "X1_coef": slope},
  }

  theta = orch._compute_theta_for_optimization(
      "spend", 0, df, scaler, best_model_data
  )

  _assert_close(theta, effect_kpi, "effect in KPI units")
