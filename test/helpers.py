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

"""Shared helpers of the doubleml_pipeline tests.

* :func:`saturating_panel`: a geo x week panel with a known tanh (or linear)
  dose response.  It is the ``D_sel``-like frame of the Stage B unit tests
  (test_alpha_selection.py), the panel of the Phase 0 progress / Diag6 tests
  (test_workflow.py) and the panel of the scale-invariance acceptance test
  (test_acceptance.py).
* :func:`deterministic_fit` / :func:`deterministic_search`: deterministic OLS
  stand-ins for the per-fold FLAML wrapper (``hparam_policy="per_fold"``) and
  for the global hyperparameter search (``hparam_policy="global"``).  FLAML
  searches are wall-clock bounded and therefore non-deterministic; with the
  stand-ins every full pipeline run of the test suite is exactly reproducible
  while everything else -- DoubleML cross-fitting, cluster keys, CATE basis,
  Phase 0, consolidation, Phase 4 -- runs as in production.
* :func:`find`: path of a named output file below an output directory.
* :func:`scaling_numbers` / :func:`scale_by_hand` / :func:`unscale_by_hand`:
  the formulas of the scaling strategies of ``CausalDataScaler`` written out
  by hand.  The scaler unit tests (test_preprocessing.py) and the checks of
  the places where the pipeline calls the scaler (test_scaling_pipeline.py)
  compare against them.
"""

import os
import types
from typing import Any

import numpy as np
import pandas as pd
from sklearn import linear_model

# The scaling strategies of ``CausalDataScaler``; the columns of a strategy
# are listed in its constructor argument ``<strategy>_cols``.
SCALING_STRATEGIES = (
    "standardize",
    "min_max",
    "median_normalize",
    "population_median_normalize",
    "population_standardize",
)
# The strategies that scale the value per head of population.
POPULATION_STRATEGIES = (
    "population_median_normalize",
    "population_standardize",
)
# A column that is in no list of the scaler.
NO_SCALING = "none"


def saturating_panel(
    seed: int = 0,
    truth: str = "tanh",
    n_geo: int = 10,
    n_weeks: int = 80,
    t_half: float = 2.0,
    theta: float = 4.0,
    resid: float = 1.0,
) -> pd.DataFrame:
  """Geo x week panel with a known dose response (tanh or linear).

  ``T = 2 exp(0.4 X1 + geo_t) x lognormal(sd = resid)`` and
  ``Y = effect(T) + 1.5 X1 - X2 + geo_level + noise`` with
  ``effect = theta tanh(ln 3 / t_half x T / 2)`` for ``truth="tanh"``
  and the linear ``0.4 theta T / t_half`` otherwise.

  Args:
    seed: Seed of the random number generator.
    truth: ``"tanh"`` or ``"linear"``.
    n_geo: Number of geos.
    n_weeks: Number of weekly periods per geo.
    t_half: Half-saturation point of the tanh truth.
    theta: Size of the effect.
    resid: Residual (log) standard deviation of ``T`` given ``X1``.

  Returns:
    A DataFrame with date, geo, item, spend, X1, X2 and sales columns.
  """
  rng = np.random.default_rng(seed)
  dates = pd.date_range("2021-01-04", periods=n_weeks, freq="7D")
  geo_level = rng.normal(0.0, 0.3, n_geo)
  geo_t = rng.normal(0.0, 0.5, n_geo)
  frames = []
  for g in range(n_geo):
    x1 = rng.normal(size=n_weeks)
    x2 = rng.normal(size=n_weeks)
    base = 2.0 * np.exp(0.4 * x1 + geo_t[g])
    t = base * np.exp(resid * rng.normal(size=n_weeks) - resid**2 / 2.0)
    if truth == "tanh":
      lam = np.log(3.0) / t_half
      effect = theta * np.tanh(lam * t / 2.0)
    else:
      effect = 0.4 * theta * t / t_half
    y = effect + 1.5 * x1 - x2 + geo_level[g] + rng.normal(size=n_weeks)
    frames.append(pd.DataFrame({
        "date": dates,
        "geo": f"G{g}",
        "item": "I0",
        "spend": t,
        "X1": x1,
        "X2": x2,
        "sales": y,
    }))
  return pd.concat(frames, ignore_index=True)


def deterministic_fit(self, X, y):  # pylint: disable=invalid-name
  """Replaces ``FlamlRegressorDoubleML.fit`` by a deterministic OLS fit.

  Patch it with ``mock.patch.object(flaml_dml.FlamlRegressorDoubleML, "fit",
  helpers.deterministic_fit)`` on the ``per_fold`` path.  The runner reads
  ``auto_ml_.best_loss`` for the nuisance RMSE columns.
  """
  self.tuned_model_ = linear_model.LinearRegression().fit(X, y)
  resid = np.asarray(y, dtype=float) - self.tuned_model_.predict(X)
  self.auto_ml_ = types.SimpleNamespace(
      best_loss=float(np.sqrt(np.mean(resid**2)))
  )
  return self


def deterministic_search(x, y, learner, time_budget, n_jobs, seed=1,
                         verbose=0):
  """Stands in for ``hparam_search.search_hyperparameters`` (OLS, no search).

  Patch it with ``mock.patch.object(hparam_search, "search_hyperparameters",
  mock.Mock(side_effect=helpers.deterministic_search))`` on the ``global``
  path.
  """
  del learner, time_budget, n_jobs, seed, verbose
  model = linear_model.LinearRegression().fit(x, y)
  resid = np.asarray(y, dtype=float) - model.predict(x)
  return {
      "estimator_cls": linear_model.LinearRegression,
      "params": {"fit_intercept": True},
      "best_loss": float(np.sqrt(np.mean(resid**2))),
      "n_trials": 1,
      "n_iterations": 1,
      "elapsed_seconds": 0.0,
      "eval_method": "cv",
      "flaml_estimator_class": "DeterministicOLS",
  }


def find(out_dir: str, name: str) -> str:
  """Path of the first file called ``name`` below ``out_dir``."""
  for root, _, files in os.walk(out_dir):
    if name in files:
      return os.path.join(root, name)
  raise FileNotFoundError(name)


def scaling_numbers(
    strategy: str, values: pd.Series, population: pd.Series
) -> tuple[float, float]:
  """The ``(shift, scale)`` a fit of ``strategy`` on ``values`` remembers.

  With ``base = values / population`` for the population strategies and
  ``base = values`` otherwise, the scaled value is ``(base - shift) / scale``:

  ===========================  ======  ====================================
  strategy                     shift   scale
  ===========================  ======  ====================================
  standardize                  mean    standard deviation
  min_max                      min     max - min
  median_normalize             0       median of the non-zero values
  population_median_normalize  0       median of the positive per-head values
  population_standardize       mean    standard deviation
  none                         0       1
  ===========================  ======  ====================================

  Args:
    strategy: One of ``SCALING_STRATEGIES`` or ``NO_SCALING``.
    values: The rows of the column the scaler is fitted on.
    population: The population of those rows.

  Returns:
    The ``(shift, scale)`` pair.
  """
  if strategy == NO_SCALING:
    return 0.0, 1.0
  base = values / population if strategy in POPULATION_STRATEGIES else values
  if strategy in ("standardize", "population_standardize"):
    return float(base.mean()), float(base.std(ddof=0))
  if strategy == "min_max":
    return float(base.min()), float(base.max() - base.min())
  if strategy == "median_normalize":
    return 0.0, float(base[base != 0].median())
  if strategy == "population_median_normalize":
    return 0.0, float(base[base > 0].median())
  raise ValueError(f"Unknown scaling strategy '{strategy}'.")


def scale_by_hand(
    strategy: str, numbers: tuple[float, float], values: Any, population: Any
) -> Any:
  """``values`` scaled once with the ``numbers`` of :func:`scaling_numbers`."""
  shift, scale = numbers
  base = values / population if strategy in POPULATION_STRATEGIES else values
  return (base - shift) / scale


def unscale_by_hand(
    strategy: str,
    numbers: tuple[float, float],
    scaled: Any,
    population: Any,
    is_difference: bool = False,
) -> Any:
  """``scaled`` values restored once to the original unit.

  Args:
    strategy: One of ``SCALING_STRATEGIES`` or ``NO_SCALING``.
    numbers: The ``(shift, scale)`` of :func:`scaling_numbers`.
    scaled: Scaled values, or -- with ``is_difference`` -- the difference of
      two scaled values.
    population: The population of the rows of ``scaled``.
    is_difference: The shift cancels in a difference and is not added back.

  Returns:
    The values (or the difference) in the original unit.
  """
  shift, scale = numbers
  base = scaled * scale + (0.0 if is_difference else shift)
  return base * population if strategy in POPULATION_STRATEGIES else base
