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
"""

import os
import types

import numpy as np
import pandas as pd
from sklearn import linear_model


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
