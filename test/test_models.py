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

"""Tests for the ``models`` package: ``flaml_dml`` and ``fixed_hparam_regressor``.

* ``FlamlRegressorDoubleML``: construction, ``sklearn.base.clone`` safety and
  the ``early_stop`` / ``n_jobs`` arguments that reach FLAML.
* ``FixedHparamRegressor`` (dml_performance_spec §3.3 [P-B1] / §3.4 [P-B1c]):
  clone identity, finite ``internal_val_rmse_``, KFold(5) / holdout
  validation, and ``build_estimator`` only passing what the class accepts.

``models.saturation`` is covered by test_saturation.py.
"""

import math
import types
import unittest
from unittest import mock

from doubleml_pipeline.models import fixed_hparam_regressor
from doubleml_pipeline.models import flaml_dml
import numpy as np
from sklearn import base
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree


class FlamlDMLTest(unittest.TestCase):
  """Construction and ``sklearn.base.clone`` safety of the wrapper."""

  def test_initialization(self):
    """Tests FlamlDML initialization."""
    model = flaml_dml.FlamlRegressorDoubleML(
        time=10,
        estimator_list=["lgbm"],
        metric="rmse",
        random_state=42,
    )
    self.assertEqual(model.time, 10)
    self.assertEqual(model.estimator_list, ["lgbm"])
    self.assertEqual(model.metric, "rmse")
    self.assertEqual(model.random_state, 42)

  def test_clone_without_sharing_mutable_state(self):
    """Tests that cloned instances do not share mutable state."""
    model = flaml_dml.FlamlRegressorDoubleML(
        time=10,
        estimator_list=["lgbm", "xgboost"],
        metric="rmse",
        random_state=42,
    )
    cloned_model = base.clone(model)
    self.assertIsNot(model.estimator_list, cloned_model.estimator_list)
    self.assertEqual(model.estimator_list, cloned_model.estimator_list)


class FlamlWrapperArgumentsTest(unittest.TestCase):
  """``early_stop`` / ``n_jobs`` reach FLAML."""

  def _run(self, **wrapper_kwargs):
    captured = {}

    class _FakeAutoML:

      def __init__(self, **kwargs):
        captured["init"] = kwargs
        self._state = types.SimpleNamespace(eval_method="holdout")
        self.config_history = {0: ("rf", {}, 0.1), 1: ("rf", {}, 0.2)}
        self.best_loss = 0.5

      def fit(self, **kwargs):
        captured["fit"] = kwargs
        est = types.SimpleNamespace(
            predict=lambda x: np.zeros(len(x)), get_params=lambda: {}
        )
        self.model = types.SimpleNamespace(estimator=est)

    with mock.patch.object(flaml_dml.flaml, "AutoML", _FakeAutoML):
      wrapper = flaml_dml.FlamlRegressorDoubleML(
          time=5, estimator_list=["rf"], metric="rmse", random_state=3,
          **wrapper_kwargs,
      )
      wrapper.fit(np.zeros((10, 2)), np.zeros(10))
    return wrapper, captured["fit"]

  def test_per_fold_passes_early_stop_and_n_jobs(self):
    wrapper, fit_kwargs = self._run(early_stop=True, n_jobs=2)
    self.assertIs(fit_kwargs["early_stop"], True)
    self.assertEqual(fit_kwargs["n_jobs"], 2)
    self.assertNotIn("max_iter", fit_kwargs)
    self.assertEqual(fit_kwargs["time_budget"], 5)
    self.assertEqual(fit_kwargs["seed"], 3)
    # The stop time / trials / eval_method are recorded for the log.
    self.assertGreaterEqual(wrapper.search_elapsed_seconds_, 0.0)
    self.assertEqual(wrapper.n_trials_, 2)
    self.assertEqual(wrapper.eval_method_, "holdout")

  def test_defaults_do_not_pass_the_new_arguments(self):
    """Callers that do not set them (Phase 0) issue the previous FLAML call."""
    _, fit_kwargs = self._run()
    self.assertNotIn("early_stop", fit_kwargs)
    self.assertNotIn("n_jobs", fit_kwargs)
    self.assertNotIn("max_iter", fit_kwargs)

  def test_get_params_carries_the_new_arguments(self):
    wrapper = flaml_dml.FlamlRegressorDoubleML(
        time=5, estimator_list=["rf"], metric="rmse", early_stop=True, n_jobs=3
    )
    params = wrapper.get_params()
    self.assertIs(params["early_stop"], True)
    self.assertEqual(params["n_jobs"], 3)
    wrapper.set_params(n_jobs=1, early_stop=False)
    self.assertEqual(wrapper.n_jobs, 1)
    self.assertIs(wrapper.early_stop, False)


class _CountingRegressor(base.BaseEstimator, base.RegressorMixin):
  """A linear fit that counts how often ``fit`` is called (class-wide)."""

  n_fits = 0

  def __init__(self, alpha=1.0, n_jobs=None, random_state=None):
    self.alpha = alpha
    self.n_jobs = n_jobs
    self.random_state = random_state

  def fit(self, X, y):  # pylint: disable=invalid-name
    _CountingRegressor.n_fits += 1
    self.model_ = linear_model.Ridge(alpha=self.alpha).fit(X, y)
    return self

  def predict(self, X):  # pylint: disable=invalid-name
    return self.model_.predict(X)


def _data(n=200, seed=0):
  rng = np.random.default_rng(seed)
  x = rng.normal(size=(n, 3))
  y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=n)
  return x, y


class FixedHparamRegressorTest(unittest.TestCase):
  """Clone identity, finite ``internal_val_rmse_``, KFold(5) validation."""

  def setUp(self):
    super().setUp()
    _CountingRegressor.n_fits = 0

  def test_get_params_identical_after_clone(self):
    """``get_params()`` survives ``sklearn.base.clone`` (DoubleML per fold)."""
    params = {"alpha": 0.3}
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor,
        params,
        n_jobs=2,
        random_state=7,
        val_method=fixed_hparam_regressor.VAL_METHOD_HOLDOUT,
        val_n_splits=4,
        val_split_ratio=0.2,
    )
    cloned = base.clone(reg)
    self.assertIsNot(cloned, reg)
    self.assertEqual(cloned.get_params(), reg.get_params())
    self.assertIs(cloned.estimator_cls, _CountingRegressor)
    # The clone must be unfitted and must not share the mutable params dict.
    self.assertFalse(hasattr(cloned, "model_"))
    self.assertIsNot(cloned.params, reg.params)
    self.assertEqual(cloned.params, params)

  def test_set_params_round_trip_and_validation(self):
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor, {"alpha": 1.0}
    )
    reg.set_params(
        n_jobs=3, val_method=fixed_hparam_regressor.VAL_METHOD_HOLDOUT
    )
    self.assertEqual(reg.n_jobs, 3)
    self.assertEqual(reg.val_method, fixed_hparam_regressor.VAL_METHOD_HOLDOUT)
    with self.assertRaises(ValueError):
      reg.set_params(not_a_param=1)

  def test_internal_val_rmse_finite_after_fit(self):
    """``internal_val_rmse_`` is a finite positive number after ``fit``."""
    x, y = _data()
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor, {"alpha": 0.1}, random_state=1
    )
    reg.fit(x, y)
    self.assertTrue(math.isfinite(reg.internal_val_rmse_))
    self.assertGreater(reg.internal_val_rmse_, 0.0)
    self.assertLess(reg.internal_val_rmse_, 1.0)
    preds = reg.predict(x)
    self.assertEqual(preds.shape, (len(y),))
    self.assertIsInstance(reg.model_, _CountingRegressor)

  def test_cv_uses_kfold_with_five_shuffled_splits(self):
    """``val_method="cv"``: ``KFold(5, shuffle=True, random_state)``."""
    x, y = _data()
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor, {"alpha": 0.1}, random_state=11
    )
    with mock.patch.object(
        fixed_hparam_regressor.model_selection,
        "KFold",
        wraps=model_selection.KFold,
    ) as kfold:
      reg.fit(x, y)
    kfold.assert_called_once_with(n_splits=5, shuffle=True, random_state=11)
    # 5 validation fits + the final fit with the fixed theta*.
    self.assertEqual(_CountingRegressor.n_fits, 6)

  def test_holdout_uses_one_split(self):
    """``val_method="holdout"``: one ``train_test_split(test_size=0.1)``."""
    x, y = _data()
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor,
        {"alpha": 0.1},
        random_state=3,
        val_method=fixed_hparam_regressor.VAL_METHOD_HOLDOUT,
    )
    with mock.patch.object(
        fixed_hparam_regressor.model_selection,
        "train_test_split",
        wraps=model_selection.train_test_split,
    ) as split:
      reg.fit(x, y)
    self.assertEqual(split.call_count, 1)
    self.assertEqual(split.call_args.kwargs["test_size"], 0.1)
    self.assertEqual(split.call_args.kwargs["random_state"], 3)
    # 1 validation fit + the final fit.
    self.assertEqual(_CountingRegressor.n_fits, 2)
    self.assertTrue(math.isfinite(reg.internal_val_rmse_))

  def test_internal_validation_matches_manual_kfold(self):
    """The cv value is the mean over the KFold splits of the fold RMSE."""
    x, y = _data(seed=5)
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor, {"alpha": 0.5}, random_state=2
    )
    reg.fit(x, y)
    folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=2)
    losses = []
    for train_idx, val_idx in folds.split(x):
      model = _CountingRegressor(alpha=0.5).fit(x[train_idx], y[train_idx])
      diff = y[val_idx] - model.predict(x[val_idx])
      losses.append(math.sqrt(float(np.mean(diff**2))))
    self.assertAlmostEqual(reg.internal_val_rmse_, float(np.mean(losses)), 12)

  def test_unknown_val_method_raises(self):
    x, y = _data()
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor, {}, val_method="bootstrap"
    )
    with self.assertRaises(ValueError):
      reg.fit(x, y)

  def test_fixed_hparam_regressors_reach_the_estimator(self):
    """``theta*`` is applied verbatim; n_jobs / random_state are overridden."""
    x, y = _data()
    reg = fixed_hparam_regressor.FixedHparamRegressor(
        _CountingRegressor,
        {"alpha": 0.25, "n_jobs": -1, "random_state": 999},
        n_jobs=4,
        random_state=1,
    )
    reg.fit(x, y)
    self.assertEqual(reg.model_.alpha, 0.25)
    self.assertEqual(reg.model_.n_jobs, 4)
    self.assertEqual(reg.model_.random_state, 1)


class BuildEstimatorTest(unittest.TestCase):
  """``build_estimator`` only passes what the class accepts."""

  def test_random_state_dropped_for_classes_without_it(self):
    est = fixed_hparam_regressor.build_estimator(
        neighbors.KNeighborsRegressor,
        {"n_neighbors": 3, "random_state": 5},
        n_jobs=2,
        random_state=1,
    )
    self.assertEqual(est.get_params()["n_neighbors"], 3)
    self.assertEqual(est.get_params()["n_jobs"], 2)
    self.assertNotIn("random_state", est.get_params())

  def test_n_jobs_dropped_for_classes_without_it(self):
    est = fixed_hparam_regressor.build_estimator(
        tree.DecisionTreeRegressor,
        {"max_depth": 2, "n_jobs": 8},
        n_jobs=2,
        random_state=1,
    )
    self.assertEqual(est.get_params()["max_depth"], 2)
    self.assertEqual(est.get_params()["random_state"], 1)
    self.assertNotIn("n_jobs", est.get_params())

  def test_callbacks_are_dropped(self):
    est = fixed_hparam_regressor.build_estimator(
        linear_model.LinearRegression, {"callbacks": [object()]}
    )
    self.assertNotIn("callbacks", est.get_params())

  def test_accepts_parameter(self):
    self.assertTrue(
        fixed_hparam_regressor.accepts_parameter(
            linear_model.LinearRegression, "n_jobs"
        )
    )
    self.assertFalse(
        fixed_hparam_regressor.accepts_parameter(
            linear_model.LinearRegression, "random_state"
        )
    )


if __name__ == "__main__":
  unittest.main()
