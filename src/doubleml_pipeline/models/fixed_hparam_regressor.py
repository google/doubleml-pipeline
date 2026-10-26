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

"""Nuisance learner with FIXED hyperparameters (the "global" policy).

``hparam_policy = "global"`` separates the hyperparameter search
from the nuisance fits: FLAML runs ONCE per (treatment, learner) on the full
sample (``evaluation.hparam_search``) and every fold of every Phase 1 / Phase 3
cross-fitting then fits the concrete estimator class with that fixed parameter
dict ``theta*``.  :class:`FixedHparamRegressor` is the scikit-learn compatible
wrapper DoubleML receives as ``ml_l`` / ``ml_m``; ``sklearn.base.clone`` (which
DoubleML calls for every fold) recreates an unfitted copy from ``get_params``.

``fit`` also computes the quantity the combined error of Phase 2 / the
best-group selection of Phase 3 is built on: the validation RMSE of ``theta*``
measured INSIDE the fold's training data, with the same method FLAML chose for
the search (``cv`` -> ``KFold(5, shuffle=True)``; ``holdout`` -> one
``train_test_split(test_size=0.1)``).  It is stored in ``internal_val_rmse_``
and read by ``evaluation.runner`` in place of FLAML's ``best_loss``.
"""

import inspect
import math
from typing import Any, Callable, Dict, Optional
import warnings

import numpy as np
from sklearn import base
from sklearn import model_selection
from sklearn.utils import validation

# Validation methods; they mirror FLAML's `eval_method`.
VAL_METHOD_CV = 'cv'
VAL_METHOD_HOLDOUT = 'holdout'
VAL_METHODS = (VAL_METHOD_CV, VAL_METHOD_HOLDOUT)

# Keys of `theta*` that are REPLACED by the constructor arguments of
# `FixedHparamRegressor` (`estimator_cls(**params, n_jobs=n_jobs,
# random_state=random_state)`) whenever that argument is not None (a None
# argument leaves FLAML's value in `theta*` untouched), and keys that are
# fit-time objects FLAML may leave on the estimator (never constructor
# parameters of a fresh fit).
OVERRIDDEN_PARAM_KEYS = ('n_jobs', 'random_state')
DROPPED_PARAM_KEYS = ('callbacks',)


def accepts_parameter(estimator_cls: Any, name: str) -> bool:
  """Whether ``estimator_cls(**{name: ...})`` is a valid constructor call.

  scikit-learn estimators list their hyperparameters explicitly in
  ``__init__``; estimators with a ``**kwargs`` constructor (e.g. XGBoost) are
  probed through ``get_params`` of a default instance.

  Args:
    estimator_cls: The estimator class.
    name: A constructor keyword (``"n_jobs"`` / ``"random_state"``).

  Returns:
    True when the keyword is accepted.
  """
  try:
    parameters = inspect.signature(estimator_cls.__init__).parameters
  except (TypeError, ValueError):  # pragma: no cover - builtins / C classes
    return False
  if name in parameters:
    return True
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
    try:
      return name in estimator_cls().get_params()
    except Exception:  # pylint: disable=broad-except
      return False
  return False


def build_estimator(
    estimator_cls: Any,
    params: Optional[Dict[str, Any]],
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Any:
  """Instantiates ``estimator_cls`` with ``theta*``, the run's threads and seed.

  The call is ``estimator_cls(**params, n_jobs=n_jobs,
  random_state=random_state)``.  ``n_jobs`` / ``random_state`` contained in
  ``params`` (FLAML's values) are replaced; either is only passed when the
  class accepts it (e.g. ``KNeighborsRegressor`` has no ``random_state``).

  Args:
    estimator_cls: Concrete estimator class (``lightgbm.LGBMRegressor``, ...).
    params: ``theta*`` = ``automl.model.estimator.get_params()``.
    n_jobs: Threads of this fit (``h`` of the parallel plan); None = leave the
      class default / the value in ``params``.
    random_state: Seed of this fit; None = leave the class default / the value
      in ``params``.

  Returns:
    An unfitted estimator.
  """
  kwargs = {
      key: value
      for key, value in dict(params or {}).items()
      if key not in DROPPED_PARAM_KEYS
  }
  for key, value in zip(OVERRIDDEN_PARAM_KEYS, (n_jobs, random_state)):
    if value is None:
      continue
    if accepts_parameter(estimator_cls, key):
      kwargs[key] = int(value)
    else:
      kwargs.pop(key, None)
  return estimator_cls(**kwargs)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  diff = (
      np.asarray(y_true, dtype=float).ravel()
      - np.asarray(y_pred, dtype=float).ravel()
  )
  return float(math.sqrt(float(np.mean(diff**2))))


def internal_validation_rmse(
    make_estimator: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    val_method: str = VAL_METHOD_CV,
    n_splits: int = 5,
    split_ratio: float = 0.1,
    random_state: Optional[int] = None,
) -> float:
  """The validation RMSE of ``theta*`` inside one fold's training data.

  * ``cv``: ``KFold(n_splits, shuffle=True, random_state)`` over ``(x, y)``;
    the estimator is fitted on each training part and the RMSE on the
    validation part is averaged over the splits (FLAML's ``cv`` loss is the
    mean of the per-fold losses as well).
  * ``holdout``: one ``train_test_split(test_size=split_ratio,
    random_state)``; the RMSE on the held-out part.

  ``n_splits`` is reduced to ``len(y)`` when the data has fewer rows; with a
  single row the value is NaN (nothing can be validated).

  Args:
    make_estimator: Factory returning a fresh unfitted estimator.
    x: Training design matrix of the fold.
    y: Training target of the fold.
    val_method: ``"cv"`` | ``"holdout"``.
    n_splits: ``KFold`` splits (``cv``).
    split_ratio: Validation share (``holdout``).
    random_state: Seed of the split.

  Returns:
    The validation RMSE (``r_l`` / ``r_m`` of this fold).

  Raises:
    ValueError: On an unknown ``val_method``.
  """
  if val_method not in VAL_METHODS:
    raise ValueError(
        f'val_method must be one of {VAL_METHODS}, got {val_method!r}.'
    )
  x = np.asarray(x)
  y = np.asarray(y, dtype=float).ravel()
  n_obs = int(len(y))
  if n_obs < 2:
    return float('nan')

  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if val_method == VAL_METHOD_CV:
      splits = max(2, min(int(n_splits), n_obs))
      folds = model_selection.KFold(
          n_splits=splits, shuffle=True, random_state=random_state
      )
      losses = []
      for train_idx, val_idx in folds.split(x):
        model = make_estimator().fit(x[train_idx], y[train_idx])
        losses.append(_rmse(y[val_idx], model.predict(x[val_idx])))
      return float(np.mean(losses))

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x, y, test_size=split_ratio, random_state=random_state
    )
    if len(y_train) == 0 or len(y_val) == 0:  # pragma: no cover - tiny n
      return float('nan')
    model = make_estimator().fit(x_train, y_train)
    return _rmse(y_val, model.predict(x_val))


class FixedHparamRegressor(base.BaseEstimator, base.RegressorMixin):
  """Fits a concrete estimator class with a fixed ``theta*``.

  Attributes set by ``fit``:
    internal_val_rmse_: The validation RMSE inside the training data
      of this fit (``r_l`` / ``r_m`` of the fold).
    model_: The fitted ``estimator_cls`` instance.
  """

  _estimator_type = 'regressor'

  def __init__(
      self,
      estimator_cls: Any,
      params: Dict[str, Any],
      n_jobs: int = 1,
      random_state: Optional[int] = None,
      val_method: str = VAL_METHOD_CV,
      val_n_splits: int = 5,
      val_split_ratio: float = 0.1,
  ):
    """Stores the constructor arguments unchanged (scikit-learn convention).

    Args:
      estimator_cls: Concrete estimator class of the FLAML winner
        (``automl.model.estimator.__class__``).
      params: ``theta*`` (``automl.model.estimator.get_params()``).
      n_jobs: Threads per fit (``h`` of the parallel plan).
      random_state: Seed of the fits and of the validation split.
      val_method: ``"cv"`` | ``"holdout"`` -- the ``eval_method`` FLAML chose
        for this learner in the global search.
      val_n_splits: ``KFold`` splits under ``cv`` (5, as FLAML).
      val_split_ratio: Held-out share under ``holdout`` (0.1, as FLAML).
    """
    self.estimator_cls = estimator_cls
    self.params = params
    self.n_jobs = n_jobs
    self.random_state = random_state
    self.val_method = val_method
    self.val_n_splits = val_n_splits
    self.val_split_ratio = val_split_ratio

  def _make_estimator(self) -> Any:
    """A fresh unfitted ``estimator_cls(**theta*, n_jobs, random_state)``."""
    return build_estimator(
        self.estimator_cls,
        self.params,
        n_jobs=self.n_jobs,
        random_state=self.random_state,
    )

  def fit(  # pylint: disable=invalid-name
      self, X: Any, y: Any
  ) -> 'FixedHparamRegressor':
    """(1) the internal validation RMSE, (2) the fit with fixed ``theta*``.

    Args:
      X: Training design matrix (the fold's training side).
      y: Training target.

    Returns:
      ``self``.
    """
    x_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=float).ravel()
    # The same quantity FLAML's `best_loss` measured, at the same
    # place: a validation RMSE inside this fold's training data.
    self.internal_val_rmse_ = internal_validation_rmse(
        self._make_estimator,
        x_arr,
        y_arr,
        val_method=self.val_method,
        n_splits=self.val_n_splits,
        split_ratio=self.val_split_ratio,
        random_state=self.random_state,
    )
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      self.model_ = self._make_estimator().fit(x_arr, y_arr)
    self.n_features_in_ = int(x_arr.shape[1]) if x_arr.ndim == 2 else 1
    return self

  def predict(self, X: Any) -> np.ndarray:  # pylint: disable=invalid-name
    """Predicts with the fitted estimator."""
    validation.check_is_fitted(self, 'model_')
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      return np.asarray(self.model_.predict(X), dtype=float).ravel()

  def get_params(self, deep: bool = True) -> Dict[str, Any]:
    """The constructor arguments (``clone`` rebuilds an unfitted copy)."""
    del deep  # No nested estimators.
    return {
        'estimator_cls': self.estimator_cls,
        'params': self.params,
        'n_jobs': self.n_jobs,
        'random_state': self.random_state,
        'val_method': self.val_method,
        'val_n_splits': self.val_n_splits,
        'val_split_ratio': self.val_split_ratio,
    }

  def set_params(self, **params: Any) -> 'FixedHparamRegressor':
    """Sets constructor arguments; unknown names raise ``ValueError``."""
    valid = self.get_params()
    for key, value in params.items():
      if key not in valid:
        raise ValueError(
            f'Invalid parameter {key!r} for FixedHparamRegressor; valid'
            f' parameters are {sorted(valid)}.'
        )
      setattr(self, key, value)
    return self
