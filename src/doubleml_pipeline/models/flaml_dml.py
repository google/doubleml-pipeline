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

"""Module wrapping FLAML estimators to be seamlessly used with DoubleML."""

import logging
import time as _time
from typing import Any, Dict, List, Optional
import warnings

import flaml
import numpy as np
from sklearn import base
from sklearn import multioutput
from sklearn.utils import validation

_LOGGER = logging.getLogger(__name__)

# Constructor arguments of `FlamlRegressorDoubleML` that are NOT forwarded to
# `flaml.AutoML(**kwargs)` (everything else in `kwargs` is).
_OWN_PARAMS = (
    "time",
    "estimator_list",
    "metric",
    "verbose",
    "random_state",
    "early_stop",
    "n_jobs",
)


def flaml_eval_method(automl: Any) -> Optional[str]:
  """Returns the evaluation method FLAML chose (``"cv"`` | ``"holdout"``).

  FLAML logs "Evaluation method: cv|holdout" and
  stores the decision in ``automl._state.eval_method``; the latter is read.

  Args:
    automl: A fitted ``flaml.AutoML``.

  Returns:
    The method, or None when it cannot be determined.
  """
  state = getattr(automl, "_state", None)
  method = getattr(state, "eval_method", None)
  return str(method) if method else None


def flaml_n_trials(automl: Any) -> Optional[int]:
  """Returns ``len(automl.config_history)``, the number of FLAML trials.

  Args:
    automl: A fitted ``flaml.AutoML``.

  Returns:
    The number of recorded trials, or None when unavailable.
  """
  history = getattr(automl, "config_history", None)
  try:
    return int(len(history)) if history is not None else None
  except TypeError:
    return None


class FlamlRegressorDoubleML(base.BaseEstimator, base.RegressorMixin):
  """A wrapper for FLAML regression models to be used as an estimator.

  This wrapper allows FLAML estimators to be seamlessly integrated
  within the DoubleML framework.

  ``early_stop`` and ``n_jobs`` (= ``h`` threads per worker of the parallel
  plan) are forwarded to ``flaml.AutoML.fit``.  Both default to ``None`` =
  "do not pass", so that callers which do not set them (the Phase 0 Stage B
  learner selection, which the parallel plan does not cover) keep issuing
  exactly the same FLAML call as before; the Phases 1-3 runner passes
  ``early_stop=True`` and ``n_jobs=h``.
  """

  _estimator_type = "regressor"

  def __init__(
      self,
      time: int,
      estimator_list: List[str],
      metric: str,
      verbose: int = 0,
      random_state: Optional[int] = None,
      early_stop: Optional[bool] = None,
      n_jobs: Optional[int] = None,
      **kwargs: Any,
  ):
    """Initializes the wrapper (time budget, estimator candidates, metric)."""
    self.time = time
    self.estimator_list = estimator_list
    self.metric = metric
    self.verbose = verbose
    self.random_state = random_state
    self.early_stop = early_stop
    self.n_jobs = n_jobs
    self.kwargs = kwargs

  def fit(self, X: Any, y: Any) -> "FlamlRegressorDoubleML":  # pylint: disable=invalid-name
    """Fits the inner FLAML AutoML model to the training data.

    One FLAML search per call (``hparam_policy = "per_fold"``: one per fold).
    The stop time of the search (``search_elapsed_seconds_``), the number of
    trials (``n_trials_``) and the evaluation method FLAML chose
    (``eval_method_``) are recorded and logged.

    Args:
      X: Training design matrix of shape ``(n, p)``.
      y: Training targets of shape ``(n,)``.

    Returns:
      The fitted wrapper (``self``).
    """
    fit_kwargs: Dict[str, Any] = {}
    # early_stop=True; `max_iter` is never passed (time_budget caps).
    if self.early_stop is not None:
      fit_kwargs["early_stop"] = bool(self.early_stop)
    # n_jobs = h (threads per worker).
    if self.n_jobs is not None:
      fit_kwargs["n_jobs"] = int(self.n_jobs)
    started = _time.monotonic()
    self.auto_ml_ = flaml.AutoML(**self.kwargs)
    self.auto_ml_.fit(
        X_train=X,
        y_train=y,
        task="regression",
        time_budget=self.time,
        estimator_list=self.estimator_list,
        metric=self.metric,
        verbose=self.verbose,
        seed=self.random_state,
        **fit_kwargs,
    )
    self.search_elapsed_seconds_ = _time.monotonic() - started
    self.n_trials_ = flaml_n_trials(self.auto_ml_)
    self.eval_method_ = flaml_eval_method(self.auto_ml_)
    self.tuned_model_ = self.auto_ml_.model.estimator
    _LOGGER.info(
        "FLAML search (per_fold) learner=%s: stopped after %.2fs"
        " (time_budget=%ss, early_stop=%s), trials=%s, eval_method=%s,"
        " best_loss=%s",
        self.estimator_list,
        self.search_elapsed_seconds_,
        self.time,
        fit_kwargs.get("early_stop"),
        self.n_trials_,
        self.eval_method_,
        getattr(self.auto_ml_, "best_loss", None),
    )
    return self

  def predict(self, X: Any) -> Any:  # pylint: disable=invalid-name
    """Predicts outcomes via the internal tuned estimator."""
    validation.check_is_fitted(self, "tuned_model_")
    return self.tuned_model_.predict(X)

  def get_params(self, deep: bool = True) -> Dict[str, Any]:
    """Gets parameters for this estimator wrapper."""
    params = {
        "time": self.time,
        "estimator_list": self.estimator_list,
        "metric": self.metric,
        "verbose": self.verbose,
        "random_state": self.random_state,
        "early_stop": self.early_stop,
        "n_jobs": self.n_jobs,
    }
    params.update(self.kwargs)
    return params

  def set_params(self, **params: Any) -> "FlamlRegressorDoubleML":
    """Sets specific parameter configurations."""
    for key in _OWN_PARAMS:
      if key in params:
        setattr(self, key, params.pop(key))
    self.kwargs.update(params)
    return self


# ---------------------------------------------------------------------------
# Additive extension for the saturation-aware DML Phase 0.  Nothing below is
# reached when the `saturation` config block is omitted.
# ---------------------------------------------------------------------------


def fit_multioutput(template: Any, x_values: Any, y_matrix: Any) -> Any:
  """Fits a *clone* of a tuned estimator on an ``(n, J)`` target matrix.

  Implements the "Multi-Output Regression of the Treatment Basis" step
  (`rloss_theory_and_algorithm_en.md` Step 2): the J B-spline
  basis targets ``b_j(T)`` are all smooth functions of the SAME treatment, so
  they are learned by ONE vector-valued model rather than by J independent
  ones.  A multi-output tree splits to minimise the variance averaged over the
  J outputs; it remains consistent for every ``E[b_j(T)|X]``, so the DML
  orthogonality argument is unaffected.

  Learners without native multi-output support (e.g. LightGBM) fall back to a
  per-output wrapper.  The fallback still avoids a hyperparameter search per
  output - it only refits a fixed configuration - but its fit count degrades
  from 1 to J, so the caller is told which path was taken.

  Args:
    template: A *fitted or unfitted* scikit-learn compatible estimator whose
      hyperparameters are copied.  It is never mutated.
    x_values: Training design matrix.
    y_matrix: Training targets, shape ``(n,)`` or ``(n, J)``.

  Returns:
    An ``(estimator, native)`` pair. ``estimator.predict`` returns ``(n, J)``;
    ``native`` is ``True`` when a single model covers all J outputs.
  """
  targets = np.asarray(y_matrix, dtype=float)
  if targets.ndim == 1:
    targets = targets.reshape(-1, 1)

  if targets.shape[1] > 1:
    try:
      estimator = base.clone(template)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_values, targets)
        probe = np.asarray(estimator.predict(x_values[:1]))
      if probe.ndim == 2 and probe.shape[1] == targets.shape[1]:
        return estimator, True
    except (ValueError, TypeError, IndexError, AttributeError):
      # The learner does not accept a 2-D target; use the fallback below.
      pass

  estimator = multioutput.MultiOutputRegressor(base.clone(template))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    estimator.fit(x_values, targets)
  return estimator, False


class MultiOutputFlamlRegressor(base.BaseEstimator, base.RegressorMixin):
  """Vector-valued FLAML regression of the treatment basis ``E[b(T)|X]``.

  In the T-basis projection the treatment-side nuisance is ONE regression
  with J outputs, not J regressions, so learner selection costs ``N_t`` AutoML
  searches instead of ``N_t * J``.

  FLAML's ``AutoML`` only accepts a 1-D target, so the hyperparameter search is
  run once against a scalar surrogate - the dominant principal component of the
  (standardised) basis targets.  Because every ``b_j(T)`` is a smooth function
  of the same treatment, the J targets are strongly collinear and PC1 is an
  excellent proxy for "how hard is it to predict the treatment basis from X".
  The resulting configuration is then refit on the full ``(n, J)`` matrix.
  """

  _estimator_type = "regressor"

  def __init__(
      self,
      time: int,
      estimator_list: List[str],
      metric: str,
      verbose: int = 0,
      random_state: Optional[int] = None,
      **kwargs: Any,
  ):
    """Initializes the wrapper (same arguments as `FlamlRegressorDoubleML`)."""
    self.time = time
    self.estimator_list = estimator_list
    self.metric = metric
    self.verbose = verbose
    self.random_state = random_state
    self.kwargs = kwargs

  @staticmethod
  def _surrogate(targets: Any) -> Any:
    """Returns the scalar surrogate target (PC1 of the standardised basis)."""
    centered = targets - targets.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = centered / scale
    if standardized.shape[1] == 1:
      return standardized[:, 0]
    try:
      _, _, right = np.linalg.svd(standardized, full_matrices=False)
      return standardized @ right[0]
    except np.linalg.LinAlgError:  # pragma: no cover - numerically extreme
      return standardized.mean(axis=1)

  def fit(self, X: Any, y: Any) -> "MultiOutputFlamlRegressor":  # pylint: disable=invalid-name
    """Runs ONE AutoML search, then refits the winner on all J outputs."""
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
      targets = targets.reshape(-1, 1)

    self.auto_ml_ = flaml.AutoML(**self.kwargs)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.auto_ml_.fit(
          X_train=X,
          y_train=self._surrogate(targets),
          task="regression",
          time_budget=self.time,
          estimator_list=self.estimator_list,
          metric=self.metric,
          verbose=self.verbose,
          seed=self.random_state,
      )
    # The tuned scikit-learn estimator; cloned, never refit in place.
    self.template_ = self.auto_ml_.model.estimator
    self.estimator_, self.native_multioutput_ = fit_multioutput(
        self.template_, X, targets
    )
    self.n_outputs_ = int(targets.shape[1])
    return self

  def predict(self, X: Any) -> Any:  # pylint: disable=invalid-name
    """Predicts all J outputs; always returns an ``(n, J)`` array."""
    validation.check_is_fitted(self, "estimator_")
    preds = np.asarray(self.estimator_.predict(X), dtype=float)
    if preds.ndim == 1:
      preds = preds.reshape(-1, 1)
    return preds

