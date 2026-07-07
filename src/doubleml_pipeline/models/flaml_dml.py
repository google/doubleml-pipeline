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

from typing import Any, Dict, List, Optional
import flaml
from sklearn import base
from sklearn.utils import validation


class FlamlRegressorDoubleML(base.BaseEstimator, base.RegressorMixin):
  """A wrapper for FLAML regression models to be used as an estimator.

  This wrapper allows FLAML estimators to be seamlessly integrated
  within the DoubleML framework.
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
    """Initializes the wrapper indicating the time budget, estimator candidates, and metrics."""
    self.time = time
    self.estimator_list = estimator_list
    self.metric = metric
    self.verbose = verbose
    self.random_state = random_state
    self.kwargs = kwargs

  def fit(self, X: Any, y: Any) -> "FlamlRegressorDoubleML":  # pylint: disable=invalid-name
    """Fits the inner FLAML AutoML model to the training data."""
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
    )
    self.tuned_model_ = self.auto_ml_.model.estimator
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
    }
    params.update(self.kwargs)
    return params

  def set_params(self, **params: Any) -> "FlamlRegressorDoubleML":
    """Sets specific parameter configurations."""
    for key in ["time", "estimator_list", "metric", "verbose", "random_state"]:
      if key in params:
        setattr(self, key, params.pop(key))
    self.kwargs.update(params)
    return self
