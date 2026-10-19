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

"""Global FLAML hyperparameter search (the "global" hyperparameter policy).

Under ``hparam_policy = "global"`` FLAML runs ONCE per (treatment ``d_col``,
learner) at the start of Phase 1, on the sample Phases 1-3 are fitted on
(``D_inf`` or the full sample according to ``final_fit``; the
scaled frame ``df_norm``), with features ``x_cols`` and the target

* ``Y`` for every element of ``ml_models_y`` (the ``ml_l`` side), and
* ``D`` for every element of ``ml_models_t`` (the ``ml_m`` side) -- the very
  column Phases 1-3 regress on: ``fbar(T; alpha_hat)`` when Phase 0 adopted a
  saturation family (``selected_family != None``), ``T`` otherwise.

The result ``theta*`` (``automl.model.estimator.get_params()``) together with
``best_loss``, the number of trials, the wall-clock time and the
``eval_method`` FLAML chose is persisted to
``<output_dir>/step1_exploration_<d_col>/hparams_<d_col>_<learner>.json`` and
turned into the
:class:`~models.fixed_hparam_regressor.FixedHparamRegressor` DoubleML
receives as ``ml_l`` / ``ml_m``.  A learner listed on both sides is searched
twice (once per target); the two results share one JSON file, keyed by side.
"""

import dataclasses
import importlib
import json
import os
import time as _time
from typing import Any, Dict, Iterable, List, Optional
import warnings

import flaml
import numpy as np

from ..models import fixed_hparam_regressor
from ..models import flaml_dml

SIDE_Y = 'Y'  # ml_l: learners of E[Y | X]
SIDE_T = 'T'  # ml_m: learners of E[D | X]
SIDES = (SIDE_Y, SIDE_T)

# The FLAML arguments of the global search.
SEARCH_TASK = 'regression'
SEARCH_METRIC = 'rmse'
SEARCH_EVAL_METHOD = 'auto'  # the same as in the per_fold searches
SEARCH_EARLY_STOP = True
SEARCH_SEED = 1

HPARAMS_JSON_TEMPLATE = 'hparams_{d_col}_{learner}.json'
JSON_SCHEMA_VERSION = 1


def hparams_json_name(d_col: str, learner: str) -> str:
  """``hparams_<d_col>_<learner>.json``."""
  return HPARAMS_JSON_TEMPLATE.format(d_col=d_col, learner=learner)


def class_path(cls: Any) -> str:
  """``module.QualName`` of a class (the JSON form of ``estimator_cls``)."""
  return f'{cls.__module__}.{cls.__qualname__}'


def resolve_class(path: str) -> Any:
  """Imports ``module.QualName`` back into the class object.

  Args:
    path: The string produced by :func:`class_path`.

  Returns:
    The class.

  Raises:
    ImportError: If the module cannot be imported.
    AttributeError: If the class is not found in the module.
  """
  module_name, _, qual_name = path.rpartition('.')
  obj: Any = importlib.import_module(module_name)
  for part in qual_name.split('.'):
    obj = getattr(obj, part)
  return obj


def _jsonable(value: Any) -> Any:
  """Converts numpy scalars / arrays so that ``json.dump`` accepts them."""
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, type):
    return class_path(value)
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  return str(value)


@dataclasses.dataclass
class HparamSearchResult:
  """The outcome of one global search (one (treatment, side, learner)).

  Attributes:
    d_col: Treatment column.
    side: ``"Y"`` (``ml_l``) or ``"T"`` (``ml_m``).
    learner: FLAML estimator name (``"lgbm"``, ``"xgboost"``, ...).
    target_col: The column the search regressed on (``Y``, or the ``D``
      column of Phases 1-3).
    estimator_cls: Concrete estimator class of the winner.
    params: ``theta*`` -- the winner's ``get_params()`` dict.
    best_loss: ``automl.best_loss`` (validation RMSE of ``theta*``).
    n_trials: ``len(automl.config_history)``.
    n_iterations: Total FLAML iterations (``automl._track_iter``), if known.
    elapsed_seconds: Wall-clock time of the search (the stop time,).
    eval_method: ``"cv"`` | ``"holdout"`` as FLAML decided (``auto``).
    n_rows: Rows of the search sample (= the Phases 1-3 sample).
    n_features: ``len(x_cols)``.
    time_budget: ``tau`` (seconds).
    seed: FLAML seed.
    n_jobs: Threads given to FLAML (``h`` of the parallel plan).
    flaml_estimator_class: FLAML wrapper class name (traceability).
  """

  d_col: str
  side: str
  learner: str
  target_col: str
  estimator_cls: Any
  params: Dict[str, Any]
  best_loss: float
  n_trials: Optional[int]
  n_iterations: Optional[int]
  elapsed_seconds: float
  eval_method: str
  n_rows: int
  n_features: int
  time_budget: float
  seed: int
  n_jobs: int
  flaml_estimator_class: str = ''

  @property
  def val_method(self) -> str:
    """The validation method = FLAML's ``eval_method``."""
    eval_method = str(self.eval_method).lower()
    if eval_method == fixed_hparam_regressor.VAL_METHOD_HOLDOUT:
      return fixed_hparam_regressor.VAL_METHOD_HOLDOUT
    return fixed_hparam_regressor.VAL_METHOD_CV

  def make_regressor(
      self, n_jobs: Optional[int] = None, random_state: Optional[int] = None
  ) -> fixed_hparam_regressor.FixedHparamRegressor:
    """The ``FixedHparamRegressor(estimator_cls, theta*, ...)`` of this result.

    Args:
      n_jobs: Threads per fit (``h``); defaults to the search's ``n_jobs``.
      random_state: Seed of the fits / validation split.

    Returns:
      An unfitted ``FixedHparamRegressor``.
    """
    return fixed_hparam_regressor.FixedHparamRegressor(
        estimator_cls=self.estimator_cls,
        params=dict(self.params),
        n_jobs=int(self.n_jobs if n_jobs is None else n_jobs),
        random_state=random_state,
        val_method=self.val_method,
    )

  def to_json_dict(self) -> Dict[str, Any]:
    """The JSON form of this result (``estimator_cls`` as a class path)."""
    data = dataclasses.asdict(self)
    data['estimator_cls'] = class_path(self.estimator_cls)
    return _jsonable(data)

  @classmethod
  def from_json_dict(cls, data: Dict[str, Any]) -> 'HparamSearchResult':
    """Rebuilds a result from :meth:`to_json_dict` (resolving the class)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs['estimator_cls'] = resolve_class(str(data['estimator_cls']))
    kwargs['params'] = dict(data.get('params') or {})
    return cls(**kwargs)


def search_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    learner: str,
    time_budget: float,
    n_jobs: int,
    seed: int = SEARCH_SEED,
    verbose: int = 0,
) -> Dict[str, Any]:
  """One FLAML search with the fixed arguments of the global policy.

  ``task="regression"``, ``estimator_list=[learner]``, ``metric="rmse"``,
  ``time_budget=tau``, ``eval_method="auto"``, ``early_stop=True``,
  ``n_jobs=h``, ``seed``.  ``max_iter`` is never passed.

  Args:
    x: Design matrix (``x_cols`` of the scaled Phases 1-3 sample).
    y: Target (``Y`` or ``D``).
    learner: FLAML estimator name.
    time_budget: ``tau`` in seconds.
    n_jobs: Threads for FLAML / the estimators (``h``).
    seed: FLAML seed.
    verbose: FLAML verbosity.

  Returns:
    ``{'estimator_cls', 'params', 'best_loss', 'n_trials', 'n_iterations',
    'elapsed_seconds', 'eval_method', 'flaml_estimator_class'}``.
  """
  started = _time.monotonic()
  automl = flaml.AutoML()
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    automl.fit(
        X_train=np.asarray(x),
        y_train=np.asarray(y, dtype=float).ravel(),
        task=SEARCH_TASK,
        estimator_list=[learner],
        metric=SEARCH_METRIC,
        time_budget=time_budget,
        eval_method=SEARCH_EVAL_METHOD,
        early_stop=SEARCH_EARLY_STOP,
        n_jobs=int(n_jobs),
        seed=seed,
        verbose=verbose,
    )
  elapsed = _time.monotonic() - started
  estimator = automl.model.estimator
  params = {
      k: v
      for k, v in estimator.get_params().items()
      if k not in fixed_hparam_regressor.DROPPED_PARAM_KEYS
  }
  eval_method = flaml_dml.flaml_eval_method(automl)
  if eval_method is None:
    warnings.warn(
        f"FLAML did not expose its eval_method for learner '{learner}';"
        " assuming 'cv' for the internal validation of FixedHparamRegressor.",
        UserWarning,
    )
    eval_method = fixed_hparam_regressor.VAL_METHOD_CV
  n_iterations = getattr(automl, '_track_iter', None)
  return {
      'estimator_cls': estimator.__class__,
      'params': params,
      'best_loss': float(automl.best_loss),
      'n_trials': flaml_dml.flaml_n_trials(automl),
      'n_iterations': int(n_iterations) if n_iterations is not None else None,
      'elapsed_seconds': float(elapsed),
      'eval_method': str(eval_method),
      'flaml_estimator_class': automl.model.__class__.__name__,
  }


def run_search(
    d_col: str,
    side: str,
    learner: str,
    target_col: str,
    x: np.ndarray,
    y: np.ndarray,
    time_budget: float,
    n_jobs: int,
    seed: int = SEARCH_SEED,
) -> HparamSearchResult:
  """Runs :func:`search_hyperparameters` and packages the result.

  Args:
    d_col: Treatment column.
    side: ``"Y"`` or ``"T"``.
    learner: FLAML estimator name.
    target_col: Name of the target column (for the JSON / the hand-over
      check of the runner).
    x: Design matrix.
    y: Target.
    time_budget: ``tau``.
    n_jobs: ``h``.
    seed: FLAML seed.

  Returns:
    The :class:`HparamSearchResult`.

  Raises:
    ValueError: On an unknown ``side``.
  """
  if side not in SIDES:
    raise ValueError(f'side must be one of {SIDES}, got {side!r}.')
  raw = search_hyperparameters(
      x=x,
      y=y,
      learner=learner,
      time_budget=time_budget,
      n_jobs=n_jobs,
      seed=seed,
  )
  x_arr = np.asarray(x)
  return HparamSearchResult(
      d_col=d_col,
      side=side,
      learner=learner,
      target_col=target_col,
      n_rows=int(x_arr.shape[0]),
      n_features=int(x_arr.shape[1]) if x_arr.ndim == 2 else 1,
      time_budget=float(time_budget),
      seed=int(seed),
      n_jobs=int(n_jobs),
      **raw,
  )


def write_hparams_json(
    path: str,
    results: Iterable[HparamSearchResult],
    context: Optional[Dict[str, Any]] = None,
) -> str:
  """Persists the search results of ONE (treatment, learner).

  Args:
    path: Target file (``hparams_<d_col>_<learner>.json``).
    results: The results of that learner (one per side it was searched on).
    context: Traceability fields (``selected_family``, ``alpha_hat``,
      ``T_ref``, ``final_fit``, sample size, ...).

  Returns:
    ``path``.
  """
  results = list(results)
  payload: Dict[str, Any] = {
      'schema_version': JSON_SCHEMA_VERSION,
      'hparam_policy': 'global',
      'd_col': results[0].d_col if results else None,
      'learner': results[0].learner if results else None,
      'context': _jsonable(context or {}),
      'sides': {r.side: r.to_json_dict() for r in results},
  }
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, default=str)
  return path


def read_hparams_json(path: str) -> Dict[str, HparamSearchResult]:
  """Reads a file written by :func:`write_hparams_json`.

  Args:
    path: The JSON file.

  Returns:
    ``{side: HparamSearchResult}``.
  """
  with open(path, encoding='utf-8') as handle:
    payload = json.load(handle)
  return {
      side: HparamSearchResult.from_json_dict(data)
      for side, data in (payload.get('sides') or {}).items()
  }


def group_by_learner(
    results: Iterable[HparamSearchResult],
) -> Dict[str, List[HparamSearchResult]]:
  """Groups results by learner name (one JSON file per learner)."""
  grouped: Dict[str, List[HparamSearchResult]] = {}
  for result in results:
    grouped.setdefault(result.learner, []).append(result)
  return grouped
