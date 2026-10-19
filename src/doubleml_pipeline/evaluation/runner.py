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

"""Core execution pipeline logic for DoubleML PLR model construction."""

import copy
import dataclasses
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import patsy

from ..models import flaml_dml
from ..preprocessing import scalers
from . import hparam_search
from . import metrics

# Column added to `df_1a` ONLY when held-out rows were scored (Phase 0 split
# with `final_fit: d_inf`): which rows the DML model was fitted on and which
# were scored with the fitted CATE afterwards.
ESTIMATION_SAMPLE_COL = 'estimation_sample'
FITTED_SAMPLE_LABEL = 'D_inf'
SCORED_SAMPLE_LABEL = 'D_sel'

# The two hyperparameter policies of Phases 1-3.
HPARAM_POLICY_GLOBAL = 'global'
HPARAM_POLICY_PER_FOLD = 'per_fold'
HPARAM_POLICIES = (HPARAM_POLICY_GLOBAL, HPARAM_POLICY_PER_FOLD)


@dataclasses.dataclass(frozen=True)
class CateStructure:
  """The resolved ``cate_structure`` of one treatment."""

  cate_type: str
  cate_df: Any
  cate_degree: Any
  cate_intercept: Any


def resolve_cate_structure(
    cate_structure: Union[str, Dict[str, Any], None],
) -> CateStructure:
  """Resolves the ``cate_structure`` argument into its four fields.

  This is the (unchanged) rule of ``DoubleMLPipeline.__init__``, factored out
  so that ``prepare_run_inputs`` builds the CATE basis once per treatment
  with exactly the same settings as ``run``.

  Args:
    cate_structure: ``'auto_additive'``, a patsy formula, or a dict with
      ``type`` / ``df`` / ``degree`` / ``include_intercept``.

  Returns:
    The resolved ``CateStructure``.
  """
  if isinstance(cate_structure, dict):
    return CateStructure(
        cate_type=cate_structure.get('type', 'auto_additive'),
        cate_df=cate_structure.get('df', 3),
        cate_degree=cate_structure.get('degree', 2),
        cate_intercept=cate_structure.get('include_intercept', False),
    )
  cate_type = str(cate_structure) if cate_structure else 'auto_additive'
  if cate_type == 'auto_additive':
    return CateStructure(cate_type, 3, 2, False)
  return CateStructure(cate_type, 'custom', 'custom', 'custom')


@dataclasses.dataclass
class PreparedInputs:
  """Per-treatment inputs of ``run``, computed ONCE and reused by every run.

  Attributes:
    df_norm: ``scaler.fit_transform(df_orig)`` -- the scaled fitting frame.
    df_norm_score: ``scaler.transform(df_score)`` (held-out rows), or None.
    cate_basis: The CATE basis ``p(X)`` on the fitting rows.
    cate_basis_score: The same design evaluated on the held-out rows, or
      None.
    n_rows: ``len(df_orig)`` (hand-over check of the global search sample).
  """

  df_norm: pd.DataFrame
  df_norm_score: Optional[pd.DataFrame]
  cate_basis: pd.DataFrame
  cate_basis_score: Optional[pd.DataFrame]
  n_rows: int


def prepare_run_inputs(
    df_orig: pd.DataFrame,
    scaler: scalers.CausalDataScaler,
    df_score: Optional[pd.DataFrame],
    covariates_for_cate: Optional[List[str]],
    cate_structure: Union[str, Dict[str, Any], None],
) -> PreparedInputs:
  """Fits the scaler and builds the CATE basis once per treatment .

  ``run()`` used to call ``scaler.fit_transform(df_orig)`` and the patsy
  spline construction in every Phase 1 task and every Phase 3 repetition;
  both depend only on the treatment's frame and CATE settings, so they are
  computed here once and handed to ``run(prepared=...)``.  The scaler is
  fitted in place (as ``run()`` did) so that ``inverse_transform_column`` and
  the raw-population lookup see the same fitted state.

  Args:
    df_orig: The rows the DML model is FITTED on.
    scaler: The pipeline scaler; fitted here on ``df_orig``.
    df_score: Held-out rows scored with the fitted CATE, or None.
    covariates_for_cate: CATE covariates of the treatment.
    cate_structure: ``cate_structure`` of the treatment.

  Returns:
    The ``PreparedInputs``.
  """
  structure = resolve_cate_structure(cate_structure)
  df_norm = scaler.fit_transform(df_orig)
  has_score = df_score is not None and len(df_score) > 0
  df_norm_score = scaler.transform(df_score) if has_score else None
  basis, basis_score = build_cate_basis_pair(
      df_norm,
      df_orig,
      df_norm_score,
      covariates_for_cate=covariates_for_cate or [],
      cate_type=structure.cate_type,
      cate_df=structure.cate_df,
      cate_degree=structure.cate_degree,
      cate_intercept=structure.cate_intercept,
  )
  return PreparedInputs(
      df_norm=df_norm,
      df_norm_score=df_norm_score,
      cate_basis=basis,
      cate_basis_score=basis_score,
      n_rows=int(len(df_orig)),
  )


def fold_validation_loss(fold_model: Any) -> Optional[float]:
  """The validation RMSE of one fold learner (``r_l`` / ``r_m`` input).

  Under ``hparam_policy = "global"`` the fold learner is a
  ``FixedHparamRegressor`` and the value is its ``internal_val_rmse_``
  (validation RMSE of ``theta*`` inside the fold's training data); under
  ``per_fold`` it is FLAML's ``auto_ml_.best_loss`` exactly as before.

  Args:
    fold_model: A stored fold learner of ``dml_model.models``.

  Returns:
    The loss, or None when the learner exposes neither quantity.
  """
  internal = getattr(fold_model, 'internal_val_rmse_', None)
  if internal is not None:
    return float(internal)
  aml = getattr(fold_model, 'auto_ml_', getattr(fold_model, 'auto_ml', None))
  if aml and hasattr(aml, 'best_loss'):
    return aml.best_loss
  return None


def _check_search_target(
    side: str, result: hparam_search.HparamSearchResult, expected: str
) -> None:
  """Checks that the global search regressed on the column this run uses."""
  if result.target_col != expected:
    raise ValueError(
        f'Global hyperparameters of the {side} side were searched on target'
        f" '{result.target_col}', but this run regresses on '{expected}'"
        ' (hand-over mismatch).'
    )


def _check_search_sample(
    result: hparam_search.HparamSearchResult, n_rows: int
) -> None:
  """Checks that the global search sample is the Phases 1-3 fit sample."""
  if result.n_rows != n_rows:
    raise ValueError(
        f"Global hyperparameters of learner '{result.learner}' were searched"
        f' on {result.n_rows} rows but the fit sample has {n_rows} rows'
        ' (hand-over mismatch).'
    )


class DoubleMLPipeline:
  """Configures and executes Partial Linear Regression (PLR) estimation."""

  def __init__(
      self,
      y_col: str,
      d_col: str,
      x_cols: List[str],
      ml_y_model: str,
      ml_t_model: str,
      n_folds: int = 5,
      n_rep: int = 1,
      time_budget: int = 15,
      cate_structure: Union[str, Dict[str, Any]] = 'auto_additive',
      covariates_for_cate: Optional[List[str]] = None,
      ground_truth_col: Optional[str] = None,
      cluster_cols: Optional[List[str]] = None,
      random_state: Optional[int] = None,
      n_jobs: Optional[int] = None,
      early_stop: Optional[bool] = True,
      hparams_y: Optional[hparam_search.HparamSearchResult] = None,
      hparams_t: Optional[hparam_search.HparamSearchResult] = None,
  ):
    """Initializes the DoubleML Pipeline defining structural causal bounds.

    Args:
      y_col: Outcome column ``Y``.
      d_col: Treatment column ``D`` (``fbar(T; alpha_hat)`` or ``T``).
      x_cols: Nuisance covariates ``X``.
      ml_y_model: FLAML learner name of the Y side (``ml_l``).
      ml_t_model: FLAML learner name of the T side (``ml_m``).
      n_folds: ``K`` folds of the cross-fitting.
      n_rep: DoubleML repetitions inside one run (1).
      time_budget: ``tau`` -- the FLAML search budget (``per_fold``).
      cate_structure: CATE structure (see ``resolve_cate_structure``).
      covariates_for_cate: CATE covariates.
      ground_truth_col: Optional ground-truth effect column.
      cluster_cols: Cluster DML keys.
      random_state: Seed of the run (sample splitting, FLAML, fixed fits).
      n_jobs: Threads per learner fit -- ``h`` of the parallel plan, passed
        to FLAML (``n_jobs=h``) and to ``FixedHparamRegressor(n_jobs=h)``.
        None leaves the library defaults.
      early_stop: FLAML ``early_stop`` of the ``per_fold`` searches (True).
        None = do not pass.
      hparams_y: The global search result of the Y-side
        learner; when given (together with ``hparams_t``) the nuisance
        learners are ``FixedHparamRegressor`` instances with its ``theta*``
        and no FLAML search runs inside the folds.
      hparams_t: The global search result of the T-side learner.

    Raises:
      ValueError: If only one of ``hparams_y`` / ``hparams_t`` is given, or
        if a given result was searched on another target column.
    """
    self.y_col = y_col
    self.d_col = d_col
    self.x_cols = x_cols
    self.n_folds = n_folds
    self.n_rep = n_rep
    self.time_budget = time_budget
    self.covariates_for_cate = covariates_for_cate or []
    self.ground_truth_col = ground_truth_col
    self.cluster_cols = cluster_cols or []
    self.random_state = random_state
    self.n_jobs = n_jobs
    self.early_stop = early_stop

    structure = resolve_cate_structure(cate_structure)
    self.cate_type = structure.cate_type
    self.cate_df = structure.cate_df
    self.cate_degree = structure.cate_degree
    self.cate_intercept = structure.cate_intercept

    self.ml_y_name = ml_y_model
    self.ml_t_name = ml_t_model
    if (hparams_y is None) != (hparams_t is None):
      raise ValueError(
          'hparams_y and hparams_t must be given together (hparam_policy ='
          ' "global") or both omitted ("per_fold").'
      )
    self.hparams_y = hparams_y
    self.hparams_t = hparams_t
    if hparams_y is not None and hparams_t is not None:
      # Hand-over check: the search target of the T side must be THE D column
      # Phases 1-3 regress on (fbar(T; alpha_hat) or T), and the Y side Y.
      _check_search_target(hparam_search.SIDE_Y, hparams_y, y_col)
      _check_search_target(hparam_search.SIDE_T, hparams_t, d_col)
      self.hparam_policy = HPARAM_POLICY_GLOBAL
      # The same theta* in every fold of every run; the learner
      # is cloned per fold by DoubleML and fitted on that fold's training
      # side exactly as the FLAML wrapper was.
      self.ml_y = hparams_y.make_regressor(
          n_jobs=n_jobs, random_state=self.random_state
      )
      self.ml_t = hparams_t.make_regressor(
          n_jobs=n_jobs, random_state=self.random_state
      )
    else:
      self.hparam_policy = HPARAM_POLICY_PER_FOLD
      self.ml_y = flaml_dml.FlamlRegressorDoubleML(
          time=time_budget,
          estimator_list=[ml_y_model],
          metric='rmse',
          random_state=self.random_state,
          early_stop=early_stop,
          n_jobs=n_jobs,
      )
      self.ml_t = flaml_dml.FlamlRegressorDoubleML(
          time=time_budget,
          estimator_list=[ml_t_model],
          metric='rmse',
          random_state=self.random_state,
          early_stop=early_stop,
          n_jobs=n_jobs,
      )

  def _build_cate_basis(
      self, df_norm: pd.DataFrame, df_orig: pd.DataFrame
  ) -> pd.DataFrame:
    """Generates spline or linear basis arrays for CATE formulas."""
    # Delegates to the module-level builder so that Phase 0 and Phases 1-3
    # construct p(X) through the very same function.
    return build_cate_basis(
        df_norm,
        df_orig,
        covariates_for_cate=self.covariates_for_cate,
        cate_type=self.cate_type,
        cate_df=self.cate_df,
        cate_degree=self.cate_degree,
        cate_intercept=self.cate_intercept,
    )

  def run(
      self,
      df_orig: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      rep_id: int = 1,
      df_score: Optional[pd.DataFrame] = None,
      row_order: Optional[pd.Index] = None,
      prepared: Optional[PreparedInputs] = None,
  ) -> Dict[str, Any]:
    """Runs the DoubleML causal execution over normalized variables.

    Args:
      df_orig: The rows the DML model is FITTED on (Phase 0's ``D_inf`` when a
        saturation split is active, otherwise the whole sample).
      scaler: The pipeline scaler; fitted here on ``df_orig`` unless
        ``prepared`` carries the already scaled frames.
      rep_id: Repetition id (seed / model-name suffix).
      df_score: Optional extra rows (Phase 0's ``D_sel``) that take no part in
        the fit but are SCORED with the fitted model: the CATE ``theta(X)`` and
        its confidence band are evaluated on them, the nuisance predictions
        come from the stored fold models (none of which saw these rows) and
        the incremental KPI is derived exactly as for the fitted rows.  They
        are appended to ``df_1a`` with ``estimation_sample = "D_sel"``
        (fitted rows carry ``"D_inf"``), so every downstream table and chart
        covers all data points.  Nothing is added when ``None``.
      row_order: Optional index giving the row order of the returned
        ``df_1a`` (the original order of the full sample).  The pipeline
        aggregates repetitions positionally, so all repetitions must return
        the same order.
      prepared: The scaled frames and the CATE basis of
        this treatment computed once by ``prepare_run_inputs`` (``scaler``
        must be the instance fitted there).  When None they are computed
        here, exactly as before.

    Returns:
      ``{'model_name', 'df_1a', 'metrics_1c', 'model_obj', 'cate_coef_dict'}``.
    """
    if len(df_orig) < 1000 and self.covariates_for_cate:
      warnings.warn(
          'Sample size is small (n < 1000). Set degree of continuous covariates'
          ' to 2 or less.',
          UserWarning,
      )

    model_name = (
        f'{self.ml_y_name}_{self.ml_t_name}_{self.n_folds}_{rep_id}'
        f'_{self.time_budget}_{len(self.covariates_for_cate)}'
    )

    # Methodological Rationale: Global pre-scaling is mathematically essential
    # to maintain "Causal Coordinate Invariance".
    #
    # Academic Citation: Chernozhukov et al. (2018), Double/debiased machine
    # learning for treatment and structural parameters, The Econometrics
    # Journal.
    #
    # Context on Leakage: While predictive ML classifies global pre-scaling as
    # distributional leakage, causal DoubleML requires this global coordinate
    # system.
    #
    # Avoiding Estimation Bias: If scaling parameters ($\mu_k, \sigma_k$) varied
    # independently across folds, pooling cross-fitted residuals to run the
    # final CATE regression would mix different coordinate ratios, introducing
    # severe estimation bias into the physical dollar treatment parameters
    # ($\theta_0$).
    has_score = df_score is not None and len(df_score) > 0
    if prepared is not None:
      # df_norm computed once per treatment; copied because the
      # prediction / KPI columns are added to it below.
      if len(prepared.df_norm) != len(df_orig) or not (
          prepared.df_norm.index.equals(df_orig.index)
      ):
        raise ValueError(
            'prepared.df_norm does not match df_orig (rows / index); the'
            ' precomputed inputs belong to another sample.'
        )
      if self.hparam_policy == HPARAM_POLICY_GLOBAL:
        # Hand-over check: the global search ran on THIS sample.
        for result in (self.hparams_y, self.hparams_t):
          if result is not None:
            _check_search_sample(result, len(df_orig))
      df_norm = prepared.df_norm.copy()
      df_norm_score = (
          prepared.df_norm_score.copy()
          if has_score and prepared.df_norm_score is not None
          else None
      )
    else:
      df_norm = scaler.fit_transform(df_orig)
      # Held-out rows are transformed with the SAME fitted parameters (never
      # refitted), so they live in the coordinate system of the fitted model.
      df_norm_score = scaler.transform(df_score) if has_score else None

    # FIX: Use DoubleMLClusterData with cluster_cols for robust inference
    missing_clusters = [
        c for c in self.cluster_cols if c not in df_norm.columns
    ]
    if missing_clusters:
      raise ValueError(
          f'Cluster columns not found in dataframe: {missing_clusters}'
      )
    v_clusters = self.cluster_cols
    if v_clusters:
      dml_data = dml.DoubleMLClusterData(
          df_norm,
          y_col=self.y_col,
          d_cols=self.d_col,
          x_cols=self.x_cols,
          cluster_cols=v_clusters,
      )
    else:
      dml_data = dml.DoubleMLData(
          df_norm, y_col=self.y_col, d_cols=self.d_col, x_cols=self.x_cols
      )

    r_state = self.random_state if self.random_state is not None else rep_id
    np.random.seed(r_state)

    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=UserWarning, module='doubleml')
      if isinstance(dml_data, dml.DoubleMLClusterData):
        dml_model = dml.DoubleMLPLR(
            dml_data,
            ml_l=self.ml_y,
            ml_m=self.ml_t,
            n_folds=self.n_folds,
            n_rep=self.n_rep,
        )
      else:
        dml_model = dml.DoubleMLPLR(
            dml_data,
            ml_l=self.ml_y,
            ml_m=self.ml_t,
            n_folds=self.n_folds,
            n_rep=self.n_rep,
        )
      dml_model.fit(store_models=True)

      if prepared is not None and prepared.cate_basis is not None:
        # The CATE basis p(X) built once per treatment.
        spline_basis = prepared.cate_basis
        score_basis = prepared.cate_basis_score if has_score else None
      else:
        spline_basis, score_basis = build_cate_basis_pair(
            df_norm,
            df_orig,
            df_norm_score,
            covariates_for_cate=self.covariates_for_cate,
            cate_type=self.cate_type,
            cate_df=self.cate_df,
            cate_degree=self.cate_degree,
            cate_intercept=self.cate_intercept,
        )
      cate_obj = dml_model.cate(basis=spline_basis)
      conf_int = cate_obj.confint(basis=spline_basis)
      # The fitted BLP evaluated on the held-out rows: p(x)'beta_hat with its
      # pointwise band, on the identical basis columns.
      conf_int_score = (
          cate_obj.confint(basis=score_basis) if has_score else None
      )

    y_pred_norm = (
        np.mean(dml_model.predictions['ml_l'], axis=1)
        if self.n_rep > 1
        else dml_model.predictions['ml_l'].reshape(-1)
    )
    t_pred_norm = (
        np.mean(dml_model.predictions['ml_m'], axis=1)
        if self.n_rep > 1
        else dml_model.predictions['ml_m'].reshape(-1)
    )

    df_results = df_orig.copy()
    df_norm['Y_pred_norm'] = y_pred_norm
    df_norm['T_pred_norm'] = t_pred_norm

    df_results['Y_pred'] = scaler.inverse_transform_column(
        df_norm, 'Y_pred_norm', param_source_col=self.y_col
    )
    df_results['T_pred'] = scaler.inverse_transform_column(
        df_norm, 'T_pred_norm', param_source_col=self.d_col
    )
    df_results['Y_res'] = df_results[self.y_col] - df_results['Y_pred']
    df_results['T_res'] = df_results[self.d_col] - df_results['T_pred']

    zero_norm = 0.0
    if self.d_col in scaler.standard_params_:
      p_val = scaler.standard_params_[self.d_col]
      zero_norm = -p_val['mean'] / p_val['scale']
    elif self.d_col in scaler.min_max_params_:
      p_val = scaler.min_max_params_[self.d_col]
      diff = p_val['data_max'] - p_val['data_min']
      zero_norm = -p_val['data_min'] / diff if diff != 0 else 0.0
    elif self.d_col in scaler.pop_standard_params_:
      p_val = scaler.pop_standard_params_[self.d_col]
      zero_norm = -p_val['mean'] / p_val['scale']

    # FIX: Append .values to bypass Pandas index alignment and force strict
    # row-by-row positional matching
    df_norm['estimated_sales'] = (df_norm[self.d_col] - zero_norm) * conf_int[
        'effect'
    ].values
    df_norm['estimated_sales_2.5%'] = (
        df_norm[self.d_col] - zero_norm
    ) * conf_int['2.5 %'].values
    df_norm['estimated_sales_97.5%'] = (
        df_norm[self.d_col] - zero_norm
    ) * conf_int['97.5 %'].values

    kpi_col = f'estimated_incremental_KPI_{self.d_col}'
    kpi_lower = f'estimated_incremental_KPI_{self.d_col}_2.5%'
    kpi_upper = f'estimated_incremental_KPI_{self.d_col}_97.5%'

    df_results[kpi_col] = scaler.inverse_transform_column(
        df_norm,
        'estimated_sales',
        param_source_col=self.y_col,
        is_difference=True,
    )
    df_results[kpi_lower] = scaler.inverse_transform_column(
        df_norm,
        'estimated_sales_2.5%',
        param_source_col=self.y_col,
        is_difference=True,
    )
    df_results[kpi_upper] = scaler.inverse_transform_column(
        df_norm,
        'estimated_sales_97.5%',
        param_source_col=self.y_col,
        is_difference=True,
    )

    n_fitted = int(len(df_results))
    n_scored = 0
    if has_score:
      df_results[ESTIMATION_SAMPLE_COL] = FITTED_SAMPLE_LABEL
      df_scored = self._score_held_out_rows(
          dml_model,
          scaler,
          df_score,
          df_norm_score,
          conf_int_score,
          zero_norm,
          kpi_col,
          kpi_lower,
          kpi_upper,
      )
      n_scored = int(len(df_scored))
      df_results = pd.concat([df_results, df_scored], axis=0)
      if row_order is not None:
        df_results = df_results.loc[row_order]

    total_input = df_results[self.d_col].sum()
    total_kpi = df_results[kpi_col].sum()
    total_sales = df_results[self.y_col].sum()

    if self.ground_truth_col and self.ground_truth_col in df_results.columns:
      gt_metrics = metrics.calculate_ground_truth_metrics(
          df_results[self.ground_truth_col], df_results[kpi_col], total_input
      )
    else:
      gt_metrics = {
          'true_total_roi': np.nan,
          'SMAPE': np.nan,
          'R_squared': np.nan,
          'MAE': np.nan,
          'RMSE': np.nan,
      }

    y_losses, t_losses = [], []
    try:
      # `fold_validation_loss` reads `internal_val_rmse_`
      # of a FixedHparamRegressor (global) or FLAML's `best_loss` (per_fold);
      # the fold averaging below is unchanged.
      for r_models in dml_model.models['ml_l'][self.d_col]:
        for fold_m in r_models:
          loss = fold_validation_loss(fold_m)
          if loss is not None:
            y_losses.append(loss)

      for r_models in dml_model.models['ml_m'][self.d_col]:
        for fold_m in r_models:
          loss = fold_validation_loss(fold_m)
          if loss is not None:
            t_losses.append(loss)
    except (KeyError, TypeError):
      pass

    y_loss = float(np.mean(y_losses)) if y_losses else np.nan
    t_loss = float(np.mean(t_losses)) if t_losses else np.nan

    cate_coef_dict = {}
    try:
      cate_summary = cate_obj.summary
      for idx, row in cate_summary.iterrows():
        cate_coef_dict[f'{idx}_coef'] = float(row['coef'])
        cate_coef_dict[f'{idx}_pvalue'] = float(row.get('P>|t|', np.nan))
        cate_coef_dict[f'{idx}_stderr'] = float(row.get('std err', np.nan))
    except (AttributeError, KeyError):
      pass

    metrics_1c = {
        'model': model_name,
        # `model_name` joins its 6 fields with '_', but FLAML learner names may
        # themselves contain '_' (e.g. 'extra_tree', 'xgb_limitdepth'), so it
        # cannot be split back apart positionally. Carry the two names through
        # as their own fields instead of making callers re-parse the string.
        'ml_y_name': self.ml_y_name,
        'ml_t_name': self.ml_t_name,
        # The CONFIGURED fold count. Distinct from 'number_of_folds' below,
        # which is DoubleML's EFFECTIVE count: for cluster data it computes
        # `n_folds ** n_cluster_vars`, so 2-way clustering with n_folds=3
        # reports 9. Callers that need to reconstruct the requested setting
        # (Phase 2 -> Phase 3) must use this one.
        'n_folds': self.n_folds,
        'd_col': self.d_col,
        'x_col': str(self.x_cols),
        'y_col': self.y_col,
        'covariates_for_cate': str(self.covariates_for_cate),
        'cate_structure': self.cate_type,
        'cate_df': self.cate_df,
        'cate_degree': self.cate_degree,
        'cate_intercept': self.cate_intercept,
        'total_input': total_input,
        'total_estimated_incremental_KPI': total_kpi,
        'total_estimated_roi': (
            total_kpi / total_input if total_input > 0 else 0
        ),
        **gt_metrics,
        'rep_id': rep_id,
        'rep_total_input': total_input,
        'rep_total_kpi': total_kpi,
        'rep_total_sales': total_sales,
        'nuisance_models_rmse_Y_model_evaluated_by_AutoML': y_loss,
        'nuisance_models_rmse_T_model_evaluated_by_AutoML': t_loss,
        'number_of_observations': dml_model.n_obs,
        'number_of_folds': dml_model.n_folds,
        'number_of_repetitions': dml_model.n_rep,
    }
    if has_score:
      # Only present when held-out rows were scored, so legacy outputs are
      # byte-identical.  The totals above cover fitted + scored rows.
      metrics_1c['number_of_fitted_rows'] = n_fitted
      metrics_1c['number_of_scored_rows'] = n_scored

    return {
        'model_name': model_name,
        'df_1a': df_results,
        'metrics_1c': metrics_1c,
        'model_obj': dml_model,
        'cate_coef_dict': cate_coef_dict,
    }

  def _score_held_out_rows(
      self,
      dml_model: Any,
      scaler: scalers.CausalDataScaler,
      df_score: pd.DataFrame,
      df_norm_score: pd.DataFrame,
      conf_int_score: pd.DataFrame,
      zero_norm: float,
      kpi_col: str,
      kpi_lower: str,
      kpi_upper: str,
  ) -> pd.DataFrame:
    """Builds the ``df_1a`` rows of the held-out (scored) sample.

    Mirrors the fitted-row arithmetic of :meth:`run` one-to-one:

    * ``effect`` / CI come from the fitted CATE evaluated on the new rows;
    * ``Y_pred`` / ``T_pred`` are the mean prediction of the stored fold
      models (every fold model was trained without these rows, so this is an
      honest out-of-sample prediction);
    * ``estimated_incremental_KPI`` is ``(T_norm - zero_norm) * effect``
      inverse-transformed with the fitted scaler parameters.

    Args:
      dml_model: The fitted ``DoubleMLPLR``.
      scaler: The scaler fitted on the fitting rows.
      df_score: Raw held-out rows.
      df_norm_score: ``df_score`` transformed with the fitted scaler.
      conf_int_score: ``cate_obj.confint(basis=score_basis)``.
      zero_norm: Normalised value of ``T = 0`` (see :meth:`run`).
      kpi_col: Name of the incremental-KPI column.
      kpi_lower: Name of its lower CI column.
      kpi_upper: Name of its upper CI column.

    Returns:
      A frame indexed like ``df_score`` with the same columns as the fitted
      rows of ``df_1a`` and ``estimation_sample = "D_sel"``.
    """
    df_out = df_score.copy()
    df_norm_s = df_norm_score.copy()

    # Population-based inverse transforms look the raw population up by index
    # on the FITTING rows; extend that lookup with the held-out rows.
    scaler_s = scaler
    raw_pop = getattr(scaler, 'raw_population_', None)
    pop_col = getattr(scaler, 'population_col', None)
    if raw_pop is not None and pop_col in df_score.columns:
      scaler_s = copy.copy(scaler)
      extra_pop = df_score[pop_col]
      scaler_s.raw_population_ = pd.concat(
          [raw_pop, extra_pop.loc[~extra_pop.index.isin(raw_pop.index)]]
      )

    x_score = df_norm_s[self.x_cols].to_numpy(dtype=float)
    df_norm_s['Y_pred_norm'] = _ensemble_predict(
        dml_model.models, 'ml_l', self.d_col, x_score
    )
    df_norm_s['T_pred_norm'] = _ensemble_predict(
        dml_model.models, 'ml_m', self.d_col, x_score
    )
    df_out['Y_pred'] = scaler_s.inverse_transform_column(
        df_norm_s, 'Y_pred_norm', param_source_col=self.y_col
    )
    df_out['T_pred'] = scaler_s.inverse_transform_column(
        df_norm_s, 'T_pred_norm', param_source_col=self.d_col
    )
    df_out['Y_res'] = df_out[self.y_col] - df_out['Y_pred']
    df_out['T_res'] = df_out[self.d_col] - df_out['T_pred']

    t_centered = df_norm_s[self.d_col] - zero_norm
    df_norm_s['estimated_sales'] = t_centered * conf_int_score['effect'].values
    df_norm_s['estimated_sales_2.5%'] = (
        t_centered * conf_int_score['2.5 %'].values
    )
    df_norm_s['estimated_sales_97.5%'] = (
        t_centered * conf_int_score['97.5 %'].values
    )
    df_out[kpi_col] = scaler_s.inverse_transform_column(
        df_norm_s, 'estimated_sales', param_source_col=self.y_col,
        is_difference=True,
    )
    df_out[kpi_lower] = scaler_s.inverse_transform_column(
        df_norm_s, 'estimated_sales_2.5%', param_source_col=self.y_col,
        is_difference=True,
    )
    df_out[kpi_upper] = scaler_s.inverse_transform_column(
        df_norm_s, 'estimated_sales_97.5%', param_source_col=self.y_col,
        is_difference=True,
    )
    df_out[ESTIMATION_SAMPLE_COL] = SCORED_SAMPLE_LABEL
    return df_out


def _ensemble_predict(
    models: Any, learner: str, d_col: str, x_new: np.ndarray
) -> np.ndarray:
  """Mean prediction of all stored (repetition, fold) nuisance models.

  Args:
    models: ``dml_model.models`` (``fit(store_models=True)``).
    learner: ``'ml_l'`` or ``'ml_m'``.
    d_col: Treatment column (key of the nested model dict).
    x_new: Design matrix of the rows to predict.

  Returns:
    The averaged predictions, or NaNs when the stored models are unavailable.
  """
  preds = []
  try:
    for r_models in models[learner][d_col]:
      for fold_m in r_models:
        if fold_m is None:
          continue
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          preds.append(np.asarray(fold_m.predict(x_new), dtype=float).ravel())
  except (KeyError, TypeError, AttributeError, ValueError) as exc:
    warnings.warn(
        f'Nuisance predictions for the scored rows are unavailable ({exc});'
        ' Y_pred / T_pred of those rows are NaN.',
        UserWarning,
    )
    preds = []
  if not preds:
    return np.full(len(x_new), np.nan)
  return np.mean(np.vstack(preds), axis=0)


def build_cate_basis(
    df_norm: pd.DataFrame,
    df_orig: pd.DataFrame,
    covariates_for_cate: Optional[List[str]],
    cate_type: str = 'auto_additive',
    cate_df: Any = 3,
    cate_degree: Any = 2,
    cate_intercept: Any = False,
) -> pd.DataFrame:
  """Builds the CATE basis p(X) shared by Phase 0 and Phases 1-3.

  This is THE single construction of
  ``p(X) = (1, B^(1)_1(X^(1)), ..., B^(q)_{d_q}(X^(q)))`` from
  ``covariates_for_cate`` / ``cate_structure``.  Phase 0 (alpha selection) and
  Phases 1-3 (final estimation) both call it, so the ``theta`` expressible at
  alpha-selection time coincides with the one of the final model.  The body is
  the unchanged legacy ``DoubleMLPipeline._build_cate_basis``.

  Args:
    df_norm: Normalised frame (the basis is evaluated on it).
    df_orig: Raw frame (used to detect binary covariates).
    covariates_for_cate: CATE covariates; empty/None gives intercept only.
    cate_type: ``'auto_additive'`` or a patsy formula.
    cate_df: B-spline df (auto_additive).
    cate_degree: B-spline degree (auto_additive).
    cate_intercept: ``include_intercept`` of patsy ``bs`` (auto_additive).

  Returns:
    The ``(n, d)`` design matrix as a DataFrame indexed like ``df_norm``.
  """
  basis, _ = build_cate_basis_pair(
      df_norm,
      df_orig,
      None,
      covariates_for_cate=covariates_for_cate,
      cate_type=cate_type,
      cate_df=cate_df,
      cate_degree=cate_degree,
      cate_intercept=cate_intercept,
  )
  return basis


def _clip_to_fitted_range(
    series_fit: pd.Series, series_new: pd.Series
) -> pd.Series:
  """Clips numeric ``series_new`` to ``[min, max]`` of ``series_fit``.

  patsy's stateful ``bs()`` transform refuses values outside the outermost
  knots it memorised on the fitting data, so held-out rows are evaluated at
  the boundary of the fitted support instead of being extrapolated.
  Non-numeric columns are returned unchanged.

  Args:
    series_fit: Column of the fitting frame whose observed ``[min, max]``
      defines the clipping bounds.
    series_new: Column of the rows to score, clipped to the range of
      ``series_fit``.

  Returns:
    ``series_new`` clipped to ``[min(series_fit), max(series_fit)]``, or
    ``series_new`` unchanged when it is non-numeric or the bounds are NaN.
  """
  if not pd.api.types.is_numeric_dtype(series_new):
    return series_new
  lo = pd.to_numeric(series_fit, errors='coerce').min()
  hi = pd.to_numeric(series_fit, errors='coerce').max()
  if pd.isna(lo) or pd.isna(hi):
    return series_new
  return series_new.clip(lower=lo, upper=hi)


def build_cate_basis_pair(
    df_norm: pd.DataFrame,
    df_orig: pd.DataFrame,
    df_norm_score: Optional[pd.DataFrame],
    covariates_for_cate: Optional[List[str]],
    cate_type: str = 'auto_additive',
    cate_df: Any = 3,
    cate_degree: Any = 2,
    cate_intercept: Any = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
  """Builds p(X) on the fitting rows and evaluates the SAME design on extras.

  The fitting design is constructed exactly as :func:`build_cate_basis` does.
  When ``df_norm_score`` is given, the design matrix of those rows is produced
  with patsy's ``build_design_matrices`` from the fitted ``design_info``, so
  the B-spline knots (and the column order DoubleML's ``confint(basis=...)``
  insists on) are those of the fitting rows.  Numeric covariates of the extra
  rows are clipped to the fitted range (see :func:`_clip_to_fitted_range`).

  Args:
    df_norm: Normalised fitting frame.
    df_orig: Raw fitting frame (binary-covariate detection).
    df_norm_score: Normalised frame of the rows to score, or ``None``.
    covariates_for_cate: CATE covariates; empty/None gives intercept only.
    cate_type: ``'auto_additive'`` or a patsy formula.
    cate_df: B-spline df (auto_additive).
    cate_degree: B-spline degree (auto_additive).
    cate_intercept: ``include_intercept`` of patsy ``bs`` (auto_additive).

  Returns:
    ``(basis_fit, basis_score)``; ``basis_score`` is ``None`` when no scoring
    frame was given.
  """
  has_score = df_norm_score is not None and len(df_norm_score) > 0

  # FIX: Assign df_norm.index to prevent Patsy from resetting indices
  if not covariates_for_cate:
    basis_fit = pd.DataFrame(
        np.ones((len(df_norm), 1)), columns=['Intercept'], index=df_norm.index
    )
    basis_score = None
    if has_score:
      basis_score = pd.DataFrame(
          np.ones((len(df_norm_score), 1)),
          columns=['Intercept'],
          index=df_norm_score.index,
      )
    return basis_fit, basis_score

  if cate_type == 'auto_additive':
    formula_parts = []
    data_fit = {}
    data_score = {}
    for col in covariates_for_cate:
      safe_col = str(col).replace(' ', '_').replace('-', '_')
      is_binary = len(df_orig[col].dropna().unique()) <= 2
      if is_binary:
        formula_parts.append(safe_col)
      else:
        formula_parts.append(
            f'bs({safe_col}, df={cate_df}, degree={cate_degree},'
            f' include_intercept={cate_intercept})'
        )
      data_fit[safe_col] = df_norm[col]
      if has_score:
        data_score[safe_col] = _clip_to_fitted_range(
            df_norm[col], df_norm_score[col]
        )
    formula = ' + '.join(formula_parts)
  else:
    formula = cate_type
    data_fit = {col: df_norm[col] for col in covariates_for_cate}
    data_score = {}
    if has_score:
      data_score = {
          col: _clip_to_fitted_range(df_norm[col], df_norm_score[col])
          for col in covariates_for_cate
      }

  mat = patsy.dmatrix(formula, data_fit)
  columns = list(mat.design_info.column_names)
  basis_fit = pd.DataFrame(
      np.asarray(mat), columns=columns, index=df_norm.index
  )
  basis_score = None
  if has_score:
    (mat_score,) = patsy.build_design_matrices([mat.design_info], data_score)
    basis_score = pd.DataFrame(
        np.asarray(mat_score), columns=columns, index=df_norm_score.index
    )
  return basis_fit, basis_score
