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

"""Module defining the core execution pipeline logic for DoubleML PLR model construction."""

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
from . import metrics


class DoubleMLPipeline:
  """Configures and executes the core structure for Partial Linear Regression estimation."""

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
  ):
    """Initializes the DoubleML Pipeline object defining structural causal bounds."""
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

    if isinstance(cate_structure, dict):
      self.cate_type = cate_structure.get('type', 'auto_additive')
      self.cate_df = cate_structure.get('df', 3)
      self.cate_degree = cate_structure.get('degree', 2)
      self.cate_intercept = cate_structure.get('include_intercept', False)
    else:
      self.cate_type = (
          str(cate_structure) if cate_structure else 'auto_additive'
      )
      if self.cate_type == 'auto_additive':
        self.cate_df = 3
        self.cate_degree = 2
        self.cate_intercept = False
      else:
        self.cate_df = 'custom'
        self.cate_degree = 'custom'
        self.cate_intercept = 'custom'

    self.ml_y_name = ml_y_model
    self.ml_t_name = ml_t_model
    self.ml_y = flaml_dml.FlamlRegressorDoubleML(
        time=time_budget,
        estimator_list=[ml_y_model],
        metric='rmse',
        random_state=self.random_state,
    )
    self.ml_t = flaml_dml.FlamlRegressorDoubleML(
        time=time_budget,
        estimator_list=[ml_t_model],
        metric='rmse',
        random_state=self.random_state,
    )

  def _build_cate_basis(
      self, df_norm: pd.DataFrame, df_orig: pd.DataFrame
  ) -> pd.DataFrame:
    """Generates spline or linear basis arrays for Conditional Average Treatment Effect formulas."""
    # FIX: Assign df_norm.index to prevent Patsy from resetting indices
    if not self.covariates_for_cate:
      return pd.DataFrame(
          np.ones((len(df_norm), 1)), columns=['Intercept'], index=df_norm.index
      )

    if self.cate_type == 'auto_additive':
      formula_parts = []
      data_dict = {}
      for col in self.covariates_for_cate:
        safe_col = str(col).replace(' ', '_').replace('-', '_')
        is_binary = len(df_orig[col].dropna().unique()) <= 2
        if is_binary:
          formula_parts.append(safe_col)
        else:
          formula_parts.append(
              f'bs({safe_col}, df={self.cate_df}, degree={self.cate_degree},'
              f' include_intercept={self.cate_intercept})'
          )
        data_dict[safe_col] = df_norm[col]

      mat = patsy.dmatrix(' + '.join(formula_parts), data_dict)
      return pd.DataFrame(
          mat, columns=mat.design_info.column_names, index=df_norm.index
      )
    else:
      mat = patsy.dmatrix(
          self.cate_type,
          {col: df_norm[col] for col in self.covariates_for_cate},
      )
      return pd.DataFrame(
          mat, columns=mat.design_info.column_names, index=df_norm.index
      )

  def run(
      self,
      df_orig: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      rep_id: int = 1,
  ) -> Dict[str, Any]:
    """Runs the DoubleML causal execution over normalized variables."""
    if len(df_orig) < 1000 and self.covariates_for_cate:
      warnings.warn(
          'Sample size is small (n < 1000). Set degree of continuous covariates'
          ' to 2 or less.',
          UserWarning,
      )

    model_name = f'{self.ml_y_name}_{self.ml_t_name}_{self.n_folds}_{rep_id}_{self.time_budget}_{len(self.covariates_for_cate)}'

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
    df_norm = scaler.fit_transform(df_orig)

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

      spline_basis = self._build_cate_basis(df_norm, df_orig)
      cate_obj = dml_model.cate(basis=spline_basis)
      conf_int = cate_obj.confint(basis=spline_basis)

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
      for r_models in dml_model.models['ml_l'][self.d_col]:
        for fold_m in r_models:
          aml = getattr(fold_m, 'auto_ml_', getattr(fold_m, 'auto_ml', None))
          if aml and hasattr(aml, 'best_loss'):
            y_losses.append(aml.best_loss)

      for r_models in dml_model.models['ml_m'][self.d_col]:
        for fold_m in r_models:
          aml = getattr(fold_m, 'auto_ml_', getattr(fold_m, 'auto_ml', None))
          if aml and hasattr(aml, 'best_loss'):
            t_losses.append(aml.best_loss)
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

    return {
        'model_name': model_name,
        'df_1a': df_results,
        'metrics_1c': metrics_1c,
        'model_obj': dml_model,
        'cate_coef_dict': cate_coef_dict,
    }
