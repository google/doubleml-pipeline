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

"""Module for evaluating model performance and calculating various causal and nuisance metrics."""

from typing import Dict
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def calculate_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
  """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between true and predicted values."""
  denominator = np.abs(y_true) + np.abs(y_pred)
  denominator = np.where(denominator == 0, 1e-9, denominator)
  smape_val = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
  return float(smape_val)


def calculate_ground_truth_metrics(
    y_true: pd.Series, y_pred: pd.Series, total_input: float
) -> Dict[str, float]:
  """Calculates accuracy metrics against ground truth data, including SMAPE, R-squared, MAE, and RMSE."""
  if y_true is None or y_true.empty:
    return {
        'true_total_roi': np.nan,
        'SMAPE': np.nan,
        'R_squared': np.nan,
        'MAE': np.nan,
        'RMSE': np.nan,
    }

  true_kpi = y_true.sum()
  true_roi = true_kpi / total_input if total_input > 0 else 0
  smape = calculate_smape(y_true, y_pred)

  r2_val = (
      metrics.r2_score(y_true, y_pred)
      if len(y_true) > 1 and np.var(y_true) > 0
      else np.nan
  )
  mae = metrics.mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

  return {
      'true_total_roi': true_roi,
      'SMAPE': smape,
      'R_squared': r2_val,
      'MAE': mae,
      'RMSE': rmse,
  }


def evaluate_nuisance_residuals(
    residuals: pd.Series,
    treatment: pd.Series,
    covariates: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
  """Evaluates the properties of nuisance residuals, including variance ratio, skewness, and normality."""
  results: Dict[str, float] = {}
  results['residual_mean'] = float(residuals.mean())
  results['residual_skewness'] = float(residuals.skew())
  t_var = treatment.var()
  results['variance_ratio'] = (
      float(residuals.var() / t_var) if t_var != 0 else np.nan
  )

  clean_residuals = residuals.dropna()
  if len(clean_residuals) >= 8:
    _, p_value = stats.normaltest(clean_residuals)
    results['normaltest_pvalue'] = float(p_value)
  else:
    results['normaltest_pvalue'] = np.nan

  if covariates is not None and not covariates.empty:
    correlations = covariates.corrwith(residuals)
    results['max_correlation'] = float(np.abs(correlations).max())
  else:
    results['max_correlation'] = np.nan
  return results


def aggregate_geo_item_metrics(
    df: pd.DataFrame,
    model_name: str,
    d_col: str,
    y_col: str,
    geo_col: Optional[str] = None,
    item_col: Optional[str] = None,
    geo_name_col: Optional[str] = None,
    item_name_col: Optional[str] = None,
    gt_col: Optional[str] = None,
) -> pd.DataFrame:
  """Aggregates estimated KPIs and ROI metrics across specified geographical and item dimensions."""
  group_cols = []
  if geo_col and geo_col in df.columns:
    group_cols.append(geo_col)
  if geo_name_col and geo_name_col in df.columns:
    group_cols.append(geo_name_col)
  if item_col and item_col in df.columns:
    group_cols.append(item_col)
  if item_name_col and item_name_col in df.columns:
    group_cols.append(item_name_col)

  if not group_cols:
    df['geo'] = 'all'
    group_cols = ['geo']

  kpi_col = f'estimated_incremental_KPI_{d_col}'

  def _compute_metrics(group_df: pd.DataFrame) -> pd.Series:
    """Computes aggregated input, KPI, sales, and ROI metrics for a single grouped dataframe."""
    total_in = group_df[d_col].sum()
    total_kpi = group_df[kpi_col].sum()
    total_sales = group_df[y_col].sum()
    est_roi = total_kpi / total_in if total_in > 0 else 0

    res = {
        'total_input': total_in,
        'total_estimated_incremental_KPI': total_kpi,
        'total_sales': total_sales,
        'total_estimated_roi': est_roi,
    }

    if gt_col and gt_col in group_df.columns:
      y_true = group_df[gt_col]
      y_pred = group_df[kpi_col]
      gt_metrics = calculate_ground_truth_metrics(y_true, y_pred, total_in)
      res.update(gt_metrics)
    else:
      res.update({
          'true_total_roi': np.nan,
          'SMAPE': np.nan,
          'R_squared': np.nan,
          'MAE': np.nan,
          'RMSE': np.nan,
      })

    return pd.Series(res)

  df_agg = (
      df.groupby(group_cols, observed=True)
      .apply(_compute_metrics)
      .reset_index()
  )
  df_agg.insert(0, 'model', model_name)
  return df_agg


def compute_nuisance_combined_error(
    df_metrics: pd.DataFrame,
    y_rmse_col: str,
    t_rmse_col: str,
) -> pd.DataFrame:
  """Computes a normalized composite error metric from Y and T model errors.

  Contextual Error Shifting: For the exact same winning model architecture, its
  computed `combined_error` will mathematically differ between Phase 1 and
  Phase 3. This is expected.

  - Phase 1 Exploration (selection.py): Normalization spans across distinct
    competing algorithmic combinations (e.g., XGBoost vs Ridge). The loss
    measures inter-model algorithmic superiority.
  - Phase 3 Ensembling (workflow.py): Normalization strictly spans across
    data-splitting runs (e.g., rep_1..20) of the *same* winning model. The loss
    measures intra-model partition stability.

  Args:
    df_metrics: DataFrame containing the RMSE columns.
    y_rmse_col: Name of the column for Y model RMSE.
    t_rmse_col: Name of the column for T model RMSE.

  Returns:
    The updated DataFrame with 'norm_Y_rmse', 'norm_T_rmse', and
    'combined_error' columns.
  """
  epsilon = 1e-9

  min_val_y = df_metrics[y_rmse_col].min()
  max_val_y = df_metrics[y_rmse_col].max()
  df_metrics['norm_Y_rmse'] = (df_metrics[y_rmse_col] - min_val_y + epsilon) / (
      max_val_y - min_val_y + epsilon
  )

  min_val_t = df_metrics[t_rmse_col].min()
  max_val_t = df_metrics[t_rmse_col].max()
  df_metrics['norm_T_rmse'] = (df_metrics[t_rmse_col] - min_val_t + epsilon) / (
      max_val_t - min_val_t + epsilon
  )

  df_metrics['combined_error'] = df_metrics['norm_T_rmse'] * (
      df_metrics['norm_T_rmse'] + df_metrics['norm_Y_rmse']
  )

  return df_metrics
