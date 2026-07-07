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

"""Module providing model selection criteria algorithms (e.g., combined RMSE)."""

import pandas as pd


def shortlist_top_models(
    df_metrics: pd.DataFrame,
    top_n: int = 5,
    y_rmse_col: str = 'nuisance_models_rmse_Y_model_evaluated_by_AutoML',
    t_rmse_col: str = 'nuisance_models_rmse_T_model_evaluated_by_AutoML',
) -> pd.DataFrame:
  """Shortlists top N performing estimator combinations using combined normalized RMSE."""
  if df_metrics.empty:
    return df_metrics

  df_copy = df_metrics.copy()
  epsilon = 1e-9

  min_val_y = df_copy[y_rmse_col].min()
  max_val_y = df_copy[y_rmse_col].max()
  df_copy['norm_Y_rmse'] = (df_copy[y_rmse_col] - min_val_y + epsilon) / (
      max_val_y - min_val_y + epsilon
  )

  min_val_t = df_copy[t_rmse_col].min()
  max_val_t = df_copy[t_rmse_col].max()
  df_copy['norm_T_rmse'] = (df_copy[t_rmse_col] - min_val_t + epsilon) / (
      max_val_t - min_val_t + epsilon
  )

  df_copy['combined_error'] = df_copy['norm_T_rmse'] * (
      df_copy['norm_T_rmse'] + df_copy['norm_Y_rmse']
  )

  df_sorted = df_copy.sort_values(
      by='combined_error', ascending=True
  ).reset_index(drop=True)

  df_sorted['shortlisted'] = 0
  shortlist_limit = min(top_n, len(df_sorted))
  if shortlist_limit > 0:
    df_sorted.loc[: shortlist_limit - 1, 'shortlisted'] = 1

  return df_sorted
