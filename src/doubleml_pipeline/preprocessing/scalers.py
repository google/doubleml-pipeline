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

"""Module providing centralized data scaling utilities for causal inference."""

from typing import Any, Dict, List, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing


class CausalDataScaler:
  """Manages various scaling operations for causal inference.

  Supports standard scaling, population-based standardization, median
  normalization, and min-max scaling.
  """

  def __init__(
      self,
      standardize_cols: Optional[List[str]] = None,
      population_standardize_cols: Optional[List[str]] = None,
      population_median_normalize_cols: Optional[List[str]] = None,
      min_max_cols: Optional[List[str]] = None,
      median_normalize_cols: Optional[List[str]] = None,
      population_col: str = 'population',
  ):
    """Initializes the CausalDataScaler with specified scaling strategies."""
    self.standardize_cols = standardize_cols or []
    self.population_standardize_cols = population_standardize_cols or []
    self.population_median_normalize_cols = (
        population_median_normalize_cols or []
    )
    self.min_max_cols = min_max_cols or []
    self.median_normalize_cols = median_normalize_cols or []
    self.population_col = population_col

    self.standard_params_: Dict[str, Dict[str, float]] = {}
    self.pop_standard_params_: Dict[str, Dict[str, float]] = {}
    self.pop_median_params_: Dict[str, Dict[str, float]] = {}
    self.min_max_params_: Dict[str, Dict[str, float]] = {}
    self.median_params_: Dict[str, Dict[str, float]] = {}

    if self.population_col in self.population_median_normalize_cols:
      warnings.warn(
          f"The population_col '{self.population_col}' is also included in "
          'population_median_normalize_cols. This will cause the column to '
          'be divided by itself, resulting in a constant value of 1.0 for '
          'all rows, and the model will disregard it. Consider moving it to '
          'standardize_cols instead.',
          UserWarning,
      )

  # TOD0: The Code doesn't Check if same column is present in multiple lists
  # This would lead to scaling being applied multiple times to the same column

  def fit(self, x: pd.DataFrame, y: Optional[Any] = None) -> 'CausalDataScaler':
    """Fits the scaler attributes to the dataset and calculates parameters."""
    del y  # Unused
    df = x
    needs_pop = (
        self.population_standardize_cols
        or self.population_median_normalize_cols
    )
    if needs_pop and self.population_col not in df.columns:
      raise ValueError(
          f"Specified population column '{self.population_col}' not found in"
          ' dataframe.'
      )

    if needs_pop:
      self.raw_population_ = df[self.population_col].copy()

    # 1. Standard Scaling
    for col in [c for c in self.standardize_cols if c in df.columns]:
      scaler = preprocessing.StandardScaler()
      scaler.fit(df[[col]])
      self.standard_params_[col] = {
          'mean': float(scaler.mean_[0]),
          'scale': float(scaler.scale_[0]),
      }

    # 2. Min-Max Scaling
    for col in [c for c in self.min_max_cols if c in df.columns]:
      scaler = preprocessing.MinMaxScaler()
      scaler.fit(df[[col]])
      self.min_max_params_[col] = {
          'data_min': float(scaler.data_min_[0]),
          'data_max': float(scaler.data_max_[0]),
      }

    # 3. Population Standardization
    for col in [c for c in self.population_standardize_cols if c in df.columns]:
      pop_series = df[self.population_col].replace(0, np.nan)
      val_per_pop = (df[col] / pop_series).replace([np.inf, -np.inf], np.nan)

      scaler = preprocessing.StandardScaler()
      scaler.fit(val_per_pop.fillna(0).to_frame())
      self.pop_standard_params_[col] = {
          'mean': float(scaler.mean_[0]),
          'scale': float(scaler.scale_[0]),
      }

    # 4. Population Median Normalization
    for col in [
        c for c in self.population_median_normalize_cols if c in df.columns
    ]:
      pop_series = df[self.population_col].replace(0, np.nan)
      val_per_pop = (df[col] / pop_series).replace([np.inf, -np.inf], np.nan)

      median_val = val_per_pop[val_per_pop > 0].median()
      if pd.isna(median_val) or median_val == 0:
        median_val = val_per_pop.median()
        if pd.isna(median_val) or median_val == 0:
          median_val = 1.0

      self.pop_median_params_[col] = {'median_val_per_pop': float(median_val)}

    # 5. Median Normalization
    for col in [c for c in self.median_normalize_cols if c in df.columns]:
      non_zero_vals = df[col][df[col] != 0]
      median_val = non_zero_vals.median()
      if pd.isna(median_val) or median_val == 0:
        median_val = df[col].median()
        if pd.isna(median_val) or median_val == 0:
          median_val = 1.0

      self.median_params_[col] = {'median_val': float(median_val)}

    return self

  def transform(self, x: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
    """Applies the fitted scaling parameters to the input data."""
    del y  # Unused parameter
    df = x
    needs_pop = (
        self.population_standardize_cols
        or self.population_median_normalize_cols
    )
    if needs_pop and self.population_col not in df.columns:
      raise ValueError(
          f"Specified population column '{self.population_col}' not found in"
          ' dataframe.'
      )

    df_norm = df.copy()

    if needs_pop:
      df_norm['_raw_population_'] = df_norm[self.population_col]

    # 1. Standard Scaling
    for col in [c for c in self.standardize_cols if c in df_norm.columns]:
      if col in self.standard_params_:
        p_val = self.standard_params_[col]
        df_norm[col] = (df_norm[col] - p_val['mean']) / p_val['scale']

    # 2. Min-Max Scaling
    for col in [c for c in self.min_max_cols if c in df_norm.columns]:
      if col in self.min_max_params_:
        p_val = self.min_max_params_[col]
        data_range = p_val['data_max'] - p_val['data_min']
        if data_range == 0:
          df_norm[col] = 0.0
        else:
          df_norm[col] = (df_norm[col] - p_val['data_min']) / data_range

    # 3. Population Standardization
    for col in [
        c for c in self.population_standardize_cols if c in df_norm.columns
    ]:
      if col in self.pop_standard_params_:
        pop_series = df_norm['_raw_population_'].replace(0, np.nan)
        val_per_pop = (
            (df_norm[col] / pop_series)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        p_val = self.pop_standard_params_[col]
        df_norm[col] = (val_per_pop - p_val['mean']) / p_val['scale']

    # 4. Population Median Normalization
    for col in [
        c for c in self.population_median_normalize_cols if c in df_norm.columns
    ]:
      if col in self.pop_median_params_:
        pop_series = df_norm['_raw_population_'].replace(0, np.nan)
        val_per_pop = (df_norm[col] / pop_series).replace(
            [np.inf, -np.inf], np.nan
        )
        p_val = self.pop_median_params_[col]
        df_norm[col] = (
            val_per_pop.fillna(p_val['median_val_per_pop'])
            / p_val['median_val_per_pop']
        )

    # 5. Median Normalization
    for col in [c for c in self.median_normalize_cols if c in df_norm.columns]:
      if col in self.median_params_:
        p_val = self.median_params_[col]
        df_norm[col] = df_norm[col] / p_val['median_val']

    # Clean up tracking workspace column prior to returning
    df_norm = df_norm.drop(columns=['_raw_population_'], errors='ignore')

    return df_norm

  def fit_transform(
      self, x: pd.DataFrame, y: Optional[Any] = None
  ) -> pd.DataFrame:
    """Fits the scaler attributes to the dataset and returns a scaled dataframe.

    Args:
      x: The dataframe to fit and transform.
      y: Optional target variable.

    Returns:
      The scaled dataframe.
    """
    return self.fit(x, y).transform(x, y)

  def inverse_transform_column(
      self,
      df_norm: pd.DataFrame,
      target_col: str,
      param_source_col: Optional[str] = None,
      is_difference: bool = False,
  ) -> pd.Series:
    """Reverts a scaled column back to its original magnitude.

    This is based on fitted parameters.

    Args:
      df_norm: The normalized dataframe.
      target_col: The name of the column to inverse transform.
      param_source_col: The name of the column whose parameters should be used.
        Defaults to target_col.
      is_difference: Whether the column represents a difference between two
        values, in which case the mean shift is not applied.

    Returns:
      A pandas series with the original magnitude.
    """
    source = param_source_col if param_source_col else target_col

    if source in self.standard_params_:
      p_val = self.standard_params_[source]
      if is_difference:
        return df_norm[target_col] * p_val['scale']
      return df_norm[target_col] * p_val['scale'] + p_val['mean']

    elif source in self.min_max_params_:
      p_val = self.min_max_params_[source]
      if is_difference:
        return df_norm[target_col] * (p_val['data_max'] - p_val['data_min'])
      return (
          df_norm[target_col] * (p_val['data_max'] - p_val['data_min'])
          + p_val['data_min']
      )

    elif source in self.median_params_:
      p_val = self.median_params_[source]
      return df_norm[target_col] * p_val['median_val']

    elif (
        source in self.pop_standard_params_ or source in self.pop_median_params_
    ):
      if self.population_col not in df_norm.columns:
        raise ValueError(
            f"Specified population column '{self.population_col}' not found in"
            ' dataframe.'
        )

      if not hasattr(self, 'raw_population_') or self.raw_population_ is None:
        raise ValueError('Fitted raw population data not found in scaler.')

      pop_series = self.raw_population_.loc[df_norm.index].replace(0, np.nan)

      if source in self.pop_standard_params_:
        p_val = self.pop_standard_params_[source]
        if is_difference:
          raw_val = df_norm[target_col] * p_val['scale']
        else:
          raw_val = df_norm[target_col] * p_val['scale'] + p_val['mean']
        return raw_val * pop_series

      elif source in self.pop_median_params_:
        p_val = self.pop_median_params_[source]
        return df_norm[target_col] * p_val['median_val_per_pop'] * pop_series

    return df_norm[target_col]
