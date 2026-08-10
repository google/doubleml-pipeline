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

"""Module providing feature engineering utilities such as lag generation."""

from typing import List, Optional, Union

import pandas as pd


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    periods: Union[int, List[int]],
    date_col: str,
    unit_col: Optional[Union[str, List[str]]] = None,
    na_fill: bool = False,
) -> pd.DataFrame:
  """Creates lagged features for specified columns in the dataframe.

  Args:
      df (pd.DataFrame): The input dataframe.
      columns (List[str]): List of column names to create lags for.
      periods (Union[int, List[int]]): Number of lags or specific lag indices to
        generate (e.g., [1, 7, 14] or 3).
      date_col (str): Column specifying the date/time for exact chronological
        lagging.
      unit_col (Optional[str]): Optional column to group by before shifting
        (essential for Geo/Panel data).
      na_fill (bool): If True, fills the resulting NaN values with 0. Default is
        False.

  Returns:
      pd.DataFrame: A new dataframe with the lagged columns added.

  Raises:
      ValueError: If date_col is not specified or not in the dataframe.
      KeyError: If any of the specified columns are not present in the
        dataframe.
  """
  if not date_col:
    raise ValueError("date_col must be specified.")
  if date_col not in df.columns:
    raise ValueError(f"date_col '{date_col}' not found in the dataframe.")

  df_out = df.copy()
  for col in columns:
    if col not in df_out.columns:
      raise KeyError(f"Column '{col}' not found in the dataframe.")

  # 1. Sort the dataframe securely
  row_sort_keys = []
  if unit_col:
    unit_cols = [unit_col] if isinstance(unit_col, str) else unit_col
    for col_name in unit_cols:
      if col_name in df_out.columns:
        row_sort_keys.append(col_name)
  row_sort_keys.append(date_col)

  df_out = df_out.sort_values(by=row_sort_keys)

  # 2. Advanced exact date matching using vectorized Reindexing
  temp_date = pd.to_datetime(df_out[date_col])

  # Infer chronological period delta dynamically
  diffs = temp_date.diff()
  period_delta = diffs[diffs > pd.Timedelta(0)].median()
  if pd.isna(period_delta):
    period_delta = pd.Timedelta(days=1)

  v_periods = (
      list(range(1, periods + 1)) if isinstance(periods, int) else periods
  )

  for col in columns:
    present_unit_cols = []
    if unit_col:
      unit_cols = [unit_col] if isinstance(unit_col, str) else unit_col
      present_unit_cols = [c for c in unit_cols if c in df_out.columns]

    if present_unit_cols:
      idx_arrays = [df_out[c] for c in present_unit_cols] + [temp_date]
      lookup = pd.Series(
          df_out[col].values, index=pd.MultiIndex.from_arrays(idx_arrays)
      )
    else:
      lookup = pd.Series(df_out[col].values, index=temp_date)

    # Ensure unique index to avoid reindex errors safely
    lookup = lookup.loc[~lookup.index.duplicated(keep="last")]

    for p in v_periods:
      lag_col_name = f"{col}_l{p}"
      target_date = temp_date - (p * period_delta)

      if present_unit_cols:
        target_idx = pd.MultiIndex.from_arrays(
            [df_out[c] for c in present_unit_cols] + [target_date]
        )
      else:
        target_idx = target_date

      # Extract exact matches; returns NaN automatically if target date does
      # not exist
      df_out[lag_col_name] = lookup.reindex(target_idx).values

      if na_fill:
        df_out[lag_col_name] = df_out[lag_col_name].fillna(0)

  return df_out


def create_amount_from_pct(
    df: pd.DataFrame, sales_col: str, d_col: str, output_col: str
) -> pd.DataFrame:
  """Converts a percentage discount/rate treatment column into a monetary amount column.

  Formula: Discount_amt = Sales_actual * (d_col / (1 - d_col))
  Clips d_col to a maximum of 0.999 to prevent division by zero.

  Args:
    df: The input DataFrame.
    sales_col: The name of the column containing sales amounts.
    d_col: The name of the column containing the percentage discount.
    output_col: The name of the column to output the calculated amount to.

  Returns:
    A new DataFrame with the added monetary amount column.
  """
  if sales_col not in df.columns:
    raise ValueError(f"sales_col '{sales_col}' not found in dataframe.")
  if d_col not in df.columns:
    raise ValueError(f"d_col '{d_col}' not found in dataframe.")

  df_out = df.copy()
  pct_clipped = df_out[d_col].clip(upper=0.999)
  df_out[output_col] = df_out[sales_col] * (pct_clipped / (1.0 - pct_clipped))
  return df_out
