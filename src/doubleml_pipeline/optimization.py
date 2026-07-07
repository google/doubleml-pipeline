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

"""Optimization module for calculating ROI metrics and generating visualizations.

This module provides the optimization capabilities for percentage treatments,
calculating ROI, and outputting both CSV metrics and visualization charts.
"""

from collections.abc import Sequence
import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def optimize(
    df_1a_consolidated: pd.DataFrame,
    treatment: str,
    treatment_amt: str,
    periods: tuple[str, str],
    threshold_roi: float = 0.0,
    other_cost_variables: Optional[Sequence[str]] = None,
    target_geo: str = 'all',
    target_item: str = 'all',
    **kwargs,
):
  """Optimization feature for percentage treatments.

  Calculates ROI and optimization metrics across specified periods and saves to
  CSV. Also generates visualization overlays highlighting the optimized spend.

  Args:
    df_1a_consolidated: Consolidated DataFrame.
    treatment: Name of the treatment variable.
    treatment_amt: Amount of treatment.
    periods: A tuple of (start_date, end_date) specifying the optimization
      periods.
    threshold_roi: Minimum ROI threshold for optimization (default: 0.0).
    other_cost_variables: List of cost variables to consider.
    target_geo: Target geography for optimization (default: 'all').
    target_item: Target item for optimization (default: 'all').
    **kwargs: Additional configuration parameters such as date_col, geo_col,
      item_col, sales_col, and out_dir.
  """
  if other_cost_variables is None:
    other_cost_variables = []

  date_col = kwargs.get('date_col', 'Date')
  geo_col = kwargs.get('geo_col', 'Geo')
  item_col = kwargs.get('item_col', 'Item')
  sales_col = kwargs.get('sales_col', 'Sales')
  out_dir = kwargs.get('out_dir', '')

  kpi_col = f'estimated_incremental_KPI_{treatment}'

  # Ensure output directory exists
  opt_dir = os.path.join(out_dir, '007_optimisation')
  os.makedirs(opt_dir, exist_ok=True)

  df_work = df_1a_consolidated.copy()
  if date_col in df_work.columns:
    df_work[date_col] = pd.to_datetime(df_work[date_col])

  start_date = pd.to_datetime(periods[0])
  end_date = pd.to_datetime(periods[1])

  # 1. Output 1: CSV metrics
  df_opt = df_work[
      (df_work[date_col] >= start_date) & (df_work[date_col] <= end_date)
  ].copy()

  geos = [target_geo] if target_geo != 'all' else ['all']
  if target_geo == 'all' and geo_col and geo_col in df_work.columns:
    geos.extend(sorted(df_work[geo_col].dropna().unique()))

  items = [target_item] if target_item != 'all' else ['all']
  if target_item == 'all' and item_col and item_col in df_work.columns:
    items.extend(sorted(df_work[item_col].dropna().unique()))

  results = []
  for g in geos:
    for i in items:
      df_sub = df_opt.copy()
      if g != 'all':
        df_sub = df_sub[df_sub[geo_col] == g]
      if i != 'all':
        df_sub = df_sub[df_sub[item_col] == i]

      if df_sub.empty:
        continue

      grouped = df_sub.groupby(date_col).sum(numeric_only=True)
      if treatment_amt not in grouped.columns or kpi_col not in grouped.columns:
        continue

      grouped['ROI'] = np.divide(
          grouped[kpi_col],
          grouped[treatment_amt],
          out=np.zeros_like(grouped[kpi_col], dtype=float),
          where=grouped[treatment_amt] != 0,
      )

      total_cost = grouped[treatment_amt].sum()
      mask = grouped['ROI'] < threshold_roi
      reduction = grouped.loc[mask, treatment_amt].sum()
      optimised_total = total_cost - reduction

      total_cost_cv = 0.0
      for cv in other_cost_variables:
        if cv in grouped.columns:
          total_cost_cv += grouped[cv].sum()

      increase_pct = reduction / total_cost_cv if total_cost_cv != 0 else 0.0

      results.append({
          'treatment': treatment,
          'other_cost_variables': other_cost_variables,
          'target_geo': g,
          'target_item': i,
          'periods_start_date': periods[0],
          'periods_end_date': periods[1],
          'threshold_roi': threshold_roi,
          'total_cost_of_treatment_in_periods': total_cost,
          'optimised_total_cost_of_treatment_in_periods': optimised_total,
          'reduction_of_cost_of_treatment': reduction,
          'total_cost_of_other_cost_variables': total_cost_cv,
          'increase_pct_of_total_cost_of_other_cost_variables': increase_pct,
      })

  if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(opt_dir, 'optimization_metrics.csv'), index=False
    )

  # 2. Output 2: Visualization overlays
  # Generate charts for Total, Geo, and Item levels across the full dataset
  # (not just periods)
  levels_list = [('total', None)]
  if geo_col and geo_col in df_work.columns:
    for g_val in sorted(df_work[geo_col].dropna().unique()):
      levels_list.append(('geo', g_val))
  if item_col and item_col in df_work.columns:
    for i_val in sorted(df_work[item_col].dropna().unique()):
      levels_list.append(('item', i_val))

  agg_dict = {
      col: 'sum'
      for col in [sales_col, treatment_amt, kpi_col]
      if col in df_work.columns
  }

  if date_col in df_work.columns:
    for level_type, level_val in levels_list:
      if level_type == 'total':
        chart_df = df_work
        prefix = 'total'
        title_prefix = 'Total'
      elif level_type == 'geo':
        chart_df = df_work[df_work[geo_col] == level_val]
        prefix = f'geo_{level_val}'
        title_prefix = f'[{level_val}]'
      elif level_type == 'item':
        chart_df = df_work[df_work[item_col] == level_val]
        prefix = f'item_{level_val}'
        title_prefix = f'[{level_val}]'
      else:
        continue

      if chart_df.empty or not agg_dict:
        continue

      df_ts = (
          chart_df.groupby(date_col, as_index=False)
          .agg(agg_dict)
          .sort_values(date_col)
      )

      # Calculate ROI and threshold curve
      if treatment_amt in df_ts.columns and kpi_col in df_ts.columns:
        roi = np.divide(
            df_ts[kpi_col].values,
            df_ts[treatment_amt].values,
            out=np.zeros_like(df_ts[kpi_col].values, dtype=float),
            where=df_ts[treatment_amt].values != 0,
        )
        mask = roi >= threshold_roi
        green_line = np.where(mask, df_ts[treatment_amt].values, 0.0)
      else:
        continue

      _, ax = plt.subplots(figsize=(24, 10))

      if treatment_amt in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[treatment_amt],
            color='orange',
            alpha=0.8,
            linewidth=2,
            label=f'Spend ({treatment_amt})',
        )
      if kpi_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[kpi_col],
            color='red',
            linewidth=2,
            label=f'Est. Incremental Sales ({kpi_col})',
        )

      # Add the new threshold line and fill
      ax.plot(
          df_ts[date_col],
          green_line,
          color='darkgreen',
          linewidth=2,
          label=f'threshold_Spend_roi_{threshold_roi}',
      )
      ax.fill_between(
          df_ts[date_col],
          df_ts[treatment_amt],
          green_line,
          color='green',
          alpha=0.3,
          label='Reduced Spend',
      )

      ax.set_title(
          f'{title_prefix} Time Series with ROI Optimization'
          f' (Threshold={threshold_roi})',
          fontsize=24,
      )
      ax.set_xlabel('Date', fontsize=20)
      ax.set_ylabel('Amount', fontsize=20)
      ax.tick_params(axis='y', labelsize=16)
      ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
      plt.xticks(rotation=90, fontsize=12)
      ax.legend(fontsize=16, loc='upper left')
      plt.grid(True, alpha=0.3)
      plt.subplots_adjust(bottom=0.25)
      plt.tight_layout()

      out_path = os.path.join(opt_dir, f'{prefix}_timeseries_promo_vs_est.png')
      plt.savefig(out_path, dpi=200)
      plt.close()
