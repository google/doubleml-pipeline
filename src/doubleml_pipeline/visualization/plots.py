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

"""Plotting functions for the DoubleML pipeline."""

import os
import re
import textwrap
from typing import List, Optional
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def _add_metrics_box(ax, df_plot, kpi_col, gt_col):
  """Helper function to overlay an evaluation metrics box on the plot."""
  if gt_col and gt_col in df_plot.columns:
    y_true = df_plot[gt_col].fillna(0)
    y_pred = df_plot[kpi_col].fillna(0)
    if len(y_true) > 0:
      denom = np.abs(y_true) + np.abs(y_pred)
      denom = np.where(denom == 0, 1e-9, denom)
      smape = np.mean(2 * np.abs(y_pred - y_true) / denom) * 100
      r2 = (
          metrics.r2_score(y_true, y_pred)
          if len(y_true) > 1 and np.var(y_true) > 0
          else np.nan
      )
      mae = metrics.mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

      textstr = (
          f'SMAPE: {smape:.2f}%\n$R^2$: {r2:.2f}\nMAE: {mae:.2f}\nRMSE:'
          f' {rmse:.2f}'
      )
      props = dict(
          boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'
      )
      ax.text(
          0.98,
          0.95,
          textstr,
          transform=ax.transAxes,
          fontsize=14,
          verticalalignment='top',
          horizontalalignment='right',
          bbox=props,
      )


def _format_title(title, df, actual_y_col=None):
  """Helper function to format titles with treatment and outcome columns."""
  treatments = [
      c.replace('estimated_incremental_KPI_', '')
      for c in df.columns
      if c.startswith('estimated_incremental_KPI_')
      and not c.endswith(('%', 'total_est_temp'))
  ]

  add_parts = []
  if treatments:
    add_parts.append(f'Treatment(s): {", ".join(treatments)}')

  if actual_y_col:
    add_parts.append(f'Outcome: {actual_y_col}')

  if add_parts:
    return f"{title}\n[{' | '.join(add_parts)}]"
  return title


def _set_dynamic_xticks(ax, n_items, is_date=False):
  """Helper to thin out x-ticks for dense plots while keeping grid intact."""
  max_ticks = 40
  if n_items > max_ticks:
    if is_date:
      ax.xaxis.set_major_locator(
          mtick.MaxNLocator(nbins=max_ticks, prune='both')
      )
    else:
      step = max(1, n_items // max_ticks)
      ticks = np.arange(0, n_items, step)
      ax.set_xticks(ticks)
  else:
    if not is_date:
      ax.set_xticks(range(n_items))


def plot_incremental_kpi_line(
    df,
    date_col,
    kpi_col,
    lower_col,
    upper_col,
    out_path,
    title,
    gt_col=None,
    actual_y_col=None,
):
  """Plots a line chart showing estimated incremental KPI over time with confidence intervals."""
  title = _format_title(title, df, actual_y_col)

  agg_cols = [kpi_col, lower_col, upper_col]
  if gt_col and gt_col in df.columns:
    agg_cols.append(gt_col)
  if actual_y_col and actual_y_col in df.columns:
    agg_cols.append(actual_y_col)

  df_plot = (
      df.groupby(date_col, as_index=False, observed=False)[agg_cols]
      .sum()
      .sort_values(date_col)
  )
  df_plot[date_col] = pd.to_datetime(df_plot[date_col])

  n_dates = len(df_plot[date_col].unique())
  dynamic_width = min(40, max(24, n_dates * 0.15))
  _, ax = plt.subplots(figsize=(dynamic_width, 12))

  ax.plot(
      df_plot[date_col],
      df_plot[kpi_col],
      label='Estimated Incremental KPI',
      color='red',
      linewidth=2,
  )
  ax.fill_between(
      df_plot[date_col],
      df_plot[lower_col],
      df_plot[upper_col],
      color='pink',
      alpha=0.4,
      label='95% Confidence Interval',
  )

  if gt_col and gt_col in df_plot.columns:
    ax.plot(
        df_plot[date_col],
        df_plot[gt_col],
        label='Ground Truth KPI',
        color='blue',
        linewidth=2,
        linestyle='--',
    )
    _add_metrics_box(ax, df_plot, kpi_col, gt_col)

  if actual_y_col and actual_y_col in df_plot.columns:
    ax.plot(
        df_plot[date_col],
        df_plot[actual_y_col],
        label='Actual Total KPI',
        color='green',
        linewidth=2,
        linestyle=':',
    )

  ax.set_xlabel('Date', fontsize=18)
  _set_dynamic_xticks(ax, n_dates, is_date=True)
  plt.xticks(rotation=90, fontsize=12)
  ax.set_ylabel('KPI', fontsize=20)
  plt.yticks(fontsize=14)
  ax.set_title(title, fontsize=26)
  ax.legend(fontsize=16, loc='upper left')
  ax.grid(
      True,
      which='both',
      color='lightgrey',
      linestyle='-',
      linewidth=0.5,
      alpha=0.7,
  )
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  plt.close()


def plot_incremental_kpi_bar_stacked(
    df, date_col, kpi_cols, out_path, title, gt_col=None, actual_y_col=None
):
  """Plots a stacked bar chart showing estimated incremental KPI components over time."""
  title = _format_title(title, df, actual_y_col)

  agg_cols = kpi_cols.copy()
  if gt_col and gt_col in df.columns:
    agg_cols.append(gt_col)
  if actual_y_col and actual_y_col in df.columns:
    agg_cols.append(actual_y_col)

  df_plot = (
      df.groupby(date_col, as_index=False, observed=False)[agg_cols]
      .sum()
      .sort_values(date_col)
  )
  df_plot[date_col] = pd.to_datetime(df_plot[date_col])

  n_dates = len(df_plot[date_col].unique())
  dynamic_width = min(40, max(24, n_dates * 0.15))
  _, ax = plt.subplots(figsize=(dynamic_width, 12))

  dates_str = df_plot[date_col].dt.strftime('%Y-%m-%d')
  bottom_vals = np.zeros(len(df_plot))
  for col in kpi_cols:
    ax.bar(
        dates_str,
        df_plot[col],
        bottom=bottom_vals,
        label=col,
    )
    bottom_vals += df_plot[col].values

  if gt_col and gt_col in df_plot.columns:
    ax.plot(
        dates_str,
        df_plot[gt_col],
        label='Ground Truth KPI (Total)',
        color='blue',
        marker='o',
        linewidth=2,
    )
    df_plot['total_est_temp'] = df_plot[kpi_cols].sum(axis=1)
    _add_metrics_box(ax, df_plot, 'total_est_temp', gt_col)

  if actual_y_col and actual_y_col in df_plot.columns:
    ax.plot(
        dates_str,
        df_plot[actual_y_col],
        label='Actual Total KPI',
        color='green',
        linewidth=2,
        linestyle=':',
    )

  ax.set_xlabel('Date', fontsize=18)
  _set_dynamic_xticks(ax, n_dates, is_date=False)
  plt.xticks(rotation=90, fontsize=12)
  ax.set_ylabel('KPI', fontsize=20)
  plt.yticks(fontsize=14)
  ax.set_title(title, fontsize=26)
  ax.legend(fontsize=14, loc='upper left')
  ax.grid(
      True,
      which='both',
      color='lightgrey',
      linestyle='-',
      linewidth=0.5,
      alpha=0.7,
  )
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  plt.close()


def plot_roi_timeseries_bar(
    df,
    date_col,
    input_col,
    kpi_col,
    lower_col,
    upper_col,
    out_path,
    title,
    gt_col=None,
    actual_y_col=None,
):
  """Plots a multi-panel bar chart showing input, incremental KPI, and ROI over time."""
  title = _format_title(title, df, actual_y_col)

  agg_cols = [input_col, kpi_col, lower_col, upper_col]
  if gt_col and gt_col in df.columns:
    agg_cols.append(gt_col)
  df_plot = (
      df.groupby(date_col, as_index=False, observed=False)[agg_cols]
      .sum()
      .sort_values(date_col)
  )
  df_plot[date_col] = pd.to_datetime(df_plot[date_col])

  dates_str = df_plot[date_col].dt.strftime('%Y-%m-%d')
  inputs, kpis = df_plot[input_col].values, df_plot[kpi_col].values
  kpi_lowers, kpi_uppers = df_plot[lower_col].values, df_plot[upper_col].values

  rois = np.divide(kpis, inputs, out=np.zeros_like(kpis), where=inputs != 0)
  roi_err_lower = np.divide(
      kpis - kpi_lowers, inputs, out=np.zeros_like(kpis), where=inputs != 0
  )
  roi_err_upper = np.divide(
      kpi_uppers - kpis, inputs, out=np.zeros_like(kpis), where=inputs != 0
  )

  n_dates = len(df_plot[date_col].unique())
  dynamic_width = min(40, max(30, n_dates * 0.15))
  fig, axes = plt.subplots(3, 1, figsize=(dynamic_width, 24))
  fig.suptitle(title, fontsize=32, y=0.99)

  axes[0].bar(dates_str, inputs, color='steelblue')
  axes[0].set_ylabel('Total Input', fontsize=22)

  axes[1].bar(
      dates_str,
      kpis,
      yerr=[np.abs(kpis - kpi_lowers), np.abs(kpi_uppers - kpis)],
      capsize=4,
      color='mediumseagreen',
      error_kw={'alpha': 0.5},
  )
  axes[1].set_ylabel('Incremental KPI', fontsize=22)

  axes[2].bar(
      dates_str,
      rois,
      yerr=[np.abs(roi_err_lower), np.abs(roi_err_upper)],
      capsize=4,
      color='coral',
      error_kw={'alpha': 0.5},
  )
  axes[2].set_ylabel('ROI', fontsize=22)
  axes[2].set_xlabel('Date', fontsize=20)

  for ax in axes:
    ax.grid(
        True,
        which='both',
        color='lightgrey',
        linestyle='-',
        linewidth=0.5,
        alpha=0.7,
    )
    _set_dynamic_xticks(ax, n_dates, is_date=False)
    ax.tick_params(axis='x', rotation=90, labelsize=12, labelbottom=True)
    ax.tick_params(axis='y', labelsize=16)

  if gt_col and gt_col in df_plot.columns:
    gt_kpis = df_plot[gt_col].values
    gt_rois = np.divide(
        gt_kpis, inputs, out=np.zeros_like(gt_kpis), where=inputs != 0
    )
    axes[1].plot(
        dates_str,
        gt_kpis,
        color='blue',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Ground Truth KPI',
    )
    axes[1].legend(loc='upper left', fontsize=14)
    axes[2].plot(
        dates_str,
        gt_rois,
        color='blue',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Ground Truth ROI',
    )
    axes[2].legend(loc='upper left', fontsize=14)
    _add_metrics_box(axes[1], df_plot, kpi_col, gt_col)

  plt.tight_layout(rect=[0, 0.03, 1, 0.94])
  plt.savefig(out_path, dpi=200)
  plt.close()


def plot_entity_roi_comparison(
    df,
    entity_col,
    input_col,
    kpi_col,
    lower_col,
    upper_col,
    out_path,
    title,
    gt_col=None,
    actual_y_col=None,
):
  """Plots a multi-panel bar chart comparing input, incremental KPI, and ROI across entities."""
  title = _format_title(title, df, actual_y_col)

  agg_cols = [input_col, kpi_col, lower_col, upper_col]
  if gt_col and gt_col in df.columns:
    agg_cols.append(gt_col)
  df_plot = (
      df.groupby(entity_col, as_index=False, observed=False)[agg_cols]
      .sum()
      .sort_values(entity_col)
  )

  entities = df_plot[entity_col].astype(str).values
  inputs, kpis = df_plot[input_col].values, df_plot[kpi_col].values
  kpi_lowers, kpi_uppers = df_plot[lower_col].values, df_plot[upper_col].values

  rois = np.divide(kpis, inputs, out=np.zeros_like(kpis), where=inputs != 0)
  total_input, total_kpi = inputs.sum(), kpis.sum()
  total_roi = total_kpi / total_input if total_input > 0 else 0

  n_entities = len(entities)
  dynamic_width = min(40, max(30, n_entities * 0.6))
  fig, axes = plt.subplots(3, 1, figsize=(dynamic_width, 30))
  fig.suptitle(title, fontsize=32, y=0.99)

  axes[0].bar(entities, inputs, color='steelblue')
  axes[0].set_ylabel('Total Input', fontsize=22)

  axes[1].bar(
      entities,
      kpis,
      yerr=[np.abs(kpis - kpi_lowers), np.abs(kpi_uppers - kpis)],
      capsize=4,
      color='mediumseagreen',
      error_kw={'alpha': 0.5},
  )
  axes[1].set_ylabel('Incremental KPI', fontsize=22)

  axes[2].bar(
      entities,
      rois,
      yerr=[
          np.abs(
              np.divide(
                  kpis - kpi_lowers,
                  inputs,
                  out=np.zeros_like(kpis),
                  where=inputs != 0,
              )
          ),
          np.abs(
              np.divide(
                  kpi_uppers - kpis,
                  inputs,
                  out=np.zeros_like(kpis),
                  where=inputs != 0,
              )
          ),
      ],
      capsize=4,
      color='coral',
      error_kw={'alpha': 0.5},
  )
  axes[2].axhline(
      total_roi, color='red', linestyle='--', label='Overall Est. ROI'
  )
  axes[2].set_ylabel('ROI', fontsize=22)
  axes[2].set_xlabel(entity_col, fontsize=22)

  for ax in axes:
    ax.grid(
        True,
        which='both',
        color='lightgrey',
        linestyle='-',
        linewidth=0.5,
        alpha=0.7,
    )
    _set_dynamic_xticks(ax, n_entities, is_date=False)

    # Ensure we only set labels for the ticks that were kept
    ticks = ax.get_xticks()
    valid_ticks = [int(t) for t in ticks if 0 <= t < n_entities]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(
        [entities[i] for i in valid_ticks],
        rotation=60,
        ha='right',
        fontsize=16,
    )

    ax.tick_params(axis='y', labelsize=16)

  if gt_col and gt_col in df_plot.columns:
    gt_kpis = df_plot[gt_col].values
    gt_rois = np.divide(
        gt_kpis, inputs, out=np.zeros_like(gt_kpis), where=inputs != 0
    )
    total_gt_kpi = gt_kpis.sum()
    total_gt_roi = total_gt_kpi / total_input if total_input > 0 else 0

    axes[1].plot(
        entities,
        gt_kpis,
        color='blue',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Ground Truth KPI',
    )
    axes[2].plot(
        entities,
        gt_rois,
        color='blue',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Ground Truth ROI',
    )
    axes[2].axhline(
        total_gt_roi, color='blue', linestyle=':', label='Overall GT ROI'
    )
    _add_metrics_box(axes[1], df_plot, kpi_col, gt_col)

  for ax in axes:
    if ax.get_legend_handles_labels()[0]:
      ax.legend(loc='upper right', fontsize=14)

  plt.tight_layout(rect=[0, 0.08, 1, 0.94])
  plt.savefig(out_path, dpi=200)
  plt.close()


def plot_sensitivity_contour(
    dml_model,
    out_path: str,
    title: str,
    benchmark_covariates: Optional[List[str]] = None,
    override_theta: Optional[float] = None,
    override_se: Optional[float] = None,
):
  """Plots sensitivity contour lines for partial linear regression treatment effect analysis."""
  try:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      dml_model.sensitivity_analysis(
          cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0
      )

      theta = (
          override_theta if override_theta is not None else dml_model.coef[0]
      )
      se = override_se if override_se is not None else dml_model.se[0]
      m_val = np.abs(theta) - 1.96 * se
      rv = 0
      if m_val > 0:
        k_val = m_val / np.abs(theta)
        rv = (-(k_val**2) + np.sqrt(k_val**4 + 4 * k_val**2)) / 2

      bx, by = 0, 0
      if benchmark_covariates:
        try:
          bench_res = dml_model.sensitivity_benchmark(
              benchmarking_set=benchmark_covariates
          )
          bx = (
              bench_res['cf_d'].iloc[0]
              if isinstance(bench_res, pd.DataFrame)
              else bench_res.get('cf_d', [0])[0]
          )
          by = (
              bench_res['cf_y'].iloc[0]
              if isinstance(bench_res, pd.DataFrame)
              else bench_res.get('cf_y', [0])[0]
          )
        except (KeyError, AttributeError, ValueError):
          pass

      max_grid = max(0.1, rv * 1.5 if rv else 0.1, bx * 1.1, by * 1.1)
      max_grid = min(max_grid, 0.99)

      cf_d_grid = np.linspace(0, max_grid, 300)
      cf_y_grid = np.linspace(0, max_grid, 300)
      cf_d_mesh, cf_y_mesh = np.meshgrid(cf_d_grid, cf_y_grid)

      bias_contour = np.sqrt(
          cf_y_mesh * cf_d_mesh / (1 - cf_d_mesh + 1e-9)
      ) * np.abs(theta)
      adjusted_lower_bound = np.abs(theta) - bias_contour - (1.96 * se)

      z_max = np.nanmax(adjusted_lower_bound)
      z_min = np.percentile(adjusted_lower_bound, 2)
      if z_max == z_min:
        z_min = z_max - 1e-5

      plot_z = np.clip(adjusted_lower_bound, z_min, z_max)

      plt.figure(figsize=(14, 12))
      cp = plt.contour(
          cf_d_mesh,
          cf_y_mesh,
          plot_z,
          levels=15,
          cmap='coolwarm',
          alpha=0.6,
      )
      plt.clabel(cp, inline=True, fontsize=12)

      if (
          np.nanmin(adjusted_lower_bound)
          <= 0
          <= np.nanmax(adjusted_lower_bound)
      ):
        plt.contour(
            cf_d_mesh,
            cf_y_mesh,
            adjusted_lower_bound,
            levels=[0],
            colors='red',
            linestyles='--',
            linewidths=3,
        )

      plt.plot([0, max_grid], [0, max_grid], 'b:', linewidth=2, label='y = x')

      if m_val > 0:
        plt.plot(
            rv,
            rv,
            marker='*',
            color='gold',
            markersize=15,
            markeredgecolor='black',
            label=f'Robustness Value: {rv:.3f}',
        )

      if benchmark_covariates and (bx > 0 or by > 0):
        plt.plot(
            bx,
            by,
            marker='D',
            markersize=12,
            color='tab:blue',
            label=f'Benchmark: {benchmark_covariates[0]}',
            linestyle='None',
        )

      plt.xlim(0, max_grid)
      plt.ylim(0, max_grid)
      plt.title(
          f'Sensitivity Contour Plot - {title}\nLower Limit of 95% Confidence'
          ' Bound',
          fontsize=18,
      )
      plt.xlabel('Treatment Partial R^2 (cf_d)', fontsize=16)
      plt.ylabel('Outcome Partial R^2 (cf_y)', fontsize=16)
      plt.grid(True, linestyle=':', alpha=0.6)
      plt.legend(loc='upper right', fontsize=14)

      def fmt_sci(v):
        return f'{v:.2e}' if abs(v) < 0.01 and v != 0 else f'{v:.3f}'

      props = dict(boxstyle='round', facecolor='white', alpha=0.9)
      plt.text(
          0.05,
          0.95,
          (
              f'Estimated GATE/ATE: {fmt_sci(theta)}\n'
              f'Robustness Value (RV): {fmt_sci(rv)}'
          ),
          transform=plt.gca().transAxes,
          fontsize=16,
          verticalalignment='top',
          bbox=props,
      )
      plt.tight_layout()
      plt.savefig(out_path, dpi=200)
      plt.close()

      return {
          'theta': theta,
          'se': se,
          'robustness_value': rv,
          'benchmark_cf_d': bx,
          'benchmark_cf_y': by,
      }
  except (ValueError, KeyError, AttributeError, TypeError, RuntimeError) as e:
    print(f'    [Warning] Contour generation skipped for {title}: {e}')
    return None


def plot_cate_scatter_matrix(
    df_1a_consolidated,
    d_cols,
    cont_cols,
    bin_cols,
    eq_str_dict,
    out_path,
    cate_stats_dict=None,
):
  """Plots a scatter matrix showing CATE estimations against covariates."""
  n_rows = len(d_cols) + 1
  n_cols = len(cont_cols) if cont_cols else 1

  fig_height = max(24, 10 * n_rows)
  fig, axes = plt.subplots(
      n_rows, n_cols, figsize=(12 * n_cols + 20, fig_height), squeeze=False
  )
  fig.suptitle(
      'CATE Scatter Matrix: Treatment vs Incremental Effect',
      fontsize=24,
      wrap=True,
      y=0.98,
  )

  patterns = []
  if bin_cols:
    patterns = []
    for _, row in df_1a_consolidated[bin_cols].drop_duplicates().iterrows():
      pattern_str = ', '.join([f'{col}={row[col]}' for col in bin_cols])
      patterns.append((pattern_str, row.to_dict()))
    unique_colors = [p[0] for p in patterns]
  else:
    unique_colors = ['All Data']

  cmap = plt.get_cmap('tab10')
  color_map = {val: cmap(i % 10) for i, val in enumerate(unique_colors)}

  for i, d_col in enumerate(d_cols):
    kpi_col = f'estimated_incremental_KPI_{d_col}'
    for j, cont_col in enumerate(cont_cols or ['dummy']):
      ax = axes[i, j]
      x_data = df_1a_consolidated[d_col].values
      y_data = df_1a_consolidated[kpi_col].values
      z_data = (
          df_1a_consolidated[cont_col].values
          if cont_cols
          else np.zeros_like(x_data)
      )

      if (
          cont_cols
          and len(np.unique(x_data)) > 2
          and len(np.unique(y_data)) > 2
      ):
        try:
          x_j = x_data + np.random.normal(0, 1e-8, size=len(x_data))
          y_j = y_data + np.random.normal(0, 1e-8, size=len(y_data))
          contour = ax.tricontourf(
              x_j, y_j, z_data, levels=20, cmap='coolwarm', alpha=0.4
          )
          if j == len(cont_cols) - 1:
            fig.colorbar(contour, ax=ax, label=f'Covariate: {cont_col}')
        except (ValueError, AttributeError, RuntimeError):
          pass

      for c_val in unique_colors:
        if bin_cols:
          pattern_dict = next(p[1] for p in patterns if p[0] == c_val)
          idx = np.ones(len(x_data), dtype=bool)
          for k, v in pattern_dict.items():
            idx &= (df_1a_consolidated[k] == v).values
        else:
          idx = np.ones(len(x_data), dtype=bool)

        if not idx.any():
          continue
        ax.scatter(
            x_data[idx],
            y_data[idx],
            color=color_map[c_val],
            alpha=0.9,
            label=c_val,
            edgecolors='white',
            s=60,
        )

      ax.set_xlabel(f'Treatment: {d_col}', fontsize=14)
      ax.set_ylabel(f'Inc. Effect: {kpi_col}', fontsize=14)
      if cont_cols:
        ax.set_title(f'Grouped by: {cont_col}', fontsize=16)
      ax.grid(True, linestyle=':', alpha=0.5)

  total_input = (
      df_1a_consolidated[[
          c for c in d_cols if not re.search(r'_l\d+$', c, flags=re.IGNORECASE)
      ]]
      .sum(axis=1)
      .values
  )
  total_kpi = df_1a_consolidated['total_estimated_incremental_KPI'].values

  for j, cont_col in enumerate(cont_cols or ['dummy']):
    ax = axes[-1, j]
    y_data = total_kpi
    z_data = (
        df_1a_consolidated[cont_col].values
        if cont_cols
        else np.zeros_like(total_input)
    )

    if (
        cont_cols
        and len(np.unique(total_input)) > 2
        and len(np.unique(y_data)) > 2
    ):
      try:
        t_j = total_input + np.random.normal(0, 1e-8, size=len(total_input))
        y_j = y_data + np.random.normal(0, 1e-8, size=len(y_data))
        contour = ax.tricontourf(
            t_j, y_j, z_data, levels=20, cmap='coolwarm', alpha=0.4
        )
        if j == len(cont_cols) - 1:
          fig.colorbar(contour, ax=ax, label=f'Covariate: {cont_col}')
      except (ValueError, AttributeError, RuntimeError):
        pass

    for c_val in unique_colors:
      if bin_cols:
        pattern_dict = next(p[1] for p in patterns if p[0] == c_val)
        idx = np.ones(len(total_input), dtype=bool)
        for k, v in pattern_dict.items():
          idx &= (df_1a_consolidated[k] == v).values
      else:
        idx = np.ones(len(total_input), dtype=bool)

      if not idx.any():
        continue
      ax.scatter(
          total_input[idx],
          y_data[idx],
          color=color_map[c_val],
          alpha=0.9,
          edgecolors='white',
          s=60,
      )

    ax.set_xlabel('Total Treatment (Excl. Lags)', fontsize=14)
    ax.set_ylabel('Total Incremental KPI', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)

  plt.tight_layout(rect=[0, 0.03, 0.55, 0.95])

  ax_right = fig.add_axes([0.58, 0.05, 0.40, 0.90])
  ax_right.axis('off')

  max_chars = 68

  handles, labels = axes[0, 0].get_legend_handles_labels()

  eq_lines = ['Estimated CATE Functions:', '-' * max_chars, '']
  for d_col, eq_str in eq_str_dict.items():
    wrapped_eq = textwrap.wrap(eq_str, width=max_chars)
    eq_lines.append(f'[{d_col}]')
    eq_lines.extend(wrapped_eq)
    eq_lines.append('')

  note_text = (
      'Note: Interpretation of coefficients of CATE function should be'
      ' directional as they are estimated based on normalised covariates.'
  )
  eq_lines.append('-' * max_chars)
  eq_lines.extend(textwrap.wrap(note_text, width=max_chars))

  eq_text = '\n'.join([line.ljust(max_chars) for line in eq_lines])

  stats_lines = ['P-value & Std Err:', '-' * max_chars, '']
  if cate_stats_dict:
    for d_col, cate_stats in cate_stats_dict.items():
      stats_lines.append(f'[{d_col}]')
      stats_lines.append(
          f'{"Covariate":<35} {"Coef":>10} {"P-val":>10} {"StdErr":>10}'
      )
      coefs = cate_stats.get('coef', {})
      pvals = cate_stats.get('pvalue', {})
      stderrs = cate_stats.get('stderr', {})

      def format_val(val):
        return str(val) if isinstance(val, str) else f'{val:.3f}'

      for cov in pvals.keys():
        coef_val = coefs.get(cov, np.nan)
        pval_val = pvals[cov]
        stderr_val = stderrs.get(cov, np.nan)

        c_str = format_val(coef_val)
        p_str = format_val(pval_val)
        s_str = format_val(stderr_val)

        cov_wrap = textwrap.wrap(cov, width=35)
        if not cov_wrap:
          cov_wrap = ['']
        stats_lines.append(
            f'{cov_wrap[0]:<35} {c_str:>10} {p_str:>10} {s_str:>10}'
        )
        for extra_line in cov_wrap[1:]:
          stats_lines.append(f'{extra_line:<35} {"":>10} {"":>10} {"":>10}')
      stats_lines.append('')

  stats_text = '\n'.join([line.ljust(max_chars) for line in stats_lines])

  eq_lines_count = len(eq_lines)
  stats_lines_count = len(stats_lines)
  legend_lines = (len(labels) * 2 + 2) if handles else 0

  total_lines = legend_lines + eq_lines_count + stats_lines_count
  line_height_pt = 0.35
  required_height = total_lines * line_height_pt * 1.5 + 4
  if required_height > fig_height:
    fig_height = required_height
    fig.set_figheight(fig_height)

  line_height_norm = line_height_pt / fig_height

  y_curr = 1.0
  if handles:
    padded_labels = [lbl.ljust(max_chars - 4) for lbl in labels]
    leg = ax_right.legend(
        handles,
        padded_labels,
        loc='upper left',
        bbox_to_anchor=(0.0, y_curr),
        title='Binary Covariates',
        prop={'family': 'monospace', 'size': 12},
        facecolor='lightgray',
        edgecolor='black',
        borderpad=0.5,
        fancybox=True,
    )
    plt.setp(leg.get_title(), family='monospace', fontsize=14)

    y_curr -= (legend_lines * line_height_norm) + 0.01

  box_format = dict(
      fontsize=12,
      family='monospace',
      linespacing=1.5,
      verticalalignment='top',
      bbox=dict(
          boxstyle='round,pad=0.5',
          facecolor='lightgray',
          edgecolor='black',
      ),
  )

  ax_right.text(
      0.0, y_curr, eq_text, transform=ax_right.transAxes, **box_format
  )

  y_curr -= (eq_lines_count * line_height_norm * 1.5) + 0.02

  ax_right.text(
      0.0, y_curr, stats_text, transform=ax_right.transAxes, **box_format
  )

  plt.savefig(out_path, dpi=200)
  plt.close()


def plot_prior_distributions(roi_array, contrib_array, out_path, d_col_name):
  """Plots fit prior distributions (Normal, LogNormal, Beta) for ROI and contribution ratios."""
  fig, axes = plt.subplots(1, 2, figsize=(24, 10))
  fig.suptitle(f'Prior Distributions - {d_col_name}', fontsize=28, y=0.98)

  def fmt(v):
    return f'{v:.2e}' if abs(v) < 0.001 and v != 0 else f'{v:.3f}'

  def _fit_and_plot(ax, data, title, is_contrib):
    data = data[np.isfinite(data)]
    if len(data) == 0:
      ax.text(0.5, 0.5, 'No valid data', ha='center', fontsize=16)
      return

    ax.hist(
        data, bins=30, density=True, alpha=0.5, color='gray', edgecolor='black'
    )
    x_min, x_max = np.min(data), np.max(data)
    x_seq = np.linspace(
        x_min - 0.2 * np.abs(x_min), x_max + 0.2 * np.abs(x_max), 200
    )

    mu, std = stats.norm.fit(data)
    ax.plot(
        x_seq,
        stats.norm.pdf(x_seq, mu, std),
        'k-',
        linewidth=2,
        label=rf'Normal ($\mu$={fmt(mu)}, $\sigma$={fmt(std)})',
    )

    valid_log = data[data > 0]
    if len(valid_log) > 0:
      shape, loc, scale = stats.lognorm.fit(valid_log, floc=0)
      ax.plot(
          x_seq,
          stats.lognorm.pdf(x_seq, shape, loc, scale),
          'b--',
          linewidth=2,
          label=(
              rf'LogNormal ($\mu$={fmt(np.log(scale))}, $\sigma$={fmt(shape)})'
          ),
      )

    if is_contrib:
      valid_beta = data[(data > 0) & (data < 1)]
      if len(valid_beta) > 10:
        a, b, loc, scale = stats.beta.fit(valid_beta, floc=0, fscale=1)
        ax.plot(
            x_seq,
            stats.beta.pdf(x_seq, a, b, loc, scale),
            'r:',
            linewidth=2,
            label=f'Beta ($\\alpha$={fmt(a)}, $\\beta$={fmt(b)})',
        )

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Value', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

  _fit_and_plot(
      axes[0],
      contrib_array,
      'Total Contribution (Inc. KPI / KPI)',
      is_contrib=True,
  )
  _fit_and_plot(
      axes[1],
      roi_array,
      'Total ROI (Inc. KPI / Input)',
      is_contrib=False,
  )

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.savefig(out_path, dpi=200)
  plt.close()


def generate_roi_charts(
    df: pd.DataFrame,
    date_col: str,
    geo_col: str,
    sales_col: str,
    spend_col: str,
    est_col: str,
    out_dir: str,
    peak_months: Optional[List[int]] = None,
    item_col: Optional[str] = None,
) -> None:
  """Generates promo analysis charts matching reference notebook formatting specifications."""
  if peak_months is None:
    peak_months = [3, 7, 11, 12]

  df_work = df.copy()
  if date_col in df_work.columns:
    df_work[date_col] = pd.to_datetime(df_work[date_col])

  lower_col = f'{est_col}_2.5%'
  upper_col = f'{est_col}_97.5%'
  has_ci = lower_col in df_work.columns and upper_col in df_work.columns

  years = (
      sorted(df_work[date_col].dt.year.unique())[-3:]
      if date_col in df_work.columns
      else [2024]
  )

  levels_list = [('total', None)]
  if geo_col and geo_col in df_work.columns:
    for g in sorted(df_work[geo_col].dropna().unique()):
      levels_list.append(('geo', g))
  if item_col and item_col in df_work.columns:
    for i in sorted(df_work[item_col].dropna().unique()):
      levels_list.append(('item', i))

  # -------------------------------------------------------------------
  # 1. Subdir: 001_time_series_charts
  # -------------------------------------------------------------------
  dir1 = os.path.join(out_dir, '001_time_series_charts')
  os.makedirs(dir1, exist_ok=True)

  agg_dict = {
      col: 'sum'
      for col in [sales_col, spend_col, est_col]
      if col in df_work.columns
  }
  if has_ci:
    if lower_col in df_work.columns:
      agg_dict[lower_col] = 'sum'
    if upper_col in df_work.columns:
      agg_dict[upper_col] = 'sum'

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
        raise ValueError(f'Unknown level_type: {level_type}')

      if chart_df.empty or not agg_dict:
        continue

      df_ts = (
          chart_df.groupby(date_col, as_index=False, observed=False)
          .agg(agg_dict)
          .sort_values(date_col)
      )

      # Chart A (Sales & Spend)
      _, ax = plt.subplots(figsize=(24, 10))
      if sales_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[sales_col],
            color='blue',
            alpha=0.8,
            linewidth=2,
            label=f'Actual Sales ({sales_col})',
        )
      if spend_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[spend_col],
            color='purple',
            alpha=0.8,
            linewidth=3,
            label=f'Spend ({spend_col})',
        )
      if est_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[est_col],
            color='red',
            linewidth=2,
            label=f'Est. Incremental Sales ({est_col})',
        )
      if has_ci and lower_col in df_ts.columns and upper_col in df_ts.columns:
        ax.fill_between(
            df_ts[date_col],
            df_ts[lower_col],
            df_ts[upper_col],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval',
        )
      ax.set_title(
          f'{title_prefix} Time Series: Sales, Spend, and Estimated Incremental'
          ' Sales',
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
      plt.savefig(
          os.path.join(dir1, f'{prefix}_timeseries_sales_and_spend.png'),
          dpi=200,
      )
      plt.close()

      # Chart B (Spend vs Est)
      _, ax = plt.subplots(figsize=(24, 10))
      if spend_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[spend_col],
            color='purple',
            alpha=0.8,
            linewidth=2,
            label=f'Spend ({spend_col})',
        )
      if est_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[est_col],
            color='red',
            linewidth=2,
            label=f'Est. Incremental Sales ({est_col})',
        )
      if has_ci and lower_col in df_ts.columns and upper_col in df_ts.columns:
        ax.fill_between(
            df_ts[date_col],
            df_ts[lower_col],
            df_ts[upper_col],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval',
        )
      ax.set_title(
          f'{title_prefix} Time Series: Spend vs. Estimated Incremental Sales',
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
      plt.savefig(
          os.path.join(dir1, f'{prefix}_timeseries_spend_vs_est.png'), dpi=200
      )
      plt.close()

  # -------------------------------------------------------------------
  # 2. Subdir: 002_result_in_sale_periods (Time Series Matrix)
  # -------------------------------------------------------------------
  dir2 = os.path.join(out_dir, '002_result_in_sale_periods')
  os.makedirs(dir2, exist_ok=True)

  if date_col in df_work.columns:
    for level_type, level_val in levels_list:
      if level_type == 'total':
        sub_df_all = df_work
        prefix = 'total'
        title_prefix = 'Total'
      elif level_type == 'geo':
        sub_df_all = df_work[df_work[geo_col] == level_val]
        prefix = f'geo_{level_val}'
        title_prefix = f'[{level_val}]'
      elif level_type == 'item':
        sub_df_all = df_work[df_work[item_col] == level_val]
        prefix = f'item_{level_val}'
        title_prefix = f'[{level_val}]'
      else:
        raise ValueError(f'Unknown level_type: {level_type}')

      if sub_df_all.empty:
        continue

      matrix_agg_dict = {
          sales_col: 'sum',
          spend_col: 'sum',
          est_col: 'sum',
      }
      if has_ci:
        matrix_agg_dict[lower_col] = 'sum'
        matrix_agg_dict[upper_col] = 'sum'

      nrows = len(years)
      ncols = len(peak_months)
      fig, axes = plt.subplots(
          nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False
      )
      fig.suptitle(
          f'{title_prefix} Promo Results in Peak Sale Periods (By Month &'
          ' Year)',
          fontsize=24,
      )

      for row_idx, y in enumerate(years):
        for col_idx, m in enumerate(peak_months):
          ax = axes[row_idx, col_idx]
          sub_df = sub_df_all[
              (sub_df_all[date_col].dt.year == y)
              & (sub_df_all[date_col].dt.month == m)
          ]
          if sub_df.empty:
            ax.set_title(f'Year {y} - Month {m}', fontsize=18)
            continue

          sub_ts = (
              sub_df.groupby(date_col, as_index=False, observed=False)
              .agg(matrix_agg_dict)
              .sort_values(date_col)
          )

          ax.plot(
              sub_ts[date_col],
              sub_ts[sales_col],
              color='blue',
              linewidth=2,
              label='Sales',
          )
          ax.plot(
              sub_ts[date_col],
              sub_ts[spend_col],
              color='purple',
              linewidth=3,
              label='Spend',
          )
          ax.plot(
              sub_ts[date_col],
              sub_ts[est_col],
              color='red',
              linewidth=2,
              label='Inc Sales',
          )
          if has_ci:
            ax.fill_between(
                sub_ts[date_col],
                sub_ts[lower_col],
                sub_ts[upper_col],
                color='red',
                alpha=0.2,
                label='95% CI',
            )

          for d_val in sub_ts[date_col]:
            if d_val.weekday() >= 5:
              ax.axvspan(
                  d_val - pd.Timedelta(hours=12),
                  d_val + pd.Timedelta(hours=12),
                  color='grey',
                  alpha=0.3,
              )

          ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
          ax.tick_params(axis='x', rotation=90, labelsize=8)
          ax.tick_params(axis='y', labelsize=10)
          ax.set_ylabel('Amount', fontsize=16)
          ax.grid(True, alpha=0.4)

          ax2 = ax.twinx()
          promo_pct = np.where(
              sub_ts[sales_col] == 0, 0.0, sub_ts[spend_col] / sub_ts[sales_col]
          )
          inc_pct = np.where(
              sub_ts[sales_col] == 0, 0.0, sub_ts[est_col] / sub_ts[sales_col]
          )

          width = pd.Timedelta(hours=10)
          ax2.bar(
              sub_ts[date_col] - pd.Timedelta(hours=5),
              promo_pct,
              width=width,
              color='red',
              alpha=0.3,
              label='Promo %',
          )
          ax2.bar(
              sub_ts[date_col] + pd.Timedelta(hours=5),
              inc_pct,
              width=width,
              color='blue',
              alpha=0.3,
              label='Inc Sales %',
          )
          ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
          ax2.set_ylabel('Percentage (%)', fontsize=16)
          ax2.tick_params(axis='y', labelsize=10)

          ax.set_title(f'Year {y} - Month {m}', fontsize=18)
          h1, l1 = ax.get_legend_handles_labels()
          h2, l2 = ax2.get_legend_handles_labels()
          ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.savefig(
          os.path.join(dir2, f'{prefix}_result_in_sale_periods_matrix.png'),
          dpi=200,
      )
      plt.close()

  # -------------------------------------------------------------------
  # 3. Subdir: 003_roi_by_month (Monthly Stats Matrix)
  # -------------------------------------------------------------------
  dir3 = os.path.join(out_dir, '003_roi_by_month')
  os.makedirs(dir3, exist_ok=True)

  if date_col in df_work.columns:
    for level_type, level_val in levels_list:

      if level_type == 'total':
        sub_df_all = df_work
        prefix = 'total'
        title_prefix = 'Total'
      elif level_type == 'geo':
        sub_df_all = df_work[df_work[geo_col] == level_val]
        prefix = f'geo_{level_val}'
        title_prefix = f'[{level_val}]'
      elif level_type == 'item':
        sub_df_all = df_work[df_work[item_col] == level_val]
        prefix = f'item_{level_val}'
        title_prefix = f'[{level_val}]'
      else:
        raise ValueError(f'Unknown level_type: {level_type}')

      if sub_df_all.empty:
        continue

      nrows = len(years)
      ncols = 3
      fig, axes = plt.subplots(
          nrows, ncols, figsize=(24, 5 * nrows), sharey='col', squeeze=False
      )
      fig.suptitle(
          f'{title_prefix} Promo ROI = Incremental KPI / Promo Spend and'
          ' Percentage Metrics by Month',
          fontsize=20,
      )

      for row_idx, y in enumerate(years):
        year_df = sub_df_all[sub_df_all[date_col].dt.year == y]
        months = np.arange(1, 13)
        m_spend = np.zeros(12)
        m_sales = np.zeros(12)
        m_est = np.zeros(12)

        if not year_df.empty:
          m_agg = year_df.groupby(
              year_df[date_col].dt.month, observed=False
          ).agg({
              sales_col: 'sum',
              spend_col: 'sum',
              est_col: 'sum',
          })
          for m_val in m_agg.index:
            if 1 <= m_val <= 12:
              m_spend[m_val - 1] = m_agg.loc[m_val, spend_col]
              m_sales[m_val - 1] = m_agg.loc[m_val, sales_col]
              m_est[m_val - 1] = m_agg.loc[m_val, est_col]
        promo_spend_pct = np.divide(
            m_spend,
            m_sales,
            out=np.zeros_like(m_spend, dtype=float),
            where=m_sales != 0,
        )
        inc_sales_pct = np.divide(
            m_est,
            m_sales,
            out=np.zeros_like(m_est, dtype=float),
            where=m_sales != 0,
        )
        promo_roi = np.divide(
            m_est,
            m_spend,
            out=np.zeros_like(m_est, dtype=float),
            where=m_spend != 0,
        )

        y_total_spend = m_spend.sum()
        y_total_sales = m_sales.sum()
        y_total_est = m_est.sum()

        y_total_spend_pct = (
            y_total_spend / y_total_sales if y_total_sales > 0 else 0.0
        )
        y_total_inc_pct = (
            y_total_est / y_total_sales if y_total_sales > 0 else 0.0
        )
        y_total_roi = y_total_est / y_total_spend if y_total_spend > 0 else 0.0

        colors = ['orange' if (m in peak_months) else 'skyblue' for m in months]

        # Col 0: Promo Spend % of Sales
        ax0 = axes[row_idx, 0]
        ax0.bar(months, promo_spend_pct, color=colors)
        ax0.axhline(
            y_total_spend_pct,
            color='darkblue',
            linestyle=':',
            linewidth=2,
            label=f'Yearly Total ({y_total_spend_pct:.1%})',
        )
        ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax0.set_title(f'Year {y}: Promo Spend % of Sales', fontsize=16)
        ax0.set_ylabel(f'Year {y} - Pct', fontsize=16)
        ax0.set_xlabel('Month', fontsize=14)
        ax0.bar(months[0], 0, color='orange', label='Sales Peak Months')
        ax0.legend(loc='upper left', fontsize=10)
        ax0.grid(True, alpha=0.3)

        # Col 1: Incremental Sales % of Sales
        ax1 = axes[row_idx, 1]
        ax1.bar(months, inc_sales_pct, color=colors)
        ax1.axhline(
            y_total_inc_pct,
            color='darkblue',
            linestyle=':',
            linewidth=2,
            label=f'Yearly Total ({y_total_inc_pct:.1%})',
        )
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.set_title(f'Year {y}: Incremental Sales % of Sales', fontsize=16)
        ax1.set_ylabel(f'Year {y} - Pct', fontsize=16)
        ax1.set_xlabel('Month', fontsize=14)
        ax1.bar(months[0], 0, color='orange', label='Sales Peak Months')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Col 2: Promo ROI
        ax2 = axes[row_idx, 2]
        ax2.bar(months, promo_roi, color=colors)
        ax2.axhline(
            y_total_roi,
            color='darkblue',
            linestyle=':',
            linewidth=2,
            label=f'Yearly Total ({y_total_roi:.2f})',
        )
        ax2.set_title(
            f'Year {y}: Promo ROI = Incremental KPI / Promo Spend', fontsize=16
        )
        ax2.set_ylabel(f'Year {y} - ROI', fontsize=16)
        ax2.set_xlabel('Month', fontsize=14)
        ax2.bar(months[0], 0, color='orange', label='Sales Peak Months')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.savefig(
          os.path.join(dir3, f'{prefix}_roi_by_month_matrix.png'), dpi=200
      )
      plt.close()

  # -------------------------------------------------------------------
  # 4. Subdir: 004_roi_by_geo (Geo Stats Matrix)
  # -------------------------------------------------------------------
  dir4 = os.path.join(out_dir, '004_roi_by_geo')
  os.makedirs(dir4, exist_ok=True)

  if date_col in df_work.columns:
    months_list = [None] + list(range(1, 13))
    for month_val in months_list:
      if month_val is None:
        month_df = df_work
        prefix = 'total'
        title_prefix = 'Total (All Months)'
      else:
        month_df = df_work[df_work[date_col].dt.month == month_val]
        prefix = f'month_{month_val}'
        title_prefix = f'Month {month_val}'

      if month_df.empty:
        continue

      nrows = len(years)
      ncols = 3
      fig, axes = plt.subplots(
          nrows, ncols, figsize=(28, 6 * nrows), sharey='col', squeeze=False
      )
      fig.suptitle(
          f'[{title_prefix}] Promo ROI = Incremental KPI / Promo Spend and'
          ' Percentage Metrics by Geo',
          fontsize=24,
      )

      for row_idx, y in enumerate(years):
        year_df = month_df[month_df[date_col].dt.year == y]
        if geo_col and geo_col in year_df.columns:
          geo_agg = (
              year_df.groupby(geo_col, as_index=False, observed=False)
              .agg({sales_col: 'sum', spend_col: 'sum', est_col: 'sum'})
              .sort_values(geo_col)
          )
          geos = geo_agg[geo_col].astype(str).values
          geo_spend = geo_agg[spend_col].values
          geo_sales = geo_agg[sales_col].values
          geo_est = geo_agg[est_col].values
        else:
          geos = np.array(['All'])
          geo_spend = np.array(
              [year_df[spend_col].sum()] if not year_df.empty else [0.0]
          )
          geo_sales = np.array(
              [year_df[sales_col].sum()] if not year_df.empty else [0.0]
          )
          geo_est = np.array(
              [year_df[est_col].sum()] if not year_df.empty else [0.0]
          )
        geo_spend_pct = np.divide(
            geo_spend,
            geo_sales,
            out=np.zeros_like(geo_spend, dtype=float),
            where=geo_sales != 0,
        )
        geo_inc_pct = np.divide(
            geo_est,
            geo_sales,
            out=np.zeros_like(geo_est, dtype=float),
            where=geo_sales != 0,
        )
        geo_roi = np.divide(
            geo_est,
            geo_spend,
            out=np.zeros_like(geo_est, dtype=float),
            where=geo_spend != 0,
        )

        o_total_spend = geo_spend.sum()
        o_total_sales = geo_sales.sum()
        o_total_est = geo_est.sum()

        o_spend_pct = (
            o_total_spend / o_total_sales if o_total_sales > 0 else 0.0
        )
        o_inc_pct = o_total_est / o_total_sales if o_total_sales > 0 else 0.0
        o_roi = o_total_est / o_total_spend if o_total_spend > 0 else 0.0

        wrapped_geos = [textwrap.fill(str(g), width=15) for g in geos]
        x_pos = np.arange(len(geos))

        # Col 0: Promo Spend % of Sales
        ax0 = axes[row_idx, 0]
        ax0.bar(x_pos, geo_spend_pct, color='darkblue', alpha=0.6)
        ax0.axhline(
            o_spend_pct,
            color='darkorange',
            linestyle=':',
            linewidth=3,
            label=f'Overall Total ({o_spend_pct:.1%})',
        )
        ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(wrapped_geos, rotation=90, ha='center', fontsize=12)
        ax0.set_title(f'Year {y}: Promo Spend % of Sales', fontsize=18)
        ax0.set_ylabel('Percentage (%)', fontsize=16)
        ax0.set_xlabel(geo_col if geo_col else 'Geo', fontsize=16)
        ax0.legend(loc='upper right', fontsize=12)
        ax0.grid(True, alpha=0.3)

        # Col 1: Incremental Sales % of Sales
        ax1 = axes[row_idx, 1]
        ax1.bar(x_pos, geo_inc_pct, color='darkblue', alpha=0.6)
        ax1.axhline(
            o_inc_pct,
            color='darkorange',
            linestyle=':',
            linewidth=3,
            label=f'Overall Total ({o_inc_pct:.1%})',
        )
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(wrapped_geos, rotation=90, ha='center', fontsize=12)
        ax1.set_title(f'Year {y}: Incremental Sales % of Sales', fontsize=18)
        ax1.set_ylabel('Percentage (%)', fontsize=16)
        ax1.set_xlabel(geo_col if geo_col else 'Geo', fontsize=16)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Col 2: Promo ROI
        ax2 = axes[row_idx, 2]
        ax2.bar(x_pos, geo_roi, color='darkblue', alpha=0.6)
        ax2.axhline(
            o_roi,
            color='darkorange',
            linestyle=':',
            linewidth=3,
            label=f'Overall Total ({o_roi:.2f})',
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(wrapped_geos, rotation=90, ha='center', fontsize=12)
        ax2.set_title(
            f'Year {y}: Promo ROI = Incremental KPI / Promo Spend', fontsize=18
        )
        ax2.set_ylabel('ROI', fontsize=16)
        ax2.set_xlabel(geo_col if geo_col else 'Geo', fontsize=16)
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.savefig(
          os.path.join(dir4, f'{prefix}_roi_by_geo_matrix.png'), dpi=200
      )
      plt.close()
  # -------------------------------------------------------------------
  # 5. Subdir: 005_roi_by_item (Item Stats Matrix)
  # -------------------------------------------------------------------
  if item_col and item_col in df_work.columns:
    dir5 = os.path.join(out_dir, '005_roi_by_item')
    os.makedirs(dir5, exist_ok=True)

    if date_col in df_work.columns:
      months_list = [None] + list(range(1, 13))
      for month_val in months_list:
        if month_val is None:
          month_df = df_work
          prefix = 'total'
          title_prefix = 'Total (All Months)'
        else:
          month_df = df_work[df_work[date_col].dt.month == month_val]
          prefix = f'month_{month_val}'
          title_prefix = f'Month {month_val}'

        if month_df.empty:
          continue

        nrows = len(years)
        ncols = 3
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(28, 6 * nrows), sharey='col', squeeze=False
        )
        fig.suptitle(
            f'[{title_prefix}] Promo ROI = Incremental KPI / Promo Spend and'
            ' Percentage Metrics by Item',
            fontsize=24,
        )

        for row_idx, y in enumerate(years):
          year_df = month_df[month_df[date_col].dt.year == y]
          item_agg = (
              year_df.groupby(item_col, as_index=False, observed=False)
              .agg({sales_col: 'sum', spend_col: 'sum', est_col: 'sum'})
              .sort_values(item_col)
          )
          items = item_agg[item_col].astype(str).values
          item_spend = item_agg[spend_col].values
          item_sales = item_agg[sales_col].values
          item_est = item_agg[est_col].values

          item_spend_pct = np.divide(
              item_spend,
              item_sales,
              out=np.zeros_like(item_spend, dtype=float),
              where=item_sales != 0,
          )
          item_inc_pct = np.divide(
              item_est,
              item_sales,
              out=np.zeros_like(item_est, dtype=float),
              where=item_sales != 0,
          )
          item_roi = np.divide(
              item_est,
              item_spend,
              out=np.zeros_like(item_est, dtype=float),
              where=item_spend != 0,
          )

          o_total_spend = item_spend.sum()
          o_total_sales = item_sales.sum()
          o_total_est = item_est.sum()

          o_spend_pct = (
              o_total_spend / o_total_sales if o_total_sales > 0 else 0.0
          )
          o_inc_pct = o_total_est / o_total_sales if o_total_sales > 0 else 0.0
          o_roi = o_total_est / o_total_spend if o_total_spend > 0 else 0.0

          wrapped_items = [
              textwrap.fill(str(i_val), width=15) for i_val in items
          ]
          x_pos = np.arange(len(items))

          # Col 0: Promo Spend % of Sales
          ax0 = axes[row_idx, 0]
          ax0.bar(x_pos, item_spend_pct, color='darkblue', alpha=0.6)
          ax0.axhline(
              o_spend_pct,
              color='darkorange',
              linestyle=':',
              linewidth=3,
              label=f'Overall Total ({o_spend_pct:.1%})',
          )
          ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
          ax0.set_xticks(x_pos)
          ax0.set_xticklabels(
              wrapped_items, rotation=90, ha='center', fontsize=12
          )
          ax0.set_title(f'Year {y}: Promo Spend % of Sales', fontsize=18)
          ax0.set_ylabel('Percentage (%)', fontsize=16)
          ax0.set_xlabel(item_col, fontsize=16)
          ax0.legend(loc='upper right', fontsize=12)
          ax0.grid(True, alpha=0.3)

          # Col 1: Incremental Sales % of Sales
          ax1 = axes[row_idx, 1]
          ax1.bar(x_pos, item_inc_pct, color='darkblue', alpha=0.6)
          ax1.axhline(
              o_inc_pct,
              color='darkorange',
              linestyle=':',
              linewidth=3,
              label=f'Overall Total ({o_inc_pct:.1%})',
          )
          ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
          ax1.set_xticks(x_pos)
          ax1.set_xticklabels(
              wrapped_items, rotation=90, ha='center', fontsize=12
          )
          ax1.set_title(f'Year {y}: Incremental Sales % of Sales', fontsize=18)
          ax1.set_ylabel('Percentage (%)', fontsize=16)
          ax1.set_xlabel(item_col, fontsize=16)
          ax1.legend(loc='upper right', fontsize=12)
          ax1.grid(True, alpha=0.3)

          # Col 2: Promo ROI
          ax2 = axes[row_idx, 2]
          ax2.bar(x_pos, item_roi, color='darkblue', alpha=0.6)
          ax2.axhline(
              o_roi,
              color='darkorange',
              linestyle=':',
              linewidth=3,
              label=f'Overall Total ({o_roi:.2f})',
          )
          ax2.set_xticks(x_pos)
          ax2.set_xticklabels(
              wrapped_items, rotation=90, ha='center', fontsize=12
          )
          ax2.set_title(
              f'Year {y}: Promo ROI = Incremental KPI / Promo Spend',
              fontsize=18,
          )
          ax2.set_ylabel('ROI', fontsize=16)
          ax2.set_xlabel(item_col, fontsize=16)
          ax2.legend(loc='upper right', fontsize=12)
          ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(dir5, f'{prefix}_roi_by_item_matrix.png'),
            dpi=200,
        )
        plt.close()
