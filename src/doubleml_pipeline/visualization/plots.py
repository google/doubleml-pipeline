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

"""Plotting functions for the DoubleML pipeline.

Besides the Phase 1-4 charts (incremental KPI, ROI, CATE, sensitivity and the
alpha profile), this module holds the Phase 0 diagnostic figures written to
``step0_saturation/``; see the "Phase 0 figures" section at the
bottom of the file.
"""

import functools
import math
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import warnings

from doubleml_pipeline.evaluation import sensitivity
from doubleml_pipeline.models import saturation
from doubleml_pipeline.utils import io as pipeline_io
import matplotlib

# The Phase 0 figures are always written to PNG files and never
# shown interactively; select the non-interactive backend before pyplot loads
# so drawing works in headless environments.
matplotlib.use('Agg')
# pylint: disable=g-import-not-at-top,wrong-import-position
from matplotlib import font_manager
from matplotlib import textpath
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

# pylint: enable=g-import-not-at-top,wrong-import-position


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
    override_n_obs: Optional[int] = None,
    treatment_variable: str = sensitivity.TREATMENT_VARIABLE_T,
    t_ref: Optional[float] = None,
    group_mask: Optional[np.ndarray] = None,
    benchmark_cf_d: Optional[float] = None,
    benchmark_cf_y: Optional[float] = None,
    benchmark_rho: Optional[float] = None,
    benchmark_scope: Optional[str] = None,
):
  """Plots sensitivity contour lines for partial linear regression treatment effect analysis.

  The contour value is the adjusted lower
  bound ``|theta_hat| - B(cf_y, cf_d) - t_star_alpha se_hat`` with
  ``B = rho_U scale sqrt(cf_y cf_d / (1 - cf_d))`` and ``scale = sqrt(sigma2
  nu2)`` from ``dml_model.sensitivity_elements`` (recomputed from the
  out-of-fold residuals when unavailable); the robustness value is ``RV =
  (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2`` with ``f_q = (|theta_hat| -
  t_star_alpha se_hat) / scale`` (0 when the effect is not significant), so
  the RV star lies on the zero contour by construction (see
  ``evaluation.sensitivity``).  A GATE override (geo / item) uses the group's
  ``theta_g``, ``se_g``, ``n_g`` and the group's own moments ``sigma2_g``,
  ``nu2_g`` over ``group_mask``; ``se sqrt(n)`` is never used as the scale.

  With ``treatment_variable = "f_bar(T)"`` the model was
  fitted with ``D = fbar(T; alpha_hat)``; the x axis then reads "Partial R^2
  w.r.t. fbar(T; alpha_hat)" and the unit of ``theta_hat`` ("increment of Y
  for moving the treatment from 0 to T_ref") is stated.

  The benchmark point
  ``(cf_{d,g}, cf_{y,g})`` is RECEIVED as arguments (computed once per
  treatment by the caller from the long / short out-of-fold residuals of the
  group); this function no longer calls ``sensitivity_benchmark`` (which
  refitted the short model for every contour).

  Args:
    dml_model: The Phase 1-3 ``DoubleMLPLR`` (fitted with ``D`` as is).
    out_path: Where the PNG is written.
    title: Scope label used in the title.
    benchmark_covariates: Observed covariates of the benchmark point (the
      first one is named in the legend).
    override_theta: ``theta_g`` of a GATE group.
    override_se: ``se_g`` of a GATE group.
    override_n_obs: ``n_g`` of a GATE group (rows in the group).
    treatment_variable: ``"f_bar(T)"`` or ``"T"`` (the PLR's ``D``).
    t_ref: ``T_ref`` (on the scale ``T / c``) for the unit note of the
      saturation version.
    group_mask: Boolean mask over the fitted rows selecting the GATE group
      (required with a GATE override).
    benchmark_cf_d: ``cf_{d,g}`` of the benchmark point (0 when None).
    benchmark_cf_y: ``cf_{y,g}`` of the benchmark point (0 when None).
    benchmark_rho: ``rho_{U,g}`` of the benchmark (reported only).
    benchmark_scope: ``"group"`` / ``"whole_sample"`` (reported only).

  Returns:
    ``{'theta', 'se', 'robustness_value', 'n_obs', 'f_q', 'sigma2', 'nu2',
    'scale', 'benchmark_cf_d', 'benchmark_cf_y', 'benchmark_rho',
    'benchmark_scope'}``, or None when the chart was skipped.
  """
  try:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      dml_model.sensitivity_analysis(
          cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0
      )

      stats_k19 = sensitivity.sensitivity_stats(
          dml_model,
          override_theta=override_theta,
          override_se=override_se,
          override_n_obs=override_n_obs,
          group_mask=group_mask,
      )
      theta = stats_k19.theta
      se = stats_k19.se
      rv = stats_k19.robustness_value
      saturated = treatment_variable == sensitivity.TREATMENT_VARIABLE_FBAR

      # The benchmark point of this scope, computed by the caller.
      bx = float(benchmark_cf_d) if benchmark_cf_d is not None else 0
      by = float(benchmark_cf_y) if benchmark_cf_y is not None else 0

      # The grid limit, derived from the RV and the benchmark point.
      max_grid = max(0.1, rv * 1.5 if rv else 0.1, bx * 1.1, by * 1.1)
      max_grid = min(max_grid, 0.99)

      cf_d_grid = np.linspace(0, max_grid, 300)
      cf_y_grid = np.linspace(0, max_grid, 300)
      cf_d_mesh, cf_y_mesh = np.meshgrid(cf_d_grid, cf_y_grid)

      # B(cf_y, cf_d) = rho_U scale sqrt(cf_y cf_d / (1 - cf_d)), with the
      # residual-based scale = sqrt(sigma2 nu2) (never se sqrt(n)).
      bias_contour = sensitivity.bias_bound(
          cf_y_mesh, cf_d_mesh, stats_k19.scale, rho=1.0
      )
      adjusted_lower_bound = (
          np.abs(theta) - bias_contour - sensitivity.T_STAR_ALPHA * se
      )

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

      if rv > 0:
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
      # cf_d is the partial R^2 of U with the residual of
      # fbar(T; alpha_hat) in the saturation version (linear version
      # unchanged).
      if saturated:
        plt.xlabel(
            r'Partial $R^2$ w.r.t. $\bar{f}(T;\hat{\alpha})$ (cf_d)',
            fontsize=16,
        )
      else:
        plt.xlabel('Treatment Partial R^2 (cf_d)', fontsize=16)
      plt.ylabel('Outcome Partial R^2 (cf_y)', fontsize=16)
      plt.grid(True, linestyle=':', alpha=0.6)
      plt.legend(loc='upper right', fontsize=14)

      def fmt_sci(v):
        return f'{v:.2e}' if abs(v) < 0.01 and v != 0 else f'{v:.3f}'

      box_lines = [
          f'Estimated GATE/ATE: {fmt_sci(theta)}',
          f'Robustness Value (RV): {fmt_sci(rv)}',
          (
              f'n = {stats_k19.n_obs}, t = {stats_k19.t_stat:.2f},'
              f' scale = sqrt(sigma2 nu2) = {fmt_sci(stats_k19.scale)},'
              f' f_q = {fmt_sci(stats_k19.f_q)}'
          ),
      ]
      if saturated:
        t_ref_text = f' = {t_ref:.4g} on T/c' if t_ref is not None else ''
        box_lines.append(
            r'Unit of $\hat{\theta}$: increment of Y for moving the'
            f' treatment from 0 to T_ref{t_ref_text}'
        )
      props = dict(boxstyle='round', facecolor='white', alpha=0.9)
      plt.text(
          0.05,
          0.95,
          '\n'.join(box_lines),
          transform=plt.gca().transAxes,
          fontsize=16,
          verticalalignment='top',
          bbox=props,
      )
      plt.tight_layout()
      plt.savefig(out_path, dpi=200)
      plt.close()

      record = stats_k19.as_record()
      record.update({
          'benchmark_cf_d': bx,
          'benchmark_cf_y': by,
          'benchmark_rho': benchmark_rho,
          'benchmark_scope': benchmark_scope,
      })
      return record
  except (ValueError, KeyError, AttributeError, TypeError, RuntimeError) as e:
    print(f'    [Warning] Contour generation skipped for {title}: {e}')
    return None


def _wrap_label_tokens(text, fits):
  """Greedy line breaking after underscores or spaces.

  ``textwrap`` splits long identifiers like ``estimated_incremental_KPI_tv``
  mid-word, which reads badly on an axis; here a line only ends after an
  underscore or a space.

  Args:
    text: The label to wrap.
    fits: ``fits(line) -> bool``, whether a candidate line fits.

  Returns:
    The lines (without trailing whitespace); a token that does not fit on its
    own is returned as a line of its own and left to the caller.
  """
  tokens = re.findall(r'[^_\s]*[_\s]+|[^_\s]+$', text)
  lines = []
  current = ''
  for token in tokens:
    if current and not fits((current + token).rstrip()):
      lines.append(current.rstrip())
      current = token
    else:
      current += token
  if current.rstrip():
    lines.append(current.rstrip())
  return lines


def _wrap_axis_label(text, width=24):
  """Wraps a label onto several lines, breaking after underscores or spaces.

  A single token longer than ``width`` (no break opportunity at all) is cut
  at ``width`` so it cannot overflow either.

  Args:
    text: The label to wrap.
    width: Maximum number of characters per line.

  Returns:
    The label with newlines inserted.
  """
  wrapped = []
  for line in _wrap_label_tokens(text, lambda line: len(line) <= width):
    wrapped.extend(
        textwrap.wrap(
            line, width, break_long_words=True, break_on_hyphens=False
        )
        or [line]
    )
  return '\n'.join(wrapped) if wrapped else text


# Safety margin on measured text widths: the advance width of the layout and
# the ink extent the renderer reports differ by about one percent.
_TEXT_WIDTH_MARGIN = 1.03
# Renderer-free text measurement (the same FreeType layout matplotlib's Agg
# renderer uses).
_TEXT_TO_PATH = textpath.TextToPath()


@functools.lru_cache(maxsize=4096)
def _text_width_in(text, fontsize):
  """Rendered width of ``text`` in inches, in the default proportional font.

  Uses the font's advance widths (FreeType layout) rather than glyph
  outlines, so it costs about a millisecond and needs no figure or renderer.

  Args:
    text: A single line of text.
    fontsize: Font size in points.

  Returns:
    The width in inches (0 for an empty string).
  """
  if not text:
    return 0.0
  width_pt, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(
      text, font_manager.FontProperties(size=fontsize), ismath=False
  )
  return _TEXT_WIDTH_MARGIN * width_pt / 72.0


def _wrap_label_to_length(text, length_in, fontsize):
  """Wraps a label so that no rendered line exceeds ``length_in`` inches.

  Lines break after underscores or spaces (:func:`_wrap_label_tokens`) based
  on measured widths; a single token wider than the length is cut by
  characters.

  Args:
    text: The label to wrap.
    length_in: Available length in inches.
    fontsize: Font size in points.

  Returns:
    The list of lines (at least one).
  """

  def fits(line):
    return _text_width_in(line, fontsize) <= length_in

  lines = []
  for line in _wrap_label_tokens(str(text), fits):
    if fits(line):
      lines.append(line)
      continue
    per_char_in = _text_width_in(line, fontsize) / max(1, len(line))
    width = max(1, int(length_in / per_char_in))
    lines.extend(
        textwrap.wrap(
            line, width, break_long_words=True, break_on_hyphens=False
        )
        or [line]
    )
  return lines or ['']


def _binary_patterns(df, bin_cols):
  """Enumerates the binary-covariate combinations present in ``df``.

  Args:
    df: The consolidated frame.
    bin_cols: Binary CATE covariates (may be empty).

  Returns:
    ``(patterns, labels)``: ``patterns`` is a list of ``(label, {col: value})``
    pairs (one per distinct combination, in order of first appearance) and
    ``labels`` the list of their labels.  Without binary covariates there is a
    single ``'All Data'`` group with an empty pattern.
  """
  if not bin_cols:
    return [('All Data', {})], ['All Data']
  patterns = []
  for _, row in df[bin_cols].drop_duplicates().iterrows():
    pattern_str = ', '.join([f'{col}={row[col]}' for col in bin_cols])
    patterns.append((pattern_str, row.to_dict()))
  return patterns, [p[0] for p in patterns]


def _pattern_mask(df, pattern, n):
  """Boolean row mask of one binary-covariate combination."""
  idx = np.ones(n, dtype=bool)
  for k, v in pattern.items():
    idx &= (df[k] == v).values
  return idx


# Fonts of the CATE matrix charts.  The HTML report scales the png down to
# the width of its container, so sizes tuned for a 1:1 view come out
# unreadable there.
_CATE_SUPTITLE_FS = 36
_CATE_TITLE_FS = 26
_CATE_LABEL_FS = 24
_CATE_TICK_FS = 20
# Monospace text of the right-hand column (Binary Covariates, Estimated CATE
# Functions, P-value & Std Err) and its line width in characters.  Every line
# of the column is wrapped to `_CATE_MAX_CHARS`, and the column is sized from
# that width, so no text can run beyond the grey boxes.
_CATE_MONO_FS = 22.0
_CATE_MONO_MIN_FS = 9.0
_CATE_MAX_CHARS = 68
# Advance width of DejaVu Sans Mono (matplotlib's default monospace font) in
# em; used to size the text column so that it is exactly as wide as the boxes.
_MONO_EM_WIDTH = 0.602
# Column widths of the "P-value & Std Err" table: covariate name + three
# numeric columns separated by single spaces (35 + 3 * 11 = 68 characters).
_CATE_STATS_NAME_WIDTH = 35
_CATE_STATS_NUM_WIDTH = 10
# Cell labels are wrapped first and only shrunk (never below this size) when
# the wrapped text would still need more lines than the margins reserve.
_CATE_MIN_CELL_FS = 14.0
# Height reserved in the margins for every extra line of a wrapped cell label.
_CATE_EXTRA_LINE_IN = 0.45


def _fit_text(text, length_in, fontsize, max_lines, min_fontsize=None):
  """Fits a proportional-font label into ``length_in`` inches.

  The label is wrapped on measured widths (after underscores or spaces, see
  :func:`_wrap_label_to_length`) at ``fontsize``.  When that takes more than
  ``max_lines`` lines the font is reduced step by step, down to
  ``min_fontsize``, until the label fits.

  Args:
    text: The label.
    length_in: Available length in inches (cell width for titles and x
      labels, cell height for y and colorbar labels).
    fontsize: Preferred font size in points.
    max_lines: Maximum number of lines before the font is reduced.
    min_fontsize: Smallest font size allowed (default
      ``_CATE_MIN_CELL_FS``).

  Returns:
    ``(wrapped_text, fontsize)``.
  """
  if min_fontsize is None:
    min_fontsize = _CATE_MIN_CELL_FS
  fs = float(fontsize)
  while True:
    lines = _wrap_label_to_length(text, length_in, fs)
    if len(lines) <= max_lines or fs <= min_fontsize:
      return '\n'.join(lines), fs
    fs = max(min_fontsize, fs - 1.0)


def _wrap_mono(text, width):
  """Wraps ``text`` to ``width`` characters (long tokens are broken too).

  Args:
    text: The text to wrap.
    width: Maximum characters per line.

  Returns:
    The wrapped lines (at least one, possibly empty).
  """
  lines = textwrap.wrap(
      str(text), width=width, break_long_words=True, break_on_hyphens=False
  )
  return lines or ['']


def _format_stat_number(val, width=_CATE_STATS_NUM_WIDTH):
  """Formats a table number so that it never exceeds ``width`` characters.

  Args:
    val: A number or a string.
    width: Column width in characters.

  Returns:
    ``'{val:.3f}'`` when it fits, otherwise a 3-digit scientific notation;
    strings are truncated to the column width.
  """
  if isinstance(val, str):
    return val[:width]
  try:
    fixed = f'{float(val):.3f}'
  except (TypeError, ValueError):
    return str(val)[:width]
  if len(fixed) <= width:
    return fixed
  return f'{float(val):.3e}'[:width]


def _cate_text_blocks(eq_str_dict, cate_stats_dict, max_chars):
  """Builds the two monospace text blocks of the right-hand column.

  Every line is wrapped to ``max_chars`` characters (treatment headers and
  covariate names included), so the blocks never overflow the grey boxes
  that are sized from that width.

  Args:
    eq_str_dict: ``{treatment: readable CATE equation}``.
    cate_stats_dict: ``{treatment: {"coef": {...}, "pvalue": {...}, "stderr":
      {...}}}`` or None.
    max_chars: Characters per line (every line is padded to this width so all
      boxes come out equally wide).

  Returns:
    ``(eq_lines, stats_lines)``: the lines of the "Estimated CATE Functions"
    block and of the "P-value & Std Err" block.
  """
  eq_lines = ['Estimated CATE Functions:', '-' * max_chars, '']
  for d_col, eq_str in (eq_str_dict or {}).items():
    eq_lines.extend(_wrap_mono(f'[{d_col}]', max_chars))
    eq_lines.extend(_wrap_mono(eq_str, max_chars))
    eq_lines.append('')
  note_text = (
      'Note: Interpretation of coefficients of CATE function should be'
      ' directional as they are estimated based on normalised covariates.'
  )
  eq_lines.append('-' * max_chars)
  eq_lines.extend(_wrap_mono(note_text, max_chars))

  name_w, num_w = _CATE_STATS_NAME_WIDTH, _CATE_STATS_NUM_WIDTH
  stats_lines = ['P-value & Std Err:', '-' * max_chars, '']
  if cate_stats_dict:
    for d_col, cate_stats in cate_stats_dict.items():
      stats_lines.extend(_wrap_mono(f'[{d_col}]', max_chars))
      stats_lines.append(
          f'{"Covariate":<{name_w}} {"Coef":>{num_w}} {"P-val":>{num_w}}'
          f' {"StdErr":>{num_w}}'
      )
      coefs = cate_stats.get('coef', {})
      pvals = cate_stats.get('pvalue', {})
      stderrs = cate_stats.get('stderr', {})

      for cov in pvals.keys():
        c_str = _format_stat_number(coefs.get(cov, np.nan))
        p_str = _format_stat_number(pvals[cov])
        s_str = _format_stat_number(stderrs.get(cov, np.nan))

        cov_wrap = _wrap_mono(cov, name_w)
        stats_lines.append(
            f'{cov_wrap[0]:<{name_w}} {c_str:>{num_w}} {p_str:>{num_w}}'
            f' {s_str:>{num_w}}'
        )
        for extra_line in cov_wrap[1:]:
          stats_lines.append(
              f'{extra_line:<{name_w}} {"":>{num_w}} {"":>{num_w}}'
              f' {"":>{num_w}}'
          )
      stats_lines.append('')
  return eq_lines, stats_lines


# Label kinds of a cell and the line budget of each before the font shrinks:
# the row gap holds a two-line title plus a two-line x label, the left margin
# a three-line y label, the colorbar gutter a two-line colorbar label.
_CATE_LABEL_BUDGET = {'title': 2, 'x': 2, 'y': 3, 'cbar': 2}


def _cate_cell_texts(d_cols, cont_cols):
  """Raw (unwrapped) labels of every cell of the two CATE matrix charts.

  Args:
    d_cols: Treatment columns.
    cont_cols: Continuous CATE covariates (may be empty).

  Returns:
    ``{'title': [...], 'x': [...], 'y': [...], 'cbar': [...]}``.
  """
  x_labels = [f'Treatment: {d}' for d in d_cols]
  x_labels.append('Total Treatment (Excl. Lags)')
  y_labels = [f'Inc. Effect: estimated_incremental_KPI_{d}' for d in d_cols]
  y_labels.append('Total Incremental KPI')
  return {
      'title': [f'Grouped by: {c}' for c in (cont_cols or [])],
      'x': x_labels,
      'y': y_labels,
      'cbar': [f'Covariate: {c}' for c in (cont_cols or [])],
  }


def _fit_cell_label(kind, text, cell_in):
  """Wraps one cell label of ``kind`` to a cell of ``cell_in`` inches.

  Args:
    kind: ``'title'``, ``'x'``, ``'y'`` or ``'cbar'``.
    text: The raw label.
    cell_in: Edge of the (square) cell in inches.

  Returns:
    ``(wrapped_text, fontsize)`` (see :func:`_fit_text`).
  """
  fontsize = _CATE_TITLE_FS if kind == 'title' else _CATE_LABEL_FS
  return _fit_text(text, cell_in, fontsize, _CATE_LABEL_BUDGET[kind])


def _cate_matrix_figure(
    n_rows, n_cols, n_legend_labels, eq_lines, stats_lines, title, cell_texts
):
  """Creates the figure shared by the CATE scatter matrix and response chart.

  One square cell per (treatment, covariate) pair plus the trailing "total"
  row, and a right-hand text column that is exactly as wide as the three
  monospace boxes it carries (so no blank space is left at the right edge).
  The monospace font is first sized to fill the column and only shrunk when
  the three boxes would not fit the figure height.  The cell titles and axis
  labels are wrapped to the cell size (:func:`_fit_cell_label`) and the
  margins between / around the cells grow with the number of wrapped lines,
  so long treatment or covariate names never run into a neighbouring cell or
  out of the figure.

  Args:
    n_rows: Number of cell rows (treatments + 1).
    n_cols: Number of cell columns (continuous covariates, at least 1).
    n_legend_labels: Number of entries of the "Binary Covariates" legend.
    eq_lines: Lines of the "Estimated CATE Functions" block.
    stats_lines: Lines of the "P-value & Std Err" block.
    title: Figure suptitle.
    cell_texts: The raw cell labels (see :func:`_cate_cell_texts`).

  Returns:
    A dict with ``fig``, ``axes`` (2-D array), ``ax_right`` (the text column
    axes, frame off), ``cell_in`` (edge of a cell in inches), ``mono_fs``
    (final monospace font size), ``line_height_norm`` (line height as a
    fraction of the figure height), ``legend_lines``, ``eq_lines_count`` and
    ``stats_lines_count``.
  """
  # Base margins, sized for single-line labels at the fonts above (the layout
  # itself -- square cells and the right-hand text column -- is unchanged by
  # the font size).
  panel_in = 10.0
  base_left_in = 2.6  # up to a three-line y label + tick labels
  base_top_in = 1.7  # suptitle + one-line cell title
  base_bottom_in = 1.1  # one-line x label + tick labels
  base_cbar_in = 2.4  # colorbar gutter to the right of the last column
  wspace = 0.55
  base_hspace = 0.38  # one-line x label + ticks + one-line title below

  grid_width = panel_in * n_cols

  def _margins(lines):
    """Margins (inches) for the given number of lines per label kind."""
    extra = {k: max(0, v - 1) * _CATE_EXTRA_LINE_IN for k, v in lines.items()}
    return {
        # The base left margin already holds the full y-label budget (three
        # lines); beyond that the font shrinks, so it never has to grow.
        'left': base_left_in,
        'top': base_top_in + extra['title'],
        'bottom': base_bottom_in + extra['x'],
        'cbar': base_cbar_in + extra['cbar'],
        'gap': extra['x'] + extra['title'],
    }

  def _cell_edge(margins):
    """Cell edge (inches) at which every cell comes out exactly square."""
    return (grid_width - margins['left'] - margins['cbar']) / (
        n_cols + (n_cols - 1) * wspace
    )

  def _line_counts(cell_in):
    """Lines of the tallest wrapped label of each kind at ``cell_in``."""
    return {
        kind: max(
            [_fit_cell_label(kind, t, cell_in)[0].count('\n') + 1 for t in ts]
            or [1]
        )
        for kind, ts in cell_texts.items()
    }

  # The wrapped line counts depend on the cell size, which depends on the
  # margins, which depend on the line counts: iterate to a fixed point.
  lines = {kind: 1 for kind in _CATE_LABEL_BUDGET}
  for _ in range(4):
    margins = _margins(lines)
    cell_in = _cell_edge(margins)
    new_lines = _line_counts(cell_in)
    if new_lines == lines:
      break
    lines = new_lines
  margins = _margins(lines)
  cell_in = _cell_edge(margins)
  left_in, top_in = margins['left'], margins['top']
  bottom_in, cbar_in = margins['bottom'], margins['cbar']
  hspace = base_hspace + margins['gap'] / cell_in
  fig_height = (
      cell_in * (n_rows + (n_rows - 1) * hspace) + top_in + bottom_in
  )

  # Fit the right-hand column vertically by shrinking the monospace font
  # rather than by stretching the figure: growing the height is what used to
  # make this chart extremely tall.
  legend_lines = (n_legend_labels * 2 + 2) if n_legend_labels else 0
  eq_lines_count = len(eq_lines)
  stats_lines_count = len(stats_lines)
  total_lines = legend_lines + eq_lines_count + stats_lines_count
  # Padding lines for the gaps between the legend and the two text boxes.
  padded_lines = total_lines + 6
  mono_fs = _CATE_MONO_FS
  available_in = 0.90 * fig_height
  line_height_in = mono_fs * 1.75 / 72.0
  if padded_lines * line_height_in > available_in:
    mono_fs = max(
        _CATE_MONO_MIN_FS,
        mono_fs * available_in / (padded_lines * line_height_in),
    )
    line_height_in = mono_fs * 1.75 / 72.0
  if padded_lines * line_height_in > available_in:
    # Still too long even at the minimum readable size; fall back to growing.
    fig_height = padded_lines * line_height_in / 0.90

  # The text column is sized from the (final) monospace font so that the
  # three boxes span it edge to edge: box = max_chars glyphs + the round-box
  # padding (0.5 em each side), plus a small gutter on both sides.
  box_in = (_CATE_MAX_CHARS * _MONO_EM_WIDTH + 1.0) * mono_fs / 72.0
  gutter_in = 0.35
  text_width = box_in + 2 * gutter_in
  fig_width = grid_width + text_width
  grid_frac = grid_width / fig_width

  fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(fig_width, fig_height),
      squeeze=False,
      # Explicit margins: ``tight_layout`` mismeasures axes that carry a forced
      # box aspect, which lets neighbouring columns overlap.
      gridspec_kw={
          'left': left_in / fig_width,
          'right': (grid_width - cbar_in) / fig_width,
          'top': 1 - top_in / fig_height,
          'bottom': bottom_in / fig_height,
          'wspace': wspace,
          'hspace': hspace,
      },
  )
  fig.suptitle(title, fontsize=_CATE_SUPTITLE_FS, wrap=True, y=0.98)

  ax_right = fig.add_axes(
      [grid_frac + gutter_in / fig_width, 0.05, box_in / fig_width, 0.90]
  )
  ax_right.axis('off')
  return {
      'fig': fig,
      'axes': axes,
      'ax_right': ax_right,
      'cell_in': cell_in,
      'mono_fs': mono_fs,
      'line_height_norm': (mono_fs * 1.5 / 72.0) / fig_height,
      'legend_lines': legend_lines,
      'eq_lines_count': eq_lines_count,
      'stats_lines_count': stats_lines_count,
  }


def _draw_cate_right_column(layout, handles, labels, eq_lines, stats_lines):
  """Draws the legend and the two text boxes with aligned left/right edges.

  The "Estimated CATE Functions" box is drawn first (every line padded to
  ``_CATE_MAX_CHARS`` characters, so both text boxes are equally wide) and its
  horizontal extent is measured; the "Binary Covariates" legend is then
  expanded to exactly that extent (``mode='expand'``), so all three boxes
  share the same left and right edge.  The vertical positions are measured
  too (legend height, text-box height), so the boxes never overlap whatever
  the font metrics are.

  Args:
    layout: The dict returned by :func:`_cate_matrix_figure`.
    handles: Legend handles (scatter or line artists), may be empty.
    labels: Legend labels (one per handle).
    eq_lines: Lines of the "Estimated CATE Functions" block.
    stats_lines: Lines of the "P-value & Std Err" block.
  """
  fig = layout['fig']
  ax_right = layout['ax_right']
  mono_fs = layout['mono_fs']
  max_chars = _CATE_MAX_CHARS

  eq_text = '\n'.join([line.ljust(max_chars) for line in eq_lines])
  stats_text = '\n'.join([line.ljust(max_chars) for line in stats_lines])

  box_format = dict(
      fontsize=mono_fs,
      family='monospace',
      linespacing=1.5,
      verticalalignment='top',
      bbox=dict(
          boxstyle='round,pad=0.5',
          facecolor='lightgray',
          edgecolor='black',
      ),
  )

  renderer = fig.canvas.get_renderer()
  inv = ax_right.transAxes.inverted()
  # The round box extends the text extent by 0.5 em on every side.
  pad_px = 0.5 * mono_fs * fig.dpi / 72.0
  ax_h_px = ax_right.get_window_extent(renderer=renderer).height
  pad_y = pad_px / ax_h_px if ax_h_px > 0 else 0.0
  gap_y = 0.012  # vertical gap between boxes (axes fraction)

  def _box(artist):
    """``(x0, y0, x1, y1)`` of a text artist's round box in axes fractions."""
    ext = artist.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = inv.transform([
        [ext.x0 - pad_px, ext.y0 - pad_px],
        [ext.x1 + pad_px, ext.y1 + pad_px],
    ])
    return x0, y0, x1, y1

  # 1) Provisional position at the top: fixes the common box width.
  y_top = 1.0 - pad_y
  eq_artist = ax_right.text(
      0.0, y_top, eq_text, transform=ax_right.transAxes, **box_format
  )
  x0, _, x1, _ = _box(eq_artist)

  # 2) Legend expanded to the text-box width, anchored at the very top; its
  #    rendered height decides where the equation box goes.
  if handles:
    # Labels longer than the box (many binary covariates) wrap onto further
    # lines instead of running past the box edge.  The handle, its padding
    # and the box padding take about 7 monospace characters, so the visible
    # text is wrapped 8 characters short of the box width.
    label_w = max_chars - 8
    padded_labels = [
        '\n'.join(line.ljust(label_w) for line in _wrap_mono(lbl, label_w))
        for lbl in labels
    ]
    leg = ax_right.legend(
        handles,
        padded_labels,
        loc='upper left',
        bbox_to_anchor=(x0, 0.0, x1 - x0, 1.0),
        bbox_transform=ax_right.transAxes,
        mode='expand',
        borderaxespad=0.0,
        title='Binary Covariates',
        prop={'family': 'monospace', 'size': mono_fs},
        facecolor='lightgray',
        edgecolor='black',
        borderpad=0.5,
        fancybox=True,
    )
    plt.setp(leg.get_title(), family='monospace', fontsize=mono_fs + 2)
    leg._legend_box.align = 'left'  # pylint: disable=protected-access
    # The legend lays itself out at draw time; draw once to measure it.
    fig.canvas.draw()
    leg_ext = leg.get_window_extent(renderer=renderer)
    leg_y0 = inv.transform([[leg_ext.x0, leg_ext.y0]])[0][1]
    y_top = leg_y0 - gap_y - pad_y
    eq_artist.set_position((0.0, y_top))

  # 3) The stats box right below the (measured) equation box.
  _, eq_y0, _, _ = _box(eq_artist)
  ax_right.text(
      0.0,
      eq_y0 - gap_y - pad_y,
      stats_text,
      transform=ax_right.transAxes,
      **box_format,
  )


def _style_cate_cell(layout, ax, x_label, y_label, title):
  """Square cell, labels and grid shared by both CATE matrix charts.

  Every label is wrapped to the cell size (:func:`_fit_cell_label`), so a
  long treatment or covariate name can neither overlap a neighbouring cell
  nor run out of the figure.

  Args:
    layout: The dict returned by :func:`_cate_matrix_figure`.
    ax: The cell axes.
    x_label: Raw x label.
    y_label: Raw y label.
    title: Raw cell title, or None for the "total" row.
  """
  cell_in = layout['cell_in']
  ax.set_box_aspect(1)
  text, fs = _fit_cell_label('x', x_label, cell_in)
  ax.set_xlabel(text, fontsize=fs)
  text, fs = _fit_cell_label('y', y_label, cell_in)
  ax.set_ylabel(text, fontsize=fs)
  ax.tick_params(axis='both', labelsize=_CATE_TICK_FS)
  if title:
    text, fs = _fit_cell_label('title', title, cell_in)
    ax.set_title(text, fontsize=fs)
  ax.grid(True, linestyle=':', alpha=0.5)


def plot_cate_scatter_matrix(
    df_1a_consolidated,
    d_cols,
    cont_cols,
    bin_cols,
    eq_str_dict,
    out_path,
    cate_stats_dict=None,
):
  """Plots a scatter matrix showing CATE estimations against covariates.

  One square cell per (treatment, continuous covariate) pair plus a trailing
  "total" row; the dots are coloured by the binary-covariate combination
  ("Binary Covariates" legend) and the background contour by the continuous
  covariate of the column.  The right-hand column carries the legend, the
  estimated CATE functions and their p-values / standard errors.

  Args:
    df_1a_consolidated: The consolidated frame.
    d_cols: Treatment columns.
    cont_cols: Continuous CATE covariates (one column per covariate).
    bin_cols: Binary CATE covariates (one colour per combination).
    eq_str_dict: ``{treatment: readable CATE equation}``.
    out_path: Destination png path.
    cate_stats_dict: Optional ``{treatment: {"coef", "pvalue", "stderr"}}``.

  Returns:
    ``{(row, col): (xlim, ylim)}`` of every cell (row ``len(d_cols)`` is the
    total row), so that :func:`plot_cate_response_curves` can reproduce the
    exact axis scales.
  """
  n_rows = len(d_cols) + 1
  n_cols = len(cont_cols) if cont_cols else 1
  patterns, unique_colors = _binary_patterns(df_1a_consolidated, bin_cols)
  cmap = plt.get_cmap('tab10')
  color_map = {val: cmap(i % 10) for i, val in enumerate(unique_colors)}

  eq_lines, stats_lines = _cate_text_blocks(
      eq_str_dict, cate_stats_dict, _CATE_MAX_CHARS
  )
  layout = _cate_matrix_figure(
      n_rows,
      n_cols,
      len(unique_colors),
      eq_lines,
      stats_lines,
      'CATE Scatter Matrix: Treatment vs Incremental Effect',
      _cate_cell_texts(d_cols, cont_cols),
  )
  fig, axes = layout['fig'], layout['axes']
  n = len(df_1a_consolidated)

  def _contour(ax, x_data, y_data, cont_col, is_last_col):
    if not cont_cols:
      return
    if len(np.unique(x_data)) <= 2 or len(np.unique(y_data)) <= 2:
      return
    z_data = df_1a_consolidated[cont_col].values
    try:
      x_j = x_data + np.random.normal(0, 1e-8, size=len(x_data))
      y_j = y_data + np.random.normal(0, 1e-8, size=len(y_data))
      contour = ax.tricontourf(
          x_j, y_j, z_data, levels=20, cmap='coolwarm', alpha=0.4
      )
      if is_last_col:
        cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
        cbar = fig.colorbar(contour, cax=cax)
        # Wrapped to the cell height like the y labels, so a long covariate
        # name cannot run past the top / bottom of the colorbar.
        text, fs = _fit_cell_label(
            'cbar', f'Covariate: {cont_col}', layout['cell_in']
        )
        cbar.set_label(text, fontsize=fs)
        cbar.ax.tick_params(labelsize=_CATE_TICK_FS)
    except (ValueError, AttributeError, RuntimeError):
      pass

  def _dots(ax, x_data, y_data, with_label):
    for c_val, pattern in patterns:
      idx = _pattern_mask(df_1a_consolidated, pattern, n)
      if not idx.any():
        continue
      ax.scatter(
          x_data[idx],
          y_data[idx],
          color=color_map[c_val],
          alpha=0.9,
          label=c_val if with_label else None,
          edgecolors='white',
          s=60,
      )

  for i, d_col in enumerate(d_cols):
    kpi_col = f'estimated_incremental_KPI_{d_col}'
    x_data = df_1a_consolidated[d_col].values
    y_data = df_1a_consolidated[kpi_col].values
    for j, cont_col in enumerate(cont_cols or ['dummy']):
      ax = axes[i, j]
      _contour(ax, x_data, y_data, cont_col, j == n_cols - 1)
      _dots(ax, x_data, y_data, with_label=(i == 0 and j == 0))
      _style_cate_cell(
          layout,
          ax,
          f'Treatment: {d_col}',
          f'Inc. Effect: {kpi_col}',
          f'Grouped by: {cont_col}' if cont_cols else None,
      )

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
    _contour(ax, total_input, total_kpi, cont_col, j == n_cols - 1)
    _dots(ax, total_input, total_kpi, with_label=False)
    _style_cate_cell(
        layout,
        ax,
        'Total Treatment (Excl. Lags)',
        'Total Incremental KPI',
        None,
    )

  handles, labels = axes[0, 0].get_legend_handles_labels()
  _draw_cate_right_column(layout, handles, labels, eq_lines, stats_lines)

  axis_limits = {
      (i, j): (axes[i, j].get_xlim(), axes[i, j].get_ylim())
      for i, j in np.ndindex(n_rows, n_cols)
  }
  plt.savefig(out_path, dpi=200)
  plt.close()
  return axis_limits


# Quantiles of the 95% prediction interval drawn around every response curve,
# and the opacity of its fill (light enough for overlapping groups to stay
# readable).
_RESPONSE_BAND_QUANTILES = (0.025, 0.975)
_RESPONSE_BAND_ALPHA = 0.15


def _response_band(f_row, kpi, curve_grid):
  """95% prediction interval of a group's row-level response curves.

  Under the PLR every row of the group satisfies ``KPI_i = theta_i * f_i``
  with ``theta_i = p(x_i)' beta`` its own CATE; inside one binary-covariate
  group ``theta_i`` varies only through the continuous CATE covariates.  The
  band is therefore the 2.5% - 97.5% range of the implied row slopes
  ``theta_i = KPI_i / f_i`` (rows with ``f_i > 0``) applied to the group's
  curve: 95% of the group's rows respond inside it.

  Args:
    f_row: Per-row transformed treatment ``f_i`` of the group's rows.
    kpi: Per-row incremental KPI of the group's rows.
    curve_grid: The group's unit-slope curve ``fbar(T / c_g)`` on the x grid.

  Returns:
    ``(lower, upper)`` arrays on the x grid, or None when fewer than two rows
    carry a usable slope.
  """
  f_row = np.asarray(f_row, dtype=float)
  kpi = np.asarray(kpi, dtype=float)
  usable = np.isfinite(f_row) & np.isfinite(kpi) & (f_row > 0)
  if int(usable.sum()) < 2:
    return None
  slopes = kpi[usable] / f_row[usable]
  q_lo, q_hi = np.quantile(slopes, _RESPONSE_BAND_QUANTILES)
  curve_grid = np.asarray(curve_grid, dtype=float)
  lo, hi = q_lo * curve_grid, q_hi * curve_grid
  return np.minimum(lo, hi), np.maximum(lo, hi)


def _fit_group_response(x_raw, kpi, spec, x_grid):
  """Fits one group's response curve and evaluates it on ``x_grid``.

  Under the (saturation or linear) PLR every row satisfies
  ``KPI_i = theta_i * f_i`` with ``f_i`` the (normalized) transformed
  treatment of the row; the group's curve uses the least-squares slope
  ``theta_g = sum(KPI_i f_i) / sum(f_i^2)`` through the origin and the group's
  median treatment divisor ``c_g``: ``y(T) = theta_g * fbar(T / c_g)``.  For a
  linear treatment ``f_i = T_i``, ``c_g = 1`` and the curve is the straight
  line ``theta_g * T``.

  Args:
    x_raw: Raw treatment of the group's rows (linear case: also ``f_i``).
    kpi: Incremental KPI of the group's rows.
    spec: ``{"f_row", "divisor_row", "curve_fn"}`` restricted to the group's
      rows, or None for a linear treatment.
    x_grid: Raw treatment values to evaluate the curve on.

  Returns:
    ``(y_grid, theta_g, band)``; ``y_grid`` is None when the group carries no
    usable signal, ``band`` is the ``(lower, upper)`` 95% prediction interval
    of :func:`_response_band` (None when it cannot be formed).
  """
  kpi = np.asarray(kpi, dtype=float)
  if spec is None:
    f_row = np.asarray(x_raw, dtype=float)
    divisor = 1.0
    curve_fn = lambda t: t  # pylint: disable=unnecessary-lambda-assignment
  else:
    f_row = np.asarray(spec['f_row'], dtype=float)
    div_row = np.asarray(spec['divisor_row'], dtype=float)
    div_row = div_row[np.isfinite(div_row) & (div_row > 0)]
    divisor = float(np.median(div_row)) if div_row.size else 1.0
    curve_fn = spec['curve_fn']
  ok = np.isfinite(f_row) & np.isfinite(kpi)
  denom = float(np.sum(f_row[ok] ** 2))
  if not ok.any() or denom <= 0:
    return None, float('nan'), None
  theta = float(np.sum(kpi[ok] * f_row[ok]) / denom)
  curve_grid = np.asarray(curve_fn(np.asarray(x_grid) / divisor), float)
  y_grid = theta * curve_grid
  return y_grid, theta, _response_band(f_row, kpi, curve_grid)


def plot_cate_response_curves(
    df_1a_consolidated,
    d_cols,
    cont_cols,
    bin_cols,
    eq_str_dict,
    out_path,
    cate_stats_dict=None,
    response_specs=None,
    axis_limits=None,
):
  """Plots the fitted response curves behind the CATE scatter matrix.

  Same figure, cells, axis scales, fonts and right-hand column as
  :func:`plot_cate_scatter_matrix`, but instead of the dots every cell shows,
  per binary-covariate combination (same colours as the dots, translucent),
  the response curve ``T -> incremental KPI`` in the ORIGINAL treatment units:
  the saturation curve ``theta_g * fbar(T / c_g; alpha_hat, T_ref)`` of the
  group when a saturation family is in force for the treatment, the straight
  line ``theta_g * T`` otherwise (see :func:`_fit_group_response`).  Every
  curve is wrapped in its 95% prediction interval (lighter fill of the same
  colour): the 2.5% - 97.5% range of the group's row-level response curves,
  i.e. of the implied row slopes ``theta_i = KPI_i / f_i``, whose spread
  inside a binary group comes from the continuous CATE covariates (see
  :func:`_response_band`).  The trailing "total" row reuses the treatment's
  curve with a single treatment and falls back to a per-group straight-line
  fit otherwise.

  Args:
    df_1a_consolidated: The consolidated frame.
    d_cols: Treatment columns.
    cont_cols: Continuous CATE covariates (one column per covariate; the
      curves do not depend on them, the columns only mirror the scatter
      matrix).
    bin_cols: Binary CATE covariates (one colour per combination).
    eq_str_dict: ``{treatment: readable CATE equation}``.
    out_path: Destination png path.
    cate_stats_dict: Optional ``{treatment: {"coef", "pvalue", "stderr"}}``.
    response_specs: Optional ``{treatment: {"f_row": per-row fbar(T_i / c_i),
      "divisor_row": per-row divisor c_i, "curve_fn": callable mapping the
      normalized treatment T / c to fbar, "label": str}}`` for the treatments
      under a saturation family; treatments missing from it are linear.
    axis_limits: Optional ``{(row, col): (xlim, ylim)}`` as returned by
      :func:`plot_cate_scatter_matrix`; applied verbatim so both charts share
      the same scales.

  Returns:
    ``{treatment: {group label: theta_g}}`` of the fitted group slopes.
  """
  response_specs = response_specs or {}
  axis_limits = axis_limits or {}
  n_rows = len(d_cols) + 1
  n_cols = len(cont_cols) if cont_cols else 1
  patterns, unique_colors = _binary_patterns(df_1a_consolidated, bin_cols)
  cmap = plt.get_cmap('tab10')
  color_map = {val: cmap(i % 10) for i, val in enumerate(unique_colors)}

  eq_lines, stats_lines = _cate_text_blocks(
      eq_str_dict, cate_stats_dict, _CATE_MAX_CHARS
  )
  layout = _cate_matrix_figure(
      n_rows,
      n_cols,
      # + 1: the "95% prediction interval" entry added to the legend below.
      len(unique_colors) + 1,
      eq_lines,
      stats_lines,
      'CATE Response Curves: Treatment vs Incremental Effect',
      _cate_cell_texts(d_cols, cont_cols),
  )
  axes = layout['axes']
  n = len(df_1a_consolidated)
  fitted = {}

  def _x_grid(i, x_data):
    lims = axis_limits.get((i, 0))
    x_max = lims[0][1] if lims else float(np.nanmax(x_data))
    if not np.isfinite(x_max) or x_max <= 0:
      x_max = 1.0
    return np.linspace(0.0, x_max, 200)

  def _curves(ax, x_data, y_data, spec, x_grid, with_label, store):
    for c_val, pattern in patterns:
      idx = _pattern_mask(df_1a_consolidated, pattern, n)
      if not idx.any():
        continue
      sub_spec = None
      if spec is not None:
        sub_spec = {
            'f_row': np.asarray(spec['f_row'])[idx],
            'divisor_row': np.asarray(spec['divisor_row'])[idx],
            'curve_fn': spec['curve_fn'],
        }
      y_grid, theta, band = _fit_group_response(
          x_data[idx], y_data[idx], sub_spec, x_grid
      )
      if store is not None:
        store[c_val] = theta
      if y_grid is None:
        continue
      if band is not None:
        # The 95% prediction interval of the group's rows, under the curve.
        ax.fill_between(
            x_grid,
            band[0],
            band[1],
            color=color_map[c_val],
            alpha=_RESPONSE_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
      ax.plot(
          x_grid,
          y_grid,
          color=color_map[c_val],
          alpha=0.55,
          linewidth=4,
          label=c_val if with_label else None,
          zorder=2,
      )

  def _apply_limits(ax, i, j):
    lims = axis_limits.get((i, j))
    if lims:
      ax.set_xlim(lims[0])
      ax.set_ylim(lims[1])

  def _response_note(ax, label):
    """Small wrapped note in the upper-right corner naming the response."""
    ax.text(
        0.98,
        0.97,
        textwrap.fill(f'response: {label}', 30),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=_CATE_TICK_FS - 2,
        color='#444444',
        zorder=5,
        bbox=dict(
            boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', alpha=0.85
        ),
    )

  for i, d_col in enumerate(d_cols):
    kpi_col = f'estimated_incremental_KPI_{d_col}'
    x_data = df_1a_consolidated[d_col].values.astype(float)
    y_data = df_1a_consolidated[kpi_col].values.astype(float)
    spec = response_specs.get(d_col)
    x_grid = _x_grid(i, x_data)
    fitted[d_col] = {}
    curve_label = (spec or {}).get('label') or 'linear'
    for j, cont_col in enumerate(cont_cols or ['dummy']):
      ax = axes[i, j]
      _curves(
          ax,
          x_data,
          y_data,
          spec,
          x_grid,
          with_label=(i == 0 and j == 0),
          store=fitted[d_col] if j == 0 else None,
      )
      title = f'Grouped by: {cont_col}' if cont_cols else None
      _style_cate_cell(
          layout, ax, f'Treatment: {d_col}', f'Inc. Effect: {kpi_col}', title
      )
      _response_note(ax, curve_label)
      _apply_limits(ax, i, j)

  total_input = (
      df_1a_consolidated[[
          c for c in d_cols if not re.search(r'_l\d+$', c, flags=re.IGNORECASE)
      ]]
      .sum(axis=1)
      .values.astype(float)
  )
  total_kpi = (
      df_1a_consolidated['total_estimated_incremental_KPI'].values.astype(float)
  )
  total_spec = response_specs.get(d_cols[0]) if len(d_cols) == 1 else None
  total_label = (
      (total_spec or {}).get('label') or 'linear'
      if len(d_cols) == 1
      else 'linear fit (sum of treatments)'
  )
  x_grid = _x_grid(n_rows - 1, total_input)
  for j in range(n_cols):
    ax = axes[-1, j]
    _curves(ax, total_input, total_kpi, total_spec, x_grid, False, None)
    _style_cate_cell(
        layout,
        ax,
        'Total Treatment (Excl. Lags)',
        'Total Incremental KPI',
        None,
    )
    _response_note(ax, total_label)
    _apply_limits(ax, n_rows - 1, j)

  handles, labels = axes[0, 0].get_legend_handles_labels()
  # One generic entry for the bands (each band wears its group's colour).
  handles.append(
      mpatches.Patch(facecolor='grey', edgecolor='none', alpha=0.35)
  )
  labels.append('95% prediction interval (rows of the group)')
  _draw_cate_right_column(layout, handles, labels, eq_lines, stats_lines)

  plt.savefig(out_path, dpi=200)
  plt.close()
  return fitted


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


# ---------------------------------------------------------------------------
# Saturation-aware DML additions (Phase 0 / Stage B diagnostics).
# The diagnostic plots (mean R-loss curve, the 1-SE set
# A_1SE and the distribution of the per-seed alpha_s) are ALWAYS emitted,
# regardless of whether the weak-identification warnings fired.
# ---------------------------------------------------------------------------
def _format_stat_value(value) -> str:
  """Renders one Phase 0 statistic for the alpha-profile stats table.

  Args:
    value: A scalar coming from `AlphaSelectionResult` (bool, number, string or
      None).

  Returns:
    A short human-readable string. NaN and None both render as `n/a`, because
    several Phase 0 diagnostics are simply not computable with a single seed.
  """
  if value is None:
    return 'n/a'
  if isinstance(value, (bool, np.bool_)):
    return 'yes' if value else 'no'
  if isinstance(value, (int, np.integer)):
    return f'{int(value):,}'
  if isinstance(value, (float, np.floating)):
    return 'n/a' if not np.isfinite(float(value)) else f'{float(value):.6g}'
  return str(value)


_FOOTNOTE_FONTSIZE = 7.6


def _notes(*lines: str) -> List[str]:
  """Collects footnote lines (one argument per line)."""
  return list(lines)


def _wrap_footnote_lines(lines, width: int) -> List[str]:
  """Wraps footnote lines; continuation lines of a bullet are indented."""
  out: List[str] = []
  for line in lines:
    line = str(line)
    if not line.strip():
      out.append('')
      continue
    indent = '    ' if line.lstrip().startswith(('•', '-', '*')) else ''
    wrapped = textwrap.wrap(
        line,
        width=width,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    out.extend(wrapped or [line])
  return out


def _add_footnote(
    fig,
    lines,
    fontsize: float = _FOOTNOTE_FONTSIZE,
    top_space: float = 0.0,
) -> None:
  """Appends a footnote block below the axes of ``fig``.

  The figure is made taller by the height of the block, so the panels keep
  their size, and ``tight_layout`` is told to leave the bottom strip free.
  The caller saves and closes the figure.

  Args:
    fig: The figure.
    lines: Footnote lines (each one is wrapped to the figure width).
    fontsize: Footnote font size in points.
    top_space: Fraction of the figure height to keep free at the top (for a
      ``suptitle``).
  """
  width_in, height_in = fig.get_size_inches()
  # ~0.55 em per glyph for the default sans-serif font.
  chars_per_line = max(40, int(width_in * 72.0 / (fontsize * 0.55)) - 4)
  wrapped = _wrap_footnote_lines(lines, chars_per_line)
  line_height_in = fontsize * 1.32 / 72.0
  block_in = len(wrapped) * line_height_in + 0.3
  new_height = float(height_in) + block_in
  fig.set_size_inches(width_in, new_height, forward=True)
  bottom = block_in / new_height
  fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0 - top_space))
  fig.text(
      0.012,
      0.012,
      '\n'.join(wrapped),
      ha='left',
      va='bottom',
      fontsize=fontsize,
      linespacing=1.32,
      color='#333333',
  )


def _draw_alpha_stats_table(
    ax, stats_dict, stats_tags=None, two_columns: bool = False
) -> None:
  """Draws the key-stats / process-metrics table of the alpha profile chart.

  Args:
    ax: The axes to draw on (its frame is removed).
    stats_dict: Ordered mapping of metric name to value.
    stats_tags: Optional mapping of metric name to the diagnostic it
      belongs to (``"diag1"`` ... ``"diag6"``); rows without an entry get a
      blank ``diagnostics`` cell.
    two_columns: Split the rows into two side-by-side tables.  Used by the
      full-width bottom cell of the alpha profile chart, where one narrow
      table would leave most of the cell empty.
  """
  ax.axis('off')
  stats_tags = stats_tags or {}
  cell_text = [
      [str(key), _format_stat_value(value), str(stats_tags.get(key, ''))]
      for key, value in stats_dict.items()
  ]
  if two_columns and len(cell_text) > 1:
    half = (len(cell_text) + 1) // 2
    chunks = [cell_text[:half], cell_text[half:]]
    gap = 0.03
    width = (1.0 - gap) / 2.0
    bboxes = [[0.0, 0.0, width, 1.0], [width + gap, 0.0, width, 1.0]]
  else:
    chunks = [cell_text]
    bboxes = [None]
  n_rows = max(len(c) for c in chunks) + 1
  for rows, bbox in zip(chunks, bboxes):
    if not rows:
      continue
    if bbox is not None:
      # Pad the shorter table so both end on the same row.
      rows = rows + [['', '', '']] * (n_rows - 1 - len(rows))
    kwargs = dict(
        cellText=rows,
        colLabels=['metric', 'value', 'diagnostics'],
        cellLoc='left',
        colLoc='left',
        colWidths=[0.50, 0.36, 0.14],
    )
    if bbox is None:
      table = ax.table(loc='upper center', **kwargs)
    else:
      table = ax.table(bbox=bbox, **kwargs)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    if bbox is None:
      table.scale(1.0, 1.32)
    for (row, _), cell in table.get_celld().items():
      cell.set_edgecolor('lightgray')
      if row == 0:
        cell.set_facecolor('#37474f')
        cell.set_text_props(color='white', fontweight='bold')
      elif row % 2 == 0:
        cell.set_facecolor('#f2f4f5')
  ax.set_title('Key stats & process metrics', fontsize=18)


# Colours of the "Fitted saturation curve" panel.  Red is reserved for the
# curve Phases 1-4 actually use (FINAL) and for the common point (T_ref, 1);
# the shapes of the smallest / median / largest grid alpha form a light-to-dark
# purple gradation so the three are told apart at a glance.
_FINAL_CURVE_COLOR = 'red'
_GRID_SHAPE_COLORS = ('#bcbddc', '#807dba', '#3f007d')


def _draw_saturation_curve(
    ax,
    family,
    alpha_hat,
    alpha_grid,
    one_se_set,
    t_normalized,
    t_ref=None,
    t_axis_label=None,
    grid_result=None,
) -> None:
  """Draws the fitted saturation curve over the observed treatment range.

  With `t_ref` given the panel draws the *normalized*
  candidate `fbar(T; alpha_hat) = f(T; alpha_hat) / f(T_ref; alpha_hat)`,
  which is what Phases 1-3 actually fit on, together with the identity
  candidate `fbar_T = T / T_ref` and the common point `(T_ref, 1)`
  every curve passes through.
  Without `t_ref` the legacy raw curve `f(T/c; alpha_hat)` is drawn.

  Plotting the curve next to the R-loss profile turns an abstract `alpha_hat`
  into the concrete diminishing-returns shape it implies. The shaded band
  spans the 1-SE set, so a flat (weakly identified) profile is immediately
  visible as a wide band.

  With `grid_result` (the alpha selection result) the panel additionally shows
  the range of shapes the grid can express (thin purple curves, light to dark,
  for the smallest / median / largest grid alpha) and names the curve Phases
  1-4 actually use (``FINAL``, drawn thick red together with the red dot at
  ``(T_ref, 1)``): the identity candidate when the decision was linear, or the
  ``fallback_half_saturation_T`` curve when that override is in force.  This
  is the content of the former ``others_grid_construction`` bottom panel.

  Args:
    ax: The axes to draw on.
    family: One of `models.saturation.SUPPORTED_FAMILIES`.
    alpha_hat: The selected alpha (on the normalized `T/c` scale).
    alpha_grid: The alpha grid `G`.
    one_se_set: Boolean mask of the 1-SE set `A_1SE`, or None.
    t_normalized: Observed normalized treatment values `T/c`.
    t_ref: Reference point `T_ref` on the same `T/c` scale, or None for the
      legacy un-normalized curve.
    t_axis_label: Optional x-axis label spelling out what `c` is (e.g.
      `normalized treatment T/c  (c = population x 0.012)`). Defaults to the
      bare `normalized treatment T/c`.
    grid_result: Optional alpha selection result (duck-typed); see above.
  """
  t_obs = np.asarray(t_normalized, dtype=float)
  t_obs = t_obs[np.isfinite(t_obs)]
  t_obs = np.maximum(t_obs, 0.0)
  if t_obs.size == 0:
    ax.axis('off')
    return

  normalized = t_ref is not None and np.isfinite(t_ref) and t_ref > 0
  t_ref = float(t_ref) if normalized else None

  # The 99th percentile keeps a handful of extreme-spend rows from flattening
  # the whole curve into the left margin. With normalization the curve must
  # at least reach T_ref so the (T_ref, 1) marker is on the chart.
  t_hi = float(np.quantile(t_obs, 0.99))
  if not np.isfinite(t_hi) or t_hi <= 0:
    t_hi = float(np.max(t_obs)) or 1.0
  if normalized:
    t_hi = max(t_hi, 1.05 * t_ref)
  t_curve = np.linspace(0.0, t_hi, 256)

  def _curve(alpha):
    if normalized:
      return saturation.normalized_saturation(
          t_curve, family, float(alpha), t_ref
      )
    return saturation.apply_saturation(t_curve, family, float(alpha))

  f_hat = _curve(alpha_hat)

  if one_se_set is not None and np.any(one_se_set):
    inside = np.asarray(alpha_grid, dtype=float)[
        np.asarray(one_se_set, dtype=bool)
    ]
    f_a = _curve(inside.min())
    f_b = _curve(inside.max())
    # The ordering of the two endpoint curves flips across T (for `power` it
    # flips at T = T_ref, or T/c = 1 un-normalized), so take the envelope
    # elementwise.
    ax.fill_between(
        t_curve,
        np.minimum(f_a, f_b),
        np.maximum(f_a, f_b),
        color='orange',
        alpha=0.20,
        label='1-SE set envelope',
    )

  if normalized:
    # Former grid-construction bottom panel: the range of shapes the grid can
    # express (purple gradation) and the curve Phases 1-4 actually use (FINAL).
    final_kind = 'saturation'
    alpha_est = float('nan')
    if grid_result is not None:
      grid_alphas = np.asarray(
          getattr(grid_result.grid, 'alphas', []), dtype=float
      )
      if grid_alphas.size:
        picks = {
            'grid min alpha': float(grid_alphas.min()),
            'grid median alpha': float(np.median(grid_alphas)),
            'grid max alpha': float(grid_alphas.max()),
        }
        for (label, alpha), color in zip(picks.items(), _GRID_SHAPE_COLORS):
          ax.plot(
              t_curve,
              _curve(alpha),
              color=color,
              linewidth=1.8,
              label=f'{label} = {alpha:.4g}',
          )
      final_kind, _, _ = _final_curve(grid_result)
      alpha_est = float(
          getattr(grid_result, 'alpha_hat_estimated', float('nan'))
      )
    # Identity candidate: the straight line through (T_ref, 1).
    if final_kind == 'linear':
      ax.plot(
          t_curve,
          t_curve / t_ref,
          color=_FINAL_CURVE_COLOR,
          linewidth=3,
          label='FINAL: identity candidate T / T_ref (linear, selected_family'
                ' = None)',
      )
      ax.plot(
          t_curve,
          _curve(alpha_est) if np.isfinite(alpha_est) else f_hat,
          color='darkgreen',
          linestyle='--',
          linewidth=2,
          label=(
              'estimated fbar(T; alpha_hat ='
              f' {alpha_est if np.isfinite(alpha_est) else float(alpha_hat):.4g})'
              ' -- NOT adopted (decision = linear)'
          ),
      )
    else:
      ax.plot(
          t_curve,
          t_curve / t_ref,
          color='grey',
          linestyle='--',
          linewidth=2,
          label='identity candidate T / T_ref (not adopted)',
      )
      if (
          grid_result is not None
          and bool(getattr(grid_result, 'fallback_half_saturation_used', False))
          and np.isfinite(alpha_est)
          and not np.isclose(alpha_est, float(alpha_hat))
      ):
        ax.plot(
            t_curve,
            _curve(alpha_est),
            color='darkgreen',
            linestyle='--',
            linewidth=2,
            label=(
                f'estimated alpha_hat = {alpha_est:.4g} (replaced by'
                ' fallback_half_saturation_T)'
            ),
        )
      final_prefix = 'FINAL: ' if grid_result is not None else ''
      ax.plot(
          t_curve,
          f_hat,
          color=_FINAL_CURVE_COLOR,
          linewidth=3,
          label=f'{final_prefix}fbar(T; alpha_hat = {float(alpha_hat):.4g})',
      )
    ax.plot(
        [t_ref],
        [1.0],
        marker='o',
        markersize=10,
        color=_FINAL_CURVE_COLOR,
        markeredgecolor='black',
        linestyle='none',
        zorder=6,
        label=f'(T_ref, 1) = ({t_ref:.3g}, 1)',
    )
  else:
    # A straight line through the right endpoint makes the curvature -- i.e.
    # how much saturation the fitted alpha actually implies -- readable at a
    # glance.
    if f_hat[-1] > 0:
      ax.plot(
          t_curve,
          t_curve * (f_hat[-1] / t_hi),
          color='grey',
          linestyle=':',
          linewidth=2,
          label='linear (no saturation) reference',
      )
    ax.plot(
        t_curve,
        f_hat,
        color=_FINAL_CURVE_COLOR,
        linewidth=3,
        label=f'f(T/c; alpha_hat = {float(alpha_hat):.4g})',
    )

  # Median-normalized treatments satisfy median(T/c) == 1 by construction, so
  # this marker is the natural "typical row" reference point.  Black dotted:
  # red is reserved for the FINAL curve.
  t_median = float(np.median(t_obs))
  if 0 < t_median <= t_hi:
    ax.axvline(t_median, color='black', linestyle=':', linewidth=1.5)
    ax.annotate(
        f'median T/c = {t_median:.3g}',
        xy=(t_median, 0.0),
        xytext=(4, 6),
        textcoords='offset points',
        rotation=90,
        va='bottom',
        fontsize=11,
        color='black',
    )

  ax_hist = ax.twinx()
  ax_hist.hist(
      t_obs[t_obs <= t_hi],
      bins=40,
      color='steelblue',
      alpha=0.22,
      zorder=0,
  )
  ax_hist.set_ylabel('observed rows (D_sel)', fontsize=13, color='steelblue')
  ax_hist.tick_params(axis='y', labelcolor='steelblue')
  ax.set_zorder(ax_hist.get_zorder() + 1)
  ax.patch.set_visible(False)

  ax.set_xlabel(t_axis_label or 'normalized treatment T/c', fontsize=16)
  if normalized:
    ax.set_ylabel(
        f'fbar(T; alpha) = f / f(T_ref; alpha)  [{family}]', fontsize=16
    )
    ax.set_title(
        'Fitted saturation curve (normalized through (T_ref, 1))', fontsize=18
    )
  else:
    ax.set_ylabel(f'f(T/c; alpha)  [{family}]', fontsize=16)
    ax.set_title('Fitted saturation curve', fontsize=18)
  ax.legend(fontsize=11, loc='upper left')
  ax.grid(True, alpha=0.3)


def _grid_construction_notes(result: Any) -> List[str]:
  """Footnote lines of the grid-construction panel.

  Args:
    result: The alpha selection result.

  Returns:
    Footnote lines: how T_ref and the quantiles are defined, how the grid was
    built and which files hold the data.
  """
  grid = result.grid
  t_ref = _t_ref(result)
  unit = _grid_unit(result)
  t_ref_def = _t_ref_definition(result)
  t_ref_pct = _t_ref_percent(result)
  alphas = np.asarray(getattr(grid, 'alphas', []), dtype=float)
  n_points = int(alphas.size)
  auto_range = grid is not None and (
      np.isfinite(grid.q50)
      and np.isfinite(grid.q95)
      and np.isfinite(grid.c_min)
      and np.isfinite(grid.c_max)
  )
  if result.family == 'log':
    unit_note = (
        ' (T_knee = the spend at which the marginal effect of the log curve'
        ' has halved)'
    )
  else:
    unit_note = (
        f' ({unit} = half-saturation point: f({unit}) = 1/2 of the ceiling'
        ' of f)'
    )
  if auto_range:
    t_half_all = np.asarray(grid.t_half, float)
    t_half_all = t_half_all[np.isfinite(t_half_all) & (t_half_all > 0)]
    if grid.extensions and t_half_all.size:
      extension_note = (
          f' Here {grid.extensions} extension(s) were applied, so the final'
          f' grid spans {unit} in [{t_half_all.min():.4g},'
          f' {t_half_all.max():.4g}] ({n_points} points).'
      )
    else:
      extension_note = f' No extension was needed here ({n_points} points).'
    grid_note = (
        '• Grid construction (grey lines): the initial grid is log-spaced in'
        f' the grid variable {unit} between c_min*q50 ='
        f' {grid.c_min:g}*q50 = {grid.c_min * grid.q50:.4g} and c_max*q95 ='
        f' {grid.c_max:g}*q95 = {grid.c_max * grid.q95:.4g}{unit_note}.'
        f' {grid.c_min:g}*q50 is the FASTEST saturation considered: the'
        ' curve already half-saturates at a quarter of the median positive'
        " spend, so f_bar is nearly flat over most of the data (an 'any"
        f" spend' effect). {grid.c_max:g}*q95 is the MILDEST: half-saturation"
        ' only at 4x the 95th percentile, so the curve is practically linear'
        ' over the observed range (fbar -> T/T_ref). Each alpha of the grid'
        f' is the family parameter implied by its {unit}. When alpha_hat'
        ' lands on a grid edge the range is extended x4 on that side (at'
        f' most 2 times).{extension_note}'
    )
  elif result.family == 'power' and n_points:
    grid_note = (
        f'• Grid construction (grey lines are omitted): {n_points} equally'
        f' spaced alpha values in [{alphas.min():.3g}, {alphas.max():.3g}]'
        ' (power is dimensionless and has no half-saturation point; alpha'
        ' -> 1 is the linear limit).'
    )
  else:
    mode = getattr(grid, 'mode', 'manual')
    grid_note = (
        f'• Grid construction (grey lines): {n_points} values of {unit} from'
        f' the manual saturation.grid settings (mode = {mode}).'
    )
  return _notes(
      f'• T_ref = {t_ref_def} = {t_ref:.4g}: the {t_ref_pct}th percentile of'
      ' the POSITIVE treatment values, computed once on the full data before'
      ' the split (saturation.t_ref_quantile). Every candidate curve is'
      ' normalized as fbar(T; alpha) = f(T; alpha) / f(T_ref; alpha), so all'
      ' curves pass through (T_ref, 1) and differ only in shape; the theta'
      ' estimated in Phases 1-3 is the effect of moving T from 0 to T_ref. T'
      ' is on the Phase 0 scale T/c (c = the treatment divisor;'
      ' divisor_formula in summary_alpha_selection.json).',
      '• Grid construction: histogram of the rows with T > 0 (log axis);'
      ' q50(T>0) and q95(T>0) are the 50th / 95th percentiles of the POSITIVE'
      ' values (zero-spend rows excluded); the orange band is the automatic'
      ' grid range. Data: diag3_identification_diagnostics.csv (q50_T_pos,'
      ' q95_T_pos, T_max, T_ref, share_T_zero), summary_alpha_grid.csv'
      ' (grid_index, half_saturation_T, alpha), summary_alpha_selection.json'
      ' (T_ref, t_ref_quantile, grid_mode, grid_extensions, grid_edge_hit,'
      ' alpha_hat, alpha_hat_estimated, fallback_half_saturation_T_used).',
      grid_note,
  )


def _draw_grid_construction_panel(ax, result: Any) -> None:
  """Draws the grid-construction panel of the alpha profile chart.

  Histogram of ``T > 0`` on a log axis with
  vertical lines at ``q50``, ``q95`` (labelled ``= T_ref``) and at every grid
  point; the automatic range ``[c_min q50, c_max q95]`` is shaded.  The grid
  mode, the number of extensions and ``grid_edge_hit`` are stated in the
  legend title.

  Args:
    ax: The axes to draw on.
    result: The alpha selection result.
  """
  family = result.family
  grid = result.grid
  t_values = np.asarray(
      getattr(result, 't_values', None)
      if getattr(result, 't_values', None) is not None
      else [],
      dtype=float,
  )
  t_pos = t_values[np.isfinite(t_values) & (t_values > 0)]
  t_ref = _t_ref(result)
  unit = _grid_unit(result)
  t_ref_def = _t_ref_definition(result)
  alphas = np.asarray(getattr(grid, 'alphas', []), dtype=float)
  n_points = int(alphas.size)
  auto_range = grid is not None and (
      np.isfinite(grid.q50)
      and np.isfinite(grid.q95)
      and np.isfinite(grid.c_min)
      and np.isfinite(grid.c_max)
  )
  if t_pos.size:
    lo = max(float(np.min(t_pos)), 1e-12)
    hi = max(float(np.max(t_pos)), lo * 1.0001)
    t_half = np.asarray(getattr(grid, 't_half', []), float)
    finite_half = t_half[np.isfinite(t_half) & (t_half > 0)]
    if finite_half.size:
      lo = min(lo, float(finite_half.min()))
      hi = max(hi, float(finite_half.max()))
    bins = np.logspace(math.log10(lo), math.log10(hi), 40)
    ax.hist(
        t_pos,
        bins=bins,
        color='#9ecae1',
        edgecolor='white',
        label='rows with T > 0',
    )
    ax.set_xscale('log')
    if auto_range:
      band_lo = grid.c_min * grid.q50
      band_hi = grid.c_max * grid.q95
      ax.axvspan(
          band_lo,
          band_hi,
          color='#fdd49e',
          alpha=0.35,
          label=(
              f'auto grid range of {unit}: [{grid.c_min:g} q50, {grid.c_max:g}'
              f' q95] = [{band_lo:.3g}, {band_hi:.3g}]'
          ),
      )
      ax.axvline(
          grid.q50,
          color='#08519c',
          lw=2,
          label=f'q50(T>0) = {grid.q50:.4g} (median positive T)',
      )
      q95_label = f'q95(T>0) = {grid.q95:.4g}'
      if np.isclose(grid.q95, t_ref, rtol=1e-6):
        q95_label += ' = T_ref'
      ax.axvline(grid.q95, color='#a50f15', lw=2, label=q95_label)
      if not np.isclose(grid.q95, t_ref, rtol=1e-6):
        ax.axvline(
            t_ref,
            color='#a50f15',
            lw=1.5,
            ls=':',
            label=f'T_ref = {t_ref_def} = {t_ref:.4g}',
        )
    else:
      ax.axvline(
          t_ref,
          color='#a50f15',
          lw=2,
          label=f'T_ref = {t_ref_def} = {t_ref:.4g}',
      )
    grid_label = f'grid points ({n_points} x {unit}, log-spaced)'
    for i, th in enumerate(finite_half):
      ax.axvline(
          th,
          color='grey',
          lw=0.6,
          alpha=0.7,
          label=grid_label if i == 0 else None,
      )
    mode = getattr(grid, 'mode', 'n/a')
    extensions = getattr(grid, 'extensions', 0)
    legend = ax.legend(
        fontsize=11,
        loc='upper right',
        title=(
            f'{family}; grid mode = {mode}, extensions = {extensions},'
            f" grid_edge_hit = {getattr(result, 'grid_edge_hit', 'n/a')}"
        ),
    )
    plt.setp(legend.get_title(), fontsize=11)
  else:
    ax.text(
        0.5, 0.5, 'no positive treatment values', ha='center', fontsize=14
    )
  ax.set_title('Grid construction', fontsize=18)
  ax.set_xlabel(
      'T on the Phase 0 scale T/c (rows with T > 0), log scale', fontsize=16
  )
  ax.set_ylabel('count', fontsize=16)
  ax.grid(True, alpha=0.3)


def plot_alpha_profile(
    alpha_grid,
    mean_loss,
    per_seed_loss,
    per_seed_alpha,
    alpha_hat,
    one_se_set,
    out_path: str,
    title: str,
    family: Optional[str] = None,
    t_normalized=None,
    stats_dict=None,
    t_ref: Optional[float] = None,
    t_axis_label: Optional[str] = None,
    stats_tags=None,
    footnote_lines=None,
    grid_result=None,
):
  """Plots the alpha profile chart ``summary_alpha_profile_<treatment>.png``.

  The per-seed panel is visualization only (no rule).
  With the optional `family` / `t_normalized` / `stats_dict` / `grid_result`
  arguments the chart is a 2 + 2 + 1 panel, from top to bottom:

  * row 1: **Grid construction** (how the half-saturation grid was built;
    needs `grid_result`) | **Seed-wise alpha distribution**;
  * row 2: **Profiled R-loss curve** | **Fitted saturation curve (normalized
    through (T_ref, 1))** (needs `family` / `t_normalized`; with
    `grid_result` it also shows the shapes the grid can express and names the
    FINAL curve);
  * row 3: **Key stats & process metrics**, one full-width cell mirroring the
    treatment's row of `summary_alpha_selection.csv`;
  * footnotes explaining every panel and naming the CSV columns behind it.

  Omitting all optional arguments reproduces the original two-panel chart
  (R-loss curve | seed-wise distribution).

  Args:
    alpha_grid: The alpha grid `G`.
    mean_loss: The aggregated curve `mean_s L_s(alpha)`.
    per_seed_loss: The `(S, G)` per-seed curves.
    per_seed_alpha: The `(S,)` per-seed argmins (diagnostic only).
    alpha_hat: The selected alpha.
    one_se_set: Boolean mask of the 1-SE set `A_1SE`.
    out_path: Destination png path.
    title: Chart title.
    family: Saturation family; required to draw the saturation-curve panel.
    t_normalized: Observed normalized treatment `T/c` on `D_sel`; required to
      draw the saturation-curve panel.
    stats_dict: Ordered mapping of metric name to value for the stats table.
    t_ref: Reference point `T_ref` (same `T/c` scale as `t_normalized`).
      When given the saturation-curve panel draws the normalized
      candidate `fbar` through `(T_ref, 1)` instead of the raw `f`.
    t_axis_label: Optional x-axis label of the saturation-curve panel that
      spells out the divisor `c` (see
      `alpha_selection.TreatmentDivisor.axis_label`).
    stats_tags: Optional mapping of stats-table metric name to the
      diagnostic it belongs to (`"diag1"` ... `"diag6"`); shown in the
      `diagnostics` column of the table.
    footnote_lines: Optional extra footnote lines appended after the standard
      panel explanations (e.g. the file list of the folder).
    grid_result: Optional alpha selection result (duck-typed: `grid`,
      `t_values`, `t_ref`, `grid_edge_hit`, ...) that drives the
      grid-construction panel and the FINAL-curve annotation.
  """
  alpha_grid = np.asarray(alpha_grid, dtype=float)
  mean_loss = np.asarray(mean_loss, dtype=float)
  per_seed_loss = np.asarray(per_seed_loss, dtype=float)

  draw_curve = family is not None and t_normalized is not None
  draw_stats = bool(stats_dict)
  draw_grid = grid_result is not None
  extended = draw_curve or draw_stats or draw_grid

  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  if extended:
    # 2 + 2 + 1 cells.  The stats row is sized from its number of rows (two
    # side-by-side tables, ~0.3 in per table row plus the title).
    panel_h = 8.5
    n_stats_rows = (len(stats_dict) + 1) // 2 + 1 if draw_stats else 2
    stats_h = max(2.5, 0.30 * n_stats_rows + 0.9)
    fig = plt.figure(figsize=(22, 2 * panel_h + stats_h))
    # No explicit hspace / wspace here: a GridSpec with local subplot params
    # is skipped by tight_layout, which `_add_footnote` relies on.
    gs = fig.add_gridspec(3, 2, height_ratios=[panel_h, panel_h, stats_h])
    ax_grid = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
  else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    ax_grid = ax3 = ax4 = None

  # --- Grid construction (row 1, left) -------------------------------------
  if ax_grid is not None:
    if draw_grid:
      _draw_grid_construction_panel(ax_grid, grid_result)
    else:
      ax_grid.axis('off')
      ax_grid.set_title('Grid construction', fontsize=18)
      ax_grid.text(
          0.5,
          0.5,
          'grid construction not available\n(no alpha selection result)',
          ha='center',
          va='center',
          fontsize=14,
          color='grey',
      )

  # --- Seed-wise alpha distribution (row 1, right) -------------------------
  per_seed_alpha = np.asarray(per_seed_alpha, dtype=float)
  if per_seed_alpha.size:
    ax2.hist(
        per_seed_alpha,
        bins=max(5, min(20, per_seed_alpha.size)),
        color='steelblue',
        alpha=0.8,
    )
  ax2.axvline(
      float(alpha_hat),
      color='red',
      linestyle='--',
      linewidth=2,
      label=f'alpha_hat = {float(alpha_hat):.4g}',
  )
  ax2.set_xlabel('alpha_hat per seed', fontsize=16)
  ax2.set_ylabel('count', fontsize=16)
  ax2.set_title('Seed-wise alpha distribution', fontsize=18)
  ax2.legend(fontsize=12)
  ax2.grid(True, alpha=0.3)

  # --- Profiled R-loss curve (row 2, left) ---------------------------------
  for row in per_seed_loss:
    ax1.plot(alpha_grid, row, color='grey', alpha=0.25, linewidth=1)
  ax1.plot(
      alpha_grid, mean_loss, color='darkgreen', linewidth=3, label='mean R-loss'
  )
  if one_se_set is not None and np.any(one_se_set):
    inside = alpha_grid[np.asarray(one_se_set, dtype=bool)]
    ax1.axvspan(
        float(inside.min()),
        float(inside.max()),
        color='orange',
        alpha=0.18,
        label='1-SE set (A_1SE)',
    )
  ax1.axvline(
      float(alpha_hat),
      color='red',
      linestyle='--',
      linewidth=2,
      label=f'alpha_hat = {float(alpha_hat):.4g}',
  )
  ax1.set_xlabel('alpha', fontsize=16)
  ax1.set_ylabel('profiled R-loss', fontsize=16)
  ax1.set_title('Profiled R-loss curve', fontsize=18)
  ax1.legend(fontsize=12)
  ax1.grid(True, alpha=0.3)

  # --- Fitted saturation curve (row 2, right) ------------------------------
  if ax3 is not None:
    if draw_curve:
      _draw_saturation_curve(
          ax3,
          family,
          alpha_hat,
          alpha_grid,
          one_se_set,
          t_normalized,
          t_ref=t_ref,
          t_axis_label=t_axis_label,
          grid_result=grid_result,
      )
    else:
      ax3.axis('off')

  # --- Key stats & process metrics (row 3, full width) ---------------------
  if ax4 is not None:
    if draw_stats:
      _draw_alpha_stats_table(
          ax4, stats_dict, stats_tags=stats_tags, two_columns=True
      )
    else:
      ax4.axis('off')

  fig.suptitle(title, fontsize=22)
  notes = _notes('Notes')
  if draw_grid:
    notes.extend(_grid_construction_notes(grid_result))
  notes.extend(_notes(
      '• Seed-wise alpha distribution: histogram of the per-seed argmins'
      ' alpha_hat_s (summary_alpha_seed-wise_alpha_distribution.csv:'
      ' alpha_hat_s). Visualization only -- alpha_hat comes from the averaged'
      ' curve, never from averaging these argmins; a wide spread is the'
      ' per-seed view of a flat profile.',
      '• Profiled R-loss curve: grey = R-loss curve L_hat_s(alpha) of each'
      ' seed s (one cluster-level K-fold partition per seed); dark green ='
      ' seed-averaged curve L_bar(alpha) = mean_s L_hat_s(alpha)'
      ' (summary_alpha_grid.csv: mean_r_loss; per-seed values in'
      ' summary_rloss_curves.csv: rloss_cate); red dashed = alpha_hat, the'
      ' argmin of L_bar (grid argmin + Brent refinement); orange band = span'
      ' of the 1-SE set A_1SE = {alpha : L_bar(alpha) - L_bar(alpha_hat) <='
      ' se(alpha)} (in_1se_set / se_alpha columns; Diag2 fires when the'
      ' half-saturation points inside A_1SE spread by more than'
      ' flatness_ratio).',
  ))
  if draw_curve:
    if t_ref is not None and np.isfinite(t_ref) and t_ref > 0:
      curve_note = (
          '• Fitted saturation curve: thick red = the FINAL curve Phases 1-4'
          ' use -- fbar(T; alpha_hat) = f(T; alpha_hat) / f(T_ref; alpha_hat),'
          ' the normalized curve Phases 1-3 regress on (theta = effect of'
          ' moving T from 0 to T_ref), or the identity candidate when the'
          ' decision was linear; grey dashed = identity candidate T / T_ref'
          ' (no saturation, not adopted); dark green dashed = an estimated'
          ' shape that was NOT adopted; red dot = (T_ref, 1), the point every'
          ' candidate passes through, T_ref = q95(T | T > 0); orange envelope'
          ' = curves of the smallest and largest alpha in A_1SE (a wide'
          ' envelope = weakly identified shape); black dotted = median of T/c;'
          ' blue histogram (right axis) = observed D_sel rows. x axis: the'
          ' normalized treatment T/c'
          + (f' ({t_axis_label})' if t_axis_label else '')
          + '.'
      )
      if draw_grid:
        _, _, final_text = _final_curve(grid_result)
        curve_note += (
            ' Thin purple lines, light to dark = fbar for the smallest /'
            ' median / largest grid alpha (the range of shapes the grid can'
            ' express). '
            + final_text
        )
      notes.append(curve_note)
    else:
      notes.append(
          '• Fitted saturation curve: thick red = f(T/c; alpha_hat); grey'
          ' dotted = straight line through the right end (no-saturation'
          ' reference); orange envelope = curves of the smallest and largest'
          ' alpha in A_1SE; black dotted = median of T/c; blue histogram'
          ' (right axis) = observed D_sel rows.'
      )
  if draw_stats:
    notes.append(
        '• Key stats & process metrics: the "diagnostics" column names the'
        ' Phase 0 diagnostic a row belongs to -- diag1 = basis'
        ' approximation error eps(J) (diag1_basis_approx_error.png), diag2 ='
        ' 1-SE flatness (diag2_rloss_profile.png), diag3 = residual-variation'
        ' ratio r_T = sd(T~)/sd(T/T_ref) (diag3_identification_diagnostics'
        '.csv), diag4 = shape quantiles phi_p(alpha_hat) = fbar(q_p(T>0);'
        ' alpha_hat) vs the linear references q_p/T_ref'
        ' (diag4_f_transform_quantiles.png), diag5 = saturation level'
        ' kappa(alpha_hat) = f(T_ref; alpha_hat) (diag5_saturation_at_T_ref'
        '.png; tanh/exp only), diag6 = unit consistency of T_ref, checked at'
        ' the start of Phase 1 (diag6_t_ref_consistency.csv). Blank = result'
        ' / process metric (summary_alpha_selection.json / .csv).'
    )
  if footnote_lines:
    notes.extend(footnote_lines)
  _add_footnote(fig, notes, fontsize=10.5, top_space=0.04)
  plt.savefig(out_path, dpi=200)
  plt.close()


# =============================================================================
# Phase 0 figures written to ``step0_saturation/``.
#
# The figures are always written by Phase 0 (regardless of warnings).  Every
# file name carries the prefix of the diagnostic it visualizes, or
# ``others_`` when no rule is attached (see :mod:`doubleml_pipeline.utils.io`):
#
# * ``diag1_basis_approx_error.png``: Diag1, ``eps(J)`` per grid alpha
#   against ``basis_tol``;
# * ``diag2_rloss_profile.png``: the averaged R-loss curves, the 1-SE set
#   ``A_1SE`` and the decision;
# * ``diag4_f_transform_quantiles.png``: Diag4, the four shape quantiles
#   ``phi_p(alpha)`` with the linear references ``q_p / T_ref``;
# * ``diag5_saturation_at_T_ref.png``: Diag5,
#   ``kappa(alpha) = f(T_ref; alpha)`` (tanh / exp only);
# * ``others_split_coverage.png``: cells per key level on the ``D_sel`` /
#   ``D_inf`` side.
#
# The grid construction and the per-seed argmins (visualization only) are
# panels of ``summary_alpha_profile_<treatment>.png``
# (:func:`plot_alpha_profile`, written by the orchestrator), not stand-alone
# files.
#
# Every figure ends with a footnote that explains its annotations and names
# the CSV columns / JSON keys it corresponds to, so the folder is readable
# on its own.
#
# The curves drawn are the normalized
# ``fbar(T; alpha) = f / f(T_ref; alpha)`` -- every candidate passes through
# ``(T_ref, 1)`` and the identity candidate is ``T / T_ref``.  The x axis of
# the profile figures is the grid variable (``T_1/2`` for tanh / exp,
# ``T_knee`` for log, alpha for power).
#
# The functions take the :class:`~doubleml_pipeline.evaluation.alpha_selection.
# AlphaSelectionResult` (duck-typed, so there is no import cycle) and a
# destination path; :func:`write_phase0_figures` writes all of them.
# =============================================================================

_DIAG4_PERCENTILES = (25, 50, 75, 90)
_POWER_FLATNESS_RANGE = 0.4


def _t_ref(result: Any) -> float:
  """``T_ref`` of the result; q95 of ``T > 0`` as a guard."""
  t_ref = float(getattr(result, 't_ref', float('nan')))
  if np.isfinite(t_ref) and t_ref > 0:
    return t_ref
  t_values = np.asarray(result.t_values, dtype=float)
  t_pos = t_values[np.isfinite(t_values) & (t_values > 0)]
  return float(np.quantile(t_pos, 0.95)) if t_pos.size else 1.0


def _x_axis(result: Any) -> Tuple[np.ndarray, str, bool]:
  """Returns the x values, label and log flag of the profile figures.

  The R-loss profile is drawn against the grid variable on a log
  axis: ``T_1/2`` for tanh / exp, ``T_knee`` for log.  The
  ``power`` family has no half-saturation point, so alpha is used instead.

  Args:
    result: The alpha selection result.

  Returns:
    ``(x_values, label, log_scale)``.
  """
  t_half = np.asarray(result.grid.t_half, dtype=float)
  if t_half.size and np.all(np.isfinite(t_half)) and np.all(t_half > 0):
    if result.family == 'log':
      return t_half, 'marginal-effect-halving point $T_{knee}$ (log)', True
    return t_half, 'half-saturation point $T_{1/2}$ (log)', True
  return np.asarray(result.grid.alphas, float), 'alpha', False


def _x_of_alpha(result: Any, alpha: float) -> float:
  """Maps an alpha value onto the x axis chosen by :func:`_x_axis`."""
  _, _, log_scale = _x_axis(result)
  if log_scale:
    return float(saturation.half_saturation_point(result.family, alpha))
  return float(alpha)


def _grid_unit(result: Any) -> str:
  """Short name of the grid variable: ``T_1/2``, ``T_knee`` or ``alpha``."""
  unit = getattr(result, 'grid_variable', '') or ''
  return unit or ('alpha' if result.family == 'power' else 'T_1/2')


def _t_ref_percent(result: Any) -> int:
  """The quantile of ``T > 0`` that defines ``T_ref``, in percent."""
  quantile = float(
      getattr(result, 't_ref_quantile', saturation.DEFAULT_T_REF_QUANTILE)
  )
  return int(round(100 * quantile))


def _t_ref_definition(result: Any) -> str:
  """``q95(T | T > 0)`` with the quantile actually configured."""
  return f'q{_t_ref_percent(result)}(T | T > 0)'


def _final_curve(result: Any) -> Tuple[str, float, str]:
  """Describes the curve Phases 1-4 actually use.

  Args:
    result: The alpha selection result.

  Returns:
    ``(kind, alpha_final, text)``: ``kind`` is ``"saturation"`` or
    ``"linear"``, ``alpha_final`` the alpha used downstream (the estimate
    when linear was adopted) and ``text`` a one-sentence statement.
  """
  selected = getattr(result, 'selected_family', None)
  alpha_est = float(getattr(result, 'alpha_hat_estimated', float('nan')))
  alpha_final = float(getattr(result, 'alpha_hat', alpha_est))
  reason = getattr(result, 'decision_reason', 'n/a')
  if selected is None:
    return (
        'linear',
        alpha_est,
        (
            'FINAL curve used by Phases 1-4: the identity candidate T / T_ref'
            ' (no saturation transform; selected_family = None, decision_reason'
            f' = {reason}). The estimated alpha_hat = {alpha_est:.4g} is shown'
            ' dashed for reference only.'
        ),
    )
  if bool(getattr(result, 'fallback_half_saturation_used', False)):
    return (
        'saturation',
        alpha_final,
        (
            'FINAL curve used by Phases 1-4: fbar(T; alpha ='
            f' {alpha_final:.4g}), fixed by'
            ' alpha_selection.fallback_half_saturation_T because Diag1 / Diag2'
            f' fired; the estimated alpha_hat = {alpha_est:.4g} is shown dashed'
            ' for reference.'
        ),
    )
  return (
      'saturation',
      alpha_final,
      (
          'FINAL curve used by Phases 1-4: fbar(T; alpha_hat ='
          f' {alpha_final:.4g}) (selected_family = {selected}, decision_reason'
          f' = {reason}).'
      ),
  )


def _shade_one_se_set(
    ax, result: Any, x_sorted: np.ndarray, order, log_scale: bool, label: str
) -> None:
  """Shades every grid point of ``A_1SE`` on a profile figure.

  Each member is shaded between the mid-points to its neighbours (geometric
  mid-points on a log axis), so a non-contiguous set shows as separate bands.

  Args:
    ax: The axes to draw on.
    result: The alpha selection result (``one_se_set`` is used).
    x_sorted: The x values of the grid points in increasing order.
    order: The permutation that sorted the grid (``np.argsort``).
    log_scale: Whether the x axis is logarithmic.
    label: Legend label of the first band.
  """
  one_se = getattr(result, 'one_se_set', None)
  if one_se is None or not np.any(one_se):
    return
  inside = np.asarray(one_se, bool)[order]
  xs = np.asarray(x_sorted, float)
  if xs.size > 1:
    if log_scale:
      mids = np.sqrt(xs[1:] * xs[:-1])
      edges_lo = np.concatenate([[xs[0] ** 2 / xs[1]], mids])
      edges_hi = np.concatenate([mids, [xs[-1] ** 2 / xs[-2]]])
    else:
      mids = (xs[1:] + xs[:-1]) / 2.0
      edges_lo = np.concatenate([[xs[0] - (xs[1] - xs[0]) / 2.0], mids])
      edges_hi = np.concatenate([mids, [xs[-1] + (xs[-1] - xs[-2]) / 2.0]])
  else:
    edges_lo = np.array([xs[0] * 0.9 if log_scale else xs[0] - 0.05])
    edges_hi = np.array([xs[0] * 1.1 if log_scale else xs[0] + 0.05])
  first = True
  for lo, hi, flag in zip(edges_lo, edges_hi, inside):
    if flag:
      ax.axvspan(
          lo, hi, color='#2ca02c', alpha=0.10, label=label if first else None
      )
      first = False


def _diag2_status(result: Any) -> str:
  """One sentence with the Diag2 statistic, threshold and outcome."""
  unit = _grid_unit(result)
  ratio = float(getattr(result, 'diag2_ratio', float('nan')))
  if result.family == 'power':
    rule = (
        f'alpha^max - alpha^min over A_1SE = {ratio:.3g} vs the fixed'
        f' threshold {_POWER_FLATNESS_RANGE:g}'
    )
  else:
    rule = (
        f'{unit}^max / {unit}^min over A_1SE = {ratio:.3g} vs flatness_ratio'
        f" = {float(getattr(result, 'flatness_ratio', 10.0)):g}"
    )
  fired = bool(getattr(result, 'diag2_fired', False))
  status = 'FIRED' if fired else 'not fired'
  if fired and getattr(result, 'selected_family', None) is None:
    status += ' (INFO only: selected_family = None)'
  return f'{rule} -> Diag2 {status}'


def plot_rloss_profile(result: Any, path: str) -> str:
  """Writes ``diag2_rloss_profile.png``.

  ``L_bar(alpha)`` with a +-1 se band (``se(alpha)`` of Diag2, paired against
  ``alpha_hat``), ``L_bar_ATE(alpha)`` (dotted; visualization only -- no
  decision and no argmin marker), the identity candidate ``L_bar_T`` as a
  horizontal line with a +-k se band, the shading of every grid point of
  ``A_1SE`` (contiguity is not required) and the marker at ``alpha_hat``.
  The title carries ``decision``; the footnote explains every element and
  the Diag2 rule.

  Args:
    result: The alpha selection result.
    path: Destination PNG path.

  Returns:
    ``path``.
  """
  x_vals, x_label, log_scale = _x_axis(result)
  order = np.argsort(x_vals)
  x_sorted = x_vals[order]
  mean_loss = np.asarray(result.mean_loss, float)[order]
  mean_ate = np.asarray(result.mean_loss_ate, float)[order]
  se_alpha = np.asarray(result.se_alpha, float)[order]
  n_seeds = int(np.size(result.per_seed_alpha))
  unit = _grid_unit(result)
  fig, ax = plt.subplots(figsize=(9.5, 5.8))
  ax.plot(
      x_sorted,
      mean_loss,
      'o-',
      color='#1f77b4',
      ms=3,
      label='$\\bar L(\\alpha)$ (CATE basis)',
  )
  band = np.where(np.isfinite(se_alpha), se_alpha, 0.0)
  ax.fill_between(
      x_sorted,
      mean_loss - band,
      mean_loss + band,
      color='#1f77b4',
      alpha=0.15,
      label='$\\pm 1\\,\\hat{se}$',
  )
  ax.plot(
      x_sorted,
      mean_ate,
      ':',
      color='#ff7f0e',
      lw=1.8,
      label='$\\bar L^{ATE}(\\alpha)$ (intercept; visualization only)',
  )
  identity = float(result.mean_loss_identity)
  if np.isfinite(identity):
    ax.axhline(
        identity,
        color='black',
        lw=1.2,
        label='$\\bar L_T$ (identity candidate $\\bar f_T = T / T_{ref}$)',
    )
    k_se = float(result.linear_test_k) * float(result.se)
    if np.isfinite(k_se):
      ax.axhspan(
          identity - k_se,
          identity + k_se,
          color='grey',
          alpha=0.15,
          label=f'$\\pm {result.linear_test_k:g}\\,\\hat{{se}}$',
      )
  _shade_one_se_set(
      ax, result, x_sorted, order, log_scale, label='$A_{1SE}$ (Diag2)'
  )
  ax.axvline(
      _x_of_alpha(result, result.alpha_hat_estimated),
      color='#d62728',
      lw=1.5,
      label='$\\hat\\alpha$',
  )
  if log_scale:
    ax.set_xscale('log')
  ax.set_xlabel(x_label)
  ax.set_ylabel('profiled R-loss')
  # decision_reason next to the decision.
  reason = getattr(result, 'decision_reason', None)
  reason_txt = f' [{reason}]' if reason else ''
  ax.set_title(
      f'R-loss profile ({result.family}) - decision: {result.decision}'
      f'{reason_txt}  (d_bar={result.d_bar:.3g}, se={result.se:.3g})'
  )
  ax.legend(fontsize=8)

  n_in = (
      int(np.count_nonzero(result.one_se_set))
      if result.one_se_set is not None
      else 0
  )
  notes = _notes(
      'Diag2 (1-SE flatness) -- notes',
      f'• L_bar(alpha) (blue): the out-of-fold profiled R-loss L_hat_s(alpha)'
      f' averaged over the S = {n_seeds} seeds, with the CATE basis p(X); x'
      f" axis = {unit} ({'log scale' if log_scale else 'linear scale'}). Lower"
      ' is better; alpha_hat (red line) is its argmin (+ Brent refinement).'
      ' Data: summary_alpha_grid.csv (alpha, half_saturation_T, mean_r_loss),'
      ' summary_rloss_curves.csv (rloss_cate per seed and grid point).',
      '• +-1 se band: se(alpha) = cluster-robust standard error of the'
      ' per-observation paired loss difference d_i(alpha) between alpha and'
      ' alpha_hat (same formula as the 1-SE decision; se_alpha column).',
      '• L_bar_ATE(alpha) (orange dotted): the same loss with the'
      ' intercept-only basis p(X) = 1, drawn to compare the profile with and'
      ' without the CATE basis; it enters no decision (rloss_ate column).',
      f'• L_bar_T (black line): loss of the identity candidate fbar_T ='
      ' T/T_ref, i.e. no saturation (rloss_identity column); grey band = +-k'
      f' se around it with k = linear_test_k = {result.linear_test_k:g}.'
      ' 1-SE decision: saturation is adopted iff d_bar = L_bar(alpha_hat) -'
      f' L_bar_T < -k se; here d_bar = {result.d_bar:.4g}, se ='
      f' {result.se:.4g} -> decision = {result.decision}'
      f" ({reason or 'n/a'}).",
      '• A_1SE (green shading) = {alpha : L_bar(alpha) - L_bar(alpha_hat) <='
      ' se(alpha)}: the grid points statistically indistinguishable from'
      f' alpha_hat (in_1se_set column; {n_in} of {int(x_vals.size)} points;'
      ' contiguity not required). Diag2 fires when the half-saturation'
      ' points inside A_1SE spread by more than flatness_ratio:'
      f' {_diag2_status(result)}. A firing means the half-saturation point is'
      ' not pinned down within one order of magnitude; Diag3 (r_T) and Diag4'
      ' show the cause.',
      '• Keys in summary_alpha_selection.json: rloss_min, d_bar, se,'
      ' linear_test_k, decision, decision_reason, diag2_fired, diag2_ratio,'
      ' flatness_ratio.',
  )
  _add_footnote(fig, notes)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def plot_f_transform_quantiles(result: Any, path: str) -> str:
  """Writes ``diag4_f_transform_quantiles.png``.

  The four shape quantiles ``phi_p(alpha) = fbar(q_p(T>0); alpha)`` of Diag4
  for ``p in {25, 50, 75, 90}`` against alpha, with horizontal lines at the
  linear references ``q_p / T_ref``; the gap between a line and its reference
  is the attenuation due to saturation at that spend level.

  Args:
    result: The alpha selection result.
    path: Destination PNG path.

  Returns:
    ``path``.
  """
  alphas = np.asarray(result.grid.alphas, float)
  order = np.argsort(alphas)
  phi = np.asarray(result.fbar_quantiles_grid, float)
  refs = np.asarray(result.linear_ref_quantiles, float)
  colors = ('#c7e9c0', '#74c476', '#31a354', '#006d2c')
  t_ref_def = _t_ref_definition(result)
  fig, ax = plt.subplots(figsize=(8.5, 5))
  for row, (p, color) in enumerate(zip(_DIAG4_PERCENTILES, colors)):
    ax.plot(
        alphas[order],
        phi[row][order],
        'o-',
        ms=3,
        color=color,
        label=f'$\\phi_{{{p}}}(\\alpha) = \\bar f(q_{{{p}}}(T>0); \\alpha)$',
    )
    if np.isfinite(refs[row]):
      ax.axhline(
          refs[row],
          color=color,
          ls='--',
          lw=1,
          label=f'linear reference $q_{{{p}}} / T_{{ref}}$',
      )
  ax.axhline(1.0, color='grey', ls=':', lw=1, label='fbar(T_ref) = 1')
  alpha_hat = float(result.alpha_hat_estimated)
  if np.isfinite(alpha_hat):
    ax.axvline(
        alpha_hat,
        color='#d62728',
        lw=1.2,
        label=f'$\\hat\\alpha$ = {alpha_hat:.4g}',
    )
  if np.all(alphas > 0) and result.grid.coord_kind != 'alpha':
    ax.set_xscale('log')
  ax.set_xlabel('alpha')
  ax.set_ylabel(r'$\bar{f}(q_p(T>0); \alpha)$')
  ax.set_title(
      f'Shape quantiles of the transformed treatment ({result.family})'
  )
  ax.legend(fontsize=7, ncol=2)
  at_hat = '; '.join(
      f'phi_{p} = {result.fbar_quantile(p):.3f} (linear ref'
      f' {result.linear_ref_quantile(p):.3f})'
      for p in _DIAG4_PERCENTILES
  )
  notes = _notes(
      'Diag4 (shape quantiles) -- notes (supporting information for Diag2;'
      ' no rule, no threshold)',
      '• phi_p(alpha) = fbar(q_p(T>0); alpha) (solid lines): the normalized'
      ' response at the p-th percentile of the POSITIVE treatment values'
      ' (p = 25, 50, 75, 90), as a function of alpha. By construction'
      f' fbar(T_ref) = 1 (grey dotted line), T_ref = {t_ref_def}.',
      '• Dashed lines: linear references q_p / T_ref = the value the identity'
      ' candidate (no saturation) takes at the same spend level. The vertical'
      ' gap between a solid line and its dashed reference is the attenuation'
      ' caused by saturation at that spend level.',
      '• Reading: phi_25 close to 1 means the lower quarter of positive spend'
      ' already achieves almost the effect of the q95 spend (practically an'
      " 'any spend' effect); phi_50 close to q50/T_ref means nearly linear."
      f' Values at alpha_hat (red line): {at_hat}.',
      '• Data: summary_alpha_grid.csv (fbar_q25, fbar_q50, fbar_q75, fbar_q90'
      ' per grid alpha); summary_alpha_selection.json (fbar_q25..fbar_q90 at'
      ' alpha_hat, linear_ref_q25..linear_ref_q90).',
  )
  _add_footnote(fig, notes)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def plot_basis_approx_error(result: Any, path: str) -> str:
  """Writes ``diag1_basis_approx_error.png``.

  ``eps(J)(alpha)``, the relative RMS error of approximating
  ``fbar(.; alpha)`` with the ``J`` T-basis functions, for every grid point,
  against ``basis_tol``; Diag1 fires when the maximum over the final grid is
  ``>= basis_tol``.

  Args:
    result: The alpha selection result.
    path: Destination PNG path.

  Returns:
    ``path``.
  """
  x_vals, x_label, log_scale = _x_axis(result)
  order = np.argsort(x_vals)
  x_sorted = x_vals[order]
  eps = getattr(result, 'basis_approx_error_alpha', None)
  n_basis = int(getattr(result, 'n_basis', 0))
  basis_tol = float(getattr(result, 'basis_tol', 1e-3))
  eps_max = float(getattr(result, 'basis_approx_error', float('nan')))
  fired = bool(getattr(result, 'diag1_fired', False))
  fig, ax = plt.subplots(figsize=(9.5, 5))
  if eps is not None and np.size(eps) == x_vals.size:
    eps_sorted = np.asarray(eps, float)[order]
    ax.plot(
        x_sorted,
        eps_sorted,
        'o-',
        color='#1f77b4',
        ms=3,
        label='$\\varepsilon(J)(\\alpha)$ per grid point',
    )
    finite = eps_sorted[np.isfinite(eps_sorted)]
    if finite.size and np.all(finite > 0):
      ax.set_yscale('log')
  else:
    ax.text(
        0.5,
        0.5,
        'eps(J) per alpha not available',
        ha='center',
        transform=ax.transAxes,
    )
  ax.axhline(
      basis_tol,
      color='#d62728',
      ls='--',
      lw=1.5,
      label=f'basis_tol = {basis_tol:g}',
  )
  _shade_one_se_set(
      ax, result, x_sorted, order, log_scale, label='$A_{1SE}$ (Diag2)'
  )
  alpha_hat = float(getattr(result, 'alpha_hat_estimated', float('nan')))
  if np.isfinite(alpha_hat):
    ax.axvline(
        _x_of_alpha(result, alpha_hat),
        color='#d62728',
        lw=1.2,
        label='$\\hat\\alpha$',
    )
  if log_scale:
    ax.set_xscale('log')
  ax.set_xlabel(x_label)
  ax.set_ylabel('$\\varepsilon(J)$: relative RMS approximation error')
  status = 'WARNING: fired' if fired else 'not fired'
  ax.set_title(
      'Diag1: basis approximation error eps(J) on the final grid (J ='
      f' {n_basis}, max eps = {eps_max:.2e}; {status})'
  )
  ax.legend(fontsize=8)
  notes = _notes(
      'Diag1 (basis approximation error) -- notes',
      '• The treatment-side nuisance E[fbar(T; alpha) | X] is obtained for'
      ' every alpha WITHOUT retraining: fbar(T; alpha) is approximated by J'
      ' B-spline basis functions b_j(T), fbar(T; alpha) ~ sum_j c_j(alpha)'
      ' b_j(T), and only the J nuisance regressions E[b_j(T) | X] are'
      ' trained.',
      '• eps(J)(alpha) = RMS of the approximation error over the observed T'
      ' divided by sd(fbar(T; alpha)) (a relative error). J is chosen on the'
      ' initial grid as the smallest J in [6, 20] with max_alpha eps(J) <'
      ' basis_tol and then kept fixed; after a grid extension eps(J) is'
      ' recomputed on the extended grid with the same J.',
      '• Diag1 fires (warning; the run continues) when max over the final'
      f' grid of eps(J) >= basis_tol = {basis_tol:g}: here J = {n_basis}, max'
      f' eps(J) = {eps_max:.2e} -> {status}. A non-negligible error would'
      ' favour the smoother alphas that are easier to approximate, i.e. bias'
      ' the comparison across alpha.',
      '• Green shading = A_1SE of Diag2 (the alphas indistinguishable from'
      ' alpha_hat), red line = alpha_hat, for orientation.',
      '• Data: summary_alpha_grid.csv (basis_approx_error_alpha per grid'
      ' point); summary_alpha_selection.json (J, basis_approx_error = max'
      ' eps(J), basis_tol, basis_mode, diag1_fired).',
  )
  _add_footnote(fig, notes)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def plot_saturation_at_t_ref(result: Any, path: str) -> str:
  """Writes ``diag5_saturation_at_T_ref.png`` (Diag5).

  ``kappa(alpha) = f(T_ref; alpha)`` over the grid for the bounded families
  (``tanh`` / ``exp``, where ``sup f = 1``): the share of the ceiling reached
  at ``T_ref``, with ``kappa(alpha_hat)`` marked.  Not defined for ``log`` /
  ``power``; the caller skips the figure for those families.

  Args:
    result: The alpha selection result.
    path: Destination PNG path.

  Returns:
    ``path``.
  """
  family = result.family
  x_vals, x_label, log_scale = _x_axis(result)
  order = np.argsort(x_vals)
  x_sorted = x_vals[order]
  alphas = np.asarray(result.grid.alphas, float)
  t_ref = _t_ref(result)
  kappa = np.asarray(
      saturation.apply_saturation(np.full(alphas.shape, t_ref), family, alphas),
      dtype=float,
  )[order]
  fig, ax = plt.subplots(figsize=(9.5, 5))
  ax.plot(
      x_sorted,
      kappa,
      'o-',
      color='#1f77b4',
      ms=3,
      label='$\\kappa(\\alpha) = f(T_{ref}; \\alpha)$ (share of ceiling)',
  )
  ax.axhline(0.5, color='grey', ls=':', lw=1, label='half of the ceiling')
  _shade_one_se_set(
      ax, result, x_sorted, order, log_scale, label='$A_{1SE}$ (Diag2)'
  )
  kappa_hat = getattr(result, 'saturation_at_t_ref', None)
  alpha_hat = float(getattr(result, 'alpha_hat', float('nan')))
  if kappa_hat is not None and np.isfinite(alpha_hat):
    x_hat = _x_of_alpha(result, alpha_hat)
    ax.axvline(x_hat, color='#d62728', lw=1.2)
    ax.plot(
        [x_hat],
        [float(kappa_hat)],
        marker='o',
        ms=8,
        color='#d62728',
        ls='none',
        label=f'$\\kappa(\\hat\\alpha)$ = {float(kappa_hat):.3f}',
    )
  if log_scale:
    ax.set_xscale('log')
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel(x_label)
  ax.set_ylabel('$\\kappa(\\alpha) = f(T_{ref}; \\alpha)$')
  kappa_txt = f'{float(kappa_hat):.3f}' if kappa_hat is not None else 'n/a'
  ax.set_title(
      f'Diag5: saturation level at T_ref ({family}); kappa(alpha_hat) ='
      f' {kappa_txt}'
  )
  ax.legend(fontsize=8)
  notes = _notes(
      'Diag5 (saturation level at T_ref) -- notes (always reported; no rule,'
      ' no threshold)',
      '• kappa(alpha) = f(T_ref; alpha): the share of the ceiling f(inf) = 1'
      f' that the raw {family} curve reaches at T_ref ='
      f' {_t_ref_definition(result)} = {t_ref:.4g}. After normalization every'
      ' fbar passes through (T_ref, 1), so the degree of saturation is not'
      ' visible in fbar itself; it appears as the ceiling fbar(inf; alpha) ='
      ' 1 / kappa(alpha).',
      f'• kappa(alpha_hat) = {kappa_txt} (red marker): e.g. 0.9 means the q95'
      ' spend already reaches 90% of the maximum attainable effect (strong'
      ' saturation inside the observed range); values far below 1 mean the'
      ' curve is still nearly linear at the top of the observed spend.',
      '• Green shading = A_1SE of Diag2: the range of kappa over it is the'
      ' uncertainty of the saturation level. Not defined for log / power'
      ' (no ceiling) -- this file is written for tanh / exp only.',
      '• Data: summary_alpha_selection.json (saturation_at_T_ref ='
      ' kappa(alpha_hat), T_ref, alpha_hat); summary_alpha_grid.csv (alpha,'
      ' half_saturation_T, in_1se_set).',
  )
  _add_footnote(fig, notes)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def plot_split_coverage(
    coverage: Optional[pd.DataFrame], path: str, result: Any = None
) -> str:
  """Writes ``others_split_coverage.png``.

  For every level of every adopted cluster key: the number of intersection
  cells on the ``D_sel`` and ``D_inf`` side (the ``n_cells_sel`` /
  ``n_cells_inf`` columns of ``others_split_coverage.csv``).  Levels that
  violate the coverage guarantee (a side with 0 cells) are printed in red.

  Args:
    coverage: The ``others_split_coverage.csv`` frame (may be empty).
    path: Destination PNG path.
    result: The alpha selection result (for ``K`` and the split unit).

  Returns:
    ``path``.
  """
  k_folds = int(getattr(result, 'k_folds', 0) or 0) if result else 0
  if coverage is None or coverage.empty:
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis('off')
    ax.text(
        0.5,
        0.5,
        'No cell-level split: the Stage A split used a single key or the'
        ' observation-level fallback,\nso others_split_coverage.csv has no'
        ' rows (coverage per key level is not defined).',
        ha='center',
        va='center',
        fontsize=10,
    )
    ax.set_title('Stage A split coverage')
  else:
    keys = list(dict.fromkeys(coverage['key'].astype(str)))
    n_levels_max = max(
        int((coverage['key'].astype(str) == k).sum()) for k in keys
    )
    height = min(14.0, max(4.5, 0.22 * n_levels_max + 1.5))
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(max(9.5, 4.8 * len(keys)), height),
        squeeze=False,
    )
    for ax, key in zip(axes[0], keys):
      sub = coverage[coverage['key'].astype(str) == key]
      y = np.arange(len(sub))
      ax.barh(
          y - 0.2,
          sub['n_cells_sel'].to_numpy(dtype=float),
          height=0.4,
          color='#1f77b4',
          label='n_cells_sel (cells in D_sel)',
      )
      ax.barh(
          y + 0.2,
          sub['n_cells_inf'].to_numpy(dtype=float),
          height=0.4,
          color='#ff7f0e',
          label='n_cells_inf (cells in D_inf)',
      )
      ax.set_yticks(y)
      labels = ax.set_yticklabels(
          sub['level'].astype(str), fontsize=7 if len(sub) > 25 else 8
      )
      covered = sub['covered'].astype(bool).to_numpy()
      for text, ok in zip(labels, covered):
        if not ok:
          text.set_color('red')
      n_cov = int(covered.sum())
      ax.set_title(
          f'key = {key} ({n_cov}/{len(sub)} levels covered)', fontsize=10
      )
      ax.set_xlabel('number of cells')
      ax.invert_yaxis()
      ax.legend(
          fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2
      )
    fig.suptitle(
        'Stage A split coverage: cells per key level on the D_sel / D_inf'
        ' side',
        fontsize=11,
    )
  notes = _notes(
      'Notes (Stage A split bookkeeping; no diagnostic rule)',
      '• The top-level split assigns whole intersection cells'
      ' (key1 x key2 of the two adopted cluster keys, e.g. geo x date block)'
      ' to D_sel (alpha selection) or D_inf (final estimation). Coverage'
      ' guarantee: every level of every key must appear on BOTH sides, so'
      ' that no geo / period is used only for one purpose.',
      '• Bars: n_cells_sel / n_cells_inf = number of cells of that level on'
      ' the D_sel / D_inf side; a level printed in red has 0 cells on one'
      ' side (covered = False). Data: others_split_coverage.csv -- key,'
      ' level, n_cells_sel, n_cells_inf, covered (both >= 1),'
      f' n_folds_covered_min (minimum over seeds of the number of the K ='
      f" {k_folds or 'K'} Stage B folds the level appears in), fold_covered"
      ' (n_folds_covered_min == K).',
      '• others_split_coverage_detail.csv lists the same cells per seed: seed,'
      ' cluster_id ("key1|key2"), key1_value, key2_value, fold (Stage B fold'
      ' of a D_sel cell; -1 for a D_inf cell), split (sel / inf; mixed only'
      ' under the observation-level fallback).',
  )
  _add_footnote(fig, notes, top_space=0.06)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def write_phase0_figures(
    result: Any, out_dir: str, coverage: Optional[pd.DataFrame] = None
) -> Dict[str, Optional[str]]:
  """Writes the Phase 0 figures; a failing figure never aborts Phase 0.

  Args:
    result: The alpha selection result.
    out_dir: ``step0_saturation`` directory (or its per-treatment subdir).
    coverage: The ``others_split_coverage.csv`` frame.

  Returns:
    Mapping ``file name -> path`` (``None`` if that figure failed).
  """
  written: Dict[str, Optional[str]] = {}
  jobs = [
      (pipeline_io.DIAG1_BASIS_APPROX_ERROR_PNG, plot_basis_approx_error),
      (pipeline_io.DIAG2_RLOSS_PROFILE_PNG, plot_rloss_profile),
      (
          pipeline_io.DIAG4_F_TRANSFORM_QUANTILES_PNG,
          plot_f_transform_quantiles,
      ),
  ]
  # Diag5: kappa is only defined for the bounded families.
  if result.family in saturation.BOUNDED_FAMILIES:
    jobs.append(
        (pipeline_io.DIAG5_SATURATION_AT_T_REF_PNG, plot_saturation_at_t_ref)
    )
  jobs.append((
      pipeline_io.SPLIT_COVERAGE_PNG,
      lambda res, p: plot_split_coverage(coverage, p, result=res),
  ))
  for name, func in jobs:
    path = os.path.join(out_dir, name)
    try:
      written[name] = func(result, path)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        KeyError,
        IndexError,
    ) as err:
      plt.close('all')
      print(f"   ⚠ Phase 0 figure '{name}' could not be drawn: {err}")
      written[name] = None
  return written
