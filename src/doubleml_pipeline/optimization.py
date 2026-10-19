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

"""Budget optimization (Phase 4): ROI metrics, allocations and their charts.

Two optimizers live here; the orchestrator (Phase 4 of the workflow) picks
one per treatment:

* :func:`optimize` -- the simple optimizer (``simple_optimization=True``):
  calculates ROI per row, cuts the spend of rows below the ROI threshold and
  outputs the CSV metrics and the ``*_timeseries_spend_vs_est.png`` charts.
* :func:`optimize_saturation` -- the saturation-aware optimizer, reached
  through ``optimize=True & simple_optimization=False`` (which additionally
  requires ``family != None``).

Both optimizers additionally write, per dimension (total / geo / item), three
two-row before / after charts -- ``*_timeseries_spend_before_after.png``
(actual spend on top, optimized spend below),
``*_timeseries_incremental_effect_before_after.png`` (red incremental effect
before, blue after) and ``*_timeseries_roi_before_after.png`` (red ROI per
date before, blue after) -- every panel on the exact width and x-axis layout
of ``*_timeseries_spend_vs_est.png`` and both panels of a chart on an
identical vertical scale (:func:`plot_before_after_charts`), the outcome
chart ``*_<outcome>.png`` (red observed outcome against the blue projected
outcome ``Y + (KPI_opt - KPI)`` under the optimized spend;
:func:`_plot_outcome_chart`, given an ``outcome_col``), and the yearly
before / after summary ``optimization_yearly_summary.csv`` (spend,
incremental KPI and ROI per calendar year and in total) that the HTML report
renders as a matrix.

.. note::
   :func:`optimize_saturation` supports ``treatment_type="spend"`` only.  The
   allocator budgets the treatment column directly, which presupposes
   ``g(T) = T`` -- the treatment *is* the money.  ``"percentage"``,
   ``"impressions"`` and ``"price"`` would need a cost model
   ``spend = g(treatment)`` to express the budget in currency; until that is
   specified the orchestrator forces ``simple_optimization=True`` for those
   types, i.e. routes them to :func:`optimize`.

The saturation-aware part

* solves the KKT allocation: one allocation unit = one row (item x geo x
  date) inside ``opt_periods`` and ``opt_geos x opt_items``; the cap
  ``T_bar_t`` = maximum actual treatment within the unit's geo x item cell;
  the ROI threshold ``r = opt_threshold_roi`` as the per-unit cap ``T_t^r``
  (``ROI_t(T) = theta_t fbar(T) / T >= r``), ``T_bar_t^eff = min(T_bar_t,
  T_t^r)``; ``budget_mode`` ``fixed`` (``K = (1 + budget_increment) x actual
  total``, one KKT pass with a single bisection on ``mu``;
  ``budget_reallocate`` chooses between ``K' = K`` with ``T_bar_t^eff`` and
  ``K' = K_+`` with ``T_bar_t`` followed by the truncation ``min(T^(1),
  T_t^r)``) and ``roi_target`` (``T_t* = T_bar_t^eff``, no budget).  The
  former greedy "exclude below the threshold -> re-solve -> iterate" loop
  no longer exists (:func:`allocate_saturation_budget`);
* writes the additional columns of
  ``consolidated_1a_full_df.csv``;
* writes ``optimization_metrics.csv`` including the incremental-KPI / ROI
  columns next to the actual and optimized total cost, and the statistics
  ``budget_mode``, ``K``, ``K_out``, ``unspent_budget``,
  ``marginal_roi_at_optimum``, ``n_units_excluded_negative_roi``,
  ``n_units_capped_by_roi``;
* draws the charts.  Each dimension (every geo, every
  item and the total) gets two two-line charts --
  ``*_original_vs_optimal_treatment.png`` (purple original vs green optimal
  treatment) and ``*_<outcome>.png`` (red original KPI vs blue estimated total
  KPI under the optimal treatment) -- plus the three before / after charts
  above; the former ``*_incremental_effect.png`` is superseded by
  ``*_timeseries_incremental_effect_before_after.png``.
"""

from collections.abc import Sequence
import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import saturation


def optimize(
    df_1a_consolidated: pd.DataFrame,
    treatment: str,
    treatment_amt: str,
    periods: tuple[str, str],
    threshold_roi: float = 1.0,
    other_cost_variables: Optional[Sequence[str]] = None,
    target_geo: str = "all",
    target_item: str = "all",
    outcome_col: Optional[str] = None,
    **kwargs,
):
  """Simple ROI-threshold optimization across treatments.

  Evaluates ROI per row (instance) in ``df_1a_consolidated`` across the
  specified ``periods``, cuts spend on rows with ``ROI < threshold_roi``, and
  saves aggregated metrics to ``optimization_metrics.csv`` along with
  visualization overlays highlighting the optimized spend.

  Args:
    df_1a_consolidated: Consolidated DataFrame.
    treatment: Name of the treatment variable.
    treatment_amt: Amount of treatment.
    periods: A tuple of (start_date, end_date) specifying the optimization
      periods.
    threshold_roi: Minimum ROI threshold per row for optimization (default:
      1.0).
    other_cost_variables: List of cost variables to consider.
    target_geo: Target geography for optimization (default: 'all').
    target_item: Target item for optimization (default: 'all').
    outcome_col: The outcome column ``Y``.  When given (and present), the
      ``<prefix>_<outcome>.png`` chart of every dimension compares the
      observed outcome with the projected outcome under the optimized spend,
      ``Y + (KPI_opt - KPI)`` -- the same chart, in the same position of the
      HTML report, as the saturation-aware optimizer's.
    **kwargs: Additional configuration parameters such as date_col, geo_col,
      item_col, sales_col, and out_dir.
  """
  if other_cost_variables is None:
    other_cost_variables = []

  date_col = kwargs.get("date_col", "Date")
  geo_col = kwargs.get("geo_col", "Geo")
  item_col = kwargs.get("item_col", "Item")
  sales_col = kwargs.get("sales_col", "Sales")
  out_dir = kwargs.get("out_dir", "")

  kpi_col = f"estimated_incremental_KPI_{treatment}"

  # Ensure output directory exists
  opt_dir = os.path.join(out_dir, "007_optimization")
  os.makedirs(opt_dir, exist_ok=True)

  df_work = df_1a_consolidated.copy()
  if date_col in df_work.columns:
    df_work[date_col] = pd.to_datetime(df_work[date_col])

  start_date = pd.to_datetime(periods[0])
  end_date = pd.to_datetime(periods[1])

  # 1. Output 1: CSV metrics (threshold_roi applied per row / instance)
  df_opt = df_work[
      (df_work[date_col] >= start_date) & (df_work[date_col] <= end_date)
  ].copy()

  geos = [target_geo] if target_geo != "all" else ["all"]
  if target_geo == "all" and geo_col and geo_col in df_work.columns:
    geos.extend(sorted(df_work[geo_col].dropna().unique()))

  items = [target_item] if target_item != "all" else ["all"]
  if target_item == "all" and item_col and item_col in df_work.columns:
    items.extend(sorted(df_work[item_col].dropna().unique()))

  results = []
  for g in geos:
    for i in items:
      df_sub = df_opt.copy()
      if g != "all":
        df_sub = df_sub[df_sub[geo_col] == g]
      if i != "all":
        df_sub = df_sub[df_sub[item_col] == i]

      if df_sub.empty:
        continue

      if (
          treatment_amt not in df_sub.columns
          or kpi_col not in df_sub.columns
          or not pd.api.types.is_numeric_dtype(df_sub[treatment_amt])
          or not pd.api.types.is_numeric_dtype(df_sub[kpi_col])
      ):
        continue

      spend_vals = (
          pd.to_numeric(df_sub[treatment_amt], errors="coerce")
          .fillna(0.0)
          .to_numpy(dtype=float)
      )
      kpi_vals = (
          pd.to_numeric(df_sub[kpi_col], errors="coerce")
          .fillna(0.0)
          .to_numpy(dtype=float)
      )
      row_roi = np.divide(
          kpi_vals,
          spend_vals,
          out=np.zeros_like(kpi_vals, dtype=float),
          where=spend_vals != 0,
      )

      total_cost = float(spend_vals.sum())
      mask = row_roi < threshold_roi
      reduction = float(spend_vals[mask].sum())
      optimized_total = total_cost - reduction

      total_cost_cv = 0.0
      for cv in other_cost_variables:
        if cv in df_sub.columns and pd.api.types.is_numeric_dtype(df_sub[cv]):
          total_cost_cv += float(
              pd.to_numeric(df_sub[cv], errors="coerce").fillna(0.0).sum()
          )

      increase_pct = reduction / total_cost_cv if total_cost_cv != 0 else 0.0

      results.append({
          "treatment": treatment,
          "other_cost_variables": other_cost_variables,
          "target_geo": g,
          "target_item": i,
          "periods_start_date": periods[0],
          "periods_end_date": periods[1],
          "threshold_roi": threshold_roi,
          "total_cost_of_treatment_in_periods": total_cost,
          "optimized_total_cost_of_treatment_in_periods": optimized_total,
          "reduction_of_cost_of_treatment": reduction,
          "total_cost_of_other_cost_variables": total_cost_cv,
          "increase_pct_of_total_cost_of_other_cost_variables": increase_pct,
      })

  if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(opt_dir, "optimization_metrics.csv"), index=False
    )

  # 2. Output 2: Visualization overlays
  # Generate charts for Total, Geo, and Item levels across the full dataset
  # (not just periods), aggregating the row-level optimized spend by date.
  levels_list = [("total", None)]
  if geo_col and geo_col in df_work.columns:
    for g_val in sorted(df_work[geo_col].dropna().unique()):
      levels_list.append(("geo", g_val))
  if item_col and item_col in df_work.columns:
    for i_val in sorted(df_work[item_col].dropna().unique()):
      levels_list.append(("item", i_val))

  has_roi_cols = (
      date_col in df_work.columns
      and treatment_amt in df_work.columns
      and kpi_col in df_work.columns
      and pd.api.types.is_numeric_dtype(df_work[treatment_amt])
      and pd.api.types.is_numeric_dtype(df_work[kpi_col])
  )
  opt_spend_col = "__optimized_spend__"
  opt_kpi_col = "__optimized_kpi__"
  opt_outcome_col = "__optimized_outcome__"
  if has_roi_cols:
    spend_all = (
        pd.to_numeric(df_work[treatment_amt], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    kpi_all = (
        pd.to_numeric(df_work[kpi_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    row_roi_all = np.divide(
        kpi_all,
        spend_all,
        out=np.zeros_like(kpi_all, dtype=float),
        where=spend_all != 0,
    )
    in_period_all = (
        (df_work[date_col] >= start_date) & (df_work[date_col] <= end_date)
    ).to_numpy()
    keep_row_mask = (~in_period_all) | (row_roi_all >= threshold_roi)
    df_work[opt_spend_col] = np.where(keep_row_mask, spend_all, 0.0)
    # A cut row loses its spend AND the incremental KPI that spend produced.
    kpi_opt_all = np.where(keep_row_mask, kpi_all, 0.0)
    df_work[opt_kpi_col] = kpi_opt_all
    if outcome_col and outcome_col in df_work.columns:
      # Projected outcome under the optimized spend, the same bridge as the
      # saturation-aware optimizer's `optimized_<outcome>`: the observed
      # outcome plus the change in incremental KPI (here: minus the KPI of
      # the cut rows).
      y_all = (
          pd.to_numeric(df_work[outcome_col], errors="coerce")
          .fillna(0.0)
          .to_numpy(dtype=float)
      )
      df_work[opt_outcome_col] = y_all + (kpi_opt_all - kpi_all)

  agg_dict = {
      col: "sum"
      for col in [
          sales_col,
          treatment_amt,
          kpi_col,
          opt_spend_col,
          opt_kpi_col,
          outcome_col,
          opt_outcome_col,
      ]
      if col in df_work.columns
  }

  if date_col in df_work.columns and has_roi_cols:
    # Yearly before / after summary (spend, incremental KPI, ROI) per
    # dimension: the data behind the matrix shown under the charts in the
    # HTML report.
    write_yearly_summary(
        df_work,
        treatment=treatment,
        date_col=date_col,
        geo_col=geo_col,
        item_col=item_col,
        spend_col=treatment_amt,
        opt_spend_col=opt_spend_col,
        kpi_col=kpi_col,
        opt_kpi_col=opt_kpi_col,
        out_dir=opt_dir,
    )
    for level_type, level_val in levels_list:
      if level_type == "total":
        chart_df = df_work
        prefix = "total"
        title_prefix = "Total"
      elif level_type == "geo":
        chart_df = df_work[df_work[geo_col] == level_val]
        prefix = f"geo_{level_val}"
        title_prefix = f"[{level_val}]"
      elif level_type == "item":
        chart_df = df_work[df_work[item_col] == level_val]
        prefix = f"item_{level_val}"
        title_prefix = f"[{level_val}]"
      else:
        continue

      if chart_df.empty or not agg_dict:
        continue

      df_ts = (
          chart_df.groupby(date_col, as_index=False, observed=False)
          .agg(agg_dict)
          .sort_values(date_col)
      )

      if opt_spend_col not in df_ts.columns:
        continue
      green_line = df_ts[opt_spend_col].to_numpy(dtype=float)

      fig, (ax,) = _timeseries_figure(1)

      if treatment_amt in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[treatment_amt],
            color="purple",
            alpha=0.8,
            linewidth=3,
            label=f"Spend ({treatment_amt})",
        )
      if kpi_col in df_ts.columns:
        ax.plot(
            df_ts[date_col],
            df_ts[kpi_col],
            color="red",
            linewidth=2,
            label=f"Est. Incremental Sales ({kpi_col})",
        )

      # Add the new threshold line and fill
      ax.plot(
          df_ts[date_col],
          green_line,
          color="darkgreen",
          linewidth=3,
          label=f"improved_Spend_with_roi_{threshold_roi}",
      )
      ax.fill_between(
          df_ts[date_col],
          df_ts[treatment_amt],
          green_line,
          color="green",
          alpha=0.3,
          label="Reduced Spend",
      )

      _style_timeseries_panel(
          ax,
          f"{title_prefix} Time Series with ROI Optimization"
          f" (Threshold={threshold_roi})",
          "Amount",
      )
      _style_date_axis(ax)

      out_path = os.path.join(opt_dir, f"{prefix}_timeseries_spend_vs_est.png")
      fig.savefig(out_path, dpi=200)
      plt.close(fig)

      # `<prefix>_<outcome>.png`: observed vs projected outcome, the chart
      # the saturation-aware optimizer shows in this position (skipped when
      # no `outcome_col` was given).
      _plot_outcome_chart(
          df_ts,
          date_col=date_col,
          prefix=prefix,
          title_prefix=title_prefix,
          outcome_col=outcome_col,
          opt_outcome_col=opt_outcome_col,
          opt_dir=opt_dir,
          threshold_roi=threshold_roi,
          start_date=start_date,
          end_date=end_date,
      )

      plot_before_after_charts(
          df_ts,
          date_col=date_col,
          prefix=prefix,
          title_prefix=title_prefix,
          spend_col=treatment_amt,
          opt_spend_col=opt_spend_col,
          kpi_col=kpi_col,
          opt_kpi_col=opt_kpi_col,
          opt_dir=opt_dir,
          threshold_roi=threshold_roi,
          start_date=start_date,
          end_date=end_date,
      )


# =============================================================================
# Before / after optimization: the 2-row spend chart and the yearly summary.
# Shared by the simple optimizer (:func:`optimize`) and the saturation-aware
# optimizer (:func:`optimize_saturation`).
# =============================================================================

# One row per (dimension, level, year | "Total"): spend, incremental KPI and
# ROI before and after the optimization.  Read by the HTML report to render
# the yearly matrix under the before / after chart.
YEARLY_SUMMARY_CSV = "optimization_yearly_summary.csv"
YEARLY_SUMMARY_COLUMNS = (
    "treatment",
    "dimension",
    "level",
    "year",
    "spend_before",
    "incremental_kpi_before",
    "roi_before",
    "spend_after",
    "incremental_kpi_after",
    "roi_after",
)


def _dimension_levels(
    frame: pd.DataFrame, geo_col: Optional[str], item_col: Optional[str]
) -> List[Tuple[str, Any]]:
  """``[("total", None), ("geo", g), ..., ("item", i), ...]`` of ``frame``."""
  levels: List[Tuple[str, Any]] = [("total", None)]
  if geo_col and geo_col in frame.columns:
    for value in sorted(frame[geo_col].dropna().unique()):
      levels.append(("geo", value))
  if item_col and item_col in frame.columns:
    for value in sorted(frame[item_col].dropna().unique()):
      levels.append(("item", value))
  return levels


def yearly_summary_frame(
    df_work: pd.DataFrame,
    treatment: str,
    date_col: str,
    geo_col: Optional[str],
    item_col: Optional[str],
    spend_col: str,
    opt_spend_col: str,
    kpi_col: str,
    opt_kpi_col: str,
) -> pd.DataFrame:
  """Builds the yearly before / after summary of an optimization.

  For every dimension (``total``, each geo, each item) and every calendar
  year of the frame plus a ``Total`` row: the spend, the incremental KPI and
  their ratio (ROI) before the optimization (the actual columns) and after it
  (the ``opt_*`` columns).  Rows outside the optimization period keep their
  actual values in the ``opt_*`` columns, so the yearly totals describe the
  whole data set exactly like the time-series charts do.

  Args:
    df_work: Row-level frame carrying all the columns below.
    treatment: Treatment column name (recorded in the ``treatment`` column).
    date_col: Date column (parsed with ``pd.to_datetime``).
    geo_col: Optional geography column.
    item_col: Optional item column.
    spend_col: Actual spend column.
    opt_spend_col: Optimized spend column.
    kpi_col: ``estimated_incremental_KPI_<treatment>``.
    opt_kpi_col: Optimized incremental KPI column.

  Returns:
    A frame with :data:`YEARLY_SUMMARY_COLUMNS`.
  """

  def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
      return np.zeros(len(frame), dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy()

  dates = pd.to_datetime(df_work[date_col], errors="coerce")
  years = dates.dt.year
  rows = []
  for dim, level in _dimension_levels(df_work, geo_col, item_col):
    if dim == "total":
      mask = np.ones(len(df_work), dtype=bool)
    elif dim == "geo":
      mask = (df_work[geo_col] == level).to_numpy()
    else:
      mask = (df_work[item_col] == level).to_numpy()
    sub = df_work.loc[mask]
    sub_years = years.loc[mask]
    year_values = [int(y) for y in sorted(sub_years.dropna().unique())]
    for year in year_values + ["Total"]:
      part = sub if year == "Total" else sub.loc[(sub_years == year).to_numpy()]
      spend_before = float(_col(part, spend_col).sum())
      kpi_before = float(_col(part, kpi_col).sum())
      spend_after = float(_col(part, opt_spend_col).sum())
      kpi_after = float(_col(part, opt_kpi_col).sum())
      rows.append({
          "treatment": treatment,
          "dimension": dim,
          "level": "all" if level is None else level,
          "year": str(year),
          "spend_before": spend_before,
          "incremental_kpi_before": kpi_before,
          "roi_before": kpi_before / spend_before if spend_before else 0.0,
          "spend_after": spend_after,
          "incremental_kpi_after": kpi_after,
          "roi_after": kpi_after / spend_after if spend_after else 0.0,
      })
  return pd.DataFrame(rows, columns=list(YEARLY_SUMMARY_COLUMNS))


def write_yearly_summary(df_work: pd.DataFrame, out_dir: str, **kwargs) -> str:
  """Writes :data:`YEARLY_SUMMARY_CSV` into ``out_dir``.

  Args:
    df_work: See :func:`yearly_summary_frame`.
    out_dir: Destination directory (``.../007_optimization``).
    **kwargs: Forwarded to :func:`yearly_summary_frame`.

  Returns:
    The CSV path.
  """
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, YEARLY_SUMMARY_CSV)
  yearly_summary_frame(df_work, **kwargs).to_csv(path, index=False)
  return path


# -----------------------------------------------------------------------------
# Shared geometry of every 007_optimization time-series chart.
#
# All charts of the folder -- the single-panel `*_timeseries_spend_vs_est.png`
# and saturation charts as well as the stacked before / after charts -- are
# built from 24 inch wide panels with margins fixed in INCHES.  The x axis of
# every png therefore starts and ends at the same pixel column whatever the
# width of its y tick labels, so the dates line up exactly when the HTML
# report stacks the images at a common width.  The single-panel charts are
# 10 in tall; every panel of the stacked before / after charts (spend,
# incremental effect, ROI) is exactly half that, so a before / after pair is
# as tall as one single-panel chart.
# -----------------------------------------------------------------------------
_TS_PANEL_WIDTH_IN = 24.0
_TS_PANEL_HEIGHT_IN = 10.0
_TS_MARGIN_LEFT_IN = 1.7  # y label + tick labels
_TS_MARGIN_RIGHT_IN = 0.4
_TS_MARGIN_TOP_IN = 0.9  # panel title
_TS_MARGIN_BOTTOM_IN = 1.7  # rotated date labels + x label
_TS_PANEL_GAP_IN = 1.3  # room for the title of the lower panel
# Height of each panel of the three stacked before / after charts.  A single
# constant so the spend, incremental effect and ROI panels always match.
_TS_BEFORE_AFTER_PANEL_HEIGHT_IN = _TS_PANEL_HEIGHT_IN / 2.0

# Colours of the before (red) / after (blue) lines and of the fill under them.
_BEFORE_COLOR, _BEFORE_FILL = "red", "#f4b6b6"
_AFTER_COLOR, _AFTER_FILL = "blue", "#b6c8f4"


def _timeseries_figure(n_panels: int, panel_height_in: Optional[float] = None):
  """A figure of ``n_panels`` stacked 24 in wide panels sharing the x axis.

  Args:
    n_panels: Number of vertically stacked panels.
    panel_height_in: Height of every panel in inches; defaults to the 10 in of
      ``*_timeseries_spend_vs_est.png``.  The width and the margins never
      change, so the x axis keeps its exact layout whatever the height.

  Returns:
    ``(fig, axes)`` with ``axes`` a list of length ``n_panels``, top first.
  """
  width = _TS_PANEL_WIDTH_IN
  panel_h = _TS_PANEL_HEIGHT_IN if panel_height_in is None else panel_height_in
  height = (
      n_panels * panel_h
      + (n_panels - 1) * _TS_PANEL_GAP_IN
      + _TS_MARGIN_TOP_IN
      + _TS_MARGIN_BOTTOM_IN
  )
  fig, axes = plt.subplots(
      n_panels,
      1,
      figsize=(width, height),
      sharex=True,
      squeeze=False,
      gridspec_kw={
          "left": _TS_MARGIN_LEFT_IN / width,
          "right": 1.0 - _TS_MARGIN_RIGHT_IN / width,
          "top": 1.0 - _TS_MARGIN_TOP_IN / height,
          "bottom": _TS_MARGIN_BOTTOM_IN / height,
          "hspace": _TS_PANEL_GAP_IN / panel_h,
      },
  )
  return fig, list(axes[:, 0])


def _style_timeseries_panel(ax, title: str, y_label: str) -> None:
  """Title, y label, tick size, legend and grid shared by every panel."""
  ax.set_title(title, fontsize=24)
  ax.set_ylabel(y_label, fontsize=20)
  ax.tick_params(axis="y", labelsize=16)
  ax.legend(fontsize=16, loc="upper left")
  ax.grid(True, alpha=0.3)


def _style_date_axis(ax) -> None:
  """The shared date axis: Monday ticks, ``%Y/%m/%d`` labels rotated 90."""
  ax.set_xlabel("Date", fontsize=20)
  ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
  plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)


def _mark_period(ax, start_date: Any, end_date: Any) -> None:
  """Dashed markers at the start and end of the optimization period."""
  if start_date is not None and end_date is not None:
    ax.axvline(start_date, color="grey", linestyle="--", alpha=0.6)
    ax.axvline(end_date, color="grey", linestyle="--", alpha=0.6)


def _numeric(values: Any) -> np.ndarray:
  """``values`` as a float array with NaN / non-numeric entries set to 0."""
  return (
      pd.to_numeric(pd.Series(values), errors="coerce")
      .fillna(0.0)
      .to_numpy(dtype=float)
  )


def _ratio(numerator: Any, denominator: Any) -> np.ndarray:
  """Elementwise ``numerator / denominator`` with ``0`` where the latter is 0.

  The pipeline's ROI convention (``optimization_yearly_summary.csv``, the ROI
  bar charts): a date without spend has ROI ``0``, not NaN.

  Args:
    numerator: The incremental KPI values.
    denominator: The spend values.

  Returns:
    The ratio array.
  """
  numerator = np.asarray(numerator, dtype=float)
  denominator = np.asarray(denominator, dtype=float)
  return np.divide(
      numerator,
      denominator,
      out=np.zeros_like(numerator, dtype=float),
      where=denominator != 0,
  )


def _plot_before_after_panels(
    dates: Any,
    before: Any,
    after: Any,
    out_path: str,
    titles: Tuple[str, str],
    labels: Tuple[str, str],
    colors: Tuple[str, str],
    fills: Tuple[str, str],
    y_label: str,
    start_date: Any = None,
    end_date: Any = None,
    reference: Optional[Tuple[float, str]] = None,
    panel_height_in: float = _TS_BEFORE_AFTER_PANEL_HEIGHT_IN,
) -> None:
  """Two stacked panels on an IDENTICAL vertical scale: before on top.

  Both panels belong to the shared 007 geometry, i.e. each one has exactly the
  width and x-axis layout of ``*_timeseries_spend_vs_est.png`` and, by
  default, half its height.  The y limits come from the data alone, so the
  vertical scale does not depend on the panel height.

  Args:
    dates: The x values (date column of the aggregated frame).
    before: Series before the optimization (top panel).
    after: Series after the optimization (bottom panel).
    out_path: Destination png path.
    titles: ``(top title, bottom title)``.
    labels: ``(top legend label, bottom legend label)``.
    colors: ``(top line colour, bottom line colour)``.
    fills: ``(top fill colour, bottom fill colour)`` under the lines.
    y_label: Y-axis label of both panels.
    start_date: Optional inclusive start of the optimization period (dashed
      marker on both panels).
    end_date: Optional inclusive end of the optimization period.
    reference: Optional ``(value, label)`` horizontal reference line drawn on
      both panels (e.g. the ROI threshold).
    panel_height_in: Height of each panel in inches (default: half the 10 in
      of ``*_timeseries_spend_vs_est.png``, shared by the spend, incremental
      effect and ROI charts).
  """
  before = _numeric(before)
  after = _numeric(after)
  extremes = [before.max(), after.max(), before.min(), after.min(), 0.0]
  if reference is not None:
    extremes.append(float(reference[0]))
  y_max = float(np.nanmax(extremes))
  y_max = y_max * 1.08 if y_max > 0 else 1.0
  y_min = float(np.nanmin(extremes))
  y_min = y_min * 1.08 if y_min < 0 else 0.0

  fig, (ax_top, ax_bot) = _timeseries_figure(2, panel_height_in)
  for ax, values, color, fill, label, title in (
      (ax_top, before, colors[0], fills[0], labels[0], titles[0]),
      (ax_bot, after, colors[1], fills[1], labels[1], titles[1]),
  ):
    ax.plot(dates, values, color=color, linewidth=3, label=label)
    ax.fill_between(dates, 0.0, values, color=fill, alpha=0.5)
    if reference is not None:
      ax.axhline(
          reference[0],
          color="black",
          linestyle=":",
          linewidth=1.5,
          label=reference[1],
      )
    _mark_period(ax, start_date, end_date)
    ax.set_ylim(y_min, y_max)
    _style_timeseries_panel(ax, title, y_label)
  _style_date_axis(ax_bot)
  fig.savefig(out_path, dpi=200)
  plt.close(fig)


def _threshold_text(threshold_roi: Optional[float]) -> str:
  return f" (Threshold={threshold_roi})" if threshold_roi is not None else ""


def plot_spend_before_after(
    df_ts: pd.DataFrame,
    date_col: str,
    spend_col: str,
    opt_spend_col: str,
    out_path: str,
    title_prefix: str,
    spend_label: str,
    threshold_roi: Optional[float] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> None:
  """Writes ``<prefix>_timeseries_spend_before_after.png``.

  Two charts stacked vertically (2 rows x 1 column) on an IDENTICAL vertical
  scale: the actual spend (``treatment_spend_cols``) on top and the optimized
  spend below, so the size of every cut (or increase) can be read directly.
  Each panel has the width and x-axis layout of
  ``*_timeseries_spend_vs_est.png`` and exactly half its height -- the same
  panel size as the incremental effect and ROI charts, so the three before /
  after charts stack evenly.

  Args:
    df_ts: Date-aggregated frame of one dimension, sorted by date.
    date_col: Date column.
    spend_col: Actual spend column.
    opt_spend_col: Optimized spend column.
    out_path: Destination png path.
    title_prefix: ``Total`` / ``[<geo>]`` / ``[<item>]``.
    spend_label: Legend label of the actual spend line.
    threshold_roi: The ROI floor, annotated in the title when given.
    start_date: Optional inclusive start of the optimization period (dashed
      marker on both charts).
    end_date: Optional inclusive end of the optimization period.
  """
  if spend_col not in df_ts.columns or opt_spend_col not in df_ts.columns:
    return
  _plot_before_after_panels(
      df_ts[date_col],
      df_ts[spend_col],
      df_ts[opt_spend_col],
      out_path,
      titles=(
          f"{title_prefix} Spend before optimization",
          f"{title_prefix} Spend after optimization"
          f"{_threshold_text(threshold_roi)}",
      ),
      labels=(spend_label, "Optimized spend"),
      colors=("purple", "darkgreen"),
      fills=("#d8b4e2", "#b7dfb9"),
      y_label="Amount",
      start_date=start_date,
      end_date=end_date,
  )


def plot_incremental_effect_before_after(
    df_ts: pd.DataFrame,
    date_col: str,
    kpi_col: str,
    opt_kpi_col: str,
    out_path: str,
    title_prefix: str,
    threshold_roi: Optional[float] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> None:
  """Writes ``<prefix>_timeseries_incremental_effect_before_after.png``.

  Two line charts stacked vertically on an IDENTICAL vertical scale: the
  estimated incremental effect of the actual spend (red, top) and the
  incremental effect under the optimized spend (blue, bottom).  Each panel has
  the width and x-axis layout of ``*_timeseries_spend_vs_est.png`` and exactly
  half its height, so the pair gives an at-a-glance overview.

  Args:
    df_ts: Date-aggregated frame of one dimension, sorted by date.
    date_col: Date column.
    kpi_col: ``estimated_incremental_KPI_<treatment>``.
    opt_kpi_col: Incremental KPI under the optimized spend.
    out_path: Destination png path.
    title_prefix: ``Total`` / ``[<geo>]`` / ``[<item>]``.
    threshold_roi: The ROI floor, annotated in the title when given.
    start_date: Optional inclusive start of the optimization period.
    end_date: Optional inclusive end of the optimization period.
  """
  if kpi_col not in df_ts.columns or opt_kpi_col not in df_ts.columns:
    return
  _plot_before_after_panels(
      df_ts[date_col],
      df_ts[kpi_col],
      df_ts[opt_kpi_col],
      out_path,
      titles=(
          f"{title_prefix} Incremental Effect before optimization",
          f"{title_prefix} Incremental Effect after optimization"
          f"{_threshold_text(threshold_roi)}",
      ),
      labels=(
          f"Incremental Effect before optimization ({kpi_col})",
          "Incremental Effect after optimization",
      ),
      colors=(_BEFORE_COLOR, _AFTER_COLOR),
      fills=(_BEFORE_FILL, _AFTER_FILL),
      y_label="Incremental KPI",
      start_date=start_date,
      end_date=end_date,
  )


def plot_roi_before_after(
    df_ts: pd.DataFrame,
    date_col: str,
    spend_col: str,
    opt_spend_col: str,
    kpi_col: str,
    opt_kpi_col: str,
    out_path: str,
    title_prefix: str,
    threshold_roi: Optional[float] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> None:
  """Writes ``<prefix>_timeseries_roi_before_after.png``.

  Two line charts stacked vertically on an IDENTICAL vertical scale: the ROI
  of every date before the optimization (red, top; incremental KPI / spend of
  the date, ``0`` when the spend is ``0``) and after it (blue, bottom;
  optimized incremental KPI / optimized spend).  The ROI threshold is drawn as
  a dotted reference line on both panels.  Each panel has the width and x-axis
  layout of ``*_timeseries_spend_vs_est.png`` and exactly half its height.

  Args:
    df_ts: Date-aggregated frame of one dimension, sorted by date.
    date_col: Date column.
    spend_col: Actual spend column.
    opt_spend_col: Optimized spend column.
    kpi_col: ``estimated_incremental_KPI_<treatment>``.
    opt_kpi_col: Incremental KPI under the optimized spend.
    out_path: Destination png path.
    title_prefix: ``Total`` / ``[<geo>]`` / ``[<item>]``.
    threshold_roi: The ROI floor (reference line, title).
    start_date: Optional inclusive start of the optimization period.
    end_date: Optional inclusive end of the optimization period.
  """
  needed = (spend_col, opt_spend_col, kpi_col, opt_kpi_col)
  if any(col not in df_ts.columns for col in needed):
    return
  roi_before = _ratio(_numeric(df_ts[kpi_col]), _numeric(df_ts[spend_col]))
  roi_after = _ratio(
      _numeric(df_ts[opt_kpi_col]), _numeric(df_ts[opt_spend_col])
  )
  reference = (
      (float(threshold_roi), f"ROI threshold = {threshold_roi}")
      if threshold_roi is not None
      else None
  )
  _plot_before_after_panels(
      df_ts[date_col],
      roi_before,
      roi_after,
      out_path,
      titles=(
          f"{title_prefix} ROI before optimization",
          f"{title_prefix} ROI after optimization"
          f"{_threshold_text(threshold_roi)}",
      ),
      labels=(
          f"ROI before optimization ({kpi_col} / {spend_col})",
          "ROI after optimization",
      ),
      colors=(_BEFORE_COLOR, _AFTER_COLOR),
      fills=(_BEFORE_FILL, _AFTER_FILL),
      y_label="ROI",
      start_date=start_date,
      end_date=end_date,
      reference=reference,
  )


def plot_before_after_charts(
    df_ts: pd.DataFrame,
    date_col: str,
    prefix: str,
    title_prefix: str,
    spend_col: str,
    opt_spend_col: str,
    kpi_col: str,
    opt_kpi_col: str,
    opt_dir: str,
    threshold_roi: Optional[float] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> None:
  """Writes the three before / after charts of one dimension.

  Shared by both optimizers: ``<prefix>_timeseries_spend_before_after.png``,
  ``<prefix>_timeseries_incremental_effect_before_after.png`` and
  ``<prefix>_timeseries_roi_before_after.png`` (see the three ``plot_*``
  functions above).  Charts whose columns are missing from ``df_ts`` are
  skipped.

  Args:
    df_ts: Date-aggregated frame of one dimension, sorted by date.
    date_col: Date column.
    prefix: Filename prefix (``total``, ``geo_<geo>`` or ``item_<item>``).
    title_prefix: ``Total`` / ``[<geo>]`` / ``[<item>]``.
    spend_col: Actual spend column.
    opt_spend_col: Optimized spend column.
    kpi_col: ``estimated_incremental_KPI_<treatment>``.
    opt_kpi_col: Incremental KPI under the optimized spend.
    opt_dir: Destination directory (``.../007_optimization``).
    threshold_roi: The ROI floor.
    start_date: Optional inclusive start of the optimization period.
    end_date: Optional inclusive end of the optimization period.
  """
  common = dict(
      date_col=date_col,
      title_prefix=title_prefix,
      threshold_roi=threshold_roi,
      start_date=start_date,
      end_date=end_date,
  )
  plot_spend_before_after(
      df_ts,
      spend_col=spend_col,
      opt_spend_col=opt_spend_col,
      out_path=os.path.join(
          opt_dir, f"{prefix}_timeseries_spend_before_after.png"
      ),
      spend_label=f"Spend ({spend_col})",
      **common,
  )
  plot_incremental_effect_before_after(
      df_ts,
      kpi_col=kpi_col,
      opt_kpi_col=opt_kpi_col,
      out_path=os.path.join(
          opt_dir, f"{prefix}_timeseries_incremental_effect_before_after.png"
      ),
      **common,
  )
  plot_roi_before_after(
      df_ts,
      spend_col=spend_col,
      opt_spend_col=opt_spend_col,
      kpi_col=kpi_col,
      opt_kpi_col=opt_kpi_col,
      out_path=os.path.join(
          opt_dir, f"{prefix}_timeseries_roi_before_after.png"
      ),
      **common,
  )


# =============================================================================
# Saturation-aware budget optimization (Phase 4), spend treatments only.
# =============================================================================


_EPS = 1e-12


def compute_basis_theta(
    basis_df: pd.DataFrame, cate_coef_dict: Dict[str, float]
) -> Optional[np.ndarray]:
  """Rebuilds ``theta(x) = p(x)' beta`` from the pooled CATE coefficients.

  Args:
    basis_df: The CATE basis ``p(X)`` (columns must match the coefficient names
      used by the existing pipeline).
    cate_coef_dict: The pooled ``{name}_coef`` dictionary produced by Phase 3.

  Returns:
    ``theta(x)`` up to a positive scale factor, or ``None`` when the
    coefficients cannot be matched to the basis columns.
  """
  if basis_df is None or not cate_coef_dict:
    return None
  coefs = {
      str(k)[:-5]: float(v)
      for k, v in cate_coef_dict.items()
      if str(k).endswith("_coef")
  }
  matched = [c for c in basis_df.columns if str(c) in coefs]
  if not matched:
    return None
  values = np.zeros(len(basis_df), dtype=float)
  for col in matched:
    values += coefs[str(col)] * np.asarray(basis_df[col], dtype=float)
  return values


def estimate_theta_kpi_units(
    kpi_values: np.ndarray,
    f_values: np.ndarray,
    basis_theta: Optional[np.ndarray] = None,
    group_keys: Optional[pd.Series] = None,
) -> np.ndarray:
  """Recovers ``theta_hat_t`` in KPI units from the Phase 3 outputs.

  Under the saturation model the estimated incremental KPI of a row is
  ``KPI_t = f(T_t; alpha_hat) * theta_hat_t`` (the pipeline multiplies the CATE
  by the treatment, which is the *transformed* treatment here).  ``theta`` is
  therefore ``KPI / f`` wherever ``f > 0``; rows with ``T_t = 0`` (where
  ``f = 0`` for all three families) are recovered from the pooled CATE
  coefficients, which are proportional to ``theta`` by construction.

  Args:
    kpi_values: ``estimated_incremental_KPI_<treatment>``.
    f_values: ``f(T_t; alpha_hat)``.
    basis_theta: Optional ``p(x)' beta`` (any positive scale).
    group_keys: Optional grouping used to fill ``f = 0`` rows when the basis
      reconstruction is unavailable.

  Returns:
    ``theta_hat_t`` expressed in KPI units.
  """
  kpi_values = np.asarray(kpi_values, dtype=float)
  f_values = np.asarray(f_values, dtype=float)
  usable = f_values > 1e-9
  ratio = np.zeros_like(kpi_values)
  with np.errstate(divide="ignore", invalid="ignore"):
    ratio[usable] = kpi_values[usable] / f_values[usable]
  ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

  if basis_theta is not None:
    basis_theta = np.nan_to_num(
        np.asarray(basis_theta, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    mask = usable & (np.abs(basis_theta) > 1e-12)
    if np.count_nonzero(mask) >= max(5, int(0.05 * len(kpi_values))):
      scale = float(np.median(ratio[mask] / basis_theta[mask]))
      candidate = scale * basis_theta
      denom = np.abs(ratio[mask]) + 1e-9
      rel_err = float(np.median(np.abs(candidate[mask] - ratio[mask]) / denom))
      if np.isfinite(scale) and rel_err < 0.05:
        return candidate

  if np.all(usable):
    return ratio

  filled = ratio.copy()
  if group_keys is not None and len(group_keys) == len(ratio):
    frame = pd.DataFrame(
        {"g": np.asarray(group_keys), "v": ratio, "ok": usable}
    )
    medians = frame[frame["ok"]].groupby("g", observed=False)["v"].median()
    for idx in np.flatnonzero(~usable):
      filled[idx] = float(medians.get(frame["g"].iloc[idx], np.nan))
  global_median = float(np.median(ratio[usable])) if np.any(usable) else 0.0
  filled = np.where(np.isfinite(filled), filled, global_median)
  filled[~usable] = np.where(
      np.isfinite(filled[~usable]), filled[~usable], global_median
  )
  return np.nan_to_num(filled, nan=global_median)


def compute_budget(
    treatment_values: np.ndarray, budget_increment: float
) -> float:
  """Computes the budget ``K = (1 + budget_increment) * actual total``.

  Args:
    treatment_values: Actual treatment of the target units.
    budget_increment: A real number ``>= -1``.

  Returns:
    The budget upper bound ``K``.

  Raises:
    ValueError: If ``budget_increment < -1``.
  """
  budget_increment = float(budget_increment)
  if budget_increment < -1.0:
    raise ValueError(
        f"budget_increment must be >= -1 (got {budget_increment})."
    )
  total = float(np.nansum(np.asarray(treatment_values, dtype=float)))
  return max(0.0, (1.0 + budget_increment) * total)


# The enumerated budget modes of
# saturation_config['optimization']['budget_mode'].
BUDGET_MODES = ("fixed", "roi_target")

# Default of saturation_config['optimization']['cap_quantile']: the support
# cap T_bar_t is the q-quantile of the POSITIVE actual treatment within the
# unit's geo x item cell, floored at the unit's own actual treatment,
# T_bar_t = max(Q_q(cell_t), T_t^actual).  q = 1 is the cell maximum (then the
# floor never binds), i.e. the historical behaviour.
DEFAULT_CAP_QUANTILE = 1.0

# The enumerated values of saturation_config['optimization']
# ['reallocate_within']: "all" = one KKT problem over every allocation unit
# (spend moves across days, geos and items); "date" = one KKT problem per
# date with the per-date budget (no cross-day movement).
REALLOCATE_WITHIN = ("all", "date")

# Tolerance used to decide whether an allocation sits on its ROI cap.
_CAP_TOL = 1e-9


@dataclasses.dataclass
class SaturationAllocation:
  """Row-level outcome of :func:`allocate_saturation_budget`.

  Attributes:
    allocation: ``T_t*`` per unit.
    budget_mode: ``"fixed"`` or ``"roi_target"``.
    budget_k: ``K = (1 + b) sum_t T_t^actual`` (all target units); NaN under
      ``roi_target``.
    budget_k_prime: ``K'`` actually handed to the solver (``K`` or ``K_+``); NaN
      under ``roi_target``.
    k_out: ``K_out = sum_t T_t*``.
    unspent_budget: ``K - sum_t T_t*``; NaN under ``roi_target``.
    marginal_roi_at_optimum: ``mu*`` of the KKT solve; NaN under ``roi_target``
      (and 0 when every unit sits at its cap).
    t_bar: The support caps ``T_bar_t``.
    t_roi_cap: The ROI caps ``T_t^r`` (``inf`` where the threshold does not
      bind).
    t_bar_eff: ``min(T_bar_t, T_t^r)``.
    negative_roi: ``Z_0 = {t : theta_t <= 0}``.
    capped_by_roi: Units with ``T_t* = T_t^r``.
    capped_by_support: Units with ``T_t* = T_bar_t`` (the support cap binds;
      ``theta_t > 0``).
    n_units: Number of allocation units.
    n_units_excluded_negative_roi: Number of units in ``Z_0`` (``theta_t <=
      0``), i.e. the count of ``negative_roi``.
    n_units_capped_by_roi: Number of units with ``T_t* = T_t^r``, i.e. the count
      of ``capped_by_roi``.
    n_units_at_support_cap: Number of units with ``T_t* = T_bar_t``, i.e. the
      count of ``capped_by_support``.
  """

  allocation: np.ndarray
  budget_mode: str
  budget_k: float
  budget_k_prime: float
  k_out: float
  unspent_budget: float
  marginal_roi_at_optimum: float
  t_bar: np.ndarray
  t_roi_cap: np.ndarray
  t_bar_eff: np.ndarray
  negative_roi: np.ndarray
  capped_by_roi: np.ndarray
  capped_by_support: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=bool)
  )

  @property
  def n_units(self) -> int:
    return int(np.size(self.allocation))

  @property
  def n_units_excluded_negative_roi(self) -> int:
    return int(np.count_nonzero(self.negative_roi))

  @property
  def n_units_capped_by_roi(self) -> int:
    return int(np.count_nonzero(self.capped_by_roi))

  @property
  def n_units_at_support_cap(self) -> int:
    return int(np.count_nonzero(self.capped_by_support))


def allocate_saturation_budget(
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    t_actual: np.ndarray,
    t_bar: Union[float, np.ndarray],
    threshold_roi: float = 1.0,
    t_ref: Optional[Union[float, np.ndarray]] = None,
    budget_mode: str = "fixed",
    budget_increment: float = 0.0,
    budget_reallocate: bool = True,
) -> SaturationAllocation:
  """The saturation-aware KKT allocation (array level).

  Notation: ``t`` = allocation unit, ``theta_t`` = CATE of
  the unit (units of ``fbar``), ``T_bar_t`` = support cap (by default the
  maximum actual treatment within the unit's geo x item cell; with
  ``optimization.cap_quantile = q < 1`` the ``q``-quantile of the positive
  actual treatment of the cell floored at the unit's own actual treatment, see
  :func:`optimize_saturation`), ``r`` = ROI threshold, ``T_t^r`` = ROI cap,
  ``T_bar_t^eff = min(T_bar_t, T_t^r)``, ``Z_0 = {t : theta_t <= 0}``,
  ``K = (1 + b) sum_t T_t^actual`` (all units),
  ``K_+ = (1 + b) sum_{t not in Z_0} T_t^actual``.

  * ``budget_mode = "fixed"``, ``budget_reallocate = True`` (default):
    ``K' = K``; one KKT solve with caps ``T_bar_t^eff``.
  * ``budget_mode = "fixed"``, ``budget_reallocate = False``:
    ``K' = K_+``; one KKT solve with caps ``T_bar_t`` giving ``T^(1)``, then
    ``T_t* = min(T_t^(1), T_t^r)`` -- the cut is NOT redistributed.
  * ``budget_mode = "roi_target"``: ``T_t* = T_bar_t^eff`` for ``theta_t >
    0`` and ``0`` otherwise; no budget, no bisection.

  In every mode ``sum_t T_t* <= K`` (fixed) and ``T_t* <= T_bar_t`` hold.

  Args:
    family: One of ``saturation.SUPPORTED_FAMILIES``.
    alpha: ``alpha_hat`` (scalar or one value per unit, raw-scale).
    theta: ``theta_hat_t`` per unit.
    t_actual: Actual treatment per unit.
    t_bar: Support cap ``T_bar_t`` (scalar or per unit).
    threshold_roi: ``r = opt_threshold_roi >= 0`` (default: 1.0).
    t_ref: ``T_ref`` (scalar or per unit), or ``None`` for the raw ``f``.
    budget_mode: ``"fixed"`` | ``"roi_target"``.
    budget_increment: ``b >= -1`` (fixed only).
    budget_reallocate: See above (fixed only; default: ``True``).

  Returns:
    The :class:`SaturationAllocation`.

  Raises:
    ValueError: On an unknown ``budget_mode``, a negative threshold, or
      ``roi_target`` with ``threshold_roi = 0``.
  """
  mode = str(budget_mode or "fixed").strip().lower()
  if mode not in BUDGET_MODES:
    raise ValueError(
        f"budget_mode must be one of {list(BUDGET_MODES)} (got"
        f" '{budget_mode}')."
    )
  r = float(threshold_roi)
  if r < 0.0:
    raise ValueError(f"opt_threshold_roi must be >= 0 (got {threshold_roi}).")
  if mode == "roi_target" and r == 0.0:
    raise ValueError(
        "budget_mode='roi_target' requires opt_threshold_roi > 0 (with r = 0"
        " the ROI cap never binds and every unit with theta_t > 0 would simply"
        " sit at its support cap T_bar_t)."
    )
  theta = np.nan_to_num(np.asarray(theta, dtype=float), nan=0.0)
  t_actual = np.nan_to_num(np.asarray(t_actual, dtype=float), nan=0.0)
  t_bar_arr = np.broadcast_to(
      np.asarray(t_bar, dtype=float), theta.shape
  ).astype(float, copy=True)
  t_bar_arr = np.where(np.isfinite(t_bar_arr), np.maximum(t_bar_arr, 0.0), 0.0)
  negative = theta <= 0.0

  # The ROI cap T_t^r and the effective cap.
  t_roi = saturation.roi_cap(
      family, alpha, theta, r, t_ref=t_ref, t_upper=t_bar_arr
  )
  t_roi = np.where(negative, 0.0, t_roi)
  t_bar_eff = np.minimum(t_bar_arr, t_roi)

  if mode == "roi_target":
    allocation = np.where(negative, 0.0, t_bar_eff)
    allocation = np.where(np.isfinite(allocation), allocation, 0.0)
    budget_k = budget_k_prime = unspent = mu_star = float("nan")
  else:
    budget_k = compute_budget(t_actual, budget_increment)
    if budget_reallocate:
      budget_k_prime = budget_k
      solution = saturation.solve_budget_allocation(
          family, alpha, theta, budget_k_prime, t_bar_eff, t_ref=t_ref
      )
      allocation = solution.allocation
    else:
      budget_k_prime = compute_budget(t_actual[~negative], budget_increment)
      solution = saturation.solve_budget_allocation(
          family, alpha, theta, budget_k_prime, t_bar_arr, t_ref=t_ref
      )
      # Cut only, no redistribution.
      allocation = np.minimum(solution.allocation, t_roi)
    mu_star = float(solution.mu)
    allocation = np.where(negative, 0.0, allocation)
    # Never exceed the caps or the budget (numerical safety only).
    allocation = np.minimum(allocation, t_bar_arr)
    total = float(allocation.sum())
    if total > budget_k and total > 0.0:
      allocation = allocation * (budget_k / total)
    unspent = budget_k - float(allocation.sum())
  k_out = float(allocation.sum())
  capped = (
      (~negative)
      & np.isfinite(t_roi)
      & (allocation >= t_roi - _CAP_TOL * np.maximum(t_roi, 1.0))
  )
  at_support = (
      (~negative)
      & (t_bar_arr > 0.0)
      & (allocation >= t_bar_arr - _CAP_TOL * np.maximum(t_bar_arr, 1.0))
  )
  return SaturationAllocation(
      allocation=allocation,
      budget_mode=mode,
      budget_k=float(budget_k),
      budget_k_prime=float(budget_k_prime),
      k_out=k_out,
      unspent_budget=float(unspent),
      marginal_roi_at_optimum=float(mu_star),
      t_bar=t_bar_arr,
      t_roi_cap=t_roi,
      t_bar_eff=t_bar_eff,
      negative_roi=negative,
      capped_by_roi=capped,
      capped_by_support=at_support,
  )


def _cell_cap(
    values: np.ndarray,
    keys: Optional[pd.DataFrame],
    quantile: float = DEFAULT_CAP_QUANTILE,
) -> np.ndarray:
  """The support cap ``T_bar_t = max(Q_q(cell_t), T_t^actual)``.

  ``Q_q(cell_t)`` is the ``q``-quantile of the POSITIVE actual treatment
  within the geo x item cell of the unit (over the whole frame; zero-spend
  rows do not shift it, consistent with ``T_ref = q95(T | T > 0)``), and the
  floor at the unit's own actual treatment means that a unit above the
  quantile (an outlier day) keeps its own actual spend as its cap: the
  optimizer may still cut it (KKT marginal condition, ROI cap) but never
  pushes any unit above ``T_bar_t``.  With ``q = 1`` the quantile is the cell
  maximum and the floor never binds (the historical behaviour).  A cell
  without positive treatment has cap ``T_t^actual`` (= 0).

  Args:
    values: Actual treatment per row.
    keys: The geo / item key columns (``None`` or no columns: one cell).
    quantile: ``q`` in ``(0, 1]``.

  Returns:
    ``T_bar_t`` per row.

  Raises:
    ValueError: If ``quantile`` is outside ``(0, 1]``.
  """
  q = float(quantile)
  if not (0.0 < q <= 1.0):
    raise ValueError(f"cap_quantile must be in (0, 1] (got {quantile}).")
  values = np.asarray(values, dtype=float)
  values = np.where(np.isfinite(values), values, 0.0)

  def _positive_quantile(v: np.ndarray) -> float:
    pos = v[v > 0.0]
    return float(np.quantile(pos, q)) if pos.size else 0.0

  if keys is None or keys.shape[1] == 0:
    cell_q = np.full(values.shape, _positive_quantile(values))
  else:
    frame = keys.copy()
    frame["__v__"] = values
    cols = [c for c in frame.columns if c != "__v__"]
    cell_q = (
        frame.groupby(cols, observed=False, dropna=False)["__v__"]
        .transform(lambda s: _positive_quantile(s.to_numpy(dtype=float)))
        .to_numpy(dtype=float)
    )
  return np.maximum(cell_q, values)


def optimize_saturation(
    df_1a_consolidated: pd.DataFrame,
    treatment: str,
    treatment_amt: str,
    periods: Tuple[Any, Any],
    family: str,
    alpha: Union[float, np.ndarray],
    theta: np.ndarray,
    gain: Optional[np.ndarray] = None,
    budget_increment: float = 0.0,
    threshold_roi: float = 1.0,
    other_cost_variables: Optional[Sequence[str]] = None,
    target_geo: str = "all",
    target_item: str = "all",
    outcome_col: Optional[str] = None,
    t_ref: Optional[Union[float, np.ndarray]] = None,
    budget_mode: str = "fixed",
    budget_reallocate: bool = True,
    opt_geos: Optional[Sequence[Any]] = None,
    opt_items: Optional[Sequence[Any]] = None,
    cap_quantile: float = DEFAULT_CAP_QUANTILE,
    reallocate_within: str = "all",
    **kwargs,
) -> pd.DataFrame:
  """Runs the saturation-aware optimization and writes the Phase 4 outputs.

  Only defined for ``treatment_type="spend"``, where the treatment column *is*
  the money being budgeted, i.e. ``g(T) = T``.  For ``"percentage"``,
  ``"impressions"`` and ``"price"`` a cost model ``spend = g(treatment)`` would
  be needed to denominate the budget in currency; until that is specified the
  orchestrator routes those types to the simple optimizer instead.

  The allocation units are the rows (geo x item x date) inside
  ``opt_periods`` that belong to ``opt_geos x opt_items`` (``None`` = all);
  the support cap of a unit is ``T_bar_t = max(Q_q(cell_t), T_t^actual)``
  with ``Q_q`` the ``cap_quantile``-quantile of the positive actual treatment
  within its geo x item cell over the whole frame (``q = 1``: the cell
  maximum; :func:`_cell_cap`); the ROI threshold is the per-unit cap
  ``T_t^r``; the problem is solved in one KKT pass by
  :func:`allocate_saturation_budget` (``reallocate_within = "all"``) or in
  one KKT pass per date with the per-date budget ``K_d = (1 + b) sum_{t in d}
  T_t^actual`` (``reallocate_within = "date"``: spend moves across geos and
  items within a day but never across days).  Rows outside the target set
  keep their pre-optimization values in the ``optimized_*`` columns.

  Carryover (adstock) is NOT modelled: the response of a row depends on that
  row's spend only, which is what lets the KKT problem separate across units.
  Under ``"all"`` the optimizer therefore moves spend freely across days as
  well as across geos and items; interpret the result at the geo x item x
  period level (totals over ``opt_periods``) rather than day by day, or use
  ``"date"`` to keep each day's total fixed.

  With ``t_ref`` (the reference point in RAW
  treatment units, ``c_i * T_ref``) the objective is
  ``sum_t theta_t fbar(T_t; alpha_i, T_ref_raw_i)`` -- exactly what Phases 1-3
  estimated ``theta`` against.  Without ``t_ref`` the pre-normalization
  ``gain * f`` objective is used (legacy call sites).

  Args:
    df_1a_consolidated: The consolidated frame (a working copy is returned; the
      input is never modified in place).
    treatment: Treatment column ``T``.
    treatment_amt: Monetary spend column associated with the treatment. Equal to
      ``treatment`` for spend treatments.
    periods: ``(start, end)`` of ``opt_periods``.
    family: The saturation family.
    alpha: The fixed ``alpha_hat`` from Phase 0, expressed in RAW treatment
      units. A scalar, or one value per row when ``population_median_normalize``
      induces a row-varying ``alpha_i`` (treatment normalization bridge).
    theta: ``theta_hat_t = p(X_t)' beta_hat`` in KPI units (one value per row),
      as estimated by Phases 1-3 against ``fbar``.
    gain: Treatment normalization bridge for the un-normalized ``f``. The
      per-row factor satisfying ``f(T/c; alpha) = gain * f(T; alpha_i)``; ``1``
      for ``exp`` / ``log`` and ``c ** (-alpha)`` for ``power``. ``None`` means
      no normalization.  Ignored when ``t_ref`` is given.
    budget_increment: ``b >= -1`` (``budget_mode = fixed`` only).
    threshold_roi: ``r = opt_threshold_roi >= 0``.
    other_cost_variables: Existing ``opt_other_cost_variables``.
    target_geo: Target geography (kept for output parity).
    target_item: Target item (kept for output parity).
    outcome_col: The outcome column, used for ``optimized_<outcome>``.
    t_ref: ``T_ref`` in RAW treatment units (scalar, or one
      value per row -- ``saturation.raw_scale_reference``).  ``None`` keeps the
      un-normalized objective.
    budget_mode: ``"fixed"`` (default) | ``"roi_target"``.
    budget_reallocate: See :func:`allocate_saturation_budget` (fixed only).
    opt_geos: Geos whose rows are allocation units (``None`` = all).
    opt_items: Items whose rows are allocation units (``None`` = all).
    cap_quantile: ``q`` of the support cap ``T_bar_t = max(Q_q(cell_t),
      T_t^actual)`` (``optimization.cap_quantile``; default 1.0 = cell
      maximum).
    reallocate_within: ``"all"`` (default: one KKT problem over all units) |
      ``"date"`` (one KKT problem per date with the per-date budget; no
      cross-day movement).  ``optimization.reallocate_within``.
    **kwargs: ``date_col``, ``geo_col``, ``item_col``, ``sales_col``,
      ``out_dir``.

  Returns:
    A copy of ``df_1a_consolidated`` carrying the ``optimized_*`` columns.

  Raises:
    ValueError: On an unknown ``budget_mode`` / ``reallocate_within`` or a
      ``cap_quantile`` outside ``(0, 1]``.
  """
  other_cost_variables = list(other_cost_variables or [])
  date_col = kwargs.get("date_col", "Date")
  geo_col = kwargs.get("geo_col", None)
  item_col = kwargs.get("item_col", None)
  out_dir = kwargs.get("out_dir", "")
  mode = str(budget_mode or "fixed").strip().lower()
  if mode not in BUDGET_MODES:
    raise ValueError(
        f"budget_mode must be one of {list(BUDGET_MODES)} (got"
        f" '{budget_mode}')."
    )
  scope = str(reallocate_within or "all").strip().lower()
  if scope not in REALLOCATE_WITHIN:
    raise ValueError(
        f"reallocate_within must be one of {list(REALLOCATE_WITHIN)} (got"
        f" '{reallocate_within}')."
    )

  kpi_col = f"estimated_incremental_KPI_{treatment}"
  opt_dir = os.path.join(out_dir, "007_optimization")
  os.makedirs(opt_dir, exist_ok=True)

  df_work = df_1a_consolidated.copy()
  if date_col and date_col in df_work.columns:
    dates = pd.to_datetime(df_work[date_col], errors="coerce")
  else:
    dates = pd.Series(pd.NaT, index=df_work.index)

  if periods and periods[0] is not None:
    start_date = pd.to_datetime(periods[0])
  else:
    start_date = dates.min()
  if periods and periods[1] is not None:
    end_date = pd.to_datetime(periods[1])
  else:
    end_date = dates.max()
  if dates.notna().any():
    period_mask = (dates >= start_date) & (dates <= end_date)
  else:
    period_mask = pd.Series(True, index=df_work.index)
  in_period = period_mask.fillna(False).to_numpy()

  # Allocation units = opt_periods AND opt_geos x opt_items.
  in_scope = in_period.copy()
  if opt_geos is not None and geo_col and geo_col in df_work.columns:
    in_scope &= df_work[geo_col].isin(list(opt_geos)).to_numpy()
  if opt_items is not None and item_col and item_col in df_work.columns:
    in_scope &= df_work[item_col].isin(list(opt_items)).to_numpy()

  t_actual = (
      pd.to_numeric(df_work[treatment], errors="coerce")
      .fillna(0.0)
      .to_numpy(dtype=float)
  )
  kpi_actual = (
      pd.to_numeric(df_work.get(kpi_col, 0.0), errors="coerce")
      .fillna(0.0)
      .to_numpy(dtype=float)
  )
  theta = np.nan_to_num(np.asarray(theta, dtype=float), nan=0.0)

  # Treatment normalization bridge.  Phases 1-3 estimated theta against
  # `fbar(T/c; alpha, T_ref)`; Phase 4 budgets in RAW units.  The identity
  # `fbar(T/c; alpha, T_ref) = fbar(T; alpha_i, c T_ref)` holds exactly, so with
  # `t_ref` the objective is `theta` against the per-row `alpha_i` and
  # `T_ref_raw_i` -- no gain.  Without `t_ref` (un-normalized f) the two are
  # reconciled by `f(T/c; alpha) = gain_i * f(T; alpha_i)`, i.e. by optimizing
  # `theta_eff = theta * gain`.
  alpha_vec = np.broadcast_to(
      np.asarray(alpha, dtype=float), t_actual.shape
  ).astype(float, copy=False)
  alpha_series = pd.Series(alpha_vec, index=df_work.index)
  if t_ref is not None:
    t_ref_vec: Optional[np.ndarray] = np.broadcast_to(
        np.asarray(t_ref, dtype=float), t_actual.shape
    ).astype(float, copy=False)
    gain_vec = np.ones_like(t_actual)
  else:
    t_ref_vec = None
    if gain is None:
      gain_vec = np.ones_like(t_actual)
    else:
      gain_vec = np.broadcast_to(
          np.asarray(gain, dtype=float), t_actual.shape
      ).astype(float, copy=False)
  theta_eff = theta * gain_vec

  def _response(t_values: np.ndarray) -> np.ndarray:
    """The response on the SAME scale Phases 1-3 used."""
    if t_ref_vec is not None:
      return saturation.normalized_saturation(
          t_values, family, alpha_vec, t_ref_vec
      )
    return gain_vec * saturation.apply_saturation(t_values, family, alpha_vec)

  # T_bar_t = max(Q_q(cell_t), T_t^actual): the cap_quantile-quantile of the
  # positive actual treatment within the geo x item cell of the unit (over
  # the whole frame), floored at the unit's own actual treatment.
  key_cols = [c for c in (geo_col, item_col) if c and c in df_work.columns]
  t_bar_all = _cell_cap(
      t_actual, df_work[key_cols] if key_cols else None, cap_quantile
  )

  # One KKT pass on the allocation units ("all"), or one per date ("date";
  # the per-date budget K_d = (1 + b) sum_{t in d} T_t^actual keeps each
  # day's total fixed).  Rows without a date form one group of their own.
  t_opt = t_actual.copy()
  unit_idx = np.flatnonzero(in_scope)
  if scope == "date" and unit_idx.size:
    date_codes, _ = pd.factorize(dates.iloc[unit_idx].astype(str))
    groups = [unit_idx[date_codes == c] for c in np.unique(date_codes)]
  else:
    groups = [unit_idx]
  group_results: List[Tuple[np.ndarray, SaturationAllocation]] = []
  for group_idx in groups:
    if not (group_idx.size and np.any(t_bar_all[group_idx] > 0)):
      continue
    group_result = allocate_saturation_budget(
        family=family,
        alpha=alpha_vec[group_idx],
        theta=theta_eff[group_idx],
        t_actual=t_actual[group_idx],
        t_bar=t_bar_all[group_idx],
        threshold_roi=threshold_roi,
        t_ref=None if t_ref_vec is None else t_ref_vec[group_idx],
        budget_mode=mode,
        budget_increment=budget_increment,
        budget_reallocate=budget_reallocate,
    )
    t_opt[group_idx] = group_result.allocation
    group_results.append((group_idx, group_result))

  # `f` is reported on the SAME scale Phases 1-3 used, i.e.
  # fbar(T/c; alpha, T_ref) or, without t_ref, f(T/c; alpha).
  f_opt = _response(t_opt)
  kpi_opt = f_opt * theta
  # Rows outside the target set keep their pre-optimization values.
  kpi_opt = np.where(in_scope, kpi_opt, kpi_actual)

  # ---- Outputs: consolidated CSV columns -----------------------------
  col_t = f"optimized_{treatment}"
  col_tf = f"optimized_{treatment}_transformed"
  col_spend = f"optimized_{treatment}_spend"
  col_kpi = f"optimized_incremental_KPI_{treatment}"
  col_roi = f"optimized_roi_{treatment}"

  df_work[col_t] = t_opt
  df_work[col_tf] = np.where(in_scope, f_opt, _response(t_actual))
  df_work[col_kpi] = kpi_opt

  # `treatment_type="spend"`: the treatment IS the spend, so g(T) = T.
  spend_opt = t_opt.copy()
  if treatment_amt in df_work.columns:
    actual_spend_full = (
        pd.to_numeric(df_work[treatment_amt], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
  else:
    actual_spend_full = t_actual
  spend_opt = np.where(in_scope, spend_opt, actual_spend_full)
  df_work[col_spend] = spend_opt

  with np.errstate(divide="ignore", invalid="ignore"):
    df_work[col_roi] = np.divide(
        kpi_opt,
        spend_opt,
        out=np.zeros_like(kpi_opt, dtype=float),
        where=spend_opt != 0,
    )

  if outcome_col and outcome_col in df_work.columns:
    y_actual = pd.to_numeric(df_work[outcome_col], errors="coerce").fillna(0.0)
    df_work[f"optimized_{outcome_col}"] = y_actual.to_numpy(dtype=float) + (
        kpi_opt - kpi_actual
    )

  # ---- Outputs: optimization_metrics.csv -----------------------------
  # Per-unit statistics kept index-aligned so that every
  # geo / item / 'all' row of the metrics aggregates exactly its own units.
  unit_frame = pd.DataFrame(
      {
          "is_unit": in_scope,
          "negative": False,
          "capped": False,
          "at_support_cap": False,
          "t_bar": t_bar_all,
      },
      index=df_work.index,
  )
  for group_idx, group_result in group_results:
    rows = df_work.index[group_idx]
    unit_frame.loc[rows, "negative"] = group_result.negative_roi
    unit_frame.loc[rows, "capped"] = group_result.capped_by_roi
    unit_frame.loc[rows, "at_support_cap"] = group_result.capped_by_support
  # mu*: one value under "all"; one per date under "date" (reported as the
  # median with its min / max; NaN under roi_target).
  mu_values = np.asarray(
      [r.marginal_roi_at_optimum for _, r in group_results], dtype=float
  )
  mu_values = mu_values[np.isfinite(mu_values)]
  if mu_values.size:
    mu_star = float(np.median(mu_values))
    mu_min, mu_max = float(mu_values.min()), float(mu_values.max())
  else:
    mu_star = mu_min = mu_max = float("nan")
  b_inc = float(budget_increment) if mode == "fixed" else None
  b_realloc = bool(budget_reallocate) if mode == "fixed" else None

  geos = [target_geo] if target_geo != "all" else ["all"]
  if target_geo == "all" and geo_col and geo_col in df_work.columns:
    geos.extend(sorted(df_work.loc[in_scope, geo_col].dropna().unique()))
  items = [target_item] if target_item != "all" else ["all"]
  if target_item == "all" and item_col and item_col in df_work.columns:
    items.extend(sorted(df_work.loc[in_scope, item_col].dropna().unique()))

  df_scope = df_work.loc[in_scope]
  results = []
  for geo_value in geos:
    for item_value in items:
      df_sub = df_scope
      if geo_value != "all":
        df_sub = df_sub[df_sub[geo_col] == geo_value]
      if item_value != "all":
        df_sub = df_sub[df_sub[item_col] == item_value]
      if df_sub.empty:
        continue

      if treatment_amt in df_sub.columns:
        total_cost = float(df_sub[treatment_amt].sum())
      else:
        total_cost = float(df_sub[treatment].sum())
      optimized_total = float(df_sub[col_spend].sum())
      inc_kpi = (
          float(df_sub[kpi_col].sum()) if kpi_col in df_sub.columns else 0.0
      )
      opt_inc_kpi = float(df_sub[col_kpi].sum())

      total_cost_cv = 0.0
      for cost_var in other_cost_variables:
        if cost_var in df_sub.columns:
          total_cost_cv += float(df_sub[cost_var].sum())
      reduction = total_cost - optimized_total
      increase_pct = reduction / total_cost_cv if total_cost_cv != 0 else 0.0

      # alpha_i is row-varying under population_median_normalize,
      # so summarise over exactly the rows this metrics row aggregates.
      sub_alpha = alpha_series.loc[df_sub.index]
      sub_pos = df_work.index.get_indexer(df_sub.index)
      sub_units = unit_frame.loc[df_sub.index]
      sub_t_actual = float(np.sum(t_actual[sub_pos]))
      k_sub = (
          (1.0 + float(budget_increment)) * sub_t_actual
          if mode == "fixed"
          else None
      )
      k_out_sub = float(np.sum(t_opt[sub_pos]))

      results.append({
          "treatment": treatment,
          "other_cost_variables": other_cost_variables,
          "target_geo": geo_value,
          "target_item": item_value,
          "periods_start_date": (
              periods[0] if periods else str(start_date.date())
          ),
          "periods_end_date": periods[1] if periods else str(end_date.date()),
          "threshold_roi": threshold_roi,
          "saturation_family": family,
          "alpha_hat": float(sub_alpha.median()),
          # Equal to alpha_hat unless the normalization made alpha_i vary.
          "alpha_hat_min": float(sub_alpha.min()),
          "alpha_hat_max": float(sub_alpha.max()),
          # The reference point of fbar in RAW units
          # (median over the aggregated rows; None for the un-normalized f).
          "T_ref_raw": (
              None
              if t_ref_vec is None
              else float(np.median(t_ref_vec[sub_pos]))
          ),
          "budget_increment": b_inc,
          # Adjacent to total_cost_of_treatment_in_periods.
          "total_cost_of_treatment_in_periods": total_cost,
          f"incremental_KPI_{treatment}": inc_kpi,
          f"roi_{treatment}": inc_kpi / total_cost if total_cost else 0.0,
          # Adjacent to optimized_total_cost_of_treatment_in_periods.
          "optimized_total_cost_of_treatment_in_periods": optimized_total,
          f"optimized_incremental_KPI_{treatment}": opt_inc_kpi,
          f"optimized_roi_{treatment}": (
              opt_inc_kpi / optimized_total if optimized_total else 0.0
          ),
          "reduction_of_cost_of_treatment": reduction,
          "total_cost_of_other_cost_variables": total_cost_cv,
          "increase_pct_of_total_cost_of_other_cost_variables": increase_pct,
          # Optimization statistics.  K / K_out /
          # unspent_budget and the unit counts are aggregated over the units
          # of this row (so a geo row may show a negative unspent_budget when
          # budget was reallocated INTO the geo); budget_mode, the scope and
          # mu* are properties of the whole problem (mu* per date under
          # reallocate_within = "date": median with min / max).
          "budget_mode": mode,
          "budget_reallocate": b_realloc,
          "reallocate_within": scope,
          "cap_quantile": float(cap_quantile),
          # Modelling premise of the allocation (documented in the README):
          # no carryover / adstock, so spend moves across the allocation
          # units (rows) as allowed by reallocate_within.
          "assumptions": "no_carryover",
          "K": k_sub,
          "K_out": k_out_sub,
          "unspent_budget": None if k_sub is None else float(k_sub - k_out_sub),
          "marginal_roi_at_optimum": (
              None if not np.isfinite(mu_star) else float(mu_star)
          ),
          "marginal_roi_at_optimum_min": (
              None if not np.isfinite(mu_min) else float(mu_min)
          ),
          "marginal_roi_at_optimum_max": (
              None if not np.isfinite(mu_max) else float(mu_max)
          ),
          "n_units": int(sub_units["is_unit"].sum()),
          "n_units_excluded_negative_roi": int(sub_units["negative"].sum()),
          "n_units_capped_by_roi": int(sub_units["capped"].sum()),
          "n_units_at_support_cap": int(sub_units["at_support_cap"].sum()),
      })

  if results:
    pd.DataFrame(results).to_csv(
        os.path.join(opt_dir, "optimization_metrics.csv"), index=False
    )

  # ---- Outputs: charts -----------------------------------------------
  opt_outcome_col = (
      f"optimized_{outcome_col}"
      if outcome_col and f"optimized_{outcome_col}" in df_work.columns
      else None
  )
  _plot_saturation_optimization(
      df_work,
      treatment=treatment,
      treatment_amt=treatment_amt,
      kpi_col=kpi_col,
      opt_treatment_col=col_t,
      opt_spend_col=col_spend,
      opt_kpi_col=col_kpi,
      date_col=date_col,
      geo_col=geo_col,
      item_col=item_col,
      sales_col=kwargs.get("sales_col", None),
      outcome_col=outcome_col,
      opt_outcome_col=opt_outcome_col,
      threshold_roi=threshold_roi,
      opt_dir=opt_dir,
      start_date=start_date,
      end_date=end_date,
  )
  return df_work


def _timeseries_line_chart(
    dates: Any,
    series: Sequence[Tuple[Any, str, str]],
    out_path: str,
    title: str,
    start_date: Any,
    end_date: Any,
) -> None:
  """Draws a date-indexed line chart with one line per entry of ``series``.

  One panel of the shared 007_optimization geometry
  (:func:`_timeseries_figure`).

  Args:
    dates: The x values (the date column of the aggregated frame).
    series: ``(values, color, label)`` triples, drawn in order.
    out_path: Destination png path.
    title: Chart title.
    start_date: Inclusive start of the optimization period (dashed marker).
    end_date: Inclusive end of the optimization period (dashed marker).
  """
  fig, (ax,) = _timeseries_figure(1)
  for values, color, label in series:
    ax.plot(dates, values, color=color, linewidth=3, label=label)
  _mark_period(ax, start_date, end_date)
  _style_timeseries_panel(ax, title, "Amount")
  _style_date_axis(ax)
  fig.savefig(out_path, dpi=200)
  plt.close(fig)


def _plot_outcome_chart(
    df_ts: pd.DataFrame,
    date_col: str,
    prefix: str,
    title_prefix: str,
    outcome_col: Optional[str],
    opt_outcome_col: Optional[str],
    opt_dir: str,
    threshold_roi: float,
    start_date: Any,
    end_date: Any,
) -> None:
  """Writes ``<prefix>_<outcome>.png`` -- the business bottom line.

  One two-line chart: the observed outcome (red, ``Original KPI``) against the
  projected outcome under the optimized spend (blue, ``Estimated total KPI
  with optimal treatment``), i.e. ``Y + (KPI_opt - KPI)`` summed per date.
  Shared by both optimizers so the HTML report shows the chart in the same
  position whatever ``simple_optimization`` is.  Nothing is written when
  either column is missing from ``df_ts``.

  Args:
    df_ts: Date-aggregated frame for this dimension, sorted by date.
    date_col: Date column name.
    prefix: Filename prefix (``total``, ``geo_<geo>`` or ``item_<item>``).
    title_prefix: Human-readable prefix for the chart title.
    outcome_col: The outcome column ``Y``, if available.
    opt_outcome_col: The projected outcome under the optimized spend, if
      available.
    opt_dir: Destination directory.
    threshold_roi: The ROI floor, annotated in the title.
    start_date: Inclusive start of the optimization period.
    end_date: Inclusive end of the optimization period.
  """
  if not (
      outcome_col
      and opt_outcome_col
      and outcome_col in df_ts.columns
      and opt_outcome_col in df_ts.columns
  ):
    return
  _timeseries_line_chart(
      df_ts[date_col],
      [
          (df_ts[outcome_col], "red", f"Original KPI ({outcome_col})"),
          (
              df_ts[opt_outcome_col],
              "blue",
              "Estimated total KPI with optimal treatment",
          ),
      ],
      os.path.join(opt_dir, f"{prefix}_{outcome_col}.png"),
      f"{title_prefix} {outcome_col} (Threshold={threshold_roi})",
      start_date,
      end_date,
  )


def _plot_spend_charts(
    df_ts: pd.DataFrame,
    date_col: str,
    prefix: str,
    title_prefix: str,
    treatment: str,
    opt_treatment_col: str,
    outcome_col: Optional[str],
    opt_outcome_col: Optional[str],
    opt_dir: str,
    threshold_roi: float,
    start_date: Any,
    end_date: Any,
) -> None:
  """Writes the ``treatment_types="spend"`` two-line charts for one dimension.

  Each chart carries exactly two lines so the comparison it is meant to
  support is unambiguous:

  * ``<prefix>_original_vs_optimal_treatment.png`` - purple actual treatment
    against green optimized treatment (the decision).
  * ``<prefix>_<outcome>.png`` - red observed outcome against blue projected
    outcome (the business bottom line; :func:`_plot_outcome_chart`, which the
    simple optimizer draws as well).

  The causal gain (incremental KPI before / after the optimization) is the
  ``<prefix>_timeseries_incremental_effect_before_after.png`` chart written by
  :func:`plot_before_after_charts` for both optimizers.

  Args:
    df_ts: Date-aggregated frame for this dimension, sorted by date.
    date_col: Date column name.
    prefix: Filename prefix (``total``, ``geo_<geo>`` or ``item_<item>``).
    title_prefix: Human-readable prefix for chart titles.
    treatment: Actual treatment column ``T``.
    opt_treatment_col: ``optimized_<treatment>``.
    outcome_col: The outcome column, if available.
    opt_outcome_col: ``optimized_<outcome>``, if available.
    opt_dir: Destination directory.
    threshold_roi: The ROI floor, annotated in the titles.
    start_date: Inclusive start of the optimization period.
    end_date: Inclusive end of the optimization period.
  """
  dates = df_ts[date_col]

  if treatment in df_ts.columns and opt_treatment_col in df_ts.columns:
    _timeseries_line_chart(
        dates,
        [
            (df_ts[treatment], "purple", f"Original treatment ({treatment})"),
            (df_ts[opt_treatment_col], "green", "Optimal treatment"),
        ],
        os.path.join(opt_dir, f"{prefix}_original_vs_optimal_treatment.png"),
        f"{title_prefix} Original vs Optimal Treatment"
        f" (Threshold={threshold_roi})",
        start_date,
        end_date,
    )

  _plot_outcome_chart(
      df_ts,
      date_col=date_col,
      prefix=prefix,
      title_prefix=title_prefix,
      outcome_col=outcome_col,
      opt_outcome_col=opt_outcome_col,
      opt_dir=opt_dir,
      threshold_roi=threshold_roi,
      start_date=start_date,
      end_date=end_date,
  )


def _plot_saturation_optimization(
    df_work: pd.DataFrame,
    treatment: str,
    treatment_amt: str,
    kpi_col: str,
    opt_treatment_col: str,
    opt_spend_col: str,
    opt_kpi_col: str,
    date_col: str,
    geo_col: Optional[str],
    item_col: Optional[str],
    sales_col: Optional[str],
    outcome_col: Optional[str],
    opt_outcome_col: Optional[str],
    threshold_roi: float,
    opt_dir: str,
    start_date: Any,
    end_date: Any,
) -> None:
  """Draws the optimization charts for every dimension.

  Emits, for ``total``, each geo and each item, the two two-line charts of
  :func:`_plot_spend_charts` and the three before / after charts shared with
  the simple optimizer (:func:`plot_before_after_charts`).  Only
  ``treatment_type="spend"`` reaches this code path; other treatment types
  are routed to the simple optimizer by the orchestrator.

  Args:
    df_work: The working frame already carrying the ``optimized_*`` columns.
    treatment: Treatment column ``T``.
    treatment_amt: Actual monetary spend column.
    kpi_col: ``estimated_incremental_KPI_<treatment>``.
    opt_treatment_col: ``optimized_<treatment>``.
    opt_spend_col: ``optimized_<treatment>_spend``.
    opt_kpi_col: ``optimized_incremental_KPI_<treatment>``.
    date_col: Date column used for the time axis.
    geo_col: Optional geography column.
    item_col: Optional item column.
    sales_col: Optional sales column, aggregated for parity with the previous
      output.
    outcome_col: The outcome column ``Y``.
    opt_outcome_col: ``optimized_<outcome>``.
    threshold_roi: The ROI floor, annotated on the charts.
    opt_dir: Destination directory (``.../007_optimization``).
    start_date: Inclusive start of the optimization period.
    end_date: Inclusive end of the optimization period.
  """
  if not date_col or date_col not in df_work.columns:
    return

  spend_col = treatment_amt if treatment_amt in df_work.columns else treatment

  chart_frame = df_work.copy()
  chart_frame[date_col] = pd.to_datetime(chart_frame[date_col], errors="coerce")

  # Yearly before / after summary (the matrix under the charts in the HTML
  # report).  `g(T) = T` for spend treatments, so the actual spend column is
  # the treatment itself unless a separate spend column was given.
  write_yearly_summary(
      chart_frame,
      treatment=treatment,
      date_col=date_col,
      geo_col=geo_col,
      item_col=item_col,
      spend_col=spend_col,
      opt_spend_col=opt_spend_col,
      kpi_col=kpi_col,
      opt_kpi_col=opt_kpi_col,
      out_dir=opt_dir,
  )

  levels = _dimension_levels(chart_frame, geo_col, item_col)

  candidate_cols = [
      sales_col,
      spend_col,
      kpi_col,
      opt_spend_col,
      opt_kpi_col,
      treatment,
      opt_treatment_col,
      outcome_col,
      opt_outcome_col,
  ]
  agg_dict = {
      col: "sum" for col in candidate_cols if col and col in chart_frame.columns
  }
  if not agg_dict:
    return

  for level_type, level_value in levels:
    if level_type == "total":
      subset = chart_frame
      prefix, title_prefix = "total", "Total"
    elif level_type == "geo":
      subset = chart_frame[chart_frame[geo_col] == level_value]
      prefix, title_prefix = f"geo_{level_value}", f"[{level_value}]"
    else:
      subset = chart_frame[chart_frame[item_col] == level_value]
      prefix, title_prefix = f"item_{level_value}", f"[{level_value}]"
    if subset.empty:
      continue

    df_ts = (
        subset.groupby(date_col, as_index=False, observed=False)
        .agg(agg_dict)
        .sort_values(date_col)
    )

    _plot_spend_charts(
        df_ts,
        date_col=date_col,
        prefix=prefix,
        title_prefix=title_prefix,
        treatment=treatment,
        opt_treatment_col=opt_treatment_col,
        outcome_col=outcome_col,
        opt_outcome_col=opt_outcome_col,
        opt_dir=opt_dir,
        threshold_roi=threshold_roi,
        start_date=start_date,
        end_date=end_date,
    )
    plot_before_after_charts(
        df_ts,
        date_col=date_col,
        prefix=prefix,
        title_prefix=title_prefix,
        spend_col=spend_col,
        opt_spend_col=opt_spend_col,
        kpi_col=kpi_col,
        opt_kpi_col=opt_kpi_col,
        opt_dir=opt_dir,
        threshold_roi=threshold_roi,
        start_date=start_date,
        end_date=end_date,
    )
