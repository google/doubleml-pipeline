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

"""Tool: get_effects_by.

One tool for every "effect by ..." question: by store or product (segment
dimensions), by period (month, day of week, ...) or by any input column
(quantile bins), optionally crossed with a second key into a 2-D table,
restricted to a date window and paged.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import base
from . import cate_model
from . import grouping

_SORT_KEYS = ("incremental_kpi", "effect_per_unit", "roi_on_spend")
_MAX_BINS = 10
_MAX_PAGE_SIZE = 100


def _primary_metric(table: pd.DataFrame) -> Optional[str]:
  """The per-unit metric whose interval drives the reliability signals."""
  for metric in ("roi_on_spend", "effect_per_unit"):
    if metric in table.columns:
      return metric
  return None


def _ci_separation(
    table: pd.DataFrame, metric: Optional[str]
) -> Optional[Dict[str, Any]]:
  """Finds pairs of groups whose intervals do not overlap.

  A pair where the lower bound of one group exceeds the upper bound of
  another is a large, well-supported difference that deserves a look at the
  data behind both groups (`get_segment_profile`).
  """
  if metric is None:
    return None
  low = table[f"{metric}_low"].to_numpy(dtype=float)
  high = table[f"{metric}_high"].to_numpy(dtype=float)
  separated = (low[:, None] > high[None, :]) & np.isfinite(low[:, None])
  separated &= np.isfinite(high[None, :])
  pairs = int(separated.sum())
  result: Dict[str, Any] = {"metric": metric, "separated_pairs": pairs}
  if pairs:
    gaps = np.where(separated, low[:, None] - high[None, :], -np.inf)
    higher, lower = np.unravel_index(int(gaps.argmax()), gaps.shape)
    result["largest_gap"] = {
        "higher_group": table.index[higher],
        "lower_group": table.index[lower],
        "gap": float(gaps[higher, lower]),
    }
  return result


def _threshold_summary(table: pd.DataFrame) -> Optional[Dict[str, Any]]:
  if "roi_vs_threshold" not in table.columns:
    return None
  verdicts = table["roi_vs_threshold"]
  return {
      base.WORTH_INVESTING: list(table.index[verdicts == base.WORTH_INVESTING]),
      base.NOT_WORTH_INVESTING: list(
          table.index[verdicts == base.NOT_WORTH_INVESTING]
      ),
      base.UNDETERMINED: int((verdicts == base.UNDETERMINED).sum()),
  }


def _low_sample_groups(table: pd.DataFrame) -> Optional[list]:
  if "low_sample" not in table.columns:
    return None
  return list(table.index[table["low_sample"]])


def _column_role(
    run: base.RunData, treatment: Optional[str], column: str
) -> str:
  """How the model used the grouping column, which bounds what a split by it

  can mean: `cate_covariate` (effect was allowed to vary with it),
  `confounder` (adjusted for only) or `unused`.
  """
  treatments = [treatment] if treatment else run.treatments
  if all(column in run.cate_covariates.get(t, []) for t in treatments):
    return "cate_covariate"
  if all(column in run.confounders.get(t, []) for t in treatments):
    return "confounder"
  return "unused"


def _heterogeneity_evidence(
    run: base.RunData, treatment: Optional[str], column: str
) -> Optional[str]:
  """Evidence that the CATE varies with `column`, if it is a CATE covariate."""
  if treatment is None or column not in run.cate_covariates.get(treatment, []):
    return None
  pvalues = cate_model.term_pvalues(run, treatment)
  return base.evidence_label(cate_model.covariate_pvalue(column, pvalues))


def _ordered(
    table: pd.DataFrame, group: grouping.Grouping, sort_by: str
) -> pd.DataFrame:
  """Orders the groups: segments by `sort_by`, the rest by their natural

  order (chronological, Q1..Qn, or by count).
  """
  if group.kind != grouping.SEGMENT:
    return table.reindex([g for g in group.order if g in table.index])
  if sort_by not in table.columns:
    raise base.InvalidArgumentError(
        f"sort_by={sort_by!r} needs a single treatment with a spend column;"
        f" use one of {[k for k in _SORT_KEYS if k in table.columns]}."
    )
  return table.sort_values(sort_by, ascending=False, na_position="last")


def _page(
    table: pd.DataFrame, page: int, page_size: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """Returns one page of an ordered table and where it sits in the whole."""
  pages = max(1, -(-len(table) // page_size))
  base.check_range("page", page, 1, pages)
  shown = table.iloc[(page - 1) * page_size : page * page_size]
  window = {
      "page": page,
      "pages": pages,
      "page_size": page_size,
      "groups_total": len(table),
      "groups_shown": len(shown),
      "first": shown.index[0],
      "last": shown.index[-1],
  }
  return shown, window


def _date_window(
    run: base.RunData, start: Optional[str], end: Optional[str]
) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
  """Parses the start/end filter.

  Returns the period of the whole data and the filter applied, for the
  result, plus the row mask of the filter (None when there is no filter).
  """
  if not run.date_col:
    if start or end:
      raise base.InvalidArgumentError(
          "start/end need a date_col, which this run does not have."
      )
    return {}, None
  dates = run.dates()
  window: Dict[str, Any] = {
      "data_period": {"start": dates.min().date(), "end": dates.max().date()}
  }
  if start is None and end is None:
    return window, None
  mask = pd.Series(True, index=dates.index)
  window["filter"] = {}
  for name, value, keep in (
      ("start", start, lambda d: dates >= d),
      ("end", end, lambda d: dates <= d),
  ):
    if value is None:
      continue
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
      raise base.InvalidArgumentError(
          f"{name}={value!r} is not a date; use YYYY-MM-DD."
      )
    window["filter"][name] = parsed.date()
    mask &= keep(parsed)
  if not mask.any():
    raise base.InvalidArgumentError(
        f"No rows between start={start!r} and end={end!r}."
    )
  return window, mask


def _with_labels(table: pd.DataFrame, group: grouping.Grouping) -> pd.DataFrame:
  table = table.rename_axis("group").reset_index()
  if group.names:
    table.insert(
        1, "name", table["group"].map(lambda g: group.names.get(str(g)))
    )
  return table


def _one_dimensional(
    run: base.RunData,
    treatment: Optional[str],
    group: grouping.Grouping,
    mask: Optional[pd.Series],
    sort_by: str,
    page: int,
    page_size: int,
) -> base.ToolResult:
  labels = group.labels.rename("group")
  if mask is not None:
    labels = labels.where(mask)
  table = run.aggregate(treatment, by=labels)
  if table.empty:
    raise base.UnavailableError(f"No rows could be grouped by {group.key!r}.")
  metric = _primary_metric(table)
  table = _ordered(table, group, sort_by)
  shown, window = _page(table, page, page_size)
  result: Dict[str, Any] = {
      "grouping": group.describe(),
      "window": window,
      "effect_direction_counts": (
          table["effect_direction"].value_counts().to_dict()
      ),
      "column_role": _column_role(run, treatment, group.column),
      "effect_drivers": run.effect_drivers(treatment, group.column, group.key),
      "heterogeneity_evidence": _heterogeneity_evidence(
          run, treatment, group.column
      ),
      "threshold_roi": run.threshold_roi,
      "threshold_summary": _threshold_summary(table),
      "ci_separation": _ci_separation(table, metric),
      "low_sample_groups": _low_sample_groups(table),
      "groups": base.records(_with_labels(shown, group)),
  }
  if metric:
    result["spread"] = {
        "metric": metric,
        **table[metric].agg(["min", "median", "max"]).to_dict(),
    }
  if group.kind == grouping.PERIOD:
    result["highest_period"] = table["incremental_kpi"].idxmax()
    result["lowest_period"] = table["incremental_kpi"].idxmin()
    result["configured_peak_months"] = run.metadata.get("peak_month")
  return result


def _two_dimensional(
    run: base.RunData,
    treatment: Optional[str],
    rows: grouping.Grouping,
    cols: grouping.Grouping,
    mask: Optional[pd.Series],
    page: int,
    page_size: int,
) -> base.ToolResult:
  row_labels = rows.labels.rename("row_group")
  if mask is not None:
    row_labels = row_labels.where(mask)
  table = run.aggregate(
      treatment, by=[row_labels, cols.labels.rename("col_group")]
  )
  if table.empty:
    raise base.UnavailableError("No rows could be grouped by both keys.")
  present = table.index
  row_order = [r for r in rows.order if r in present.get_level_values(0)]
  col_order = [c for c in cols.order if c in present.get_level_values(1)]
  ordered = table.reindex(
      [(r, c) for r in row_order for c in col_order if (r, c) in present]
  )
  shown, window = _page(ordered, page, page_size)
  return {
      "rows": rows.describe(),
      "columns": cols.describe(),
      "window": window,
      "row_groups": row_order,
      "col_groups": col_order,
      "column_role": {
          rows.key: _column_role(run, treatment, rows.column),
          cols.key: _column_role(run, treatment, cols.column),
      },
      "effect_drivers": {
          rows.key: run.effect_drivers(treatment, rows.column, rows.key),
          cols.key: run.effect_drivers(treatment, cols.column, cols.key),
      },
      "heterogeneity_evidence": {
          rows.key: _heterogeneity_evidence(run, treatment, rows.column),
          cols.key: _heterogeneity_evidence(run, treatment, cols.column),
      },
      "threshold_roi": run.threshold_roi,
      "threshold_summary": _threshold_summary(table),
      "low_sample_groups": _low_sample_groups(table),
      "cells": base.records(shown.reset_index()),
  }


def _run_effects_by(
    run: base.RunData,
    by: str,
    treatment: Optional[str],
    cross_with: Optional[str],
    bins: int,
    within: Optional[str],
    start: Optional[str],
    end: Optional[str],
    sort_by: str,
    page: int,
    page_size: int,
) -> base.ToolResult:
  base.check_range("bins", bins, 2, _MAX_BINS)
  base.check_range("page_size", page_size, 1, _MAX_PAGE_SIZE)
  base.check_choice("sort_by", sort_by, _SORT_KEYS)
  treatment = run.resolve_treatment(treatment, allow_all=True)
  window, mask = _date_window(run, start, end)
  rows = grouping.resolve(run, by, bins, within)
  if cross_with is None:
    result = _one_dimensional(
        run, treatment, rows, mask, sort_by, page, page_size
    )
  else:
    if cross_with == by:
      raise base.InvalidArgumentError("cross_with must differ from by.")
    cols = grouping.resolve(run, cross_with, bins)
    result = _two_dimensional(run, treatment, rows, cols, mask, page, page_size)
  return {**window, **result}


@base.agent_tool
def get_effects_by(
    self,
    by: str,
    treatment: Optional[str] = None,
    cross_with: Optional[str] = None,
    bins: int = 4,
    within: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sort_by: str = "incremental_kpi",
    page: int = 1,
    page_size: int = 50,
) -> base.ToolResult:
  """Breaks the causal effect down by segment, period or any input column.

  Args:
    by: What to group by. A segment dimension ("geo" for stores or regions,
      "item" for products); a period ("week", "month", "quarter", "year",
      "month_of_year" to pool calendar months across years, "day_of_week" for
      daily data); or the name of an input column, binned into quantiles when
      numeric (for example media spend or discount depth) and into its most
      frequent values otherwise.
    treatment: Treatment to analyze. Required when the results span several
      runs. Within a single run it defaults to the only treatment, or to all of
      the run's treatments combined.
    cross_with: A second key of the same kinds, for a 2-D table of cells (for
      example media spend quartile x discount depth quartile).
    bins: Quantile bins per numeric column, 2-10.
    within: "geo" or "item": cut the quantile bins of a numeric `by` column
      within each segment, so each store or product is ranked against its own
      history rather than against the others.
    start: First date to include (YYYY-MM-DD). Needs a date column.
    end: Last date to include (YYYY-MM-DD).
    sort_by: For segments: "incremental_kpi", "effect_per_unit" or
      "roi_on_spend", descending. Periods and bins keep their natural order
      (chronological, Q1..Qn, most frequent first).
    page: Which page of groups (or cells) to return, from 1.
    page_size: Groups per page (1-100). Summaries always cover all groups.

  Returns:
    data_period (first and last date in the data) and filter (the start/end
    applied, if any); window (page, pages, groups_total, and the first and
    last group on this page, so the range being shown is always explicit);
    groups (or cells), each with the metrics of get_treatment_effects plus the
    reliability signals roi_vs_threshold (the ROI interval lies below the
    threshold: not_worth_investing; above it: worth_investing; otherwise
    undetermined), low_sample and plausibility_flags. Also threshold_summary
    (which groups are worth or not worth investing in), ci_separation (pairs
    of groups whose intervals do not overlap, a sign of a large difference to
    investigate with get_segment_profile), low_sample_groups, the spread of
    the per-unit metric, column_role (cate_covariate, confounder or unused: a
    split by an unused column, especially one the treatment itself can move
    such as search interest or traffic, is descriptive only), effect_drivers
    (whether the model estimates group-specific effects at all) and
    heterogeneity_evidence when `by` is a CATE covariate (strong/weak/none
    with its p-value). For numeric columns, grouping.cut_points gives the bin
    edges, per segment when `within` is set.
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {
      "run": name,
      "treatment": treatment or "all",
      **_run_effects_by(
          run,
          by,
          treatment,
          cross_with,
          bins,
          within,
          start,
          end,
          sort_by,
          page,
          page_size,
      ),
  }
