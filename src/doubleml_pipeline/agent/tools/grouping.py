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

"""Resolves a grouping key into one group label per row of a run.

`get_effects_by` accepts three families of keys and this module turns each
into a `Grouping`: a segment dimension (`geo`, `item`), a calendar period
(`week` ... `day_of_week`) or any input column, which is binned by quantiles
when numeric and by its most frequent values otherwise.
"""

import dataclasses
from typing import Any, Dict, List, Optional

import pandas as pd

from . import base

SEGMENT = "segment"
PERIOD = "period"
BIN = "bin"

_PERIOD_FREQUENCIES = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}
MONTH_OF_YEAR = "month_of_year"
DAY_OF_WEEK = "day_of_week"
PERIOD_KEYS = (*_PERIOD_FREQUENCIES, MONTH_OF_YEAR, DAY_OF_WEEK)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# Label of the bin that pools the rare values of a categorical column; the
# parentheses keep it from colliding with a real category.
OTHER = "(other)"


@dataclasses.dataclass(frozen=True)
class Grouping:
  """One group label per row, with what the labels mean."""

  key: str
  kind: str
  # The data column the groups are built from (the date column for periods).
  column: str
  labels: pd.Series
  # Group labels in presentation order (chronological, Q1..Qn, or by count).
  order: List[Any]
  # Group code -> display name, for segments with a name column.
  names: Dict[str, Any] = dataclasses.field(default_factory=dict)
  # How a column was binned: "quantiles", "distinct_values" (few distinct
  # values, kept as they are) or "categories".
  binning: Optional[str] = None
  # Quantile bins: {bin: {low, high}}, or {segment: {bin: ...}} within
  # segments.
  cut_points: Optional[Dict[str, Any]] = None
  # Which dimension the bins were cut within, if any.
  within: Optional[str] = None

  def describe(self) -> Dict[str, Any]:
    """The grouping's metadata, as returned next to the groups."""
    description: Dict[str, Any] = {
        "key": self.key,
        "kind": self.kind,
        "column": self.column,
    }
    if self.binning:
      description["binning"] = self.binning
    if self.within:
      description["within"] = self.within
    if self.cut_points is not None:
      description["cut_points"] = self.cut_points
    return description


def available_keys(run: base.RunData) -> Dict[str, List[str]]:
  """The keys `resolve` accepts for `run`, by family."""
  periods: List[str] = []
  if run.date_col:
    periods = [k for k in PERIOD_KEYS if k != DAY_OF_WEEK]
    if run.data_frequency == base.DAILY:
      periods.append(DAY_OF_WEEK)
  return {
      "segments": list(run.dimensions),
      "periods": periods,
      "columns": run.input_columns(),
  }


def resolve(
    run: base.RunData,
    key: str,
    bins: int,
    within: Optional[str] = None,
) -> Grouping:
  """Builds the `Grouping` for `key`.

  Args:
    run: The pipeline run.
    key: A segment dimension, a period key or an input column name. Segment and
      period keys take precedence over a column of the same name.
    bins: Number of quantile bins for numeric columns, and the number of
      categories (including `other`) kept for non-numeric ones.
    within: Cut the quantile bins of a numeric column within each segment of
      this dimension, so that each segment is ranked against its own history.

  Raises:
    InvalidArgumentError: If `key` or `within` is unknown.
    UnavailableError: If the run lacks what the key needs (date column,
      daily data, configured dimension).
  """
  if within is not None:
    base.check_choice("within", within, list(run.dimensions))
  if key in base.DIMENSIONS:
    return _segment_grouping(run, key)
  if key in PERIOD_KEYS:
    return _period_grouping(run, key)
  if key in run.input_columns():
    return _column_grouping(run, key, bins, within)
  raise base.InvalidArgumentError(
      f"Unknown grouping key {key!r}. Use a segment dimension, a period or an"
      f" input column: {available_keys(run)}."
  )


def _segment_grouping(run: base.RunData, dimension: str) -> Grouping:
  column = run.dimensions.get(dimension)
  if not column:
    raise base.UnavailableError(
        f"No {dimension}_col was configured for this pipeline run."
    )
  labels = run.full_df()[column]
  return Grouping(
      key=dimension,
      kind=SEGMENT,
      column=column,
      labels=labels,
      order=sorted(labels.dropna().unique(), key=str),
      names=run.segment_names(dimension),
  )


def _period_grouping(run: base.RunData, key: str) -> Grouping:
  dates = run.dates()
  if key == DAY_OF_WEEK:
    if run.data_frequency != base.DAILY:
      raise base.UnavailableError(
          "Day-of-week analysis needs daily data; this run's data is"
          f" {run.data_frequency}."
      )
    labels = dates.dt.dayofweek.map(dict(enumerate(_WEEKDAYS)))
    order: List[Any] = list(_WEEKDAYS)
  elif key == MONTH_OF_YEAR:
    labels = dates.dt.month
    order = list(range(1, 13))
  else:
    labels = dates.dt.to_period(_PERIOD_FREQUENCIES[key])
    order = sorted(labels.dropna().unique())
  present = set(labels.dropna().unique())
  return Grouping(
      key=key,
      kind=PERIOD,
      column=run.date_col,
      labels=labels,
      order=[o for o in order if o in present],
  )


def _column_grouping(
    run: base.RunData, column: str, bins: int, within: Optional[str]
) -> Grouping:
  values = run.full_df()[column]
  if not pd.api.types.is_numeric_dtype(values):
    if within:
      raise base.InvalidArgumentError(
          f"within applies to numeric columns; {column!r} is not numeric."
      )
    return _categorical_grouping(column, values, bins)
  if values.nunique() <= bins and not within:
    return Grouping(
        key=column,
        kind=BIN,
        column=column,
        labels=values,
        order=sorted(values.dropna().unique()),
        binning="distinct_values",
    )
  if within:
    segments = run.full_df()[run.dimensions[within]]
    labels = pd.Series(index=values.index, dtype=object)
    cut_points: Dict[str, Any] = {}
    for segment, group in values.groupby(segments, observed=True):
      binned, edges = _quantile_bins(group, bins)
      labels.loc[group.index] = binned
      cut_points[str(segment)] = edges
  else:
    labels, cut_points = _quantile_bins(values, bins)
  order = sorted(labels.dropna().unique(), key=_bin_rank)
  return Grouping(
      key=column,
      kind=BIN,
      column=column,
      labels=labels,
      order=order,
      binning="quantiles",
      cut_points=cut_points,
      within=within,
  )


def _categorical_grouping(
    column: str, values: pd.Series, bins: int
) -> Grouping:
  """Keeps the `bins - 1` most frequent values and pools the rest."""
  counts = values.value_counts()
  if len(counts) > bins:
    frequent = counts.index[: bins - 1]
    # Missing values stay missing (and are dropped) rather than pooled.
    values = values.where(values.isna() | values.isin(frequent), OTHER)
    counts = values.value_counts()
  return Grouping(
      key=column,
      kind=BIN,
      column=column,
      labels=values,
      order=list(counts.index),
      binning="categories",
  )


def _quantile_bins(values: pd.Series, bins: int):
  """Labels values Q1..Qn by quantile and returns the bin edges.

  Ties collapse bins (`duplicates="drop"`), so a column with few distinct
  values yields fewer than `bins` groups; a constant column yields one.
  """
  if values.nunique() < 2:
    value = float(values.dropna().iloc[0]) if values.notna().any() else None
    labels = values.where(values.isna(), "Q1")
    return labels, {"Q1": {"low": value, "high": value}}
  codes, edges = pd.qcut(
      values, q=bins, labels=False, retbins=True, duplicates="drop"
  )
  labels = codes.map(lambda c: f"Q{int(c) + 1}" if pd.notna(c) else None)
  cut_points = {
      f"Q{i + 1}": {"low": float(edges[i]), "high": float(edges[i + 1])}
      for i in range(len(edges) - 1)
  }
  return labels, cut_points


def _bin_rank(label: Any) -> int:
  return int(str(label)[1:])
