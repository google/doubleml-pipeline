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

"""Benchmark prices, price gaps and price investments of ``price`` treatments.

The pipeline reports the incremental KPI of a row as ``theta(X) * D``, i.e.
against the counterfactual ``D = 0``.  With a raw price as the treatment that
counterfactual is a price of zero, far outside the data, and the "incremental"
KPI becomes a linear extrapolation of the size of the sales themselves.  A
``price`` treatment is therefore estimated as the gap between a benchmark
price and the observed price, so that ``D = 0`` means "sold at the benchmark
price".

Vocabulary (shared with the orchestrator and ``docs/advanced.md``):

* group: the rows with the same ``item_col`` and ``geo_col`` values;
* ``p_{g,t}``: the price of group ``g`` at date ``t`` (the treatment column as
  given by the caller);
* ``b_{g,t}``: the benchmark price of that row;
* ``D_{g,t} = b_{g,t} - p_{g,t}``: the gap, the treatment the pipeline
  estimates;
* ``U_{g,t} = sales_{g,t} / p_{g,t}``: the quantity sold;
* ``C_{g,t} = U_{g,t} * max(D_{g,t}, 0)``: the price investment, the cost the
  ROI of the treatment is computed against.

The benchmark is configured inside ``treatment_types`` (no extra
orchestrator argument)::

    treatment_types = {
        "tv_spend": "spend",
        "price": {"type": "price", "benchmark": "q_95"},
    }

``benchmark`` accepts (see :func:`parse_benchmark`):

* omitted -- automatic resolution (:func:`resolve_auto`): a column whose
  smallest non-NaN value is ``<= 0`` is taken to be a gap already
  (``"as_is"``), otherwise the group maximum ``"q_100"`` is used;
* ``"as_is"`` -- the column already is ``D``; nothing is transformed;
* ``"q_k"`` with an integer ``1 <= k <= 100`` -- the ``k``-th percentile of
  the prices of the group, taken as the smallest observed price at or above
  the percentile (``q_100`` is the maximum);
* ``"rolling_max"`` with ``"window": "<n>W"`` or ``"<n>D"`` -- the maximum
  price of the group over the calendar window ``(t - W, t]`` ending at the
  row's own date;
* a number -- a constant benchmark price;
* any other string -- the name of a column holding the benchmark price.
"""

import dataclasses
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# The treatment type whose benchmark this module resolves.
PRICE_TYPE = "price"

# ``benchmark`` keywords.
AS_IS = "as_is"
ROLLING_MAX = "rolling_max"

# ``BenchmarkSpec.method`` values.
METHOD_AUTO = "auto"
METHOD_AS_IS = "as_is"
METHOD_QUANTILE = "quantile"
METHOD_ROLLING_MAX = "rolling_max"
METHOD_COLUMN = "column"
METHOD_CONSTANT = "constant"

# Keys accepted in the dict form of a ``treatment_types`` entry.
TREATMENT_TYPE_KEYS = frozenset({"type", "benchmark", "window"})

# Suffix of the output columns.
GAP_SUFFIX = "_gap"
BENCHMARK_SUFFIX = "_benchmark"
INVESTMENT_SUFFIX = "_investment"

_QUANTILE_PATTERN = re.compile(r"^q_(\d+)$")
_WINDOW_PATTERN = re.compile(r"^(\d+)([WD])$")
_LAG_PATTERN = re.compile(r"^(?P<base>.+)_l(?P<lag>\d+)$")


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
  """How the benchmark price ``b_{g,t}`` of a ``price`` treatment is set.

  Attributes:
    method: One of ``METHOD_AUTO``, ``METHOD_AS_IS``, ``METHOD_QUANTILE``,
      ``METHOD_ROLLING_MAX``, ``METHOD_COLUMN`` or ``METHOD_CONSTANT``.
    percentile: The ``k`` of ``q_k`` (``METHOD_QUANTILE`` only).
    window: The calendar window, ``"<n>W"`` or ``"<n>D"``
      (``METHOD_ROLLING_MAX`` only).
    column: The benchmark price column (``METHOD_COLUMN`` only).
    value: The constant benchmark price (``METHOD_CONSTANT`` only).
  """

  method: str
  percentile: Optional[int] = None
  window: Optional[str] = None
  column: Optional[str] = None
  value: Optional[float] = None

  def describe(self) -> str:
    """Returns the specification in the notation of the documentation."""
    if self.method == METHOD_QUANTILE:
      return f"q_{self.percentile}"
    if self.method == METHOD_ROLLING_MAX:
      return f"rolling_max(window={self.window})"
    if self.method == METHOD_COLUMN:
      return f"column '{self.column}'"
    if self.method == METHOD_CONSTANT:
      return f"constant {self.value:g}"
    return self.method

  def is_time_varying(self) -> bool:
    """True when ``b_{g,t}`` can differ between the rows of one group."""
    return self.method in (METHOD_ROLLING_MAX, METHOD_COLUMN)

  def to_metadata(self) -> Dict[str, Any]:
    """Returns the JSON-serializable form written to ``metadata.json``."""
    return {
        "method": self.method,
        "percentile": self.percentile,
        "window": self.window,
        "column": self.column,
        "value": self.value,
        "description": self.describe(),
    }


def parse_benchmark(
    value: Any, window: Any = None, where: str = "treatment_types"
) -> BenchmarkSpec:
  """Parses the ``benchmark`` (and ``window``) entry of a ``price`` treatment.

  Args:
    value: The ``benchmark`` value: ``None`` (automatic), ``"as_is"``,
      ``"q_k"``, ``"rolling_max"``, a number or a column name.
    window: The ``window`` value, required with ``"rolling_max"`` and not
      allowed otherwise.
    where: Prefix of the error messages.

  Returns:
    The parsed :class:`BenchmarkSpec`.

  Raises:
    ValueError: On an invalid combination or value.
    TypeError: When ``value`` is neither ``None``, a string nor a number.
  """
  if value is None:
    spec = BenchmarkSpec(METHOD_AUTO)
  elif isinstance(value, bool):
    raise TypeError(f"{where}: benchmark must not be a bool (got {value!r}).")
  elif isinstance(value, (int, float, np.integer, np.floating)):
    const = float(value)
    if not np.isfinite(const) or const <= 0.0:
      raise ValueError(
          f"{where}: a constant benchmark price must be finite and > 0 (got"
          f" {value!r})."
      )
    spec = BenchmarkSpec(METHOD_CONSTANT, value=const)
  elif isinstance(value, str):
    text = value.strip()
    match = _QUANTILE_PATTERN.match(text)
    if text == AS_IS:
      spec = BenchmarkSpec(METHOD_AS_IS)
    elif match:
      k = int(match.group(1))
      if not 1 <= k <= 100:
        raise ValueError(
            f"{where}: the percentile of benchmark 'q_k' must satisfy 1 <= k"
            f" <= 100 (got {text!r})."
        )
      spec = BenchmarkSpec(METHOD_QUANTILE, percentile=k)
    elif text == ROLLING_MAX:
      if window is None:
        raise ValueError(
            f"{where}: benchmark 'rolling_max' requires 'window' (e.g."
            " '26W' or '180D')."
        )
      spec = BenchmarkSpec(METHOD_ROLLING_MAX, window=_parse_window(window))
    elif not text:
      raise ValueError(f"{where}: benchmark must not be an empty string.")
    else:
      spec = BenchmarkSpec(METHOD_COLUMN, column=text)
  else:
    raise TypeError(
        f"{where}: benchmark must be None, a string or a number (got"
        f" {type(value).__name__})."
    )
  if window is not None and spec.method != METHOD_ROLLING_MAX:
    raise ValueError(
        f"{where}: 'window' is only allowed with benchmark 'rolling_max'"
        f" (got benchmark {value!r})."
    )
  return spec


def _parse_window(window: Any) -> str:
  """Validates ``"<n>W"`` / ``"<n>D"`` and returns it normalized."""
  if not isinstance(window, str):
    raise TypeError(
        "window must be a string of the form '<n>W' (weeks) or '<n>D' (days)"
        f" (got {type(window).__name__})."
    )
  text = window.strip().upper()
  match = _WINDOW_PATTERN.match(text)
  if not match or int(match.group(1)) < 1:
    raise ValueError(
        "window must be a positive integer followed by 'W' (weeks) or 'D'"
        f" (days), e.g. '26W' or '180D' (got {window!r})."
    )
  return text


def _window_length(window: str) -> pd.Timedelta:
  """Returns the calendar length ``W`` of a validated ``window`` string."""
  return pd.Timedelta(_parse_window(window))


def parse_treatment_type(
    d_col: str, value: Any
) -> Tuple[str, Optional[BenchmarkSpec]]:
  """Splits one ``treatment_types`` entry into its type and benchmark spec.

  Args:
    d_col: The treatment column the entry belongs to (error messages).
    value: The entry: the type as a string (``"spend"``, ``"price"``, ...) or
      a dict ``{"type": ..., "benchmark": ..., "window": ...}``.

  Returns:
    ``(type, spec)``: the type string, and the :class:`BenchmarkSpec` when the
    type is ``"price"`` (``METHOD_AUTO`` when no benchmark was given), else
    ``None``.

  Raises:
    ValueError: When the dict has unknown keys, lacks ``type``, or gives a
      ``benchmark`` / ``window`` for a type other than ``"price"``.
    TypeError: When the entry is neither a string nor a dict.
  """
  where = f"treatment_types['{d_col}']"
  if isinstance(value, str):
    ttype = value
    benchmark = None
    window = None
  elif isinstance(value, dict):
    unknown = sorted(set(value) - TREATMENT_TYPE_KEYS)
    if unknown:
      raise ValueError(
          f"{where}: unknown key(s) {unknown}; allowed keys are"
          f" {sorted(TREATMENT_TYPE_KEYS)}."
      )
    if "type" not in value:
      raise ValueError(f"{where}: the dict form requires the 'type' key.")
    ttype = value["type"]
    if not isinstance(ttype, str):
      raise TypeError(f"{where}: 'type' must be a string (got {ttype!r}).")
    benchmark = value.get("benchmark")
    window = value.get("window")
  else:
    raise TypeError(
        f"{where}: expected the type as a string or a dict with 'type' (got"
        f" {type(value).__name__})."
    )
  if ttype != PRICE_TYPE:
    if benchmark is not None or window is not None:
      raise ValueError(
          f"{where}: 'benchmark' / 'window' are only allowed for type"
          f" '{PRICE_TYPE}' (got type {ttype!r})."
      )
    return ttype, None
  return ttype, parse_benchmark(benchmark, window, where=where)


def split_lag(column: str) -> Tuple[str, int]:
  """Splits ``"<base>_l<n>"`` into ``(base, n)``; ``(column, 0)`` otherwise.

  ``preprocessing.features.create_lagged_features`` names the lag ``n`` of
  column ``c`` ``f"{c}_l{n}"``.

  Args:
    column: A column name.

  Returns:
    The base column and the lag order (``0`` for a column that is not a lag).
  """
  match = _LAG_PATTERN.match(str(column))
  if not match:
    return str(column), 0
  return match.group("base"), int(match.group("lag"))


def resolve_auto(spec: BenchmarkSpec, prices: pd.Series) -> BenchmarkSpec:
  """Resolves ``METHOD_AUTO`` from the values of the treatment column.

  A column whose smallest non-NaN value is ``<= 0`` cannot be a price, so it
  is taken to be a gap the caller computed already (``as_is``); otherwise the
  group maximum ``q_100`` is used.  Any other ``spec`` is returned unchanged.

  Args:
    spec: The parsed specification.
    prices: The treatment column as given.

  Returns:
    A specification whose ``method`` is not ``METHOD_AUTO``.

  Raises:
    ValueError: When the column has no numeric value.
  """
  if spec.method != METHOD_AUTO:
    return spec
  values = pd.to_numeric(prices, errors="coerce").dropna()
  if values.empty:
    raise ValueError(
        f"Treatment column '{prices.name}' has no numeric value; the price"
        " benchmark cannot be resolved."
    )
  if float(values.min()) <= 0.0:
    return BenchmarkSpec(METHOD_AS_IS)
  return BenchmarkSpec(METHOD_QUANTILE, percentile=100)


def reserved_column_collisions(df: pd.DataFrame) -> List[str]:
  """Returns the columns of ``df`` whose name is a benchmark keyword.

  Such a column could not be named as a benchmark column (the keyword wins),
  so the caller rejects the configuration instead of guessing.

  Args:
    df: The input DataFrame.

  Returns:
    The offending column names (``as_is``, ``rolling_max`` or ``q_<k>``).
  """
  found = []
  for col in df.columns:
    name = str(col)
    if name in (AS_IS, ROLLING_MAX) or _QUANTILE_PATTERN.match(name):
      found.append(name)
  return found


def compute_benchmark(
    df: pd.DataFrame,
    price_col: str,
    spec: BenchmarkSpec,
    group_cols: List[str],
    date_col: Optional[str] = None,
) -> pd.Series:
  """Computes ``b_{g,t}`` for every row of ``df``.

  Args:
    df: The input DataFrame.
    price_col: The price column ``p``.
    spec: A resolved specification (``METHOD_QUANTILE``,
      ``METHOD_ROLLING_MAX``, ``METHOD_COLUMN`` or ``METHOD_CONSTANT``).
    group_cols: The columns defining a group (``item_col`` and ``geo_col``,
      those that are set); empty for a single group.
    date_col: The date column (required by ``METHOD_ROLLING_MAX``).

  Returns:
    The benchmark prices, aligned with ``df.index`` (float; NaN where the
    benchmark is undefined, e.g. a NaN benchmark column value).

  Raises:
    ValueError: For an unresolved ``spec``, a missing benchmark column, or
      ``METHOD_ROLLING_MAX`` without ``date_col``.
  """
  prices = pd.to_numeric(df[price_col], errors="coerce").astype(float)
  if spec.method == METHOD_CONSTANT:
    return pd.Series(float(spec.value), index=df.index, dtype=float)
  if spec.method == METHOD_COLUMN:
    if spec.column not in df.columns:
      raise ValueError(
          f"Benchmark column '{spec.column}' of price treatment"
          f" '{price_col}' is not a column of the DataFrame."
      )
    return pd.to_numeric(df[spec.column], errors="coerce").astype(float)
  keys = [df[c] for c in group_cols]
  if spec.method == METHOD_QUANTILE:
    q = spec.percentile / 100.0
    if keys:
      return prices.groupby(keys, sort=False, dropna=False).transform(
          lambda s: s.quantile(q, interpolation="higher")
      )
    return pd.Series(
        prices.quantile(q, interpolation="higher"), index=df.index, dtype=float
    )
  if spec.method == METHOD_ROLLING_MAX:
    if not date_col:
      raise ValueError(
          "benchmark 'rolling_max' requires date_col (the calendar window is"
          " taken over the dates of the group)."
      )
    return _rolling_max(prices, df, group_cols, date_col, spec.window)
  raise ValueError(
      f"The benchmark of '{price_col}' is not resolved ({spec.describe()})."
  )


def _rolling_max(
    prices: pd.Series,
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    window: str,
) -> pd.Series:
  """``max(p_{g,s} : t - W < s <= t)`` per row, over the observed dates."""
  length = _window_length(window)
  dates = pd.to_datetime(df[date_col])
  out = pd.Series(np.nan, index=df.index, dtype=float)
  if group_cols:
    positions = df.groupby(
        [df[c] for c in group_cols], sort=False, dropna=False
    ).indices.values()
  else:
    positions = [np.arange(len(df))]
  for idx in positions:
    idx = np.asarray(idx)
    order = np.argsort(dates.values[idx], kind="stable")
    ordered = idx[order]
    series = pd.Series(
        prices.values[ordered], index=pd.DatetimeIndex(dates.values[ordered])
    )
    # closed="right": the window is (t - W, t], the row's own date included,
    # so b_{g,t} >= p_{g,t} and the gap is never negative.
    rolled = series.rolling(length, closed="right", min_periods=1).max()
    out.iloc[ordered] = rolled.values
  return out


def period_delta(
    df: pd.DataFrame, date_col: str, group_cols: List[str]
) -> pd.Timedelta:
  """The length of one period, as ``create_lagged_features`` infers it.

  The rows are sorted by ``group_cols`` then ``date_col`` and the median of
  the positive differences between consecutive dates is taken; 1 day when no
  positive difference exists.

  Args:
    df: The input DataFrame.
    date_col: The date column.
    group_cols: The columns identifying a unit (sorted before the date).

  Returns:
    The period length.
  """
  ordered = df.sort_values(by=list(group_cols) + [date_col], kind="stable")
  dates = pd.to_datetime(ordered[date_col])
  diffs = dates.diff()
  delta = diffs[diffs > pd.Timedelta(0)].median()
  if pd.isna(delta):
    return pd.Timedelta(days=1)
  return delta


def shift_within_groups(
    df: pd.DataFrame,
    column: str,
    n_periods: int,
    date_col: str,
    group_cols: List[str],
    delta: pd.Timedelta,
    fill_value: float = 0.0,
) -> pd.Series:
  """The value of ``column`` in the row of the same group ``n_periods`` earlier.

  Row ``(g, t)`` receives the value of row ``(g, t - n_periods * delta)``;
  ``fill_value`` when no such row exists (the same exact-date matching as
  ``create_lagged_features``).  A NaN value of an existing row is kept.

  Args:
    df: The input DataFrame.
    column: The column to shift.
    n_periods: The lag order ``n >= 1``.
    date_col: The date column.
    group_cols: The columns defining a group; empty for a single group.
    delta: The period length (see :func:`period_delta`).
    fill_value: The value of rows whose earlier row does not exist.

  Returns:
    The shifted values, aligned with ``df.index``.
  """
  dates = pd.to_datetime(df[date_col])
  keys = [df[c] for c in group_cols]
  if keys:
    index = pd.MultiIndex.from_arrays(keys + [dates])
    target = pd.MultiIndex.from_arrays(keys + [dates - n_periods * delta])
  else:
    index = pd.Index(dates)
    target = pd.Index(dates - n_periods * delta)
  lookup = pd.Series(df[column].values, index=index)
  lookup = lookup.loc[~lookup.index.duplicated(keep="last")]
  exists = target.isin(lookup.index)
  values = lookup.reindex(target).values.astype(float)
  values = np.where(exists, values, fill_value)
  return pd.Series(values, index=df.index, dtype=float)


def investment(
    gap: pd.Series,
    cost: Optional[pd.Series] = None,
    sales: Optional[pd.Series] = None,
    prices: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
  """The price investment ``C`` with ``C = 0`` where the gap is ``<= 0``.

  Either ``cost`` (a cost column given by the caller) or ``sales`` and
  ``prices`` (the automatic ``C = sales / p * max(D, 0)``) must be given.

  Args:
    gap: The gap ``D`` of every row.
    cost: The caller's cost column; its value is kept where ``D > 0`` and
      replaced by ``0`` elsewhere.
    sales: The sales amounts of the automatic computation.
    prices: The prices of the automatic computation (all ``> 0``).

  Returns:
    ``(C, counts)`` with ``counts["n_rows_cost_zeroed"]`` the rows where
    ``D <= 0`` and the given cost was not ``0`` (caller's column only) and
    ``counts["n_rows_sales_missing"]`` the rows whose sales were NaN
    (automatic computation only; their ``C`` is ``0``).

  Raises:
    ValueError: When neither input is given, or a price is ``<= 0`` or NaN
      where the gap is defined.
  """
  gap_values = pd.to_numeric(gap, errors="coerce").astype(float)
  positive = gap_values > 0.0
  counts = {"n_rows_cost_zeroed": 0, "n_rows_sales_missing": 0}
  if cost is not None:
    cost_values = pd.to_numeric(cost, errors="coerce").astype(float)
    zeroed = (~positive) & cost_values.notna() & (cost_values != 0.0)
    counts["n_rows_cost_zeroed"] = int(zeroed.sum())
    result = cost_values.where(positive, 0.0)
    return result.astype(float), counts
  if sales is None or prices is None:
    raise ValueError(
        "investment() needs either `cost` or `sales` and `prices`."
    )
  price_values = pd.to_numeric(prices, errors="coerce").astype(float)
  invalid = gap_values.notna() & ~(price_values > 0.0)
  if invalid.any():
    raise ValueError(
        "The price investment cannot be computed: the price column has"
        f" {int(invalid.sum())} row(s) with a value <= 0 or NaN where the gap"
        " is defined. Provide treatment_spend_cols for this treatment instead."
    )
  sales_values = pd.to_numeric(sales, errors="coerce").astype(float)
  missing = sales_values.isna() & gap_values.notna()
  counts["n_rows_sales_missing"] = int(missing.sum())
  quantity = (sales_values / price_values).fillna(0.0)
  result = quantity * gap_values.clip(lower=0.0).fillna(0.0)
  result = result.where(gap_values.notna(), np.nan)
  return result.astype(float), counts


def gap_counts(gap: pd.Series) -> Dict[str, int]:
  """Counts the rows with ``D == 0`` and ``D < 0`` (NaN excluded)."""
  values = pd.to_numeric(gap, errors="coerce")
  return {
      "n_rows_gap_zero": int((values == 0.0).sum()),
      "n_rows_gap_negative": int((values < 0.0).sum()),
  }


def output_names(base: str, lag: int = 0) -> Dict[str, str]:
  """The output column names of the price treatment ``base`` (lag ``lag``).

  The lag order stays a ``_l<n>`` suffix (``price_gap_l1``), the convention of
  ``create_lagged_features`` that the orchestrator relies on to group the lags
  of a treatment with their base.

  Args:
    base: The base price column.
    lag: The lag order (``0`` for the base column itself).

  Returns:
    ``{"gap", "benchmark", "investment"}`` -> column name.
  """
  suffix = f"_l{lag}" if lag else ""
  return {
      "gap": f"{base}{GAP_SUFFIX}{suffix}",
      "benchmark": f"{base}{BENCHMARK_SUFFIX}",
      "investment": f"{base}{INVESTMENT_SUFFIX}{suffix}",
  }
