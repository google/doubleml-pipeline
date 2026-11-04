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

"""Unit tests of the ``price`` treatment benchmark, gap and investment.

* ``preprocessing.price_benchmark``: parsing of the ``treatment_types`` entry
  (string / dict form, the ``benchmark`` keywords and their errors), the
  automatic resolution, the group quantile and rolling-max benchmarks, the
  lag shift, the price investment and the output names.
* ``CausalWorkflowOrchestrator._apply_price_treatments``: the renames of
  ``d_cols`` / ``treatment_types`` / ``treatment_spend_cols``, the benchmark
  column added to ``x_cols``, the investment column (caller's cost column or
  automatic), the lag treatments, the scaler lists, the metadata records and
  the restore before a second run.  Nothing here runs a pipeline.
* ``models.saturation``: the odd extension ``f(T) = -f(-T)`` of the families,
  of the derivative and of the T-basis.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

from doubleml_pipeline import workflow
from doubleml_pipeline.models import saturation
from doubleml_pipeline.preprocessing import features
from doubleml_pipeline.preprocessing import price_benchmark as pb
from doubleml_pipeline.preprocessing import scalers
import numpy as np
import pandas as pd


def _weekly_panel() -> pd.DataFrame:
  """2 items x 2 geos x 6 weeks; item A costs 100, item B 200 at list price.

  Discounts: item A is sold at 80 in weeks 3-4 of geo X and at 90 in week 5 of
  geo Y; item B at 150 in week 2 of both geos.  ``sales = price * units``.

  Returns:
    A DataFrame with date, item, geo, price, units, sales, X1, X2 and Y
    columns (24 rows).
  """
  dates = pd.date_range("2024-03-01", periods=6, freq="7D")
  rows = []
  for item, list_price in (("A", 100.0), ("B", 200.0)):
    for geo in ("X", "Y"):
      for week, date in enumerate(dates):
        price = list_price
        if item == "A" and geo == "X" and week in (2, 3):
          price = 80.0
        if item == "A" and geo == "Y" and week == 4:
          price = 90.0
        if item == "B" and week == 1:
          price = 150.0
        units = 10.0 + week + (5.0 if price < list_price else 0.0)
        rows.append({
            "date": date,
            "item": item,
            "geo": geo,
            "price": price,
            "units": units,
            "sales": price * units,
            "X1": float(week),
            "X2": 1.0 if geo == "X" else 0.0,
            "Y": units,
        })
  return pd.DataFrame(rows)


class ParseTreatmentTypeTest(unittest.TestCase):
  """The string and dict forms of a ``treatment_types`` entry."""

  def test_string_type_without_benchmark(self):
    """A plain type string: no spec, except the automatic one for price."""
    self.assertEqual(pb.parse_treatment_type("d", "spend"), ("spend", None))
    ttype, spec = pb.parse_treatment_type("d", "price")
    self.assertEqual(ttype, "price")
    self.assertEqual(spec.method, pb.METHOD_AUTO)

  def test_dict_forms(self):
    """Every accepted ``benchmark`` value maps to its ``BenchmarkSpec``."""
    cases = {
        "q_95": pb.BenchmarkSpec(pb.METHOD_QUANTILE, percentile=95),
        "q_100": pb.BenchmarkSpec(pb.METHOD_QUANTILE, percentile=100),
        "as_is": pb.BenchmarkSpec(pb.METHOD_AS_IS),
        "list_price": pb.BenchmarkSpec(pb.METHOD_COLUMN, column="list_price"),
        1200: pb.BenchmarkSpec(pb.METHOD_CONSTANT, value=1200.0),
    }
    for value, expected in cases.items():
      ttype, spec = pb.parse_treatment_type(
          "d", {"type": "price", "benchmark": value}
      )
      self.assertEqual(ttype, "price")
      self.assertEqual(spec, expected, msg=str(value))
    _, spec = pb.parse_treatment_type(
        "d", {"type": "price", "benchmark": "rolling_max", "window": "26W"}
    )
    self.assertEqual(
        spec, pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="26W")
    )
    self.assertEqual(spec.describe(), "rolling_max(window=26W)")
    self.assertTrue(spec.is_time_varying())
    _, spec = pb.parse_treatment_type("d", {"type": "price"})
    self.assertEqual(spec.method, pb.METHOD_AUTO)
    self.assertEqual(
        pb.parse_treatment_type("d", {"type": "spend"})[0], "spend"
    )

  def test_describe_uses_the_documented_notation(self):
    """``describe()`` (metadata ``description``, logs, errors) per method."""
    self.assertEqual(
        pb.BenchmarkSpec(pb.METHOD_QUANTILE, percentile=95).describe(), "q_95"
    )
    self.assertEqual(
        pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="26W").describe(),
        "rolling_max(window=26W)",
    )
    self.assertEqual(
        pb.BenchmarkSpec(pb.METHOD_COLUMN, column="list_price").describe(),
        "column 'list_price'",
    )
    self.assertEqual(
        pb.BenchmarkSpec(pb.METHOD_CONSTANT, value=1200.0).describe(),
        "constant 1200",
    )
    self.assertEqual(pb.BenchmarkSpec(pb.METHOD_AS_IS).describe(), "as_is")
    self.assertEqual(pb.BenchmarkSpec(pb.METHOD_AUTO).describe(), "auto")

  def test_errors(self):
    """Invalid entries are rejected with a message naming the problem."""
    bad = [
        ({"type": "spend", "benchmark": "q_95"}, ValueError, "only allowed"),
        ({"benchmark": "q_95"}, ValueError, "requires the 'type' key"),
        ({"type": "price", "foo": 1}, ValueError, "unknown key"),
        ({"type": "price", "window": "26W"}, ValueError, "only allowed with"),
        ({"type": "price", "benchmark": "rolling_max"}, ValueError, "window"),
        (
            {"type": "price", "benchmark": "rolling_max", "window": "26M"},
            ValueError,
            "window must be",
        ),
        (
            {"type": "price", "benchmark": "rolling_max", "window": "0W"},
            ValueError,
            "window must be",
        ),
        ({"type": "price", "benchmark": "q_0"}, ValueError, "1 <= k <= 100"),
        ({"type": "price", "benchmark": "q_101"}, ValueError, "1 <= k"),
        ({"type": "price", "benchmark": -5}, ValueError, "finite and > 0"),
        ({"type": "price", "benchmark": True}, TypeError, "bool"),
        ({"type": "price", "benchmark": ""}, ValueError, "empty"),
        ({"type": 3}, TypeError, "'type' must be a string"),
        (3, TypeError, "expected the type"),
    ]
    for value, error, pattern in bad:
      with self.assertRaisesRegex(error, pattern, msg=str(value)):
        pb.parse_treatment_type("d", value)


class ResolveAutoTest(unittest.TestCase):
  """The automatic resolution of a ``price`` treatment without benchmark."""

  def test_as_is_when_a_value_is_not_positive(self):
    """A value <= 0 means the column is a gap already."""
    spec = pb.BenchmarkSpec(pb.METHOD_AUTO)
    self.assertEqual(
        pb.resolve_auto(spec, pd.Series([5.0, 0.0, 3.0])).method,
        pb.METHOD_AS_IS,
    )
    self.assertEqual(
        pb.resolve_auto(spec, pd.Series([5.0, -1.0, np.nan])).method,
        pb.METHOD_AS_IS,
    )

  def test_q100_when_all_values_are_positive(self):
    """All values > 0 means a price: the group maximum is the benchmark."""
    resolved = pb.resolve_auto(
        pb.BenchmarkSpec(pb.METHOD_AUTO), pd.Series([5.0, np.nan, 3.0])
    )
    self.assertEqual(resolved, pb.BenchmarkSpec(pb.METHOD_QUANTILE, 100))
    self.assertEqual(resolved.describe(), "q_100")

  def test_explicit_is_unchanged_and_empty_raises(self):
    """An explicit spec passes through; a column without values fails."""
    explicit = pb.BenchmarkSpec(pb.METHOD_QUANTILE, percentile=95)
    self.assertIs(pb.resolve_auto(explicit, pd.Series([0.0])), explicit)
    with self.assertRaisesRegex(ValueError, "no numeric value"):
      pb.resolve_auto(
          pb.BenchmarkSpec(pb.METHOD_AUTO), pd.Series([np.nan], name="p")
      )


class ComputeBenchmarkTest(unittest.TestCase):
  """``compute_benchmark`` for every resolved method."""

  def setUp(self):
    """Builds the weekly panel."""
    super().setUp()
    self.df = _weekly_panel()

  def test_group_maximum(self):
    """``q_100`` is the maximum price of the group; the gap is never < 0."""
    bench = pb.compute_benchmark(
        self.df, "price", pb.BenchmarkSpec(pb.METHOD_QUANTILE, 100),
        ["item", "geo"],
    )
    expected = self.df.groupby(["item", "geo"])["price"].transform("max")
    pd.testing.assert_series_equal(bench, expected.astype(float))
    gap = bench - self.df["price"]
    self.assertTrue((gap >= 0).all())
    # Item A / geo X: list 100, discounted to 80 twice -> gap 20 on those rows.
    a_x = self.df[(self.df["item"] == "A") & (self.df["geo"] == "X")].index
    self.assertEqual(sorted(gap.loc[a_x].tolist()), [0, 0, 0, 0, 20, 20])

  def test_quantile_is_an_observed_price(self):
    """``q_k`` is the smallest observed price at or above the percentile."""
    df = pd.DataFrame({"g": ["a"] * 5, "p": [10.0, 20.0, 30.0, 40.0, 50.0]})
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_QUANTILE, 75), ["g"]
    )
    # Linear interpolation would give 40; "higher" takes the smallest observed
    # price at or above the 75th percentile.
    self.assertEqual(bench.tolist(), [40.0] * 5)
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_QUANTILE, 70), ["g"]
    )
    self.assertEqual(bench.tolist(), [40.0] * 5)
    self.assertIn(40.0, df["p"].tolist())

  def test_single_group_without_group_cols(self):
    """Without group columns the whole frame is one group."""
    bench = pb.compute_benchmark(
        self.df, "price", pb.BenchmarkSpec(pb.METHOD_QUANTILE, 100), []
    )
    self.assertEqual(bench.tolist(), [200.0] * len(self.df))

  def test_rolling_max_calendar_window_including_the_row(self):
    """The window is ``(t - W, t]``: the row's own date is included."""
    dates = pd.to_datetime(
        ["2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22"]
    )
    df = pd.DataFrame(
        {"g": "a", "date": dates, "p": [100.0, 120.0, 100.0, 100.0]}
    )
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="2W"), ["g"],
        date_col="date",
    )
    # (t - 14 days, t]: 3/1 -> {3/1}; 3/8 -> {3/1, 3/8}; 3/15 -> {3/8, 3/15};
    # 3/22 -> {3/15, 3/22}.
    self.assertEqual(bench.tolist(), [100.0, 120.0, 120.0, 100.0])
    gap = bench - df["p"]
    self.assertEqual(gap.tolist(), [0.0, 0.0, 20.0, 0.0])

  def test_rolling_max_skips_missing_dates_and_unsorted_rows(self):
    """Missing dates are absent from the window; row order is irrelevant."""
    # 3/8 is missing: on 3/15 the window (3/1, 3/15] holds 3/15 only.
    dates = pd.to_datetime(["2024-03-22", "2024-03-01", "2024-03-15"])
    df = pd.DataFrame({"g": "a", "date": dates, "p": [100.0, 120.0, 100.0]})
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="2W"), ["g"],
        date_col="date",
    )
    self.assertEqual(bench.tolist(), [100.0, 120.0, 100.0])

  def test_rolling_max_is_per_group(self):
    """The window of a group never sees the prices of another group."""
    dates = pd.to_datetime(["2024-03-01", "2024-03-08"] * 2)
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b"],
        "date": dates,
        "p": [100.0, 90.0, 50.0, 40.0],
    })
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="30D"), ["g"],
        date_col="date",
    )
    self.assertEqual(bench.tolist(), [100.0, 100.0, 50.0, 50.0])

  def test_rolling_max_requires_date_col(self):
    """``rolling_max`` without ``date_col`` is an error."""
    with self.assertRaisesRegex(ValueError, "requires date_col"):
      pb.compute_benchmark(
          self.df, "price",
          pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="2W"), ["item"],
      )

  def test_rolling_max_without_group_cols(self):
    """Without group columns the window runs over the whole frame."""
    dates = pd.to_datetime(["2024-03-01", "2024-03-08", "2024-03-15"])
    df = pd.DataFrame({"date": dates, "p": [100.0, 120.0, 90.0]})
    bench = pb.compute_benchmark(
        df, "p", pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="2W"), [],
        date_col="date",
    )
    # (t - 14 days, t]: 3/15 -> {3/8, 3/15}, so the 120 of 3/8 still counts.
    self.assertEqual(bench.tolist(), [100.0, 120.0, 120.0])

  def test_column_and_constant(self):
    """A benchmark column is used as given; a constant fills every row."""
    df = self.df.assign(
        list_price=np.where(self.df["item"] == "A", 110.0, 210.0)
    )
    bench = pb.compute_benchmark(
        df, "price", pb.BenchmarkSpec(pb.METHOD_COLUMN, column="list_price"),
        ["item"],
    )
    pd.testing.assert_series_equal(
        bench, df["list_price"].astype(float), check_names=False
    )
    bench = pb.compute_benchmark(
        df, "price", pb.BenchmarkSpec(pb.METHOD_CONSTANT, value=150.0), []
    )
    self.assertEqual(set(bench), {150.0})
    with self.assertRaisesRegex(ValueError, "not a column"):
      pb.compute_benchmark(
          df, "price", pb.BenchmarkSpec(pb.METHOD_COLUMN, column="nope"), []
      )

  def test_unresolved_spec_raises(self):
    """``METHOD_AUTO`` must be resolved before computing a benchmark."""
    with self.assertRaisesRegex(ValueError, "not resolved"):
      pb.compute_benchmark(
          self.df, "price", pb.BenchmarkSpec(pb.METHOD_AUTO), []
      )


class ShiftTest(unittest.TestCase):
  """The period length, the lag shift and the output names."""

  def test_period_delta_and_shift(self):
    """The shift takes the row exactly ``n`` periods earlier, else 0."""
    df = _weekly_panel()
    delta = pb.period_delta(df, "date", ["item", "geo"])
    self.assertEqual(delta, pd.Timedelta(days=7))
    # Drop one row to create a missing earlier date.
    missing = df[(df["item"] == "A") & (df["geo"] == "X")].index[1]
    df2 = df.drop(index=missing).reset_index(drop=True)
    df2["v"] = np.arange(len(df2), dtype=float)
    shifted = pb.shift_within_groups(
        df2, "v", 1, "date", ["item", "geo"], delta
    )
    for pos in range(len(df2)):
      row = df2.iloc[pos]
      earlier = df2[
          (df2["item"] == row["item"])
          & (df2["geo"] == row["geo"])
          & (df2["date"] == row["date"] - delta)
      ]
      expected = float(earlier["v"].iloc[0]) if len(earlier) else 0.0
      self.assertEqual(shifted.iloc[pos], expected)
    # The value is taken n periods earlier, not n rows earlier: the row after
    # the dropped date gets 0.
    after_missing = df2[
        (df2["item"] == "A")
        & (df2["geo"] == "X")
        & (df2["date"] == df["date"].iloc[2])
    ].index[0]
    self.assertEqual(shifted.iloc[after_missing], 0.0)

  def test_shift_keeps_nan_of_an_existing_row(self):
    """Only a missing earlier row is filled; an existing NaN stays NaN."""
    dates = pd.to_datetime(["2024-03-01", "2024-03-08"])
    df = pd.DataFrame({"date": dates, "v": [np.nan, 1.0]})
    shifted = pb.shift_within_groups(
        df, "v", 1, "date", [], pd.Timedelta(days=7)
    )
    self.assertEqual(shifted.iloc[0], 0.0)  # No earlier row.
    self.assertTrue(np.isnan(shifted.iloc[1]))  # Earlier row exists, NaN.

  def test_period_delta_falls_back_to_one_day(self):
    """With a single date there is no positive difference: one day."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01"] * 3),
        "g": ["a", "b", "c"],
    })
    self.assertEqual(pb.period_delta(df, "date", ["g"]), pd.Timedelta(days=1))

  def test_split_lag_and_output_names(self):
    """``<base>_l<n>`` is split; output names keep the lag suffix last."""
    self.assertEqual(pb.split_lag("price"), ("price", 0))
    self.assertEqual(pb.split_lag("price_l3"), ("price", 3))
    self.assertEqual(pb.split_lag("tv_l1_x"), ("tv_l1_x", 0))
    self.assertEqual(
        pb.output_names("price"),
        {
            "gap": "price_gap",
            "benchmark": "price_benchmark",
            "investment": "price_investment",
        },
    )
    names = pb.output_names("price", 2)
    self.assertEqual(names["gap"], "price_gap_l2")
    self.assertEqual(names["investment"], "price_investment_l2")
    # The lag suffix stays at the end: the orchestrator groups the lags of a
    # treatment with their base by stripping `_l<n>`.
    self.assertEqual(pb.split_lag(names["gap"]), ("price_gap", 2))


class InvestmentTest(unittest.TestCase):
  """The price investment ``C`` and the small counting helpers."""

  def test_callers_cost_is_zeroed_where_gap_is_not_positive(self):
    """The caller's cost is kept where D > 0 and replaced by 0 elsewhere."""
    gap = pd.Series([10.0, 0.0, -5.0, np.nan, 3.0], name="g")
    cost = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0])
    values, counts = pb.investment(gap, cost=cost)
    self.assertEqual(values.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
    self.assertEqual(counts["n_rows_cost_zeroed"], 3)
    self.assertEqual(counts["n_rows_sales_missing"], 0)

  def test_automatic_investment(self):
    """``C = sales / p * max(D, 0)``; NaN sales give 0, NaN gap stays NaN."""
    gap = pd.Series([20.0, 0.0, -10.0, 5.0, np.nan])
    prices = pd.Series([80.0, 100.0, 110.0, 95.0, 100.0])
    sales = pd.Series([800.0, 1000.0, 1100.0, np.nan, 500.0])
    values, counts = pb.investment(gap, sales=sales, prices=prices)
    # U = sales / p = 10, 10, 10, (NaN -> 0), 5; C = U * max(D, 0).
    self.assertEqual(values.tolist()[:4], [200.0, 0.0, 0.0, 0.0])
    self.assertTrue(np.isnan(values.iloc[4]))
    self.assertEqual(counts["n_rows_sales_missing"], 1)
    self.assertEqual(counts["n_rows_cost_zeroed"], 0)

  def test_non_positive_price_raises(self):
    """A price <= 0 (no quantity) and missing inputs are errors."""
    gap = pd.Series([20.0, 100.0])
    with self.assertRaisesRegex(ValueError, "<= 0 or NaN"):
      pb.investment(
          gap, sales=pd.Series([1.0, 1.0]), prices=pd.Series([80.0, 0.0])
      )
    with self.assertRaisesRegex(ValueError, "needs either"):
      pb.investment(gap)

  def test_gap_counts_and_reserved_names(self):
    """``gap_counts`` ignores NaN; keyword-named columns are detected."""
    counts = pb.gap_counts(pd.Series([0.0, 0.0, -1.0, 2.0, np.nan]))
    self.assertEqual(counts, {"n_rows_gap_zero": 2, "n_rows_gap_negative": 1})
    df = pd.DataFrame(columns=["price", "q_95", "as_is", "rolling_max", "q"])
    self.assertEqual(
        pb.reserved_column_collisions(df), ["q_95", "as_is", "rolling_max"]
    )


class _StopAtPhase0Error(Exception):
  """Raised by the Phase 0 stub that cuts a run short after the price step."""


class OrchestratorPriceTreatmentTest(unittest.TestCase):
  """`_apply_price_treatments` on a small panel (no pipeline run)."""

  # pylint: disable=protected-access

  def setUp(self):
    """Creates the output directory and the weekly panel."""
    super().setUp()
    self.out_dir = tempfile.mkdtemp()
    self.df = _weekly_panel()

  def tearDown(self):
    """Removes the output directory."""
    super().tearDown()
    shutil.rmtree(self.out_dir, ignore_errors=True)

  def _orchestrator(self, d_cols, treatment_types, **kwargs):
    """An orchestrator on the panel's columns; ``kwargs`` override."""
    params = dict(
        y_col="Y",
        d_cols=list(d_cols),
        x_cols_list=[["X1", "X2"] for _ in d_cols],
        ml_models_y=["lgbm"],
        ml_models_t=["lgbm"],
        n_folds_list=[2],
        output_dir=self.out_dir,
        date_col="date",
        geo_col="geo",
        item_col="item",
        sales_col="sales",
        treatment_types=treatment_types,
        generate_html=False,
    )
    params.update(kwargs)
    return workflow.CausalWorkflowOrchestrator(**params)

  def _apply(self, orch, df=None, scaler=None):
    """Runs the validation and the price step; returns (frame, warnings)."""
    df = self.df if df is None else df
    scaler = scaler or scalers.CausalDataScaler()
    orch._validate_columns(df)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      out = orch._apply_price_treatments(df, scaler)
    orch._validate_treatments(out)
    return out, [str(w.message) for w in caught]

  def test_constructor_normalizes_the_dict_form(self):
    """The type string is kept; the benchmark spec is stored separately."""
    orch = self._orchestrator(
        ["price", "tv"],
        {"price": {"type": "price", "benchmark": "q_95"}, "tv": "spend"},
    )
    self.assertEqual(orch.treatment_types, {"price": "price", "tv": "spend"})
    self.assertEqual(
        orch.price_benchmark_specs["price"],
        pb.BenchmarkSpec(pb.METHOD_QUANTILE, percentile=95),
    )
    self.assertNotIn("tv", orch.price_benchmark_specs)
    # Non-spend treatments force the simple optimizer as before.
    self.assertTrue(orch.simple_optimization)
    with self.assertRaisesRegex(ValueError, "only allowed for type 'price'"):
      self._orchestrator(
          ["tv"], {"tv": {"type": "spend", "benchmark": "q_95"}}
      )

  def test_constructor_rejects_keys_that_are_not_treatments(self):
    """A key of either dict that is not in ``d_cols`` is an error.

    Without the check the treatment would silently keep the default type
    ``spend`` (no benchmark step, no header line) and its own column as cost.
    """
    with self.assertRaisesRegex(
        ValueError,
        r"treatment_types: key\(s\) 'price' \(did you mean 'UnitPriceSM'\?\)"
        r" are not in d_cols \['UnitPriceSM'\]",
    ):
      self._orchestrator(
          ["UnitPriceSM"], {"price": {"type": "price", "benchmark": "q_95"}}
      )
    with self.assertRaisesRegex(
        ValueError,
        r"treatment_spend_cols: key\(s\) 'pricee' \(did you mean 'price'\?\)",
    ):
      self._orchestrator(
          ["price"], {"price": "price"}, treatment_spend_cols={"pricee": "Y"}
      )
    # A key with no close match gets no hint, but is rejected all the same.
    with self.assertRaisesRegex(ValueError, r"key\(s\) 'zzz' are not in"):
      self._orchestrator(["price"], {"price": "price", "zzz": "spend"})

  def test_constructor_rejects_a_lag_without_its_own_entry(self):
    """A lag of a ``price`` treatment must be typed itself, not defaulted."""
    with self.assertRaisesRegex(
        ValueError,
        r"'price_l1' has no entry while its base column 'price' is 'price'",
    ):
      self._orchestrator(["price", "price_l1"], {"price": "price"})
    # A lag of a `spend` treatment defaults to `spend` as before.
    orch = self._orchestrator(["tv", "tv_l1"], {"tv": "spend"})
    self.assertEqual(orch.treatment_types, {"tv": "spend", "tv_l1": "spend"})

  def test_auto_path_transforms_a_raw_price(self):
    """A raw price gets ``q_100``, the renames, the investment, the record."""
    orch = self._orchestrator(["price"], {"price": "price"})
    original = self.df.copy()
    out, messages = self._apply(orch)
    # The caller's frame is untouched; the new columns live in the copy.
    pd.testing.assert_frame_equal(self.df, original)
    for col in ("price_gap", "price_benchmark", "price_investment"):
      self.assertIn(col, out.columns)
      self.assertNotIn(col, self.df.columns)
    # Renames.
    self.assertEqual(orch.d_cols, ["price_gap"])
    self.assertEqual(orch.treatment_types, {"price_gap": "price"})
    self.assertEqual(
        orch.treatment_spend_cols, {"price_gap": "price_investment"}
    )
    self.assertEqual(orch.x_cols_list, [["X1", "X2", "price_benchmark"]])
    # Values: benchmark = group maximum, gap >= 0, investment = units * gap.
    expected_bench = out.groupby(["item", "geo"])["price"].transform("max")
    np.testing.assert_allclose(out["price_benchmark"], expected_bench)
    np.testing.assert_allclose(out["price_gap"], expected_bench - out["price"])
    np.testing.assert_allclose(
        out["price_investment"], out["units"] * out["price_gap"]
    )
    # The automatic decision is announced.
    self.assertTrue(
        any("gives no benchmark" in m and "q_100" in m for m in messages)
    )
    # Metadata record.
    record = orch._price_benchmark_records["price_gap"]
    self.assertEqual(record["source_column"], "price")
    self.assertEqual(record["lag"], 0)
    self.assertEqual(record["benchmark"]["description"], "q_100")
    self.assertEqual(record["benchmark_resolution"], "auto")
    self.assertEqual(record["benchmark_column"], "price_benchmark")
    self.assertEqual(record["group_cols"], ["item", "geo"])
    self.assertEqual(record["n_rows_gap_negative"], 0)
    self.assertEqual(
        record["n_rows_gap_zero"], int((out["price_gap"] == 0).sum())
    )
    self.assertEqual(record["cost_source"], "auto")
    self.assertEqual(record["cost_column"], "price_investment")
    self.assertEqual(record["cost_input_column"], "sales")
    # The benchmark table.
    path = os.path.join(self.out_dir, "price_benchmark", "price_benchmark.csv")
    table = pd.read_csv(path)
    self.assertEqual(list(table.columns), ["item", "geo", "price_benchmark"])
    self.assertEqual(len(table), 4)

  def test_auto_path_keeps_a_gap_column_as_is(self):
    """A column with a 0 is a gap already: nothing is renamed or built."""
    df = self.df.copy()
    df["price"] = df.groupby(["item", "geo"])["price"].transform("max") - df[
        "price"
    ]
    orch = self._orchestrator(["price"], {"price": "price"})
    out, messages = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["price"])
    self.assertEqual(orch.x_cols_list, [["X1", "X2"]])
    self.assertNotIn("price_gap", out.columns)
    # No price column -> no automatic investment; Phase 4 would skip the ROI.
    self.assertEqual(orch.treatment_spend_cols, {})
    self.assertTrue(any("used as is" in m for m in messages))
    record = orch._price_benchmark_records["price"]
    self.assertEqual(record["benchmark"]["method"], "as_is")
    self.assertEqual(record["cost_source"], "none")

  def test_as_is_with_callers_cost_column_is_zeroed_on_non_positive_gap(self):
    """With ``as_is`` the caller's cost column is still zeroed on D <= 0."""
    df = self.df.copy()
    df["gap"] = df.groupby(["item", "geo"])["price"].transform("max") - df[
        "price"
    ]
    df["cost"] = 7.0
    orch = self._orchestrator(
        ["gap"],
        {"gap": {"type": "price", "benchmark": "as_is"}},
        treatment_spend_cols={"gap": "cost"},
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["gap"])
    self.assertEqual(orch.treatment_spend_cols, {"gap": "gap_investment"})
    expected = np.where(out["gap"] > 0, 7.0, 0.0)
    np.testing.assert_allclose(out["gap_investment"], expected)
    record = orch._price_benchmark_records["gap"]
    self.assertEqual(record["cost_source"], "user")
    self.assertEqual(
        record["n_rows_cost_zeroed"], int((out["gap"] <= 0).sum())
    )

  def test_explicit_quantile_keeps_negative_gaps_and_zeroes_their_cost(self):
    """A price above ``q_50`` keeps its negative gap and gets cost 0."""
    # Item A / geo X: prices 100 x4, 80 x2 -> q_50 ("higher") is 100; add a
    # week above the list price so that a negative gap appears.
    df = self.df.copy()
    idx = df[(df["item"] == "A") & (df["geo"] == "X")].index[-1]
    df.loc[idx, "price"] = 130.0
    df.loc[idx, "sales"] = 130.0 * df.loc[idx, "units"]
    orch = self._orchestrator(
        ["price"], {"price": {"type": "price", "benchmark": "q_50"}}
    )
    out, messages = self._apply(orch, df)
    self.assertEqual(out.loc[idx, "price_benchmark"], 100.0)
    self.assertEqual(out.loc[idx, "price_gap"], -30.0)
    self.assertEqual(out.loc[idx, "price_investment"], 0.0)
    record = orch._price_benchmark_records["price_gap"]
    self.assertEqual(record["benchmark_resolution"], "explicit")
    self.assertEqual(record["n_rows_gap_negative"], 1)
    self.assertTrue(any("negative gap" in m for m in messages))
    self.assertFalse(any("gives no benchmark" in m for m in messages))

  def test_callers_cost_column_takes_precedence(self):
    """``treatment_spend_cols`` wins over the automatic investment."""
    df = self.df.assign(discount_cost=3.0)
    orch = self._orchestrator(
        ["price"],
        {"price": "price"},
        treatment_spend_cols={"price": "discount_cost"},
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(
        orch.treatment_spend_cols, {"price_gap": "price_investment"}
    )
    expected = np.where(out["price_gap"] > 0, 3.0, 0.0)
    np.testing.assert_allclose(out["price_investment"], expected)
    self.assertEqual(
        orch._price_benchmark_records["price_gap"]["cost_source"], "user"
    )

  def test_no_sales_col_means_no_investment(self):
    """Without ``sales_col`` the gap is built but no investment column."""
    orch = self._orchestrator(["price"], {"price": "price"}, sales_col=None)
    out, _ = self._apply(orch)
    self.assertIn("price_gap", out.columns)
    self.assertNotIn("price_investment", out.columns)
    self.assertEqual(orch.treatment_spend_cols, {})
    self.assertEqual(
        orch._price_benchmark_records["price_gap"]["cost_source"], "none"
    )

  def test_non_positive_price_with_explicit_benchmark_raises(self):
    """A price <= 0 breaks the automatic quantity, not a caller's cost."""
    df = self.df.copy()
    df.loc[0, "price"] = 0.0
    orch = self._orchestrator(
        ["price"], {"price": {"type": "price", "benchmark": "q_100"}}
    )
    with self.assertRaisesRegex(ValueError, "<= 0 or NaN"):
      self._apply(orch, df)
    # With a caller's cost column the quantity is not needed: no error.
    df["cost"] = 1.0
    orch = self._orchestrator(
        ["price"],
        {"price": {"type": "price", "benchmark": "q_100"}},
        treatment_spend_cols={"price": "cost"},
    )
    out, _ = self._apply(orch, df)
    self.assertIn("price_investment", out.columns)

  def test_rolling_max_and_constant_and_column_benchmarks(self):
    """The three explicit benchmark kinds and their x_cols / table shape."""
    df = self.df.assign(
        list_price=np.where(self.df["item"] == "A", 100.0, 200.0)
    )
    orch = self._orchestrator(
        ["price"],
        {
            "price": {
                "type": "price",
                "benchmark": "rolling_max",
                "window": "2W",
            }
        },
    )
    out, _ = self._apply(orch, df)
    expected = pb.compute_benchmark(
        df, "price", pb.BenchmarkSpec(pb.METHOD_ROLLING_MAX, window="2W"),
        ["item", "geo"], "date",
    )
    np.testing.assert_allclose(out["price_benchmark"], expected)
    table = pd.read_csv(
        os.path.join(self.out_dir, "price_benchmark", "price_benchmark.csv")
    )
    self.assertEqual(
        list(table.columns), ["item", "geo", "date", "price_benchmark"]
    )
    # A constant benchmark is not added to x_cols (no information).
    orch = self._orchestrator(
        ["price"], {"price": {"type": "price", "benchmark": 250}}
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(set(out["price_benchmark"]), {250.0})
    self.assertEqual(orch.x_cols_list, [["X1", "X2"]])
    # A column benchmark.
    orch = self._orchestrator(
        ["price"], {"price": {"type": "price", "benchmark": "list_price"}}
    )
    out, _ = self._apply(orch, df)
    np.testing.assert_allclose(out["price_benchmark"], df["list_price"])
    self.assertEqual(orch.x_cols_list, [["X1", "X2", "price_benchmark"]])

  def test_benchmark_table_is_sorted_by_group_and_date(self):
    """The CSV lists the groups in order whatever the row order of ``df``."""
    # The panel reversed: the last group (item B / geo Y) and the last week
    # come first.
    df = self.df.iloc[::-1].reset_index(drop=True)
    path = os.path.join(self.out_dir, "price_benchmark", "price_benchmark.csv")
    orch = self._orchestrator(["price"], {"price": "price"})
    self._apply(orch, df)
    table = pd.read_csv(path)
    self.assertEqual(
        table[["item", "geo"]].values.tolist(),
        [["A", "X"], ["A", "Y"], ["B", "X"], ["B", "Y"]],
    )
    # A time-varying benchmark: one row per group and date, the dates
    # ascending within each group.
    orch = self._orchestrator(
        ["price"],
        {
            "price": {
                "type": "price",
                "benchmark": "rolling_max",
                "window": "2W",
            }
        },
    )
    self._apply(orch, df)
    table = pd.read_csv(path, parse_dates=["date"])
    self.assertEqual(len(table), len(df))
    pd.testing.assert_frame_equal(
        table,
        table.sort_values(["item", "geo", "date"], kind="stable").reset_index(
            drop=True
        ),
    )

  def test_reserved_column_name_raises(self):
    """A column named like a benchmark keyword is rejected."""
    df = self.df.assign(q_95=1.0)
    orch = self._orchestrator(["price"], {"price": "price"})
    with self.assertRaisesRegex(ValueError, "benchmark keyword"):
      self._apply(orch, df)

  def test_lag_treatment_shifts_gap_and_investment(self):
    """``price_l1`` gets the gap and investment of one period earlier."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    orch = self._orchestrator(
        ["price", "price_l1"],
        {
            "price": {"type": "price", "benchmark": "q_100"},
            "price_l1": "price",
        },
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["price_gap", "price_gap_l1"])
    self.assertEqual(
        orch.treatment_spend_cols,
        {
            "price_gap": "price_investment",
            "price_gap_l1": "price_investment_l1",
        },
    )
    self.assertEqual(
        orch.x_cols_list,
        [["X1", "X2", "price_benchmark"], ["X1", "X2", "price_benchmark"]],
    )
    delta = pd.Timedelta(days=7)
    for col in ("gap", "investment"):
      shifted = pb.shift_within_groups(
          out, f"price_{col}", 1, "date", ["item", "geo"], delta
      )
      np.testing.assert_allclose(out[f"price_{col}_l1"], shifted)
    # First week of every group: no earlier row -> 0.
    first = (
        out.groupby(["item", "geo"])["date"].transform("min") == out["date"]
    )
    self.assertTrue((out.loc[first, "price_gap_l1"] == 0).all())
    record = orch._price_benchmark_records["price_gap_l1"]
    self.assertEqual(record["lag"], 1)
    self.assertEqual(record["source_column"], "price")
    self.assertEqual(record["cost_column"], "price_investment_l1")

  def test_lag_only_treatment_uses_the_base_column(self):
    """A lag alone is computed from its base column, which must exist."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    orch = self._orchestrator(
        ["price_l1"], {"price_l1": {"type": "price", "benchmark": "q_100"}}
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["price_gap_l1"])
    self.assertIn("price_gap", out.columns)
    with self.assertRaisesRegex(ValueError, "not a column of the DataFrame"):
      orch2 = self._orchestrator(["price_l1"], {"price_l1": "price"})
      self._apply(orch2, df.drop(columns=["price"]))

  def test_lag_with_callers_cost_column(self):
    """A lag's own ``treatment_spend_cols`` entry wins over the shifted one."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    df["lag_cost"] = 4.0
    orch = self._orchestrator(
        ["price", "price_l1"],
        {"price": "price", "price_l1": "price"},
        treatment_spend_cols={"price_l1": "lag_cost"},
    )
    out, _ = self._apply(orch, df)
    # The base keeps the automatic investment; the lag gets the caller's
    # column, zeroed where its (shifted) gap is <= 0.
    self.assertEqual(
        orch.treatment_spend_cols,
        {
            "price_gap": "price_investment",
            "price_gap_l1": "price_investment_l1",
        },
    )
    np.testing.assert_allclose(
        out["price_investment"], out["units"] * out["price_gap"]
    )
    np.testing.assert_allclose(
        out["price_investment_l1"], np.where(out["price_gap_l1"] > 0, 4.0, 0.0)
    )
    record = orch._price_benchmark_records["price_gap_l1"]
    self.assertEqual(record["cost_source"], "user")
    self.assertEqual(record["cost_input_column"], "lag_cost")

  def test_lag_without_any_cost_column(self):
    """Without ``sales_col`` neither the base nor the lag has an investment."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    orch = self._orchestrator(
        ["price", "price_l1"],
        {"price": "price", "price_l1": "price"},
        sales_col=None,
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["price_gap", "price_gap_l1"])
    self.assertIn("price_gap_l1", out.columns)
    self.assertNotIn("price_investment", out.columns)
    self.assertNotIn("price_investment_l1", out.columns)
    self.assertEqual(orch.treatment_spend_cols, {})
    for gap_col in ("price_gap", "price_gap_l1"):
      record = orch._price_benchmark_records[gap_col]
      self.assertEqual(record["cost_source"], "none")
      self.assertIsNone(record["cost_column"])

  def test_nan_sales_are_counted_and_logged(self):
    """A NaN ``sales_col`` row gets investment 0 and is named in the log."""
    df = self.df.copy()
    idx = df[(df["item"] == "A") & (df["geo"] == "X")].index[2]
    self.assertGreater(
        df.groupby(["item", "geo"])["price"].transform("max")[idx]
        - df.loc[idx, "price"],
        0,
    )
    df.loc[idx, "sales"] = np.nan
    orch = self._orchestrator(["price"], {"price": "price"})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
      out, _ = self._apply(orch, df)
    self.assertEqual(out.loc[idx, "price_investment"], 0.0)
    record = orch._price_benchmark_records["price_gap"]
    self.assertEqual(record["n_rows_sales_missing"], 1)
    self.assertIn("1 row(s) with NaN sales set to 0", log.getvalue())

  def test_conflicting_lag_benchmarks_raise(self):
    """A base column and its lag must agree on the benchmark."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    orch = self._orchestrator(
        ["price", "price_l1"],
        {
            "price": {"type": "price", "benchmark": "q_100"},
            "price_l1": {"type": "price", "benchmark": "q_95"},
        },
    )
    with self.assertRaisesRegex(ValueError, "different benchmarks"):
      self._apply(orch, df)

  def test_scaler_lists_are_extended(self):
    """A scaler list naming the price column receives gap and benchmark."""
    scaler = scalers.CausalDataScaler(standardize_cols=["price", "X1"])
    orch = self._orchestrator(["price"], {"price": "price"})
    self._apply(orch, scaler=scaler)
    self.assertEqual(
        scaler.standardize_cols,
        ["price", "X1", "price_gap", "price_benchmark"],
    )
    untouched = scalers.CausalDataScaler(standardize_cols=["X1"])
    orch = self._orchestrator(["price"], {"price": "price"})
    self._apply(orch, scaler=untouched)
    self.assertEqual(untouched.standardize_cols, ["X1"])

  def test_restore_before_a_second_run(self):
    """The renames are undone before a second run and redone identically."""
    orch = self._orchestrator(
        ["price", "tv"], {"price": "price", "tv": "spend"}
    )
    df = self.df.assign(tv=1.0)
    out1, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["price_gap", "tv"])
    orch._restore_price_config()
    self.assertEqual(orch.d_cols, ["price", "tv"])
    self.assertEqual(orch.treatment_types, {"price": "price", "tv": "spend"})
    self.assertEqual(orch.treatment_spend_cols, {"tv": "tv"})
    self.assertEqual(orch.x_cols_list, [["X1", "X2"], ["X1", "X2"]])
    out2, _ = self._apply(orch, df)
    pd.testing.assert_frame_equal(out1, out2)
    self.assertEqual(orch.d_cols, ["price_gap", "tv"])
    self.assertEqual(orch.x_cols_list[1], ["X1", "X2"])

  def test_run_full_pipeline_restores_the_config_before_a_second_run(self):
    """``run_full_pipeline`` starts a second run from the caller's config.

    Each run is cut short at Phase 0 (replaced by a stub that raises), right
    after the price step.  Without the restore the second call would look for
    the renamed treatment ``price_gap`` in the caller's frame and fail its
    column validation instead.
    """
    orch = self._orchestrator(["price"], {"price": "price"})
    scaler = scalers.CausalDataScaler()

    def run_until_phase0():
      with mock.patch.object(
          orch, "_run_phase0", side_effect=_StopAtPhase0Error
      ), warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        with self.assertRaises(_StopAtPhase0Error):
          orch.run_full_pipeline(df=self.df, scaler=scaler)

    run_until_phase0()
    self.assertEqual(orch.d_cols, ["price_gap"])
    run_until_phase0()
    self.assertEqual(orch.d_cols, ["price_gap"])
    self.assertEqual(orch.treatment_types, {"price_gap": "price"})
    self.assertEqual(
        orch.treatment_spend_cols, {"price_gap": "price_investment"}
    )
    self.assertEqual(orch.x_cols_list, [["X1", "X2", "price_benchmark"]])

  def test_no_group_cols_warns_and_uses_one_group(self):
    """Without item_col and geo_col the whole frame is one group."""
    orch = self._orchestrator(
        ["price"], {"price": "price"}, geo_col=None, item_col=None
    )
    out, messages = self._apply(orch)
    self.assertEqual(set(out["price_benchmark"]), {200.0})
    self.assertTrue(any("one group" in m for m in messages))

  def test_as_is_lag_needs_neither_base_column_nor_date_col(self):
    """An explicit ``as_is`` lag column is accepted on its own."""
    df = self.df.copy()
    df["gap_l1"] = 5.0
    df.loc[df.index[:3], "gap_l1"] = 0.0
    orch = self._orchestrator(
        ["gap_l1"], {"gap_l1": {"type": "price", "benchmark": "as_is"}},
        date_col=None,
    )
    out, _ = self._apply(orch, df)
    self.assertEqual(orch.d_cols, ["gap_l1"])
    self.assertNotIn("gap", out.columns)
    self.assertEqual(
        orch._price_benchmark_records["gap_l1"]["benchmark"]["method"], "as_is"
    )

  def test_lag_treatment_requires_date_col(self):
    """A lag that must be shifted needs ``date_col``."""
    df = features.create_lagged_features(
        self.df, ["price"], 1, "date", unit_col=["item", "geo"], na_fill=True
    ).reset_index(drop=True)
    orch = self._orchestrator(
        ["price_l1"], {"price_l1": "price"}, date_col=None
    )
    with self.assertRaisesRegex(ValueError, "require date_col"):
      self._apply(orch, df)


class OddExtensionTest(unittest.TestCase):
  """``f(T) = -f(-T)`` for the families, the derivative and the T-basis."""

  def test_families_are_odd(self):
    """Every family and ``fbar`` are odd; the derivative is even and > 0."""
    t = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    for family in saturation.SUPPORTED_FAMILIES:
      f_pos = saturation.apply_saturation(np.abs(t), family, 0.7)
      f_val = saturation.apply_saturation(t, family, 0.7)
      np.testing.assert_allclose(f_val, np.sign(t) * f_pos, err_msg=family)
      fbar = saturation.normalized_saturation(t, family, 0.7, 2.0)
      np.testing.assert_allclose(fbar, -fbar[::-1], err_msg=family)
      self.assertEqual(fbar[3], 0.0)
      deriv = saturation.saturation_derivative(t, family, 0.7)
      np.testing.assert_allclose(deriv, deriv[::-1], err_msg=family)
      self.assertTrue(np.all(deriv > 0.0), msg=family)

  def test_basis_is_odd_and_t_ref_uses_magnitudes(self):
    """The T-basis mirrors ``fbar``; T_ref / T_max are taken on |T|."""
    rng = np.random.default_rng(0)
    t_values = np.concatenate([rng.uniform(0.0, 10.0, 300), [-4.0, -1.0]])
    basis = saturation.TreatmentBasis(t_values, basis_dim=8, t_ref=5.0)
    t = np.array([0.5, 2.0, 7.5])
    np.testing.assert_allclose(basis.transform(-t), -basis.transform(t))
    coefs = basis.project_family("tanh", np.array([0.6]))
    approx = basis.transform(-t) @ coefs[:, 0]
    exact = saturation.normalized_saturation(-t, "tanh", 0.6, 5.0)
    np.testing.assert_allclose(approx, exact, atol=1e-2)
    self.assertEqual(
        saturation.reference_point(np.array([-4.0, 1.0, 2.0]), 1.0), 4.0
    )
    self.assertEqual(
        saturation.basis_domain_max(np.array([-4.0, 1.0, 2.0]), 1.0), 4.0
    )
    # Observations beyond T_max are counted on |T|.
    small = saturation.TreatmentBasis(
        np.array([-5.0, 1.0, 2.0, 3.0]), basis_dim=4, t_ref=2.0, t_max=3.0
    )
    self.assertEqual(small.n_above_t_max, 1)


if __name__ == "__main__":
  unittest.main()
