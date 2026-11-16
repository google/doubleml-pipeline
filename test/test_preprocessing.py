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

"""Tests for the ``preprocessing`` package: ``features`` and ``scalers``."""

import re
import unittest

from doubleml_pipeline.preprocessing import features
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
import numpy as np
import pandas as pd
import pytest


class FeaturesTest(unittest.TestCase):
  """Tests for feature engineering functions."""

  def test_create_lagged_features_basic(self):
    """Tests basic functionality of create_lagged_features."""
    df = pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=5),
        "value": [1, 2, 3, 4, 5],
        "geo": ["A", "A", "A", "A", "A"],
    })
    df_lagged = features.create_lagged_features(
        df=df,
        columns=["value"],
        periods=2,
        date_col="date",
        unit_col="geo",
        na_fill=True,
    )

    self.assertIn("value_l1", df_lagged.columns)
    self.assertIn("value_l2", df_lagged.columns)

    # Check first lag
    self.assertEqual(df_lagged.iloc[1]["value_l1"], 1.0)
    self.assertEqual(df_lagged.iloc[2]["value_l1"], 2.0)

    # Check second lag
    self.assertEqual(df_lagged.iloc[2]["value_l2"], 1.0)
    self.assertEqual(df_lagged.iloc[3]["value_l2"], 2.0)

    # Check NA fill for the first row
    self.assertEqual(df_lagged.iloc[0]["value_l1"], 0.0)
    self.assertEqual(df_lagged.iloc[0]["value_l2"], 0.0)

  def test_create_lagged_features_multiple_unit_cols(self):
    """Tests create_lagged_features with multiple unit_cols."""
    df = pd.DataFrame({
        "date": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
        ),
        "value": [1, 2, 3, 4],
        "geo1": ["A", "A", "B", "B"],
        "geo2": ["X", "X", "Y", "Y"],
    })
    df_lagged = features.create_lagged_features(
        df=df,
        columns=["value"],
        periods=1,
        date_col="date",
        unit_col=["geo1", "geo2"],
        na_fill=True,
    )

    df_lagged = df_lagged.sort_values(by=["geo1", "geo2", "date"]).reset_index(
        drop=True
    )

    self.assertIn("value_l1", df_lagged.columns)

    # geo1="A", geo2="X", date="2020-01-01" -> no lag
    self.assertEqual(df_lagged.iloc[0]["value_l1"], 0.0)
    # geo1="A", geo2="X", date="2020-01-02" -> lag is 1
    self.assertEqual(df_lagged.iloc[1]["value_l1"], 1.0)
    # geo1="B", geo2="Y", date="2020-01-01" -> no lag
    self.assertEqual(df_lagged.iloc[2]["value_l1"], 0.0)
    # geo1="B", geo2="Y", date="2020-01-02" -> lag is 3
    self.assertEqual(df_lagged.iloc[3]["value_l1"], 3.0)

  def test_create_lagged_features_missing_unit_col_raises(self):
    """A misspelled unit_col raises instead of silently dropping the grouping."""
    df = pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=4),
        "value": [1, 2, 3, 4],
        "geo": ["A", "A", "B", "B"],
    })
    with self.assertRaisesRegex(ValueError, r"unit_col \['geo_typo'\]"):
      features.create_lagged_features(
          df=df,
          columns=["value"],
          periods=1,
          date_col="date",
          unit_col="geo_typo",
      )
    # Only the absent entries of a list are reported.
    with self.assertRaisesRegex(ValueError, r"unit_col \['item'\]"):
      features.create_lagged_features(
          df=df,
          columns=["value"],
          periods=1,
          date_col="date",
          unit_col=["geo", "item"],
      )


@pytest.mark.parametrize("freq", ["W-MON", "W-FRI", "W-SAT", "W-SUN"])
def test_create_lagged_features_weekly(freq):
  """Tests create_lagged_features with various weekly frequencies."""
  df = pd.DataFrame({
      "date": pd.date_range(start="2020-01-01", periods=5, freq=freq),
      "value": [1, 2, 3, 4, 5],
      "geo": ["A", "A", "A", "A", "A"],
  })
  df_lagged = features.create_lagged_features(
      df=df,
      columns=["value"],
      periods=2,
      date_col="date",
      unit_col="geo",
      na_fill=True,
  )
  assert not df_lagged.isnull().values.any()


class ScalersTest(unittest.TestCase):
  """Test suite for causal data scaling functionality."""

  def test_causal_data_scaler_fit_transform(self):
    """Tests the fit_transform method of CausalDataScaler."""
    scaler = scalers.CausalDataScaler(
        standardize_cols=["value1"],
        population_median_normalize_cols=["value2"],
        population_col="pop",
    )
    df = pd.DataFrame({
        "value1": [10, 20, 30],
        "value2": [100, 200, 300],
        "pop": [1000, 2000, 3000],
    })

    df_scaled = scaler.fit_transform(df)

    self.assertIn("value1", df_scaled.columns)
    self.assertIn("value2", df_scaled.columns)
    self.assertTrue(hasattr(scaler, "raw_population_"))

    # value1 should be standardized (mean=20, std=10)
    # value2 should be population median normalized
    self.assertAlmostEqual(df_scaled["value1"].mean(), 0.0, places=5)

  def test_inverse_transform(self):
    """Tests the inverse_transform_column method of CausalDataScaler."""
    scaler = scalers.CausalDataScaler(
        standardize_cols=["value1"], population_col="pop"
    )
    df = pd.DataFrame({"value1": [10, 20, 30], "pop": [1000, 2000, 3000]})

    df_scaled = scaler.fit_transform(df)
    df_restored = scaler.inverse_transform_column(
        df_scaled, "value1", param_source_col="value1"
    )

    self.assertAlmostEqual(df_restored.iloc[0], 10.0, places=5)
    self.assertAlmostEqual(df_restored.iloc[1], 20.0, places=5)
    self.assertAlmostEqual(df_restored.iloc[2], 30.0, places=5)

  def test_minmax_scaler_forward_backward(self):
    """Tests forward and backward passes for MinMax Scaling."""
    scaler = scalers.CausalDataScaler(
        min_max_cols=["value1"],
        population_standardize_cols=["value2"],
        population_col="pop",
    )
    df = pd.DataFrame({
        "value1": [10, 20, 30],
        "value2": [100, 200, 300],
        "pop": [1000, 2000, 3000],
    })

    # Forward pass
    df_scaled = scaler.fit_transform(df)

    # Check that _raw_population_ is completely eliminated
    self.assertNotIn("_raw_population_", df_scaled.columns)

    # Backward pass
    df_restored = scaler.inverse_transform_column(df_scaled, "value1")

    # mathematically matches original
    self.assertAlmostEqual(df_restored.iloc[0], 10.0, places=5)
    self.assertAlmostEqual(df_restored.iloc[1], 20.0, places=5)
    self.assertAlmostEqual(df_restored.iloc[2], 30.0, places=5)

  def test_fit_raises_for_missing_scaling_column(self):
    """A misspelled scaling column raises instead of being silently skipped."""
    scaler = scalers.CausalDataScaler(standardize_cols=["value1", "value_typo"])
    df = pd.DataFrame({"value1": [10, 20, 30]})
    with self.assertRaisesRegex(
        ValueError, r"standardize_cols: \['value_typo'\]"
    ):
      scaler.fit(df)
    # Nothing was fitted.
    self.assertEqual(scaler.standard_params_, {})

  def test_fit_reports_every_scaling_list_with_missing_columns(self):
    """Every list with a missing column is named; complete lists are not."""
    scaler = scalers.CausalDataScaler(
        standardize_cols=["value1"],
        min_max_cols=["a_typo"],
        median_normalize_cols=["b_typo"],
    )
    df = pd.DataFrame({"value1": [10, 20, 30]})
    with self.assertRaises(ValueError) as ctx:
      scaler.fit(df)
    message = str(ctx.exception)
    self.assertIn("min_max_cols: ['a_typo']", message)
    self.assertIn("median_normalize_cols: ['b_typo']", message)
    self.assertNotIn("standardize_cols", message)

  def test_transform_still_tolerates_sub_frames(self):
    """Only fit is strict: transform keeps working on column sub-frames."""
    scaler = scalers.CausalDataScaler(standardize_cols=["value1", "value2"])
    df = pd.DataFrame({"value1": [10, 20, 30], "value2": [1, 2, 3]})
    scaler.fit(df)
    df_scaled = scaler.transform(df[["value1"]])
    self.assertEqual(list(df_scaled.columns), ["value1"])
    self.assertAlmostEqual(df_scaled["value1"].mean(), 0.0, places=5)

  def test_column_in_two_lists_is_rejected_at_construction(self):
    """A column under two strategies would be scaled twice and undone once."""
    with self.assertRaisesRegex(
        ValueError,
        r"'value1' \(population_median_normalize_cols, median_normalize_cols\)",
    ):
      scalers.CausalDataScaler(
          median_normalize_cols=["value1"],
          population_median_normalize_cols=["value1", "value2"],
          population_col="pop",
      )

  def test_column_repeated_in_one_list_is_rejected_at_construction(self):
    """``transform`` loops over the list, so a repeated name scales twice."""
    with self.assertRaisesRegex(
        ValueError, r"'value1' \(standardize_cols, standardize_cols\)"
    ):
      scalers.CausalDataScaler(standardize_cols=["value1", "value2", "value1"])

  def test_repeated_column_error_names_only_the_repeated_columns(self):
    """Every repeated column is named with its lists; the others are not."""
    with self.assertRaises(ValueError) as ctx:
      scalers.CausalDataScaler(
          standardize_cols=["a", "b"],
          min_max_cols=["a"],
          median_normalize_cols=["c", "c"],
          population_median_normalize_cols=["d"],
      )
    message = str(ctx.exception)
    self.assertIn("'a' (standardize_cols, min_max_cols)", message)
    self.assertIn(
        "'c' (median_normalize_cols, median_normalize_cols)", message
    )
    self.assertNotIn("'b'", message)
    self.assertNotIn("'d'", message)

  def test_fit_rejects_a_column_added_to_a_second_list(self):
    """The lists are mutable (the price step extends them), so fit re-checks."""
    scaler = scalers.CausalDataScaler(
        median_normalize_cols=["value1"], population_col="pop"
    )
    scaler.population_median_normalize_cols.append("value1")
    df = pd.DataFrame({"value1": [10.0, 20.0, 30.0], "pop": [1.0, 2.0, 4.0]})
    with self.assertRaisesRegex(ValueError, "listed more than once"):
      scaler.fit(df)
    # Nothing was fitted.
    self.assertEqual(scaler.median_params_, {})
    self.assertEqual(scaler.pop_median_params_, {})

  def test_transform_rejects_a_list_edited_after_fit(self):
    """A column listed twice after ``fit`` is caught before it is scaled.

    ``fit`` keeps the parameters of earlier fits, so a column moved from one
    list to another and then listed under both would be scaled by both.
    """
    df = pd.DataFrame({"value1": [10.0, 20.0, 30.0], "pop": [1.0, 2.0, 4.0]})
    scaler = scalers.CausalDataScaler(
        median_normalize_cols=["value1"], population_col="pop"
    )
    scaler.fit(df)
    scaler.median_normalize_cols = []
    scaler.population_median_normalize_cols = ["value1"]
    scaler.fit(df)
    scaler.median_normalize_cols = ["value1"]
    with self.assertRaisesRegex(ValueError, "listed more than once"):
      scaler.transform(df)


# ---------------------------------------------------------------------------
# The operations of the scaler, each for every scaling strategy and on two
# columns: `spend` has a row with 0 (the two median strategies skip it) and
# `sales` has none, with a minimum other than 0 (so that a forgotten shift
# shows).  Expected values are the hand formulas of test/helpers.py.
# ---------------------------------------------------------------------------

_COLUMNS = ("spend", "sales")


def _fit_frame() -> pd.DataFrame:
  """The rows the scaler is fitted on; the population differs by row."""
  return pd.DataFrame({
      "spend": [0.0, 200.0, 600.0, 900.0, 3000.0, 9000.0],
      "sales": [700.0, 400.0, 1500.0, 1200.0, 5200.0, 8100.0],
      "pop": [100.0, 100.0, 200.0, 150.0, 500.0, 1000.0],
  })


def _other_frame() -> pd.DataFrame:
  """Rows that take no part in the fit: other values, other population."""
  return pd.DataFrame(
      {
          "spend": [1500.0, 0.0, 2500.0],
          "sales": [900.0, 3000.0, 6000.0],
          "pop": [500.0, 250.0, 400.0],
      },
      index=[100, 101, 102],
  )


def _second_fit_frame() -> pd.DataFrame:
  """The rows of a second fit, far from the values of `_fit_frame`."""
  return pd.DataFrame({
      "spend": [0.0, 30.0, 50.0, 80.0, 110.0, 140.0],
      "sales": [60.0, 20.0, 90.0, 75.0, 130.0, 45.0],
      "pop": [10.0, 20.0, 10.0, 40.0, 20.0, 50.0],
  })


def _scaler_for(strategy: str, column: str) -> scalers.CausalDataScaler:
  """A scaler with `column` as the only column, listed under `strategy`."""
  return scalers.CausalDataScaler(
      population_col="pop", **{f"{strategy}_cols": [column]}
  )


def _by_hand(strategy: str, fitted: pd.DataFrame, rows: pd.DataFrame, column):
  """`rows[column]` scaled by hand with the numbers of a fit on `fitted`."""
  numbers = helpers.scaling_numbers(strategy, fitted[column], fitted["pop"])
  return helpers.scale_by_hand(strategy, numbers, rows[column], rows["pop"])


def _assert_close(actual, expected):
  np.testing.assert_allclose(
      np.asarray(actual, dtype=float),
      np.asarray(expected, dtype=float),
      rtol=1e-9,
      atol=1e-9,
  )


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        # mean 200, standard deviation 100
        ("standardize", [-1.0, 1.0]),
        # min 100, max 300
        ("min_max", [0.0, 1.0]),
        # median 200
        ("median_normalize", [0.5, 1.5]),
        # per head 4 and 6: median 5
        ("population_median_normalize", [0.8, 1.2]),
        # per head 4 and 6: mean 5, standard deviation 1
        ("population_standardize", [-1.0, 1.0]),
    ],
)
def test_transform_of_two_rows_small_enough_to_check_by_eye(strategy, expected):
  """The formula of every strategy on numbers that need no calculator."""
  df = pd.DataFrame({"value": [100.0, 300.0], "pop": [25.0, 50.0]})
  scaler = _scaler_for(strategy, "value")

  scaled = scaler.fit_transform(df)

  _assert_close(scaled["value"], expected)


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_transform_matches_the_hand_formula(strategy, column):
  df = _fit_frame()
  scaler = _scaler_for(strategy, column)

  scaled = scaler.fit_transform(df)

  _assert_close(scaled[column], _by_hand(strategy, df, df, column))


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_inverse_restores_the_fitted_rows(strategy, column):
  df = _fit_frame()
  scaler = _scaler_for(strategy, column)
  scaled = scaler.fit_transform(df)

  restored = scaler.inverse_transform_column(scaled, column)

  _assert_close(restored, df[column])


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_inverse_of_a_difference_is_the_difference_of_the_inverses(
    strategy, column
):
  """`is_difference=True` restores the difference of two scaled values."""
  df = _fit_frame()
  scaler = _scaler_for(strategy, column)
  scaled = scaler.fit_transform(df)
  # Two scaled quantities of the column and their difference, like the
  # prediction with and without the treatment (the incremental KPI).
  scaled["a"] = scaled[column] + 0.25
  scaled["b"] = 0.5 * scaled[column]
  scaled["a_minus_b"] = scaled["a"] - scaled["b"]

  difference = scaler.inverse_transform_column(
      scaled, "a_minus_b", param_source_col=column, is_difference=True
  )

  restored_a = scaler.inverse_transform_column(
      scaled, "a", param_source_col=column
  )
  restored_b = scaler.inverse_transform_column(
      scaled, "b", param_source_col=column
  )
  _assert_close(difference, restored_a - restored_b)
  numbers = helpers.scaling_numbers(strategy, df[column], df["pop"])
  _assert_close(
      difference,
      helpers.unscale_by_hand(
          strategy, numbers, scaled["a_minus_b"], df["pop"], is_difference=True
      ),
  )


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_transform_of_rows_not_used_in_the_fit(strategy, column):
  fitted, other = _fit_frame(), _other_frame()
  scaler = _scaler_for(strategy, column).fit(fitted)

  scaled = scaler.transform(other)

  _assert_close(scaled[column], _by_hand(strategy, fitted, other, column))


# The strategies that restore a value without the population of its row.
_STRATEGIES_WITHOUT_POPULATION = (
    "standardize",
    "min_max",
    "median_normalize",
)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param([100, 101, 102], id="new_labels"),
        pytest.param([0, 1, 2], id="labels_of_fitted_rows"),
    ],
)
@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", _STRATEGIES_WITHOUT_POPULATION)
def test_inverse_restores_rows_not_used_in_the_fit(strategy, column, index):
  """Rows that took no part in the fit are restored with its numbers.

  The pipeline scales and restores the rows it only scores (Phase 0's `D_sel`)
  with the numbers of the rows it fits on (`D_inf`).  The two population
  strategies are not part of this test: `inverse_transform_column` multiplies
  by the population that `fit` remembered, looked up by index label, so a
  caller adds the population of the other rows to `raw_population_` before it
  restores them.  The pipeline does; test_scaling_pipeline.py checks the
  result.
  """
  fitted = _fit_frame()
  other = _other_frame().set_axis(index, axis="index")
  scaler = _scaler_for(strategy, column).fit(fitted)
  scaled = scaler.transform(other)

  restored = scaler.inverse_transform_column(scaled, column)

  _assert_close(restored, other[column])


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.POPULATION_STRATEGIES)
def test_population_strategies_stop_without_the_population_column(
    strategy, column
):
  """`fit`, `transform` and `inverse_transform_column` all need the column."""
  df = _fit_frame()
  without_population = df.drop(columns="pop")
  message = re.escape(
      "Specified population column 'pop' not found in dataframe."
  )

  with pytest.raises(ValueError, match=message):
    _scaler_for(strategy, column).fit(without_population)

  scaler = _scaler_for(strategy, column).fit(df)
  with pytest.raises(ValueError, match=message):
    scaler.transform(without_population)
  scaled_without_population = scaler.transform(df).drop(columns="pop")
  with pytest.raises(ValueError, match=message):
    scaler.inverse_transform_column(scaled_without_population, column)


# A scaler that is fitted a second time must behave like a scaler fitted only
# on the second frame.  The pipeline fits the scaler it is given at the start
# of Phases 1-3; the first fit is one the caller ran beforehand.


@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_second_fit_replaces_the_numbers_of_the_first(strategy, column):
  first, second = _fit_frame(), _second_fit_frame()
  scaler = _scaler_for(strategy, column).fit(first)

  scaled = scaler.fit_transform(second)
  restored = scaler.inverse_transform_column(scaled, column)

  _assert_close(scaled[column], _by_hand(strategy, second, second, column))
  _assert_close(restored, second[column])


_KEPT_NUMBERS_REASON = (
    "fit keeps the numbers of every earlier fit, and"
    " inverse_transform_column uses the first numbers it finds for a column"
    " (standardize, min_max, median_normalize, population_standardize,"
    " population_median_normalize, in this order) whatever list the column"
    " is in now."
)
# The order in which `inverse_transform_column` looks for the numbers of a
# column.
_INVERSE_LOOKUP_ORDER = (
    "standardize",
    "min_max",
    "median_normalize",
    "population_standardize",
    "population_median_normalize",
)


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_KEPT_NUMBERS_REASON
)
@pytest.mark.parametrize("column", _COLUMNS)
@pytest.mark.parametrize("strategy", helpers.SCALING_STRATEGIES)
def test_column_taken_off_its_list_before_the_second_fit_is_left_alone(
    strategy, column
):
  """The pipeline takes a saturated treatment off the lists before its fit."""
  first, second = _fit_frame(), _second_fit_frame()
  scaler = _scaler_for(strategy, column).fit(first)
  setattr(scaler, f"{strategy}_cols", [])

  scaled = scaler.fit_transform(second)
  restored = scaler.inverse_transform_column(scaled, column)

  _assert_close(scaled[column], second[column])
  _assert_close(restored, second[column])


def _moved_column_cases():
  """Every ordered pair of strategies; the pairs that currently fail."""
  cases = []
  for old in helpers.SCALING_STRATEGIES:
    for new in helpers.SCALING_STRATEGIES:
      if old == new:
        continue
      for column in _COLUMNS:
        marks = ()
        # The numbers of `old` are found before those of `new`.
        if _INVERSE_LOOKUP_ORDER.index(old) < _INVERSE_LOOKUP_ORDER.index(new):
          marks = pytest.mark.xfail(
              strict=True, raises=AssertionError, reason=_KEPT_NUMBERS_REASON
          )
        cases.append(
            pytest.param(
                old, new, column, marks=marks, id=f"{old}-to-{new}-{column}"
            )
        )
  return cases


@pytest.mark.parametrize(("old", "new", "column"), _moved_column_cases())
def test_column_moved_to_another_list_before_the_second_fit(old, new, column):
  """The numbers of the former strategy are not used any more."""
  first, second = _fit_frame(), _second_fit_frame()
  scaler = _scaler_for(old, column).fit(first)
  setattr(scaler, f"{old}_cols", [])
  setattr(scaler, f"{new}_cols", [column])

  scaled = scaler.fit_transform(second)
  restored = scaler.inverse_transform_column(scaled, column)

  _assert_close(scaled[column], _by_hand(new, second, second, column))
  _assert_close(restored, second[column])


def test_column_in_no_list_is_left_alone():
  """`transform` and `inverse_transform_column` pass the column through."""
  df = _fit_frame().assign(other=[5.0, -1.0, 0.0, 12.0, 7.0, 3.0])
  scaler = scalers.CausalDataScaler(
      standardize_cols=["spend"],
      population_median_normalize_cols=["sales"],
      population_col="pop",
  )

  scaled = scaler.fit_transform(df)
  restored = scaler.inverse_transform_column(scaled, "other")

  np.testing.assert_array_equal(scaled["other"], df["other"])
  np.testing.assert_array_equal(scaled["pop"], df["pop"])
  np.testing.assert_array_equal(restored, df["other"])


if __name__ == "__main__":
  unittest.main()
