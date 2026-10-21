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

import unittest

from doubleml_pipeline.preprocessing import features
from doubleml_pipeline.preprocessing import scalers
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


if __name__ == "__main__":
  unittest.main()
