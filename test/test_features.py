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

"""Tests for doubleml_pipeline.preprocessing.features."""

import unittest

import pandas as pd
import pytest

from doubleml_pipeline.preprocessing import features


class TestFeatures(unittest.TestCase):
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


if __name__ == "__main__":
  unittest.main()
