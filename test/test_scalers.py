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

"""Tests for scalers in the doubleml_pipeline preprocessing module."""

import unittest

from doubleml_pipeline.preprocessing import scalers
import pandas as pd


class TestScalers(unittest.TestCase):
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


if __name__ == "__main__":
  unittest.main()
