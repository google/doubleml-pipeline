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

"""Tests for optimization module."""

import os
import unittest
from unittest import mock

from doubleml_pipeline import optimization
import pandas as pd


class TestOptimization(unittest.TestCase):
  """Test suite for optimization."""

  @mock.patch("matplotlib.pyplot.savefig")
  def test_optimize(self, mock_savefig):
    """Tests optimization."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "Geo": ["A", "A", "A"],
        "Item": ["X", "X", "X"],
        "Sales": [100, 200, 300],
        "Spend": [10, 20, 30],
        "estimated_incremental_KPI_Spend": [15, 25, 45],
    })

    optimization.optimize(
        df_1a_consolidated=df,
        treatment="Spend",
        treatment_amt="Spend",
        periods=("2020-01-01", "2020-01-02"),
        threshold_roi=1.0,
        out_dir="/tmp",
    )

    self.assertTrue(
        os.path.exists("/tmp/007_optimisation/optimization_metrics.csv")
    )
    mock_savefig.assert_called()


if __name__ == "__main__":
  unittest.main()
