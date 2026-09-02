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

"""Tests for metrics calculation module."""

import unittest

from doubleml_pipeline.evaluation import metrics
import pandas as pd


class TestMetrics(unittest.TestCase):
  """Test suite for metrics calculation functionality."""

  def test_calculate_smape(self):
    y_true = pd.Series([10.0, 20.0, 0.0, 100.0])
    y_pred = pd.Series([12.0, 18.0, 0.0, 100.0])

    smape = metrics.calculate_smape(y_true, y_pred)

    self.assertIsInstance(smape, float)
    self.assertGreaterEqual(smape, 0.0)

  def test_calculate_ground_truth_metrics(self):
    """Tests the calculate_ground_truth_metrics function."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 29.0])

    result_metrics = metrics.calculate_ground_truth_metrics(
        y_true=y_true, y_pred=y_pred, total_input=100.0
    )

    self.assertIn("SMAPE", result_metrics)
    self.assertIn("R_squared", result_metrics)
    self.assertIn("MAE", result_metrics)
    self.assertIn("RMSE", result_metrics)
    self.assertIn("true_total_roi", result_metrics)

    # sum(y_true)=60, input=100 -> 60/100
    self.assertEqual(result_metrics["true_total_roi"], 0.6)


if __name__ == "__main__":
  unittest.main()
