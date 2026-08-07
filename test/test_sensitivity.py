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

"""Tests for sensitivity in the doubleml_pipeline evaluation module."""

import unittest
from unittest import mock

import pandas as pd

from doubleml_pipeline.evaluation import sensitivity


class TestSensitivity(unittest.TestCase):
  """Test suite for sensitivity analysis."""

  def test_perform_gate_and_sensitivity_analysis(self):
    """Tests sensitivity analysis function."""
    df_orig = pd.DataFrame({"gate1": ["A", "B", "A"], "gate2": ["X", "Y", "X"]})

    mock_model = mock.Mock()
    mock_model.gate.return_value = mock.Mock(summary="gate_summary")
    mock_model.sensitivity_elements = "sensitivity_elements"
    mock_model.sensitivity_benchmark.return_value = pd.DataFrame(
        {"cf_y": [0.1], "cf_d": [0.2]}
    )

    res = sensitivity.perform_gate_and_sensitivity_analysis(
        mock_model,
        df_orig,
        gate_cols=["gate1", "gate2"],
        benchmark_covariates=["cov1"],
    )

    self.assertEqual(res["gate_summary"], "gate_summary")
    self.assertEqual(res["sensitivity_elements"], "sensitivity_elements")
    self.assertIn("cov1", res["benchmarks"])
    self.assertEqual(res["benchmarks"]["cov1"]["x"], 0.2)
    self.assertEqual(res["benchmarks"]["cov1"]["y"], 0.1)


if __name__ == "__main__":
  unittest.main()
