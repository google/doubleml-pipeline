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

"""Tests for FLAML DML model in the doubleml_pipeline models module."""

import unittest

from doubleml_pipeline.models.flaml_dml import FlamlRegressorDoubleML


class TestFlamlDML(unittest.TestCase):
  """Test suite for FlamlDML."""

  def test_initialization(self):
    """Tests FlamlDML initialization."""
    model = FlamlRegressorDoubleML(
        time=10,
        estimator_list=["lgbm"],
        metric="rmse",
        random_state=42,
    )
    self.assertEqual(model.time, 10)
    self.assertEqual(model.estimator_list, ["lgbm"])
    self.assertEqual(model.metric, "rmse")
    self.assertEqual(model.random_state, 42)


if __name__ == "__main__":
  unittest.main()
