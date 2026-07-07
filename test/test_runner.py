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

"""Tests for runner in the doubleml_pipeline evaluation module."""

import unittest

from doubleml_pipeline.evaluation import runner


class TestRunner(unittest.TestCase):
  """Test suite for evaluation runner."""

  def test_initialization(self):
    """Tests the initialization of DoubleMLPipeline."""
    pipeline = runner.DoubleMLPipeline(
        y_col="Y",
        d_col="T",
        x_cols=["X1", "X2"],
        ml_y_model="lgbm",
        ml_t_model="lgbm",
        n_folds=3,
    )
    self.assertEqual(pipeline.y_col, "Y")
    self.assertEqual(pipeline.d_col, "T")
    self.assertEqual(pipeline.x_cols, ["X1", "X2"])
    self.assertEqual(pipeline.ml_y_name, "lgbm")
    self.assertEqual(pipeline.ml_t_name, "lgbm")
    self.assertEqual(pipeline.n_folds, 3)


if __name__ == "__main__":
  unittest.main()
