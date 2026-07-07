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

"""Tests for selection in the doubleml_pipeline evaluation module."""

import unittest

from doubleml_pipeline.evaluation import selection
import pandas as pd


class TestSelection(unittest.TestCase):
  """Test suite for model selection."""

  def test_shortlist_top_models(self):
    """Tests the shortlist_top_models function."""
    df = pd.DataFrame({
        "model": ["m1", "m2", "m3"],
        "nuisance_models_rmse_Y_model_evaluated_by_AutoML": [0.1, 0.2, 0.3],
        "nuisance_models_rmse_T_model_evaluated_by_AutoML": [0.2, 0.1, 0.3],
    })
    result = selection.shortlist_top_models(df, top_n=2)
    self.assertEqual(len(result), 3)
    self.assertEqual(result["shortlisted"].sum(), 2)

    # Check that it picked the best combined errors (m1 and m2 have lower error
    # than m3)
    shortlisted_models = result[result["shortlisted"] == 1]["model"].tolist()
    self.assertIn("m1", shortlisted_models)
    self.assertIn("m2", shortlisted_models)
    self.assertNotIn("m3", shortlisted_models)


if __name__ == "__main__":
  unittest.main()
