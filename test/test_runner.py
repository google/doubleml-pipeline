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
from unittest import mock

import numpy as np
import pandas as pd

from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.preprocessing import scalers


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

  @mock.patch("doubleml.DoubleMLPLR")
  def test_multi_repetition_aggregation(self, mock_plr_class):
    """Tests multi-repetition behavior in DoubleMLPipeline."""
    n_obs = 10
    n_rep = 3

    mock_plr = mock.MagicMock()
    mock_plr.predictions = {
        "ml_l": np.random.rand(n_obs, n_rep),
        "ml_m": np.random.rand(n_obs, n_rep),
    }
    mock_plr.models = {"ml_l": {"T": []}, "ml_m": {"T": []}}
    mock_plr.n_obs = n_obs
    mock_plr.n_folds = 5
    mock_plr.n_rep = n_rep

    mock_cate = mock.MagicMock()
    mock_cate.confint.return_value = pd.DataFrame({
        "effect": np.zeros(n_obs),
        "2.5 %": np.zeros(n_obs),
        "97.5 %": np.zeros(n_obs),
    })
    mock_cate.summary = pd.DataFrame({"coef": [1.0]}, index=["Intercept"])
    mock_plr.cate.return_value = mock_cate

    mock_plr_class.return_value = mock_plr

    pipeline = runner.DoubleMLPipeline(
        y_col="Y",
        d_col="T",
        x_cols=["X1"],
        ml_y_model="lgbm",
        ml_t_model="lgbm",
        n_folds=5,
        n_rep=n_rep,
    )

    df_orig = pd.DataFrame({
        "Y": np.random.rand(n_obs),
        "T": np.random.rand(n_obs),
        "X1": np.random.rand(n_obs),
    })
    scaler = scalers.CausalDataScaler()

    results = pipeline.run(df_orig, scaler=scaler)

    df_res = results["df_1a"]
    self.assertEqual(len(df_res), n_obs)
    self.assertIn("Y_pred", df_res.columns)
    self.assertIn("T_pred", df_res.columns)


if __name__ == "__main__":
  unittest.main()
