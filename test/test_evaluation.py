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

"""Tests for the small ``evaluation`` helpers and the step-output writers.

* ``metrics`` (SMAPE, ground-truth metrics), ``selection``
  (``shortlist_top_models``) and ``runner.DoubleMLPipeline`` (initialization,
  multi-repetition aggregation with a mocked ``DoubleMLPLR``).
* ``utils.io.save_step1_outputs`` / ``save_step2_outputs``: the CSVs written
  per exploration run under the default and the detailed ``process_log``.

``alpha_selection`` and ``sensitivity`` have their own files
(test_alpha_selection.py and test_sensitivity.py).
"""

import os
import unittest
from unittest import mock

from doubleml_pipeline.evaluation import metrics
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.evaluation import selection
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.utils import io
import numpy as np
import pandas as pd


class MetricsTest(unittest.TestCase):
  """SMAPE and the ground-truth metrics."""

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


class SelectionTest(unittest.TestCase):
  """The Phase 2 shortlist."""

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


class RunnerTest(unittest.TestCase):
  """``runner.DoubleMLPipeline``."""

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


class StepOutputWriterTest(unittest.TestCase):
  """``save_step1_outputs`` / ``save_step2_outputs``."""

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step1_outputs_default_processlog(
      self, mock_to_csv, mock_makedirs
  ):
    """Tests save_step1_outputs default behavior (lightweight process_log)."""
    df = pd.DataFrame({"a": [1, 2]})
    io.save_step1_outputs(
        df, df, df, "test_model", "/tmp/out"
    )
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 2)
    calls = [
        mock.call(
            os.path.join("/tmp/out", "1b_geo_item_test_model.csv"), index=False
        ),
        mock.call(
            os.path.join("/tmp/out", "1c_metrics_test_model.csv"), index=False
        ),
    ]
    mock_to_csv.assert_has_calls(calls, any_order=True)

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step1_outputs_custom_processlog(
      self, mock_to_csv, mock_makedirs
  ):
    """Tests save_step1_outputs with custom process_log."""
    df = pd.DataFrame({"a": [1, 2]})
    io.save_step1_outputs(
        df, df, df, "test_model", "/tmp/out", process_log="detailed"
    )
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 3)
    calls = [
        mock.call(
            os.path.join("/tmp/out", "1a_full_df_test_model.csv"), index=False
        ),
        mock.call(
            os.path.join("/tmp/out", "1b_geo_item_test_model.csv"), index=False
        ),
        mock.call(
            os.path.join("/tmp/out", "1c_metrics_test_model.csv"), index=False
        ),
    ]
    mock_to_csv.assert_has_calls(calls, any_order=True)

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step2_outputs(self, mock_to_csv, mock_makedirs):
    """Tests save_step2_outputs."""
    df = pd.DataFrame({"a": [1, 2]})
    io.save_step2_outputs(df, "/tmp/out")
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 1)


if __name__ == "__main__":
  unittest.main()
