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

"""Tests for workflow module."""

import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from doubleml_pipeline import workflow


class TestWorkflow(unittest.TestCase):
  """Test suite for workflow."""

  # pylint: disable=protected-access

  def setUp(self):
    """Sets up temporary directory and basic orchestrator for tests."""
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.orchestrator = workflow.CausalWorkflowOrchestrator(
        y_col="Y",
        d_cols=["T1"],
        x_cols_list=[["X1", "X2"]],
        ml_models_y=["lgbm"],
        ml_models_t=["lgbm"],
        n_folds_list=[3],
        output_dir=self.test_dir,
    )

    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    self.df = pd.DataFrame({
        "Y": np.random.normal(0, 1, n_samples),
        "T1": np.random.uniform(0, 1, n_samples),
        "X1": np.random.normal(0, 1, n_samples),
        "X2": np.random.normal(0, 1, n_samples),
        "sales": np.random.uniform(10, 100, n_samples),
        "date": pd.date_range(start="2020-01-01", periods=n_samples),
        "geo": np.random.choice(["A", "B", "C"], n_samples),
    })

  def tearDown(self):
    """Cleans up temporary directory after tests."""
    super().tearDown()
    shutil.rmtree(self.test_dir)

  def test_initialization(self):
    """Tests workflow initialization."""
    self.assertEqual(self.orchestrator.y_col, "Y")
    self.assertEqual(self.orchestrator.d_cols, ["T1"])
    self.assertEqual(self.orchestrator.x_cols_list, [["X1", "X2"]])
    self.assertEqual(self.orchestrator.output_dir, self.test_dir)

  def test_validate_treatments_spend(self):
    """Tests treatment validation for spend types."""
    self.orchestrator.treatment_types = {"T1": "spend"}
    self.orchestrator.treatment_spend_cols = {"T1": "T1"}

    # Should not raise an error
    self.orchestrator._validate_treatments(self.df)

  def test_validate_treatments_percentage_valid(self):
    """Tests treatment validation for valid percentage types."""
    self.orchestrator.treatment_types = {"T1": "percentage"}
    self.orchestrator.sales_col = "sales"

    # Should not raise an error
    self.orchestrator._validate_treatments(self.df)

  def test_validate_treatments_percentage_invalid_values(self):
    """Tests treatment validation for invalid percentage values."""
    self.orchestrator.treatment_types = {"T1": "percentage"}
    self.orchestrator.sales_col = "sales"

    invalid_df = self.df.copy()
    invalid_df.loc[0, "T1"] = 1.5  # Outside [0.0, 1.0]

    with self.assertRaisesRegex(ValueError, "contains values outside"):
      self.orchestrator._validate_treatments(invalid_df)

  def test_validate_treatments_missing_sales_col(self):
    """Tests treatment validation missing sales column for percentage type."""
    self.orchestrator.treatment_types = {"T1": "percentage"}
    self.orchestrator.sales_col = None

    with self.assertRaisesRegex(ValueError, "valid 'sales_col'"):
      self.orchestrator._validate_treatments(self.df)

  def test_format_cate_equation(self):
    """Tests formatting of CATE equation from coefficients."""
    coef_dict = {
        "Intercept": 0.5,
        "bs(X1)": -1.2,
        "bs(X2)[0]": 0.00005,
    }
    equation = self.orchestrator.format_cate_equation(coef_dict)

    self.assertIn("0.5000", equation)
    self.assertIn("-1.2000 * Spline_(X1)", equation)
    self.assertIn("+5.00e-05 * Spline_0(X2)", equation)

  def test_format_cate_equation_empty(self):
    """Tests CATE equation formatting for near-zero coefficients."""
    coef_dict = {
        "Intercept": 1e-11,
        "X1": 1e-12,
    }
    equation = self.orchestrator.format_cate_equation(coef_dict)
    self.assertEqual(equation, "Constant Effect")

  def test_extract_priors_roi_normal(self):
    """Tests extraction of Normal priors for ROI."""
    self.orchestrator.date_col = "date"
    self.df["estimated_incremental_KPI_T1"] = self.df[
        "T1"
    ] * 2.0 + np.random.normal(0, 0.1, len(self.df))

    priors = self.orchestrator.extract_priors(
        self.df, "T1", prior_type="roi", distribution_type="Normal"
    )

    self.assertIn("mu", priors)
    self.assertIn("sigma", priors)
    self.assertAlmostEqual(priors["mu"], 2.0, delta=0.5)

  def test_extract_priors_empty(self):
    """Tests prior extraction when group_cols are missing."""
    priors = self.orchestrator.extract_priors(
        self.df, "T2", prior_type="roi", distribution_type="Normal"
    )
    self.assertEqual(priors, {})

  @mock.patch(
      ""
      "doubleml_pipeline.workflow.runner.DoubleMLPipeline"
  )
  @mock.patch(
      ""
      "doubleml_pipeline.workflow.metrics.aggregate_geo_item_metrics"
  )
  @mock.patch(
      ""
      "doubleml_pipeline.workflow.io.save_step1_outputs"
  )
  def test_run_single_exploration(self, mock_save, mock_agg, mock_pipeline):
    """Tests a single exploration run."""
    # Setup mock returns
    mock_instance = mock.MagicMock()
    mock_pipeline.return_value = mock_instance

    mock_res = {
        "df_1a": pd.DataFrame(),
        "model_name": "mock_model",
        "metrics_1c": {"key": "value"},
    }
    mock_instance.run.return_value = mock_res

    mock_df_1b = pd.DataFrame(columns=["model"])
    mock_agg.return_value = mock_df_1b

    # Mock scaler
    mock_scaler = mock.MagicMock()

    res = self.orchestrator._run_single_exploration(
        ml_y="lgbm",
        ml_t="lgbm",
        n_folds=3,
        d_col="T1",
        x_cols=["X1"],
        cate_cols=[],
        cate_structure=None,
        gt_col=None,
        df=self.df,
        scaler=mock_scaler,
    )

    self.assertEqual(res, {"key": "value"})
    mock_pipeline.assert_called_once()
    mock_instance.run.assert_called_once()
    mock_agg.assert_called_once()
    mock_save.assert_called_once()


if __name__ == "__main__":
  unittest.main()
