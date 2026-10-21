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

"""Unit tests of ``workflow.CausalWorkflowOrchestrator``.

* Initialization, treatment validation, CATE equation formatting, prior
  extraction and a single (mocked) exploration run.
* The ``saturation_config`` schema: the legacy defaults, the derived
  ``simple_optimization``, the default family resolution, the accepted keys of
  every block and the rejection of unknown / retired keys and out-of-range
  values.
* The optimization settings: ``opt_threshold_roi``, ``budget_mode`` /
  ``budget_reallocate``, ``cap_quantile`` / ``reallocate_within``, ``opt_geos``
  / ``opt_items``, the treatment-lag (no carryover) warning and the fact that
  an explicit ``simple_optimization=False`` is never rewritten at
  initialization.
* The ``hparam_policy`` argument, the worker / thread allocation of the
  parallel plan and the joblib parallel section.
* ``_model_group_label``: FLAML learner names that contain an underscore
  (``extra_tree``, ``xgb_limitdepth``) must not collide.
* The Phase 0 progress display (the eight display items, their order and
  formats) and Diag6 (unit consistency of T_ref at the Phase 1 hand-over; the
  mismatch case stops ``run_full_pipeline`` before Phase 1 starts).

Nothing here runs a complete pipeline: the full ``run_full_pipeline`` guards
live in test_workflow_e2e.py, the spec acceptance tests in test_acceptance.py
and the hyperparameter-policy tests in test_hparam_policy.py.
"""

import contextlib
import copy
import io
import os
import re
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import warnings

from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import runner
from doubleml_pipeline.models import saturation
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.test import helpers
import numpy as np
import pandas as pd


class WorkflowTest(unittest.TestCase):
  """Initialization, validation, formatting and one mocked run."""

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
    self.assertTrue(self.orchestrator.generate_html)
    self.assertTrue(self.orchestrator.generate_agent)

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

  def test_validate_columns_passes_when_all_present(self):
    """All required columns present and optional ones unset: no error."""
    self.orchestrator._validate_columns(self.df)

  def test_validate_columns_missing_y_col(self):
    """A missing outcome column is reported under y_col."""
    self.orchestrator.y_col = "Y_typo"
    with self.assertRaisesRegex(ValueError, r"y_col: 'Y_typo'"):
      self.orchestrator._validate_columns(self.df)

  def test_validate_columns_missing_d_col(self):
    """A missing treatment column is reported under d_cols."""
    self.orchestrator.d_cols = ["T1", "T_typo"]
    with self.assertRaisesRegex(ValueError, r"d_cols: 'T_typo'"):
      self.orchestrator._validate_columns(self.df)

  def test_validate_columns_reports_every_missing_column_at_once(self):
    """Typos across several parameters surface in ONE message."""
    self.orchestrator.x_cols_list = [["X1", "X_bad"]]
    self.orchestrator.covariates_for_cate_list = [["C_bad"]]
    self.orchestrator.geo_col = "geo_bad"
    with self.assertRaises(ValueError) as ctx:
      self.orchestrator._validate_columns(self.df)
    message = str(ctx.exception)
    self.assertIn("x_cols_list: 'X_bad'", message)
    self.assertIn("covariates_for_cate_list: 'C_bad'", message)
    self.assertIn("geo_col: 'geo_bad'", message)
    # Present columns are not reported as missing.
    reported = message.split("\nAvailable columns")[0]
    self.assertNotIn("'X1'", reported)
    self.assertNotIn("y_col", reported)
    self.assertNotIn("d_cols", reported)
    # The available columns are listed to help the user.
    self.assertIn("Available columns (7)", message)

  def test_validate_columns_did_you_mean_hint(self):
    """A near miss (here a case difference) gets a did-you-mean hint."""
    self.orchestrator.geo_col = "Geo"
    with self.assertRaisesRegex(
        ValueError, r"geo_col: 'Geo' \(did you mean 'geo'\?\)"
    ):
      self.orchestrator._validate_columns(self.df)

  def test_validate_columns_optional_params_unset_are_skipped(self):
    """Unset optional columns are not required to exist."""
    optional = (
        "date_col",
        "geo_col",
        "item_col",
        "geo_name_col",
        "item_name_col",
        "sales_col",
    )
    for name in optional:
      self.assertIsNone(getattr(self.orchestrator, name))
    self.orchestrator._validate_columns(self.df)

  def test_validate_columns_optional_param_set_but_missing_raises(self):
    """An optional column that IS set must exist: no silent fallback."""
    optional = (
        "date_col",
        "geo_col",
        "item_col",
        "geo_name_col",
        "item_name_col",
        "sales_col",
    )
    for name in optional:
      with self.subTest(name=name):
        setattr(self.orchestrator, name, "nope")
        with self.assertRaisesRegex(ValueError, rf"{name}: 'nope'"):
          self.orchestrator._validate_columns(self.df)
        setattr(self.orchestrator, name, None)

  def test_validate_columns_ground_truth_only_when_enabled(self):
    """ground_truth_effect_column is checked only if ground_truth_existence."""
    self.orchestrator.ground_truth_effect_column = ["gt_missing"]
    self.orchestrator.ground_truth_existence = False
    self.orchestrator._validate_columns(self.df)  # Not required: no error.
    self.orchestrator.ground_truth_existence = True
    with self.assertRaisesRegex(
        ValueError, r"ground_truth_effect_column: 'gt_missing'"
    ):
      self.orchestrator._validate_columns(self.df)

  def test_validate_columns_opt_other_cost_variables(self):
    """Cost variables that do not exist are reported, present ones are not."""
    self.orchestrator.opt_other_cost_variables = ["sales", "cost_typo"]
    with self.assertRaises(ValueError) as ctx:
      self.orchestrator._validate_columns(self.df)
    message = str(ctx.exception)
    self.assertIn("opt_other_cost_variables: 'cost_typo'", message)
    self.assertNotIn("'sales'", message.split("\nAvailable columns")[0])

  def test_run_full_pipeline_stops_on_missing_column_before_any_work(self):
    """run_full_pipeline raises before Phase 0, the scaler fit or any output."""
    self.orchestrator.x_cols_list = [["X1", "X_bad"]]
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2"])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
      with self.assertRaisesRegex(ValueError, r"x_cols_list: 'X_bad'"):
        self.orchestrator.run_full_pipeline(df=self.df, scaler=scaler)
    self.assertNotIn("Starting DML Pipeline", buffer.getvalue())
    self.assertNotIn("Phase 0", buffer.getvalue())
    # The scaler was never fitted.
    self.assertEqual(scaler.standard_params_, {})

  def test_format_cate_equation(self):
    """Tests formatting of CATE equation from coefficients."""
    coef_dict = {
        "Intercept_coef": 0.5,
        "bs(X1)_coef": -1.2,
        "bs(X2)[0]_coef": 0.00005,
        "Intercept_pvalue": 0.01,
        "Intercept_stderr": 0.05,
    }
    equation = self.orchestrator.format_cate_equation(coef_dict)

    self.assertIn("0.5000", equation)
    self.assertIn("-1.2000 * Spline_(X1)", equation)
    self.assertIn("+5.00e-05 * Spline_0(X2)", equation)
    # Ensure pvalue and stderr are ignored
    self.assertNotIn("0.01", equation)
    self.assertNotIn("0.05", equation)

  def test_format_cate_equation_empty(self):
    """Tests CATE equation formatting for near-zero coefficients."""
    coef_dict = {
        "Intercept_coef": 1e-11,
        "X1_coef": 1e-12,
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

  @mock.patch("doubleml_pipeline.workflow.runner.DoubleMLPipeline")
  @mock.patch("doubleml_pipeline.workflow.metrics.aggregate_geo_item_metrics")
  @mock.patch("doubleml_pipeline.workflow.io.save_step1_outputs")
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

    # The FULL run result is returned (it is reused as Phase 3 repetition 0),
    # not only `metrics_1c`.
    self.assertIs(res, mock_res)
    self.assertEqual(res["metrics_1c"], {"key": "value"})
    mock_pipeline.assert_called_once()
    mock_instance.run.assert_called_once()
    mock_agg.assert_called_once()
    mock_save.assert_called_once()


class SaturationConfigTest(unittest.TestCase):
  """The ``saturation_config`` rules and schema of the orchestrator."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.base_kwargs = dict(
        y_col="Y",
        d_cols=["T1"],
        x_cols_list=[["X1", "X2"]],
        ml_models_y=["lgbm"],
        ml_models_t=["lgbm"],
        n_folds_list=[3],
        output_dir=self.test_dir,
    )

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.test_dir)

  def _orch(self, saturation_config, **extra):
    kwargs = dict(self.base_kwargs, **extra)
    return workflow.CausalWorkflowOrchestrator(
        saturation_config=saturation_config, **kwargs
    )

  def test_defaults_keep_the_legacy_path(self):
    """Without a saturation block, Phase 0 must be inert."""
    orch = workflow.CausalWorkflowOrchestrator(**self.base_kwargs)
    self.assertIsNone(orch.sat_family)
    self.assertEqual(orch._alpha_hat, {})
    self.assertTrue(orch.simple_optimization)

  def test_detailed_optimization_without_saturation_raises(self):
    """The saturation-aware optimizer requires a saturation family."""
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          optimize=True,
          simple_optimization=False,
          **self.base_kwargs,
      )

  def test_detailed_optimization_with_saturation_is_accepted(self):
    """An explicit simple_optimization=False is kept (never rewritten)."""
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        simple_optimization=False,
        saturation_config={"saturation": {"family": "power"}},
        **self.base_kwargs,
    )
    self.assertEqual(orch.sat_family, "power")
    self.assertFalse(orch.simple_optimization)
    self.assertEqual(orch.budget_mode, "fixed")
    self.assertTrue(orch.budget_reallocate)

  def test_several_treatments_keep_simple_optimization_false(self):
    """Several treatments do not rewrite simple_optimization at init.

    The per-treatment fallback (and its warning) happens in Phase 4 and is
    exercised end to end by test_workflow_e2e.py.
    """
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      orch = self._orch(
          {"saturation": {"family": "exp"}},
          optimize=True,
          simple_optimization=False,
          d_cols=["T1", "T2"],
          x_cols_list=[["X1"], ["X1"]],
      )
    self.assertFalse(orch.simple_optimization)
    self.assertEqual(len(orch.d_cols), 2)
    self.assertFalse(caught)

  def test_simple_optimization_without_saturation_is_unchanged(self):
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True, simple_optimization=True, **self.base_kwargs
    )
    self.assertIsNone(orch.sat_family)
    self.assertTrue(orch.optimize)

  def test_unsupported_family_raises(self):
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"saturation": {"family": "hill"}},
          **self.base_kwargs,
      )

  def test_family_is_the_only_required_key(self):
    """Enabling saturation must be a ONE-key change; the rest defaults."""
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "exp"}},
        **self.base_kwargs,
    )
    self.assertEqual(orch.sat_family, "exp")
    # Everything else falls back to its documented default.
    self.assertEqual(orch.final_fit, "d_inf")
    self.assertEqual(orch.budget_increment, 0.0)
    self.assertEqual(orch.alpha_selection_cfg, {})
    self.assertEqual(orch.saturation_cfg, {"family": "exp"})

  def test_simple_optimization_is_derived_from_the_family(self):
    """The caller should not have to remember a second flag."""
    with_sat = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "exp"}},
        **self.base_kwargs,
    )
    self.assertFalse(with_sat.simple_optimization)

    without_sat = workflow.CausalWorkflowOrchestrator(
        optimize=True, **self.base_kwargs
    )
    self.assertTrue(without_sat.simple_optimization)

  def test_explicit_simple_optimization_still_wins(self):
    """An explicit value overrides the derived one, but warns."""
    with self.assertWarns(UserWarning):
      orch = workflow.CausalWorkflowOrchestrator(
          optimize=True,
          simple_optimization=True,
          saturation_config={"saturation": {"family": "exp"}},
          **self.base_kwargs,
      )
    self.assertTrue(orch.simple_optimization)

  def test_full_config_is_read_from_the_single_dict(self):
    """The whole setup is passed as ONE config object."""
    orch = workflow.CausalWorkflowOrchestrator(
        saturation_config={
            "saturation": {
                "family": "exp",
                "grid": {"min": 0.1, "max": 2.0, "num": 7, "scale": "log"},
                "basis_dim_T": 9,
                "ridge": 1e-3,
            },
            "alpha_selection": {
                "sel_fraction": 0.3,
                "n_seeds": 4,
                "k_folds": 3,
                "refine": "none",
                "flatness_ratio": 10.0,
                "t_ref_tolerance": 1.0e-9,
                "fallback_half_saturation_T": None,
                "time_budget": 7,
            },
            "final_fit": {"data": "full_sample"},
            "optimization": {"budget_increment": 0.25, "budget_mode": "fixed",
                             "budget_reallocate": True},
        },
        **self.base_kwargs,
    )
    self.assertEqual(orch.sat_family, "exp")
    self.assertEqual(orch.saturation_cfg["basis_dim_T"], 9)
    self.assertEqual(orch.alpha_selection_cfg["n_seeds"], 4)
    self.assertEqual(orch.final_fit, "full_sample")
    self.assertEqual(orch.budget_increment, 0.25)
    self.assertEqual(orch.budget_mode, "fixed")
    self.assertTrue(orch.budget_reallocate)

  def test_unknown_config_block_raises(self):
    """A typo must fail loudly, not silently skip Phase 0."""
    with self.assertRaises(ValueError) as ctx:
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"saturaton": {"family": "exp"}}, **self.base_kwargs
      )
    self.assertIn("saturaton", str(ctx.exception))

  def test_unknown_config_key_raises(self):
    for block, bad in (
        ("saturation", {"family": "exp", "basis_dim": 9}),
        ("alpha_selection", {"nseeds": 4}),
        ("final_fit", {"dataset": "d_inf"}),
        ("optimization", {"budget": 0.1}),
    ):
      with self.assertRaises(ValueError, msg=block):
        workflow.CausalWorkflowOrchestrator(
            saturation_config={block: bad}, **self.base_kwargs
        )

  def test_unknown_grid_key_raises(self):
    config = {"saturation": {"family": "exp", "grid": {"lo": 0.1}}}
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config=config,
          **self.base_kwargs,
      )

  def test_non_dict_config_block_raises(self):
    with self.assertRaises(TypeError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"saturation": "exp"}, **self.base_kwargs
      )

  def test_config_does_not_mutate_the_callers_dict(self):
    """The caller's config object must never be modified in place."""
    cfg = {"saturation": {"family": "exp"}}
    snapshot = copy.deepcopy(cfg)
    workflow.CausalWorkflowOrchestrator(
        saturation_config=cfg, **self.base_kwargs
    )
    self.assertEqual(cfg, snapshot)

  def test_invalid_final_fit_raises(self):
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"final_fit": {"data": "d_sel"}}, **self.base_kwargs
      )

  def test_budget_increment_below_minus_one_raises(self):
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"optimization": {"budget_increment": -1.5}},
          **self.base_kwargs,
      )

  def test_family_resolution(self):
    self.assertIsNone(self._orch(None).sat_family)
    self.assertEqual(self._orch({"saturation": {}}).sat_family, "tanh")
    self.assertEqual(
        self._orch({"saturation": {"grid": {"num": 9}}}).sat_family, "tanh"
    )
    self.assertIsNone(self._orch({"saturation": {"family": None}}).sat_family)
    self.assertEqual(
        self._orch({"saturation": {"family": "exp"}}).sat_family, "exp"
    )
    self.assertEqual(saturation.DEFAULT_FAMILY, "tanh")
    self.assertIn("tanh", saturation.SUPPORTED_FAMILIES)

  def test_new_config_keys_are_accepted(self):
    """All new saturation and alpha_selection config keys pass validation."""
    orch = self._orch({
        "saturation": {
            "family": "tanh",
            "grid": {"mode": "auto", "c_min": 0.25, "c_max": 4.0, "num": 25,
                     "max_extensions": 2},
            "basis_dim_T": "auto",
            "basis_tol": 1e-3,
            "basis_dim_range": [6, 20],
        },
        "alpha_selection": {
            "linear_test_k": 2.0,
            "flatness_ratio": 10.0,
            "t_ref_tolerance": 1.0e-9,
            "fallback_half_saturation_T": None,
            "output_dir": "step0_saturation",
            "learners": {"learner_selection": False,
                         "small_sample_threshold": 2000},
        },
    })
    self.assertEqual(orch.sat_family, "tanh")

  def test_bad_new_keys_are_rejected(self):
    with self.assertRaises(ValueError):
      self._orch({"saturation": {"grid": {"c_mni": 0.2}}})
    with self.assertRaises(ValueError):
      self._orch({"alpha_selection": {"learners": {"y_modle": {}}}})
    with self.assertRaises(ValueError):
      self._orch({"saturation": {"grid": {"mode": "optuna"}}})

  def test_alpha_selection_keys_accepted(self):
    """The alpha_selection keys pass config validation."""
    orch = self._orch({
        "saturation": {"family": "tanh"},
        "alpha_selection": {
            "flatness_ratio": 10.0,
            "t_ref_tolerance": 1.0e-9,
            "fallback_half_saturation_T": None,
            "linear_test_k": 1.0,
        },
    })
    self.assertEqual(orch.sat_family, "tanh")
    self._orch({
        "saturation": {"family": "tanh"},
        "alpha_selection": {"fallback_half_saturation_T": 12.5},
    })

  def test_retired_keys_rejected(self):
    """Abolished alpha_selection keys are rejected with ValueError."""
    for key in ("se_" + "type", "flatness_" + "threshold",
                "dispersion_" + "threshold", "resid_ratio_" + "threshold",
                "fallback_" + "alpha"):
      with self.assertRaises(ValueError, msg=key):
        self._orch({"alpha_selection": {key: 1}})

  def test_out_of_range_values_rejected(self):
    """Out-of-range values for the new keys are rejected with ValueError."""
    with self.assertRaises(ValueError):
      self._orch({"alpha_selection": {"flatness_ratio": 0.5}})
    with self.assertRaises(ValueError):
      self._orch({"alpha_selection": {"t_ref_tolerance": 0.0}})
    with self.assertRaises(ValueError):
      self._orch({"alpha_selection": {"fallback_half_saturation_T": -1.0}})


class OptimizationSettingsTest(unittest.TestCase):
  """Validation and defaults of the optimization settings at init."""

  def _kwargs(self, **extra):
    kwargs = dict(
        y_col="Y",
        d_cols=["T1"],
        x_cols_list=[["X1"]],
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[2],
        output_dir="/tmp/unused_k17",
    )
    kwargs.update(extra)
    return kwargs

  def test_negative_threshold_raises(self):
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          opt_threshold_roi=-0.5, **self._kwargs()
      )

  def test_roi_target_needs_a_positive_threshold(self):
    """budget_mode='roi_target' rejects opt_threshold_roi=0 and accepts > 0."""
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          opt_threshold_roi=0.0,
          saturation_config={
              "saturation": {"family": "tanh"},
              "optimization": {"budget_mode": "roi_target"},
          },
          **self._kwargs(),
      )
    # Omitting opt_threshold_roi uses DEFAULT_THRESHOLD_ROI = 1.0 (> 0), so
    # roi_target works out of the box without extra parameters.
    orch_default = workflow.CausalWorkflowOrchestrator(
        saturation_config={
            "saturation": {"family": "tanh"},
            "optimization": {"budget_mode": "roi_target"},
        },
        **self._kwargs(),
    )
    self.assertEqual(orch_default.budget_mode, "roi_target")
    self.assertEqual(
        orch_default.opt_threshold_roi, workflow.DEFAULT_THRESHOLD_ROI
    )
    self.assertTrue(orch_default.budget_reallocate)

    orch = workflow.CausalWorkflowOrchestrator(
        opt_threshold_roi=1.5,
        saturation_config={
            "saturation": {"family": "tanh"},
            "optimization": {
                "budget_mode": "roi_target",
                "budget_reallocate": False,
            },
        },
        **self._kwargs(),
    )
    self.assertEqual(orch.budget_mode, "roi_target")
    self.assertEqual(orch.opt_threshold_roi, 1.5)
    self.assertFalse(orch.budget_reallocate)

  def test_cap_quantile_and_reallocate_within_settings(self):
    """Defaults (1.0 / "all"), parsing and validation of the two new keys."""
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "tanh"}},
        **self._kwargs(),
    )
    self.assertEqual(orch.cap_quantile, 1.0)
    self.assertEqual(orch.reallocate_within, "all")
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={
            "saturation": {"family": "tanh"},
            "optimization": {"cap_quantile": 0.95,
                             "reallocate_within": "Date"},
        },
        **self._kwargs(),
    )
    self.assertEqual(orch.cap_quantile, 0.95)
    self.assertEqual(orch.reallocate_within, "date")
    for bad in ({"cap_quantile": 0.0}, {"cap_quantile": 1.5},
                {"cap_quantile": "0.9"}, {"cap_quantile": True},
                {"reallocate_within": "week"}, {"cap_quantil": 0.9}):
      with self.subTest(bad=bad):
        with self.assertRaises(ValueError):
          workflow.CausalWorkflowOrchestrator(
              optimize=True,
              saturation_config={
                  "saturation": {"family": "tanh"},
                  "optimization": bad,
              },
              **self._kwargs(),
          )

  def test_treatment_lag_warning(self):
    """A <d_col>_l<p> covariate triggers the no-carryover warning in Phase 4."""
    orch = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "tanh"}},
        **self._kwargs(x_cols_list=[["X1", "T1_l1", "T1_l2", "T1_lag"]]),
    )
    self.assertEqual(
        orch.treatment_lag_covariates(), {"T1": ["T1_l1", "T1_l2"]}
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      orch._warn_treatment_lag_covariates()  # pylint: disable=protected-access
    self.assertEqual(len(caught), 1)
    self.assertIn("no carryover", str(caught[0].message))
    self.assertIn("T1_l1", str(caught[0].message))
    # Lags in covariates_for_cate count too.
    orch_cate = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "tanh"}},
        covariates_for_cate_list=[["T1_l3"]],
        **self._kwargs(x_cols_list=[["X1", "T1_l3"]]),
    )
    self.assertEqual(orch_cate.treatment_lag_covariates(), {"T1": ["T1_l3"]})
    # No lag, or the linear / simple path: silent.
    orch_plain = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "tanh"}},
        **self._kwargs(),
    )
    self.assertEqual(orch_plain.treatment_lag_covariates(), {})
    orch_linear = workflow.CausalWorkflowOrchestrator(
        **self._kwargs(x_cols_list=[["X1", "T1_l1"]])
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      orch_plain._warn_treatment_lag_covariates()  # pylint: disable=protected-access
      orch_linear._warn_treatment_lag_covariates()  # pylint: disable=protected-access
    self.assertEqual(len(caught), 0)

  def test_threshold_roi_default_is_one_for_every_path(self):
    """opt_threshold_roi defaults to 1.0 regardless of optimizer mode."""
    # Linear / simple_optimization=True path (no saturation_config).
    orch_linear = workflow.CausalWorkflowOrchestrator(**self._kwargs())
    self.assertTrue(orch_linear.simple_optimization)
    self.assertEqual(orch_linear.opt_threshold_roi, 1.0)
    self.assertTrue(orch_linear.budget_reallocate)

    # Saturation-aware path (family set, simple_optimization derived False).
    orch_sat = workflow.CausalWorkflowOrchestrator(
        optimize=True,
        saturation_config={"saturation": {"family": "tanh"}},
        **self._kwargs(),
    )
    self.assertFalse(orch_sat.simple_optimization)
    self.assertEqual(orch_sat.opt_threshold_roi, 1.0)
    self.assertTrue(orch_sat.budget_reallocate)

    # Explicit simple_optimization=True with a saturation family.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      orch_simple = workflow.CausalWorkflowOrchestrator(
          optimize=True,
          simple_optimization=True,
          saturation_config={"saturation": {"family": "tanh"}},
          **self._kwargs(),
      )
    self.assertTrue(orch_simple.simple_optimization)
    self.assertEqual(orch_simple.opt_threshold_roi, 1.0)

    # An explicit value (including 0.0 for fixed mode) is respected.
    orch_zero = workflow.CausalWorkflowOrchestrator(
        opt_threshold_roi=0.0,
        saturation_config={"saturation": {"family": "tanh"}},
        **self._kwargs(),
    )
    self.assertEqual(orch_zero.opt_threshold_roi, 0.0)

  def test_budget_reallocate_defaults_to_true(self):
    """budget_reallocate defaults to True; False is read from the block."""
    orch = workflow.CausalWorkflowOrchestrator(
        saturation_config={"saturation": {"family": "exp"}},
        **self._kwargs(),
    )
    self.assertTrue(orch.budget_reallocate)

    orch_false = workflow.CausalWorkflowOrchestrator(
        saturation_config={
            "saturation": {"family": "exp"},
            "optimization": {"budget_reallocate": False},
        },
        **self._kwargs(),
    )
    self.assertFalse(orch_false.budget_reallocate)

  def test_bad_optimization_keys_and_values(self):
    """Rejects unknown optimization keys and invalid mode/reallocate values."""
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"optimization": {"budget_mode": "greedy"}},
          **self._kwargs(),
      )
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"optimization": {"budget_reallocate": "yes"}},
          **self._kwargs(),
      )
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(
          saturation_config={"optimization": {"roi_" + "threshold": 1.0}},
          **self._kwargs(),
      )

  def test_opt_geos_and_items(self):
    """Validates opt_geos and opt_items checks and normalization."""
    with self.assertRaises(ValueError):
      workflow.CausalWorkflowOrchestrator(opt_geos=["G0"], **self._kwargs())
    with self.assertRaises(TypeError):
      workflow.CausalWorkflowOrchestrator(
          opt_geos="G0", geo_col="geo", **self._kwargs()
      )
    orch = workflow.CausalWorkflowOrchestrator(
        opt_geos=["G0"],
        opt_items=("I1",),
        geo_col="geo",
        item_col="item",
        **self._kwargs(),
    )
    self.assertEqual(orch.opt_geos, ["G0"])
    self.assertEqual(orch.opt_items, ["I1"])
    none = workflow.CausalWorkflowOrchestrator(geo_col="geo", **self._kwargs())
    self.assertIsNone(none.opt_geos)
    self.assertIsNone(none.opt_items)


class HparamPolicyArgumentTest(unittest.TestCase):
  """``hparam_policy`` defaults to "global"; invalid values raise."""

  def _kwargs(self, **extra):
    base = dict(
        y_col="Y",
        d_cols=["T"],
        x_cols_list=[["X"]],
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[3],
        output_dir="/tmp/unused",
    )
    base.update(extra)
    return base

  def test_default_is_global(self):
    orch = workflow.CausalWorkflowOrchestrator(**self._kwargs())
    self.assertEqual(orch.hparam_policy, "global")

  def test_per_fold_is_accepted(self):
    orch = workflow.CausalWorkflowOrchestrator(
        **self._kwargs(hparam_policy="per_fold")
    )
    self.assertEqual(orch.hparam_policy, "per_fold")

  def test_invalid_value_raises_at_init(self):
    for bad in ("Global", "perfold", "", None, 1):
      with self.subTest(value=bad):
        with self.assertRaises(ValueError):
          workflow.CausalWorkflowOrchestrator(
              **self._kwargs(hparam_policy=bad)
          )


class WorkersThreadsTest(unittest.TestCase):
  """``W = min(N_task, n_cores)``, ``h = n_cores // W``.

  The parallel plan ``(W, h)`` must equal the reference table for
  ``(n_cores, N_task) = (8, 4), (10, 4), (12, 4), (8, 27), (12, 27)``.
  """

  def test_table_of_section_4_2(self):
    expected = {
        (8, 4): (4, 2),
        (10, 4): (4, 2),
        (12, 4): (4, 3),
        (8, 27): (8, 1),
        (12, 27): (12, 1),
    }
    for (n_cores, n_tasks), want in expected.items():
      with self.subTest(n_cores=n_cores, n_tasks=n_tasks):
        self.assertEqual(
            workflow.compute_workers_threads(n_tasks, n_cores), want
        )

  def test_degenerate_inputs(self):
    # No tasks: one worker, all threads.
    self.assertEqual(workflow.compute_workers_threads(0, 8), (1, 8))
    # One core: one worker, one thread whatever the task count.
    self.assertEqual(workflow.compute_workers_threads(27, 1), (1, 1))
    # Tasks == cores: one thread each.
    self.assertEqual(workflow.compute_workers_threads(16, 16), (16, 1))

  def test_resolve_n_cores(self):
    with mock.patch.object(workflow.os, "cpu_count", return_value=10):
      self.assertEqual(workflow.resolve_n_cores(-1), 10)
      self.assertEqual(workflow.resolve_n_cores(None), 10)
      # A positive n_jobs is used in place of n_cores.
      self.assertEqual(workflow.resolve_n_cores(4), 4)
      # joblib's "all but one".
      self.assertEqual(workflow.resolve_n_cores(-2), 9)

  def test_orchestrator_parallel_plan_uses_n_jobs(self):
    """The orchestrator's (workers, threads) plan is derived from n_jobs."""
    orch = workflow.CausalWorkflowOrchestrator(
        y_col="Y",
        d_cols=["T"],
        x_cols_list=[["X"]],
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[3],
        output_dir="/tmp/unused",
        n_jobs=8,
    )
    self.assertEqual(orch._parallel_plan(4), (4, 2))  # pylint: disable=protected-access
    self.assertEqual(orch._parallel_plan(27), (8, 1))  # pylint: disable=protected-access


class ParallelSectionTest(unittest.TestCase):
  """The joblib configuration of a parallel section."""

  def test_default_is_loky_with_inner_max_num_threads(self):
    with mock.patch.dict(workflow.os.environ, {}, clear=False):
      workflow.os.environ.pop(workflow.PARALLEL_BACKEND_ENV, None)
      with workflow.parallel_section(4, 2):
        backend, n_jobs = workflow.joblib.parallel.get_active_backend()
        self.assertEqual(backend.__class__.__name__, "LokyBackend")
        self.assertEqual(n_jobs, 4)
        self.assertEqual(backend.inner_max_num_threads, 2)

  def test_env_var_selects_the_fork_backend_and_caps_threads(self):
    """The backend env var selects multiprocessing and caps OMP/MKL threads."""
    env = {workflow.PARALLEL_BACKEND_ENV: "multiprocessing",
           "OMP_NUM_THREADS": "7"}
    with mock.patch.dict(workflow.os.environ, env, clear=False):
      workflow.os.environ.pop("MKL_NUM_THREADS", None)
      with workflow.parallel_section(3, 2):
        backend, n_jobs = workflow.joblib.parallel.get_active_backend()
        self.assertEqual(backend.__class__.__name__, "MultiprocessingBackend")
        self.assertEqual(n_jobs, 3)
        # h is applied through the environment the forked workers inherit.
        self.assertEqual(workflow.os.environ["OMP_NUM_THREADS"], "2")
        self.assertEqual(workflow.os.environ["MKL_NUM_THREADS"], "2")
      # ... and restored afterwards.
      self.assertEqual(workflow.os.environ["OMP_NUM_THREADS"], "7")
      self.assertNotIn("MKL_NUM_THREADS", workflow.os.environ)


# FLAML's documented regression learners; see third_party/py/flaml/automl.
_SAFE_NAMES = ("lgbm", "rf", "xgboost", "catboost")


_UNDERSCORE_NAMES = ("extra_tree", "xgb_limitdepth")


def _metrics_1c(ml_y, ml_t, n_folds, rep_id=1, time_budget=15, n_cate=1,
                n_cluster_vars=2):
  """Mirrors the subset of runner.metrics_1c that the label depends on.

  Note the two distinct fold fields. `n_folds` is what the caller configured
  and is what `model_name` embeds; `number_of_folds` is DoubleML's EFFECTIVE
  count, which for cluster data is `n_folds ** n_cluster_vars`.

  Args:
    ml_y: FLAML learner name for the outcome model.
    ml_t: FLAML learner name for the treatment model.
    n_folds: Fold count as configured by the caller.
    rep_id: Repetition index.
    time_budget: FLAML time budget, in seconds.
    n_cate: Number of CATE variables.
    n_cluster_vars: Number of cluster variables, used to derive the effective
      fold count.

  Returns:
    A `metrics_1c`-shaped dict holding the `model` name string alongside the
    structured fields (`ml_y_name`, `ml_t_name`, `n_folds`,
    `number_of_folds`) that `_model_group_label` reads.
  """
  return {
      "model": (
          f"{ml_y}_{ml_t}_{n_folds}_{rep_id}_{time_budget}_{n_cate}"
      ),
      "ml_y_name": ml_y,
      "ml_t_name": ml_t,
      "n_folds": n_folds,
      "number_of_folds": n_folds**n_cluster_vars,
  }


class ModelGroupLabelTest(unittest.TestCase):
  """Guards `_model_group_label` against separator collisions.

  `evaluation.runner` builds `model_name` by joining six fields with '_':

      f'{ml_y_name}_{ml_t_name}_{n_folds}_{rep_id}_{time_budget}_{n_cate}'

  Two of FLAML's official learner names contain '_' themselves --
  `extra_tree` and `xgb_limitdepth` -- so the encoding is ambiguous and
  cannot be decoded by position. Doing so used to cause two distinct
  failures:

    1. a crash, because `int(name.split("_")[2])` received a learner name;
    2. a SILENT one, because the `<ml_y>_<ml_t>_<n_folds>` group label was
       built as `"_".join(split("_")[:3])`, so `n_folds` fell out of the
       label and runs differing only in fold count were averaged together.

  Both are now avoided by carrying the fields through as structured data on
  `metrics_1c` rather than re-parsing the string. These tests pin that.
  """

  # `_model_group_label` is module-private but is the unit under test; this
  # matches the convention used by TestWorkflow above.
  # pylint: disable=protected-access

  def test_label_is_unchanged_for_underscore_free_names(self):
    """The fix must not alter existing outputs for the common case."""
    for ml_y in _SAFE_NAMES:
      for ml_t in _SAFE_NAMES:
        m_data = _metrics_1c(ml_y, ml_t, 5)
        legacy = "_".join(m_data["model"].split("_")[:3])
        self.assertEqual(workflow._model_group_label(m_data), legacy)

  def test_label_keeps_the_full_learner_name(self):
    """A learner name containing '_' must survive into the label intact."""
    m_data = _metrics_1c("extra_tree", "rf", 5)
    self.assertEqual(workflow._model_group_label(m_data), "extra_tree_rf_5")
    # The legacy implementation truncated to the first three '_'-fields.
    self.assertEqual(
        "_".join(m_data["model"].split("_")[:3]), "extra_tree_rf"
    )

  def test_fold_counts_do_not_collide(self):
    """The silent bug: 5-fold and 10-fold must not share a group label."""
    for ml_y in _UNDERSCORE_NAMES:
      for ml_t in _SAFE_NAMES:
        label_5 = workflow._model_group_label(_metrics_1c(ml_y, ml_t, 5))
        label_10 = workflow._model_group_label(_metrics_1c(ml_y, ml_t, 10))
        self.assertNotEqual(label_5, label_10, f"{ml_y}/{ml_t}")

  def test_label_uses_configured_not_effective_fold_count(self):
    """DoubleML squares n_folds for cluster data; the label must not.

    `double_ml.py` sets `_n_folds = n_folds ** n_cluster_vars`, so a run
    configured with 3 folds and 2-way clustering reports `number_of_folds=9`.
    The label has to mirror `model_name`, which embeds the configured 3.
    """
    m_data = _metrics_1c("rf", "lgbm", 3, n_cluster_vars=2)
    self.assertEqual(m_data["number_of_folds"], 9)
    self.assertEqual(workflow._model_group_label(m_data), "rf_lgbm_3")
    self.assertEqual(
        workflow._model_group_label(m_data),
        "_".join(m_data["model"].split("_")[:3]),
    )

  def test_label_is_unique_per_learner_pair(self):
    """No two (ml_y, ml_t) pairs may share a group label."""
    seen = {}
    for ml_y in _SAFE_NAMES + _UNDERSCORE_NAMES:
      for ml_t in _SAFE_NAMES + _UNDERSCORE_NAMES:
        label = workflow._model_group_label(_metrics_1c(ml_y, ml_t, 5))
        self.assertNotIn(label, seen, f"{(ml_y, ml_t)} collides {seen}")
        seen[label] = (ml_y, ml_t)


class ShortlistFieldTest(unittest.TestCase):
  """Phase 2 reads these fields off the frame; runner must supply them."""

  def test_runner_emits_the_structured_fields(self):
    """The pipeline exposes learner names and fold count as attributes."""
    # Note the constructor keywords are ml_y_model / ml_t_model, but they are
    # stored as self.ml_y_name / self.ml_t_name, which is what metrics_1c and
    # the model_name format string both read.
    pipeline = runner.DoubleMLPipeline(
        y_col="Y",
        d_col="D",
        x_cols=["X1"],
        ml_y_model="extra_tree",
        ml_t_model="xgb_limitdepth",
        n_folds=5,
        time_budget=1,
    )
    # The names must be retrievable without parsing `model_name`.
    self.assertEqual(pipeline.ml_y_name, "extra_tree")
    self.assertEqual(pipeline.ml_t_name, "xgb_limitdepth")
    self.assertEqual(pipeline.n_folds, 5)


def make_orchestrator(out_dir: str, n_seeds: int = 2, k_folds: int = 2,
                      d_cols=("spend",), **extra_selection):
  selection = {"n_seeds": n_seeds, "k_folds": k_folds}
  selection.update(extra_selection)
  return workflow.CausalWorkflowOrchestrator(
      y_col="sales", d_cols=list(d_cols),
      x_cols_list=[["X1", "X2"] for _ in d_cols],
      ml_models_y=["rf"], ml_models_t=["rf"], n_folds_list=[2],
      output_dir=out_dir, covariates_for_cate_list=[["X1"] for _ in d_cols],
      time_budget=1, n_jobs=1, date_col="date", geo_col="geo",
      # This test patches the per-fold FLAML search: keep the previous
      # hyperparameter policy (the default is "global").
      hparam_policy="per_fold",
      item_col="item", run_sensitivity=False, generate_html=False,
      treatment_types={d: "spend" for d in d_cols},
      treatment_spend_cols={d: d for d in d_cols}, sales_col="sales",
      optimize=True,
      saturation_config={
          "saturation": {"family": "tanh"},
          "alpha_selection": selection,
      },
  )


# The Phase 0 display items 1)-8), one regular expression per line kind.
_TAG = r"\[[^\]]+\]"


ITEM_PATTERNS = {
    1: re.compile(
        r"^  ├─ Stage B: Number of ML trainings = (\d+) x 2 nuisance models x"
        r" (\d+) x (\d+)(?: \+ .*)? = (\d+)$"
    ),
    2: re.compile(
        r"^\[Phase 0 Progress\] elapsed \d+:\d\d:\d\d \| \d+/\d+ ML trainings"
        r" completed \| ETA .+$"
    ),
    3: re.compile(
        rf"^  │   {_TAG} Learners: Y-model = \S+, T-model = \S+ \(n_sel = \d+"
        r" (<|>=) small_sample_threshold = \d+\)$"
    ),
    4: re.compile(
        rf"^  │   {_TAG} Grid: mode = (auto|manual), \S+ range = \[.+\]"
        r" \(\d+ points\), T_ref = \S+, extensions applied = \d+,"
        r" grid_edge_hit = (none|upper|lower)$"
    ),
    5: re.compile(
        rf"^  │   {_TAG} Diag1( WARNING)?: J = \d+, eps\(J\) = \S+ (<|>=)"
        r" basis_tol = \S+, cond\(B\) = \S+ (<|>=) \S+, T_max = \S+"
        r" \(q\S+ of T>0; \d+ D_sel rows clipped\)$"
    ),
    6: re.compile(
        rf"^  │   {_TAG} alpha: grid-argmin alpha = \S+( \(.+\))?,"
        r" (Brent-refined alpha_hat|alpha_hat \(no Brent refinement\)) = \S+"
        r"( \(.+\))?, L_bar\(alpha_hat\) = \S+$"
    ),
    "7_diag2": re.compile(
        rf"^  │   {_TAG} Diag2( WARNING| INFO)?: 1-SE flatness"
        r" (fired|not fired), .+ (>|<=) .+ \(A_1SE: \d+ of \d+ grid points\)$"
    ),
    "7_diag3": re.compile(
        rf"^  │   {_TAG} Diag3: r_T = sd\(T~\)/sd\(T/T_ref\) = .+$"
    ),
    "7_diag4": re.compile(
        rf"^  │   {_TAG} Diag4: shape quantiles at alpha_hat: .+$"
    ),
    "7_diag5": re.compile(
        rf"^  │   {_TAG} Diag5: saturation level at T_ref kappa\(alpha_hat\) ="
        r" f\(T_ref; alpha_hat\) = .+$"
    ),
    "7_decision": re.compile(
        rf"^  │   {_TAG} 1-SE decision: decision = (saturation|linear)"
        r" \(decision_reason = (1se_pass|1se_fail|grid_edge_upper"
        r"|basis_ill_conditioned)\),"
        r" d_bar = \S+, se = \S+ \((2-way \S+ x \S+; 1-way \S+ = \S+"
        r"|1-way \S+|iid)\), crit = \S+ \(k = \S+, (normal|t with df = \d+)\)"
        r"( \[1-way se would decide: (saturation|linear)\])?"
        r" -> selected_family = .+$"
    ),
    8: re.compile(r"^  └─ Phase 0 complete\. .+$"),
}


def parse_stage_b(log: str):
  """Returns the Stage B lines (from item 1 onwards) of a Phase 0 log."""
  lines = [ln.rstrip("\n") for ln in log.splitlines() if ln.strip()]
  starts = [i for i, ln in enumerate(lines) if ITEM_PATTERNS[1].match(ln)]
  if len(starts) != 1:
    raise AssertionError(f"item 1) must appear exactly once; log:\n{log}")
  return lines[starts[0]:]


def check_item_order(test, stage_b_lines, n_treatments):
  """Asserts the display order: 1, 2*, (3,4,5,6,7...) x treatments, 8."""
  idx = 0
  m = ITEM_PATTERNS[1].match(stage_b_lines[idx])
  test.assertIsNotNone(m, stage_b_lines[idx])
  idx += 1
  while ITEM_PATTERNS[2].match(stage_b_lines[idx]):
    idx += 1
  for _ in range(n_treatments):
    for item in (3, 4, 5, 6):
      test.assertRegex(stage_b_lines[idx], ITEM_PATTERNS[item])
      idx += 1
    diag2 = stage_b_lines[idx]
    test.assertRegex(diag2, ITEM_PATTERNS["7_diag2"])
    idx += 1
    if "fired," in diag2 and "not fired" not in diag2:
      test.assertRegex(stage_b_lines[idx], ITEM_PATTERNS["7_diag3"])
      idx += 1
      test.assertRegex(stage_b_lines[idx], ITEM_PATTERNS["7_diag4"])
      idx += 1
    if ITEM_PATTERNS["7_diag5"].match(stage_b_lines[idx]):
      idx += 1
    test.assertRegex(stage_b_lines[idx], ITEM_PATTERNS["7_decision"])
    idx += 1
  test.assertRegex(stage_b_lines[idx], ITEM_PATTERNS[8])
  idx += 1
  test.assertEqual(
      idx, len(stage_b_lines),
      "lines after item 8):\n" + "\n".join(stage_b_lines[idx:]),
  )
  return m


class ProgressDisplayTest(unittest.TestCase):
  """The Phase 0 progress display: item order, counts and line formats.

  A Phase 0 run prints, after the (unchanged) Stage A block, exactly the
  eight display items in their order and nothing else; no internal marker
  appears in the console output; the ML-training count equals
  ``treatments x 2 x S x K``.
  """

  def _run_phase0(self, n_seeds, k_folds, d_cols=("spend",)):
    """Runs Phase 0 on a synthetic panel; returns (stdout log, orchestrator)."""
    df = helpers.saturating_panel(n_weeks=60, theta=6.0)
    if len(d_cols) > 1:
      rng = np.random.default_rng(1)
      for extra in d_cols[1:]:
        df[extra] = df["spend"] * rng.uniform(0.5, 1.5, len(df))
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    with tempfile.TemporaryDirectory() as out_dir:
      orch = make_orchestrator(out_dir, n_seeds, k_folds, d_cols)
      buffer = io.StringIO()
      with warnings.catch_warnings(), contextlib.redirect_stdout(buffer):
        warnings.simplefilter("ignore")
        orch._run_phase0(df, scaler)  # pylint: disable=protected-access
    return buffer.getvalue(), orch

  def test_items_one_to_eight_in_order_and_nothing_else(self):
    """The Stage B block prints the eight items in order and nothing else."""
    log, orch = self._run_phase0(n_seeds=2, k_folds=2)
    print(log)
    self.assertIn("=== Phase 0: Saturation-Aware Setup (family=tanh) ===", log)
    self.assertIn("  ├─ Stage A: Top-Level Split", log)
    stage_b = parse_stage_b(log)
    m = check_item_order(self, stage_b, n_treatments=1)
    n_treat, s, k, total = (int(v) for v in m.groups())
    self.assertEqual((n_treat, s, k), (1, 2, 2))
    self.assertEqual(total, 1 * 2 * 2 * 2)
    alpha_results = orch._alpha_results  # pylint: disable=protected-access
    self.assertEqual(total, sum(r.n_ml_fits for r in alpha_results.values()))
    # The console output must not carry internal markers (tag-like
    # prefixes or section pointers), nor internal step logs.
    self.assertNotRegex(log, r"\[v4-[A-Z]")
    self.assertNotIn("spec §", log)
    for banned in ("Stage B schedules", "one unit = one ML training",
                   "Alpha is estimated on the normalized treatment",
                   "grid extended x4", "[Phase 0 / Stage B]"):
      self.assertNotIn(banned, log)
    # Item 7 states the linear fallback explicitly when it happens.
    decision_line = next(
        ln for ln in stage_b if ITEM_PATTERNS["7_decision"].match(ln)
    )
    result = orch._alpha_results["spend"]  # pylint: disable=protected-access
    if result.selected_family is None:
      self.assertIn("FALLBACK TO LINEAR DML", decision_line)
    else:
      self.assertIn("selected_family = tanh", decision_line)

  def test_count_tracks_treatments_seeds_and_folds(self):
    log, orch = self._run_phase0(n_seeds=3, k_folds=2, d_cols=("spend", "tv"))
    stage_b = parse_stage_b(log)
    m = check_item_order(self, stage_b, n_treatments=2)
    n_treat, s, k, total = (int(v) for v in m.groups())
    self.assertEqual((n_treat, s, k), (2, 3, 2))
    self.assertEqual(total, 2 * 2 * 3 * 2)
    alpha_results = orch._alpha_results  # pylint: disable=protected-access
    self.assertEqual(total, sum(r.n_ml_fits for r in alpha_results.values()))

  def test_progress_line_format(self):
    """Item 2): elapsed, completed / total, ETA -- one line per interval."""
    orch = make_orchestrator("/tmp/unused_k16")
    with tempfile.TemporaryDirectory() as marker_dir:
      for name in ("a.done", "b.done"):
        with open(os.path.join(marker_dir, name), "w") as fh:
          fh.write("")
      stop = threading.Event()
      buffer = io.StringIO()
      with contextlib.redirect_stdout(buffer):
        thread = threading.Thread(
            target=orch._monitor_phase0_progress,  # pylint: disable=protected-access
            args=(marker_dir, 8, stop, 1),
        )
        thread.start()
        thread.join(timeout=5.0)
        stop.set()
        thread.join()
    lines = [ln for ln in buffer.getvalue().splitlines() if ln.strip()]
    self.assertTrue(lines)
    for line in lines:
      self.assertRegex(line, ITEM_PATTERNS[2])
    self.assertIn("2/8 ML trainings completed", lines[0])


class Diag6Test(unittest.TestCase):
  """Diag6: unit consistency of T_ref at Phase 1.

  Feeding Phase 1 a rescaled treatment column (x100) stops the pipeline with
  an error at the start of Phase 1 (ERROR line, nothing of Phase 1 runs); the
  untouched hand-over passes silently.
  """

  def test_rescaled_treatment_column_stops_the_pipeline(self):
    """A x100 unit mismatch at the Phase 1 hand-over stops the pipeline."""
    df = helpers.saturating_panel(n_weeks=60, theta=6.0)
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    with tempfile.TemporaryDirectory() as out_dir:
      # The 1-SE rule is forced so that a family is adopted (Diag6 runs only
      # when selected_family != None).
      orch = make_orchestrator(out_dir, linear_test_k=-1e12)
      original = orch._run_phase0  # pylint: disable=protected-access

      def rescaled_phase0(frame, sc):
        df_fit, df_est = original(frame, sc)
        # A unit mismatch between Phase 0 and Phase 1: the column handed to
        # Phase 1 is in different units (x100) than the one Phase 0 saw.
        df_fit = df_fit.copy()
        df_fit["spend"] = df_fit["spend"] * 100.0
        if orch._score_frame is not None:  # pylint: disable=protected-access
          score = orch._score_frame.copy()  # pylint: disable=protected-access
          score["spend"] = score["spend"] * 100.0
          orch._score_frame = score  # pylint: disable=protected-access
        return df_fit, df_est

      buffer = io.StringIO()
      with mock.patch.object(orch, "_run_phase0", rescaled_phase0), (
          warnings.catch_warnings()
      ), contextlib.redirect_stdout(buffer):
        warnings.simplefilter("ignore")
        with self.assertRaises(workflow.TRefMismatchError) as ctx:
          orch.run_full_pipeline(df=df, scaler=scaler, top_n=1, n_reps=1)
      log = buffer.getvalue()
      self.assertEqual(orch._selected_family["spend"], "tanh")  # pylint: disable=protected-access
      message = str(ctx.exception)
      self.assertIn("Diag6", message)
      self.assertIn("T_ref^(0) =", message)
      self.assertIn("T_ref^(1) =", message)
      self.assertIn("column 'spend'", message)
      self.assertIn("[ERROR] Diag6", log)
      # The stop happens before anything of Phase 1 runs.
      self.assertNotIn("=== Phase 1", log)

  def test_consistent_handover_is_silent(self):
    """An untouched hand-over passes Diag6 without any output."""
    df = helpers.saturating_panel(n_weeks=60, theta=6.0)
    scaler = scalers.CausalDataScaler(standardize_cols=["X1", "X2", "sales"])
    with tempfile.TemporaryDirectory() as out_dir:
      orch = make_orchestrator(out_dir, linear_test_k=-1e12)
      buffer = io.StringIO()
      with warnings.catch_warnings(), contextlib.redirect_stdout(buffer):
        warnings.simplefilter("ignore")
        df_fit, _ = orch._run_phase0(df, scaler)  # pylint: disable=protected-access
        df_fit = orch._apply_saturation_transform(df_fit)  # pylint: disable=protected-access
      self.assertNotIn("Diag6", buffer.getvalue())
      # The treatment Phases 1-3 receive is fbar in [0, >1), not raw spend.
      self.assertLess(float(df_fit["spend"].max()), 10.0)
      self.assertGreater(float(df["spend"].max()), 10.0)


if __name__ == "__main__":
  unittest.main()
