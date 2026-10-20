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

"""Module handling input and output operations for saving workflow results.

Besides the step 1-3 writers this module is the single place that names the
files of the Phase 0 output folder ``step0_saturation/``, so the
writer (``evaluation.saturation_evaluation``), the figures
(``visualization.plots``), the orchestrator (``workflow``) and the HTML
report (``visualization.html_extraction``) agree on the names without
importing each other.

Every ``step0_saturation/`` file name carries a prefix that states which part
of Phase 0 it belongs to, so the folder can be read on its own:

* ``diag1_`` ... ``diag6_``: the six Phase 0 identification diagnostics
  (Diag1 basis approximation error, Diag2 1-SE flatness, Diag3
  residual-variation ratio ``r_T``, Diag4 shape quantiles, Diag5 saturation
  level at ``T_ref``, Diag6 unit consistency of ``T_ref``);
* ``summary_``: the result of the alpha selection itself (the summary JSON /
  CSV, the per-grid-point table, the per-seed R-loss curves, the per-seed
  argmins behind the "Seed-wise alpha distribution" panel and the five-panel
  alpha profile chart);
* ``others_``: supporting material without a diagnostic rule (the Stage A
  split coverage per key level -- summary CSV, per-cell detail CSV and the
  bar chart).

The alpha profile chart ``summary_alpha_profile_<treatment>.png`` is the one
figure that summarises Phase 0 for a treatment: grid construction, seed-wise
alpha distribution, profiled R-loss curve, fitted saturation curve and the
key stats / process metrics.  (The former stand-alone
``others_grid_construction.png`` and ``others_alpha_seed_distribution.png``
duplicated two of its panels and are no longer written.)
"""

import os
from typing import Any, Dict, Tuple
import cloudpickle as cp
import pandas as pd

# ---------------------------------------------------------------------------
# step0_saturation/ -- summary_: the result of the alpha selection.
# ---------------------------------------------------------------------------
# The per-treatment summary (Phases 1-3 / Phase 4 read T_ref here).
SUMMARY_JSON = "summary_alpha_selection.json"
# One row per treatment (written once, at the top of the Phase 0 folder).
SUMMARY_CSV = "summary_alpha_selection.csv"
# Stage A split, cluster-key plan and every Phase 0 warning of the run.
PHASE0_DIAGNOSTICS_JSON = "summary_phase0_diagnostics.json"
# One row per grid point (alpha, T_1/2, mean R-loss, 1-SE flag,
# the Diag4 shape quantiles and the Diag1 error per alpha).
ALPHA_GRID_CSV = "summary_alpha_grid.csv"
# The per-seed R-loss curves (CATE / intercept / identity).
RLOSS_CURVES_CSV = "summary_rloss_curves.csv"
# The per-seed argmins alpha_hat_s: the data behind the "Seed-wise alpha
# distribution" panel (top right) of the alpha profile chart.
ALPHA_SEED_DISTRIBUTION_CSV = "summary_alpha_seed-wise_alpha_distribution.csv"
# The five-panel alpha profile chart of one treatment (2 + 2 + 1 panels:
# grid construction | seed-wise alpha distribution / profiled R-loss curve |
# fitted saturation curve / key stats & process metrics).
ALPHA_PROFILE_PNG_TEMPLATE = "summary_alpha_profile_{treatment}.png"

# ---------------------------------------------------------------------------
# step0_saturation/ -- diagN_: the identification diagnostics Diag1 - Diag6.
# ---------------------------------------------------------------------------
DIAG1_BASIS_APPROX_ERROR_PNG = "diag1_basis_approx_error.png"
DIAG2_RLOSS_PROFILE_PNG = "diag2_rloss_profile.png"
DIAG3_IDENTIFICATION_CSV = "diag3_identification_diagnostics.csv"
DIAG4_F_TRANSFORM_QUANTILES_PNG = "diag4_f_transform_quantiles.png"
# tanh / exp only (Diag5: kappa is undefined for log / power).
DIAG5_SATURATION_AT_T_REF_PNG = "diag5_saturation_at_T_ref.png"
# Written at the start of Phase 1 (Diag6 runs there).
DIAG6_T_REF_CONSISTENCY_CSV = "diag6_t_ref_consistency.csv"

# ---------------------------------------------------------------------------
# step0_saturation/ -- others_: supporting material without a diagnostic rule
# -- the Stage A split coverage.  The three files
# share the ``others_split_coverage`` stem: the per-key-level summary CSV, the
# per-cell detail CSV (Stage B fold per (seed, cell) and the split side) and
# the bar chart drawn from the summary CSV.
# ---------------------------------------------------------------------------
COVERAGE_CHECK_CSV = "others_split_coverage.csv"
FOLD_ASSIGNMENTS_CSV = "others_split_coverage_detail.csv"
SPLIT_COVERAGE_PNG = "others_split_coverage.png"

# Columns of the Diag6 record written at the start of
# Phase 1 (one row per treatment).
DIAG6_COLUMNS = (
    "treatment",
    "status",
    "T_ref_phase0",
    "T_ref_phase1",
    "rel_error",
    "t_ref_tolerance",
    "t_ref_quantile",
    "n_rows_phase1",
    "note",
)


def alpha_profile_png(treatment: str) -> str:
  """Name of the five-panel alpha profile chart of ``treatment``."""
  return ALPHA_PROFILE_PNG_TEMPLATE.format(treatment=treatment)


# Descriptions of ``FILE_INDEX`` that do not fit on one line.  They are kept
# out of the tuple so that no element is two adjacent string literals, which
# gpylint flags as a probable missing comma (``implicit-str-concat``).
_ALPHA_PROFILE_DESCRIPTION = (
    "grid construction, seed-wise alpha distribution, profiled R-loss"
    " curve, fitted saturation curve, key stats & process metrics"
)
_ALPHA_GRID_DESCRIPTION = (
    "per grid point: alpha, T_1/2, mean_r_loss, se_alpha, in_1se_set,"
    " fbar_q25..q90, basis_approx_error_alpha"
)
_ALPHA_SEED_DISTRIBUTION_DESCRIPTION = (
    "per seed: alpha_hat_s, rloss_at_alpha_hat_s, rloss_identity_s, d_s"
    " (the seed-wise alpha distribution panel)"
)

# (file name, category, one-line description) of every per-treatment file,
# in the order a reader should open them.  ``diag5_*`` exists only for
# ``tanh`` / ``exp`` and ``diag6_*`` is written by Phase 1.
FILE_INDEX: Tuple[Tuple[str, str, str], ...] = (
    (SUMMARY_JSON, "summary",
     "decision, alpha_hat, T_ref, d_bar / se, Diag1-Diag5 values"),
    (ALPHA_PROFILE_PNG_TEMPLATE, "summary", _ALPHA_PROFILE_DESCRIPTION),
    (ALPHA_GRID_CSV, "summary", _ALPHA_GRID_DESCRIPTION),
    (RLOSS_CURVES_CSV, "summary",
     "per seed x grid point: rloss_cate, rloss_ate, rloss_identity"),
    (ALPHA_SEED_DISTRIBUTION_CSV, "summary",
     _ALPHA_SEED_DISTRIBUTION_DESCRIPTION),
    (DIAG1_BASIS_APPROX_ERROR_PNG, "diag1",
     "eps(J) per grid alpha against basis_tol"),
    (DIAG2_RLOSS_PROFILE_PNG, "diag2",
     "R-loss profile with the +-1 se band, the identity level and A_1SE"),
    (DIAG3_IDENTIFICATION_CSV, "diag3",
     "r_T = sd(T~) / sd(T / T_ref) and the treatment-support statistics"),
    (DIAG4_F_TRANSFORM_QUANTILES_PNG, "diag4",
     "phi_p(alpha) = fbar(q_p(T>0); alpha) with the linear references"),
    (DIAG5_SATURATION_AT_T_REF_PNG, "diag5",
     "kappa(alpha) = f(T_ref; alpha) over the grid (tanh / exp only)"),
    (DIAG6_T_REF_CONSISTENCY_CSV, "diag6",
     "T_ref^(0) vs T_ref^(1) at the start of Phase 1 (pass / fail / skipped)"),
    (COVERAGE_CHECK_CSV, "others",
     "per key level: n_cells_sel, n_cells_inf, covered, fold coverage"),
    (FOLD_ASSIGNMENTS_CSV, "others",
     "per (seed, cell): Stage B fold and the Stage A split side"),
    (SPLIT_COVERAGE_PNG, "others",
     "bar chart of n_cells_sel / n_cells_inf per key level"),
)

METADATA_FILE = "metadata.json"
PHASE4_DIR = "step4_consolidated_results"
FULL_DF_FILE = f"{PHASE4_DIR}/consolidated_1a_full_df.csv"
REPETITIONS_FILE = f"{PHASE4_DIR}/consolidated_3a_detail.csv"
SUMMARY_FILE = f"{PHASE4_DIR}/consolidated_3b_summary.csv"
GEO_SUMMARY_FILE = f"{PHASE4_DIR}/consolidated_3b_summary_geo.csv"
ITEM_SUMMARY_FILE = f"{PHASE4_DIR}/consolidated_3b_summary_item.csv"
CATE_COEF_FILE = f"{PHASE4_DIR}/graph/004_cate/cate_coef_dict_pooling.csv"
PRIOR_FILE = f"{PHASE4_DIR}/graph/005_prior/prior_parameters.csv"
SENSITIVITY_FILE = (
    f"{PHASE4_DIR}/graph/006_sensitivity/sensitivity_metrics.csv"
)
OPTIMIZATION_FILE = (
    f"{PHASE4_DIR}/graph/007_optimization/optimization_metrics.csv"
)
# Where non-spend treatments (percentage, impressions, price) write the same
# file; format with `treatment=<d_col>`.
ROI_OUTPUT_OPTIMIZATION_FILE = (
    f"{PHASE4_DIR}/roi_output/{{treatment}}/007_optimization/"
    "optimization_metrics.csv"
)

DIMENSIONS = ("geo", "item")


def save_step1_outputs(
    df_1a: pd.DataFrame,
    df_1b: pd.DataFrame,
    df_1c: pd.DataFrame,
    model_name: str,
    out_dir: str,
    process_log: str = "lightweight",
) -> None:
  """Saves outputs generated during Step 1 of the causal evaluation.

  Args:
    df_1a: The full dataset.
    df_1b: The geo-item level aggregated dataset.
    df_1c: The evaluation metrics summary dataset.
    model_name: Name identifier for the model.
    out_dir: Output directory where the results should be saved.
    process_log: Log level. If 'lightweight', skip generating 1a_full_df.
  """
  os.makedirs(out_dir, exist_ok=True)
  if process_log != "lightweight":
    df_1a.to_csv(
        os.path.join(out_dir, f"1a_full_df_{model_name}.csv"), index=False
    )
  df_1b.to_csv(
      os.path.join(out_dir, f"1b_geo_item_{model_name}.csv"), index=False
  )
  df_1c.to_csv(
      os.path.join(out_dir, f"1c_metrics_{model_name}.csv"), index=False
  )


def save_step2_outputs(shortlisted_1c: pd.DataFrame, out_dir: str) -> None:
  """Saves the shortlisted models data generated during Step 2.

  Args:
    shortlisted_1c: The shortlisted models metrics.
    out_dir: Output directory where the results should be saved.
  """
  os.makedirs(out_dir, exist_ok=True)
  shortlisted_1c.to_csv(
      os.path.join(out_dir, "2_shortlisted_models_metrics.csv"), index=False
  )


def save_step3_outputs(
    df_3a_detail: pd.DataFrame,
    df_3b_summary: pd.DataFrame,
    best_model_data: Dict[str, Any],
    out_dir: str,
) -> None:
  """Saves the Step 3 iteration details and the best model binary.

  Args:
    df_3a_detail: Iteration detail dataframe.
    df_3b_summary: Iteration summary dataframe.
    best_model_data: Dictionary containing best model's datasets and objects.
    out_dir: Output directory where the results should be saved.
  """
  os.makedirs(out_dir, exist_ok=True)
  df_3a_detail.to_csv(
      os.path.join(out_dir, "3a_iteration_detail.csv"), index=False
  )
  df_3b_summary.to_csv(
      os.path.join(out_dir, "3b_iteration_summary.csv"), index=False
  )

  best_dir = os.path.join(out_dir, "best_model")
  os.makedirs(best_dir, exist_ok=True)

  best_model_name = best_model_data["model_name"]
  best_model_data["df_1a"].to_csv(
      os.path.join(best_dir, f"3c_best_1a_df_{best_model_name}.csv"),
      index=False,
  )
  best_model_data["df_1b"].to_csv(
      os.path.join(best_dir, f"3c_best_1b_geo_item_{best_model_name}.csv"),
      index=False,
  )

  with open(
      os.path.join(best_dir, f"best_dml_object_{best_model_name}.pkl"), "wb"
  ) as f:
    cp.dump(best_model_data["model_obj"], f)
