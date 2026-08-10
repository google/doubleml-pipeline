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

"""Module handling input and output operations for saving workflow results."""

import os
from typing import Any, Dict
import cloudpickle as cp
import pandas as pd


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
  """Saves execution iteration details and the best model binary generated during Step 3.

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
