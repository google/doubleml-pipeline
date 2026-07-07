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

"""Module acting as the Orchestrator for the Causal Inference workflow."""

import glob
import os
import re
import shutil
import threading
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from . import optimization
from .evaluation import metrics
from .evaluation import runner
from .evaluation import selection
from .preprocessing import scalers
from .utils import io
from .visualization import plots


class CausalWorkflowOrchestrator:
  """Main orchestrator handling causal inference modeling.

  Handles execution, parallelization, selection, and consolidation.
  """

  def __init__(
      self,
      y_col: str,
      d_cols: List[str],
      x_cols_list: List[List[str]],
      ml_models_y: List[str],
      ml_models_t: List[str],
      n_folds_list: List[int],
      output_dir: str,
      n_jobs: int = -1,
      covariates_for_cate_list: Optional[List[List[str]]] = None,
      cate_structure_list: Optional[List[Any]] = None,
      time_budget: int = 15,
      date_col: Optional[str] = None,
      geo_col: Optional[str] = None,
      item_col: Optional[str] = None,
      geo_name_col: Optional[str] = None,
      item_name_col: Optional[str] = None,
      run_sensitivity: bool = False,
      sensitivity_scope: Optional[List[str]] = None,
      ground_truth_existence: bool = False,
      ground_truth_effect_column: Optional[List[str]] = None,
      treatment_types: Optional[Dict[str, str]] = None,
      treatment_spend_cols: Optional[Dict[str, str]] = None,
      sales_col: Optional[str] = None,
      optimize: bool = False,
      simple_optimization: bool = True,
      opt_periods: Optional[tuple[Any, Any]] = None,
      opt_threshold_roi: float = 0.0,
      opt_other_cost_variables: Optional[List[str]] = None,
  ):
    """Initializes the central parameters for orchestration tasks."""
    # Validate parameters
    if not isinstance(d_cols, list):
      raise TypeError("d_cols must be a list of strings.")

    if not isinstance(x_cols_list, list):
      raise TypeError("x_cols_list must be a list of lists.")
    if not all(isinstance(x, list) for x in x_cols_list):
      raise TypeError("Each element in x_cols_list must be a list.")
    if len(x_cols_list) != len(d_cols):
      raise ValueError(
          f"Length of x_cols_list ({len(x_cols_list)}) must match length of"
          f" d_cols ({len(d_cols)})."
      )

    if covariates_for_cate_list is not None:
      if not isinstance(covariates_for_cate_list, list):
        raise TypeError("covariates_for_cate_list must be a list of lists.")
      if not all(isinstance(x, list) for x in covariates_for_cate_list):
        raise TypeError(
            "Each element in covariates_for_cate_list must be a list."
        )
      if len(covariates_for_cate_list) != len(d_cols):
        raise ValueError(
            "Length of covariates_for_cate_list "
            f"({len(covariates_for_cate_list)}) must match length of "
            f"d_cols ({len(d_cols)})."
        )

    self.y_col = y_col
    self.d_cols = d_cols
    self.x_cols_list = x_cols_list
    self.ml_models_y = ml_models_y
    self.ml_models_t = ml_models_t
    self.n_folds_list = n_folds_list
    self.output_dir = output_dir
    self.n_jobs = n_jobs
    self.covariates_for_cate_list = covariates_for_cate_list or [
        [] for _ in d_cols
    ]
    self.cate_structure_list = cate_structure_list or [
        {
            "type": "auto_additive",
            "df": 3,
            "degree": 2,
            "include_intercept": False,
        }
        for _ in d_cols
    ]
    self.time_budget = time_budget
    self.date_col = date_col
    self.geo_col = geo_col
    self.item_col = item_col
    self.geo_name_col = geo_name_col
    self.item_name_col = item_name_col
    self.run_sensitivity = run_sensitivity
    self.sensitivity_scope = sensitivity_scope or ["total"]
    self.ground_truth_existence = ground_truth_existence
    self.ground_truth_effect_column = ground_truth_effect_column or []
    self.treatment_types = treatment_types or {}
    for d_col in self.d_cols:
      if d_col not in self.treatment_types:
        self.treatment_types[d_col] = "spend"

    self.treatment_spend_cols = treatment_spend_cols or {}
    for d_col in self.d_cols:
      if self.treatment_types[d_col] == "spend":
        if d_col not in self.treatment_spend_cols:
          self.treatment_spend_cols[d_col] = d_col
    self.sales_col = sales_col
    self.optimize = optimize
    self.simple_optimization = simple_optimization
    self.opt_periods = opt_periods
    self.opt_threshold_roi = opt_threshold_roi
    self.opt_other_cost_variables = opt_other_cost_variables or []

  def _validate_treatments(self, df: pd.DataFrame) -> None:
    """Validates treatment types, percentage bounds, sales_col presence, and target spend columns."""
    for d_col in self.d_cols:
      ttype = self.treatment_types.get(d_col, "spend")
      if ttype == "percentage":
        if d_col in df.columns:
          vals = df[d_col].dropna().values
          if len(vals) > 0 and (np.any(vals < 0.0) or np.any(vals > 1.0)):
            raise ValueError(
                f"Treatment column '{d_col}' is marked as 'percentage', but"
                " contains values outside [0.0, 1.0]."
            )
        if not self.sales_col or self.sales_col not in df.columns:
          raise ValueError(
              f"Treatment column '{d_col}' is marked as 'percentage', but valid"
              f" 'sales_col' ('{self.sales_col}') was not provided in"
              " dataframe."
          )
      elif ttype in ["spend", "impressions"]:
        if d_col in df.columns:
          vals = df[d_col].dropna().values
          if len(vals) > 0 and np.all((vals >= 0.0) & (vals <= 1.0)):
            lower_d_col = d_col.lower()
            if any(sub in lower_d_col for sub in ["discount", "pct", "rate"]):
              warnings.warn(
                  f"Treatment column '{d_col}' has values in [0.0, 1.0] and"
                  " name containing discount/pct/rate, but is set to"
                  f" '{ttype}'. Consider setting treatment_types['{d_col}'] ="
                  " 'percentage'.",
                  UserWarning,
              )

      if d_col in self.treatment_spend_cols:
        target_spend = self.treatment_spend_cols[d_col]
        if target_spend not in df.columns:
          raise ValueError(
              f"Specified spend column '{target_spend}' for treatment"
              f" '{d_col}' not found in dataframe."
          )

  def _monitor_phase1_progress(
      self,
      target_dir: str,
      total_tasks: int,
      stop_event: threading.Event,
      interval: int = 60,
  ) -> None:
    """Polls directory to update progress messages for Phase 1."""
    while not stop_event.is_set():
      for _ in range(interval * 10):
        if stop_event.is_set():
          break
        time.sleep(0.1)
      if stop_event.is_set():
        break
      if os.path.exists(target_dir):
        metrics_files = sorted(
            glob.glob(os.path.join(target_dir, "1c_metrics_*.csv")),
            key=os.path.getmtime,
        )
        completed = len(metrics_files)
        pct = (completed / total_tasks) * 100
        if completed > 0:
          latest_model = (
              os.path.basename(metrics_files[-1])
              .replace("1c_metrics_", "")
              .replace(".csv", "")
          )
          print(
              f"[Phase 1 Progress] {completed}/{total_tasks} ({pct:.1f}%)"
              f" completed. Latest estimated model: {latest_model}",
              flush=True,
          )
        else:
          print(
              f"[Phase 1 Progress] 0/{total_tasks} (0.0%) completed...",
              flush=True,
          )
    print(
        f"[Phase 1 Progress] {total_tasks}/{total_tasks} (100.0%) completed.",
        flush=True,
    )

  def _monitor_phase3_progress(
      self,
      target_dir: str,
      total_reps: int,
      model_name: str,
      stop_event: threading.Event,
      interval: int = 60,
  ) -> None:
    """Polls directory to update progress messages for Phase 3."""
    while not stop_event.is_set():
      for _ in range(interval * 10):
        if stop_event.is_set():
          break
        time.sleep(0.1)
      if stop_event.is_set():
        break
      if os.path.exists(target_dir):
        completed = len(glob.glob(os.path.join(target_dir, "*.done")))
        pct = (completed / total_reps) * 100
        print(
            f"[Phase 3 Progress - {model_name}] {completed}/{total_reps}"
            f" ({pct:.1f}%) completed.",
            flush=True,
        )
    print(
        f"[Phase 3 Progress - {model_name}] {total_reps}/{total_reps} (100.0%)"
        " completed.",
        flush=True,
    )

  def _insert_metadata_columns(
      self, df_1b: pd.DataFrame, meta_dict: Dict[str, Any]
  ) -> pd.DataFrame:
    """Injects tracking metadata cleanly into aggregated dataframes."""
    try:
      model_idx = list(df_1b.columns).index("model")
      for i, (k_val, v_val) in enumerate(meta_dict.items()):
        if k_val not in df_1b.columns:
          df_1b.insert(model_idx + 1 + i, k_val, v_val)
    except ValueError:
      for k_val, v_val in meta_dict.items():
        if k_val not in df_1b.columns:
          df_1b[k_val] = v_val
    return df_1b

  def _run_single_exploration(
      self,
      ml_y: str,
      ml_t: str,
      n_folds: int,
      d_col: str,
      x_cols: List[str],
      cate_cols: List[str],
      cate_structure: Any,
      gt_col: str,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
  ) -> Dict[str, Any]:
    """Executes a single structural estimation run across cross-validation folds."""
    # FIX: Extract valid cluster columns for cluster robust inference
    cluster_cols = []
    if self.geo_col and self.geo_col in df.columns:
      cluster_cols.append(self.geo_col)
    if self.item_col and self.item_col in df.columns:
      cluster_cols.append(self.item_col)

    pipeline = runner.DoubleMLPipeline(
        self.y_col,
        d_col,
        x_cols,
        ml_y,
        ml_t,
        n_folds=n_folds,
        time_budget=self.time_budget,
        cate_structure=cate_structure,
        covariates_for_cate=cate_cols,
        ground_truth_col=gt_col,
        cluster_cols=cluster_cols,
        random_state=1,
    )
    res = pipeline.run(df, scaler, rep_id=1)
    df_1b = metrics.aggregate_geo_item_metrics(
        res["df_1a"],
        res["model_name"],
        d_col,
        self.y_col,
        geo_col=self.geo_col,
        item_col=self.item_col,
        geo_name_col=self.geo_name_col,
        item_name_col=self.item_name_col,
        gt_col=gt_col,
    )

    run_metrics = res["metrics_1c"]
    meta_dict = {
        "d_col": d_col,
        "x_col": str(x_cols),
        "y_col": self.y_col,
        "covariates_for_cate": str(cate_cols),
        "cate_structure": run_metrics.get("cate_structure", ""),
        "cate_df": run_metrics.get("cate_df", ""),
        "cate_degree": run_metrics.get("cate_degree", ""),
        "cate_intercept": run_metrics.get("cate_intercept", ""),
    }
    df_1b = self._insert_metadata_columns(df_1b, meta_dict)

    out_dir = os.path.join(self.output_dir, f"step1_exploration_{d_col}")
    io.save_step1_outputs(
        res["df_1a"],
        df_1b,
        pd.DataFrame([res["metrics_1c"]]),
        res["model_name"],
        out_dir,
    )
    return res["metrics_1c"]

  def _run_single_iteration(
      self,
      row: pd.Series,
      d_col: str,
      x_cols: List[str],
      cate_cols: List[str],
      cate_structure: Any,
      gt_col: str,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      rep: int,
      temp_dir: str,
  ) -> Dict[str, Any]:
    """Spins up iteration tasks inside Phase 3's robust ensemble structure."""
    # FIX: Extract valid cluster columns for cluster robust inference
    cluster_cols = []
    if self.geo_col and self.geo_col in df.columns:
      cluster_cols.append(self.geo_col)
    if self.item_col and self.item_col in df.columns:
      cluster_cols.append(self.item_col)

    pipeline = runner.DoubleMLPipeline(
        self.y_col,
        d_col,
        x_cols,
        row["ML_model_Y"],
        row["ML_model_T"],
        n_folds=row["n_folds"],
        time_budget=self.time_budget,
        cate_structure=cate_structure,
        covariates_for_cate=cate_cols,
        ground_truth_col=gt_col,
        cluster_cols=cluster_cols,
        random_state=rep + 1,
    )
    res = pipeline.run(df, scaler, rep_id=rep + 1)
    res["model_group"] = row["model"]
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, f"rep_{rep}.done"), "w") as f_obj:
      f_obj.write("")
    return res

  def format_cate_equation(self, coef_dict: Dict[str, float]) -> str:
    """Transforms regression dictionary parameters into a readable equation text string."""
    eq_parts = []
    for name, val in coef_dict.items():
      if abs(val) < 1e-10:
        continue
      clean_name = re.sub(
          r"bs\(([^,]+)[^\)]*\)(?:\[(\d+)\])?", r"Spline_\2(\1)", name
      )
      clean_name = (
          clean_name.replace("[T.", "_").replace("]", "").replace(":", " * ")
      )

      if abs(val) < 0.0001:
        val_str = f"{val:+.2e}"
      else:
        val_str = f"{val:+.4f}"

      if clean_name == "Intercept":
        eq_parts.append(val_str)
      else:
        eq_parts.append(f"{val_str} * {clean_name}")

    if not eq_parts:
      return "Constant Effect"
    res = " ".join(eq_parts).strip()
    if res.startswith("+"):
      res = res[1:].strip()
    return res

  def extract_priors(
      self,
      df_1a_consolidated: pd.DataFrame,
      treatment_base_name: str,
      prior_type: str = "roi",
      distribution_type: str = "Normal",
  ) -> Dict[str, float]:
    """Generates shape parameters (mean, standard deviation, alpha, beta) for probabilistic prior matching."""
    if self.date_col and self.date_col in df_1a_consolidated.columns:
      ts_df = df_1a_consolidated.groupby(self.date_col).sum()
    else:
      ts_df = df_1a_consolidated

    group_cols = [
        c
        for c in self.d_cols
        if re.sub(r"_l\d+$", "", c) == treatment_base_name
    ]
    if not group_cols:
      return {}

    kpi_cols = [f"estimated_incremental_KPI_{c}" for c in group_cols]
    total_inc_kpi = ts_df[kpi_cols].sum(axis=1)
    input_col = (
        ts_df[treatment_base_name]
        if treatment_base_name in ts_df.columns
        else pd.Series(np.zeros(len(ts_df)), index=ts_df.index)
    )

    if prior_type == "roi":
      mask = input_col > 0
      data = (total_inc_kpi.loc[mask] / input_col.loc[mask]).values
    else:
      mask = (input_col > 0) & (ts_df[self.y_col] > 0)
      data = (total_inc_kpi.loc[mask] / ts_df.loc[mask, self.y_col]).values

    data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values
    if len(data) == 0:
      return {}

    if distribution_type == "Normal":
      mu_val, std = stats.norm.fit(data)
      return {"mu": mu_val, "sigma": std}
    elif distribution_type == "LogNormal":
      valid_data = data[data > 0]
      if len(valid_data) == 0:
        return {}
      shape, _, scale = stats.lognorm.fit(valid_data, floc=0)
      return {"mu": np.log(scale), "sigma": shape}
    elif distribution_type == "Beta":
      valid_data = data[(data > 0) & (data < 1)]
      if len(valid_data) == 0:
        return {}
      a_val, b_val, _, _ = stats.beta.fit(valid_data, floc=0, fscale=1)
      return {"alpha": a_val, "beta": b_val}
    return {}

  def _generate_dim_summary(
      self,
      dim_col: str,
      df_1a_consolidated: pd.DataFrame,
      best_model_data_dict: Dict[str, Any],
      df_3b_concat: pd.DataFrame,
  ) -> Optional[pd.DataFrame]:
    """Consolidates metric outputs over individual dimension groups (geo, item)."""
    if not dim_col or dim_col not in df_1a_consolidated.columns:
      return None
    rows = []
    for dim_val in df_1a_consolidated[dim_col].dropna().unique():
      dim_df = df_1a_consolidated[df_1a_consolidated[dim_col] == dim_val]
      sum_input_total = 0
      sum_kpi_total = 0

      y_true_ts_tot = np.zeros(len(dim_df))
      y_pred_ts_tot = np.zeros(len(dim_df))

      for i, d_col in enumerate(self.d_cols):
        m_data = best_model_data_dict[d_col]["metrics_1c"]
        kpi_col = f"estimated_incremental_KPI_{d_col}"

        t_input = dim_df[d_col].sum()
        t_kpi = dim_df[kpi_col].fillna(0).sum()
        m_group = "_".join(m_data.get("model", "").split("_")[:3])

        channel_summary = df_3b_concat[df_3b_concat["Channel"] == d_col]
        if not channel_summary.empty:
          c_err_med = channel_summary["combined_error_median"].iloc[0]
          c_err_mean = channel_summary["combined_error_mean"].iloc[0]
          c_err_sd = channel_summary["combined_error_sd"].iloc[0]
        else:
          c_err_med = c_err_mean = c_err_sd = np.nan

        row = {
            dim_col: dim_val,
            "Channel": d_col,
            "Model Group": m_group,
            "x_col": m_data.get("x_col"),
            "y_col": m_data.get("y_col"),
            "covariates_for_cate": m_data.get("covariates_for_cate"),
            "cate_structure": m_data.get("cate_structure"),
            "cate_df": m_data.get("cate_df"),
            "cate_degree": m_data.get("cate_degree"),
            "cate_intercept": m_data.get("cate_intercept"),
            "total_input": t_input,
            "total_estimated_incremental_KPI": t_kpi,
            "total_estimated_roi": t_kpi / t_input if t_input > 0 else 0,
            "combined_error_median": c_err_med,
            "combined_error_mean": c_err_mean,
            "combined_error_sd": c_err_sd,
            "best_model": 1,
        }

        if self.ground_truth_existence:
          gt_col = self.ground_truth_effect_column[i]
          y_true_ts = (
              dim_df[gt_col].fillna(0).values
              if gt_col in dim_df.columns
              else np.zeros(len(dim_df))
          )
          y_pred_ts = dim_df[kpi_col].fillna(0).values
          row.update(
              metrics.calculate_ground_truth_metrics(
                  pd.Series(y_true_ts), pd.Series(y_pred_ts), t_input
              )
          )
          y_true_ts_tot += y_true_ts
          y_pred_ts_tot += y_pred_ts
        else:
          row.update({
              "true_total_roi": np.nan,
              "SMAPE": np.nan,
              "R_squared": np.nan,
              "MAE": np.nan,
              "RMSE": np.nan,
          })

        rows.append(row)
        if not re.search(r"_l\d+$", d_col):
          sum_input_total += t_input
        sum_kpi_total += t_kpi

      tot_row = {
          dim_col: dim_val,
          "Channel": "Total",
          "Model Group": "-",
          "x_col": "-",
          "y_col": self.y_col,
          "covariates_for_cate": "-",
          "cate_structure": "-",
          "cate_df": np.nan,
          "cate_degree": np.nan,
          "cate_intercept": np.nan,
          "total_input": sum_input_total,
          "total_estimated_incremental_KPI": sum_kpi_total,
          "total_estimated_roi": (
              sum_kpi_total / sum_input_total if sum_input_total > 0 else 0
          ),
          "combined_error_median": np.nan,
          "combined_error_mean": np.nan,
          "combined_error_sd": np.nan,
          "best_model": 1,
      }
      if self.ground_truth_existence:
        tot_row.update(
            metrics.calculate_ground_truth_metrics(
                pd.Series(y_true_ts_tot),
                pd.Series(y_pred_ts_tot),
                sum_input_total,
            )
        )
      else:
        tot_row.update({
            "true_total_roi": np.nan,
            "SMAPE": np.nan,
            "R_squared": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        })

      rows.append(tot_row)

    # ---------------------------------------------------------
    # Overall "Total" Dimension Block (e.g., Geo = "Total")
    # ---------------------------------------------------------
    dim_val_total = "Total"
    sum_input_total_all = 0
    sum_kpi_total_all = 0

    use_date_agg = (
        self.ground_truth_existence
        and self.date_col
        and self.date_col in df_1a_consolidated.columns
    )

    df_date_agg = None
    y_true_ts_tot_all = None
    y_pred_ts_tot_all = None

    if use_date_agg:
      df_date_agg = df_1a_consolidated.groupby(
          self.date_col, as_index=False
      ).sum()
      y_true_ts_tot_all = np.zeros(len(df_date_agg))
      y_pred_ts_tot_all = np.zeros(len(df_date_agg))
    elif self.ground_truth_existence:
      y_true_ts_tot_all = np.zeros(len(df_1a_consolidated))
      y_pred_ts_tot_all = np.zeros(len(df_1a_consolidated))

    for i, d_col in enumerate(self.d_cols):
      m_data = best_model_data_dict[d_col]["metrics_1c"]
      kpi_col = f"estimated_incremental_KPI_{d_col}"

      t_input = df_1a_consolidated[d_col].sum()
      t_kpi = df_1a_consolidated[kpi_col].fillna(0).sum()
      m_group = "_".join(m_data.get("model", "").split("_")[:3])

      channel_summary = df_3b_concat[df_3b_concat["Channel"] == d_col]
      if not channel_summary.empty:
        c_err_med = channel_summary["combined_error_median"].iloc[0]
        c_err_mean = channel_summary["combined_error_mean"].iloc[0]
        c_err_sd = channel_summary["combined_error_sd"].iloc[0]
      else:
        c_err_med = c_err_mean = c_err_sd = np.nan

      row = {
          dim_col: dim_val_total,
          "Channel": d_col,
          "Model Group": m_group,
          "x_col": m_data.get("x_col"),
          "y_col": m_data.get("y_col"),
          "covariates_for_cate": m_data.get("covariates_for_cate"),
          "cate_structure": m_data.get("cate_structure"),
          "cate_df": m_data.get("cate_df"),
          "cate_degree": m_data.get("cate_degree"),
          "cate_intercept": m_data.get("cate_intercept"),
          "total_input": t_input,
          "total_estimated_incremental_KPI": t_kpi,
          "total_estimated_roi": t_kpi / t_input if t_input > 0 else 0,
          "combined_error_median": c_err_med,
          "combined_error_mean": c_err_mean,
          "combined_error_sd": c_err_sd,
          "best_model": 1,
      }

      if self.ground_truth_existence:
        gt_col = self.ground_truth_effect_column[i]
        if use_date_agg and df_date_agg is not None:
          y_true_ts = (
              df_date_agg[gt_col].fillna(0).values
              if gt_col in df_date_agg.columns
              else np.zeros(len(df_date_agg))
          )
          y_pred_ts = (
              df_date_agg[kpi_col].fillna(0).values
              if kpi_col in df_date_agg.columns
              else np.zeros(len(df_date_agg))
          )
        else:
          y_true_ts = (
              df_1a_consolidated[gt_col].fillna(0).values
              if gt_col in df_1a_consolidated.columns
              else np.zeros(len(df_1a_consolidated))
          )
          y_pred_ts = df_1a_consolidated[kpi_col].fillna(0).values

        row.update(
            metrics.calculate_ground_truth_metrics(
                pd.Series(y_true_ts), pd.Series(y_pred_ts), t_input
            )
        )
        y_true_ts_tot_all += y_true_ts
        y_pred_ts_tot_all += y_pred_ts
      else:
        row.update({
            "true_total_roi": np.nan,
            "SMAPE": np.nan,
            "R_squared": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        })

      rows.append(row)
      if not re.search(r"_l\d+$", d_col):
        sum_input_total_all += t_input
      sum_kpi_total_all += t_kpi

    tot_row_all = {
        dim_col: dim_val_total,
        "Channel": "Total",
        "Model Group": "-",
        "x_col": "-",
        "y_col": self.y_col,
        "covariates_for_cate": "-",
        "cate_structure": "-",
        "cate_df": np.nan,
        "cate_degree": np.nan,
        "cate_intercept": np.nan,
        "total_input": sum_input_total_all,
        "total_estimated_incremental_KPI": sum_kpi_total_all,
        "total_estimated_roi": (
            sum_kpi_total_all / sum_input_total_all
            if sum_input_total_all > 0
            else 0
        ),
        "combined_error_median": np.nan,
        "combined_error_mean": np.nan,
        "combined_error_sd": np.nan,
        "best_model": 1,
    }

    if self.ground_truth_existence:
      tot_row_all.update(
          metrics.calculate_ground_truth_metrics(
              pd.Series(y_true_ts_tot_all),
              pd.Series(y_pred_ts_tot_all),
              sum_input_total_all,
          )
      )
    else:
      tot_row_all.update({
          "true_total_roi": np.nan,
          "SMAPE": np.nan,
          "R_squared": np.nan,
          "MAE": np.nan,
          "RMSE": np.nan,
      })

    rows.append(tot_row_all)
    return pd.DataFrame(rows)

  def run_full_pipeline(
      self,
      df: pd.DataFrame,
      scaler: scalers.CausalDataScaler,
      top_n: int = 5,
      n_reps: int = 20,
  ) -> None:
    """Orchestrates and executes the complete 4-phase DoubleML causal framework modeling sequence."""
    self._validate_treatments(df)
    actual_cores = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
    print(
        "\n======================================================================"
    )
    print(
        "Model Naming Convention"
        " Note:\n[Y-model]_[T-model]_[number_of_folds]_[number_of_repetitions]_[time_budget]_[number_of_covariates_for_cate]."
    )
    print(
        "======================================================================\n",
        flush=True,
    )
    print(
        f"★★★ Starting DML Pipeline Execution on {actual_cores} Parallel"
        " Cores ★★★"
    )

    best_model_data_dict = {}
    all_best_models_3b = []
    all_3a_details = []

    for idx, d_col in enumerate(self.d_cols):
      x_cols = self.x_cols_list[idx]
      cate_cols = self.covariates_for_cate_list[idx]
      cate_structure = self.cate_structure_list[idx]
      gt_col = (
          self.ground_truth_effect_column[idx]
          if self.ground_truth_existence
          else None
      )

      kpi_col = f"estimated_incremental_KPI_{d_col}"
      kpi_lower = f"estimated_incremental_KPI_{d_col}_2.5%"
      kpi_upper = f"estimated_incremental_KPI_{d_col}_97.5%"

      total_p1_tasks = (
          len(self.ml_models_y) * len(self.ml_models_t) * len(self.n_folds_list)
      )
      print(f"\n>>> Executing Channel Workflow: [{d_col}] <<<")
      print(
          f"=== Phase 1: Grid Search Exploration ({total_p1_tasks} total"
          " combinations of models) ===",
          flush=True,
      )

      out_dir_p1 = os.path.join(self.output_dir, f"step1_exploration_{d_col}")
      if os.path.exists(out_dir_p1):
        shutil.rmtree(out_dir_p1, ignore_errors=True)
      os.makedirs(out_dir_p1, exist_ok=True)

      tasks_p1 = []
      for y_val in self.ml_models_y:
        for t_val in self.ml_models_t:
          for f_val in self.n_folds_list:
            tasks_p1.append({"ml_y": y_val, "ml_t": t_val, "n_folds": f_val})
      stop_event = threading.Event()
      monitor_thread = threading.Thread(
          target=self._monitor_phase1_progress,
          args=(out_dir_p1, total_p1_tasks, stop_event, 60),
      )
      monitor_thread.start()

      try:
        exploration_metrics = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
            joblib.delayed(self._run_single_exploration)(
                t_dict["ml_y"],
                t_dict["ml_t"],
                t_dict["n_folds"],
                d_col,
                x_cols,
                cate_cols,
                cate_structure,
                gt_col,
                df,
                scaler,
            )
            for t_dict in tasks_p1
        )
      finally:
        stop_event.set()
        monitor_thread.join()

      df_1c_all = pd.DataFrame(exploration_metrics)
      print(f"\n=== Phase 2: Shortlisting Top {top_n} Performing Models ===")
      df_shortlisted = selection.shortlist_top_models(df_1c_all, top_n=top_n)
      df_shortlisted["ML_model_Y"] = df_shortlisted["model"].apply(
          lambda x: x.split("_")[0]
      )
      df_shortlisted["ML_model_T"] = df_shortlisted["model"].apply(
          lambda x: x.split("_")[1]
      )
      df_shortlisted["n_folds"] = df_shortlisted["model"].apply(
          lambda x: int(x.split("_")[2])
      )
      io.save_step2_outputs(
          df_shortlisted,
          os.path.join(self.output_dir, f"step2_shortlisted_{d_col}"),
      )

      print(
          f"\n=== Phase 3: Robust Ensemble Estimation ({n_reps} repetitions per"
          " model) ===",
          flush=True,
      )
      shortlisted_rows = df_shortlisted[df_shortlisted["shortlisted"] == 1]
      all_iter_results_for_channel = []

      for model_idx, (_, row) in enumerate(shortlisted_rows.iterrows(), 1):
        model_name_for_print = row["model"]
        print(
            f"\n>>> Phase 3 Candidate {model_idx}/{len(shortlisted_rows)}:"
            f" Evaluating Model '{model_name_for_print}' <<<",
            flush=True,
        )
        temp_p3_dir = os.path.join(
            self.output_dir, f"temp_p3_{d_col}_{model_idx}"
        )
        if os.path.exists(temp_p3_dir):
          shutil.rmtree(temp_p3_dir, ignore_errors=True)
        os.makedirs(temp_p3_dir, exist_ok=True)

        stop_event_p3 = threading.Event()
        monitor_thread_p3 = threading.Thread(
            target=self._monitor_phase3_progress,
            args=(temp_p3_dir, n_reps, model_name_for_print, stop_event_p3, 60),
        )
        monitor_thread_p3.start()

        try:
          iter_results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
              joblib.delayed(self._run_single_iteration)(
                  row,
                  d_col,
                  x_cols,
                  cate_cols,
                  cate_structure,
                  gt_col,
                  df,
                  scaler,
                  rep,
                  temp_p3_dir,
              )
              for rep in range(n_reps)
          )
        finally:
          stop_event_p3.set()
          monitor_thread_p3.join()
          shutil.rmtree(temp_p3_dir, ignore_errors=True)
        all_iter_results_for_channel.extend(iter_results)

      df_3a_detail = pd.DataFrame(
          [r["metrics_1c"] for r in all_iter_results_for_channel]
      )
      df_3a_detail["Model Group"] = [
          r["model_group"] for r in all_iter_results_for_channel
      ]

      epsilon = 1e-9
      y_col_rmse = "nuisance_models_rmse_Y_model_evaluated_by_AutoML"
      t_col_rmse = "nuisance_models_rmse_T_model_evaluated_by_AutoML"
      df_3a_detail["norm_Y_rmse"] = (
          df_3a_detail[y_col_rmse] - df_3a_detail[y_col_rmse].min() + epsilon
      ) / (
          df_3a_detail[y_col_rmse].max()
          - df_3a_detail[y_col_rmse].min()
          + epsilon
      )
      df_3a_detail["norm_T_rmse"] = (
          df_3a_detail[t_col_rmse] - df_3a_detail[t_col_rmse].min() + epsilon
      ) / (
          df_3a_detail[t_col_rmse].max()
          - df_3a_detail[t_col_rmse].min()
          + epsilon
      )
      df_3a_detail["combined_error"] = df_3a_detail["norm_T_rmse"] * (
          df_3a_detail["norm_T_rmse"] + df_3a_detail["norm_Y_rmse"]
      )

      iteration_summaries = []
      for m_group, df_group in df_3a_detail.groupby("Model Group"):
        group_results = [
            r
            for r in all_iter_results_for_channel
            if r["model_group"] == m_group
        ]

        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          median_est_sales = np.nanmedian(
              np.array([r["df_1a"][kpi_col].values for r in group_results]),
              axis=0,
          )

        total_input = np.nansum(group_results[0]["df_1a"][d_col].values)
        total_kpi = np.nansum(median_est_sales)

        sum_res = {
            "Channel": d_col,
            "Model Group": m_group,
            "x_col": str(x_cols),
            "y_col": self.y_col,
            "covariates_for_cate": str(cate_cols),
            "cate_structure": df_group["cate_structure"].iloc[0],
            "cate_df": df_group["cate_df"].iloc[0],
            "cate_degree": df_group["cate_degree"].iloc[0],
            "cate_intercept": df_group["cate_intercept"].iloc[0],
            "total_input": total_input,
            "total_estimated_incremental_KPI": total_kpi,
            "total_estimated_roi": (
                total_kpi / total_input if total_input > 0 else 0
            ),
        }
        if gt_col and gt_col in group_results[0]["df_1a"].columns:
          sum_res.update(
              metrics.calculate_ground_truth_metrics(
                  group_results[0]["df_1a"][gt_col],
                  pd.Series(median_est_sales),
                  total_input,
              )
          )
        else:
          sum_res.update({
              "true_total_roi": np.nan,
              "SMAPE": np.nan,
              "R_squared": np.nan,
              "MAE": np.nan,
              "RMSE": np.nan,
          })

        sum_res.update({
            "combined_error_median": df_group["combined_error"].median(),
            "combined_error_mean": df_group["combined_error"].mean(),
            "combined_error_sd": df_group["combined_error"].std(),
            "best_model": 0,
        })
        iteration_summaries.append(sum_res)

      df_3b_summary = pd.DataFrame(iteration_summaries)
      best_model_group = df_3b_summary.loc[
          df_3b_summary["combined_error_median"].idxmin(), "Model Group"
      ]
      df_3b_summary.loc[
          df_3b_summary["Model Group"] == best_model_group, "best_model"
      ] = 1
      df_3a_detail["best_model"] = (
          df_3a_detail["Model Group"] == best_model_group
      ).astype(int)

      best_group_results = [
          r
          for r in all_iter_results_for_channel
          if r["model_group"] == best_model_group
      ]
      best_model_data = best_group_results[0].copy()

      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        point_estimates = np.array(
            [r["df_1a"][kpi_col].values for r in best_group_results]
        )
        lower_bounds = np.array(
            [r["df_1a"][kpi_lower].values for r in best_group_results]
        )
        upper_bounds = np.array(
            [r["df_1a"][kpi_upper].values for r in best_group_results]
        )

        # Chernozhukov et al. (2018), Section 3.4:
        # Median Method for Repeated Cross-Fitting
        tilde_theta = np.nanmedian(point_estimates, axis=0)

        # Back-calculate SE for each repetition b:
        # SE_b = (Upper_b - Lower_b) / (2 * 1.96)
        se_b = (upper_bounds - lower_bounds) / (2.0 * 1.96)
        var_b = se_b**2

        # Penalty for deviation from median point estimate:
        # (theta_b - tilde_theta)^2
        deviation_penalty = (point_estimates - tilde_theta) ** 2

        # Combined variance for each repetition
        combined_var_b = var_b + deviation_penalty

        # Aggregated variance across repetitions: Median_b(combined_var_b)
        tilde_var = np.nanmedian(combined_var_b, axis=0)
        tilde_sigma = np.sqrt(np.maximum(0.0, tilde_var))

        best_model_data["df_1a"][kpi_col] = tilde_theta
        best_model_data["df_1a"][kpi_lower] = tilde_theta - 1.96 * tilde_sigma
        best_model_data["df_1a"][kpi_upper] = tilde_theta + 1.96 * tilde_sigma

      df_1b_base = metrics.aggregate_geo_item_metrics(
          best_model_data["df_1a"],
          best_model_data["model_name"],
          d_col,
          self.y_col,
          geo_col=self.geo_col,
          item_col=self.item_col,
          geo_name_col=self.geo_name_col,
          item_name_col=self.item_name_col,
          gt_col=gt_col,
      )
      bm_metrics = best_model_data["metrics_1c"]
      meta_dict = {
          "d_col": d_col,
          "x_col": str(x_cols),
          "y_col": self.y_col,
          "covariates_for_cate": str(cate_cols),
          "cate_structure": bm_metrics.get("cate_structure", ""),
          "cate_df": bm_metrics.get("cate_df", ""),
          "cate_degree": bm_metrics.get("cate_degree", ""),
          "cate_intercept": bm_metrics.get("cate_intercept", ""),
      }
      df_1b_base = self._insert_metadata_columns(df_1b_base, meta_dict)
      best_model_data["df_1b"] = df_1b_base

      all_best_models_3b.append(
          df_3b_summary[df_3b_summary["best_model"] == 1].copy()
      )
      all_3a_details.append(
          df_3a_detail[df_3a_detail["best_model"] == 1].copy()
      )
      best_model_data_dict[d_col] = best_model_data

      io.save_step3_outputs(
          df_3a_detail,
          df_3b_summary,
          best_model_data,
          os.path.join(self.output_dir, f"step3_final_{d_col}"),
      )

    # Phase 4
    print(
        "\n======================================================================"
    )
    print("=== Phase 4: Consolidating Results & Visualizing Impact ===")
    print(
        "======================================================================"
    )

    phase4_dir = os.path.join(
        self.output_dir, "step4_phase4_consolidated_results"
    )
    graph_dir = os.path.join(phase4_dir, "graph")
    total_graph_dir = os.path.join(graph_dir, "001_total")
    geo_graph_dir = os.path.join(graph_dir, "002_geo")
    item_graph_dir = os.path.join(graph_dir, "003_item")
    cate_graph_dir = os.path.join(graph_dir, "004_cate")
    prior_graph_dir = os.path.join(graph_dir, "005_prior")
    sens_graph_dir = os.path.join(graph_dir, "006_sensitivity")

    for dir_path in [
        phase4_dir,
        total_graph_dir,
        geo_graph_dir,
        item_graph_dir,
        sens_graph_dir,
        cate_graph_dir,
        prior_graph_dir,
    ]:
      os.makedirs(dir_path, exist_ok=True)

    pd.concat(all_3a_details, ignore_index=True).to_csv(
        os.path.join(phase4_dir, "consolidated_3a_detail.csv"), index=False
    )
    df_3b_concat = pd.concat(all_best_models_3b, ignore_index=True)
    is_lag = df_3b_concat["Channel"].str.contains(r"_l\d+$", regex=True)
    sum_input = df_3b_concat.loc[~is_lag, "total_input"].sum()
    sum_kpi = df_3b_concat["total_estimated_incremental_KPI"].sum()

    total_row_dict = {
        "Channel": "Total",
        "Model Group": "-",
        "x_col": "-",
        "y_col": self.y_col,
        "covariates_for_cate": "-",
        "cate_structure": "-",
        "cate_df": np.nan,
        "cate_degree": np.nan,
        "cate_intercept": np.nan,
        "total_input": sum_input,
        "total_estimated_incremental_KPI": sum_kpi,
        "total_estimated_roi": sum_kpi / sum_input if sum_input > 0 else 0,
        "combined_error_median": np.nan,
        "combined_error_mean": np.nan,
        "combined_error_sd": np.nan,
        "best_model": 1,
    }

    if self.ground_truth_existence:
      y_true_ts = np.zeros(len(df))
      y_pred_ts = np.zeros(len(df))
      for i, c in enumerate(self.d_cols):
        b_data = best_model_data_dict[c]["df_1a"]
        kpi_col = f"estimated_incremental_KPI_{c}"
        gt_col = self.ground_truth_effect_column[i]
        y_pred_ts += b_data[kpi_col].fillna(0).values
        if gt_col in b_data.columns:
          y_true_ts += b_data[gt_col].fillna(0).values
      total_row_dict.update(
          metrics.calculate_ground_truth_metrics(
              pd.Series(y_true_ts), pd.Series(y_pred_ts), sum_input
          )
      )
    else:
      total_row_dict.update({
          "true_total_roi": np.nan,
          "SMAPE": np.nan,
          "R_squared": np.nan,
          "MAE": np.nan,
          "RMSE": np.nan,
      })

    pd.concat(
        [df_3b_concat, pd.DataFrame([total_row_dict])], ignore_index=True
    ).to_csv(
        os.path.join(phase4_dir, "consolidated_3b_summary.csv"), index=False
    )

    df_1a_consolidated = df.copy()
    total_est_sales, total_lower_ci, total_upper_ci = (
        np.zeros(len(df)),
        np.zeros(len(df)),
        np.zeros(len(df)),
    )
    total_input_series = np.zeros(len(df))
    total_gt_sales = np.zeros(len(df)) if self.ground_truth_existence else None
    individual_kpi_cols = []
    cate_eq_dict = {}

    for idx, (d_col, b_data) in enumerate(best_model_data_dict.items()):
      c_kpi, c_low, c_upp = (
          f"estimated_incremental_KPI_{d_col}",
          f"estimated_incremental_KPI_{d_col}_2.5%",
          f"estimated_incremental_KPI_{d_col}_97.5%",
      )
      df_1a_consolidated[c_kpi] = b_data["df_1a"][c_kpi].values
      df_1a_consolidated[c_low] = b_data["df_1a"][c_low].values
      df_1a_consolidated[c_upp] = b_data["df_1a"][c_upp].values
      individual_kpi_cols.append(c_kpi)

      total_est_sales += b_data["df_1a"][c_kpi].values
      total_lower_ci += b_data["df_1a"][c_low].values
      total_upper_ci += b_data["df_1a"][c_upp].values
      if not re.search(r"_l\d+$", d_col, flags=re.IGNORECASE):
        total_input_series += b_data["df_1a"][d_col].values
      if self.ground_truth_existence:
        total_gt_sales += (
            b_data["df_1a"][self.ground_truth_effect_column[idx]]
            .fillna(0)
            .values
        )
      if "cate_coef_dict" in b_data:
        cate_eq_dict[d_col] = self.format_cate_equation(
            b_data["cate_coef_dict"]
        )

    df_1a_consolidated["total_estimated_incremental_KPI"] = total_est_sales
    df_1a_consolidated["total_estimated_incremental_KPI_2.5%"] = total_lower_ci
    df_1a_consolidated["total_estimated_incremental_KPI_97.5%"] = total_upper_ci
    df_1a_consolidated["total_input_aggregate"] = total_input_series
    consolidated_gt_col = (
        "total_ground_truth_KPI" if self.ground_truth_existence else None
    )
    if consolidated_gt_col:
      df_1a_consolidated[consolidated_gt_col] = total_gt_sales

    if self.optimize:
      if len(self.d_cols) > 1:
        warnings.warn(
            "Optimization with multiple treatments is not yet supported."
        )
      elif not self.simple_optimization:
        warnings.warn("Detailed optimization is not yet supported.")

    for d_col in self.d_cols:
      ttype = self.treatment_types[d_col]
      if ttype in ["percentage", "impressions", "spend"]:
        financial_spend_col = self.treatment_spend_cols[d_col]

        if (
            financial_spend_col
            and financial_spend_col in df_1a_consolidated.columns
        ):
          kpi_col = f"estimated_incremental_KPI_{d_col}"
          spend_vals = df_1a_consolidated[financial_spend_col].values
          kpi_vals = df_1a_consolidated[kpi_col].values
          df_1a_consolidated[f"roi_amount_based_{d_col}"] = np.divide(
              kpi_vals,
              spend_vals,
              out=np.zeros_like(kpi_vals, dtype=float),
              where=spend_vals != 0,
          )

          roi_dir = os.path.join(phase4_dir, "roi_output", d_col)
          if ttype != "spend":
            plots.generate_roi_charts(
                df_1a_consolidated,
                self.date_col,
                self.geo_col,
                self.sales_col,
                financial_spend_col,
                kpi_col,
                roi_dir,
                item_col=getattr(self, "item_col", None),
            )

          if (
              self.optimize
              and self.simple_optimization
              and len(self.d_cols) <= 1
          ):
            if self.date_col and self.date_col in df_1a_consolidated.columns:
              min_d = df_1a_consolidated[self.date_col].min()
              max_d = df_1a_consolidated[self.date_col].max()
            else:
              min_d, max_d = None, None

            opt_periods = (
                self.opt_periods
                if self.opt_periods is not None
                else (min_d, max_d)
            )

            optimization.optimize(
                df_1a_consolidated=df_1a_consolidated,
                treatment=d_col,
                treatment_amt=financial_spend_col,
                periods=opt_periods,
                threshold_roi=self.opt_threshold_roi,
                other_cost_variables=self.opt_other_cost_variables,
                target_geo="all",
                target_item="all",
                date_col=self.date_col,
                geo_col=self.geo_col,
                item_col=getattr(self, "item_col", None),
                sales_col=self.sales_col,
                out_dir=roi_dir,
            )

          # Updated Prior for roi spend
          if self.date_col and self.date_col in df_1a_consolidated.columns:
            ts_promo = df_1a_consolidated.groupby(self.date_col).sum()
          else:
            ts_promo = df_1a_consolidated

          if financial_spend_col in ts_promo.columns:
            promo_spend_series = ts_promo[financial_spend_col]
            promo_kpi_series = ts_promo[kpi_col]
            roi_mask_p = promo_spend_series > 0
            roi_data_p = (
                (
                    promo_kpi_series.loc[roi_mask_p]
                    / promo_spend_series.loc[roi_mask_p]
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            contrib_mask_p = (promo_spend_series > 0) & (
                ts_promo[self.y_col] > 0
            )
            contrib_data_p = (
                (
                    promo_kpi_series.loc[contrib_mask_p]
                    / ts_promo.loc[contrib_mask_p, self.y_col]
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            updated_prior_dir = os.path.join(roi_dir, "006_updated_prior")
            os.makedirs(updated_prior_dir, exist_ok=True)
            plots.plot_prior_distributions(
                roi_data_p,
                contrib_data_p,
                os.path.join(
                    updated_prior_dir, f"prior_distributions_{d_col}.png"
                ),
                d_col,
            )
            p_recs = []
            for is_contrib, p_t, p_d in [
                (True, "contribution", contrib_data_p),
                (False, "roi", roi_data_p),
            ]:
              if len(p_d) > 0:
                mu_v, std_v = stats.norm.fit(p_d)
                p_recs.append({
                    "treatment": d_col,
                    "type": p_t,
                    "distribution": "Normal",
                    "mu": mu_v,
                    "sigma": std_v,
                    "alpha": np.nan,
                    "beta": np.nan,
                })

                p_log = p_d[p_d > 0]
                if len(p_log) > 0:
                  shape, _, scale = stats.lognorm.fit(p_log, floc=0)
                  p_recs.append({
                      "treatment": d_col,
                      "type": p_t,
                      "distribution": "LogNormal",
                      "mu": np.log(scale),
                      "sigma": shape,
                      "alpha": np.nan,
                      "beta": np.nan,
                  })

                if is_contrib:
                  p_beta = p_d[(p_d > 0) & (p_d < 1)]
                  if len(p_beta) > 10:
                    a_val, b_val, _, _ = stats.beta.fit(
                        p_beta, floc=0, fscale=1
                    )
                    p_recs.append({
                        "treatment": d_col,
                        "type": p_t,
                        "distribution": "Beta",
                        "mu": np.nan,
                        "sigma": np.nan,
                        "alpha": a_val,
                        "beta": b_val,
                    })

            if p_recs:
              pd.DataFrame(p_recs).to_csv(
                  os.path.join(updated_prior_dir, "prior_parameters.csv"),
                  index=False,
              )

    df_1a_consolidated.to_csv(
        os.path.join(phase4_dir, "consolidated_1a_full_df.csv"), index=False
    )

    geo_summary_df = self._generate_dim_summary(
        self.geo_col, df_1a_consolidated, best_model_data_dict, df_3b_concat
    )
    if geo_summary_df is not None:
      geo_summary_df.to_csv(
          os.path.join(phase4_dir, "consolidated_3b_summary_geo.csv"),
          index=False,
      )

    item_summary_df = self._generate_dim_summary(
        self.item_col, df_1a_consolidated, best_model_data_dict, df_3b_concat
    )
    if item_summary_df is not None:
      item_summary_df.to_csv(
          os.path.join(phase4_dir, "consolidated_3b_summary_item.csv"),
          index=False,
      )

    if self.date_col and self.date_col in df_1a_consolidated.columns:
      print("  ├─ Generating timeseries & ROI charts...", flush=True)
      plots.plot_incremental_kpi_line(
          df_1a_consolidated,
          self.date_col,
          "total_estimated_incremental_KPI",
          "total_estimated_incremental_KPI_2.5%",
          "total_estimated_incremental_KPI_97.5%",
          os.path.join(
              total_graph_dir, "total_incremental_KPI_timeseries_line.png"
          ),
          "Total Incremental KPI Timeseries (Line)",
          consolidated_gt_col,
          actual_y_col=self.y_col,
      )
      plots.plot_incremental_kpi_bar_stacked(
          df_1a_consolidated,
          self.date_col,
          individual_kpi_cols,
          os.path.join(
              total_graph_dir, "total_incremental_KPI_timeseries_bar.png"
          ),
          "Total Incremental KPI Component Stacked (Bar)",
          consolidated_gt_col,
          actual_y_col=self.y_col,
      )
      plots.plot_roi_timeseries_bar(
          df_1a_consolidated,
          self.date_col,
          "total_input_aggregate",
          "total_estimated_incremental_KPI",
          "total_estimated_incremental_KPI_2.5%",
          "total_estimated_incremental_KPI_97.5%",
          os.path.join(total_graph_dir, "total_ROI_timeseries_bar.png"),
          "Total Input, KPI, and ROI Timeseries",
          consolidated_gt_col,
      )

      for e_col, e_dir in [
          (self.geo_col, geo_graph_dir),
          (self.item_col, item_graph_dir),
      ]:
        if e_col and e_col in df_1a_consolidated.columns:
          for i, e_val in enumerate(
              df_1a_consolidated[e_col].dropna().unique(), 1
          ):
            e_df = df_1a_consolidated[df_1a_consolidated[e_col] == e_val]
            plots.plot_incremental_kpi_line(
                e_df,
                self.date_col,
                "total_estimated_incremental_KPI",
                "total_estimated_incremental_KPI_2.5%",
                "total_estimated_incremental_KPI_97.5%",
                os.path.join(
                    e_dir,
                    f"{i:03d}_{e_val}_incremental_KPI_timeseries_line.png",
                ),
                f"[{e_val}] Total Incremental KPI",
                consolidated_gt_col,
                actual_y_col=self.y_col,
            )
            plots.plot_incremental_kpi_bar_stacked(
                e_df,
                self.date_col,
                individual_kpi_cols,
                os.path.join(
                    e_dir, f"{i:03d}_{e_val}_incremental_KPI_timeseries_bar.png"
                ),
                f"[{e_val}] Stacked Component KPI",
                consolidated_gt_col,
                actual_y_col=self.y_col,
            )
            plots.plot_roi_timeseries_bar(
                e_df,
                self.date_col,
                "total_input_aggregate",
                "total_estimated_incremental_KPI",
                "total_estimated_incremental_KPI_2.5%",
                "total_estimated_incremental_KPI_97.5%",
                os.path.join(e_dir, f"{i:03d}_{e_val}_ROI_timeseries_bar.png"),
                f"[{e_val}] Input, KPI & ROI",
                consolidated_gt_col,
            )
          plots.plot_entity_roi_comparison(
              df_1a_consolidated,
              e_col,
              "total_input_aggregate",
              "total_estimated_incremental_KPI",
              "total_estimated_incremental_KPI_2.5%",
              "total_estimated_incremental_KPI_97.5%",
              os.path.join(e_dir, f"000_{e_col}_ROI_comparison.png"),
              f"Cross-{e_col} ROI & Impact Comparison",
              consolidated_gt_col,
          )

    all_cate_covs = list(set().union(*self.covariates_for_cate_list))
    cont_cols = [c for c in all_cate_covs if df[c].nunique() > 2]
    bin_cols = [c for c in all_cate_covs if df[c].nunique() <= 2]
    print("  ├─ Generating CATE Scatter Matrix...", flush=True)
    plots.plot_cate_scatter_matrix(
        df_1a_consolidated,
        self.d_cols,
        cont_cols,
        bin_cols,
        cate_eq_dict,
        os.path.join(cate_graph_dir, "cate_scatter_matrix.png"),
    )

    print("  ├─ Generating Prior Distributions & CSV...", flush=True)
    prior_records = []

    if self.date_col and self.date_col in df_1a_consolidated.columns:
      ts_df = df_1a_consolidated.groupby(self.date_col).sum()
    else:
      ts_df = df_1a_consolidated

    base_channels = {}
    for d_col in self.d_cols:
      base_name = re.sub(r"_l\d+$", "", d_col)
      if base_name not in base_channels:
        base_channels[base_name] = []
      base_channels[base_name].append(d_col)

    for base_name, group_cols in base_channels.items():
      kpi_cols = [f"estimated_incremental_KPI_{c}" for c in group_cols]
      total_inc_kpi = ts_df[kpi_cols].sum(axis=1)

      input_col = (
          ts_df[base_name]
          if base_name in ts_df.columns
          else pd.Series(np.zeros(len(ts_df)), index=ts_df.index)
      )

      roi_mask = input_col > 0
      roi_data = (
          (total_inc_kpi.loc[roi_mask] / input_col.loc[roi_mask])
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
          .values
      )

      contrib_mask = (input_col > 0) & (ts_df[self.y_col] > 0)
      contrib_data = (
          (
              total_inc_kpi.loc[contrib_mask]
              / ts_df.loc[contrib_mask, self.y_col]
          )
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
          .values
      )

      plots.plot_prior_distributions(
          roi_data,
          contrib_data,
          os.path.join(prior_graph_dir, f"prior_distributions_{base_name}.png"),
          base_name,
      )

      for is_contrib, p_type, p_data in [
          (True, "contribution", contrib_data),
          (False, "roi", roi_data),
      ]:
        if len(p_data) == 0:
          continue
        mu_val, std = stats.norm.fit(p_data)
        prior_records.append({
            "treatment": base_name,
            "type": p_type,
            "distribution": "Normal",
            "mu": mu_val,
            "sigma": std,
            "alpha": np.nan,
            "beta": np.nan,
        })

        p_log = p_data[p_data > 0]
        if len(p_log) > 0:
          shape, _, scale = stats.lognorm.fit(p_log, floc=0)
          prior_records.append({
              "treatment": base_name,
              "type": p_type,
              "distribution": "LogNormal",
              "mu": np.log(scale),
              "sigma": shape,
              "alpha": np.nan,
              "beta": np.nan,
          })

        if is_contrib:
          p_beta = p_data[(p_data > 0) & (p_data < 1)]
          if len(p_beta) > 10:
            a_val, b_val, _, _ = stats.beta.fit(p_beta, floc=0, fscale=1)
            prior_records.append({
                "treatment": base_name,
                "type": p_type,
                "distribution": "Beta",
                "mu": np.nan,
                "sigma": np.nan,
                "alpha": a_val,
                "beta": b_val,
            })

    if prior_records:
      pd.DataFrame(prior_records).to_csv(
          os.path.join(prior_graph_dir, "prior_parameters.csv"), index=False
      )

    if self.run_sensitivity:
      print("  ├─ Running Sensitivity Analysis Contours...", flush=True)
      for idx, d_col in enumerate(self.d_cols):
        dml_model = best_model_data_dict[d_col]["model_obj"]
        benchmarks = (
            self.x_cols_list[idx] if idx < len(self.x_cols_list) else []
        )

        top_bench = None
        if benchmarks:
          num_bench = [
              b
              for b in benchmarks
              if pd.api.types.is_numeric_dtype(df[b]) and b != d_col
          ]
          if num_bench:
            top_bench = [df[num_bench].corrwith(df[d_col]).abs().idxmax()]

        if "total" in self.sensitivity_scope:
          plots.plot_sensitivity_contour(
              dml_model,
              os.path.join(
                  sens_graph_dir, f"{d_col}_total_sensitivity_contour.png"
              ),
              f"{d_col} (Total)",
              top_bench,
          )
        for scope, col in [("geo", self.geo_col), ("item", self.item_col)]:
          if scope in self.sensitivity_scope and col in df.columns:
            try:
              dml_model.gate(groups=pd.get_dummies(df[col], drop_first=True))
              plots.plot_sensitivity_contour(
                  dml_model,
                  os.path.join(
                      sens_graph_dir, f"{d_col}_{scope}_sensitivity_contour.png"
                  ),
                  f"{d_col} ({scope.capitalize()} Level)",
                  top_bench,
              )
            except (
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
                RuntimeError,
            ):
              pass

    print(
        "======================================================================\n★★★"
        " Full Pipeline Execution Finished ★★★\n",
        flush=True,
    )
