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

"""Module for extracting outputs and generating an HTML report."""

import base64
import json
import os
import pandas as pd


def generate_html_report(output_dir: str) -> str | None:
  """Generates an HTML report for DoubleML pipeline results."""

  image_dict = {}
  for subdir in ["graph", "roi_output"]:
    s_dir = os.path.join(output_dir, "step4_consolidated_results", subdir)
    if os.path.exists(s_dir):
      for root, _, files in os.walk(s_dir):
        for file in files:
          if file.endswith(".png"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, output_dir).replace("\\", "/")
            try:
              with open(full_path, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                image_dict[rel_path] = f"data:image/png;base64,{b64_data}"
            except OSError:
              pass

  def get_b64(rel_path):
    return image_dict.get(rel_path, "")

  metadata_path = os.path.join(output_dir, "metadata.json")
  if not os.path.exists(metadata_path):
    print(f"Metadata file not found at {metadata_path}.")
    return

  with open(metadata_path, "r") as f:
    metadata = json.load(f)

  treatment_cols = metadata.get("treatment_cols", [])
  outcome_col = metadata.get("outcome_col", "")
  date_col_min = str(metadata.get("date_col_min", ""))[:10]
  date_col_max = str(metadata.get("date_col_max", ""))[:10]
  date_col = metadata.get("date_col")
  date_col_str = str(date_col) if date_col else "None"
  population_col = metadata.get("population_col")
  population_col_str = (
      str(population_col) if population_col else "None"
  )
  x_cols_list = metadata.get("x_cols_list", [])
  treatment_types = metadata.get("treatment_types", {})
  treatment_spend_cols = metadata.get("treatment_spend_cols", {})
  treatment_spend_totals = metadata.get("treatment_spend_totals", {})
  geo_col = metadata.get("geo_col")
  geo_col_str = str(geo_col) if geo_col else "None"
  item_col = metadata.get("item_col")
  item_col_str = str(item_col) if item_col else "None"
  geos = metadata.get("geos", [])
  items = metadata.get("items", [])
  peak_month = metadata.get("peak_month")
  if not peak_month:
    peak_month = [3, 7, 11, 12]
  ground_truth_existence = metadata.get("ground_truth_existence", False)
  ground_truth_metrics = metadata.get("ground_truth_metrics")
  optimisation = metadata.get("optimisation", True)

  summary_csv_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "consolidated_3b_summary.csv",
  )
  geo_csv_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "consolidated_3b_summary_geo.csv",
  )
  item_csv_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "consolidated_3b_summary_item.csv",
  )
  df_1a_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "consolidated_1a_full_df.csv",
  )
  prior_parameters_csv_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "graph",
      "005_prior",
      "prior_parameters.csv",
  )
  optimization_metrics_csv_path = os.path.join(
      output_dir,
      "step4_consolidated_results",
      "graph",
      "007_optimisation",
      "optimization_metrics.csv",
  )

  geos = geos or []
  items = items or []

  total_input = 0
  total_kpi = 0
  total_roi = 0.0
  total_input_spend = 0.0
  model_groups, covariates_for_cate, cate_structure = [], [], []
  cate_df, cate_degree, cate_intercept = [], [], []
  summary_metrics_dict = {}
  gt_metrics_dict_extracted = {}

  df_1a = None
  if os.path.exists(df_1a_path):
    try:
      df_1a = pd.read_csv(df_1a_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
      pass

  is_all_other_treatment = False
  if treatment_types:
    is_all_other_treatment = all(
        ttype in ["percentage", "impressions", "price"]
        for ttype in treatment_types.values()
    )

  def extract_metrics_from_row(
      row, key, dim_col=None, dim_val=None, ch="Total"
  ):
    inp = float(row.get("total_input", 0))
    kpi = float(row.get("total_estimated_incremental_KPI", 0))
    roi = float(row.get("total_estimated_roi", 0.0))

    input_val = int(round(inp))
    input_spend = 0.0

    if is_all_other_treatment:
      if ch == "Total":
        if dim_col and df_1a is not None and dim_col in df_1a.columns:
          val_df = df_1a[df_1a[dim_col] == dim_val]
          for s_col in treatment_spend_cols.values():
            if s_col in val_df.columns:
              input_spend += float(val_df[s_col].sum())
        else:
          input_spend = sum(float(v) for v in treatment_spend_totals.values())
      else:
        s_col = treatment_spend_cols.get(ch)
        if dim_col and df_1a is not None and dim_col in df_1a.columns:
          val_df = df_1a[df_1a[dim_col] == dim_val]
          if s_col and s_col in val_df.columns:
            input_spend = float(val_df[s_col].sum())
        else:
          input_spend = float(treatment_spend_totals.get(ch, 0))

      if input_spend > 0:
        roi = kpi / input_spend
      else:
        roi = 0.0

    summary_metrics_dict[key] = {
        "input": input_val,
        "input_spend": int(round(input_spend)) if is_all_other_treatment else 0,
        "kpi": int(round(kpi)),
        "roi": round(roi, 2),
    }

    has_gt = any(k in row for k in ["SMAPE", "R_squared", "MAE", "RMSE"])
    if has_gt:
      gt_metrics_dict_extracted[key] = {
          k: float(row[k]) if pd.notnull(row[k]) else None
          for k in ["SMAPE", "R_squared", "MAE", "RMSE"]
          if k in row
      }

  try:
    if os.path.exists(summary_csv_path):
      df_summary = pd.read_csv(summary_csv_path)
      if "Channel" in df_summary.columns:
        for _, row in df_summary.iterrows():
          ch = str(row["Channel"])
          key = "total" if ch == "Total" else ch
          extract_metrics_from_row(row, key, None, None, ch)

          if ch == "Total":
            total_input = summary_metrics_dict["total"]["input"]
            total_kpi = summary_metrics_dict["total"]["kpi"]
            total_roi = summary_metrics_dict["total"]["roi"]
            total_input_spend = summary_metrics_dict["total"]["input_spend"]

        non_total_rows = df_summary[df_summary["Channel"] != "Total"]
        model_groups = (
            non_total_rows.get("Model Group", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
        covariates_for_cate = (
            non_total_rows.get("covariates_for_cate", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
        cate_structure = (
            non_total_rows.get("cate_structure", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
        cate_df = (
            non_total_rows.get("cate_df", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
        cate_degree = (
            non_total_rows.get("cate_degree", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
        cate_intercept = (
            non_total_rows.get("cate_intercept", pd.Series())
            .dropna()
            .unique()
            .tolist()
        )
  except (
      OSError,
      pd.errors.EmptyDataError,
      pd.errors.ParserError,
      ValueError,
      KeyError,
  ) as e:
    print(f"Error reading summary csv: {e}")

  def extract_dim_metrics(csv_path, dim_col, is_geo=True):
    if not os.path.exists(csv_path) or not dim_col:
      return
    try:
      df_dim = pd.read_csv(csv_path)
      if dim_col not in df_dim.columns:
        return
      for val in df_dim[dim_col].unique():
        df_val = df_dim[df_dim[dim_col] == val]
        for _, row in df_val.iterrows():
          ch = str(row.get("Channel", "Total"))
          if ch == "Total":
            key = f"geo:{val}" if is_geo else f"item:{val}"
          else:
            key = f"geo:{val}:{ch}" if is_geo else f"item:{val}:{ch}"
          extract_metrics_from_row(row, key, dim_col, val, ch)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        KeyError,
    ) as e:
      print(f"Error reading {csv_path}: {e}")

  extract_dim_metrics(geo_csv_path, geo_col, is_geo=True)
  extract_dim_metrics(item_csv_path, item_col, is_geo=False)

  def csv_to_html_table(csv_path):
    if csv_path and os.path.exists(csv_path):
      try:
        df = pd.read_csv(csv_path)
        return df.to_html(index=False, classes="data-table")
      except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return "<p>Error loading table data.</p>"
    return "<p>No table data available.</p>"

  prior_table = csv_to_html_table(prior_parameters_csv_path)

  gt_metrics_json_str = "{}"
  if gt_metrics_dict_extracted:
    gt_metrics_json_str = json.dumps(gt_metrics_dict_extracted)
  elif ground_truth_existence and ground_truth_metrics:
    if "SMAPE" in ground_truth_metrics or "MAE" in ground_truth_metrics:
      gt_metrics_json_str = json.dumps({"total": ground_truth_metrics})
    else:
      gt_metrics_json_str = json.dumps(ground_truth_metrics)

  has_spend = any(t == "spend" for t in treatment_types.values())
  has_other_treatment = any(
      t in ["percentage", "impressions", "price"]
      for t in treatment_types.values()
  )

  dim_opts = "<option value='total'>Total</option>"
  for g in geos:
    dim_opts += f"<option value='geo:{g}'>geo:{g}</option>"
  for i in items:
    dim_opts += f"<option value='item:{i}'>item:{i}</option>"

  geo_opts = "".join([f"<option value='{g}'>{g}</option>" for g in geos])
  if geos:
    geo_opts = "<option value='total'>Total</option>" + geo_opts

  item_opts = "".join([f"<option value='{i}'>{i}</option>" for i in items])
  if items:
    item_opts = "<option value='total'>Total</option>" + item_opts

  month_opts_1_12 = "<option value='total'>Total</option>"
  for i in range(1, 13):
    month_opts_1_12 += f"<option value='month:{i}'>month:{i}</option>"

  treatment_opts = "".join(
      f"<option value='{t}'>{t}</option>" for t in treatment_cols
  )
  if len(treatment_cols) == 1:
    display_str = treatment_cols[0]
  else:
    display_str = f"[{', '.join(treatment_cols)}]"
  treatment_opts_005 = (
      f'<option value="{treatment_cols}">{display_str}</option>'
  )

  geo_index_map = {g: f"{(i+1):03d}" for i, g in enumerate(geos)}
  item_index_map = {item: f"{(i+1):03d}" for i, item in enumerate(items)}

  treatment_str = ", ".join(treatment_cols)
  treatment_types_str = ", ".join(
      [f"{k}: {v}" for k, v in treatment_types.items()]
  )
  treatment_vars_html = (
      f"<strong>Treatment Variable(s):</strong> {treatment_str} <br>\n"
      "            <strong>Treatment Types:</strong> "
      f"{treatment_types_str}"
  )
  if has_other_treatment and treatment_spend_cols:
    spend_cols_list = [f"{k}: {v}" for k, v in treatment_spend_cols.items()]
    spend_cols_str = ", ".join(spend_cols_list)
    treatment_vars_html += (
        " <br>\n            <strong>Treatment Spend Col(s):</strong> "
        f"{spend_cols_str}"
    )

  if is_all_other_treatment:
    header_metrics = (
        f"Input (treatment): {total_input} | "
        f"Input (treatment spend): {int(round(total_input_spend))} | "
        f"Incremental KPI: {total_kpi} | "
        f"ROI (incremental KPI / treatment spend): {total_roi:.2f}"
    )
  else:
    header_metrics = (
        f"Input: {total_input} | Incremental KPI: {total_kpi} | "
        f"ROI: {total_roi:.2f}"
    )

  def extract_opt_metrics(csv_path):
    if not os.path.exists(csv_path):
      return {}
    try:
      df = pd.read_csv(csv_path)
      metrics = {}
      for _, row in df.iterrows():
        geo = str(row.get("target_geo", "all"))
        item = str(row.get("target_item", "all"))
        key = "total"
        if geo != "all":
          key = f"geo:{geo}"
        elif item != "all":
          key = f"item:{item}"
        increase_pct = row.get(
            "increase_pct_of_total_cost_of_other_cost_variables", 0
        )
        if pd.notnull(increase_pct):
          increase_pct_str = f"{increase_pct * 100:.2f}%"
        else:
          increase_pct_str = "0.00%"
        metrics[key] = {
            "start": str(row.get("periods_start_date", "")).replace("-", "/"),
            "end": str(row.get("periods_end_date", "")).replace("-", "/"),
            "cost": int(
                round(float(row.get("total_cost_of_treatment_in_periods", 0)))
            ),
            "opt_cost": int(
                round(
                    float(
                        row.get(
                            "optimised_total_cost_of_treatment_in_periods", 0
                        )
                    )
                )
            ),
            "reduction": int(
                round(float(row.get("reduction_of_cost_of_treatment", 0)))
            ),
            "other_vars": str(row.get("other_cost_variables", "[]")),
            "other_cost": int(
                round(float(row.get("total_cost_of_other_cost_variables", 0)))
            ),
            "increase_pct": increase_pct_str,
        }
      return metrics
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as e:
      print(f"Error reading {csv_path}: {e}")
      return {}

  trt_opt_metrics = {}
  for t, ttype in treatment_types.items():
    if ttype in ["percentage", "impressions", "price"]:
      path = os.path.join(
          output_dir,
          "step4_consolidated_results",
          "roi_output",
          t,
          "007_optimisation",
          "optimization_metrics.csv",
      )
      trt_opt_metrics[t] = extract_opt_metrics(path)

  trt_opt_metrics["main"] = extract_opt_metrics(optimization_metrics_csv_path)

  opt_metrics_json = json.dumps(trt_opt_metrics)

  html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DoubleML Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; color: #2e7d32; background-color: #f9f9f9; padding: 20px; }}
        .top-banner {{ background-color: #1b5e20; color: white; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; border-radius: 8px; margin-bottom: 20px; }}
        .top-banner a {{ color: #c8e6c9; text-decoration: none; }}
        .top-banner a:hover {{ text-decoration: underline; }}
        .header-bg {{ background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .row-top {{ font-size: 32px; font-weight: bold; margin-bottom: 10px; color: #1b5e20; }}
        .row-mid {{ font-size: 16px; margin-bottom: 5px; color: #388e3c; line-height: 1.5; }}
        .tab-buttons {{ overflow: hidden; border-bottom: 2px solid #2e7d32; margin-top: 20px; }}
        .tab-button {{ background-color: #e8f5e9; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; font-size: 16px; font-weight: bold; color: #2e7d32; transition: 0.3s; margin-right: 5px; border-radius: 5px 5px 0 0; }}
        .tab-button:hover {{ background-color: #c8e6c9; }}
        .tab-button.active {{ background-color: #2e7d32; color: white; }}
        .tab-content {{ display: none; padding: 20px; border: 1px solid #c8e6c9; border-top: none; background-color: white; }}
        .data-table {{ border-collapse: collapse; width: 50%; margin-top: 15px; margin-bottom: 15px; }}
        .data-table th, .data-table td {{ text-align: left; padding: 8px; border: 1px solid #c8e6c9; color: #000; }}
        .data-table th {{ background-color: #2e7d32; color: white; }}
        select {{ font-size: 16px; padding: 5px; margin-right: 10px; border: 1px solid #2e7d32; color: #2e7d32; border-radius: 4px; }}
        .chart-img {{ max-width: 100%; height: auto; margin-top: 15px; cursor: zoom-in; border: 1px solid #e8f5e9; border-radius: 4px; transition: transform 0.2s; }}
        .chart-img:hover {{ box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); }}
        .modal-content {{ margin: auto; display: block; max-width: 90%; max-height: 90%; margin-top: 2%; }}
        .close {{ position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }}
    </style>
</head>
<body>
    
    <div id="imageModal" class="modal" onclick="this.style.display='none'">
      <span class="close">&times;</span>
      <img class="modal-content" id="expandedImg">
    </div>

    <div class="top-banner">
        doubleml-pipeline Output Summary (<a href="https://github.com/google/doubleml-pipeline" target="_blank">https://github.com/google/doubleml-pipeline</a>)
    </div>

    <div class="header-bg">
        <div class="row-top">
            {header_metrics}
        </div>
        <div class="row-mid">
            {treatment_vars_html} <br>
            <strong>Outcome Variable:</strong> {outcome_col} <br>
            <strong>Data Period:</strong> {date_col_min} to {date_col_max} <br>
            <strong>Date Col:</strong> {date_col_str} | <strong>Population Col:</strong> {population_col_str} <br>
            <strong>Geo Col:</strong> {geo_col_str} | <strong>Item Col:</strong> {item_col_str} <br>
            <strong>X Cols List:</strong> {', '.join(str(x) for x in x_cols_list)}
        </div>
        <div class="row-mid">
            <strong>Model Group(s):</strong> {', '.join(map(str, model_groups))} <br>
            <strong>Model Naming Convention Note:</strong> [Y-model]_[T-model]_[number_of_folds]_[number_of_repetitions]_[time_budget]_[number_of_covariates_for_cate]. <br>
            <strong>Covariates for CATE:</strong> {', '.join(map(str, covariates_for_cate))} <br>
            <strong>CATE Structure:</strong> {', '.join(map(str, cate_structure))} <br>
            <strong>CATE DF:</strong> {', '.join(map(str, cate_df))} <br>
            <strong>CATE Degree:</strong> {', '.join(map(str, cate_degree))} <br>
            <strong>CATE Intercept:</strong> {', '.join(map(str, cate_intercept))}
        </div>
    </div>

    <div class="tab-buttons">
        <button class="tab-button active" onclick="openTab(event, 'tab_001_total')">001_total</button>
        <button class="tab-button" onclick="openTab(event, 'tab_002_geo')">002_geo</button>
        <button class="tab-button" onclick="openTab(event, 'tab_003_item')">003_item</button>
        <button class="tab-button" onclick="openTab(event, 'tab_004_cate')">004_cate</button>
        <button class="tab-button" onclick="openTab(event, 'tab_005_prior')">005_prior</button>
        <button class="tab-button" onclick="openTab(event, 'tab_006_sensitivity')">006_sensitivity</button>
"""
  if has_spend:
    html += """        <button class="tab-button" onclick="openTab(event, 'tab_007_optimisation')">007_optimisation</button>\n"""

  if has_other_treatment:
    for t, ttype in treatment_types.items():
      if ttype in ["percentage", "impressions", "price"]:
        html += f"""
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_001')">roi_output_001_time_series_charts ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_002')">roi_output_002_result_in_sale_periods ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_003')">roi_output_003_roi_by_month ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_004')">roi_output_004_roi_by_geo ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_005')">roi_output_005_roi_by_item ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_006')">roi_output_006_updated_prior ({t})</button>
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_007')">roi_output_007_optimisation ({t})</button>
"""

  html += """    </div>\n\n"""

  html += f"""
    <div id="tab_001_total" class="tab-content" style="display:block;">
        <h3>Total Results</h3>
        <select id="sel_001" onchange="updateImg001()">
            <option value='total_ROI_timeseries_bar'>ROI_timeseries</option>
            <option value='total_incremental_KPI_timeseries_line'>timeseries_line</option>
            <option value='total_incremental_KPI_timeseries_bar'>timeseries_bar</option>
        </select>
        <br>
        <div id="metrics_001"></div>
        <img id="img_001" class="chart-img" src="{get_b64('step4_consolidated_results/graph/001_total/total_ROI_timeseries_bar.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""

  if geo_col and geos:
    html += f"""
    <div id="tab_002_geo" class="tab-content">
        <h3>Geo Results</h3>
        <select id="sel_002_fmt" onchange="updateImg002()">
            <option value='ROI_timeseries'>ROI_timeseries</option>
            <option value='timeseries_line'>timeseries_line</option>
            <option value='timeseries_bar'>timeseries_bar</option>
        </select>
        <select id="sel_002_geo" onchange="updateImg002()">
            {geo_opts}
        </select>
        <br>
        <div id="metrics_002"></div>
        <img id="img_002" class="chart-img" src="{get_b64(f'step4_consolidated_results/graph/002_geo/000_{geo_col}_ROI_comparison.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
  else:
    html += """
    <div id="tab_002_geo" class="tab-content">
        <h3>Geo Results</h3>
        <p>No table data available.</p>
    </div>
"""

  if item_col and items:
    html += f"""
    <div id="tab_003_item" class="tab-content">
        <h3>Item Results</h3>
        <select id="sel_003_fmt" onchange="updateImg003()">
            <option value='ROI_timeseries'>ROI_timeseries</option>
            <option value='timeseries_line'>timeseries_line</option>
            <option value='timeseries_bar'>timeseries_bar</option>
        </select>
        <select id="sel_003_item" onchange="updateImg003()">
            {item_opts}
        </select>
        <br>
        <div id="metrics_003"></div>
        <img id="img_003" class="chart-img" src="{get_b64(f'step4_consolidated_results/graph/003_item/000_{item_col}_ROI_comparison.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
  else:
    html += """
    <div id="tab_003_item" class="tab-content">
        <h3>Item Results</h3>
        <p>No table data available.</p>
    </div>
"""

  html += f"""
    <div id="tab_004_cate" class="tab-content">
        <h3>CATE Estimation</h3>
        <img class="chart-img" src="{get_b64('step4_consolidated_results/graph/004_cate/cate_scatter_matrix.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""

  prior_images = []
  for t in treatment_cols:
    img_path = f"step4_consolidated_results/graph/005_prior/prior_distributions_{t}.png"
    prior_images.append(
        f'<img class="chart-img" src="{get_b64(img_path)}"'
        ' onclick="expandImage(this.src)"'
        " onerror=\"this.style.display='none'\">"
    )
  prior_images_html = "".join(prior_images)

  html += f"""
    <div id="tab_005_prior" class="tab-content">
        <h3>Prior Distributions</h3>
        <select id="sel_005_trt">
            {treatment_opts_005}
        </select>
        <br>
        {prior_images_html}
        {prior_table}
    </div>
"""

  html += f"""
    <div id="tab_006_sensitivity" class="tab-content">
        <h3>Sensitivity Analysis</h3>
        <select id="sel_006_trt" onchange="updateImgByPath('img_006', 'step4_consolidated_results/graph/006_sensitivity/' + this.value + '_total_sensitivity_contour.png')">
            {treatment_opts}
        </select>
        <br>
        <img id="img_006" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""

  if has_spend:
    html += """
    <div id="tab_007_optimisation" class="tab-content">
        <h3>Optimisation</h3>
"""
    if len(treatment_cols) > 1:
      html += (
          "<p>Optimization for multiple treatments is currently not available"
          " in doubleml-pipeline.</p>\n"
      )
    elif not optimisation:
      html += (
          "<p>Optimization is skipped because 'optimize' is set to False.</p>\n"
      )
    else:
      if geos or items:
        html += f"""
        <select id="sel_007_dim" onchange="updateImgOpt007_main()">
            {dim_opts}
        </select>
"""
      html += """
        <br>
        <div id="opt_metrics_007_main"></div>
        <img id="img_007" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
"""
    html += "    </div>\n"

  if has_other_treatment:
    for t, ttype in treatment_types.items():
      if ttype in ["percentage", "impressions", "price"]:
        bpath = f"step4_consolidated_results/roi_output/{t}"

        trt_prior_table = csv_to_html_table(
            os.path.join(
                output_dir,
                "step4_consolidated_results",
                "roi_output",
                t,
                "006_updated_prior",
                "prior_parameters.csv",
            )
        )

        html += f"""
    <div id="tab_{t}_roi_output_001" class="tab-content">
        <h3>ROI Output 001: Time Series Charts ({t})</h3>
        <select id="sel_{t}_001_chart" onchange="updateImgOpt001('{t}')">
            <option value="spend_vs_est">spend_vs_est</option>
            <option value="sales_and_spend">sales_and_spend</option>
        </select>
        <select id="sel_{t}_001_dim" onchange="updateImgOpt001('{t}')">
            {dim_opts}
        </select>
        <br>
        <div id="metrics_{t}_001"></div>
        <img id="img_{t}_001" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
        html += f"""
    <div id="tab_{t}_roi_output_002" class="tab-content">
        <h3>ROI Output 002: Result in Sale Periods ({t})</h3>
        <p>peak_month = {peak_month} in CausalWorkflowOrchestrator()</p>
        <select id="sel_{t}_002_dim" onchange="updateImgOpt002('{t}')">
            {dim_opts}
        </select>
        <br>
        <img id="img_{t}_002" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
        html += f"""
    <div id="tab_{t}_roi_output_003" class="tab-content">
        <h3>ROI Output 003: ROI by Month ({t})</h3>
        <p>peak_month = {peak_month} in CausalWorkflowOrchestrator()</p>
        <select id="sel_{t}_003_dim" onchange="updateImgOpt003('{t}')">
            {dim_opts}
        </select>
        <br>
        <div id="metrics_{t}_003"></div>
        <img id="img_{t}_003" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
        if geo_col:
          html += f"""
    <div id="tab_{t}_roi_output_004" class="tab-content">
        <h3>ROI Output 004: ROI by Geo ({t})</h3>
        <select id="sel_{t}_004_month" onchange="updateImgOpt004('{t}')">
            {month_opts_1_12}
        </select>
        <br>
        <img id="img_{t}_004" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
        if item_col:
          html += f"""
    <div id="tab_{t}_roi_output_005" class="tab-content">
        <h3>ROI Output 005: ROI by Item ({t})</h3>
        <select id="sel_{t}_005_month" onchange="updateImgOpt005('{t}')">
            {month_opts_1_12}
        </select>
        <br>
        <img id="img_{t}_005" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""
        html += f"""
    <div id="tab_{t}_roi_output_006" class="tab-content">
        <h3>ROI Output 006: Updated Prior ({t})</h3>
        <img class="chart-img" src="{get_b64(f'{bpath}/006_updated_prior/prior_distributions_{t}.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        {trt_prior_table}
    </div>
"""
        html += f"""
    <div id="tab_{t}_roi_output_007" class="tab-content">
        <h3>ROI Output 007: Optimisation ({t})</h3>
        <select id="sel_{t}_007_dim" onchange="updateImgOpt007('{t}')">
            {dim_opts}
        </select>
        <br>
        <div id="opt_metrics_{t}_007"></div>
        <img id="img_{t}_007" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""

  image_dict_json = json.dumps(image_dict)
  geo_index_json = json.dumps(geo_index_map)
  item_index_json = json.dumps(item_index_map)
  summary_metrics_json = json.dumps(summary_metrics_dict)

  html += f"""
    <script>
        const imageDict = {image_dict_json};
        const geoIndexMap = {geo_index_json};
        const itemIndexMap = {item_index_json};
        const summaryMetrics = {summary_metrics_json};
        const gtMetrics = {gt_metrics_json_str};
        const metadata_geo_col = "{geo_col}";
        const metadata_item_col = "{item_col}";
        const isAllOtherTreatment = {str(is_all_other_treatment).lower()};
        const optMetricsData = {opt_metrics_json};
        const optThresholdRoi = "{metadata.get('opt_threshold_roi', 'None')}";

        function renderMetrics(divId, key) {{
            var div = document.getElementById(divId);
            if (!div) return;
            var html = '';

            if (summaryMetrics[key]) {{
                var m = summaryMetrics[key];
                if (isAllOtherTreatment) {{
                    html += "<p><strong>Input (treatment):</strong> " + m.input + " | <strong>Input (treatment spend):</strong> " + m.input_spend + " | <strong>Incremental KPI:</strong> " + m.kpi + " | <strong>ROI (incremental KPI / treatment spend):</strong> " + m.roi + "</p>";
                }} else {{
                    html += "<p><strong>Input:</strong> " + m.input + " | <strong>Incremental KPI:</strong> " + m.kpi + " | <strong>ROI:</strong> " + m.roi + "</p>";
                }}
            }}

            if (gtMetrics[key]) {{
                html += "<table class='data-table'><tr><th>Metric</th><th>Value</th></tr>";
                var g = gtMetrics[key];
                for (const k in g) {{
                    var disp = (k === "R_squared") ? "R2" : k;
                    var val = g[k];
                    if (val === null) {{
                        val = "null (no ground truth data)";
                    }} else if (typeof val === 'number') {{
                        val = parseFloat(val.toPrecision(4));
                    }}
                    html += "<tr><td>" + disp + "</td><td>" + val + "</td></tr>";
                }}
                html += "</table>";
            }}
            div.innerHTML = html;
        }}

        function renderOptMetrics(divId, t, dim) {{
            var div = document.getElementById(divId);
            if (!div) return;
            var html = '';
            if (optMetricsData[t] && optMetricsData[t][dim]) {{
                var m = optMetricsData[t][dim];
                html += "<p><strong>Optimisation ROI Threshold:</strong> " + optThresholdRoi + "<br>";
                html += "<strong>Optimisation Period:</strong> " + m.start + " - " + m.end + "<br>";
                html += "<strong>Total Cost of Treatment in the Period:</strong> " + m.cost + "<br>";
                html += "<strong>Optimised Total Cost of Treatment in the Period:</strong> " + m.opt_cost + "<br>";
                html += "<strong>Reduction of Cost of Treatment:</strong> " + m.reduction + "<br>";
                html += "<strong>Other Cost Variables:</strong> " + m.other_vars + "<br>";
                html += "<strong>Total Cost of Other Cost Variables:</strong> " + m.other_cost + "<br>";
                html += "<strong>Increase Percentage of Total Cost of Other Cost Variables:</strong> " + m.increase_pct + "</p>";
            }}
            div.innerHTML = html;
        }}

        function updateImgByPath(imgId, path) {{
            var img = document.getElementById(imgId);
            if (img) {{
                if (imageDict[path]) {{
                    img.src = imageDict[path];
                    img.style.display = 'block';
                }} else {{
                    img.src = '';
                    img.style.display = 'none';
                }}
            }}
        }}

        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}

        function updateImg001() {{
            var val = document.getElementById('sel_001') ? document.getElementById('sel_001').value : 'total_ROI_timeseries_bar';
            updateImgByPath('img_001', 'step4_consolidated_results/graph/001_total/' + val + '.png');
            renderMetrics('metrics_001', 'total');
        }}

        function updateImg002() {{
            var chartType = document.getElementById('sel_002_fmt').value;
            var geo = document.getElementById('sel_002_geo').value;
            if (geo === 'total') {{
                updateImgByPath('img_002', 'step4_consolidated_results/graph/002_geo/000_' + metadata_geo_col + '_ROI_comparison.png');
                renderMetrics('metrics_002', 'total');
            }} else {{
                var suffix = '';
                if (chartType === 'timeseries_bar') suffix = '_incremental_KPI_timeseries_bar';
                else if (chartType === 'timeseries_line') suffix = '_incremental_KPI_timeseries_line';
                else if (chartType === 'ROI_timeseries') suffix = '_ROI_timeseries_bar';
                updateImgByPath('img_002', 'step4_consolidated_results/graph/002_geo/' + geoIndexMap[geo] + '_' + geo + suffix + '.png');
                renderMetrics('metrics_002', 'geo:' + geo);
            }}
        }}

        function updateImg003() {{
            var chartType = document.getElementById('sel_003_fmt').value;
            var item = document.getElementById('sel_003_item').value;
            if (item === 'total') {{
                updateImgByPath('img_003', 'step4_consolidated_results/graph/003_item/000_' + metadata_item_col + '_ROI_comparison.png');
                renderMetrics('metrics_003', 'total');
            }} else {{
                var suffix = '';
                if (chartType === 'timeseries_bar') suffix = '_incremental_KPI_timeseries_bar';
                else if (chartType === 'timeseries_line') suffix = '_incremental_KPI_timeseries_line';
                else if (chartType === 'ROI_timeseries') suffix = '_ROI_timeseries_bar';
                updateImgByPath('img_003', 'step4_consolidated_results/graph/003_item/' + itemIndexMap[item] + '_' + item + suffix + '.png');
                renderMetrics('metrics_003', 'item:' + item);
            }}
        }}

        function updateImgOpt007_main() {{
            var sel = document.getElementById('sel_007_dim');
            if (!sel) return;
            var dim = sel.value;
            var path = 'step4_consolidated_results/graph/007_optimisation/';
            if (dim === 'total') {{
                path += 'total_timeseries_spend_vs_est.png';
            }} else if (dim.startsWith('geo:')) {{
                path += 'geo_' + dim.substring(4) + '_timeseries_spend_vs_est.png';
            }} else if (dim.startsWith('item:')) {{
                path += 'item_' + dim.substring(5) + '_timeseries_spend_vs_est.png';
            }}
            updateImgByPath('img_007', path);
            renderOptMetrics('opt_metrics_007_main', 'main', dim);
        }}

        function updateImgOpt001(t) {{
            var chart = document.getElementById('sel_' + t + '_001_chart').value;
            var dim = document.getElementById('sel_' + t + '_001_dim').value;
            var path = '';
            if (dim === 'total') {{
                path = 'step4_consolidated_results/roi_output/' + t + '/001_time_series_charts/total_timeseries_' + chart + '.png';
            }} else if (dim.startsWith('geo:')) {{
                path = 'step4_consolidated_results/roi_output/' + t + '/001_time_series_charts/geo_' + dim.substring(4) + '_timeseries_' + chart + '.png';
            }} else if (dim.startsWith('item:')) {{
                path = 'step4_consolidated_results/roi_output/' + t + '/001_time_series_charts/item_' + dim.substring(5) + '_timeseries_' + chart + '.png';
            }}
            updateImgByPath('img_' + t + '_001', path);
            var metricKey = dim === 'total' ? t : dim + ':' + t;
            renderMetrics('metrics_' + t + '_001', metricKey);
        }}

        function updateImgOpt002(t) {{
            var dim = document.getElementById('sel_' + t + '_002_dim').value;
            var path = 'step4_consolidated_results/roi_output/' + t + '/002_result_in_sale_periods/';
            if (dim === 'total') {{
                path += 'total_result_in_sale_periods_matrix.png';
            }} else if (dim.startsWith('geo:')) {{
                path += 'geo_' + dim.substring(4) + '_result_in_sale_periods_matrix.png';
            }} else if (dim.startsWith('item:')) {{
                path += 'item_' + dim.substring(5) + '_result_in_sale_periods_matrix.png';
            }}
            updateImgByPath('img_' + t + '_002', path);
        }}

        function updateImgOpt003(t) {{
            var dim = document.getElementById('sel_' + t + '_003_dim').value;
            var path = 'step4_consolidated_results/roi_output/' + t + '/003_roi_by_month/';
            if (dim === 'total') {{
                path += 'total_roi_by_month_matrix.png';
            }} else if (dim.startsWith('geo:')) {{
                path += 'geo_' + dim.substring(4) + '_roi_by_month_matrix.png';
            }} else if (dim.startsWith('item:')) {{
                path += 'item_' + dim.substring(5) + '_roi_by_month_matrix.png';
            }}
            updateImgByPath('img_' + t + '_003', path);
            var metricKey = dim === 'total' ? t : dim + ':' + t;
            renderMetrics('metrics_' + t + '_003', metricKey);
        }}

        function updateImgOpt004(t) {{
            var month = document.getElementById('sel_' + t + '_004_month').value;
            var path = 'step4_consolidated_results/roi_output/' + t + '/004_roi_by_geo/';
            if (month === 'total') {{
                path += 'total_roi_by_geo_matrix.png';
            }} else if (month.startsWith('month:')) {{
                var m = month.substring(6);
                path += 'month_' + m + '_roi_by_geo_matrix.png';
            }}
            updateImgByPath('img_' + t + '_004', path);
        }}

        function updateImgOpt005(t) {{
            var month = document.getElementById('sel_' + t + '_005_month').value;
            var path = 'step4_consolidated_results/roi_output/' + t + '/005_roi_by_item/';
            if (month === 'total') {{
                path += 'total_roi_by_item_matrix.png';
            }} else if (month.startsWith('month:')) {{
                var m = month.substring(6);
                path += 'month_' + m + '_roi_by_item_matrix.png';
            }}
            updateImgByPath('img_' + t + '_005', path);
        }}

        function updateImgOpt007(t) {{
            var dim = document.getElementById('sel_' + t + '_007_dim').value;
            var path = 'step4_consolidated_results/roi_output/' + t + '/007_optimisation/';
            if (dim === 'total') {{
                path += 'total_timeseries_spend_vs_est.png';
            }} else if (dim.startsWith('geo:')) {{
                path += 'geo_' + dim.substring(4) + '_timeseries_spend_vs_est.png';
            }} else if (dim.startsWith('item:')) {{
                path += 'item_' + dim.substring(5) + '_timeseries_spend_vs_est.png';
            }}
            updateImgByPath('img_' + t + '_007', path);
            renderOptMetrics('opt_metrics_' + t + '_007', t, dim);
        }}

        function expandImage(src) {{
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('expandedImg');
            modal.style.display = "block";
            modalImg.src = src;
        }}

        window.onload = function() {{
            updateImg001();
            if(document.getElementById('sel_002_fmt')) updateImg002();
            if(document.getElementById('sel_003_fmt')) updateImg003();
            if(document.getElementById('sel_006_trt')) {{ var sel = document.getElementById('sel_006_trt'); sel.onchange(); }}
            if(document.getElementById('sel_007_geo')) {{ updateImgOpt007_main(); }}
            
            var selects = document.getElementsByTagName('select');
            for(var i=0; i<selects.length; i++) {{
                if (selects[i].id.indexOf('sel_') === 0 && selects[i].onchange) {{
                    if (selects[i].id !== 'sel_001' && selects[i].id !== 'sel_002_fmt' && selects[i].id !== 'sel_002_geo' && selects[i].id !== 'sel_003_fmt' && selects[i].id !== 'sel_003_item' && selects[i].id !== 'sel_005_trt' && selects[i].id !== 'sel_006_trt' && selects[i].id !== 'sel_007_geo') {{
                        selects[i].onchange();
                    }}
                }}
            }}
        }};
    </script>
</body>
</html>
"""

  out_path = os.path.join(output_dir, "report.html")
  with open(out_path, "w") as f:
    f.write(html)

  return out_path
