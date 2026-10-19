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
import re
from typing import Any

from doubleml_pipeline.utils import io as pipeline_io
import pandas as pd


def _sanitize_for_filename(value: str) -> str:
  """Reduces a column name to characters that are safe in a filename."""
  cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._-")
  return cleaned


def _report_filename(
    treatment_cols: list[str],
    outcome_col: str,
    saturation_family: str | None,
) -> str:
  """Builds the report filename ``<treatment>_<outcome>_<family>.html``.

  Multiple treatments are joined with ``-``. The family segment is omitted
  entirely on the legacy path where ``saturation.family`` is unset.

  Args:
    treatment_cols: The treatment columns of the run, in pipeline order.
    outcome_col: The outcome column of the run.
    saturation_family: The saturation family, or ``None``.

  Returns:
    The filename (not a full path).
  """
  treatment_part = "-".join(
      p for p in (_sanitize_for_filename(t) for t in treatment_cols) if p
  )
  parts = [
      p for p in (treatment_part, _sanitize_for_filename(outcome_col)) if p
  ]
  if saturation_family:
    parts.append(_sanitize_for_filename(saturation_family))
  if not parts:
    return "report.html"
  return "_".join(parts) + ".html"


def _png_data_uri(path: str) -> str:
  """Returns a base64 data URI for a PNG file ('' when unreadable)."""
  try:
    with open(path, "rb") as img_file:
      return "data:image/png;base64," + base64.b64encode(
          img_file.read()
      ).decode("utf-8")
  except OSError:
    return ""


def _fmt_num(value, digits: int = 4) -> str:
  """Formats a JSON number for the descriptive block (``n/a`` for null)."""
  if value is None:
    return "n/a"
  try:
    return f"{float(value):.{digits}g}"
  except (TypeError, ValueError):
    return str(value)


def _phase0_tab_html(
    output_dir: str, phase0_meta: dict[str, dict[str, Any]]
) -> str:
  """Builds the ``000_saturation`` tab of the report.

  Embeds ``diag2_rloss_profile.png`` and the alpha profile chart
  ``summary_alpha_profile_<treatment>.png`` (grid construction, seed-wise
  alpha distribution, profiled R-loss curve, fitted saturation curve and the
  key stats) and shows ``decision``, ``decision_reason``, ``alpha_hat``,
  ``half_saturation_T``, ``T_ref``, ``d_bar``, ``se``, ``diag2_ratio`` and
  ``saturation_at_T_ref`` from ``summary_alpha_selection.json`` in the
  descriptive block.

  Args:
    output_dir: The pipeline output root.
    phase0_meta: ``metadata['saturation_phase0']`` (treatment -> info).

  Returns:
    The tab's HTML (a ``tab-content`` div).
  """
  blocks = []
  for treatment, info in phase0_meta.items():
    rel_dir = (info or {}).get("output_dir") or "step0_saturation"
    t_dir = os.path.join(output_dir, rel_dir)
    summary = {}
    summary_path = os.path.join(t_dir, pipeline_io.SUMMARY_JSON)
    if os.path.exists(summary_path):
      try:
        with open(summary_path, "r") as handle:
          summary = json.load(handle)
      except (OSError, ValueError):
        summary = {}
    summary = summary or dict(info or {})
    # The log family's grid variable is T_knee.
    t_half_label = (
        "Marginal-effect-halving point T_knee"
        if summary.get("family") == "log"
        else "Half-saturation point T_1/2"
    )
    fields = [
        ("Decision", summary.get("decision")),
        ("Decision reason", summary.get("decision_reason")),
        ("Selected family", summary.get("selected_family")),
        ("alpha_hat", _fmt_num(summary.get("alpha_hat"), 6)),
        (t_half_label, _fmt_num(summary.get("half_saturation_T"), 6)),
        ("T_ref (reference point of fbar)", _fmt_num(summary.get("T_ref"), 6)),
        # 1-SE linear test: d_bar against the identity candidate
        # L_T; se is the cluster-robust se_obs.
        ("d_bar (L(alpha_hat) - L_T)", _fmt_num(summary.get("d_bar"))),
        ("se (cluster-robust se_obs)", _fmt_num(summary.get("se"))),
        # Diag2 ratio and Diag5 saturation level.
        (
            "diag2_ratio (1-SE flatness, T_1/2 max/min)",
            _fmt_num(summary.get("diag2_ratio"), 3),
        ),
        (
            "saturation_at_T_ref (kappa = f(T_ref; alpha_hat))",
            _fmt_num(summary.get("saturation_at_T_ref"), 3),
        ),
        ("r_T = sd(T~)/sd(T/T_ref)", _fmt_num(summary.get("r_T"), 3)),
    ]
    # What the "normalized treatment T/c" of alpha_hat / T_ref / the charts
    # concretely is: the formula of the scaler divisor c and its meaning.
    if summary.get("divisor_formula"):
      fields.append(
          ("Normalized treatment scale (T/c)", summary.get("divisor_formula"))
      )
    if summary.get("divisor_description"):
      fields.append(("Meaning of T/c", summary.get("divisor_description")))
    rows = " <br>".join(
        f"<strong>{label}:</strong> {value}" for label, value in fields
    )
    # The alpha profile chart is written next to the per-treatment folder
    # (directly in step0_saturation/ with several treatments), so look in
    # both places.
    profile_name = pipeline_io.alpha_profile_png(treatment)
    profile_candidates = [
        os.path.join(t_dir, profile_name),
        os.path.join(os.path.dirname(t_dir.rstrip(os.sep)), profile_name),
    ]
    profile_path = next(
        (p for p in profile_candidates if os.path.exists(p)),
        profile_candidates[0],
    )
    image_paths = [
        os.path.join(t_dir, pipeline_io.DIAG2_RLOSS_PROFILE_PNG),
        profile_path,
    ]
    images = "".join(
        f'<img class="chart-img" src="{_png_data_uri(p)}"'
        ' onclick="expandImage(this.src)"'
        " onerror=\"this.style.display='none'\">"
        for p in image_paths
    )
    blocks.append(
        f"<h4>{treatment}</h4><div class=\"row-mid\">{rows}</div>{images}"
    )
  return (
      '\n    <div id="tab_000_saturation" class="tab-content">\n'
      "        <h3>Phase 0: Saturation / Alpha Selection</h3>\n        "
      + "".join(blocks)
      + "\n    </div>\n"
  )


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
  optimization = metadata.get("optimization", True)
  sensitivity_scope = metadata.get("sensitivity_scope", ["total"])
  saturation_family = metadata.get("saturation_family")
  # Only present when a family is configured,
  # so legacy (family=None) reports render exactly as before.
  saturation_phase0 = metadata.get("saturation_phase0") or {}
  phase4_optimization = metadata.get("phase4_optimization") or {}
  # The Cluster DML keys (selected automatically for every run,
  # with or without saturation_config).  Older metadata files have no block,
  # in which case the header line is simply omitted.
  cluster_dml = metadata.get("cluster_dml") or {}
  cluster_keys_html = ""
  if cluster_dml:
    counts = cluster_dml.get("cluster_counts") or {}
    used = cluster_dml.get("keys_used") or cluster_dml.get("keys") or []
    if used:
      used_txt = ", ".join(f"{k} ({counts.get(k, '?')} clusters)" for k in used)
      used_txt = f"{len(used)}-way: {used_txt}"
    else:
      used_txt = "none (observation-level inference)"
    rejected = cluster_dml.get("not_adopted") or {}
    if rejected:
      used_txt += " | not adopted: " + ", ".join(
          f"{k} ({n} clusters)" for k, n in rejected.items()
      )
    cluster_keys_html = (
        "<strong>Cluster DML Keys (auto-selected; date/geo/item cols above"
        f" stay the visualization dimensions):</strong> {used_txt} <br>\n"
        "            "
    )

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
      "007_optimization",
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
          val_df = df_1a[df_1a[dim_col].astype(str) == str(dim_val)]
          for s_col in treatment_spend_cols.values():
            if s_col in val_df.columns:
              input_spend += float(val_df[s_col].sum())
        else:
          input_spend = sum(float(v) for v in treatment_spend_totals.values())
      else:
        s_col = treatment_spend_cols.get(ch)
        if dim_col and df_1a is not None and dim_col in df_1a.columns:
          val_df = df_1a[df_1a[dim_col].astype(str) == str(dim_val)]
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
            .tolist()
        )
        covariates_for_cate = (
            non_total_rows.get("covariates_for_cate", pd.Series())
            .dropna()
            .tolist()
        )
        cate_structure = (
            non_total_rows.get("cate_structure", pd.Series())
            .dropna()
            .tolist()
        )
        cate_df = (
            non_total_rows.get("cate_df", pd.Series())
            .dropna()
            .tolist()
        )
        cate_degree = (
            non_total_rows.get("cate_degree", pd.Series())
            .dropna()
            .tolist()
        )
        cate_intercept = (
            non_total_rows.get("cate_intercept", pd.Series())
            .dropna()
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

  sens_dim_opts = ""
  if "total" in sensitivity_scope:
    sens_dim_opts += "<option value='total'>Total</option>"
  if "geo" in sensitivity_scope:
    for g in geos:
      sens_dim_opts += f"<option value='geo:{g}'>geo:{g}</option>"
  if "item" in sensitivity_scope:
    for i in items:
      sens_dim_opts += f"<option value='item:{i}'>item:{i}</option>"

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
        f"Input (treatment): {total_input:,} | "
        f"Input (treatment spend): {int(round(total_input_spend)):,} | "
        f"Incremental KPI: {total_kpi:,} | "
        f"ROI (incremental KPI / treatment spend): {total_roi:.2f}"
    )
  else:
    header_metrics = (
        f"Input: {total_input:,} | Incremental KPI: {total_kpi:,} | "
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
        if geo != "all" and item != "all":
          # geo x item combinations have no selector entry; keying them as
          # `geo:<geo>` would overwrite the geo-level row.
          continue
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
                            "optimized_total_cost_of_treatment_in_periods", 0
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

        # Saturation-only fields. They are attached ONLY when the
        # optimizer wrote them, so legacy reports render exactly as before.
        treatment_name = str(row.get("treatment", "")).strip()
        extra_specs = [
            ("inc_kpi", f"incremental_KPI_{treatment_name}"),
            ("roi", f"roi_{treatment_name}"),
            ("opt_inc_kpi", f"optimized_incremental_KPI_{treatment_name}"),
            ("opt_roi", f"optimized_roi_{treatment_name}"),
        ]
        if treatment_name:
          for out_key, col_name in extra_specs:
            if col_name in df.columns and pd.notnull(row.get(col_name)):
              metrics[key][out_key] = float(row[col_name])
        # Keys whose values are free-form text; every other key is numeric.
        string_keys = frozenset(
            {"sat_family", "budget_mode", "reallocate_within", "assumptions"}
        )
        for out_key, col_name in [
            ("sat_family", "saturation_family"),
            ("alpha_hat", "alpha_hat"),
            ("budget_increment", "budget_increment"),
            # Optimization statistics shown in the descriptive block.
            ("budget_mode", "budget_mode"),
            ("budget_K", "K"),
            ("budget_K_out", "K_out"),
            ("unspent_budget", "unspent_budget"),
            ("marginal_roi_at_optimum", "marginal_roi_at_optimum"),
            ("n_units_excluded_negative_roi", "n_units_excluded_negative_roi"),
            ("n_units_capped_by_roi", "n_units_capped_by_roi"),
            # Allocation scope, support cap and the stated premise.
            ("reallocate_within", "reallocate_within"),
            ("cap_quantile", "cap_quantile"),
            ("assumptions", "assumptions"),
            ("n_units_at_support_cap", "n_units_at_support_cap"),
        ]:
          if col_name in df.columns and pd.notnull(row.get(col_name)):
            value = row[col_name]
            if out_key in string_keys:
              metrics[key][out_key] = str(value)
            else:
              metrics[key][out_key] = float(value)
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
          "007_optimization",
          "optimization_metrics.csv",
      )
      trt_opt_metrics[t] = extract_opt_metrics(path)

  trt_opt_metrics["main"] = extract_opt_metrics(optimization_metrics_csv_path)

  opt_metrics_json = json.dumps(trt_opt_metrics)

  def extract_yearly_summary(csv_path):
    """Reads ``optimization_yearly_summary.csv`` into the matrix payload.

    Args:
      csv_path: Path to ``optimization_yearly_summary.csv``.

    Returns:
      ``{dim_key: {"years": [...], "before": [{spend, kpi, roi}, ...],
      "after": [...]}}`` with ``dim_key`` in ``total`` / ``geo:<geo>`` /
      ``item:<item>`` (the values of the dimension selector). An empty dict
      when the file is missing, unreadable or lacks the required columns.
    """
    if not os.path.exists(csv_path):
      return {}
    try:
      df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
      print(f"Error reading {csv_path}: {e}")
      return {}
    needed = {"dimension", "level", "year", "spend_before", "spend_after"}
    if not needed.issubset(df.columns):
      return {}
    payload = {}
    for _, row in df.iterrows():
      dim = str(row["dimension"])
      level = str(row["level"])
      if dim == "total":
        key = "total"
      elif dim == "geo":
        key = f"geo:{level}"
      elif dim == "item":
        key = f"item:{level}"
      else:
        continue
      entry = payload.setdefault(key, {"years": [], "before": [], "after": []})
      entry["years"].append(str(row["year"]))
      for side in ("before", "after"):
        entry[side].append({
            "spend": float(row.get(f"spend_{side}", 0.0) or 0.0),
            "kpi": float(row.get(f"incremental_kpi_{side}", 0.0) or 0.0),
            "roi": float(row.get(f"roi_{side}", 0.0) or 0.0),
        })
    return payload

  yearly_summary = {}
  for t, ttype in treatment_types.items():
    if ttype in ["percentage", "impressions", "price"]:
      yearly_summary[t] = extract_yearly_summary(
          os.path.join(
              output_dir,
              "step4_consolidated_results",
              "roi_output",
              t,
              "007_optimization",
              "optimization_yearly_summary.csv",
          )
      )
  yearly_summary["main"] = extract_yearly_summary(
      os.path.join(
          output_dir,
          "step4_consolidated_results",
          "graph",
          "007_optimization",
          "optimization_yearly_summary.csv",
      )
  )
  yearly_summary_json = json.dumps(yearly_summary)
  # Which treatments were fitted with D = fbar(T;
  # alpha_hat) (selected_family != None) -- the sensitivity tab states that
  # their bounds are conditional on alpha_hat.  Empty on the family=None path.
  sens_selected_family_json = json.dumps(
      metadata.get("saturation_selected_family") or {}
  )

  # Reported in the header: True = the simple ROI-threshold optimizer, False
  # = the saturation-aware optimizer.  Older metadata files lack the key.
  simple_optimization_flag = metadata.get("simple_optimization")
  simple_optimization_str = (
      "n/a" if simple_optimization_flag is None else str(
          bool(simple_optimization_flag)
      )
  )

  # 000_saturation tab (saturation runs only).
  phase0_button = ""
  if saturation_phase0:
    phase0_button = (
        "        <button class=\"tab-button\" onclick=\"openTab(event,"
        " 'tab_000_saturation')\">000_saturation</button>\n"
    )
  # Linear-fallback notice for the optimization descriptive block, keyed
  # like `optMetricsData` ("main" for spend, the treatment name otherwise).
  phase4_fallback = {}
  for t_name, info in phase4_optimization.items():
    if (info or {}).get("phase4_fallback_to_linear"):
      ttype = treatment_types.get(t_name)
      key = (
          t_name if ttype in ["percentage", "impressions", "price"] else "main"
      )
      phase4_fallback[key] = info
  phase4_fallback_json = json.dumps(phase4_fallback)

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
        /* Yearly before / after matrix: exactly the width of the tab; the
           two label columns are fixed, the year columns share the rest and the
           renderer picks a font size at which the widest number fits.  Only
           the label cells may wrap; numbers are never broken. */
        .yearly-matrix {{ border-collapse: collapse; width: 100%; table-layout: fixed; margin-top: 15px; margin-bottom: 15px; }}
        .yearly-matrix th, .yearly-matrix td {{ text-align: right; padding: 5px 8px; border: 1px solid #c8e6c9; color: #000; vertical-align: middle; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .yearly-matrix th {{ background-color: #2e7d32; color: white; text-align: center; }}
        .yearly-matrix td.row-label {{ font-weight: bold; background-color: #e8f5e9; color: #1b5e20; text-align: left; white-space: normal; }}
        .yearly-matrix td.metric-label {{ text-align: left; color: #333; background-color: #f9fbf9; white-space: normal; }}
        .yearly-matrix td.total-col {{ background-color: #f1f8e9; font-weight: bold; }}
        .yearly-matrix tr.roi-row td {{ border-bottom: 2px solid #2e7d32; }}
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
            {cluster_keys_html}<strong>X Cols List:</strong> {', '.join(str(x) for x in x_cols_list)}
        </div>
        <div class="row-mid">
            <strong>Model Group(s):</strong> {', '.join(map(str, model_groups))} <br>
            <strong>Model Naming Convention Note:</strong> [Y-model]_[T-model]_[number_of_folds]_[number_of_repetitions]_[time_budget]_[number_of_covariates_for_cate]. <br>
            <strong>Covariates for CATE:</strong> {', '.join(map(str, covariates_for_cate))} <br>
            <strong>CATE Structure:</strong> {', '.join(map(str, cate_structure))} <br>
            <strong>CATE DF:</strong> {', '.join(map(str, cate_df))} <br>
            <strong>CATE Degree:</strong> {', '.join(map(str, cate_degree))} <br>
            <strong>CATE Intercept:</strong> {', '.join(map(str, cate_intercept))} <br>
            <strong>Simple Optimization:</strong> {simple_optimization_str}
        </div>
    </div>

    <div class="tab-buttons">
{phase0_button}        <button class="tab-button active" onclick="openTab(event, 'tab_001_total')">001_total</button>
        <button class="tab-button" onclick="openTab(event, 'tab_002_geo')">002_geo</button>
        <button class="tab-button" onclick="openTab(event, 'tab_003_item')">003_item</button>
        <button class="tab-button" onclick="openTab(event, 'tab_004_cate')">004_cate</button>
        <button class="tab-button" onclick="openTab(event, 'tab_005_prior')">005_prior</button>
        <button class="tab-button" onclick="openTab(event, 'tab_006_sensitivity')">006_sensitivity</button>
"""
  if has_spend:
    html += """        <button class="tab-button" onclick="openTab(event, 'tab_007_optimization')">007_optimization</button>\n"""

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
        <button class="tab-button" onclick="openTab(event, 'tab_{t}_roi_output_007')">roi_output_007_optimization ({t})</button>
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
        <img class="chart-img" src="{get_b64('step4_consolidated_results/graph/004_cate/cate_response_curves.png')}" onclick="expandImage(this.src)" onerror="this.style.display='none'">
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
        <select id="sel_006_trt" onchange="updateImgSens006()">
            {treatment_opts}
        </select>
"""
  if "geo" in sensitivity_scope or "item" in sensitivity_scope:
    html += f"""
        <select id="sel_006_dim" onchange="updateImgSens006()">
            {sens_dim_opts}
        </select>
"""
  html += """
        <br>
        <p id="sens_note_006" class="row-mid" style="display:none;"></p>
        <img id="img_006" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
    </div>
"""

  if saturation_phase0:
    html += _phase0_tab_html(output_dir, saturation_phase0)

  if has_spend:
    html += """
    <div id="tab_007_optimization" class="tab-content">
        <h3>Optimization</h3>
"""
    if len(treatment_cols) > 1:
      html += (
          "<p>Optimization for multiple treatments is currently not available"
          " in doubleml-pipeline.</p>\n"
      )
    elif not optimization:
      html += (
          "<p>Optimization is skipped because 'optimize' is set to False.</p>\n"
      )
    else:
      # Same structure as the roi_output_007_optimization (<treatment>) tab:
      # selector, descriptive metrics, the spend_vs_est chart (simple
      # optimizer), the original-vs-optimal treatment chart (saturation
      # optimizer only), the <outcome> chart (both optimizers: observed vs
      # projected outcome), the three before / after charts (spend,
      # incremental effect, ROI) and the yearly matrix.  Every image is hidden
      # when its file does not exist.
      html += f"""
        <select id="sel_007_dim" onchange="updateImgOpt007_main()">
            {dim_opts}
        </select>
"""
      html += """
        <br>
        <div id="opt_metrics_007_main"></div>
        <img id="img_007_spend_vs_est" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_007_treatment" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_007_outcome" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_007_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_007_incremental_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_007_roi_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <div id="yearly_matrix_007_main"></div>
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
        <h3>ROI Output 007: Optimization ({t})</h3>
        <select id="sel_{t}_007_dim" onchange="updateImgOpt007('{t}')">
            {dim_opts}
        </select>
        <br>
        <div id="opt_metrics_{t}_007"></div>
        <img id="img_{t}_007" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_{t}_007_outcome" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_{t}_007_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_{t}_007_incremental_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <img id="img_{t}_007_roi_before_after" class="chart-img" src="" onclick="expandImage(this.src)" onerror="this.style.display='none'">
        <div id="yearly_matrix_{t}_007"></div>
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
        const metadata_outcome_col = "{outcome_col}";
        const isAllOtherTreatment = {str(is_all_other_treatment).lower()};
        const optMetricsData = {opt_metrics_json};
        const optThresholdRoi = "{metadata.get('opt_threshold_roi', 'None')}";
        const phase4Fallback = {phase4_fallback_json};
        const yearlySummaryData = {yearly_summary_json};
        const sensSelectedFamily = {sens_selected_family_json};

        function renderMetrics(divId, key) {{
            var div = document.getElementById(divId);
            if (!div) return;
            var html = '';

            if (summaryMetrics[key]) {{
                var m = summaryMetrics[key];
                if (isAllOtherTreatment) {{
                    html += "<p><strong>Input (treatment):</strong> " + Number(m.input).toLocaleString('en-US') + " | <strong>Input (treatment spend):</strong> " + Number(m.input_spend).toLocaleString('en-US') + " | <strong>Incremental KPI:</strong> " + Number(m.kpi).toLocaleString('en-US') + " | <strong>ROI (incremental KPI / treatment spend):</strong> " + m.roi + "</p>";
                }} else {{
                    html += "<p><strong>Input:</strong> " + Number(m.input).toLocaleString('en-US') + " | <strong>Incremental KPI:</strong> " + Number(m.kpi).toLocaleString('en-US') + " | <strong>ROI:</strong> " + m.roi + "</p>";
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
                html += "<p><strong>Optimization ROI Threshold:</strong> " + optThresholdRoi + "<br>";
                html += "<strong>Optimization Period:</strong> " + m.start + " - " + m.end + "<br>";
                html += "<strong>Total Cost of Treatment in the Period:</strong> " + m.cost + "<br>";
                html += "<strong>Optimized Total Cost of Treatment in the Period:</strong> " + m.opt_cost + "<br>";
                html += "<strong>Reduction of Cost of Treatment:</strong> " + m.reduction + "<br>";
                html += "<strong>Other Cost Variables:</strong> " + m.other_vars + "<br>";
                html += "<strong>Total Cost of Other Cost Variables:</strong> " + m.other_cost + "<br>";
                html += "<strong>Increase Percentage of Total Cost of Other Cost Variables:</strong> " + m.increase_pct + "</p>";
                // Saturation-only rows, rendered only when the
                // saturation optimizer supplied them.
                if (m.sat_family !== undefined) {{
                    html += "<p><strong>Saturation Family:</strong> " + m.sat_family;
                    if (m.alpha_hat !== undefined) {{
                        html += "<br><strong>Estimated alpha (alpha_hat):</strong> " + m.alpha_hat.toPrecision(6);
                    }}
                    if (m.budget_increment !== undefined) {{
                        html += "<br><strong>Budget Increment:</strong> " + m.budget_increment;
                    }}
                    // Budget mode and the optimization statistics.
                    var fmtOpt = function(v) {{ return (v === null || v === undefined) ? "n/a" : Number(v).toLocaleString('en-US', {{maximumFractionDigits: 4}}); }};
                    if (m.budget_mode !== undefined) {{
                        html += "<br><strong>Budget mode:</strong> " + m.budget_mode;
                    }}
                    if (m.budget_K !== undefined) {{
                        html += "<br><strong>Budget K:</strong> " + fmtOpt(m.budget_K);
                    }}
                    if (m.budget_K_out !== undefined) {{
                        html += "<br><strong>Allocated total K_out:</strong> " + fmtOpt(m.budget_K_out);
                    }}
                    if (m.unspent_budget !== undefined) {{
                        html += "<br><strong>Unspent budget:</strong> " + fmtOpt(m.unspent_budget);
                    }}
                    if (m.marginal_roi_at_optimum !== undefined) {{
                        html += "<br><strong>Marginal ROI at the optimum (mu*):</strong> " + fmtOpt(m.marginal_roi_at_optimum);
                    }}
                    if (m.n_units_excluded_negative_roi !== undefined) {{
                        html += "<br><strong>Units excluded (negative ROI):</strong> " + m.n_units_excluded_negative_roi;
                    }}
                    if (m.n_units_capped_by_roi !== undefined) {{
                        html += "<br><strong>Units capped by the ROI threshold:</strong> " + m.n_units_capped_by_roi;
                    }}
                    if (m.n_units_at_support_cap !== undefined) {{
                        html += "<br><strong>Units at their support cap T_bar_t:</strong> " + m.n_units_at_support_cap;
                    }}
                    if (m.cap_quantile !== undefined) {{
                        html += "<br><strong>Support cap quantile (cap_quantile):</strong> " + fmtOpt(m.cap_quantile) + " (1 = cell maximum; T_bar_t = max(Q_q(geo x item cell, T&gt;0), actual))";
                    }}
                    if (m.reallocate_within !== undefined) {{
                        html += "<br><strong>Reallocation scope (reallocate_within):</strong> " + m.reallocate_within + (m.reallocate_within === "date" ? " (one KKT problem per date; no cross-day movement)" : " (one KKT problem; spend moves across days, geos and items)");
                    }}
                    if (m.assumptions !== undefined) {{
                        html += "<br><strong>Premise:</strong> " + m.assumptions + " &mdash; the response of a unit (geo x item x date row) depends on that row's spend only; no carryover / adstock is modelled. Interpret the allocation at the geo x item x period level rather than day by day.";
                    }}
                    html += "</p>";
                }}
                if (m.inc_kpi !== undefined || m.opt_inc_kpi !== undefined) {{
                    html += "<p>";
                    if (m.inc_kpi !== undefined) {{
                        html += "<strong>Incremental KPI in the Period:</strong> " + Math.round(m.inc_kpi).toLocaleString() + "<br>";
                    }}
                    if (m.roi !== undefined) {{
                        html += "<strong>ROI:</strong> " + m.roi.toFixed(4) + "<br>";
                    }}
                    if (m.opt_inc_kpi !== undefined) {{
                        html += "<strong>Optimized Incremental KPI in the Period:</strong> " + Math.round(m.opt_inc_kpi).toLocaleString() + "<br>";
                    }}
                    if (m.opt_roi !== undefined) {{
                        html += "<strong>Optimized ROI:</strong> " + m.opt_roi.toFixed(4) + "<br>";
                    }}
                    if (m.inc_kpi !== undefined && m.opt_inc_kpi !== undefined) {{
                        html += "<strong>Incremental KPI Uplift:</strong> " + Math.round(m.opt_inc_kpi - m.inc_kpi).toLocaleString();
                    }}
                    html += "</p>";
                }}
            }}
            // Phase 0 selected linear -> linear fallback.
            if (phase4Fallback[t]) {{
                var fb = phase4Fallback[t];
                var fmt = function(v) {{ return (v === null || v === undefined) ? "n/a" : Number(v).toPrecision(4); }};
                html += "<p><strong>Saturation Fallback:</strong> " + fb.phase4_fallback_reason;
                html += "<br><strong>Decision basis:</strong> decision_reason = " + (fb.phase4_fallback_decision_reason || "n/a") + ", d_bar = " + fmt(fb.phase4_fallback_d_bar) + ", se = " + fmt(fb.phase4_fallback_se) + ", k = " + fb.phase4_fallback_linear_test_k;
                html += "<br><strong>simple_optimization (config, unchanged):</strong> " + fb.simple_optimization_config + "</p>";
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

        // Yearly before / after matrix.  Columns: "Before / After", the
        // metric (Spend, Incremental KPI, ROI), one column per year and the
        // Total; every numeric cell holds the number only.  The table spans
        // the tab width (CSS: width 100%, fixed layout) and the font size is
        // chosen so that the widest number fits its column.
        var YEARLY_LABEL_COL_PX = 165;   // "Before optimization" column
        var YEARLY_METRIC_COL_PX = 140;  // "Incremental KPI" column
        var YEARLY_LABEL_COL_MIN_PX = 80;   // narrowest, labels wrap
        var YEARLY_METRIC_COL_MIN_PX = 80;
        var YEARLY_FONT_MAX_PX = 14;
        var YEARLY_FONT_MIN_PX = 9;
        var YEARLY_CELL_PAD_PX = 18;  // horizontal padding + border
        var YEARLY_EM_PER_CHAR = 0.62;  // digits / separators in Arial
        function renderYearlyMatrix(divId, t, dim) {{
            var div = document.getElementById(divId);
            if (!div) return;
            var data = yearlySummaryData[t] ? yearlySummaryData[t][dim] : null;
            if (!data || !data.years || !data.years.length) {{
                div.innerHTML = '';
                return;
            }}
            var fmtInt = function(v) {{ return Math.round(Number(v)).toLocaleString('en-US'); }};
            var fmtRoi = function(v) {{ return Number(v).toFixed(2); }};
            var metrics = [
                ['Spend', function(m) {{ return fmtInt(m.spend); }}, ''],
                ['Incremental KPI', function(m) {{ return fmtInt(m.kpi); }}, ''],
                ['ROI', function(m) {{ return fmtRoi(m.roi); }}, ' class="roi-row"']
            ];
            var nYears = data.years.length;
            var maxLen = 6;  // widest rendered number, in characters
            var html = '<table class="yearly-matrix">';
            html += '<colgroup><col class="col-label"><col class="col-metric">';
            for (var c = 0; c < nYears; c++) {{ html += '<col>'; }}
            html += '</colgroup>';
            html += '<tr><th></th><th>Metric</th>';
            for (var i = 0; i < nYears; i++) {{
                html += '<th>' + data.years[i] + '</th>';
            }}
            html += '</tr>';
            var groups = [['Before optimization', data.before], ['After optimization', data.after]];
            for (var g = 0; g < groups.length; g++) {{
                for (var r = 0; r < metrics.length; r++) {{
                    html += '<tr' + metrics[r][2] + '>';
                    if (r === 0) {{
                        html += '<td class="row-label" rowspan="' + metrics.length + '">' + groups[g][0] + '</td>';
                    }}
                    html += '<td class="metric-label">' + metrics[r][0] + '</td>';
                    for (var j = 0; j < nYears; j++) {{
                        var text = metrics[r][1](groups[g][1][j]);
                        if (text.length > maxLen) maxLen = text.length;
                        var cls = data.years[j] === 'Total' ? ' class="total-col"' : '';
                        html += '<td' + cls + '>' + text + '</td>';
                    }}
                    html += '</tr>';
                }}
            }}
            html += '</table>';
            div.innerHTML = html;
            var table = div.querySelector('table');
            table.dataset.maxlen = maxLen;
            table.dataset.nyears = nYears;
            fitYearlyMatrix(table);
        }}

        // Fit a yearly matrix to the width of its container: the year columns
        // share what the two label columns leave; the font shrinks (14 px
        // down to 9 px) until the widest number (~0.62 em per character plus
        // the cell padding) fits one column.  If even 9 px is too wide the
        // two label columns are narrowed (their text wraps) and, as a last
        // resort, the matrix scrolls horizontally inside its container, so
        // the numbers are never cut and nothing spills outside the page.
        // Re-run when a tab is opened (hidden tabs have no width) and when
        // the window is resized.
        function fitYearlyMatrix(table) {{
            var div = table.parentNode;
            if (!div || !div.clientWidth) return;
            var nYears = Number(table.dataset.nyears) || 1;
            var maxLen = Number(table.dataset.maxlen) || 6;
            var numberPx = function(fontPx) {{ return maxLen * YEARLY_EM_PER_CHAR * fontPx + YEARLY_CELL_PAD_PX; }};
            var labelPx = YEARLY_LABEL_COL_PX, metricPx = YEARLY_METRIC_COL_PX;
            var perCol = (div.clientWidth - labelPx - metricPx) / nYears;
            var fontPx = Math.floor((perCol - YEARLY_CELL_PAD_PX) / (maxLen * YEARLY_EM_PER_CHAR));
            fontPx = Math.max(YEARLY_FONT_MIN_PX, Math.min(YEARLY_FONT_MAX_PX, fontPx));
            var needed = numberPx(fontPx) * nYears;
            var spare = div.clientWidth - needed;  // what the label columns may use
            if (spare < labelPx + metricPx) {{
                var ratio = Math.max(0, spare) / (labelPx + metricPx);
                labelPx = Math.max(YEARLY_LABEL_COL_MIN_PX, Math.floor(labelPx * ratio));
                metricPx = Math.max(YEARLY_METRIC_COL_MIN_PX, Math.floor(metricPx * ratio));
            }}
            table.style.fontSize = fontPx + 'px';
            var cols = table.querySelectorAll('col');
            if (cols.length >= 2) {{
                cols[0].style.width = labelPx + 'px';
                cols[1].style.width = metricPx + 'px';
            }}
            var minWidth = Math.ceil(needed + labelPx + metricPx);
            if (minWidth > div.clientWidth) {{
                table.style.minWidth = minWidth + 'px';
                div.style.overflowX = 'auto';
            }} else {{
                table.style.minWidth = '';
                div.style.overflowX = '';
            }}
        }}

        function fitYearlyMatricesIn(root) {{
            var tables = (root || document).querySelectorAll('table.yearly-matrix');
            for (var i = 0; i < tables.length; i++) {{ fitYearlyMatrix(tables[i]); }}
        }}
        window.addEventListener('resize', function() {{ fitYearlyMatricesIn(document); }});

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
            var opened = document.getElementById(tabName);
            opened.style.display = "block";
            evt.currentTarget.className += " active";
            // The matrix was rendered while the tab was hidden (width 0).
            fitYearlyMatricesIn(opened);
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
            var dim = sel ? sel.value : 'total';
            var base = 'step4_consolidated_results/graph/007_optimization/';
            var prefix = 'total';
            if (dim.startsWith('geo:')) {{
                prefix = 'geo_' + dim.substring(4);
            }} else if (dim.startsWith('item:')) {{
                prefix = 'item_' + dim.substring(5);
            }}
            // Simple optimizer (simple_optimization=True / linear fallback).
            updateImgByPath('img_007_spend_vs_est',
                base + prefix + '_timeseries_spend_vs_est.png');
            // Saturation-aware optimizer only (hidden when the file does not
            // exist).
            updateImgByPath('img_007_treatment',
                base + prefix + '_original_vs_optimal_treatment.png');
            // Both optimizers: the <outcome> chart (observed vs projected
            // outcome), the before / after charts (spend, incremental effect,
            // ROI) and the yearly matrix.
            updateImgByPath('img_007_outcome',
                base + prefix + '_' + metadata_outcome_col + '.png');
            updateImgByPath('img_007_before_after',
                base + prefix + '_timeseries_spend_before_after.png');
            updateImgByPath('img_007_incremental_before_after',
                base + prefix + '_timeseries_incremental_effect_before_after.png');
            updateImgByPath('img_007_roi_before_after',
                base + prefix + '_timeseries_roi_before_after.png');
            renderOptMetrics('opt_metrics_007_main', 'main', dim);
            renderYearlyMatrix('yearly_matrix_007_main', 'main', dim);
        }}

        function updateImgSens006() {{
            var trt = document.getElementById('sel_006_trt').value;
            var dimSel = document.getElementById('sel_006_dim');
            var dim = dimSel ? dimSel.value : 'total';
            var safe_dim = dim.replace(/\\//g, '_').replace(/\\\\/g, '_');

            var scopeStr = 'total';
            if (safe_dim.startsWith('geo:')) {{
                scopeStr = safe_dim.substring(4);
            }} else if (safe_dim.startsWith('item:')) {{
                scopeStr = safe_dim.substring(5);
            }}

            var path = 'step4_consolidated_results/graph/006_sensitivity/' + trt + '_' + scopeStr + '_sensitivity_contour.png';
            updateImgByPath('img_006', path);
            // With D = fbar(T; alpha_hat) the OVB bounds
            // are conditional on the selected shape.
            var note = document.getElementById('sens_note_006');
            if (note) {{
                var fam = sensSelectedFamily[trt];
                if (fam) {{
                    note.innerHTML = '<strong>Note:</strong> the treatment variable of the PLR is f&#772;(T; &alpha;&#770;) (family: ' + fam + '); the bounds are conditional on the selected shape &alpha;&#770; (see Diag2 in the 000_saturation tab for shape uncertainty).';
                    note.style.display = 'block';
                }} else {{
                    note.style.display = 'none';
                }}
            }}
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
            var base = 'step4_consolidated_results/roi_output/' + t + '/007_optimization/';
            var prefix = 'total';
            if (dim.startsWith('geo:')) {{
                prefix = 'geo_' + dim.substring(4);
            }} else if (dim.startsWith('item:')) {{
                prefix = 'item_' + dim.substring(5);
            }}
            updateImgByPath('img_' + t + '_007', base + prefix + '_timeseries_spend_vs_est.png');
            updateImgByPath('img_' + t + '_007_outcome', base + prefix + '_' + metadata_outcome_col + '.png');
            updateImgByPath('img_' + t + '_007_before_after', base + prefix + '_timeseries_spend_before_after.png');
            updateImgByPath('img_' + t + '_007_incremental_before_after', base + prefix + '_timeseries_incremental_effect_before_after.png');
            updateImgByPath('img_' + t + '_007_roi_before_after', base + prefix + '_timeseries_roi_before_after.png');
            renderOptMetrics('opt_metrics_' + t + '_007', t, dim);
            renderYearlyMatrix('yearly_matrix_' + t + '_007', t, dim);
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
            if(document.getElementById('opt_metrics_007_main')) {{ updateImgOpt007_main(); }}

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

  out_path = os.path.join(
      output_dir,
      _report_filename(treatment_cols, outcome_col, saturation_family),
  )
  with open(out_path, "w") as f:
    f.write(html)

  return out_path
