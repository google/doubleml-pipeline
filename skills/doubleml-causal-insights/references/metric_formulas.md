# Metric Formulas for `doubleml-pipeline` Outputs

Use these exact snippets when inspecting an `<output_dir>` so row-level 95%
bounds, monetary ROI, CATE p-value grades, segment profiles, and column
exclusions match pipeline semantics across any domain.

## Table of Contents

-   [1. Load Metadata, Exclude Ground Truth, and Summarize Any Slice](#1-load-metadata-exclude-ground-truth-and-summarize-any-slice)
-   [2. Within-Entity Quantile Bins, Column Roles, and Pooled CATE Grades](#2-within-entity-quantile-bins-column-roles-and-pooled-cate-grades)
-   [3. Diagnostics, Segment Profiles, and Phase 4 Optimization](#3-diagnostics-segment-profiles-and-phase-4-optimization)

## 1. Load Metadata, Exclude Ground Truth, and Summarize Any Slice

```python
import json
import pathlib
import re
import numpy as np
import pandas as pd

_GT_EXACT = {
    "true_total_roi",
    "SMAPE",
    "R_squared",
    "MAE",
    "RMSE",
    "total_ground_truth_KPI",
}
_NON_INPUT_PREFIXES = (
    "TrueContrib_",
    "true_incremental_KPI_",
    "ground_truth_",
    "estimated_incremental_KPI_",
    "total_estimated_incremental_KPI",
    "total_input_aggregate",
    "roi_amount_based_",
    "optimized_",
    "estimation_sample",
    "Y_pred",
    "T_pred",
    "Y_res",
    "T_res",
)


def usable_input_columns(df: pd.DataFrame, meta: dict[str, object]) -> list[str]:
  derived = set(meta.get("cluster_dml", {}).get("derived_columns", []))
  reserved = {
      meta.get("outcome_col"),
      meta.get("date_col"),
      meta.get("geo_col"),
      meta.get("item_col"),
      meta.get("geo_name_col"),
      meta.get("item_name_col"),
      *meta.get("treatment_cols", []),
      *(meta.get("treatment_spend_cols") or {}).values(),
  }
  return [
      c
      for c in df.columns
      if c not in _GT_EXACT
      and c not in derived
      and c not in reserved
      and not c.startswith(_NON_INPUT_PREFIXES)
  ]


def summarize_slice(
    df: pd.DataFrame,
    d_col: str,
    y_col: str,
    spend_col: str | None,
    threshold_roi: float | None,
) -> dict[str, object]:
  kpi_col = f"estimated_incremental_KPI_{d_col}"
  lo_row = np.minimum(df[f"{kpi_col}_2.5%"], df[f"{kpi_col}_97.5%"])
  hi_row = np.maximum(df[f"{kpi_col}_2.5%"], df[f"{kpi_col}_97.5%"])
  kpi, kpi_lo, kpi_hi = (
      float(df[kpi_col].sum()),
      float(lo_row.sum()),
      float(hi_row.sum()),
  )
  direction = "positive" if kpi > 0 else ("negative" if kpi < 0 else "zero")
  outcome = float(df[y_col].sum())
  t_vals = df[d_col]
  treated = int((t_vals != 0).sum())
  mixed_sign = bool((t_vals > 0).any() and (t_vals < 0).any())
  t_sum = float(t_vals.sum())
  if mixed_sign or t_sum == 0:
    t_input, per_unit, per_unit_lo, per_unit_hi = None, None, None, None
  else:
    t_input = t_sum
    per_unit = kpi / t_sum
    per_unit_lo = min(kpi_lo / t_sum, kpi_hi / t_sum)
    per_unit_hi = max(kpi_lo / t_sum, kpi_hi / t_sum)
  spend = (
      float(df[spend_col].sum())
      if spend_col and spend_col in df.columns
      else None
  )
  roi = (kpi / spend) if spend is not None and spend > 0 else None
  roi_lo = (kpi_lo / spend) if spend is not None and spend > 0 else None
  roi_hi = (kpi_hi / spend) if spend is not None and spend > 0 else None
  verdict = None
  if threshold_roi is not None and roi_lo is not None and roi_hi is not None:
    if roi_hi < threshold_roi:
      verdict = "not_worth_investing"
    elif roi_lo > threshold_roi:
      verdict = "worth_investing"
    else:
      verdict = "undetermined"
  flags: list[str] = []
  if spend is not None and spend < 0:
    flags.append("negative_spend")
  if spend_col == d_col and spend == 0 and kpi != 0:
    flags.append("kpi_without_spend")
  if kpi > 0 and kpi > max(outcome, 0.0):
    flags.append("incremental_kpi_exceeds_outcome")
  return {
      "rows": len(df),
      "treated_rows": treated,
      "low_sample": treated < 30,
      "outcome_total": outcome,
      "treatment_input": t_input,
      "spend": spend,
      "incremental_kpi": kpi,
      "incremental_kpi_low": kpi_lo,
      "incremental_kpi_high": kpi_hi,
      "effect_direction": direction,
      "share_of_outcome": (kpi / outcome) if outcome > 0 else None,
      "effect_per_unit": per_unit,
      "effect_per_unit_low": per_unit_lo,
      "effect_per_unit_high": per_unit_hi,
      "roi_on_spend": roi,
      "roi_on_spend_low": roi_lo,
      "roi_on_spend_high": roi_hi,
      "roi_vs_threshold": verdict,
      "plausibility_flags": flags,
  }
```

## 2. Within-Entity Quantile Bins, Column Roles, and Pooled CATE Grades

```python
def within_quantile_bins(
    df: pd.DataFrame, col: str, within_col: str, q: int = 4
) -> pd.Series:
  labels = [f"Q{i + 1}" for i in range(q)]
  return df.groupby(within_col)[col].transform(
      lambda s: pd.qcut(s.rank(method="first"), q=q, labels=labels)
  )


def column_role(meta: dict[str, object], d_col: str, col: str) -> str:
  treatments = list(meta.get("treatment_cols") or [])
  if d_col not in treatments:
    return "unused"
  idx = treatments.index(d_col)
  cate_lists = list(meta.get("covariates_for_cate_list") or [])
  x_lists = list(meta.get("x_cols_list") or [])
  if idx < len(cate_lists) and col in (cate_lists[idx] or []):
    return "cate_covariate"
  if idx < len(x_lists) and col in (x_lists[idx] or []):
    return "confounder"
  return "unused"


def cate_evidence(
    cate_csv: pathlib.Path, d_col: str, covariates: list[str]
) -> dict[str, str]:
  if not cate_csv.is_file():
    return {}
  cdf = pd.read_csv(cate_csv)
  row = cdf[(cdf["d_col"] == d_col) & (cdf["model"] == "median")]
  if row.empty:
    return {}
  pdict = json.loads(row.iloc[0]["cate_pvalue_dict"])
  grades: dict[str, str] = {}
  for cov in covariates:
    norm = re.sub(r"[\s-]+", "_", cov)
    pat = re.compile(
        rf"^(?:{re.escape(cov)}|{re.escape(norm)})$"
        rf"|\b(?:bs|C)\((?:{re.escape(cov)}|{re.escape(norm)})\b"
    )
    ps = [
        float(v)
        for k, v in pdict.items()
        if k.endswith("_pvalue")
        and k != "Intercept_pvalue"
        and pat.search(k[:-7])
    ]
    if ps:
      p_adj = min(1.0, min(ps) * len(ps))
      tier = "strong" if p_adj < 0.05 else ("weak" if p_adj < 0.10 else "none")
      p_str = "p<0.001" if p_adj < 0.001 else f"p={p_adj:.2g}"
      grades[cov] = f"{tier} ({p_str})"
  return grades
```

## 3. Diagnostics, Segment Profiles, and Phase 4 Optimization

```python
def repetition_stability(rep_csv: pathlib.Path, d_col: str) -> dict[str, float]:
  rdf = pd.read_csv(rep_csv)
  vals = rdf.loc[rdf["d_col"] == d_col, "total_estimated_incremental_KPI"]
  mean_val = float(vals.mean())
  return {
      "n_reps": int(len(vals)),
      "share_positive": float((vals > 0).mean()),
      "cv": (
          float(vals.std(ddof=1) / abs(mean_val))
          if len(vals) > 1 and mean_val != 0
          else 0.0
      ),
  }


def sensitivity_summary(
    sens_csv: pathlib.Path, d_col: str
) -> dict[str, object]:
  if not sens_csv.is_file():
    return {}
  sdf = pd.read_csv(sens_csv)
  sdf = sdf[sdf["treatment"] == d_col].copy()
  if sdf.empty:
    return {}
  strongest = sdf[["benchmark_cf_d", "benchmark_cf_y"]].max(axis=1)
  has_bench = sdf["benchmark_covariate"].notna() & (strongest > 0)
  sdf["robust_to_benchmark"] = (sdf["robustness_value"] > strongest).where(
      has_bench
  )
  total = sdf[sdf["scope"] == "total"].to_dict(orient="records")
  least = (
      sdf[sdf["scope"] != "total"]
      .nsmallest(5, "robustness_value")
      .to_dict(orient="records")
  )
  return {"overall": total, "least_robust_segments": least}


def segment_covariate_gaps(
    df: pd.DataFrame, mask: pd.Series, cols: list[str]
) -> list[dict[str, object]]:
  out: list[dict[str, object]] = []
  for col in cols:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
      continue
    mean_all, std_all = float(s.mean()), float(s.std())
    mean_seg = float(s[mask].mean())
    out.append({
        "column": col,
        "segment_mean": mean_seg,
        "overall_mean": mean_all,
        "ratio_to_overall": (mean_seg / mean_all) if mean_all else None,
        "gap_in_row_sd": (mean_seg - mean_all) / std_all if std_all else None,
    })
  return out


def optimization_dir(
    out_dir: pathlib.Path, meta: dict[str, object], d_col: str
) -> pathlib.Path:
  p4 = out_dir / "step4_consolidated_results"
  if (meta.get("treatment_types") or {}).get(d_col) == "spend":
    return p4 / "graph" / "007_optimization"
  return p4 / "roi_output" / d_col / "007_optimization"
```

-   **Simple vs. Saturation Optimizer & Warnings:** Check
    `optimization_yearly_summary.csv` (`dimension == "total"`, `geo`, or `item`)
    and reconcile against `optimization_metrics.csv` (`target_geo == "all"`,
    `target_item == "all"`). Under simple optimization (`simple_optimization ==
    true` or `linear_fallback`), rows in `periods_start_date..periods_end_date`
    with `kpi / spend < threshold_roi` have spend and KPI cut to zero (no
    redistribution); under saturation optimization, inspect
    `optimized_<d>_spend` and `optimized_incremental_KPI_<d>` in
    `consolidated_1a_full_df.csv` plus `K`, `K_out`, `unspent_budget`, and
    `marginal_roi_at_optimum` in `optimization_metrics.csv`. Flag explicitly if
    `kpi_after > 2.0 * kpi_before` (model-implied upper bound), if `kpi_before <=
    0 < kpi_after`, or if `> 30%` of spending units are cut to zero.
-   **Phase 0 Multi-Treatment Path:** Read
    `step0_saturation/summary_alpha_selection.json` for single-treatment runs,
    or `step0_saturation/<d>/summary_alpha_selection.json` when multiple
    treatments are modeled, alongside
    `step0_saturation/diag6_t_ref_consistency.csv`.
