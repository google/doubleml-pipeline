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

"""Tool: get_budget_optimization.

Reads the Phase 4 optimization of a run (`optimize=True`) at row level, that
is per geo x item x date allocation unit, and aggregates it by date, geo or
item. The saturation-aware optimizer writes the optimized spend and KPI of
every row to the consolidated CSV. The simple optimizer does not, but its rule
is deterministic (a row keeps its spend and incremental KPI only if its ROI
reaches the threshold), so it is reproduced here from the saved columns. In
both cases the period totals are checked against `optimization_metrics.csv`.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import base

_DIMENSIONS = ("total", "date", *base.DIMENSIONS)
_PERIOD_FREQUENCIES = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}
_MAX_PERIODS = 120
_MAX_TOP_N = 50
# Relative tolerance of the reconciliation with optimization_metrics.csv.
_RECONCILE_RTOL = 1e-6
# Thresholds of the plausibility warnings.
_KPI_RATIO_WARNING = 2.0
_CUT_SHARE_WARNING = 0.3
_SATURATION = "saturation"
_SIMPLE = "simple"
_LINEAR_FALLBACK = "linear_fallback"
# optimization_metrics.csv statistics of the saturation-aware optimizer.
_SATURATION_STATISTICS = (
    "K",
    "K_out",
    "unspent_budget",
    "marginal_roi_at_optimum",
    "n_units_excluded_negative_roi",
    "n_units_capped_by_roi",
    "n_units_at_support_cap",
)


def _optimization_file(run: base.RunData, treatment: str) -> Optional[str]:
  """Returns the optimization_metrics.csv path of `treatment`, if written."""
  for path in (
      base.OPTIMIZATION_FILE,
      base.ROI_OUTPUT_OPTIMIZATION_FILE.format(treatment=treatment),
  ):
    if run.has_file(path):
      metrics = run.read_csv(path)
      if (metrics["treatment"].astype(str) == treatment).any():
        return path
  return None


def optimization_available(run: base.RunData) -> bool:
  """Whether any treatment of `run` has optimization results."""
  return bool(run.metadata.get("optimization")) and any(
      _optimization_file(run, t) for t in run.treatments
  )


def _overall_metrics_row(
    run: base.RunData, path: str, treatment: str
) -> pd.Series:
  """Returns the all-geo, all-item row of optimization_metrics.csv."""
  metrics = run.read_csv(path)
  rows = metrics[
      (metrics["treatment"].astype(str) == treatment)
      & (metrics["target_geo"].astype(str) == "all")
      & (metrics["target_item"].astype(str) == "all")
  ]
  if rows.empty:
    raise base.UnavailableError(
        f"{path} has no overall row for {treatment!r}; re-run the pipeline."
    )
  return rows.iloc[0]


def _require_optimization(run: base.RunData, treatment: str) -> str:
  """Returns the metrics path, or explains why no optimization exists."""
  if not run.metadata.get("optimization"):
    raise base.UnavailableError(
        "Budget optimization was not run (optimize=False). Re-run the"
        " pipeline with optimize=True to get an optimized allocation."
    )
  if len(run.treatments) > 1:
    raise base.UnavailableError(
        "The pipeline optimizes only runs with a single treatment; this run"
        f" has {run.treatments}. Run one orchestrator per treatment with"
        " optimize=True."
    )
  path = _optimization_file(run, treatment)
  if path is None:
    raise base.UnavailableError(
        f"No optimization results for {treatment!r}. Optimization needs"
        " date_col and a treatment_spend_cols entry, plus sales_col for"
        " percentage, impressions and price treatments."
    )
  return path


def _optimizer_mode(
    run: base.RunData, treatment: str, df: pd.DataFrame
) -> str:
  """Returns "saturation", "simple" or "linear_fallback"."""
  recorded = (
      (run.metadata.get("phase4_optimization") or {}).get(treatment) or {}
  ).get("phase4_optimization")
  if recorded in (_SATURATION, _SIMPLE, _LINEAR_FALLBACK):
    return recorded
  saturation_cols = (
      f"optimized_{treatment}_spend",
      f"optimized_incremental_KPI_{treatment}",
  )
  if all(c in df.columns for c in saturation_cols):
    return _SATURATION
  return _SIMPLE


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
  return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _scope_mask(
    run: base.RunData,
    df: pd.DataFrame,
    overall: pd.Series,
    mode: str,
    settings: Dict[str, Any],
) -> pd.Series:
  """Rows the optimizer could change: the period (and the saturation scope)."""
  mask = pd.Series(True, index=df.index)
  date_col = run.metadata.get("date_col")
  if date_col and date_col in df.columns:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    start = pd.to_datetime(overall.get("periods_start_date"), errors="coerce")
    end = pd.to_datetime(overall.get("periods_end_date"), errors="coerce")
    if pd.notna(start):
      mask &= dates >= start
    if pd.notna(end):
      mask &= dates <= end
  if mode == _SATURATION:
    for dim, key in (("geo", "opt_geos"), ("item", "opt_items")):
      values = settings.get(key)
      column = run.dimensions.get(dim)
      if values is not None and column in df.columns:
        mask &= df[column].astype(str).isin([str(v) for v in values])
  return mask


def _row_plan(
    run: base.RunData,
    treatment: str,
    df: pd.DataFrame,
    mode: str,
    threshold: float,
) -> pd.DataFrame:
  """Returns spend and incremental KPI before and after, per row."""
  spend_col = (run.metadata.get("treatment_spend_cols") or {}).get(treatment)
  kpi_col = f"estimated_incremental_KPI_{treatment}"
  base.require_columns(df, [c for c in (spend_col, kpi_col) if c])
  if not spend_col:
    raise base.UnavailableError(
        f"No spend column is mapped for {treatment!r}."
    )
  spend = _numeric(df, spend_col)
  kpi = _numeric(df, kpi_col)
  if mode == _SATURATION:
    after_cols = [
        f"optimized_{treatment}_spend",
        f"optimized_incremental_KPI_{treatment}",
    ]
    base.require_columns(df, after_cols)
    spend_after = _numeric(df, after_cols[0])
    kpi_after = _numeric(df, after_cols[1])
  else:
    # The simple optimizer's rule (`optimization.optimize`): a row keeps its
    # spend and incremental KPI only if KPI / spend >= threshold, with ROI 0
    # where spend is 0.
    roi = kpi / spend.where(spend != 0)
    keep = roi.fillna(0.0) >= threshold
    spend_after = spend.where(keep, 0.0)
    kpi_after = kpi.where(keep, 0.0)
  return pd.DataFrame(
      {
          "spend_before": spend,
          "spend_after": spend_after,
          "incremental_kpi_before": kpi,
          "incremental_kpi_after": kpi_after,
      },
      index=df.index,
  )


def _summarize(plan: pd.DataFrame, by: Optional[pd.Series]) -> pd.DataFrame:
  """Sums the row plan, overall or per group, and derives change metrics."""
  tol = 1e-9 * max(float(plan["spend_before"].abs().max() or 0.0), 1.0)
  frame = plan.assign(
      allocation_units=1,
      units_with_actual_spend=plan["spend_before"] > tol,
      units_spend_increased=plan["spend_after"] > plan["spend_before"] + tol,
      units_spend_reduced=(plan["spend_after"] < plan["spend_before"] - tol)
      & (plan["spend_after"] > tol),
      units_spend_cut_to_zero=(plan["spend_before"] > tol)
      & (plan["spend_after"] <= tol),
  )
  if by is None:
    sums = frame.sum().to_frame().T
  else:
    sums = frame.groupby(by, observed=True).sum()
  count_cols = [
      c for c in sums.columns if c.startswith(("units_", "allocation_"))
  ]
  sums[count_cols] = sums[count_cols].astype(int)
  sums["spend_change"] = sums["spend_after"] - sums["spend_before"]
  sums["spend_change_pct"] = base.ratio(
      sums["spend_change"], sums["spend_before"]
  )
  sums["incremental_kpi_change"] = (
      sums["incremental_kpi_after"] - sums["incremental_kpi_before"]
  )
  sums["roi_before"] = base.ratio(
      sums["incremental_kpi_before"], sums["spend_before"]
  )
  sums["roi_after"] = base.ratio(
      sums["incremental_kpi_after"], sums["spend_after"]
  )
  return sums


def _settings(
    run: base.RunData, treatment: str, overall: pd.Series, mode: str
) -> Dict[str, Any]:
  """Returns the optimizer settings of the run."""
  metadata = run.metadata
  settings: Dict[str, Any] = {
      "optimizer": mode,
      "threshold_roi": float(overall.get("threshold_roi")),
      "period_start": overall.get("periods_start_date"),
      "period_end": overall.get("periods_end_date"),
  }
  if mode == _SATURATION:
    recorded = (metadata.get("phase4_optimization") or {}).get(treatment) or {}
    settings.update({
        "saturation_family": overall.get("saturation_family")
        or metadata.get("saturation_family"),
        "budget_mode": overall.get("budget_mode")
        or metadata.get("budget_mode"),
        "budget_increment": overall.get("budget_increment"),
        "budget_reallocate": overall.get("budget_reallocate"),
        "cap_quantile": overall.get("cap_quantile"),
        "reallocate_within": overall.get("reallocate_within"),
        "opt_geos": recorded.get("opt_geos", metadata.get("opt_geos")),
        "opt_items": recorded.get("opt_items", metadata.get("opt_items")),
    })
  return settings


def _assumptions(settings: Dict[str, Any]) -> List[str]:
  """States what the optimized allocation assumes, in plain words."""
  assumptions = [
      "Retrospective what-if: it re-plans the actual spend of the"
      " optimization period with the fitted model. It is not a forecast for"
      " future periods.",
      "The optimized incremental KPI is computed by the same model from point"
      " estimates. It is not observed, and no 95% interval applies to it.",
      "Each treatment is optimized on its own; spend is never moved between"
      " treatments or runs.",
  ]
  if settings["optimizer"] == _SATURATION:
    assumptions += [
        f"Diminishing returns follow the {settings['saturation_family']}"
        " saturation curve selected by the pipeline.",
        "A row's spend never exceeds the historical level of its geo x item"
        f" cell (cap_quantile={settings['cap_quantile']}), so no unobserved"
        " spend level is extrapolated.",
        "No carryover: spend in one period does not affect later periods.",
    ]
    if settings["reallocate_within"] == "date":
      assumptions.append("Each date keeps its total spend.")
    else:
      assumptions.append("Spend may move across dates, geos and items.")
    if settings["budget_mode"] == "roi_target":
      assumptions.append(
          "No budget limit: each row receives spend up to where its ROI"
          " reaches the threshold."
      )
    else:
      assumptions.append(
          "Budget = (1 + budget_increment) x actual spend in scope"
          f" (budget_increment={settings['budget_increment']})."
      )
  else:
    assumptions += [
        "Rows whose estimated ROI is below the threshold lose all their spend"
        " and its incremental KPI; the freed budget is not reallocated.",
        "The effect is linear in spend within a row (no saturation).",
    ]
  return assumptions


def _warnings(
    overall: Dict[str, Any], mode: str, reconciled: Optional[bool]
) -> List[str]:
  """Flags results that need care before they are presented as a plan."""
  warnings = []
  before = overall["incremental_kpi_before"]
  after = overall["incremental_kpi_after"]
  if before <= 0 < after:
    warnings.append(
        "The optimized incremental KPI turns positive although the estimate"
        " for the actual spend is not. The gain is model-implied: it comes"
        " from removing spend where the model estimates a negative effect."
    )
  elif before > 0 and after > _KPI_RATIO_WARNING * before:
    warnings.append(
        f"The optimized incremental KPI is {after / before:.1f} times the"
        " estimate for the actual spend. The gain is model-implied, mostly"
        " from removing spend the model scores low; present it as an upper"
        " bound, not as an expected gain."
    )
  spending_units = overall["units_with_actual_spend"]
  if spending_units and (
      overall["units_spend_cut_to_zero"] / spending_units > _CUT_SHARE_WARNING
  ):
    warnings.append(
        f"{overall['units_spend_cut_to_zero']} of the {spending_units}"
        " allocation units with actual spend lose all of it. Check whether"
        " this is operationally possible before recommending it."
    )
  if mode == _LINEAR_FALLBACK:
    warnings.append(
        "Saturation was configured but not identified in the data, so the"
        " simple ROI-threshold rule was used instead."
    )
  if reconciled is False:
    warnings.append(
        "The recomputed period totals differ from the pipeline's"
        " optimization_metrics.csv; re-run the pipeline before using them."
    )
  return warnings


def _reconcile(totals: Dict[str, Any], overall: pd.Series) -> Optional[bool]:
  """Checks the recomputed totals against optimization_metrics.csv."""
  pairs: List[Tuple[float, Any]] = [(
      totals["spend_after"],
      overall.get("optimized_total_cost_of_treatment_in_periods"),
  ), (
      totals["spend_before"],
      overall.get("total_cost_of_treatment_in_periods"),
  )]
  checked = [(a, float(b)) for a, b in pairs if pd.notna(b)]
  if not checked:
    return None
  return all(
      abs(a - b) <= _RECONCILE_RTOL * max(abs(b), 1.0) for a, b in checked
  )


def _ranked_segments(
    run: base.RunData, summary: pd.DataFrame, dimension: str, top_n: int
) -> Dict[str, Any]:
  """Returns the segments with the largest spend increases and decreases."""
  names = run.segment_names(dimension)
  if names:
    summary.insert(0, "name", summary.index.map(lambda c: names.get(str(c))))
  ranked = (
      summary.sort_values("spend_change", ascending=False)
      .rename_axis("segment")
      .reset_index()
  )
  increases = ranked[ranked["spend_change"] > 0]
  decreases = ranked[ranked["spend_change"] < 0].iloc[::-1]
  return {
      "segments": len(ranked),
      "segments_with_more_spend": len(increases),
      "segments_with_less_spend": len(decreases),
      "largest_increases": base.records(increases.head(top_n)),
      "largest_decreases": base.records(decreases.head(top_n)),
  }


def _run_budget_optimization(
    run: base.RunData,
    treatment: Optional[str],
    dimension: str,
    granularity: str,
    top_n: int,
) -> base.ToolResult:
  """Returns the optimized allocation of a single run."""
  base.check_choice("dimension", dimension, _DIMENSIONS)
  base.check_choice("granularity", granularity, tuple(_PERIOD_FREQUENCIES))
  base.check_range("top_n", top_n, 1, _MAX_TOP_N)
  treatment = run.resolve_treatment(treatment, allow_all=False)
  path = _require_optimization(run, treatment)
  overall_metrics = _overall_metrics_row(run, path, treatment)
  df = run.full_df()
  mode = _optimizer_mode(run, treatment, df)
  settings = _settings(run, treatment, overall_metrics, mode)
  scope = _scope_mask(run, df, overall_metrics, mode, settings)
  plan = _row_plan(
      run, treatment, df, mode, settings["threshold_roi"]
  ).loc[scope]
  if plan.empty:
    raise base.UnavailableError("No rows fall in the optimization period.")

  totals = _summarize(plan, None).iloc[0].to_dict()
  # A row of mixed dtypes upcasts the counts to float.
  totals = {
      k: int(v) if k.startswith(("units_", "allocation_")) else v
      for k, v in totals.items()
  }
  reconciled = _reconcile(totals, overall_metrics)
  result: Dict[str, Any] = {
      "treatment": treatment,
      "spend_column": (run.metadata.get("treatment_spend_cols") or {}).get(
          treatment
      ),
      "settings": settings,
      "assumptions": _assumptions(settings),
      "overall": totals,
      "matches_pipeline_totals": reconciled,
      "warnings": _warnings(totals, mode, reconciled),
  }
  if mode == _SATURATION:
    result["optimizer_statistics"] = {
        key: overall_metrics.get(key)
        for key in _SATURATION_STATISTICS
        if key in overall_metrics.index
    }

  if dimension == "total":
    return result
  result["dimension"] = dimension
  if dimension == "date":
    date_col = run.metadata.get("date_col")
    if not date_col or date_col not in df.columns:
      raise base.UnavailableError("No date_col was configured for this run.")
    periods = (
        pd.to_datetime(df.loc[plan.index, date_col], errors="coerce")
        .dt.to_period(_PERIOD_FREQUENCIES[granularity])
        .rename("period")
    )
    summary = _summarize(plan, periods)
    if len(summary) > _MAX_PERIODS:
      raise base.InvalidArgumentError(
          f"{len(summary)} periods exceed the limit of {_MAX_PERIODS}; use a"
          " coarser granularity."
      )
    result["granularity"] = granularity
    result["periods"] = base.records(summary.reset_index())
    return result

  column = run.dimensions.get(dimension)
  if not column:
    raise base.UnavailableError(
        f"No {dimension}_col was configured for this pipeline run."
    )
  result["column"] = column
  summary = _summarize(plan, df.loc[plan.index, column])
  result.update(_ranked_segments(run, summary, dimension, top_n))
  return result


@base.agent_tool
def get_budget_optimization(
    self,
    treatment: Optional[str] = None,
    dimension: str = "total",
    granularity: str = "month",
    top_n: int = 10,
) -> base.ToolResult:
  """Returns the pipeline's optimized budget allocation for a treatment.

  Available only when the run was made with optimize=True (single-treatment
  runs). The pipeline re-plans the actual spend of its optimization period,
  row by row (geo x item x date): the simple optimizer removes spend where the
  estimated ROI is below the threshold; the saturation-aware optimizer
  redistributes the budget along the estimated diminishing-returns curve.

  Args:
    treatment: Treatment to analyze. Required when the results span several
      runs; defaults to the run's only treatment.
    dimension: "total" (overall only), "date", "geo" or "item".
    granularity: For dimension="date": "week", "month", "quarter" or "year".
    top_n: For "geo" or "item": number of segments with the largest spend
      increases and decreases to return (1-50).

  Returns:
    settings (optimizer, ROI threshold, period and, for the saturation-aware
    optimizer, budget settings), assumptions to state with any
    recommendation, overall (spend, incremental KPI and ROI before and after,
    and counts of allocation units whose spend increases, is reduced or is cut
    to zero), warnings, matches_pipeline_totals and, depending on dimension,
    periods (one row per period) or the segments with the largest spend
    increases and decreases. Optimized values are model-implied, not observed.
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {
      "run": name,
      **_run_budget_optimization(
          run, treatment, dimension, granularity, top_n
      ),
  }
