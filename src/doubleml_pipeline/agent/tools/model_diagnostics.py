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

"""Tool: get_model_diagnostics."""

from typing import Any, Dict, List, Optional

import pandas as pd

from . import base
from . import cate_model

_LEAST_ROBUST_SEGMENTS = 5
_SENSITIVITY_COLUMNS = (
    "treatment",
    "scope",
    "group",
    "robustness_value",
    "benchmark_cf_d",
    "benchmark_cf_y",
    "benchmark_covariate",
)


def _repetition_stability(
    run: base.RunData, treatments: List[str]
) -> List[Dict[str, Any]]:
  """Summarizes the total effect across cross-fitting repetitions."""
  reps = run.read_csv(base.REPETITIONS_FILE, "Re-run the full pipeline.")
  base.require_columns(
      reps, ["d_col", "Model Group", "total_estimated_incremental_KPI"]
  )
  reps = reps[reps["d_col"].isin(treatments)]
  kpi = reps.groupby("d_col")["total_estimated_incremental_KPI"]
  stability = pd.DataFrame({
      "model_group": reps.groupby("d_col")["Model Group"].first(),
      "repetitions": kpi.size(),
      "median_incremental_kpi": kpi.median(),
      "min_incremental_kpi": kpi.min(),
      "max_incremental_kpi": kpi.max(),
      "share_positive": (
          (reps["total_estimated_incremental_KPI"] > 0)
          .groupby(reps["d_col"])
          .mean()
      ),
      "coefficient_of_variation": base.ratio(kpi.std(), kpi.mean().abs()),
  })
  return base.records(stability.rename_axis("treatment").reset_index())


def _confounding_sensitivity(
    run: base.RunData, treatments: List[str]
) -> Dict[str, Any]:
  """Summarizes the omitted-variable-bias sensitivity analysis."""
  sensitivity = run.read_csv(
      base.SENSITIVITY_FILE, "Re-run the pipeline with run_sensitivity=True."
  )
  base.require_columns(sensitivity, _SENSITIVITY_COLUMNS)
  sensitivity = sensitivity[sensitivity["treatment"].isin(treatments)]
  if sensitivity.empty:
    raise base.UnavailableError(f"No sensitivity results for {treatments}.")
  strongest = sensitivity[["benchmark_cf_d", "benchmark_cf_y"]].max(axis=1)
  # The pipeline writes zero strengths when no benchmark could be computed.
  has_benchmark = sensitivity["benchmark_covariate"].notna() & (strongest > 0)
  sensitivity = sensitivity.assign(
      robust_to_benchmark_confounder=(
          sensitivity["robustness_value"] > strongest
      ).where(has_benchmark)
  )
  names = {dim: run.segment_names(dim) for dim in base.DIMENSIONS}
  if any(names.values()):
    sensitivity = sensitivity.assign(
        group_name=[
            names.get(scope, {}).get(str(group))
            for scope, group in zip(sensitivity["scope"], sensitivity["group"])
        ]
    )
  is_overall = sensitivity["scope"] == "total"
  by_segment = [
      {
          "treatment": treatment,
          "scope": scope,
          "segments": len(rows),
          "robust_segments": int(
              rows["robust_to_benchmark_confounder"].eq(True).sum()
          ),
          "least_robust": base.records(
              rows.nsmallest(_LEAST_ROBUST_SEGMENTS, "robustness_value")
          ),
      }
      for (treatment, scope), rows in sensitivity[~is_overall].groupby(
          ["treatment", "scope"]
      )
  ]
  return {
      "overall": base.records(sensitivity[is_overall]),
      "by_segment": by_segment,
  }


def _cate_models(run: base.RunData, treatments: List[str]) -> Dict[str, Any]:
  """The CATE specification per treatment, or why it is unavailable."""
  models = {}
  for treatment in treatments:
    try:
      models[treatment] = cate_model.describe(run, treatment)
    except base.UnavailableError as e:
      models[treatment] = {"status": "unavailable", "reason": str(e)}
  return models


def _run_model_diagnostics(
    run: base.RunData, treatment: Optional[str] = None
) -> base.ToolResult:
  """Returns the diagnostics of a single run."""
  treatment = run.resolve_treatment(treatment, allow_all=True)
  treatments = [treatment] if treatment else run.treatments
  result: Dict[str, Any] = {}
  for key, check in (
      ("repetition_stability", _repetition_stability),
      ("confounding_sensitivity", _confounding_sensitivity),
  ):
    try:
      result[key] = check(run, treatments)
    except base.UnavailableError as e:
      result[key] = {"status": "unavailable", "reason": str(e)}
  result["cate_model"] = _cate_models(run, treatments)
  return result


@base.agent_tool
def get_model_diagnostics(
    self, treatment: Optional[str] = None
) -> base.ToolResult:
  """Assesses how far the estimates can be trusted and what they depend on.

  repetition_stability: spread of each treatment's total incremental KPI
  across repeated cross-fitting runs (share_positive near 0 or 1 and a low
  coefficient_of_variation indicate a stable estimate).
  confounding_sensitivity: Cinelli & Hazlett (2020) analysis. theta and se
  are the effect estimate and its standard error; robustness_value is the
  partial R2 an unobserved confounder would need with both treatment and
  outcome to explain the effect away; robust_to_benchmark_confounder is true
  when a confounder as strong as the strongest observed covariate
  (benchmark_covariate) could not. It cannot rule out stronger omitted
  drivers, such as other channels that were not adjusted for. Segment-level
  results are summarized with their least robust segments.
  cate_model: the fitted conditional-effect specification per treatment: its
  structure, basis terms with coefficients and p-values, covariate_evidence
  (strong/weak/none with the p-value) telling which covariates the per-unit
  effect depends on, and the distribution of the row-level per-unit effect.

  Args:
    treatment: Treatment to check. Defaults to all treatments of all runs.
  """
  if treatment is None:
    return {
        "runs": [
            {"run": name, **_run_model_diagnostics(run, None)}
            for name, run in self._runs.items()  # pylint: disable=protected-access
        ]
    }
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {"run": name, **_run_model_diagnostics(run, treatment)}
