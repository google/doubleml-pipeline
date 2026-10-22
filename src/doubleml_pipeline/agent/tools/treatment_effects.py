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

"""Tool: get_treatment_effects."""

from . import base


def _run_treatment_effects(run: base.RunData) -> base.ToolResult:
  all_treatments = run.aggregate(None).iloc[0]
  total_kpi = all_treatments["incremental_kpi"]
  effects = []
  for treatment in run.treatments:
    row = run.aggregate(treatment).iloc[0].to_dict()
    row["share_of_total_incremental_kpi"] = (
        row["incremental_kpi"] / total_kpi if total_kpi else None
    )
    effects.append({"treatment": treatment, **row})
  return {
      "outcome": run.outcome,
      "outcome_total": run.full_df()[run.outcome].sum(),
      "treatments": effects,
      "all_treatments": all_treatments.to_dict(),
  }


@base.agent_tool
def get_treatment_effects(self) -> base.ToolResult:
  """Returns the overall causal effect of every treatment, grouped by run.

  Per treatment: incremental_kpi with its 95% interval
  (incremental_kpi_low/high; a conservative range that is expected to include
  negative values, not a significance test) and effect_direction (the sign of
  the point estimate: positive, negative or zero), treatment_input,
  effect_per_unit (incremental KPI per unit of the treatment variable as
  recorded), spend and roi_on_spend (when a spend column is mapped), each with
  a 95% interval (_low/_high), share_of_outcome (incremental KPI / total
  outcome) and share_of_total_incremental_kpi within its run. Reliability
  signals: roi_vs_threshold (the ROI interval lies entirely below the run's
  ROI threshold: not_worth_investing; entirely above: worth_investing;
  otherwise undetermined), low_sample and plausibility_flags (for example an
  incremental KPI larger than the outcome itself, which means the estimate
  should not be trusted). all_treatments combines the treatments of one run
  only.
  """
  return {
      "runs": [
          {"run": name, **_run_treatment_effects(run)}
          for name, run in self._runs.items()  # pylint: disable=protected-access
      ]
  }
