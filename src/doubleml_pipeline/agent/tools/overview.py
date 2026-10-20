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

"""Tool: get_analysis_overview."""

import re
from typing import Dict

from . import base
from . import budget_optimization

_LAG_PATTERN = re.compile(r"_l\d+$")


def _analysis_availability(run: base.RunData) -> Dict[str, bool]:
  return {
      "get_treatment_effects": run.has_file(base.FULL_DF_FILE),
      "get_segment_effects": bool(run.dimensions),
      "get_effect_trend": bool(run.metadata.get("date_col")),
      "get_effect_heterogeneity": any(run.cate_covariates.values()),
      "get_robustness_checks": (
          run.has_file(base.REPETITIONS_FILE)
          or run.has_file(base.SENSITIVITY_FILE)
      ),
      "get_budget_optimization": budget_optimization.optimization_available(
          run
      ),
  }


def _run_overview(run: base.RunData) -> base.ToolResult:
  """Returns analysis overview for a single run."""
  metadata = run.metadata
  date_col = metadata.get("date_col")
  treatment_types = metadata.get("treatment_types") or {}
  spend_cols = metadata.get("treatment_spend_cols") or {}
  confounders = dict(zip(run.treatments, metadata.get("x_cols_list") or []))
  availability = _analysis_availability(run)
  period = None
  if date_col:
    period = {
        "date_column": date_col,
        "start": metadata.get("date_col_min"),
        "end": metadata.get("date_col_max"),
    }
  return {
      "outcome": run.outcome,
      "period": period,
      "treatments": [
          {
              "name": treatment,
              "type": treatment_types.get(treatment, "spend"),
              "spend_column": spend_cols.get(treatment),
              "is_lagged": bool(_LAG_PATTERN.search(treatment)),
              "adjusted_for": confounders.get(treatment, []),
              "cate_covariates": run.cate_covariates.get(treatment, []),
          }
          for treatment in run.treatments
      ],
      "segment_dimensions": {
          dim: {
              "column": column,
              "segments": len(metadata.get(f"{dim}s") or []),
          }
          for dim, column in run.dimensions.items()
      },
      "available_analyses": [k for k, ok in availability.items() if ok],
      "unavailable_analyses": [k for k, ok in availability.items() if not ok],
  }


@base.agent_tool
def get_analysis_overview(self) -> base.ToolResult:
  """Describes each pipeline run and which analyses are available.

  Call this first. Per run: the outcome KPI, the analysis period, each
  treatment (type, spend column, confounders adjusted for, CATE covariates),
  the segment dimensions, and the available and unavailable analysis tools.
  Runs are estimated separately: compare their treatments side by side, but
  never add their effects up.
  """
  return {
      "runs": [
          {"run": name, **_run_overview(run)}
          for name, run in self._runs.items()  # pylint: disable=protected-access
      ]
  }
