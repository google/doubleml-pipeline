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
from typing import Any, Dict

from . import base
from . import budget_optimization
from . import grouping

_LAG_PATTERN = re.compile(r"_l\d+$")


def _analysis_availability(run: base.RunData) -> Dict[str, bool]:
  has_results = run.has_file(base.FULL_DF_FILE)
  return {
      "get_treatment_effects": has_results,
      "get_effects_by": has_results,
      "get_segment_profile": has_results and bool(run.dimensions),
      "get_model_diagnostics": (
          run.has_file(base.REPETITIONS_FILE)
          or run.has_file(base.SENSITIVITY_FILE)
          or run.has_file(base.CATE_COEF_FILE)
      ),
      "get_budget_optimization": budget_optimization.optimization_available(
          run
      ),
  }


def _data_summary(run: base.RunData) -> Dict[str, Any]:
  """Shape of the input data: rows, period, frequency and segment counts."""
  metadata = run.metadata
  summary: Dict[str, Any] = {
      "rows": len(run.full_df()) if run.has_file(base.FULL_DF_FILE) else None,
      "frequency": run.data_frequency,
      "period": None,
      "segment_dimensions": {
          dim: {
              "column": column,
              "segments": len(metadata.get(f"{dim}s") or []),
          }
          for dim, column in run.dimensions.items()
      },
  }
  if run.date_col:
    summary["period"] = {
        "date_column": run.date_col,
        "start": metadata.get("date_col_min"),
        "end": metadata.get("date_col_max"),
    }
  return summary


def _run_overview(run: base.RunData) -> base.ToolResult:
  """Returns analysis overview for a single run."""
  metadata = run.metadata
  treatment_types = metadata.get("treatment_types") or {}
  availability = _analysis_availability(run)
  return {
      "outcome": run.outcome,
      "data": _data_summary(run),
      "treatments": [
          {
              "name": treatment,
              "type": treatment_types.get(treatment, "spend"),
              "spend_column": run.spend_cols.get(treatment),
              "is_lagged": bool(_LAG_PATTERN.search(treatment)),
              "saturation_family": run.saturation_family.get(treatment),
              "adjusted_for": run.confounders.get(treatment, []),
              "cate_covariates": run.cate_covariates.get(treatment, []),
          }
          for treatment in run.treatments
      ],
      "threshold_roi": run.threshold_roi,
      "effects_by_keys": (
          grouping.available_keys(run)
          if availability["get_effects_by"]
          else None
      ),
      "available_analyses": [k for k, ok in availability.items() if ok],
      "unavailable_analyses": [k for k, ok in availability.items() if not ok],
  }


@base.agent_tool
def get_analysis_overview(self) -> base.ToolResult:
  """Describes each pipeline run and which analyses are available.

  Call this first. Per run: the outcome KPI; the data (rows, sampling
  frequency such as daily or weekly, period, segment dimensions); each
  treatment (type, spend column, confounders adjusted for, CATE covariates);
  the ROI threshold that roi_vs_threshold verdicts refer to; the keys that
  get_effects_by accepts (segments, periods and input columns); and the
  available and unavailable analysis tools. Runs are estimated separately:
  compare their treatments side by side, but never add their effects up.
  """
  return {
      "runs": [
          {"run": name, **_run_overview(run)}
          for name, run in self._runs.items()  # pylint: disable=protected-access
      ]
  }
