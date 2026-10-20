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

"""Tool: get_effect_trend."""

from typing import Optional
import pandas as pd

from . import base

_PERIOD_FREQUENCIES = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}
_MONTH_OF_YEAR = "month_of_year"
_MAX_PERIODS = 120


def _run_effect_trend(
    run: base.RunData,
    treatment: Optional[str] = None,
    granularity: str = "month",
) -> base.ToolResult:
  """Returns effect trend over time for a single run."""
  base.check_choice(
      "granularity", granularity, (*_PERIOD_FREQUENCIES, _MONTH_OF_YEAR)
  )
  date_col = run.metadata.get("date_col")
  if not date_col:
    raise base.UnavailableError("No date_col was configured for this run.")
  treatment = run.resolve_treatment(treatment, allow_all=True)
  dates = pd.to_datetime(run.full_df()[date_col])
  if granularity == _MONTH_OF_YEAR:
    periods = dates.dt.month
  else:
    periods = dates.dt.to_period(_PERIOD_FREQUENCIES[granularity])

  trend = run.aggregate(treatment, by=periods.rename("period"))
  if trend.empty:
    raise base.UnavailableError(f"No valid dates in {date_col!r}.")
  if len(trend) > _MAX_PERIODS:
    raise base.InvalidArgumentError(
        f"{len(trend)} periods exceed the limit of {_MAX_PERIODS}; use a"
        " coarser granularity."
    )
  return {
      "treatment": treatment or "all",
      "granularity": granularity,
      "periods": base.records(trend.reset_index()),
      "highest_period": trend["incremental_kpi"].idxmax(),
      "lowest_period": trend["incremental_kpi"].idxmin(),
      "configured_peak_months": run.metadata.get("peak_month"),
      "effect_drivers": run.effect_drivers(treatment, date_col, "period"),
  }


@base.agent_tool
def get_effect_trend(
    self, treatment: Optional[str] = None, granularity: str = "month"
) -> base.ToolResult:
  """Returns how the causal effect evolves over time or across the year.

  Args:
    treatment: Treatment to analyze. Required when the results span several
      runs. Within a single run it defaults to the only treatment, or to all
      of the run's treatments combined.
    granularity: "week", "month", "quarter", "year", or "month_of_year" to
      pool each calendar month across years (seasonality profile).

  Returns:
    One row per period with the same metrics as get_treatment_effects, the
    periods with the highest and lowest incremental KPI, the peak months
    configured for the run, and effect_drivers: the only variables through
    which the per-unit effect can change over time.
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {"run": name, **_run_effect_trend(run, treatment, granularity)}
