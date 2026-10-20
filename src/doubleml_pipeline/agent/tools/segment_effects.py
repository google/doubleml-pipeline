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

"""Tool: get_segment_effects."""

from typing import Optional

from . import base

_SORT_KEYS = ("incremental_kpi", "effect_per_unit", "roi_on_spend")
_MAX_TOP_N = 50


def _run_segment_effects(
    run: base.RunData,
    dimension: str,
    treatment: Optional[str] = None,
    sort_by: str = "incremental_kpi",
    top_n: int = 10,
) -> base.ToolResult:
  """Returns top and bottom segment effects for a single run."""
  base.check_choice("dimension", dimension, base.DIMENSIONS)
  if dimension not in run.dimensions:
    raise base.UnavailableError(
        f"No {dimension}_col was configured for this pipeline run."
    )
  base.check_choice("sort_by", sort_by, _SORT_KEYS)
  base.check_range("top_n", top_n, 1, _MAX_TOP_N)
  treatment = run.resolve_treatment(treatment, allow_all=True)
  column = run.dimensions[dimension]
  df = run.full_df()

  segments = run.aggregate(treatment, by=df[column])
  if sort_by not in segments.columns:
    raise base.InvalidArgumentError(
        f"sort_by={sort_by!r} is not available for"
        f" treatment={treatment!r}; use one of"
        f" {[k for k in _SORT_KEYS if k in segments.columns]}."
    )
  names = run.segment_names(dimension)
  if names:
    segments.insert(
        0, "name", segments.index.map(lambda c: names.get(str(c)))
    )
  ranked = (
      segments.sort_values(sort_by, ascending=False)
      .rename_axis("segment")
      .reset_index()
  )
  n_lowest = min(top_n, len(ranked) - top_n)
  return {
      "dimension": dimension,
      "column": column,
      "treatment": treatment or "all",
      "sort_by": sort_by,
      "segments": len(ranked),
      "effect_direction_counts": (
          ranked["effect_direction"].value_counts().to_dict()
      ),
      "spread": ranked[sort_by].agg(["min", "median", "max"]).to_dict(),
      "effect_drivers": run.effect_drivers(treatment, column, dimension),
      "highest": base.records(ranked.head(top_n)),
      "lowest": (
          base.records(ranked.tail(n_lowest).iloc[::-1]) if n_lowest > 0 else []
      ),
  }


@base.agent_tool
def get_segment_effects(
    self,
    dimension: str,
    treatment: Optional[str] = None,
    sort_by: str = "incremental_kpi",
    top_n: int = 10,
) -> base.ToolResult:
  """Ranks segments, for example stores, regions or products, by effect.

  Args:
    dimension: Segment dimension, "geo" or "item".
    treatment: Treatment to analyze. Required when the results span several
      runs. Within a single run it defaults to the only treatment, or to all
      of the run's treatments combined.
    sort_by: Ranking metric: "incremental_kpi", "effect_per_unit" or
      "roi_on_spend". Per-unit metrics require a single treatment.
    top_n: Number of segments to return from each end of the ranking (1-50).

  Returns:
    The highest and lowest ranked segments with the same metrics as
    get_treatment_effects, the count of segments per effect_direction, the
    spread (min, median, max) of the ranking metric across all segments, and
    effect_drivers: whether the model estimates segment-specific effects at
    all. If it does not, differences in per-unit metrics between segments are
    not explained by this analysis.
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {
      "run": name,
      **_run_segment_effects(run, dimension, treatment, sort_by, top_n),
  }
