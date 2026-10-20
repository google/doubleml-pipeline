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

"""Deterministic, read-only analysis tools over doubleml-pipeline outputs.

`CausalResultsReader` answers questions about the causal estimates written by
`CausalWorkflowOrchestrator.run_full_pipeline`. Each tool's implementation and
schema live under `doubleml_pipeline.agent.tools`.
"""

import os
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from .tools import base
from .tools import budget_optimization
from .tools import effect_heterogeneity
from .tools import effect_trend
from .tools import overview
from .tools import robustness_checks
from .tools import segment_effects
from .tools import treatment_effects

ToolResult = base.ToolResult

# Methods of CausalResultsReader exposed as agent tools, in prompt order.
TOOL_NAMES = (
    "get_analysis_overview",
    "get_treatment_effects",
    "get_segment_effects",
    "get_effect_trend",
    "get_effect_heterogeneity",
    "get_robustness_checks",
    "get_budget_optimization",
)


class CausalResultsReader:
  """Answers questions about the causal estimates of one or more pipeline runs.

  Each run is the `output_dir` of one `CausalWorkflowOrchestrator` run. Tools
  route a treatment to the run that estimated it, so treatment names must be
  unique across runs. Runs are estimated separately, so their effects are
  reported side by side and never added up.

  Typical use is to register `tools()` with an agent:

    reader = CausalResultsReader({"tv": "results/tv", "flyer": "results/flyer"})
    agent = Agent(..., tools=reader.tools())
  """

  def __init__(
      self,
      results: Union[
          str, "os.PathLike[str]", Mapping[str, Union[str, "os.PathLike[str]"]]
      ],
  ):
    """Initializes the reader.

    Args:
      results: The `output_dir` of a completed run, or a mapping from run name
        to `output_dir` for several runs.

    Raises:
      FileNotFoundError: If a run has no metadata.json.
      ValueError: If no run is given or a treatment appears in several runs.
    """
    if not isinstance(results, Mapping):
      results = {pathlib.Path(results).resolve().name: results}
    if not results:
      raise ValueError("At least one pipeline run is required.")
    self._runs = {name: base.RunData(path) for name, path in results.items()}
    self._run_of: Dict[str, str] = {}
    for name, run in self._runs.items():
      for treatment in run.treatments:
        if treatment in self._run_of:
          raise ValueError(
              f"Treatment {treatment!r} appears in runs"
              f" {self._run_of[treatment]!r} and {name!r}; treatment names"
              " must be unique across runs."
          )
        self._run_of[treatment] = name

  def tools(self) -> List[Callable[..., ToolResult]]:
    """Returns the bound tool methods to register with an agent."""
    return [getattr(self, name) for name in TOOL_NAMES]

  def snapshot(self) -> Dict[str, Any]:
    """Returns precomputed outputs of all tools for embedding in HTML reports."""
    overview_res = self.get_analysis_overview()
    treatments = list(self._run_of)
    segment_res: Dict[str, Any] = {}
    trend_res: Dict[str, Any] = {}
    if len(self._runs) == 1:
      segment_res["overall"] = {
          dim: self.get_segment_effects(dim) for dim in base.DIMENSIONS
      }
      trend_res["overall"] = {
          gran: self.get_effect_trend(granularity=gran)
          for gran in ("month", "quarter", "year", "month_of_year")
      }
    if len(treatments) > 1 or len(self._runs) > 1:
      segment_res["by_treatment"] = {
          t: {
              dim: self.get_segment_effects(dim, treatment=t)
              for dim in base.DIMENSIONS
          }
          for t in treatments
      }
      trend_res["by_treatment"] = {
          t: {
              gran: self.get_effect_trend(treatment=t, granularity=gran)
              for gran in ("month", "month_of_year")
          }
          for t in treatments
      }
    return {
        "get_analysis_overview": overview_res,
        "get_treatment_effects": self.get_treatment_effects(),
        "get_segment_effects": segment_res,
        "get_effect_trend": trend_res,
        "get_effect_heterogeneity": {
            t: self.get_effect_heterogeneity(treatment=t) for t in treatments
        },
        "get_robustness_checks": self.get_robustness_checks(),
        "get_budget_optimization": self._optimization_snapshot(treatments),
    }

  def _optimization_snapshot(self, treatments: List[str]) -> Dict[str, Any]:
    """Returns get_budget_optimization per treatment and dimension."""
    snapshot: Dict[str, Any] = {}
    for treatment in treatments:
      total = self.get_budget_optimization(treatment=treatment)
      if total["status"] != "ok":
        snapshot[treatment] = {"total": total}
        continue
      snapshot[treatment] = {
          "total": total,
          "date": self.get_budget_optimization(
              treatment=treatment, dimension="date"
          ),
          **{
              dim: self.get_budget_optimization(
                  treatment=treatment, dimension=dim
              )
              for dim in base.DIMENSIONS
          },
      }
    return snapshot

  get_analysis_overview = overview.get_analysis_overview
  get_treatment_effects = treatment_effects.get_treatment_effects
  get_segment_effects = segment_effects.get_segment_effects
  get_effect_trend = effect_trend.get_effect_trend
  get_effect_heterogeneity = effect_heterogeneity.get_effect_heterogeneity
  get_robustness_checks = robustness_checks.get_robustness_checks
  get_budget_optimization = budget_optimization.get_budget_optimization

  def _route(self, treatment: Optional[str]) -> Tuple[str, base.RunData]:
    """Returns the run name and `RunData` that estimated `treatment`."""
    if treatment is None:
      if len(self._runs) > 1:
        raise base.InvalidArgumentError(
            "The results span several runs; specify a treatment, one of"
            f" {list(self._run_of)}."
        )
      return next(iter(self._runs.items()))
    base.check_choice("treatment", treatment, list(self._run_of))
    name = self._run_of[treatment]
    return name, self._runs[name]
