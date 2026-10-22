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
from .tools import effects_by
from .tools import grouping
from .tools import model_diagnostics
from .tools import overview
from .tools import segment_profile
from .tools import treatment_effects

ToolResult = base.ToolResult

# Methods of CausalResultsReader exposed as agent tools, in prompt order.
TOOL_NAMES = (
    "get_analysis_overview",
    "get_treatment_effects",
    "get_effects_by",
    "get_segment_profile",
    "get_model_diagnostics",
    "get_budget_optimization",
)

# Period keys precomputed for the browser snapshot; finer ones would bloat it.
_SNAPSHOT_PERIODS = (
    "month",
    "quarter",
    "year",
    grouping.MONTH_OF_YEAR,
    grouping.DAY_OF_WEEK,
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
    """Returns precomputed outputs of all tools for embedding in HTML reports.

    `get_segment_profile` takes a free-form segment and is not precomputed.
    """
    treatments = list(self._run_of)
    return {
        "get_analysis_overview": self.get_analysis_overview(),
        "get_treatment_effects": self.get_treatment_effects(),
        "get_effects_by": {t: self._effects_by_snapshot(t) for t in treatments},
        "get_model_diagnostics": self.get_model_diagnostics(),
        "get_budget_optimization": {
            t: self._optimization_snapshot(t) for t in treatments
        },
    }

  def _effects_by_snapshot(self, treatment: str) -> Dict[str, Any]:
    """get_effects_by per segment dimension, period and CATE covariate."""
    _, run = self._route(treatment)
    keys = grouping.available_keys(run)
    snapshot: Dict[str, Any] = {}
    for key in keys["segments"]:
      snapshot[key] = self.get_effects_by(key, treatment=treatment)
    for key in keys["periods"]:
      if key in _SNAPSHOT_PERIODS:
        snapshot[key] = self.get_effects_by(key, treatment=treatment)
    for covariate in run.cate_covariates.get(treatment, []):
      if covariate in keys["columns"] and covariate not in snapshot:
        snapshot[covariate] = self.get_effects_by(
            covariate, treatment=treatment
        )
    return snapshot

  def _optimization_snapshot(self, treatment: str) -> Dict[str, Any]:
    """get_budget_optimization for the treatment, per dimension."""
    total = self.get_budget_optimization(treatment=treatment)
    if total["status"] != "ok":
      return {"total": total}
    return {
        "total": total,
        **{
            dim: self.get_budget_optimization(
                treatment=treatment, dimension=dim
            )
            for dim in ("date", *base.DIMENSIONS)
        },
    }

  get_analysis_overview = overview.get_analysis_overview
  get_treatment_effects = treatment_effects.get_treatment_effects
  get_effects_by = effects_by.get_effects_by
  get_segment_profile = segment_profile.get_segment_profile
  get_model_diagnostics = model_diagnostics.get_model_diagnostics
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
