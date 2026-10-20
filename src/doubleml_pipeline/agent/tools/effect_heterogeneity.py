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

"""Tool: get_effect_heterogeneity."""

import json
import math
from typing import Dict, Optional
import pandas as pd

from . import base

_MAX_BINS = 10


def covariate_pvalue(
    covariate: str, term_pvalues: Dict[str, float]
) -> Optional[float]:
  """Returns the Bonferroni-adjusted minimum p-value of a covariate's terms."""
  name = covariate.replace(" ", "_").replace("-", "_")
  pvalues = [
      p
      for term, p in term_pvalues.items()
      if (term == name or term.startswith((f"bs({name},", f"{name}[")))
      and p is not None
      and math.isfinite(p)
  ]
  if not pvalues:
    return None
  return min(1.0, min(pvalues) * len(pvalues))


def _cate_term_pvalues(run: base.RunData, treatment: str) -> Dict[str, float]:
  """Returns pooled CATE p-values keyed by basis term, if recorded."""
  if not run.has_file(base.CATE_COEF_FILE):
    return {}
  coefs = run.read_csv(base.CATE_COEF_FILE)
  pooled = coefs[(coefs["d_col"] == treatment) & (coefs["model"] == "median")]
  if pooled.empty:
    return {}
  return {
      term.removesuffix("_pvalue"): pvalue
      for term, pvalue in json.loads(
          pooled["cate_pvalue_dict"].iloc[0]
      ).items()
  }


def _run_effect_heterogeneity(
    run: base.RunData, treatment: Optional[str] = None, bins: int = 4
) -> base.ToolResult:
  """Returns effect heterogeneity for a single run."""
  base.check_range("bins", bins, 2, _MAX_BINS)
  treatment = run.resolve_treatment(treatment, allow_all=False)
  covariates = run.cate_covariates.get(treatment)
  if not covariates:
    raise base.UnavailableError(
        f"No covariates_for_cate were configured for {treatment!r}."
    )
  term_pvalues = _cate_term_pvalues(run, treatment)
  df = run.full_df()
  base.require_columns(df, covariates)

  drivers = []
  for covariate in covariates:
    values = df[covariate]
    if values.nunique() > bins:
      if pd.api.types.is_numeric_dtype(values):
        values = pd.qcut(values, q=bins, duplicates="drop")
      else:
        frequent = values.value_counts().index[: bins - 1]
        values = values.where(values.isin(frequent), "other")
    table = run.aggregate(treatment, by=values.rename("bin"))
    drivers.append({
        "covariate": covariate,
        "heterogeneity_pvalue": covariate_pvalue(covariate, term_pvalues),
        "bins": base.records(table.reset_index()),
    })
  return {"treatment": treatment, "covariates": drivers}


@base.agent_tool
def get_effect_heterogeneity(
    self, treatment: Optional[str] = None, bins: int = 4
) -> base.ToolResult:
  """Shows how the per-unit effect (CATE) varies with each CATE covariate.

  Observations are split into quantile bins of every covariate configured in
  covariates_for_cate_list (or by value when it has at most `bins` distinct
  values). Each bin reports effect_per_unit with its interval and direction.
  heterogeneity_pvalue is the Bonferroni-adjusted p-value of the CATE basis
  terms of the covariate: below 0.05 means the effect significantly varies
  with it.

  Args:
    treatment: Treatment to analyze. Required unless there is only one.
    bins: Number of quantile bins per covariate (2-10).
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {"run": name, **_run_effect_heterogeneity(run, treatment, bins)}
