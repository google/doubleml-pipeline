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

"""Reads the pooled CATE specification of a run.

The pipeline fits the conditional effect as a linear model over basis
expansions (splines, dummies) of the CATE covariates and records the pooled
coefficients, standard errors and p-values in `cate_coef_dict_pooling.csv`.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import base

_POOLED_MODEL = "median"
_SUFFIXES = {
    "coef": "cate_coef_dict",
    "stderr": "cate_stderr_dict",
    "pvalue": "cate_pvalue_dict",
}
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _pooled_row(run: base.RunData, treatment: str) -> Optional[pd.Series]:
  if not run.has_file(base.CATE_COEF_FILE):
    return None
  coefs = run.read_csv(base.CATE_COEF_FILE)
  pooled = coefs[
      (coefs["d_col"] == treatment) & (coefs["model"] == _POOLED_MODEL)
  ]
  return None if pooled.empty else pooled.iloc[0]


def _term_dict(row: pd.Series, column: str, suffix: str) -> Dict[str, Any]:
  """Parses a `{term_<suffix>: value}` JSON column into `{term: value}`."""
  if column not in row.index or pd.isna(row[column]):
    return {}
  return {
      term.removesuffix(f"_{suffix}"): value
      for term, value in json.loads(row[column]).items()
  }


def term_pvalues(run: base.RunData, treatment: str) -> Dict[str, float]:
  """Pooled CATE p-values keyed by basis term, or {} if not recorded."""
  row = _pooled_row(run, treatment)
  return {} if row is None else _term_dict(row, _SUFFIXES["pvalue"], "pvalue")


def covariate_pvalue(
    covariate: str, pvalues: Dict[str, float]
) -> Optional[float]:
  """Bonferroni-adjusted minimum p-value over a covariate's basis terms."""
  name = covariate.replace(" ", "_").replace("-", "_")
  matching = [
      p
      for term, p in pvalues.items()
      if _term_covariate(term) == name and p is not None and math.isfinite(p)
  ]
  if not matching:
    return None
  return min(1.0, min(matching) * len(matching))


def _term_covariate(term: str) -> str:
  """Returns the covariate behind a basis term.

  For example `bs(demand, df=3)[1]` -> `demand`, `holiday[T.True]` ->
  `holiday`.
  """
  match = re.match(r"^(?:bs|C)\((.+?)[,)]", term)
  if match:
    return match.group(1).strip()
  return term.split("[", 1)[0]


def describe(run: base.RunData, treatment: str) -> Dict[str, Any]:
  """The CATE specification of `treatment` and the evidence per covariate.

  Raises:
    UnavailableError: If the run has no CATE covariates for the treatment.
  """
  covariates = run.cate_covariates.get(treatment) or []
  if not covariates:
    raise base.UnavailableError(
        f"No covariates_for_cate were configured for {treatment!r}; the model"
        " assumes a constant per-unit effect."
    )
  row = _pooled_row(run, treatment)
  if row is None:
    raise base.UnavailableError(
        f"{base.CATE_COEF_FILE} has no pooled CATE row for {treatment!r}."
    )
  parsed = {
      name: _term_dict(row, column, name) for name, column in _SUFFIXES.items()
  }
  terms = [
      {
          "term": term,
          "covariate": _term_covariate(term),
          "coef": coef,
          "stderr": parsed["stderr"].get(term),
          "pvalue": parsed["pvalue"].get(term),
      }
      for term, coef in parsed["coef"].items()
  ]
  return {
      "treatment": treatment,
      "structure": row.get("cate_structure"),
      "spline_df": row.get("cate_df"),
      "spline_degree": row.get("cate_degree"),
      "covariates": covariates,
      "covariate_evidence": {
          c: base.evidence_label(covariate_pvalue(c, parsed["pvalue"]))
          for c in covariates
      },
      "terms": terms,
      "effect_per_unit_distribution": _per_unit_distribution(run, treatment),
  }


def _per_unit_distribution(run: base.RunData, treatment: str) -> Dict[str, Any]:
  """Quantiles of the row-level per-unit effect over treated rows."""
  df = run.full_df()
  kpi_col = f"estimated_incremental_KPI_{treatment}"
  base.require_columns(df, [treatment, kpi_col])
  treated = df[treatment].fillna(0).ne(0)
  per_unit = (
      (df.loc[treated, kpi_col] / df.loc[treated, treatment])
      .replace([np.inf, -np.inf], np.nan)
      .dropna()
  )
  if per_unit.empty:
    return {"treated_observations": 0}
  quantiles: List[float] = per_unit.quantile(_QUANTILES).tolist()
  return {
      "treated_observations": int(treated.sum()),
      **{f"p{int(q * 100)}": v for q, v in zip(_QUANTILES, quantiles)},
      "share_negative": float((per_unit < 0).mean()),
  }
