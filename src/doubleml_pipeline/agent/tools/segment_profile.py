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

"""Tool: get_segment_profile.

Describes one store, region or product in the data the pipeline was given,
before any estimation: how much it sells, how it was treated and how its
covariates compare with the rest of the data. It is the check to run when a
segment's estimate looks implausible or stands far apart from the others.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import base

_MAX_COLUMNS = 40
_QUANTILES = (0.25, 0.5, 0.75)


def _resolve_segment(run: base.RunData, dimension: str, segment: str) -> Any:
  """Returns the segment code matching `segment` as a code or a name."""
  codes = run.full_df()[run.dimensions[dimension]].dropna().unique()
  by_code = {str(code): code for code in codes}
  if segment in by_code:
    return by_code[segment]
  by_name = {
      str(name): by_code[code]
      for code, name in run.segment_names(dimension).items()
      if code in by_code
  }
  if segment in by_name:
    return by_name[segment]
  raise base.InvalidArgumentError(
      f"Unknown {dimension} {segment!r}; use one of {sorted(by_code)}"
      + (f" or a name among {sorted(by_name)}." if by_name else ".")
  )


def _profile_columns(
    run: base.RunData, treatment: Optional[str], columns: Optional[List[str]]
) -> List[str]:
  """Confounders and CATE covariates of the treatment(s), plus `columns`."""
  treatments = [treatment] if treatment else run.treatments
  default: List[str] = []
  for t in treatments:
    default += run.confounders.get(t, []) + run.cate_covariates.get(t, [])
  requested = list(columns or [])
  inputs = set(run.input_columns())
  unknown = [c for c in requested if c not in inputs]
  if unknown:
    raise base.InvalidArgumentError(
        f"Columns {unknown} are not input columns of this run."
    )
  selected = list(dict.fromkeys(default + requested))
  if len(selected) > _MAX_COLUMNS:
    raise base.InvalidArgumentError(
        f"{len(selected)} columns exceed the limit of {_MAX_COLUMNS}."
    )
  return selected


def _compare_columns(
    df: pd.DataFrame, mask: pd.Series, columns: List[str]
) -> List[Dict[str, Any]]:
  """Compares the segment mean of each numeric column with the overall mean.

  `gap_in_row_sd` is the gap in row-level standard deviations of the column,
  a scale-free size of the difference (not a z-score of the mean).
  """
  comparisons = []
  for column in columns:
    values = df[column]
    if not pd.api.types.is_numeric_dtype(values):
      comparisons.append({"column": column, "note": "not numeric"})
      continue
    overall_mean, overall_std = values.mean(), values.std()
    segment_mean = values[mask].mean()
    comparisons.append({
        "column": column,
        "segment_mean": segment_mean,
        "overall_mean": overall_mean,
        "ratio_to_overall": (
            segment_mean / overall_mean if overall_mean else None
        ),
        "gap_in_row_sd": (
            (segment_mean - overall_mean) / overall_std if overall_std else None
        ),
    })
  return comparisons


def _treatment_summary(
    df: pd.DataFrame, mask: pd.Series, treatment: str, spend_col: Optional[str]
) -> Dict[str, Any]:
  values = pd.to_numeric(df[treatment], errors="coerce")
  active = values.fillna(0).ne(0)
  segment_active = values[mask & active]
  summary: Dict[str, Any] = {
      "treatment": treatment,
      "treated_share": float(active[mask].mean()),
      "overall_treated_share": float(active.mean()),
      "mean_when_treated": (
          float(segment_active.mean()) if len(segment_active) else None
      ),
      "overall_mean_when_treated": (
          float(values[active].mean()) if active.any() else None
      ),
      **{
          f"p{int(q * 100)}_when_treated": v
          for q, v in zip(_QUANTILES, segment_active.quantile(_QUANTILES))
      },
  }
  if spend_col and spend_col in df.columns:
    spend = pd.to_numeric(df[spend_col], errors="coerce")
    total = spend.sum()
    summary["spend"] = float(spend[mask].sum())
    summary["share_of_spend"] = (
        float(spend[mask].sum() / total) if total else None
    )
  return summary


def _run_segment_profile(
    run: base.RunData,
    dimension: str,
    segment: str,
    treatment: Optional[str],
    columns: Optional[List[str]],
) -> base.ToolResult:
  base.check_choice("dimension", dimension, base.DIMENSIONS)
  if dimension not in run.dimensions:
    raise base.UnavailableError(
        f"No {dimension}_col was configured for this pipeline run."
    )
  treatment = run.resolve_treatment(treatment, allow_all=True)
  code = _resolve_segment(run, dimension, segment)
  df = run.full_df()
  mask = df[run.dimensions[dimension]] == code
  outcome = pd.to_numeric(df[run.outcome], errors="coerce")
  treatments = [treatment] if treatment else run.treatments
  result: Dict[str, Any] = {
      "dimension": dimension,
      "segment": code,
      "name": run.segment_names(dimension).get(str(code)),
      "observations": int(mask.sum()),
      "share_of_observations": float(mask.mean()),
      "outcome": {
          "column": run.outcome,
          "total": float(outcome[mask].sum()),
          "share_of_total": (
              float(outcome[mask].sum() / outcome.sum())
              if outcome.sum()
              else None
          ),
          "mean_per_row": float(outcome[mask].mean()),
          "overall_mean_per_row": float(outcome.mean()),
      },
      "treatments": [
          _treatment_summary(df, mask, t, run.spend_cols.get(t))
          for t in treatments
      ],
      "covariates": _compare_columns(
          df, mask, _profile_columns(run, treatment, columns)
      ),
  }
  if run.date_col:
    dates = run.dates()[mask]
    result["period"] = {
        "start": dates.min(),
        "end": dates.max(),
        "distinct_dates": int(dates.nunique()),
    }
  return result


@base.agent_tool
def get_segment_profile(
    self,
    dimension: str,
    segment: str,
    treatment: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> base.ToolResult:
  """Describes one store, region or product in the input data.

  The profile is computed before any estimation and compares the segment
  with the rest of the data.

  Use it to look behind an estimate that stands out: a segment whose
  interval does not overlap the others (ci_separation), that carries
  plausibility_flags, or whose effect is extreme. It shows whether the
  segment is special in the data itself, for example few treated rows, an
  unusual treatment intensity, or covariates far from the overall mean.

  Args:
    dimension: "geo" or "item".
    segment: The segment's code, or its name when a name column is mapped.
    treatment: Treatment whose confounders and CATE covariates to compare.
      Required when the results span several runs; defaults to all treatments of
      a single run.
    columns: Additional input columns to compare.

  Returns:
    observations and their share, the outcome total and share, per treatment
    the treated share, typical intensity when treated and spend share, the
    covered period, and for each covariate the segment mean, the overall
    mean, their ratio and gap_in_row_sd, the gap in row-level standard
    deviations of the column (a scale-free size of the difference).
  """
  name, run = self._route(treatment)  # pylint: disable=protected-access
  return {
      "run": name,
      **_run_segment_profile(run, dimension, segment, treatment, columns),
  }
