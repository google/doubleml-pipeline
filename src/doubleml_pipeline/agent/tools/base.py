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

"""Shared run data loader, effect aggregation, and tool helpers."""

import functools
import json
import math
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...utils import io

METADATA_FILE = io.METADATA_FILE
PHASE4_DIR = io.PHASE4_DIR
FULL_DF_FILE = io.FULL_DF_FILE
REPETITIONS_FILE = io.REPETITIONS_FILE
CATE_COEF_FILE = io.CATE_COEF_FILE
SENSITIVITY_FILE = io.SENSITIVITY_FILE
OPTIMIZATION_FILE = io.OPTIMIZATION_FILE
ROI_OUTPUT_OPTIMIZATION_FILE = io.ROI_OUTPUT_OPTIMIZATION_FILE
DIMENSIONS = io.DIMENSIONS

ToolResult = Dict[str, Any]


class UnavailableError(Exception):
  """The pipeline run did not produce the output that a tool needs."""


class InvalidArgumentError(Exception):
  """A tool was called with an argument outside its documented domain."""


def agent_tool(
    method: Callable[..., ToolResult],
) -> Callable[..., ToolResult]:
  """Adds the status envelope and JSON-safe conversion to a tool method."""

  @functools.wraps(method)
  def wrapper(self, *args, **kwargs) -> ToolResult:
    try:
      result = {"status": "ok", **method(self, *args, **kwargs)}
    except UnavailableError as e:
      result = {"status": "unavailable", "reason": str(e)}
    except InvalidArgumentError as e:
      result = {"status": "invalid_argument", "reason": str(e)}
    return to_json_safe(result)

  return wrapper


def to_json_safe(value: Any) -> Any:
  """Recursively converts pandas/numpy values to compact JSON-safe values."""
  if isinstance(value, dict):
    return {str(k): to_json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [to_json_safe(v) for v in value]
  if isinstance(value, np.generic):
    value = value.item()
  if isinstance(value, float):
    # Drops floating-point noise such as 0.07000000000000001.
    return float(f"{value:.10g}") if math.isfinite(value) else None
  if isinstance(value, (pd.Timestamp, pd.Period, pd.Interval)):
    return str(value)
  if value is pd.NA or value is pd.NaT:
    return None
  return value


def check_choice(name: str, value: Any, choices: Sequence[Any]) -> None:
  if value not in choices:
    raise InvalidArgumentError(
        f"{name} must be one of {list(choices)}, got {value!r}."
    )


def check_range(name: str, value: int, low: int, high: int) -> None:
  if not low <= value <= high:
    raise InvalidArgumentError(
        f"{name} must be between {low} and {high}, got {value!r}."
    )


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
  missing = [c for c in columns if c not in df.columns]
  if missing:
    raise UnavailableError(
        f"Columns {missing} are missing from the consolidated results; re-run"
        " the pipeline."
    )


def records(df: pd.DataFrame) -> List[Dict[str, Any]]:
  return df.to_dict(orient="records")


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
  """Element-wise ratio, NaN (null in JSON) where the denominator is 0."""
  return numerator / denominator.where(denominator != 0)


def _add_per_unit_metric(
    sums: pd.DataFrame, name: str, denominator: pd.Series
) -> None:
  """Adds incremental KPI per unit of `denominator`, with its 95% interval."""
  sums[name] = ratio(sums["incremental_kpi"], denominator)
  low = ratio(sums["incremental_kpi_low"], denominator)
  high = ratio(sums["incremental_kpi_high"], denominator)
  sums[f"{name}_low"] = np.fmin(low, high)
  sums[f"{name}_high"] = np.fmax(low, high)


class RunData:
  """Loads and caches the outputs of a single pipeline run."""

  def __init__(self, results_dir: Union[str, "os.PathLike[str]"]):
    self.dir = pathlib.Path(results_dir)
    metadata_path = self.dir / METADATA_FILE
    if not metadata_path.is_file():
      raise FileNotFoundError(
          f"{metadata_path} not found. Pass the output_dir of a completed"
          " pipeline run."
      )
    self.metadata: Dict[str, Any] = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    self.treatments: List[str] = list(self.metadata["treatment_cols"])
    self.outcome: str = self.metadata["outcome_col"]
    self.dimensions: Dict[str, str] = {
        dim: self.metadata[f"{dim}_col"]
        for dim in DIMENSIONS
        if self.metadata.get(f"{dim}_col")
    }
    self.cate_covariates: Dict[str, List[str]] = dict(
        zip(
            self.treatments,
            self.metadata.get("covariates_for_cate_list") or [],
        )
    )
    self._csv_cache: Dict[str, pd.DataFrame] = {}

  def read_csv(self, relative_path: str, hint: str = "") -> pd.DataFrame:
    """Reads (and caches) a pipeline output CSV."""
    if relative_path not in self._csv_cache:
      path = self.dir / relative_path
      if not path.is_file():
        raise UnavailableError(
            f"{relative_path} was not produced by this pipeline run. {hint}"
            .strip()
        )
      self._csv_cache[relative_path] = pd.read_csv(path, low_memory=False)
    return self._csv_cache[relative_path]

  def full_df(self) -> pd.DataFrame:
    return self.read_csv(FULL_DF_FILE, "Re-run the full pipeline.")

  def has_file(self, relative_path: str) -> bool:
    return (self.dir / relative_path).is_file()

  def resolve_treatment(
      self, treatment: Optional[str], *, allow_all: bool
  ) -> Optional[str]:
    """Validates a treatment argument; None means all treatments combined."""
    if treatment is None:
      if len(self.treatments) == 1:
        return self.treatments[0]
      if allow_all:
        return None
      raise InvalidArgumentError(
          f"Specify a treatment, one of {self.treatments}."
      )
    check_choice("treatment", treatment, self.treatments)
    return treatment

  def segment_names(self, dimension: str) -> Dict[str, Any]:
    """Maps segment codes, as strings, to names if a name column is set."""
    column = self.dimensions.get(dimension)
    name_col = self.metadata.get(f"{dimension}_name_col")
    if not column or not name_col:
      return {}
    df = self.full_df()
    if name_col not in df.columns:
      return {}
    names = df.groupby(column)[name_col].first()
    return {str(code): name for code, name in names.items()}

  def effect_drivers(
      self, treatment: Optional[str], column: str, dimension: str
  ) -> Dict[str, Any]:
    """Describes what can make the per-unit effect differ across a dimension."""
    treatments = [treatment] if treatment else self.treatments
    covariates = {t: self.cate_covariates.get(t, []) for t in treatments}
    modeled = all(column in c for c in covariates.values())
    drivers = sorted(set().union(*covariates.values()))
    if modeled:
      explanation = f"The model estimates {dimension}-specific effects."
    elif drivers:
      explanation = (
          f"The model does not estimate {dimension}-specific effects. Per-unit"
          f" effects differ across each {dimension} only through {drivers};"
          f" any other {dimension} difference cannot be explained by this"
          f" analysis. To estimate it, add {column!r} or a variable that"
          " captures the difference to covariates_for_cate_list and re-run"
          " the pipeline."
      )
    else:
      explanation = (
          "The model assumes a constant per-unit effect, so per-unit"
          f" differences across each {dimension} cannot be explained by this"
          " analysis; incremental_kpi differs only with treatment volume."
      )
    return {
        "cate_covariates": covariates,
        "dimension_modeled": modeled,
        "explanation": explanation,
    }

  def aggregate(
      self, treatment: Optional[str], by: Optional[pd.Series] = None
  ) -> pd.DataFrame:
    """Sums effects, overall or per group of `by`, and derives metrics."""
    df = self.full_df()
    treatments = [treatment] if treatment else self.treatments
    kpi_cols = [f"estimated_incremental_KPI_{t}" for t in treatments]
    lower_cols = [f"{c}_2.5%" for c in kpi_cols]
    upper_cols = [f"{c}_97.5%" for c in kpi_cols]
    require_columns(df, [self.outcome, *kpi_cols, *lower_cols, *upper_cols])
    # Row bounds swap where the treatment is below its zero point (for example
    # negative price gaps), so order them per row before summing.
    lower = df[lower_cols].to_numpy()
    upper = df[upper_cols].to_numpy()
    frame = pd.DataFrame(
        {
            "incremental_kpi": df[kpi_cols].sum(axis=1),
            "incremental_kpi_low": np.nansum(np.fmin(lower, upper), axis=1),
            "incremental_kpi_high": np.nansum(np.fmax(lower, upper), axis=1),
            "outcome": df[self.outcome],
            "observations": 1,
        },
        index=df.index,
    )
    if treatment:
      require_columns(df, [treatment])
      frame["treatment_input"] = df[treatment]
      frame["treated_observations"] = df[treatment].fillna(0).ne(0)
      spend = (self.metadata.get("treatment_spend_cols") or {}).get(treatment)
      if spend in df.columns:
        frame["spend"] = df[spend]
    if by is None:
      sums = frame.sum().to_frame().T
    else:
      sums = frame.groupby(by, observed=True).sum()

    # Sign of the point estimate. The 95% interval is a conservative sum of
    # row-level bounds and is expected to include negative values, so it is
    # not used as a significance test here.
    sums["effect_direction"] = np.select(
        [sums["incremental_kpi"] > 0, sums["incremental_kpi"] < 0],
        ["positive", "negative"],
        default="zero",
    )
    sums["share_of_outcome"] = ratio(
        sums["incremental_kpi"], sums.pop("outcome")
    )
    if treatment:
      _add_per_unit_metric(sums, "effect_per_unit", sums["treatment_input"])
    if "spend" in sums:
      _add_per_unit_metric(sums, "roi_on_spend", sums["spend"])
    return sums
