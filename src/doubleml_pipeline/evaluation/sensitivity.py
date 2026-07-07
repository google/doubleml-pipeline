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

"""Module defining utilities for executing sensitivity analysis in causal graphs."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import doubleml as dml
import pandas as pd


def perform_gate_and_sensitivity_analysis(
    dml_model: dml.DoubleMLPLR,
    df_orig: pd.DataFrame,
    gate_cols: List[str],
    benchmark_covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Performs GATE (Group Average Treatment Effect) and robust sensitivity bounds checks."""
  benchmark_covariates = benchmark_covariates or []

  if len(gate_cols) == 1:
    group_series = df_orig[gate_cols[0]].astype(str)
  else:
    group_series = df_orig[gate_cols[0]].astype(str)
    for col in gate_cols[1:]:
      group_series = group_series + '_' + df_orig[col].astype(str)

  groups_df = pd.get_dummies(group_series)
  gate_est = dml_model.gate(groups=groups_df)

  dml_model.sensitivity_analysis(
      cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0
  )

  benchmarks_dict = {}
  for cov in benchmark_covariates:
    try:
      benchmarks = dml_model.sensitivity_benchmark(benchmarking_set=[cov])
      if isinstance(benchmarks, pd.DataFrame):
        bench_x = benchmarks['cf_d'].iloc[0]
        bench_y = benchmarks['cf_y'].iloc[0]
      else:
        bench_x = None
        bench_y = None
      benchmarks_dict[cov] = {'x': bench_x, 'y': bench_y}
    except (
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        RuntimeError,
    ) as err:
      benchmarks_dict[cov] = {'x': None, 'y': None, 'error': str(err)}

  return {
      'gate_summary': gate_est.summary,
      'sensitivity_elements': dml_model.sensitivity_elements,
      'benchmarks': benchmarks_dict,
      'groups_df': groups_df,
  }
