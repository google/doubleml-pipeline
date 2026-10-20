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

"""End-to-end test: a real pipeline run is analyzable by the generated agent."""

import json
import pathlib
import shutil
import tempfile
import unittest

from doubleml_pipeline import workflow
from doubleml_pipeline.agent import results_reader
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.visualization import html_extraction
import numpy as np
import pandas as pd


def _simulate(n_weeks: int = 40, seed: int = 0) -> pd.DataFrame:
  """Panel of geos x items x weeks with a known, demand-dependent effect."""
  rng = np.random.default_rng(seed)
  index = pd.MultiIndex.from_product(
      [["G1", "G2", "G3"], ["I1", "I2"], range(n_weeks)],
      names=["geo", "item", "week"],
  )
  df = index.to_frame(index=False)
  n = len(df)
  df["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
      df["week"] * 7, unit="D"
  )
  df["X1"] = rng.normal(size=n)
  df["demand"] = rng.uniform(0.5, 1.5, size=n)
  df["spend"] = np.maximum(0.0, 10 + 3 * df["X1"] + rng.normal(size=n))
  df["sales"] = (
      100 + 2 * df["spend"] * df["demand"] + 5 * df["X1"] + rng.normal(size=n)
  )
  return df.drop(columns="week")


class AgentEndToEndTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.output_dir = pathlib.Path(tempfile.mkdtemp())
    orchestrator = workflow.CausalWorkflowOrchestrator(
        y_col="sales",
        d_cols=["spend"],
        x_cols_list=[["X1", "demand"]],
        # Scikit-learn based, so the test needs no LightGBM or XGBoost.
        ml_models_y=["rf"],
        ml_models_t=["rf"],
        n_folds_list=[2],
        output_dir=str(cls.output_dir),
        n_jobs=1,
        covariates_for_cate_list=[["demand"]],
        time_budget=1,
        date_col="date",
        geo_col="geo",
        item_col="item",
        run_sensitivity=True,
        sensitivity_scope=["total", "geo"],
        treatment_spend_cols={"spend": "spend"},
        sales_col="sales",
        optimize=True,
        opt_threshold_roi=1.0,
        generate_html=False,
    )
    orchestrator.run_full_pipeline(
        df=_simulate(),
        scaler=scalers.CausalDataScaler(
            standardize_cols=["sales", "spend", "X1", "demand"]
        ),
        top_n=1,
        n_reps=2,
    )
    cls.reader = results_reader.CausalResultsReader(cls.output_dir)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.output_dir)
    super().tearDownClass()

  def test_agent_is_generated_by_default(self):
    self.assertTrue(
        (
            self.output_dir / "agents" / "causal_ml_insight_agent" / "agent.py"
        ).is_file()
    )

  def test_every_analysis_is_available(self):
    (overview,) = self.reader.get_analysis_overview()["runs"]
    self.assertEqual(overview["unavailable_analyses"], [])

  def test_every_tool_reads_the_pipeline_outputs(self):
    calls = {
        "get_analysis_overview": {},
        "get_treatment_effects": {},
        "get_segment_effects": {"dimension": "geo"},
        "get_effect_trend": {"granularity": "quarter"},
        "get_effect_heterogeneity": {},
        "get_robustness_checks": {},
        "get_budget_optimization": {"dimension": "geo"},
    }
    self.assertCountEqual(calls, results_reader.TOOL_NAMES)
    for name, kwargs in calls.items():
      with self.subTest(tool=name):
        result = getattr(self.reader, name)(**kwargs)
        self.assertEqual(result["status"], "ok", result)
        json.dumps(result, allow_nan=False)

  def test_budget_optimization_matches_pipeline_outputs(self):
    for dimension in ("total", "date", "geo", "item"):
      with self.subTest(dimension=dimension):
        result = self.reader.get_budget_optimization(dimension=dimension)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["settings"]["optimizer"], "simple")
        self.assertTrue(result["matches_pipeline_totals"])

  def test_recovers_positive_effect(self):
    (run,) = self.reader.get_treatment_effects()["runs"]
    (effect,) = run["treatments"]
    self.assertEqual(effect["effect_direction"], "positive")

  def test_html_report_embeds_interactive_ai_assistant(self):
    report_path = html_extraction.generate_html_report(str(self.output_dir))
    self.assertIsNotNone(report_path)
    html_text = pathlib.Path(report_path).read_text(encoding="utf-8")
    self.assertIn("tab_ai_assistant", html_text)
    self.assertIn("causalAnalysisSnapshot", html_text)
    self.assertIn("aiSystemInstruction", html_text)
    self.assertIn("sendAiChatMessage", html_text)


if __name__ == "__main__":
  unittest.main()
