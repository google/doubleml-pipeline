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

"""Tests for the bundled causal-insights Agent Skill and formulas."""

import ast
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
from types import ModuleType
import unittest

from doubleml_pipeline import workflow
import pandas as pd

_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$")
_CODE_FENCE_RE = re.compile(r"^(\s*)```(\w+)?\s*$")
_RETIRED_TOKENS = (
    "se_" + "seed",
    "se_" + "type",
    "crit" + "erion",
    "rloss_" + "linear",
    "fallback_" + "alpha",
    "roi_" + "threshold",
    "flatness_" + "threshold",
    "dispersion_" + "threshold",
    "resid_ratio_" + "threshold",
    "fbar_median_" + "obs",
    "fbar_q90_" + "obs",
    "linear " + "candidate",
    "combine_" + "se",
    "allocate_with_" + "roi",
    "alpha_hat_" + "ate",
)


_HARDCODED_CHANNEL_RE = re.compile(
    r"\b(?:tv|flyers?|media|promotions?|promos?|pantry|skus?|stores?)\b",
    re.IGNORECASE,
)


def _find_skills_dir() -> pathlib.Path | None:
  """Locates the repository `skills/` directory across source and runfiles.

  Returns:
    Path to `skills/` when available, or `None` in installed wheel environments
    that omit repository test data.
  """
  package_dir = pathlib.Path(os.path.abspath(workflow.__file__)).parent
  candidates = (
      pathlib.Path(os.path.abspath(__file__)).parent.parent,
      package_dir,
      package_dir.parent.parent,
  )
  for candidate in candidates:
    skills_dir = candidate / "skills"
    if (skills_dir / "causal-insights" / "SKILL.md").is_file():
      return skills_dir
  return None


_SKILLS_DIR = _find_skills_dir()
_SKILL_DIR = (
    (_SKILLS_DIR / "causal-insights")
    if _SKILLS_DIR is not None
    else pathlib.Path()
)


def _github_anchor(heading: str) -> str:
  """Converts a Markdown heading text into its GitHub-style URL slug."""
  text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
  text = re.sub(r"[`*~]", "", text).strip().lower()
  text = re.sub(r"[^\w\s-]", "", text)
  return re.sub(r"\s+", "-", text)


def _anchors(md_path: pathlib.Path) -> set[str]:
  """Extracts all heading anchors outside fenced code blocks in `md_path`."""
  anchors: set[str] = set()
  in_fence = False
  for line in md_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("```"):
      in_fence = not in_fence
      continue
    if not in_fence:
      match = _HEADING_RE.match(line)
      if match:
        anchors.add(_github_anchor(match.group(1)))
  return anchors


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
  """Parses flat YAML frontmatter and Markdown body from `SKILL.md` text."""
  if not text.startswith("---\n"):
    raise ValueError("Missing YAML frontmatter opening delimiter")
  end = text.find("\n---\n", 4)
  if end == -1:
    raise ValueError("Missing YAML frontmatter closing delimiter")
  parsed: dict[str, str] = {}
  current_key: str | None = None
  lines_acc: list[str] = []
  for line in text[4:end].splitlines():
    if line and not line[0].isspace() and ":" in line:
      if current_key is not None:
        parsed[current_key] = " ".join(lines_acc).strip()
      key, val = line.split(":", 1)
      current_key = key.strip()
      val = val.strip()
      lines_acc = [] if val in (">-", ">", "|") else [val]
    elif current_key is not None:
      lines_acc.append(line.strip())
  if current_key is not None:
    parsed[current_key] = " ".join(lines_acc).strip()
  return parsed, text[end + 5 :]


def _python_blocks(md_path: pathlib.Path) -> list[str]:
  """Extracts fenced Python code blocks from `md_path`."""
  blocks: list[str] = []
  in_block = False
  lang = ""
  indent = ""
  buf: list[str] = []
  for line in md_path.read_text(encoding="utf-8").splitlines():
    match = _CODE_FENCE_RE.match(line)
    if not in_block:
      if match and match.group(2):
        in_block = True
        indent = match.group(1)
        lang = match.group(2).lower()
        buf = []
    elif match and match.group(2) is None and match.group(1) == indent:
      if lang in ("python", "py"):
        blocks.append(
            "\n".join(
                item[len(indent) :] if item.startswith(indent) else item
                for item in buf
            )
        )
      in_block = False
    else:
      buf.append(line)
  return blocks


def _load_formulas_module(tmp_dir: pathlib.Path) -> ModuleType:
  """Compiles and imports `references/metric_formulas.md` Python snippets."""
  formulas_md = _SKILL_DIR / "references" / "metric_formulas.md"
  blocks = _python_blocks(formulas_md)
  if not blocks:
    raise ValueError("No Python blocks found in metric_formulas.md")
  for block in blocks:
    ast.parse(block)
  mod_path = tmp_dir / "metric_formulas_loaded.py"
  mod_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
  spec = importlib.util.spec_from_file_location(
      "doubleml_metric_formulas_test", mod_path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not create module spec for {mod_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


@unittest.skipIf(_SKILLS_DIR is None, "skills/ directory is not available")
class BundledSkillStructureTest(unittest.TestCase):
  """Agent Skills specification compliance for the bundled skill."""

  def test_frontmatter_and_length(self):
    skill_md = _SKILL_DIR / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(text)
    self.assertEqual(frontmatter.get("name"), _SKILL_DIR.name)
    self.assertRegex(frontmatter["name"], r"^[a-z0-9]+(-[a-z0-9]+)*$")
    self.assertLessEqual(len(frontmatter["name"]), 64)

    description = frontmatter.get("description", "")
    self.assertTrue(description)
    self.assertLessEqual(len(description), 1024)
    self.assertIn("Use when", description)
    self.assertIn("Don't use for", description)
    self.assertLess(len(text.splitlines()), 500)
    self.assertTrue(body.strip())

  def test_minimal_footprint_and_all_references_linked(self):
    self.assertFalse((_SKILL_DIR / "README.md").exists())
    self.assertFalse((_SKILL_DIR / "scripts").exists())
    refs_dir = _SKILL_DIR / "references"
    self.assertTrue(refs_dir.is_dir())
    skill_text = (_SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for child in sorted(refs_dir.iterdir()):
      if child.name in ("__pycache__", "__init__.py"):
        continue
      self.assertTrue(child.is_file(), f"Unexpected nested dir: {child}")
      rel = f"references/{child.name}"
      self.assertIn(rel, skill_text, f"{rel} is not linked from SKILL.md")

  def test_relative_links_anchors_and_reference_tocs(self):
    md_files = [
        _SKILL_DIR / "SKILL.md",
        *sorted((_SKILL_DIR / "references").glob("*.md")),
    ]
    for md_path in md_files:
      text = md_path.read_text(encoding="utf-8")
      if md_path.parent.name == "references" and len(text.splitlines()) > 100:
        self.assertIn(
            "## Table of Contents",
            text,
            f"{md_path.name} exceeds 100 lines without a Table of Contents",
        )
      for target in _LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
          continue
        file_part, _, anchor = target.partition("#")
        resolved = (
            md_path
            if not file_part
            else pathlib.Path(os.path.normpath(md_path.parent / file_part))
        )
        self.assertTrue(
            resolved.exists(), f"{md_path.name}: {target} does not exist"
        )
        if anchor and resolved.suffix == ".md":
          self.assertIn(
              anchor,
              _anchors(resolved),
              f"{md_path.name}: missing #{anchor} in {resolved.name}",
          )

  def test_no_retired_or_obsolete_terminology(self):
    pattern = re.compile(
        "|".join(re.escape(token) for token in _RETIRED_TOKENS), re.IGNORECASE
    )
    md_files = [
        _SKILL_DIR / "SKILL.md",
        *sorted((_SKILL_DIR / "references").glob("*.md")),
    ]
    for md_path in md_files:
      text = md_path.read_text(encoding="utf-8")
      hits = sorted({m.group(0) for m in pattern.finditer(text)})
      self.assertFalse(hits, f"{md_path.name} uses retired terms: {hits}")
      self.assertNotIn("Phase 5", text)
      self.assertNotIn("report.html", text)

  def test_channel_and_domain_agnostic_wording(self):
    md_files = [
        _SKILL_DIR / "SKILL.md",
        *sorted((_SKILL_DIR / "references").glob("*.md")),
    ]
    for md_path in md_files:
      text = md_path.read_text(encoding="utf-8")
      hits = sorted({m.group(0) for m in _HARDCODED_CHANNEL_RE.finditer(text)})
      self.assertFalse(
          hits,
          f"{md_path.name} hardcodes channel/retail-specific terms: {hits}",
      )


@unittest.skipIf(_SKILLS_DIR is None, "skills/ directory is not available")
class MetricFormulasExecutionTest(unittest.TestCase):
  """Executes the documented metric_formulas.md functions on synthetic outputs."""

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.tmp)
    self.formulas = _load_formulas_module(self.tmp)

  def test_column_filtering_excludes_ground_truth_and_derived_keys(self):
    df = pd.DataFrame({
        "date": ["2024-01-07"],
        "entity": ["A"],
        "outcome": [100.0],
        "treat_rate": [0.1],
        "treat_cost": [10.0],
        "driver_demand": [500.0],
        "is_peak": [1],
        "estimated_incremental_KPI_treat_rate": [20.0],
        "TrueContrib_treat_rate": [18.0],
        "total_ground_truth_KPI": [18.0],
        "true_total_roi": [1.8],
        "cluster_date_block": [0],
    })
    meta = {
        "outcome_col": "outcome",
        "date_col": "date",
        "geo_col": "entity",
        "treatment_cols": ["treat_rate"],
        "treatment_spend_cols": {"treat_rate": "treat_cost"},
        "cluster_dml": {"derived_columns": ["cluster_date_block"]},
    }
    usable = self.formulas.usable_input_columns(df, meta)
    self.assertEqual(usable, ["driver_demand", "is_peak"])

  def test_summarize_slice_orders_swapped_bounds_and_computes_spend_roi(self):
    df = pd.DataFrame({
        "outcome": [100.0, 200.0, 300.0],
        "treat_rate": [0.1, 0.2, -0.05],
        "treat_cost": [10.0, 20.0, 10.0],
        "estimated_incremental_KPI_treat_rate": [25.0, 45.0, 10.0],
        # Third row simulates swapped bounds when T < zero_point.
        "estimated_incremental_KPI_treat_rate_2.5%": [15.0, 35.0, 18.0],
        "estimated_incremental_KPI_treat_rate_97.5%": [35.0, 55.0, 2.0],
    })
    summary = self.formulas.summarize_slice(
        df,
        d_col="treat_rate",
        y_col="outcome",
        spend_col="treat_cost",
        threshold_roi=1.0,
    )
    self.assertEqual(summary["incremental_kpi"], 80.0)
    self.assertEqual(summary["incremental_kpi_low"], 52.0)
    self.assertEqual(summary["incremental_kpi_high"], 108.0)
    self.assertIsNone(summary["effect_per_unit"])
    self.assertEqual(summary["roi_on_spend"], 2.0)
    self.assertAlmostEqual(summary["roi_on_spend_low"], 1.3)
    self.assertAlmostEqual(summary["roi_on_spend_high"], 2.7)
    self.assertEqual(summary["roi_vs_threshold"], "worth_investing")
    self.assertTrue(summary["low_sample"])
    self.assertEqual(summary["plausibility_flags"], [])

    no_thr = self.formulas.summarize_slice(
        df.iloc[:1],
        d_col="treat_rate",
        y_col="outcome",
        spend_col="treat_cost",
        threshold_roi=None,
    )
    self.assertIsNone(no_thr["roi_vs_threshold"])
    self.assertAlmostEqual(no_thr["effect_per_unit"], 250.0)
    self.assertAlmostEqual(no_thr["effect_per_unit_low"], 150.0)
    self.assertAlmostEqual(no_thr["effect_per_unit_high"], 350.0)
    self.assertEqual(no_thr["effect_direction"], "positive")

    neg_treat = self.formulas.summarize_slice(
        df.iloc[2:],
        d_col="treat_rate",
        y_col="outcome",
        spend_col=None,
        threshold_roi=1.0,
    )
    self.assertAlmostEqual(neg_treat["effect_per_unit"], -200.0)
    self.assertAlmostEqual(neg_treat["effect_per_unit_low"], -360.0)
    self.assertAlmostEqual(neg_treat["effect_per_unit_high"], -40.0)
    self.assertIsNone(neg_treat["roi_on_spend"])

  def test_plausibility_flags_detect_invalid_slices(self):
    df = pd.DataFrame({
        "outcome": [50.0],
        "spend": [0.0],
        "estimated_incremental_KPI_spend": [80.0],
        "estimated_incremental_KPI_spend_2.5%": [70.0],
        "estimated_incremental_KPI_spend_97.5%": [90.0],
    })
    flagged = self.formulas.summarize_slice(
        df, d_col="spend", y_col="outcome", spend_col="spend", threshold_roi=1.0
    )
    self.assertEqual(
        flagged["plausibility_flags"],
        ["kpi_without_spend", "incremental_kpi_exceeds_outcome"],
    )

  def test_within_quantiles_cate_evidence_and_diagnostics_helpers(self):
    panel = pd.DataFrame({
        "entity": ["A"] * 4 + ["B"] * 4,
        "driver_x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    })
    bins = self.formulas.within_quantile_bins(panel, "driver_x", "entity", q=2)
    self.assertEqual(list(bins), ["Q1", "Q1", "Q2", "Q2"] * 2)

    cate_csv = self.tmp / "cate_coef_dict_pooling.csv"
    spline = "bs(driver_demand, df=3, degree=2, include_intercept=False)"
    pd.DataFrame({
        "model": ["rep_1", "median"],
        "d_col": ["treat_rate", "treat_rate"],
        "cate_pvalue_dict": [
            json.dumps({"Intercept_pvalue": 0.9}),
            json.dumps({
                "Intercept_pvalue": 0.001,
                f"{spline}[0]_pvalue": 0.004,
                f"{spline}[1]_pvalue": 0.00072,
                f"{spline}[2]_pvalue": 0.25,
                "is_peak_pvalue": 0.14,
            }),
        ],
    }).to_csv(cate_csv, index=False)
    grades = self.formulas.cate_evidence(
        cate_csv, "treat_rate", ["driver_demand", "is_peak"]
    )
    self.assertEqual(
        grades,
        {
            "driver_demand": "strong (p=0.0022)",
            "is_peak": "none (p=0.14)",
        },
    )
    meta_roles = {
        "treatment_cols": ["treat_rate"],
        "x_cols_list": [["driver_demand", "confounder_z"]],
        "covariates_for_cate_list": [["driver_demand"]],
    }
    self.assertEqual(
        self.formulas.column_role(meta_roles, "treat_rate", "driver_demand"),
        "cate_covariate",
    )
    self.assertEqual(
        self.formulas.column_role(meta_roles, "treat_rate", "confounder_z"),
        "confounder",
    )
    self.assertEqual(
        self.formulas.column_role(meta_roles, "treat_rate", "post_metric"),
        "unused",
    )

    sens_csv = self.tmp / "sensitivity_metrics.csv"
    pd.DataFrame({
        "treatment": ["treat_rate", "treat_rate"],
        "scope": ["total", "geo"],
        "group": ["all", "A"],
        "robustness_value": [0.12, 0.03],
        "benchmark_cf_d": [0.05, 0.08],
        "benchmark_cf_y": [0.04, 0.06],
        "benchmark_covariate": ["driver_demand", "driver_demand"],
    }).to_csv(sens_csv, index=False)
    sens = self.formulas.sensitivity_summary(sens_csv, "treat_rate")
    self.assertTrue(sens["overall"][0]["robust_to_benchmark"])
    self.assertFalse(sens["least_robust_segments"][0]["robust_to_benchmark"])

    gaps = self.formulas.segment_covariate_gaps(
        panel, panel["entity"] == "B", ["driver_x"]
    )
    self.assertEqual(len(gaps), 1)
    self.assertGreater(gaps[0]["gap_in_row_sd"], 0.5)

    rep_csv = self.tmp / "consolidated_3a_detail.csv"
    pd.DataFrame({
        "d_col": ["treat_rate"] * 3,
        "total_estimated_incremental_KPI": [60.0, 70.0, 80.0],
    }).to_csv(rep_csv, index=False)
    rep = self.formulas.repetition_stability(rep_csv, "treat_rate")
    self.assertEqual(rep["n_reps"], 3)
    self.assertEqual(rep["share_positive"], 1.0)
    self.assertAlmostEqual(rep["cv"], 10.0 / 70.0)

    spend_opt = self.formulas.optimization_dir(
        self.tmp, {"treatment_types": {"treat_spend": "spend"}}, "treat_spend"
    )
    pct_opt = self.formulas.optimization_dir(
        self.tmp,
        {"treatment_types": {"treat_rate": "percentage"}},
        "treat_rate",
    )
    self.assertEqual(
        spend_opt,
        self.tmp / "step4_consolidated_results" / "graph" / "007_optimization",
    )
    self.assertEqual(
        pct_opt,
        self.tmp
        / "step4_consolidated_results"
        / "roi_output"
        / "treat_rate"
        / "007_optimization",
    )


if __name__ == "__main__":
  unittest.main()
