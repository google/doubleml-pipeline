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

"""Tests for the ADK agent generator and the generated agent."""

import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest

from doubleml_pipeline.agent import generator
from doubleml_pipeline.agent import results_reader
from google.adk import agents
from google.adk.tools import function_tool
from google.adk.tools import skill_toolset


def _load_module(path: pathlib.Path):
  spec = importlib.util.spec_from_file_location("generated_agent", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class GenerateAgentTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.output_dir)
    (self.output_dir / "metadata.json").write_text(
        json.dumps({"treatment_cols": ["T1"], "outcome_col": "sales"})
    )

  def test_writes_agent_package_with_bundled_skills(self):
    agent_dir = generator.generate_agent(self.output_dir)

    self.assertEqual(
        agent_dir,
        self.output_dir / "agents" / generator.DEFAULT_AGENT_NAME,
    )
    self.assertCountEqual(
        [p.name for p in agent_dir.iterdir()],
        ["__init__.py", "agent.py", ".env.example", "skills"],
    )
    skill_dir = agent_dir / "skills" / "retail-analytics"
    self.assertTrue((skill_dir / "SKILL.md").is_file())
    # The output schema is maintained in docs/ and bundled on generation.
    self.assertEqual(
        (skill_dir / "references" / "output_csv_schema.md").read_text(
            encoding="utf-8"
        ),
        generator.OUTPUT_SCHEMA_DOC.read_text(encoding="utf-8"),
    )

  def test_generated_agent_is_loadable_by_adk(self):
    agent_dir = generator.generate_agent(
        self.output_dir, model="gemini-test", agent_name="my_agent"
    )

    root_agent = _load_module(agent_dir / "agent.py").root_agent

    self.assertIsInstance(root_agent, agents.LlmAgent)
    self.assertEqual(root_agent.name, "my_agent")
    self.assertEqual(root_agent.model, "gemini-test")
    *reader_tools, skills = root_agent.tools
    self.assertEqual(len(reader_tools), len(results_reader.TOOL_NAMES))
    self.assertIsInstance(skills, skill_toolset.SkillToolset)
    for tool in reader_tools:
      with self.subTest(tool=tool.__name__):
        # pylint: disable-next=protected-access
        declaration = function_tool.FunctionTool(tool)._get_declaration()
        self.assertEqual(declaration.name, tool.__name__)
        self.assertTrue(declaration.description)

  def test_generated_agent_follows_symlinks(self):
    agent_dir = generator.generate_agent(self.output_dir)
    link = pathlib.Path(tempfile.mkdtemp()) / "agents"
    self.addCleanup(shutil.rmtree, link.parent)
    link.symlink_to(agent_dir.parent)

    module = _load_module(link / agent_dir.name / "agent.py")

    overview = module.root_agent.tools[0]()
    self.assertEqual(overview["status"], "ok")

  def test_multiple_runs(self):
    runs = [self._write_run("tv", "TV"), self._write_run("flyer", "Flyer")]

    agent_dir = generator.generate_agent(runs, output_dir=self.output_dir)

    self.assertEqual(
        agent_dir, self.output_dir / "agents" / generator.DEFAULT_AGENT_NAME
    )
    self.assertIn(
        '"flyer": _AGENT_DIR / "../../flyer"',
        (agent_dir / "agent.py").read_text(),
    )
    overview = _load_module(agent_dir / "agent.py").root_agent.tools[0]()
    self.assertEqual([r["run"] for r in overview["runs"]], ["tv", "flyer"])

  def test_multiple_runs_validation(self):
    tv, flyer = self._write_run("tv", "TV"), self._write_run("flyer", "Flyer")
    tv_again = self._write_run("tv_again", "TV")
    cases = {
        "missing output_dir": {"results_dirs": [tv, flyer]},
        "duplicate run names": {
            "results_dirs": [tv, flyer],
            "output_dir": self.output_dir,
            "run_names": ["a", "a"],
        },
        "shared treatment": {
            "results_dirs": [tv, tv_again],
            "output_dir": self.output_dir,
        },
    }
    for name, kwargs in cases.items():
      with self.subTest(name):
        with self.assertRaises(ValueError):
          generator.generate_agent(**kwargs)

  def _write_run(self, name: str, treatment: str) -> pathlib.Path:
    run = self.output_dir / name
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"treatment_cols": [treatment], "outcome_col": "sales"})
    )
    return run

  def test_regeneration_replaces_skills(self):
    agent_dir = generator.generate_agent(self.output_dir)
    stale = agent_dir / "skills" / "stale-skill"
    stale.mkdir()

    generator.generate_agent(self.output_dir)

    self.assertFalse(stale.exists())

  def test_custom_skills_dir(self):
    skills_dir = self.output_dir / "my_skills"
    (skills_dir / "cpg-analytics").mkdir(parents=True)
    (skills_dir / "cpg-analytics" / "SKILL.md").write_text("---\n---\n")
    (skills_dir / "not-a-skill").mkdir()

    agent_dir = generator.generate_agent(self.output_dir, skills_dir=skills_dir)

    self.assertEqual(
        [p.name for p in (agent_dir / "skills").iterdir()], ["cpg-analytics"]
    )

  def test_rejects_skills_dir_inside_generated_agent(self):
    agent_dir = generator.generate_agent(self.output_dir)

    with self.assertRaises(ValueError):
      generator.generate_agent(self.output_dir, skills_dir=agent_dir / "skills")
    self.assertTrue((agent_dir / "skills" / "retail-analytics").is_dir())

  def test_invalid_arguments(self):
    cases = {
        "missing results": (
            FileNotFoundError,
            {"results_dirs": self.output_dir / "missing"},
        ),
        "agent name": (ValueError, {"agent_name": "my-agent"}),
        "model": (ValueError, {"model": ""}),
        "missing skills dir": (
            FileNotFoundError,
            {"skills_dir": self.output_dir / "missing"},
        ),
        "empty skills dir": (ValueError, {"skills_dir": self.output_dir}),
    }
    for name, (error, kwargs) in cases.items():
      with self.subTest(name):
        with self.assertRaises(error):
          generator.generate_agent(
              **{"results_dirs": self.output_dir, **kwargs}
          )


if __name__ == "__main__":
  unittest.main()
