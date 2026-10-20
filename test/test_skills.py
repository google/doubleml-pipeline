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

"""Tests for the skills bundled with doubleml-pipeline."""

import re
import unittest

from doubleml_pipeline.agent import generator
from doubleml_pipeline.agent import instruction
from doubleml_pipeline.agent import results_reader
from google.adk import skills

_TOOL_REFERENCE = re.compile(r"\bget_[a-z_]+\b")


class BundledSkillsTest(unittest.TestCase):

  def test_bundled_skills_load_in_adk(self):
    loaded = skills.load_skills_from_dir(generator.BUNDLED_SKILLS_DIR)
    self.assertIn("retail-analytics", [skill.name for skill in loaded])

  def test_referenced_tools_exist(self):
    """Guards the skills and the instruction against tool renames."""
    documents = {"instruction": instruction.INSTRUCTION}
    for path in generator.BUNDLED_SKILLS_DIR.rglob("*.md"):
      documents[path.name] = path.read_text(encoding="utf-8")
    for name, text in documents.items():
      with self.subTest(document=name):
        self.assertLessEqual(
            set(_TOOL_REFERENCE.findall(text)), set(results_reader.TOOL_NAMES)
        )


if __name__ == "__main__":
  unittest.main()
