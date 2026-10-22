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

"""Tests for the notebook chat helper."""

import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from doubleml_pipeline.agent import chat
from doubleml_pipeline.test import test_results_reader
from google.adk.events import event as event_lib
from google.genai import types


def _event(parts, author="causal_ml_insight_agent") -> event_lib.Event:
  return event_lib.Event(
      author=author, content=types.Content(role="model", parts=parts)
  )


# The tests reach into the chat's agent and runner on purpose.
# pylint: disable=protected-access


def _tool_names(session: chat.Chat):
  return [t.__name__ for t in session._agent.tools if callable(t)]


class ChatTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)
    test_results_reader._write_results(self.root)
    # Chat() refuses to start without credentials; give the tests a key.
    env = mock.patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    env.start()
    self.addCleanup(env.stop)

  def test_builds_the_generated_agent_in_process(self):
    session = chat.Chat(self.root)
    self.assertIn("get_effects_by", _tool_names(session))
    self.assertIn("get_segment_profile", _tool_names(session))

  def test_explicit_api_key_takes_precedence_over_environment(self):
    with mock.patch.dict(
        "os.environ",
        {"GOOGLE_GENAI_USE_VERTEXAI": "TRUE", "GOOGLE_CLOUD_PROJECT": "p"},
    ):
      with_key = chat.Chat(self.root, api_key="AIzaKey")
      # Older google-adk releases build their own client from the environment,
      # so the explicit key must be mirrored there too.
      self.assertEqual(os.environ["GOOGLE_API_KEY"], "AIzaKey")
      self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "FALSE")
    with mock.patch.dict(
        "os.environ",
        {"GOOGLE_GENAI_USE_VERTEXAI": "TRUE", "GOOGLE_CLOUD_PROJECT": "p"},
    ):
      from_env = chat.Chat(self.root)
    self.assertIsNotNone(with_key._agent.model.client)
    self.assertFalse(with_key._agent.model.client.vertexai)
    self.assertIsNone(from_env._agent.model.client)

  def test_missing_credentials_fail_early_with_guidance(self):
    with mock.patch.dict("os.environ", {}, clear=True):
      with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
        chat.Chat(self.root)
    with mock.patch.dict("os.environ", {"GOOGLE_API_KEY": "AIzaKey"}, clear=True):
      self.assertIsNone(chat.Chat(self.root)._agent.model.client)

  def test_vertex_express_key_uses_vertex_backend_when_asked(self):
    with mock.patch.dict(
        "os.environ", {"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_API_KEY": "old"}
    ):
      session = chat.Chat(self.root, api_key="express", vertexai=True)
      self.assertEqual(os.environ["GOOGLE_API_KEY"], "express")
      self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "TRUE")
      # An express mode key must not be paired with a project.
      self.assertNotIn("GOOGLE_CLOUD_PROJECT", os.environ)
    self.assertTrue(session._agent.model.client.vertexai)
    # Keys default to the Gemini API whatever they look like.
    default = chat.Chat(self.root, api_key="AQ.looks-like-express")
    self.assertFalse(default._agent.model.client.vertexai)

  async def test_ask_collects_tool_calls_and_answer_per_session(self):
    session = chat.Chat(self.root)
    seen = []

    async def fake_run_async(*, user_id, session_id, new_message):
      del user_id
      seen.append((session_id, new_message.parts[0].text))
      yield _event([
          types.Part(
              function_call=types.FunctionCall(
                  name="get_effects_by", args={"by": "geo"}
              )
          )
      ])
      yield _event([types.Part.from_text(text="Store B pays back.")])

    with mock.patch.object(session._runner, "run_async", fake_run_async):
      first = await session.ask("Where does it pay back?")
      second = await session.ask("And the worst store?")
      session.reset()
      third = await session.ask("Start over.")

    self.assertEqual(str(first), "Store B pays back.")
    self.assertEqual(
        first.tool_calls, (chat.ToolCall("get_effects_by", {"by": "geo"}),)
    )
    self.assertIn("get_effects_by(by='geo')", first._repr_markdown_())
    self.assertEqual(len(session.history), 1)
    self.assertEqual(seen[0][0], seen[1][0])  # Same conversation.
    self.assertNotEqual(seen[1][0], seen[2][0])  # New one after reset.
    self.assertEqual(third.question, "Start over.")
    json.dumps(first.tool_calls[0].args)

  def test_turn_renders_as_markdown(self):
    turn = chat.Turn("Q?", "A.", ())
    self.assertEqual(turn._repr_markdown_(), "**You:** Q?\n\nA.")


if __name__ == "__main__":
  unittest.main()
