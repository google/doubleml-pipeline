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

"""Chat with the causal insight agent from a notebook.

`Chat` runs the same agent that `generate_agent` writes to disk, in process,
so it works in Colab and Jupyter without `adk web`:

    from doubleml_pipeline.agent import Chat

    chat = Chat("results")                     # one Chat = one conversation
    await chat.ask("Where does the promotion not pay back?")
    await chat.ask("Profile the worst region.")  # remembers the turn above
    chat.ui()                                  # or a chat widget

Authentication follows the agent's `.env.example`: Vertex AI when
`GOOGLE_GENAI_USE_VERTEXAI=TRUE` (with `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION`), otherwise `GOOGLE_API_KEY`. An explicit `api_key`
argument takes precedence over both.
"""

import asyncio
import dataclasses
import html
import os
from typing import Any, List, Mapping, Optional, Sequence, Union
import uuid

from google import genai
from google.adk import skills as adk_skills
from google.adk.agents import llm_agent
from google.adk.models import google_llm
from google.adk.runners import InMemoryRunner
from google.adk.tools import skill_toolset
from google.genai import types

from . import generator
from . import instruction
from . import results_reader

_APP_NAME = "doubleml_pipeline"
_USER_ID = "notebook"

PathLike = Union[str, "os.PathLike[str]"]
Results = Union[PathLike, Mapping[str, PathLike]]


@dataclasses.dataclass(frozen=True)
class ToolCall:
  """One analysis the agent ran while answering."""

  name: str
  args: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Turn:
  """A question, the analyses the agent ran, and its answer.

  Notebooks render it as Markdown; `str(turn)` is the answer text.
  """

  question: str
  answer: str
  tool_calls: Sequence[ToolCall]

  def __str__(self) -> str:
    return self.answer

  def _repr_markdown_(self) -> str:
    lines = [f"**You:** {self.question}", ""]
    if self.tool_calls:
      calls = ", ".join(_format_call(c) for c in self.tool_calls)
      lines += [f"<sub>Analyses run: {calls}</sub>", ""]
    lines.append(self.answer)
    return "\n".join(lines)


def _format_call(call: ToolCall) -> str:
  args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
  return f"`{call.name}({args})`"


class Chat:
  """A multi-turn conversation with the causal insight agent."""

  def __init__(
      self,
      results: Results,
      *,
      api_key: Optional[str] = None,
      model: str = generator.DEFAULT_AGENT_MODEL,
      skills_dir: Optional[PathLike] = None,
  ):
    """Builds the agent over one or more pipeline runs.

    Args:
      results: The `output_dir` of a completed run, or a mapping from run name
        to `output_dir` for several runs (treatment names must be unique).
      api_key: Gemini API key. When given it is used regardless of the
        environment; otherwise the environment decides between Vertex AI and
        `GOOGLE_API_KEY`, as in the generated agent's `.env.example`.
      model: Gemini model name.
      skills_dir: Directory of skills to load. Defaults to the skills bundled
        with the package.
    """
    reader = results_reader.CausalResultsReader(results)
    skills_root = (
        generator.BUNDLED_SKILLS_DIR if skills_dir is None else skills_dir
    )
    if api_key:
      # The client is pinned to the Gemini API: the SDK would otherwise still
      # switch to Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is set.
      client = genai.Client(api_key=api_key, vertexai=False)
      llm = google_llm.Gemini(model=model, client=client)
    else:
      llm = google_llm.Gemini(model=model)
    self._agent = llm_agent.Agent(
        name=generator.DEFAULT_AGENT_NAME,
        model=llm,
        description=generator.AGENT_DESCRIPTION,
        instruction=instruction.INSTRUCTION,
        tools=[
            *reader.tools(),
            skill_toolset.SkillToolset(
                skills=adk_skills.load_skills_from_dir(skills_root)
            ),
        ],
    )
    self._runner = InMemoryRunner(agent=self._agent, app_name=_APP_NAME)
    self._session_id: Optional[str] = None
    self.history: List[Turn] = []

  async def ask(self, question: str) -> Turn:
    """Sends `question` in the current conversation and returns the turn."""
    session_id = await self._ensure_session()
    answer_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=question)]
    )
    async for event in self._runner.run_async(
        user_id=_USER_ID, session_id=session_id, new_message=message
    ):
      for call in event.get_function_calls():
        tool_calls.append(ToolCall(call.name, dict(call.args or {})))
      if event.is_final_response() and event.content:
        answer_parts += [p.text for p in event.content.parts if p.text]
    turn = Turn(question, "\n".join(answer_parts).strip(), tuple(tool_calls))
    self.history.append(turn)
    return turn

  def reset(self) -> None:
    """Starts a new conversation; the agent forgets previous turns."""
    self._session_id = None
    self.history = []

  async def _ensure_session(self) -> str:
    if self._session_id is None:
      session = await self._runner.session_service.create_session(
          app_name=_APP_NAME, user_id=_USER_ID, session_id=uuid.uuid4().hex
      )
      self._session_id = session.id
    return self._session_id

  def ui(self, placeholder: str = "Ask about the causal estimates..."):
    """Returns a chat widget for notebooks (requires ipywidgets).

    Each turn shows the question, the analyses the agent ran and the answer
    as Markdown. Press Enter or the Send button to ask; Shift+Enter adds a
    line.
    """
    try:
      import ipywidgets as widgets  # pylint: disable=g-import-not-at-top
      from IPython import display  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          "Chat.ui() needs ipywidgets and IPython; use `await chat.ask(...)`"
          " instead."
      ) from e

    transcript = widgets.Output()
    status = widgets.HTML()
    prompt = widgets.Textarea(
        placeholder=placeholder, layout=widgets.Layout(width="100%")
    )
    send = widgets.Button(description="Send", button_style="primary")
    new_chat = widgets.Button(description="New chat")

    def render(turn: Turn) -> None:
      markdown = turn._repr_markdown_()  # pylint: disable=protected-access
      with transcript:
        display.display(display.Markdown(markdown))
        display.display(display.HTML("<hr>"))

    async def submit() -> None:
      question = prompt.value.strip()
      if not question:
        return
      prompt.value = ""
      send.disabled = True
      status.value = "<i>Thinking...</i>"
      try:
        render(await self.ask(question))
        status.value = ""
      except Exception as e:  # pylint: disable=broad-exception-caught
        status.value = f'<span style="color:#b00">{html.escape(str(e))}</span>'
      finally:
        send.disabled = False

    def on_send(_) -> None:
      asyncio.ensure_future(submit())

    def on_new_chat(_) -> None:
      self.reset()
      transcript.clear_output()
      status.value = ""

    send.on_click(on_send)
    new_chat.on_click(on_new_chat)
    for turn in self.history:
      render(turn)
    return widgets.VBox([
        transcript,
        status,
        prompt,
        widgets.HBox([send, new_chat]),
    ])


async def ask(
    results: Results, question: str, *, api_key: Optional[str] = None, **kwargs
) -> Turn:
  """Asks a single question about a run, with no conversation history.

  Equivalent to `await Chat(results, api_key=api_key, **kwargs).ask(question)`.
  """
  return await Chat(results, api_key=api_key, **kwargs).ask(question)
