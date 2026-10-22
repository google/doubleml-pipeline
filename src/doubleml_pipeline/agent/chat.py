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
argument takes precedence over both; it is a Gemini API key (AI Studio) unless
`vertexai=True` marks it as a Vertex AI express mode key.
"""

import asyncio
import dataclasses
import html
import os
import threading
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


def _environment_has_credentials() -> bool:
  """True when the SDK can authenticate from the environment alone."""
  if os.environ.get("GOOGLE_API_KEY"):
    return True
  return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().upper() in (
      "TRUE",
      "1",
  )


def _export_api_key(api_key: str, vertexai: bool) -> None:
  """Mirrors an explicit key into the environment.

  Older google-adk releases ignore the `client` passed to `Gemini` and build
  their own `genai.Client` from the environment, which then fails with
  "No API key was provided". Exporting the key (and the matching backend
  switch) keeps the explicit key working on every release. An express mode key
  must not be combined with a project/location, so those are cleared.
  """
  os.environ["GOOGLE_API_KEY"] = api_key
  os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE" if vertexai else "FALSE"
  if vertexai:
    os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    os.environ.pop("GOOGLE_CLOUD_LOCATION", None)


class Chat:
  """A multi-turn conversation with the causal insight agent."""

  def __init__(
      self,
      results: Results,
      *,
      api_key: Optional[str] = None,
      vertexai: Optional[bool] = None,
      model: str = generator.DEFAULT_AGENT_MODEL,
      skills_dir: Optional[PathLike] = None,
  ):
    """Builds the agent over one or more pipeline runs.

    Args:
      results: The `output_dir` of a completed run, or a mapping from run name
        to `output_dir` for several runs (treatment names must be unique).
      api_key: API key. When given it is used regardless of the environment;
        otherwise the environment decides between Vertex AI and
        `GOOGLE_API_KEY`, as in the generated agent's `.env.example`.
      vertexai: Which backend the key belongs to. Defaults to the Gemini API
        (keys from AI Studio). Pass `True` for a Vertex AI express mode key.
        A key that is restricted to the other service fails with
        `403 API_KEY_SERVICE_BLOCKED`; switch this flag in that case.
      model: Gemini model name.
      skills_dir: Directory of skills to load. Defaults to the skills bundled
        with the package.
    """
    reader = results_reader.CausalResultsReader(results)
    skills_root = (
        generator.BUNDLED_SKILLS_DIR if skills_dir is None else skills_dir
    )
    if api_key:
      vertexai = bool(vertexai)
      _export_api_key(api_key, vertexai)
      # The client is pinned to the chosen backend: the SDK would otherwise
      # still switch to Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is set.
      client = genai.Client(api_key=api_key, vertexai=vertexai)
      llm = google_llm.Gemini(model=model, client=client)
    else:
      if not _environment_has_credentials():
        raise ValueError(
            "No API key was given and the environment has no credentials."
            " Pass api_key=... (in Colab: add a secret named GOOGLE_API_KEY,"
            " enable notebook access for it, and read it with"
            " userdata.get('GOOGLE_API_KEY')), or set"
            " GOOGLE_GENAI_USE_VERTEXAI=TRUE with GOOGLE_CLOUD_PROJECT and"
            " GOOGLE_CLOUD_LOCATION for Vertex AI."
        )
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

  def _run_blocking(self, coro):
    """Runs `coro` to completion from synchronous code such as a widget callback.

    Notebook kernels already run an event loop on the main thread, so the
    coroutine is driven on that same loop through `nest_asyncio` when it is
    available (Colab and Jupyter ship it); this keeps the agent's HTTP client
    on the loop it was created on. Without `nest_asyncio` the coroutine runs
    on a dedicated background loop instead.
    """
    try:
      loop = asyncio.get_event_loop()
    except RuntimeError:
      loop = None
    if loop is None or loop.is_closed():
      return asyncio.run(coro)
    if not loop.is_running():
      return loop.run_until_complete(coro)
    try:
      import nest_asyncio  # pylint: disable=g-import-not-at-top
    except ImportError:
      return asyncio.run_coroutine_threadsafe(coro, self._ui_loop()).result()
    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)

  def _ui_loop(self) -> asyncio.AbstractEventLoop:
    """Event loop on a daemon thread, used when nest_asyncio is unavailable."""
    loop = getattr(self, "_ui_loop_instance", None)
    if loop is None or loop.is_closed():
      loop = asyncio.new_event_loop()
      threading.Thread(
          target=loop.run_forever, name="doubleml-chat-ui", daemon=True
      ).start()
      self._ui_loop_instance = loop
    return loop

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

    def on_send(_) -> None:
      question = prompt.value.strip()
      if not question:
        return
      prompt.value = ""
      send.disabled = True
      status.value = "<i>Thinking...</i>"
      try:
        # Answered synchronously inside the click handler: in Colab, widget
        # updates made from another thread, or from a task scheduled on the
        # kernel loop, are not shown until the next cell runs.
        render(self._run_blocking(self.ask(question)))
        status.value = ""
      except Exception as e:  # pylint: disable=broad-exception-caught
        status.value = f'<span style="color:#b00">{html.escape(str(e))}</span>'
      finally:
        send.disabled = False

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
    results: Results,
    question: str,
    *,
    api_key: Optional[str] = None,
    vertexai: Optional[bool] = None,
    **kwargs,
) -> Turn:
  """Asks a single question about a run, with no conversation history.

  Equivalent to
  `await Chat(results, api_key=api_key, vertexai=vertexai, **kwargs).ask(question)`.
  """
  chat = Chat(results, api_key=api_key, vertexai=vertexai, **kwargs)
  return await chat.ask(question)
