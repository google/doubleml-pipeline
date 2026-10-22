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

"""ADK agent integration for analyzing doubleml-pipeline results."""

from .generator import generate_agent
from .instruction import INSTRUCTION
from .results_reader import CausalResultsReader

# `Chat`, `Turn` and `ask` import google.adk, which the pipeline itself does
# not need, so they are loaded on first use.
_CHAT_EXPORTS = ("Chat", "Turn", "ask")


def __getattr__(name: str):
  if name in _CHAT_EXPORTS:
    from . import chat  # pylint: disable=g-import-not-at-top

    return getattr(chat, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CausalResultsReader",
    "INSTRUCTION",
    "generate_agent",
    *_CHAT_EXPORTS,
]
