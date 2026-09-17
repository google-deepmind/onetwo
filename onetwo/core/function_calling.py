# Copyright 2026 DeepMind Technologies Limited.
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

"""Core data types for function calling.

These types describe *which tools are available*, *what the model asked for*,
and *what the tools returned*. They are deliberately free of any rendering,
parsing, execution or transport concerns, and depend only on the standard
library and `onetwo.core.content`, so that they can be shared by:

  * the text-parsing tool-use strategies in `onetwo.stdlib.tool_use`, which
    recover a `FunctionCall` from prose emitted by the model, and
  * the model-native function-calling builtins in `onetwo.builtins`, which
    receive a `FunctionCall` as structured data from the backend.

Every backend, and every source of tools, has its own way of spelling these
concepts. The types here are a narrow waist between them: each tool source is
adapted to them once and each backend lowers them once, so that an agent
written against them runs unchanged across backends.

A conversation is a sequence of `ConversationTurn`s, one per message on the
wire. An exchange in which the model consults a single tool has four of them:

  ConversationTurn(role=USER, content='Who directed Alien?')
  ConversationTurn(
      role=MODEL,
      thought='I am not sure offhand, so I will look it up.',
      calls=[FunctionCall('search', kwargs={'query': 'Alien director'})],
  )
  ConversationTurn(
      role=TOOL,
      responses=[FunctionResponse('search', response='Ridley Scott')],
  )
  ConversationTurn(role=MODEL, content='Ridley Scott directed Alien.')

The MODEL turns are produced by a backend from the model's reply, and the TOOL
turn by whoever executes the calls. An exchange ends with a MODEL turn that
requests no calls.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any

from onetwo.core import content as content_lib


@dataclasses.dataclass(frozen=True)
class ToolSpec:
  """A backend-neutral specification of a tool available to the model.

  A `ToolSpec` is a declaration, not an implementation: it states what the
  model may call and with which arguments, and carries no code. The executable
  counterpart lives in `onetwo.stdlib.tool_use`, which additionally holds the
  callable and its prompting and error-handling policy; a `ToolSpec` is the
  projection of such a tool onto the model API. Separating them is what lets
  the backends, which must not depend on `onetwo.stdlib`, see the declaration
  without seeing the code.

  `parameters` is JSON Schema, the dialect that tool providers and model APIs
  already share, so lowering a spec onto a particular backend is a narrowing
  rather than a translation.

  Attributes:
    name: The unique name of the tool (e.g., 'search', 'calculator').
    description: Natural language description of what the tool does.
    parameters: JSON Schema (as a Mapping) describing the accepted keyword
      arguments, or None if the tool takes no parameters.
  """

  name: str
  description: str = ''
  parameters: Mapping[str, Any] | None = None


class ConversationRole(str, enum.Enum):
  """Role of a turn in a function-calling conversation."""

  USER = 'user'
  MODEL = 'model'
  TOOL = 'tool'


@dataclasses.dataclass(frozen=True)
class ConversationTurn:
  """A single turn in a function-calling conversation.

  Represents either:
  * A user message (`role == ConversationRole.USER`), with text/multimodal
    `content`.
  * A model response (`role == ConversationRole.MODEL`), which may include
    `thought`, text/multimodal `content`, and/or one or more function `calls`.
  * A tool execution turn (`role == ConversationRole.TOOL`), containing the
    `responses` from the executed function calls.

  Attributes:
    role: The role of this turn (USER, MODEL, or TOOL).
    content: Optional text or multimodal content (e.g., user question or model
      final text answer).
    calls: Sequence of function calls requested by the model (when `role ==
      ConversationRole.MODEL`). A model may request several calls in one turn,
      so this is a sequence rather than a single optional call.
    responses: Sequence of function responses returned by the environment (when
      `role == ConversationRole.TOOL`).
    thought: Optional internal reasoning emitted by the model before or
      alongside its function calls. A turn holds at most one, since the model
      emits one turn per generation: reasoning that the model does after seeing
      a tool result belongs to the next MODEL turn.
  """

  role: ConversationRole
  content: str | content_lib.ChunkList | None = None
  calls: Sequence['FunctionCall'] = ()
  responses: Sequence['FunctionResponse'] = ()
  # TODO: Thinking models attach an opaque signature to their
  # reasoning, which has to be echoed back verbatim for the model to keep its
  # reasoning state across turns. A `ChunkList` can carry one in the metadata
  # of its chunks; wire that up when lowering turns onto the backends.
  thought: str | content_lib.ChunkList | None = None


@dataclasses.dataclass
class FunctionCall:
  """A single function call (e.g., as may appear in one step of tool usage).

  Attributes:
    function_name: The name of the function to be called. Should match the name
      of a function that is registered in `onetwo.function_registry` (or when
      using a tool handler like `PythonToolUseEnvironment`, the name of a tool
      as per its `Tool.name`).
    args: The arguments of the function. Model-native function calling APIs
      express all arguments as keyword arguments, so this is typically empty for
      calls that originate from a backend, and is populated only by the
      text-parsing strategies, which can recover positional arguments.
    kwargs: The keyword arguments of the function.
  """

  function_name: str = ''
  args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FunctionResponse:
  """The outcome of executing a single `FunctionCall`.

  Attributes:
    function_name: The name of the function that was called, matching
      `FunctionCall.function_name`.
    response: The value returned by the function. `None` when `error` is set.
    error: A description of why the call failed, or `None` if it succeeded.
      Failures are represented rather than raised so that they can be reported
      back to the model, giving it the opportunity to correct the call.
  """

  function_name: str = ''
  response: Any = None
  error: str | None = None
