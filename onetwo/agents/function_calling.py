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

"""Model-native function calling, implemented once for every backend.

A model that supports function calling does not execute anything: it replies
with a request to call a tool, and waits to be told what happened. Turning that
into a useful answer takes a loop -- call the model, run whatever it asked for,
give it the results, call it again -- which ends when the model replies without
asking for anything further.

That loop is the same whichever model is answering, so it lives here rather
than in the backends. A backend contributes exactly one thing to it: an
implementation of `llm.generate_with_tools`, which performs a single model call
and knows how to spell a tool declaration, a conversation, and a function call
in whatever dialect its API uses. Everything else -- dispatching the calls,
reporting failures back to the model, deciding when the exchange is over -- is
here, written once.

Usage:

```python
  backend.register()
  agent = function_calling.FCAgent(tools=[get_weather, search])
  turn = ot.run(agent('What is the weather in London?'))
  print(turn.content)
```
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
from typing import Any

from onetwo.agents import agents_base
from onetwo.builtins import llm
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.core import tracing
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import tool_schema


def _traced(response: fc_types.FunctionResponse) -> Any:
  """Returns the outcome of a call in a form the execution trace can carry.

  Args:
    response: The outcome of one call.

  Returns:
    The value the tool returned, or the reason it failed.
  """
  if response.error:
    return {'error': response.error}
  return response.response


@dataclasses.dataclass
class FCStep:
  """One model turn, together with the tool turn that answers it.

  A step is the unit in which the conversation grows: the model says something
  (possibly requesting calls), and the calls it requested are answered. The
  final step of an exchange has no `tool_turn`, since the model asked for
  nothing.

  Attributes:
    model_turn: What the model replied, with `role=MODEL`.
    tool_turn: The results of the calls requested in `model_turn`, with
      `role=TOOL`, or None if no calls were requested (or none were executed).
  """

  model_turn: fc_types.ConversationTurn
  tool_turn: fc_types.ConversationTurn | None = None

  @property
  def turns(self) -> list[fc_types.ConversationTurn]:
    """Returns the turns this step appends to the conversation."""
    return [self.model_turn] + ([self.tool_turn] if self.tool_turn else [])

  def __hash__(self) -> int:
    """Required in case we want to compare updates for deduplication."""
    return hash((str(self.model_turn), str(self.tool_turn)))


@dataclasses.dataclass
class FCState:
  """The conversation so far.

  Attributes:
    history: Every turn exchanged, in order, starting with the user's request.
      This is what is sent to the model on the next call, so anything the model
      needs to see again -- including its own earlier calls and thoughts -- must
      be kept here.
    num_steps: How many steps have been taken, used to enforce a bound on the
      length of the exchange.
  """

  history: list[fc_types.ConversationTurn] = dataclasses.field(
      default_factory=list
  )
  num_steps: int = 0

  def __add__(self, update: FCStep) -> FCState:
    """Required overload for supporting accumulation of updates."""
    new_state = copy.deepcopy(self)
    new_state += update
    return new_state

  def __iadd__(self, update: FCStep) -> FCState:
    """Required overload for supporting accumulation of updates."""
    self.history.extend(update.turns)
    self.num_steps += 1
    return self

  def __hash__(self) -> int:
    """Required in case we want to compare states for deduplication."""
    return hash(tuple(str(turn) for turn in self.history))


@dataclasses.dataclass
class FCAgent(
    agents_base.SingleSampleAgent[
        str | Sequence[fc_types.ConversationTurn],  # _I (inputs)
        fc_types.ConversationTurn,  # _O (output)
        FCState,  # _S (state)
        FCStep,  # _U (update)
        None,  # _E (environment)
    ]
):
  """Agent that runs model-native function calling to completion.

  The exchange ends when the model replies without requesting any call, or when
  `max_steps` steps have been taken. Use `steps_exhausted` to tell those two
  apart: the second means the model was still working when it was cut off.

  A single step is available by asking for one, which is how to inspect what
  the model wants to call without running it:

  ```python
    turn = await agent('What is the weather in London?', max_steps=1)
    print(turn.calls)
  ```

  Attributes:
    tools: The tools the model may call, either as `Tool`s or as plain
      callables, which are introspected into `Tool`s. Their declarations are
      sent to the model; their code is called from here and never leaves this
      process.
    max_steps: Bound on the number of model calls, so that a model which keeps
      asking for tools cannot loop forever.
    parallel_calls: Whether to execute the calls of a single turn concurrently.
      Models routinely request several at once and tools are usually IO-bound,
      so this is on by default. Results are always reported in the order the
      model asked for them, whatever order they complete in.
    stop_on_tool_error: If True, a failing tool aborts the exchange by raising.
      By default failures are reported to the model instead, which lets it
      correct a mistaken call (see `_execute_call`).
    generate_kwargs: Generation parameters passed through to the backend on
      every call, e.g. `{'temperature': 0.0}`.
  """

  tools: Sequence[llm_tool_use.Tool | Callable[..., Any]] = ()
  max_steps: int = 10
  parallel_calls: bool = True
  stop_on_tool_error: bool = False
  generate_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # Derived from `tools` in `__post_init__`: the executable form, indexed by
  # the name the model will use, and the declaration-only form sent to the
  # model. The split is deliberate -- a backend receives only the latter, so it
  # cannot execute anything.
  _tools_by_name: dict[str, llm_tool_use.Tool] = dataclasses.field(
      init=False, default_factory=dict
  )
  _specs: list[fc_types.ToolSpec] = dataclasses.field(
      init=False, default_factory=list
  )

  def __post_init__(self):
    for tool in self.tools:
      if not isinstance(tool, llm_tool_use.Tool):
        tool = tool_schema.tool_from_callable(tool)
      elif tool.parameters is None and tool.function is not None:
        # A hand-written `Tool` need not spell out its parameters; recover them
        # from the function, since the model cannot call what is not declared.
        # Only the missing pieces are filled in, so that the policy fields the
        # author did set, such as `irrecoverable_error_types`, are preserved.
        derived = tool_schema.tool_from_callable(tool.function, name=tool.name)
        tool = dataclasses.replace(
            tool,
            parameters=derived.parameters,
            description=tool.description or derived.description,
        )
      self._tools_by_name[tool.name] = tool
      self._specs.append(tool.to_spec())

  def steps_exhausted(self, state: FCState) -> bool:
    """Returns whether the exchange stopped only because it ran out of steps."""
    return state.num_steps >= self.max_steps

  @executing.make_executable(copy_self=False)
  async def _execute_call(
      self, call: fc_types.FunctionCall
  ) -> fc_types.FunctionResponse:
    """Runs one call requested by the model and reports what happened.

    A failure is returned rather than raised, so that it can be shown to the
    model on the next turn: a model that calls a tool that does not exist, or
    passes an argument that a tool rejects, is usually able to correct itself
    once told. Aborting instead would throw away a recoverable conversation.

    The exceptions to that are failures there is no point in showing the model,
    which the tool declares in `Tool.irrecoverable_error_types`, and the case
    where the caller has asked to fail fast with `stop_on_tool_error`.

    Args:
      call: The call requested by the model.

    Returns:
      The result of the call, or a description of why it failed.
    """
    tool = self._tools_by_name.get(call.function_name)
    if tool is None:
      available = ', '.join(sorted(self._tools_by_name)) or 'none'
      return fc_types.FunctionResponse(
          function_name=call.function_name,
          error=(
              f'Unknown tool: {call.function_name!r}. Available tools:'
              f' {available}.'
          ),
          id=call.id,
      )
    try:
      result = await tool(*call.args, **call.kwargs)
    except Exception as e:  # pylint: disable=broad-except
      irrecoverable = tool.irrecoverable_error_types or []
      if self.stop_on_tool_error or type(e).__name__ in irrecoverable:
        raise
      return fc_types.FunctionResponse(
          function_name=call.function_name,
          error=f'{type(e).__name__}: {e}',
          id=call.id,
      )
    return fc_types.FunctionResponse(
        function_name=call.function_name, response=result, id=call.id
    )

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def initialize_state(
      self,
      inputs: str | Sequence[fc_types.ConversationTurn],
      environment: None = None,
  ) -> FCState:
    """Returns a newly initialized state based on the input.

    Overridden from base class (Agent).

    Args:
      inputs: Either the user's request, or a conversation to continue.
      environment: Unused; this agent does not need an environment.
    """
    del environment
    if isinstance(inputs, str):
      history = [
          fc_types.ConversationTurn(
              role=fc_types.ConversationRole.USER, content=inputs
          )
      ]
    else:
      history = list(inputs)
    return FCState(history=history)

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(name='FCAgent.step')
  async def _sample_single_next_step(
      self, *, state: FCState, environment: None = None
  ) -> FCStep:
    """Performs one model call and answers whatever it asks for.

    Overridden from base class (SingleSampleAgent).

    Args:
      state: Current state of the agent.
      environment: Unused; this agent does not need an environment.

    Returns:
      The model's turn, plus the tool turn answering it if it requested calls.
    """
    del environment
    model_turn = await llm.generate_with_tools(
        messages=state.history, tools=self._specs, **self.generate_kwargs
    )
    if not model_turn.calls:
      self._record_step_in_trace(model_turn, ())
      return FCStep(model_turn=model_turn)

    executables = [self._execute_call(call) for call in model_turn.calls]
    if self.parallel_calls:
      # `par_iter` preserves the order of its inputs, so the responses line up
      # with the calls even when the tools finish out of order. That matters
      # for backends whose calls carry no id, where a response is matched to
      # its call by position.
      responses = list(await executing.par_iter(executables))
    else:
      responses = [await executable for executable in executables]
    self._record_step_in_trace(model_turn, responses)
    return FCStep(
        model_turn=model_turn,
        tool_turn=fc_types.ConversationTurn(
            role=fc_types.ConversationRole.TOOL, responses=responses
        ),
    )

  def _record_step_in_trace(
      self,
      model_turn: fc_types.ConversationTurn,
      responses: Sequence[fc_types.FunctionResponse],
  ) -> None:
    """Records what this step asked for, and what came back, into the trace.

    Args:
      model_turn: What the model replied.
      responses: The outcome of each call the model requested, in the order it
        requested them. Empty when it requested none.
    """
    details: list[fc_types.ToolCallDetail] = []
    if model_turn.thought:
      details.append(fc_types.ThoughtDetail(text=str(model_turn.thought)))
    for index, call in enumerate(model_turn.calls):
      response = responses[index] if index < len(responses) else None
      details.append(
          fc_types.FunctionCallDetail(
              name=call.function_name,
              args=dict(call.kwargs),
              # `id` is absent on backends that do not correlate calls; the
              # detail then omits it rather than carrying an empty string.
              id=call.id or None,
              response=None if response is None else _traced(response),
          )
      )
    info: dict[str, Any] = {
        fc_types.TOOL_CALLS_INFO_KEY: len(model_turn.calls),
    }
    if details:
      info[fc_types.TOOL_DETAILS_INFO_KEY] = [d.to_dict() for d in details]
    tracing.update_info(info)

  def is_finished(self, state: FCState) -> bool:
    """Returns whether the exchange is over.

    Overridden from base class (Agent).

    The exchange is over once the model replies without requesting any call. A
    turn that contains both text and calls is not final: the model is thinking
    aloud on its way to using a tool.

    Args:
      state: Current state of the agent.
    """
    if state.num_steps >= self.max_steps:
      return True
    if not state.history:
      return False
    last_turn = state.history[-1]
    is_model_turn = last_turn.role == fc_types.ConversationRole.MODEL
    return is_model_turn and not last_turn.calls

  def extract_output(self, state: FCState) -> fc_types.ConversationTurn | None:
    """Returns the model's most recent turn.

    Overridden from base class (Agent).

    Args:
      state: Current (presumably final) state of the agent.
    """
    for turn in reversed(state.history):
      if turn.role == fc_types.ConversationRole.MODEL:
        return turn
    return None
