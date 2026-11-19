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

"""Model-native function calling, compliant with every backend.

A model that supports function calling does not execute anything: it replies
with a request to call a tool, and waits to be told what happened. Turning that
into a useful answer takes a loop -- call the model, run whatever it asked for,
give it the results, call it again -- which ends when the model replies without
asking for anything further.

That loop is the same whichever model / backend is answering. A backend
contributes exactly one thing to it: an implementation of
`llm.generate_with_tools`, which performs a single model call and knows how to
spell a tool declaration, a conversation, and a function call in whatever
dialect its API uses. Everything else -- dispatching the calls, reporting
failures back to the model, deciding when the exchange is over -- is here,
written once.

Usage:

```python
  backend.register()
  agent = agent_lib.Agent(tools=[get_weather, search])
  answer = ot.run(agent('What is the weather in London?'))
  print(answer)
```

A tool need not live in this process. An MCP server can be named in the same
list, and is connected to for the length of a run:

```python
  agent = agent_lib.Agent(
      tools=[get_weather, mcp_tools.server('http://localhost:8777/mcp')]
  )
```

To see what the model called and what came back, run with tracing:

```python
  result, trace = ot.run(agent(question), enable_tracing=True)
  print(agent_lib.format_tool_calls(trace))
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
import contextlib
import copy
import dataclasses
import re
from typing import Any, TypeAlias

from onetwo.agents import agents_base
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.core import results
from onetwo.core import tracing
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import tool_schema

# What a caller may name as a tool of the agent.
AgentTool: TypeAlias = (
    llm_tool_use.Tool | Callable[..., Any] | llm_tool_use.ToolSource
)

DEFAULT_FINAL_INSTRUCTION = (
    'You can no longer use tools. Answer now, using what you have found so far.'
)
DEFAULT_EMPTY_REPLY_INSTRUCTION = (
    'Your previous response was empty. Either call a tool or provide your'
    ' answer.'
)


def _is_empty_reply(turn: fc_types.ConversationTurn) -> bool:
  """Returns whether `turn` is a model turn with neither calls nor content."""
  return (
      turn.role == fc_types.ConversationRole.MODEL
      and not turn.calls
      and not str(turn.content or '').strip()
  )


def _answered(state: AgentState) -> bool:
  """Returns whether the model's latest turn is a reply without tool calls."""
  if not state.turns:
    return False
  last_turn = state.turns[-1]
  is_model_turn = last_turn.role == fc_types.ConversationRole.MODEL
  return is_model_turn and not last_turn.calls


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


def _as_tool(tool: llm_tool_use.Tool | Callable[..., Any]) -> llm_tool_use.Tool:
  """Returns the given tool in the fully declared form the model needs.

  A plain callable is introspected. A hand-written `Tool` need not declare its
  parameters; they are recovered from the function it wraps, since the model
  cannot call what it has not been told about. Only the missing pieces are
  filled in, so that the policy fields the author did set, such as
  `irrecoverable_error_types`, are preserved.

  Args:
    tool: A `Tool`, or a callable to make one from.

  Returns:
    A `Tool` with its parameters declared.
  """
  if not isinstance(tool, llm_tool_use.Tool):
    return tool_schema.tool_from_callable(tool)
  if tool.parameters is not None or tool.function is None:
    return tool
  derived = tool_schema.tool_from_callable(tool.function, name=tool.name)
  return dataclasses.replace(
      tool,
      parameters=derived.parameters,
      description=tool.description or derived.description,
  )


@dataclasses.dataclass
class AgentStep:
  """One model turn, together with the turns around it.

  A step is the unit in which the conversation grows. It appends up to three
  turns, in this order: `[user_turn, model_turn, tool_turn]`, leaving out those
  that are None. The model says something (possibly requesting calls), and the
  calls it requested are answered. The final step of an exchange has no
  `tool_turn`, since the model asked for nothing.

  Attributes:
    model_turn: What the model replied, with `role=MODEL`.
    tool_turn: The results of the calls requested in `model_turn`, with
      `role=TOOL`, or None if no calls were requested (or none were executed).
    user_turn: A turn with `role=USER` that the agent itself sends the model
      ahead of `model_turn`, or None. It is set on the call made with
      `Agent.final_instruction` once `max_steps` has run out, and with
      `Agent.empty_reply_instruction` after an empty model reply. The user's own
      request is not one: it opens `AgentState.turns` instead.
  """

  model_turn: fc_types.ConversationTurn
  tool_turn: fc_types.ConversationTurn | None = None
  user_turn: fc_types.ConversationTurn | None = None

  @property
  def turns(self) -> list[fc_types.ConversationTurn]:
    """Returns the turns this step appends to the conversation."""
    turns = [self.user_turn, self.model_turn, self.tool_turn]
    return [turn for turn in turns if turn is not None]

  def __hash__(self) -> int:
    """Required in case we want to compare updates for deduplication."""
    return hash(tuple(str(turn) for turn in self.turns))


@dataclasses.dataclass
class AgentState:
  """The conversation so far.

  Attributes:
    turns: Every turn exchanged, in order, starting with the user's request.
      This is what is sent to the model on the next call, so anything the model
      needs to see again -- including its own earlier calls and thoughts -- must
      be kept here.
    num_steps: How many steps have been taken, used to enforce a bound on the
      length of the exchange.
  """

  turns: list[fc_types.ConversationTurn] = dataclasses.field(
      default_factory=list
  )
  num_steps: int = 0

  def __add__(self, update: AgentStep) -> AgentState:
    """Required overload for supporting accumulation of updates."""
    new_state = copy.deepcopy(self)
    new_state += update
    return new_state

  def __iadd__(self, update: AgentStep) -> AgentState:
    """Required overload for supporting accumulation of updates."""
    self.turns.extend(update.turns)
    self.num_steps += 1
    return self

  def __hash__(self) -> int:
    """Required in case we want to compare states for deduplication."""
    return hash(tuple(str(turn) for turn in self.turns))


@dataclasses.dataclass(frozen=True)
class AgentEnvironment:
  """The tools of one run, in the two forms the loop needs them in.

  Most tools are fixed when the agent is built, but not all: a tool reached
  over a connection is callable only while that connection is open, so which
  tools exist, and whether they still work, is a property of the run rather
  than of the agent. This is that set, opened by `Agent.start_environment`
  when a run begins and released when it ends.

  Attributes:
    tools_by_name: The executable tools, indexed by the name the model uses to
      ask for them.
    specs: The declaration-only form, which is all the model is sent. The split
      is deliberate: a backend receives only this, so it cannot execute
      anything.
  """

  tools_by_name: Mapping[str, llm_tool_use.Tool]
  specs: Sequence[fc_types.ToolSpec]


# TODO: Improve the system prompt to inform the agent about the
# number of tool calls available to it.
@dataclasses.dataclass
class Agent(
    agents_base.SingleSampleAgent[
        str | content_lib.ChunkList | Sequence[fc_types.ConversationTurn],  # _I
        str,  # _O (output)
        AgentState,  # _S (state)
        AgentStep,  # _U (update)
        AgentEnvironment,  # _E (environment)
    ]
):
  """Agent that runs model-native function calling to completion.

  The exchange ends when the model replies without requesting any call, or when
  it has used up `max_steps` (see `final_instruction` for giving it one last
  chance to answer). Use `steps_exhausted` to tell those apart: it means the
  model was still working when it was cut off.

  The output of a run is the model's answer. The exchange that led to it is in
  the final state, which is also how to inspect what the model wants to call
  without running it:

  ```python
    _, state = await agent(
        'What is the weather in London?', max_steps=1, return_final_state=True
    )
    print(state.turns[-2].calls)  # The model turn, before its tool turn.
  ```

  Attributes:
    tools: The tools the model may call, either as `Tool`s or as plain
      callables, which are introspected into `Tool`s. Their declarations are
      sent to the model; their code is called from here and never leaves this
      process. A `ToolSource` may be listed among them to stand in for tools
      that must be fetched when a run starts, such as those of an MCP server:
      `tools=[get_weather, mcp_tools.server('http://localhost:8777/mcp')]`.
    max_steps: Bound on the number of steps in which the model may call tools,
      so that a model which keeps asking for them cannot loop forever.
    final_instruction: If set, and the model is still calling tools when
      `max_steps` runs out, it is given one more call, without tools and with
      this instruction, to answer from what it has found so far, e.g. 'Answer
      now, using what you have found.'. The instruction is recorded in the
      conversation like any other user turn. By default the exchange is cut off
      instead.
    empty_reply_instruction: If set, sent as a user turn to prompt the model
      again when it replies with neither text nor tool calls, e.g. 'Your
      previous response was empty. Either call a tool or provide your answer.'.
      By default (`''`) an empty reply ends the exchange.
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

  tools: Sequence[AgentTool] = ()
  max_steps: int = 10
  final_instruction: str = ''
  empty_reply_instruction: str = ''
  parallel_calls: bool = True
  stop_on_tool_error: bool = False
  generate_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # The two halves of `tools`, separated in `__post_init__`: those that could
  # be resolved there and then, and those that can only be had while a run is
  # in progress. See `start_environment`, which puts them back together.
  _tools_by_name: dict[str, llm_tool_use.Tool] = dataclasses.field(
      init=False, default_factory=dict
  )
  _specs: list[fc_types.ToolSpec] = dataclasses.field(
      init=False, default_factory=list
  )
  _sources: list[llm_tool_use.ToolSource] = dataclasses.field(
      init=False, default_factory=list
  )

  def __post_init__(self):
    for tool in self.tools:
      if isinstance(tool, llm_tool_use.ToolSource):
        # Not a tool but a promise of some, kept until a run starts.
        self._sources.append(tool)
        continue
      tool = _as_tool(tool)
      self._tools_by_name[tool.name] = tool
      self._specs.append(tool.to_spec())

  @contextlib.asynccontextmanager
  async def start_environment(self) -> AsyncIterator[AgentEnvironment]:
    """Gathers the tools of one run, and releases them when it ends.

    Overridden from base class (Agent).

    Tools given directly were resolved once, when the agent was built, and are
    the whole environment when no source was given -- the usual case, which
    opens nothing and costs nothing. Each source is asked for its tools here
    and told to release them when the run ends, whether it ended in an answer
    or in an exception.

    A source that names a tool already given directly replaces it, on the
    grounds that whatever was just fetched is the more current of the two.

    Yields:
      The tools of this run.
    """
    if not self._sources:
      yield AgentEnvironment(
          tools_by_name=self._tools_by_name, specs=self._specs
      )
      return
    async with contextlib.AsyncExitStack() as stack:
      tools_by_name = dict(self._tools_by_name)
      for source in self._sources:
        for tool in await stack.enter_async_context(source.open()):
          tool = _as_tool(tool)
          tools_by_name[tool.name] = tool
      # Declarations are derived from the same mapping the calls are dispatched
      # against, so that a replaced tool is declared once, in the place the
      # tool it replaced held, rather than twice over.
      yield AgentEnvironment(
          tools_by_name=tools_by_name,
          specs=[tool.to_spec() for tool in tools_by_name.values()],
      )

  def _should_retry_empty_reply(self, state: AgentState) -> bool:
    """Returns whether the last turn is an empty reply to re-prompt."""
    return (
        bool(self.empty_reply_instruction)
        and bool(state.turns)
        and _is_empty_reply(state.turns[-1])
    )

  def _done_answering(self, state: AgentState) -> bool:
    """Returns whether the model has replied and no retry is due."""
    return _answered(state) and not self._should_retry_empty_reply(state)

  def steps_exhausted(self, state: AgentState) -> bool:
    """Returns whether the exchange ran out of steps without an answer."""
    return state.num_steps >= self.max_steps and not self._done_answering(state)

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def _execute_call(
      self, call: fc_types.FunctionCall, *, environment: AgentEnvironment
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
      environment: The tools of this run. Not copied, since a tool reached over
        a connection is bound to it and a copy of it would not work.

    Returns:
      The result of the call, or a description of why it failed.
    """
    tool = environment.tools_by_name.get(call.function_name)
    if tool is None:
      available = ', '.join(sorted(environment.tools_by_name)) or 'none'
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
      inputs: str | content_lib.ChunkList | Sequence[fc_types.ConversationTurn],
      environment: AgentEnvironment | None = None,
  ) -> AgentState:
    """Returns a newly initialized state based on the input.

    Overridden from base class (Agent).

    Args:
      inputs: Either the user's request, as text or as multimodal content, or a
        conversation to continue.
      environment: Unused; the state of an exchange does not depend on which
        tools are available to it.
    """
    del environment
    if isinstance(inputs, (str, content_lib.ChunkList)):
      turns = [
          fc_types.ConversationTurn(
              role=fc_types.ConversationRole.USER, content=inputs
          )
      ]
    else:
      turns = list(inputs)
    return AgentState(turns=turns)

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(name='Agent.step')
  async def _sample_single_next_step(
      self, *, state: AgentState, environment: AgentEnvironment | None = None
  ) -> AgentStep:
    """Performs one model call and answers whatever it asks for.

    Overridden from base class (SingleSampleAgent).

    Args:
      state: Current state of the agent.
      environment: The tools of this run, as gathered by `start_environment`.
        Defaults to those given directly to the agent, which is all of them
        unless it was given a source.

    Returns:
      The model's turn, plus the tool turn answering it if it requested calls.
    """
    if environment is None:
      environment = AgentEnvironment(
          tools_by_name=self._tools_by_name, specs=self._specs
      )
    # Past the budget is only reached with a `final_instruction` (see
    # `is_finished`). Tools are withheld and the model is told so: a model
    # that has been calling tools all along goes on doing so otherwise,
    # declared or not.
    is_final_call = state.num_steps >= self.max_steps
    turns, tools, user_turn = state.turns, environment.specs, None
    if is_final_call:
      user_turn = fc_types.ConversationTurn(
          role=fc_types.ConversationRole.USER, content=self.final_instruction
      )
      turns, tools = turns + [user_turn], ()
    elif self._should_retry_empty_reply(state):
      # A model turn with neither calls nor text (e.g. thoughts only) did not
      # finish the run; inject a user nudge so turn roles alternate cleanly.
      user_turn = fc_types.ConversationTurn(
          role=fc_types.ConversationRole.USER,
          content=self.empty_reply_instruction,
      )
      turns = turns + [user_turn]
    model_turn = await llm.generate_with_tools(
        turns=turns, tools=tools, **self.generate_kwargs
    )
    if not model_turn.calls or is_final_call:
      self._record_step_in_trace(model_turn, ())
      return AgentStep(model_turn=model_turn, user_turn=user_turn)

    executables = [
        self._execute_call(call, environment=environment)
        for call in model_turn.calls
    ]
    if self.parallel_calls:
      # `par_iter` preserves the order of its inputs, so the responses line up
      # with the calls even when the tools finish out of order. That matters
      # for backends whose calls carry no id, where a response is matched to
      # its call by position.
      responses = list(await executing.par_iter(executables))
    else:
      responses = [await executable for executable in executables]
    self._record_step_in_trace(model_turn, responses)
    return AgentStep(
        model_turn=model_turn,
        tool_turn=fc_types.ConversationTurn(
            role=fc_types.ConversationRole.TOOL, responses=responses
        ),
        user_turn=user_turn,
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

  def is_finished(self, state: AgentState) -> bool:
    """Returns whether the exchange is over.

    Overridden from base class (Agent).

    The exchange is over once the model replies without requesting any call. A
    turn that contains both text and calls is not final: the model is thinking
    aloud on its way to using a tool. It is also over once `max_steps` steps
    have been taken, plus one when there is a `final_instruction` to give.

    Args:
      state: Current state of the agent.
    """
    if self._done_answering(state):
      return True
    budget = self.max_steps + (1 if self.final_instruction else 0)
    return state.num_steps >= budget

  def extract_output(self, state: AgentState) -> str:
    """Returns the model's answer as text, or '' if it has not given one.

    Overridden from base class (Agent).

    The answer is the content of the last turn, provided that it is a model
    turn without calls. There is none when the exchange was cut off while the
    model was still calling tools (see `steps_exhausted`). Either way, every
    turn of the exchange, with its calls and thoughts, is in `state.turns`.

    Args:
      state: Current (presumably final) state of the agent.
    """
    if not _answered(state) or state.turns[-1].content is None:
      return ''
    return str(state.turns[-1].content)


def _clip(value: Any, max_chars: int) -> str:
  """Returns `value` as text of at most `max_chars` characters."""
  text = str(value)
  if len(text) <= max_chars:
    return text
  if max_chars < len('...'):
    return text[:max_chars]
  return text[: max_chars - len('...')] + '...'


def format_tool_calls(
    execution_result: results.ExecutionResult, max_chars: int = 160
) -> str:
  """Returns the thoughts and tool calls recorded in a trace, in order.

  Each call is shown with what it returned, or with why it failed:

    thought: I should look that up.
    get_weather({'location': 'London'})
      -> {'temp': 15}
    explode({'reason': 'boom'})
      !! ValueError: boom

  A call without arguments is shown as `name()`. Calls are not grouped by step,
  since the trace details do not say where one step ends.

  Args:
    execution_result: The trace of a run, as returned by `ot.run` with
      `enable_tracing=True`.
    max_chars: Longer thoughts, results and errors are cut to this length.

  Returns:
    One line per thought and per call, and one per call's outcome.
  """
  lines = []
  for detail in fc_types.collect_tool_call_details(execution_result):
    detail_type = detail.get('type')
    if detail_type == 'thought':
      lines.append(f'thought: {_clip(detail.get("text", ""), max_chars)}')
      continue
    if detail_type == 'function_call':
      lines.append(f'{detail["name"]}({detail.get("args") or ""})')
    elif detail_type == 'function_response':
      # A response the backend could not match to any call it recorded.
      lines.append(f'{detail["name"]} (no matching call)')
    else:
      raise ValueError(f'Unknown detail type: {detail_type}')
    response = detail.get('response')
    if isinstance(response, dict) and set(response) == {'error'}:
      lines.append(f'  !! {_clip(response["error"], max_chars)}')
    else:
      lines.append(f'  -> {_clip(response, max_chars)}')
  return '\n'.join(lines)


# The docids of the hits that retrieval tools return, e.g. the JSON headers
# `{"docid": "...", "score": ...}` of the OneTwo index MCP server's `search`.
_DOCID_PATTERN = re.compile(r'"docid": "([^"]+)"')


def _docids_or_clipped(result: Any, max_chars: int) -> str:
  """Returns the docids in `result`, or else `result` as text, clipped."""
  docids = _DOCID_PATTERN.findall(str(result))
  if docids:
    return ', '.join(docids)
  return _clip(result, max_chars)


def format_agent_steps(
    state: AgentState,
    format_result: Callable[[Any], str] | None = None,
    max_chars: int = 160,
) -> str:
  """Returns the steps of a run, one line per tool call, and the answer.

  Unlike `format_tool_calls`, which reads a trace, this reads the final state of
  a run (`agent(..., return_final_state=True)`), so it needs no tracing and
  also shows the answer:

    1. get_weather('London') -> {'temp': 15}
    2. explode('boom') -> error: ValueError: boom
    3. search('lions') -> lion_chunk1, lion_chunk2
    4. No tool call, the model answers: It is 15 degrees.

  The calls of a single step share its number. A call that was not executed
  (one requested on the final call, see `Agent.final_instruction`) is shown as
  such.

  Args:
    state: The final state of a run.
    format_result: Formats what a tool returned. By default, a result with
      `"docid": "..."` entries (e.g. the hits of an index MCP server's `search`)
      is shown as its docids, and any other result as text, cut to `max_chars`.
    max_chars: Longer results, errors and answers are cut to this length. Not
      applied to the output of `format_result`.

  Returns:
    One line per tool call and one for the answer, if the model gave one.
  """
  if format_result is None:
    format_result = lambda result: _docids_or_clipped(result, max_chars)
  lines = []
  step = 0
  for index, turn in enumerate(state.turns):
    if turn.role != fc_types.ConversationRole.MODEL:
      continue
    step += 1
    if not turn.calls:
      answer = _clip(turn.content or '', max_chars)
      lines.append(f'{step}. No tool call, the model answers: {answer}')
      continue
    # The results of this turn's calls are in the tool turn that follows it.
    responses = []
    if index + 1 < len(state.turns):
      next_turn = state.turns[index + 1]
      if next_turn.role == fc_types.ConversationRole.TOOL:
        responses = list(next_turn.responses)
    for call_index, call in enumerate(turn.calls):
      args = ', '.join(
          [repr(value) for value in call.args]
          + [repr(value) for value in call.kwargs.values()]
      )
      if call_index >= len(responses):
        result = 'not executed'
      elif responses[call_index].error:
        result = f'error: {_clip(responses[call_index].error, max_chars)}'
      else:
        result = format_result(responses[call_index].response)
      lines.append(f'{step}. {call.function_name}({args}) -> {result}')
  return '\n'.join(lines)
