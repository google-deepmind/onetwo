# Copyright 2024 DeepMind Technologies Limited.
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

"""Implementation of a Python Planning strategy using the Agent framework.

In the Python Planning strategy, we iteratively prompt an LLM to generate blocks
of Python code, which may include calls to "tools" that are provided in the form
of functions. In each step, the LLM is shown the results of executing the
previous steps, so that the LLM can either build on the previously generated
code (in case of success), or else correct errors that became apparent from the
execution results.

Inspired by various research in Python-based tool orchestration, such as
ViperGPT (https://arxiv.org/pdf/2303.08128.pdf) and AdaPlanner
(https://arxiv.org/pdf/2305.16653.pdf)

Adopting the "Agent" framework for the strategy implementation means that:
* The full state of the strategy at each step is encapsulated in a
  serializable state object (PythonPlanningState), which in this case is
  represented as a series of steps, each of which involve generation and
  execution of a Python program.
* The prompt template that defines the calls made to the LLM in each step takes
  such a state object as input and performs exactly one step of the strategy.
  The step-loop is implemented in the agent code, rather than in the prompt
  template.
* This ensures that in addition to running the full Python Planning strategy
  end-to-end, we are also able to support stop/restart, stepwise execution, and
  composition of Python Planning with other agent strategies such as beam search
  for pursuing multiple possible trajectories in parallel.
"""

import abc
from collections.abc import AsyncIterator
import contextlib
import dataclasses
import re
from typing import Protocol, Sequence, TypeAlias

from onetwo.agents import agents_base
from onetwo.builtins import composables
from onetwo.builtins import prompt_templating
from onetwo.core import composing
from onetwo.core import constants
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import templating
from onetwo.core import tracing
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

# Aliases for brevity.
_ExecutionStatus: TypeAlias = python_execution.ExecutionStatus
_PythonSandboxState: TypeAlias = python_tool_use.PythonSandboxState


@dataclasses.dataclass
class PythonPlanningStep:
  """One step of a Python Planning agent state.

  Attributes:
    is_finished: Whether this is intended to be the final state. If True, then
      the observation can be treated as the final answer.
    code: Python code to be executed.
    result: The output produced when executing the Python code.
    execution_status: Status indicating whether the code execution succeeded.
      Will be `None` if the code has not been executed yet.
  """

  is_finished: bool = False
  code: str = ''
  result: str = ''
  execution_status: _ExecutionStatus | None = None


# In the Python Planning strategy, the state consists of a monotonically
# increasing sequence of steps (each of which involve generation and execution
# of a Python program).
PythonPlanningState = agents_base.UpdateListState[str, PythonPlanningStep]


def _get_result_str(
    result: python_execution.SandboxResult, is_irrecoverable_error: bool
) -> str:
  """Returns a string representation of the result for showing to the LLM."""
  parts = []

  # Content that the sandbox printed to stdout.
  if result.stdout:
    parts.append(result.stdout)

  # Value of the final statement (if any). This is important to output in
  # particular in the case where the Python sandbox is used like a calculator,
  # e.g., where the last line is a simple expression like `x + 2`. If the final
  # expression value is `None`, that usually means that the last statement was
  # not an evaluable expression (e.g., was an `if` statement or something like
  # that). In those cases, we only output `None` in the case where the sandbox
  # did not print anything else to `stdout`.
  if result.final_expression_value or (
      not result.stdout and result.execution_status == _ExecutionStatus.SUCCESS
  ):
    parts.append(str(result.final_expression_value))

  # Error message (prompting for retry, where relevant).
  if (
      result.execution_status == _ExecutionStatus.EXECUTION_ERROR
      and not is_irrecoverable_error
  ):
    parts.append(
        'The python code above raises an exception:\n\n'
        f'{result.status_message}\n\n'
        'Rewrite the python code to eliminate the exception.\n'
        'Do not start from scratch, you can assume that the variables from '
        'previous steps are available.\n'
    )
  elif result.execution_status != _ExecutionStatus.SUCCESS:
    parts.append(
        f'{str(result.execution_status.value)}: {result.status_message}'
    )

  return '\n'.join(parts)


def _sandbox_state_from_agent_state(
    agent_state: PythonPlanningState,
) -> _PythonSandboxState:
  """Returns a sandbox state (code sequence) from the given agent state."""
  # The state of the PythonSandbox is determined solely by the sequence of code
  # snippets that have been executed on it so far.
  code_sequence = tuple(step.code for step in agent_state.updates)
  return code_sequence


def _regex_search(start_fence: str, end_fence: str, text: str) -> str:
  """Returns substring between the given fences, or itself if nothing found."""
  found = re.search(f'(?s){start_fence}(.*?){end_fence}', text)
  if found:
    text = found.group().replace(start_fence, '').replace(end_fence, '').strip()
  return text


def _parse_llm_reply_code(llm_reply: str) -> str:
  """Parses the LLM reply to remove fences that aren't strict code.

  The Flash sv2 gemini models tend to start their code with '```python'
  '```tool_code', etc, instead of just outputting the code, so we must
  extract those out if we encounter them. We also remove any trailing text after
  the next ``` if we find one. Older models don't have these issues,
  and this code should do nothing besides .strip() in that case.

  Args:
    llm_reply: The LLM reply to parse.

  Returns:
    The stripped LLM reply with the start fence removed if found, and anything
    after the next '```' removed as well.
  """
  llm_reply = llm_reply.strip()
  first_line = llm_reply.split('\n')[0]
  # If we find ``` in the first line, we remove the line, assuming it's a fence.
  if '```' == first_line[:3]:
    llm_reply = llm_reply[len(first_line) :]
  # Similarly if the first line is simply 'tool_code' or 'python', then it was
  # likely intended to be part of a fence (since we end the prompt with ```).
  elif 'tool_code' == first_line or 'python' == first_line:
    llm_reply = llm_reply[len(first_line) :]

  # Any '```' now is treated as the stop fence. Anything beyond is discarded.
  llm_reply = _regex_search('', '```', llm_reply)

  return llm_reply.strip()


DEFAULT_PYTHON_PLANNING_INSTRUCTION = """\
Your task is to answer questions through the help of a Python interpreter and \
a set of tools that can be invoked via Python function calls. You do not need \
to use all of the tools for every question, and you can also call the same \
tool multiple times -- whatever is most effective for answering the question \
accurately and factually. You will answer the question in one or more steps. \
In each step, you should output a block of Python code enclosed in triple back \
ticks, after which the code will automatically be executed, and the output \
will be printed immediately afterward. You can then proceed with outputting \
the next Python code block, where appropriate. Each code block can reference \
variables defined in earlier code blocks. At the end, you should print the \
final answer and then call the function `exit()` to indicate that you are \
done.\
"""


DEFAULT_PYTHON_PLANNING_PROMPT_TEXT = """\
{#- Preamble: Instructions and tools description -#}
{%- role name='system' -%}
{% if instruction -%}
{{ instruction + '\n\n' }}
{%- endif -%}
Here is a list of available tools:
{%- for tool in tools -%}
{{ '\n' }}
Tool name: {{ tool.name }}
Tool description: {{ tool.description }}
{% if tool.example -%}
  Tool example: {{ tool.example_str }}
{%- endif -%}
{%- endfor -%}

{#- Preamble: Few-shot exemplars -#}
{%- for exemplar in exemplars -%}
{{ '\n' }}
**Question**: {{ exemplar.inputs + '\n' }}
{%- for step in exemplar.updates -%}
```
{{ step.code }}
```
{{ step.result | trim + '\n\n' }}
{%- if step.is_finished -%}
**Done**
{%- endif -%}
{%- endfor -%}
{%- endfor -%}
{%- endrole -%}

{#- Start of the processing of the actual inputs. -#}

{#- Render the original question. -#}
{{ '\n' }}
**Question**: {{ state.inputs + '\n' }}

{#- Render the current state (i.e., any steps performed up till now). -#}
{%- for step in state.updates -%}
```
{{ step.code }}
```
{{ step.result | trim + '\n\n' }}
{%- if step.is_finished -%}
**Done**
{%- endif -%}
{%- endfor -%}

{#- Get a response from the LLM for the next step and return it. -#}
```{{ store('llm_reply', generate_text(stop=['```']) | trim) }}
"""

# Default set of exemplars that can be used for calls to a Python Planning
# prompt.
DEFAULT_PYTHON_PLANNING_EXEMPLARS = [
    PythonPlanningState(
        inputs='How much taller is Everest than Mount Fuji?',
        updates=[
            PythonPlanningStep(
                is_finished=False,
                code="""\
# First we need to find out how tall are Everest and Mount Fuji.
height1 = search('how tall is Everest?')
height2 = search('how tall is Mount Fuji?')
print(f'Everest: {height1}, Mount Fuji: {height2}')""",
                result='Everest: 8,849m, Mount Fuji: 3,776m',
                execution_status=_ExecutionStatus.SUCCESS,
            ),
            PythonPlanningStep(
                is_finished=False,
                code="""\
# Now we extract the numbers.
num1 = firstnumber(height1)
num2 = firstnumber(height2)
# And we compare them.
print(num1 - num2)""",
                result='5073',
                execution_status=_ExecutionStatus.SUCCESS,
            ),
            PythonPlanningStep(
                is_finished=True,
                code="""\
# We can now give the answer and exit.
print('Everest is 5,073 meters taller than Mount Fuji.')
exit()""",
                result='Everest is 5,073 meters taller than Mount Fuji.',
                execution_status=_ExecutionStatus.SUCCESS,
            ),
        ],
    ),
]


class PythonPlanningPromptProtocol(Protocol):
  """Interface for prompt usable with PythonPlanningAgent.prompt."""

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @abc.abstractmethod
  async def __call__(
      self,
      exemplars: list[PythonPlanningState],
      state: PythonPlanningState,
      tools: Sequence[llm_tool_use.Tool],
  ) -> str:
    """Executes the prompt template on the given args and returns the result.

    Args:
      exemplars: Few-shot exemplars to include in the prompt.
      state: Current state of the agent, i.e., the original inputs and sequence
        of steps (if any) that have been performed so far.
      tools: The tools that are available to be used and whose descriptions are
        to be listed in the prompt. Any tools referenced in the `exemplars`
        should be registered here, although it is not strictly required for all
        of the tools to be illustrated in `exemplars`.

    Returns:
      The LLM reply, which is presumably a block of Python code. The caller is
      responsible for verifying it is valid Python code, stripping off any
      enclosing "``` ... ```" marks, and then executing it.
    """


@dataclasses.dataclass
class PythonPlanningPromptJ2(
    PythonPlanningPromptProtocol, prompt_templating.JinjaTemplateWithCallbacks
):
  """JinjaTemplate usable with PythonPlanningAgent.prompt.

  Attributes:
    instruction: The instruction to use in the prompt. Does not need to begin
      or end with a newline, as these will be handled automatically.
  """

  # Overriding default value of attribute defined in templating.JinjaTemplate.
  text: str = DEFAULT_PYTHON_PLANNING_PROMPT_TEXT

  instruction: str = DEFAULT_PYTHON_PLANNING_INSTRUCTION

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def __call__(
      self,
      exemplars: list[PythonPlanningState],
      state: PythonPlanningState,
      tools: Sequence[llm_tool_use.Tool],
  ) -> str:
    """See PythonPlanningPromptProtocol."""
    result = await self.render(
        instruction=self.instruction,
        exemplars=exemplars,
        state=state,
        tools=tools,
    )
    if 'llm_reply' in result:
      # If the prompt succeeded in running to the end, we should come here.
      return result['llm_reply']
    elif templating.ERROR in result:
      # If an error is raised whle processing the prompt, we should come here.
      return f'{constants.ERROR_STRING}: {result[templating.ERROR]}'
    else:
      # If we come here, then there must be a bug somewhere.
      raise ValueError(
          "PythonPlanning prompt result missing 'llm_reply' and lacking error"
          f' message to explain why: {result}'
      )


@dataclasses.dataclass
class PythonPlanningPromptComposable(PythonPlanningPromptProtocol):
  """Composable prompt for PythonPlanningAgent.prompt.

  Currently the question supports multimodal inputs.
  In the future, if we have models or tools that can output multimodal content,
  we can extend this to support multimodal PythonPlanning states too.

  Attributes:
    instruction: The instruction to use in the prompt. Does not need to begin
      or end with a newline, as these will be handled automatically.
    instruction_role: The role to use for the instruction parts of the prompt,
      which describe the available tools or the overall task.
    response_role: The role to use for the response parts of the prompt, i.e.
      those that will be provided by the model.
  """

  instruction: str = DEFAULT_PYTHON_PLANNING_INSTRUCTION
  instruction_role: content_lib.PredefinedRole = content_lib.PredefinedRole.USER
  response_role: content_lib.PredefinedRole = content_lib.PredefinedRole.MODEL

  def _render_tool(self, tool: llm_tool_use.Tool) -> composing.Composable:
    """Returns a composable with the prompt content describing a single tool."""
    e = composables.c(
        f'\nTool name: {tool.name}\n',
        role=self.instruction_role,
    ) + composables.c(
        f'Tool description: {tool.description}\n',
        role=self.instruction_role,
    )
    if tool.example:
      e += composables.c(
          f'Tool example: {tool.example_str}\n',
          role=self.instruction_role,
      )
    return e

  def _render_tools(
      self,
      tools: Sequence[llm_tool_use.Tool],
  ) -> composing.Composable:
    """Returns a composable with the prompt content describing all tools."""
    e = composables.f(
        'Here is a list of available tools:\n', role=self.instruction_role
    )
    for tool in tools:
      e += self._render_tool(tool)
    return e

  def _render_step(
      self, step: PythonPlanningStep
  ) -> composing.Composable:
    """Returns a composable for rendering a single PythonPlanning step."""
    e = composables.c(content_lib.ChunkList())
    e += composables.c(
        f'```\n{step.code}\n```\n',
        role=self.response_role,
    )
    e += composables.c(
        f'{step.result.strip()}\n\n',
        role=self.instruction_role,
    )
    if step.is_finished:
      e += composables.c(
          '**Done**\n',
          role=content_lib.PredefinedRole.MODEL,
      )
    return e

  def _render_state(self, state: PythonPlanningState) -> composing.Composable:
    """Returns a composable for rendering the agent state for one example."""
    e = composables.c('\n**Question**: ', role=content_lib.PredefinedRole.USER)
    e += composables.c(state.inputs, role=content_lib.PredefinedRole.USER)
    e += composables.f('\n', role=content_lib.PredefinedRole.USER)
    for step in state.updates:
      e += self._render_step(step)
    return e

  def _render_exemplars(
      self, exemplars: Sequence[PythonPlanningState]
  ) -> composing.Composable:
    """Returns a composable for rendering all of the few-shot exemplars."""
    e = composables.c(content_lib.ChunkList())
    for exemplar in exemplars:
      e += self._render_state(exemplar)
    return e

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def __call__(
      self,
      exemplars: list[PythonPlanningState],
      state: PythonPlanningState,
      tools: Sequence[llm_tool_use.Tool],
  ) -> content_lib.ChunkList | str:
    e = composables.c(content_lib.ChunkList())
    # Instruction
    if self.instruction:
      e += composables.c(
          self.instruction + '\n\n', role=self.instruction_role)

    e += self._render_tools(tools)
    e += self._render_exemplars(exemplars)
    e += self._render_state(state)
    e += composables.f('```', role=self.response_role)
    # Using chat instead of generate_text to return ChunkList.
    e += composables.store('llm_reply', composables.chat(stop=['\n```']))

    try:
      _ = await e
    except ValueError as error_message:
      e = {}
      e['llm_reply'] = f'#ERROR#: {error_message}'
    # Checking 'llm_reply' in e itself results in a ValueError.
    # So just try to return it for now.
    return e['llm_reply'].strip()


# TODO: Generalize the output type from `str` to `Any`.
@dataclasses.dataclass
class PythonPlanningAgent(
    agents_base.SingleSampleAgent[
        str,  # _I (inputs)
        str,  # _O (outputs)
        PythonPlanningState,  # _S (state)
        PythonPlanningStep,  # _U (update)
        python_tool_use.PythonToolUseEnvironment,  # _E (environment)
    ]
):
  """Agent that repeatedly generates/executes Python code for tool use.

  Attributes:
    prompt: Prompt template used for prompting the LLM at each step.
    exemplars: Few-shot exemplars to include in the prompt.
    environment_config: Config controlling the behavior of environments created
      by this agent. This includes list of the tools that are available to be
      used and whose descriptions are to be listed in the prompt. Any tools
      referenced in the `exemplars` should be registered here, although it is
      not strictly required for all of the tools to be illustrated in
      `exemplars`.
    max_steps: Maximum number of iterations.
  """

  # TODO: Make `JinjaTemplate` fully stateless. Currently it is almost
  # stateless, but not quite, as executing JinjaTemplate causes changes to its
  # internal `_context` (an instance of `PromptTemplateContext`). The prompt
  # template context should be treated as part of the "environment", similarly
  # to `tool_handler` and `_sandbox`.
  prompt: PythonPlanningPromptProtocol = dataclasses.field(
      default_factory=PythonPlanningPromptJ2
  )
  exemplars: list[PythonPlanningState] = dataclasses.field(default_factory=list)
  # TODO: Decouple the choice of environment from the agent.
  environment_config: python_tool_use.PythonToolUseEnvironmentConfig = (
      dataclasses.field(
          default_factory=python_tool_use.PythonToolUseEnvironmentConfig
      )
  )
  max_steps: int = 10

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def initialize_state(
      self,
      inputs: str,
      environment: python_tool_use.PythonToolUseEnvironment | None = None,
  ) -> PythonPlanningState:
    """Returns a newly initialized state based on the input question.

    Overridden from base class (Agent).

    Args:
      inputs: Input to the agent, representing the overall goal that the agent
        is trying to achieve.
      environment: Environment in which to perform the operation.
    """
    del environment
    return PythonPlanningState(inputs=inputs)

  @contextlib.asynccontextmanager
  async def start_environment(
      self,
  ) -> AsyncIterator[python_tool_use.PythonToolUseEnvironment]:
    """Context manager to start the environment.

    Usage:
    ```
      agent = ...
      async with agent.start_environment() as env:
         # In here, we can call other methods on `agent` using `env` as the
         # environment.
    ```

    Yields:
      Environment object, which will be automatically cleaned up when exiting
      the `with` block.
    """
    with python_tool_use.PythonToolUseEnvironment(
        config=self.environment_config
    ) as env:
      yield env

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(
      'PythonPlanningAgent._sample_single_next_step', skip=['environment']
  )
  async def _sample_single_next_step(
      self,
      *,
      state: PythonPlanningState,
      environment: python_tool_use.PythonToolUseEnvironment,
  ) -> PythonPlanningStep:
    """Runs one step of the strategy and returns a new resulting state.

    Overridden from base class (SingleSampleAgent).

    Args:
      state: Current state of the agent.
      environment: Environment in which to perform the operation.

    Returns:
      An incremental update to the agent state that would occur as a result of
      performing the given step.
    """
    if environment is None:
      raise ValueError('Environment must be specified for this agent.')

    # Prompt the LLM to determine the next action to take.
    llm_reply = await self.prompt(
        exemplars=self.exemplars,
        state=state,
        tools=environment.config.tools,
    )

    sandbox_state = _sandbox_state_from_agent_state(state)
    result = await environment.run_code(  # pytype: disable=wrong-keyword-args
        sandbox_state=sandbox_state, code=_parse_llm_reply_code(llm_reply)
    )

    irrecoverable_error = False
    if (
        result.sandbox_result.execution_status
        == _ExecutionStatus.EXECUTION_ERROR
        and result.sandbox_result.failure_details is not None
        and result.sandbox_result.failure_details.hook_name is not None
    ):
      # Get the exception type from the failure details.
      failure_details = result.sandbox_result.failure_details
      exception_class = failure_details.exception_class

      # Check if the exception type is listed as an irrecoverable error in the
      # tool config.
      tool_obj = next(
          filter(
              lambda tool: tool.name == failure_details.hook_name,
              environment.config.tools,
          ),
          None,
      )
      if (
          tool_obj
          and tool_obj.irrecoverable_error_types
          and exception_class in tool_obj.irrecoverable_error_types
      ):
        irrecoverable_error = True

    irrecoverable_error = (
        irrecoverable_error
        or result.sandbox_result.execution_status
        in (
            _ExecutionStatus.PROGRAM_ERROR,
            _ExecutionStatus.SANDBOX_ERROR,
            _ExecutionStatus.SANDBOX_TIMEOUT,
        )
    )

    return PythonPlanningStep(
        code=result.code,
        result=_get_result_str(result.sandbox_result, irrecoverable_error),
        execution_status=result.sandbox_result.execution_status,
        is_finished=result.exit_hook_called or irrecoverable_error,
    )

  def is_finished(self, state: PythonPlanningState) -> bool:
    """Returns whether the strategy is in finished state.

    Overridden from base class (Agent).

    Args:
      state: Current state of the agent.
    """
    return bool(state.updates and state.updates[-1].is_finished) or (
        len(state.updates) >= self.max_steps
    )

  def extract_output(self, state: PythonPlanningState) -> str:
    """Returns the final output from the strategy, based on the state.

    Overridden from base class (Agent).

    Args:
      state: Current (presumably final) state of the agent.
    """
    # TODO: Add support for non-string outputs.
    if state.updates:
      answer = state.updates[-1].result
    else:
      answer = None
    return str(answer).strip()
