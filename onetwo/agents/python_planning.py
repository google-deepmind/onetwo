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

Inspired by the approach taken in (internal link).

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
  composition of Python Planning with other agent strategies such as
  tree-of-thought, for pursuing multiple possible trajectories in parallel.
"""

import abc
from collections.abc import AsyncIterator, Sequence
import contextlib
import dataclasses
from typing import Protocol, TypeAlias

from onetwo.agents import base as agents_base
from onetwo.builtins import prompt_templating
from onetwo.core import executing
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


def _get_result_str(result: python_execution.SandboxResult) -> str:
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
  if result.execution_status == _ExecutionStatus.EXECUTION_ERROR:
    parts.append(
        'The python code above raises an exception:\n\n'
        f'{result.status_message}\n\n'
        'Rewrite the python code to eliminate the exception.\n'
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


# TODO: Add more exemplars to make the prompt more robust.
DEFAULT_PYTHON_PLANNING_PROMPT_TEXT = """\
{#- Preamble: Tools description -#}
{%- role name='system' -%}
Here is a list of available tools:
{% for tool in tools %}
Tool name: {{ tool.name }}
Tool description: {{ tool.description }}
{% if tool.example -%}
  Tool example: {{ tool.example_str }}
{%- endif -%}
{%- endfor -%}

{#- Preamble: Few-shot exemplars -#}
{{ '\n' }}
Here is an example of how these tools can be used to solve a task:
{% for exemplar in exemplars %}
**Question**: {{ exemplar.inputs + '\n' }}
{%- for step in exemplar.updates %}
```
{{ step.code }}
```
{{ step.result + '\n' }}
{%- if step.is_finished -%}
**Done**
{% endif -%}
{%- endfor -%}
{%- endfor -%}

{{ '\n\n' }}
Write code in Python using the above tools to solve the following question:
{% endrole -%}

{#- Start of the processing of the actual inputs. -#}

{#- Render the original question. -#}
**Question**: {{ state.inputs + '\n' }}

{#- Render the current state (i.e., any steps performed up till now). -#}
{%- for step in state.updates -%}
```
{{ step.code }}
```
{{ step.result + '\n' }}
{%- if step.is_finished -%}
**Done**
{% endif -%}
{%- endfor -%}

{#- Get a response from the LLM for the next step and return it. -#}
```
{{ store('llm_reply', generate_text(stop=['```']) | trim) }}
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

  @executing.make_executable
  @abc.abstractmethod
  async def __call__(
      self,
      exemplars: list[PythonPlanningState],
      state: PythonPlanningState,
      tools: Sequence[llm_tool_use.ToolSpec],
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
  """JinjaTemplate usable with PythonPlanningAgent.prompt."""

  # Overriding default value of attribute defined in templating.JinjaTemplate.
  text: str = DEFAULT_PYTHON_PLANNING_PROMPT_TEXT

  @executing.make_executable
  async def __call__(
      self,
      exemplars: list[PythonPlanningState],
      state: PythonPlanningState,
      tools: Sequence[llm_tool_use.ToolSpec],
  ) -> str:
    """See PythonPlanningPromptProtocol."""
    result = await self.render(
        exemplars=exemplars,
        state=state,
        tools=tools,
    )
    return result['llm_reply']


# TODO: Does the output type have to be `str` or could it be `Any`?
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

  @executing.make_executable(copy_self=False)
  async def initialize_state(self, inputs: str) -> PythonPlanningState:
    """Returns a newly initialized state based on the input question.

    Overridden from base class (Agent).

    Args:
      inputs: Input to the agent, representing the overall goal that the agent
        is trying to achieve.
    """
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
    result = await environment.run_code(
        sandbox_state=sandbox_state, code=llm_reply.strip()
    )

    irrecoverable_error = result.sandbox_result.execution_status in (
        _ExecutionStatus.PROGRAM_ERROR,
        _ExecutionStatus.SANDBOX_ERROR,
        _ExecutionStatus.SANDBOX_TIMEOUT,
    )

    return PythonPlanningStep(
        code=result.code,
        result=_get_result_str(result.sandbox_result),
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
    # TODO: Consider how to support non-string outputs.
    if state.updates:
      answer = state.updates[-1].result
    else:
      answer = None
    return str(answer).strip()
