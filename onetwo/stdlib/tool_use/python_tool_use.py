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

"""Library for executing Python code and calling tools directly or via code."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import functools
import logging
import traceback
from typing import Any, Self, TypeAlias

from onetwo.builtins import tool_use
from onetwo.core import constants
from onetwo.core import tracing
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_safe_subset
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import stateful_caching

# Names of hard-coded hook functions.
HOOK_EXIT = 'exit'

# Aliases for brevity.
_ExecutionStatus = python_execution.ExecutionStatus
_SandboxStatus = python_execution.SandboxStatus

# The state of the PythonSandbox is determined solely by the sequence of code
# snippets that have been executed on it so far. We represent this as a sequence
# of strings.
PythonSandboxState: TypeAlias = tuple[str, ...]


@dataclasses.dataclass
class RunCodeResult:
  """The result of a call to `PythonToolUseEnvironment.run_code`.

  Attributes:
    code: The code that was executed.
    sandbox_result: Result returned by `sandbox.run`.
    exit_hook_called: Whether `ExitHook` was called while executing the current
      code block.
  """
  code: str = ''
  sandbox_result: python_execution.SandboxResult = dataclasses.field(
      default_factory=python_execution.SandboxResult
  )
  exit_hook_called: bool = False


@dataclasses.dataclass
class PythonToolUseEnvironmentConfig:
  """Config controlling the behavior of a PythonToolUseEnvironment.

  This config will often be created initially as a member of an agent or other
  prompting strategy class, and will then be copied over to the environment
  whenever a new environment instance is spawned. Portions of the environment
  config will also be copied when spawning new Python sandbox or tool instances.
  For this reason, it is important that the config is copiable, although not
  necessarily serializable. (E.g., it is allowed to contain things like function
  pointers.) Also, once the config object is fully set up, it should be
  considered immutable, so as to ensure that it is thread-safe (which is a
  requirement for storing it as part of an agent class).

  Attributes:
    sandbox_factory: Factory for creating Python sandboxes.
    tools: Definitions of the tools that are to be made available in the
      environment for calling either directly via `run_tool` or indirectly from
      Python code executed via `run_code`.
    imports: A list of import statements to add to the Python sandbox.
    sandbox_timeout: Amount of time after which `PythonSandbox.run` should time
      out.
    max_retries_on_timeout: Number of times to retry code execution in the case
      of a sandbox timeout. This is mostly useful as a way to address tail
      latency issues that can occur in certain RPC-based tools that may be
      called by the Python sandbox as hooks.
  """
  sandbox_factory: python_execution.PythonSandboxFactory = dataclasses.field(
      default_factory=python_execution_safe_subset.PythonSandboxSafeSubsetFactory
  )
  # TODO: While the `tools` below (which contain both tool name and
  # tool function) are sufficient for managing stateless tools, for supporting
  # stateful tools, we will need to have a tool factory for each tool, rather
  # than just a single instance of each tool function. Or more precisely, in the
  # case where multiple tools may be methods of the same object and thus share
  # a common state, we will need to explicitly group these tool functions by
  # tool object, and have a tool factory for each such object, so that we can
  # construct a new instance of the relevant tool object on demand, depending
  # on the desired starting state.
  tools: Sequence[llm_tool_use.Tool] = dataclasses.field(
      default_factory=list
  )
  imports: list[str] = dataclasses.field(default_factory=list)
  sandbox_timeout: datetime.timedelta = datetime.timedelta(seconds=20)
  max_retries_on_timeout: int = 1


@dataclasses.dataclass
class ExitHook:
  """Hook function to indicate that the agent has reached a solution.

  Attributes:
    called: Indicates whether the hook was called during the current invocation
      of the Python sandbox. This is expected to be manually reset to `False`
      each time before running code in the sandbox.
  """
  called: bool = False

  def __call__(self) -> None:
    self.called = True


PythonSandboxCache: TypeAlias = stateful_caching.StatefulObjectCache[
    python_execution.PythonSandbox, PythonSandboxState
]


# TODO: Extend `caching.CacheEnabled` and configure caching behavior
# in the `Tool`. Make sure that caching works robustly with stateful tools.
@dataclasses.dataclass
class PythonToolUseEnvironment:
  """Stateful environment supporting Python code execution and tool calls.

  The environment is stateful in the sense that when an agent executes an action
  (in this case, in the form of a block of Python code), this can affect the
  values of global variables stored in the Python sandbox, and these variables
  can then be referred to in Python code that is executed as part of subsequent
  actions.

  Unlike an agent state, however, the environment does not need to support
  serialization or deserialization, since it can always be recreated from the
  other state content + agent config (e.g., by initializing a fresh Python
  sandbox and re-executing all of the steps in the agent state up to that point,
  etc.). For the same reason, there is no need to support deep-copying of the
  environment.

  This class is not thread-safe. It is generally safe, however, to share the
  same environment across multiple parallel asyncio branches, as long as they
  are all executed on a single thread. This is true at least with regard to the
  Python sandbox, for which we have an explicit mechanism to maintain multiple
  sandboxes in parallel where necessary; if other tools registered in the tool
  handler are themselves stateful, however, there could be interference between
  the different parallel execution branches, as we currently maintain only one
  copy of each tool per environment.

  Attributes:
    config: Config controlling the behavior of the PythonToolUseEnvironment.
      Typically cloned from an identical config object owned by the agent.
  """
  config: PythonToolUseEnvironmentConfig = dataclasses.field(
      default_factory=PythonToolUseEnvironmentConfig
  )

  # Flag indicating whether we are already inside of a `with env` block.
  _started: bool = dataclasses.field(default=False)

  # The sandboxes in which to execute the code. The cache/environment will
  # control the life cycle of the sandboxes internally and will ensure that the
  # sandboxes have access to the set of tools defined in the environment config.
  # Users of the environment should interact with the sandboxes only indirectly
  # via `PythonToolUseEnvironment.run_code`.
  _sandbox_cache: PythonSandboxCache = dataclasses.field(init=False)

  _stateless_tools: dict[str, llm_tool_use.Tool] = dataclasses.field(
      default_factory=dict
  )
  # TODO: When supporting stateful tools, we will need to group them
  # by tool object, and maintain a StatefulObjectCache for each such object,
  # similarly to how we maintain the PythonSandboxCache above.

  def __post_init__(self):
    """Overridden from dataclasses.dataclass."""
    self._sandbox_cache = PythonSandboxCache(
        create_object_function=self._create_sandbox_for_state,
        destroy_object_function=lambda sandbox: sandbox.stop(),
    )
    for tool in self.config.tools:
      # TODO: Add support for stateful tools.
      self._stateless_tools[tool.name] = tool

  def __enter__(self) -> Self:
    """Starts the environment on context manager enter."""
    if self._started:
      raise ValueError('Environment already started.')
    self._started = True
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Stops and cleans up the environment on context manager exit."""
    self._sandbox_cache.destroy()
    self._started = False

  def is_tool_supported(self, tool_name: str) -> bool:
    """Returns whether the given tool is supported by this environment."""
    # TODO: Add support for stateful tools.
    return tool_name in self._stateless_tools

  def register(self) -> None:
    """Registers the relevant methods in the builtin registry."""
    tool_use.run_tool.configure(self.run_tool)
    # TODO: If we switch to storing an environment state token in a
    # contextvar, we can remove the `sandbox_state` parameter from
    # `self.run_code` and then configure `self.run_code` as a builtin like this:
    # tool_use.run_python.configure(self.run_code)

  def _prepare_hooks(self) -> Mapping[str, Callable[..., Any]]:
    """Prepares hooks for the sandbox, including hooks for calling tools."""

    # Wrapper function to convert directly-specified args into tuple+dict.
    async def _run_tool_wrapper(tool_name: str, *args, **kwargs) -> Any:
      return await self.run_tool(
          tool_name=tool_name, tool_args=args, tool_kwargs=kwargs
      )

    # Special hook to indicate that the Python sandbox has reached a solution.
    hooks = {HOOK_EXIT: ExitHook()}
    # Hooks for calling tools.
    for tool in self.config.tools:
      # Note that although the `tool` itself is already a function, we are
      # having the sandbox invoke the tool via the `run_tool` wrapper rather
      # than calling the tool directly, to ensure that we have a centralized
      # place where we can implement all of the necessary bookkeeping for
      # managing tool states.
      hooks[tool.name] = functools.partial(_run_tool_wrapper, tool.name)
    return hooks

  async def _create_sandbox_for_state(
      self, sandbox_state: PythonSandboxState
  ) -> python_execution.PythonSandbox:
    """Returns a newly created sandbox for the given state.

    Transfers ownership of the sandbox to the caller, who needs to be sure to
    call `stop` on the sandbox when done with it.

    Args:
      sandbox_state: The state to which the sandbox should correspond. (I.e.,
        the sequence of code blocks that should have been executed up to that
        point.)
    """
    # Create the sandbox with the necessary hooks and imports.
    hooks = self._prepare_hooks()
    sandbox = self.config.sandbox_factory.create_sandbox(
        timeout=self.config.sandbox_timeout,
        imports=self.config.imports,
        hooks=hooks,
        hook_objects={HOOK_EXIT: hooks[HOOK_EXIT]},
        allow_restarts=False,
    )
    sandbox = await sandbox.start_unsafe()
    # Reconstruct the desired sandbox state by reexecuting the full code history
    # up till now.
    try:
      for code in sandbox_state:
        new_result = await self._run_code_with_sandbox(
            code=code, sandbox=sandbox
        )
        sandbox_status = new_result.sandbox_result.sandbox_status
        if sandbox_status != _SandboxStatus.AFTER_RUNNING_CODE:
          logging.warning(
              'Re-executing code step failed to update sandbox state: %s.',
              new_result,
          )
    except Exception as e:
      sandbox.stop()
      raise e
    return sandbox

  async def _run_code_with_sandbox(
      self,
      code: str,
      sandbox: python_execution.PythonSandbox,
  ) -> RunCodeResult:
    """Sends code to the sandbox, and postprocesses and returns results.

    Args:
      code: The python code to execute.
      sandbox: The sandbox in which to execute the code.

    Returns:
      Result of executing the code, including effects on hook objects.
    """
    # We reset `hook_objects[HOOK_EXIT].called` at both the beginning and end of
    # each call to `run_code`, since we use it as a signal of whether the
    # sandbox called the `exit()` hook during the current `run_code` invocation.
    sandbox.get_hook_object(HOOK_EXIT).called = False

    try:
      sandbox_result: python_execution.SandboxResult = await sandbox.run(code)
      return RunCodeResult(
          code=code,
          sandbox_result=sandbox_result,
          exit_hook_called=sandbox.get_hook_object(HOOK_EXIT).called,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Catch when an unexpected exception occurred; a retry is unlikely to
      # help in this scenario, so we just report the exception directly and
      # indicate that we're done.
      logging.info('Exception thrown by the sandbox: %s', e)
      return RunCodeResult(
          code=code,
          sandbox_result=python_execution.SandboxResult(
              execution_status=_ExecutionStatus.PROGRAM_ERROR,
              status_message=''.join(traceback.format_exception(e)),
          )
      )
    finally:
      # We reset `hook_objects[HOOK_EXIT].called` at both the beginning and end
      # of each call to `run_code`, since we use it as a signal of whether the
      # sandbox called the `exit()` hook during the current `run_code`
      # invocation.
      sandbox.get_hook_object(HOOK_EXIT).called = False

  async def _run_code_with_retries(
      self, sandbox_state: PythonSandboxState, code: str, max_retries: int
  ) -> RunCodeResult:
    """Recursive helper function for `run_code`. Retries on timeout."""
    sandbox = await self._sandbox_cache.get(sandbox_state)
    result = await self._run_code_with_sandbox(code=code, sandbox=sandbox)

    # Since we are done with the sandbox for now, we can put it back in the
    # cache, associated with the relevant new sandbox state.
    sandbox_status = result.sandbox_result.sandbox_status
    if sandbox_status == _SandboxStatus.BEFORE_RUNNING_CODE:
      # Code was never run. Same sandbox state as before.
      self._sandbox_cache.cache_object(sandbox_state, sandbox)
    elif sandbox_status == _SandboxStatus.AFTER_RUNNING_CODE:
      # Code was run. Update the sandbox state accordingly.
      new_sandbox_state = sandbox_state + (code,)
      self._sandbox_cache.cache_object(new_sandbox_state, sandbox)
    elif sandbox_status == _SandboxStatus.CLEAN:
      # For some reason, we were given back a clean sandbox (e.g., because the
      # sandbox is stateless, or because it restarted in a clean state).
      clean_sandbox_state = tuple()
      self._sandbox_cache.cache_object(clean_sandbox_state, sandbox)
    elif sandbox_status == _SandboxStatus.INVALID:
      # Sandbox crashed or is in an otherwise unusable state. Safest thing to
      # do here is to simply discard the sandbox and start over with a fresh
      # one in which we can reconstruct the intended state in a controlled
      # manner.
      self._sandbox_cache.discard_object(sandbox)
    else:
      raise ValueError(f'Unexpected sandbox status: {sandbox_status}')

    # Automatically retry in case of a timeout.
    execution_status = result.sandbox_result.execution_status
    if execution_status == _ExecutionStatus.SANDBOX_TIMEOUT and max_retries > 0:
      return await self._run_code_with_retries(
          sandbox_state, code, max_retries - 1
      )

    return result

  @tracing.trace(name='run_code')
  async def run_code(
      self, sandbox_state: PythonSandboxState, code: str
  ) -> RunCodeResult:
    """Fetches sandbox, sends code to it, and postprocesses and returns results.

    Should always be called within a `with environment:` block.

    Args:
      sandbox_state: Current sandbox state, containing the sequence of past code
        blocks that are assumed to have been run on the Python sandbox up till
        now. The environment is responsible for either identifying an existing
        Python sandbox whose state is consistent with that history, or else
        reconstructing such a sandbox on the fly by creating a fresh sandbox and
        then re-executing the sequence of past code blocks.
      code: The Python code to execute.

    Returns:
      Result of executing the code, including effects on hook objects.
    """
    if not self._started:
      raise ValueError(
          'Environment not started. Did you forget to wrap your call to'
          ' `run_code()` in a `with env` block?'
      )

    if code.startswith(constants.ERROR_STRING):
      return RunCodeResult(
          code=code,
          sandbox_result=python_execution.SandboxResult(
              execution_status=_ExecutionStatus.PROGRAM_ERROR,
              status_message=(
                  'Code began with error string (maybe due to an issue in'
                  ' parsing the LLM reply?).'
              ),
          ),
      )

    return await self._run_code_with_retries(
        sandbox_state, code, max_retries=self.config.max_retries_on_timeout
    )

  @tracing.trace(name='run_tool')
  async def run_tool(
      self,
      tool_name: str,
      tool_args: tuple[Any, ...],
      tool_kwargs: dict[str, Any],
  ) -> Any:
    """Runs the specified tool and returns the result.

    Can be configured as an implementation of the `run_tool` builtin.

    Args:
      tool_name: The name of the tool to run. Should match the `Tool.name` of
        one of tools managed by this environment.
      tool_args: Position args to pass to the tool function.
      tool_kwargs: Keyword args to pass to the tool function.

    Returns:
      The return value of the tool function.
    """
    if tool_name == constants.ERROR_STRING:
      # ERROR_STRING as the tool_name is a special case, where we are expected
      # to simply echo the error message stored in the tool argument.
      if len(tool_args) != 1:
        raise ValueError(
            'When tool_name is ERROR_STRING, we expect there to be exactly one'
            ' argument containing the detailed error message (e.g., an error'
            ' that occurred when parsing the LLM response to determine the'
            f' tool call). Instead found {len(tool_args)} arguments:'
            f' {tool_name=},  {tool_args=},  {tool_kwargs=}'
        )
      return tool_args[0]

    try:
      if not self.is_tool_supported(tool_name):
        raise ValueError(
            f'Function {tool_name} is not registered in the environment'
            f' (tools={self.config.tools}).'
        )

      # TODO: When using stateful tools, this is where we would need
      # to fetch the appropriate instance of the tool based on the current world
      # state. After the tool call, we would then need to update both the world
      # state and the state of the given tool instance.

      tool = self._stateless_tools[tool_name]
      return await tool(*tool_args, **tool_kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f'{constants.ERROR_STRING}: {e}'
