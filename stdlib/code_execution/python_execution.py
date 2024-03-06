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

"""Core interfaces and shared libraries for Python code execution."""

import abc
import ast
from collections.abc import AsyncIterator, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import enum
import logging
import textwrap
from typing import Any, Callable, Self

from onetwo.core import executing


def adjust_code_to_set_final_expression_value(
    code: str, variable_name: str = 'result', default_value: Any = None,
) -> str:
  r"""Returns adjusted code that store the final expression value in a variable.

  Ignores trailing comment lines when determining the "final expression value".

  If the provided code is not valid Python code, then will return the original
  code unchanged. (Ideally, the caller should perform validation of the code
  prior to calling this function.)

  Examples (assuming variable_name = 'result'):
  * 'x = 2\nx + 3' => 'x = 2\nresult = x + 3'.
  * 'x = 2\ndel x' => 'x = 2\ndel x\nresult = None'.
  * 'x = 2\n# Comment' => 'result = x = 2\n# Comment'.

  Args:
    code: String containing the original unadjusted Python code.
    variable_name: The variable name into which to store the final expression
      value.
    default_value: Default value to assign to `variable_name` in the case where
      the actual final expression value was undefined.
  """
  dedented_code = textwrap.dedent(code.rstrip())
  lines = dedented_code.split('\n')

  # Special case when `code` is empty or consists of only whitespace.
  if len(lines) == 1 and not lines[0].strip():
    lines = []

  try:
    parse_tree = ast.parse(dedented_code)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.warning(
        'Failed to parse Python code:\n```\n%s\n```',
        dedented_code,
        exc_info=True,
    )
    return code

  result_idx = None
  if parse_tree and parse_tree.body:
    # ast.AST.lineno is 1-indexed. We subtract 1 to make it 0-based.
    last_statement = parse_tree.body[-1]
    # The only Python statements that are compatible with variable assignment
    # (as far as we are aware...) are expressions and assignments, e.g.:
    # * Expression: `2 + 3` ==> `result = 2 + 3`
    # * Assignment: `y = x + 2` ==> `result = y = x + 2`
    # Other Python statements that are incompatible with variable assignment
    # include compound statements (`if`, `while`, `for`, etc.) and various
    # simple non-expression / non-assignment statements (e.g., `assert`, `del`,
    # `import`, `raise`, etc.).
    # For background, see: https://docs.python.org/3/reference/index.html
    if isinstance(last_statement, ast.Assign) or isinstance(
        last_statement, ast.Expr
    ):
      result_idx = last_statement.lineno - 1

  if result_idx is None:
    # In this case, since the last statement does not return a value that can be
    # assigned to the result variable, we just set the result equal to None.
    lines.append(f'{variable_name} = {default_value}')
  else:
    # If the last statement does return a value, then we can simply set the
    # result variable equal to that.
    lines[result_idx] = f'{variable_name} = ' + lines[result_idx]

  return '\n'.join(lines)


@enum.unique
class ExecutionStatus(enum.Enum):
  """Code execution status to indicate success or error encountered.

  Attributes:
    SUCCESS: Code ran with no Errors. Note: Code that completes without error
      but has no output is considered successful.
    COMPILE_ERROR: The code could not be compiled (i.e., was not valid Python
      code).
    EXECUTION_ERROR: An exception was thrown when running the code. This could
      be directly from the provided code or from code within a hook. In the
      future, hook errors may be broken out into their own type.
    PROGRAM_ERROR: An error occurred on the client side either while preparing
      the execution request or while processing the results.
    SANDBOX_TIMEOUT: The sandbox timed out.
    SANDBOX_ERROR: An error (such as a security violation) occurred in the
      sandbox.
  """

  SUCCESS = 'SUCCESS'
  COMPILE_ERROR = 'COMPILE_ERROR'
  EXECUTION_ERROR = 'EXECUTION_ERROR'
  # TODO: Do we still need PROGRAM_ERROR, or would all the cases of
  # interest be covered by SANDBOX_ERROR and EXECUTION_ERROR?
  PROGRAM_ERROR = 'PROGRAM_ERROR'
  SANDBOX_TIMEOUT = 'SANDBOX_TIMEOUT'
  SANDBOX_ERROR = 'SANDBOX_ERROR'


@enum.unique
class SandboxStatus(enum.Enum):
  """Status of a Python sandbox after attempting a call to `run`.

  Attributes:
    AFTER_RUNNING_CODE: The code was run, and the value of variables have been
      updated accordingly.
    BEFORE_RUNNING_CODE: The code was not run, and the values of variables have
      not been changed.
    CLEAN: The sandbox is in the same state as if no code had ever been run on
      it. This is the typical status of a stateless sandbox.
    INVALID: The sandbox is unusable or unsafe to use, e.g., due to crashing or
      becoming unresponsive, or because for whatever reason we were not able to
      determine whether the effect of the attempted call to `run` was reflected
      in the value of the variables in the sandbox.
  """
  AFTER_RUNNING_CODE = 'AFTER_RUNNING_CODE'
  BEFORE_RUNNING_CODE = 'BEFORE_RUNNING_CODE'
  CLEAN = 'CLEAN'
  INVALID = 'INVALID'


@dataclasses.dataclass
class SandboxResultTiming:
  """Information on time elapsed while processing a sandbox request.

  Attributes:
    since_start: Total amount of time that elapsed from the time the code
      execution request was sent to the sandbox until the reply was received or
      the request timed out.
    since_last_interaction: Amount of time that elapsed from the last recorded
      interaction with the internal sandbox (i.e., since the last "sign of
      life") until the reply was received or the request timed out. If no
      callback hooks are used, then this should be the same as `total`. In the
      case where a callback to a hook function was made, the time of the "last
      interaction" will be the time of the last such call.
  """
  since_start: datetime.timedelta = datetime.timedelta()
  since_last_interaction: datetime.timedelta = datetime.timedelta()


@dataclasses.dataclass
class SandboxResult:
  r"""The result of executing code in the sandbox.

  Attributes:
    final_expression_value: If the last statement of the executed code was an
      evaluable expression, then this will be the value of that expression,
      otherwise will be `None`. E.g.: for the code `x = 3\nx + 2`, the value of
      the final expression would be 5. Similarly, for `x = 3`, the value would
      be `3`. For programs that end in a complex construct like a for-loop or
      if-statement, or that end in an assertion or print statement, etc.,
      however, the final expression value would be `None`. (To see the output of
      a print statement, see `stdout`.)
    stdout: Content that the code wrote to stdout during execution.
    sandbox_status: Sandbox status after the code execution attempt.
    execution_status: Code execution status (success/failure).
    status_message: A message describing details of non-SUCCESS return statuses.
      For exceptions, this is typically the message of the exception combined
      with the exception type.
    timing: Information on time elapsed while processing the sandbox request.
  """

  final_expression_value: Any = None
  stdout: str = ''
  # TODO: Add `stderr` here if for some of our sandbox implementations
  # we are able to capture the contents written to `stderr`. (So far we have not
  # succeeded in doing this in sandbox2, as `logging.error` leads to a sandbox
  # violation, and `contextlib.redirect_stderr` always seems to result in just
  # an empty string otherwise.)
  sandbox_status: SandboxStatus = SandboxStatus.AFTER_RUNNING_CODE
  execution_status: ExecutionStatus = ExecutionStatus.SUCCESS
  status_message: str = ''
  timing: SandboxResultTiming | None = None

  def __str__(self) -> str:
    parts = []
    if self.stdout:
      parts.append(self.stdout)
    if self.execution_status != ExecutionStatus.SUCCESS:
      parts.append(f'{self.execution_status.value}: {self.status_message}')
    return '\n'.join(parts)


@dataclasses.dataclass
class PythonSandbox(metaclass=abc.ABCMeta):
  """Generic interface for a Python sandbox.

  A Python sandbox that implements this interface can be used in any of the
  following ways:
  (A) Called directly/programmatically from within a prompting strategy.
  (B) Used as one of many tools in a tool use strategy like ReAct.
  (C) Used as the mechanism for orchestrating other tools calls in a
      Python-based tool use strategy like PythonPlanningAgent.
  """

  @abc.abstractmethod
  def is_stateful(self) -> bool:
    """Returns whether variables are carried over from one call to the next."""

  @contextlib.asynccontextmanager
  async def start(self) -> AsyncIterator[Self]:
    """Context manager for starting the sandbox and cleaning up at the end.

    Yields:
      The current sandbox (in a started state).
    """
    try:
      await self.start_unsafe()
      yield self
    except Exception as e:  # pylint: disable=broad-exception-caught
      # We attempt to cleanly stop the sandbox before passing the exception.
      self.stop(e)
    else:
      self.stop()

  @abc.abstractmethod
  async def start_unsafe(self) -> Self:
    """Starts the sandbox (but does not automatically clean it up).

    It is the responsibility of the caller to call `stop` when they are done
    with the sandbox.
    """

  @abc.abstractmethod
  def stop(self, e: Exception | None = None) -> None:
    """Stops the sandbox and any associated threads, queues, etc.

    When using the `start` context manager, the `stop` function will be called
    automatically when exiting the context. When starting the sandbox with
    `start_unsafe`, however, the caller is responsible for calling `stop`
    manually when done using the sandbox.

    Args:
      e: Exception that led to the sandbox being stopped (in the case where the
        sandbox had to be stopped prematurely due to an exception).
    """

  @executing.make_executable
  @abc.abstractmethod
  def run(self, code: str) -> SandboxResult:
    """Returns the result after running the given code in the sandbox.

    Args:
      code: The code to execute.
    """

  @abc.abstractmethod
  async def set_variables(self, **variables: Any) -> None:
    """Sets the variables in the sandbox.

    Args:
      **variables: The variables to set (in the form of `key=value` args).
    """

  @abc.abstractmethod
  async def get_variables(self, *names: str) -> Mapping[str, Any]:
    """Returns the values of the variables in the sandbox.

    Args:
      *names: Names of the variables to fetch.

    Returns:
      A mapping from variable name to its value in the sandbox (or None if that
      variable hasn't been set).

    Raises:
      ValueError if the JSON reply from the sandbox cannot be decoded (e.g. if
      the variables cannot be JSON serialized).
    """

  @abc.abstractmethod
  def get_hook_object(self, key: str) -> Any:
    """Returns the (mutable) hook object associated with the given key.

    If no hook object is associated with that key, then returns `None`.

    Args:
      key: The key (i.e., variable name) under which the given hook object was
        stored in the `hook_objects` mapping that was passed to
        `PythonSandboxFactory.create_sandbox` when the sandbox was created.
    """


class PythonSandboxFactory(metaclass=abc.ABCMeta):
  """Generic interface for a factory that creates Python sandboxes."""

  @abc.abstractmethod
  def create_sandbox(
      self,
      *,
      timeout: datetime.timedelta = datetime.timedelta(seconds=10),
      imports: Sequence[str] | str = tuple(),
      hooks: Mapping[str, Callable[..., Any]] | None = None,
      hook_objects: Mapping[str, Any] | None = None,
      allow_restarts: bool = False,
  ) -> PythonSandbox:
    r"""Returns a newly-created Python sandbox.

    Usage example of `hook_objects`:
    ```
      @dataclasses.dataclass
      class MyObject:
        value: int = 0
        def set_value(self, value: int) -> None:
          self.value = value

      sandbox_factory = ...

      context = {'my_object': MyObject()}
      with sandbox_factory.create_sandbox(
          hook_objects=context,
          hooks={'my_hook': context['my_object'].set_value},
      ).start() as sb:
        sb.run('my_hook(42)')
        print(context['my_object'].value). # This should print '42'.
    ```

    Args:
      timeout: Amount of time after which `PythonSandbox.run` should time out.
      imports: Additional import statements to run at the start of the sandbox
        (on top of any libraries the sandbox imports by default), represented as
        either a list (e.g.,  `['import re', 'from typing import Optional']`) or
        as a block of text (e.g., "import re\nfrom typing import Optional").
        Note that depending on the sandbox implementation, there may be
        limitations regarding what libraries can be imported (see the individual
        sandbox's docstring for details). If an unsupported import statement is
        specified, then `create_sandbox` should ideally detect this proactively
        and raise a ValueError. If the invalid import could not be detected
        proactively (or if the import itself was valid, but usage of certain of
        the imported functions is not allowed), then the issue may alternatively
        manifest itself as a SANDBOX_ERROR when the first problematic code is
        actually run.
      hooks: Mapping from string to functions for the hooks the sandbox can
        call.
      hook_objects: Objects containing modifiable state of the hook functions.
        Takes the form of a mapping of variable name to value. By default, this
        is empty, but users of the sandbox are free to use this as a way to
        bundle together with the sandbox some instances of objects whose life
        cycle they want to have bound together with that of the sandbox, and
        whose contents can be modified by the hook functions. One typical usage
        pattern would be in the case where one of the hooks is a method of an
        object -- in that case, we can store the object itself here. (See usage
        example above.)
      allow_restarts: Set True to return a sandbox that continues accepting
        requests after a restart. Only select this option if you don't expect
        multiple calls to [safe_]run() to depend on each other.
    """
