# Copyright 2025 DeepMind Technologies Limited.
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

"""Core interfaces for Python code execution.

For library code prefer `python_execution_utils` instead.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import enum
from typing import Any, Callable, Final

from onetwo.core import executing

# Helper keys for building the exception failure details in the sandbox.
_EX_FAILURE_DETAILS_KEY: Final[str] = 'failure_details'
EX_HOOK_NAME_KEY: Final[str] = 'hook_name'
EX_CLASS_KEY: Final[str] = 'exception_class'
EX_MESSAGE_KEY: Final[str] = 'exception_message'


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
class FailureDetails:
  r"""Details of a failure that occurred while executing a sandbox code.

  Attributes:
    hook_name: The name of the hook that resulted in the failure if any.
    exception_class: The type of the exception that was raised.
    exception_message: The message of the exception that was raised.
  """

  hook_name: str | None = None
  exception_class: str = ''
  exception_message: str = ''


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
    failure_details: Details of a failure that occurred while executing a
      sandbox code.
    timing: Information on time elapsed while processing the sandbox request.
  """

  final_expression_value: Any = None
  stdout: str = ''
  sandbox_status: SandboxStatus = SandboxStatus.AFTER_RUNNING_CODE
  execution_status: ExecutionStatus = ExecutionStatus.SUCCESS
  status_message: str = ''
  failure_details: FailureDetails | None = None
  timing: SandboxResultTiming | None = None

  def __str__(self) -> str:
    """Returns a string representation of the result's most salient content."""
    parts = []
    if self.stdout:
      parts.append(self.stdout)
    if self.final_expression_value is not None:
      # If `final_expression_value` is non-None, then it must have been
      # explicitly populated by the sandbox execution. To make sure we don't
      # hide any relevant information, we thus include this value as part of
      # result string, even if the executed Python code had already printed
      # some other information earlier to stdout. This behavior is consistent
      # with colab, which also outputs the value of the final expression of any
      # given code block, regardless of whether additional content had been
      # explicitly printed to stdout earlier in the code block.
      parts.append(str(self.final_expression_value))
    elif not self.stdout and self.execution_status == ExecutionStatus.SUCCESS:
      # If `final_expression_value` is None, we currently don't have a robust
      # way of distinguishing whether the last line of code actually evaluated
      # to `None` vs. whether the last line had an undefined value (as is the
      # case, for example, with a `print` statement) or was never even executed
      # due to an error. As an approximation, we thus include the value `None`
      # in the result string only in the case where the status is `SUCCESS`
      # (which indicates that the last line really was executed) and where
      # `self.stdout` is empty (which rules out the case where the last line was
      # a print statement, or more generally, where the code block was designed
      # purely around print statements). Note that this heuristic is sufficient
      # to ensure that we always return a non-empty result string, including in
      # the case where the Python sandbox is used in the style of a calculator,
      # where the entire code block is designed to evaluate a single expression.
      parts.append(str(self.final_expression_value))
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
  async def start(self) -> AsyncIterator[PythonSandbox]:
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

  async def start_unsafe(self) -> PythonSandbox:
    """Starts the sandbox (but does not automatically clean it up).

    It is the responsibility of the caller to call `stop` when they are done
    with the sandbox.

    Returns:
      The current sandbox (in a started state).
    """
    return self

  def stop(self, e: Exception | None = None) -> None:
    """Stops the sandbox and any associated threads, queues, etc.

    Override this method if your implementation needs to do any cleanup.

    When using the `start` context manager, the `stop` function will be called
    automatically when exiting the context. When starting the sandbox with
    `start_unsafe`, however, the caller is responsible for calling `stop`
    manually when done using the sandbox.

    Args:
      e: Exception that led to the sandbox being stopped (in the case where the
        sandbox had to be stopped prematurely due to an exception).
    """

  @executing.make_executable  # pytype: disable=wrong-arg-types
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
        print(context['my_object'].value)  # This should print '42'.
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


def parse_failure_details(
    result: dict[str, Any],
) -> FailureDetails | None:
  """Parses the failure details from the sandbox result, if present.

  Args:
    result: The result returned by the sandbox when there is a
      failure/exception.

  Returns:
    The parsed failure details, or None if not present.
  """
  failure_details = result.get(_EX_FAILURE_DETAILS_KEY)
  if failure_details is None:
    return None
  return FailureDetails(
      hook_name=failure_details[EX_HOOK_NAME_KEY],
      exception_class=failure_details[EX_CLASS_KEY],
      exception_message=failure_details[EX_MESSAGE_KEY],
  )
