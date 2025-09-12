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

"""Core interfaces for Python code execution with async communication."""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import datetime
import json
import logging
from multiprocessing import connection
import queue
from typing import Any, Callable, Mapping, TypeAlias

from onetwo.core import executing
from onetwo.core import utils
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_utils

# Aliases
_ExecutionStatus: TypeAlias = python_execution.ExecutionStatus
_SandboxStatus: TypeAlias = python_execution.SandboxStatus
_SandboxResult: TypeAlias = python_execution.SandboxResult

# Constants for hook communication
DATA_KEY: str = python_execution_utils.DATA_KEY
EXCEPTION_KEY: str = python_execution_utils.EXCEPTION_KEY
EX_ARGS_KEY: str = python_execution_utils.EX_ARGS_KEY
EX_CLASS_KEY: str = python_execution_utils.EX_CLASS_KEY
EX_HOOK_NAME_KEY: str = python_execution_utils.EX_HOOK_NAME_KEY

current_timing = python_execution_utils.current_timing


async def handle_hook_message(
    hooks: Mapping[str, Callable[..., Any]],
    message: str,
    sandbox_connection: Any,
    start_time: datetime.datetime,
) -> _SandboxResult | None:
  """Processes a single hook message from the sandbox.

  This function parses the message, executes the corresponding hook, and sends
  the result (or exception information) back to the sandbox.

  Args:
    hooks: A mapping of hook names to callable functions.
    message: The JSON string message received from the sandbox.
    sandbox_connection: The connection object used to send replies back to the
      sandbox.
    start_time: The timestamp when the current 'run' operation started.

  Returns:
    A _SandboxResult indicating an error if communication with the sandbox
    fails, otherwise None.
  """
  reply = {}
  # We got a message, we reset the base time for timeout.
  base_time = datetime.datetime.now()
  message = json.loads(message)
  logging.info('Sandbox hook call: %s', str(message))
  try:
    res = await utils.call_and_maybe_await(
        hooks[message['hook']],
        *message['args'],
        **message['kwargs'],
    )
    reply[DATA_KEY] = res
    logging.info('Sandbox hook result: %s', str(res))

  except Exception as e:  # pylint: disable=broad-exception-caught
    # Exceptions happening in hook execution are passed back to the
    # sandbox where the hook wrapper will throw them, allowing them
    # to be handled like exceptions originating in the sandbox.
    logging.info(
        'Sandbox hook raised exception: %s',
        f'{e.__class__.__name__} - {e.args}',
        exc_info=True,
    )
    reply[EXCEPTION_KEY] = {
        python_execution.EX_CLASS_KEY: e.__class__.__name__,
        EX_ARGS_KEY: e.args,
        python_execution.EX_HOOK_NAME_KEY: message['hook'],
    }
    try:
      _ = json.dumps(reply)
    except TypeError:
      reply[EX_ARGS_KEY] = (str(e.args),)

  try:
    sandbox_connection.send(json.dumps(reply))
  except BrokenPipeError:
    logging.info('Sandbox connection closed while sending result.')
    return _SandboxResult(
        stdout='',
        sandbox_status=_SandboxStatus.INVALID,
        execution_status=_ExecutionStatus.SANDBOX_ERROR,
        status_message='Code execution dropped (BrokenPipeError).',
        timing=current_timing(start=start_time, base=base_time),
    )


@dataclasses.dataclass
class BaseMultiProcessSandbox(python_execution.PythonSandbox, abc.ABC):
  """Base class for sandboxes using a separate process.

  This class provides a framework for sandboxes that run code in a separate
  process, communicating using queues for results and a connection object
  (e.g., a pipe) for bidirectional hook callbacks.

  This class implements the `run` method from `python_execution.PythonSandbox`,
  providing the core loop for managing asynchronous communication,
  including sending code, polling for results, handling hook messages, and
  managing timeouts.

  Attributes:
    timeout: The maximum time to wait for a single code execution to complete.
    hooks: A mapping of function names to callable functions that can be invoked
      from within the sandbox.
  """

  timeout: datetime.timedelta = datetime.timedelta(seconds=10)
  hooks: Mapping[str, Callable[..., Any]] = dataclasses.field(
      default_factory=dict
  )

  @abc.abstractmethod
  def _send_request(self, code: str) -> None:
    """Sends an execution request to the sandbox worker.

    Args:
      code: The Python code string to be executed.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _get_result_queue(self) -> Any:
    """Returns the queue used to receive results from the sandbox worker.

    Returns:
      The result queue object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _parse_result(
      self,
      result_payload: Any,
      start_time: datetime.datetime,
      base_time: datetime.datetime,
  ) -> _SandboxResult:
    """Parses the raw result from the queue into a SandboxResult object.

    Args:
      result_payload: The data received from the result queue.
      start_time: The time the execution request was sent.
      base_time: The time of the last interaction with the sandbox.

    Returns:
      A _SandboxResult object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _is_sandbox_alive(self) -> bool:
    """Checks if the sandbox worker process/thread is still running.

    Returns:
      True if the sandbox is alive, False otherwise.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _get_sandbox_connection(self) -> connection.Connection | None:
    """Returns the connection object for hook communications.

    This connection is used for bidirectional messaging between the main
    process and the sandboxed process, primarily for handling hook calls. The
    object should implement the `multiprocessing.connection.Connection`
    interface.

    The following methods are expected to be available:
    - `send(obj)`: To send JSON-serializable objects.
    - `recv()`: To receive objects.
    - `poll([timeout])`: To check for available data.

    Returns:
      A `multiprocessing.connection.Connection` object or None if hooks are
      not enabled or the connection is not established.
    """
    raise NotImplementedError

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def run(self, code: str) -> _SandboxResult:
    """Runs the code in the sandbox, handling timeouts and hook messages.

    This loop sends the code to the sandbox, waits for results, and processes
    any hook-related messages in the meantime.

    Args:
      code: The Python code string to execute.

    Returns:
      The _SandboxResult from the execution.
    """
    # Precheck code syntax
    try:
      compile(code, '<string>', 'exec')
    except SyntaxError as e:
      logging.info('Fast fail on bad syntax: %s \nCode: %s', e, code)
      return _SandboxResult(
          stdout='',
          sandbox_status=_SandboxStatus.BEFORE_RUNNING_CODE,
          execution_status=_ExecutionStatus.COMPILE_ERROR,
          status_message=f'{e.__class__.__name__} - {str(e)}',
      )

    logging.info('Sending code to sandbox: %s', code)
    self._send_request(code)
    start_time = datetime.datetime.now()
    base_time = start_time  # Time of last interaction with the sandbox.
    sandbox_connection = self._get_sandbox_connection()

    while True:
      await asyncio.sleep(0)
      try:
        result_queue = self._get_result_queue()
        result_str = result_queue.get(block=False)
        result_queue.task_done()
        result_pkg = self._parse_result(result_str, start_time, base_time)
        return result_pkg

      except queue.Empty:
        timeout_plus_1_second = self.timeout + datetime.timedelta(seconds=1)
        if datetime.datetime.now() - base_time > timeout_plus_1_second:
          # The sandbox may have crashed.
          logging.info('Timed out waiting for result.')
          return _SandboxResult(
              stdout='',
              sandbox_status=_SandboxStatus.INVALID,
              execution_status=_ExecutionStatus.SANDBOX_TIMEOUT,
              status_message='Code execution timed out (never received reply).',
              timing=current_timing(start=start_time, base=base_time),
          )
        if self.hooks and sandbox_connection:
          if sandbox_connection.poll():
            try:
              message = sandbox_connection.recv()
            except EOFError:
              logging.info('Sandbox connection closed while receiving.')
              return _SandboxResult(
                  stdout='',
                  sandbox_status=_SandboxStatus.INVALID,
                  execution_status=_ExecutionStatus.SANDBOX_ERROR,
                  status_message='Code execution dropped (EOFError).',
                  timing=current_timing(start=start_time, base=base_time),
              )
            error_result = await handle_hook_message(
                self.hooks, message, sandbox_connection, start_time
            )
            if error_result:
              return error_result
