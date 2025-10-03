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
import multiprocessing
from multiprocessing import connection
import queue
import threading
from typing import Any, Callable, Final, Mapping, TypeAlias

from onetwo.core import executing
from onetwo.core import utils
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_utils

import multiprocessing as mp_context

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

# Authentication for the connection between the sandbox and the listener in the
# main thread.
_AUTHKEY: Final[bytes] = b'pwd'

current_timing = python_execution_utils.current_timing


def _parse_result_str(
    result_str: str,
    start_time: datetime.datetime,
    base_time: datetime.datetime,
) -> _SandboxResult:
  """Parses a JSON string from the sandbox process into a SandboxResult.

  Args:
    result_str: The string returned by the sandbox.
    start_time: The time at which the sandbox started executing the code.
    base_time: The time at which the sandbox most recently received a message
      from the main thread (either a code request or a reply to a previous
      message).

  Returns:
    The parsed SandboxResult.
  """
  timing = python_execution_utils.current_timing(
      start=start_time, base=base_time
  )

  sandbox_result = python_execution_utils.parse_sandbox_result_json(
      result_str, start_time, base_time
  )

  if sandbox_result:
    return sandbox_result

  # Received an arbitrary string (not a full SandboxResult). Means something
  # must have gone wrong, and we didn't exit the sandbox via the route we
  # expected.
  return _SandboxResult(
      stdout=result_str,
      sandbox_status=_SandboxStatus.INVALID,
      execution_status=_ExecutionStatus.SANDBOX_ERROR,
      status_message=(
          'Sandbox returned an arbitrary string rather than a full'
          ' SandboxResult. This is unexpected.'
      ),
      timing=timing,
  )


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


def _sandbox_process_target(
    requests_queue: multiprocessing.JoinableQueue,
    results_queue: multiprocessing.JoinableQueue,
    listener_port: int | None,
    hook_names: list[str],
    timeout_seconds: float,
    inner_sandbox_factory: python_execution.PythonSandboxFactory,
):
  """Target function run in a separate process to execute code in a sandbox.

  This function establishes the sandboxed environment. It continuously waits
  for code execution requests from the `requests_queue`. Each request
  contains the code to be executed and the current context.

  The code is executed using the `run` method of an inner sandbox instance
  created by the `inner_sandbox_factory`. Standard output is captured by the
  inner sandbox.

  If `listener_port` and `hook_names` are provided, it connects back to the
  main process to enable hook function calls from within the sandboxed code.
  These hooks allow the sandboxed code to interact with the main process
  in a controlled manner.

  Execution results, including the final expression value, captured stdout,
  status, and any exceptions, are packaged and sent back to the main process
  via the `results_queue`. The updated execution context is also returned.

  The process loop terminates when a '##done##' sentinel is received on the
  `requests_queue`.

  Args:
    requests_queue: A queue used to receive execution requests from the main
      process. Each request is expected to be a dict containing 'code' and
      'context' (the variables).
    results_queue: A queue used to send results back to the main process. Each
      result is a dict containing 'result_json' (a JSON string of the
      SandboxResult) and 'context' (the updated variables).
    listener_port: The port number on 'localhost' where the main process is
      listening for hook connections. If None or if hook_names is empty, hook
      support is disabled.
    hook_names: A list of function names that the sandboxed code is allowed to
      call as hooks. These functions are executed in the main process.
    timeout_seconds: The maximum time in seconds allowed for a single code
      execution call within the inner sandbox.
    inner_sandbox_factory: A factory instance to create the inner sandbox used
      for code execution.
  """
  conn = None
  if listener_port and hook_names:
    try:
      address = ('localhost', listener_port)
      conn = connection.Client(address, authkey=_AUTHKEY)

    except Exception as e:
      raise ValueError('Cannot connect to the main code, aborting') from e

  hook_callables = {}
  if conn:
    for hook_name in hook_names:
      hook_callables[hook_name] = (
          python_execution_utils.create_sandbox_hook_callable(hook_name, conn)
      )

  # Instantiate the inner sandbox
  inner_sandbox = inner_sandbox_factory.create_sandbox(hooks=hook_callables)
  if not isinstance(inner_sandbox, python_execution.PythonSandbox):
    raise TypeError(
        'The provided inner_sandbox_factory did not create an instance of'
        ' PythonSandbox.'
    )

  while True:
    request = requests_queue.get()
    # TODO: We should delay this call to `Queue.task_done` until
    # after we actually finish processing the request, so as to ensure the
    # thread doesn't get shut down before we have a chance to calculate the
    # result and put it in the results_queue. One way to do this could be to
    # move the below if-statement into the `try` block and then move the
    # call to `requests_queue.task_done()` into a `finally` block.
    requests_queue.task_done()
    if request == '##done##':
      logging.info('Sandbox process received ##done##.')
      break

    code = request.get('code')
    context = request.get('context', {})

    if code is None:
      continue

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def execute_in_inner_sandbox():
      await inner_sandbox.set_variables(**context)
      sandbox_result = await inner_sandbox.run(code)
      # Retrieve all variables from the inner sandbox's context,
      if hasattr(inner_sandbox, 'context'):
        updated_context = await inner_sandbox.get_variables(
            *inner_sandbox.context.keys()
        )
      else:
        # Fallback: Only retrieve variables originally in the context.
        # New variables created in the sandbox won't be reflected.
        logging.warning(
            'Inner sandbox type %s does not have a .context attribute. '
            'Only fetching initially set variables.',
            type(inner_sandbox),
        )
        updated_context = await inner_sandbox.get_variables(*context.keys())
      return sandbox_result, updated_context

    try:
      sandbox_result, updated_context = loop.run_until_complete(
          asyncio.wait_for(execute_in_inner_sandbox(), timeout=timeout_seconds)
      )

      failure_details = sandbox_result.failure_details
      if failure_details:
        failure_details_dict = {
            python_execution.EX_HOOK_NAME_KEY: failure_details.hook_name,
            python_execution.EX_CLASS_KEY: failure_details.exception_class,
            python_execution.EX_MESSAGE_KEY: failure_details.exception_message,
        }
      else:
        failure_details_dict = None
      sandbox_result_dict = {
          'final_expression_value': sandbox_result.final_expression_value,
          'stdout': sandbox_result.stdout,
          'execution_status': sandbox_result.execution_status.value,
          'status_message': sandbox_result.status_message,
          'failure_details': failure_details_dict,
      }
    except asyncio.TimeoutError:
      sandbox_result_dict = {
          'stdout': '',  # Stdout is lost on timeout with inner_sandbox.run
          'execution_status': _ExecutionStatus.SANDBOX_TIMEOUT.value,
          'status_message': (
              f'Sandbox run timed out after {timeout_seconds} seconds.'
          ),
      }
      updated_context = context
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Catch errors during set_variables, get_variables, or unexpected
      # inner_sandbox issues
      sandbox_result_dict = {
          'stdout': '',
          'execution_status': _ExecutionStatus.SANDBOX_ERROR.value,
          'status_message': (
              f'Error in sandbox process: {e.__class__.__name__} - {str(e)}'
          ),
      }
      updated_context = context
    finally:
      if loop and not loop.is_closed():
        loop.close()

    result_json = json.dumps(sandbox_result_dict)
    results_queue.put({
        'result_json': result_json,
        'context': updated_context,
    })

  if conn:
    conn.close()


@dataclasses.dataclass
class PythonSandboxMultiProcessWrapper(BaseMultiProcessSandbox):
  """A generic wrapper for running a PythonSandbox in a separate process.

  This sandbox executes code in a separate, persistent process, leveraging the
  asynchronous communication and hook management features provided by
  `BaseMultiProcessSandbox`. The code execution within the child process
  is performed by an 'inner sandbox' instance, which is created by the
  provided `inner_sandbox_factory`. This class is stateful,
  meaning variable assignments persist across `run` calls.

  Key Features:
    - Process Isolation: Code runs in a child process managed by Python's
      `multiprocessing` module (using 'spawn' context where possible). This
      protects the main process from crashes or hangs in the sandboxed code.
    - Stateful Execution: Variables and their values persist within the
      sandbox process across multiple calls to `run()`. To populate or modify
      the sandbox's context, use the `set_variables()` method *after* the
      sandbox has been started using `.start()`. The `get_variables()` method
      can be used to retrieve variable values from the sandbox.
    - Hook Support: Inherits hook mechanism from `BaseMultiProcessSandbox`.
    - Pluggable Inner Sandbox: The actual execution sandbox within the
      subprocess is configurable via the `inner_sandbox_factory`.

  Attributes:
    timeout: Maximum duration for a single `run()` call before it times out.
      This includes any time spent waiting for hook calls to complete.
    hooks: Mapping from string to functions for the hooks the sandbox can call.
      These functions are executed in the main process, allowing the sandboxed
      code to trigger actions outside the sandbox in a controlled manner.
    hook_objects: Objects containing modifiable state of the hook functions.
      This allows hooks to interact with stateful objects in the main process.
      See `PythonSandboxFactory` for more details.
    inner_sandbox_factory: A factory instance responsible for creating the inner
      sandbox within the child process. This must be provided.
  """

  timeout: datetime.timedelta = dataclasses.field(
      default=datetime.timedelta(seconds=10)
  )
  hooks: Mapping[str, Callable[..., Any]] = dataclasses.field(
      default_factory=dict
  )
  hook_objects: dict[str, Any] = dataclasses.field(default_factory=dict)
  inner_sandbox_factory: python_execution.PythonSandboxFactory | None = (
      dataclasses.field(default=None)
  )

  _requests_queue: multiprocessing.JoinableQueue | None = dataclasses.field(
      init=False, default=None
  )
  _results_queue: multiprocessing.JoinableQueue | None = dataclasses.field(
      init=False, default=None
  )
  _sandbox_process: multiprocessing.Process | None = dataclasses.field(
      init=False, default=None
  )
  _listener: connection.Listener | None = dataclasses.field(
      init=False, default=None
  )
  _sandbox_connection: Any = dataclasses.field(init=False, default=None)
  _port: int = dataclasses.field(init=False, default=8080)
  _context: dict[str, Any] = dataclasses.field(default_factory=dict)
  _mp_context: Any = dataclasses.field(init=False, default=None)

  def __post_init__(self):
    """Initializes the multiprocessing context."""
    try:
      self._mp_context = mp_context.get_context('spawn')
    except ValueError:
      logging.warning(
          "multiprocessing 'spawn' context not available, falling back to"
          ' default.'
      )
      self._mp_context = mp_context.get_context()

  def is_stateful(self) -> bool:
    """Returns whether variables are carried over from one call to the next.

    Overridden from base class (PythonSandbox).
    """
    return True

  async def start_unsafe(self) -> PythonSandboxMultiProcessWrapper:
    if self._sandbox_process is not None:
      raise ValueError('Sandbox already running.')

    self._requests_queue = self._mp_context.JoinableQueue()
    self._results_queue = self._mp_context.JoinableQueue()
    self._context = {}

    # We start a connection to which the sandbox can send requests to run
    # tools.
    listener_thread = None
    if self.hooks:
      address = ('localhost', 0)
      listener = connection.Listener(address=address, authkey=_AUTHKEY)
      self._listener = listener
      self._port = int(self._listener.address[1])
      logging.info('Sandbox listener setup on port %s.', str(self._port))

      def start_connection():
        nonlocal listener
        self._sandbox_connection = listener.accept()
        logging.info('Sandbox listener accepting connections.')

      listener_thread = threading.Thread(target=start_connection)
      listener_thread.start()

    self._sandbox_process = self._mp_context.Process(
        target=_sandbox_process_target,
        args=(
            self._requests_queue,
            self._results_queue,
            self._port,
            list(self.hooks.keys()),
            self.timeout.total_seconds(),
            self.inner_sandbox_factory,
        ),
    )
    self._sandbox_process.start()
    logging.info('Sandbox process initiated.')

    if self.hooks:
      if listener_thread:
        listener_thread.join(timeout=30.0)  # Join the thread here
        if listener_thread.is_alive():
          self.stop()
          raise RuntimeError(
              'Timeout waiting for sandbox to connect back for hooks (listener'
              ' thread still alive).'
          )

      if not self._sandbox_connection:
        self.stop()
        raise RuntimeError(
            'Failed to establish hook connection from sandbox. Check sandbox'
            ' logs.'
        )
      logging.info('Sandbox hook connection established.')
    else:
      logging.info('Sandbox started without hooks.')

    return self

  def stop(self, e: Exception | None = None) -> None:
    logging.info('Sandbox closing.')
    if self._requests_queue is not None:
      self._requests_queue.put('##done##')
      self._requests_queue.join()
      self._requests_queue = None
    logging.info('Request queue closed')
    if self._results_queue is not None:
      # TODO: call join after clearing the queue
      # self.results_queue.join()
      self._results_queue = None
    logging.info('Results queue closed')
    if self._sandbox_process:
      self._sandbox_process.join()
      self._sandbox_process = None
    logging.info('Sandbox Process stopped.')
    if self._listener:
      self._listener.close()
      self._listener = None
    logging.info('Sandbox listener closed.')
    if self._sandbox_connection:
      self._sandbox_connection.close()
      self._sandbox_connection = None
    logging.info('Sandbox process stopped.')
    if e is not None:
      logging.info('Exception raised: %s', e)
      raise e

  def _send_request(self, code: str) -> None:
    """Sends an execution request to the sandbox worker.

    Overridden from base class (BaseAsyncSandbox).

    Args:
      code: The Python code string to be executed.
    """
    assert self._requests_queue is not None
    request = {'code': code, 'context': self._context.copy()}
    self._requests_queue.put(request)

  def _get_result_queue(self) -> Any:
    """Returns the queue used to receive results from the sandbox worker.

    Overridden from base class (BaseAsyncSandbox).

    Returns:
      The result queue object.
    """
    assert self._results_queue is not None
    return self._results_queue

  def _parse_result(
      self,
      result_payload: Any,
      start_time: datetime.datetime,
      base_time: datetime.datetime,
  ) -> _SandboxResult:
    """Parses the raw result from the queue into a SandboxResult object.

    Overridden from base class (BaseAsyncSandbox).

    Args:
      result_payload: The data received from the result queue.
      start_time: The time the execution request was sent.
      base_time: The time of the last interaction with the sandbox.

    Returns:
      A _SandboxResult object.
    """
    result_str = result_payload.get('result_json')
    self._context = result_payload.get('context', self._context)

    if result_str is None:
      return _SandboxResult(
          sandbox_status=_SandboxStatus.INVALID,
          execution_status=_ExecutionStatus.SANDBOX_ERROR,
          status_message=(
              "Internal Error: 'result_json' key not found in sandbox response."
          ),
          timing=python_execution_utils.current_timing(
              start=start_time, base=base_time
          ),
      )

    return _parse_result_str(result_str, start_time, base_time)

  def _is_sandbox_alive(self) -> bool:
    """Checks if the sandbox worker process/thread is still running.

    Overridden from base class (BaseAsyncSandbox).

    Returns:
      True if the sandbox is alive, False otherwise.
    """
    return (
        self._sandbox_process is not None and self._sandbox_process.is_alive()
    )

  def _get_sandbox_connection(self) -> Any | None:
    """Returns the connection object for hook communications.

    This connection is used for sending/receiving data related to hook calls
    between the main process and the sandbox.

    Overridden from base class (BaseAsyncSandbox).

    Returns:
      The connection object, or None if hooks are not supported or a connection
      is not established.
    """
    return self._sandbox_connection

  async def set_variables(self, **variables: Any) -> None:
    """Sets the variables in the sandbox.

    Overridden from base class (PythonSandbox).

    Args:
      **variables: The variables to set (in the form of `key=value` args).
    """
    self._context.update(variables)

  async def get_variables(self, *names: str) -> Mapping[str, Any]:
    """Returns the values of the variables in the sandbox.

    Overridden from base class (PythonSandbox).

    Args:
      *names: Names of the variables to fetch.

    Returns:
      A mapping from variable name to its value in the sandbox (or None if that
      variable hasn't been set).
    """
    return {name: self._context.get(name) for name in names}

  def get_hook_object(self, key: str) -> Any:
    """Returns the (mutable) hook object associated with the given key.

    If no hook object is associated with that key, then returns `None`.

    Overridden from base class (PythonSandbox).

    Args:
      key: The key (i.e., variable name) under which the given hook object was
        stored in the `hook_objects` mapping that was passed to
        `PythonSandboxFactory.create_sandbox` when the sandbox was created.
    """
    return self.hook_objects.get(key)
