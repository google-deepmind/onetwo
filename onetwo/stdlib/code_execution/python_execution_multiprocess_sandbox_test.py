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

import asyncio
import datetime
import json
from multiprocessing import connection
import queue
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_multiprocess_sandbox

# Aliases
_ExecutionStatus = python_execution.ExecutionStatus
_SandboxStatus = python_execution.SandboxStatus
_SandboxResult = python_execution.SandboxResult


class HandleHookMessageTest(parameterized.TestCase):
  """Tests for the handle_hook_message function."""

  def setUp(self):
    """Sets up the test case."""
    super().setUp()
    self.mock_connection = mock.MagicMock()
    self.start_time = datetime.datetime.now()

  def test_successful_sync_hook(self):
    """Tests handling a successful synchronous hook call."""
    hooks = {'my_hook': lambda x: x * 2}
    message = json.dumps({'hook': 'my_hook', 'args': [2], 'kwargs': {}})
    result = asyncio.run(
        python_execution_multiprocess_sandbox.handle_hook_message(
            hooks, message, self.mock_connection, self.start_time
        )
    )
    self.assertIsNone(result)
    self.mock_connection.send.assert_called_once()
    sent_arg = self.mock_connection.send.call_args[0][0]
    self.assertEqual(json.loads(sent_arg), {'data': 4})

  def test_successful_async_hook(self):
    """Tests handling a successful asynchronous hook call."""

    async def my_async_hook(name):
      await asyncio.sleep(0.01)
      return f'Hello {name}'

    hooks = {'greet': my_async_hook}
    message = json.dumps({'hook': 'greet', 'args': ['World'], 'kwargs': {}})
    result = asyncio.run(
        python_execution_multiprocess_sandbox.handle_hook_message(
            hooks, message, self.mock_connection, self.start_time
        )
    )
    self.assertIsNone(result)
    self.mock_connection.send.assert_called_once()
    sent_arg = self.mock_connection.send.call_args[0][0]
    self.assertEqual(json.loads(sent_arg), {'data': 'Hello World'})

  def test_hook_not_found(self):
    """Tests handling a call to a hook that does not exist."""
    hooks = {'real_hook': lambda: 1}
    message = json.dumps({'hook': 'fake_hook', 'args': [], 'kwargs': {}})
    result = asyncio.run(
        python_execution_multiprocess_sandbox.handle_hook_message(
            hooks, message, self.mock_connection, self.start_time
        )
    )
    self.assertIsNone(result)
    self.mock_connection.send.assert_called_once()
    sent_arg = self.mock_connection.send.call_args[0][0]
    self.assertEqual(
        json.loads(sent_arg),
        {
            'exception': {
                'exception_class': 'KeyError',
                'args': ['fake_hook'],
                'hook_name': 'fake_hook',
            }
        },
    )

  def test_hook_raises_exception(self):
    """Tests handling a hook call that raises an exception."""

    def error_hook():
      raise ValueError('Test error')

    hooks = {'error_hook': error_hook}
    message = json.dumps({'hook': 'error_hook', 'args': [], 'kwargs': {}})
    result = asyncio.run(
        python_execution_multiprocess_sandbox.handle_hook_message(
            hooks, message, self.mock_connection, self.start_time
        )
    )
    self.assertIsNone(result)
    self.mock_connection.send.assert_called_once()
    sent_arg = self.mock_connection.send.call_args[0][0]
    self.assertEqual(
        json.loads(sent_arg),
        {
            'exception': {
                'exception_class': 'ValueError',
                'args': ['Test error'],
                'hook_name': 'error_hook',
            }
        },
    )

  def test_send_reply_broken_pipe(self):
    """Tests handling a BrokenPipeError when sending reply to the sandbox."""
    hooks = {'my_hook': lambda: 'ok'}
    message = json.dumps({'hook': 'my_hook', 'args': [], 'kwargs': {}})
    self.mock_connection.send.side_effect = BrokenPipeError()

    result = asyncio.run(
        python_execution_multiprocess_sandbox.handle_hook_message(
            hooks, message, self.mock_connection, self.start_time
        )
    )
    self.assertIsNotNone(result)
    self.assertEqual(result.execution_status, _ExecutionStatus.SANDBOX_ERROR)
    self.assertEqual(
        result.status_message, 'Code execution dropped (BrokenPipeError).'
    )


class MockMultiProcessSandbox(
    python_execution_multiprocess_sandbox.BaseMultiProcessSandbox
):
  """A mock implementation of BaseMultiProcessSandbox for testing."""

  def __init__(self, *args, **kwargs):
    """Initializes the mock sandbox."""
    super().__init__(*args, **kwargs)
    self.mock_requests = []
    self.mock_result_queue = queue.Queue()
    self.mock_connection = mock.MagicMock(spec=connection.Connection)
    self.sandbox_alive = True
    self._parse_result_fn = lambda payload, start, base: _SandboxResult(
        stdout=str(payload)
    )

  def _send_request(self, code: str) -> None:
    """Mocks sending a request."""
    self.mock_requests.append(code)

  def _get_result_queue(self) -> Any:
    """Returns the mock result queue."""
    return self.mock_result_queue

  def _parse_result(
      self,
      result_payload: Any,
      start_time: datetime.datetime,
      base_time: datetime.datetime,
  ) -> _SandboxResult:
    """Mocks parsing the result."""
    return self._parse_result_fn(result_payload, start_time, base_time)

  def _is_sandbox_alive(self) -> bool:
    """Returns the mock alive status."""
    return self.sandbox_alive

  def _get_sandbox_connection(self) -> connection.Connection | None:
    """Returns the mock connection."""
    return self.mock_connection

  # Methods not under test
  def is_stateful(self) -> bool:
    return True

  async def set_variables(self, **variables: Any) -> None:
    pass

  async def get_variables(self, *names: str) -> dict[str, Any]:
    return {}

  def get_hook_object(self, key: str) -> Any:
    return None


class BaseMultiProcessSandboxTest(parameterized.TestCase):
  """Tests for the BaseMultiProcessSandbox.run method."""

  def test_run_success(self):
    """Tests the run method successfully executing code."""
    sandbox = MockMultiProcessSandbox()
    sandbox.mock_result_queue.put('{"stdout": "OK"}')

    result = executing.run(sandbox.run('print("Hello")'))

    self.assertEqual(sandbox.mock_requests, ['print("Hello")'])
    self.assertEqual(result.stdout, '{"stdout": "OK"}')

  def test_run_compile_error(self):
    """Tests the run method handling a syntax error in the code."""
    sandbox = MockMultiProcessSandbox()
    code = 'a = 1 +'
    result = executing.run(sandbox.run(code))

    self.assertEqual(result.execution_status, _ExecutionStatus.COMPILE_ERROR)
    self.assertIn('SyntaxError', result.status_message)
    self.assertEmpty(sandbox.mock_requests)

  def test_run_timeout(self):
    """Tests the run method timing out when no result is received."""
    sandbox = MockMultiProcessSandbox(
        timeout=datetime.timedelta(milliseconds=10)
    )

    with mock.patch.object(asyncio, 'sleep', autospec=True):
      result = executing.run(sandbox.run('print("Timeout")'))

    self.assertEqual(result.execution_status, _ExecutionStatus.SANDBOX_TIMEOUT)
    self.assertEqual(
        result.status_message,
        'Code execution timed out (never received reply).',
    )

  def test_run_hook_message(self):
    """Tests the run method handling a hook message from the sandbox."""
    hook_called = False

    def my_hook():
      nonlocal hook_called
      hook_called = True
      return 'hook_ok'

    sandbox = MockMultiProcessSandbox(hooks={'my_hook': my_hook})

    hook_message = json.dumps({'hook': 'my_hook', 'args': [], 'kwargs': {}})

    # Configure mock_connection.poll to return True only on the first call
    sandbox.mock_connection.poll.side_effect = [True, False, False]
    sandbox.mock_connection.recv.return_value = hook_message

    sleep_call_count = 0
    result_added = False
    async def mock_sleep(duration):
      nonlocal sleep_call_count, result_added
      del duration
      sleep_call_count += 1
      # Add the final result to the queue on the *second* call to mock_sleep.
      # This ensures the first pass of the run loop processes the hook message.
      if sleep_call_count == 2 and not result_added:
        sandbox.mock_result_queue.put('{"stdout": "Final Result"}')
        result_added = True

    with mock.patch.object(
        asyncio, 'sleep', side_effect=mock_sleep, autospec=True
    ):
      with mock.patch.object(
          python_execution_multiprocess_sandbox,
          'handle_hook_message',
          wraps=python_execution_multiprocess_sandbox.handle_hook_message,
      ) as wrapped_handle:
        result = executing.run(sandbox.run('call_hook()'))

        wrapped_handle.assert_awaited_once()
        self.assertTrue(hook_called)
        sandbox.mock_connection.send.assert_called_once()
        sent_arg = sandbox.mock_connection.send.call_args[0][0]
        self.assertEqual(json.loads(sent_arg), {'data': 'hook_ok'})

        self.assertEqual(result.stdout, '{"stdout": "Final Result"}')

  def test_run_handle_hook_error(self):
    """Tests the run method when handle_hook_message returns an error result."""
    sandbox = MockMultiProcessSandbox(hooks={'my_hook': lambda: 1})
    error_result = _SandboxResult(
        execution_status=_ExecutionStatus.SANDBOX_ERROR,
        status_message='Hook handling failed',
    )
    with mock.patch.object(
        python_execution_multiprocess_sandbox,
        'handle_hook_message',
        return_value=error_result,
    ) as mock_handle:
      sandbox.mock_connection.poll.return_value = True
      sandbox.mock_connection.recv.return_value = json.dumps(
          {'hook': 'my_hook', 'args': [], 'kwargs': {}}
      )

      with mock.patch.object(asyncio, 'sleep', autospec=True):
        result = executing.run(sandbox.run('call_hook()'))

        mock_handle.assert_awaited_once()
        self.assertEqual(result, error_result)


if __name__ == '__main__':
  absltest.main()
