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

import datetime
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_utils

# Aliases
_ExecutionStatus = python_execution.ExecutionStatus
_SandboxStatus = python_execution.SandboxStatus
_SandboxResult = python_execution.SandboxResult
_SandboxResultTiming = python_execution.SandboxResultTiming


class PythonExecutionUtilsTest(parameterized.TestCase):

  def test_current_timing(self):
    start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    base_time = datetime.datetime(2023, 1, 1, 12, 0, 30)

    with mock.patch.object(
        python_execution_utils, 'datetime'
    ) as mock_datetime:
      mock_datetime.datetime.now.return_value = datetime.datetime(
          2023, 1, 1, 12, 1, 0
      )
      mock_datetime.timedelta = datetime.timedelta  # Restore timedelta

      timing = python_execution_utils.current_timing(
          start=start_time, base=base_time
      )

      self.assertIsInstance(timing, _SandboxResultTiming)
      self.assertEqual(timing.since_start, datetime.timedelta(seconds=60))
      self.assertEqual(
          timing.since_last_interaction, datetime.timedelta(seconds=30)
      )

  @parameterized.named_parameters(
      (
          'valid_result',
          json.dumps({
              'final_expression_value': 'ok',
              'stdout': 'some output',
              'sandbox_status': 'AFTER_RUNNING_CODE',
              'execution_status': 'SUCCESS',
              'status_message': '',
          }),
          _SandboxResult(
              final_expression_value='ok',
              stdout='some output',
              sandbox_status=_SandboxStatus.AFTER_RUNNING_CODE,
              execution_status=_ExecutionStatus.SUCCESS,
              status_message='',
          ),
      ),
      (
          'minimal_valid_result',
          json.dumps({'stdout': 'minimal'}),
          _SandboxResult(
              stdout='minimal',
              sandbox_status=_SandboxStatus.AFTER_RUNNING_CODE,
              execution_status=_ExecutionStatus.SUCCESS,
          ),
      ),
      (
          'valid_result_with_failure',
          json.dumps({
              'stdout': 'error output',
              'execution_status': 'EXECUTION_ERROR',
              'status_message': 'Something bad',
              'failure_details': {
                  'exception_class': 'ValueError',
                  'exception_message': 'Test error',
                  'hook_name': 'my_hook',
              },
          }),
          _SandboxResult(
              stdout='error output',
              execution_status=_ExecutionStatus.EXECUTION_ERROR,
              status_message='Something bad',
              failure_details=python_execution.FailureDetails(
                  exception_class='ValueError',
                  exception_message='Test error',
                  hook_name='my_hook',
              ),
          ),
      ),
      ('invalid_json', 'this is not json', None),
      ('not_a_dict', json.dumps('just a string'), None),
      ('missing_stdout', json.dumps({'final_expression_value': 'ok'}), None),
      (
          'string_containing_stdout',
          json.dumps('a string with stdout in it'),
          None,
      ),
  )
  def test_parse_sandbox_result_json(self, result_str, expected_result):
    start_time = datetime.datetime.now()
    base_time = start_time

    with mock.patch.object(
        python_execution_utils, 'current_timing'
    ) as mock_current_timing:
      mock_timing = _SandboxResultTiming(
          since_start=datetime.timedelta(0),
          since_last_interaction=datetime.timedelta(0),
      )
      mock_current_timing.return_value = mock_timing

      result = python_execution_utils.parse_sandbox_result_json(
          result_str, start_time, base_time
      )

      if expected_result is None:
        self.assertIsNone(result)
      else:
        expected_result.timing = mock_timing
        self.assertEqual(result, expected_result)

  def test_create_sandbox_hook_callable_success(self):
    mock_conn = mock.MagicMock()
    hook_name = 'test_hook'

    # Simulate a successful response from the main process
    mock_conn.recv.return_value = json.dumps({'data': 'hook result'})

    hook_callable = python_execution_utils.create_sandbox_hook_callable(
        hook_name, mock_conn
    )

    result = hook_callable(1, 'a', key='value')
    self.assertEqual(result, 'hook result')
    expected_message = {
        'hook': hook_name,
        'args': [1, 'a'],
        'kwargs': {'key': 'value'},
    }
    mock_conn.send.assert_called_once()
    sent_arg = mock_conn.send.call_args[0][0]
    self.assertEqual(json.loads(sent_arg), expected_message)

  @parameterized.named_parameters(
      (
          'builtin_exception',
          'ValueError',
          ValueError,
          'Test error message',
      ),
      (
          'runtime_exception',
          'UnknownError',
          RuntimeError,
          'Another error',
      ),
  )
  def test_create_sandbox_hook_callable_exception(
      self, ex_class, expected_exception, ex_message
  ):
    mock_conn = mock.MagicMock()
    hook_name = 'error_hook'
    exception_details = {
        python_execution_utils.EXCEPTION_KEY: {
            python_execution_utils.EX_CLASS_KEY: ex_class,
            python_execution_utils.EX_ARGS_KEY: [ex_message],
            python_execution_utils.EX_HOOK_NAME_KEY: hook_name,
        }
    }
    mock_conn.recv.return_value = json.dumps(exception_details)

    hook_callable = python_execution_utils.create_sandbox_hook_callable(
        hook_name, mock_conn
    )

    # Call the hook and assert an exception is raised
    with self.assertRaises(expected_exception) as context:
      hook_callable('test_arg')

    self.assertEqual(str(context.exception), ex_message)
    self.assertEqual(getattr(context.exception, '_hook_name', None), hook_name)
    expected_message = {
        'hook': hook_name,
        'args': ['test_arg'],
        'kwargs': {},
    }
    mock_conn.send.assert_called_once()
    sent_arg = mock_conn.send.call_args[0][0]
    self.assertEqual(json.loads(sent_arg), expected_message)

  @parameterized.named_parameters(
      ('empty', '', 'result = None'),
      ('single_line', '2 + 3', 'result = 2 + 3'),
      ('trailing_whitespace', '2 + 3 \n ', 'result = 2 + 3'),
      ('multiple_lines', 'x = 2\ny = 3\nx + y', 'x = 2\ny = 3\nresult = x + y'),
      (
          'trailing_comments',
          'x = 2\ny = 3\nx + y\n# C1\n# C2',
          'x = 2\ny = 3\nresult = x + y\n# C1\n# C2',
      ),
      # Note that the below is the same as `help(print)\nresult = None`.
      ('help', 'help(print)', 'result = help(print)'),
      ('entire_program_indented', '  x = 2\n  x + 3', 'x = 2\nresult = x + 3'),
      (
          'multi_line_expression_ending_in_backslash',
          'x = """\\\na\\\n"""',
          'result = x = """\\\na\\\n"""',
      ),
      (
          'multi_line_expression_with_indentation_brackets',
          'x = [\n  1,\n  2]',
          'result = x = [\n  1,\n  2]',
      ),
      (
          'multi_line_expression_with_brackets_but_last_line_not_indented',
          'x = [\n  1,\n  2\n]',
          'result = x = [\n  1,\n  2\n]',
      ),
      (
          'multi_line_expression_with_parentheses_but_last_line_not_indented',
          'x = f(\n  y,\n  z\n)',
          'result = x = f(\n  y,\n  z\n)',
      ),
      (
          'multi_line_expression_without_indentation_multi_line_string',
          'x = """\na\nb\n"""',
          'result = x = """\na\nb\n"""',
      ),
  )
  def test_adjust_code_to_set_final_expression_value(self, code, expected):
    # This covers cases where we prepend `result = ` before the last statement
    # in the code, and any other cases that involve more than simply appending
    # `result = None` at the end.
    actual = python_execution_utils.adjust_code_to_set_final_expression_value(
        code=code
    )
    self.assertEqual(expected, actual)

  @parameterized.named_parameters(
      ('comments_only', '# Comment'),
      # (internal link) start
      ('+=', 'x = 2\nx += 1'),
      ('-=', 'x = 2\nx -= 1'),
      ('assert', 'x = 2\nassert x == 2'),
      ('class', 'x = 2\nclass Y:\n  var: int = 0'),
      ('def', 'def f(x):\n  return x'),
      ('del', 'x = 2\ndel x'),
      ('elif', 'if True:\n  return 1\nelif False:\n  return 2'),
      ('else', 'if True:\n  return 1\nelse:\n  return 2'),
      ('except_1', 'try:\n  x = 2\nexcept:\n  x = None'),
      ('except_2', 'try:\n  x = 2\nexcept ValueError:\n  x = None'),
      ('finally', 'try:\n  x = 2\nfinally:\n  print(x)'),
      ('for', 'x = 1\nfor i in range(4):\n  x += 1'),
      ('from', 'x = 2\nfrom typing import Any'),
      ('if', 'if True:\n  return 1'),
      ('import', 'x = 2\nimport typing'),
      ('raise', 'raise ValueError("Bad value.")'),
      ('while', 'x = 1\nwhile x < 4:\n  x += 1'),
      # (internal link) end
  )
  def test_adjust_code_to_set_final_expression_value_result_none(self, code):
    # This covers cases where the final expression cannot be assigned to a
    # variable. In that scenario, we simply append `result = None` at the end.
    actual = python_execution_utils.adjust_code_to_set_final_expression_value(
        code=code
    )
    expected = f'{code}\nresult = None'
    self.assertEqual(expected, actual)

  def test_adjust_code_to_set_final_expression_value_invalid_python_unchanged(
      self,
  ):
    code = '2+'
    actual = python_execution_utils.adjust_code_to_set_final_expression_value(
        code=code
    )
    self.assertEqual(code, actual)

  def test_adjust_code_to_set_final_expression_value_custom_variable_name(self):
    code = 'x = 2\ny = 3\nx + y'
    variable_name = 'final_expression_value'
    expected = 'x = 2\ny = 3\nfinal_expression_value = x + y'
    actual = python_execution_utils.adjust_code_to_set_final_expression_value(
        code=code, variable_name=variable_name
    )
    self.assertEqual(expected, actual)

if __name__ == '__main__':
  absltest.main()
