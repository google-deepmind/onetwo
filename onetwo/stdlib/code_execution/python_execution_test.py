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

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.stdlib.code_execution import python_execution

_ExecutionStatus = python_execution.ExecutionStatus
_SandboxResult = python_execution.SandboxResult


class SandboxResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'success_no_output',
          python_execution.SandboxResult(),
          'None',
      ),
      (
          'success_stdout_only',
          python_execution.SandboxResult(stdout='Hi'),
          'Hi',
      ),
      (
          'success_final_expression_value_only',
          python_execution.SandboxResult(final_expression_value=2),
          '2',
      ),
      (
          'success_stdout_and_final_expression_value',
          python_execution.SandboxResult(final_expression_value=2, stdout='Hi'),
          'Hi\n2',
      ),
      (
          'error_no_output',
          python_execution.SandboxResult(
              execution_status=_ExecutionStatus.EXECUTION_ERROR,
              status_message='Error message.',
          ),
          'EXECUTION_ERROR: Error message.',
      ),
      (
          'error_stdout_only',
          python_execution.SandboxResult(
              stdout='Hi',
              execution_status=_ExecutionStatus.EXECUTION_ERROR,
              status_message='Error message.',
          ),
          'Hi\nEXECUTION_ERROR: Error message.',
      ),
      (
          'error_final_expression_value_only',
          python_execution.SandboxResult(
              final_expression_value=2,
              execution_status=_ExecutionStatus.EXECUTION_ERROR,
              status_message='Error message.',
          ),
          '2\nEXECUTION_ERROR: Error message.',
      ),
      (
          'error_stdout_and_final_expression_value',
          python_execution.SandboxResult(
              final_expression_value=2,
              stdout='Hi',
              execution_status=_ExecutionStatus.EXECUTION_ERROR,
              status_message='Error message.',
          ),
          'Hi\n2\nEXECUTION_ERROR: Error message.',
      ),
  )
  def test_to_string(self, result, expected_string):
    self.assertEqual(expected_string, str(result))


if __name__ == '__main__':
  absltest.main()
