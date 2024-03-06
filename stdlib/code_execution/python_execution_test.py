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

from absl.testing import absltest
from absl.testing import parameterized

from onetwo.stdlib.code_execution import python_execution


class PythonExecutionTest(parameterized.TestCase):

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
    actual = python_execution.adjust_code_to_set_final_expression_value(
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
    actual = python_execution.adjust_code_to_set_final_expression_value(
        code=code
    )
    expected = f'{code}\nresult = None'
    self.assertEqual(expected, actual)

  def test_adjust_code_to_set_final_expression_value_invalid_python_unchanged(
      self,
  ):
    code = '2+'
    actual = python_execution.adjust_code_to_set_final_expression_value(
        code=code
    )
    self.assertEqual(code, actual)

  def test_adjust_code_to_set_final_expression_value_custom_variable_name(self):
    code = 'x = 2\ny = 3\nx + y'
    variable_name = 'final_expression_value'
    expected = 'x = 2\ny = 3\nfinal_expression_value = x + y'
    actual = python_execution.adjust_code_to_set_final_expression_value(
        code=code, variable_name=variable_name
    )
    self.assertEqual(expected, actual)

if __name__ == '__main__':
  absltest.main()
