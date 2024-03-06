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
from onetwo.stdlib.code_execution import python_execution_safe_subset


class ArithmeticEvalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('add', '1+2', 3),
      ('add_with_negative_number', '-4+2', -2),
      ('add_with_spaces', ' 5 +  6  ', 11),
      ('subtract', '1-2', -1),
      ('subtract_with_negative_number', '-4-2', -6),
      ('multiply', '2*7', 14),
      ('multiply_with_decimals', '3.5*4.7', 3.5 * 4.7),
      ('divide_integers', '1/2', 1 / 2, type(1 / 2)),
      ('divide_float_by_integer', '1./2', 1. / 2),
      ('floor_divide_integers', '1//2', 0, type(1 // 2)),
      ('floor_divide_float_by_integer', '1.//2', 0),
      ('power', '2**3', 2**3, type(2**3)),
      ('power_with_fractional_exponent', '2**0.5', 2**0.5, type(2**0.5)),
      ('unary_plus', '+42', 42),
      ('unary_minus', '-42', -42),
      ('mixed', '1*100 / (99.0 + 2.0-1) + 1**3', 2.0),
  )
  def test_arithmetic_eval(
      self, expression, expected_value, expected_type=None
  ):
    actual_value = python_execution_safe_subset.arithmetic_eval(expression)
    with self.subTest('correct_value'):
      self.assertEqual(expected_value, actual_value)

    if expected_type is not None:
      with self.subTest('correct_type'):
        self.assertEqual(
            expected_type,
            type(actual_value)
        )

  def test_arithmetic_eval_syntax_error(self):
    # https://mail.python.org/pipermail/tutor/2004-December/033828.html
    infinite_loop = '(lambda l: l(l)) (lambda l: l(l))'
    self.assertRaises(
        SyntaxError,
        lambda: python_execution_safe_subset.arithmetic_eval(infinite_loop),
    )

if __name__ == '__main__':
  absltest.main()
