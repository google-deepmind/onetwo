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

import asyncio
import dataclasses
import math
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.core import iterating
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_safe_subset
from onetwo.stdlib.code_execution import python_execution_test_utils


# Convenience wrapper to treat `safe_eval` as an ordinary function.
def _safe_eval(*args, **kwargs) -> Any:
  return executing.run(python_execution_safe_subset.safe_eval(*args, **kwargs))


# Trivial class with attributes, for using in testing.
@dataclasses.dataclass
class Attrs:
  id: Any
  flag: Any


# Trivial class with a property, for using in testing.
class HasProperty:
  flag: Any = True

  @property
  def prop(self) -> Any:
    return self.flag


class ArithmeticEvalTest(parameterized.TestCase):
  """Tests for arithmetic eval behavior in `save_eval` and `arithmetic_eval`."""

  @parameterized.named_parameters(
      ('add', '1+2', 3),
      ('add_with_negative_number', '-4+2', -2),
      ('add_with_spaces', ' 5 +  6  ', 11),
      ('subtract', '1-2', -1),
      ('subtract_with_negative_number', '-4-2', -6),
      ('multiply', '2*7', 14),
      ('multiply_with_decimals', '3.5*4.7', 3.5 * 4.7),
      ('divide_integers', '1/2', 1 / 2, type(1 / 2)),
      ('divide_float_by_integer', '1./2', 1.0 / 2),
      ('floor_divide_integers', '1//2', 0, type(1 // 2)),
      ('floor_divide_float_by_integer', '1.//2', 0),
      ('mod', '5 % 2', 1),
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
        self.assertEqual(expected_type, type(actual_value))

    actual_value = _safe_eval(expression)
    with self.subTest('safe_eval_should_subsume_arithmetic_eval'):
      self.assertEqual(expected_value, actual_value)

  def test_error_for_infinite_loop_lambda(self):
    # https://mail.python.org/pipermail/tutor/2004-December/033828.html
    infinite_loop = '(lambda l: l(l)) (lambda l: l(l))'

    with self.subTest('arithmetic_eval'):
      self.assertRaises(
          SyntaxError,
          lambda: python_execution_safe_subset.arithmetic_eval(infinite_loop),
      )

    with self.subTest('safe_eval'):
      self.assertRaises(
          SyntaxError,
          lambda: _safe_eval(infinite_loop),
      )


class PredicateEvalTest(parameterized.TestCase):
  """Tests for predicate eval behavior in `safe_eval`."""

  def test_true_false(self):
    self.assertTrue(_safe_eval('True'))
    self.assertFalse(_safe_eval('False'))

  def test_negation(self):
    self.assertFalse(_safe_eval('not True'))
    self.assertTrue(_safe_eval('not False'))

  def test_simple_and_truth(self):
    self.assertTrue(_safe_eval('True and True'))
    self.assertFalse(_safe_eval('True and False'))
    self.assertFalse(_safe_eval('False and True'))
    self.assertFalse(_safe_eval('False and False'))

  def test_simple_or_truth(self):
    self.assertTrue(_safe_eval('True or True'))
    self.assertTrue(_safe_eval('True or False'))
    self.assertTrue(_safe_eval('False or True'))
    self.assertFalse(_safe_eval('False and False'))

  def test_comparisons(self):
    self.assertTrue(_safe_eval('2 + 3 > 4'))
    self.assertFalse(_safe_eval('2 - 3 > 4'))
    self.assertTrue(_safe_eval('-3 < +4'))
    self.assertTrue(_safe_eval('-3 * -2 > +4'))
    self.assertFalse(_safe_eval('(-3 * -2 > +4) == False'))
    self.assertTrue(_safe_eval('3 > 4 or 5 > 2'))
    self.assertTrue(_safe_eval('5 / 2 == 2.5'))
    self.assertTrue(_safe_eval('5 // 2 == 2'))

  def test_short_circuiting_and(self):
    self.assertFalse(
        _safe_eval(
            'False and error()', allowed_callables={'error': lambda: 1 / 0}
        )
    )
    with self.assertRaises(ZeroDivisionError):
      _safe_eval('True and error()', allowed_callables={'error': lambda: 1 / 0})

  def test_short_circuiting_or(self):
    self.assertTrue(
        _safe_eval(
            'True or error()', allowed_callables={'error': lambda: 1 / 0}
        )
    )
    self.assertTrue(
        _safe_eval(
            '1 > 0 or error()', allowed_callables={'error': lambda: 1 / 0}
        )
    )
    with self.assertRaises(ZeroDivisionError):
      _safe_eval('False or error()', allowed_callables={'error': lambda: 1 / 0})
    with self.assertRaises(ZeroDivisionError):
      _safe_eval(
          '2 >= 4 or error()', allowed_callables={'error': lambda: 1 / 0}
      )

  def test_variables_booleans(self):
    self.assertTrue(_safe_eval('x', context={'x': True}))
    self.assertFalse(_safe_eval('x', context={'x': False}))
    self.assertTrue(
        _safe_eval('x and (y or z)', context={'x': True, 'y': False, 'z': True})
    )
    self.assertFalse(
        _safe_eval(
            'x and (y or z)', context={'x': True, 'y': False, 'z': False}
        )
    )
    self.assertTrue(
        _safe_eval(
            'x and (y or not z)', context={'x': True, 'y': False, 'z': False}
        )
    )

  def test_error_from_undefined_variable(self):
    with self.assertRaisesRegex(NameError, "name 'foobar' is not defined"):
      _safe_eval('foobar')

  def test_chained_comparison_count(self):
    # Checks that, when doing chained comparisons, we hold onto the old results
    # to avoid recomputation.
    count_f = [0]
    count_y = [0]
    count_z = [0]

    def f():
      count_f[0] += 1
      return 10

    def g():
      count_y[0] += 1
      return 20

    def h():
      count_z[0] += 1
      return 30

    self.assertTrue(
        _safe_eval(
            'f() < g() < h()', allowed_callables={'f': f, 'g': g, 'h': h}
        )
    )
    self.assertEqual(count_f[0], 1)
    self.assertEqual(count_y[0], 1)
    self.assertEqual(count_z[0], 1)

  def test_calls_and_conjunctions(self):
    self.assertTrue(
        _safe_eval(
            'func_1(foo) and not func_2(bar)',
            context={'foo': True, 'bar': True},
            allowed_callables={
                'func_1': lambda x: x,
                'func_2': lambda x: not x,
            },
        )
    )

    self.assertFalse(
        _safe_eval(
            'func_1(foo) and func_2(bar)',
            context={'foo': True, 'bar': True},
            allowed_callables={
                'func_1': lambda x: x,
                'func_2': lambda x: not x,
            },
        )
    )

  def test_attributes_booleans(self):
    obj = Attrs(id='hello', flag=True)
    self.assertTrue(_safe_eval('obj.flag', context={'obj': obj}))
    self.assertFalse(_safe_eval('not(obj.flag)', context={'obj': obj}))

  def test_attributes_mixed_with_callables(self):
    obj = Attrs(id=22, flag=False)
    self.assertTrue(
        _safe_eval(
            'plus1(obj.id) == 23',
            context={'obj': obj},
            allowed_callables={'plus1': lambda n: n + 1},
        )
    )
    self.assertFalse(
        _safe_eval(
            '(plus1(obj.id) == 23) and obj.flag',
            context={'obj': obj},
            allowed_callables={'plus1': lambda n: n + 1},
        )
    )

  def test_attributes_strings(self):
    obj = Attrs(id='hello', flag=None)
    self.assertTrue(_safe_eval('obj.id == "hello"', context={'obj': obj}))
    self.assertFalse(_safe_eval('obj.id != "hello"', context={'obj': obj}))
    self.assertFalse(_safe_eval('obj.id == "Hello"', context={'obj': obj}))

  def test_attributes_three_chained_lookups(self):
    obj = Attrs(id=Attrs(id=None, flag=True), flag=None)
    self.assertTrue(_safe_eval('obj.id.flag', context={'obj': obj}))

  def test_properties_booleans(self):
    obj = HasProperty()
    self.assertTrue(_safe_eval('obj.flag == obj.prop', context={'obj': obj}))
    self.assertFalse(_safe_eval('obj.flag != obj.prop', context={'obj': obj}))

  def test_properties_assigned_values(self):
    obj = HasProperty()
    obj.flag = 5
    self.assertTrue(_safe_eval('obj.flag == 5', context={'obj': obj}))
    self.assertFalse(_safe_eval('obj.flag != 5', context={'obj': obj}))

  def test_attribute_errors_prohibited_attribute(self):
    with self.assertRaises(SyntaxError):
      # Prohibited attribute
      _safe_eval('obj.__class__ == obj.__class__', context={'obj': 'test'})

  def test_attribute_errors_attribute_error(self):
    obj = Attrs(id=Attrs(id=None, flag=True), flag=None)
    with self.assertRaises(AttributeError):
      _safe_eval('obj.id.id.id', context={'obj': obj})
    with self.assertRaises(AttributeError):
      # Attribute reference fails
      _safe_eval('obj.non_existing_attr', context={'obj': 'test'})

  def test_attribute_errors_syntax_error(self):
    obj = Attrs(id='id ', flag=None)
    with self.assertRaises(SyntaxError):
      _safe_eval('obj.id.strip() == "id"', context={'obj': obj})
    with self.assertRaises(SyntaxError):
      _safe_eval('obj.func().attr', context={'obj': obj})


class SemiliteralEvalTest(parameterized.TestCase):
  """Tests for semi-literal eval behavior in `safe_eval`."""

  def test_string(self):
    self.assertEqual(_safe_eval('"hello world"'), 'hello world')
    self.assertEqual(_safe_eval("'hello world'"), 'hello world')

  def test_bytes(self):
    self.assertEqual(_safe_eval('b"hello world"'), b'hello world')
    self.assertEqual(_safe_eval("b'hello world'"), b'hello world')

  def test_number(self):
    self.assertEqual(_safe_eval('42'), 42)
    self.assertEqual(_safe_eval('-42'), -42)

  def test_tuple(self):
    self.assertEqual(_safe_eval('(3, "four")'), (3, 'four'))
    self.assertEqual(_safe_eval('()'), ())

  def test_list(self):
    self.assertEqual(_safe_eval('[1, 2, 3]'), [1, 2, 3])
    self.assertEqual(_safe_eval('[[]]'), [[]])

  def test_dict(self):
    self.assertEqual(_safe_eval('{"x":16}'), {'x': 16})
    self.assertEqual(_safe_eval('{"x":{"y": "z"}}'), {'x': {'y': 'z'}})
    self.assertEqual(_safe_eval('{}'), {})

  def test_set(self):
    self.assertEqual(_safe_eval('{3, 4, 5}'), {3, 4, 5})
    self.assertEqual(_safe_eval('{"unity"}'), {'unity'})

  def test_constant(self):
    self.assertEqual(_safe_eval('True'), True)
    self.assertEqual(_safe_eval('False'), False)
    self.assertIsNone(_safe_eval('None'))

    self.assertRaises(NameError, lambda: _safe_eval('true'))
    self.assertEqual(_safe_eval('true', context={'true': True}), True)

  def test_callable(self):
    self.assertEqual(
        _safe_eval('[1, "two", set([3])]', allowed_callables={'set': set}),
        [1, 'two', set([3])],
    )

    self.assertEqual(
        _safe_eval(
            'cons(42, (cons(43, None)))',
            allowed_callables={'cons': lambda x, y: [x, y]},
        ),
        [42, [43, None]],
    )

    self.assertEqual(
        _safe_eval(
            'cons(y=42, x=(cons(y=43, x=None)))',
            allowed_callables={'cons': lambda x, y: [x, y]},
        ),
        [[None, 43], 42],
    )

  def test_dotted_names_in_callable(self):
    # Dotted names are allowed in callable position, but must be explicitly
    # listed as a fully-qualified name in the callables dictionary.  This is to
    # curtail arbitrary attribute lookup.
    self.assertEqual(
        _safe_eval('a.B(42)', allowed_callables={'a.B': 'result is {}'.format}),
        'result is 42',
    )

  def test_dotted_names_not_allowed_in_variable(self):
    # Dotted names are allowed outside of callable position, but are treated as
    # attribute lookups (see corresponding tests cases above). Directly
    # assigning a value to a dotted name in the context dictionary is not
    # supported.
    with self.assertRaisesRegex(KeyError, 'foo'):
      _safe_eval('foo.BAR', context={'foo.BAR': 42})

  def test_unknown_callables(self):
    self.assertRaises(SyntaxError, lambda: _safe_eval('[1, "two", set([3])]'))
    self.assertRaises(SyntaxError, lambda: _safe_eval('[1, "two", True()]'))

  def test_callables_only_in_callable_position(self):
    self.assertRaises(
        NameError, lambda: _safe_eval('set', allowed_callables={'set': set})
    )


class SafeEvalAdditionalBehaviorTest(parameterized.TestCase):
  """Tests for additional behavior in `safe_eval`."""

  @parameterized.named_parameters(
      ('assign_int', 'x = 2', {}, 2, {'x': 2}),
      ('assign_string', 'x = "a"', {}, 'a', {'x': 'a'}),
      ('assign_tuple', 'x = ("a", 2)', {}, ('a', 2), {'x': ('a', 2)}),
      ('assign_list', 'x = ["a", "b"]', {}, ['a', 'b'], {'x': ['a', 'b']}),
      ('assign_from_var', 'x = y + 1', {'y': 2}, 3, {'x': 3, 'y': 2}),
      ('assign_overwrite', 'x = 3', {'x': 2}, 3, {'x': 3}),
      ('iadd', 'x += 1', {'x': 2}, None, {'x': 3}),
      ('iand', 'x &= {1, 3}', {'x': {1, 2}}, None, {'x': {1}}),
      ('iconcat', 'x += ["b"]', {'x': ['a']}, None, {'x': ['a', 'b']}),
      ('ifloordiv', 'x //= 2', {'x': 5}, None, {'x': 2}),
      ('imod', 'x %= 2', {'x': 5}, None, {'x': 1}),
      ('imul', 'x *= 3', {'x': 2}, None, {'x': 6}),
      ('ior', 'x |= {2}', {'x': {1}}, None, {'x': {1, 2}}),
      ('ipow', 'x **= 2', {'x': 3}, None, {'x': 9}),
      ('isub', 'x -= 1', {'x': 2}, None, {'x': 1}),
      ('itruediv', 'x /= 2', {'x': 5}, None, {'x': 5 / 2}),
      ('ixor', 'x ^= {1, 3}', {'x': {1, 2}}, None, {'x': {2, 3}}),
      ('multiple_assignment', 'y = x = 2', {}, 2, {'x': 2, 'y': 2}),
      ('multiple_statements', 'x = 2\ny = x + 1', {}, 3, {'x': 2, 'y': 3}),
  )
  def test_variable_assignment(
      self, code, context, expected_value, expected_context
  ):
    value = _safe_eval(code, context=context)
    with self.subTest(name='correct_value'):
      self.assertEqual(expected_value, value)
    with self.subTest(name='correct_context'):
      self.assertEqual(expected_context, context)

  @parameterized.named_parameters(
      {
          'testcase_name': 'dotted_name_representing_function',
          'code': 'math.exp(0)',
          'context': {},
          'allowed_callables': {'math.exp': math.exp},
          'expected_value': 1,
          'expected_context': {},
      },
      {
          'testcase_name': 'dotted_name_representing_method',
          'code': "x.append('b')\nx",
          'context': {'x': ['a']},
          'allowed_callables': {'list.append': list.append},
          'expected_value': ['a', 'b'],
          'expected_context': {'x': ['a', 'b']},
      },
  )
  def test_function_calls(
      self, code, context, allowed_callables, expected_value, expected_context
  ):
    value = _safe_eval(
        code, context=context, allowed_callables=allowed_callables
    )
    with self.subTest(name='correct_value'):
      self.assertEqual(expected_value, value)
    with self.subTest(name='correct_context'):
      self.assertEqual(expected_context, context)

  @parameterized.named_parameters(
      ('if_exp_true', '1 if True else 2', {}, 1),
      ('if_exp_false', '1 if False else 2', {}, 2),
      ('if_statement_true', 'x = 2\nif True:\n  x = 1\nx', {}, 1),
      ('if_statement_false', 'x = 2\nif False:\n  x = 1\nx', {}, 2),
      ('if_else_true', 'if True:\n  x = 1\nelse:\n  x = 2\nx', {}, 1),
      ('if_else_false', 'if False:\n  x = 1\nelse:\n  x = 2\nx', {}, 2),
      (
          'if_else_elif',
          'if False:\n  x = 1\nelif True:\n  x = 2\nelse:\n  x = 3\nx',
          {},
          2,
      ),
      ('format_with_string_mod', "'x=%s' % x", {'x': 'Zürich'}, 'x=Zürich'),
      ('fstring_default', "f'x={x}'", {'x': 'Zürich'}, 'x=Zürich'),
      ('fstring_s', "f'x={x!s}'", {'x': 'Zürich'}, 'x=Zürich'),
      ('fstring_r', "f'x={x!r}'", {'x': 'Zürich'}, "x='Zürich'"),
      ('fstring_a', "f'x={x!a}'", {'x': 'Zürich'}, "x='Z\\xfcrich'"),
      ('set_difference', '{1, 2} - {1, 3}', None, {2}),
      ('set_intersection', '{1, 2} & {1, 3}', None, {1}),
      ('set_symmetric_diff', '{1, 2} ^ {1, 3}', None, {2, 3}),
      ('set_union', '{1} | {2}', None, {1, 2}),
  )
  def test_miscellaneous_syntax(self, code, context, expected_value):
    value = _safe_eval(code, context=context)
    self.assertEqual(expected_value, value)

  @parameterized.named_parameters(
      ('insert_in_dict', 'x["a"] = 2', {'x': {}}, 'ast.Subscript'),
      (
          'for_loop',
          'result = 0\nfor x in range(2):\n  result += x\nresult',
          {},
          'ast.For',
          {'range': range},
      ),
      ('list_comprehension', '[str(x) for x in (1, 2)]', {}, 'ast.ListComp'),
      (
          'while_loop',
          'x = 1\nwhile x < 10:  x *= 2\nx',
          {},
          'ast.While',
      ),
      (
          'star_arguments',
          'args = [1]\nstr(*args)',
          {},
          'Star arguments not supported',
          {'str': str},
      ),
      (
          'write_to_attribute',
          'x.id = 2',
          {'x': Attrs(id=1, flag=False)},
          'Unexpected assignment target type',
      ),
  )
  def test_syntax_that_is_not_supported_now_but_may_be_in_the_future(
      self, code, context, error_regex, allowed_callables=None
  ):
    with self.assertRaisesRegex(SyntaxError, error_regex):
      _safe_eval(
          code=code, context=context, allowed_callables=allowed_callables
      )

  @parameterized.named_parameters(
      ('class_definition', 'class C:\n  pass', {}, 'type=ClassDef'),
      ('function_definition', 'def f(x):\n  return x', {}, 'type=FunctionDef'),
      ('lambda', 'f = lambda x: x', {}, 'ast.Lambda'),
  )
  def test_syntax_that_is_permanently_not_supported(
      self, code, context, error_regex, allowed_callables=None
  ):
    with self.assertRaisesRegex(SyntaxError, error_regex):
      _safe_eval(
          code=code, context=context, allowed_callables=allowed_callables
      )


def f1(a, b, c):
  """Normal test hook."""
  return a + b + c


@executing.make_executable
async def f2(a, b, c):
  """Executable test hook."""
  return a + b + c


@iterating.to_thread
def f3(a, b, c):
  """Thread test hook."""
  return a + b + c


def f4(a, b, c):
  """ValueError test hook."""
  del a, b, c
  raise ValueError('Some error.')


def f5(a, b, c):
  """TypeError test hook."""
  del a, b, c
  raise TypeError('Crash in hook.')


class PythonSandboxSafeSubsetTest(
    parameterized.TestCase, python_execution_test_utils.SandboxResultAssertions
):

  @parameterized.named_parameters(
      ('simple', '2+3', 5),
      ('parenthesized', '2*(2+3)+1', 11),
      ('variables', 'a=1+1\na', 2),
      ('variables2', 'a=1+1\nb=a*a\na+b', 6),
      ('strings', "a='word'\nlen(a)", 4),
      ('strings2', "a='word'\nb=a + ' other'\nb", 'word other'),
      # Note that for assignment operations like (`a = 2`), we are returning the
      # assigned value (e.g., `2`) as the value of the expression, motivated by
      # the fact that one can write `b = a = 2` and get the 2 as the value of b.
      # If we were entirely strict about this, however, the value of the
      # assignment expression in Python is actually undefined, as can be seen by
      # the fact, for example, that it is not valid to write `b = (a = 2)`.
      ('assignment', 'a = 2', 2),
      ('multiple_assignment', 'b = a = 2', 2),
      ('callable_bool', 'bool(2)', True),
      ('callable_dict', "dict([('x', 1), ('y', 2)])", {'x': 1, 'y': 2}),
      ('callable_int', 'int(3.14)', 3),
      ('callable_float', '(float(1.0) / 3) * 3', 1.0),
      ('callable_len', "len(['a', 'b'])", 2),
      ('callable_list', 'list((0, 1, 2))', [0, 1, 2]),
      ('callable_range', 'list(range(3))', [0, 1, 2]),
      ('callable_set', "set(['a', 'b', 'a'])", {'a', 'b'}),
      ('callable_str', 'str(2)', '2'),
      ('method_dict_clear', "d = {'x': 1, 'y': 2}\nd.clear()\nd", {}),
      ('method_dict_copy', "d = {'x': 1, 'y': 2}\nd.copy()", {'x': 1, 'y': 2}),
      ('method_dict_fromkeys', "dict.fromkeys(('x',), 0)", {'x': 0}),
      ('method_dict_get', "d = {'x': 1, 'y': 2}\nd.get('x')", 1),
      ('method_dict_items', "d = {'x': 1}\nlist(d.items())", [('x', 1)]),
      ('method_dict_keys', "d = {'x': 1}\nlist(d.keys())", ['x']),
      ('method_dict_pop', "d = {'x': 1, 'y': 2}\nd.pop('x')\nd", {'y': 2}),
      ('meth0d_dict_popitem', "d = {'x': 1, 'y': 2}\nd.popitem()\nd", {'x': 1}),
      ('method_dict_setdefault', "d = {}\nd.setdefault('x', 1)\nd", {'x': 1}),
      ('method_dict_update', "d = {'x': 1}\nd.update({'x': 2})\nd", {'x': 2}),
      ('method_dict_values', "d = {'x': 1}\nlist(d.values())", [1]),
      ('method_list_append', 'x = ["a"]\nx.append("b")\nx', ['a', 'b']),
      ('method_list_clear', 'x = ["a"]\nx.clear()\nx', []),
      ('method_list_copy', 'x = ["a"]\nx.copy()', ['a']),
      ('method_list_count', 'x = ["a", "b"]\nx.count("a")', 1),
      ('method_list_extend', 'x = ["a"]\nx.extend(["b"])\nx', ['a', 'b']),
      ('method_list_index', 'x = ["a", "b"]\nx.index("b")', 1),
      ('method_list_insert', 'x = ["a"]\nx.insert(0, "b")\nx', ['b', 'a']),
      ('method_list_pop', 'x = ["a", "b"]\nx.pop(0)', 'a'),
      ('method_list_remove', 'x = ["a", "b"]\nx.remove("b")\nx', ['a']),
      ('method_list_reverse', 'x = ["a", "b"]\nx.reverse()\nx', ['b', 'a']),
      ('method_list_sort', 'x = ["b", "a"]\nx.sort()\nx', ['a', 'b']),
  )
  def test_run_on_simple_expressions(self, code, expected_value):
    expected_result = python_execution.SandboxResult(
        final_expression_value=expected_value
    )

    sb = python_execution_safe_subset.PythonSandboxSafeSubset()

    @sb.start()
    async def wrapper(code: str):
      result = await sb.run(code)
      return result

    result = asyncio.run(wrapper(code))
    self.assertSandboxResultEqualIgnoringTiming(expected_result, result)

  @parameterized.named_parameters(
      (
          'print_string',
          'print("hello")',
          python_execution.SandboxResult(stdout='hello\n'),
      ),
      (
          'print_arithmetic_expression',
          'print(2+3)',
          python_execution.SandboxResult(stdout='5\n'),
      ),
      (
          'multiple_prints',
          'print("hello")\nprint(2+3)',
          python_execution.SandboxResult(stdout='hello\n5\n'),
      ),
      (
          'print_followed_by_expression',
          'print("hello")\n2+3',
          python_execution.SandboxResult(
              stdout='hello\n', final_expression_value=5
          ),
      ),
      (
          'print_with_hook_call',
          'print(f())',
          python_execution.SandboxResult(
              stdout='Hello world!\n',
          ),
      ),
  )
  def test_run_with_print_statements(self, code, expected_result):

    def f():
      return 'Hello world!'

    sb = python_execution_safe_subset.PythonSandboxSafeSubset(hooks={'f': f})

    @sb.start()
    async def wrapper(code: str):
      result = await sb.run(code)
      return result

    result = asyncio.run(wrapper(code))
    self.assertSandboxResultEqualIgnoringTiming(expected_result, result)

  @parameterized.named_parameters(
      (
          'return_exception',
          'print("Hello world!")\na=b',
          python_execution.SandboxResult(
              stdout='Hello world!\n',
              execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
              status_message="NameError - name 'b' is not defined",
              failure_details=python_execution.FailureDetails(
                  exception_class='NameError',
                  exception_message="name 'b' is not defined",
              ),
          ),
      ),
      (
          'syntax_error',
          'print("Hello world!)',
          python_execution.SandboxResult(
              sandbox_status=python_execution.SandboxStatus.BEFORE_RUNNING_CODE,
              execution_status=python_execution.ExecutionStatus.COMPILE_ERROR,
              status_message=(
                  'SyntaxError - unterminated string literal (detected at line'
                  ' 1) (<string>, line 1)'
              ),
          ),
      ),
      (
          'invalid_literal_expression',
          'print(int("not an int"))',
          python_execution.SandboxResult(
              execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
              status_message=(
                  "ValueError - invalid literal for int() with base 10: 'not an"
                  " int'"
              ),
              failure_details=python_execution.FailureDetails(
                  exception_class='ValueError',
                  exception_message=(
                      "invalid literal for int() with base 10: 'not an int'"
                  ),
              ),
          ),
      ),
      (
          'hook_exception',
          'print(f())',
          python_execution.SandboxResult(
              execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
              status_message='ValueError - value error',
              failure_details=python_execution.FailureDetails(
                  hook_name='f',
                  exception_class='ValueError',
                  exception_message='value error',
              ),
          ),
      ),
  )
  def test_run_error(self, code: str, expected: python_execution.SandboxResult):

    def f():
      raise ValueError('value error')

    sb = python_execution_safe_subset.PythonSandboxSafeSubset(hooks={'f': f})

    @sb.start()
    async def wrapper(code: str):
      result = await sb.run(code)
      return result

    result = asyncio.run(wrapper(code))
    self.assertSandboxResultEqualIgnoringTiming(expected, result)

  @parameterized.named_parameters(
      (
          'normal_test_hook',
          'f1',
          f1,
          python_execution.SandboxResult(stdout='Hello world!\n'),
      ),
      (
          'executable_test_hook',
          'f2',
          f2,
          python_execution.SandboxResult(stdout='Hello world!\n'),
      ),
      (
          'thread_test_hook',
          'f3',
          f3,
          python_execution.SandboxResult(stdout='Hello world!\n'),
      ),
      (
          'value_error_test_hook',
          'f4',
          f4,
          python_execution.SandboxResult(
              execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
              status_message='ValueError - Some error.',
              failure_details=python_execution.FailureDetails(
                  hook_name='f4',
                  exception_class='ValueError',
                  exception_message='Some error.',
              ),
          ),
      ),
      (
          'type_error_test_hook',
          'f5',
          f5,
          python_execution.SandboxResult(
              execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
              status_message='TypeError - Crash in hook.',
              failure_details=python_execution.FailureDetails(
                  hook_name='f5',
                  exception_class='TypeError',
                  exception_message='Crash in hook.',
              ),
          ),
      ),
  )
  def test_run_with_hooks(self, fn_name, fn, expected):
    code = f'r = {fn_name}("Hello", c="!", b=" world")\nprint(r)'
    sb = python_execution_safe_subset.PythonSandboxSafeSubset(
        hooks={fn_name: fn}
    )

    @sb.start()
    async def wrapper(code: str):
      result = await sb.run(code)
      return result

    result = asyncio.run(wrapper(code))
    self.assertSandboxResultEqualIgnoringTiming(expected, result)

  def test_run_statefully(self):
    sb = python_execution_safe_subset.PythonSandboxSafeSubset()

    @sb.start()
    async def wrapper(code: str):
      result = await sb.run(code)
      return result

    result = None
    code_blocks = ['x = 2', 'x *= 2\ny = 3', 'x + y']
    for code in code_blocks:
      result = asyncio.run(wrapper(code))

    with self.subTest(name='final_result'):
      self.assertSandboxResultEqualIgnoringTiming(
          python_execution.SandboxResult(final_expression_value=7), result
      )

    with self.subTest(name='final_context'):
      self.assertDictEqual({'x': 4, 'y': 3}, sb.context)

  def test_variables(self):
    code = textwrap.dedent("""\
        print(var1)
        print(var2)
        var1 += [1]
        var2 += 1
    """)
    sb = python_execution_safe_subset.PythonSandboxSafeSubset()

    @sb.start()
    async def wrapper():
      await sb.set_variables(var1=[], var2=0)
      result = await sb.run(code)
      variables = await sb.get_variables('var1', 'var2')
      return result, variables

    result, variables = asyncio.run(wrapper())

    expected_result = python_execution.SandboxResult(stdout='[]\n0\n')

    with self.subTest('result_is_correct'):
      self.assertSandboxResultEqualIgnoringTiming(expected_result, result)

    with self.subTest('variables_are_correct'):
      self.assertEqual({'var1': [1], 'var2': 1}, variables, variables)


if __name__ == '__main__':
  absltest.main()
