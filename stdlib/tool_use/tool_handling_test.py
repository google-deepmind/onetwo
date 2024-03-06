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

import pprint
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import tool_use
from onetwo.core import executing
from onetwo.core import routing
from onetwo.stdlib.tool_use import tool_handling

_ArgumentFormat = tool_handling.ArgumentFormat

# Default reply for LanguageModelEngineForTest to return when it receives a
# prompt that it was not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


def _add_sync(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


async def _add_async(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


@executing.make_executable
def _add_executable(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


class ToolUseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'simple_yaml',
          _ArgumentFormat.YAML,
          (
              'run:\n  code: "def fn(x):\\n  return x + 1\\nresult ='
              ' fn(10)\\n"\n  language: python\n'
          ),
      ),
      (
          'code_yaml',
          _ArgumentFormat.YAML_CODE,
          (
              'run:\n'
              '  code: |\n'
              '    def fn(x):\n'
              '      return x + 1\n'
              '    result = fn(10)\n'
              '  language: python\n'
          ),
      ),
  )
  def test_render_yaml_multiline(self, fmt, expected):
    kwargs = {
        'language': 'python',
        'code': textwrap.dedent("""\
          def fn(x):
            return x + 1
          result = fn(10)
        """),
    }
    rendered = tool_handling._render_call_content(fmt, 'run', **kwargs)
    self.assertEqual(rendered, expected, pprint.pformat(rendered))

  def test_render_parse_roundtrip(self):
    arg_list = (1, 'a', [1, 2], {'a': 1})
    arg_dict = {
        'a': 1,
        'b': 'f ',
        'c': [1, 2],
        'd': {'a': 1},
        'code': """\
    def my_f():
      return 1
    """,
    }
    for fmt in _ArgumentFormat:
      rendered = tool_handling._render_call_content(
          fmt, 'f', *arg_list, **arg_dict
      )
      (_, fn, args, kwargs) = tool_handling._parse_call_content(
          fmt, rendered, {}
      )

      with self.subTest(f'parse_call_correctly_{fmt.value}'):
        self.assertEqual(fn, 'f')
        self.assertEqual(args, arg_list)
        self.assertEqual(kwargs, arg_dict)

      with self.subTest(f'roundtrip_consistent_{fmt.value}'):
        re_rendered = tool_handling._render_call_content(
            fmt, fn, *args, **kwargs
        )
        self.assertEqual(rendered, re_rendered)

      parse_by_rendered = {
          _ArgumentFormat.PYTHON: _ArgumentFormat.PYTHON,
          _ArgumentFormat.YAML: _ArgumentFormat.YAML,
          _ArgumentFormat.YAML_CODE: _ArgumentFormat.YAML,
          _ArgumentFormat.JSON: _ArgumentFormat.JSON,
          _ArgumentFormat.JSON_SINGLE: _ArgumentFormat.JSON_SINGLE,
      }

      expected_remainder = '\nSome continuation'
      with self.subTest(f'parse_and_consume_{fmt.value}'):
        rendered = tool_handling.render_call(
            fmt, 'my_function', *arg_list, **arg_dict
        )
        rendered += expected_remainder
        _, fn, args, kwargs, inferred_fmt, index = (
            tool_handling.parse_and_consume_call(  # pylint: disable=line-too-long
                rendered, context_vars={}
            )
        )
        remainder = rendered[index:]
        self.assertEqual(fn, 'my_function')
        self.assertEqual(args, arg_list)
        self.assertEqual(kwargs, arg_dict)
        self.assertEqual(remainder, expected_remainder)
        self.assertEqual(inferred_fmt, parse_by_rendered[fmt])

      with self.subTest(f'full_roundtrip_consistent_{fmt.value}'):
        # We only check for formats that can be fully determined automatically.
        if fmt == inferred_fmt:
          re_rendered = tool_handling.render_call(
              inferred_fmt, fn, *args, **kwargs
          )
          re_rendered += expected_remainder
          self.assertEqual(rendered, re_rendered)

  @parameterized.named_parameters(
      ('positional', 'Python(1)', (1,), {}),
      ('keyword', 'Python(a=1, b="a")', tuple(), {'a': 1, 'b': 'a'}),
  )
  def test_parse_python_format(self, act_text, expected_args, expected_kwargs):
    _, fn, args, kwargs, fmt, index = tool_handling.parse_and_consume_call(
        act_text, context_vars={}
    )
    self.assertEqual(fn, 'Python')
    self.assertEqual(args, expected_args)
    self.assertEqual(kwargs, expected_kwargs)
    self.assertEqual(fmt, _ArgumentFormat.PYTHON)
    self.assertLen(act_text, index)

  @parameterized.named_parameters(
      dict(
          testcase_name='Two variables',
          act_text='Add(a1, a2)',
          context_vars={'a1': 1, 'a2': 2},
          expected_name='Add',
          expected_args=(1, 2),
          expected_kwargs={},
      ),
      dict(
          testcase_name='Double use single variable',
          act_text='Add(a1, a1)',
          context_vars={'a1': 1},
          expected_name='Add',
          expected_args=(1, 1),
          expected_kwargs={},
      ),
      dict(
          testcase_name='One variables, one literal',
          act_text='Add(a1, 5)',
          context_vars={'a1': 1},
          expected_name='Add',
          expected_args=(1, 5),
          expected_kwargs={},
      ),
  )
  def test_parse_variable_as_arg(
      self,
      act_text,
      context_vars,
      expected_name,
      expected_args,
      expected_kwargs,
  ):
    def add(arg1: Any, arg2: Any) -> Any:
      return arg1 + arg2

    routing.function_registry['add'] = add
    tool_handler = tool_handling.ToolHandler()
    tool_handler.register()
    tool_handler.register_tool(
        tool_handling.ToolSpec(
            name='Add',
            name_in_registry='add',
            description='Tool description.',
            example='Tool example.',
        )
    )

    _, name, args, kwargs, _, _ = tool_handling.parse_and_consume_call(
        text=act_text, context_vars=context_vars
    )

    with self.subTest('correct_name'):
      self.assertEqual(expected_name, name)

    with self.subTest('correct_args'):
      self.assertSequenceEqual(expected_args, args)

    with self.subTest('correct_kwargs'):
      self.assertDictEqual(expected_kwargs, kwargs)

  def test_parse_variable_as_arg_not_in_context_fails(self):
    # Simple example of a function that takes two arguments.
    tool_name = 'Add'

    def add(arg1: Any, arg2: Any) -> Any:
      return arg1 + arg2

    routing.function_registry['add'] = add
    tool_handler = tool_handling.ToolHandler()
    tool_handler.register()
    tool_handler.register_tool(
        tool_handling.ToolSpec(
            name=tool_name,
            name_in_registry='add',
            description='Tool description.',
            example='Tool example.',
        )
    )

    with self.assertRaisesRegex(ValueError, 'does not exist in context'):
      tool_handling.parse_and_consume_call(text='Add(a1, a1)', context_vars={})

  def test_register(self):
    def f_in_registry(arg1: Any, arg2: Any) -> str:
      return f'f_in_registry: {arg1} {arg2}'

    def f_in_tool_handler(arg1: Any, arg2: Any) -> Any:
      return f'f_in_tool_handler: {arg1} {arg2}'

    routing.function_registry['f'] = f_in_registry
    routing.function_registry['g'] = f_in_tool_handler
    tool_handler = tool_handling.ToolHandler()
    tool_handler.register_tool(
        tool_handling.ToolSpec(name='f', name_in_registry='g')
    )

    args = ['a', 'b']
    kwargs = {}

    # Before we call `tool_handler.register()`, `tool_use.run_tool` should
    # invoke the default implementation, which simply looks up 'f' in the
    # function registry.
    result_before = executing.run(tool_use.run_tool('f', args, kwargs))
    with self.subTest('before_registering_tool_handler'):
      self.assertEqual('f_in_registry: a b', result_before)

    # After we call `tool_handler.register()`, `tool_use.run_tool()` should
    # invoke `tool_handler.run_tool()`, which will call the tool that is
    # registered under the name 'f' in the tool handler.
    tool_handler.register()
    result_after = executing.run(tool_use.run_tool('f', args, kwargs))
    with self.subTest('before_registering_tool_handler'):
      self.assertEqual('f_in_tool_handler: a b', result_after)

  @parameterized.named_parameters(
      ('sync', _add_sync),
      ('async', _add_async),
      ('executable', _add_executable),
  )
  def test_run_tool_function_types(self, add_function):
    routing.function_registry['add'] = add_function

    tool_handler = tool_handling.ToolHandler()
    tool_handler.register()
    tool_handler.register_tool(
        tool_handling.ToolSpec(
            name='add',
            description='Tool description.',
            example='Tool example.',
        )
    )

    result = executing.run(
        tool_handler.run_tool(
            tool_name='add', tool_args=['a', 'b'], tool_kwargs={}
        )
    )
    self.assertEqual('ab', result)

  @parameterized.named_parameters(
      ('positional_args_numeric', [1, 2], {}, 3),
      ('keyword_args_numeric', [], {'arg1': 1, 'arg2': 2}, 3),
      ('positional_args_string', ['1', '2'], {}, '12'),
  )
  def test_run_tool_args_types(self, args, kwargs, expected_result):
    def add(arg1: Any, arg2: Any) -> Any:
      return arg1 + arg2

    routing.function_registry['add'] = add
    tool_handler = tool_handling.ToolHandler()
    tool_handler.register()
    tool_handler.register_tool(
        tool_handling.ToolSpec(
            name='Add',
            name_in_registry='add',
            description='Tool description.',
            example='Tool example.',
        )
    )

    result = executing.run(
        tool_handler.run_tool(
            tool_name='Add', tool_args=args, tool_kwargs=kwargs
        )
    )
    self.assertEqual(expected_result, result)

  @parameterized.named_parameters(
      ('example_type_str', 'Search(pi) = "3.14".', 'Search(pi) = "3.14".'),
      (
          'example_type_tool_example',
          tool_handling.ToolExample(
              function_call=tool_handling.FunctionCall(
                  function_name='Search', args=('pi',), kwargs={}
              ),
              response='3.14',
          ),
          "Search('pi')\nwill return: '3.14'",
      ),
      ('example_type_none', None, 'None'),
  )
  def test_tool_spec_example_str(
      self,
      tool_example: str | tool_handling.ToolExample | None,
      expected_example_str: str,
  ):
    tool_spec = tool_handling.ToolSpec(
        name='Search',
        name_in_registry='Search("<query>")',
        description='Description.',
        example=tool_example,
    )
    self.assertEqual(expected_example_str, tool_spec.example_str)


if __name__ == '__main__':
  absltest.main()
