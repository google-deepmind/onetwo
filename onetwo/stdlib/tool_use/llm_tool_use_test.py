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

import pprint
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.core import routing
from onetwo.stdlib.tool_use import llm_tool_use

_ArgumentFormat = llm_tool_use.ArgumentFormat


def _add_sync(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


async def _add_async(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


@executing.make_executable  # pytype: disable=wrong-arg-types
def _add_executable(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


ARG_LIST = (1, 'a', [1, 2], {'a': 1})
ARG_DICT = {
    'a': 1,
    'b': 'f ',
    'c': [1, 2],
    'd': {'a': 1},
    'code': """\
def my_f():
  return 1
""",
}
ARG_LIST_MD = ('1', 'a', '[1, 2]', "{'a': 1}", 'def my_f():\n  return 1')


class LlmToolUseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests routing.function_registry. In case `import routing` is
    # not executed (this may happen when running `pytest` with multiple tests
    # that import `llm` module) the `function_registry` may be already filled
    # with various functions elsewhere in unexpected ways. We manually remove
    # all the keys to make sure it is empty.
    # TODO:` or such
    # for better control of reproducibility.
    routing.function_registry.clear()

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
    rendered = llm_tool_use._render_call_content(fmt, 'run', **kwargs)
    self.assertEqual(
        rendered,
        expected,
        pprint.pformat(rendered) + '\n' + pprint.pformat(expected),
    )

  def test_render_markdown(self):
    code = textwrap.dedent("""\
        def fn(x):
          return x + 1
        result = fn(10)
        """).rstrip()
    rendered = llm_tool_use._render_call_content(
        _ArgumentFormat.MARKDOWN, 'tool_code', code
    )
    self.assertEqual(rendered, f'tool_code\n{code}\n')

  # Only formats that support both positional and keyword arguments
  # (i.e., not `MARKDOWN`).
  @parameterized.named_parameters(
      ('PYTHON', _ArgumentFormat.PYTHON),
      ('YAML', _ArgumentFormat.YAML),
      ('YAML_CODE', _ArgumentFormat.YAML_CODE),
      ('JSON', _ArgumentFormat.JSON),
      ('JSON_SINGLE', _ArgumentFormat.JSON_SINGLE),
  )
  def test_render_call_content_roundtrip(self, fmt):

    rendered_call_content = llm_tool_use._render_call_content(
        fmt, 'f', *ARG_LIST, **ARG_DICT
    )
    (_, fn, args, kwargs) = llm_tool_use._parse_call_content(
        fmt, rendered_call_content, {}
    )

    with self.subTest('parse_call_correctly'):
      self.assertEqual(fn, 'f')
      self.assertEqual(args, ARG_LIST)
      self.assertEqual(kwargs, ARG_DICT)

    re_rendered_call_content = llm_tool_use._render_call_content(
        fmt, fn, *args, **kwargs
    )
    self.assertEqual(rendered_call_content, re_rendered_call_content)

  # Only formats that support both positional and keyword arguments
  # (i.e., not `MARKDOWN`).
  @parameterized.named_parameters(
      ('PYTHON', _ArgumentFormat.PYTHON, _ArgumentFormat.PYTHON),
      ('YAML', _ArgumentFormat.YAML, _ArgumentFormat.YAML),
      ('YAML_CODE', _ArgumentFormat.YAML_CODE, _ArgumentFormat.YAML),
      ('JSON', _ArgumentFormat.JSON, _ArgumentFormat.JSON),
      ('JSON_SINGLE', _ArgumentFormat.JSON_SINGLE, _ArgumentFormat.JSON_SINGLE),
  )
  def test_render_call_roundtrip(self, fmt, parse_by_rendered):

    expected_remainder = '\nSome continuation'
    rendered_call = llm_tool_use.render_call(
        fmt, 'my_function', *ARG_LIST, **ARG_DICT
    )
    rendered_call += expected_remainder
    _, fn, args, kwargs, inferred_fmt, index = (
        llm_tool_use.parse_and_consume_call(rendered_call, context_vars={})
    )
    remainder = rendered_call[index:]
    with self.subTest('parse_and_consume'):
      self.assertEqual(fn, 'my_function')
      self.assertEqual(args, ARG_LIST)
      self.assertEqual(kwargs, ARG_DICT)
      self.assertEqual(remainder, expected_remainder)
      self.assertEqual(inferred_fmt, parse_by_rendered)

    if fmt == inferred_fmt:
      # We only check for formats that can be fully determined automatically.
      re_rendered_call = llm_tool_use.render_call(
          inferred_fmt, fn, *args, **kwargs
      )
      re_rendered_call += expected_remainder
      self.assertEqual(rendered_call, re_rendered_call)

  # Similar to the previous test, but specifically for markdown, which only
  # supports one positional argument.
  @parameterized.parameters(*ARG_LIST_MD)
  def test_render_call_content_roundtrip_markdown(
      self, arg, fmt=_ArgumentFormat.MARKDOWN
  ):

    rendered = llm_tool_use._render_call_content(fmt, 'f', arg)
    (_, fn, roundtrip_arg, _) = llm_tool_use._parse_call_content(
        fmt, rendered, {}
    )

    with self.subTest('parse_call_correctly'):
      self.assertEqual(fn, 'f')
      self.assertEqual(roundtrip_arg, (arg,))

    re_rendered = llm_tool_use._render_call_content(fmt, fn, arg)
    self.assertEqual(rendered, re_rendered)

  @parameterized.parameters(*ARG_LIST_MD)
  def test_render_call_roundtrip_markdown(
      self, arg, fmt=_ArgumentFormat.MARKDOWN
  ):

    expected_remainder = '\nSome continuation'
    rendered = llm_tool_use.render_call(fmt, 'my_function', arg)
    rendered += expected_remainder
    _, fn, args, _, inferred_fmt, index = llm_tool_use.parse_and_consume_call(
        rendered, context_vars={}
    )
    remainder = rendered[index:]
    with self.subTest('parse_and_consume'):
      self.assertEqual(fn, 'my_function')
      self.assertEqual(args, (arg,))
      self.assertEqual(remainder, expected_remainder)
      self.assertEqual(inferred_fmt, _ArgumentFormat.MARKDOWN)

    re_rendered = llm_tool_use.render_call(inferred_fmt, fn, arg)
    re_rendered += expected_remainder
    self.assertEqual(rendered, re_rendered)

  @parameterized.named_parameters(
      ('positional', 'tool_code(1)', (1,), {}),
      ('keyword', 'tool_code(a=1, b="a")', tuple(), {'a': 1, 'b': 'a'}),
  )
  def test_parse_python_format(self, act_text, expected_args, expected_kwargs):
    _, fn, args, kwargs, fmt, index = llm_tool_use.parse_and_consume_call(
        act_text, context_vars={}
    )
    self.assertEqual(fn, 'tool_code')
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
    _, name, args, kwargs, _, _ = llm_tool_use.parse_and_consume_call(
        text=act_text, context_vars=context_vars
    )

    with self.subTest('correct_name'):
      self.assertEqual(expected_name, name)

    with self.subTest('correct_args'):
      self.assertSequenceEqual(expected_args, args)

    with self.subTest('correct_kwargs'):
      self.assertDictEqual(expected_kwargs, kwargs)

  def test_parse_variable_as_arg_not_in_context_fails(self):
    with self.assertRaisesRegex(ValueError, 'does not exist in context'):
      llm_tool_use.parse_and_consume_call(text='Add(a1, a1)', context_vars={})


class ToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests routing.function_registry. In case `import routing` is
    # not executed (this may happen when running `pytest` with multiple tests
    # that import `llm` module) the `function_registry` may be already filled
    # with various functions elsewhere in unexpected ways. We manually remove
    # all the keys to make sure it is empty.
    routing.function_registry.clear()

  @parameterized.named_parameters(
      ('example_type_str', 'Search(pi) = "3.14".', 'Search(pi) = "3.14".'),
      (
          'example_type_tool_example',
          llm_tool_use.ToolExample(
              function_call=llm_tool_use.FunctionCall(
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
      tool_example: str | llm_tool_use.ToolExample | None,
      expected_example_str: str,
  ):
    tool_spec = llm_tool_use.Tool(name='Search', example=tool_example)
    self.assertEqual(expected_example_str, tool_spec.example_str)

  @parameterized.named_parameters(
      ('sync', _add_sync),
      ('async', _add_async),
      ('executable', _add_executable),
  )
  def test_call_tool_via_function_in_tool_spec(self, add_function):
    tool = llm_tool_use.Tool(name='add', function=add_function)

    with self.subTest('positional_args'):
      self.assertEqual(5, executing.run(tool(2, 3)))

    with self.subTest('kwargs_args'):
      self.assertEqual(5, executing.run(tool(arg1=2, arg2=3)))

  @parameterized.named_parameters(
      ('sync', _add_sync),
      ('async', _add_async),
      ('executable', _add_executable),
  )
  def test_call_tool_via_tool_name_and_function_registry(self, add_function):
    routing.function_registry['add'] = add_function
    tool = llm_tool_use.Tool(name='add')

    with self.subTest('positional_args'):
      self.assertEqual(5, executing.run(tool(2, 3)))

    with self.subTest('kwargs_args'):
      self.assertEqual(5, executing.run(tool(arg1=2, arg2=3)))


if __name__ == '__main__':
  absltest.main()
