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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import tool_use
from onetwo.core import executing
from onetwo.core import routing


def _add_sync(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


async def _add_async(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


@executing.make_executable
def _add_executable(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


class ToolUseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests routing.function_registry. In case `import routing` is
    # not executed (this may happen when running `pytest` with multiple tests
    # that import `llm` module) the `function_registry` may be already filled
    # with various functions elsewhere in unexpected ways. We manually remove
    # all the keys to make sure it is empty.
    routing.function_registry.clear()
    # Unfortunately, we also removed all the builtins configured when importing
    # tool_use. Let's re-set them.
    # TODO:` or such
    # for better control of reproducibility.
    tool_use.reset_defaults()

  @parameterized.named_parameters(
      ('sync', _add_sync),
      ('async', _add_async),
      ('executable', _add_executable),
  )
  def test_default_run_tool_function_types(self, add_function):
    routing.function_registry['add'] = add_function
    result = executing.run(
        tool_use.run_tool(
            tool_name='add', tool_args=['a', 'b'], tool_kwargs={}
        )
    )
    self.assertEqual('ab', result)

  @parameterized.named_parameters(
      ('positional_args_numeric', [1, 2], {}, 3),
      ('keyword_args_numeric', [], {'arg1': 1, 'arg2': 2}, 3),
      ('positional_args_string', ['1', '2'], {}, '12'),
  )
  def test_default_run_tool_args_types(self, args, kwargs, expected_result):
    def add(arg1: Any, arg2: Any) -> Any:
      return arg1 + arg2

    routing.function_registry['add'] = add
    result = executing.run(
        tool_use.run_tool(
            tool_name='add', tool_args=args, tool_kwargs=kwargs
        )
    )
    self.assertEqual(expected_result, result)

if __name__ == '__main__':
  absltest.main()
