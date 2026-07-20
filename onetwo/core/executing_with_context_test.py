# Copyright 2026 DeepMind Technologies Limited.
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

from __future__ import annotations
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.core import executing_with_context
from onetwo.core import updating
from typing_extensions import override


def _collect_stream(
    executable: executing.Executable,
    depth: int = 1,
) -> list[str]:
  res = []
  with executing.safe_stream(executable, iteration_depth=depth) as stream:
    updates = updating.Update[str]()
    for update in stream:
      updates += update
      result = updates.to_result()
      assert isinstance(result, str) or isinstance(result, list)
      res.append(result)
  return res


@dataclasses.dataclass
class ContextForTest:
  content: str = dataclasses.field(default_factory=str)


@dataclasses.dataclass
class ExecutableWithContextForTest(
    executing_with_context.ExecutableWithContext[ContextForTest, str]
):
  content: str = dataclasses.field(default_factory=str)

  @override
  def initialize_context(self, *args, **kwargs) -> ContextForTest:
    return ContextForTest(*args)

  @classmethod
  @override
  def wrap(cls, other: str) -> ExecutableWithContextForTest:
    return ExecutableWithContextForTest(content=other)

  @classmethod
  @override
  def get_result(cls, context: ContextForTest) -> str:
    return context.content

  @override
  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def execute(
      self,
      context: ContextForTest,
  ) -> str:
    # Add the content of this node to the context.
    context.content += self.content
    # Return the content between quotes.
    return f'"{self.content}"'


@dataclasses.dataclass
class SerialContextForTest:
  content: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SerialExecutableWithContextForTest(
    executing_with_context.SerialExecutableWithContext[  # pyrefly: ignore[bad-specialization]
        SerialContextForTest, list[str]
    ]
):

  @override
  @classmethod
  def empty_result(cls) -> list[str]:
    return []

  @override
  def initialize_context(self, *args, **kwargs) -> SerialContextForTest:
    return SerialContextForTest(list(*args))

  @classmethod
  @override
  def wrap(cls, other: str) -> SingleStepSerialExecutableWithContextForTest:  # pyrefly: ignore[bad-override]
    return SingleStepSerialExecutableWithContextForTest(content=other)

  @classmethod
  @override
  def get_result(cls, context: SerialContextForTest) -> str:  # pyrefly: ignore[bad-override]
    return ''.join(context.content)


@dataclasses.dataclass
class SingleStepSerialExecutableWithContextForTest(
    SerialExecutableWithContextForTest
):
  content: str = dataclasses.field(default_factory=str)

  @override
  @executing.make_executable  # pytype: disable=wrong-arg-types
  def execute(self, context: SerialContextForTest) -> list[str]:
    # Add the content of this node to the context.
    context.content.append(self.content)
    # Return the content between quotes.
    return [f'"{self.content}"']

  @override
  @executing.make_executable  # pytype: disable=wrong-arg-types
  def iterate(
      self, context: SerialContextForTest, iteration_depth: int = 1
  ) -> list[str]:
    del iteration_depth
    # Add the content of this node to the context.
    context.content.append(self.content)
    # Return the content between quotes.
    return [f'"{self.content}"']


class ExecutingWithContextTest(parameterized.TestCase):

  def test_execute_simple(self):
    e = ExecutableWithContextForTest(content='hello')
    with self.subTest('run_directly'):
      res = executing.run(e)
      self.assertEqual(res, 'hello')

    with self.subTest('result_has_quotes'):
      res = executing.run(e.execute(ContextForTest()))  # pyrefly: ignore[bad-argument-count, no-matching-overload]
      self.assertEqual(res, '"hello"')

    with self.subTest('run_with_arguments'):
      res = executing.run(e('prefix '))
      self.assertEqual(res, 'prefix hello')

    with self.subTest('run_with_resetting'):
      res = executing.run(e())
      self.assertEqual(res, 'hello')

    with self.subTest('stream'):
      res = _collect_stream(e())
      self.assertListEqual(res, ['hello'])

    with self.subTest('stream_iterate'):
      res = _collect_stream(e.iterate(ContextForTest()))  # pyrefly: ignore[bad-argument-count, bad-argument-type]
      self.assertListEqual(res, ['"hello"'])

  def test_execute_serial(self):
    e = SerialExecutableWithContextForTest()
    e += 'hello '
    e += 'world'

    with self.subTest('run_directly'):
      res = executing.run(e)
      self.assertEqual(res, 'hello world')

    with self.subTest('result_is_a_list_with_quotes'):
      res = executing.run(e.execute(SerialContextForTest()))  # pyrefly: ignore[bad-argument-count, no-matching-overload]
      self.assertListEqual(res, ['"hello "', '"world"'])

    with self.subTest('stored_result_is_correct_after_run'):
      _ = executing.run(e)
      result_field = e._result
      assert isinstance(result_field, list)
      self.assertListEqual(result_field, ['"hello "', '"world"'])

    with self.subTest('stored_result_is_correct_after_stream'):
      _ = _collect_stream(e())
      self.assertListEqual(e._result, ['"hello "', '"world"'])  # pyrefly: ignore[bad-argument-type]

    with self.subTest('run_with_arguments'):
      res = executing.run(e(['prefix ']))
      self.assertEqual(res, 'prefix hello world')

    with self.subTest('result_with_arguments'):
      res = executing.run(e.execute(SerialContextForTest(['prefix '])))  # pyrefly: ignore[bad-argument-count, no-matching-overload]
      self.assertEqual(res, ['"hello "', '"world"'])

    with self.subTest('run_with_resetting'):
      res = executing.run(e())
      self.assertEqual(res, 'hello world')

    with self.subTest('stream_iterate_returns_lists'):
      res = _collect_stream(e.iterate(SerialContextForTest()))  # pyrefly: ignore[bad-argument-count, bad-argument-type]
      self.assertListEqual(res, [['"hello "'], ['"hello "', '"world"']])

  def test_right_addition(self):
    e = 'hello '
    e += SerialExecutableWithContextForTest()
    e += 'world'

    with self.subTest('run_directly'):
      res = executing.run(e)
      self.assertEqual(res, 'hello world')

    with self.subTest('result_is_a_list_with_quotes'):
      res = executing.run(e.execute(SerialContextForTest()))  # pyrefly: ignore[bad-argument-count, no-matching-overload]
      self.assertListEqual(res, ['"hello "', '"world"'])

    with self.subTest('run_with_arguments'):
      res = executing.run(e(['prefix ']))
      self.assertEqual(res, 'prefix hello world')

    with self.subTest('run_with_resetting'):
      res = executing.run(e())
      self.assertEqual(res, 'hello world')

    with self.subTest('stream_iterate'):
      res = _collect_stream(e.iterate(SerialContextForTest()))  # pyrefly: ignore[bad-argument-count, bad-argument-type]
      self.assertListEqual(res, [['"hello "'], ['"hello "', '"world"']])


if __name__ == '__main__':
  absltest.main()
