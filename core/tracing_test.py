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
from collections.abc import AsyncIterator, Iterator, Mapping
import copy
import functools
import io
import pprint
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import results
from onetwo.core import tracing
from onetwo.core import utils


ExecutionResult = results.ExecutionResult


@tracing.trace
def f(a: str) -> dict[str, str]:
  return {'b': a + ' done'}


def g(a: str) -> str:
  f(a + ' other')
  f(a + ' another')
  return a + ' from g'


@tracing.trace
def h(b: str) -> str:
  g(b)
  return g(b + '2')


@tracing.trace
async def af(a: str) -> dict[str, str]:
  await asyncio.sleep(0.1)
  return {'b': a + ' done'}


async def ag(a: str) -> str:
  par = asyncio.gather(af(a + ' other'), af(a + ' another'))
  await par
  return a + ' from g'


@tracing.trace
async def ah(b: str) -> str:
  await ag(b)
  r = await ag(b + '2')
  return r


@tracing.trace
def with_update(a: str) -> str:
  tracing.update_info({'a': a})
  return a


class Test:
  @tracing.trace
  def m(self, a: str) -> str:
    return a + ' done'


class TestStringTracer(tracing.Tracer):
  def __init__(self):
    self.buffer = io.StringIO()
    self.inner_tracer = tracing.StringTracer(self.buffer)

  def get_result(self) -> str:
    return self.buffer.getvalue()

  def add_stage(self) -> tracing.Tracer:
    return self.inner_tracer.add_stage()

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    self.inner_tracer.set_inputs(name, inputs)

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    self.inner_tracer.update_outputs(name, outputs)


class TestExecutionResultTracer(tracing.Tracer):
  def __init__(self):
    self.execution_result = ExecutionResult()
    self.inner_tracer = tracing.ExecutionResultTracer(self.execution_result)

  def get_result(self) -> ExecutionResult:
    return self.execution_result

  def add_stage(self) -> tracing.Tracer:
    return self.inner_tracer.add_stage()

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    self.inner_tracer.set_inputs(name, inputs)

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    self.inner_tracer.update_outputs(name, outputs)


class TracingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_tracer', None, None),
      (
          'string_tracer',
          TestStringTracer(),
          "  ff: {'a': 'test'}\n  ff: {'output': 'other done'}\n",
      ),
      ('execution_result_tracer', TestExecutionResultTracer(), None),
  )
  def test_skip_inputs(self, tracer, expected_result):
    @tracing.trace(skip=['b'])
    def ff(a, b):
      del a
      return b + ' done'

    expected_execution_result = ExecutionResult(
        stage_name='ff',
        inputs={'a': 'test'},
        outputs={results.MAIN_OUTPUT: 'other done'},
        stages=[],
    )

    _, execution_result = tracing.run(
        functools.partial(ff, 'test', 'other'), tracer=tracer
    )
    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          expected_execution_result,
          pprint.pformat(execution_result),
      )
      if tracer is not None:
        result = tracer.get_result()
        if isinstance(tracer, TestExecutionResultTracer):
          expected_result = ExecutionResult(stages=[expected_execution_result])
        self.assertEqual(
            result,
            expected_result,
            pprint.pformat(result),
        )

  def test_change_name(self):
    @tracing.trace(name='f2')
    def ff(a):
      return a + ' done'

    _, execution_result = tracing.run(functools.partial(ff, 'test'))
    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          ExecutionResult(
              stage_name='f2',
              inputs={'a': 'test'},
              outputs={results.MAIN_OUTPUT: 'test done'},
              stages=[],
          ),
          pprint.pformat(execution_result),
      )

  def test_update_info(self):
    _, execution_result = tracing.run(
        functools.partial(with_update, 'test')
    )
    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          ExecutionResult(
              stage_name='with_update',
              inputs={'a': 'test'},
              outputs={results.MAIN_OUTPUT: 'test'},
              info={'a': 'test'},
          ),
          pprint.pformat(execution_result),
      )

  def test_decorate_method(self):
    _, execution_result = tracing.run(functools.partial(Test().m, 'test'))
    with self.subTest('should_not_include_self'):
      self.assertEqual(
          execution_result,
          ExecutionResult(
              stage_name='m',
              inputs={'a': 'test'},
              outputs={results.MAIN_OUTPUT: 'test done'},
              stages=[],
          ),
          pprint.pformat(execution_result),
      )

  @parameterized.named_parameters(
      ('no_tracer', None, None),
      (
          'string_tracer',
          TestStringTracer(),
          (
              "  h: {'b': 'test'}\n"
              "    f: {'a': 'test other'}\n"
              "    f: {'b': 'test other done'}\n"
              "    f: {'a': 'test another'}\n"
              "    f: {'b': 'test another done'}\n"
              "    f: {'a': 'test2 other'}\n"
              "    f: {'b': 'test2 other done'}\n"
              "    f: {'a': 'test2 another'}\n"
              "    f: {'b': 'test2 another done'}\n"
              "  h: {'output': 'test2 from g'}\n"
          ),
      ),
      ('execution_result_tracer', TestExecutionResultTracer(), None),
  )
  def test_tracing(self, tracer, expected_result):
    _, execution_result = tracing.run(
        functools.partial(h, 'test'), tracer=tracer
    )

    expected_execution_result = ExecutionResult(
        stage_name='h',
        inputs={'b': 'test'},
        outputs={results.MAIN_OUTPUT: 'test2 from g'},
        stages=[
            ExecutionResult(
                stage_name='f',
                inputs={'a': 'test other'},
                outputs={'b': 'test other done'},
            ),
            ExecutionResult(
                stage_name='f',
                inputs={'a': 'test another'},
                outputs={'b': 'test another done'},
            ),
            ExecutionResult(
                stage_name='f',
                inputs={'a': 'test2 other'},
                outputs={'b': 'test2 other done'},
            ),
            ExecutionResult(
                stage_name='f',
                inputs={'a': 'test2 another'},
                outputs={'b': 'test2 another done'},
            ),
        ],
    )

    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          expected_execution_result,
          pprint.pformat(execution_result),
      )
      if tracer is not None:
        result = tracer.get_result()
        if isinstance(tracer, TestExecutionResultTracer):
          expected_result = ExecutionResult(stages=[expected_execution_result])
        self.assertEqual(
            result,
            expected_result,
            pprint.pformat(result),
        )

  def test_missing_decorator_tracing(self):
    @tracing.trace
    def f1(a):
      return f2(a)

    def f2(a):
      return f3(a)

    @tracing.trace
    def f3(a):
      return a

    _, execution_result = tracing.run(functools.partial(f1, 'test'))
    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          ExecutionResult(
              stage_name='f1',
              inputs={'a': 'test'},
              outputs={results.MAIN_OUTPUT: 'test'},
              stages=[
                  ExecutionResult(
                      stage_name='f3',
                      inputs={'a': 'test'},
                      outputs={results.MAIN_OUTPUT: 'test'},
                  )
              ],
          ),
          pprint.pformat(execution_result),
      )

  @parameterized.named_parameters(
      ('no_tracer', None, None),
      (
          'string_tracer',
          TestStringTracer(),
          (
              "  ah: {'b': 'test'}\n"
              "    af: {'a': 'test other'}\n"
              "    af: {'a': 'test another'}\n"
              "    af: {'b': 'test other done'}\n"
              "    af: {'b': 'test another done'}\n"
              "    af: {'a': 'test2 other'}\n"
              "    af: {'a': 'test2 another'}\n"
              "    af: {'b': 'test2 other done'}\n"
              "    af: {'b': 'test2 another done'}\n"
              "  ah: {'output': 'test2 from g'}\n"
          ),
      ),
      ('execution_result_tracer', TestExecutionResultTracer(), None),
  )
  def test_async_tracing(self, tracer, expected_result):
    _, execution_result = tracing.run(
        functools.partial(ah, 'test'), tracer=tracer
    )

    expected_execution_result = ExecutionResult(
        stage_name='ah',
        inputs={'b': 'test'},
        outputs={results.MAIN_OUTPUT: 'test2 from g'},
        stages=[
            ExecutionResult(
                stage_name='af',
                inputs={'a': 'test other'},
                outputs={'b': 'test other done'},
            ),
            ExecutionResult(
                stage_name='af',
                inputs={'a': 'test another'},
                outputs={'b': 'test another done'},
            ),
            ExecutionResult(
                stage_name='af',
                inputs={'a': 'test2 other'},
                outputs={'b': 'test2 other done'},
            ),
            ExecutionResult(
                stage_name='af',
                inputs={'a': 'test2 another'},
                outputs={'b': 'test2 another done'},
            ),
        ],
    )

    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          expected_execution_result,
          pprint.pformat(execution_result),
      )
      if tracer is not None:
        result = tracer.get_result()
        if isinstance(tracer, TestExecutionResultTracer):
          expected_result = ExecutionResult(stages=[expected_execution_result])
        self.assertEqual(
            result,
            expected_result,
            pprint.pformat(result),
        )

  def test_async_iterator(self):
    @tracing.trace
    async def fn(a: str) -> AsyncIterator[str]:
      for i in range(len(a)):
        yield a[: i + 1]

    trace = []

    async def wrapper():
      nonlocal trace
      tracing.execution_context.set(None)
      async for _ in fn('test'):
        trace.append(copy.deepcopy(tracing.execution_context.get(None)))

    expected_trace = [
        ExecutionResult(
            stage_name='fn',
            inputs={'a': 'test'},
            outputs={'output': 't'},
            stages=[],
        ),
        ExecutionResult(
            stage_name='fn',
            inputs={'a': 'test'},
            outputs={'output': 'te'},
            stages=[],
        ),
        ExecutionResult(
            stage_name='fn',
            inputs={'a': 'test'},
            outputs={'output': 'tes'},
            stages=[],
        ),
        ExecutionResult(
            stage_name='fn',
            inputs={'a': 'test'},
            outputs={'output': 'test'},
            stages=[],
        ),
    ]

    asyncio.run(wrapper())
    with self.subTest('should_return_correct_results'):
      self.assertListEqual(trace, expected_trace, pprint.pformat(trace))

  def test_par_tracing(self):
    @tracing.trace
    async def f1(a):
      return await asyncio.gather(f2(a + '1'), f2(a + '2'))

    async def f2(a):
      return await asyncio.gather(f3(a + '1'), f3(a + '2'))

    @tracing.trace
    async def f3(a):
      return a + ' done'

    _, execution_result = tracing.run(functools.partial(f1, 'test'))

    with self.subTest('should_return_correct_results'):
      self.assertEqual(
          execution_result,
          ExecutionResult(
              stage_name='f1',
              inputs={'a': 'test'},
              outputs={
                  results.MAIN_OUTPUT: [
                      ['test11 done', 'test12 done'],
                      ['test21 done', 'test22 done'],
                  ]
              },
              stages=[
                  ExecutionResult(
                      stage_name='f3',
                      inputs={'a': 'test11'},
                      outputs={results.MAIN_OUTPUT: 'test11 done'},
                  ),
                  ExecutionResult(
                      stage_name='f3',
                      inputs={'a': 'test12'},
                      outputs={results.MAIN_OUTPUT: 'test12 done'},
                  ),
                  ExecutionResult(
                      stage_name='f3',
                      inputs={'a': 'test21'},
                      outputs={results.MAIN_OUTPUT: 'test21 done'},
                  ),
                  ExecutionResult(
                      stage_name='f3',
                      inputs={'a': 'test22'},
                      outputs={results.MAIN_OUTPUT: 'test22 done'},
                  ),
              ],
          ),
          pprint.pformat(execution_result),
      )

  def test_from_instance_name(self):
    class C:
      def __init__(self, name: str):
        self.name = name

      @tracing.trace(name=utils.FromInstance('name'))
      def f(self, x: int) -> int:
        return x + 1

    c = C('test')
    _, execution_result = tracing.run(functools.partial(c.f, 1))
    expected_execution_result = ExecutionResult(
        stage_name='test',
        inputs={'x': 1},
        outputs={results.MAIN_OUTPUT: 2},
    )
    self.assertEqual(
        execution_result,
        expected_execution_result,
        pprint.pformat(execution_result),
    )

  def test_queue_trace(self):

    class TestTracer(tracing.QueueTracer):
      def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
        self.callback(f'{2-self.depth} - {name}: {inputs}')

      def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
        self.callback(f'{2-self.depth} - {name}: {outputs}')

    @tracing.trace
    def f0(a: str) -> str:
      return a + ' done'

    @tracing.trace
    def f1(a: int) -> Iterator[str]:
      for i in range(a):
        yield str(i)

    @tracing.trace
    async def f2(a: str) -> str:
      return f0(a) + ' ' + f0(a + '2') + str(list(f1(3)))

    updates = []
    iterator = tracing.stream(f2('test'), TestTracer)
    for update in iterator:
      updates.append(update)
    result = iterator.value

    with self.subTest('should_return_correct_updates'):
      self.assertListEqual(
          updates,
          [
              "1 - f2: {'a': 'test'}",
              "2 - f0: {'a': 'test'}",
              "2 - f0: {'output': 'test done'}",
              "2 - f0: {'a': 'test2'}",
              "2 - f0: {'output': 'test2 done'}",
              "2 - f1: {'a': 3}",
              "2 - f1: {'output': '0'}",
              "2 - f1: {'output': '1'}",
              "2 - f1: {'output': '2'}",
              "1 - f2: {'output': \"test done test2 done['0', '1', '2']\"}",
          ],
      )

    with self.subTest('should_return_correct_result'):
      self.assertEqual(result, "test done test2 done['0', '1', '2']")

  def test_queue_trace_errors(self):

    class TestTracer(tracing.QueueTracer):
      def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
        self.callback(f'{2-self.depth} - {name}: {inputs}')

      def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
        self.callback(f'{2-self.depth} - {name}: {outputs}')

    @tracing.trace
    async def fn(a: str) -> str:
      return a

    updates = []
    iterator = tracing.stream(fn('test'), TestTracer)
    for update in iterator:
      updates.append(update)
    result = iterator.value

    with self.subTest('should_return_correct_updates'):
      self.assertListEqual(
          updates, ["1 - fn: {'a': 'test'}", "1 - fn: {'output': 'test'}"]
      )

    with self.subTest('should_return_correct_result'):
      self.assertEqual(result, 'test')

    updates = []
    with self.subTest('should_raise_error'):
      with self.assertRaises(ValueError):
        iterator = tracing.stream(fn('test'), TestTracer)
        for update in iterator:
          updates.append(update)
          raise ValueError()
      self.assertListEqual(
          updates,
          [
              "1 - fn: {'a': 'test'}",
          ],
      )

    with self.subTest('should_continue'):
      for update in iterator:
        updates.append(update)
      self.assertListEqual(
          updates,
          [
              "1 - fn: {'a': 'test'}",
              "1 - fn: {'output': 'test'}",
          ],
      )

  @parameterized.named_parameters(
      ('depth_0', 0, []),
      ('depth_1', 1, ['test', "['0 done', '1 done', '2 done']"]),
      (
          'depth_2',
          2,
          [
              'test',
              'update0',
              '0 done',
              'update1',
              '1 done',
              'update2',
              '2 done',
              "['0 done', '1 done', '2 done']",
          ],
      ),
      (
          'depth_minus_1',
          -1,
          [
              'test',  # From f2 report_update.
              'update0',  # From f1 report_update.
              'f00',  # From f0 report_update.
              '0 done',  # From f0 return.
              '0 done',  # From f1 yield.
              'update1',  # From f1 report_update.
              'f01',  # From f0 report_update.
              '1 done',  # From f0 return.
              '1 done',  # From f1 yield.
              'update2',  # From f1 report_update.
              'f02',  # From f0 report_update.
              '2 done',  # From f0 return.
              '2 done',  # From f1 yield.
              "['0 done', '1 done', '2 done']",  # From f2 return.
          ],
      ),
  )
  def test_stream_updates(self, depth: int, expected_updates: list[str]):
    @tracing.trace
    async def f0(a: str) -> str:
      await tracing.report_update('f0' + a)
      return a + ' done'

    @tracing.trace
    async def f1(a: int) -> AsyncIterator[str]:
      for i in range(a):
        await tracing.report_update('update' + str(i))
        yield (await f0(str(i)))

    @tracing.trace
    async def f2(a: str) -> str:
      await tracing.report_update(a)
      res = []
      async for r in f1(3):
        res.append(r)
      return str(res)

    updates = []
    iterator = tracing.stream_updates(f2('test'), iteration_depth=depth)
    for update in iterator:
      updates.append(update)
    result = iterator.value

    with self.subTest('should_return_correct_updates'):
      self.assertListEqual(
          updates, expected_updates
      )

    with self.subTest('should_return_correct_result'):
      self.assertEqual(result, "['0 done', '1 done', '2 done']")


if __name__ == '__main__':
  absltest.main()
