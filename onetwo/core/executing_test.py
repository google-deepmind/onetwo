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

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
import copy
import pprint
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import updating


# We define a bunch of type shortcuts for convenience.
ExecutableWithStr = executing.Executable[str]
make_executable = executing.make_executable
parallel = executing.parallel
par_iter = executing.par_iter
serial = executing.serial


class IterableReply(executing.ExecutableWithCallback[str]):
  def __init__(self, text: str):
    self.text = text

  async def _iterate(self) -> AsyncIterator[updating.Update[str]]:
    for i in range(1, len(self.text) + 1):
      yield updating.Update(self.text[:i])


class UncopiableObject():
  def __init__(self, text: str):
    self.text = text

  def __copy__(self):
    raise NotImplementedError('UncopiableObject is not copyable.')

  def __deepcopy__(self, memo):
    raise NotImplementedError('UncopiableObject is not deepcopyable.')


def _wrap_request(request: str, streaming: bool = False) -> ExecutableWithStr:
  if streaming:
    return process_stream(request)
  else:
    return process(request)


@make_executable
def process(request: str) -> str:
  return request.replace('req', 'rep')


@make_executable
def process_stream(request: str) -> IterableReply:
  reply = request.replace('req', 'rep')
  return IterableReply(reply)


@make_executable
async def process_updates(request: str) -> str:
  reply = request.replace('req', 'rep')
  for i in range(1, len(reply) + 1):
    await tracing.report_update(reply[:i])
  return reply


# An arbitrary function can be turned into an executable.
@make_executable
def my_function(text: str) -> str:
  return f'done: {text}'


@make_executable
async def async_wrap_function_without_call(text: str) -> str:
  return await my_function(text)


@make_executable
async def async_wrap_function_multiple_args(text1: str, text2: str) -> str:
  res1 = await my_function(text1)
  res2 = await my_function(text2)
  return f'{res1} {res2}'


@make_executable
def list_function(texts):
  return f'done: {texts[0]}'


@make_executable(non_copied_args=['uncopiable'])
def my_function_with_non_copied_args(
    texts: list[str], uncopiable: UncopiableObject
) -> str:
  return f'done: {texts[0]} with {uncopiable.text}'


@make_executable('other')
async def my_function_with_input(
    text: str, other: ExecutableWithStr
) -> str:
  other_result = await other
  return f'done: {text} with {other_result}'


def my_function_with_two_args(
    one: ExecutableWithStr,
    two: ExecutableWithStr,
) -> Any:
  return one, two


def _process(request: str) -> str:
  return request.replace('req', 'rep')


def _process_stream(request: str) -> IterableReply:
  reply = request.replace('req', 'rep')
  return IterableReply(reply)


class CallRecorder:
  def __init__(self, use_threads: bool = False, streaming: bool = False):
    self.calls = []
    self.use_threads = use_threads
    self.streaming = streaming

  def process(self, request: str) -> ExecutableWithStr:
    return self._process(request)

  @make_executable
  async def _process(self, request: str) -> str | IterableReply:
    self.calls.append(request)
    fn = _process if self.streaming else _process_stream
    if self.use_threads:
      return await asyncio.to_thread(fn, request)
    else:
      return fn(request)


class TestObject:

  def __init__(self, some_list: list[str]):
    self.some_list = some_list

  def not_decorated_method(self, req):
    return self.some_list[0] + req

  @make_executable
  def method(self, req):
    return self.some_list[0] + req

  @make_executable(copy_self=False)
  def method_without_copy(self, req):
    return self.some_list[0] + req

  @make_executable
  async def async_method(self, req):
    return await self.method(req)

  @make_executable
  async def async_method_without_call(self, req):
    return await self.method(req)

  @make_executable
  def list_method(self, texts):
    return self.some_list[0] + texts[0]

  @make_executable(non_copied_args=['uncopiable'])
  def method_with_non_copied_args(
      self, texts: list[str], uncopiable: UncopiableObject
  ) -> str:
    return self.some_list[0] + texts[0] + uncopiable.text


class ExecutionTest(parameterized.TestCase):

  def test_executable_object(self):
    """Tests the subclassing of Executable."""

    class TestExecutable(ExecutableWithStr):
      """Simple Executable issuing a sequence of requests."""

      def __init__(self, requests: list[str]):
        self.requests = requests

      @tracing.trace
      def process(self, request: str) -> str:
        return request.replace('req', 'rep')

      async def _aexec(self) -> Sequence[str]:
        """Execute all requests."""
        return [self.process(request) for request in self.requests]

      async def _aiterate(self, iteration_depth: int = 1) -> AsyncIterator[Any]:
        del iteration_depth
        for request in self.requests:
          yield self.process(request)

    e = TestExecutable(requests=['req1', 'req2', 'req3'])
    expected_results = ['rep1', 'rep2', 'rep3']

    run_results = executing.run(e)

    with self.subTest('run'):
      self.assertEqual(
          run_results, expected_results, msg=pprint.pformat(run_results)
      )

    with executing.stream_updates(e) as iterator:
      stream_results = list(iterator)
    with self.subTest('stream'):
      self.assertEqual(
          stream_results, expected_results, msg=pprint.pformat(stream_results)
      )

    with executing.safe_stream(e) as iterator:
      stream_results = list(iterator)
    with self.subTest('stream'):
      self.assertEqual(
          stream_results, expected_results, msg=pprint.pformat(stream_results)
      )

    results = []

    def cb(res) -> None:
      nonlocal results
      results.append(res)

    final_result = executing.stream_with_callback(e, cb)
    with self.subTest('stream_with_callback_list'):
      self.assertEqual(
          results, expected_results, msg=pprint.pformat(results)
      )

    with self.subTest('stream_with_callback_final'):
      self.assertEqual(
          final_result, expected_results[-1], msg=pprint.pformat(final_result)
      )

  def test_repeated_decoration(self):
    decorated_fn_one = make_executable('one')(my_function_with_two_args)
    decorated_fn_two = make_executable('two')(decorated_fn_one)
    first_exec_arg = my_function('1')
    second_exec_arg = my_function('2')
    tuple_one = executing.run(
        decorated_fn_one(first_exec_arg, second_exec_arg)
    )
    tuple_two = executing.run(
        decorated_fn_two(first_exec_arg, second_exec_arg)
    )
    with self.subTest('repeated_decoration_is_a_no_op'):
      self.assertEqual(tuple_one, tuple_two)

  def test_make_executable_functions(self):
    """Tests the make_executable decorator on various function types."""
    # Whether we decorate a normal function, a bound method, an async function,
    # an iterator, or an async iterator, they should all be executable with run
    # or stream.
    @make_executable
    def test_fn(text: str) -> str:
      return f'done: {text}'

    @make_executable
    async def async_test_fn(text: str) -> str:
      return await test_fn(text)  # tn should be awaitable

    @make_executable
    @tracing.trace
    def test_iterator(text: str) -> Iterator[str]:
      for i in range(1, len(text) + 1):
        res = 'done: ' + text[:i]
        yield res

    @make_executable
    async def async_test_iterator(text: str) -> AsyncIterator[str]:
      async for s in test_iterator(text):  # test_iterator should be async. pylint: disable=not-an-iterable
        yield s

    bound_fn = make_executable(TestObject(['done: ']).not_decorated_method)

    cb_result = []
    def cb(res) -> None:
      nonlocal cb_result
      cb_result.append(res)

    for fn, is_iterative in [
        (test_fn, False),
        (bound_fn, False),
        (async_test_fn, False),
        (test_iterator, True),
        (async_test_iterator, True),
    ]:
      with self.subTest(f'run_{fn.__name__}'):
        result = executing.run(fn('test'))
        expected_result = 'done: test'
        self.assertEqual(result, expected_result, msg=pprint.pformat(result))

      with self.subTest(f'safe_stream_{fn.__name__}'):
        with executing.safe_stream(fn('test')) as iterator:
          result = list(iterator)
        expected_result = [
            'done: t',
            'done: te',
            'done: tes',
            'done: test',
        ]
        self.assertEqual(
            result,
            expected_result if is_iterative else [expected_result[-1]],
            msg=pprint.pformat(result),
        )

      with self.subTest(f'stream_updates_{fn.__name__}'):
        with executing.stream_updates(fn('test')) as iterator:
          result = list(iterator)
        expected_result = [
            'done: t',
            'done: te',
            'done: tes',
            'done: test',
        ]
        result = result if is_iterative else iterator.value
        self.assertEqual(
            result,
            expected_result if is_iterative else expected_result[-1],
            msg=pprint.pformat(result),
        )

      with self.subTest(f'stream_with_callback_{fn.__name__}'):
        cb_result = []
        final_result = executing.stream_with_callback(fn('test'), cb)
        expected_result = [
            'done: t',
            'done: te',
            'done: tes',
            'done: test',
        ]
        expected_final_result = expected_result[-1]
        self.assertEqual(
            cb_result,
            expected_result if is_iterative else [expected_final_result],
            msg=pprint.pformat(cb_result),
        )
        self.assertEqual(
            final_result,
            expected_final_result,
            msg=pprint.pformat(final_result),
        )

  def test_methods_are_executable(self):
    inner_text = ['req1']
    obj = TestObject(inner_text)
    result1 = executing.run(obj.method('_0'))
    result2 = executing.run(obj.async_method('_1'))

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(result1, 'req1_0')
      self.assertEqual(result2, 'req1_1')

  def test_method_executable_copy_when_decorated_with_atsign(self):
    inner_text = ['req1']
    obj = TestObject(inner_text)
    obj2 = copy.deepcopy(obj)
    executable = obj.method('_0')
    executable2 = copy.deepcopy(executable)
    executable3 = obj.method_without_copy('_0')
    executable4 = copy.deepcopy(executable3)
    inner_text[0] = 'req2'

    with self.subTest('copy_is_deep_on_object'):
      self.assertEqual(obj.some_list, ['req2'])
      self.assertEqual(obj2.some_list, ['req1'])

    with self.subTest('copy_is_deep_on_executable'):
      self.assertEqual(executing.run(executable), 'req2_0')
      self.assertEqual(executing.run(executable2), 'req1_0')

    with self.subTest('copy_is_deep_on_self'):
      self.assertNotEqual(executable.args[0], executable2.args[0])

    with self.subTest('copy_is_not_deep_with_self'):
      self.assertEqual(executing.run(executable3), 'req2_0')
      self.assertEqual(executing.run(executable4), 'req2_0')
      self.assertEqual(executable3.args[0], executable4.args[0])

  def test_method_executable_copy_when_decorated_inline(self):
    inner_text = ['req1']
    obj = TestObject(inner_text)
    executable = make_executable(obj.not_decorated_method)('_0')
    executable2 = copy.deepcopy(executable)
    executable3 = make_executable(copy_self=False)(obj.not_decorated_method)(
        '_0'
    )
    executable4 = copy.deepcopy(executable3)
    inner_text[0] = 'req2'

    with self.subTest('copy_is_deep_on_executable'):
      self.assertEqual(executing.run(executable), 'req2_0')
      self.assertEqual(executing.run(executable2), 'req1_0')

    with self.subTest('copy_is_deep_on_self'):
      self.assertNotEqual(executable.args[0], executable2.args[0])

    with self.subTest('copy_is_not_deep_with_self'):
      self.assertEqual(executing.run(executable3), 'req2_0')
      self.assertEqual(executing.run(executable4), 'req2_0')
      self.assertEqual(executable3.args[0], executable4.args[0])

  def test_method_args_copy(self):
    obj = TestObject(['req'])
    texts = ['_0']
    executable = obj.list_method(texts)
    executable2 = copy.deepcopy(executable)
    texts[0] = '_1'

    with self.subTest('arguments_are_copied'):
      self.assertEqual(executing.run(executable), 'req_1')
      self.assertEqual(executing.run(executable2), 'req_0')

  def test_function_args_copy(self):
    texts = ['r0']
    executable = list_function(texts)
    executable2 = copy.deepcopy(executable)
    texts[0] = 'r1'

    with self.subTest('arguments_are_copied'):
      self.assertEqual(executing.run(executable), 'done: r1')
      self.assertEqual(executing.run(executable2), 'done: r0')

  def test_method_args_copy_with_non_copied_args(self):
    obj = TestObject(['req'])
    texts = ['_0']
    uncopiable = UncopiableObject('_u0')
    executable = obj.method_with_non_copied_args(texts, uncopiable=uncopiable)
    executable2 = copy.deepcopy(executable)
    texts[0] = '_1'
    uncopiable.text = '_u1'

    with self.subTest('non_copied_args_are_not_copied'):
      # The original `executable` is the one whose args we modified directly,
      # so we see the updated values for all args.
      self.assertEqual(executing.run(executable), 'req_1_u1')
      # The copied `executable2` shows the original value of the normal arg,
      # but the updated value of the uncopied one
      self.assertEqual(executing.run(executable2), 'req_0_u1')

  def test_function_args_copy_with_non_copied_args(self):
    texts = ['r0']
    uncopiable = UncopiableObject('u0')
    executable = my_function_with_non_copied_args(
        texts, uncopiable=uncopiable
    )
    executable2 = copy.deepcopy(executable)
    texts[0] = 'r1'
    uncopiable.text = 'u1'

    with self.subTest('non_copied_args_are_not_copied_but_others_are'):
      # The original `executable` is the one whose args we modified directly,
      # so we see the updated values for all args.
      self.assertEqual(executing.run(executable), 'done: r1 with u1')
      # The copied `executable2` shows the original value of the normal arg,
      # but the updated value of the uncopied one
      self.assertEqual(executing.run(executable2), 'done: r0 with u1')

  def test_serial_streaming(self):
    executable = serial(
        _wrap_request('req_0'), _wrap_request('req_1')
    )
    with executing.stream_updates(executable) as iterator:
      result = sum(iterator, start=updating.Update()).to_result()
    self.assertListEqual(result, ['rep_0', 'rep_1'])

  def test_serial_streaming_with_callback(self):
    executable = serial(
        _wrap_request('req_0'), _wrap_request('req_1')
    )
    updates = executing.Update()

    def cb(update: executing.Update) -> None:
      nonlocal updates
      updates += update

    returned_result = executing.stream_with_callback(executable, cb)
    result = updates.to_result()
    self.assertEqual(returned_result, result)
    self.assertListEqual(result, ['rep_0', 'rep_1'])

  def test_no_need_to_call_wrapped_functions(self):
    reply = executing.run(async_wrap_function_without_call('r'))

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(reply, 'done: r')

  def test_no_need_to_call_methods(self):
    inner_text = ['req1']
    obj = TestObject(inner_text)
    result1 = executing.run(obj.method('_0'))
    result2 = executing.run(obj.async_method_without_call('_1'))

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(result1, 'req1_0')
      self.assertEqual(result2, 'req1_1')

  def test_wrap_with_executable_parameter(self):
    @make_executable
    def pass_through(text: str) -> str:
      return text

    reply1 = executing.run(my_function_with_input(
        pass_through('r'), my_function('s')
    ))
    reply2 = executing.run(my_function_with_input('r', pass_through('s')))
    reply3 = executing.run(my_function_with_input('r', my_function('s')))
    reply4 = executing.run(my_function_with_input(
        my_function('r'), my_function_with_input('s', pass_through('t'))
    ))

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(reply1, 'done: r with done: s')
      self.assertEqual(reply2, 'done: r with s')
      self.assertEqual(reply3, 'done: r with done: s')
      self.assertEqual(reply4, 'done: done: r with done: s with t')

  def test_multiple_arguments(self):
    res1 = executing.run(async_wrap_function_multiple_args('r', 's'))
    res2 = executing.run(async_wrap_function_multiple_args(
        _wrap_request('r_0'), 's'
    ))
    res3 = executing.run(async_wrap_function_multiple_args(
        'r', _wrap_request('s_0')
    ))
    res4 = executing.run(async_wrap_function_multiple_args(
        _wrap_request('r_0'), _wrap_request('s_0')
    ))

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(res1, 'done: r done: s')
      self.assertEqual(res2, 'done: r_0 done: s')
      self.assertEqual(res3, 'done: r done: s_0')
      self.assertEqual(res4, 'done: r_0 done: s_0')

  @parameterized.named_parameters(
      (
          'use_threads',
          True,
          ['req_0', 'req_3', 'req_1', 'req_4', 'req_2', 'req_5'],
      ),
      (
          'no_threads',
          False,
          ['req_0', 'req_3', 'req_1', 'req_4', 'req_2', 'req_5'],
      ),
  )
  def test_serial_parallel_run(self, use_threads, expected_calls):
    """Test the interleaving of two parallel series of calls."""
    recorder = CallRecorder(use_threads=use_threads)
    # We execute two sequences of requests in parallel.
    # If we don't use threads, since asyncio uses only one thread, and
    # we are prioritizing executing the tasks in the order they are created
    # there will be no interleaving.
    # If we use threads, this means the execution can really be parallelized
    # and the calls will then be interleaved.
    executable1 = serial(*[recorder.process(f'req_{i}') for i in range(0, 3)])
    executable2 = serial(*[recorder.process(f'req_{i}') for i in range(3, 6)])
    expected_results = [
        ['rep_0', 'rep_1', 'rep_2'],
        ['rep_3', 'rep_4', 'rep_5'],
    ]
    results = executing.run(parallel(executable1, executable2))

    with self.subTest('should_order_calls'):
      self.assertEqual(
          expected_calls,
          recorder.calls,
          pprint.pformat(recorder.calls),
      )

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(expected_results, results)

  def test_par_iter(self):
    executable = par_iter((
        _wrap_request('req_0'),
        _wrap_request('req_1'),
        _wrap_request('req_2'),
    ))
    with executing.stream_updates(executable) as iterator:
      result = sum(iterator, start=updating.Update()).to_result()
    result2 = iterator.value

    with self.subTest('should_produce_correct_results'):
      self.assertEqual(result, ['rep_0', 'rep_1', 'rep_2'])
      self.assertEqual(result2, result)

  def test_serial_parallel_stream(self):
    """Test the interleaving of two parallel series of calls."""
    recorder = CallRecorder(use_threads=False)
    # Since we stream with iteration_depth=2 the calls will be interleaved.
    executable1 = serial(*[recorder.process(f'req_{i}') for i in range(0, 3)])
    executable2 = serial(*[recorder.process(f'req_{i}') for i in range(3, 6)])
    executable = parallel(executable1, executable2)
    expected_results = [
        ['rep_0', 'rep_1', 'rep_2'],
        ['rep_3', 'rep_4', 'rep_5'],
    ]
    expected_calls = ['req_0', 'req_3', 'req_1', 'req_4', 'req_2', 'req_5']
    expected_partial_results = [
        [['rep_0']],
        [['rep_0'], ['rep_3']],
        [['rep_0', 'rep_1'], ['rep_3']],
        [['rep_0', 'rep_1'], ['rep_3', 'rep_4']],
        [['rep_0', 'rep_1', 'rep_2'], ['rep_3', 'rep_4']],
        [['rep_0', 'rep_1', 'rep_2'], ['rep_3', 'rep_4', 'rep_5']],
    ]
    with self.subTest('safe_stream'):
      partial_results = []
      updates = executing.Update()
      with executing.safe_stream(executable, iteration_depth=2) as iterator:
        for update in iterator:
          updates += update
          partial_results.append(updates.to_result())
      results = updates.to_result()

      with self.subTest('should_order_calls'):
        self.assertEqual(
            expected_calls,
            recorder.calls,
            pprint.pformat(recorder.calls),
        )

      with self.subTest('should_produce_correct_results'):
        self.assertEqual(expected_results, results)

      with self.subTest('should_produce_partial_results'):
        self.assertEqual(expected_partial_results, partial_results)

    with self.subTest('stream_updates'):
      executable1 = serial(*[process(f'req_{i}') for i in range(0, 3)])
      executable2 = serial(*[process(f'req_{i}') for i in range(3, 6)])
      executable = parallel(executable1, executable2)
      partial_results = []
      updates = executing.Update()
      with executing.stream_updates(executable, iteration_depth=1) as iterator:
        for update in iterator:
          updates += update
          partial_results.append(updates.to_result())
        results = iterator.value

      with self.subTest('should_produce_correct_results'):
        self.assertEqual(expected_results, results)

      with self.subTest('should_produce_partial_results'):
        expected_partial_results = [
            [['rep_0', 'rep_1', 'rep_2']],
            [['rep_0', 'rep_1', 'rep_2'], ['rep_3', 'rep_4', 'rep_5']],
        ]
        self.assertListEqual(
            expected_partial_results,
            partial_results,
            pprint.pformat(partial_results),
        )

  def test_no_calls(self):
    @executing.make_executable
    def fn1() -> str:
      return 'test'

    @executing.make_executable
    def fn2(inputs: str) -> str:
      return inputs + ' done'

    executable = parallel(fn1(), fn2(fn1()), fn2('other'))
    self.assertEqual(
        executing.run(executable), ['test', 'test done', 'other done']
    )

  def test_corner_cases(self):
    @executing.make_executable
    def fn1() -> str:
      return 'test'

    @executing.make_executable
    def fn2(inputs: list[str]) -> str:
      if inputs:
        return inputs[0] + ' done'
      else:
        return 'done'

    executable = parallel(fn1())
    executable = fn2(executable)
    with self.subTest('parallel_of_one'):
      self.assertEqual(
          executing.run(executable), 'test done'
      )

    executable = par_iter([])

    with self.subTest('empty_iterable'):
      self.assertEqual(executing.run(executable), [])

    executable = parallel(parallel(fn1()))
    with self.subTest('nested_parallel'):
      self.assertEqual(
          executing.run(executable), [['test']]
      )

  def test_wrap_asyncgen(self):

    @executing.make_executable
    @tracing.trace
    async def iterate(num: int) -> AsyncIterator[str]:
      for i in range(num):
        request = _wrap_request(f'{i}_0')
        reply = await request
        yield reply  # pytype: disable=attribute-error

    @executing.make_executable
    async def run(num: int) -> list[str]:
      result = []
      for i in range(num):
        request = _wrap_request(f'{i}_0')
        reply = await request
        result.append(reply)  # pytype: disable=attribute-error
      return result

    result1 = executing.run(iterate(3))
    result2 = executing.run(run(3))

    incremental1 = []
    with executing.stream_updates(iterate(3)) as iterator:
      for r in iterator:
        incremental1.append(r)

    incremental2 = []
    with executing.safe_stream(run(3)) as iterator:
      for r in iterator:
        incremental2.append(r)

    with executing.stream_updates(run(3)) as iterator:
      list(iterator)
    incremenatal2b = iterator.value

    incremental3 = []
    with executing.safe_stream(iterate(3), iteration_depth=0) as iterator:
      for r in iterator:
        incremental3.append(r)

    incremental3b = []
    with executing.stream_updates(iterate(3), iteration_depth=0) as iterator:
      for r in iterator:
        incremental3b.append(r)
      result3b = iterator.value

    incremental4 = []
    with executing.safe_stream(run(3), iteration_depth=0) as iterator:
      for r in iterator:
        incremental4.append(r)

    incremental4b = []
    with executing.stream_updates(run(3), iteration_depth=0) as iterator:
      for r in iterator:
        incremental4b.append(r)
      result4b = iterator.value

    with self.subTest('iterate_should_return_item_on_run'):
      self.assertEqual(result1, '2_0')

    with self.subTest('run_should_return_list_on_run'):
      self.assertListEqual(result2, ['0_0', '1_0', '2_0'])

    with self.subTest('should_return_list_on_stream'):
      self.assertListEqual(incremental1, ['0_0', '1_0', '2_0'])
      self.assertListEqual(incremental2, [['0_0', '1_0', '2_0']])
      self.assertListEqual(incremenatal2b, ['0_0', '1_0', '2_0'])

    with self.subTest('iteration_depth_0_should_behave_as_run'):
      self.assertListEqual(incremental3, [executing.Update('2_0')])
      self.assertListEqual(
          incremental4,
          [executing.Update(['0_0', '1_0', '2_0'])],
      )
      self.assertListEqual(incremental3b, [])
      self.assertListEqual(incremental4b, [])
      self.assertEqual(result3b, '2_0')
      self.assertListEqual(result4b, ['0_0', '1_0', '2_0'])

  def test_iterable_reply(self):
    request = _wrap_request('req_0', streaming=True)
    with executing.stream_updates(request, iteration_depth=-1) as iterator:
      for i, reply_update in enumerate(iterator):
        assert isinstance(reply_update, executing.Update)  # Type hint.
        self.assertEqual(reply_update.to_result(), 'rep_0'[: i + 1])
    reply = executing.run(request)
    self.assertEqual(reply, 'rep_0')

  def test_iterable_parallel_reply(self):
    request1 = _wrap_request('req_0', streaming=True)
    request2 = _wrap_request('other_req_1', streaming=True)
    executable = parallel(request1, request2)
    results = []
    expected_results = [
        ('r', 0),
        ('o', 1),
        ('re', 0),
        ('ot', 1),
        ('rep', 0),
        ('oth', 1),
        ('rep_', 0),
        ('othe', 1),
        ('rep_0', 0),
        ('other', 1),
        ('other_', 1),
        ('other_r', 1),
        ('other_re', 1),
        ('other_rep', 1),
        ('other_rep_', 1),
        ('other_rep_1', 1),
    ]

    with executing.safe_stream(executable, iteration_depth=-1) as iterator:
      for update in iterator:
        assert isinstance(update, executing.ListUpdate)  # Type hint.
        results.append(
            (copy.deepcopy(update.payload[0][0].payload), update.payload[0][1])
        )
    self.assertListEqual(results, expected_results, pprint.pformat(results))

    executable = parallel(request1, request2)
    reply = list(executing.run(executable))
    self.assertListEqual(reply, ['rep_0', 'other_rep_1'])

    executable = parallel(
        process_updates('req_0'),
        process_updates('o_req_1'),
    )
    results = []
    with executing.stream_updates(executable, iteration_depth=-1) as iterator:
      for update in iterator:
        if isinstance(update, updating.Update):
          results.append(update.to_result())
        else:
          results.append(update)
    result = iterator.value
    expected_results = [
        'r',
        'o',
        're',
        'o_',
        'rep',
        'o_r',
        'rep_',
        'o_re',
        'rep_0',
        'o_rep',
        ['rep_0'],
        'o_rep_',
        'o_rep_1',
        [None, 'o_rep_1'],
    ]
    self.assertListEqual(results, expected_results, pprint.pformat(results))
    self.assertListEqual(result, ['rep_0', 'o_rep_1'], pprint.pformat(result))

  def test_parallel_iterators(self):
    @executing.make_executable
    async def iter1():
      requests = [_wrap_request(f'0req_{i}', streaming=True) for i in range(3)]
      executable = serial(*requests)
      async for result in executable.with_depth(-1):
        # We introduce a "fake" request whose reply is not used in order to
        # yield control to the other iterator.
        await _wrap_request('x_0')
        yield result

    @executing.make_executable
    async def iter2():
      requests = [
          _wrap_request(f'1req_{i}', streaming=True) for i in range(3, 6)
      ]
      executable = serial(*requests)
      async for result in executable.with_depth(-1):
        yield result
        # We introduce a "fake" request whose reply is not used in order to
        # yield control to the other iterator.
        await _wrap_request('y_1')

    executable = parallel(iter1(), iter2())
    results = []
    with executing.safe_stream(executable, iteration_depth=-1) as iterator:
      for update in iterator:
        results.append((
            copy.deepcopy(update.to_simplified_result()[0][0]),
            update.payload[0][1],
        ))
    # In particular we want to make sure that the replies are correctly mapped
    # back to the iterator that issued the requests, so the first character of
    # the reply should match with the index of the iterator.
    expected_result = [
        ('0', 0),
        ('1', 1),
        ('0r', 0),
        ('1r', 1),
        ('0re', 0),
        ('1re', 1),
        ('0rep', 0),
        ('1rep', 1),
        ('0rep_', 0),
        ('1rep_', 1),
        ('0rep_0', 0),
        ('1rep_3', 1),
        ('0', 0),
        ('1', 1),
        ('0r', 0),
        ('1r', 1),
        ('0re', 0),
        ('1re', 1),
        ('0rep', 0),
        ('1rep', 1),
        ('0rep_', 0),
        ('1rep_', 1),
        ('0rep_1', 0),
        ('1rep_4', 1),
        ('0', 0),
        ('1', 1),
        ('0r', 0),
        ('1r', 1),
        ('0re', 0),
        ('1re', 1),
        ('0rep', 0),
        ('1rep', 1),
        ('0rep_', 0),
        ('1rep_', 1),
        ('0rep_2', 0),
        ('1rep_5', 1),
    ]
    self.assertListEqual(results, expected_result, pprint.pformat(results))

  @parameterized.named_parameters(
      (
          'depth_0',
          0,
          [[
              ['one_rep_0', 'two_rep_1'],
              ['one_rep_2', 'two_rep_3'],
          ]],
      ),
      (
          'depth_1',
          1,
          [
              [['one_rep_0', 'two_rep_1']],
              [['one_rep_2', 'two_rep_3']],
          ],
      ),
      (
          'depth_2',
          2,
          [
              [['one_rep_0']],
              [['two_rep_1']],
              [['one_rep_2']],
              [['two_rep_3']],
          ],
      ),
      (
          'depth_infinite',
          -1,
          [
              [['o']],
              [['t']],
              [['on']],
              [['tw']],
              [['one']],
              [['two']],
              [['one_']],
              [['two_']],
              [['one_r']],
              [['two_r']],
              [['one_re']],
              [['two_re']],
              [['one_rep']],
              [['two_rep']],
              [['one_rep_']],
              [['two_rep_']],
              [['one_rep_0']],
              [['two_rep_1']],
              [['o']],
              [['t']],
              [['on']],
              [['tw']],
              [['one']],
              [['two']],
              [['one_']],
              [['two_']],
              [['one_r']],
              [['two_r']],
              [['one_re']],
              [['two_re']],
              [['one_rep']],
              [['two_rep']],
              [['one_rep_']],
              [['two_rep_']],
              [['one_rep_2']],
              [['two_rep_3']],
          ],
      ),
  )
  def test_iteration_depth(self, iteration_depth, expected_result):
    request1 = _wrap_request('one_req_0', streaming=True)
    request2 = _wrap_request('two_req_1', streaming=True)
    request3 = _wrap_request('one_req_2', streaming=True)
    request4 = _wrap_request('two_req_3', streaming=True)
    executable1 = parallel(request1, request2)
    executable2 = parallel(request3, request4)
    executable = serial(executable1, executable2)
    results = []
    with executing.safe_stream(
        executable, iteration_depth=iteration_depth
    ) as iterator:
      for update in iterator:
        if isinstance(update, updating.Update):
          results.append(copy.deepcopy(update.to_simplified_result()))
        else:
          results.append(copy.deepcopy(update))
    self.assertListEqual(results, expected_result, pprint.pformat(results))

  def test_executable_with_postprocessing(self):
    @executing.make_executable
    def iterator(length: int) -> Iterator[str]:
      for i in range(length):
        yield f'done {i}'

    e1 = executing.ExecutableWithPostprocessing(
        wrapped=iterator(3), postprocessing_callback=lambda x: x + ' post'
    )
    e2 = executing.ExecutableWithPostprocessing(
        wrapped=iterator(3), postprocessing_callback=lambda x: x + ' post'
    )
    e3 = executing.ExecutableWithPostprocessing(
        wrapped=iterator(3),
        postprocessing_callback=lambda x: x + ' post',
        update_callback=lambda x: x + ' update',
    )
    e4 = executing.ExecutableWithPostprocessing(
        wrapped=iterator(3),
        postprocessing_callback=lambda x: x + ' post',
        update_callback=lambda x: x + ' update',
    )

    with self.subTest('run_with_postprocessing_callback'):
      res1 = executing.run(e1)
      self.assertEqual(res1, 'done 2 post')

    with self.subTest('run_with_postprocessing_and_update_callbacks'):
      res3 = executing.run(e3)
      self.assertEqual(res3, 'done 2 post')

    with self.subTest('stream_with_postprocessing_callback'):
      res2 = updating.Update()
      updates2 = []
      with executing.safe_stream(e2) as iterator:
        for update in iterator:
          res2 += update
          updates2.append(update.to_result())
      self.assertEqual(res2.to_result(), 'done 2 post')
      self.assertListEqual(
          updates2, ['done 0 post', 'done 1 post', 'done 2 post']
      )

    with self.subTest('stream_with_postprocessing_and_update_callbacks'):
      res4 = updating.Update()
      updates4 = []
      with executing.safe_stream(e4) as iterator:
        for update in iterator:
          res4 += update
          updates4.append(update)

      self.assertEqual(res4.to_result(), 'done 2 update')
      self.assertListEqual(
          updates4, ['done 0 update', 'done 1 update', 'done 2 update']
      )

  @parameterized.named_parameters(
      ('run_return_exceptions', 'run', True),
      ('safe_stream_return_exceptions', 'safe_stream', True),
      ('stream_updates_return_exceptions', 'stream_updates', True),
      ('run_no_return_exceptions', 'run', False),
      ('safe_stream_no_return_exceptions', 'safe_stream', False),
      ('stream_updates_no_return_exceptions', 'stream_updates', False),
  )
  def test_parallel_exceptions(self, run_mode, return_exceptions):
    @executing.make_executable
    def compute(s: str) -> str:
      if s == 'error':
        raise ValueError(s)
      else:
        return s + ' done'

    executables = [compute(s) for s in ['test1', 'error', 'test2']]
    executable = executing.par_iter(
        executables, return_exceptions=return_exceptions
    )

    if not return_exceptions:
      with self.assertRaises(ValueError):
        match run_mode:
          case 'run':
            _ = executing.run(executable)
          case 'safe_stream':
            with executing.safe_stream(executable) as iterator:
              _ = sum(iterator, start=updating.Update()).to_result()
          case 'stream_updates':
            with executing.stream_updates(executable) as iterator:
              _ = sum(iterator, start=updating.Update()).to_result()
    else:
      match run_mode:
        case 'run':
          res = executing.run(executable)
        case 'safe_stream':
          with executing.safe_stream(executable) as iterator:
            res = sum(iterator, start=updating.Update()).to_result()
        case 'stream_updates':
          with executing.stream_updates(executable) as iterator:
            res = sum(iterator, start=updating.Update()).to_result()
        case _:
          res = None
      with self.assertRaises(ValueError):
        raise res[1]

  def test_chain_with_iterate_argument(self):
    @executing.make_executable(iterate_argument='prefix')
    def append(prefix: str, postfix: str) -> str:
      return prefix + postfix

    @executing.make_executable('prefix', iterate_argument='prefix')
    def append_non_exec(prefix: str, postfix: str) -> str:
      return prefix + postfix

    @executing.make_executable(iterate_argument='prefix')
    def append_it(prefix: str, postfix: str) -> Iterator[str]:
      postfix_parts = postfix.split(' ')
      for i in range(1, len(postfix_parts) + 1):
        # We iterate while removing spaces from the postfix.
        yield prefix + ''.join(postfix_parts[:i])

    executable = append(append('a', 'b'), append('c', 'd'))
    executable2 = copy.deepcopy(executable)

    result = executing.run(executable)
    with self.subTest('execute_append'):
      self.assertEqual(result, 'abcd')

    result = []
    with executing.safe_stream(executable2) as iterator:
      for update in iterator:
        result.append(updating.Update(update).to_result())

    with self.subTest('iterate_append'):
      # We iterate the prefix but not from the postfix.
      self.assertListEqual(result, ['a', 'ab', 'abcd'])

    executable = append(
        append_non_exec('a', 'b'), append_non_exec('c', 'd')
    )
    executable2 = copy.deepcopy(executable)

    result = executing.run(executable)
    with self.subTest('execute_append_non_exec'):
      self.assertEqual(result, 'abcd')

    result = []
    with executing.safe_stream(executable2) as iterator:
      for update in iterator:
        result.append(updating.Update(update).to_result())

    with self.subTest('iterate_append_non_exec'):
      # We iterate the prefix but not from the postfix.
      self.assertListEqual(result, ['a', 'ab', 'abcd'])

    executable = append_it(append_it('a ', 'b c'), append_it(' d ', 'e f'))
    executable2 = copy.deepcopy(executable)

    result = executing.run(executable)
    with self.subTest('execute_append_it'):
      # Spaces have been removed from the postfixes but not the prefixes.
      self.assertEqual(result, 'a bcdef')

    result = []
    with executing.safe_stream(executable2) as iterator:
      for update in iterator:
        result.append(updating.Update(update).to_result())

    with self.subTest('iterate_append_it'):
      # We iterate the prefix but not from the postfix.
      self.assertListEqual(
          result,
          [
              'a ',  # Prefix of prefix of function call
              'a b',  # Iterate 1 of postfix of prefix
              'a bc',  # Iterate 2 of postfix of prefix
              'a bc',  # Prefix of function call
              'a bcd',  # First part of postfix
              'a bcdef',  # Second part of postfix
          ],
      )

  def test_iterator_with_var_positional_args(self):
    @make_executable
    async def iterator(*args):
      for arg in args:
        yield arg

    result = executing.run(iterator('a', 'b', 'c', 'd'))
    self.assertEqual(result, 'd')

    with executing.safe_stream(iterator('a', 'b', 'c', 'd')) as iterator:
      result = []
      for update in iterator:
        result.append(updating.Update(update).to_result())
      self.assertEqual(result, ['a', 'b', 'c', 'd'])

  def test_pre_execute(self):
    @make_executable
    async def iterating_function(n):
      for i in range(n):
        yield f'done {i}'

    @make_executable
    async def wrapper(n):
      return iterating_function(n)

    with self.subTest('execute_run'):
      result = executing.run(wrapper(3))
      self.assertEqual(result, 'done 2')

    with self.subTest('execute_stream'):
      with executing.safe_stream(wrapper(3)) as iterator:
        result = []
        for update in iterator:
          result.append(updating.Update(update).to_result())
        self.assertEqual(result, ['done 0', 'done 1', 'done 2'])

    with self.subTest('pre_execute_run'):
      pre_result = executing.run(wrapper(3).pre_execute())
      result = executing.run(pre_result)
      self.assertEqual(result, 'done 2')

    with self.subTest('pre_execute_stream'):
      pre_result = executing.run(wrapper(3).pre_execute())
      with executing.safe_stream(pre_result) as iterator:
        result = []
        for update in iterator:
          result.append(updating.Update(update).to_result())
        self.assertEqual(result, ['done 0', 'done 1', 'done 2'])

  def test_execution_of_var_positional(self):
    @executing.make_executable
    def idle(a):
      return a

    def fn(a, *args, other):
      return a + ';' + ','.join(args) + ' ' + other

    f1 = executing.make_executable(fn)
    f2 = executing.make_executable('args')(fn)

    with self.subTest('execute_all_positional'):
      result = executing.run(
          f1(idle('a'), idle('b'), idle('c'), other=idle('d'))
      )
      self.assertEqual(result, 'a;b,c d')

    with self.subTest('fails_to_execute'):
      with self.assertRaises(TypeError):
        # This raises a TypeError because the arguments will not be executed
        # hence the function will call `join` on objects that are executables
        # instead of strings.
        _ = executing.run(f2('a', 'b', idle('c'), other=idle('d')))

    with self.subTest('execute_non_positional'):
      result = executing.run(f2('a', 'b', 'c', other=idle('d')))
      self.assertEqual(result, 'a;b,c d')

  def test_callable_object(self):
    class _CallableObject:

      def __call__(self, *args, **kwargs):
        return f'called({args}, {kwargs})'

    obj = make_executable(_CallableObject())
    result = executing.run(obj(1, b=3))
    self.assertEqual(result, "called((1,), {'b': 3})")

  def test_async_callable_object(self):
    class _AsyncCallableObject:

      async def __call__(self, *args, **kwargs):
        return f'called({args}, {kwargs})'

    obj = make_executable(_AsyncCallableObject())
    result = executing.run(obj(1, b=3))
    self.assertEqual(result, "called((1,), {'b': 3})")

  @parameterized.named_parameters(
      ('depth_0', 0, []),
      ('depth_1', 1, ['start', 3]),
      ('depth_2', 2, ['start', 0, 1, 2, 3]),
  )
  def test_stream_updates(self, depth: int, expected_updates: list[str | int]):
    """Test streaming updates."""

    @executing.make_executable
    @tracing.trace
    def inner(value):
      return value

    @executing.make_executable
    @tracing.trace
    async def plan(value):
      res = 0
      await tracing.report_update('start')
      for i in range(value):
        res += await inner(i)
      return res

    with executing.stream_updates(plan(3), iteration_depth=depth) as it:
      res = list(it)
    final = it.value
    self.assertListEqual(res, expected_updates)
    self.assertEqual(final, 3)


if __name__ == '__main__':
  absltest.main()
