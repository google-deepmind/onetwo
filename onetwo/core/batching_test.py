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

import asyncio
from collections.abc import Mapping
import functools
import pprint
import threading
import time
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import batching
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils


# A function operating on batches.
@batching.batch_function(batch_size=3, debug=True)
def my_fn(requests: Sequence[dict[str, Any]]) -> Sequence[str]:
  return [p['s'] + f" d{p['d']}" + f" f{p['f']}" + ' done' for p in requests]


# The signature of the function for single calls.
@batching.batchable_function(implementation=my_fn)
def my_single_fn(s: str, d: int, f: float = 0.0):
  # Just to give the signature, the implementation is done by my_fn.
  del s, d, f
  pass


# A function determining how to batch calls.
# It puts together request whose field 's' has the same first character.
def can_batch(
    requests: Sequence[dict[str, Any]], req: dict[str, Any]
) -> tuple[bool, bool]:
  # Batch is never full.
  is_full_after_adding = False
  # If batch is empty, we can add the request.
  can_add_to_batch = not requests
  if requests:
    # We want only requests with the same starting character.
    can_add_to_batch = req['s'][0] == requests[0]['s'][0]
  return can_add_to_batch, is_full_after_adding


# Another function using can_batch to decide how to batch.
@batching.batch_function(batching_function=can_batch, debug=True)
def my_other_fn(requests: Sequence[dict[str, Any]]) -> Sequence[str]:
  return [p['s'] + f" d{p['d']}" + f" f{p['f']}" + ' done' for p in requests]


@batching.add_batching
class ClassWithBatch:

  @batching.batchable_method(implementation=utils.FromInstance('my_fn'))
  def my_single_fn(self, s: str, d: int, f: float = 0.0):
    # Just to give the signature.
    pass

  @batching.batch_method(batch_size=3, debug=True)
  def my_fn(self, requests: Sequence[dict[str, Any]]) -> Sequence[str]:
    return [p['s'] + f" d{p['d']}" + f" f{p['f']}" + ' done' for p in requests]

  # A method determining how to batch calls.
  # It puts together request whose field 's' has the same first character.
  def can_batch_method(
      self, requests: Sequence[dict[str, Any]], req: dict[str, Any]
  ) -> tuple[bool, bool]:
    # Batch is empty, we can add the request.
    can_add_to_batch = not requests
    # Batch is never full.
    is_full_after_adding = False
    if requests:
      # We want requests with the same starting character.
      can_add_to_batch = req['s'][0] == requests[0]['s'][0]
    return can_add_to_batch, is_full_after_adding

  @batching.batch_method(
      batching_function=utils.FromInstance('can_batch_method'), debug=True
  )
  def my_other_fn(self, requests: Sequence[dict[str, Any]]) -> Sequence[str]:
    return [p['s'] + f" d{p['d']}" + f" f{p['f']}" + ' done' for p in requests]


class BatchingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We make sure the queues are stopped so that individual test cases that
    # fail do not affect other test cases.
    asyncio.run(batching._finish_queues(force=True))

  @parameterized.named_parameters(
      ('method_batchsize_list', True, False, False),
      ('method_batchsize_single', True, False, True),
      ('method_batchingfunction_list', True, True, False),
      ('function_batchsize_list', False, False, False),
      ('function_batchsize_single', False, False, True),
      ('function_batchingfunction_list', False, True, False),
  )
  def test_batching_decorators(
      self, method, has_batching_function, call_normal_signature):
    c = ClassWithBatch()

    async def serial_plan(i):
      for r in ['a', 'b', 'c']:
        if call_normal_signature:
          # If we call the normal signature we directly provide arguments.
          if method:
            result = await c.my_single_fn(r + str(i), 0)
          else:
            result = await my_single_fn(r + str(i), 0)
        else:
          # Otherwise we have to pass a list containing a dict of argument names
          # and their values.
          dict_with_args = {
              's': r + str(i),
              'd': 0,
              'f': 0.0,
          }
          if method and has_batching_function:
            result = await c.my_other_fn([dict_with_args])
          elif method and not has_batching_function:
            result = await c.my_fn([dict_with_args])
          elif not method and has_batching_function:
            result = await my_other_fn([dict_with_args])
          elif not method and not has_batching_function:
            result = await my_fn([dict_with_args])
          else:
            assert False
      return result

    async def parallel_plan():
      coroutines = []
      for i in range(4):
        coroutines.append(serial_plan(i))
      result = await asyncio.gather(*coroutines)
      return result

    results = batching.run(parallel_plan())

    expected_batchsize_calls = [
        [
            {'s': 'a0', 'd': 0, 'f': 0.0},
            {'s': 'a1', 'd': 0, 'f': 0.0},
            {'s': 'a2', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'b0', 'd': 0, 'f': 0.0},
            {'s': 'b1', 'd': 0, 'f': 0.0},
            {'s': 'a3', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'c0', 'd': 0, 'f': 0.0},
            {'s': 'c1', 'd': 0, 'f': 0.0},
            {'s': 'b2', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'c2', 'd': 0, 'f': 0.0},
            {'s': 'b3', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'c3', 'd': 0, 'f': 0.0},
        ],
    ]

    expected_batchfunction_calls = [
        [
            {'s': 'a0', 'd': 0, 'f': 0.0},
            {'s': 'a1', 'd': 0, 'f': 0.0},
            {'s': 'a2', 'd': 0, 'f': 0.0},
            {'s': 'a3', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'b0', 'd': 0, 'f': 0.0},
            {'s': 'b1', 'd': 0, 'f': 0.0},
            {'s': 'b2', 'd': 0, 'f': 0.0},
            {'s': 'b3', 'd': 0, 'f': 0.0},
        ],
        [
            {'s': 'c0', 'd': 0, 'f': 0.0},
            {'s': 'c1', 'd': 0, 'f': 0.0},
            {'s': 'c2', 'd': 0, 'f': 0.0},
            {'s': 'c3', 'd': 0, 'f': 0.0},
        ],
    ]
    expected_calls = (
        expected_batchfunction_calls
        if has_batching_function
        else expected_batchsize_calls
    )

    expected_dict_results = [
        ['c0 d0 f0.0 done'],
        ['c1 d0 f0.0 done'],
        ['c2 d0 f0.0 done'],
        ['c3 d0 f0.0 done'],
    ]
    expected_single_results = [
        'c0 d0 f0.0 done',
        'c1 d0 f0.0 done',
        'c2 d0 f0.0 done',
        'c3 d0 f0.0 done',
    ]
    expected_results = (
        expected_single_results if call_normal_signature
        else expected_dict_results
    )

    # We retrieve the underlying queue from the decorated function.
    if call_normal_signature:
      if method:
        inner = c.my_other_fn if has_batching_function else c.my_fn
      else:
        inner = my_other_fn if has_batching_function else my_fn
    else:
      if method and has_batching_function:
        inner = c.my_other_fn
      elif method and not has_batching_function:
        inner = c.my_fn
      elif not method and has_batching_function:
        inner = my_other_fn
      elif not method and not has_batching_function:
        inner = my_fn
      else:
        assert False
    if method:
      calls = inner.__self__.batching_queues[inner.__qualname__].calls  # pytype: disable=attribute-error
    else:
      calls = inner.__self__.calls  # pytype: disable=attribute-error

    with self.subTest('should_return_enough_results'):
      self.assertLen(results, 4)

    with self.subTest('should_make_batched_calls'):
      self.assertListEqual(calls, expected_calls, pprint.pformat(calls))

    with self.subTest('should_return_the_right_results'):
      self.assertListEqual(results, expected_results, pprint.pformat(results))

  @parameterized.named_parameters(
      ('batching_on', True, [[0, 1], [2], [3, 4], [5]]),
      ('batching_off', False, [[0, 1, 2], [3, 4, 5]]),
  )
  def test_run_disable_batching(self, enable_batching, expected_results):
    processed = []

    @batching.batch_function(batch_size=2)
    def process(requests):
      processed.append(requests)
      return requests

    async def plan():
      _ = await process(list(range(3)))
      _ = await process(list(range(3, 6)))

    batching.run(plan(), enable_batching=enable_batching)
    with self.subTest('should_return_correct_results'):
      self.assertListEqual(
          processed, expected_results, pprint.pformat(processed)
      )

  @parameterized.named_parameters(
      ('batching_on', True, [[0, 1], [2], [3, 4], [5]]),
      ('batching_off', False, [[0, 1, 2], [3, 4, 5]]),
  )
  def test_stream_disable_batching(self, enable_batching, expected_results):
    processed = []

    @batching.batch_function(batch_size=2)
    def process(requests):
      processed.append(requests)
      return requests

    async def plan():
      _ = await process(list(range(3)))
      yield 0
      _ = await process(list(range(3, 6)))
      yield 1

    for _ in batching.stream(plan(), enable_batching=enable_batching):
      pass

    with self.subTest('should_return_correct_results'):
      self.assertListEqual(
          processed, expected_results, pprint.pformat(processed)
      )

  def test_call_with_list(self):
    @batching.batch_function(batch_size=3)
    def process(requests):
      return requests

    result = list(batching.run(process(list(range(10)))))
    self.assertListEqual(result, list(range(10)), pprint.pformat(result))

  def test_wrong_batching_function(self):
    @batching.batch_function(batching_function=lambda x, y: (False, True))
    def process(requests):
      return requests

    async def plan():
      result = await asyncio.gather(process([1]), process([2]), process([3, 4]))
      return result

    with self.assertRaisesRegex(
        ValueError,
        'Batching queue not empty but none got batched',
    ):
      batching.run(plan())

  def test_wrong_batch_function_return_type(self):
    @batching.batch_function(batch_size=3)
    def process(unused_requests):
      return 0  # Return type is not a Sequence.

    async def plan():
      result = await asyncio.gather(process([1]), process([2]), process([3, 4]))
      return result

    with self.assertRaisesRegex(
        ValueError, 'should return a Sequence of results'
    ):
      batching.run(plan())

  def test_wrong_batch_function_return_length(self):
    @batching.batch_function(batch_size=3)
    def process(requests):
      return requests[0:1]  # Return fewer results than requests

    async def plan():
      result = await asyncio.gather(process([1]), process([2]), process([3, 4]))
      return result

    with self.assertRaisesRegex(
        ValueError, 'should return as many results as requests'
    ):
      batching.run(plan())

  def test_run_twice(self):
    @batching.batch_function(batch_size=3)
    def process(requests):
      return requests[0:1]  # Return fewer results than requests

    async def something():
      pass

    async def plan():
      result = await process([1])
      batching.run(something())
      return result

    with self.assertRaisesRegex(
        ValueError, 'Cannot call run or stream more than once per thread.'
    ):
      batching.run(plan())

  def test_stream_and_run(self):
    # We check that calling run/stream while streaming is not possible.
    @batching.batch_function(batch_size=3)
    async def process(requests):
      time.sleep(1)
      return requests

    async def plan():
      for i in range(10):
        result = await process([i])
        yield result

    for _ in batching.stream(plan()):
      with self.assertRaisesRegex(
          ValueError, 'Cannot call run or stream more than once per thread.'
      ):
        # Indeed, run expects Awaitable, and here we are sending AsyncGenerator.
        # Exception will be raised before the `run` actually being executed, so
        # let's just ignore the pytype error.
        batching.run(plan())  # pytype: disable=wrong-arg-types
      with self.assertRaisesRegex(
          ValueError, 'Cannot call run or stream more than once per thread.'
      ):
        for _ in batching.stream(plan()):
          pass

  def test_run_in_threads(self):
    @batching.batch_function(batch_size=3)
    def process(requests):
      return requests[0:1]  # Return fewer results than requests

    def something():
      async def inner():
        return await process([2])
      return batching.run(inner())

    async def plan():
      result = list(await process([1]))
      result.extend(await asyncio.to_thread(something))
      return result

    with self.subTest('should_run_in_subthread'):
      res = batching.run(plan())
      self.assertEqual([1, 2], res, res)

  def test_run_from_threads(self):
    """Checks that one can run from two threads if enable_batching=False."""
    processed = []

    @batching.batch_function(batch_size=2)
    def process(requests):
      time.sleep(1)
      processed.append(requests)
      return requests

    def send(requests):
      async def plan():
        result = await asyncio.gather(*[process([req]) for req in requests])
        return result

      return batching.run(plan(), enable_batching=False)

    thread1 = threading.Thread(
        target=functools.partial(send, [1, 2, 3])
    )
    thread2 = threading.Thread(
        target=functools.partial(send, [4, 5, 6])
    )
    thread1.start()
    # We ensure the second thread starts a bit after the first.
    time.sleep(0.5)
    thread2.start()
    thread1.join()
    thread2.join()
    self.assertListEqual(processed, [[1], [4], [2], [5], [3], [6]])

  def test_async_batched_function(self):
    @batching.batch_function(batch_size=3)
    async def process(requests):
      return requests

    async def plan():
      result = await process(list(range(10)))
      return result

    result = list(batching.run(plan()))
    self.assertListEqual(result, list(range(10)), pprint.pformat(result))

  def test_stream(self):
    batches = []

    @batching.batch_function(batch_size=3)
    async def process(requests):
      batches.append(requests)
      return requests

    async def plan():
      for i in range(10):
        result = await process([i])
        yield result

    results = []
    for r in batching.stream(plan()):
      results.append(r)

    self.assertListEqual(
        results, [[i] for i in range(10)], pprint.pformat(results)
    )
    self.assertListEqual(
        batches, [[i] for i in range(10)], pprint.pformat(batches)
    )

  def test_safe_stream(self):
    batches = []

    @batching.batch_function(batch_size=3)
    async def process(requests):
      batches.append(requests)
      return requests

    async def plan():
      for i in range(10):
        result = await process([i])
        yield result

    results = []
    with batching.safe_stream(plan()) as iterator:
      for r in iterator:
        results.append(r)

    self.assertListEqual(
        results, [[i] for i in range(10)], pprint.pformat(results)
    )
    self.assertListEqual(
        batches, [[i] for i in range(10)], pprint.pformat(batches)
    )

    with self.assertRaisesRegex(ValueError, 'test'):
      with batching.safe_stream(plan()) as iterator:
        for r in iterator:
          results.append(r)
          raise ValueError('test')

  def test_stream_updates(self):

    # This will trace the output.
    @tracing.trace  # pytype: disable=wrong-arg-types
    async def process(value):
      return value + 1

    async def plan():
      for i in range(4):
        await tracing.report_update(f'{i} sent')
        result = await process(i)
      return result

    results = []
    with batching.stream_updates(plan(), iteration_depth=-1) as iterator:
      for update in iterator:
        results.append(update)
      final_result = iterator.value

    with self.subTest('should_produce_the_right_result'):
      self.assertEqual(final_result, 4)

    with self.subTest('should_produce_the_right_updates'):
      expected_updates = ['0 sent', 1, '1 sent', 2, '2 sent', 3, '3 sent', 4]
      self.assertListEqual(
          results, expected_updates, pprint.pformat(results)
      )

    with self.subTest('should_raise_on_loop_interruption'):
      with self.assertRaisesRegex(ValueError, 'test'):
        with batching.stream_updates(plan()) as iterator:
          for r in iterator:
            results.append(r)
            raise ValueError('test')

  def test_stream_updates_with_batching(self):
    batches = []

    @batching.batch_function(batch_size=3)
    @tracing.trace  # pytype: disable=wrong-arg-types
    async def process(requests):
      batches.append(requests)
      return requests

    async def plan():
      for i in range(2):
        executables = [process([i * 5 + j]) for j in range(5)]
        result = await asyncio.gather(*executables)
        await tracing.report_update(result)
      return result

    results = []
    with batching.stream_updates(plan(), iteration_depth=-1) as iterator:
      for update in iterator:
        results.append(update)
      final_result = iterator.value

    with self.subTest('should_produce_the_right_result'):
      self.assertEqual(final_result, [[5], [6], [7], [8], [9]])

    with self.subTest('should_produce_the_right_updates'):
      expected_updates = [
          [0, 1, 2],  # First batch sent.
          [3, 4],  # Second batch sent.
          [[0], [1], [2], [3], [4]],  # First result returned.
          [5, 6, 7],  # Third batch sent.
          [8, 9],  # Fourth batch sent.
          [[5], [6], [7], [8], [9]],  # Second result returned.
      ]
      self.assertListEqual(results, expected_updates, pprint.pformat(results))

    with self.subTest('should_produce_the_right_batches'):
      self.assertListEqual(
          batches,
          [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9]],
          pprint.pformat(batches),
      )

    with self.subTest('should_raise_on_loop_interruption'):
      with self.assertRaisesRegex(ValueError, 'test'):
        with batching.stream_updates(plan()) as iterator:
          for r in iterator:
            results.append(r)
            raise ValueError('test')

  def test_run_in_separate_threads(self):
    """Checks that batching.run() can be run in separate threads."""
    batches = []
    calls = {}
    call_ids = []
    @batching.batch_function(batch_size=3, debug=True)
    async def process(requests):
      nonlocal batches
      batches.append(requests)
      time.sleep(1)
      return requests

    async def plan(name):
      nonlocal calls, call_ids
      coroutines = []
      for i in range(5):
        coroutines.append(process([f'{name}{i}']))
      result = await asyncio.gather(*coroutines)
      call_ids.append(list(batching._thread_data.calls.keys())[0])
      calls[name] = list(batching._thread_data.calls.values())[0]
      return result

    def execute(name):
      return batching.run(plan(name))

    t1 = threading.Thread(target=functools.partial(execute, 'a'))
    t2 = threading.Thread(target=functools.partial(execute, 'b'))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    with self.subTest('should_produce_the_right_batches'):
      # We check that all batches are produced and interleaved between the
      # two threads.
      self.assertCountEqual(  # We compare the lists without checking order.
          batches,
          [
              ['a0', 'a1', 'a2'],
              ['b0', 'b1', 'b2'],
              ['a3', 'a4'],
              ['b3', 'b4'],
          ],
          pprint.pformat(batches),
      )

    with self.subTest('should_operate_on_the_same_object'):
      self.assertEqual(call_ids[0], call_ids[1])

    with self.subTest('should_produce_the_right_distinct_calls'):
      self.assertListEqual(
          calls['a'],
          [['a0', 'a1', 'a2'], ['a3', 'a4']],
          pprint.pformat(calls['a']),
      )
      self.assertListEqual(
          calls['b'],
          [['b0', 'b1', 'b2'], ['b3', 'b4']],
          pprint.pformat(calls['b']),
      )

  def test_cleaning_on_exception(self):
    @batching.batch_function(batch_size=2)
    async def process(requests):
      return requests

    async def stream_plan(raises: bool = True):
      for i in range(3):
        result = await process([i] * 2)
        yield result
        if raises:
          raise KeyError('test')

    async def run_plan(raises: bool = True):
      res = []
      for i in range(3):
        result = await process([i] * 2)
        res.append(result)
        if raises:
          raise KeyError('test')
      return res

    with self.subTest('should_stop_run_on_exception'):
      with self.assertRaisesRegex(KeyError, 'test'):
        _ = batching.run(run_plan(raises=True))

    with self.subTest('should_restart_run_normally'):
      run_results = batching.run(run_plan(raises=False))
      self.assertListEqual(
          run_results, [[0, 0], [1, 1], [2, 2]], pprint.pformat(run_results)
      )

    stream_results = []
    with self.subTest('should_stop_stream_on_exception'):
      with self.assertRaisesRegex(KeyError, 'test'):
        for r in batching.stream(stream_plan(raises=True)):
          stream_results.append(r)

    with self.subTest('stream_should_produce_partial_results'):
      self.assertListEqual(
          stream_results, [[0, 0]], pprint.pformat(stream_results)
      )

    with self.subTest('should_restart_stream_normally'):
      for r in batching.stream(stream_plan(raises=False)):
        stream_results.append(r)

    with self.subTest('stream_should_produce_full_results'):
      self.assertListEqual(
          stream_results,
          [[0, 0], [0, 0], [1, 1], [2, 2]],
          pprint.pformat(stream_results),
      )

  def test_stop_queues(self):
    @batching.add_batching
    class ClassWithBatchForTest:

      @batching.batch_method(batch_size=3)
      def process(self, requests):
        return requests

    @batching.batch_function(batch_size=3)
    async def process(requests):
      return requests

    async def stream_plan():
      for i in range(10):
        result = await process([i] * 5)
        yield result

    async def run_plan():
      res = []
      for i in range(10):
        result = await process([i] * 5)
        res.append(result)
      return res

    instance = ClassWithBatchForTest()

    async def method_stream_plan():
      for i in range(10):
        result = await instance.process([i] * 5)
        yield result

    async def method_run_plan():
      res = []
      for i in range(10):
        result = await instance.process([i] * 5)
        res.append(result)
      return res

    for _ in range(3):
      # We repeat three times to make sure that each run doesn't leak into
      # the next one, i.e. everything is properly closed and cleaned up after
      # each run/stream call.
      stream_results = []
      for r in batching.stream(stream_plan()):
        stream_results.append(r)
      # This will fail if stream didn't properly close the queues.
      run_results = batching.run(run_plan())

      self.assertListEqual(
          stream_results,
          [[i] * 5 for i in range(10)],
          pprint.pformat(stream_results),
      )
      self.assertListEqual(
          run_results, [[i] * 5 for i in range(10)], pprint.pformat(run_results)
      )
      stream_results = []
      for r in batching.stream(method_stream_plan()):
        stream_results.append(r)
      # This will fail if stream didn't properly close the queues.
      run_results = batching.run(method_run_plan())

      self.assertListEqual(
          stream_results,
          [[i] * 5 for i in range(10)],
          pprint.pformat(stream_results),
      )
      self.assertListEqual(
          run_results, [[i] * 5 for i in range(10)], pprint.pformat(run_results)
      )

  def test_run_function_in_threadpool(self):
    @batching.run_function_in_threadpool
    def process(request):
      return request

    @batching.run_function_in_threadpool
    def process_with_error(request):
      if request == 1 or request == 3:
        raise KeyError(str(request))
      return request

    results = process([{'request': i} for i in range(10)])

    self.assertListEqual(
        list(range(10)), list(results), pprint.pformat(results)
    )

    with self.assertRaisesRegex(ValueError, 'KeyError'):
      _ = process_with_error([{'request': i} for i in range(4)])

  def test_run_method_in_threadpool(self):
    class C:
      @batching.run_method_in_threadpool
      def process(self, request):
        return request

      @batching.run_method_in_threadpool
      def process_with_error(self, request):
        if request == 1 or request == 3:
          raise KeyError(str(request))
        return request

    results = C().process([{'request': i} for i in range(10)])

    self.assertListEqual(
        list(range(10)), list(results), pprint.pformat(results)
    )

    with self.assertRaisesRegex(ValueError, 'KeyError'):
      _ = C().process_with_error([{'request': i} for i in range(4)])

  def test_batching_function_with_threadpool(self):
    calls = []

    @batching.batch_function(batch_size=3)
    @batching.run_function_in_threadpool
    def process_batch(request):
      nonlocal calls
      t = time.time()
      calls.append((request, int(t)))
      time.sleep(1)
      return request

    @batching.batchable_function(implementation=process_batch)
    def process(request):
      del request

    async def run_plan():
      coroutines = [process(i) for i in range(10)]
      return await asyncio.gather(*coroutines)

    run_results = batching.run(run_plan())

    with self.subTest('should_produce_the_right_results'):
      self.assertListEqual(
          run_results, [i for i in range(10)], pprint.pformat(run_results)
      )

    with self.subTest('should_produce_4_batches_in_less_than_5_seconds'):
      # Each call is tagged by its timestamp, and the processing takes 1 second,
      # so we can check that the 10 requests are processed in batches of 3 by
      # checking that the timestamps of all the requests are spread over 3 to 5
      # seconds max.
      times = [c[1] for c in calls]
      span = max(times) - min(times)
      self.assertBetween(span, 3, 5, msg=pprint.pformat(times))

  def test_batching_method_with_threadpool(self):
    calls = []

    @batching.add_batching
    class C:
      @batching.batch_method(batch_size=3)
      @batching.run_method_in_threadpool
      def process_batch(self, request):
        nonlocal calls
        t = time.time()
        calls.append((request, int(t)))
        time.sleep(1)
        return request

      @batching.batchable_method(
          implementation=utils.FromInstance('process_batch')
      )
      def process(self, request):
        del request

    async def run_plan():
      c = C()
      coroutines = [c.process(i) for i in range(10)]
      return await asyncio.gather(*coroutines)

    run_results = batching.run(run_plan())

    with self.subTest('should_produce_the_right_results'):
      self.assertListEqual(
          run_results, [i for i in range(10)], pprint.pformat(run_results)
      )

    with self.subTest('should_produce_4_batches_in_less_than_5_seconds'):
      # Each call is tagged by its timestamp, and the processing takes 1 second,
      # so we can check that the 10 requests are processed in batches of 3 by
      # checking that the timestamps of all the requests are spread over 3 to 5
      # seconds max.
      times = [c[1] for c in calls]
      span = max(times) - min(times)
      self.assertBetween(span, 3, 5, msg=pprint.pformat(times))

  def test_batching_with_threadpool_single_decorator(self):
    calls = []
    num_calls = 0
    num_batches = 0

    def wrapper(function):
      nonlocal num_batches
      @functools.wraps(function)
      def wrapped_function(requests: Sequence[Any]) -> Sequence[Any]:
        nonlocal num_batches
        num_batches += 1
        return function(requests)

      return wrapped_function

    @batching.batch_function_with_threadpool(batch_size=3, wrapper=wrapper)
    def process(request):
      nonlocal calls, num_calls
      t = time.time()
      calls.append((request, int(t)))
      num_calls += 1
      time.sleep(1)
      return request

    async def run_plan():
      coroutines = [process(i) for i in range(10)]
      return await asyncio.gather(*coroutines)

    run_results = batching.run(run_plan())

    with self.subTest('should_produce_the_right_results'):
      self.assertListEqual(
          run_results, [i for i in range(10)], pprint.pformat(run_results)
      )

    with self.subTest('should_produce_4_batches_in_less_than_5_seconds'):
      # Each call is tagged by its timestamp, and the processing takes 1 second,
      # so we can check that the 10 requests are processed in batches of 3 by
      # checking that the timestamps of all the requests are spread over 3 to 5
      # seconds max.
      times = [c[1] for c in calls]
      span = max(times) - min(times)
      self.assertBetween(span, 3, 5, msg=pprint.pformat(times))

    with self.subTest('should_produce_the_right_counters'):
      self.assertEqual(num_batches, 4)
      self.assertEqual(num_calls, 10)

  def test_batching_method_with_threadpool_single_decorator(self):
    calls = []

    def wrapper(method):
      @functools.wraps(method)
      def wrapped_method(self, requests: Sequence[Any]) -> Sequence[Any]:
        self.batches += 1
        return method(self, requests)

      return wrapped_method

    @batching.add_batching
    class C:
      def __init__(self):
        self.batches = 0
        self.calls = 0

      @batching.batch_method_with_threadpool(batch_size=3, wrapper=wrapper)
      def process(self, request):
        nonlocal calls
        t = time.time()
        calls.append((request, int(t)))
        self.calls += 1
        time.sleep(1)
        return request

    c_instance = C()
    async def run_plan():
      nonlocal c_instance
      coroutines = [c_instance.process(i) for i in range(10)]
      return await asyncio.gather(*coroutines)

    run_results = batching.run(run_plan())

    with self.subTest('should_produce_the_right_results'):
      self.assertListEqual(
          run_results, [i for i in range(10)], pprint.pformat(run_results)
      )

    with self.subTest('should_produce_4_batches_in_less_than_5_seconds'):
      # Each call is tagged by its timestamp, and the processing takes 1 second,
      # so we can check that the 10 requests are processed in batches of 3 by
      # checking that the timestamps of all the requests are spread over 3 to 5
      # seconds max.
      times = [c[1] for c in calls]
      span = max(times) - min(times)
      self.assertBetween(span, 3, 5, msg=pprint.pformat(times))

    with self.subTest('should_produce_the_right_counters'):
      self.assertEqual(c_instance.batches, 4)
      self.assertEqual(c_instance.calls, 10)

  def test_fills_batches_in_par_of_par(self):
    @batching.add_batching
    class BatchClass:
      def __init__(self):
        self.results = []

      @batching.batchable_method(implementation=utils.FromInstance('my_fn'))
      def my_single_fn(self, i: int, j: int, k: int) -> str:
        # Just to give the signature.
        del i, j, k
        return ''

      @batching.batch_method(batch_size=10, debug=True)
      def my_fn(
          self, requests: Sequence[Mapping[str, Any]]
      ) -> Sequence[str]:
        # We simulate that each step takes a bit of time.
        time.sleep(0.1)
        results = [str((r['i'], r['j'], r['k'])) for r in requests]
        self.results.append(results)
        return results

    batch_class = BatchClass()
    target = batch_class.my_single_fn

    async def coro(i, j):
      nonlocal target
      for k in range(3):
        # We simulate that each step.
        await asyncio.sleep(0.02)
        await target(i, j, k)

    async def par(*args):
      await asyncio.gather(*(coro(*a) for a in args))

    async def ppar(*args):
      await asyncio.gather(*(par(*a) for a in args))

    l = [[(i, j) for j in range(4)] for i in range(4)]
    batching.run(ppar(*l))
    batches_sizes = [len(l) for l in batch_class.results]

    # We send a total of 4*4*3=48 requests, with batches of size 10,
    # but there is some serialization (each coro sends 3 in series and we
    # have 4*4 of these in nested parallel fashion).
    # Due to the way tasks are prioritized, we cannot garanty that we will
    # always have full batches, but at least the queue monitoring task should
    # not read the queue too fast (otherwise it will start executing while it
    # could have waited a bit longer for the queue to fill up).
    pprint.pprint(batch_class.results)
    # We check that we at least get a few batches full. Indeed, this may
    # depend on the exact prioritization and timing, but if the queue does not
    # get read too fast, it has time to fill up.
    self.assertListEqual(
        batches_sizes[:3], [10, 10, 10], pprint.pformat(batches_sizes)
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_error_batching_enabled',
          'inputs': [1, 2, 3],
          'enable_batching': True,
          'expected_processed': [[1, 2], [3]],
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_enabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': True,
          'expected_processed': [[1, 'error2'], [3]],
          # When an error occurs in a batch, it affects all of the other
          # requests in that batch (but not requests from other batches).
          'expected_results': ['error2', 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
      {
          'testcase_name': 'no_error_batching_disabled',
          'inputs': [1, 2, 3],
          'enable_batching': False,
          'expected_processed': [[1], [2], [3]],
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_disabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': False,
          'expected_processed': [[1], ['error2'], [3]],
          # When an error occurs in a batch, it affects all of the other
          # requests in that batch (but not requests from other batches).
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
  )
  def test_exception_handling_batch_function(
      self,
      inputs,
      enable_batching,
      expected_processed,
      expected_results,
      expected_error_message_if_not_caught,
  ):
    processed = []

    @batching.batch_function(batch_size=2)
    def process_batch(requests: Sequence[dict[str, Any]]) -> Sequence[int]:
      requests = [r['request'] for r in requests]
      processed.append(
          [(str(r) if isinstance(r, Exception) else r) for r in requests]
      )
      replies = []
      for r in requests:
        if isinstance(r, Exception):
          raise r
        else:
          replies.append(r)
      return replies

    @executing.make_executable  # pytype: disable=wrong-arg-types
    @batching.batchable_function(implementation=process_batch)
    def process(request: int | Exception) -> int:
      del request
      pass  # pytype: disable=bad-return-type

    @executing.make_executable  # pytype: disable=wrong-arg-types
    async def process_with_error_handling(x: int | Exception) -> int | str:
      try:
        return await process(x)
      except Exception as e:  # pylint: disable=broad-except
        return str(e)

    async def process_all_without_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int]:
      executables = [process(x) for x in inputs]
      return await executing.parallel(*executables)

    async def process_all_with_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int | str]:
      executables = [process_with_error_handling(x) for x in inputs]
      return await executing.parallel(*executables)

    results = batching.run(
        process_all_with_error_handling(inputs), enable_batching=enable_batching
    )
    with self.subTest('processed_with_correct_batching'):
      self.assertSequenceEqual(
          expected_processed, processed, pprint.pformat(processed)
      )
    with self.subTest('returns_correct_results'):
      self.assertSequenceEqual(
          expected_results, results, pprint.pformat(results)
      )

    if expected_error_message_if_not_caught is not None:
      with self.subTest('raises_informative_error_if_not_caught'):
        # Here we verify that if the caller does not implement any error
        # handling, the original error gets raised (the same as it would if no
        # batching were applied), rather than getting masked by an error like
        # "ValueError: Attempting to finish a non-empty queue".
        with self.assertRaisesRegex(
            ValueError, expected_error_message_if_not_caught
        ):
          batching.run(
              process_all_without_error_handling(inputs),
              enable_batching=enable_batching,
          )

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_error_batching_enabled',
          'inputs': [1, 2, 3],
          'enable_batching': True,
          'expected_processed': [[1, 2], [3]],
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_enabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': True,
          'expected_processed': [[1, 'error2'], [3]],
          # When an error occurs in a batch, it affects all of the other
          # requests in that batch (but not requests from other batches).
          'expected_results': ['error2', 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
      {
          'testcase_name': 'no_error_batching_disabled',
          'inputs': [1, 2, 3],
          'enable_batching': False,
          'expected_processed': [[1], [2], [3]],
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_disabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': False,
          'expected_processed': [[1], ['error2'], [3]],
          # When an error occurs in a batch, it affects all of the other
          # requests in that batch (but not requests from other batches).
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
  )
  def test_exception_handling_batch_method(
      self,
      inputs,
      enable_batching,
      expected_processed,
      expected_results,
      expected_error_message_if_not_caught,
  ):
    processed = []

    @batching.add_batching
    class C:

      @batching.batch_method(batch_size=2)
      def process_batch(
          self, requests: Sequence[dict[str, Any]]
      ) -> Sequence[int]:
        requests = [r['request'] for r in requests]
        processed.append(
            [(str(r) if isinstance(r, Exception) else r) for r in requests]
        )
        replies = []
        for r in requests:
          if isinstance(r, Exception):
            raise r
          else:
            replies.append(r)
        return replies

      @executing.make_executable  # pytype: disable=wrong-arg-types
      @batching.batchable_method(
          implementation=utils.FromInstance('process_batch')
      )
      def process(self, request: int | Exception) -> int:
        del request
        pass  # pytype: disable=bad-return-type

      @executing.make_executable  # pytype: disable=wrong-arg-types
      async def process_with_error_handling(
          self, x: int | Exception
      ) -> int | str:
        try:
          return await self.process(x)
        except Exception as e:  # pylint: disable=broad-except
          return str(e)

    async def process_all_without_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int]:
      c = C()
      executables = [c.process(x) for x in inputs]
      return await executing.parallel(*executables)

    async def process_all_with_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int | str]:
      c = C()
      executables = [c.process_with_error_handling(x) for x in inputs]
      return await executing.parallel(*executables)

    results = batching.run(
        process_all_with_error_handling(inputs),
        enable_batching=enable_batching,
    )
    with self.subTest('processed_with_correct_batching'):
      self.assertSequenceEqual(
          expected_processed, processed, pprint.pformat(processed)
      )
    with self.subTest('returns_correct_results'):
      self.assertSequenceEqual(
          expected_results, results, pprint.pformat(results)
      )

    if expected_error_message_if_not_caught is not None:
      with self.subTest('raises_informative_error_if_not_caught'):
        # Here we verify that if the caller does not implement any error
        # handling, the original error gets raised (the same as it would if no
        # batching were applied), rather than getting masked by an error like
        # "ValueError: Attempting to finish a non-empty queue".
        with self.assertRaisesRegex(
            ValueError, expected_error_message_if_not_caught
        ):
          batching.run(
              process_all_without_error_handling(inputs),
              enable_batching=enable_batching,
          )

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_error_batching_enabled',
          'inputs': [1, 2, 3],
          'enable_batching': True,
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_enabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': True,
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
      {
          'testcase_name': 'no_error_batching_disabled',
          'inputs': [1, 2, 3],
          'enable_batching': False,
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_disabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': False,
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
  )
  def test_exception_handling_batch_function_with_threadpool(
      self,
      inputs,
      enable_batching,
      expected_results,
      expected_error_message_if_not_caught,
  ):
    @executing.make_executable  # pytype: disable=wrong-arg-types
    @batching.batch_function_with_threadpool(batch_size=2)
    def process(request: int | Exception) -> int:
      if isinstance(request, Exception):
        raise request
      else:
        return request

    @executing.make_executable  # pytype: disable=wrong-arg-types
    async def process_with_error_handling(x: int | Exception) -> int | str:
      try:
        return await process(x)
      except Exception as e:  # pylint: disable=broad-except
        return str(e)

    async def process_all_without_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int]:
      executables = [process(x) for x in inputs]
      return await executing.parallel(*executables)

    async def process_all_with_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int | str]:
      executables = [process_with_error_handling(x) for x in inputs]
      return await executing.parallel(*executables)

    results = batching.run(
        process_all_with_error_handling(inputs), enable_batching=enable_batching
    )
    self.assertSequenceEqual(expected_results, results, pprint.pformat(results))

    if expected_error_message_if_not_caught is not None:
      with self.subTest('raises_informative_error_if_not_caught'):
        # Here we verify that if the caller does not implement any error
        # handling, the original error gets raised (the same as it would if no
        # batching were applied), rather than getting masked by an error like
        # "ValueError: Attempting to finish a non-empty queue".
        with self.assertRaisesRegex(
            ValueError, expected_error_message_if_not_caught
        ):
          batching.run(
              process_all_without_error_handling(inputs),
              enable_batching=enable_batching,
          )

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_error_batching_enabled',
          'inputs': [1, 2, 3],
          'enable_batching': True,
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_enabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': True,
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
      {
          'testcase_name': 'no_error_batching_disabled',
          'inputs': [1, 2, 3],
          'enable_batching': False,
          'expected_results': [1, 2, 3],
          'expected_error_message_if_not_caught': None,
      },
      {
          'testcase_name': 'single_error_batching_disabled',
          'inputs': [1, ValueError('error2'), 3],
          'enable_batching': False,
          'expected_results': [1, 'error2', 3],
          'expected_error_message_if_not_caught': 'error2',
      },
  )
  def test_exception_handling_batch_method_with_threadpool(
      self,
      inputs,
      enable_batching,
      expected_results,
      expected_error_message_if_not_caught,
  ):
    @batching.add_batching
    class C:

      @executing.make_executable  # pytype: disable=wrong-arg-types
      @batching.batch_method_with_threadpool(batch_size=2)
      def process(self, request: int | Exception) -> int:
        if isinstance(request, Exception):
          raise request
        else:
          return request

      @executing.make_executable  # pytype: disable=wrong-arg-types
      async def process_with_error_handling(
          self, x: int | Exception
      ) -> int | str:
        try:
          return await self.process(x)
        except Exception as e:  # pylint: disable=broad-except
          return str(e)

    async def process_all_without_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int]:
      c = C()
      executables = [c.process(x) for x in inputs]
      return await executing.parallel(*executables)

    async def process_all_with_error_handling(
        inputs: Sequence[int | Exception],
    ) -> Sequence[int | str]:
      c = C()
      executables = [c.process_with_error_handling(x) for x in inputs]
      return await executing.parallel(*executables)

    results = batching.run(
        process_all_with_error_handling(inputs), enable_batching=enable_batching
    )
    self.assertSequenceEqual(expected_results, results, pprint.pformat(results))

    if expected_error_message_if_not_caught is not None:
      with self.subTest('raises_informative_error_if_not_caught'):
        # Here we verify that if the caller does not implement any error
        # handling, the original error gets raised (the same as it would if no
        # batching were applied), rather than getting masked by an error like
        # "ValueError: Attempting to finish a non-empty queue".
        with self.assertRaisesRegex(
            ValueError, expected_error_message_if_not_caught
        ):
          batching.run(
              process_all_without_error_handling(inputs),
              enable_batching=enable_batching,
          )

  def test_to_thread_pool_method(self):
    class C:
      processed = []

      @batching.to_thread_pool_method(num_workers=2)
      def process(self, request: int, sleep_time: int = 0) -> int:
        time.sleep(sleep_time)
        self.processed.append(request)
        return request

    async def process_all(
        inputs: Sequence[tuple[int, int]],
    ) -> tuple[Sequence[int], Sequence[int]]:
      c = C()
      coroutines = [c.process(*x) for x in inputs]
      return await asyncio.gather(*coroutines), c.processed

    results, processed = asyncio.run(
        process_all([(1, 1), (2, 0), (3, 0), (4, 1)])
    )
    with self.subTest('results_are_correct'):
      self.assertEqual([1, 2, 3, 4], results)

    with self.subTest('processing_order_is_correct'):
      self.assertEqual([2, 3, 1, 4], processed)

  def test_to_thread_pool_method_cascading(self):
    """Test thread pools with parallel/sequential calls."""

    class C:
      def __init__(self):
        self.submitted = []
        self.processed = []
        self.lock = threading.Lock()

      @batching.to_thread_pool_method(num_workers=2)
      def process(self, request: int, sleep_time: int = 0) -> int:
        with self.lock:
          self.submitted.append(request)
        time.sleep(sleep_time)
        with self.lock:
          self.processed.append(request)
        return request

      async def sequential(
          self, inputs: Sequence[tuple[int, int]]
      ) -> Sequence[int]:
        res = []
        for x in inputs:
          res.append(await self.process(*x))
        return res

      async def parallel(
          self, inputs: Sequence[tuple[int, int]]
      ) -> Sequence[Sequence[int]]:
        coroutines = [
            self.sequential(inputs[i : i + 2])
            for i in range(0, len(inputs) - 1, 2)
        ]
        return await asyncio.gather(*coroutines)

    c = C()
    results = asyncio.run(
        c.parallel([(1, 1), (2, 1), (3, 0), (4, 1)])
    )
    with self.subTest('results_are_correct'):
      self.assertEqual([[1, 2], [3, 4]], results)

    with self.subTest('submitting_order_is_correct'):
      # We are processing (1,2) in parallel with (3,4)
      # so we will create first the tasks 1 and 3,
      # and 3 will return first, at which point we will create the task 4,
      # then 1 will return, at which point we create the task 2.
      self.assertEqual([1, 3, 4, 2], c.submitted)

    with self.subTest('processing_order_is_correct'):
      # Since we submit tasks in the order 1, 3, 4, 2 and
      # we use two workers, we will get the results in the order 3 (after
      # 0 seconds), 1 (after 1 second), 4, 2.
      self.assertEqual([3, 1, 4, 2], c.processed)

  def test_to_thread_pool_method_made_executable(self):
    class C:
      processed = []

      @executing.make_executable  # pytype: disable=wrong-arg-types
      @batching.to_thread_pool_method(num_workers=2)
      def process(self, request: int, sleep_time: int = 0) -> int:
        time.sleep(sleep_time)
        self.processed.append(request)
        return request

    c = C()
    executables = [c.process(*x) for x in [(1, 1), (2, 0), (3, 0), (4, 1)]]
    results = batching.run(executing.parallel(*executables))
    with self.subTest('results_are_correct'):
      self.assertSequenceEqual([1, 2, 3, 4], results)

    with self.subTest('processing_order_is_correct'):
      self.assertEqual([2, 3, 1, 4], c.processed)

  @parameterized.named_parameters(
      ('return_exceptions', True),
      ('no_return_exceptions', False),
  )
  def test_to_thread_pool_method_with_exceptions(
      self, return_exceptions
  ):
    class C:

      @executing.make_executable  # pytype: disable=wrong-arg-types
      @batching.to_thread_pool_method(num_workers=2)
      def process(self, request: int) -> int:
        if request == 2:
          raise ValueError('error')
        else:
          return request

    c = C()
    executables = [c.process(x) for x in [1, 2, 3]]
    if return_exceptions:
      results = batching.run(
          executing.parallel(*executables, return_exceptions=return_exceptions)
      )
      self.assertLen(results, 3)
      self.assertIsInstance(results[1], ValueError)
      self.assertSequenceEqual(
          [1, 3], [r for r in results if not isinstance(r, ValueError)]
      )
    else:
      with self.assertRaisesRegex(ValueError, 'error'):
        batching.run(
            executing.parallel(
                *executables, return_exceptions=return_exceptions
            )
        )


if __name__ == '__main__':
  absltest.main()
