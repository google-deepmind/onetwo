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
import itertools
import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import iterating


class IteratingTest(parameterized.TestCase):

  def test_asyncio_run_wrapper_outside_loop(self):
    async def f(i: int):
      if i:
        return i
      else:
        raise ValueError('zero')

    with self.subTest('runs_without_error'):
      self.assertEqual(iterating.asyncio_run_wrapper(f(1)), 1)

    with self.subTest('raises_error'):
      with self.assertRaisesRegex(ValueError, 'zero'):
        iterating.asyncio_run_wrapper(f(0))

  def test_asyncio_run_wrapper_inside_loop(self):

    async def outer_wrapper():
      async def f(i: int):
        if i:
          return i
        else:
          raise ValueError('zero')

      with self.subTest('runs_without_error'):
        self.assertEqual(iterating.asyncio_run_wrapper(f(1)), 1)

      with self.subTest('raises_error'):
        with self.assertRaisesRegex(ValueError, 'zero'):
          iterating.asyncio_run_wrapper(f(0))

      with self.subTest('direct_asyncio_fails'):
        with self.assertRaises(RuntimeError):
          asyncio.run(f(1))

    asyncio.run(outer_wrapper())

  def test_async_iterator_to_sync(self):
    async def wrap():
      for i in range(5):
        yield i

    result = []
    with iterating.async_iterator_to_sync(wrap()) as iterator:
      for i in iterator:
        result.append(i)

    self.assertListEqual(result, list(range(5)), result)

  def test_async_iterator_to_sync_with_error_in_iterator(self):
    # Even though it looks like the exception should be raised from the iterator
    # to the main code, because the iterator is actually executed in a thread,
    # this will not happen automatically, so we test that the mechanism of
    # catching the exception inside the sub thread and propagating it to the
    # main thread works.
    async def wrap():
      for i in range(5):
        yield i
        raise ValueError()

    with self.assertRaises(ValueError):
      result = []
      with iterating.async_iterator_to_sync(wrap()) as iterator:
        for i in iterator:
          result.append(i)

  @parameterized.named_parameters(
      # Note that KeyboardInterrupt is not a subclass of Exception, so
      # additional care is needed to make sure it is handled.
      ('KeyboardInterrupt', KeyboardInterrupt),
      ('ValueError', ValueError),
  )
  def test_async_iterator_to_sync_with_error_in_loop(self, exception_type):
    # We make sure that the context manager that produces the iterator properly
    # handles the exception and promptly and cleanly terminates the thread
    # that it started.
    iteration_when_terminated = -2
    max_iterations = 100
    async def wrap():
      nonlocal iteration_when_terminated
      i = -1
      try:
        for i in range(max_iterations):
          yield i
      finally:
        iteration_when_terminated = i

    with self.assertRaises(exception_type):
      with  iterating.async_iterator_to_sync(wrap()) as iterator:
        for _ in iterator:
          raise exception_type()

    with self.subTest('child_thread_is_terminated_promptly'):
      # Since the stop signal is sent to the child thread asynchronously, we
      # cannot guarantee the exact iteration at which the child thread will be
      # terminated, but it should be very near to the beginning (i.e., long
      # before the halfway point).
      self.assertLess(iteration_when_terminated, max_iterations / 2)

  def test_function_with_callback_to_async_iterator(self):
    def f(cb):
      for i in range(5):
        cb(i)

    async def wrap():
      res = []
      with iterating.function_with_callback_to_async_iterator(f) as it:
        async for i in it:
          res.append(i)
      return res

    result = asyncio.run(wrap())
    self.assertListEqual(result, list(range(5)), result)

  def test_function_with_callback_with_error_in_callback(self):
    # Even though it looks like the exception should be raised from the callback
    # to the main code, because the callback is actually executed in a thread,
    # this will not happen automatically, so we test that the mechanism of
    # catching the exception inside the sub thread and propagating it to the
    # main thread works.
    def f(cb):
      for i in range(5):
        cb(i)
        raise ValueError()

    async def wrap():
      res = []
      with iterating.function_with_callback_to_async_iterator(f) as it:
        async for i in it:
          res.append(i)
      return res

    with self.assertRaises(ValueError):
      _ = asyncio.run(wrap())

  def test_function_with_callback_with_error_in_loop(self):
    # We make sure that the context manager that produces the iterator properly
    # handles exceptions and terminates the thread that it started cleanly.
    def f(cb):
      for i in range(5):
        cb(i)

    async def wrap():
      res = []
      with iterating.function_with_callback_to_async_iterator(f) as it:
        async for i in it:
          res.append(i)
          raise ValueError()
      return res

    with self.assertRaises(ValueError):
      _ = asyncio.run(wrap())

  def test_sync_iterator_to_async(self):
    def it():
      for i in range(5):
        yield i

    async def wrap():
      res = []
      with iterating.sync_iterator_to_async(it()) as iterator:
        async for i in iterator:
          res.append(i)
      return res

    result = asyncio.run(wrap())
    self.assertListEqual(result, list(range(5)), result)

  def test_sync_iterator_to_async_with_error_in_iterator(self):
    # Even though it looks like the exception should be raised from the iterator
    # to the main code, because the iterator is actually executed in a thread,
    # this will not happen automatically, so we test that the mechanism of
    # catching the exception inside the sub thread and propagating it to the
    # main thread works.
    def it():
      for i in range(5):
        yield i
        raise ValueError()

    async def wrap():
      res = []
      with iterating.sync_iterator_to_async(it()) as iterator:
        async for i in iterator:
          res.append(i)
      return res

    with self.assertRaises(ValueError):
      _ = asyncio.run(wrap())

  def test_sync_iterator_to_async_with_error_in_loop(self):
    # We make sure that the context manager that produces the iterator properly
    # handles exceptions and terminates the thread that it started cleanly.
    def it():
      for i in range(5):
        yield i

    async def wrap():
      res = []
      with iterating.sync_iterator_to_async(it()) as iterator:
        async for i in iterator:
          res.append(i)
          raise ValueError()
      return res

    with self.assertRaises(ValueError):
      _ = asyncio.run(wrap())

  def test_merge(self):

    async def iterator():
      for i in range(5):
        yield i

    async def wrapper():
      result = []
      async for res, index in iterating.merge(
          iterator(), iterator(), iterator()
      ):
        result.append((res, index))
      return result

    result = asyncio.run(wrapper())
    self.assertListEqual(
        result, list(itertools.product(range(5), range(3))), result
    )

  def test_merge_iter(self):
    iterations = 3
    chunk_size = 2
    num_iterators = 10

    async def iterator():
      for i in range(iterations):
        yield i

    all_iter = (iterator() for _ in range(num_iterators))

    async def wrapper():
      result = []
      async for res, index in iterating.merge_iter(
          all_iter, chunk_size=chunk_size
      ):
        result.append((res, index))
      return result

    chunks = (
        range(i, i + chunk_size) for i in range(0, num_iterators, chunk_size)
    )
    expected = list(
        itertools.chain(
            *[itertools.product(range(iterations), c) for c in chunks]
        )
    )
    result = asyncio.run(wrapper())
    self.assertListEqual(result, expected, result)

  def test_merge_exceptions(self):
    iterations = 3
    num_iterators = 4

    async def iterator(num: int):
      for i in range(iterations):
        if (i + num) % 4 == 0:
          raise ValueError(f'{i}')
        yield i

    all_iter = (iterator(i) for i in range(num_iterators))

    async def wrapper():
      per_iterator = [[] for _ in range(num_iterators)]
      async for res, index in iterating.merge_iter(
          all_iter, return_exceptions=True
      ):
        per_iterator[index].append(res)
      return per_iterator

    expected = [
        [ValueError('0')],
        [0, 1, 2],
        [0, 1, ValueError('2')],
        [0, ValueError('1')],
    ]
    result = asyncio.run(wrapper())
    with self.subTest('stops_at_exception_without_failing'):
      self.assertListEqual([str(r) for r in result], [str(r) for r in expected])

  def test_to_thread_function(self):
    result = []
    lock = threading.Lock()

    def safe_append(a):
      with lock:
        result.append(a)

    @iterating.to_thread
    def f():
      for i in range(5):
        time.sleep(1)
        safe_append(i)

    async def wrap():
      await asyncio.gather(f(), f())

    expected = list(itertools.chain(*zip(range(5), range(5))))
    asyncio.run(wrap())
    self.assertListEqual(result, expected, result)

  def test_to_thread_with_timeout_success(self):
    timeout = 1.0

    @iterating.to_thread(timeout=timeout)
    def func():
      return 42

    self.assertEqual(asyncio.run(func()), 42)

  def test_to_thread_times_out(self):
    timeout = 1.0

    @iterating.to_thread(timeout=timeout)
    def func():
      time.sleep(2 * timeout)

    with self.assertRaises(asyncio.exceptions.TimeoutError):
      asyncio.run(func())

  def test_to_thread_iterator(self):
    result = []

    @iterating.to_thread_iterator
    def f():
      for i in range(5):
        time.sleep(1)
        yield i

    merged = iterating.merge(f(), f())

    async def wrap():
      async for res, index in merged:
        result.append((res, index))
      return result

    expected = list(itertools.product(range(5), range(2)))
    asyncio.run(wrap())
    self.assertListEqual(result, expected, result)

if __name__ == '__main__':
  absltest.main()
