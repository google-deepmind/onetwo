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
import copy
import dataclasses
import functools
import multiprocessing.pool
import time
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from onetwo.core import content as content_lib
from onetwo.core import utils
import PIL.Image


_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList
_Message: TypeAlias = content_lib.Message


def f_empty():
  return None


def f_pos_kw(x, unused_y):
  return x


def f_kwargs(unused_x, **other):
  return other['y']


def f_varpos(*args, unused_y=0):
  return args[0]


def f_complex(a, b, /, c, d='d', *args, e='e', f, **kwargs):
  return a + b + c + d + e + f + args[0] + kwargs['h']


def direct_call(f, args, kwargs):
  return f(*args, **kwargs)


def indirect_call(f, args, kwargs):
  exp_args = utils.get_expanded_arguments(f, True, args, kwargs)
  call_args, call_kwargs = utils.get_calling_args_and_kwargs(f, exp_args)
  return f(*call_args, **call_kwargs)


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('f_empty', False, f_empty, [], {}, {}, None),
      (
          'f_pos_kw',
          False,
          f_pos_kw,
          [1, 2],
          {},
          {'x': 1, 'unused_y': 2},
          1,
      ),
      (
          'f_pos_kw2',
          False,
          f_pos_kw,
          [1],
          {'unused_y': 2},
          {'x': 1, 'unused_y': 2},
          1,
      ),
      (
          'f_kwargs',
          False,
          f_kwargs,
          [1],
          {'y': 2},
          {'unused_x': 1, 'y': 2},
          2,
      ),
      (
          'f_varpos_with_defaults',
          True,
          f_varpos,
          [1, 2],
          {},
          {'args': (1, 2), 'unused_y': 0},
          1,
      ),
      (
          'f_varpos',
          False,
          f_varpos,
          [1, 2],
          {},
          {'args': (1, 2)},
          1,
      ),
      (
          'f_complex',
          False,
          f_complex,
          ['a', 'b', 'c', 'd2', 'd3'],
          {'e': 'e2', 'f': 'f2', 'h': 'h2'},
          {
              'a': 'a',
              'args': ('d3',),
              'b': 'b',
              'c': 'c',
              'd': 'd2',
              'e': 'e2',
              'f': 'f2',
              'h': 'h2',
          },
          'abcd2e2f2d3h2',
      ),
      (
          'f_complex_with_defaults',
          True,
          f_complex,
          ['a', 'b', 'c', 'd2', 'd3'],
          {'f': 'f2', 'h': 'h2'},
          {
              'a': 'a',
              'args': ('d3',),
              'b': 'b',
              'c': 'c',
              'd': 'd2',
              'e': 'e',
              'f': 'f2',
              'h': 'h2',
          },
          'abcd2ef2d3h2',
      ),
  )
  def test_get_args(
      self,
      use_defaults,
      fun,
      args,
      kwargs,
      expected,
      expected_call_res,
  ):
    result = utils.get_expanded_arguments(fun, use_defaults, args, kwargs)
    with self.subTest('should_return_correct_result'):
      self.assertEqual(result, expected)

    with self.subTest('should_give_the_right_return_value'):
      direct_res = direct_call(fun, args, kwargs)
      self.assertEqual(direct_res, expected_call_res)

    with self.subTest('should_give_the_right_return_value_with_indirect_call'):
      indirect_res = indirect_call(fun, args, kwargs)
      self.assertEqual(indirect_res, expected_call_res)

  @parameterized.named_parameters(
      ('include_all_set', True, (1, 2, 'a'), {}, {'a': 1, 'b': 2, 'c': 'a'}),
      (
          'include_some_set',
          True,
          (),
          {'a': 1, 'c': 'a'},
          {'a': 1, 'b': 0, 'c': 'a'},
      ),
      ('exclude_all_set', False, (1, 2, 'a'), {}, {'a': 1, 'b': 2, 'c': 'a'}),
      (
          'exclude_some_set',
          False,
          (),
          {'a': 1, 'c': 'a'},
          {'a': 1, 'c': 'a'},
      ),
  )
  def test_defaults(self, include_defaults, args, kwargs, expected):
    def f(a, b: int = 0, c: str = ''):
      del b, c
      return a

    result = utils.get_expanded_arguments(f, include_defaults, args, kwargs)
    self.assertEqual(result, expected)

  def test_from_instance(self):
    class C:

      def __init__(self, test_attribute: str):
        self.test_attribute = test_attribute

    with self.subTest('can_retrieve_by_name'):
      param = utils.FromInstance(name='test_attribute')
      self.assertEqual('x', param(C('x')))

    with self.subTest('can_retrieve_by_function'):
      param = utils.FromInstance(function=lambda x: x.test_attribute * 2)
      self.assertEqual('xx', param(C('x')))

    with self.subTest('FROM_INSTANCE_CLASS_NAME_retrieves_class_name'):
      self.assertEqual('C', utils.FROM_INSTANCE_CLASS_NAME(C('x')))

    with self.subTest('raises_error_if_name_and_function_both_specified'):
      with self.assertRaisesRegex(ValueError, 'Only one of'):
        param = utils.FromInstance(
            name='test_attribute', function=lambda x: x.__class__.__name__
        )
        param(C('x'))

    with self.subTest('raises_error_if_name_and_function_are_both_missing'):
      with self.assertRaisesRegex(ValueError, 'Only one of'):
        param = utils.FromInstance()
        param(C('x'))

  def test_rate_limit_function(self):
    @utils.rate_limit_function(2.0)
    def f(i: int):
      return i

    @utils.rate_limit_function(2.0)
    async def af(i: int):
      return i

    async def plan():
      return await asyncio.gather(*[af(i) for i in range(10)])

    start = time.perf_counter()
    res = []
    for i in range(10):
      res.append(f(i))
    end = time.perf_counter()

    with self.subTest('blocking_should_return_correct_results'):
      self.assertListEqual(res, list(range(10)))

    with self.subTest('should_limit_blocking_calls'):
      # We run 10 queries with 2qps, it should take no less than 4.5s.
      self.assertLess(4.5, end - start)

    start = time.perf_counter()
    res = asyncio.run(plan())
    end = time.perf_counter()

    with self.subTest('async_should_return_correct_results'):
      self.assertListEqual(res, list(range(10)))

    with self.subTest('should_limit_async_calls'):
      # We run 10 queries with 2qps, it should take no less than 4.5s.
      self.assertLess(4.5, end - start)

  def test_threaded_rate_limit_function(self):
    @utils.rate_limit_function(2.0)
    def f(i: int):
      return i

    @utils.rate_limit_function(2.0)
    async def af(i: int):
      return i

    async def plan(start):
      return await asyncio.gather(*[af(i) for i in range(start, start + 5)])

    start = time.perf_counter()
    with multiprocessing.pool.ThreadPool(10) as pool:
      res = list(pool.map(f, list(range(10))))
      pool.close()
      pool.join()
    end = time.perf_counter()

    with self.subTest('blocking_should_return_correct_results'):
      self.assertListEqual(res, list(range(10)))

    with self.subTest('should_limit_blocking_calls'):
      # We run 10 queries with 2qps, it should take no less than 4.5s.
      self.assertLess(4.5, end - start)

    start = time.perf_counter()
    with multiprocessing.pool.ThreadPool(10) as pool:
      res = list(
          pool.map(lambda x: asyncio.run(plan(x)), list(range(0, 20, 5)))
      )
      pool.close()
      pool.join()
    end = time.perf_counter()

    with self.subTest('async_should_return_correct_results'):
      self.assertListEqual(
          res, [list(range(start, start + 5)) for start in range(0, 20, 5)]
      )

    with self.subTest('should_limit_async_calls'):
      # We run 20 queries with 2qps, it should take no less than 9.5s.
      self.assertLess(9.5, end - start)

  def test_rate_limit_method(self):
    class ClassForTest:

      @utils.rate_limit_method(2.0)
      def f(self, i: int):
        return i

      @utils.rate_limit_method(2.0)
      async def af(self, i: int):
        return i

    # We create two instances to check that each instance is rate-limited
    # separately.
    t1 = ClassForTest()
    t2 = ClassForTest()

    async def plan(instance):
      return await asyncio.gather(*[instance.af(i) for i in range(10)])

    async def run_both():
      return await asyncio.gather(plan(t1), plan(t2))

    start = time.perf_counter()
    res = []
    for i in range(0, 20, 2):
      res.append(t1.f(i))
      res.append(t2.f(i + 1))
    end = time.perf_counter()

    with self.subTest('blocking_should_return_correct_results'):
      self.assertListEqual(res, list(range(20)))

    with self.subTest('should_limit_blocking_calls'):
      # We run 20 queries with 2qps, on two objects, it should take ~5s
      self.assertBetween(end - start, 5, 6)

    start = time.perf_counter()
    res = asyncio.run(run_both())
    end = time.perf_counter()

    with self.subTest('async_should_return_correct_results'):
      self.assertListEqual(res, [list(range(10))] * 2)

    with self.subTest('should_limit_async_calls'):
      # We run 20 queries with 2qps, on two objects, it should take ~5s.
      self.assertBetween(end - start, 5, 6)

  def test_with_retry_sync(self):
    max_retries = 12
    # We set the delays to be small, so that the test doesn't take too long.
    initial_base_delay = 0.01
    max_base_delay = 0.05

    @dataclasses.dataclass
    class FailOnFirstNCalls:
      n: int = 0

      @utils.with_retry(
          max_retries=max_retries,
          initial_base_delay=initial_base_delay,
          max_base_delay=max_base_delay,
      )
      def __call__(self, value: str) -> str:
        if self.n > 0:
          self.n -= 1
          raise ValueError(f'error: {self.n} tries left')
        return value

    with self.subTest('should_succeed_when_retrying_enough_times'):
      self.assertEqual('x', FailOnFirstNCalls(n=max_retries)('x'))

    with self.subTest('should_raise_underlying_error_when_not_enough_retries'):
      with self.assertRaisesRegex(ValueError, 'error'):
        FailOnFirstNCalls(n=max_retries + 1)('x')

    with self.subTest('should_introduce_no_delay_if_no_retry_is_needed'):
      start_time = time.perf_counter()
      _ = FailOnFirstNCalls(n=0)('x')
      end_time = time.perf_counter()
      self.assertLess(end_time - start_time, 0.5 * initial_base_delay)

    with self.subTest('first_retry_should_take_around_initial_base_delay'):
      # More precisely, the delay should be a random value somewhere between
      # initial_base_delay and 2 * initial_base_delay.
      start_time = time.perf_counter()
      _ = FailOnFirstNCalls(n=1)('x')
      end_time = time.perf_counter()
      self.assertGreaterEqual(end_time - start_time, initial_base_delay)
      self.assertLess(end_time - start_time, 2.1 * initial_base_delay)

    with self.subTest('second_retry_should_double_the_base_delay'):
      # This is because we increase the base delay after two consecutive errors.
      start_time = time.perf_counter()
      _ = FailOnFirstNCalls(n=2)('x')
      end_time = time.perf_counter()
      # Minimum: (1 + 2) * initial_base_delay
      self.assertGreaterEqual(end_time - start_time, 3 * initial_base_delay)
      # Maximum delay is double the minimum delay (due to the randomization).
      self.assertLess(end_time - start_time, 6.1 * initial_base_delay)

    with self.subTest('third_retry_should_keep_base_delay_constant'):
      # We don't increase the base delay in this case because we only increase
      # the base delay after two consecutive errors at the same base delay.
      start_time = time.perf_counter()
      _ = FailOnFirstNCalls(n=3)('x')
      end_time = time.perf_counter()
      # Minimum: (1 + 2 + 2) * initial_base_delay
      self.assertGreaterEqual(end_time - start_time, 5 * initial_base_delay)
      # Maximum delay is double the minimum delay (due to the randomization).
      self.assertLess(end_time - start_time, 10.1 * initial_base_delay)

    with self.subTest('base_delay_should_be_capped_at_max_base_delay'):
      # This is because we increase the base delay after two consecutive errors.
      start_time = time.perf_counter()
      _ = FailOnFirstNCalls(n=10)('x')
      end_time = time.perf_counter()
      self.assertLessEqual(end_time - start_time, 2 * 10 * max_base_delay)

  def test_with_retry_async(self):
    max_retries = 12
    # We set the delays to be small, so that the test doesn't take too long.
    initial_base_delay = 0.01
    max_base_delay = 0.05

    @dataclasses.dataclass
    class FailOnFirstNCalls:
      n: int = 0

      @utils.with_retry(
          max_retries=max_retries,
          initial_base_delay=initial_base_delay,
          max_base_delay=max_base_delay,
      )
      async def __call__(self, value: str) -> str:
        if self.n > 0:
          self.n -= 1
          raise ValueError(f'error: {self.n} tries left')
        return value

    with self.subTest('should_succeed_when_retrying_enough_times'):
      self.assertEqual('x', asyncio.run(FailOnFirstNCalls(n=max_retries)('x')))

    with self.subTest('should_raise_underlying_error_when_not_enough_retries'):
      with self.assertRaisesRegex(ValueError, 'error'):
        asyncio.run(FailOnFirstNCalls(n=max_retries + 1)('x'))

    with self.subTest('should_introduce_no_delay_if_no_retry_is_needed'):
      start_time = time.perf_counter()
      _ = asyncio.run(FailOnFirstNCalls(n=0)('x'))
      end_time = time.perf_counter()
      self.assertLess(end_time - start_time, 0.5 * initial_base_delay)

    with self.subTest('first_retry_should_take_around_initial_base_delay'):
      # More precisely, the delay should be a random value somewhere between
      # initial_base_delay and 2 * initial_base_delay.
      start_time = time.perf_counter()
      _ = asyncio.run(FailOnFirstNCalls(n=1)('x'))
      end_time = time.perf_counter()
      self.assertGreaterEqual(end_time - start_time, initial_base_delay)
      self.assertLess(end_time - start_time, 2.1 * initial_base_delay)

    with self.subTest('second_retry_should_double_the_base_delay'):
      # This is because we increase the base delay after two consecutive errors.
      start_time = time.perf_counter()
      _ = asyncio.run(FailOnFirstNCalls(n=2)('x'))
      end_time = time.perf_counter()
      # Minimum: (1 + 2) * initial_base_delay
      self.assertGreaterEqual(end_time - start_time, 3 * initial_base_delay)
      # Maximum delay is double the minimum delay (due to the randomization).
      self.assertLess(end_time - start_time, 6.1 * initial_base_delay)

    with self.subTest('third_retry_should_keep_base_delay_constant'):
      # We don't increase the base delay in this case because we only increase
      # the base delay after two consecutive errors at the same base delay.
      start_time = time.perf_counter()
      _ = asyncio.run(FailOnFirstNCalls(n=3)('x'))
      end_time = time.perf_counter()
      # Minimum: (1 + 2 + 2) * initial_base_delay
      self.assertGreaterEqual(end_time - start_time, 5 * initial_base_delay)
      # Maximum delay is double the minimum delay (due to the randomization).
      self.assertLess(end_time - start_time, 10.1 * initial_base_delay)

    with self.subTest('base_delay_should_be_capped_at_max_base_delay'):
      # This is because we increase the base delay after two consecutive errors.
      start_time = time.perf_counter()
      _ = asyncio.run(FailOnFirstNCalls(n=10)('x'))
      end_time = time.perf_counter()
      self.assertLessEqual(end_time - start_time, 2 * 10 * max_base_delay)

  @parameterized.named_parameters(
      ('str', 'key1', False, b'key1'),
      ('list', [1, 2, 3, 1.0, True, 'a'], False, b"[1, 2, 3, 1.0, True, 'a']"),
      ('tuple', (1, 2, 3), False, b'(1, 2, 3)'),
      ('set', {'b', 'a'}, False, b"['a', 'b']"),
      (
          'dict',
          {'b': 2, 'a': 1, 'c': (1.0, 2.0), 'e': True, 'd': 's', 'f': [1, 'a']},
          False,
          (
              b"[('a', 1), ('b', 2), ('c', (1.0, 2.0)), ('d', 's'), ('e',"
              b" True), ('f', [1, 'a'])]"
          ),
      ),
      ('bytes', b'a', False, b'a'),
      (
          'deep_recursion',
          [{'a': 1, 'b': 2}, ([1, 2], {'c': 3, 'b': b'abc', 1: 2})],
          False,
          (
              b"[[('a', 1), ('b', 2)], ([1, 2], [('1', 2), ('b', b'abc'), ('c',"
              b' 3)])]'
          ),
      ),
      ('chunk', _Chunk('abc'), False, b'abc'),
      (
          'chunk_list',
          _ChunkList(chunks=[_Chunk('ab'), _Chunk(b'cd'), _Chunk('ef')]),
          False,
          b"['ab', b'cd', 'ef']",
      ),
      (
          'message',
          _Message(content='abc', role='user'),
          False,
          b'Message(role=user, content=abc)',
      ),
      (
          'messages',
          [
              _Message(content='abc', role='user'),
              _Message(content='def', role='model'),
          ],
          False,
          (
              b"['Message(role=user, content=abc)', 'Message(role=model,"
              b" content=def)']"
          ),
      ),
      (
          'messages_with_bytes',
          [
              _Message(
                  content=_ChunkList([_Chunk(b'abc', content_type='bytes')]),
                  role='user',
              ),
              _Message(content='def', role='user'),
          ],
          False,
          (
              b"[\"Message(role=user, content=[b'abc'])\", 'Message(role=user,"
              b" content=def)']"
          ),
      ),
      (
          'np.array',
          np.array([1, 2, 3]),
          False,
          b'\x01\x00\x00\x00\x00\x00\x00\x00'
          + b'\x02\x00\x00\x00\x00\x00\x00\x00'
          + b'\x03\x00\x00\x00\x00\x00\x00\x00',
      ),
      ('str_with_fallback', 'key1', True, b'key1'),
      (
          'list_with_fallback',
          [1, 2, 3, 1.0, True, 'a'],
          True,
          b"[1, 2, 3, 1.0, True, 'a']",
      ),
      ('tuple_with_fallback', (1, 2, 3), True, b'(1, 2, 3)'),
      ('set_with_fallback', {'b', 'a'}, True, b"['a', 'b']"),
      (
          'dict_with_fallback',
          {'b': 2, 'a': 1, 'c': (1.0, 2.0), 'e': True, 'd': 's', 'f': [1, 'a']},
          False,
          (
              b"[('a', 1), ('b', 2), ('c', (1.0, 2.0)), ('d', 's'), ('e',"
              b" True), ('f', [1, 'a'])]"
          ),
      ),
      ('bytes_with_fallback', b'a', True, b'a'),
      ('chunk_with_fallback', _Chunk('abc'), True, b'abc'),
      (
          'chunk_list_with_fallback',
          _ChunkList(chunks=[_Chunk('ab'), _Chunk(b'cd'), _Chunk('ef')]),
          True,
          b'abcdef',
      ),
      (
          'message_with_fallback',
          _Message(content='abc', role='user'),
          True,
          b"Message(role='user', content='abc')",
      ),
      (
          'messages_with_fallback',
          [
              _Message(content='abc', role='user'),
              _Message(content='def', role='model'),
          ],
          True,
          (
              b"[Message(role='user', content='abc'),"
              b" Message(role='model', content='def')]"
          ),
      ),
      (
          'messages_with_bytes_with_fallback',  # Does not use fallback.
          [
              _Message(
                  content=_ChunkList([_Chunk(b'abc', content_type='bytes')]),
                  role='user',
              ),
              _Message(content='def', role='user'),
          ],
          True,
          (
              b"[\"Message(role=user, content=[b'abc'])\", 'Message(role=user,"
              b" content=def)']"
          ),
      ),
      (
          'message_with_bytes_with_fallback',  # Does not use fallback.
          _Message(
              content=_ChunkList([_Chunk(b'abc', content_type='bytes')]),
              role='user',
          ),
          True,
          b"Message(role=user, content=[b'abc'])",
      ),
      (
          'np.array_with_fallback',
          np.array([1, 2, 3]),
          True,
          b'\x01\x00\x00\x00\x00\x00\x00\x00'
          + b'\x02\x00\x00\x00\x00\x00\x00\x00'
          + b'\x03\x00\x00\x00\x00\x00\x00\x00',
      ),
  )
  def test_get_bytes_for_hashing(self, key, fallback_if_safe, expected_bytes):
    got = utils.get_bytes_for_hashing(key, fallback_if_safe)
    self.assertEqual(expected_bytes, got, got)

  @parameterized.named_parameters(
      ('str', 'key1', 'key1', 'key2'),
      (
          'list',
          [1, 2, 3, True, 1.0],
          [1, 2, 3, True, 1.0],
          [3, 2, 1, True, 1.0],
      ),
      ('set', {'a', 'b'}, {'b', 'a'}, {'a', 'c'}),
      ('dict', {'a': 1, 'b': 2}, {'b': 2, 'a': 1}, {'a': 1, 'b': 3}),
      ('bytes', b'a', b'a', b'b'),
      (
          'pil',
          PIL.Image.new(mode='RGB', size=(2, 2)),
          PIL.Image.new(mode='RGB', size=(2, 2)),
          PIL.Image.new(mode='RGB', size=(2, 3)),
      ),
      ('chunk', _Chunk('abc'), _Chunk('abc'), _Chunk('cba')),
      (
          'chunk_with_pil',
          _Chunk(PIL.Image.new(mode='RGB', size=(2, 2))),
          _Chunk(PIL.Image.new(mode='RGB', size=(2, 2))),
          _Chunk(PIL.Image.new(mode='RGB', size=(2, 3))),
      ),
      (
          'chunk_list',
          _ChunkList(chunks=[_Chunk('ab'), _Chunk(b'ab'), _Chunk('bc')]),
          _ChunkList(chunks=[_Chunk('ab'), _Chunk(b'ab'), _Chunk('bc')]),
          _ChunkList(chunks=[_Chunk('bc'), _Chunk(b'ab'), _Chunk('ab')]),
      ),
      (
          'message',
          _Message(content='abc', role='user'),
          _Message(content='abc', role='user'),
          _Message(content='cba', role='user'),
      ),
      (
          'messages_with_roles',
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'def', content_type='bytes')]),
                  role='user',
              ),
          ],
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'def', content_type='bytes')]),
                  role='user',
              ),
          ],
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'def', content_type='bytes')]),
                  role='assistant',
              ),
          ],
      ),
      (
          'messages_with_bytes',
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'def', content_type='bytes')]),
                  role='user',
              ),
          ],
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'def', content_type='bytes')]),
                  role='user',
              ),
          ],
          [
              _Message(content='abc', role='user'),
              _Message(
                  content=_ChunkList([_Chunk(b'xyz', content_type='bytes')]),
                  role='user',
              ),
          ],
      ),
      (
          'np.array',
          np.array([1, 2, 3]),
          np.array([1, 2, 3]),
          np.array([3, 2, 1]),
      ),
  )
  def test_get_hash(self, key, similar_key, other_key):
    # We create a simple copy of the key.
    c = copy.deepcopy(key)
    key_hash = utils.get_str_hash(key)
    copy_hash = utils.get_str_hash(c)
    similar_hash = utils.get_str_hash(similar_key)
    other_hash = utils.get_str_hash(other_key)
    # We check we obtain the same hash (this is not a very strong test
    # as ideally one would compare hashes on different machines etc... but
    # this is a first attempt at checking that some reasonable hash is created).
    self.assertEqual(copy_hash, key_hash)
    self.assertEqual(similar_hash, key_hash)
    # We also check that the hashes are different for different keys (this
    # would catch mistakes such as accidentally mapping all keys to the same
    # hash).
    self.assertNotEqual(other_hash, key_hash)

  def test_is_method(self):
    results = {}

    def decorator(f):
      nonlocal results
      results[f.__name__] = utils.is_method(f)
      return f

    def wrapping_decorator(f):
      @functools.wraps(f)
      def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

      return wrapper

    def incorrect_wrapping_decorator(f):
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

      return wrapper

    @decorator
    def simple_function():  # pylint: disable=unused-variable
      return None

    class ClassForTest:  # pylint: disable=unused-variable

      @decorator
      def method(self):
        return None

      @decorator
      @wrapping_decorator
      def wrapped_method(self):
        return None

      @decorator
      @incorrect_wrapping_decorator
      def incorrect_wrapped_method(self):
        return None

    self.assertDictEqual(
        results,
        {
            'simple_function': False,
            'method': True,
            'wrapped_method': True,
            'incorrect_wrapped_method': False,
        },
    )

  def test_returning_raised_exception(self):
    @utils.returning_raised_exception
    def f(x: int, y: str):
      raise ValueError(f'error: {x}, {y}')

    @utils.returning_raised_exception
    async def af(x: int, y: str):
      raise ValueError(f'error: {x}, {y}')

    class C:

      @utils.returning_raised_exception
      def f(self, x: int, y: str):
        raise ValueError(f'error: {x}, {y}')

      @utils.returning_raised_exception
      async def af(self, x: int, y: str):
        raise ValueError(f'error: {x}, {y}')

    o = C()

    with self.subTest('regular_function'):
      result = f(1, y='a')
      self.assertEqual(ValueError, type(result))
      self.assertEqual('error: 1, a', str(result))

    with self.subTest('async_function'):
      result = asyncio.run(af(1, y='a'))
      self.assertEqual(ValueError, type(result))
      self.assertEqual('error: 1, a', str(result))

    with self.subTest('regular_method'):
      result = o.f(1, y='a')
      self.assertEqual(ValueError, type(result))
      self.assertEqual('error: 1, a', str(result))

    with self.subTest('async_method'):
      result = asyncio.run(o.af(1, y='a'))
      self.assertEqual(ValueError, type(result))
      self.assertEqual('error: 1, a', str(result))

  def test_raising_returned_exception(self):
    # Here we verify that `raising_returned_exception` acts as a direct inverse
    # of `returning_raised_exception`.
    @utils.raising_returned_exception
    @utils.returning_raised_exception
    def f(x: int, y: str):
      raise ValueError(f'error: {x}, {y}')

    @utils.raising_returned_exception
    @utils.returning_raised_exception
    async def af(x: int, y: str):
      raise ValueError(f'error: {x}, {y}')

    class C:

      @utils.raising_returned_exception
      @utils.returning_raised_exception
      def f(self, x: int, y: str):
        raise ValueError(f'error: {x}, {y}')

      @utils.raising_returned_exception
      @utils.returning_raised_exception
      async def af(self, x: int, y: str):
        raise ValueError(f'error: {x}, {y}')

    o = C()

    with self.subTest('regular_function'):
      with self.assertRaisesRegex(ValueError, 'error: 1, a'):
        f(1, y='a')

    with self.subTest('async_function'):
      with self.assertRaisesRegex(ValueError, 'error: 1, a'):
        asyncio.run(af(1, y='a'))

    with self.subTest('regular_method'):
      with self.assertRaisesRegex(ValueError, 'error: 1, a'):
        o.f(1, y='a')

    with self.subTest('async_method'):
      with self.assertRaisesRegex(ValueError, 'error: 1, a'):
        asyncio.run(o.af(1, y='a'))


if __name__ == '__main__':
  absltest.main()
