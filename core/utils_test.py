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
import functools
import multiprocessing.pool
import time

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import utils


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
          'f_kwargs', False,
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
      return await asyncio.gather(*[af(i) for i in range(start, start+5)])

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
    class TestClass:
      @utils.rate_limit_method(2.0)
      def f(self, i: int):
        return i

      @utils.rate_limit_method(2.0)
      async def af(self, i: int):
        return i

    # We create two instances to check that each instance is rate-limited
    # separately.
    t1 = TestClass()
    t2 = TestClass()

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

    class TestClass:  # pylint: disable=unused-variable
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


if __name__ == '__main__':
  absltest.main()
