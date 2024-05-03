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

import functools

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import agents_test_utils
from onetwo.core import executing
from onetwo.core import executing_impl
from onetwo.core import tracing


def ordinary_function(x, y):
  return x + y


async def async_function(x, y):
  return x + y


@executing.make_executable
def executable_function(x, y):
  return x + y


@executing.make_executable
async def async_executable_function(x, y):
  return x + y


@tracing.trace
def traced_ordinary_function(x, y):
  return x + y


@tracing.trace
async def traced_async_function(x, y):
  return x + y


@executing.make_executable
@tracing.trace
def executable_traced_function(x, y):
  return x + y


@tracing.trace
@executing.make_executable
def traced_executable_function(x, y):
  return x + y


def executable_function_that_does_not_use_make_executable_decorator(x, y):
  return executing.serial(executable_function(x, y), executable_function(x, y))


class C:
  def ordinary_method(self, x, y):
    return x + y

  async def async_method(self, x, y):
    return x + y

  @executing.make_executable
  def executable_method(self, x, y):
    return x + y

  @executing.make_executable
  async def async_executable_method(self, x, y):
    return x + y


class ExecutingImplTest(parameterized.TestCase):

  def test_set_decorated_with_make_executable(self):
    def f(x):
      return x

    with self.subTest('false_before_setting'):
      self.assertFalse(executing_impl.is_decorated_with_make_executable(f))

    executing_impl.set_decorated_with_make_executable(f)

    with self.subTest('true_after_setting'):
      self.assertTrue(executing_impl.is_decorated_with_make_executable(f))

  @parameterized.named_parameters(
      ('ordinary_function', ordinary_function, False),
      ('async_function', async_function, False),
      ('executable_function', executable_function, True),
      ('async_executable_function', async_executable_function, True),
      ('traced_ordinary_function', traced_ordinary_function, False),
      ('traced_async_function', traced_async_function, False),
      ('executable_traced_function', executable_traced_function, True),
      ('traced_executable_function', traced_executable_function, True),
      ('agent', agents_test_utils.StringAgent(), True),
      ('ordinary_method', C().ordinary_method, False),
      ('async_method', C().async_method, False),
      ('executable_method', C().executable_method, True),
      ('async_executable_method', C().async_executable_method, True),
      (
          'executable_function_that_does_not_use_make_executable_decorator',
          executable_function_that_does_not_use_make_executable_decorator,
          False,
      ),
  )
  def test_is_decorated_with_make_executable(self, function, expected_result):
    self.assertEqual(
        expected_result,
        executing_impl.is_decorated_with_make_executable(function),
    )

  @parameterized.named_parameters(
      ('ordinary_function', ordinary_function, False),
      ('async_function', async_function, True),
      ('executable_function', executable_function, True),
      ('async_executable_function', async_executable_function, True),
      ('traced_ordinary_function', traced_ordinary_function, False),
      ('traced_async_function', traced_async_function, True),
      ('executable_traced_function', executable_traced_function, True),
      ('traced_executable_function', traced_executable_function, True),
      ('agent', agents_test_utils.StringAgent(), True),
      ('ordinary_method', C().ordinary_method, False),
      ('async_method', C().async_method, True),
      ('executable_method', C().executable_method, True),
      ('async_executable_method', C().async_executable_method, True),
      # TODO: The following case should ideally return True but does
      # not currently, as it is difficult to predict that the function returns
      # an Executable in this case without actually calling it.
      (
          'executable_function_that_does_not_use_make_executable_decorator',
          executable_function_that_does_not_use_make_executable_decorator,
          False,
      ),
  )
  def test_returns_awaitable(self, function, expected_result):
    self.assertEqual(
        expected_result,
        executing_impl.returns_awaitable(function),
    )

  @parameterized.named_parameters(
      ('ordinary_function', ordinary_function),
      ('async_function', async_function),
      ('executable_function', executable_function),
      ('async_executable_function', async_executable_function),
      ('traced_ordinary_function', traced_ordinary_function),
      ('traced_async_function', traced_async_function),
      ('executable_traced_function', executable_traced_function),
      ('traced_executable_function', traced_executable_function),
      ('ordinary_method', C().ordinary_method),
      ('async_method', C().async_method),
      ('executable_method', C().executable_method),
      ('async_executable_method', C().async_executable_method),
  )
  def test_call_and_maybe_await(self, function):
    result = executing.run(
        executing_impl.call_and_maybe_await(function, 'x', 'y')
    )
    with self.subTest('positional_args'):
      self.assertEqual('xy', result)

    result = executing.run(
        executing_impl.call_and_maybe_await(function, x='x', y='y')
    )
    with self.subTest('keyword_args'):
      self.assertEqual('xy', result)

    partial_function = functools.partial(function, x='x')
    result = executing.run(
        executing_impl.call_and_maybe_await(partial_function, y='y')
    )
    with self.subTest('partial_function'):
      self.assertEqual('xy', result)

  def test_call_and_maybe_await_agent(self):
    function = agents_test_utils.StringAgent(sequence=['x', 'y'])
    result = executing.run(
        executing_impl.call_and_maybe_await(function, 'some_input')
    )
    self.assertEqual('x y', result)

  def test_call_and_maybe_await_executable_function_that_does_not_use_make_executable_decorator(
      self,
  ):
    # Note that this case is handled correctly, even though `returns_awaitable`
    # does not return the correct value for this function.
    function = executable_function_that_does_not_use_make_executable_decorator
    result = executing.run(
        executing_impl.call_and_maybe_await(function, 'x', 'y')
    )
    self.assertEqual(['xy', 'xy'], result)


if __name__ == '__main__':
  absltest.main()
