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

from collections.abc import AsyncIterator, Sequence
import functools
import inspect
from typing import Any, TypeVar
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import builtins_base
from onetwo.core import executing
from onetwo.core import routing


_T = TypeVar('_T')
_T2 = TypeVar('_T2')


class ExecutableAssertions(unittest.TestCase):
  """Mixin class for executing.Executable assertions."""

  # pylint: disable=invalid-name
  def assertExecutableResultEqual(
      self,
      executable: executing.Executable,
      value: Any,
  ) -> None:
    executable_value = executing.run(executable)
    msg = f'Expected {value}, got `{executable_value}'
    return self.assertEqual(executable_value, value, msg)


def normal_function(a: int) -> int:
  return a


def normal_function_that_returns_iterable(a: int) -> Sequence[int]:
  return (a + 1, a + 2, a + 3)


async def coroutine_function(a: int) -> int:
  return a


async def asyncgenerator_function(a: int) -> AsyncIterator[int]:
  for el in (a + 1, a + 2, a + 3):
    yield el


def normal_function_with_typevar(a: _T, b: list[_T]) -> _T:
  return a + b[0]


@executing.make_executable()
async def make_executable_function(a: int) -> int:
  return a


@executing.make_executable()
async def make_executable_async_generator(a: int) -> AsyncIterator[int]:
  for el in (a + 1, a + 2, a + 3):
    yield el


class BaseTest(parameterized.TestCase, ExecutableAssertions):

  @parameterized.named_parameters(
      ('works_with_normal_function', normal_function, 1),
      (
          'works_with_normal_function_that_returns_iterable',
          normal_function_that_returns_iterable,
          (2, 3, 4),
      ),
      ('works_with_coroutine_function', coroutine_function, 1),
      (
          'works_with_asyncgenerator_function',
          asyncgenerator_function,
          4,  # Last yielded value.
      ),
      ('works_with_make_executable_function', make_executable_function, 1),
      (
          'works_with_make_executable_async_generator',
          make_executable_async_generator,
          4,  # Last yielded value.
      ),
  )
  def test_configure_with_function(self, impl, expected_output):
    @builtins_base.Builtin[int]
    def fn(a: int) -> int:
      del a
      raise NotImplementedError('This is just a signature.')

    fn.configure(impl)

    self.assertExecutableResultEqual(fn(1), expected_output)

  def test_configurable_with_typevar(self):
    @builtins_base.Builtin[int]
    def fn(a: _T2, b: list[_T2]) -> int:
      del a, b
      raise NotImplementedError('This is just a signature.')

    fn.configure(normal_function_with_typevar)

    self.assertExecutableResultEqual(fn(1, [2, 3]), 3)

  def test_configure_with_partial(self):
    """Verifies error gets raised when configuring with functools.partial."""
    @builtins_base.Builtin[int]
    def fn(a: int) -> int:
      del a
      raise NotImplementedError('This is just a signature.')

    def fn_impl(a: int, b: int) -> int:
      del a, b
      raise NotImplementedError('Not needed.')

    with self.assertRaisesRegex(
        ValueError,
        'Looks like `configure` method of a builtin was called with',
    ):
      fn.configure(functools.partial(fn_impl, b=0))

  def test_configure_with_method(self):
    @builtins_base.Builtin[int]
    def fn(a: int) -> int:
      del a
      raise NotImplementedError('This is just a signature.')

    class Provider:
      def add_one(self, a: int) -> int:
        return a + 1

      @executing.make_executable
      def add_one_decorated(self, a: int) -> int:
        return a + 1

    provider = Provider()

    # The following command ends up calling `executing.make_executable` on the
    # implementation and follows the branch `elif inspect.ismethod(function):`.
    fn.configure(provider.add_one)

    with self.subTest('works_without_make_executable_decorator'):
      self.assertExecutableResultEqual(fn(1), 2)

    fn.configure(provider.add_one_decorated)

    with self.subTest('works_with_make_executable_decorator'):
      self.assertExecutableResultEqual(fn(1), 2)

  def test_binds_at_execution(self):
    @builtins_base.Builtin[int]
    def fn(a: int, b: int) -> int:
      del a, b
      raise NotImplementedError('This is just a signature.')

    def f1(a: int, b: int = 0) -> int:
      return a + 1 + b

    def f2(a: int, b: int = 0) -> int:
      return a - 1 + b

    # We want to ensure that if we switch the function after creating the
    # executable, we will point to the new function.
    fn.configure(f1)
    e = fn(1)
    with self.subTest('executes_with_f1'):
      self.assertExecutableResultEqual(e, 2)
    fn.configure(f2)
    with self.subTest('executes_with_f2'):
      self.assertExecutableResultEqual(e, 0)

    # Similarly, if we change the defaults, these are taken into account.
    fn.configure(f1, b=1)
    with self.subTest('executes_with_f1_and_b_is_1'):
      self.assertExecutableResultEqual(e, 3)

  def test_signature(self):
    @builtins_base.Builtin[int]
    def fn(a: int, b: str = '') -> int:
      del b
      return a

    def correct(a: int, b: str = '') -> int:
      del b
      return a + 1

    def correct_nodefault(a: int) -> int:
      return a + 1

    def correct_additional(a: int, c: int) -> int:
      return a + c

    def correct_additional_optional(a: int, c: int = 2) -> int:
      return a + c

    def incorrect_return(a: int, b: str = '') -> str:
      del b
      return str(a + 1)

    def incorrect_missing(b: str = '') -> int:
      del b
      return 1

    def incorrect_type(a: str) -> int:
      return int(a) + 1

    with self.subTest('calls_the_right_function'):
      fn.configure(correct)
      self.assertExecutableResultEqual(fn(1), 2)

      fn.configure(correct_nodefault)
      self.assertExecutableResultEqual(fn(1), 2)
      # Test that we can pass arguments that are not implemented
      # (they will be ignored).
      self.assertExecutableResultEqual(fn(a=1, b=''), 2)

    with self.subTest('configure_error_if_additional_args_have_no_values'):
      with self.assertRaisesRegex(
          ValueError,
          'does not have. The value of this non-optional ',
      ):
        fn.configure(correct_additional)

    with self.subTest('configure_correctly_if_additional_args_have_values'):
      fn.configure(correct_additional_optional)
      self.assertExecutableResultEqual(fn(a=1), 3)
      fn.configure(correct_additional, c=2)
      self.assertExecutableResultEqual(fn(a=1), 3)

    fn.configure(correct_additional_optional)
    with self.subTest('configure_error_when_args_not_in_base_signature'):
      with self.assertRaisesRegex(
          ValueError,
          'Unknown argument c passed when calling the builtin',
      ):
        _ = executing.run(fn(a=1, c=2))

    with self.subTest('accepts_wrong_return_type'):
      # Note that the return type is not checked at decoration time.
      # We disable the static type check.
      fn.configure(incorrect_return)  # pytype: disable=wrong-arg-types
      # This should work even though the return type did not match
      # the original signature.
      self.assertExecutableResultEqual(fn(1, 1), '2')

    with self.subTest('raises_error_if_signature_does_not_match'):
      with self.assertRaisesRegex(ValueError, 'missing'):
        fn.configure(incorrect_missing)

      with self.assertRaisesRegex(ValueError, 'with type'):
        fn.configure(incorrect_type)

  def test_defaults(self):
    @builtins_base.Builtin[int]
    def fn(a: int, b: int | None = None, c: int = 0) -> int:
      del a, b, c
      return 0

    def add(a: int, b: int | None = None, c: int = 0) -> int:
      return a + b + c

    with self.subTest('can_not_update_if_not_configured'):
      with self.assertRaises(ValueError):
        fn.update(c=1)

    with self.subTest('canfigure_unknown_args_raises_error'):
      with self.assertRaisesRegex(
          ValueError,
          'Trying to set a default value for argument d',
      ):
        fn.configure(add, d=1)
    fn.configure(add, b=1)
    with self.subTest('configured_b_value_is_used'):
      self.assertExecutableResultEqual(fn(1, c=2), 4)
      self.assertExecutableResultEqual(fn(1), 2)

    with self.subTest('configured_b_value_is_overridden'):
      self.assertExecutableResultEqual(fn(1, c=2, b=3), 6)
      self.assertExecutableResultEqual(fn(1, b=3), 4)

    fn.update(b=2)
    with self.subTest('updated_b_value_is_used'):
      self.assertExecutableResultEqual(fn(1, c=2), 5)
      self.assertExecutableResultEqual(fn(1), 3)

    with self.subTest('update_unknown_args_raises_error'):
      with self.assertRaisesRegex(
          ValueError,
          'Trying to set a default value for argument e',
      ):
        fn.update(e=1)

    fn.configure(add, c=1)
    with self.subTest('None_parameter_must_have_a_default'):
      with self.assertRaises(TypeError):
        _ = executing.run(fn(1))
      self.assertExecutableResultEqual(fn(1, 2), 4)

    fn.configure(add, b=1, c=1)
    with self.subTest('multiple_values_configured'):
      self.assertExecutableResultEqual(fn(1), 3)
      self.assertExecutableResultEqual(fn(1, b=2), 4)

    fn.update(c=2)
    with self.subTest('multiple_values_configured'):
      self.assertExecutableResultEqual(fn(1), 4)
      self.assertExecutableResultEqual(fn(1, b=2), 5)

    with self.subTest('None_value_reverts_to_default'):
      # If None is passed as the value of a parameter but we had set a default
      # for it, we use the default.
      self.assertExecutableResultEqual(fn(1, b=None), 4)

  def test_extra_args(self):
    @builtins_base.Builtin[int]
    def fn(a: int, b: int | None = None) -> int:
      del a, b
      return 0

    def implementation(a: int, b: int | None = None, **kwargs) -> int:
      return a + b + kwargs['c']

    fn.configure(implementation, a=1)
    with self.subTest('should_pass_extra_args'):
      self.assertExecutableResultEqual(fn(b=2, c=3), 6)

    with self.subTest('accepts_unused_args'):
      self.assertExecutableResultEqual(fn(b=2, c=3, d=4), 6)

    with self.subTest('override_preset_vars'):
      self.assertExecutableResultEqual(fn(a=2, b=2, c=3), 7)

  @parameterized.named_parameters(
      ('with_defaults', {'c': 2}, 4),
      ('without_defaults', {}, 2),
  )
  def test_executable(self, defaults, expected):
    """Tests that we can configure a Builtin function with an executable."""
    @builtins_base.Builtin[int]
    def fn(a: int, b: int | None = None, c: int = 0) -> int:
      del a, b, c
      return 0

    @executing.make_executable
    def add(a: int, b: int | None = None, c: int = 0) -> int:
      return a + b + c

    fn.configure(add, b=1)  # The `add` executable will be used.
    self.assertExecutableResultEqual(fn(1, **defaults), expected)

  def test_wrapping(self):
    """Tests that we the properties of the wrapped function."""
    @builtins_base.Builtin[int]
    def fn(a: int, *, b: int | None = None, c: int = 0) -> int:
      del b, c
      return a

    arguments = list(inspect.signature(fn).parameters.keys())
    self.assertListEqual(arguments, ['a', 'b', 'c'])
    self.assertEqual(fn.__name__, 'fn')

  def test_configure_update_copy_registry(self):
    @builtins_base.Builtin[int]
    def fn(a: int, b: int | None = None, c: int = 0) -> int:
      del a, b, c
      return 0

    def add(a: int, b: int | None = None, c: int = 0) -> int:
      return a + b + c

    # We create 2 registries with different defaults.
    fn.configure(add, b=1)
    r1 = routing.copy_registry()
    fn.update(b=2)
    r2 = routing.copy_registry()
    # We create an executable to run in two different contexts.
    executable = fn(1, c=2)
    # We switch the context and store the results for each context.
    with routing.RegistryContext():
      routing.set_registry(r1)
      res1 = executing.run(executable)
    with routing.RegistryContext():
      routing.set_registry(r2)
      res2 = executing.run(executable)
    with routing.RegistryContext():
      routing.set_registry(r1)
      res3 = executing.run(executable)
    self.assertEqual(res1, 4)
    self.assertEqual(res2, 5)
    self.assertEqual(res3, 4)

if __name__ == '__main__':
  absltest.main()
