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

"""Core content from `executing.py` factored out to avoid circular dependencies.

The intention is to keep the content in this file to the minimum needed to avoid
circular dependencies between `executing.py` and other low-level libraries like
`utils.py` or `batching.py`. In particular, we should minimize the dependencies
from this file to other OneTwo libraries.

The existence of this file should be treated as an implementation detail from
the perspective of general users of OneTwo. So rather than importing this file
directly, most code (both inside and outside of the OneTwo core) should instead
import `executing.py` and/or `utils.py`, which contain public aliases to the
classes and functions defined here.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
import functools
import inspect
from typing import Any, final, Generic, ParamSpec, TypeVar

from onetwo.core import updating


# Basic type variables that need to be specified when using this library.
_Result = TypeVar('_Result')
_Args = ParamSpec('_Args')

_Update = updating.Update


class Executable(
    Generic[_Result],
    Awaitable[_Result],
    AsyncIterator[_Update[_Result]],
    metaclass=abc.ABCMeta,
):
  """Interface for a process that can be executed step by step.

  Executable supports both `await` and `async for` statements. If the underlying
  implementation supports only `await`, we wrap it into the async iterator that
  yields only that one item. Similarly, if the underlying implementation is an
  async iterator and only supports `async for`, we implement `await` by manually
  iterating through all of the values and returning the final one.

  When awaited, executable returns a value of type Result. However, when using
  `async for`, executable produces instances of class (or subclasses of)
  `updating.Update`. Class Update helps maintaining (`accumulate`) intermediate
  resuts and obtaining the final result (`to_result`) based on accumulated
  information.
  """

  @abc.abstractmethod
  async def _aexec(self) -> _Result:
    """Implementation as an Awaitable."""

  @abc.abstractmethod
  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[_Update[_Result]]:
    """Implementation as an AsyncIterator with configurable depth."""
    yield _Update()  # For correct typing

  @final
  def with_depth(self, iteration_depth: int) -> AsyncIterator[_Update[_Result]]:
    return self._aiterate(iteration_depth)

  @final
  def __await__(self) -> Generator[Any, Any, _Result]:
    """Method that is called when using `await executable`."""
    result = yield from self._aexec().__await__()
    return result

  @final
  def __aiter__(self) -> AsyncIterator[_Update[_Result]]:
    """Method that is called when using `async for executable`."""
    return self._aiterate().__aiter__()

  @final
  async def __anext__(self) -> _Update[_Result]:
    """Method that is called when using `async for executable`."""
    return await self._aiterate().__anext__()


def set_decorated_with_make_executable(f: Callable[..., Any]) -> None:
  """Marks the callable as being decorated with @executing.make_executable.

  Args:
    f: An arbitrary callable.
  """
  f.decorated_with_make_executable = True


def is_decorated_with_make_executable(f: Callable[..., Any]) -> bool:
  """Returns whether the callable is decorated with @executing.make_executable.

  Args:
    f: An arbitrary callable.
  """
  # Special handling for partial functions.
  while isinstance(f, functools.partial):
    f = f.func

  # Special handling for callable objects (e.g., sub-classes of Agent).
  if (
      hasattr(f, '__call__')
      and not inspect.isfunction(f)
      and not inspect.ismethod(f)
  ):
    f = f.__call__

  return getattr(f, 'decorated_with_make_executable', False)


def returns_awaitable(f: Callable[..., Any]) -> bool:
  """Returns whether the callable returns something that is awaitable.

  Note that while using this function is more reliable than calling
  `inspect.iscoroutinefunction` directly, there are still some cases that it
  does not catch, such as when `f` is manually implemented to return an
  Executable, without using the `@executing.make_executable` decorator. To
  catch these cases, it is recommended to use `call_and_maybe_await` where
  possible, rather than depending on `returns_awaitable`.

  Args:
    f: An arbitrary callable.
  """
  if (
      hasattr(f, '__call__')
      and not inspect.isfunction(f)
      and not inspect.ismethod(f)
      and not isinstance(f, functools.partial)
  ):
    f = f.__call__

  return inspect.iscoroutinefunction(f) or is_decorated_with_make_executable(f)


async def call_and_maybe_await(
    f: Callable[_Args, _Result | Awaitable[_Result] | Executable[_Result]],
    *args: _Args.args,
    **kwargs: _Args.kwargs,
) -> _Result:
  """"Calls the callable and awaits the result if appropriate."""
  if returns_awaitable(f):
    result = await f(*args, **kwargs)
  else:
    result = f(*args, **kwargs)
  # The below condition is needed in case `f` was manually implemented to return
  # an Executable, without using the `@executing.make_executable` decorator.
  if isinstance(result, Executable):
    result = await result
  return result
