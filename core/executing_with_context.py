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

"""Define Executables that can be concatenated and carry over context."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
import copy
import dataclasses
from typing import Any, Generic, TypeVar, Protocol

from onetwo.core import executing
from typing_extensions import override


# Represents the context type of an ExecutableWithContext
_C = TypeVar('_C')
# Represents the output type of an ExecutableWithContext
_T = TypeVar('_T')


class AddableSequence(Protocol):
  """Protocol for sequences of objects that can be added with a + operator."""

  def __add__(self, other: Any) -> AddableSequence:
    ...

_A = TypeVar('_A', bound=AddableSequence)


@dataclasses.dataclass
class ExecutableWithContext(
    Generic[_C, _T], executing.Executable[_T], metaclass=abc.ABCMeta
):
  """Executable that can use/modify a context object as it executes."""
  _final_context: _C | None = dataclasses.field(default=None, init=False)
  _executed: bool = dataclasses.field(default=False, init=False)
  _result: _T | None = dataclasses.field(default=None, init=False)
  _stored_initial_context: _C | None = dataclasses.field(
      default=None, init=False
  )

  @abc.abstractmethod
  def initialize_context(self, *args, **kwargs) -> _C:
    """Create a context object from the specified arguments."""

  @classmethod
  @abc.abstractmethod
  def wrap(cls, other: _T) -> ExecutableWithContext[_C, _T]:
    """Create a new ExecutableWithContext from an arbitrary object.

    This is used for example when we want to compose ExecutableWithContext
    by having functions that can take in either a basic type (e.g. a string) or
    an ExecutableWithContext that will be executed before passing in the result.
    This will define the behaviour of passing a basic type (how does the
    context gets modified by consuming this basic type).

    Args:
      other: Arbitrary object that we know how to wrap into an
        ExecutableWithContext.

    Returns:
      ExecutableWithContext whose execution output will be the basic object.
    """

  @classmethod
  @abc.abstractmethod
  def get_result(cls, context: _C) -> _T:
    """Extract the final result from the context."""

  @classmethod
  async def maybe_execute(cls, content: _T, context: _C) -> _T:
    if isinstance(content, ExecutableWithContext):
      return await content.execute(context)
    else:
      return await cls.wrap(content).execute(context)

  @classmethod
  async def maybe_iterate(
      cls,
      content: Any,
      context: _C,
      iteration_depth: int = 1,
  ) -> AsyncIterator[_T]:
    if isinstance(content, ExecutableWithContext):
      async for res in content.iterate(context, iteration_depth):
        yield res
    else:
      yield (await cls.maybe_execute(content, context))

  @abc.abstractmethod
  @executing.make_executable
  async def execute(
      self,
      context: _C,
  ) -> _T:
    ...

  @executing.make_executable
  async def iterate(
      self, context: _C, iteration_depth: int = 1
  ) -> AsyncIterator[_T]:
    # Default implementation: single step execution.
    del iteration_depth
    res = await self.execute(context)
    yield res

  @override
  async def _aexec(self) -> _T:
    if self._stored_initial_context is None:
      context = self.initialize_context()
    else:
      context = copy.deepcopy(self._stored_initial_context)
    _ = await self.execute(context)
    return self.get_result(context)

  @override
  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[_T]:
    if iteration_depth == 0:
      res = await self
      yield res
    else:
      if self._stored_initial_context is None:
        context = self.initialize_context()
      else:
        context = copy.deepcopy(self._stored_initial_context)
      current_result = self.get_result(context)
      async for _ in self.iterate(
          context=context, iteration_depth=iteration_depth
      ):
        new_result = self.get_result(context)
        if current_result != new_result:
          yield new_result
          current_result = new_result

  def __call__(self, *args, **kwargs) -> ExecutableWithContext[_C, _T]:
    """Convenience function to store the context before execution."""
    self._stored_initial_context = self.initialize_context(*args, **kwargs)
    return self


@dataclasses.dataclass
class SerialExecutableWithContext(
    Generic[_C, _A], ExecutableWithContext[_C, _A]
):
  """Represents a Sequence of ExecutableWithContext to be executed serially.

  When executing this object, each of the nodes will be executed one after
  the other and the context will be passed around (and possibly modified by
  each node). The results of each of the nodes will be put in a list.
  This class supports concatenation with the `+` operator.

  Attributes:
    nodes: List of ExecutableWithContext.
  """

  nodes: list[ExecutableWithContext[_C, _A]] = dataclasses.field(
      default_factory=list
  )

  @classmethod
  @abc.abstractmethod
  def empty_result(cls) -> _A:
    """Returns an empty list of nodes."""

  def __post_init__(self):
    if self.nodes is None:
      self.nodes = self.empty_result()

  @override
  @executing.make_executable
  async def execute(self, context: _C) -> _A:
    if len(self.nodes) == 1:
      result = await self.nodes[0].execute(context)
    else:
      result = self.empty_result()
      for node in self.nodes:
        node_result = await node.execute(context)
        try:
          result += node_result
        except ValueError:
          # This is in case we have a result which is a generic python object
          # and cannot be added to the ChunkList.
          result += str(node_result)
    self._final_context = copy.deepcopy(context)
    self._executed = True
    self._result = result
    return result

  @executing.make_executable
  async def iterate(
      self, context: _C, iteration_depth: int = 1
  ) -> AsyncIterator[_A]:
    if iteration_depth == 0:
      yield (await self.execute(context))
    else:
      current_result = self.empty_result()
      for node in self.nodes:
        node_result = None
        async for update in node.iterate(context, iteration_depth - 1):
          node_result = update
          yield current_result + update
        if node_result is not None:
          current_result += node_result
      self._result = current_result
      self._final_context = copy.deepcopy(context)
      self._executed = True

  def __iadd__(self, other: Any) -> SerialExecutableWithContext:
    match other:
      case SerialExecutableWithContext():
        self.nodes += other.nodes
      case _:
        self.nodes.append(self.wrap(other))
    return self

  # Pytype complains that we try to instantiate an abstract class, but when
  # called on a concrete child this works.
  # pytype: disable=not-instantiable
  def __add__(self, other: Any) -> SerialExecutableWithContext:
    match other:
      case SerialExecutableWithContext():
        return self.__class__(self.nodes + other.nodes)
      case _:
        return self.__class__(self.nodes + [self.wrap(other)])

  def __radd__(self, other: Any) -> SerialExecutableWithContext:
    return self.__class__([self.wrap(other)] + self.nodes)
  # pytype: enable=not-instantiable

