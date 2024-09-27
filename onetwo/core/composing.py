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

"""Define Composable objects as Executables that can be concatenated.

TODO.

This is inspired by the Guidance library where one can create prompts for an
llm with expressions like
```
llm += 'some prompt' + gen()
```
So one can compose functions with strings via the `+` operator and obtain
new functions which, upon execution actually query the LLM.

Here we introduce a Composable class which inherits from Executable and returns,
upon execution, objects of type ChunkList (that can be added together,
like strings
or lists). These Composable objects can themselves be chained together with the
`+` operator.

So for example if `gen()` is a function returning a Composable object, one can
do
```
executable = 'some prompt' + gen()
onetwo.run(executable)
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Sequence
import contextlib
import copy
import dataclasses
import functools
import inspect
from typing import Any, TypeAlias, TypeVar

from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import executing_with_context
from onetwo.core import updating
from onetwo.core import utils
from typing_extensions import override


_T = TypeVar('_T')

Chunk = content_lib.Chunk
ChunkList = content_lib.ChunkList


def _safe_add(
    left: ChunkList, right: ChunkList
) -> ChunkList:
  """Append two sequences together and flatten the right one."""
  if isinstance(right, list):
    result = left
    for element in right:
      result = _safe_add(result, element)
    return result
  else:
    if left is None:
      return right
    if right is None:
      return left
    return left + right


@dataclasses.dataclass
class Context:
  """Context of execution for a Composable.

  This class carries the various variables that are processed by the successive
  elements of the Composable.
  Each element may read from or write into this context.
  It can be used as plain dict[str, Any] to hold values of variables.

  Attributes:
    prefix: The incrementally built prefix that results from executing the
      Composable from left to right.
    chunk_list: Incrementally built list of chunks, i.e. results from each part.
  """

  _variables: dict[str, Any] = dataclasses.field(default_factory=dict)
  prefix: ChunkList = dataclasses.field(default_factory=ChunkList)

  @property
  def variables(self) -> dict[str, Any]:
    """Returns the variables in this context."""
    return self._variables

  def to_messages(self) -> list[content_lib.Message]:
    """Converts current prefix into a list of Messages to be used with llm.chat.

    Chunks with no role are assumed to have the USER role. Each consecutive
    collection of chunks with the same role are grouped into a single Message.
    Messages containing just a single string chunk are normalized to represent
    the content as the string itself rather than a ChunkList of length 1.

    Returns:
      List of messages.
    """
    current_role = content_lib.PredefinedRole.USER
    current_chunk_list = []
    messages = []
    for chunk in self.prefix:
      chunk.role = chunk.role or content_lib.PredefinedRole.USER
      if chunk.role == current_role:
        current_chunk_list.append(chunk)
      else:
        if current_chunk_list:
          messages.append(
              content_lib.Message.create_normalized(
                  role=current_role, content=current_chunk_list)
          )
        current_chunk_list = [chunk]
        current_role = chunk.role
    if current_chunk_list:
      messages.append(
          content_lib.Message.create_normalized(
              role=current_role,
              content=content_lib.ChunkList(current_chunk_list),
          )
      )

    return messages

  def __getitem__(self, key: str, /) -> Any:
    if key not in self._variables:
      raise ValueError(f'Context key "{key}" not found in {self}')
    return self._variables[key]

  def __contains__(self, key: str) -> bool:
    return key in self._variables

  def __setitem__(self, key: str, value: Any) -> None:
    self._variables[key] = value

  def replace(self, other: dict[str, Any] | Context) -> None:
    """Replace the contents of self with the contents of other.

    Args:
      other: Other Context that will replace self. If a dict is passed,
        only the dict part of this context is replaced, otherwise both
        the dict part and the prefix are replaced.
    """
    if isinstance(other, Context):
      self._variables = other._variables  # pylint: disable=protected-access
      self.prefix = other.prefix
    else:
      self._variables = other


@dataclasses.dataclass
class Composable(
    executing_with_context.SerialExecutableWithContext[
        Context, ChunkList
    ]
):
  """An Executable that can be concatenated.

  A Composable is a node in a computational graph that can be executed or
  iterated through.
  Unlike the basic Executable, a Composable has the additional property that
  it can be added to other Composable (with the + operator) to create a
  concatenation chain. The result of previous elements of the chain is
  automatically passed as the first positional argument to the next element in
  the chain, and there is a Context object that can be passed down the chain.
  Note that the computational graph can be generated dynamically: for example
  a node in the chain can, instead of producing a result to be appended, produce
  another node or even a full graph, that will be inserted in its place and
  executed before proceeding. So the graph's structure can be generated as
  execution proceeds.
  """

  def initialize_context(self, *args, **kwargs) -> Context:
    return Context(kwargs)

  @classmethod
  def empty_result(cls) -> ChunkList:
    return ChunkList()

  @classmethod
  def get_result(cls, context: Context) -> str:
    return str(context.prefix)

  @classmethod
  def wrap(cls, other: str | Chunk) -> Composable:
    return _AppendNode(content=other)

  def __getitem__(self, key: str) -> Any:
    if not self._executed:
      raise ValueError(
          f'Impossible to access key {key}. Composable {self} has not been'
          ' executed.'
      )
    return self._final_context[key]

  def __str__(self) -> str:
    return ' + '.join(str(node) for node in self.nodes)

  def __repr__(self) -> str:
    """String representation for debugging."""
    return str(self)

  @contextlib.contextmanager
  def section(self, name: str, hidden: bool = False) -> Iterator[None]:
    self += section_start(name, hidden)
    yield
    self += section_end()


@dataclasses.dataclass
class _AppendNode(Composable):
  """ComposableNode that wraps a Chunk and appends it."""

  content: str | Chunk | ChunkList | None = None

  @override
  @executing.make_executable
  async def iterate(
      self, context: Context, iteration_depth: int = 1
  ) -> AsyncIterator[list[ChunkList]]:
    # Default implementation: single step execution.
    del iteration_depth
    return (await self.execute(context))

  @override
  @executing.make_executable
  def execute(
      self, context: Context
  ) -> Any:
    match self.content:
      case str():
        context.prefix += self.content
        return ChunkList([Chunk(self.content)])
      case Chunk():
        context.prefix += self.content
        return ChunkList([self.content])
      case ChunkList():
        context.prefix += self.content
        return self.content
      case None:
        return ChunkList([])
      case _:
        # Generic python object: we convert it to a string.
        str_content = str(self.content)
        context.prefix += str_content
        # But we try and return it as is.
        return self.content

  def __str__(self) -> str:
    return str(self.content)

  def __repr__(self) -> str:
    """String representation for debugging."""
    return str(self)


@executing.make_executable
def _noop():
  pass


@dataclasses.dataclass
class _FunctionNode(Composable):
  """ComposableNode that wraps a function.

  Attributes:
    executable: An Executable of type FunctionExecWrapper (e.g. a function
      decorated with make_executable) which takes in a prefix as a first
      argument, and returns a result of the same type. This is where the
      processing will happen in a left-to-right fashion: the various elements
      that are added together will be bundled into a single executable which
      will be executed by passing in the results from one element to the next.
    context_arg_name: This is the name of the argument (first positional
      argument) to which we will pass the context.
  """

  executable: executing.FunctionExecWrapper[ChunkList] = (
      dataclasses.field(default=_noop())
  )
  context_arg_name: str = dataclasses.field(default_factory=str)

  # We disable protected-access as we need to manipulate the _arguments field
  # of the wrapped executable.
  # pylint: disable=protected-access
  @property
  def _context(self) -> Context:
    return self.executable._arguments[self.context_arg_name]

  @_context.setter
  def _context(self, value: Context) -> None:
    self.executable._arguments[self.context_arg_name] = value

  # # pylint: enable=protected-access

  @override
  async def execute(self, context: Context) -> ChunkList:
    self._context = context
    # We execute the wrapped Executable.
    result = await self.executable
    while isinstance(result, executing.Executable) and not isinstance(
        result, Composable
    ):
      result = await result
    # We may get a Composable in which case we execute it with the right
    # context.
    result = await self.maybe_execute(result, context)
    assert not isinstance(result, Composable)
    # We may get yet another Executable and again we execute it.
    while isinstance(result, executing.Executable):
      result = await result
    return result

  async def iterate_through_executable(
      self,
      executable: executing.Executable,
      context: Context,
      iteration_depth: int = 1,
  ) -> AsyncIterator[ChunkList]:
    # We iterate through the executable and if we get a stream of
    # non-executables we yield them (but reset the context at each step since
    # the different yields are for an increasingly long sequence of chunks)
    done = False
    updated_context = copy.deepcopy(context)
    updates = updating.Update()
    async for content_update in executable.with_depth(iteration_depth):
      updates += content_update
      content = updates.to_result()
      if not isinstance(content, executing.Executable):
        pre_context = copy.deepcopy(context)
        yield await self.maybe_execute(content, context)
        updated_context = copy.deepcopy(context)
        context.replace(pre_context)
        done = True
      else:
        yield content
    if done:
      context.replace(updated_context)

  @override
  async def iterate(  # pytype: disable=signature-mismatch
      self, context: Context, iteration_depth: int = 1
  ) -> AsyncIterator[ChunkList]:
    if iteration_depth == 0:
      res = await self.execute(context)
      yield res
    else:
      self._context = context
      content = None
      done = False
      async for content_update in self.iterate_through_executable(
          self.executable, context, iteration_depth=iteration_depth
      ):
        content = content_update
        if (not isinstance(content, executing.Executable)):
          yield content_update
          done = True
      if not done:
        # This is the case where the executable possibly produced a Composable.
        if isinstance(content, Composable):
          async for content_update in content.iterate(
              context,
              iteration_depth=iteration_depth,
          ):
            content = content_update
            if (not isinstance(content_update, executing.Executable)):
              yield content_update
        if isinstance(content, executing.Executable):
          async for content_update in self.iterate_through_executable(
              content, context, iteration_depth=iteration_depth
          ):
            yield content_update
        else:
          if content is not None:
            yield await self.maybe_execute(content, context)

  def __str__(self) -> str:
    return str(self.executable)

  def __repr__(self) -> str:
    """String representation for debugging."""
    return str(self)


@utils.decorator_with_optional_args
def make_composable(
    fn: Callable[..., executing.FunctionExecWrapper],
    *composable_args: str,
) -> Callable[..., Composable]:
  """Allows the function to be composed by addition.

  It returns a function which has one less argument (the first positional
  argument is removed) than the decorated function.

  Usage:
    ```
    @make_composable
    def my_function(context: Context, some_arg: int) -> str:
      # Do something using the context (read or write)
      ...
      return some_result

    res = 'hello' + my_function(1) + 'finished'
    ```

  Args:
    fn: The function to be decorated. Its first argument should be a Context.
      Also this function should return an executing.FunctionExecWrapper, so it
      should be for example a function or method decorated with
      @executing.make_executable.
    *composable_args: Tuple of argument names (as strings) that should not be
      executed as they can be passed as Composables.

  Returns:
    A function that can be used in a composition chain.

  Raises:
    ValueError: If the decorator is not applied to a function, that returns a
      FunctionExecWrapper.
  """
  signature = inspect.signature(fn)
  parameter_names = list(signature.parameters.keys())
  if len(parameter_names) < 1:
    raise ValueError(
        'Expected at least one argument, got'
        f' {len(parameter_names)}. The @make_composable decorator'
        ' should be applied to a function whose first argument represents the'
        ' result from previous elements of the chain.'
    )
  first_arg_name = parameter_names[0]

  if not utils.is_decorated_with_make_executable(fn):
    fn = executing.make_executable(
        first_arg_name, *composable_args, execute_result=False
    )(fn)

  @functools.wraps(fn)
  def inner(*args, **kwargs) -> Composable:
    to_wrap = fn(Context(), *args, **kwargs)
    assert isinstance(to_wrap, executing.FunctionExecWrapper)
    # The composable arguments should not be executed automatically.
    for arg_name in composable_args:
      if arg_name not in to_wrap.non_executed_args:
        to_wrap.non_executed_args.append(arg_name)
    # The context argument should not be executed automatically as we take
    # care of the execution in Composable._aexec().
    if first_arg_name not in to_wrap.non_executed_args:
      to_wrap.non_executed_args.append(first_arg_name)
    # We make sure there is no iterate_argument specified since we want to
    # handle the iteration through the prefix in a special way.
    to_wrap.iterate_argument = None
    # We don't execute the result automatically since it may be a Composable,
    # in which case we need to pass the prefix and context to it before
    # executing.
    to_wrap.execute_result = False
    return Composable(
        nodes=[
            _FunctionNode(
                executable=to_wrap,
                context_arg_name=first_arg_name,
            )
        ]
    )

  return inner


@make_composable
def set_context(
    context: Context, context_to_set: dict[str, Any] | Context
) -> ChunkList:
  """Composable that sets the context for the rest of the chain."""
  context.replace(copy.deepcopy(context_to_set))
  return ChunkList()


@make_composable
def get_context(
    context: Context, context_holder: Context
) -> ChunkList:
  """Composable that sets the context for the rest of the chain."""
  context_holder.replace(context)
  return ChunkList()


@dataclasses.dataclass
class SectionInfo:
  name: str
  start_index: int
  prefix: ChunkList
  hidden: bool


@make_composable
def section_start(
    context: Context, name: str, hidden: bool = False
) -> ChunkList:
  """Introduces a section start into the chain.

  Args:
    context: Execution context of the chain.
    name: Name of the section to be created (this is the name of the variable in
      the context dictionary that will contain the content of the section once
      we reach the first `section_end`).
    hidden: True if this section should be hidden after execution (i.e. the
      prefix will be reset to what it was before the beginning of the section).

  Returns:
    A Composable object that can be added to a chain (and does not return
    anything upon execution).
  """
  if '_sections' not in context:
    context['_sections'] = []
  context['_sections'].append(
      SectionInfo(
          name,
          len(context.prefix) if context.prefix is not None else 0,
          copy.deepcopy(context.prefix),
          hidden,
      )
  )
  return ChunkList()


@make_composable
def section_end(context: Context) -> ChunkList:
  """Signals the end of a section.

  This will close the most recently opened section and store the content of the
  section into the variable whose name is the name of the section (provided via
  `section_start`).

  Args:
    context: Execution context of the chain.

  Returns:
    A Composable object that can be added to a chain (and does not return
    anything upon execution).
  """
  section_info = context['_sections'].pop(-1)
  context[section_info.name] = None
  if context.prefix is not None:
    context[section_info.name] = context.prefix[section_info.start_index :]
  if section_info.hidden:
    context.prefix = section_info.prefix
  return ChunkList()


@make_composable
def get_var(context: Context, var_name: str) -> ChunkList:
  """Composable that returns the value of a context variable.

  Args:
    context: Execution context of the chain.
    var_name: Name of the variable to read from the context.

  Returns:
    A Composable object that can be added to a chain (and returns the value
    of the variable upon execution).
  """
  return context[var_name]


@make_composable('content')
async def store(
    context: Context,
    name: str,
    content: ChunkList | Composable,
) -> ChunkList:
  """Composable that stores the result of a subgraph into a context variable.

  Args:
    context: Execution context of the chain.
    name: Name of the variable to store the content in.
    content: The data to be stored. This can be a Composable which will be
      executed with the prefix of the store node as its prefix before storing
      its result in the variable.

  Returns:
    The result of executing the content.
  """
  saved_prefix = copy.deepcopy(context.prefix)
  if isinstance(content, Composable):
    content = await content.execute(context)
  else:
    content = await _AppendNode(content=content).execute(context)
  context.prefix = saved_prefix
  # The content can either be a ChunkList or a generic python object.
  # If it is a ChunkList, we store a string version of it into the context
  # variable but return the ChunkList.
  # If it is a generic python object we store this object directly into the
  # context variable so that it can be reused as is. But we return a string
  # version of this object embedded in a ChunkList.
  if isinstance(content, ChunkList):
    context[name] = str(content)
    return content
  else:
    context[name] = content
    return ChunkList([Chunk(str(content))])


_Bundle: TypeAlias = tuple[ChunkList, Context]


@make_composable('branches')
async def _fork(
    context: Context,
    *branches: ChunkList | Composable,
    joiner: Callable[
        [Context, Sequence[_Bundle]], _Bundle | Coroutine[Any, Any, _Bundle]
    ],
) -> ChunkList:
  """Parallel composition of composables.

  Args:
    context: Context that will be duplicated and passed to each branch.
    *branches: Composables to be executed.
    joiner: Function to be executed to produce the result from all the options,
      and the final context.

  Returns:
    The result of calling (or awaiting) fn on the executed options.
  """

  @executing.make_executable
  def run_all(
      *to_be_executed: ChunkList | executing.Executable,
  ) -> Sequence[ChunkList]:
    # We use this function with a make_executable decorator to automatically
    # execute all arguments in parallel.
    # The type after execution is indeed ChunkList.
    return to_be_executed  # pytype: disable=bad-return-type

  # Make a copy of the context for each branch.
  # The prefix is already executed, we can't use it to pass context, so
  # we need to create a copy of the context for each branch.
  contexts = [copy.deepcopy(context) for _ in range(len(branches))]
  saved_prefix = copy.deepcopy(context.prefix)

  branches_executables = (
      branch.execute(ctx) if isinstance(branch, Composable) else branch
      for ctx, branch in zip(contexts, branches)
  )
  executed_branches = await run_all(*branches_executables)
  executed_branches = [branch for branch in executed_branches]
  result = joiner(context, list(zip(executed_branches, contexts)))
  if isinstance(result, Coroutine):
    result, new_context = await result
  else:
    result, new_context = result
  # Set context to the one returned by the joiner function.
  context.replace(new_context)
  context.prefix = saved_prefix
  return result


def make_join_composable(
    joiner: Callable[
        [_Bundle, Sequence[_Bundle]], _Bundle | Coroutine[Any, Any, _Bundle]
    ],
) -> Callable[..., Composable]:
  """Decorator that creates a composable that runs branches and joins them.

  Usage:
    This decorator applies to a function that takes in multiple branches (which
    can be the result of Composable objects) and returns some final result by
    combining those branches.
    For example, if we want to create a concatenation of all the branches,
    we would do:
    ```
    @composing.make_join_composable
    def join(*options):
      return ','.join(options)
    ```
    and we can then use it for example as
    ```
    executing.run('prefix' + join('branch1' + gen(), 'branch2'))
    ```
    Alternately we may want to just pick one of the branches:
    ```
    @composing.make_join_composable
    def select(*options):
      return options[0]
    ```

  Args:
    joiner: A function or async function (or make_executable decorated function)
      which takes the result of the execution of each branch and processes them
      to produce a single output.

  Returns:
    A composable which takes a tuple of composables as arguments, runs them
    and applies the joiner to the result.
  """
  return functools.partial(_fork, joiner=joiner)
