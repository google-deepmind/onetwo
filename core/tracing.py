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

"""Library for tracing the execution tree as an ExecutionResult."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator, Mapping, Sequence
import contextvars
import copy
import dataclasses
import functools
import inspect
import io
from typing import Any, Generic, ParamSpec, TypeVar

from onetwo.core import iterating
from onetwo.core import results
from onetwo.core import utils


_ParamType = ParamSpec('_ParamType')
_ReturnType = TypeVar('_ReturnType')

_FunctionToDecorate = (
    Callable[_ParamType, _ReturnType]
    | Callable[_ParamType, Awaitable[_ReturnType]]
    | Callable[_ParamType, AsyncIterator[_ReturnType]]
)

# This is the context variable that will contain the current execution result.
execution_context = contextvars.ContextVar('execution_context')


class Tracer(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def add_stage(self) -> Tracer:
    """Adds a stage."""

  @abc.abstractmethod
  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    """Sets the inputs."""

  @abc.abstractmethod
  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    """Updates the outputs."""

execution_tracer = contextvars.ContextVar('execution_tracer')


@dataclasses.dataclass
class ExecutionResultTracer(Tracer):
  """Tracer that creates and updates an ExecutionResult.

  Attributes:
    execution_result: The ExecutionResult structure being used for tracing.
  """
  execution_result: results.ExecutionResult = dataclasses.field(
      default_factory=results.ExecutionResult
  )

  def add_stage(self) -> Tracer:
    self.execution_result.stages.append(results.ExecutionResult())
    return ExecutionResultTracer(self.execution_result.stages[-1])

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    self.execution_result.stage_name = name
    self.execution_result.inputs = inputs

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    self.execution_result.outputs.update(outputs)


@dataclasses.dataclass
class StringTracer(Tracer):
  """Tracer that prints the execution on stdout.

  Attributes:
    stream: The IO stream to write to (can be sys.stdout).
    indent: Indentation level (to display output as a tree)
  """
  stream: io.TextIOBase
  indent: int = 0

  def add_stage(self) -> Tracer:
    return StringTracer(self.stream, self.indent + 2)

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    indents = ' ' * self.indent
    self.stream.writelines([f'{indents}{name}: {inputs}\n'])

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    indents = ' ' * self.indent
    self.stream.writelines([f'{indents}{name}: {outputs}\n'])


@dataclasses.dataclass
class QueueTracer(Tracer):
  """Tracer that sends messages to a queue.

  Attributes:
    callback: The queue to write updates to.
    depth: Iteration level.
  """
  callback: Callable[[Any], None]
  depth: int = 1

  def add_stage(self) -> Tracer:
    updated_depth = self.depth - 1 if self.depth > 0 else self.depth
    return self.__class__(self.callback, updated_depth)

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    raise NotImplementedError()

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    raise NotImplementedError()


_T = TypeVar('_T')
_V = TypeVar('_V')


@dataclasses.dataclass
class IteratorWithReturnValue(Generic[_T, _V], Iterator[_T]):
  """Wraps a generator as an iterator and stores the final value.

  Attributes:
    generator: Generator to be wrapped.
    value: Return value of the generator.
  """
  generator: Generator[None, _T, Any]
  value: _V | None = None

  def __iter__(self) -> Iterator[_T]:
    # Initialize the generator.
    self.generator = self.generator.__iter__()
    return self

  def __next__(self) -> _T:
    try:
      return self.generator.__next__()
    except StopIteration as e:
      self.value = e.value
      raise e


def stream(
    coroutine: Awaitable[_V],
    tracer: type[QueueTracer] = QueueTracer,
    iteration_depth: int = 1,
) -> IteratorWithReturnValue[Any, _V]:
  """Runs a coroutine and streams its updates to a queue.

  Args:
    coroutine: Coroutine to execute.
    tracer: Tracer class to use for recording the execution.
    iteration_depth: Depth of the coroutine execution to be traced. If negative,
      all updates will be returned. If zero, no update will be produced.

  Returns:
    An IteratorWithReturnValue that will contain the final return value of the
      coroutine. The iterator can be used to iterate over the updates from the
      coroutine execution.
  """

  def _stream_updates(
      callback_wrapper: Callable[[Callable[[_T], None]], Awaitable[_V]]
  ) -> Generator[_T, None, _V]:
    with iterating.coroutine_with_callback_to_sync_iterator(
        callback_wrapper
    ) as it:
      wrapper = IteratorWithReturnValue(it)
      for i in wrapper:
        yield i
    return wrapper.value

  async def wrapper(callback: Callable[[_T], None]) -> Awaitable[_V]:
    execution_context.set(None)
    # We add 1 because the first @trace call will decrease the depth by 1,
    # but we leave -1 unchanged.
    depth = iteration_depth + 1 if iteration_depth >= 0 else iteration_depth
    execution_tracer.set(tracer(callback=callback, depth=depth))
    return await coroutine

  return IteratorWithReturnValue(_stream_updates(wrapper))


@dataclasses.dataclass
class OutputTracer(QueueTracer):
  """Tracer that is used by stream_updates to record the returned values."""

  def set_inputs(self, name: str, inputs: Mapping[str, Any]) -> None:
    # We don't want to record the inputs, so we don't call the callback
    # when we enter a traced function, only when we exit it (see
    # update_outputs).
    pass

  def update_outputs(self, name: str, outputs: Mapping[str, Any]) -> None:
    # Whenever we exit a function, we call the callback with the outputs.
    if self.depth != 0:
      if results.MAIN_OUTPUT in outputs and len(outputs) == 1:
        self.callback(outputs[results.MAIN_OUTPUT])
      else:
        self.callback(outputs)


def stream_updates(
    coroutine: Awaitable[_V],
    iteration_depth: int = 1,
) -> IteratorWithReturnValue[Any, _V]:
  """Runs a coroutine and iterates over its updates.

  This is similar to stream(), but it will only iterate over the updates from
  the coroutine execution, which can either be produced by an inner function
  decorated with @trace or by explicitly calling report_update(). The final
  return value of the coroutine will be available as the value field of the
  returned IteratorWithReturnValue.
  The updates can be of any type.

  Args:
    coroutine: Coroutine to execute.
    iteration_depth: Depth of the coroutine execution to be traced. If negative,
      all updates will be returned. If zero, no update will be produced.

  Returns:
    An IteratorWithReturnValue that will contain the final return value of the
      coroutine. The iterator can be used to iterate over the updates from the
      coroutine execution.
  """

  return stream(coroutine, OutputTracer, iteration_depth)


async def report_update(update: Any, name: str | None = None):
  tracer = execution_tracer.get()
  # With this sleep, we make sure that the calls to update_outputs can be
  # interleaved if there are multiple parallel calls to report_update.
  await asyncio.sleep(0)
  if isinstance(tracer, QueueTracer):
    tracer.update_outputs(name or '', {results.MAIN_OUTPUT: update})


def update_info(info: Mapping[str, Any]) -> None:
  """Add entries into the info field of the current ExecutionResult.

  Args:
    info: Mapping from string to values to update the info field of the current
      ExecutionResult tracking the execution progress.
  """
  execution_result = execution_context.get(None)
  if execution_result is not None:
    execution_result.info.update(info)


def run(
    coroutine: Callable[[], _ReturnType],
    tracer: Tracer | None = None,
) -> tuple[_ReturnType, results.ExecutionResult]:
  """Run a coroutine while enabling tracing of its execution.

  Args:
    coroutine: Function or coroutine to execute. It will be called to execute,
      so if it is a function defined with async def, the return value will be
      a coroutine that will be awaited.
    tracer: Optional Tracer for custom recording of the trace.

  Returns:
    A pair (result, execution_result) of the result from the executed function
      and an ExecutionResult containing the trace of execution.
  """
  # The execution_context variable will contain the ExecutionResult. We set it
  # to None to indicate that it is empty, and it will be initialized by the
  # first encountered function decorated by @tracing.
  execution_context.set(None)
  execution_tracer.set(tracer)
  return_value = coroutine()
  if isinstance(return_value, Awaitable):
    async def wrapper():
      execution_context.set(None)
      execution_tracer.set(tracer)
      result = await return_value
      return result, execution_context.get(None)
    return asyncio.run(wrapper())

  return return_value, execution_context.get(None)


@utils.decorator_with_optional_args
def trace(
    function: _FunctionToDecorate | None = None,
    name: str | utils.FromInstance[str] | None = None,
    skip: Sequence[str] | None = None,
) -> _FunctionToDecorate:
  """Decorator to enable producing an ExecutionResult of the execution tree.

  Usage:

  ```
  @trace
  def my_f(x, y):
    ...

  @trace(skip=['x'])
  def my_f(x, y):
    ...

  output, execution_result = tracing.run(functools.partial(my_f, x, y))
  ```

  Args:
    function: The function or method to be decorated (can be a normal function,
      a coroutine, or an asynchronous iterator).
    name: Optional name to use (instead of the decorated function name) as the
      stage_name in the ExecutionResult.
    skip: Optional Sequence of names of parameters to be skipped, i.e. not
      included in the `inputs` field of the ExecutionResult (note that in the
      case of a method, 'self' will always be skipped). This may also contain
      the name of output parameters. For example if it contains
      results.MAIN_OUTPUT and the return value is not a Mapping, nothing will
      be added to the outputs. If the return value is a Mapping, the fields that
      are mentioned in skip will not be included in the outputs.

  Returns:
    A function of the same type and signature as the decorated one.
  """

  def _prepare_and_populate_inputs(
      *args, **kwargs
  ) -> tuple[
      str,
      results.ExecutionResult,
      contextvars.Token[results.ExecutionResult],
      contextvars.Token[ExecutionResultTracer] | None,
  ]:
    execution_result = execution_context.get(None)
    if execution_result is None:
      execution_result = results.ExecutionResult()
      # We set it here and below as well so that the token refers back to this
      # value.
      execution_context.set(execution_result)
    else:
      execution_result.stages.append(results.ExecutionResult())
      execution_result = execution_result.stages[-1]

    nonlocal name
    if isinstance(name, utils.FromInstance):
      if 'self' in kwargs:
        obj = kwargs['self']
      else:
        obj = args[0]
      name = utils.RuntimeParameter[str](name, obj).value()

    stage_name = name or function.__name__
    inputs = copy.copy(
        utils.get_expanded_arguments(function, False, args, kwargs)
    )
    if utils.is_method(function):
      del inputs['self']
    if skip is not None:
      for param in skip:
        inputs.pop(param, None)
    try:
      inputs = copy.deepcopy(inputs)
    except TypeError as e:
      raise ValueError(
          f'Unable to trace the inputs to {name} ({function.__name__}),'
          f' {args=} {kwargs=}'
      ) from e
    execution_result.stage_name = stage_name
    execution_result.inputs = inputs

    tracer = execution_tracer.get(None)
    if tracer is not None:
      new_tracer = tracer.add_stage()
      new_tracer.set_inputs(stage_name, inputs)
      tracer_token = execution_tracer.set(new_tracer)
    else:
      tracer_token = None

    token = execution_context.set(execution_result)
    return stage_name, execution_result, token, tracer_token

  def _update_outputs(
      execution_result: results.ExecutionResult | None, value: Any, name: str
  ) -> None:
    if execution_result is None:
      return
    # If the returned value is a Mapping, we use that directly as the outputs
    # dictionary in the execution_result, otherwise we create a single field
    # called results.MAIN_OUTPUT containing the returned value.
    if not isinstance(value, Mapping):
      value = {results.MAIN_OUTPUT: value}
      # Possibly remove the entries that have to be skipped.
    if skip is not None:
      value = copy.copy(value)
      for param in skip:
        value.pop(param, None)
    # We use update here because the function may have already written some
    # outputs on its own.
    execution_result.outputs.update({**value})

    tracer = execution_tracer.get(None)
    if tracer is not None:
      if skip is None or results.MAIN_OUTPUT not in skip:
        tracer.update_outputs(name, execution_result.outputs)

  @functools.wraps(function)
  async def awrapper(*args, **kwargs) -> _ReturnType:
    stage_name, execution_result, token, tracer_token = (
        _prepare_and_populate_inputs(*args, **kwargs)
    )
    # Call
    return_value = await function(*args, **kwargs)
    _update_outputs(execution_result, return_value, stage_name)
    execution_context.reset(token)
    if tracer_token is not None:
      execution_tracer.reset(tracer_token)
    return return_value

  @functools.wraps(function)
  async def iwrapper(*args, **kwargs) -> AsyncIterator[_ReturnType]:
    stage_name, execution_result, token, tracer_token = (
        _prepare_and_populate_inputs(*args, **kwargs)
    )
    # Call
    async for return_value in function(*args, **kwargs):  # pytype: disable=attribute-error
      _update_outputs(execution_result, return_value, stage_name)
      yield return_value
    execution_context.reset(token)
    if tracer_token is not None:
      execution_tracer.reset(tracer_token)

  @functools.wraps(function)
  def gwrapper(*args, **kwargs) -> Iterator[_ReturnType]:
    stage_name, execution_result, token, tracer_token = (
        _prepare_and_populate_inputs(*args, **kwargs)
    )
    # Call
    for return_value in function(*args, **kwargs):  # pytype: disable=attribute-error
      _update_outputs(execution_result, return_value, stage_name)
      yield return_value
    execution_context.reset(token)
    if tracer_token is not None:
      execution_tracer.reset(tracer_token)

  @functools.wraps(function)
  def wrapper(*args, **kwargs) -> _ReturnType:
    stage_name, execution_result, token, tracer_token = (
        _prepare_and_populate_inputs(*args, **kwargs)
    )
    # Call
    return_value = function(*args, **kwargs)
    _update_outputs(execution_result, return_value, stage_name)
    execution_context.reset(token)
    if tracer_token is not None:
      execution_tracer.reset(tracer_token)
    return return_value

  if isinstance(name, utils.FromInstance) and not utils.is_method(function):
    raise ValueError(
        '@trace decorator has a FromInstance argument but is applied to a'
        ' function and not a method'
    )

  if inspect.iscoroutinefunction(function):
    return awrapper
  elif inspect.isasyncgenfunction(function):
    return iwrapper
  elif inspect.isgeneratorfunction(function):
    return gwrapper
  else:
    return wrapper
