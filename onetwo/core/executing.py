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

"""Layer on top of asyncio to enable handling async iterators like coroutines.

"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
import contextlib
import copy
import dataclasses
import functools
import inspect
import itertools
from typing import Any, Final, Generic, Literal, ParamSpec, TypeVar, cast, final, overload

from onetwo.core import batching
from onetwo.core import executing_impl
from onetwo.core import iterating
from onetwo.core import results as results_lib
from onetwo.core import tracing
from onetwo.core import updating
from onetwo.core import utils


# Basic type variables that need to be specified when using this library.
Result = TypeVar('Result')
_Args = ParamSpec('_Args')

run = batching.run

Executable = executing_impl.Executable
ExecutableWithPostprocessing = executing_impl.ExecutableWithPostprocessing
Update = updating.Update
ListUpdate = updating.ListUpdate

_DEFAULT_CHUNK_SIZE: Final[int] = 100


@dataclasses.dataclass
class ExecutableWithCallback(
    Generic[Result], Executable[Result], metaclass=abc.ABCMeta
):
  """An executable with a callback executed at the end of the processing.

  This can be used for example to represent the return value of a streaming
  call. Indeed, this return value would be an executable that can be iterated
  through and with a final callback to cache the result at the end of the
  iterations.

  Concrete implementations of this class should override the `_iterate()`
  abstract method which yields intermediate values.

  Attributes:
    final_value_callback: A callback to call at the end of the iterations to
      ensure that the final value is handled properly (e.g. in the case where it
      needs to be cached).
    final_value: After the iterations are finished, this field contains the last
      produced value.
  """

  final_value_callback: Callable[[Result], None] | None = None
  final_value: Result | None = None

  @final
  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[Update[Result]]:
    """Yields the intermediate values and calls the final_value_callback."""
    updates = Update()
    async for update in self._iterate():
      updates += update
      if iteration_depth != 0:
        yield update
    self.final_value = updates.to_result()
    if self.final_value_callback is not None:
      self.final_value_callback(self.final_value)
    if iteration_depth == 0:
      yield Update(self.final_value)

  @final
  async def _aexec(self) -> Result:
    """Iterate this value until done (including calling final_value_callback).

    Returns:
      The final value given by the AsyncIterator _inner().
    """
    async for _ in self._aiterate():
      pass
    return self.final_value

  @abc.abstractmethod
  async def _iterate(self) -> AsyncIterator[Update[Result]]:
    """Produces the intermediate values."""
    yield Update(self.final_value)  # To force the correct signature.
    raise NotImplementedError()


@contextlib.contextmanager
def _safe_stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Iterator[batching.ResultType[Update[Result]]]]:
  """See safe_stream."""
  if iteration_depth == 0:
    result = batching.run(
        executable,
        enable_batching=enable_batching,
        enable_tracing=enable_tracing,
        tracer=tracer,
    )
    def iterator():
      if enable_tracing:
        yield Update(result[0]), result[1]
      else:
        yield Update(result)

    yield iterator()
  else:
    # We distinguish the enable_tracing cases to enable static type checking
    # and we explicitly type cast the return value as the type checker gets
    # confused otherwise.
    if enable_tracing:
      with batching.safe_stream(
          executable.with_depth(iteration_depth),
          enable_batching=enable_batching,
          enable_tracing=True,
          tracer=tracer,
      ) as iterator:
        yield cast(
            Iterator[tuple[Update[Result], results_lib.ExecutionResult]],
            iterator,
        )
    else:
      with batching.safe_stream(
          executable.with_depth(iteration_depth),
          enable_batching=enable_batching,
          enable_tracing=False,
          tracer=tracer,
      ) as iterator:
        yield cast(Iterator[Update[Result]], iterator)


@overload
def safe_stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> contextlib.AbstractContextManager[Iterator[Update[Result]]]: ...


@overload
def safe_stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> contextlib.AbstractContextManager[
    Iterator[tuple[Update[Result], results_lib.ExecutionResult]]
]:
  ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def safe_stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> contextlib.AbstractContextManager[
    Iterator[batching.ResultType[Update[Result]]]
]:
  ...


# Since this method is overloaded, it cannot be decorated with
# @contextlib.contextmanager as this would confuse the type checker. Instead
# we wrap the _safe_stream method which is the one decorated and we define
# the return type of safe_stream as an AbstractContextManager.
def safe_stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> contextlib.AbstractContextManager[
    Iterator[batching.ResultType[Update[Result]]]
]:
  """Main entry point to run an AsyncIterator with automatic batching.

  This uses threading to convert an AsyncIterator into a sync one.
  The AsyncIterator is run in a separate thread, while the main thread is
  watching for updates from that separate thread.
  In order to properly stop the child thread in case there is an exception in
  the loop body, this does not directly return an Iterator but instead should
  be used as a context manager, i.e. within a `with` statement:
  ```
  with safe_stream(iterator) as it:
    for item in it:
      ...
  ```
  Note that there is an alternative `stream` function that directly returns
  an iterator and can thus be used as `for item in stream(iterator):`
  but it will not properly stop the threads in case of an error and may hang.

  Args:
    executable: The iterator to be executed.
    iteration_depth: When doing recursive calls to other executables, they can
      be executed in one step or iteratively, if the depth is 0 the execution
      will be in one step, but if depth is n the iterative execution of
      subexecutables will be done with depth n-1. If set to -1, this means no
      limitation of the iteration depth.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine;
      if True, returns both the coroutine result and an ExecutionResult tracing
      the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Returns:
    A context manager that provides an iterator over the results from the
    executable.
    At each step, this yields the yielded value of the executable if
    enable_tracing=False, and a pair (yielded_value, execution_result) if
    enable_tracing=True.
    The yielded value is wrapped into an Update because __aiter__ and __anext__
    methods of class Executable return Updates.
  """
  context_manager = _safe_stream(
      executable,
      iteration_depth=iteration_depth,
      enable_batching=enable_batching,
      enable_tracing=enable_tracing,
      tracer=tracer,
  )
  if enable_tracing:
    return cast(
        contextlib.AbstractContextManager[
            Iterator[tuple[Update[Result], results_lib.ExecutionResult]]
        ],
        context_manager,
    )
  else:
    return cast(
        contextlib.AbstractContextManager[Iterator[Update[Result]]],
        context_manager,
    )


@contextlib.contextmanager
def stream_updates(  # pytype: disable=invalid-annotation
    executable: Executable[Result],
    *,
    enable_batching: bool = True,
    iteration_depth: int = 1,
) -> Iterator[tracing.IteratorWithReturnValue[Any, Result]]:
  """Stream updates from a coroutine.

  Args:
    executable: The Executable whose updates are to be streamed.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    iteration_depth: The depth of the nested coroutines to stream.

  Yields:
    An IteratorWithReturnValue that produces the updates from the executable
    and its final result.
  """
  with batching.stream_updates(
      executable,
      enable_batching=enable_batching,
      iteration_depth=iteration_depth,
  ) as it:
    yield it


@overload
def stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Update[Result]]: ...


@overload
def stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> Iterator[tuple[Update[Result], results_lib.ExecutionResult]]: ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[batching.ResultType[Update[Result]]]: ...


def stream(
    executable: Executable[Result],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[batching.ResultType[Update[Result]]]:
  """Runs the executable one step at a time, returning intermetiate results.

  **WARNING**: this will not properly end the threads if an exception is raised
  in the loop body. So consider this function as DEPRECATED.
  If you use it in a normal python program or in a test, this may lead to the
  program hanging indefinitely.
  Use `safe_stream` to avoid this issue.

  More precisely, what is happening is the following: even though this function
  wraps `safe_stream`, if you use it as follows:
  ```
  for update in stream(...):
    assert False
  ```
  The exception will be propagated back to outside the for loop, but will not
  be caught by the with statement inside the definition of `stream`.
  The only way to properly catch it is to surround the for loop with a
  try-except or a context manager, hence one should do
  ```
  with safe_stream(...) as iterator:
    for update in iterator:
      ...
  ```
  Note that it is still ok to use this function in a colab (the colab runtime
  will take care of closing the thread).

  Args:
    executable: Executable to stream.
    iteration_depth: When doing recursive calls to other executables, they can
      be executed in one step or iteratively, if the depth is 0 the execution
      will be in one step, but if depth is n the iterative execution of
      subexecutables will be done with depth n-1. If set to -1, this means no
      limitation of the iteration depth.
    enable_batching: If False, all calls to batched functions/methods are
      sent directly, if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine,
      and if True returns both the coroutine result and an ExecutionResult
      tracing the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Yields:
    At each step, this yields the yielded value of the executable if
    enable_tracing=False, and a pair (yielded_value, execution_result) if
    enable_tracing=True.
    The yielded value is wrapped into an Update because __aiter__ and __anext__
    methods of class Executable return Updates.
  """
  with safe_stream(
      executable,
      iteration_depth=iteration_depth,
      enable_batching=enable_batching,
      enable_tracing=enable_tracing,
      tracer=tracer,
  ) as iterator:
    yield from iterator


@overload
def stream_with_callback(
    executable: Executable[Result],
    callback: Callable[[batching.ResultType], None],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> Result: ...


@overload
def stream_with_callback(
    executable: Executable[Result],
    callback: Callable[[batching.ResultType], None],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> tuple[Result, results_lib.ExecutionResult]: ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def stream_with_callback(
    executable: Executable[Result],
    callback: Callable[[batching.ResultType], None],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> batching.ResultType[Result]: ...


def stream_with_callback(
    executable: Executable[Result],
    callback: Callable[[batching.ResultType], None],
    *,
    iteration_depth: int = 1,
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> batching.ResultType[Result]:
  """Runs the executable, calls the callback at each step, returns the result.

  For more information see `run`.

  Args:
    executable: Executable to stream.
    callback: Function to call when a new result/update is available.
      At each step, the callback is given the yielded value of the executable
      if enable_tracing=False, and a pair (yielded_value, execution_result) if
      enable_tracing=True.
      The yielded value will be possibly wrapped into an Update.
    iteration_depth: When doing recursive calls to other executables, they can
      be executed in one step or iteratively, if the depth is 0 the execution
      will be in one step, but if depth is n the iterative execution of
      subexecutables will be done with depth n-1. If set to -1, this means no
      limitation of the iteration depth.
    enable_batching: If False, all calls to batched functions/methods are
      sent directly, if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine,
      and if True returns both the coroutine result and an ExecutionResult
      tracing the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Returns:
    The final result (after accumulation) and the last execution_result (if
    enable_tracing=True).
  """
  final_execution_result = None
  updates = Update()

  async def wrapper(result: batching.ResultType[Update]) -> None:
    nonlocal updates, final_execution_result
    if enable_tracing:
      assert isinstance(result, tuple)  # Type hint.
      update, final_execution_result = result
    else:
      update = result
    updates += update
    res = callback(result)
    if isinstance(res, Awaitable):
      await res

  batching.stream_with_callback(
      executable.with_depth(iteration_depth),
      callback=wrapper,
      enable_batching=enable_batching,
      enable_tracing=enable_tracing,
      tracer=tracer,
  )
  if enable_tracing:
    return updates.to_result(), final_execution_result
  else:
    return updates.to_result()


# TODO: This decorator has many arguments now, and the fact that
# non_executed_args are passed positionally is a bit confusing. We should
# replace this by a keyword argument taking in an Iterable of str.
@utils.decorator_with_optional_args
def make_executable(
    function: Callable[_Args, Result],
    *non_executed_args: str,
    non_copied_args: Iterable[str] = tuple(),
    copy_self: bool = True,
    execute_result: bool = True,
    iterate_argument: str | None = None,
) -> Callable[_Args, Executable[Result]]:
  """Decorates a method or function by transforming it into an Executable.

  This decorator can be used with or without arguments.
  It performs several changes to the decorated function or method (which can
  be sync or async):
  1. By default, it makes sure to execute any argument of the function that is
    of type Executable before calling the function, unless their name is passed
    as an argument of the decorator.
    **Important**: when this decorator is applied to a method or a bound method,
    the object that is obtained when calling the method can be deep-copied (e.g.
    when using sampling.repeat) if one needs to execute several distinct
    instances, and the effect of this deep-copy will be to deep-copy the
    instance on which the method is defined. This is necessary, for example,
    when the execution of the method has side effects on its own object. But
    there are cases where one may want to avoid copying the method's instance.
    For example, if the method has no side effects and is defined on an object
    one doesn't want to copy. In this case, one can set copy_self to False to
    ensure the instance is not copied.
  2. It changes the return type of the decorated function to be an Executable.
    This means that when calling the function, we get back an object that can be
    executed (with `await ...`) or iterated (with `async for ...). The typical
    use case is to decorate a function such that one can then use the resulting
    Executable to execute the function asynchronosely and/or in parallel with
    other functions (via the `parallel` or `par_iter` functions).
  3. The other effect of returning an Executable is that the returned value of
    the decorated function can directly be executed either with `executing.run`,
    `executing.stream` or `executing.stream_with_callback`.

  Repeated decoration is a no op:
  ```
    make_executable(*args2)(make_executable(*args1)(fn))
  ```
  has the same result as
  ```
    make_executable(*args1)(fn)
  ```
  no matter what `args2` are.

  Args:
    function: Function or (bound) method to be decorated.
    *non_executed_args: Strings containing the names of the parameters of the
      decorated function that we do not want the decorator to execute before
      calling the function.
    non_copied_args: Strings containing the names of the parameters of the
      decorated function that we do not want to have deep-copied when the
      resulting Executable is deep-copied. For the parameters in this list,
      references to the exact same objects that were passed as args to the
      original Executable will be passed as-is to the deep-copied Executable.
      Should only be used for arguments that are thread-safe and which are able
      to maintain correct semantics even when accessed in parallel from multiple
      different execution branches.
    copy_self: This is only relevant in case the decorator is applied to a
      method (bound or not). If copy_self is True and the resulting Executable
      is deep-copied, the instance on which this method is defined will be
      deep-copied too. If copy_self is False, we will ensure the instance is not
      copied. The default is to copy the instance.
    execute_result: If True, the returned value from the wrapped function will
      not be returned as is but will be inspected. If it is an Executable, then
      it will be executed (either through await or async for). So if the
      decorated function is called with async for and it returns an Executable
      the iterations will iterate through this Executable directly. If False,
      the returned value from the wrapped function is passed as is. Note that
      even when execute_result is set to True, there is a way to avoid the
      execution of the returned value at runtime by calling the `pre_execute()`
      method which will just return that value without executing it. This can be
      useful when the execution of this return value should be done separately.
    iterate_argument: This is the name of one of the arguments of the function.
      When we are in iterating mode we will yield the updates from this
      argument (either yielding the argument itself if it should not be executed
      or yielding the updates from iterating through this argument). This
      enables some chaining of iterable functions whereby we get the iterates
      from one function after the other in a chain `f1(f2(f3()))`.

  Returns:
    A function with the same signature as the decorated one, but where each
    argument of type T can be provided as an Executable[T] instead, and which
    upon calling returns an Executable[Result] which, when executed will
    asynchronously execute all Executable arguments and execute the body of the
    decorated function.
  """

  if (
      hasattr(function, '__call__')
      and not inspect.isfunction(function)
      and not inspect.ismethod(function)
  ):
    function = function.__call__

  if utils.is_decorated_with_make_executable(function):
    return function

  non_copied_args = list(non_copied_args)
  if not copy_self and 'self' not in non_copied_args:
    non_copied_args.append('self')

  @functools.wraps(function)
  def inner_m(self, *args: _Args.args, **kwargs: _Args.kwargs):
    nonlocal non_executed_args, non_copied_args
    return FunctionExecWrapper(
        function,
        (self,) + args,
        kwargs,
        non_executed_args=non_executed_args,
        non_copied_args=non_copied_args,
        execute_result=execute_result,
        iterate_argument=iterate_argument,
    )

  @functools.wraps(function)
  def inner_f(*args: _Args.args, **kwargs: _Args.kwargs):
    return FunctionExecWrapper(
        function,
        args,
        kwargs,
        non_executed_args=non_executed_args,
        non_copied_args=non_copied_args,
        execute_result=execute_result,
        iterate_argument=iterate_argument,
    )

  utils.set_decorated_with_make_executable(inner_m)
  utils.set_decorated_with_make_executable(inner_f)

  if inspect.isfunction(function):
    # Decorating a function or decorating a method with "@".
    if utils.is_method(function):
      return inner_m
    else:
      return inner_f
  elif inspect.ismethod(function):
    # Decorating a bound method inline (as opposed to doing it with "@"")
    # using `make_executable(class_instance.some_method)`.
    unbound_function = function.__func__
    obj = function.__self__
    @functools.wraps(function)
    def inner_bound(*args: _Args.args, **kwargs: _Args.kwargs):
      nonlocal non_executed_args, non_copied_args
      return FunctionExecWrapper(
          unbound_function,
          (obj,) + args,
          kwargs,
          non_executed_args=non_executed_args,
          non_copied_args=non_copied_args,
          execute_result=execute_result,
          iterate_argument=iterate_argument,
      )
    utils.set_decorated_with_make_executable(inner_bound)
    return inner_bound
  else:
    raise ValueError('Decorator must be applied to a method or function.')


# We use "Any" for `serial`, `ser_iter`, `parallel`, and `par_iter` below,
# because we could do things like `serial(exec1, parallel(exec2, exec3))`, where
# exec1, exec2, and exec3 are all Executable over the same result type.
# TODO: Could we do a better type check here?
def serial(*args: Executable[Any]) -> SerialExecutable:
  """Chains executables serially."""
  return SerialExecutable(iter(args))


def ser_iter(executables: Iterable[Executable[Any]]) -> SerialExecutable:
  """Lazily chains executables serially from an iterable of executables."""
  return SerialExecutable(iter(executables))


def parallel(
    *args: Executable[Any],
    chunk_size: int | None = None,
    return_exceptions: bool = False,
) -> ParallelExecutable:
  """Chains executables in parallel. See `par_iter` for details."""
  return ParallelExecutable(
      iter(args), chunk_size=chunk_size, return_exceptions=return_exceptions
  )


def par_iter(
    executables: Iterable[Executable[Any]],
    chunk_size: int | None = None,
    return_exceptions: bool = False,
) -> ParallelExecutable:
  """Lazily chains executables in parallel from an iterable of executables.

  This executes the executables from the iterable in chunks, so that the
  iterable doesn't need to be fully materialized.

  Args:
    executables: Iterable of executables to execute in parallel.
    chunk_size: Size of the chunks of executables to be executed in parallel. If
      None, the default depends on whether the input is a list or an iterator.
      Indeed, we don't want to materialize the whole list if it is provided as
      an iterator, so we use a default of _DEFAULT_CHUNK_SIZE. However, if the
      input is already fully materialized, we use as chunk_size the length of
      the list. Note that there are specific cases when one may want to have
      smaller chunks than the size of the list, e.g. to force more interleaving
      of various steps in a big pipeline, so the value can be provided
      explicitly.
    return_exceptions: If False, any exception will stop the processing and no
      result will be returned. If True, exceptions produced by individual
      executables will be used as their returned result, so that the processing
      will not be interrupted.

  Returns:
    An executable that produces a list of results (or exceptions) upon
    execution.
  """
  return ParallelExecutable(
      executables, chunk_size=chunk_size, return_exceptions=return_exceptions
  )


class FunctionExecWrapper(Generic[Result], Executable[Result]):
  """Convenience wrapper to make a function/method into an executable.

  Attributes:
    wrapped: Function to be wrapped (with return type Result).
    args: *args from the wrapped function when it was called.
    kwargs: *kwargs from the wrapped function when it was called.
    non_executed_args: Sequence of names of those parameters that should be
      passed to the wrapped method without attempting to execute them
      (indeed, if some of the parameters are of type Executable, and not
      mentioned in this list, the default behaviour is to execute them and
      pass the resulting value to the wrapped method).
    non_copied_args: Strings containing the names of the parameters of the
      decorated function that we do not want to have deep-copied when the
      resulting Executable is deep-copied. For the parameters in this list,
      references to the exact same objects that were passed as args to the
      original Executable will be passed as-is to the deep-copied Executable.
      Should only be used for arguments that are thread-safe and which are able
      to maintain correct semantics even when accessed in parallel from multiple
      different execution branches.
    execute_result: If True, the returned value from the wrapped function will
      not be returned as is but will be inspected. If it is an Executable, then
      it will be executed (either through await or async for). So if the
      decorated function is called with async for and it returns an Executable
      the iterations will iterate through this Executable directly.
      If False, the returned value from the wrapped function is passed as is.
      Note that even when execute_result is set to True, there is a way to
      avoid the execution of the returned value at runtime by calling the
      `pre_execute()` method which will just return that value without executing
      it. This can be useful when the execution of this return value should be
      done separately.
    iterate_argument: This is the name of one of the arguments of the function.
      When we are in iterating mode we will yield the updates from this
      argument (either yielding the argument itself if it should not be executed
      or yielding the updates from iterating through this argument). This
      enables some chaining of iterable functions whereby we get the iterates
      from one function after the other in a chain `f1(f2(f3()))`.
  """

  def __init__(
      self,
      wrapped: Callable[..., Result],
      args: tuple,  # pylint: disable=g-bare-generic
      kwargs: dict[str, Any],
      *,
      non_executed_args: Sequence[str],
      non_copied_args: Iterable[str],
      execute_result: bool,
      iterate_argument: str | None,
  ):
    self.wrapped = wrapped
    # If there are executable arguments, we fill in the arguments dictionary
    # from args and kwargs so that we know which ones to execute when being
    # called.
    self.args = args
    self.kwargs = kwargs
    self.iterate_argument = iterate_argument
    self.non_executed_args = list(non_executed_args)
    self.non_copied_args = list(non_copied_args)
    # From args and kwargs, we create a list of bound arguments, which means we
    # create a map from argument name to the corresponding value, using
    # default values for arguments that are not specified.
    self._arguments = utils.get_expanded_arguments(
        self.wrapped, True, args, kwargs
    )
    # Try and determine whether one of the parameters is of kind VAR_POSITIONAL.
    # We will have to execute its individual elements.
    parameters = inspect.signature(self.wrapped).parameters
    self.var_positional = None
    for name, parameter in parameters.items():
      if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
        self.var_positional = name
        break
    # In case there is a `self` argument we also avoid executing it.
    if 'self' in self._arguments and 'self' not in non_executed_args:
      self.non_executed_args.append('self')
    if iterate_argument is not None:
      if iterate_argument not in self._arguments:
        raise ValueError(
            f'Argument {iterate_argument} not found in arguments of function'
            f' {wrapped.__name__}'
        )
    self.execute_result = execute_result

    if non_copied_args:
      self.__deepcopy__ = self._custom_deepcopy

  def _custom_deepcopy(self, memo):
    """Custom deepcopy function supporting `non_copied_args`.

    Args:
      memo: Memo to keep track of already copied instances.

    Returns:
      A copied instance.
    """
    copied_expanded_arguments = {}
    for name, value in self._arguments.items():
      if name in self.non_copied_args:
        copied_expanded_arguments[name] = value
      else:
        copied_expanded_arguments[name] = copy.deepcopy(value, memo)
    copied_args, copied_kwargs = utils.get_calling_args_and_kwargs(
        self.wrapped, copied_expanded_arguments
    )
    other = type(self)(
        self.wrapped,
        copied_args,
        copied_kwargs,
        non_executed_args=copy.deepcopy(self.non_executed_args, memo),
        non_copied_args=copy.deepcopy(self.non_copied_args, memo),
        execute_result=copy.deepcopy(self.execute_result, memo),
        iterate_argument=copy.deepcopy(self.iterate_argument, memo),
    )
    memo[id(self)] = other
    return other

  def __str__(self) -> str:
    """String representation for debugging."""
    args = ', '.join(
        [str(a) for a in self.args]
        + [f'{k}={v}' for k, v in self.kwargs.items()]
    )
    return f'[{self.wrapped.__name__}]({args})'

  def __repr__(self) -> str:
    """String representation for debugging."""
    return str(self)

  @functools.cached_property
  def _is_iterator(self) -> bool:
    return inspect.isgeneratorfunction(
        self.wrapped
    ) or inspect.isasyncgenfunction(self.wrapped)

  async def _process_arguments(
      self, execute_iterate_argument: bool = True
  ) -> None:
    """Execute the executable arguments unless marked explicitly.

    This will execute each argument repeatedly until we obtain something that
    is not of type Executable.

    Args:
      execute_iterate_argument: If True, the argument marked as
        `iterate_argument` will be executed.
    """
    while (await self._process_arguments_one_pass(execute_iterate_argument)):
      pass

  async def _process_arguments_one_pass(
      self, execute_iterate_argument: bool = True
  ) -> bool:
    """Execute once each of the executable arguments.

    Each argument which is not part of the non_executed_args and that is of
    type Executable will be executed once. The result may still be of type
    Executable.

    Args:
      execute_iterate_argument: If True, the argument marked as
        `iterate_argument` will be executed.

    Returns:
      True if at least one argument was executed.
    """
    # We identify all the arguments that are of type Executable and that are
    # not in the non_executed_args list, as they will need to be executed first.
    to_execute = []
    if self.var_positional is not None:
      # The var_positional argument is a tuple. Since we want to execute its
      # values (hence change entries in the tuple), we convert it to a list
      # and will convert it back to a tuple at the end.
      self._arguments[self.var_positional] = list(
          self._arguments[self.var_positional]
      )
    for name, value in self._arguments.items():
      if name not in self.non_executed_args:
        if isinstance(value, Executable):
          # By default we execute arguments that are executable and not marked
          # as non_executed_args except if the argument name is specified in the
          # `iterate_argument` parameter and execute_iterate_argument=False.
          if name != self.iterate_argument or execute_iterate_argument:
            to_execute.append((name, value))
        elif self.var_positional is not None and name == self.var_positional:
          for index, v in enumerate(value):
            if isinstance(v, Executable):
              to_execute.append((f'#{index}', v))
    if to_execute:
      if len(to_execute) == 1:
        arg_values = [await to_execute[0][1]]
        arg_names = [to_execute[0][0]]
      else:
        # We run in parallel all arguments that have to be executed.
        arg_values = await parallel(*(value for (_, value) in to_execute))
        arg_names = (name for (name, _) in to_execute)
      # We store executed arguments back into self._arguments with the correct
      # format in case we have a var_positional tuple.
      for name, value in zip(arg_names, arg_values):
        if name.startswith('#'):
          index = int(name[1:])
          self._arguments[self.var_positional][index] = value
        else:
          self._arguments[name] = value
    if self.var_positional is not None:
      # We restore the var_positional argument into its original form as a
      # tuple.
      self._arguments[self.var_positional] = tuple(
          self._arguments[self.var_positional]
      )
    return bool(to_execute)

  async def _call_or_await(self) -> Any:
    """Calls the wrapped function if possible.

    In particular this method does not execute the result of the wrapped
    function so that this can be done separately.
    Also it does not call the wrapped function if it is a generator function
    since this would not return a single output.

    Returns:
      A tuple(bool, result): where the bool is True if the wrapped function
      has been executed and False otherwise. When True, the result is the return
      value of the wrapped function.
    """
    assert not self._is_iterator
    args, kwargs = utils.get_calling_args_and_kwargs(
        self.wrapped, self._arguments
    )
    if inspect.iscoroutinefunction(self.wrapped):  # Awaitable.
      result = await self.wrapped(*args, **kwargs)
      return result
    else:  # Normal function.
      result = self.wrapped(*args, **kwargs)
      return result

  async def _iterate_and_accumulate(self) -> Result:
    # When the wrapped function is a generator function, we run it to
    # exhaustion and accumulate the updates if the yielded elements are updates,
    # otherwise we return the last element.
    assert self._is_iterator
    args, kwargs = utils.get_calling_args_and_kwargs(
        self.wrapped, self._arguments
    )
    updates = Update()
    if inspect.isasyncgenfunction(self.wrapped):  # Async iterator.
      async for update_or_result in self.wrapped(*args, **kwargs):
        if isinstance(update_or_result, Update):
          updates += update_or_result
        else:
          # If the underlying object doesn't return updates we simply keep the
          # latest value.
          updates = Update(copy.deepcopy(update_or_result))
    else:  # Normal iterator.
      for update_or_result in self.wrapped(*args, **kwargs):
        if isinstance(update_or_result, Update):
          updates += update_or_result
        else:
          # If the underlying object doesn't return updates we simply keep the
          # latest value.
          updates = Update(copy.deepcopy(update_or_result))
    return updates.to_result()

  async def pre_execute(self) -> Result:
    """Executes the arguments and calls the wrapped function.

    In particular this method does not execute the result of the wrapped
    function so that this can be done separately.
    If the wrapped method is a generator function (sync or async), then we
    iterate through it and return the final value (accumulating the intermediate
    values if they are updates), otherwise we call/await the wrapped method and
    return its returned value as is.

    Returns:
      The returned value of the wrapped function after calling, awaiting or
      iterating through it.
    """
    await self._process_arguments()
    if self._is_iterator:
      # The wrapped function returns an iterator or async iterator.
      # We run through it and return the final value or accumulated value.
      return await self._iterate_and_accumulate()
    else:
      return await self._call_or_await()

  async def _aexec(self) -> Result:
    """Executes the arguments of and calls the wrapped function.

    If the wrapped function is a normal function or a coroutine (hence producing
    a single result), this will call/await it and if the returned result is an
    Executable this executable will be executed and its result returned.
    If the wrapped function is an AsyncIterator, this will iterate
    through it and accumulate the results to only return the final result.

    Returns:
      The final result produced by the wrapped function.
    """
    await self._process_arguments()
    if self._is_iterator:
      # The wrapped function returns an iterator or async iterator.
      # We run through it and return the final value or accumulated value.
      return await self._iterate_and_accumulate()
    else:
      result = await self._call_or_await()
      if (
          isinstance(result, Executable) or isinstance(result, Awaitable)
      ) and self.execute_result:
        return await result
      else:
        return result

  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[Update[Result] | Result]:
    """Executes the arguments of and iterates through the wrapped function.

    If the wrapped function is a normal function or a coroutine (hence producing
    a single result), this will yield its result and finish, unless the result
    is itself an Executable in which case this will iterate through that
    Executable.
    If the wrapped function is itself an AsyncIterator, this will iterate
    through it and pass on the results.

    Args:
      iteration_depth: When doing recursive calls to other executables, they
        can be executed in one step or iteratively, if the depth is 0 the
        execution will be in one step, but if depth is n the iterative
        execution of subexecutables will be done with depth n-1.
        If set to -1, this means no limitation of the iteration depth.

    Yields:
      Stream of updates produced by the wrapped function.
    """
    if iteration_depth == 0:
      result = await self  # Call _aexec().
      yield result
    else:
      # We execute all arguments except possibly the one that is marked as
      # iterate_argument.
      await self._process_arguments(execute_iterate_argument=False)
      if self.iterate_argument is not None:
        if self.iterate_argument in self.non_executed_args or not isinstance(
            self._arguments[self.iterate_argument], Executable
        ):
          # If the argument is not executable or part of the non_executable_args
          # we yield it.
          yield self._arguments[self.iterate_argument]
        else:
          # Otherwise we yield from the argument.
          updates = updating.Update()
          async for update in self._arguments[self.iterate_argument].with_depth(
              iteration_depth
          ):
            updates += update
            yield update
          # And store the final value
          self._arguments[self.iterate_argument] = updates.to_result()
      # Now we can iterate through the function.
      if self._is_iterator:
        # The wrapped function returns an iterator or async iterator.
        # We iterate through it.
        args, kwargs = utils.get_calling_args_and_kwargs(
            self.wrapped, self._arguments
        )
        if inspect.isasyncgenfunction(self.wrapped):  # Async iterator.
          async for result in self.wrapped(*args, **kwargs):
            yield result
        else:
          # Should be a normal iterator.
          for result in self.wrapped(*args, **kwargs):
            yield result
      else:
        result = await self._call_or_await()
        if isinstance(result, Executable) and self.execute_result:
          async for update in result.with_depth(iteration_depth):
            yield update
        elif isinstance(result, Awaitable) and self.execute_result:
          result = await result
          yield result
        else:
          yield result


async def _wrap_executable(
    executable: Executable[Any], index: int, report_exceptions: bool = False
) -> Any:
  """Wraps an Executable to report its result as a ListUpdate.

  Args:
    executable: Executable to be wrapped.
    index: Index in the original list.
    report_exceptions: If True exceptions are reported as if they were results
      but also raised.

  Returns:
    The result of the executable (and reports via the tracer a corresponding
    update).
  """
  # We manually add a stage to the execution_tracer instead of using the
  # @tracing.trace decorator because we don't want this state to show up
  # in the ExecutionResults.
  tracer = tracing.execution_tracer.get(None)
  tracer_token = None
  if tracer is not None:
    new_tracer = tracer.add_stage()
    tracer_token = tracing.execution_tracer.set(new_tracer)

  try:
    res = await executable._aexec()  # pylint: disable=protected-access
  except Exception as e:
    if report_exceptions:
      # We report the exception.
      await tracing.report_update(ListUpdate([(e, index)]))
    raise e
  else:
    # No exception, we report the result.
    await tracing.report_update(ListUpdate([(res, index)]))
  finally:
    if tracer is not None:
      tracing.execution_tracer.reset(tracer_token)

  return res


class SerialExecutable(Executable[Sequence[Any]]):
  """Wraps an Iterable of Executables that are executed serially."""

  def __init__(self, executables: Iterable[Executable[Any]]):
    self._executables = iter(executables)

  async def _aexec(self) -> Sequence[Any]:
    results = []
    for index, executable in enumerate(self._executables):
      result = await _wrap_executable(executable, index, True)
      results.append(result)
    return results

  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[ListUpdate[Any]]:
    if iteration_depth == 0:
      result = await self
      yield ListUpdate([(Update(r), i) for i, r in enumerate(result)])
    else:
      for i, executable in enumerate(self._executables):
        async for update in executable.with_depth(iteration_depth - 1):
          yield ListUpdate([(update, i)])


class ParallelExecutable(Executable[Sequence[Any]]):
  """Wrapper to execute Executables in parallel.

  Attributes:
    iterable: An iterable of executables to process in parallel.
    chunk_size: Size of the chunks of executables to be executed in parallel. If
      None, the default depends on whether the input is a list or an iterator.
      Indeed, we don't want to materialize the whole list if it is provided as
      an iterator, so we use a default of _DEFAULT_CHUNK_SIZE. However, if the
      input is already fully materialized, we use as chunk_size the length of
      the list. Note that there are specific cases when one may want to have
      smaller chunks than the size of the list, e.g. to force more interleaving
      of various steps in a big pipeline, so the value can be provided
      explicitly.
    return_exceptions: If False, any exception will stop the processing
      and no result will be returned. If True, exceptions produced by individual
      executables will be used as their returned result, so that the processing
      will not be interrupted.
  """

  def __init__(
      self, iterable: Iterable[Executable[Any]], chunk_size: int | None = None,
      return_exceptions: bool = False,
  ):
    if chunk_size is None:
      if isinstance(iterable, list):
        # If the iterable is fully materialized as a list already, we can take
        # the whole list as a single chunk.
        chunk_size = len(iterable)
      else:
        chunk_size = _DEFAULT_CHUNK_SIZE
    self.chunk_size = chunk_size
    # We actually make the input an iterable so that we can chunk it.
    self.iterable = iter(iterable)
    self.return_exceptions = return_exceptions

  async def _aexec(self) -> Sequence[Any]:
    results = []
    while True:
      base_index = len(results)
      chunk = list(itertools.islice(self.iterable, self.chunk_size))
      if not chunk:
        break
      slice_results = await asyncio.gather(
          *[
              _wrap_executable(e, i + base_index, self.return_exceptions)
              for i, e in enumerate(chunk)
          ],
          return_exceptions=self.return_exceptions,
      )
      results.extend(slice_results)
    return results

  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[Update[Any]]:
    if iteration_depth == 0:
      yield Update(await self)
    else:
      async for res, index in iterating.merge_iter(
          (e.with_depth(iteration_depth - 1) for e in self.iterable),
          chunk_size=self.chunk_size,
          return_exceptions=self.return_exceptions,
      ):
        yield ListUpdate([(res, index)])
