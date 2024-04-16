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

"""Library for making batched functions and methods with decorators.

In this file we use words `request` and `reply` to mean function (or method)
inputs and outputs respectively.

# Usage:

## Decorate a function

```
@batch_function(batch_size=10)
def my_function(requests: Sequence[_RequestT]) -> Sequence[_ReplyT]:
  ...

# my_function should be now awaited.
async def wrapper():
  res = []
  for req in requests:
    result = await my_function([req])
    res.append(result)
  return res

results = run(wrapper())
```
The execution will automatically become asynchronous and will happen only when
the batches are full or if there are no more calls.
These calls can be made in arbitrary asynchronous programs.


## Decorate a method

In the case of a method, the arguments passed to the decorator can depend on
other attributes or methods of the same class. This would be indicated with
`FromInstance('param_name')` which indicates which parameter (or method) is
to be used for the value of that argument at runtime.
In addition, it is required that the class itself has batching enabled as it
will need to create queues possibly for each different instance from that class.
This is done with the `@add_batching` decorator.

```
@add_batching
class SomeClass:
  my_batch_size: int

  @batch_method(batch_size=FromInstance('my_batch_size'))
  def my_method(requests: Sequence[_RequestT]) -> Sequence[_ReplyT]:
    ...

# my_method should be now awaited.
async def wrapper():
  res = []
  for req in requests:
    result = await SomeClass().my_method([req])
    res.append(result)
  return res

results = run(wrapper())
```

## Batching function (batching_function)

One parameter that can be passed to the decorators is a batching function.
It helps to compose batches out of multiple separate calls to the function  (or
method). Assuming we have a function `my_fn(a, b)` that takes certain arguments,
batching_function decides whether a new request `my_fn(a=1, b='bla')` (expressed
as `request = {'a':1, 'b':'bla}`) can be appended to the already existing batch
`batch = (req`, re2, ...)` of requests (i.e. whether the new request is
"compatible" with other requests in the batch) and whether the batch is full (or
will be full after appending a new request). The notion of "compatible" can be
different between the use cases. User has an option of not specifying
batching_function. In that case use must specify the batch_size and default
batching mechanism is used, where any requests are assumed compatible and we add
new requests to the batch as long as batch_size has not been reached.

## Separate the call and implementation

Another decorator is provided to separate specifying the signature of the
function to be called with a single request, and the implementation of the
function that processes a list of requests.
The reason to have this is when registering the function to be used in the rest
of the code, it is preferable to register the function that takes parameters
for a single request, so that the rest of the code will not have to know that
there is batching happening, and what will be exposed is directly the signature
of how to provide one request instead of a list-of-dict signature.

In particular, the code may have calls of the `single_function` (see example
below) peppered throughout, some of which may be wrapped in loops at various
levels, some may be in different coroutines, etc... and all these calls would
end up being batched automatically, and what would be called eventually is the
`my_batch_function`.

```
# `implementation` should point to a function decorated with `batch_function`.
@batchable_function(implementation=my_batch_function)
def single_function(a: int, b: str):
  # No implementation needed, this simply dictates the signature.
  pass

@batch_function(batch_size=10)
def my_batch_function(requests: Sequence[Mapping[str, Any]]):
  # Actually implement the processing in batches.
  # Requests are passed as a Sequence of dict of {arg_name: arg_value}.

# Now in the rest of the code, my_batch_function does not have to be called, and
# one can just use single_function everywhere, as it would only always send one
# request at a time (the batching would happen automatically when possible).
# Note that the single_function must be awaited, as it is now a coroutine
# (allowing asynchronous execution hence batching).

...
result = await single_function(a=1, b='z')
...
result = await single_function(a=2, b='a')
```

## Debug

An additional parameter can be passed to the `@batch_function` and
`@batch_method` decorators: `debug=True`. This indicates that the calls to the
underlying batch function should be stored for debugging purposes.

TODO: Currently it is pretty complicated to access the underlying
queue from the decorated functions (see batching_test.py). Ideally there should
be a uniform and straightforward mechanism.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator, Mapping
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import logging
from multiprocessing import pool as mp_pool
import queue
import threading
import types
from typing import Any, Generic, Literal, ParamSpec, Sequence, Type, TypeAlias, TypeVar, overload

from onetwo.core import iterating
from onetwo.core import results
from onetwo.core import tracing
from onetwo.core import utils


_T = TypeVar('_T')
# Following TypeVars are needed to correctly type hint batch_function,
# batch_method, batchable_function, and batchable_method.
_RequestT = TypeVar('_RequestT')
_ReplyT = TypeVar('_ReplyT')

# Following ParamSpec is required to correctly type hint batchable_function and
# batchable_method.
_Args = ParamSpec('_Args')

_BatchingFunction: TypeAlias = (
    Callable[[Sequence[_RequestT], _RequestT], tuple[bool, bool]]
)
_BatchedFunction: TypeAlias = (
    Callable[[Sequence[_RequestT]], Sequence[_ReplyT | Sequence[_ReplyT]]]
)
_BatchedMethod: TypeAlias = (
    Callable[[Any, Sequence[_RequestT]], Sequence[_ReplyT | Sequence[_ReplyT]]]
)
# Reply possibly appended with tracing (debug) information.
ResultType: TypeAlias = _ReplyT | tuple[_ReplyT, results.ExecutionResult]

Parameters: TypeAlias = Mapping[str, Any]
ParametersBatch: TypeAlias = Sequence[Parameters]

_enable_batching = contextvars.ContextVar('enable_batching', default=True)
_thread_data = threading.local()


class _Container:
  """Input (request) and output (reply) of the function to be batched.

  Attributes:
    payload: The input to the function, represented as a Mapping from parameter
      name to parameter value.
    result: Whatever the function will return (assuming no Exception is raised).
    exception: Exception that the function will raise, if any. If this is
      populated, then `result` will normally be `None`.
    done: True when the result has been obtained from the function.
  """

  def __init__(self, payload: Parameters):
    self.payload = payload
    self.result = None
    self.exception = None
    self.done = False

  def __repr__(self) -> str:
    if self.result is None and self.exception is None:
      return repr(self.payload)
    elif self.exception is not None:
      return (
          f'_Container({repr(self.payload)}, exception={repr(self.exception)})'
      )
    else:
      return f'_Container({repr(self.payload)}, {repr(self.result)})'


@dataclasses.dataclass(order=True)
class PrioritizedItem:
  priority: int
  item: Any = dataclasses.field(compare=False)


def default_batching(
    batch_size: int, batch: Sequence[Any], unused_req: Any
) -> tuple[bool, bool]:
  """Default batching function.

  Allows requests to be added to the `batch` while its size is lower than
  `batch_size`. Says that `batch` is full if the `batch_size` is reached after
  adding `unused_req` to it.

  Args:
    batch_size: Maximal batch size.
    batch: Current batch of requests.

  Returns:
    A tuple of can_add, is_full bools.
  """
  del unused_req
  return (len(batch) < batch_size, len(batch) >= batch_size - 1)


@dataclasses.dataclass
class BatchQueue(Generic[_RequestT, _ReplyT]):
  """Implements a generic queue of batched calls.

  Not to be directly instantiated, but rather using one of the decorators.

  Note that such an object can be called from different threads, so all the
  thread-specific properties are stored as thread-local variables indexed
  by the object id.

  Attributes:
    wrapped: The wrapped method that executes the batches.
    debug: True if the calls are stored for debugging.
    batch_size: If provided (to the method decorator), the size of the batches
      to send to the wrapped function.
    batching_function: Helps to compose batches out of multiple separate calls
      to the function  (or method). Assuming we have a function `my_fn(a, b)`
      that takes certain arguments, batching_function decides whether a new
      request `my_fn(a=1, b='bla')` (expressed as `request = {'a':1, 'b':'bla}`)
      can be appended to the already existing batch `batch = (req`, re2, ...)`
      of requests (i.e. whether the new request is "compatible" with other
      requests in the batch) and whether the batch is full (or will be full
      after appending a new request). Meaning of "compatible" can be different
      between the use cases. It returns a tuple (can_add, is_full), where
      can_add indicates whether the new request can be added to the batch, and
      is_full indicates whether the batch is already full (True) or will be full
      after adding the new request (True) otherwise False.
    request_queue: (thread-local) The current list of requests to be executed.
    calls: (thread-local) List of calls performed to the wrapped function
      (if debug=True).
    task: (thread-local) asyncio Task for watching the queue (start method).
  """

  wrapped: Callable[..., Any]
  batch_size: utils.RuntimeParameter[int]
  batching_function: utils.RuntimeParameter[_BatchingFunction[_RequestT]]
  debug: bool = dataclasses.field(default=False)

  def _maybe_create_request_queue(self):
    if not hasattr(_thread_data, 'request_queues'):
      _thread_data.request_queues = {}
    if id(self) not in _thread_data.request_queues:
      # If this is the first call, we create a queue.
      # It will have to be closed at the end.
      _thread_data.request_queues[id(self)] = queue.PriorityQueue()
      # There should not be any watcher task running.
      if self.task is not None:
        raise ValueError('Queue watcher already running.')
      self.task = asyncio.create_task(self.start())
      self.calls = []

  @property
  def request_queue(self) -> queue.PriorityQueue:
    return _thread_data.request_queues.get(id(self), None)

  @property
  def task(self) -> asyncio.Task | None:
    if not hasattr(_thread_data, 'tasks'):
      _thread_data.tasks = {}
    return _thread_data.tasks.get(id(self), None)

  @task.setter  # Setter for property `task`.
  def task(self, value: asyncio.Task):
    if not hasattr(_thread_data, 'tasks'):
      _thread_data.tasks = {}
    _thread_data.tasks[id(self)] = value

  @property
  def calls(self) -> list[_RequestT]:
    if not hasattr(_thread_data, 'calls'):
      _thread_data.calls = {}
    return _thread_data.calls.get(id(self), None)

  @calls.setter
  def calls(self, value: Sequence[_RequestT]):
    if not hasattr(_thread_data, 'calls'):
      _thread_data.calls = {}
    _thread_data.calls[id(self)] = value

  async def coroutine(self, requests: Sequence[_RequestT]) -> Sequence[_ReplyT]:
    """Coroutine to be called (wraps the underlying function)."""
    enable_batching = _enable_batching.get()
    if enable_batching:
      if (
          not hasattr(_thread_data, 'running_queues')
          or _thread_data.running_queues is None
      ):
        raise ValueError(
            'Cannot call a function with a batch decorator outside of'
            ' `batching.run()`.'
        )
      # We make sure this object is in the running_queues for the local thread.
      _thread_data.running_queues[id(self)] = self
      self._maybe_create_request_queue()
      # For convenience we wrap the requests into _Container objects and will
      # carry those around so that we can be sure that the result is attached
      # to the corresponding request.
      wrapped_requests = [_Container(r) for r in requests]
      for r in wrapped_requests:
        item = PrioritizedItem(priority=self.request_queue.qsize(), item=r)
        self.request_queue.put(item)
      while True:
        # This is the typical way one yields control to the asyncio event loop.
        # It is like saying "if there is anything else you can do, go do it and
        # come back after".
        await asyncio.sleep(0)
        # Now we check whether all the results are ready and return if they are.
        if self.task is None:
          raise ValueError('No task watching for the queue.')
        # Since the start() coroutine is run as a task, if it raises an
        # exception it won't get passed, so we have to periodically check for
        # it.
        try:
          exception = self.task.exception()
          raise exception
        except asyncio.InvalidStateError:
          pass
        if all([r.done for r in wrapped_requests]):
          # If an exception occurred when processing the request, we raise it
          # here in the main thread so as to mimic as closely as possible the
          # behavior the caller would have seen if they had executed the
          # request directly without going via the batch queue. Note that it is
          # important that we raise the exception here in the main thread,
          # rather than in the processing thread, to ensure that the processing
          # thread isn't killed before processing the whole request queue, and
          # to ensure that the caller has the option to handle the exceptions
          # while continuing with processing.
          for r in wrapped_requests:
            if r.exception is not None:
              raise r.exception
          return [r.result for r in wrapped_requests]
        elif self.task.done():
          raise ValueError(
              'Watcher task is done but not all requests are processed.'
          )
    else:
      wrapped_requests = [_Container(r) for r in requests]
      await self.process(wrapped_requests)
      for r in wrapped_requests:
        if r.exception is not None:
          raise r.exception
      return [r.result for r in wrapped_requests]

  async def finish(self, force: bool = False) -> None:
    """Sends a signal to close the queue and stop its processing."""
    if not force:
      # If force is False, we try to finish everything in a clean way, by
      # closing the queue and letting the watcher task finish its job.
      if self.request_queue is not None:
        if not self.request_queue.empty():
          raise ValueError('Attempting to finish a non-empty queue')
        self.request_queue.put(
            PrioritizedItem(priority=0, item=None)
        )
      if self.task is not None and not self.task.done():
        await self.task
        if not self.task.done():
          raise ValueError('Could not close watcher task on queue.')
      self.request_queue.join()
    # Delete the task.
    self.task = None
    # Close and remove the request queue.
    _thread_data.request_queues.pop(id(self))

  async def process(self, requests: Sequence[_Container]):
    """Actually send a batch of requests to the underlying function."""
    if self.debug:
      self.calls.append([req.payload for req in requests])
    # Actually call the wrapped function with a batch of requests.
    try:
      if inspect.iscoroutinefunction(self.wrapped):
        replies = await self.wrapped([req.payload for req in requests])
      else:
        replies = self.wrapped([req.payload for req in requests])
      if not isinstance(replies, Sequence):
        raise ValueError(
            f'Batch function {self.wrapped} should return a Sequence of'
            f' results, got {replies=} for requests {requests=}.'
        )
      if len(replies) != len(requests):
        raise ValueError(
            f'Batch function {self.wrapped} should return as many results as'
            f' requests, got {replies=} for requests {requests=}.'
        )
      # Store the results into the container objects.
      for container, reply in zip(requests, replies):
        container.result = reply
        container.done = True
    except Exception as e:  # pylint: disable=broad-exception-caught
      # If an exception is raised, then we store it in the container objects in
      # place of the results. It is important that we don't raise the exception
      # directly here, so as to ensure that we don't accidentally kill the
      # processing thread before it has finished processing all of the requests
      # in the queue.
      for container in requests:
        container.exception = e
        container.done = True

  def get_batching_function(self) -> _BatchingFunction:
    if self.batching_function.value() is None:
      batch_size = self.batch_size.value()  # pytype: disable=attribute-error
      return functools.partial(default_batching, batch_size)
    else:
      return self.batching_function.value()  # pytype: disable=attribute-error

  async def start(self) -> None:
    """Start the watcher monitoring the call queue."""
    # This is where most of the work happens: we look into the queue, attempt
    # to form a batch and then call the process function to execute the
    # requests in the batch.

    # We look up the batching function when we start, as it may be defined at
    # runtime (the self.batching_function parameter may be a FromInstance
    # object).
    batching_function = self.get_batching_function()
    batch = []
    got_finish_signal = False
    while not got_finish_signal:
      # Possibly do something else to allow the queue to be filled in.
      await asyncio.sleep(0)

      queue_has_elements_to_process = False
      skipped_elements = []

      # We go through the queue once and extract a batch from it.
      while True:
        try:
          if self.request_queue is None:
            # Request queue likely has been killed, we stop going through it.
            break
          # We introduce a small delay to allow the queue to fill up otherwise
          # we may process it too soon (and get half-full batches).
          # TODO: This could become a bottleneck if we use batching
          # on functions that execute really fast, so we might need to make
          # this customizable.
          await asyncio.sleep(0.01)
          pitem = self.request_queue.get_nowait()
          request = pitem.item
          self.request_queue.task_done()
        except queue.Empty:
          # Queue is empty, we stop going through it.
          break
        if request is not None:
          queue_has_elements_to_process = True
          batch_requests = [b.payload for b in batch]
          current_request = request.payload
          can_add, is_full = batching_function(
              batch_requests, current_request
          )
          if can_add:
            # Adding to the batch.
            batch.append(request)
          else:
            # We keep the skipped_element to put it back in the queue.
            skipped_elements.append(pitem)
          if is_full:
            # Batch is full, we stop going through the queue.
            break
        else:
          # We have a None request, we should stop after processing
          # this final batch.
          got_finish_signal = True
          break
      # Before processing the batch, we put back the skipped elements into the
      # queue for the next round, and we prioritize them according to their
      # original priority.
      # In a multithread setting, it is possible that another thread would have
      # injected new elements in the queue while we were going through it,
      # so those injected elements could have higher priority artificially.
      for pitem in skipped_elements:
        self.request_queue.put(pitem)

      # Batch is ready, we process it.
      if batch:
        await self.process(batch)
        batch = []
      elif queue_has_elements_to_process:
        raise ValueError(
            'Batching queue not empty but none got batched by the batching'
            ' function.'
        )


async def _finish_queues(force: bool = False) -> None:
  """Sends the finish signal to all queues and erases the running_queues dict.

  Args:
    force: If True, we do not wait for the watcher tasks to finish and simply
      reset them.
  """
  if hasattr(_thread_data, 'running_queues'):
    running_queues = _thread_data.running_queues
    if running_queues is not None:
      # Send the finish signal to all queues.
      active_queues = [q.finish(force) for q in running_queues.values()]
      _thread_data.running_queues = None
      if active_queues:
        await asyncio.gather(*active_queues)


def force_reset() -> None:
  """Forces the reset of batching queues.

  Meant to be used in a colab when execution has been interrupted and one wants
  to run again without having to restart the runtime.
  This may leave hanging threads in the background so one may still need to
  restart the runtime if too many runs have been interrupted.
  """
  iterating.asyncio_run_wrapper(_finish_queues(force=True))


@contextlib.asynccontextmanager
async def running_context(
    enable_batching: bool = True, tracer: tracing.Tracer | None = None
):
  """Context manager that takes care of starting and stopping the queues.

  Args:
    enable_batching: True if batching is enabled (in which case the queues are
      used). If False, the requests to the batched functions happen as they are
      sent.
    tracer: Optional Tracer for recording the execution.

  Yields:
    None
  Raises:
    ValueError if there are running queues in the current thread.
  """
  if (
      hasattr(_thread_data, 'running_queues') and
      _thread_data.running_queues is not None and
      enable_batching
  ):
    raise ValueError('There should be no running queues.')
  # Initialize the list of queues.
  _thread_data.running_queues = {}
  _enable_batching.set(enable_batching)
  tracing.execution_context.set(results.ExecutionResult())
  tracing.execution_tracer.set(tracer)
  try:
    yield None
    await _finish_queues(force=False)
  finally:
    # We ensure that even if the processing was interrupted, the global
    # variable is cleaned up. Indeed, when running unit tests, the test may
    # keep running and we need to ensure the state is reset for the next test
    # to run as it were started independently.
    await _finish_queues(force=True)
    _thread_data.running_queues = None


@contextlib.contextmanager
def run_once():
  """Context manager that ensures that the run or stream is only run once."""
  if hasattr(_thread_data, 'running') and _thread_data.running:
    raise ValueError('Cannot call run or stream more than once per thread.')
  _thread_data.running = True
  try:
    yield None
  finally:
    _thread_data.running = False


# Following `overload` statements exist exclusively for static pytype checker.
# See https://mypy.readthedocs.io/en/stable/literal_types.html.
# TL;DR: Return type of `run` depends on the `enable_tracing` argument. This
# requires narrowing down its return type everywhere in code so that pytype can
# properly infer types.
@overload
def run(
    executable: Awaitable[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> _ReplyT: ...


@overload
def run(
    executable: Awaitable[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> tuple[_ReplyT, results.ExecutionResult]: ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def run(
    executable: Awaitable[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> ResultType[_ReplyT]: ...


def run(
    executable: Awaitable[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> ResultType[_ReplyT]:
  """Main entry point to run a coroutine with automatic batching.

  There should not be any nested calls to `run()`, `stream()` or
  `stream_with_callback()` within other such calls, or within async functions or
  within an `executing.Executable`.
  In order to combine multiple coroutines or `executing.Executable`(s), one
  should for example create a list of executables and create a single one out
  of them with `parallel(*executables)` or `serial(*executables)`.

  This is very similar in spirit to how `asyncio.run` should be used.

  In particular we use a context manager to guarantee that there is only one
  call.

  Args:
    executable: The awaitable to be executed.
    enable_batching: If False, all calls to batched functions/methods are
      sent directly, if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine,
      and if True returns both the coroutine result and an ExecutionResult
      tracing the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Returns:
    The result value of the executable if enable_tracing=False or a tuple
    (return_value, execution_result) if enable_tracing=True.
  """

  async def wrap():
    async with running_context(enable_batching=enable_batching, tracer=tracer):
      # Run the coroutine.
      result = await executable
      if enable_tracing:
        return result, tracing.execution_context.get(None)
      else:
        return result

  with run_once():
    return iterating.asyncio_run_wrapper(wrap())


def stream_with_callback(
    iterator: AsyncIterator[_ReplyT],
    callback: Callable[[ResultType[_ReplyT]], Any],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> None:
  """Execute an AsyncIterator and the batching queues with a callback.

  Instead of using a thread (as in `stream`), uses a callback at every yield
  from the iterator.

  Args:
    iterator: The AsyncIterator to run.
    callback: The callback function to call at each yield.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine;
      if True, returns both the coroutine result and an ExecutionResult tracing
      the whole execution.
    tracer: Optional custom tracer to use for recording the execution.
  """

  async def wrapper():
    async with running_context(enable_batching=enable_batching, tracer=tracer):
      execution_result = tracing.execution_context.get(None)
      async for result in iterator:
        if enable_tracing:
          res = callback((result, execution_result))
        else:
          res = callback(result)
        if isinstance(res, Awaitable):
          await res

  with run_once():
    iterating.asyncio_run_wrapper(wrapper())


@overload
def safe_stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Iterator[_ReplyT]]: ...


@overload
def safe_stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Iterator[tuple[_ReplyT, results.ExecutionResult]]]: ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def safe_stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Iterator[ResultType[_ReplyT]]]: ...


@contextlib.contextmanager
def safe_stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[Iterator[ResultType[_ReplyT]]]:
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
  an iterator and can thus be used directly as `for item in stream(iterator):`
  but it will not properly stop the threads in case of an error and may hang.

  Args:
    iterator: The iterator to be executed.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine;
      if True, returns both the coroutine result and an ExecutionResult tracing
      the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Yields:
    An iterator that yields the results yielded by the iterator or a pair
    (result, execution_result).
  """
  async def wrap():
    async with running_context(enable_batching=enable_batching, tracer=tracer):
      execution_result = tracing.execution_context.get(None)
      async for result in iterator:
        if enable_tracing:
          full_result = (result, execution_result)
        else:
          full_result = result
        yield full_result

  with run_once():
    with iterating.async_iterator_to_sync(wrap()) as it:
      yield it


@contextlib.contextmanager
def stream_updates(  # pytype: disable=invalid-annotation
    coroutine: Awaitable[_ReplyT],
    enable_batching: bool = True,
    iteration_depth: int = 1,
) -> Iterator[tracing.IteratorWithReturnValue[Any, _ReplyT]]:
  """Stream updates from a coroutine.

  Args:
    coroutine: The coroutine to stream.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    iteration_depth: The depth of the nested coroutines to stream.

  Yields:
    An IteratorWithReturnValue that produces the updates from the coroutine
    and its final result.
  """
  def _stream_updates(
      callback_wrapper: Callable[[Callable[[Any], None]], Awaitable[_ReplyT]]
  ) -> Generator[Any, None, _ReplyT]:
    with iterating.coroutine_with_callback_to_sync_iterator(
        callback_wrapper
    ) as it:
      wrapper = tracing.IteratorWithReturnValue(it)
      for i in wrapper:
        yield i
    return wrapper.value

  async def wrapper(callback: Callable[[Any], None]) -> Awaitable[_ReplyT]:
    # We add 1 because the first @trace call will decrease the depth by 1,
    # but we leave -1 unchanged.
    depth = iteration_depth + 1 if iteration_depth >= 0 else iteration_depth
    tracer = tracing.OutputTracer(callback=callback, depth=depth)
    async with running_context(enable_batching=enable_batching, tracer=tracer):
      return await coroutine

  with run_once():
    yield tracing.IteratorWithReturnValue(_stream_updates(wrapper))


@overload
def stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[False] = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[_ReplyT]: ...


@overload
def stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: Literal[True] = True,
    tracer: tracing.Tracer | None = None,
) -> Iterator[tuple[_ReplyT, results.ExecutionResult]]: ...


# The last overload is a fallback in case the caller provides a regular bool:
@overload
def stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[ResultType[_ReplyT]]: ...


def stream(
    iterator: AsyncIterator[_ReplyT],
    enable_batching: bool = True,
    enable_tracing: bool = False,
    tracer: tracing.Tracer | None = None,
) -> Iterator[ResultType[_ReplyT]]:
  """Main entry point to run an AsyncIterator with automatic batching.

  **WARNING**: this will not properly end the threads if an exception is raised
  in the loop body.
  This is fine to use in a colab, but if you use it in a normal python program
  or in a test, this may lead to the program hanging indefinitely.
  Use instead safe_stream to avoid this issue.

  This uses threading to convert an AsyncIterator into a sync one.
  The AsyncIterator is run in a separate thread, while the main thread is
  watching for updates from that separate thread.

  Args:
    iterator: The iterator to be executed.
    enable_batching: If False, all requests to batched functions/methods are
      sent directly; if True, the batching queues are used.
    enable_tracing: If False, directly returns the result from the coroutine;
      if True, returns both the coroutine result and an ExecutionResult tracing
      the whole execution.
    tracer: Optional custom tracer to use for recording the execution.

  Yields:
    The results yielded by the iterator or a pair (result, execution_result).
  """
  with safe_stream(iterator, enable_batching, enable_tracing, tracer) as it:
    yield from it


def add_batching(my_class: Type[_T]) -> Type[_T]:
  """Class decorator for classes whose methods can be batched."""
  my_class.__old_init__ = my_class.__init__

  @functools.wraps(my_class.__init__)
  def new_init(obj, /, *args, **kwargs):
    # Perform the regular __init__ of the wrapped object.
    # We call __old_init__ via getattr to avoid a pylint error when
    # doing my_class.__old_init__.
    getattr(my_class, '__old_init__')(obj, *args, **kwargs)
    # Retrieve methods that have been decorated with @batch_method.
    decorated_methods = [
        method
        for _, method in inspect.getmembers(my_class, inspect.isfunction)
        if hasattr(method, 'is_batched')
    ]
    # Bind those methods to the instance.
    bound_methods = [
        types.MethodType(
            method.original_function, obj  # pytype: disable=attribute-error
        )
        for method in decorated_methods
    ]
    # Create batching queues for each of those methods (unless this inherits
    # from an already decorated object).
    if not hasattr(obj, 'batching_queues'):
      obj.batching_queues = {}
    for original, bound in zip(decorated_methods, bound_methods):
      obj.batching_queues[original.__qualname__] = BatchQueue(
          wrapped=bound,
          batch_size=utils.RuntimeParameter(original.batch_size, obj),
          batching_function=utils.RuntimeParameter(
              original.batching_function, obj
          ),
          debug=original.debug,
      )

  my_class.__init__ = new_init
  return my_class


def validate_parameters(
    batch_size: int | None = None,
    batching_function: _BatchingFunction | None = None,
    allow_from_instance: bool = False,
) -> None:
  """Checks that the parameters of the decorator are correctly provided."""
  if batch_size is None and batching_function is None:
    raise ValueError(
        'Should provide at least one of batch_size or batching_function.'
    )
  if batch_size is not None and batching_function is not None:
    raise ValueError(
        'Should provide either batch_size or batching_function.'
    )
  if not allow_from_instance:
    if isinstance(batch_size, utils.FromInstance) or isinstance(
        batching_function, utils.FromInstance
    ):
      raise ValueError('Arguments of a function decorator cannot be inferred at'
                       ' runtime.')


# Function decorator, replaces the function with an awaitable.
def batch_function(
    batch_size: int | None = None,
    batching_function: _BatchingFunction[_RequestT] | None = None,
    debug: bool = False,
) -> Callable[
    [_BatchedFunction[_RequestT, _ReplyT]],
    Callable[[Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]
]:
  """Function decorator to make it executed in batches.

  The function to be decorated should take a list of objects and process them
  entirely.
  Once decorated, it can be called with lists of arbitrary sizes, but the
  processing will only happen once it has received enough objects to form a
  batch (whose size can be controlled either simply with the batch_size
  parameter or with a user-provided batching_function).

  Args:
    batch_size: If provided, batching will be done simply by checking the size.
    batching_function: If provided, this is used to dermine whether new requests
      can be added to an existing batch. See BatchQueue for more information.
    debug: If True, the calls to the function will be saved for debugging.

  Returns:
    A wrapper that makes the function into a batched coroutine.
  """
  validate_parameters(batch_size, batching_function, allow_from_instance=False)

  def wrapper(
      wrapped: _BatchedFunction[_RequestT, _ReplyT]
  ) -> Callable[[Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]:
    batch_queue = BatchQueue(
        wrapped=wrapped,
        batch_size=utils.RuntimeParameter(batch_size),
        batching_function=utils.RuntimeParameter(batching_function),
        debug=debug,
    )
    # Disabling pytype, see https://github.com/google/pytype/issues/527.
    return batch_queue.coroutine  # pytype: disable=bad-return-type

  return wrapper


# Method decorator, replaces the method with a coroutine.
def batch_method(
    batch_size: int | utils.FromInstance[int] | None = None,
    batching_function: _BatchingFunction[_RequestT]
    | utils.FromInstance[_BatchingFunction[_RequestT]]
    | None = None,
    debug: bool = False,
) -> Callable[
    [_BatchedMethod[_RequestT, _ReplyT]],
    Callable[[Any, Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]
]:
  """Function decorator to make it executed in batches.

  The method to be decorated should take a Sequence of objects and process them
  entirely. The class of that method should be decorated with `add_batching`.
  Once decorated, it can be called with lists of arbitrary sizes, but the
  processing will only happen once it has received enough objects to form a
  batch (whose size can be controlled either simply with the batch_size
  parameter or with a user-provided batching_function).

  Args:
    batch_size: If provided, batching will be done simply by checking the size.
    batching_function: If provided, this is used to determine whether new
      requests can be added to an existing batch. See BatchQueue for more
      information.
    debug: If True, the calls to the function will be saved for debugging.

  Returns:
    A wrapper that makes the function into a batched coroutine.
  """
  # Check params at decoration time.
  validate_parameters(batch_size, batching_function, allow_from_instance=True)

  def inner(
      method: _BatchedMethod[_RequestT, _ReplyT]
  ) -> Callable[[Any, Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]:
    # We tag the method for the `add_batching` decorator to know it has been
    # batched.
    method.is_batched = True
    # We attach parameters to the decorated function so that they
    # can be retrieved when executing the function.
    method.batching_function = batching_function
    method.batch_size = batch_size
    # We keep the original function, as it is the one that will have to be
    # called.
    method.original_function = method
    method.debug = debug

    @functools.wraps(method)
    async def wrapped(self, requests: Sequence[_RequestT]) -> Sequence[_ReplyT]:
      if (
          not hasattr(self, 'batching_queues')
          or method.__qualname__ not in self.batching_queues
      ):
        keys = (
            str(self.batching_queues.keys())
            if hasattr(self, 'batching_queues')
            else 'None'
        )
        raise ValueError(
            f'Cannot access the queue for method {method.__qualname__} from '
            f'the object {self.__class__.__qualname__}, please check that you '
            f'decorated it with @add_batching. Existing queues: {keys}'
        )
      batch_queue = self.batching_queues[method.__qualname__]
      return await batch_queue.coroutine(requests)

    # Disabling pytype, see https://github.com/google/pytype/issues/527.
    return wrapped  # pytype: disable=bad-return-type

  return inner


# Unfortunately, pytype does not properly support decorators that change the
# signature: (internal link).
def batchable_function(
    implementation: Callable[[Sequence[Any]], Awaitable[Sequence[_ReplyT]]],
) -> Callable[
    [Callable[..., _ReplyT | Awaitable[_ReplyT]]],
    Callable[..., Awaitable[_ReplyT]],
]:
  """Decorator that redirects calls to the batched version of the function.

  Args:
    implementation: A function decorated with `batch_function` that actually
      implements the batched execution of the function. It should take in a
      Sequence[dict[str, Any]] and there should be one-to-one correspondence
      between names of the arguments of the decorated function and keys in the
      dictionaries. In other words, every element of the Sequence is a
      dictionary that stores argument names and their values for the decorated
      function.

  Returns:
    A function that actually calls the batched implementation under the hood.
  """

  def inner(
      function: Callable[_Args, _ReplyT | Awaitable[_ReplyT]]
  ) -> Callable[_Args, Awaitable[_ReplyT]]:
    @functools.wraps(function)
    async def wrapped(*args: _Args.args, **kwargs: _Args.kwargs) -> _ReplyT:
      arguments = utils.get_expanded_arguments(function, True, args, kwargs)
      replies = await implementation([arguments])
      return replies[0]
    return wrapped

  return inner


# Unfortunately, pytype does not properly support decorators that change the
# signature: (internal link). Resorting to `...`.
def batchable_method(
    implementation: (
        Callable[[Any, Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]
        | utils.FromInstance[
            Callable[[Any, Sequence[_RequestT]], Awaitable[Sequence[_ReplyT]]]
        ]
    ),
    pass_self: bool = False,
) -> Callable[
    [Callable[..., _ReplyT | Awaitable[_ReplyT]]],
    Callable[..., Awaitable[_ReplyT]],
]:
  """Decorator that redirects calls to the batched version of the method.

    Assume we have a class decorated with `add_batching` and we want to decorate
    one of its methods with `batchable_method`. We need to specify the actual
    batched implementation of this method. Often the implementation itself
    will be a method of the same class. For cases like this we let
    `implementation` be of type FromInstance.

    For more details, see batchable_function.

  Args:
    implementation: A method decorated with `batch_method` that actually
      implements the batched execution of the method.
    pass_self: If True, the implementation method will be passed the `self`
      argument explicitly, otherwise this would happen through the instantiation
      of the FromInstance object.

  Returns:
    A method that actually calls the batched implementation under the hood.
  """
  def inner(
      method: Callable[_Args, _ReplyT | Awaitable[_ReplyT]]
  ) -> Callable[_Args, Awaitable[_ReplyT]]:
    if not utils.is_method(method):
      raise ValueError('batchable_method can decorate only methods.')
    @functools.wraps(method)
    async def wrapped(self, *args, **kwargs):
      arguments = utils.get_expanded_arguments(
          method, True, (self,) + args, kwargs
      )
      assert 'self' in arguments
      del arguments['self']
      batched_function = utils.RuntimeParameter(implementation, self).value()
      if pass_self:
        replies = await batched_function(self, [arguments])
      else:
        replies = await batched_function([arguments])
      return replies[0]

    return wrapped

  return inner


def _render_exceptions(exceptions: Sequence[Exception]) -> str:
  """Renders a sequence of exceptions into a string.

  Args:
    exceptions: A sequence of exceptions.

  Returns:
    A string representation of the exceptions.
  """
  result = ''
  for e in exceptions:
    trace = ''
    tb = e.__traceback__
    while tb is not None:
      trace += (
          f'filename: {tb.tb_frame.f_code.co_filename}\nname:'
          f' {tb.tb_frame.f_code.co_name}\nlineno: {tb.tb_lineno}\n'
      )
      tb = tb.tb_next
    result += f'type: {type(e).__name__}\nmessage: {e}\ntrace: {trace}\n'
  return result


@overload
def run_function_in_threadpool(
    function: Callable[..., _ReplyT],
    return_exceptions: Literal[False] = False,
) -> _BatchedFunction[Parameters, _ReplyT | None]: ...


@overload
def run_function_in_threadpool(
    function: Callable[..., _ReplyT],
    return_exceptions: Literal[True] = True,
) -> _BatchedFunction[Parameters, _ReplyT | None | Exception]: ...


def run_function_in_threadpool(
    function: Callable[..., _ReplyT],
    return_exceptions: bool = False,
) -> _BatchedFunction[Parameters, _ReplyT | None]:
  """Decorator to run a function on a list of inputs in a threadpool.

  Example:
  ```
  @run_function_in_threadpool
  def my_fn(s: str, i: int): str
    ...

  results = my_fn([{'s': 'some str', 'i', 0}, {'s': 'other str', 'i', 1}])
  ```

  Args:
    function: The function to be decorated.
    return_exceptions: If False, any exception will stop the processing and no
      result will be returned. If True, exceptions produced by individual
      requests will be used as their returned result, so that the processing
      will not be interrupted.

  Returns:
    A function which takes a sequence of inputs (each input being a dict
    of parameter values for the decorated function), runs the decorated
    function on each of these inputs using a threadpool, and returns all
    the results as a sequence.

  Raises:
    ValueError if one of the calls to the underlying function failed.
  """
  if return_exceptions:
    function = utils.returning_raised_exception(function)

  @functools.wraps(function)
  def wrapper(requests: ParametersBatch) -> Sequence[_ReplyT | None]:
    exceptions = []
    def executor(kwargs: Parameters) -> _ReplyT | None | Exception:
      nonlocal exceptions
      try:
        return function(*(), **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught
        exceptions.append(e)
        return None

    with mp_pool.ThreadPool(len(requests)) as pool:
      replies = list(pool.map(executor, requests))
      pool.close()
      pool.join()
    if exceptions:
      raise ValueError(
          'One or more exceptions happened during running'
          f' {function.__name__} in threadpool:\n'  # pylint: disable=attribute-error
          + _render_exceptions(exceptions)
      ) from exceptions[0]
    return replies

  return wrapper


@overload
def run_method_in_threadpool(
    method: Callable[..., _ReplyT],
    return_exceptions: Literal[False] = False,
) -> _BatchedMethod[Parameters, _ReplyT | None]: ...


@overload
def run_method_in_threadpool(
    method: Callable[..., _ReplyT],
    return_exceptions: Literal[True] = True,
) -> _BatchedMethod[Parameters, _ReplyT | None | Exception]: ...


def run_method_in_threadpool(
    method: Callable[..., _ReplyT],
    return_exceptions: bool = False,
) -> _BatchedMethod[Parameters, _ReplyT | None]:
  """Decorator to run a function on a list of inputs in a threadpool.

  Example:
  ```
  class C:
    @run_method_in_threadpool
    def my_fn(self, s: str, i: int): str
    ...

  c = C()
  results = c.my_fn([{'s': 'some str', 'i', 0}, {'s': 'other str', 'i', 1}])
  ```

  Args:
    method: The function to be decorated.
    return_exceptions: If False, any exception will stop the processing and no
      result will be returned. If True, exceptions produced by individual
      requests will be used as their returned result, so that the processing
      will not be interrupted.

  Returns:
    A method which takes a sequence of inputs (each input being a dict
    of parameter values for the decorated function), runs the decorated
    method on each of these inputs using a threadpool, and returns all
    the results as a sequence.

  Raises:
    ValueError if one of the calls to the underlying method fails.
  """
  if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
    raise ValueError(
        '@run_method_in_threadpool can not be used with coroutines (i.e.'
        ' functions defined with async def).'
    )

  if return_exceptions:
    method = utils.returning_raised_exception(method)

  @functools.wraps(method)
  def wrapper(self, requests: ParametersBatch) -> Sequence[_ReplyT | None]:
    exceptions = []

    def executor(kwargs: Parameters) -> _ReplyT | None:
      nonlocal exceptions
      try:
        return method(self, *(), **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught
        exceptions.append(e)
        return None

    with mp_pool.ThreadPool(len(requests)) as pool:
      replies = list(pool.map(executor, requests))
      pool.close()
      pool.join()
    if exceptions:
      raise ValueError(
          'One or more exceptions happened during running'
          f' {method.__name__} in threadpool:\n'  # pylint: disable=attribute-error
          + _render_exceptions(exceptions)
      ) from exceptions[0]
    return replies

  return wrapper


def batch_function_with_threadpool(
    batch_size: int | None = None,
    batching_function: _BatchingFunction[_RequestT] | None = None,
    wrapper: (
        Callable[
            [_BatchedFunction[_RequestT, _ReplyT | Exception]],
            _BatchedFunction[_RequestT, _ReplyT | Exception],
        ] | None
    ) = None,
    debug: bool = False,
) -> Callable[
    [Callable[..., _ReplyT]],  # TODO: Replace with _Args.
    Callable[..., Awaitable[_ReplyT]]  # TODO: Replace with _Args.
]:
  """Convenience decorator to create a batched function with a threadpool.

  Writing the following code
  ```
  @batch_function_with_threadpool(batch_size=3, wrapper=my_wrapper)
  def my_fn(s: str, i: int): str
    # Actual implementation of the function on one instance.
    ...
  ```
  is equivalent to writing the following longer code
  ```
  @batchable_function(implementation=my_fn_batch)
  def my_fn(s: str, i: int): str
    pass

  @batch_function(batch_size=3)
  @my_wrapper
  @run_function_in_threadpool
  def my_fn_batch(s: str, i: int): str
    # Actual implementation of the function on one instance.
    ...
  ```

  Args:
    batch_size: If provided, batching will be done simply by checking the size.
    batching_function: If provided, this is used to dermine whether new requests
      can be added to an existing batch. See BatchQueue for more information.
    wrapper: If not None, a wrapper for the batched method (e.g. to perform
      some batch-level logging).
    debug: If True, the calls to the function will be saved for debugging.

  Returns:
    A wrapper that makes the function into a batched coroutine.
  """

  def inner(
      function: Callable[..., _ReplyT]
  ) -> Callable[..., Awaitable[_ReplyT]]:
    threadpool_function = run_function_in_threadpool(
        function, return_exceptions=True
    )
    if wrapper is not None:
      signature = inspect.signature(threadpool_function)
      threadpool_function = wrapper(threadpool_function)
      other_signature = inspect.signature(threadpool_function)
      if signature != other_signature:
        raise ValueError(
            f'The wrapper {wrapper} changed the signature from {signature} to'
            f' {other_signature}.'
        )
    return utils.raising_returned_exception(
        batchable_function(
            implementation=batch_function(
                batch_size=batch_size,
                batching_function=batching_function,
                debug=debug,
            )(threadpool_function)
        )(threadpool_function)
    )

  return inner


def batch_method_with_threadpool(
    batch_size: int | utils.FromInstance[int] | None = None,
    batching_function: (
        _BatchingFunction[_RequestT]
        | utils.FromInstance[_BatchingFunction[_RequestT]]
        | None
    ) = None,
    wrapper: (
        Callable[
            [_BatchedMethod[_RequestT, _ReplyT | Exception]],
            _BatchedMethod[_RequestT, _ReplyT | Exception],
        ] | None
    ) = None,
    debug: bool = False,
) -> Callable[
    [Callable[..., _ReplyT]],  # TODO: Replace with _Args.
    Callable[..., Awaitable[_ReplyT]],  # TODO: Replace with _Args.
]:
  """Convenience decorator to create a batched function with a threadpool.

  Writing the following code
  ```
  @add_batching
  class C:
    @batch_method_with_threadpool(batch_size=3, wrapper=my_wrapper)
    def my_fn(self, s: str, i: int): str
      # Actual implementation of the function on one instance.
      ...
  ```
  is equivalent to writing the following longer code
  ```
  @add_batching
  class C:
    @batchable_method(implementation=utils.FromInstance('my_fn_batch')
    def my_fn(self, s: str, i: int): str
      pass

    @batch_method(batch_size=3)
    @my_wrapper
    @run_method_in_threadpool
    def my_fn_batch(self, s: str, i: int): str
      # Actual implementation of the function on one instance.
      ...
  ```

  Args:
    batch_size: If provided, batching will be done simply by checking the size.
    batching_function: If provided, this is used to determine whether new
      requests can be added to an existing batch. See BatchQueue for more
      information.
    wrapper: If not None, a wrapper for the batched method (e.g. to perform
      some batch-level logging).
    debug: If True, the calls to the function will be saved for debugging.

  Returns:
    A wrapper that makes the function into a batched coroutine.
  """

  def inner(
      method: Callable[..., _ReplyT]
  ) -> Callable[..., Awaitable[_ReplyT]]:
    threadpool_method = run_method_in_threadpool(method, return_exceptions=True)
    if wrapper is not None:
      signature = inspect.signature(threadpool_method)
      threadpool_method = wrapper(threadpool_method)
      other_signature = inspect.signature(threadpool_method)
      if signature != other_signature:
        raise ValueError(
            f'The wrapper {wrapper} changed the signature from {signature} to'
            f' {other_signature}.'
        )
    return utils.raising_returned_exception(
        batchable_method(
            implementation=batch_method(
                batch_size=batch_size,
                batching_function=batching_function,
                debug=debug,
            )(threadpool_method),
            pass_self=True,
        )(threadpool_method)
    )

  return inner


def add_logging(method):
  """Wraps a batch method to add logging and counting calls.

  Usage example:
  ```
    @batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=add_logging,
  )
  ```
  Note that the class on which the method is defined should have a _counters
  field of type collections.Counter. For a dataclass, one can define it as such:
  ```
    _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )
  ```

  Args:
    method: The method to be decorated.

  Returns:
    A decorated method that records calls into the self._counters object.
  """
  @functools.wraps(method)
  def wrapped_method(self, requests: Sequence[Any]) -> Sequence[Any]:
    logging.info(
        'Executing a batch of %d %s requests', len(requests), method.__name__
    )
    # This is a wrapper for a method that should have access to private members.
    # pylint: disable=protected-access
    self._counters[f'{method.__name__}_batches'] += 1
    # pylint: enable=protected-access
    result = method(self, requests)
    logging.info('Received results for %s requests', method.__name__)
    return result

  return wrapped_method
