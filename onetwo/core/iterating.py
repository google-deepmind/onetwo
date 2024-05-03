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

"""Library with convenience classes for iterating through coroutines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable, Iterator, Generator
import contextlib
import contextvars
import copy
import functools
import inspect
import itertools
import logging
import queue
import threading
from typing import Any, Generic, ParamSpec, TypeVar

from onetwo.core import utils


T = TypeVar('T')
Args = ParamSpec('Args')


class _ThreadWithAsyncioLoop(Generic[T], threading.Thread):
  """A thread that runs an asyncio loop to execute a coroutine."""

  def __init__(
      self,
      name: str,
      coroutine: Coroutine[Any, Any, T],
      loop: asyncio.AbstractEventLoop,
  ):
    super().__init__()
    self.name = name
    self.coroutine = coroutine
    self.loop = loop
    self.result = None

  def run(self):
    try:
      self.result = self.loop.run_until_complete(self.coroutine)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.result = e


def asyncio_run_wrapper(coroutine: Coroutine[Any, Any, T]) -> T:
  """Runs a coroutine in a new thread if the current thread is running asyncio.

  By default this will simply call asyncio.run, but in the case where this is
  called from an already running event loop (e.g. this is what happens
  when running in Google colab, or in Jupyter notebooks), then a new thread
  will be created to run the coroutine in a separate event loop.
  This is to avoid the user to have to call await on the coroutine explicitly.

  Args:
    coroutine: The coroutine to be run.

  Returns:
    The result of the coroutine.

  Raises:
    Any exception that the coroutine may have raised.
  """
  try:
    is_running = asyncio.get_event_loop().is_running()
  except RuntimeError:
    is_running = False
  if is_running:
    loop = asyncio.new_event_loop()
    thr = _ThreadWithAsyncioLoop[T](
        'Thread running an asyncio loop', coroutine, loop
    )
    thr.start()
    thr.join()
    if isinstance(thr.result, Exception):
      raise thr.result
    else:
      return thr.result
  else:
    return asyncio.run(coroutine)


# Note that we use Any as a type hint for the AsyncIterator because we can
# merge several different iterators with different return types. Hence we do
# not want to restrict the merging to be done with iterators returning the
# same type.
async def merge_iter(
    iterable: Iterable[AsyncIterator[Any]],
    chunk_size: int = 100,
    return_exceptions: bool = False,
) -> AsyncIterator[tuple[Any, int]]:
  """Merges an iterable of AsyncIterators into a single one.

  Args:
    iterable: AsyncIterators to be merged (i.e. ran in parallel).
    chunk_size: The number of AsyncIterators to be merged in parallel.
    return_exceptions: If False, any exception will stop the processing and no
      result will be returned. If True, exceptions produced by individual
      iterators will be used as their returned result, so that the processing
      will not be interrupted.

  Yields:
    An AsyncIterator going through the original iterators in parallel and
    wrapping their results into tuples with an index corresponding to which
    iterator produced it.
  """
  offset = 0
  it = iter(iterable)
  while True:
    chunk = list(itertools.islice(it, chunk_size))
    if not chunk:
      break
    async for res, index in merge(*chunk, return_exceptions=return_exceptions):
      yield res, index + offset
    offset += len(chunk)


async def merge(
    *iterables: AsyncIterator[Any],
    return_exceptions: bool = False,
) -> AsyncIterator[tuple[Any, int]]:
  """Merges a list of AsyncIterators into a single one.

  Args:
    *iterables: AsyncIterators to be merged (i.e. ran in parallel).
    return_exceptions: If False, any exception will stop the processing
      and no result will be returned. If True, exceptions produced by individual
      iterators will be used as their returned result, so that the processing
      will not be interrupted.

  Yields:
    An AsyncIterator going through the original iterators in parallel and
    wrapping their results into tuples with an index corresponding to which
    iterator produced it.
  """
  # We keep a list of iterators with their anext() coroutines,
  # and will run these coroutines in parallel.
  to_process = [(it.__aiter__(), None) for it in iterables]
  while any([it is not None for it, _ in to_process]):
    for index, (it, it_anext) in enumerate(to_process):
      if it is not None and it_anext is None:
        future = asyncio.ensure_future(it.__anext__())
        future.index = index
        to_process[index] = (to_process[index][0], future)
    # We execute the anext coroutines and return as soon as one is
    # done.
    done, _ = await asyncio.wait(
        [f for it, f in to_process if it is not None],
        return_when=asyncio.FIRST_COMPLETED,
    )
    results = []
    for future in done:
      to_process[future.index] = (to_process[future.index][0], None)
      try:
        result = future.result()
        results.append((result, future.index))
      except Exception as e:  # pylint: disable=broad-exception-caught
        # An exception has been raised, it is either a processing error or just
        # the end of this iterator, we remove it from the list.
        to_process[future.index] = (None, None)
        if not isinstance(e, StopAsyncIteration):
          # If the exception is of type StopAsyncIteration, it just means we
          # reached the end of the iterator, so there is nothing to be done and
          # we can continue with the other iterators, but if it is of any other
          # type, we need to handle it.
          if not return_exceptions:
            # If return_exceptions is False, we just propagate the exception,
            # which means all results are lost.
            raise e
          else:
            # If return_exceptions is set to True, we simply interrupt the
            # processing of the current iterator and replace its result by the
            # exception it raised.
            results.append((e, future.index))
    # We sort the results by index of the iterator that generated it to ensure
    # reproducibility.
    results.sort(key=lambda x: x[1])
    for result, index in results:
      yield result, index


ProducesT = (
    AsyncIterator[T]
    | Iterator[T]
    | Callable[[Callable[[T], None]], Awaitable[Any] | None]
)
TQueue = queue.Queue[T | object | Exception]
ThreadTarget = Callable[..., None]


def put_or_shutdown(
    q: TQueue[T], item: T | object | Exception, shutdown: threading.Event
) -> bool:
  """Waits and puts an item into a queue unless the shutdown signal is set.

  Note that we use queues with size 1 so that in order to be able to put an
  item into the queue, one has to wait until the queue is empty. Since we want
  to be able to interrupt the processing (via the shutdown signal), this
  function loops and checks the shutdown signal until it can put the item
  into the queue.

  Args:
    q: The queue to put the item into.
    item: The item to put into the queue.
    shutdown: A threading.Event that is set when the processing should stop.

  Returns:
    True if the item has been put, False if the processing has been stopped.
  """
  done = False
  while not done and not shutdown.is_set():
    try:
      q.put_nowait(item)
      done = True
    except queue.Full:
      pass
  return done


def _run_async_iterator_in_queue(
    shutdown: threading.Event,
    iterator: AsyncIterator[T],
    result_queue: TQueue,
    job_done: object,
    context: contextvars.Context,
) -> None:
  """Runs an async iterator and puts its outputs into a queue.

  Args:
    shutdown: A threading.Event to indicate if the processing should be
      interrupted.
    iterator: The iterator to be run.
    result_queue: The queue to put the results of the iterator.
    job_done: The job_done object to put in the queue.
    context: The context to run the iterator in.
  """

  async def wrapper():
    try:
      async for res in iterator:
        # We need to copy the result before putting it the queue since we will
        # give back control to the iterator which may later override this
        # result before the outer thread having had a chance to get it from
        # the queue.
        if not put_or_shutdown(result_queue, copy.deepcopy(res), shutdown):
          return
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Exception raised in async iterator thread: %s', e)
      put_or_shutdown(result_queue, e, shutdown)
      return
    put_or_shutdown(result_queue, job_done, shutdown)

  context.run(asyncio_run_wrapper, wrapper())


def _run_iterator_in_queue(
    shutdown: threading.Event,
    iterator: Iterator[T],
    result_queue: TQueue,
    job_done: object,
) -> None:
  """Runs an iterator and puts its outputs into a queue.

  Args:
    shutdown: A threading.Event to indicate if the processing should be
      interrupted.
    iterator: The iterator to be run.
    result_queue: The queue to put the results of the iterator.
    job_done: The job_done object to put in the queue.
  """
  try:
    for res in iterator:
      # We need to copy the result before putting it the queue since we will
      # give back control to the iterator which may later override this
      # result before the outer thread having had a chance to get it from
      # the queue.
      if not put_or_shutdown(result_queue, copy.deepcopy(res), shutdown):
        return
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info('Exception raised in async iterator thread: %s', e)
    put_or_shutdown(result_queue, e, shutdown)
    return
  put_or_shutdown(result_queue, job_done, shutdown)


def _run_function_with_callback_in_queue(
    shutdown: threading.Event,
    function: Callable[[Callable[[T], None]], None],
    result_queue: TQueue,
    job_done: object,
) -> None:
  """Runs a function with a callback putting the callback outputs into a queue.

  Args:
    shutdown: A threading.Event to indicate if the processing should be
      interrupted.
    function: The function to be run.
    result_queue: The queue to put the results of the function.
    job_done: The job_done object to put in the queue.
  """
  def callback(data: Any):
    nonlocal result_queue
    put_or_shutdown(result_queue, data, shutdown)

  try:
    function(callback)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info('Exception raised in function with callback thread: %s', e)
    put_or_shutdown(result_queue, e, shutdown)
    return
  put_or_shutdown(result_queue, job_done, shutdown)


def _run_coroutine_with_callback_in_queue(
    shutdown: threading.Event,
    function: Callable[[Callable[[T], None]], Awaitable[Any]],
    result_queue: TQueue,
    job_done: object,
    context: contextvars.Context,
) -> Any:
  """Runs a function with a callback putting the callback outputs into a queue.

  Args:
    shutdown: A threading.Event to indicate if the processing should be
      interrupted.
    function: The function to be run.
    result_queue: The queue to put the results of the function.
    job_done: The job_done object to put in the queue.
    context: The context to run the function in.
  """
  def callback(data: Any):
    nonlocal result_queue
    put_or_shutdown(result_queue, data, shutdown)

  async def wrapper() -> Any:
    try:
      result = await function(callback)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Exception raised in function with callback thread: %s', e)
      put_or_shutdown(result_queue, e, shutdown)
      return
    put_or_shutdown(result_queue, job_done, shutdown)
    return result

  final_result = context.run(asyncio_run_wrapper, wrapper())
  put_or_shutdown(result_queue, final_result, shutdown)


@contextlib.contextmanager
def _start_queue_and_thread(
    target: ThreadTarget, wrapped: ProducesT, max_size: int = 1, **kwargs
) -> Iterator[tuple[TQueue, object]]:
  """Starts a thread and returns a queue and a job_done object.

  Args:
    target: The target function to be run in the thread.
    wrapped: The wrapped function that the thread will process.
    max_size: The queue size. By default it is 1 which means that every item
      that will be put there will block the queue until it is retrieved. This
      guarantees that no item will be missed. Set it to 0 in order not to have
      a limit (which makes addition to the queue non-blocking).
    **kwargs: Additional keyword arguments to be passed to the target.

  Yields:
    A tuple of the queue and job_done object.
  """
  # We create a queue with maxsize=1 to ensure that the underlying thread
  # will not proceed until the last emitted result is retrieved from the
  # queue.
  q = queue.Queue(maxsize=max_size)
  job_done = object()

  shutdown = threading.Event()
  t = threading.Thread(
      target=target, args=(shutdown, wrapped, q, job_done), kwargs=kwargs
  )
  t.start()
  try:
    yield q, job_done
  except Exception as e:
    # An exception was raised in the main code (e.g. in the body of the stream
    # loop).
    shutdown.set()
    t.join()  # Wait for the end of the thread, but don't wait for the queue.
    raise e
  t.join()  # Wait for the end of the thread.
  q.join()  # Wait for the queue to be processed.


async def _async_process_queue(
    result_queue: TQueue[T], job_done: object
) -> AsyncIterator[T]:
  """Reads from the queue as an async iterator.

  Args:
    result_queue: The queue to read items from.
    job_done: An object indicating the end of the queue.

  Yields:
    Items from the queue.
  """
  while True:
    try:
      # Check whether we have something in the queue.
      result = result_queue.get_nowait()
      result_queue.task_done()
    except queue.Empty:
      continue
    if isinstance(result, Exception):
      raise result
    if result is job_done:
      break
    yield result


def _process_queue(result_queue: TQueue[T], job_done: object) -> Iterator[T]:
  """Reads from the queue as an iterator.

  Args:
    result_queue: The queue to read items from.
    job_done: An object indicating the end of the queue.

  Yields:
    Items from the queue.
  """
  while True:
    try:
      # Check whether we have something in the queue.
      result = result_queue.get_nowait()
      result_queue.task_done()
    except queue.Empty:
      continue
    if isinstance(result, Exception):
      raise result
    if result is job_done:
      break
    yield result


def _process_queue_with_result(
    result_queue: TQueue[T], job_done: object
) -> Generator[T, None, Any]:
  """Reads from the queue as an iterator.

  Args:
    result_queue: The queue to read items from.
    job_done: An object indicating the end of the queue.

  Yields:
    Items from the queue.
  """
  while True:
    try:
      # Check whether we have something in the queue.
      result = result_queue.get_nowait()
      result_queue.task_done()
    except queue.Empty:
      continue
    if isinstance(result, Exception):
      raise result
    if result is job_done:
      break
    yield result
  # Extract the final result from the queue
  result = result_queue.get()
  result_queue.task_done()
  return result


@contextlib.contextmanager
def async_iterator_to_sync(iterator: AsyncIterator[T]) -> Iterator[Iterator[T]]:
  """Converts an async iterator into a sync iterator.

  This is calling asyncio.run in a thread to execute the async iterator and
  pass its results to a queue which is read in the main thread.
  This should be used as a context manager (so that the thread is properly
  stopped in case of an exception in the loop body), hence:
  ```
  with async_iterator_to_sync(iterator) as it:
    for item in it:
      ...
  ```

  Args:
    iterator: The AsyncIterator to be executed.

  Yields:
    An Iterator that yields items from the AsyncIterator.
  """
  with _start_queue_and_thread(
      _run_async_iterator_in_queue,
      iterator,
      max_size=1,
      context=contextvars.copy_context(),
  ) as (
      result_queue,
      job_done,
  ):
    yield _process_queue(result_queue, job_done)


@contextlib.contextmanager
def sync_iterator_to_async(
    iterator: Iterator[T],
) -> Iterator[AsyncIterator[T]]:
  """Converts a sync iterator into an async iterator.

  This is running the sync iterator in a thread so that the iterator is not
  blocking the main thread which is thus able to perform other tasks while
  waiting for the next input from the iterator.
  This should be used as a context manager (so that the thread is properly
  stopped in case of an exception in the loop body), hence:
  ```
  with sync_iterator_to_async(iterator) as it:
    async for item in it:
      ...
  ```

  Args:
    iterator: The Iterator to be executed.

  Yields:
    An AsyncIterator that yields items from the Iterator.
  """
  with _start_queue_and_thread(
      _run_iterator_in_queue, iterator, max_size=1
  ) as (
      result_queue,
      job_done,
  ):
    yield _async_process_queue(result_queue, job_done)


@contextlib.contextmanager
def function_with_callback_to_async_iterator(
    function: Callable[[Callable[[T], None]], None]
) -> Iterator[AsyncIterator[T]]:
  """Converts a function with a callback into an async iterator.

  This is running the function in a thread and everytime the callback is called
  by the function, the result is passed to a queue which is monitored in the
  main thread.
  This should be used as a context manager (so that the thread is properly
  stopped in case of an exception in the loop body), hence:
  ```
  with function_with_callback_to_async_iterator(function) as it:
    async for item in it:
      ...
  ```

  Args:
    function: The function to be run in the thread.

  Yields:
    An AsyncIterator that yields items from the callback.
  """
  with _start_queue_and_thread(
      _run_function_with_callback_in_queue, function, max_size=1
  ) as (
      result_queue,
      job_done,
  ):
    yield _async_process_queue(result_queue, job_done)

R = TypeVar('R')


@contextlib.contextmanager
def coroutine_with_callback_to_sync_iterator(
    coroutine: Callable[[Callable[[T], None]], Awaitable[R]]
) -> Iterator[Generator[T, None, R]]:
  """Converts a function with a callback into a sync iterator.

  This is running the function in a thread and everytime the callback is called
  by the function, the result is passed to a queue which is monitored in the
  main thread.
  This should be used as a context manager (so that the thread is properly
  stopped in case of an exception in the loop body), hence:
  ```
  with coroutine_with_callback_to_sync_iterator(function) as it:
    for item in it:
      ...
  ```

  Args:
    coroutine: The function to be run in the thread.

  Yields:
    An Iterator that yields items from the callback.
  """

  with _start_queue_and_thread(
      _run_coroutine_with_callback_in_queue,
      coroutine,
      max_size=0,
      context=contextvars.copy_context(),
  ) as (
      result_queue,
      job_done,
  ):
    yield _process_queue_with_result(result_queue, job_done)


def to_thread_iterator(
    function: Callable[Args, Iterator[T]]
) -> Callable[Args, AsyncIterator[T]]:
  """Decorator to make the iterator execute in a thread.

  This is to be used for functions which call some external engine or do some
  complex I/O operation so that several of them can be executed in parallel.
  Functions decorated with this decorator should not affect the state of global
  variables.

  The effect of decorating an iterator will be to return an async iterator.
  So if one decorates, say:
  ```
  @to_thread
  def my_fn(x: int):
    ...
    yield return_value
  ```
  Then `my_fn` will have to be executed as an async iterator as:
  ```
  for res in my_fn(value):
    ...
  ```

  Args:
    function: A normal (non-async) iterator to be executed in a thread.

  Returns:
    A function returning an AsyncIterator.
    The input signature is the same as that of the decorated function.

  Raises:
    ValueError when attempting to decorate an async generator function.
  """
  if inspect.isasyncgenfunction(function):
    raise ValueError(
        '@to_thread_iterator decorator should be applied to a blocking iterator'
        ' not an async generator function.'
    )

  @functools.wraps(function)
  async def wrapper(*args, **kwargs) -> AsyncIterator[Any]:
    with sync_iterator_to_async(function(*args, **kwargs)) as iterator:
      async for res in iterator:
        yield res

  return wrapper


def to_thread(
    function: Callable[Args, Any] | None = None,
    /,
    *,
    timeout: float | None = None,
):
  """Decorator to make the function execute in a thread.

  This is to be used for functions which call some external engine or do some
  complex I/O operation so that several of them can be executed in parallel.
  Functions decorated with this decorator should not affect the state of global
  variables.

  The effect of decorating a (normal) function with this decorator is to return
  a coroutine function.
  So if one decorates, say:
  ```
  @to_thread
  def my_fn(x: int):
    ...
  ```
  Then `my_fn` will have to be awaited as follows:
  ```
  result = await my_fn(value)
  ```

  This decorator can be used with or without args - i.e. `@to_thread` or
  `@to_thread(timeout=10)`

  Args:
    function: A normal (non-async) function to be executed in a thread.
    timeout: Optional timeout to wait for the function to finish.

  Returns:
    A coroutine function with the same input signature as that of the decorated
    function.

  Raises:
    ValueError when attempting to decorate an async function.
  """

  def wrap(func):
    if utils.returns_awaitable(func):
      raise ValueError(
          '@to_thread decorator should be applied to a blocking function, not a'
          ' coroutine function.'
      )

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
      return await asyncio.wait_for(
          asyncio.to_thread(func, *args, **kwargs), timeout
      )

    return wrapper

  if function is None:  # this means it was used as `@to_thread(timeout=...)`
    return wrap

  # this means it was used as `@to_thread``
  return wrap(function)
