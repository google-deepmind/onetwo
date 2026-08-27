# Copyright 2026 DeepMind Technologies Limited.
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
import io
import logging
import os
import queue
import threading
import traceback
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
      context: contextvars.Context,
  ):
    super().__init__()
    self.name = name
    self.coroutine = coroutine
    self.loop = loop
    self.context = context
    self.result = None

  def run(self):
    try:
      self.result = self.context.run(
          self.loop.run_until_complete, self.coroutine
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.result = e


_THREAD_LOCAL = threading.local()


def _get_thread_local_runner() -> asyncio.Runner:
  """Returns a persistent asyncio.Runner for the current thread and PID."""
  current_pid = os.getpid()
  runner = getattr(_THREAD_LOCAL, 'runner', None)
  saved_pid = getattr(_THREAD_LOCAL, 'pid', None)

  # Recreate after a fork: the inherited Runner still references the parent's
  # loop and (now-invalid) file descriptors. Runner is lazy, so constructing
  # one here does not create a loop until the first run().
  if runner is None or saved_pid != current_pid:
    runner = asyncio.Runner()
    _THREAD_LOCAL.runner = runner
    _THREAD_LOCAL.pid = current_pid

  return runner


def asyncio_run_wrapper(coroutine: Coroutine[Any, Any, T]) -> T:
  """Runs a coroutine synchronously with per-call context isolation + loop reuse.

  By default this will run the coroutine on a persistent thread-local Runner,
  inheriting the caller's contextvars via a fresh copy while preserving loop and
  socket persistence. If called from an already running event loop (e.g. Google
  colab or Jupyter notebooks), a new thread will be created to run the coroutine
  in a separate event loop.

  Args:
    coroutine: The coroutine to be run.

  Returns:
    The result of the coroutine.

  Raises:
    Any exception that the coroutine may have raised.
  """
  try:
    is_running = asyncio.get_running_loop().is_running()
  except RuntimeError:
    is_running = False
  if is_running:
    with contextlib.closing(asyncio.new_event_loop()) as loop:
      context = contextvars.copy_context()
      thr = _ThreadWithAsyncioLoop[T](
          'Thread running an asyncio loop', coroutine, loop, context
      )
      thr.start()
      try:
        thr.join()
      except BaseException as e:  # pylint: disable=broad-exception-caught
        # The loop may still be running, so we need to explicitly stop it.
        loop.stop()
        thr.join()
        raise e
      if isinstance(thr.result, Exception):
        raise thr.result
      else:
        return thr.result  # pyrefly: ignore[bad-return]
  else:
    # Standard thread (Flume, ThreadPoolExecutor, CLI): reuse the persistent
    # loop (keeps sockets alive) and copy context per call for isolation.
    runner = _get_thread_local_runner()
    try:
      return runner.run(coroutine, context=contextvars.copy_context())
    finally:
      loop = runner.get_loop()
      to_cancel = [t for t in asyncio.all_tasks(loop) if not t.done()]
      if to_cancel:
        for t in to_cancel:
          t.cancel()
        loop.run_until_complete(
            asyncio.gather(*to_cancel, return_exceptions=True)
        )


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

  Maintains up to `chunk_size` active iterators in parallel, continuously
  refilling worker slots as iterators finish rather than waiting for whole
  chunks to complete.

  Args:
    iterable: AsyncIterators to be merged (i.e. ran in parallel).
    chunk_size: Maximum number of AsyncIterators to run in parallel.
    return_exceptions: If False, any exception will stop the processing and no
      result will be returned. If True, exceptions produced by individual
      iterators will be used as their returned result, so that the processing
      will not be interrupted.

  Yields:
    An AsyncIterator going through the original iterators in parallel and
    wrapping their results into tuples with an index corresponding to which
    iterator produced it.
  """
  it = iter(iterable)
  # Tracks the original index of each iterator to preserve order in results.
  global_index = 0
  # Maps active asyncio.Tasks to their corresponding AsyncIterator and its
  # original index.
  to_process = {}

  def _schedule_next() -> bool:
    """Retrieves the next iterator from the source, schedules its first step, and adds it to the pool."""
    nonlocal global_index
    try:
      next_async_it = aiter(next(it))
      idx = global_index
      global_index += 1
      # Schedule the first step of this new iterator.
      task = asyncio.ensure_future(anext(next_async_it))
      to_process[task] = (next_async_it, idx)
      return True
    except StopIteration:
      return False

  # Fill the pool with up to `chunk_size` active iterators.
  for _ in range(chunk_size):
    if not _schedule_next():
      break

  # Process active iterators until the pool is empty.
  while to_process:
    done, _ = await asyncio.wait(
        list(to_process.keys()),
        return_when=asyncio.FIRST_COMPLETED,
    )
    completed_results = []
    completed_tasks = []
    # Remove completed tasks from the active pool.
    for task in done:
      async_it, idx = to_process.pop(task)
      completed_tasks.append((task, async_it, idx))

    # Sort completed tasks by their original index before processing.
    # This ensures deterministic scheduling order of the next steps.
    completed_tasks.sort(key=lambda x: x[2])

    for task, async_it, idx in completed_tasks:
      try:
        res = task.result()
        completed_results.append((res, idx))
        # The iterator yielded a value; schedule its next step.
        next_task = asyncio.ensure_future(anext(async_it))
        to_process[next_task] = (async_it, idx)
      except Exception as e:  # pylint: disable=broad-exception-caught
        if isinstance(e, StopAsyncIteration):
          # The iterator is exhausted; refill the slot with a new iterator.
          _schedule_next()
        else:
          f = io.StringIO()
          traceback.print_tb(e.__traceback__, file=f)
          e.add_note(f.getvalue())
          if not return_exceptions:
            for t in to_process:
              t.cancel()
            raise e
          else:
            # Treat the exception as a result and refill the slot.
            completed_results.append((e, idx))
            _schedule_next()

    # Sort results by their original index to ensure reproducibility of outputs.
    completed_results.sort(key=lambda x: x[1])
    for res, idx in completed_results:
      yield res, idx


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
        future.index = index  # pyrefly: ignore[missing-attribute]
        to_process[index] = (to_process[index][0], future)  # pyrefly: ignore[unsupported-operation]
    # We execute the anext coroutines and return as soon as one is
    # done.
    done, _ = await asyncio.wait(  # pyrefly: ignore[bad-specialization]
        [f for it, f in to_process if it is not None],
        return_when=asyncio.FIRST_COMPLETED,
    )
    results = []
    for future in done:
      to_process[future.index] = (to_process[future.index][0], None)  # pyrefly: ignore[missing-attribute]
      try:
        result = future.result()  # pyrefly: ignore[missing-attribute]
        results.append((result, future.index))  # pyrefly: ignore[missing-attribute]
      except Exception as e:  # pylint: disable=broad-exception-caught
        # An exception has been raised, it is either a processing error or just
        # the end of this iterator, we remove it from the list.
        to_process[future.index] = (None, None)  # pyrefly: ignore[missing-attribute, unsupported-operation]
        if not isinstance(e, StopAsyncIteration):
          # If the exception is of type StopAsyncIteration, it just means we
          # reached the end of the iterator, so there is nothing to be done and
          # we can continue with the other iterators, but if it is of any other
          # type, we need to handle it.
          # We also add the traceback of the exception to the notes of the
          # exception so that it can be recovered by the caller. Indeed the
          # exception may be copied when passed back into the results, and
          # copying does not retain the __cause__ or __traceback__ information.
          f = io.StringIO()
          traceback.print_tb(e.__traceback__, file=f)
          e.add_note(f.getvalue())
          if not return_exceptions:
            # If return_exceptions is False, we just propagate the exception,
            # which means all results are lost.
            raise e
          else:
            # If return_exceptions is set to True, we simply interrupt the
            # processing of the current iterator and replace its result by the
            # exception it raised.
            results.append((e, future.index))  # pyrefly: ignore[missing-attribute]
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
  except (Exception, KeyboardInterrupt) as e:
    # An exception was raised in the main code (e.g. in the body of the stream
    # loop).
    logging.info('Shutting down queue thread due to exception: %s', e)
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
    yield result  # pyrefly: ignore[invalid-yield]


def _process_queue(result_queue: TQueue[T], job_done: object) -> Iterator[T]:
  """Reads from the queue as an iterator.

  Args:
    result_queue: The queue to read items from.
    job_done: An object indicating the end of the queue.

  Yields:
    Items from the queue.
  """
  while True:
    result = result_queue.get()
    result_queue.task_done()
    if isinstance(result, Exception):
      raise result
    if result is job_done:
      break
    yield result  # pyrefly: ignore[invalid-yield]


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
    result = result_queue.get()
    result_queue.task_done()
    if isinstance(result, Exception):
      raise result
    if result is job_done:
      break
    yield result  # pyrefly: ignore[invalid-yield]
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
