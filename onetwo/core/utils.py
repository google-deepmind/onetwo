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

"""Utility functions for the OneTwo core module."""

import asyncio
import collections
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
import copy
import dataclasses
import functools
import hashlib
import inspect
import io
import threading
import time
from typing import cast, overload, Any, Concatenate, Final, Generic, ParamSpec, TypeAlias, TypeVar

from onetwo.core import content as content_lib
import PIL.Image

_T = TypeVar('_T')
_Args = ParamSpec('_Args')

_ParamType = ParamSpec('_ParamType')
_ReturnType = TypeVar('_ReturnType')

_FunctionToDecorate: TypeAlias = (
    Callable[_ParamType, _ReturnType]
    | Callable[_ParamType, Awaitable[_ReturnType]]
    | Callable[_ParamType, AsyncIterator[_ReturnType]]
)


def is_method(function: Callable[..., Any]) -> bool:
  """Determines whether its argument is a method or a regular function.

  To be used in decorators to determine whether they are applied to a function
  or a method. This determination is static and thus relies on the decorated
  object to have a first argument called `self`.

  Args:
    function: The Callable to be tested.

  Returns:
    True if the argument is a method.
  """
  # Note that we cannot use `inspect.ismethod` since this would return True
  # only when passed a method bound to a class or an instance.
  # So one would have to do `inspect.ismethod(instance.method)` for it to
  # work. But when we apply a decorator to a method the method is really just
  # a function defined within a class.
  # For this to work when multiple decorators are applied sequentially to a
  # method, we have to ensure that the decorators are implemented with something
  # like
  # `def wrapped(self, *args, **kwargs)` instead of the simpler
  # `wrapped(*args, **kwargs)` (which would hide the self argument).
  args = inspect.getfullargspec(function).args
  if not args:
    return False
  return args[0] == 'self'


def decorator_with_optional_args(
    decorator: Callable[
        [Concatenate[_FunctionToDecorate, _Args]], _T
    ]
) -> Callable[_Args, Callable[[_FunctionToDecorate], _T]]:
  """Decorator of decorators to make passing arguments easier.

  Assume you want to write a decorator @my_dec that can take optional arguments
  and you want one to be able to use either of the following syntaxes:
  @my_dec
  @my_dec()
  @my_dec(arg1,...)
  Then this decorator should be written in a way that can either accept a
  function to be decorated as sole argument (first case), or accept arguments
  and return some decorator.
  To simplify this, we provide this meta-decorator.
  So basically you can write the decorator as a function taking in both
  the function to be decorated and the arguments:
  ```
  @decorator_with_optional_args
  def my_dec(function, expected_optional_arguments):
    ...
  ```
  And then my_dec can be used in either of the three ways above.

  Args:
    decorator: Decorator to be meta-decorated. It should be a function that
      takes in a function as first argument.

  Returns:
    New decorator that can be used in the three ways above.
  """
  def wrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> Any:
    # If the only argument is a function to be wrapped, we just wrap it.
    if len(args) == 1 and callable(args[0]):
      return decorator(args[0])
    else:
      def sub_wrapper(function) -> Any:
        return decorator(function, *args, **kwargs)
      return sub_wrapper
  return wrapper


def get_expanded_arguments(
    f: Callable[_Args, Any],
    include_defaults: bool,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> collections.OrderedDict[str, Any]:
  """Expands the arguments to the function.

  Utility function to be used in decorators and in particular when transforming
  the signature of a function for batching.
  This is similar to calling signature.bind(*args, **kwargs), but handles the
  case where the original signature has a **kwargs argument (i.e. a parameter
  of type VAR_KEYWORD) by flattening its contents.

  Args:
    f: A function whose signature is to be populated.
    include_defaults: if True, also includes the parameters that have been set
      implicitly using defaults, otherwise only includes parameters explicitly
      set by the caller.
    args: Positional arguments used in the call to the function.
    kwargs: Keyword arguments used in the call to the function.

  Returns:
    An ordered dict of {argument_name: argument_value} extracted from the
    inputs, with defaults specified. If the function f has some *args parameter,
    the dict will contain {'args': (value1, value2,...)}, and if the function
    has some **kwargs parameter, the dict will contain the individual arguments.
    In order to call the function, one needs to use get_calling_args_and_kwargs,
    which converts the ordered dict into a pair (args, kwargs).
  """
  signature = inspect.signature(f)
  bound_signature = signature.bind_partial(*args)
  if bound_signature is None:
    result = {}
  else:
    if include_defaults:
      bound_signature.apply_defaults()
    result = bound_signature.arguments
  result.update(kwargs)
  for name, param in signature.parameters.items():
    if param.kind == inspect.Parameter.VAR_KEYWORD:
      # We expand the **kwargs argument into its components.
      if name in result:
        value = copy.deepcopy(result[name])
        del result[name]
        result.update(value)
  return result


def get_calling_args_and_kwargs(
    f: Callable[_Args, Any],
    arguments: collections.OrderedDict[str, Any],
) -> _Args:
  """Generates positional/keyword arguments for calling a function.

  When using `get_expanded_arguments` we get an ordered dict of argument
  names and values. If we want to call the function, we need to figure out
  which arguments to pass positionally and which to pass by keyword.
  This is what this function provides.

  Args:
    f: A function whose signature is to be populated.
    arguments: An ordered dict produced by `get_expanded_arguments`.

  Returns:
    A pair whose first item is the tuple of positional arguments, and the
    second item is the dict of keyword-based arguments.
    This enables to call the function as follows:
    ```
    args, kwargs = get_calling_args_and_kwargs(f, arguments)
    result = f(*args, **kwargs)
    ```
  """
  signature = inspect.signature(f)
  args = ()
  kwargs = dict(copy.copy(arguments))
  for name, param in signature.parameters.items():
    if (
        param.kind == inspect.Parameter.POSITIONAL_ONLY
        or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ):
      if name in kwargs:
        # Positional arguments are moved from the kwargs dict to the args list.
        args += (kwargs[name],)
        del kwargs[name]
    elif param.kind == inspect.Parameter.VAR_POSITIONAL:
      if name in kwargs:
        # Var positional arguments are removed from the kwargs dict and
        # appended to the args list.
        args += kwargs[name]
        del kwargs[name]
  return args, kwargs


@dataclasses.dataclass(frozen=True)
class FromInstance(Generic[_T]):
  """Wrapper to indicate that a decorator parameter should be read at runtime.

    There are two ways to specify how to determine the value of the parameter
    from a given instance -- via either `name` or `function`. Only one of these
    should be specified.

    Attributes:
      name: A string representing a name of an attribute or method that is to be
        passed as the argument to the decorator.
      function: A function that will be applied to the object instance in order
        to determine the value that is to be passed as the argument to the
        decorator.
  """
  name: str | None = None
  function: Callable[[object], _T] | None = None

  def __call__(self, instance: object) -> _T:
    """Returns the value of the parameter based on the given instance."""
    if (self.name is not None) == (self.function is not None):
      raise ValueError('Only one of `name` or `function` should be specified.')
    if self.function is not None:
      return self.function(instance)
    else:
      return getattr(instance, self.name)


FROM_INSTANCE_CLASS_NAME: Final[FromInstance[str]] = FromInstance(
    function=lambda x: x.__class__.__name__
)


@dataclasses.dataclass(frozen=True)
class RuntimeParameter(Generic[_T]):
  """Convenience wrapper to read an argument value for a decorator."""

  parameter: _T | FromInstance[_T]
  instance: object | None = None

  def value(self) -> _T:
    if isinstance(self.parameter, FromInstance) and self.instance is not None:
      return self.parameter(self.instance)
    else:
      return self.parameter  # pytype: disable=bad-return-type


def rate_limit_function(
    qps: float | None,
) -> Callable[[Callable[_Args, _T]], Callable[_Args, _T]]:
  """Decorator that limits the frequency at which the function can be called.

  This is thread-safe in the sense that if the wrapped function is called from
  multiple threads, the qps limit is still enforced.

  Args:
    qps: Maximum number of calls per second as a float. If None, the decorator
      is a no-op.

  Returns:
    When applied to a function, it returns a function with the same signature
    which blocks for a certain time to guarantee the qps limit.
    When applied to a coroutine function, it returns a coroutine function with
    the same signature which yields control to the event loop until enough time
    has passed to guarantee the qps limit.
  """
  if qps is None:
    return lambda x: x

  lock = threading.Lock()
  interval = 1.0 / qps

  def decorate(function: Callable[_Args, _T]) -> Callable[_Args, _T]:
    last_call = time.perf_counter()

    @functools.wraps(function)
    def wrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> _T:
      nonlocal last_call
      lock.acquire()
      current = time.perf_counter()
      try:
        if current < last_call + interval:
          time.sleep(interval - current + last_call)
        return function(*args, **kwargs)
      finally:
        last_call = time.perf_counter()
        lock.release()

    @functools.wraps(function)
    async def awrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> _T:
      nonlocal last_call
      while not lock.acquire(blocking=False):
        await asyncio.sleep(0)
      current = time.perf_counter()
      try:
        if current < last_call + interval:
          await asyncio.sleep(interval - current + last_call)
        return await function(*args, **kwargs)
      finally:
        last_call = time.perf_counter()
        lock.release()

    if inspect.iscoroutinefunction(function):
      return awrapper
    else:
      return wrapper
  return decorate


def rate_limit_method(
    qps: float | None | FromInstance[float | None],
) -> Callable[[Callable[_Args, _T]], Callable[_Args, _T]]:
  """Decorator that limits the frequency at which the method can be called.

  This is thread-safe in the sense that if the wrapped method is called from
  multiple threads, the qps limit is still enforced.
  The qps limiting happens separately for each instance of the object on which
  this method is defined.

  Args:
    qps: Maximum number of calls per second as a float. If None, the decorator
      is a no-op.

  Returns:
    When applied to a method, it returns a method with the same signature
    which blocks for a certain time to guarantee the qps limit.
    When applied to a coroutine method, it returns a coroutine method with
    the same signature which yields control to the event loop until enough time
    has passed to guarantee the qps limit.
  """
  if qps is None:
    return lambda x: x

  class_lock = threading.Lock()

  def decorate(method: Callable[_Args, _T]) -> Callable[_Args, _T]:
    if not is_method(method):
      raise ValueError('rate_limit_method can decorate only methods.')
    def get_params(self, qps_value):
      # We retrieve the instance information from self, and create it if
      # this is the first call.
      class_lock.acquire()
      instance_lock_name = f'_lock_{method.__name__}'
      last_call_name = f'_last_call_{method.__name__}'
      interval_name = f'_interval_{method.__name__}'
      if not hasattr(self, instance_lock_name):
        # This is the first call, the instance doesn't have any attributes
        # for this method, we create them.
        setattr(self, last_call_name, time.perf_counter())
        setattr(self, interval_name, 1.0 / qps_value)
        setattr(self, instance_lock_name, threading.Lock())
      class_lock.release()

      instance_lock = getattr(self, instance_lock_name)
      interval = getattr(self, interval_name)
      return instance_lock, interval

    def get_last_call(self):
      last_call_name = f'_last_call_{method.__name__}'
      return getattr(self, last_call_name)

    def set_last_call(self):
      last_call_name = f'_last_call_{method.__name__}'
      setattr(self, last_call_name, time.perf_counter())

    @functools.wraps(method)
    def wrapper(self, *args: _Args.args, **kwargs: _Args.kwargs) -> _T:
      qps_value = RuntimeParameter[float | None](qps, self).value()
      if qps_value is None:
        # Here and below we use `method(*((self,) + args), **kwargs)` instead of
        # `method(self, *args, **kwargs)` because of python Lint error. The two
        # are equivalent.
        return method(*((self,) + args), **kwargs)
      instance_lock, interval = get_params(self, qps_value)
      instance_lock.acquire()
      last_call = get_last_call(self)
      current = time.perf_counter()
      try:
        if current < last_call + interval:
          time.sleep(interval - current + last_call)
        return method(*((self,) + args), **kwargs)
      finally:
        set_last_call(self)
        instance_lock.release()

    @functools.wraps(method)
    async def awrapper(self, *args: _Args.args, **kwargs: _Args.kwargs) -> _T:
      qps_value = RuntimeParameter[float | None](qps, self).value()
      if qps_value is None:
        return await method(*((self,) + args), **kwargs)
      instance_lock, interval = get_params(self, qps_value)
      while not instance_lock.acquire(blocking=False):
        await asyncio.sleep(0)
      last_call = get_last_call(self)
      current = time.perf_counter()
      try:
        if current < last_call + interval:
          await asyncio.sleep(interval - current + last_call)
        return await method(*((self,) + args), **kwargs)
      finally:
        set_last_call(self)
        instance_lock.release()

    if inspect.iscoroutinefunction(method):
      return awrapper
    else:
      return wrapper
  return decorate


@overload
def returning_raised_exception(
    function: Callable[_Args, _ReturnType],
) -> Callable[_Args, _ReturnType | Exception]:
  ...


@overload
def returning_raised_exception(
    function: Callable[_Args, Awaitable[_ReturnType]],
) -> Callable[_Args, Awaitable[_ReturnType | Exception]]:
  ...


def returning_raised_exception(function):
  """Decorator to make a function return an exception instead of raising it."""
  @functools.wraps(function)
  def wrapper(*args, **kwargs):
    try:
      return function(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return e

  @functools.wraps(function)
  async def awrapper(*args, **kwargs):
    try:
      return await function(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return e

  @functools.wraps(function)
  def wrapper_m(self, *args, **kwargs):
    try:
      return function(self, *args, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return e

  @functools.wraps(function)
  async def awrapper_m(self, *args, **kwargs):
    try:
      return await function(self, *args, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return e

  if inspect.iscoroutinefunction(function):
    if is_method(function):
      return awrapper_m
    else:
      return awrapper
  else:
    if is_method(function):
      return wrapper_m
    else:
      return wrapper


@overload
def raising_returned_exception(
    function: Callable[_Args, _ReturnType | Exception],
) -> Callable[_Args, _ReturnType]:
  ...


@overload
def raising_returned_exception(
    function: Callable[_Args, Awaitable[_ReturnType | Exception]],
) -> Callable[_Args, Awaitable[_ReturnType]]:
  ...


def raising_returned_exception(function):
  """Decorator to make a function raise an exception instead of returning it."""
  @functools.wraps(function)
  def wrapper(*args, **kwargs):
    result = function(*args, **kwargs)
    if isinstance(result, Exception):
      raise result
    else:
      return result

  @functools.wraps(function)
  async def awrapper(*args, **kwargs):
    # For some reason, pytype seems to get confused here and thinks we are
    # trying to call `__await__` on an `Exception`.
    result = await function(*args, **kwargs)  # pytype: disable=bad-return-type
    if isinstance(result, Exception):
      raise result
    else:
      return result

  @functools.wraps(function)
  def wrapper_m(self, *args, **kwargs):
    result = function(self, *args, **kwargs)
    if isinstance(result, Exception):
      raise result
    else:
      return result

  @functools.wraps(function)
  async def awrapper_m(self, *args, **kwargs):
    # For some reason, pytype seems to get confused here and thinks we are
    # trying to call `__await__` on an `Exception`.
    result = await function(self, *args, **kwargs)  # pytype: disable=bad-return-type
    if isinstance(result, Exception):
      raise result
    else:
      return result

  if inspect.iscoroutinefunction(function):
    if is_method(function):
      return awrapper_m
    else:
      return awrapper
  else:
    if is_method(function):
      return wrapper_m
    else:
      return wrapper


def _get_bytes_for_hashing(key: Any) -> bytes:
  """Best-effort conversion of key to bytes for further hashing."""
  match key:
    # The `hash` function for python `str` and `bytes` by default (starting
    # from python 3.3) adds a random seed to the hash. This means that between
    # two runs `hash(obj)` is not deterministic. While for many applications
    # such a behaviour would be ok, it clearly does not work well with caching,
    # where we want same deterministic hashes between the runs.
    case str():
      bytes_value = key.encode('utf-8')
    case bytes():
      bytes_value = key
    case list() | tuple():
      bytes_value = str(key).encode('utf-8')
    case dict():
      bytes_value = str(sorted(key.items())).encode('utf-8')
    case set():
      bytes_value = str(sorted(list(key))).encode('utf-8')
    case PIL.Image.Image():
      # Extract bytes from the PIL Image object and hash it.
      bytes_io = io.BytesIO()
      cast(PIL.Image.Image, key).save(bytes_io, 'JPEG')
      bytes_value = bytes_io.getvalue()
    case content_lib.Chunk():
      bytes_value = _get_bytes_for_hashing(key.content)
    case content_lib.ChunkList():
      bytes_value = b''.join(
          [_get_bytes_for_hashing(chunk) for chunk in key.chunks]
      )
    # case Hashable():
    #   Let us never add this case! This means we want to rely on `__hash__`
    #   method of the type, but as pointed out above doing so is often
    #   dangerous.
    case _:
      if hasattr(key, 'tobytes'):  # Type `str` has no such attribute.
        # This handles the case of a np.ndarray.
        bytes_value = key.tobytes()
      else:
        raise ValueError(f'Unsupported key type: {type(key)}')
  return bytes_value


def get_str_hash(key: Any) -> str:
  """Best-effort hashing of various kinds of objects.

  This function is used mainly by `core/caching.py` when computing the hash keys
  of the function calls.

  Args:
    key: Any object that we want to hash.

  Returns:
    Unique string valued hash of the object. Main goal in the context of
    `caching.py` is that identical calls to the cached functions end up being
    properly recognized.
  """
  bytes_value = _get_bytes_for_hashing(key)
  return hashlib.sha224(bytes_value).hexdigest()
