# Copyright 2025 DeepMind Technologies Limited.
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

"""Utility functions for adding caching capabilities to methods."""

from __future__ import annotations

import abc
import asyncio
import collections
from collections.abc import ByteString, Callable, Coroutine, Mapping, Sequence
import contextvars
import copy
import dataclasses
import functools
import inspect
import json
import logging
import os
import threading
import traceback
from typing import Any, Final, Generic, ParamSpec, TypeVar

import dataclasses_json
from onetwo.core import constants
from onetwo.core import executing
from onetwo.core import utils





_DEFAULT_SAMPLING_KEY: Final[str] = ''

# TODO: Move sampling context and SamplingKeyUpdater to
# sampling.py.

# Context variable used to tag different samples from sampled functions.
# This key is common to all functions in the context, meaning that if we call
# multiple distinct sampled functions, the replies will be tagged with the same
# key for consistency (but the mapping from the keys to the index in the cache
# may be different for each cache instance since not all functions are called
# at the same frequency).
# The cache contains for each key a list of samples, and maintains a mapping
# from sampling_key to the index in the list.
context_sampling_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    'context_sampling_key', default=_DEFAULT_SAMPLING_KEY
)


# Type of the value that is cached, e.g. str.
CachedType = TypeVar('CachedType')

_Args = ParamSpec('_Args')


def get_key_for_logging(key: str, sampling_key: str | None = None) -> str:
  """Produces key and sampling_key information for display in the logs."""
  rkey = repr(key)
  if len(rkey) > 50:
    rkey = rkey[:20] + f'[{len(rkey)-40} chars]' + rkey[-20:]
  return (
      f'{rkey} (hash: {utils.get_str_hash(key)} - '
      f'sampling key:{repr(sampling_key)})'
  )


def _hint_tuple_encoder(arg: Any) -> Any:
  """Encoder that helps (best effort) to preserve python tuples.

  Python tuples are converted to lists in json. This encoder stores tuples in
  such a way that later they can be properly decoded into tuples. Inspired by
  https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json.

  Args:
    arg: Any JSON-compatible object (i.e., on which `json.dumps` can be called)
      that needs to be encoded.

  Returns:
    Encoded object where every tuple `obj` is replaced with a dictionary
    `{'__tuple__': True, 'items': obj}`.
  """
  if isinstance(arg, tuple):
    return {
        '__tuple__': True,
        'items': [_hint_tuple_encoder(value) for value in arg],
    }
  if isinstance(arg, list):
    return [_hint_tuple_encoder(value) for value in arg]
  if isinstance(arg, dict):
    return {key: _hint_tuple_encoder(value) for key, value in arg.items()}
  return arg


def _hint_tuple_decoder(arg: Any) -> Any:
  """Decoder that helps (best effort) to preserve python tuples."""
  if isinstance(arg, dict):
    if '__tuple__' in arg:
      return tuple(_hint_tuple_decoder(value) for value in arg['items'])
    return {key: _hint_tuple_decoder(value) for key, value in arg.items()}
  if isinstance(arg, list):
    return [_hint_tuple_decoder(value) for value in arg]
  return arg


def nested_defaultdict_decoder(
    arg: collections.defaultdict,
) -> collections.defaultdict:
  """Decodes a defaultdict of int to defaultdict of str to int."""
  decoded = dict(arg)
  return collections.defaultdict(
      functools.partial(collections.defaultdict, int),
      {k: collections.defaultdict(int, v) for k, v in decoded.items()},
  )


def defaultdict_decoder(
    arg: collections.defaultdict,
) -> collections.defaultdict:
  """Decodes a defaultdict of int to int."""
  decoded = dict(arg)
  return collections.defaultdict(
      int,
      {k: v for k, v in decoded.items()},
  )


def nested_defaultdict_initializer(
    unused_arg: collections.defaultdict | None = None,
) -> collections.defaultdict:
  """Returns an empty defaultdict of defaultdict of int."""
  return collections.defaultdict(
      functools.partial(collections.defaultdict, int)
  )


class SimpleCache(Generic[CachedType], metaclass=abc.ABCMeta):
  """Interface for a generic cache.

  A Mapping from key to a Sequence of values of type CachedType. The actual
  value that is returned depends on the sampling_key (via a specific mechanism
  that maps sampling_keys to indices in the Sequence, but that is up to the
  implementation of the two-level cache). We map from (key, sampling_key) to
  values by first mapping key to Sequence of values and then choosing a specific
  value in that Sequence using sampling_key. The `cache_value` method adds to
  the cache (or updates a cache entry), and the `get_cached_value` method
  retrieves from the cache, returning None if no entry has been found.

  There may be three scenarios when looking up a cached value for
  a given (key, sampling_key) pair:
  1. The pair (key, sampling_key) is flagged as "in progress". This can happen
    when the method is batched, another coroutine is already processing the same
    request but did not return the result yet. We wait for the result (without
    blocking) and then return the value.
  2. Key is not found: we need to actually run the method;
  3. Key is found, there is no sample assigned to sampling_key, and there are no
    free samples available: we need to actually run the method;
  4. Key is found and either the value is already assigned to sampling_key or
    there is a free unused sample: we return the value.

  In order to implement the non-blocking behaviour in 1. we need
  get_cached_value to be async, i.e. it returns an awaitable value.

  Implementations of SimpleCache may also declare a `calls_in_progress`
  attribute of type `set`. If present, the cache_method decorator will add keys
  and sampling keys into this set before processing the corresponding calls and
  remove them once the results are obtained. See SimpleFunctionCache.
  """

  @abc.abstractmethod
  def cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: CachedType,
  ) -> None:
    """Stores the keys-value pair in the cache."""

  @abc.abstractmethod
  async def get_cached_value(
      self,
      key: str,
      sampling_key: str | None,
  ) -> CachedType | None:
    """Retrieves a value from the cache if it exists."""

  @abc.abstractmethod
  def get_key_count(self) -> int:
    """Returns the number of keys in the cache (a measure of cache size)."""


@dataclasses.dataclass
class SimpleFileCache(
    Generic[CachedType], SimpleCache[CachedType], metaclass=abc.ABCMeta
):
  """Interface for a generic file cache."""

  cache_filename: str | None = None

  @abc.abstractmethod
  def load(
      self,
      *,
      restore_mapping: bool = False,
      overwrite: bool = True,
      cache_filename: str | None = None,
  ):
    """Loads the cache from file.

    Args:
      restore_mapping: If True, will try and restore the mapping between
        sampling_key and sample_ids from disk, otherwise creates a new one. Only
        relevant if `overwrite=True` (otherwise nevers restores mapping).
      overwrite: If True, then any existing cache contents will be discarded and
        completely overwritten by the contents of the cache file. If False, then
        the existing cache contents will be preserved and the contents of the
        cache file will be merged into it.
      cache_filename: If specified, then will load from the given file path;
        otherwise, by default will load from `self.cache_filename`.
    """

  @abc.abstractmethod
  def save(
      self,
      *,
      overwrite: bool = False,
      cache_filename: str | None = None,
  ) -> None:
    """Saves the cache to file.

    Args:
      overwrite: If False (default) right before saving cache in the file we
        check if file with that name already exists and raise error if it does.
        If True we overwrite.
      cache_filename: If specified, then will save to the given file path;
        otherwise, by default will save to `self.cache_filename`.
    """


@dataclasses.dataclass
class CacheEnabled(Generic[CachedType]):
  """Interface for a class that has a cache attached to it.

  Methods of (any implementation of) this class can be decorated with
  `cache_method` to enable caching. There can be multiple decorated methods, in
  which case a single cache will handle all of them.

  Attributes:
    disable_caching: Whether caching is enabled for this object.
    raise_exceptions_as_is: Whether the exceptions raised by the cached methods
      using this object are raised as is. The default behaviour is to convert
      the exceptions into 'ValueError's before being raised again.
    cache: Cache handler to use for caching.
  """

  disable_caching: bool = False
  raise_exceptions_as_is: bool = False
  cache: SimpleCache[CachedType] | None = dataclasses.field(
      default=None,
  )

  # TODO: Remove the setters and getters once we have migrated all
  # usages to the new cache field.
  @property
  def _cache_handler(self) -> SimpleCache[CachedType]:
    cache = self.cache
    if cache is None:
      raise ValueError('Cache handler is not initialized.')
    else:
      return cache

  @_cache_handler.setter
  def _cache_handler(self, value: SimpleCache[CachedType]):
    self.cache = value


@dataclasses.dataclass
class FileCacheEnabled(Generic[CachedType], CacheEnabled[CachedType]):
  """Interface for a class that has a cache attached to it.

  Methods of (any implementation of) this class can be decorated with
  `cache_method` to enable caching. There can be multiple decorated methods, in
  which case a single cache will handle all of them.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored.
  """

  cache_filename: str | None = None
  # Override the type of cache handler.
  cache: SimpleFileCache[CachedType] | None = dataclasses.field(
      default=None,
  )

  # TODO: Remove the setters and getters once we have migrated all
  # usages to the new cache field.
  @property
  def _cache_handler(self) -> SimpleFileCache[CachedType]:
    cache = self.cache
    if cache is None:
      raise ValueError('Cache handler is not initialized.')
    else:
      return cache

  @_cache_handler.setter
  def _cache_handler(self, value: SimpleFileCache[CachedType]):
    self.cache = value

  def load_cache(
      self,
      *,
      restore_mapping: bool = False,
      overwrite: bool = True,
      cache_filename: str | None = None,
  ):
    """Loads the cache from file.

    Args:
      restore_mapping: If True, will try and restore the mapping between
        sampling_key and sample_ids from disk, otherwise creates a new one. Only
        relevant if `overwrite=True` (otherwise nevers restores mapping).
      overwrite: If True, then any existing cache contents will be discarded and
        completely overwritten by the contents of the cache file. If False, then
        the existing cache contents will be preserved and the contents of the
        cache file will be merged into it.
      cache_filename: If specified, then will load from the given file path;
        otherwise, by default will load from `self.cache_filename`.
    """
    if self.cache is None:
      raise ValueError('Cache handler is not initialized.')
    self.cache.load(
        restore_mapping=restore_mapping,
        overwrite=overwrite,
        cache_filename=cache_filename,
    )  # pytype: disable=attribute-error

  def save_cache(
      self,
      *,
      overwrite: bool = False,
      cache_filename: str | None = None,
  ):
    """Save the cache to file.

    Args:
      overwrite: If False (default) right before saving cache in the file we
        check if file with that name already exists and raise error if it does.
        If True we overwrite.
      cache_filename: If specified, then will save to the given file path;
        otherwise, by default will save to the same file path from which the
        cache was originally loaded.
    """
    if self.cache is None:
      raise ValueError('Cache handler is not initialized.')
    # Hint for the type checker.
    assert self.cache is not None
    self.cache.save(overwrite=overwrite, cache_filename=cache_filename)  # pytype: disable=attribute-error


def _create_cache_key(
    name: str,
    arguments: Mapping[str, Any],
    hashed: Sequence[str] | None = None,
) -> str:
  """Creates a cache key for arguments to a function.

  Args:
    name: Name of the function being called (i.e. destination).
    arguments: Arguments that the function has been called with.
    hashed: Sequence of arguments that should be hashed because they tend to be
      too big to be part of the key.

  Returns:
    Key to lookup in the cache.
  """

  def maybe_hash(name: str, value: Any) -> Any:
    if hashed is not None and name in hashed:
      return utils.get_str_hash(value)
    return value

  arg_value_by_name = {
      name: maybe_hash(name, value) for name, value in arguments.items()
  }
  # Add function name to the dictionary.
  arg_value_by_name[constants.CACHING_FUNCTION_NAME_KEY] = name
  # Serialize it.
  return json.dumps(arg_value_by_name, sort_keys=True, default=repr)


@dataclasses.dataclass
class CacheKeyMaker(Generic[CachedType]):
  """Class that converts inputs of a method into cache key.

  Attributes:
    hashed: A Sequence of strings representing the names of parameters that
      should be converted to bytes and hashed when computing the cache key. For
      example this would be needed if the parameter is an np.ndarray or some
      complex data type like this, to avoid the cache key to be huge.
    dropped: A Sequence of strings corresponding to the names of parameters that
      should not be used as part of the cache key.
    is_initialized: Whether method initialize was already called.
    _method: The method whose inputs have to be converted (set automatically by
      the cache_method decorator).
    _name: The name of the method (for the cache -- set automatically by the
      cache_method decorator).
  """

  # Constructor parameters.
  hashed: Sequence[str] | None = None
  dropped: Sequence[str] | None = None
  # Parameters set via the initialize method (will be set by the cache_method
  # decorator).
  _method: Callable[..., CachedType] | None = dataclasses.field(
      default=None, init=False
  )
  _name: str | None = dataclasses.field(default=None, init=False)

  @property
  def is_initialized(self) -> bool:
    if self._name is None:
      return False
    else:
      return True

  def initialize(
      self,
      method: Callable[..., CachedType],
      name: str,
  ):
    """Initialization at decoration time.

    Args:
      method: Method of class CacheEnabled that we are decorating with cache.
      name: Arbitrary name specified by the user when decorating the method with
        cache. Will be used as part of the cache key. it plays an important role
        in distinguishing between different methods decorated with caching, in
        case there are more than one.
    """
    self._method = method
    self._name = name

  def create_key(
      self,
      obj_with_cache: CacheEnabled[CachedType],
      args: tuple[Any, ...],
      kwargs: Mapping[str, Any],
  ) -> str:
    """Create cache key based on the method and arguments.

    Args:
      obj_with_cache: The instance on which the method is called.
      args: Positional arguments with which the method is called (not including
        `self`).
      kwargs: Keyword arguments with which the method is called.

    Returns:
      Cache key to lookup.
    """
    arguments = utils.get_expanded_arguments(
        self._method, True, (obj_with_cache,) + args, kwargs
    )
    arguments.pop('self', None)
    if self.dropped is not None:
      for name in self.dropped:
        # Remove arguments with that name from the main arguments.
        arguments.pop(name, None)
    key = _create_cache_key(self._name, arguments, self.hashed)
    return key


ReturnType = TypeVar('ReturnType')


def return_first_and_cache_remaining(
    values: Sequence[CachedType],
    disable_caching: bool,
    cache_value_callback: Callable[[CachedType], None],
) -> CachedType:
  """Returns the first value and possibly caches the remaining ones.

  We make sure that values is indeed a Sequence (like list or tuple). Note that
  str and ByteString are instances of Sequence in python, so we check for this
  as well.

  Args:
    values: Sequence of values.
    disable_caching: If True just return the first value and don't cache. If
      False (default) we cache.
    cache_value_callback: Callback that takes a single argument and caches it.

  Returns:
    First value in values Sequence.
  """
  if (
      isinstance(values, str)  # values = 'abcd' won't do.
      or isinstance(values, ByteString)  # values = b'abcd' won't do.
      or not isinstance(values, Sequence)
  ):
    raise ValueError(
        'Method that is decorated with cache_method(cache_extra_replies'
        '=True) needs to return a Sequence of values. The first value '
        f'will be returned and the rest will be cached. Returned {values}'
    )
  if not values:
    raise ValueError(
        'Method decorated with cache_method(cache_extra_replies=True) '
        'returned an empty sequence.'
    )
  if disable_caching:
    return values[0]
  if isinstance(values[0], executing.ExecutableWithCallback):
    # TODO: Handle this case.
    logging.warning(
        'Received replies of type %s. Caching extra replies of type '
        'executing.ExecutableWithCallback not yet supported. We discard '
        'them for now.',
        type(values[0]),
    )
  else:
    for value in values[1:]:
      cache_value_callback(value)
  return values[0]


def _get_cache(
    obj_with_cache: CacheEnabled[CachedType],
) -> SimpleCache[CachedType]:
  """Gets the cache handler of the given object.

  Checks that the class of the method supports caching.
  It would be best to do this at decoration time, but the method
  decorators are executed before the class is created.

  Args:
    obj_with_cache: Object of class CacheEnabled. We are decorating its method.

  Returns:
    The cache handler of the given object.
  """
  if not isinstance(obj_with_cache, CacheEnabled):
    raise ValueError(
        "Decorator @cache_method is applied to a method whose class doesn't"
        ' inherit from CacheEnabled.'
    )
  if getattr(obj_with_cache, 'cache', None) is None:
    raise ValueError(
        "Decorator @cache_method is applied to a method whose class doesn't"
        ' have a cache handler.'
    )
  cache = obj_with_cache.cache
  if cache is None:
    raise ValueError(
        "Decorator @cache_method is applied to a method whose class doesn't"
        ' have a cache handler.'
    )
  return cache


# Unfortunately, pytype does not properly support decorators that change the
# signature: (internal link). Resorting to `...`.
def cache_method(
    name: str | None = None,
    is_sampled: bool = False,
    cache_key_maker: CacheKeyMaker | Callable[[], CacheKeyMaker] | None = None,
    cache_extra_replies: bool | utils.FromInstance[bool] = False,
) -> Callable[
    [
        # Before caching.
        Callable[..., ReturnType | Coroutine[None, None, ReturnType]]
    ],
    Callable[..., Coroutine[None, None, ReturnType]],  # After caching.
]:
  """Decorator that adds caching to a method.

  This decorator is very flexible in terms of how to obtain a cache key from the
  method and its arguments.

  It should only be used to decorate methods (not functions), but can decorate
  both sync and async methods. A decorated method is async (i.e. the value it
  returns is awaitable). The class of the method that is decorated should
  inherit from CacheEnabled, i.e. it should have a cache_handler attached to it
  (to actually implement the cache).

  Args:
    name: Name to be used as part of the cache key to identify this method. In
      case when a class has more than one decorated method, it is important that
      they are decorated with different names in order to avoid caching
      conflicts.
    is_sampled: True if the result of the function is a sample (i.e. calls with
      the same arguments may return different results), in which case a
      sampling_key is passed as an extra parameter (the decorated function needs
      to have a sampling_key parameter).
    cache_key_maker: An optional CacheKeyMaker which will take care of
      converting the inputs to the method into cache keys of the appropriate
      format. If None, we will instantiate an object of type CacheKeyMaker. This
      can also be provided as a Callable[[None], CacheKeyMaker] to be
      instantiated at decoration time.
    cache_extra_replies: The cached method may return more than one reply when
      in fact asked for only one. For example, some LLMs may generate 5 replies
      at once when running a single `generate` request. In these cases
      discarding the unused replies seems wasteful. Instead, we can cache these
      values for later use. When False (default), we return replies as is. If
      True it is assumed that the method returns a Sequence of values (remember:
      be careful with string and bytestring return types as they *are*
      considered Sequences in Python); we cache all the values and return only
      the first one.

  Returns:
    A decorated method that:
    (0) Returns an awaitable.
    (1) Looks up the key (computed from the input arguments) in the cache.
    (2) Executes the method if necessary.
    (3) Caches the result of the call if an execution took place.
    (4) Optionally, if the method returns more replies than needed, only returns
      the first reply and caches the remaining ones for future use.
  """

  def method_wrapper(
      method: Callable[..., ReturnType | Coroutine[None, None, ReturnType]],
  ) -> Callable[..., ReturnType | Coroutine[None, None, ReturnType]]:
    """Actual decorator replacing the method with a version that uses cache."""
    nonlocal name
    # name will be used to separate between different cached methods. Make sure
    # it is not empty.
    if name is None:
      name = method.__name__  # pytype: disable=attribute-error
    # Check that the wrapped object is a method.
    if not utils.is_method(method):
      raise ValueError(
          'Decorator @cache_method should be applied to a method, was applied'
          f' to {method}.'
      )
    # Determine which object to use to make cache keys.
    if cache_key_maker is not None:
      if isinstance(cache_key_maker, CacheKeyMaker):
        if cache_key_maker.is_initialized:
          raise ValueError(
              'Cannot use the same CacheKeyMaker instance to decorate different'
              ' methods.'
          )
        else:
          maker = cache_key_maker
      else:
        maker = cache_key_maker()
    else:
      maker = CacheKeyMaker()
    # Set the parameters of the cache key maker object.
    maker.initialize(method, name)

    async def lookup(
        obj_with_cache: CacheEnabled[CachedType],
        maker: CacheKeyMaker,
        args: Any,
        kwargs: Any,
    ) -> tuple[CachedType, tuple[str, str | None]]:
      """Performs the lookup in the cache.

      Args:
        obj_with_cache: Object of class CacheEnabled. We are decorating its
          method.
        maker: Object used to make the cache keys.
        args: Args with which the decorated method is called (without `self`).
        kwargs: Kwargs with which the decorated method is called.

      Returns:
        The value found in cache if it exists or None, along with the cache key
        computed from the input arguments of the method and the sampling key.

      Raises:
        ValueError if the object on which the method is called does not inherit
        from CacheEnabled.
      """
      if is_sampled:
        # Get sampling key from context.
        sampling_key = context_sampling_key.get()
      else:
        sampling_key = None
      key = maker.create_key(obj_with_cache, args, kwargs)
      cache_handler = _get_cache(obj_with_cache)
      if obj_with_cache.disable_caching:
        value = None
      else:
        value = await cache_handler.get_cached_value(
            key=key,
            sampling_key=sampling_key,
        )
      return value, (key, sampling_key)

    def store(
        obj_with_cache: CacheEnabled[CachedType],
        value: CachedType,
        key: str,
        sampling_key: str | None,
    ) -> CachedType:
      """Stores the returned value or registers a callback to do so.

      Args:
        obj_with_cache: Object of class CacheEnabled. We are decorating its
          method.
        value: Return value of the method to be stored in the cache. If this is
          an Executable, instead of storing it directly, we attach to it a
          postprocessing callback to store it after the end of the iterations
          and return it.
        key: Cache key.
        sampling_key: Sampling key.

      Returns:
        The return value of the method. If it is an Executable we wrap it
        into an ExecutableWithPostprocessing which will call
        `cache_handler.cache_value()` after execution.
      """
      if obj_with_cache.disable_caching:
        return value

      # If the value is iterable, it will be cached only after completion.
      # We pass a callback that will take care of caching the final state
      # of the value once iterated through.
      if isinstance(value, executing.Executable):

        def callback(v: CachedType) -> CachedType:
          _get_cache(obj_with_cache).cache_value(
              key=key,
              sampling_key=sampling_key,
              value=v,
          )
          return v

        value = executing.ExecutableWithPostprocessing(
            wrapped=value, postprocessing_callback=callback
        )
      else:
        _get_cache(obj_with_cache).cache_value(
            key=key,
            sampling_key=sampling_key,
            value=value,
        )
      return value

    @functools.wraps(method)
    async def ainner(self, *args, **kwargs) -> ReturnType:
      """Async method decorated with cache."""
      # We access a protected member of `self`.
      # pylint: disable=protected-access

      # Here "self" is the object of class CacheEnabled. We are decorating one
      # of its methods.
      value, (key, sampling_key) = await lookup(self, maker, args, kwargs)
      if value is not None:
        return value
      # Actually process the call. But first indicate that we are processing it.
      if hasattr(self._cache_handler, '_calls_in_progress'):
        self._cache_handler._calls_in_progress |= {(key, sampling_key)}
      # Call/await depending on whether decorating a sync or async method.
      try:
        if inspect.iscoroutinefunction(method):
          # pytype expects *args and **kwargs.
          value = await method(*((self,) + args), **kwargs)
        else:
          value = method(*((self,) + args), **kwargs)
      except Exception as err:  # pylint: disable=broad-except
        error_message = f'Error raised while executing method {method}:\n{err}'
        if self.raise_exceptions_as_is:
          logging.error(error_message)
          raise err
        raise ValueError(f'{error_message}\n') from err
      except KeyboardInterrupt as err:
        # Note that KeyboardInterrupt is not a subclass of Exception, so we need
        # to handle it separately.
        raise err
      else:
        # We may need to resolve cache_extra_replies's value at runtime.
        do_cache_extra = utils.RuntimeParameter[bool](
            cache_extra_replies, self
        ).value()
        if do_cache_extra:
          # Method returns a Sequence of CachedType elements. Handle this case.
          logging.info('Caching extra replies for method %s.', method.__name__)  # pytype: disable=attribute-error
          value = return_first_and_cache_remaining(
              values=value,
              disable_caching=self.disable_caching,
              cache_value_callback=functools.partial(
                  self._cache_handler.cache_value,  # pytype: disable=attribute-error
                  key,
                  None,  # No sampling_key set.
              ),
          )
        result = store(self, value, key, sampling_key)
        return result
      finally:
        # Finally remove the call from `_calls_in_progress` to unblock other
        # coroutines waiting for the value. Note that we do this both in the
        # success case (after the call has been fully processed and the value
        # cached) and the failure case (to avoid leaving orphaned calls in
        # `_calls_in_progress` if execution is unexpectedly interrupted).
        if hasattr(self._cache_handler, '_calls_in_progress'):
          self._cache_handler._calls_in_progress -= {(key, sampling_key)}

    return ainner
    # pylint: enable=protected-access

  return method_wrapper


class SamplingKeyUpdater(executing.Executable):
  """Executable to set the sampling key of requests.

  This works by wrapping an executable, and updating the sampling_key context
  variable right before executing the wrapped executable, and restoring it
  afterwards. Anything triggered by the wrapped executable will inherit the
  value of this variable.

  Attributes:
    postfix: The postfix to append to the current value of the sampling_key
      variable.
    wrapped: Executable to be wrapped.
    base_key: Value of the sampling_key when the object was created.
  """

  def __init__(self, postfix: str, wrapped: executing.Executable):
    self.postfix = postfix
    self.wrapped = wrapped

  def _update_sampling_key(self) -> str:
    # We fetch the sampling key from the context only when this object
    # is executed (i.e. _aexec or _aiterate is called), so that if the object
    # is copied, the sampling key is not copied and instead it is dynamically
    # assigned.
    self.base_key = context_sampling_key.get()
    if self.base_key == '0' or not self.base_key:
      # We replace the '0' key by the default sampling_key so that it always
      # maps to the default sample.
      new_key = self.postfix if self.postfix != '0' else _DEFAULT_SAMPLING_KEY
    else:
      # Handle nested sampling. For example one could get `3#5` if there is a
      # two-level sampling scheme (i.e. X and Y both are sampled, but Y depends
      # on X) with 3 repeats of "outer" sampling and 5 repeats of "inner" one.
      new_key = f'{self.base_key}#{self.postfix}'
    return new_key

  async def _aexec(self) -> Any:
    context_sampling_key.set(self._update_sampling_key())
    try:
      result = await self.wrapped
    finally:
      context_sampling_key.set(self.base_key)
    return result

  async def _aiterate(self, iteration_depth: int = 1) -> Any:
    context_sampling_key.set(self._update_sampling_key())
    try:
      async for r in self.wrapped.with_depth(iteration_depth):
        yield r
    finally:
      context_sampling_key.set(self.base_key)


@dataclasses.dataclass
class _CacheData(
    Generic[CachedType],
    dataclasses_json.DataClassJsonMixin,
):
  """Cache data.

  Attributes:
    counters: Counts of cache hits and misses, etc.
    values_by_key: Mapping of (hashed) cache keys to the existing values.
    num_used_values_by_key: Keeps track of how many of the existing values for a
      given (hashed) cache key have been mapped to sampling_keys.
    sample_id_by_sampling_key_by_key: Mapping from a (hashed) cache and sampling
      keys to the index in the list of samples we have for that call.
  """

  # Note: The order of the attributes here determines the order in which they
  # appear in the JSON file. Smaller attributes should go at the top.
  counters: collections.Counter[str] = dataclasses.field(
      default_factory=collections.Counter,
      metadata=dataclasses_json.config(decoder=collections.Counter),
  )
  values_by_key: dict[str, list[CachedType]] = dataclasses.field(
      default_factory=dict,
      # This is the field that stores all the actual cached values. The cached
      # values can be almost of any type and they may store python tuples.
      # Unfortunately by default json does not support python tuple and it gets
      # converted to a list. Here we make a best effort encoding that is meant
      # to preserve python tuples.
      metadata=dataclasses_json.config(
          encoder=_hint_tuple_encoder,
          decoder=_hint_tuple_decoder,
      ),
  )
  num_used_values_by_key: collections.defaultdict[str, int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(int),
      metadata=dataclasses_json.config(decoder=defaultdict_decoder),
  )
  sample_id_by_sampling_key_by_key: collections.defaultdict[
      str, collections.defaultdict[str, int]
  ] = dataclasses.field(
      default_factory=nested_defaultdict_initializer,
      metadata=dataclasses_json.config(decoder=nested_defaultdict_decoder),
  )

  @classmethod
  def create_from_file(
      cls,
      cache_file_path: str,
      restore_mapping: bool = False,
      cached_value_decoder: Callable[[Any], Any] | None = None,
  ) -> _CacheData[CachedType]:
    """Returns a new instance with the contents of the cache file.

    Args:
      cache_file_path: Path of the file to read the cache from.
      restore_mapping: If True, restore the mapping from sampling_key to sample
        ids in the stored lists of values from the file. Otherwise, reset all
        the relevant structures.
      cached_value_decoder: A function that can restore the values from their
        serialized version into an object of appropriate type. See
        SimpleFunctionCache class. If None, identity decoder is used.

    Returns:
      A new cache object with contents stored in the cache file.
    """
    if cached_value_decoder is None:
      cached_value_decoder = lambda x: x

    with open(cache_file_path) as f:
      file_contents = f.read()

    try:
      cache = cls.from_json(
          file_contents,
          infer_missing=True,
      )
      # pylint: disable=protected-access
      for key_hash, cached_values in cache.values_by_key.items():
        # When reading from json, the values objects are created as dict or
        # other basic types. We convert them to the appropriate object using
        # the provided decoder.
        # Note that the encoding is done simply by calling `to_json` on the
        # cache (which corresponds to `to_json` method of `DataClassJsonMixin`
        # in `third_party/py/dataclasses_json/api.py`) so the decoder has to be
        # able to read whatever calling `from_json(to_json())` produces.
        cache.values_by_key[key_hash] = [
            cached_value_decoder(cached_value) for cached_value in cached_values
        ]
      if restore_mapping:
        logging.info('Restored sampling_key mapping from file.')
      else:
        logging.info('Create new sampling_key mapping.')
        cache.num_used_values_by_key = collections.defaultdict(int)
        cache.sample_id_by_sampling_key_by_key = (
            nested_defaultdict_initializer()
        )
      # pylint: enable=protected-access
    except Exception as error:  # pylint: disable=broad-except
      traceback.print_exc()
      raise RuntimeError(
          'Error parsing cache file %s.' % cache_file_path
      ) from error
    return cache

  def cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: CachedType,
      key_for_logging: str,
  ) -> None:
    """Stores the key-value pair in the cache.

    Each cache key may be associated with multiple values, for example when the
    function returns different samples. To distinguish between those, we may
    provide a sampling_key. In this case we try and use it to determine whether
    this is a new sample or whether we should override an existing sample.

    Args:
      key: The cache key to associate the value with.
      sampling_key: Optional sampling key (stored in a context variable), see
        `is_sampled` argument in `cache_method` decorator.
      value: The value to be stored in the cache.
      key_for_logging: Human readable version of the cache key.
    """
    # In case a cached function returns several "samples" in one call (for
    # example, LLMs can return N randomly sampled completions for the same
    # prefix), the unused samples are stored for future use. This is
    # implemented in the cache_method decorators by setting
    # cache_extra_replies=True. Each extra sample is cached explicitly using
    # the cache_value method with sampling_key=None.
    if key in self.values_by_key:
      existing_values = self.values_by_key[key]
      # We check whether we have an entry for the sampling_key.
      if key in self.sample_id_by_sampling_key_by_key:
        if sampling_key in self.sample_id_by_sampling_key_by_key[key]:
          sample_id = self.sample_id_by_sampling_key_by_key[key][sampling_key]
          # We found an entry, we assume it points to a valid entry in the list.
          assert sample_id < len(existing_values)
          # We replace this entry.
          existing_value = existing_values[sample_id]
          if value == existing_value:
            logging.info('No result added for key %s', key_for_logging)
            self.counters['add_redundant'] += 1
          else:
            # We have a match for both cache and sampling keys, but the already
            # stored value is different from the one we are trying to store.
            # This branch can not be reached if `cache_value` is called via
            # `cache_method` decorator. It can only be reached when called
            # explicitly.
            self.counters['add_overwrote'] += 1
            existing_values[sample_id] = value
            logging.info('Overwriting cached value for key %s', key_for_logging)
            logging.info('Old value: %s', repr(existing_value))
            logging.info('New value: %s', repr(value))
          return

      # The cache key is not new but the sampling_key is. This can happen for
      # two reasons: (a) sampling_key is None or (b) sampling_key is not None
      # and it is new. Case (a) corresponds either to caching deterministic
      # functions (with `is_sampled=False`) or explicit caching of extra unused
      # samples, as described above. Case (b) is when we obtained a true new
      # sample from a decorated stochastic function and want to store it.
      # We append the new value to the list of values in any case.
      logging.info('Adding new sample for key %s', key_for_logging)
      self.counters['add_new_sample'] += 1
      existing_values.append(value)
      if sampling_key is not None:
        # We also assign the sampling_key to the next available (not used) value
        # in existing_values. This value does not necessarily need to be the one
        # we have just appended to the list, for example if
        # len(existing_values) > _num_used_values_by_key[key_hash].
        self.sample_id_by_sampling_key_by_key[key][sampling_key] = (
            self.num_used_values_by_key[key]
        )
        self.num_used_values_by_key[key] += 1
    else:
      # The key is new, we add a new entry in the cache.
      logging.info('Adding new key %s', key_for_logging)
      self.counters['add_new'] += 1
      self.values_by_key[key] = [value]
      if sampling_key is not None:
        # This new entry is mapped from the sampling_key.
        self.sample_id_by_sampling_key_by_key[key][sampling_key] = 0
        # And we thus consider that we have that sample_id already used for this
        # sampling key.
        self.num_used_values_by_key[key] = 1

  def key_exists(self, key: str) -> bool:
    """Returns whether the given key exists in the cache."""
    return key in self.values_by_key

  def get_cached_value(
      self,
      key: str,
      sampling_key: str | None,
      key_for_logging: str,
      update_counters: bool = True,
  ) -> CachedType | None:
    """Retrieves a value for cache key from the cache.

    Args:
      key: The cache key to retrieve the value.
      sampling_key: Optional sampling key. See `is_sampled` of `cache_method`
        decorator.
      key_for_logging: Human readable version of the cache key.
      update_counters: Whether to update the counters: this should not be set to
        False and it is only used for the special case when we want to fallback
        to an old-style cache key for backwards compatibility.

    Returns:
      The value found in the cache if any or None otherwise.
    """
    if key in self.values_by_key:
      # The key is in the cache, we check that we have an entry for the
      # sampling_key.
      existing_values = self.values_by_key[key]
      sample_id = 0
      found = False
      if sampling_key is None:
        # Deterministic method is cached. Grab the first reply.
        sample_id = 0
        found = True
      if key in self.sample_id_by_sampling_key_by_key:
        if sampling_key in self.sample_id_by_sampling_key_by_key[key]:
          sample_id = self.sample_id_by_sampling_key_by_key[key][sampling_key]
          found = True
      if not found:
        # Do we have unused elements in the cache that we could associate with
        # this sampling key?
        # We check how many samples have already been mapped.
        if self.num_used_values_by_key[key] < len(existing_values):
          # We map a new one.
          sample_id = self.num_used_values_by_key[key]
          self.num_used_values_by_key[key] += 1
          self.sample_id_by_sampling_key_by_key[key][sampling_key] = sample_id
        else:
          if update_counters:
            self.counters['get_hit_miss_sample'] += 1
          logging.info(
              'Found key but not the sampling key: %s', key_for_logging
          )
          return None

      # We have a sample_id, we return the corresponding value.
      # The sample_id should correspond to a valid index in the list of
      # values.
      assert sample_id < len(existing_values)
      if update_counters:
        self.counters['get_hit'] += 1
      logging.info(
          'Found sample_id %d for key %s',
          sample_id,
          key_for_logging,
      )
      return existing_values[sample_id]
    else:
      # The key is not in the cache and no other coroutine is processing the
      # same call.
      if update_counters:
        self.counters['get_miss'] += 1
      logging.info('Key not found: %s', key_for_logging)
      return None

  def __iadd__(self, other: _CacheData[CachedType]) -> _CacheData[CachedType]:
    """Merges the contents of the given cache into this one.

    In case of a key collision, the values for the given key will be merged,
    preserving the value indices, and giving precedence to the existing value
    for any given index, where present.

    Args:
      other: The cache to merge into this one. Only the values_by_key will be
        merged in, not the mapping of sample_id and sample_key.

    Returns:
      This cache object, with the merged values.
    """
    for key_hash, cached_values in other.values_by_key.items():
      if key_hash not in self.values_by_key:
        self.values_by_key[key_hash] = []
      for i, cached_value in enumerate(cached_values):
        if i >= len(self.values_by_key[key_hash]):
          self.values_by_key[key_hash].append(cached_value)
    return self

  def __repr__(self):
    """Returns a string representation of a `_CacheData` object.

    The representation ignores the `values_by_key` attribute, since it clutters
    the original representation and makes it unusable for debugging purposes.
    """
    class_name = self.__class__.__name__

    fields = [
        f'{f.name}={getattr(self, f.name)!r}'
        for f in dataclasses.fields(self)
        if f.name != 'values_by_key'
    ]
    fields.append(f'values_by_key=<dict with {len(self.values_by_key)} items>')

    # _CacheData(counters=..., num_used_values_by_key=...,
    #            sample_id_by_sampling_key_by_key=...,
    #            values_by_key=...)
    return f"{class_name}_repr({', '.join(fields)})"


@dataclasses.dataclass
class SimpleFunctionCache(
    Generic[CachedType],
    SimpleFileCache[CachedType],
):
  """Simple implementation of a thread-safe generic cache.

  Attributes:
    cache_filename: Full path to the JSON cache file.
    cached_value_decoder: A function that can restore the values from their
      serialized version into an object of appropriate type. When written to
      disk, the cached values are json-serialized, so objects get converted into
      dictionaries, hence the cached_value_decoder should convert those
      dictionaries back to objects of the appropriate type (i.e. CachedType).
  """

  cache_filename: str | None = None
  # We add an empty encoder to make this work on functions that may be methods
  # from a non-serializeable object.
  cached_value_decoder: Callable[[Any], CachedType] | None = dataclasses.field(
      default=None,
  )
  _cache_data: _CacheData[CachedType] = dataclasses.field(
      default_factory=_CacheData
  )
  _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
  _calls_in_progress: set[tuple[str, str | None]] = dataclasses.field(
      default_factory=set,
  )

  def cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: CachedType,
  ) -> None:
    """See base class (SimpleCache)."""
    try:
      # Sanity check: Ensure the value can be safely cached and restored.
      # We attempt to deep-copy the value before caching it. This acts as a
      # robust test to verify that the object doesn't contain un-cacheable
      # state, such as file handles or thread locks. An object that cannot be
      # deep-copied would also fail to serialize correctly to our JSON cache
      # file, so this check prevents corrupting the cache with problematic data.
      _ = copy.deepcopy(value)
    except Exception as e:
      raise ValueError(
          f'Failed to copy value to be cached: {value}. Make sure the caching'
          ' decorator is set properly. Values that cannot be copied will not'
          ' be properly restored from cache.'
      ) from e
    # A human readable version of the cache key.
    key_for_logging = get_key_for_logging(key, sampling_key)
    logging.info('Inserting result for key %s', key_for_logging)
    # A hash of the key is used whenever we need to map from the key.
    key_hash = utils.get_str_hash(key)
    with self._lock:
      self._cache_data.cache_value(
          key_hash, sampling_key, value, key_for_logging
      )

  async def get_cached_value(
      self,
      key: str,
      sampling_key: str | None,
  ) -> CachedType | None:
    """Retrieves a value for cache key from the cache.

    Overridden from base class (SimpleCache).

    Args:
      key: The cache key to retrieve the value.
      sampling_key: Optional sampling key. See `is_sampled` of `cache_method`
        decorator.

    Returns:
      The value found in the cache if any or None otherwise.
    """
    # A human readable version of the cache key.
    key_for_logging = get_key_for_logging(key, sampling_key)
    # A hash of the key is used whenever we need to map from the key.
    key_hash = utils.get_str_hash(key, fallback_if_safe=False)
    key_hash_fallback = utils.get_str_hash(key, fallback_if_safe=True)

    logging.info('Looking up key %s', key_for_logging)
    if (key, sampling_key) in self._calls_in_progress:
      logging.info(
          'Key-sampling_key pair %s is already in progress.',
          key_for_logging,
      )
      # Another coroutine has already started processing the same call.
      while True:
        await asyncio.sleep(0)
        if (key, sampling_key) not in self._calls_in_progress:
          # Another coroutine has finished and cached the value.
          if not self._cache_data.key_exists(
              key_hash
          ) and not self._cache_data.key_exists(key_hash_fallback):
            raise ValueError(
                'Another coroutine processed the same call but did not cache '
                'the obtained value.'
            )
          break
    with self._lock:
      result = self._cache_data.get_cached_value(
          key_hash, sampling_key, key_for_logging
      )
      if result is None:
        # We do not have a cached value for this key, but we can try to get a
        # cached value for an old-style key, at least when it is safe to create
        # such a key, which means in particular that there is no multimodal data
        # in the arguments of the function.
        # TODO: Remove this fallback once we are confident that most
        # existing caches have been updated. The change of cache key to handle
        # multimodal data was introduced on 20250224, so this means it should be
        # fine to remove this fallback on 20250524.
        result = self._cache_data.get_cached_value(
            key_hash_fallback,
            sampling_key,
            key_for_logging,
            update_counters=False,
        )
    return result

  def get_key_count(self):
    """Returns the number of keys in the cache (a measure of cache size).

    Overridden from base class (SimpleCache).
    """
    return len(self._cache_data.values_by_key)

  def load(
      self,
      *,
      restore_mapping: bool = False,
      overwrite: bool = True,
      cache_filename: str | None = None,
  ):
    """Loads the cache from file.

    Overridden from base class (SimpleFileCache).

    Args:
      restore_mapping: If True, will try and restore the mapping between
        sampling_key and sample_ids from disk, otherwise creates a new one. Only
        relevant if `overwrite=True` (otherwise nevers restores mapping).
      overwrite: If True, then any existing cache contents will be discarded and
        completely overwritten by the contents of the cache file. If False, then
        the existing cache contents will be preserved and the contents of the
        cache file will be merged into it.
      cache_filename: If specified, then will load from the given file path;
        otherwise, by default will load from `self.cache_filename`.
    """
    if not cache_filename:
      cache_filename = self.cache_filename
    if not cache_filename:
      raise ValueError(
          'Cache filename must be provided when loading from disk.'
      )
    new_cache_data = _CacheData.create_from_file(
        cache_filename,
        restore_mapping=restore_mapping,
        cached_value_decoder=self.cached_value_decoder,
    )
    if overwrite:
      self._cache_data = new_cache_data
    else:
      self._cache_data += new_cache_data

  def save(
      self,
      *,
      overwrite: bool = False,
      cache_filename: str | None = None,
  ) -> None:
    """Writes the contents of the cache to the given directory.

    Overridden from base class (SimpleFileCache).

    Args:
      overwrite: If False (default) right before saving cache in the file we
        check if file with that name already exists and raise error if it does.
        If True we overwrite.
      cache_filename: If specified, then will save to the given file path;
        otherwise, by default will save to `self.cache_filename`.

    Raises:
      FileExistsError: If trying to save cache in file that already exists.
    """
    if not cache_filename:
      cache_filename = self.cache_filename
    if not cache_filename:
      raise ValueError('Cache filename must be provided when storing on disk.')
    if os.path.exists(cache_filename) and not overwrite:
      raise FileExistsError(f'File {cache_filename} already exists.')
    # Create the directory if it doesn't exist yet
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    with open(cache_filename, 'w') as f:
      logging.info('Writing cache to file: %s', cache_filename)
      # The following call corresponds to `to_json` method of
      # `DataClassJsonMixin` in `third_party/py/dataclasses_json/api.py`. This
      # method in particular applies all the custom encoder transofrmations to
      # the individual fields provided via `metadata`.
      f.write(self._cache_data.to_json())
