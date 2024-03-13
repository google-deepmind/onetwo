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

"""Utility functions for adding caching capabilities to methods."""

import abc
import asyncio
import collections
from collections.abc import ByteString, Callable, Coroutine, Hashable, Mapping, Sequence
import contextvars
import copy
import dataclasses
import functools
import hashlib
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


def _get_hash(key: Any) -> str:
  """Best-effort hashing of various kinds of objects."""
  match key:
    case str():
      bytes_value = key.encode('utf-8')
    case list() | tuple():
      bytes_value = str(key).encode('utf-8')
    case dict():
      bytes_value = str(sorted(key.items())).encode('utf-8')
    case set():
      bytes_value = str(sorted(list(key))).encode('utf-8')
    case Hashable():
      bytes_value = str(hash(key)).encode('utf-8')
    case _:
      if hasattr(key, 'tobytes'):  # Type `str` has no such attribute.
        # This handles the case of a np.ndarray.
        bytes_value = key.tobytes()
      else:
        bytes_value = bytes(key, 'utf-8')
  return hashlib.sha224(bytes_value).hexdigest()


def get_key_for_logging(key: str, sampling_key: str | None = None) -> str:
  """Produces key and sampling_key information for display in the logs."""
  rkey = repr(key)
  if len(rkey) > 50:
    rkey = rkey[:20] + f'[{len(rkey)-40} chars]' + rkey[-20:]
  return (
      f'{rkey} (hash: {_get_hash(key)} - sampling key:{repr(sampling_key)})'
  )


def nested_defaultdict_decoder(
    arg: collections.defaultdict
) -> collections.defaultdict:
  """Decodes a defaultdict of int to defaultdict of str to int."""
  decoded = dict(arg)
  return collections.defaultdict(
      functools.partial(collections.defaultdict, int),
      {k: collections.defaultdict(int, v) for k, v in decoded.items()},
  )


def defaultdict_decoder(
    arg: collections.defaultdict
) -> collections.defaultdict:
  """Decodes a defaultdict of int to int."""
  decoded = dict(arg)
  return collections.defaultdict(
      int,
      {k: v for k, v in decoded.items()},
  )


def nested_defaultdict_initializer(
    unused_arg: collections.defaultdict | None = None
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


class CacheEnabled(Generic[CachedType], metaclass=abc.ABCMeta):
  """Interface for a class that has a cache attached to it.

  Methods of (any implementation of) this class can be decorated with
  `cache_method` to enable caching. There can be multiple decorated methods, in
  which case a single cache will handle all of them.
  """

  @property
  @abc.abstractmethod
  def cache_handler(self) -> SimpleCache[CachedType]:
    """Cache attached to this object."""

  @property
  @abc.abstractmethod
  def disable_caching(self) -> bool:
    """Whether caching is enabled for this object."""


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
      return _get_hash(value)
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
  _method: (
      Callable[..., CachedType] | None
  ) = dataclasses.field(default=None, init=False)
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
    key = _create_cache_key(
        self._name, arguments, self.hashed
    )
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
      isinstance(values, str) or  # values = 'abcd' won't do.
      isinstance(values, ByteString) or  # values = b'abcd' won't do.
      not isinstance(values, Sequence)
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
        type(values[0])
    )
  else:
    for value in values[1:]:
      cache_value_callback(value)
  return values[0]


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
      method: Callable[..., ReturnType | Coroutine[None, None, ReturnType]]
  ) -> Callable[..., ReturnType | Coroutine[None, None, ReturnType]]:
    """Actual decorator replacing the method with a version that uses cache."""
    nonlocal name
    # name will be used to separate between different cached methods. Make sure
    # it is not empty.
    if name is None:
      name = method.__name__
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
      # Check that the class of the method supports caching.
      # It would be best to do this at decoration time, but the method
      # decorators are executed before the class is created.
      if not isinstance(obj_with_cache, CacheEnabled):  # pytype: disable=attribute-error
        raise ValueError(
            "Decorator @cache_method is applied to a method whose class doesn't"
            ' inherit from CacheEnabled.'
        )
      if is_sampled:
        # Get sampling key from context.
        sampling_key = context_sampling_key.get()
      else:
        sampling_key = None
      key = maker.create_key(obj_with_cache, args, kwargs)
      if obj_with_cache.disable_caching:
        value = None
      else:
        value = await obj_with_cache.cache_handler.get_cached_value(
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
          an Executable, instead of storing it directly, we attach
          to it a postprocessing callback to store it after the end of the
          iterations and return it.
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
          obj_with_cache.cache_handler.cache_value(
              key=key,
              sampling_key=sampling_key,
              value=v,
          )
          return v

        value = executing.ExecutableWithPostprocessing(
            wrapped=value, postprocessing_callback=callback
        )
      else:
        obj_with_cache.cache_handler.cache_value(
            key=key,
            sampling_key=sampling_key,
            value=value,
        )
      return value

    @functools.wraps(method)
    async def ainner(self, *args, **kwargs) -> ReturnType:
      """Async method decorated with cache."""
      nonlocal cache_extra_replies
      # Here "self" is the object of class CacheEnabled. We are decorating one
      # of its methods.
      value, (key, sampling_key) = await lookup(self, maker, args, kwargs)
      if value is not None:
        return value
      # Actually process the call. But first indicate that we are processing it.
      if hasattr(self.cache_handler, 'calls_in_progress'):
        self.cache_handler.calls_in_progress |= {(key, sampling_key)}
      # Call/await depending on whether decorating a sync or async method.
      try:
        if inspect.iscoroutinefunction(method):
          # pytype expects *args and **kwargs.
          value = await method(*((self,) + args), **kwargs)
        else:
          value = method(*((self,) + args), **kwargs)
      except Exception as err:  # pylint: disable=broad-except
        if hasattr(self.cache_handler, 'calls_in_progress'):
          # We failed to process the call. Unblock other coroutines waiting for
          # the value.
          self.cache_handler.calls_in_progress -= {(key, sampling_key)}
        raise ValueError(
            f'Error raised while executing method {method}:\n{err}\n'
        ) from err
      # We may need to resolve cache_extra_replies's value at runtime.
      do_cache_extra = utils.RuntimeParameter[bool](
          cache_extra_replies, self
      ).value()
      if do_cache_extra:
        # Method returns a Sequence of CachedType elements. Handle this case.
        logging.info('Caching extra replies for method %s.', method.__name__)
        value = return_first_and_cache_remaining(
            values=value,
            disable_caching=self.disable_caching,
            cache_value_callback=functools.partial(
                self.cache_handler.cache_value,
                key,
                None,  # No sampling_key set.
            )
        )
      result = store(self, value, key, sampling_key)
      # Finally indicate that we have processed the call and cached the value.
      if hasattr(self.cache_handler, 'calls_in_progress'):
        self.cache_handler.calls_in_progress -= {(key, sampling_key)}
      return result

    return ainner

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
    if (self.base_key == '0' or not self.base_key):
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
    result = await self.wrapped
    context_sampling_key.set(self.base_key)
    return result

  async def _aiterate(self, iteration_depth: int = 1) -> Any:
    context_sampling_key.set(self._update_sampling_key())
    async for r in self.wrapped.with_depth(iteration_depth):
      yield r
    context_sampling_key.set(self.base_key)


def add_json_extension(cache_filename: str) -> str:
  if cache_filename.endswith('.json'):
    return cache_filename
  return f'{cache_filename}.json'


@dataclasses.dataclass
class SimpleFunctionCache(
    Generic[CachedType],
    dataclasses_json.DataClassJsonMixin,
    SimpleCache[CachedType],
):
  """Simple implementation of a generic cache.

  Attributes:
    cache_filename: Any user-defined string that identifies the cache object.
      Used as a name of the file when storing cache on disk.
    cached_value_decoder: A function that can restore the values from their
      serialized version into an object of appropriate type. When written to
      disk, the cached values are json-serialized, so objects get converted into
      dictionaries, hence the cached_value_decoder should convert those
      dictionaries back to objects of the appropriate type (i.e. CachedType).
    _counters: Counts of cache hits and misses, etc.
    _values_by_key: Mapping of (hashed) cache keys to the existing values.
    _num_used_values_by_key: Keeps track of how many of the existing values for
      a given (hashed) cache key have been mapped to sampling_keys.
    _sample_id_by_sampling_key_by_key: Mapping from a (hashed) cache and
      sampling keys to the index in the list of samples we have for that
      call.
  """
  # Note: The order of the attributes here determines the order in which they
  # appear in the JSON file. Smaller attributes should go at the top.
  cache_filename: str = dataclasses.field(default_factory=str)
  # Note: `exclude=dataclasses_json.Exclude.ALWAYS` is not sufficient. Indeed,
  # even if this field is skipped from the json output, the `to_json()` function
  # will still call `as_dict()` on it before skipping it, see `value = _asdict(`
  # line in dataclasses_json/core.py.
  # We add an empty encoder to make this work on functions that may be methods
  # from a non-serializeable object.
  cached_value_decoder: Callable[[Any], CachedType] | None = dataclasses.field(
      metadata=dataclasses_json.config(
          exclude=dataclasses_json.Exclude.ALWAYS, encoder=lambda _: None,
      ),  # We don't store this on disk.
      default=None,
  )
  _counters: collections.Counter[str] = dataclasses.field(
      default_factory=collections.Counter,
      metadata=dataclasses_json.config(decoder=collections.Counter),
  )
  _values_by_key: dict[str, Any] = dataclasses.field(
      default_factory=dict,
  )
  _num_used_values_by_key: collections.defaultdict[str, int] = (
      dataclasses.field(
          default_factory=lambda: collections.defaultdict(int),
          metadata=dataclasses_json.config(decoder=defaultdict_decoder),
      )
  )
  _sample_id_by_sampling_key_by_key: collections.defaultdict[
      str, collections.defaultdict[str, int]
  ] = dataclasses.field(
      default_factory=nested_defaultdict_initializer,
      metadata=dataclasses_json.config(decoder=nested_defaultdict_decoder),
  )
  lock: threading.Lock = dataclasses.field(
      metadata=dataclasses_json.config(
          exclude=dataclasses_json.Exclude.ALWAYS,
          encoder=lambda _: None,
      ),  # We don't store this on disk.
      default_factory=threading.Lock,
  )
  calls_in_progress: set[tuple[str, str | None]] = dataclasses.field(
      metadata=dataclasses_json.config(
          exclude=dataclasses_json.Exclude.ALWAYS,
          encoder=lambda _: None,
      ),  # We don't store this on disk.
      default_factory=set,
  )

  def _cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: CachedType,
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
    """

    # In case a cached function returns several "samples" per one call (for
    # example, LLMs can return N randomly sampled completions for the same
    # prefix), the unused samples are stored for future use. This is implemented
    # in cache_method for cache_extra_replies=True. Each extra sample is cached
    # explicitly using cache_value method with sampling_key=None.
    try:
      # We attempt to copy the value in order to see if it can be pickled.
      # If this fails, we will not be able to store the cache and it may lead
      # to issues when trying to load the cache data.
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
    key_hash = _get_hash(key)
    if key_hash in self._values_by_key:
      existing_values = self._values_by_key[key_hash]
      # We check whether we have an entry for the sampling_key.
      if key_hash in self._sample_id_by_sampling_key_by_key:
        if sampling_key in self._sample_id_by_sampling_key_by_key[key_hash]:
          sample_id = self._sample_id_by_sampling_key_by_key[key_hash][
              sampling_key
          ]
          # We found an entry, we assume it points to a valid entry in the list.
          assert sample_id < len(existing_values)
          # We replace this entry.
          existing_value = existing_values[sample_id]
          if value == existing_value:
            logging.info('No result added for key %s', key_for_logging)
            self._counters['add_redundant'] += 1
          else:
            # We have a match for both cache and sampling keys, but the already
            # stored value is different from the one we are trying to store.
            # This branch can not be reached if `cache_value` is called via
            # `cache_method` decorator. It can only be reached when called
            # explicitly.
            self._counters['add_overwrote'] += 1
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
      self._counters['add_new_sample'] += 1
      existing_values.append(value)
      if sampling_key is not None:
        # We also assign the sampling_key to the next available (not used) value
        # in existing_values. This value does not necessarily need to be the one
        # we have just appended to the list, for example if
        # len(existing_values) > _num_used_values_by_key[key_hash].
        self._sample_id_by_sampling_key_by_key[key_hash][sampling_key] = (
            self._num_used_values_by_key[key_hash]
        )
        self._num_used_values_by_key[key_hash] += 1
    else:
      # The key is new, we add a new entry in the cache.
      logging.info('Adding new key %s', key_for_logging)
      self._counters['add_new'] += 1
      self._values_by_key[key_hash] = [value]
      if sampling_key is not None:
        # This new entry is mapped from the sampling_key.
        self._sample_id_by_sampling_key_by_key[key_hash][sampling_key] = 0
        # And we thus consider that we have that sample_id already used for this
        # sampling key.
        self._num_used_values_by_key[key_hash] = 1

  def cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: CachedType,
  ) -> None:
    """See parent class."""
    with self.lock:
      self._cache_value(
          key,
          sampling_key,
          value,
      )

  async def get_cached_value(
      self,
      key: str,
      sampling_key: str | None,
  ) -> CachedType | None:
    """Retrieves a value for cache key from the cache.

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
    key_hash = _get_hash(key)
    logging.info('Looking up key %s', key_for_logging)
    if (key, sampling_key) in self.calls_in_progress:
      logging.info(
          'Key-sampling_key pair %s is already in progress.',
          key_for_logging,
      )
      # Another coroutine has already started processing the same call.
      while True:
        await asyncio.sleep(0)
        if (key, sampling_key) not in self.calls_in_progress:
          # Another coroutine has finished and cached the value.
          if key_hash not in self._values_by_key:
            raise ValueError(
                'Another coroutine processed the same call but did not cache '
                'the obtained value.'
            )
          break
    with self.lock:
      if key_hash in self._values_by_key:
        # The key is in the cache, we check that we have an entry for the
        # sampling_key.
        existing_values = self._values_by_key[key_hash]
        sample_id = 0
        found = False
        if sampling_key is None:
          # Deterministic method is cached. Grab the first reply.
          sample_id = 0
          found = True
        if key_hash in self._sample_id_by_sampling_key_by_key:
          if (
              sampling_key
              in self._sample_id_by_sampling_key_by_key[key_hash]
          ):
            sample_id = self._sample_id_by_sampling_key_by_key[key_hash][
                sampling_key
            ]
            found = True
        if not found:
          # Do we have unused elements in the cache that we could associate with
          # this sampling key?
          # We check how many samples have already been mapped.
          if self._num_used_values_by_key[key_hash] < len(existing_values):
            # We map a new one.
            sample_id = self._num_used_values_by_key[key_hash]
            self._num_used_values_by_key[key_hash] += 1
            self._sample_id_by_sampling_key_by_key[key_hash][
                sampling_key
            ] = sample_id
          else:
            self._counters['get_hit_miss_sample'] += 1
            logging.info(
                'Found key but not the sampling key: %s', key_for_logging
            )
            return None

        # We have a sample_id, we return the corresponding value.
        # The sample_id should correspond to a valid index in the list of
        # values.
        assert sample_id < len(existing_values)
        self._counters['get_hit'] += 1
        logging.info(
            'Found sample_id %d for key %s',
            sample_id,
            key_for_logging,
        )
        return existing_values[sample_id]
      else:
        # The key is not in the cache and no other coroutine is processing the
        # same call.
        self._counters['get_miss'] += 1
        logging.info('Key not found: %s', key_for_logging)
        return None

  def write_to_directory(
      self,
      output_dir: str,
      overwrite: bool = False,
  ) -> None:
    """Writes the contents of the cache to the given directory.

    Args:
      output_dir: Fully qualified path of directory to be written to.
      overwrite: If False (default) right before saving cache in the file we
        check if file with that name already exists and raise error if it does.
        If True we overwrite.

    Raises:
      FileExistsError: If trying to save cache in file that already exists.
    """
    if not self.cache_filename:
      raise ValueError('Cache filename must be provided when storing on disk.')
    os.path.makedirs(output_dir)  # Create all dirs and subdirs.
    filename = add_json_extension(self.cache_filename)
    cache_file_path = os.path.join(output_dir, filename)
    if os.path.exists(cache_file_path) and not overwrite:
      raise FileExistsError(f'File {cache_file_path} already exists.')
    with open(cache_file_path, 'w') as f:
      logging.info('Writing cache to file: %s', cache_file_path)
      f.write(self.to_json())

  @classmethod
  def create_from_file(
      cls,
      cache_file_path: str,
      restore_mapping: bool = False,
      cached_value_decoder: Callable[[Any], Any] | None = None
  ) -> 'SimpleFunctionCache':
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
      cache = cls.from_json(file_contents, infer_missing=True)
      # pylint: disable=protected-access
      for key_hash, cached_values in cache._values_by_key.items():
        # When reading from json, the values objects are created as dict or
        # other basic types. We convert them to the appropriate object using
        # the provided decoder.
        # Note that the encoding is done simply by calling `to_json` on the
        # cache so the decoder has to be able to read whatever calling
        # `from_json(to_json())` produces.
        cache._values_by_key[key_hash] = [
            cached_value_decoder(cached_value) for cached_value in cached_values
        ]

        if restore_mapping:
          logging.info('Restored sampling_key mapping from file.')
        else:
          logging.info('Create new sampling_key mapping.')
          cache._num_used_values_by_key = collections.defaultdict(int)
          cache._sample_id_by_sampling_key_by_key = (
              nested_defaultdict_initializer()
          )
        cache.cached_value_decoder = cached_value_decoder
      # pylint: enable=protected-access
    except Exception as error:  # pylint: disable=broad-except
      traceback.print_exc()
      raise RuntimeError(
          'Error parsing cache file %s.' % cache_file_path
      ) from error
    return cache
