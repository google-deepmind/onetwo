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

"""Utility functions that are shared across many of the OneTwo colabs.

Since these are intended primarily for use in colabs, they may contain `print`
statements, in cases where such output tends to be useful in a colab workflow.
(In colab, the output of `print` statements will be seen by default, whereas
logged content requires extra work in order to access.)
"""

import collections
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import os
from typing import Any

from onetwo.core import caching






def get_cache_filename(backend_name: Any) -> str:
  """Returns an appropriately formatted cache filename for the given backend.

  Args:
    backend_name: An arbitrary string or enum that would uniquely identify the
      backend within the context of the set of backends whose caches are being
      managed in the same BackendCaches object (or whose caches will be saved in
      the same directory). E.g., 'gemini-1.0-pro', 'gemini-1.5-flash', etc.
  """
  if not isinstance(backend_name, str):
    backend_name = str(backend_name)
  backend_name = backend_name.replace('.', '_')
  backend_name = backend_name.replace('-', '_')
  return f'{backend_name}.json'


def _get_cache_size(backend: Any) -> int:
  """Returns the number of keys in the cache of the given backend."""
  # TODO: Provide a clean way to access the size of a cache without
  # accessing the private `_cache_handler` attribute.
  cache_handler = backend._cache_handler  # pylint: disable=protected-access
  return cache_handler.get_key_count()


def diff_cache_data(
    *, before: caching._CacheData, after: caching._CacheData  # pylint: disable=protected-access
) -> caching._CacheData:
  """Returns the difference between the two given cache data objects.

  Note that this is not a perfect diff, as values_by_key maps key to a list of
  entries, and counters map to counts of the events, but this function leaves
  all entries which are in some way modified (ex. if there was a new entry in
  values_by_key that has a different sampling key).

  Args:
    before: The cache data object to compare.
    after: The cache data object to compare.

  Returns:
    The difference between the two given cache data objects.
  """

  def _diff_dict(before: dict[Any, Any], after: dict[Any, Any]):
    return {k: v for k, v in after.items() if k not in before or v != before[k]}

  return caching._CacheData(  # pylint: disable=protected-access
      counters=collections.Counter(_diff_dict(before.counters, after.counters)),
      values_by_key=_diff_dict(before.values_by_key, after.values_by_key),
      num_used_values_by_key=_diff_dict(
          before.num_used_values_by_key, after.num_used_values_by_key
      ),
      sample_id_by_sampling_key_by_key=_diff_dict(
          before.sample_id_by_sampling_key_by_key,
          after.sample_id_by_sampling_key_by_key,
      ),
  )


# TODO: Consider moving the bulk of this class to `core/caching.py`,
# and then just providing a thin wrapper here that adds the print statements
# and other colab-specific code.
@dataclasses.dataclass(kw_only=True)
class CachedBackends(Mapping[str, caching.FileCacheEnabled]):
  """Manages a set of backends with their caches.

  Can be treated as a mapping from backend name (arbitrary name for display and
  lookup purposes) to backend object, while also providing methods for managing
  the caches. The assumption is that all of the backend caches are intended to
  be stored in the same directory. In the case where one is working as part of
  a group, the assumption is that the group maintains one shared cache directory
  (which is updated only occasionally, via careful coordination), and on top of
  that, each individual would have their own cache directory, which follows the
  same structure as the shared cache directory, and which they can freely
  update. Normally, the `cache_filename` attribute of each backend should be a
  path returned by the `get_cache_path()` method of this object.

  Typical syntax for creating a `CachedBackends` object:
  ```
    cached_backends = CachedBackends(
        own_cache_directory=...,
        shared_cache_directory=...,
        additional_cache_directories=[...],
    )
    cached_backends['unique_backend_name_1'] = SomeLLMBackendClass(
        ...,
        cache_filename=cached_backends.get_cache_path('unique_backend_name_1')
    cached_backends['unique_backend_name_2'] = SomeOtherBackendClass(
        ...,
        cache_filename=caches.get_cache_path('unique_backend_name_2'))
    ...
  ```
  and so on for each backend.

  Attributes:
    own_cache_directory: A directory owned by the user in which they can freely
      save updates to the caches, without fear of clobbering anyone else's
      changes. When we perform `save_backend_caches()`, it will write to
      `own_cache_directory`.
    shared_cache_directory: A directory containing an "official" cache file for
      each of the relevant backends. If specified, then we will read from the
      shared cache directory (in addition to the `own_cache_directory`) and give
      precedence to its contents, but we only write to it under special
      circumstances.
    additional_cache_directories: If you want to automatically merge in content
      from any of your teammates' cache directories or from a cache that was
      output by a batch eval run, you can list the additional directories here.
      These will be treated in a stritctly read-only manner.
  """

  own_cache_directory: str
  shared_cache_directory: str | None = None
  additional_cache_directories: Sequence[str] | None = None

  # The backends whose caches are being managed by this object. Mapping of
  # backend name to backend object.
  _backends: dict[str, caching.FileCacheEnabled] = dataclasses.field(
      default_factory=dict
  )

  def __getitem__(self, key: str):
    return self._backends[key]

  def __setitem__(self, key: str, item: caching.FileCacheEnabled):
    self._backends[key] = item

  def __delitem__(self, key: str):
    del self._backends[key]

  def __iter__(self):
    return iter(self._backends)

  def __len__(self):
    return len(self._backends)

  def get_cache_path(self, backend_name: str) -> str:
    """Returns the path for caching a given backend.

    Args:
      backend_name: An arbitrary string identifying the backend, upon which the
        cache filename will be based (as per `get_cache_filename()`). The name
        should be sufficient to uniquely identify the backend within the context
        of the set of backends whose caches are being managed in the same
        BackendCaches object (or whose caches will be saved in the same
        directory). E.g., 'gemini_1_0_pro', 'gemini_1_5_flash', 'search_engine',
        etc.
    """
    return os.path.join(
        self.own_cache_directory, get_cache_filename(backend_name)
    )

  def load_backend_cache(self, backend_name: str, *, overwrite: bool = True):
    """Checks if the cache file(s) already exist, in which case we load them.

    Args:
      backend_name: Name of the backend for which to load the cache.
      overwrite: If True, then completely replaces the current in-memory cache
        with the contents of the cache file(s). If False, then preserves the
        current in-memory cache contents, while merging in any additional
        content from the file(s).
    """
    if backend_name not in self._backends:
      raise ValueError(
          f'No matching backend found for backend name: {backend_name}'
      )
    backend = self._backends[backend_name]
    if not backend.cache_filename:
      print(
          f'No cache filename specified for {backend_name}. Not loading.'
      )
      return

    cache_filenames = []
    cache_file_basename = os.path.basename(backend.cache_filename)
    if self.shared_cache_directory:
      shared_cache_filename = os.path.join(
          self.shared_cache_directory, cache_file_basename
      )
      cache_filenames.append(shared_cache_filename)
    if backend.cache_filename and backend.cache_filename not in cache_filenames:
      cache_filenames.append(backend.cache_filename)
    if self.additional_cache_directories:
      for cache_directory in self.additional_cache_directories:
        cache_filename = os.path.join(cache_directory, cache_file_basename)
        if cache_filename not in cache_filenames:
          cache_filenames.append(cache_filename)

    if cache_filenames:
      print(f'Loading {len(cache_filenames)} cache file(s) for {backend_name}.')
    else:
      print(f'No cache files specified for {backend_name}.')

    for cache_filename in cache_filenames:
      if os.path.exists(cache_filename):
        print(f'Loading cache from {cache_filename} ({overwrite=}).')
        cache_size_before = _get_cache_size(backend)
        backend.load_cache(overwrite=overwrite, cache_filename=cache_filename)
        cache_size_after = _get_cache_size(backend)
        print(
            f'Loaded {cache_size_after - cache_size_before} items: '
            f'{cache_size_before} => {cache_size_after}.'
        )
        overwrite = False
      else:
        print(f'Cache file does not exist: {cache_filename}')

  def load_caches(self, *, overwrite: bool = True):
    """Loads the caches of all the currently managed backends."""
    for backend_name in self._backends:
      self.load_backend_cache(backend_name, overwrite=overwrite)

  # pylint: disable=protected-access
  def extract_cache_data(self) -> dict[str, caching._CacheData]:
    """Returns a mapping of backend name to a copy of its cache data."""
    result = {}
    for backend_name in self._backends:
      cache_handler = self._backends[backend_name]._cache_handler
      if not isinstance(cache_handler, caching.SimpleFunctionCache):
        raise ValueError(
            f'Backend {backend_name} has an unsupported cache type'
            f' ({type(cache_handler)}). Current implementation of BackendCaches'
            ' only supports SimpleFunctionCache.'
        )
      cache_data = cache_handler._cache_data
      result[backend_name] = copy.deepcopy(cache_data)
    return result

  def subtract_cache_data(self, cache_data: dict[str, caching._CacheData]):
    """Updates all currently managed backend caches to contain the just diff.

    This can be used, for example, to determine which specific contents were
    added to the backend caches during a given experiment run. E.g., if one
    were to run a large number of experiments in parallel, all starting from
    the same (potentially large) existing cache file, it can be useful at the
    end of the experiments to save to file just the contents that were added
    to each backend cache during the given experiment run (to avoid expensive
    file I/O on large amounts of duplicate content) and then merge all of those
    diffs together with the original cache file to create a single new
    all-encompassing cache file for each backend at the very end.

    Args:
      cache_data: The cache data to diff against. In the motivating use case,
        this would typically represent the "original" contents of the backend
        caches prior to the current experiment run.
    """
    for backend_name in self._backends:
      self_cache_handler = self._backends[backend_name]._cache_handler
      if not isinstance(self_cache_handler, caching.SimpleFunctionCache):
        print(f'Backend {backend_name} is not a SimpleFunctionCache.')
        continue
      self_cache_data = self_cache_handler._cache_data
      self_cache_size = _get_cache_size(self._backends[backend_name])

      if backend_name not in cache_data:
        print(f'Backend {backend_name} not found in cache data to subtract.')
        continue
      other_cache_data = cache_data[backend_name]

      self_cache_handler._cache_data = diff_cache_data(
          after=self_cache_data, before=other_cache_data
      )

      new_cache_size = _get_cache_size(self._backends[backend_name])

      print(f'{backend_name}: {self_cache_size} => {new_cache_size}')
      # pylint: enable=protected-access

  def save_caches(
      self,
      *,
      cache_directory: str | None = None,
  ):
    """Saves the caches of all the currently managed backends.

    Args:
      cache_directory: If specified, then will save the caches to this
        directory. Otherwise, will save the caches to the location configured
        when the backend was created (typically under `own_cache_directory`).
    """
    for backend in self._backends.values():
      cache_filename = backend.cache_filename
      if cache_directory:
        cache_filename = os.path.join(
            cache_directory, os.path.basename(cache_filename))
      print(f'Saving cache to {cache_filename}.')
      backend.save_cache(cache_filename=cache_filename, overwrite=True)

  def print_cache_summary(self):
    """Prints a summary of the caches of all the currently managed backends."""
    print('Cache summary:')
    for backend_name, backend in self._backends.items():
      # TODO: Provide a clean way to generate these kind of
      # diagnostics without accessing private attributes.
      cache_handler = backend._cache_handler  # pylint: disable=protected-access
      if not isinstance(cache_handler, caching.SimpleFunctionCache):
        raise ValueError(
            f'Backend {backend_name} has an unsupported cache type'
            f' ({type(cache_handler)}). Current implementation of BackendCaches'
            ' only supports SimpleFunctionCache.'
        )
      cache_data = cache_handler._cache_data  # pylint: disable=protected-access
      calls_in_progress = cache_handler._calls_in_progress  # pylint: disable=protected-access

      print(f'* {backend_name}: {backend.cache_filename}')
      print(f'  * Cache contains {_get_cache_size(backend)} items.')
      print(f'  * Counters: {cache_data.counters}')
      num_calls_in_progress = len(calls_in_progress)
      print(f'  * Calls in progress: {num_calls_in_progress}')

  def clear_all_calls_in_progress(self):
    for backend_name, backend in self._backends.items():
      cache_handler = backend._cache_handler  # pylint: disable=protected-access
      if not isinstance(cache_handler, caching.SimpleFunctionCache):
        # Only SimpleFunctionCache has a `_calls_in_progress` field.
        continue
      calls_in_progress = cache_handler._calls_in_progress  # pylint: disable=protected-access
      if calls_in_progress:
        print(f'Clearing calls in progress for {backend_name}.')
        print(f'BEFORE: {len(calls_in_progress)}')
        calls_in_progress.clear()
        print(f'AFTER: {len(calls_in_progress)}')
