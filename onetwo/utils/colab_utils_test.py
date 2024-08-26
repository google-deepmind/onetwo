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

import asyncio
import dataclasses
import os

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backends_base
from onetwo.core import caching
from onetwo.utils import colab_utils


@dataclasses.dataclass
class TestBackend(caching.FileCacheEnabled, backends_base.Backend):
  """A simple test backend that has a cache and a configurable mock method."""

  return_value: str = 'default_return_value'

  def __post_init__(self) -> None:
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )

  @caching.cache_method(name='get_value')
  def get_value(self, arg: str) -> str:
    del arg
    return self.return_value


class CachedBackendsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base_case', 'search_engine', 'search_engine.json'),
      ('converts_dashes_and_periods', 'gemini-1.0-pro', 'gemini_1_0_pro.json'),
  )
  def test_get_cache_filename(self, backend_name, expected_cache_filename):
    self.assertEqual(
        expected_cache_filename,
        colab_utils.get_cache_filename(backend_name),
    )

  def test_should_act_like_an_order_preserving_mapping_of_name_to_backend(self):
    cached_backends = colab_utils.CachedBackends(
        own_cache_directory=self.create_tempdir().full_path,
        shared_cache_directory=self.create_tempdir().full_path,
    )
    backend1 = TestBackend(
        cache_filename=cached_backends.get_cache_path('model_1_0'),
        return_value='a',
    )
    backend2 = TestBackend(
        cache_filename=cached_backends.get_cache_path('model_2_0'),
        return_value='b',
    )
    cached_backends['model_1_0'] = backend1
    cached_backends['model_2_0'] = backend2

    with self.subTest('get'):
      self.assertEqual(backend1, cached_backends['model_1_0'])
      self.assertEqual(backend2, cached_backends['model_2_0'])

    with self.subTest('len'):
      self.assertLen(cached_backends, 2)

    with self.subTest('keys'):
      self.assertEqual(['model_1_0', 'model_2_0'], list(cached_backends.keys()))

    with self.subTest('values'):
      self.assertEqual([backend1, backend2], list(cached_backends.values()))

    with self.subTest('items'):
      self.assertEqual(
          [('model_1_0', backend1), ('model_2_0', backend2)],
          list(cached_backends.items()),
      )
    with self.subTest('iteration'):
      items = []
      for backend_name, backend in cached_backends.items():
        items.append((backend_name, backend))
      self.assertEqual(
          [('model_1_0', backend1), ('model_2_0', backend2)],
          items,
      )

  def test_save_and_load_caches(self):
    shared_cache_dir = self.create_tempdir()
    user1_cache_dir = self.create_tempdir()
    user2_cache_dir = self.create_tempdir()

    shared_cache_filename = os.path.join(
        shared_cache_dir.full_path, 'model_1_0.json'
    )
    user1_cache_filename = os.path.join(
        user1_cache_dir.full_path, 'model_1_0.json'
    )
    user2_cache_filename = os.path.join(
        user2_cache_dir.full_path, 'model_1_0.json'
    )

    # `CachedBackends` object for user1. When this user makes calls to the
    # backend, it will always return 'a' (unless reading from cache).
    cached_backends = colab_utils.CachedBackends(
        own_cache_directory=user1_cache_dir.full_path,
        shared_cache_directory=shared_cache_dir.full_path,
        additional_cache_directories=[user2_cache_dir.full_path],
    )
    backend1 = TestBackend(
        cache_filename=cached_backends.get_cache_path('model_1_0'),
        return_value='a',
    )
    cached_backends['model_1_0'] = backend1

    # Simulate some calls by user1. None of these should hit the cache.
    with self.subTest('user1_call_for_arg1'):
      self.assertEqual('a', asyncio.run(backend1.get_value('arg1')))

    # Save the cache to the default location (i.e., user's own cache directory).
    cached_backends.save_caches()
    with self.subTest('by_default_should_only_write_to_own_cache_dir'):
      self.assertTrue(os.path.exists(user1_cache_filename))
      self.assertFalse(os.path.exists(shared_cache_filename))
      self.assertFalse(os.path.exists(user2_cache_filename))

    # If we choose, though, we can also save the cache to the shared cache
    # directory. After this step, the two directories should have the same
    # contents.
    cached_backends.save_caches(
        cache_directory=cached_backends.shared_cache_directory
    )
    with self.subTest('can_optionally_save_to_the_shared_directory'):
      self.assertTrue(os.path.exists(shared_cache_filename))

    # Populate the cache with some more values.
    with self.subTest('user1_call_for_arg2'):
      self.assertEqual('a', asyncio.run(backend1.get_value('arg2')))

    # If we save the caches again, then the user's own cache directory should
    # get updated (to contain both 'arg1' and 'arg2'), but the shared cached
    # directory should be left untouched (i.e., with just 'arg1').
    cached_backends.save_caches()

    # `BackendCaches` object for user2. When this user makes calls to the
    # backend, it will always return 'b' (unless reading from cache).
    caches2 = colab_utils.CachedBackends(
        own_cache_directory=user2_cache_dir.full_path,
        shared_cache_directory=shared_cache_dir.full_path,
    )
    backend2 = TestBackend(
        cache_filename=caches2.get_cache_path('model_1_0'),
        return_value='b',
    )
    caches2['model_1_0'] = backend2
    caches2.load_caches()

    # Simulate some calls by user2. Arg1 should hit the cache because it was
    # saved to the shared cache directory.
    with self.subTest('user2_call_for_arg1_should_hit_cache'):
      self.assertEqual('a', asyncio.run(backend2.get_value('arg1')))

    # Arg2 should miss the cache because it was only saved to user1's personal
    # cache directory.
    with self.subTest('user2_call_for_arg2_should_miss_cache'):
      self.assertEqual('b', asyncio.run(backend2.get_value('arg2')))

    # Arg3 should miss the cache as well because it is new.
    with self.subTest('user2_call_for_arg3_should_miss_cache'):
      self.assertEqual('b', asyncio.run(backend2.get_value('arg3')))

    # Now let's save the cache for user2.
    caches2.save_caches()
    with self.subTest('user2_cache_should_be_saved_in_its_own_directory'):
      self.assertTrue(os.path.exists(user2_cache_filename))

    # If we reload the caches for user1, we should now have 3 different cache
    # files to merge together (user1's own cache, the shared cache, and user2's
    # cache as an "additional_cache_directory").
    cached_backends.load_caches()

    # Arg1 should be saved to all 3 cache directories. Shared takes precedence.
    with self.subTest('user1_2nd_call_for_arg1_should_hit_own_cache'):
      self.assertEqual('a', asyncio.run(backend1.get_value('arg1')))

    # Arg2 should be saved only to user1's own cache.
    with self.subTest('user1_2nd_call_for_arg2_should_hit_own_cache'):
      self.assertEqual('a', asyncio.run(backend1.get_value('arg2')))

    # Arg3 should be saved only to the additional cache directory, so we fall
    # back to it.
    with self.subTest('user1_2nd_call_for_arg3_should_hit_additional_cache'):
      self.assertEqual('b', asyncio.run(backend1.get_value('arg3')))

if __name__ == '__main__':
  absltest.main()
