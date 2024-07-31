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
import collections
from collections.abc import Sequence
import copy
import dataclasses
import os
import pprint

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import constants
from onetwo.core import executing
from onetwo.core import sampling
from onetwo.core import updating
from onetwo.core import utils


class CacheForTest(caching.SimpleCache[str]):
  """Implementation of abstract SimpleCache for tests.

  Uses a tuple (key, sampling_key) as a cache key.
  """

  def __init__(self, append_when_caching=False):
    self.contents = {}
    self.append_when_caching = append_when_caching

  def cache_value(
      self,
      key: str,
      sampling_key: str | None,
      value: str,
  ) -> None:
    if self.append_when_caching:
      if (key, sampling_key) not in self.contents:
        self.contents[(key, sampling_key)] = []
      self.contents[(key, sampling_key)].append(value)
    else:
      self.contents[(key, sampling_key)] = value

  async def get_cached_value(
      self,
      key: str,
      sampling_key: str | None,
  ) -> str | None:
    res = self.contents.get((key, sampling_key), None)
    if res is None:
      return None
    if self.append_when_caching:
      res = res[0]
    assert isinstance(res, str)  # pytype hint.
    return res


@dataclasses.dataclass
class KeyMakerForTest(caching.CacheKeyMaker):
  """Simplified version of CacheKeyMaker for tests.

  A version of CacheKeyMaker with simpler implementation of create_key.
  """

  def create_key(self, obj_with_cache, args, kwargs) -> str:
    del obj_with_cache
    if len(args) == 2:
      return args[0] + args[1]
    else:
      return args[0] + kwargs['b']


@dataclasses.dataclass
class ClassWithCachedMethods(caching.CacheEnabled[str]):
  """An implementation of abstract CacheEnabled that uses CacheForTest."""

  append_when_caching: bool = False

  def __post_init__(self):
    self._cache_handler = CacheForTest(
        append_when_caching=self.append_when_caching
    )

  @caching.cache_method(
      name='method_decorated_with_default_cache_key_maker', is_sampled=True
  )
  def method_decorated_with_default_cache_key_maker(
      self, a: str, b: str
  ) -> str:
    return a + b

  @caching.cache_method(
      name='method_decorated_with_cache_extra_replies',
      is_sampled=True,
      cache_extra_replies=True,
  )
  def method_decorated_with_cache_extra_replies(
      self, a: str, b: str
  ) -> Sequence[str]:
    return [a + b, 'extra1 ' + a + b, 'extra2 ' + a + b]

  @caching.cache_method(
      name='method_does_not_return_sequence',
      is_sampled=True,
      cache_extra_replies=True,
  )
  def method_does_not_return_sequence(
      self, a: str, b: str
  ) -> str:
    return a + b

  @caching.cache_method(
      name='method_with_default_args',
      is_sampled=True
  )
  def method_with_default_args(
      self, a: str, b: str, c: str = ''
  ) -> str:
    del c
    return a + b

  @caching.cache_method(
      name='method_decorated_with_test_cache_key_maker',
      is_sampled=True,
      cache_key_maker=KeyMakerForTest
  )
  def method_decorated_with_test_cache_key_maker(self, a: str, b: str) -> str:
    return a + b

  @caching.cache_method(
      name='method_decorated_with_hash_and_kwargs',
      is_sampled=True,
      cache_key_maker=caching.CacheKeyMaker(hashed=['a']),
  )
  def method_decorated_with_hash_and_kwargs(self, a: str, **extra) -> str:
    return a + extra['b']

  @caching.cache_method(
      name='method_with_var_positional',
      is_sampled=True,
  )
  def method_with_var_positional(self, a: str, b: str, *args) -> str:
    del args
    return a + b

  @caching.cache_method(
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(dropped=['a']),
  )
  def method_with_explicit_arg(self, a: str, b: str) -> str:
    return a + b

  @caching.cache_method(
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(dropped=['a']),
  )
  def method_with_implicit_arg(self, **kwargs) -> str:
    return kwargs['a'] + kwargs['b']

  @executing.make_executable
  @caching.cache_method(is_sampled=False)
  def method_which_returns_executable(self, text: str) -> str:
    @executing.make_executable
    def stream():
      for i in range(1, len(text)):
        yield text[:i]
      yield text + ' done'

    return stream()


class SomeClass:

  @caching.cache_method(
      name='method_not_of_cacheenabled',
      is_sampled=True,
  )
  def method_not_of_cacheenabled(self, a: str, b: str) -> str:
    del a, b
    return ''


class CacheDecorationTest(parameterized.TestCase):
  """Tests cache_method with CacheKeyMaker, SimpleCache, and CacheEnabled."""

  @parameterized.named_parameters(
      (
          'no_key_maker',
          'method_decorated_with_default_cache_key_maker',
          (
              (
                  f'{{"{constants.CACHING_FUNCTION_NAME_KEY}": '
                  '"method_decorated_with_default_cache_key_maker", '
                  '"a": "test", "b": " done"}',
                  '',
              )
          ),
      ),
      (
          'default_args_with_key_maker',
          'method_with_default_args',
          (
              (
                  f'{{"{constants.CACHING_FUNCTION_NAME_KEY}": '
                  '"method_with_default_args", '
                  '"a": "test", "b": " done", "c": ""}',
                  '',
              )
          ),
      ),
      (
          'use_KeyMakerForTest',
          'method_decorated_with_test_cache_key_maker',
          ('test done', ''),
      ),
      (
          'key_maker_with_hash_and_kwargs',
          'method_decorated_with_hash_and_kwargs',
          (
              (
                  f'{{"{constants.CACHING_FUNCTION_NAME_KEY}": '
                  '"method_decorated_with_hash_and_kwargs", "a":'
                  ' "90a3ed9e32b2aaf4c61c410eb925426119e1a9dc53d4286ade99a809",'
                  ' "b": " done"}'
              ),
              '',
          ),
      ),
  )
  def test_cache_key(self, method, expected_keys):
    backend = ClassWithCachedMethods()
    result = asyncio.run(
        getattr(ClassWithCachedMethods, method)(backend, 'test', b=' done')
    )
    handler: CacheForTest = getattr(backend, '_cache_handler')
    list_keys = list(handler.contents.keys())
    # Hint: we only have one key in the cache.
    assert len(list_keys) == 1
    keys = list_keys[0]
    self.assertEqual(keys, expected_keys)
    self.assertEqual(result, 'test done')

  def test_method_with_var_positional(self):
    backend = ClassWithCachedMethods()
    result = asyncio.run(backend.method_with_var_positional('a', 'b', 'c'))
    handler: CacheForTest = getattr(backend, '_cache_handler')
    keys = list(handler.contents.keys())
    keys = keys[0]
    self.assertEqual(
        keys,
        (
            (
                f'{{"{constants.CACHING_FUNCTION_NAME_KEY}":'
                ' "method_with_var_positional", "a": "a", "args": ["c"], "b":'
                ' "b"}'
            ),
            '',
        ),
        repr(keys),
    )
    self.assertEqual(result, 'ab')

  @parameterized.named_parameters(
      (
          'disable_cache_true',
          True,
          [],
      ),
      (
          'disable_cache_false',
          False,
          [
              'test done',
              'extra1 test done',
              'extra2 test done'
          ],
      ),
  )
  def test_cache_extra_replies(self, disable_cache, expected_cached_values):
    backend = ClassWithCachedMethods(
        append_when_caching=True, disable_caching=disable_cache
    )
    method = 'method_decorated_with_cache_extra_replies'
    result = asyncio.run(
        getattr(ClassWithCachedMethods, method)(backend, 'test', b=' done')
    )
    self.assertEqual(result, 'test done')
    # Check that two extra values were cached.
    handler: CacheForTest = getattr(backend, '_cache_handler')
    values = list(handler.contents.values())
    if not disable_cache:
      # We have two keys: first reply with sampling_key='', second with None.
      assert len(values) == 2, pprint.pformat(handler.contents)
      cached_values = values[0] + values[1]
      self.assertCountEqual(
          cached_values,
          expected_cached_values,
          pprint.pformat(cached_values)
      )
    else:
      # Make sure nothing is cached.
      assert not values, pprint.pformat(handler.contents)

  def test_does_not_return_seq_cache_extra_replies_raises_exception(self):
    backend = ClassWithCachedMethods()
    with self.assertRaisesRegex(
        ValueError,
        'Method that is decorated with cache_method(cache_extra_replies=True)*'
    ):
      _ = asyncio.run(
          backend.method_does_not_return_sequence('test', b=' done')
      )

  def test_decorate_non_method_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'Decorator @cache_method should be applied to a method*'
    ):
      @caching.cache_method(
          name='_',
          is_sampled=True,
      )
      def _(a: str, b: str) -> str:
        del a, b
        return ''

  def test_decorate_method_not_of_cacheenabled_raises_exception(self):
    backend = SomeClass()
    with self.assertRaisesRegex(ValueError, '.*inherit from CacheEnabled'):
      _ = asyncio.run(backend.method_not_of_cacheenabled('test', b=' done'))

  def test_drop_keys(self):
    backend = ClassWithCachedMethods()
    results = []
    results.append(
        asyncio.run(backend.method_with_explicit_arg(a='test', b=' done'))
    )
    results.append(
        asyncio.run(backend.method_with_explicit_arg(a='test2', b=' done'))
    )
    results.append(
        asyncio.run(backend.method_with_implicit_arg(a='test', b=' done'))
    )
    results.append(
        asyncio.run(backend.method_with_implicit_arg(a='test2', b=' done'))
    )
    handler: CacheForTest = getattr(backend, '_cache_handler')
    values = list(handler.contents.values())
    with self.subTest('ignores_a_as_cache_key'):
      self.assertListEqual(results, ['test done'] * 4)
    with self.subTest('stores_two_values'):
      self.assertListEqual(values, ['test done'] * 2)

  def test_cache_after_execution(self):
    backend = ClassWithCachedMethods()
    handler: CacheForTest = getattr(backend, '_cache_handler')
    contents = handler.contents

    # Full execution which should be handled by the make_executable decorator,
    # hence the cache_method decorator does not see an Executable as the return
    # value.
    result = executing.run(backend.method_which_returns_executable('test'))
    with self.subTest('cached_the_value'):
      self.assertEqual(result, 'test done')
      self.assertListEqual(list(contents.values()), ['test done'])

    # Iterative execution
    stream = []
    @executing.make_executable
    async def wrapper():
      nonlocal stream
      # Fill in the cache (by full execution).
      _ = await backend.method_which_returns_executable('it')
      # This will read from the cache and get directly a string.
      e = await backend.method_which_returns_executable('it').pre_execute()
      stream.append(e)
      # This will not read from the cache and thus return an Executable.
      e = await backend.method_which_returns_executable('it2').pre_execute()
      result = None
      async for result in e:
        stream.append(result)
      return result

    _ = executing.run(wrapper())
    with self.subTest('cached_the_value'):
      self.assertEqual(
          stream,
          [
              'it done',
              updating.Update(payload='i'),
              updating.Update(payload='it'),
              updating.Update(payload='it2 done'),
          ],
      )
      self.assertListEqual(
          list(contents.values()), ['test done', 'it done', 'it2 done']
      )


@batching.add_batching
@dataclasses.dataclass
class ClassCachedWithSimpleFunctionCache(caching.CacheEnabled[str]):

  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def __post_init__(self):
    self._cache_handler = caching.SimpleFunctionCache(cache_filename='test')

  @caching.cache_method(is_sampled=False)
  def method_that_may_raise_errors(
      self,
      a: str,
      raise_error: bool = False,
  ) -> str:
    if raise_error:
      raise ValueError('We raise error.')
    return f'{a} result'

  @caching.cache_method(is_sampled=False)
  def method_that_raises_keyboard_interrupt(
      self,
      a: str,
  ) -> str:
    del a
    raise KeyboardInterrupt('Interrupt execution')

  @executing.make_executable(copy_self=False)
  @caching.cache_method(is_sampled=False)
  @batching.batch_method_with_threadpool(
      batch_size=5,
      wrapper=batching.add_logging,
  )
  def method_that_is_batched(self, a: str):
    self._counters['method_that_is_batched_calls'] += 1
    return f'{a} result'


class SimpleFunctionCacheTest(parameterized.TestCase):
  """Tests SimpleFunctionCache implementation of abstract SimpleCache."""

  def setUp(self):
    super().setUp()
    # The `self.create_tempdir` method uses command line flag and such flags
    # are not marked as parsed by default when running with pytest. Marking as
    # parsed directly here to make the pytest run pass.
    flags.FLAGS.mark_as_parsed()

  def test_cache_file_path_exists_raises_exception(self):
    cache_dir = self.create_tempdir()
    cache_filename = 'my_cache.json'
    # This file has the same path as the one that `write_to_directory` uses.
    tmp_filename = os.path.join(cache_dir.full_path, cache_filename)
    _ = self.create_tempfile(tmp_filename)
    function_cache = caching.SimpleFunctionCache(
        cache_filename=tmp_filename,
        cached_value_decoder=lambda x: x,
    )
    with self.assertRaisesRegex(FileExistsError, '.*already exists.'):
      function_cache.save()

  def test_cache_and_get_cached_value(self):

    function_cache = caching.SimpleFunctionCache(
        cache_filename='my_cache',
        cached_value_decoder=lambda x: x,
    )
    sampling_key_none = None
    # New cache key.
    function_cache.cache_value('key1', sampling_key_none, 'value_1')
    # New cache key.
    function_cache.cache_value('key2', sampling_key_none, 'value_2')
    with self.subTest('cache_new_cache_key'):
      self.assertEqual(
          {
              'add_new': 2,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {}),
          function_cache._cache_data.num_used_values_by_key,
      )
    # New cache key, sample_id updates.
    function_cache.cache_value('key3', 'sampling_key_1', 'value_3')
    with self.subTest('cache_new_cache_key_with_sample_update'):
      self.assertEqual(
          {
              'add_new': 3,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {utils.get_str_hash('key3'): 1}),
          function_cache._cache_data.num_used_values_by_key,
      )
    # Matched cache key, no sampling_key, append value.
    function_cache.cache_value('key1', sampling_key_none, 'value_4')
    with self.subTest('cache_matched_cache_key'):
      self.assertEqual(
          {
              'add_new': 3,
              'add_new_sample': 1,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {utils.get_str_hash('key3'): 1}),
          function_cache._cache_data.num_used_values_by_key,
      )
    # Matched cache key, new sampling_key, append value, sample_id updates.
    function_cache.cache_value('key1', 'sampling_key_1', 'value_5')
    with self.subTest('cache_matched_cache_key_with_sample_update'):
      self.assertEqual(
          {
              'add_new': 3,
              'add_new_sample': 2,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {
              utils.get_str_hash('key3'): 1,
              utils.get_str_hash('key1'): 1,
          }),
          function_cache._cache_data.num_used_values_by_key,
      )
      # By now we have 3 values for this key.
      self.assertEqual(
          ['value_1', 'value_4', 'value_5'],
          function_cache._cache_data.values_by_key[utils.get_str_hash('key1')],
      )
    with self.subTest(
        'get_cached_existing_sampling_key_returns_not_the_last_element'
    ):
      # For the existing sampling_key we get the very first value.
      self.assertEqual(
          asyncio.run(
              function_cache.get_cached_value('key1', 'sampling_key_1')
          ),
          'value_1',
      )
      # And sample ids don't change.
      self.assertEqual(
          collections.defaultdict(int, {
              utils.get_str_hash('key3'): 1,
              utils.get_str_hash('key1'): 1,
          }),
          function_cache._cache_data.num_used_values_by_key,
      )
    with self.subTest(
        'get_cached_new_sampling_key_maps_new_sample'
    ):
      # For the new sampling_key we get the second value mapped.
      self.assertEqual(
          asyncio.run(
              function_cache.get_cached_value('key1', 'sampling_key_2')
          ),
          'value_4',
      )
      # And sampling ids change.
      self.assertEqual(
          collections.defaultdict(int, {
              utils.get_str_hash('key3'): 1,
              utils.get_str_hash('key1'): 2,
          }),
          function_cache._cache_data.num_used_values_by_key,
      )
    # Matched cache key, matched sampling_key, matched value. Redundant.
    function_cache.cache_value('key1', 'sampling_key_1', 'value_1')
    with self.subTest(
        'cache_matched_cache_key_matched_sampling_key_same_value'
    ):
      self.assertEqual(
          {
              'add_new': 3,
              'add_new_sample': 2,
              'add_redundant': 1,
              'get_hit': 2,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {
              utils.get_str_hash('key3'): 1,
              utils.get_str_hash('key1'): 2,
          }),
          function_cache._cache_data.num_used_values_by_key,
      )
    # Matched cache key, matched sampling_key, new value. Overwrite.
    function_cache.cache_value('key1', 'sampling_key_1', 'value_2')
    with self.subTest(
        'cache_matched_cache_key_matched_sampling_key_new_value'
    ):
      self.assertEqual(
          {
              'add_new': 3,
              'add_new_sample': 2,
              'add_redundant': 1,
              'add_overwrote': 1,
              'get_hit': 2,
          },
          function_cache._cache_data.counters,
      )
      self.assertEqual(
          collections.defaultdict(int, {
              utils.get_str_hash('key3'): 1,
              utils.get_str_hash('key1'): 2,
          }),
          function_cache._cache_data.num_used_values_by_key,
      )
    # Deterministic function.
    function_cache.cache_value('key_det', sampling_key_none, 'value_1')
    with self.subTest(
        'get_cached_deterministic_one_value_no_sampling_key'
    ):
      self.assertEqual(
          asyncio.run(
              function_cache.get_cached_value('key_det', sampling_key_none)
          ),
          'value_1',
      )

  def test_decorated_method_raises_error(self):
    backend = ClassCachedWithSimpleFunctionCache()
    with self.subTest('cache_decorator_properly_handles_exceptions'):
      with self.assertRaisesRegex(ValueError, 'We raise error*'):
        _ = asyncio.run(
            backend.method_that_may_raise_errors(a='some', raise_error=True)
        )
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.subTest('should_clear_calls_in_progress'):
      self.assertEqual(handler._calls_in_progress, set())

  def test_keyboard_interrupt(self):
    # Note that KeyboardInterrupt is not a subclass of Exception, so additional
    # care is needed to make sure it is handled.
    backend = ClassCachedWithSimpleFunctionCache()
    with self.subTest('cache_decorator_properly_handles_exceptions'):
      with self.assertRaises(KeyboardInterrupt):
        _ = asyncio.run(
            backend.method_that_raises_keyboard_interrupt(a='some')
        )
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.subTest('should_clear_calls_in_progress'):
      self.assertEqual(handler._calls_in_progress, set())

  def test_does_not_repeat_calls_in_progress(self):
    backend = ClassCachedWithSimpleFunctionCache()

    exectuable = backend.method_that_is_batched(a='some')
    executables = sampling.repeat(exectuable, 5)
    result = executing.run(executing.par_iter(executables))

    with self.subTest('result_is_correct'):
      self.assertEqual(result, ['some result'] * 5)

    with self.subTest('called_method_only_once'):
      self.assertDictEqual(
          backend._counters,
          {
              'method_that_is_batched_batches': 1,
              'method_that_is_batched_calls': 1,
          },
      )

  def test_write_to_and_load_from_disk(self):
    cache_filename = 'my_cache.json'
    cache_dir = self.create_tempdir()
    cache_filename = os.path.join(cache_dir.full_path, cache_filename)
    function_cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    sampling_key_none = None
    function_cache.cache_value('key1', sampling_key_none, 'value_1')
    function_cache.cache_value('key1', sampling_key_none, 'value_2')
    function_cache.cache_value('key1', sampling_key_none, 'value_3')
    function_cache.cache_value('key1', 'sampling_key_1', 'value_4')
    function_cache.cache_value('key1', 'sampling_key_2', 'value_5')
    _ = asyncio.run(function_cache.get_cached_value('key1', 'sampling_key_3'))
    function_cache.cache_value('key2', sampling_key_none, 'value_6')
    function_cache.cache_value('key2', 'sampling_key_4', 'value_7')
    function_cache.cache_value('key3', 'sampling_key_5', 'value_8')
    function_cache.save()
    with self.subTest('cache_file_exists'):
      self.assertTrue(os.path.exists(cache_filename))

    # Cache with restored sample id mappings.
    cache_1 = caching.SimpleFunctionCache(cache_filename=cache_filename)
    cache_1.load(restore_mapping=True)
    with self.subTest('cache_restored_properly_with_sample_mapping'):
      self.assertEqual(
          asyncio.run(cache_1.get_cached_value('key1', 'sampling_key_2')),
          'value_2',
      )
      self.assertEqual(
          asyncio.run(cache_1.get_cached_value('key1', 'sampling_key_1')),
          'value_1',
      )

    # Cache with fresh sample id mappings.
    cache_2 = caching.SimpleFunctionCache(cache_filename=cache_filename)
    cache_2.load(restore_mapping=False)
    with self.subTest('cache_restored_properly_with_fresh_sample_mapping'):
      self.assertEqual(
          asyncio.run(cache_2.get_cached_value('key1', 'sampling_key_2')),
          'value_1',
      )
      self.assertEqual(
          asyncio.run(cache_2.get_cached_value('key1', 'sampling_key_1')),
          'value_2',
      )

  def test_write_to_specified_location(self):
    # Create a cache with some data (not yet saved to disk).
    cache_dir = self.create_tempdir()
    cache_filename1 = os.path.join(cache_dir.full_path, 'cache1.json')
    cache_filename2 = os.path.join(cache_dir.full_path, 'cache2.json')
    cache1 = caching.SimpleFunctionCache(cache_filename=cache_filename1)
    cache1.cache_value('key1', 'sampling_key_1', 'val1')
    cache1.cache_value('key2', 'sampling_key_1', 'val2')

    # Save the cache to a different location from the one specified at creation
    # time. The write to the secondary location but not to the default location.
    cache1.save(cache_filename=cache_filename2)
    with self.subTest('save_with_arg_should_write_to_the_specified_location'):
      self.assertTrue(os.path.exists(cache_filename2))
    with self.subTest('save_with_arg_should_not_write_to_the_default_location'):
      self.assertFalse(os.path.exists(cache_filename1))

    # Now save the cache to the default location. Should write to the default
    # location but not to the secondary location.
    cache1.cache_value('key3', 'sampling_key_1', 'val3')
    cache1.save()
    with self.subTest('save_without_arg_should_write_to_the_default_location'):
      self.assertTrue(os.path.exists(cache_filename1))

    # Now restore the cache from each location to verify the contents.
    cache_restored_1 = caching.SimpleFunctionCache(
        cache_filename=cache_filename1)
    cache_restored_1.load(restore_mapping=True)
    cache_restored_2 = caching.SimpleFunctionCache(
        cache_filename=cache_filename2)
    cache_restored_2.load(restore_mapping=True)

    with self.subTest('restore_from_specified_location_should_work'):
      self.assertLen(cache_restored_2._cache_data.values_by_key.values(), 2)
    with self.subTest('restore_from_default_location_should_work'):
      self.assertLen(cache_restored_1._cache_data.values_by_key.values(), 3)

  def test_load_and_merge_two_cache_files(self):
    # This test follows a similar pattern as `CacheDataTest.test_iadd`,
    # but the cache files are used instead of the `_CacheData` objects.
    cache_dir = self.create_tempdir()

    cache_filename1 = os.path.join(cache_dir.full_path, 'cache1.json')
    cache1 = caching.SimpleFunctionCache(cache_filename=cache_filename1)
    cache1.cache_value('key1', 'sampling_key_1', 'val1')
    cache1.cache_value('key1', 'sampling_key_2', 'val2')
    cache1.cache_value('key2', 'sampling_key_1', 'val3')
    cache1.cache_value('key2', 'sampling_key_2', 'val4')
    cache1.cache_value('key3', 'sampling_key_1', 'val5')
    cache1.cache_value('key3', 'sampling_key_2', 'val6')
    cache1.save()

    # Note that the sampling keys from cache2 are expected to be ignored when
    # loading from the second cache file. We verify this by changing or
    # shuffling the sampling keys in some of the cases.
    cache_filename2 = os.path.join(cache_dir.full_path, 'cache2.json')
    cache2 = caching.SimpleFunctionCache(cache_filename=cache_filename2)
    cache2.cache_value('key1', 'sampling_key_1', 'val7')
    cache2.cache_value('key1', 'sampling_key_4', 'val8')
    cache2.cache_value('key1', 'sampling_key_2', 'val9')
    cache2.cache_value('key1', 'sampling_key_3', 'val10')
    cache2.cache_value('key2', 'other_sampling_key_1', 'val11')
    cache2.cache_value('key4', 'sampling_key_1', 'val12')
    cache2.cache_value('key4', 'sampling_key_2', 'val13')
    cache2.save()

    # We restore the sample id mappings only when loading from the first cache
    # file (when `overwrite=True`), not the second cache file (when
    # `overwrite=False`), which leads to sampling_key=`None`.
    expected_merged_cache = caching.SimpleFunctionCache(
        cache_filename=cache_filename1)
    expected_merged_cache.cache_value('key1', 'sampling_key_1', 'val1')
    expected_merged_cache.cache_value('key1', 'sampling_key_2', 'val2')
    expected_merged_cache.cache_value('key1', None, 'val9')
    expected_merged_cache.cache_value('key1', None, 'val10')
    expected_merged_cache.cache_value('key2', 'sampling_key_1', 'val3')
    expected_merged_cache.cache_value('key2', 'sampling_key_2', 'val4')
    expected_merged_cache.cache_value('key3', 'sampling_key_1', 'val5')
    expected_merged_cache.cache_value('key3', 'sampling_key_2', 'val6')
    expected_merged_cache.cache_value('key4', None, 'val12')
    expected_merged_cache.cache_value('key4', None, 'val13')

    overwritten_cache = caching.SimpleFunctionCache(
        cache_filename=cache_filename1)
    overwritten_cache.load(restore_mapping=True)
    overwritten_cache.load(overwrite=True, cache_filename=cache_filename2)

    merged_cache = caching.SimpleFunctionCache(cache_filename=cache_filename1)
    merged_cache.load(restore_mapping=True)
    merged_cache.load(overwrite=False, cache_filename=cache_filename2)

    with self.subTest('overwrites_cache_when_overwrite_is_true'):
      self.assertEqual(
          cache2._cache_data.values_by_key,
          overwritten_cache._cache_data.values_by_key,
          f'\nActual: {overwritten_cache._cache_data.values_by_key}',
      )

    with self.subTest('merges_caches_when_overwrite_is_false'):
      self.assertEqual(
          expected_merged_cache._cache_data.values_by_key,
          merged_cache._cache_data.values_by_key,
          f'\nActual: {merged_cache._cache_data.values_by_key}',
      )


class CacheDataTest(parameterized.TestCase):
  """Tests _CacheData class."""

  def test_tuples_preserved_when_encoding_and_decoding_values_by_key(self):
    cache_data = caching._CacheData()

    values_by_key = {
        'key1': ['val1', 'val2'],
        'key2': [['val3', 'val4'], ['val5', 'val6']],
        'key3': [('val7', 'val8'), ('val9', 'val10')],
        'key4': [{'a': 'b'}, {'c': 'd'}],
        'key5': [{'a': ['val11', 'val12']}, {'b': ['val11', 'val12']}],
        'key6': [{'a': ('val11', 'val12')}, {'b': ('val11', 'val12')}],
        'key7': [('val7', ('a', 'b')), ('val9', ('a', 'b'))],
    }
    cache_data.values_by_key = copy.deepcopy(values_by_key)
    json_serialized = cache_data.to_json()
    decoded = caching._CacheData.from_json(
        json_serialized,
        infer_missing=True,
    )
    self.assertEqual(decoded.values_by_key, values_by_key)

  def test_iadd(self):
    cache1 = caching._CacheData(
        values_by_key={
            'key1': ['val1', 'val2'],
            'key2': ['val3', 'val4'],
            'key3': ['val5', 'val6'],
        }
    )
    cache2 = caching._CacheData(
        values_by_key={
            'key1': ['val7', 'val8', 'val9', 'val10'],
            'key2': ['val11'],
            'key4': ['val12', 'val13'],
        }
    )
    expected_result = caching._CacheData(
        values_by_key={
            'key1': ['val1', 'val2', 'val9', 'val10'],
            'key2': ['val3', 'val4'],
            'key3': ['val5', 'val6'],
            'key4': ['val12', 'val13'],
        }
    )
    cache1 += cache2
    self.assertEqual(expected_result, cache1, cache1)

if __name__ == '__main__':
  absltest.main()
