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

import collections
import dataclasses
import pprint
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.stdlib.tool_use import stateful_caching


@dataclasses.dataclass(frozen=True)
class _CState:
  """Simple "state" class for use in testing. Hashable version of `_C`."""
  steps: tuple[str, ...] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass
class _C:
  """Simple "stateful" class for use in testing. Not hashable."""
  steps: list[str] = dataclasses.field(default_factory=list)
  destroyed: bool = False

  def get_state(self) -> _CState:
    return _CState(steps=tuple(self.steps))


def _add_sync(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


async def _add_async(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


@executing.make_executable  # pytype: disable=wrong-arg-types
def _add_executable(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


class CacheCounters(collections.Counter[str]):
  """Counters of creation/destruction for use with caches of type `_C`."""

  def create_object(self, state: _CState) -> _C:
    self['num_created'] += 1
    return _C(steps=list(state.steps))

  def destroy_object(self, obj: _C) -> None:
    obj.destroyed = True
    self['num_destroyed'] += 1

  @property
  def num_created(self) -> int:
    return self['num_created']

  @property
  def num_destroyed(self) -> int:
    return self['num_destroyed']


class StatefulObjectCacheTest(parameterized.TestCase):

  def _assertLenCachedObjectsForState(  # pylint: disable=invalid-name
      self,
      cache: stateful_caching.StatefulObjectCache[_C, _CState],
      state: _CState,
      expected_len: int,
  ) -> None:
    """Assertion helper that prints detailed debug info."""
    self.assertLen(
        cache._objects_by_state.get(state, []), expected_len,
        msg=pprint.pformat(cache._objects_by_state),
        )

  def test_get_and_cache_object_sequential(self):
   # Start with an empty cache.
    counters = CacheCounters()
    cache = stateful_caching.StatefulObjectCache[_C, _CState](
        create_object_function=counters.create_object,
        destroy_object_function=counters.destroy_object,
    )

    # Try to get an object with empty state. Cache miss will cause a new object
    # to be created.
    state = _CState()
    obj = executing.run(cache.get(_CState()))
    with self.subTest('after_first_get_should_have_one_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 1)
    with self.subTest('after_first_get_should_have_no_cached_objects'):
      # The newly created object is not cached because it is already lent out.
      self.assertEmpty(cache.get_all_cached_objects())
      self.assertEmpty(cache._objects_by_state.get(state, []))

    # Perform one step and then recache the modified object with the new state.
    obj.steps.append('1')
    state = obj.get_state()
    cache.cache_object(state, obj)
    with self.subTest('after_recaching_should_still_have_one_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 1)
    with self.subTest('after_recaching_should_have_one_cached_object'):
      self.assertLen(cache.get_all_cached_objects(), 1)
      self._assertLenCachedObjectsForState(cache, state, 1)

    # Try to get an object from the cache with the new state (cache hit).
    obj = executing.run(cache.get(state))
    with self.subTest('after_second_get_should_still_have_one_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 1)
    with self.subTest('after_second_get_should_still_have_no_cached_objects'):
      self.assertEmpty(cache.get_all_cached_objects())
      self._assertLenCachedObjectsForState(cache, state, 0)
    with self.subTest('retrieved_object_should_have_the_specified_state'):
      self.assertEqual(state, obj.get_state())

    # Perform a second step and then recache the modified object.
    obj.steps.append('2')
    state = obj.get_state()
    cache.cache_object(state, obj)
    with self.subTest('at_end_should_still_have_one_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 1)
    with self.subTest('at_end_should_have_one_cached_object'):
      self.assertLen(cache.get_all_cached_objects(), 1)
      self._assertLenCachedObjectsForState(cache, state, 1)

    # Now destroy the cache.
    cache.destroy()
    with self.subTest('after_destroying_cache_should_have_no_managed_objects'):
      self.assertEmpty(cache.get_all_managed_objects())
    with self.subTest('after_destroying_cache_should_have_no_cached_objects'):
      self.assertEmpty(cache.get_all_cached_objects())
      self._assertLenCachedObjectsForState(cache, _CState(), 0)
    with self.subTest('should_create_one_object_in_total'):
      self.assertEqual(1, counters.num_created)
    with self.subTest('should_destroy_one_object_in_total'):
      self.assertEqual(1, counters.num_destroyed)

  def test_get_and_cache_object_for_non_sequential(self):
    # Start with an empty cache.
    counters = CacheCounters()
    cache = stateful_caching.StatefulObjectCache[_C, _CState](
        create_object_function=counters.create_object,
        destroy_object_function=counters.destroy_object,
    )

    # Try to get four different objects with empty state. Cache misses will
    # cause four new objects to be created.
    state = _CState()
    obj1 = executing.run(cache.get(state))
    obj2 = executing.run(cache.get(state))
    obj3 = executing.run(cache.get(state))
    obj4 = executing.run(cache.get(state))
    with self.subTest('after_first_gets_should_have_four_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 4)
    with self.subTest('after_first_gets_should_have_no_cached_objects'):
      self.assertEmpty(cache.get_all_cached_objects())

    # Perform some different steps and then recache the modified objects with
    # the new states:
    # * One object that stays at initial state.
    # * Two objects with same new state ['1a'].
    # * One object with a different new state ['1b'].
    obj2.steps.append('1a')
    obj3.steps.append('1a')
    obj4.steps.append('1b')
    cache.cache_object(obj1.get_state(), obj1)
    cache.cache_object(obj2.get_state(), obj2)
    cache.cache_object(obj3.get_state(), obj3)
    cache.cache_object(obj4.get_state(), obj4)
    with self.subTest('after_recaching_should_still_have_four_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 4)
    with self.subTest('after_recaching_should_have_four_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 4)
      self._assertLenCachedObjectsForState(cache, _CState(), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a',)), 2)
      self._assertLenCachedObjectsForState(cache, _CState(('1b',)), 1)

    # Get one of the objects with state '1a' and perform a second step '2a'.
    obj = executing.run(cache.get(_CState(('1a',))))
    obj.steps.append('2a')
    cache.cache_object(obj.get_state(), obj)
    with self.subTest('after_second_step_should_have_same_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 4)
    with self.subTest('after_second_step_should_have_correct_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 4)
      self._assertLenCachedObjectsForState(cache, _CState(), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a',)), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1b',)), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a', '2a')), 1)

    # Get the object with empty state and move it forward to '1a'.
    obj = executing.run(cache.get(_CState()))
    obj.steps.append('1a')
    cache.cache_object(obj.get_state(), obj)
    with self.subTest('after_third_step_should_have_same_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 4)
    with self.subTest('after_third_step_should_have_correct_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 4)
      self._assertLenCachedObjectsForState(cache, _CState(), 0)
      self._assertLenCachedObjectsForState(cache, _CState(('1a',)), 2)
      self._assertLenCachedObjectsForState(cache, _CState(('1b',)), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a', '2a')), 1)

    # Get an object with empty state (cache miss) and move it forward to '1a'.
    obj = executing.run(cache.get(_CState()))
    obj.steps.append('1a')
    cache.cache_object(obj.get_state(), obj)
    with self.subTest('after_fourth_step_should_have_another_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 5)
    with self.subTest('after_fourth_step_should_have_correct_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 5)
      self._assertLenCachedObjectsForState(cache, _CState(), 0)
      self._assertLenCachedObjectsForState(cache, _CState(('1a',)), 3)
      self._assertLenCachedObjectsForState(cache, _CState(('1b',)), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a', '2a')), 1)

    # Get an object with complex state (cache miss) and cache it as-is.
    obj = executing.run(cache.get(_CState(('1c', '2c'))))
    cache.cache_object(obj.get_state(), obj)
    with self.subTest('after_fifth_step_should_have_another_managed_object'):
      self.assertLen(cache.get_all_managed_objects(), 6)
    with self.subTest('after_fifth_step_should_have_correct_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 6)
      self._assertLenCachedObjectsForState(cache, _CState(), 0)
      self._assertLenCachedObjectsForState(cache, _CState(('1a',)), 3)
      self._assertLenCachedObjectsForState(cache, _CState(('1b',)), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1a', '2a')), 1)
      self._assertLenCachedObjectsForState(cache, _CState(('1c', '2c')), 1)

    # This time don't destroy the cache.
    with self.subTest('should_create_the_correct_number_of_objects'):
      self.assertEqual(6, counters.num_created)
    with self.subTest('should_destroy_no_objects_if_cache_not_destroyed'):
      self.assertEqual(0, counters.num_destroyed)

  def test_destroy_cache_when_objects_are_lent_out(self):
    # Start with an empty cache.
    counters = CacheCounters()
    cache = stateful_caching.StatefulObjectCache[_C, _CState](
        create_object_function=counters.create_object,
        destroy_object_function=counters.destroy_object,
    )

    # Try to get three different objects with empty state. Cache misses will
    # cause three new objects to be created. Recache only one of them.
    state = _CState()
    obj1 = executing.run(cache.get(state))
    obj2 = executing.run(cache.get(state))
    obj3 = executing.run(cache.get(state))
    cache.cache_object(obj1.get_state(), obj1)
    with self.subTest('initially_should_have_three_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 3)
    with self.subTest('initially_should_have_one_cached_object'):
      self.assertLen(cache.get_all_cached_objects(), 1)

    # Now destroy the cache and verify that all three objects were destroyed.
    cache.destroy()
    with self.subTest('destroying_cache_should_empty_the_cache'):
      self.assertEmpty(cache.get_all_managed_objects())
      self.assertEmpty(cache.get_all_cached_objects())
    with self.subTest('destroying_cache_should_destroy_all_managed_objects'):
      self.assertEqual(3, counters.num_destroyed)
      self.assertTrue(obj1.destroyed)
      self.assertTrue(obj2.destroyed)
      self.assertTrue(obj3.destroyed)

  def test_discard_object(self):
    # Start with an empty cache.
    counters = CacheCounters()
    cache = stateful_caching.StatefulObjectCache[_C, _CState](
        create_object_function=counters.create_object,
        destroy_object_function=counters.destroy_object,
    )

    # Get several objects with variou states. Recache only some of them.
    state1 = _CState()
    state2 = _CState(('x', 'y'))
    cached_obj1_1 = executing.run(cache.get(state1))
    cached_obj1_2 = executing.run(cache.get(state1))
    lent_obj1_1 = executing.run(cache.get(state1))
    lent_obj1_2 = executing.run(cache.get(state1))
    cached_obj2_1 = executing.run(cache.get(state2))
    cached_obj2_2 = executing.run(cache.get(state2))
    lent_obj2_1 = executing.run(cache.get(state2))
    lent_obj2_2 = executing.run(cache.get(state2))
    cache.cache_object(cached_obj1_1.get_state(), cached_obj1_1)
    cache.cache_object(cached_obj1_2.get_state(), cached_obj1_2)
    cache.cache_object(cached_obj2_1.get_state(), cached_obj2_1)
    cache.cache_object(cached_obj2_2.get_state(), cached_obj2_2)
    with self.subTest('initial_num_managed_objects'):
      self.assertLen(cache.get_all_managed_objects(), 8)
    with self.subTest('initial_num_cached_objects'):
      self.assertLen(cache.get_all_cached_objects(), 4)

    # Now discard one of the objects that was still lent out.
    cache.discard_object(lent_obj1_1)
    with self.subTest('discarding_lent_object_should_destroy_it'):
      self.assertEqual(1, counters.num_destroyed)
      self.assertTrue(lent_obj1_1.destroyed)
    with self.subTest('discarding_lent_object_should_unmanage_it'):
      self.assertLen(cache.get_all_managed_objects(), 7)
    with self.subTest('discarding_lent_object_should_not_affect_cached_ones'):
      self.assertLen(cache.get_all_cached_objects(), 4)

    # Now discard one of the cached objects.
    cache.discard_object(cached_obj1_1)
    with self.subTest('discarding_cache_object_should_destroy_it'):
      self.assertEqual(2, counters.num_destroyed)
      self.assertTrue(cached_obj1_1.destroyed)
    with self.subTest('discarding_cache_object_should_unmanage_it'):
      self.assertLen(cache.get_all_managed_objects(), 6)
    with self.subTest('discarding_cache_object_should_uncache_it'):
      self.assertLen(cache.get_all_cached_objects(), 3)

    # Non-discarded objects should be untouched.
    with self.subTest('non_discarded_objects_should_not_be_destroyed'):
      self.assertFalse(lent_obj1_2.destroyed)
      self.assertFalse(lent_obj2_1.destroyed)
      self.assertFalse(lent_obj2_2.destroyed)
      self.assertFalse(cached_obj1_2.destroyed)
      self.assertFalse(cached_obj2_1.destroyed)
      self.assertFalse(cached_obj2_2.destroyed)

if __name__ == '__main__':
  absltest.main()
