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

"""Library for managing caches of stateful objects."""

from collections.abc import Callable
import copy
import dataclasses
import itertools
from typing import Generic, TypeVar

from onetwo.core import utils

# Type used to represent an object in a StatefulObjectCache.
_Obj = TypeVar('_Obj')

# Type used to represent the state of an object in a StatefulObjectCache.
_ObjState = TypeVar('_ObjState')


@dataclasses.dataclass
class StatefulObjectCache(Generic[_Obj, _ObjState]):
  """Repository of stateful objects, which are stored and retrieved by state.

  The objects (_Obj) can be arbitrary objects, i.e., not necessarily copiable or
  serializable, etc., as long as there is a way of constructing a new object
  instance for a given state on demand. The state (_ObjState) should be an
  immutable / hashable / copiable / serializable representation of the object's
  internal state.

  The canonical motivating example is where the object is a PythonSandbox, whose
  internal state can be represented in terms of the sequence of code blocks that
  have been executed on that sandbox up till now. In that case, we would have:
    _Obj=PythonSandbox
    _ObjState=tuple[str, ...]

  Attributes:
    create_object_function: Function for creating a new object corresponding to
      a given state. The cache will use this function to create new objects on
      demand whenever there is a cache miss.
    destroy_object_function: If specified, the cache will call this function on
      each object that it owns when the cache is destroyed, or when the given
      object is explicitly discarded.
  """

  create_object_function: Callable[[_ObjState], _Obj]
  destroy_object_function: Callable[[_Obj], None] | None = None

  # TODO: Configure a maximum size of the cache (e.g., `max_size` and
  # `max_size_per_state`), track the last accessed time of each object, and when
  # the cache is full, automatically discard the least recently used object.
  # Conceptually this would mean maintaining a kind of LRU cache of stateful
  # objects. Note that it may not be strightforward to use an off-the-shelf
  # LRU cache implementation like `cache_tools.LRUCache`, however, due to the
  # following requirements:
  # * We can potentially have multiple different objects corresponding to the
  #   same state (e.g., if we are caching Python sandboxes, and if we have
  #   multiple parallel execution branches that request a Python sandbox in
  #   identical state, we may need to temporarily create multiple such identical
  #   sandboxes so that each execution branch has control of its own; if in the
  #   next step those execution branches happen to each execute the same piece
  #   of code, then when the sandboxes get returned to the cache, we will end up
  #   with multiple cached sandbox instances associated with the same state).
  # * We need to automatically destroy the object in the case where it is
  #   automatically discarded by the LRU cache, but would not want to destroy it
  #   in the case where it is popped from the LRU cache in the usual way for the
  #   purpose of being lent out.
  # * Separately from the objects cached by state, we need to also keep track of
  #   the objects that are currently lent out (and which are not associated with
  #   any given state), so that they can so that we can ensure that all objects
  #   owned by the cache are cleaned up properly when the cache itself is
  #   destroyed, even if for some reason some of the objects never got re-cached
  #   after being lent out.

  # The objects that were created by the cache (plus any other objects for which
  # the cache took over ownership). The cache will control the life cycle of the
  # objects internally and will create new ones on demand using the
  # `create_object_function`.
  _objects: list[_Obj] = dataclasses.field(default_factory=list)

  # The below mapping is used for tracking which objects correspond to which
  # state. In cases where an object is in the process of being modified, it will
  # temporarily be removed from the below mapping, and then reinserted
  # (associated with its new state) once the modification is complete. Note that
  # the object being modified will continue to be listed in `_objects` above,
  # however, so as to ensure that it gets cleaned up properly when the cache
  # itself is destroyed at the end.
  _objects_by_state: dict[_ObjState, list[_Obj]] = dataclasses.field(
      default_factory=dict
  )

  async def get(self, state: _ObjState) -> _Obj:
    """Returns an object for the given state, transferring control to caller.

    Note that while we transfer temporary control of the object to the caller
    (in the sense that the caller is free to modify the object, and the cache
    will not give the object to any other caller in the meantime), the cache
    still retains long-term ownership of the object, in the sense that it will
    automatically destroy the object at the time that the cache is destroyed.
    Normally, the caller is expected to return control of the object to the
    cache by calling `cache_object` again once it is done modifying the object.

    Args:
      state: The state to which the object should correspond.
    """
    if self._objects_by_state.setdefault(state, []):
      # Object already cached. Since we will transfer control of the object to
      # the caller (who may modify it by performing more operations on it),
      # we need to remove the object from the cache, so that it is no longer
      # associated with the current state.
      return self._objects_by_state[state].pop()
    else:
      # Object not cached. Need to create a new one.
      # Note: We are assigning `self.create_object_function` to a local variable
      # here to avoid an error saying `Calling TypeGuard function
      # 'inspect.iscoroutinefunction' with an arbitrary expression not
      # supported yet [not-supported-yet] Please assign the expression to a
      # local variable.`
      create_object_function = self.create_object_function
      obj = await utils.call_and_maybe_await(create_object_function, state)
      self._objects.append(obj)
      return obj  # pytype: disable=bad-return-type

  def cache_object(self, state: _ObjState, obj: _Obj) -> None:
    """Caches the object for the given state. Caller relinquishes control.

    In the typical usage, the given object would be one that was retrieved from
    the cache earlier and which is thus already in the list of objects that the
    cache manages and owns. In this case, the call to `cache_object` indicates
    that the caller is done modifying the object, and it is now safe for the
    cache to reindex the object with its updated state and lend the object to
    other callers as needed. If the object is not already owned by the cache,
    then calling `cache_object` is understood to also transfer long-term
    ownership of the object to the cache.

    Args:
      state: The serializable representation of the object's current state.
      obj: The object that is to be cached. The understanding is that by
        caching the object, the caller relinquishes control of the object,
        which means they should not perform any additional actions on it. To
        reclaim control of the object (or of an equivalently reconstructed one),
        the caller should call `get`.
    """
    if obj not in self._objects:
      self._objects.append(obj)
    self._objects_by_state.setdefault(state, []).append(obj)

  def discard_object(self, obj: _Obj) -> None:
    """Destroys the given object and removes it from the managed list."""
    for objects in self._objects_by_state.values():
      # Note that we don't use `obj in objects` or `object.remove(obj)` here
      # because we want to check for object identity, not equality.
      for i in range(len(objects)):
        if obj is objects[i]:
          del objects[i]
          break
    if self.destroy_object_function:
      self.destroy_object_function(obj)
    self._objects.remove(obj)

  def destroy(self) -> None:
    """Destroys all objects owned by the cache. To be called on shutdown."""
    self._objects_by_state = {}
    objects = copy.copy(self._objects)
    for obj in objects:
      self.discard_object(obj)
    del objects

  def get_all_managed_objects(self) -> tuple[_Obj, ...]:
    """Returns all objects managed by the cache. Mostly for debugging."""
    return tuple(self._objects)

  def get_all_cached_objects(self) -> tuple[_Obj, ...]:
    """Returns all currently cached/usable pnkects. Mostly for debugging."""
    return tuple(
        itertools.chain.from_iterable(self._objects_by_state.values())
    )
