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

"""Library for objects representing updates of results.

When executing a long computation, we may want to monitor its progress. The
Update class and its subclasses allow one to receive periodic updates on the
progress and compose these updates into a result representing everything done so
far.

Assuming we have an iterator over updates of an appropriate type (subclass of
`Update` class with relevant implementation of `__add__` and `to_result`), we
can compose a result in the following way:

```
updates = Update()
for update in process_yielding_updates:
  updates += update  # `update` of type MyUpdate.
  print('Current result: ', updates.to_result())

print('Final result', updates.to_result())
```

Note that the + operator is overloaded, so one can also use this syntax:

```
updates = Update()
for update in process_yielding_updates:
  updates += update
  print('Current result: ', updates.to_result())
```
Or one can use the `sum` operator as well:

```
final_result = sum(process_yielding_updates, start=Update()).to_result()
```
"""

from __future__ import annotations

import dataclasses
from typing import Generic, Protocol, TypeVar, cast


_T = TypeVar('_T')


class _Addable(Protocol):
  """Protocol for objects that can be added together with a + operator."""

  def __add__(self: _T, other: _T) -> _T: ...


_Tadd = TypeVar('_Tadd', bound=_Addable)


@dataclasses.dataclass
class Update(Generic[_T]):
  """Updates that can be accumulated and used to get the result.

  Basic implementation where payload and result are of the same type.

  Attributes:
    payload: Content of the update.
  """

  payload: _T | None = None

  def __add__(self, other: Update[_T] | _T) -> Update[_T]:
    """Incorporate this update. By default we overwrite all previous updates."""
    if isinstance(other, Update):
      return other
    else:
      return Update(other)

  def to_result(self) -> _T | None:
    """Produce a final result from the accumulated updates."""
    if isinstance(self.payload, Update):
      return cast(Update[_T], self.payload).to_result()
    else:
      return self.payload

  def to_simplified_result(self) -> _T:
    """Produces a simplified result from the accumulated updates."""
    if isinstance(self.payload, Update):
      return cast(Update[_T], self.payload).to_simplified_result()
    else:
      return self.payload


@dataclasses.dataclass
class AddableUpdate(Generic[_Tadd], Update[_Tadd]):
  """Update class for adding objects together with a + operator.

  This could be used for lists, but unlike the ListUpdate the order is not
  necessarily preserved: if several processes yield such updates in parallel
  there is no guarantee that the final list will have retain the order of the
  processes for example.
  """

  payload: _Tadd | None = None

  def __add__(
      self, other: AddableUpdate[_Tadd] | _Tadd
  ) -> AddableUpdate[_Tadd]:
    """See base class."""
    if isinstance(other, AddableUpdate):
      self.payload += other.payload
    else:
      self.payload += other
    return self


@dataclasses.dataclass
class ListUpdate(Generic[_T], Update[_T]):
  """Update class for maintaining a list.

  End result is of type list[T]. Every intermediate update (payload) provides
  a list of values and indices where these values need to go in the final
  result.

  For instance, ListUpdate([('ab', 10), (Update('b'), 23)]) tells us that final
  result is of type list['str'] and that elements in the final result with
  indices 10 and 23 should be set to 'ab' and 'b' respectively.

  We can also maintain nested lists. For example:
    ListUpdate([
        (ListUpdate([32, 1]), 0),
        (ListUpdate(10, 2), 1),
    ])
  tells us that final result is of type list[list[int]] and after this update
  (assuming it is the only update we accumulated) the final result will be:
  [[None, 32], [None, None, 10]].

  Attributes:
    payload: List of tuples [value_or_update, index] that contain a value
      (possibly wrapped in an Update) together with index where this value
      should go in the final result.
  """
  payload: list[
      tuple[Update[_T] | _T, int]
  ] = dataclasses.field(default_factory=list)

  def __add__(self, other: ListUpdate[_T]) -> ListUpdate[_T]:
    """See base class."""
    for update_or_value, index in other.payload:
      # Indices accumulated so far.
      accumulated_indices = [u[1] for u in self.payload]
      if index in accumulated_indices:
        index_in_payload = [u[1] for u in self.payload].index(index)
        found_update_or_value = self.payload[index_in_payload][0]
        if isinstance(found_update_or_value, ListUpdate):
          # Nested list case.
          self.payload[index_in_payload] = (
              found_update_or_value + update_or_value, index
          )
        else:
          # If the content is not a nested list update we just use the value to
          # replace the current element.
          self.payload[index_in_payload] = (update_or_value, index)
      else:
        # If we never saw the index before, just append it to the payload.
        self.payload.append((update_or_value, index))
    return self

  def to_result(self) -> list[_T]:
    """See base class."""
    if not self.payload:
      return []
    largest_index = max([u[1] for u in self.payload])
    result = [None] * (largest_index + 1)
    for update_or_value, index in self.payload:
      if isinstance(update_or_value, Update):
        result[index] = update_or_value.to_result()
      else:
        result[index] = update_or_value
    return result

  def to_simplified_result(self) -> list[_T]:
    """See base class."""
    if not self.payload:
      return []
    largest_index = max([u[1] for u in self.payload])
    result = [None] * (largest_index + 1)
    for update_or_value, index in self.payload:
      if isinstance(update_or_value, Update):
        result[index] = update_or_value.to_simplified_result()
      else:
        result[index] = update_or_value
    return [r for r in result if r is not None]
