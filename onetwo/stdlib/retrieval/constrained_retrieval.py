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

"""Classes and protocols for constrained retrieval.

Standard retrieval typically performs a broad semantic search across the entire
index. In contrast, 'Constrained Retrieval' applies structured filters to
narrow the search space **before** the retrieval or ranking process begins.

By defining constraints upfront, the system can exclude irrelevant documents
at the index level. This 'pre-filtering' approach is more efficient than
retrieving a large set of results and filtering them afterward, as it
significantly reduces the computational overhead of semantic similarity
calculations.

This module provides the schema for defining these filters (Constraints) and
the interface for Retrievers that are capable of applying them before the
retrieval process.

An example use case could be: Given an index comprising news articles about
technology, travel and sports, one might want to filter the search of relevant
articles to "sports" if the query is "When did Lionel Messi win his last
Ballon d'or?"
"""

import abc
from collections.abc import Iterable
import dataclasses
import enum
from typing import Any, Protocol
from onetwo.core import executing
from onetwo.stdlib.retrieval import retrieval

QueryT = retrieval.QueryT
RetrievalResultT = retrieval.RetrievalResultT
DocT = retrieval.DocT


class RetrievalConstraintType(enum.Enum):
  """Defines the logic used to evaluate document fields against constraints.

  Attributes:
    EQUALS: The document field value must exactly match the provided value.
    LIST_CONTAINS_ANY: Used when a document field contains a list of values.
      The constraint is satisfied if at least one of the provided values
      is present in that list.
  """
  EQUALS = 'equals'
  LIST_CONTAINS_ANY = 'list_contains_any'


@dataclasses.dataclass(kw_only=True)
class RetrievalConstraint:
  """A single pre-filter rule used to prune the retrieval search space.

  Each constraint defines a requirement that a document must satisfy to be
  eligible for retrieval. These rules are usually applied to the document's
  metadata fields.

  Attributes:
    field_name: The identifier that corresponds to a value extracted or derived
      from each document, which the retrieval system uses for filtering.
    constraint_type: The type of comparison to perform, as defined by the
      `RetrievalConstraintType` enum.
    value: The value to use for the comparison. The expected type and structure
      of this value depend on the `constraint_type`.
  """

  field_name: str
  constraint_type: RetrievalConstraintType
  value: Any


@dataclasses.dataclass(kw_only=True)
class RetrievalConstraints:
  """A collection of pre-filter rules that define the active search space.

  Documents must satisfy **ALL** individual constraints in this collection
  (logical AND) to be visible to the retriever. If a document fails any
  constraint, it is ignored during the semantic search phase.

  Attributes:
    constraints: A list of RetrievalConstraint objects. The retrieved documents
      must satisfy ALL constraints in this list (implicit AND).
  """

  constraints: list[RetrievalConstraint] = dataclasses.field(
      default_factory=list
  )


def get_indices_for_constraint(
    constraint: RetrievalConstraint,
    value_to_id_map: dict[Any, list[int]],
) -> set[int]:
  """Returns the set of IDs satisfying a single retrieval constraint.

  Args:
    constraint: The constraint to evaluate.
    value_to_id_map: A dictionary mapping discrete field values to the list of
      IDs associated with that value. Example: {'red': [1, 101], 'blue': [100]}
  """
  c_type = constraint.constraint_type
  c_value = constraint.value
  matching_indices = set()

  if c_type == RetrievalConstraintType.EQUALS:
    matching_indices.update(value_to_id_map.get(c_value, []))
  elif c_type == RetrievalConstraintType.LIST_CONTAINS_ANY:
    if not isinstance(c_value, list):
      raise ValueError(f'Value for LIST_CONTAINS_ANY must be a list: {c_value}')
    for val in c_value:
      matching_indices.update(value_to_id_map.get(val, []))
  else:
    raise ValueError(f'Unhandled constraint type {c_type}')
  return matching_indices


def get_candidate_indices(
    constraints: RetrievalConstraints | None,
    field_to_val_id_map: dict[str, dict[Any, list[int]]],
) -> list[int] | None:
  """Returns a sorted list of IDs satisfying all constraints (AND logic).

  Args:
    constraints: The collection of constraints to apply.
    field_to_val_id_map: A nested dictionary where the outer key is the field
      name and the inner dictionary is the mapping of values to IDs. Example:
      {'color': {'red': [101, 102], 'blue': [103]}, 'size': {'S': [101, 103]}}
  """
  if not constraints or not constraints.constraints:
    return None

  candidate_set: set[int] | None = None

  for constraint in constraints.constraints:
    field_name = constraint.field_name
    if field_name not in field_to_val_id_map:
      print(f"Warning: Constraint on unknown field '{field_name}'")
      return []

    val_to_id_map = field_to_val_id_map.get(field_name)
    current_matches = get_indices_for_constraint(constraint, val_to_id_map)

    if candidate_set is None:
      candidate_set = current_matches
    else:
      candidate_set.intersection_update(current_matches)
      if not candidate_set:
        return []

  return sorted(list(candidate_set))


class ConstrainedRetriever(
    retrieval.Retriever[QueryT, RetrievalResultT],
    Protocol[QueryT, RetrievalResultT],
):
  """Interface for Retrievers that support constraint-based filtering."""

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: RetrievalConstraints | None = None,
      **kwargs,
  ) -> Iterable[RetrievalResultT]:
    """Retrieves results based on the query, filtered by constraints."""
    pass

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: RetrievalConstraints | None = None,
      **kwargs,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Retrieves results with scores, filtered by constraints."""
    pass
