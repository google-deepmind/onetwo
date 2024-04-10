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

"""Interfaces and utilities for scoring, ranking and selecting updates.

These can be used in combination with agents to implement optimization
algorithms.
"""

import abc
from collections.abc import Sequence
from typing import Generic, TypeAlias, TypeVar

from onetwo.agents import agents_base
from onetwo.core import executing


# Type used to represent a state.
_S = TypeVar('_S')

# Type used to represent an incremental update of a state.
_U = TypeVar('_U')


class ScoringFunction(Generic[_S, _U], metaclass=abc.ABCMeta):
  """Interface for a scoring functions."""

  @executing.make_executable(copy_self=False)
  @abc.abstractmethod
  async def __call__(self, state: _S, update: _U) -> float:
    """Returns an absolute score of the update given the current state.

    The score represents the quality of the pair (state, update), i.e. how good
    the state after update, which means how good is the result of doing
    `state + update` (since updates can be added to states via the `+` operator
    to form new states).
    Ideally it should be comparable across different (state, update) pairs, but
    there could be cases where the starting state is identical across different
    updates (e.g. we are comparing different updates to the same state) in which
    case it is okay to return a score that does not take the state into account,
    but this is not recommended.

    Args:
      state: The current state.
      update: An incremental update of the state.

    Returns:
      A float representing the score of the state after update.
      The score is not normalized and can take any (positive or negative) value.
      Higher means better.
    """
    return 0.0


class RankingFunction(Generic[_S, _U], metaclass=abc.ABCMeta):
  """Interface for a ranking function."""

  @executing.make_executable(copy_self=False)
  @abc.abstractmethod
  async def __call__(
      self, states_and_updates: Sequence[tuple[_S, _U]]
  ) -> list[int]:
    """Ranks a list of states and updates and returns indices.

    The order is from best to worst.
    When considering a pair (state, update), the value of the pair corresponds
    to how good the state after update is, i.e. the value of `state + update`.
    The order should thus make sense even when comparing pairs with different
    starting states.

    Args:
      states_and_updates: A list of states and updates to be ranked.

    Returns:
      A list of indices corresponding to the ranking of the states and updates.
      The indices refer to the order in the input list.
    """
    return []


class SelectingFunction(Generic[_S, _U], metaclass=abc.ABCMeta):
  """Interface for a selecting function."""

  @executing.make_executable(copy_self=False)
  @abc.abstractmethod
  async def __call__(
      self, states_and_updates: Sequence[tuple[_S, _U]]
  ) -> int:
    """Returns the index of the best (state, update) pair.

    When considering a pair (state, update), the value of the pair corresponds
    to how good the state after update is, i.e. the value of `state + update`.
    The notion of "best" should thus make sense even when comparing pairs with
    different starting states.

    Args:
      states_and_updates: A list of states and updates to select from.

    Returns:
      The index of the selected pair in the input list.
    """
    return 0


def ranker_from_scorer(
    scorer: ScoringFunction[_S, _U]
) -> RankingFunction[_S, _U]:
  """Converts a ScoringFunction into a RankingFunction.

  The RankingFunction ranks the states and updates by decreasing score.

  Args:
    scorer: A ScoringFunction.

  Returns:
    A RankingFunction.
  """
  @executing.make_executable
  async def ranker(states_and_updates: Sequence[tuple[_S, _U]]) -> list[int]:
    executables = [scorer(s[0], s[1]) for s in states_and_updates]
    scores = await executing.par_iter(executables)
    sorted_updates = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )
    return [update[0] for update in sorted_updates]

  return ranker


async def _select_k_best(
    states_and_updates: Sequence[tuple[_S, _U]],
    critic: SelectingFunction[_S, _U],
    k: int,
) -> list[int]:
  """Selects the k best states and updates."""
  current_list = list(states_and_updates)
  selected = []
  while len(selected) < k and current_list:
    if len(current_list) == 1:
      # If there is only one element left, we can stop the loop.
      selected.append(0)
      break
    else:
      best_index = await critic(current_list)
      current_list.pop(best_index)
      selected.append(best_index)
  return selected


def ranker_from_selector(
    selector: SelectingFunction[_S, _U]
) -> RankingFunction[_S, _U]:
  """Converts a SelectingFunction into a RankingFunction.

  The RankingFunction repeatedly calls the SelectingFunction to select the best
  state/update pair and removes it from the list.

  Args:
    selector: A SelectingFunction.

  Returns:
    A RankingFunction.
  """

  @executing.make_executable
  async def ranker(states_and_updates: Sequence[tuple[_S, _U]]) -> list[int]:
    return await _select_k_best(
        states_and_updates, selector, len(states_and_updates)
    )

  return ranker


class ScoreFromUpdates(
    Generic[_S, _U], ScoringFunction[_S, agents_base.ScoredUpdate[_U]]
):
  """Scoring function that extracts the score directly from the update."""

  @executing.make_executable
  async def __call__(
      self, state: _S, update: agents_base.ScoredUpdate[_U]
  ) -> float:
    return update.score


# Type used to represent inputs.
_I = TypeVar('_I')

_ListOfScoredUpdates: TypeAlias = agents_base.UpdateListState[
    _I, agents_base.ScoredUpdate[_U]
]
_ScoredUpdate: TypeAlias = agents_base.ScoredUpdate[_U]


class ScoreFromUpdateList(
    Generic[_I, _U], ScoringFunction[_ListOfScoredUpdates, _ScoredUpdate]
):
  """Scoring function that extracts the score from an update list."""

  @executing.make_executable
  async def __call__(
      self, state: _ListOfScoredUpdates, update: _ScoredUpdate
  ) -> float:
    """Sums the scores of the updates in the list and the new update.

    This scoring function can be used when for example we have an agent
    that associates a score to each of the updates it performs. In this case
    the score of a state can be the sum of the scores of all the updates.
    For example if we have a sequential distribution as the agent, it may
    assign a probability to each update, which we can convert (by taking the
    log) into a score that can be added across the whole sequence.

    Args:
      state: The current state, i.e. a list of scored updates.
      update: The latest scored update.

    Returns:
      The sum of the scores of all the updates in the list and the new update.
    """
    updated_state = state + update
    score = 0.0
    for s in updated_state.updates:
      score += s.score
    return score

