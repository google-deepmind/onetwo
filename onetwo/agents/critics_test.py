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

from collections.abc import Sequence
import dataclasses
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import agents_base
from onetwo.agents import critics
from onetwo.agents import distribution
from onetwo.core import executing


_SU: TypeAlias = agents_base.ScoredUpdate[str]
_ULS: TypeAlias = agents_base.UpdateListState[str, _SU]


class ScorerForTest(critics.ScoringFunction[str, float]):

  @executing.make_executable(copy_self=False)
  async def __call__(self, state: str, update: float) -> float:
    """Assigns as a score the update value."""
    return update


class RankerForTest(critics.RankingFunction[str, float]):

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, states_and_updates: Sequence[tuple[str, float]]
  ) -> Sequence[int]:
    """Returns the indices of the states sorted by the update value."""
    sorted_states_and_updates = sorted(
        enumerate(states_and_updates), key=lambda x: x[1][1], reverse=True
    )
    return [i for i, _ in sorted_states_and_updates]


class SelectorForTest(critics.SelectingFunction[str, float]):

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, states_and_updates: Sequence[tuple[str, float]]
  ) -> int:
    """Returns the index of the state with the highest update value."""
    return max(enumerate(states_and_updates), key=lambda x: x[1][1])[0]


@dataclasses.dataclass
class DistributionAgentForTest(
    distribution.DistributionAgent[
        str, str, str, agents_base.ScoredUpdate[str], None
    ]
):
  distribution: list[tuple[str, float]] = dataclasses.field(
      default_factory=list
  )

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def initialize_state(self, inputs: str) -> str:
    """Overridden from base class (Agent)."""
    return inputs

  def extract_output(self, state: str) -> str:
    """Overridden from base class (Agent)."""
    return state

  def is_finished(self, state: str) -> bool:
    """Overridden from base class (Agent)."""
    return bool(state)

  @executing.make_executable(copy_self=False)
  async def get_next_step_distribution(
      self, state: str, environment: None = None
  ) -> list[agents_base.ScoredUpdate[str]]:
    """Overridden from base class (DistributionAgent)."""
    return [
        agents_base.ScoredUpdate(update=d[0], score=d[1])
        for d in self.distribution
    ]


class CriticsTest(parameterized.TestCase):

  def test_scorer_to_ranker(self):
    scorer = ScorerForTest()
    converted_ranker = critics.ranker_from_scorer(scorer)
    ranker = RankerForTest()
    states_and_updates = [
        ('a', 1.0),
        ('b', 2.0),
        ('c', 3.0),
        ('d', 4.0),
        ('e', 5.0),
    ]
    res = executing.run(ranker(states_and_updates))  # pytype: disable=wrong-arg-count
    res2 = executing.run(converted_ranker(states_and_updates))  # pytype: disable=wrong-arg-count
    self.assertEqual(res, res2)
    self.assertEqual(res, [4, 3, 2, 1, 0])

  def test_selector_to_ranker(self):
    selector = SelectorForTest()
    converted_ranker = critics.ranker_from_selector(selector)
    ranker = RankerForTest()
    states_and_updates = [
        ('a', 1.0),
        ('b', 2.0),
        ('c', 3.0),
        ('d', 4.0),
        ('e', 5.0),
    ]
    res = executing.run(ranker(states_and_updates))  # pytype: disable=wrong-arg-count
    res2 = executing.run(converted_ranker(states_and_updates))  # pytype: disable=wrong-arg-count
    self.assertEqual(res, res2)
    self.assertEqual(res, [4, 3, 2, 1, 0])

  def test_score_from_update(self):
    """Use an agent with distribution to score its states."""
    distrib = [('a', 0.1), ('b', 0.3), ('c', 0.6)]
    dist_agent = DistributionAgentForTest(distribution=distrib)
    dist_map = {k: v for k, v in distrib}
    scoring_function = critics.ScoreFromUpdates()
    state = 'a'
    scores = []
    expected_scores = []

    async def wrapper():
      nonlocal scores
      updates = await dist_agent.sample_next_step(state=state, num_candidates=3)  # pytype: disable=wrong-keyword-args
      for update in updates:
        scores.append(
            await scoring_function(state, update)
        )
        expected_scores.append(dist_map[update.update])

    executing.run(wrapper())
    self.assertEqual(scores, expected_scores)

  def test_score_from_update_list(self):
    """Use an agent with distribution to score its states."""
    scoring_function = critics.ScoreFromUpdateList()
    state = _ULS('', [_SU('a', 0.1), _SU('b', 0.3), _SU('c', 0.6)])
    update = _SU('d', 0.1)

    res = executing.run(scoring_function(state, update))
    self.assertEqual(res, 1.1)


if __name__ == '__main__':
  absltest.main()
