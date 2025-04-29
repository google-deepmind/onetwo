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

from collections.abc import Sequence
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import agents_base
from onetwo.agents import distribution
from onetwo.core import executing


StringAgentState = agents_base.UpdateListState[str, str]


@dataclasses.dataclass
class StringDistributionAgent(
    distribution.DistributionAgent[
        str, str, StringAgentState, agents_base.ScoredUpdate[str], None
    ]
):
  """Simple test agent, whose input / updates are hard-coded strings.

  The update strings are of the form '1', '2', '3', etc.

  Its output is a concatenation of the update strings, separate by space.

  Attributes:
    max_length: Maximum length of the agent's state (i.e., of its update list).
      If specified, then will finish when this length is reached. If None, then
      will by default run forever.
  """

  end_of_sequence: int = 0
  vocab_size: int = 10
  distribution: list[tuple[str, float]] = dataclasses.field(
      default_factory=list
  )

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def initialize_state(
      self, inputs: str, environment: None = None
  ) -> StringAgentState:
    return StringAgentState(inputs=inputs)

  def extract_output(self, state: StringAgentState) -> str:
    """Overridden from base class (Agent)."""
    return ' '.join(state.updates)

  def is_finished(self, state: StringAgentState) -> bool:
    """Overridden from base class (Agent)."""
    if not state.updates:
      return False
    return state.updates[-1] == self.end_of_sequence

  @executing.make_executable(copy_self=False)
  async def get_next_step_distribution(
      self, state: StringAgentState, environment: None = None
  ) -> list[agents_base.ScoredUpdate[str]]:
    if self.distribution:
      return [
          agents_base.ScoredUpdate(update=d[0], score=d[1])
          for d in self.distribution
      ]
    else:
      # Return a uniform distribution over the vocabulary.
      return [
          agents_base.ScoredUpdate(update=str(i), score=1.0 / self.vocab_size)
          for i in range(self.vocab_size)
      ]


def _scored_updates_to_tuples(
    scored_updates: Sequence[agents_base.ScoredUpdate[str]],
) -> list[tuple[str, float]]:
  return [(s.update, s.score) for s in scored_updates]


class DistributionAgentTest(parameterized.TestCase):

  def test_get_next_step_distribution(self):
    agent = StringDistributionAgent(vocab_size=10)

    with self.subTest('distribution_is_uniform'):
      dist = executing.run(
          agent.get_next_step_distribution(  # pytype: disable=wrong-keyword-args
              state=agent.initialize_state(inputs='1')  # pytype: disable=wrong-keyword-args
          )
      )
      self.assertSequenceEqual(
          [(str(i), 0.1) for i in range(10)], _scored_updates_to_tuples(dist)
      )

    with self.subTest('samples_are_correct'):
      samples = executing.run(
          agent.sample_next_step(  # pytype: disable=wrong-keyword-args
              state=agent.initialize_state(inputs='1'), num_candidates=5  # pytype: disable=wrong-keyword-args
          )
      )
      samples = [s.update for s in samples]
      self.assertContainsSubset(set(samples), set(str(i) for i in range(10)))

    with self.subTest('samples_and_scores_are_correct'):
      samples_with_scores = executing.run(
          agent.sample_next_step(  # pytype: disable=wrong-keyword-args
              state=agent.initialize_state(inputs='1'), num_candidates=5  # pytype: disable=wrong-keyword-args
          )
      )
      self.assertContainsSubset(
          set(_scored_updates_to_tuples(samples_with_scores)),
          set((str(i), 0.1) for i in range(10)),
      )


class ReweightedDistributionAgentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('top_k', None, None, 2, [('1', 0.62), ('2', 0.0), ('3', 0.38)]),
      ('top_p', None, 0.1, None, [('1', 1.0), ('2', 0.0), ('3', 0.0)]),
      ('top_k_top_p', None, 0.1, 2, [('1', 1.0), ('2', 0.0), ('3', 0.0)]),
      ('temp_0', 0.0, None, None, [('1', 1.0), ('2', 0.0), ('3', 0.0)]),
      ('temp_1', 1.0, None, None, [('1', 0.5), ('2', 0.2), ('3', 0.3)]),
      ('temp_5', 5.0, None, None, [('1', 0.37), ('2', 0.3), ('3', 0.33)]),
      ('temp_100', 100.0, None, None, [('1', 0.33), ('2', 0.33), ('3', 0.33)]),
      ('temp_1_top_k', 1.0, None, 2, [('1', 0.63), ('2', 0.0), ('3', 0.38)]),
      ('temp_0_top_p', 0.0, 0.5, None, [('1', 1.0), ('2', 0.0), ('3', 0.0)]),
      (
          'temp_100_top_p',
          100.0,
          0.5,
          None,
          [('1', 0.5), ('2', 0.0), ('3', 0.5)],
      ),
  )
  def test_get_distribution(
      self,
      temperature: float,
      top_p: float,
      top_k: int,
      expected_distribution: list[tuple[str, float]],
  ):
    inner_agent = StringDistributionAgent(
        distribution=[('1', 0.5), ('2', 0.2), ('3', 0.3)]
    )
    outer_agent = distribution.ReweightedDistributionAgent(
        inner_agent=inner_agent,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    state = outer_agent.initialize_state('test')  # pytype: disable=wrong-arg-count
    result = executing.run(outer_agent.get_next_step_distribution(state=state))  # pytype: disable=wrong-keyword-args
    # We round off the distribution to make the comparison easier.
    result = [(s.update, round(s.score, 2)) for s in result]
    # Also to avoid issues when comparing floats, we convert everything to
    # strings.
    self.assertEqual(str(expected_distribution), str(result))


if __name__ == '__main__':
  absltest.main()
