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
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from onetwo.agents import agents_base
from onetwo.agents import agents_test_utils
from onetwo.agents import critics
from onetwo.agents import optimization
from onetwo.core import executing

_StringAgentState: TypeAlias = agents_base.UpdateListState[str, str]
_SU: TypeAlias = agents_base.ScoredUpdate[str]
_ULS: TypeAlias = agents_base.UpdateListState[str, _SU]


@dataclasses.dataclass
class LogScoredListAgent(
    agents_base.Agent[str, str, _ULS, _SU, None]
):
  """Agent that wraps a DistributionAgentForTest to form states that are lists.

  The distribution agent returns updates with scores i.e. they are of type
  `ScoredUpdate[str]`, and the wrapper arranges all these updates into
  lists to form a state of type `UpdateListState[str, ScoredUpdate[str]]`.
  But since we assume that the inner agent returns scores that are
  probabilities, we convert those scores into log-probabilities so that they
  can be added over the updates to compute a score for the list (i.e for the
  full state).

  Attributes:
    inner_agent: The agent to be wrapped.
    sampling_is_deterministic: If True, instead of actually returning a sample
      the agent will deterministically return the top num_candidate updates
      using the inner agent's distribution.
  """

  inner_agent: agents_test_utils.DistributionAgentForTest
  sampling_is_deterministic: bool = False

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def initialize_state(
      self, inputs: str, environment: None = None
  ) -> _ULS:
    """Overridden from base class (Agent)."""
    return _ULS(
        inputs=inputs,
        updates=[
            _SU(
                update=inputs,
                score=np.log(self.inner_agent.score_state(inputs)),
            )
        ],
    )

  def extract_output(self, state: _ULS) -> str:
    """Overridden from base class (Agent)."""
    return ''.join(u.update for u in state.updates)

  def is_finished(self, state: _ULS) -> bool:
    """Overridden from base class (Agent)."""
    return self.inner_agent.is_finished(self.extract_output(state))

  @executing.make_executable(copy_self=False)
  async def sample_next_step(
      self, state: _ULS, num_candidates: int, environment: None = None
  ) -> list[_SU]:
    """Overridden from base class (Agent)."""
    if not self.sampling_is_deterministic:
      steps = await self.inner_agent.sample_next_step(  # pytype: disable=wrong-keyword-args
          state=self.extract_output(state), num_candidates=num_candidates
      )
    else:
      dist = await self.inner_agent.get_next_step_distribution(  # pytype: disable=wrong-arg-count
          self.extract_output(state)
      )
      steps = sorted(dist, key=lambda x: x.score, reverse=True)[:num_candidates]
    return [
        _SU(update=step.update, score=np.log(step.score)) for step in steps
    ]


class ScorerForTest(critics.ScoringFunction):

  @executing.make_executable(copy_self=False)
  async def __call__(self, state: _StringAgentState, update: str) -> float:
    content = ' '.join(state.updates + [update])
    # We assign a score that depends on the number of letters 'a' vs 'b'.
    score = 0.0
    for t in content:
      if t == 'a':
        score += 1.0
      elif t == 'b':
        score -= 1.0
    return score


class RankerForTest(critics.RankingFunction):

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, states_and_updates: Sequence[tuple[_StringAgentState, str]]
  ) -> Sequence[int]:
    contents = [
        ' '.join(state.updates + [update])
        for state, update in states_and_updates
    ]
    # Sort the states alphabetically.
    sorted_list = sorted(enumerate(contents), key=lambda x: x[1])
    return [i for i, _ in sorted_list]


class SelectorForTest(critics.SelectingFunction):

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, states_and_updates: Sequence[tuple[_StringAgentState, str]]
  ) -> int:
    contents = [
        ' '.join(state.updates + [update])
        for state, update in states_and_updates
    ]
    # Sort the states alphabetically.
    sorted_list = sorted(enumerate(contents), key=lambda x: x[1])
    return sorted_list[0][0]


class ResamplingAgentTest(parameterized.TestCase):

  def test_sample_next_step_with_scorer(self):
    inner_agent = agents_test_utils.StringAgent(
        max_length=3, sequence=['a', 'a', 'b', 'c', 'b', 'b', 'c', 'd', 'a']
    )
    critic = ScorerForTest()
    ranker = critics.ranker_from_scorer(critic)
    agent = optimization.ResamplingAgent(
        inner_agent,
        ranker,
        extra_sampling_factor=3,
    )
    # Since we use extra_sampling_factor=3, the samples will be
    # ['a', 'a', 'b'], ['c', 'b', 'b'], ['c', 'd', 'a']
    # The scorer prefers 'a' over 'c' or 'd' and 'c'or 'd' over 'b'.
    # At the first step, 'a' will be picked, then 'c', then 'a'.
    result = executing.run(agent(inputs='test'))  # pytype: disable=wrong-keyword-args
    self.assertEqual(result, 'a c a')

  def test_sample_next_step_with_ranker(self):
    inner_agent = agents_test_utils.StringAgent(
        max_length=3, sequence=['a', 'a', 'b', 'c', 'b', 'b', 'c', 'd', 'a']
    )
    critic = RankerForTest()
    agent = optimization.ResamplingAgent(
        inner_agent,
        critic,
        extra_sampling_factor=3,
    )
    # Since we use extra_sampling_factor=3, the samples will be
    # ['a', 'a', 'b'], ['c', 'b', 'b'], ['c', 'd', 'a']
    # The ranker will pick according to alphabetical order, so it will pick
    # 'a', then 'a b', then 'a b a'.
    result = executing.run(agent(inputs='test'))  # pytype: disable=wrong-keyword-args
    self.assertEqual(result, 'a b a')

  def test_sample_next_step_with_selector(self):
    inner_agent = agents_test_utils.StringAgent(
        max_length=3, sequence=['a', 'a', 'b', 'c', 'b', 'b', 'c', 'd', 'a']
    )
    critic = SelectorForTest()
    selector = critics.ranker_from_selector(critic)
    agent = optimization.ResamplingAgent(
        inner_agent,
        selector,
        extra_sampling_factor=3,
    )
    # Since we use extra_sampling_factor=3, the samples will be
    # ['a', 'a', 'b'], ['c', 'b', 'b'], ['c', 'd', 'a']
    # The ranker will pick according to alphabetical order, so it will pick
    # 'a', then 'a b', then 'a b a'.
    result = executing.run(agent(inputs='test'))  # pytype: disable=wrong-keyword-args
    self.assertEqual(result, 'a b a')


class BeamSearchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'no_beam_diversity',
          False,
          'a b a',
      ),
      (
          'with_beam_diversity',
          True,
          'a b a',
      ),
  )
  def test_beam_search_with_critic(self, diversify_beam, expected_result):
    critic = RankerForTest()
    inner_agent = agents_test_utils.StringAgent(
        max_length=3,
        sequence=[
            'a', 'a', 'a', 'b',
            'b', 'c', 'c', 'd',
            'a', 'b', 'b', 'c'
        ],
    )
    agent = optimization.BeamSearch(
        inner_agent, critic, beam_size=2, max_candidates=2,
        diversify_beam=diversify_beam
    )
    result = executing.run(agent(inputs='test'))  # pytype: disable=wrong-keyword-args
    self.assertEqual(expected_result, result)

  @parameterized.named_parameters(
      (
          'beam_1_max_1',
          1,
          1,
          'hello_world$',
          ['hello_world$'],
      ),
      (
          'beam_2_max_1',
          2,
          1,
          'world$',
          ['world$', 'hello_'],
      ),
      (
          'beam_2_max_2',
          3,
          2,
          'world$',
          ['world$', 'hallo$', 'hello_'],
      ),
  )
  def test_beam_search_with_deterministic_distribution(
      self,
      beam_size,
      max_candidates,
      expected_result,
      expected_final_states,
  ):
    inner_agent = agents_test_utils.DistributionAgentForTest(
        {'hello': 0.1, 'hallo': 0.25, 'hello_world': 0.2, 'world': 0.45}
    )
    # With the above distribution, we have the following conditionals:
    # For the first letter: P(h)=0.55, P(w)=0.45
    # For the second letter: P(e|h)=0.55, P(a|h)=0.45, P(o|w)=1.0
    # With determinisitc sampling with beam_size 1, we should start with
    # h and follow with a, to get 'hallo$' (score 0.3).
    # With beam size 2, we will have 'h' and 'w' at the first step, and
    # expand them to 'he', 'ha', 'wo'
    wrapper = LogScoredListAgent(inner_agent, True)
    critic = critics.ScoreFromUpdateList()
    ranker = critics.ranker_from_scorer(critic)
    agent = optimization.BeamSearch(
        wrapper, ranker, beam_size=beam_size, max_candidates=max_candidates,
        diversify_beam=True  # Useful for deterministic sampling.
    )
    result, final_state = executing.run(
        agent(inputs='', return_final_state=True)  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('produces_expected_result'):
      self.assertEqual(expected_result, result)
    with self.subTest('produces_expected_final_state'):
      self.assertLen(final_state.states, beam_size)
      outputs = [
          wrapper.extract_output(state) for state in final_state.states
      ]
      self.assertEqual(expected_final_states, outputs)

  def test_beam_search_with_non_deterministic_distribution(
      self,
  ):
    inner_agent = agents_test_utils.DistributionAgentForTest(
        {'hello': 0.05, 'hallo': 0.3, 'hello_world': 0.2, 'world': 0.45}
    )
    wrapper = LogScoredListAgent(inner_agent, False)
    critic = critics.ScoreFromUpdateList()
    ranker = critics.ranker_from_scorer(critic)
    agent = optimization.BeamSearch(
        wrapper, ranker, beam_size=2, max_candidates=2,
    )
    result = executing.run(agent(inputs=''))  # pytype: disable=wrong-keyword-args
    with self.subTest('produces_expected_result'):
      self.assertIn(result, ['hello$', 'hallo$', 'hello_world$', 'world$'])


if __name__ == '__main__':
  absltest.main()
