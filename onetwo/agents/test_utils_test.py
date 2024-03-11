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


from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized

from onetwo.agents import base as agents_base
from onetwo.agents import test_utils
from onetwo.core import executing


StringAgentState = agents_base.UpdateListState[str, str]


_SU: TypeAlias = agents_base.ScoredUpdate[str]
_ULS: TypeAlias = agents_base.UpdateListState[str, _SU]


class DistributionAgentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_end', {'hello': 1.0}, 'hello', {'$': 1.0}),
      ('impossible', {'hello': 1.0}, 'x', {'': 1.0}),
      ('single_start', {'hello': 1.0}, '', {'h': 1.0}),
      ('double_start', {'hello': 0.7, 'hallo': 0.3}, 'h', {'e': 0.7, 'a': 0.3}),
      ('double_end', {'hello': 0.7, 'hallo': 0.3}, 'hello', {'$': 1.0}),
      (
          'triple',
          {'hello': 0.1, 'helly': 0.2, 'hallo': 0.2, 'world': 0.5},
          'h',
          {'e': 0.6, 'a': 0.4},
      ),
  )
  def test_next_step_distribution(self, words, prefix, expected):
    async def wrapper(words: dict[str, float], prefix: str) -> list[_SU]:
      agent = test_utils.DistributionAgentForTest(words)
      state = await agent.initialize_state(prefix)
      return await agent.get_next_step_distribution(state)

    res = executing.run(wrapper(words, prefix))
    res_as_dict = {
        scored_update.update: round(scored_update.score, 2)
        for scored_update in res
    }
    self.assertDictEqual(expected, res_as_dict)

  @parameterized.named_parameters(
      ('start', '', 1.0),
      ('impossible', 'x', 0.0),
      ('incomplete', 'h', 0.5),
      ('hello', 'hello', 0.3),
      ('hallo', 'hallo', 0.2),
      ('hello$', 'hello$', 0.1),
  )
  def test_score_state(self, state, expected):
    score = test_utils.DistributionAgentForTest(
        {'hello': 0.1, 'hello_world': 0.2, 'hallo': 0.2, 'world': 0.5}
    ).score_state(state)
    self.assertEqual(
        expected,
        round(score, 2),
    )

  @parameterized.named_parameters(
      ('start', '', False),
      ('impossible', 'x', True),
      ('incomplete', 'h', False),
      ('hello', 'hello', False),
      ('hallo', 'hallo', False),
      ('hello$', 'hello$', True),
  )
  def test_is_finished(self, state, expected):
    self.assertEqual(
        expected,
        test_utils.DistributionAgentForTest(
            {'hello': 0.1, 'hello_world': 0.2, 'hallo': 0.2, 'world': 0.5}
        ).is_finished(state),
    )

  @parameterized.named_parameters(
      ('empty', '', ['hello$', 'hallo$', 'hello_world$', 'world$']),
      ('h', 'h', ['hello$', 'hallo$', 'hello_world$']),
      ('he', 'he', ['hello$', 'hello_world$']),
      ('hello', 'hello', ['hello$', 'hello_world$']),
      ('hello$', 'hello$', ['hello$']),
      ('x', 'x', ['x']),
  )
  def test_execute(self, prefix, expected):
    agent = test_utils.DistributionAgentForTest(
        {'hello': 0.1, 'hallo': 0.2, 'hello_world': 0.4, 'world': 0.3}
    )
    res = executing.run(agent(inputs=prefix))
    self.assertIn(res, expected)


class StringAgentTest(parameterized.TestCase):

  def test_sample_next_step(self):
    agent = test_utils.StringAgent(sequence=['a', 'b', 'c', 'd'])
    state = executing.run(agent.initialize_state(inputs='test'))
    result = executing.run(
        agent.sample_next_step(state=state, num_candidates=2)
    )
    self.assertEqual(result, ['a', 'b'])
    result = executing.run(
        agent.sample_next_step(state=state, num_candidates=2)
    )
    self.assertEqual(result, ['c', 'd'])

  def test_execute(self):
    agent = test_utils.StringAgent(
        max_length=5, sequence=['a', 'a', 'b', 'b', 'a']
    )
    result = executing.run(agent(inputs='test'))
    self.assertEqual(result, 'a a b b a')


if __name__ == '__main__':
  absltest.main()
