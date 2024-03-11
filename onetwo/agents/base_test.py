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

from collections.abc import AsyncIterator
import contextlib
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import base as agents_base
from onetwo.core import executing


# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'

StringAgentState = agents_base.UpdateListState[str, str]


@dataclasses.dataclass
class StringAgent(
    agents_base.SingleSampleAgent[str, str, StringAgentState, str, None]
):
  """Simple test agent, whose input / updates are hard-coded strings.

  The update strings are of the form '1', '2', '3', etc.

  Its output is a concatenation of the update strings, separate by space.

  Attributes:
    max_length: Maximum length of the agent's state (i.e., of its update list).
      If specified, then will finish when this length is reached. If None, then
      will by default run forever.
  """

  max_length: int | None = None

  @executing.make_executable(copy_self=False)
  async def initialize_state(
      self, inputs: str
  ) -> StringAgentState:
    return StringAgentState(inputs=inputs)

  def extract_output(self, state: StringAgentState) -> str:
    """Overridden from base class (Agent)."""
    return ' '.join(state.updates)

  def is_finished(self, state: StringAgentState) -> bool:
    """Overridden from base class (Agent)."""
    if self.max_length is not None:
      return len(state.updates) >= self.max_length
    else:
      return super().is_finished(state)

  @executing.make_executable(copy_self=False)
  async def _sample_single_next_step(
      self, state: StringAgentState, environment: None = None
  ) -> str:
    """Overridden from base class (SingleSampleAgent)."""
    return str(len(state.updates)+1)


@dataclasses.dataclass
class StringAgentWithEnvironment(
    agents_base.SingleSampleAgent[str, str, StringAgentState, str, list[str]]
):
  """Simple test agent, whose input / updates are hard-coded strings.

  The update strings are of the form '1', '2', '3', etc.

  Its output is a concatenation of the update strings, separate by space.

  Attributes:
    max_length: Maximum length of the agent's state (i.e., of its update list).
      If specified, then will finish when this length is reached. If None, then
      will by default run forever.
  """

  max_length: int | None = None

  @executing.make_executable(copy_self=False)
  async def initialize_state(
      self, inputs: str
  ) -> StringAgentState:
    return StringAgentState(inputs=inputs)

  @contextlib.asynccontextmanager
  async def start_environment(self) -> AsyncIterator[list[str]]:
    yield []

  def extract_output(self, state: StringAgentState) -> str:
    """Overridden from base class (Agent)."""
    return ' '.join(state.updates)

  def is_finished(self, state: StringAgentState) -> bool:
    """Overridden from base class (Agent)."""
    if self.max_length is not None:
      return len(state.updates) >= self.max_length
    else:
      return super().is_finished(state)

  @executing.make_executable(copy_self=False)
  async def _sample_single_next_step(
      self, state: StringAgentState, environment: list[str] | None = None
  ) -> str:
    """Overridden from base class (SingleSampleAgent)."""
    if environment is None:
      raise ValueError('Environment must be specified for this agent.')
    next_step = str(len(state.updates)+1)
    environment += next_step
    return next_step


class AgentTest(parameterized.TestCase):

  def test_stream_updates(self):
    agent = StringAgent(max_length=5)

    # First we construct an initial state from the inputs.
    state_0 = executing.run(agent.initialize_state(inputs='q'))

    with self.subTest('state_should_initially_contain_just_the_inputs'):
      self.assertEqual(StringAgentState(inputs='q', updates=[]), state_0)

    with self.subTest('is_finished_should_initially_be_false'):
      self.assertFalse(agent.is_finished(state=state_0))

    # Now we run the agent for 2 steps. (The agent will by default run forever,
    # so we need to specify either max_steps or a stop_condition.)
    # Note that when calling `Agent.stream_updates`, it is important to use
    # `executing.stream`, rather than `executing.run`, otherwise we will only
    # get the final update.
    max_steps = 2
    updates = list(
        executing.stream(
            agent.stream_updates(initial_state=state_0, max_steps=max_steps)
        )
    )

    # To get the final state, we can simply sum the updates.
    state_2 = sum(updates, state_0)

    # Now let's run for another 2 steps.
    second_round_of_updates = list(
        executing.stream(
            agent.stream_updates(initial_state=state_2, max_steps=max_steps)
        )
    )
    state_4 = sum(second_round_of_updates, state_2)

    # Now we'll request it to run for another 2 steps, but we expect it to only
    # run for 1 more step, because we will have then reached the max_length
    # configured within the agent itself.
    third_round_of_updates = list(
        executing.stream(
            agent.stream_updates(initial_state=state_4, max_steps=max_steps)
        )
    )
    state_5 = sum(third_round_of_updates, state_4)

    with self.subTest('should_run_for_the_specified_max_steps'):
      self.assertLen(updates, max_steps)

    with self.subTest('should_stream_the_sequence_of_updates'):
      self.assertSequenceEqual(['1', '2'], updates)

    with self.subTest('should_be_able_to_sum_the_updates_to_get_final_state'):
      self.assertEqual(
          StringAgentState(inputs='q', updates=['1', '2']), state_2
      )

    with self.subTest('should_do_an_additional_max_steps_in_second_call'):
      self.assertLen(second_round_of_updates, max_steps)
      self.assertLen(state_4.updates, 2 * max_steps)

    with self.subTest('correct_final_state_after_second_round_of_updates'):
      self.assertEqual(
          StringAgentState(inputs='q', updates=['1', '2', '3', '4']), state_4
      )

    with self.subTest('is_finished_should_be_false_until_reaching_max_length'):
      self.assertFalse(agent.is_finished(state=state_2))
      self.assertFalse(agent.is_finished(state=state_4))

    with self.subTest('is_finished_should_be_true_after_reaching_max_length'):
      self.assertTrue(agent.is_finished(state=state_5))

    with self.subTest('should_stop_once_is_finished_is_true'):
      self.assertLen(third_round_of_updates, 1)
      self.assertLen(state_5.updates, 5)

    with self.subTest('correct_final_state_after_second_round_of_updates'):
      self.assertEqual(
          StringAgentState(inputs='q', updates=['1', '2', '3', '4', '5']),
          state_5,
      )

  def test_stream_states(self):
    # Here we show how to stream the intermediate states of an agent.
    agent = StringAgent(max_length=5)
    state_0 = executing.run(agent.initialize_state(inputs='q'))
    max_steps = 2
    states = list(
        executing.stream(
            agent.stream_states(initial_state=state_0, max_steps=max_steps)
        )
    )

    with self.subTest('stream_should_yield_the_sequence_of_states'):
      self.assertSequenceEqual(
          [
              StringAgentState(inputs='q', updates=['1']),
              StringAgentState(inputs='q', updates=['1', '2']),
          ],
          states,
      )

    final_state = executing.run(
        agent.stream_states(initial_state=state_0, max_steps=max_steps)
    )

    with self.subTest('run_should_return_the_sequence_of_states'):
      self.assertEqual(
          StringAgentState(inputs='q', updates=['1', '2']), final_state
      )

  def test_stream_states_with_environment(self):
    agent = StringAgentWithEnvironment(max_length=5)
    state_0 = executing.run(agent.initialize_state(inputs='q'))
    max_steps = 2

    @executing.make_executable
    async def wrapper() -> tuple[list[str], list[str], list[str]]:
      async with agent.start_environment() as env:
        # Here we call `stream_states` twice with the same environment and then
        # inspect the environment at the end to illustrate how the environment
        # is stateful and can accumulate modifications over the course of its
        # lifetime.
        states_1 = []
        async for state in agent.stream_states(
            initial_state=state_0, max_steps=max_steps, environment=env
        ):
          states_1.append(state)
        states_2 = []
        async for state in agent.stream_states(
            initial_state=state_0, max_steps=max_steps, environment=env
        ):
          states_2.append(state)
        final_env = env
      return states_1, states_2, final_env

    states_1, states_2, final_env = executing.run(wrapper())

    with self.subTest('stream_should_yield_the_sequence_of_states_1st_call'):
      self.assertSequenceEqual(
          [
              StringAgentState(inputs='q', updates=['1']),
              StringAgentState(inputs='q', updates=['1', '2']),
          ],
          states_1,
      )

    with self.subTest('stream_should_yield_the_sequence_of_states_2nd_call'):
      self.assertSequenceEqual(
          [
              StringAgentState(inputs='q', updates=['1']),
              StringAgentState(inputs='q', updates=['1', '2']),
          ],
          states_2,
      )

    with self.subTest('environment_maintains_state_over_multiple_calls'):
      self.assertSequenceEqual(
          ['1', '2', '1', '2'],
          final_env,
      )

  def test_sample_next_step(self):
    agent = StringAgent()
    next_step_candidates = executing.run(
        agent.sample_next_step(
            state=StringAgentState(inputs='q', updates=['1']), num_candidates=3
        )
    )

    # StringAgentForTest proposes multiple steps by simply returning the same
    # hard-coded string N times. Note that in practice, however, for agents that
    # we want to sample multiple candidate next steps from, we will typically
    # under the hood be using an LLM with temperature > 0, in which case we
    # should get multiple different candidates.
    self.assertSequenceEqual(['2', '2', '2'], next_step_candidates)

  @parameterized.named_parameters(
      ('finish_condition_but_no_max_steps', None, 5, '1 2 3 4 5'),
      ('max_steps_but_no_finish_condition', 3, None, '1 2 3'),
      ('max_steps_binding', 3, 4, '1 2 3'),
      ('finish_condition_binding', 3, 2, '1 2'),
  )
  def test_execute(self, max_steps, max_length, expected_output):
    agent = StringAgent(max_length=max_length)
    output = executing.run(agent(inputs='q', max_steps=max_steps))
    self.assertEqual(expected_output, output)

  @parameterized.named_parameters(
      ('max_steps_binding', 1, '1'),
      ('stop_condition_binding', 3, '1 2'),
  )
  def test_execute_with_stop_condition(self, max_steps, expected_output):
    agent = StringAgent(max_length=5)
    stop_condition = lambda x: x.updates and x.updates[-1] == '2'
    output = executing.run(
        agent(inputs='q', max_steps=max_steps, stop_condition=stop_condition)
    )
    self.assertEqual(expected_output, output)

  def test_execute_return_final_state(self):
    agent = StringAgent()
    output, final_state = executing.run(
        agent(inputs='q', max_steps=3, return_final_state=True)
    )

    with self.subTest('correct_output'):
      self.assertEqual('1 2 3', output)

    with self.subTest('correct_final_state'):
      self.assertEqual(
          StringAgentState(inputs='q', updates=['1', '2', '3']),
          final_state,
      )

  def test_execute_with_environment(self):
    agent = StringAgentWithEnvironment()
    output = executing.run(agent(inputs='q', max_steps=3))

    with self.subTest('correct_output'):
      self.assertEqual('1 2 3', output)


if __name__ == '__main__':
  absltest.main()
