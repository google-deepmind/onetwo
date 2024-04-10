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

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import iterative_thought
from onetwo.backends import backends_test_utils
from onetwo.core import executing

# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


class IterativeThoughtTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_state', [], 'Step #1:'),
      ('non_empty_state', ['t1', 't2'], 'Step #1: t1\nStep #2: t2\nStep #3:'),
  )
  def test_prompt(self, steps_so_far, expected_request_suffix):
    # Some minimal prompt inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtState(
            inputs='q1',
            updates=['t1.1', 't1.2'],
        ),
        iterative_thought.IterativeThoughtState(
            inputs='q2',
            updates=['t2.1'],
        ),
    ]
    prompt = iterative_thought.IterativeThoughtPromptJ2()

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request = """\
This is your task.
Input: q1
Step #1: t1.1
Input: q1
Step #1: t1.1
Step #2: t1.2
Input: q2
Step #1: t2.1
Input: qN
"""
    expected_request += expected_request_suffix
    simulated_reply = 'new thought'

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={expected_request: simulated_reply},
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    next_step = executing.run(
        prompt(
            description=description,
            few_shots=few_shots,
            state=iterative_thought.IterativeThoughtState(
                inputs='qN', updates=steps_so_far
            ),
        )
    )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_llm_reply_as_next_step'):
      self.assertEqual(simulated_reply, next_step)

  def test_stream_updates(self):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtState(
            inputs='q1',
            updates=['t1.1', 't1.2'],
        ),
    ]
    inputs = 'qN'

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': ['thought1', 'thought2', 'thought3', 'thought4']
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # First we construct an initial state from the inputs.
    agent = iterative_thought.IterativeThoughtAgent(
        description=description, few_shots=few_shots
    )

    state_0 = executing.run(agent.initialize_state(inputs=inputs))

    with self.subTest('state_should_initially_contain_just_the_inputs'):
      self.assertEqual(
          iterative_thought.IterativeThoughtState(
              inputs='qN',
              updates=[],
          ),
          state_0,
      )

    with self.subTest('is_finished_should_initially_be_false'):
      self.assertFalse(agent.is_finished(state=state_0))

    # Now we run the agent for 2 steps. (IterativeThoughtAgent will by default
    # run forever, so we need to specify either max_steps or a stop_condition.)
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

    with self.subTest('should_stop_when_reaching_max_steps'):
      self.assertLen(updates, max_steps)

    with self.subTest('should_return_the_generated_thoughts_as_the_updates'):
      self.assertSequenceEqual(['thought1', 'thought2'], updates)

    with self.subTest('should_be_able_to_sum_the_updates_to_get_final_state'):
      self.assertEqual(
          iterative_thought.IterativeThoughtState(
              inputs='qN',
              updates=['thought1', 'thought2'],
          ),
          state_2,
      )

    with self.subTest('should_do_an_additional_max_steps_in_second_call'):
      self.assertLen(second_round_of_updates, max_steps)
      self.assertLen(state_4.updates, 2 * max_steps)

    with self.subTest('correct_final_state_after_second_round_of_updates'):
      self.assertEqual(
          iterative_thought.IterativeThoughtState(
              inputs='qN',
              updates=['thought1', 'thought2', 'thought3', 'thought4'],
          ),
          state_4,
      )

    with self.subTest('is_finished_should_stay_false_indefinitely'):
      self.assertFalse(agent.is_finished(state=state_2))
      self.assertFalse(agent.is_finished(state=state_4))

  def test_sample_next_step(self):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtState(
            inputs='q1',
            updates=['t1.1', 't1.2'],
        ),
    ]
    prev_state = iterative_thought.IterativeThoughtState(
        inputs='qN', updates=['t1']
    )
    num_candidates = 3

    # IterativeThoughtAgent proposes multiple steps by simply sampling N times
    # using the same prompt. Note that while we configure the test LLM in this
    # case to return the same thought 't2' all 3 times, in practice, if
    # temperature > 0, we should get multiple different candidates.
    expected_next_step_candidates = ['t2', 't2', 't2']

    # We hard-code the LLM to return the same thought every time.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            'Step #2:$': 't2',
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = iterative_thought.IterativeThoughtAgent(
        description=description, few_shots=few_shots
    )

    next_step_candidates = executing.run(
        agent.sample_next_step(state=prev_state, num_candidates=num_candidates)
    )

    with self.subTest('should_return_the_expected_next_step_candidates'):
      self.assertSequenceEqual(
          expected_next_step_candidates, next_step_candidates
      )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

  @parameterized.named_parameters(
      (
          'max_steps_binding',
          3,
          None,
          ['thought1', 'thought2', 'thought3'],
      ),
      (
          'stop_condition_binding',
          3,
          lambda x: x.updates and x.updates[-1] == 'thought2',
          ['thought1', 'thought2'],
      ),
  )
  def test_execute(self, max_steps, stop_condition, expected_output):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtState(
            inputs='q1',
            updates=['t1.1', 't1.2'],
        ),
    ]
    inputs = 'qN'

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': ['thought1', 'thought2', 'thought3', 'thought4']
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = iterative_thought.IterativeThoughtAgent(
        description=description, few_shots=few_shots
    )

    output = executing.run(
        agent(inputs=inputs, max_steps=max_steps, stop_condition=stop_condition)
    )

    with self.subTest('correct_output'):
      self.assertEqual(expected_output, output)

  def test_execute_return_final_state(self):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtState(
            inputs='q1',
            updates=['t1.1', 't1.2'],
        ),
    ]
    inputs = 'qN'
    max_steps = 3
    expected_output = ['thought1', 'thought2', 'thought3']

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': ['thought1', 'thought2', 'thought3', 'thought4']
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = iterative_thought.IterativeThoughtAgent(
        description=description, few_shots=few_shots
    )

    output, final_state = executing.run(
        agent(inputs=inputs, max_steps=max_steps, return_final_state=True)
    )

    with self.subTest('correct_output'):
      self.assertEqual(expected_output, output)

    with self.subTest('correct_final_state'):
      self.assertEqual(
          iterative_thought.IterativeThoughtState(
              inputs='qN',
              updates=expected_output,
          ),
          final_state,
      )


class IterativeThoughtProposerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'empty_state',
          [],
          'The next step could be one of the following (one per row):\n',
      ),
      (
          'non_empty_state',
          ['t1'],
          (
              'Step #1: t1\n'
              + 'The next step could be one of the following (one per row):\n'
          ),
      ),
  )
  def test_prompt(self, steps_so_far, expected_request_suffix):
    # Some minimal prompt inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtProposerExemplar(
            state=iterative_thought.IterativeThoughtState(
                inputs='q1', updates=['t1.1', 't1.2']
            ),
            next_steps=['t1a', 't1b'],
        ),
        iterative_thought.IterativeThoughtProposerExemplar(
            state=iterative_thought.IterativeThoughtState(
                inputs='q2', updates=[]
            ),
            next_steps=['t2a', 't2b'],
        ),
    ]
    prompt = iterative_thought.IterativeThoughtProposerPromptJ2()

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request = """\
This is your task.
Input: q1
Step #1: t1.1
Step #2: t1.2
The next step could be one of the following (one per row):
t1a
t1b

Input: q2
The next step could be one of the following (one per row):
t2a
t2b

Input: qN
"""
    expected_request += expected_request_suffix
    simulated_reply = """\
ta
tb\
"""
    expected_next_steps = ['ta', 'tb']

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={expected_request: simulated_reply},
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    next_steps = executing.run(
        prompt(
            description=description,
            few_shots=few_shots,
            state=iterative_thought.IterativeThoughtState(
                inputs='qN', updates=steps_so_far
            ),
        )
    )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_parsed_llm_reply_as_next_steps'):
      self.assertEqual(expected_next_steps, next_steps)

  @parameterized.named_parameters(
      # If the LLM returns fewer than the requested number of candidates, then
      # we just use whatever the LLM returned.
      (
          'llm_returns_fewer_than_num_candidates',
          ['t1'],
          3,
          ['t2a', 't2b'],
      ),
      # If the LLM returns more than the requested number of candidates, then
      # we use just the first num_candidates.
      (
          'llm_returns_more_than_num_candidates',
          ['t1'],
          1,
          ['t2a'],
      ),
  )
  def test_sample_next_step(
      self,
      steps_so_far,
      num_candidates,
      expected_next_step_candidates,
  ):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtProposerExemplar(
            state=iterative_thought.IterativeThoughtState(
                inputs='q1', updates=['t1.1', 't1.2']
            ),
            next_steps=['t1a', 't1b'],
        ),
    ]
    prev_state = iterative_thought.IterativeThoughtState(
        inputs='qN', updates=steps_so_far
    )

    # We hard-code the LLM to return a list of two candidate thoughts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            'The next step could be one of the following': 't2a\nt2b',
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = iterative_thought.IterativeThoughtProposerAgent(
        description=description, few_shots=few_shots
    )

    next_step_candidates = executing.run(
        agent.sample_next_step(state=prev_state, num_candidates=num_candidates)
    )

    with self.subTest('should_return_the_expected_next_step_candidates'):
      self.assertSequenceEqual(
          expected_next_step_candidates, next_step_candidates
      )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

  def test_execute(self):
    # Some minimal agent configuration and inputs.
    description = 'This is your task.'
    few_shots = [
        iterative_thought.IterativeThoughtProposerExemplar(
            state=iterative_thought.IterativeThoughtState(
                inputs='q1', updates=['t1.1', 't1.2']
            ),
            next_steps=['t1a', 't1b'],
        ),
    ]
    inputs = 'qN'
    max_steps = 3

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={'': ['thought1', 'thought2', 'thought3']},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = iterative_thought.IterativeThoughtProposerAgent(
        description=description, few_shots=few_shots
    )

    output = executing.run(agent(inputs=inputs, max_steps=max_steps))

    with self.subTest('should_make_repeated_llm_calls_up_to_max_steps'):
      self.assertLen(llm_backend.prompts, max_steps)

    with self.subTest('correct_output'):
      self.assertEqual(['thought1', 'thought2', 'thought3'], output)


if __name__ == '__main__':
  absltest.main()
