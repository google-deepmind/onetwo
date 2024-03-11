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
from onetwo.agents import game_of_24
from onetwo.agents import iterative_thought
from onetwo.backends import test_utils
from onetwo.core import executing


# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


class GameOf24Test(absltest.TestCase):

  def test_game_of_24_iterative_thought_prompt(self):
    # Some typical Game of 24 prompt inputs.
    description = game_of_24.GAME_OF_24_DESCRIPTION
    few_shots = game_of_24.GAME_OF_24_EXAMPLES
    state = iterative_thought.IterativeThoughtState(
        inputs='8 12 4 5', updates=['thought1']
    )
    prompt = iterative_thought.IterativeThoughtPromptJ2()

    # Now we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests and simulated replies,
    # however, we will just depend on a single default reply.
    llm_backend = test_utils.LLMForTest(
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt contained the
    # expected content. (Although we don't verify all of the prompt formatting,
    # these assertions should be sufficient to catch many basic bugs where we
    # omitted a for-loop, or failed to include some of the fields due to a typo,
    # etc.)
    next_step, result = executing.run(
        prompt(description=description, few_shots=few_shots, state=state),
        enable_tracing=True,
    )
    prefix = result.stages[0].outputs['prefix']

    with self.subTest('prompt_should_contain_the_task_description'):
      self.assertIn(description, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_inputs'):
      self.assertIn(few_shots[0].inputs, prefix)
      self.assertIn(few_shots[-1].inputs, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_steps_so_far'):
      self.assertIn(few_shots[0].updates[0], prefix)
      self.assertIn(few_shots[-1].updates[-1], prefix)

    with self.subTest('prompt_should_contain_the_actual_inputs'):
      self.assertIn(state.inputs, prefix)

    if state.updates:
      with self.subTest('prompt_should_contain_the_actual_steps_so_far'):
        self.assertIn(state.updates[0], prefix)
        self.assertIn(state.updates[-1], prefix)

    with self.subTest('should_return_the_llm_reply_as_next_step'):
      self.assertEqual(DEFAULT_REPLY, next_step)

  def test_game_of_24_iterative_thought_proposer_prompt(self):
    # Some typical Game of 24 propose prompt inputs.
    description = game_of_24.GAME_OF_24_DESCRIPTION
    few_shots = game_of_24.GAME_OF_24_PROPOSER_EXAMPLES
    state = iterative_thought.IterativeThoughtState(
        inputs='8 12 4 5', updates=['thought1']
    )
    prompt = iterative_thought.IterativeThoughtProposerPromptJ2()

    # Now we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests and simulated replies,
    # however, we will just depend on a single default reply.
    llm_backend = test_utils.LLMForTest(
        default_reply='ta\ntb',
        default_score=0.0,
    )
    llm_backend.register()

    expected_next_steps = ['ta', 'tb']

    # Now we execute the prompt and verify that the prompt contained the
    # expected content. (Although we don't verify all of the prompt formatting,
    # these assertions should be sufficient to catch many basic bugs where we
    # omitted a for-loop, or failed to include some of the fields due to a typo,
    # etc.)
    next_steps, result = executing.run(
        prompt(description=description, few_shots=few_shots, state=state),
        enable_tracing=True,
    )
    prefix = result.stages[0].outputs['prefix']

    with self.subTest('prompt_should_contain_the_task_description'):
      self.assertIn(description, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_inputs'):
      self.assertIn(few_shots[0].state.inputs, prefix)
      self.assertIn(few_shots[-1].state.inputs, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_state'):
      if few_shots[0].state.updates:
        self.assertIn(few_shots[0].state.updates[0], prefix)
      if few_shots[-1].state.updates:
        self.assertIn(few_shots[-1].state.updates[-1], prefix)

    with self.subTest('prompt_should_contain_the_exemplar_next_steps'):
      self.assertIn(few_shots[0].next_steps[0], prefix)
      self.assertIn(few_shots[-1].next_steps[-1], prefix)

    with self.subTest('prompt_should_contain_the_actual_inputs'):
      self.assertIn(state.inputs, prefix)

    if state.updates:
      with self.subTest('prompt_should_contain_the_actual_steps_so_far'):
        self.assertIn(state.updates[0], prefix)
        self.assertIn(state.updates[-1], prefix)

    with self.subTest('should_return_the_parsed_llm_reply_as_next_steps'):
      self.assertEqual(expected_next_steps, next_steps)

if __name__ == '__main__':
  absltest.main()
