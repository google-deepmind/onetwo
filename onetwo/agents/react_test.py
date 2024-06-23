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

import functools

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import react
from onetwo.backends import backends_test_utils
from onetwo.core import executing
from onetwo.stdlib.code_execution import python_execution_safe_subset
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


def _get_environment_config_with_python() -> (
    python_tool_use.PythonToolUseEnvironmentConfig
):
  """Returns an environment config with a Python tool."""
  return python_tool_use.PythonToolUseEnvironmentConfig(
      tools=[
          llm_tool_use.Tool(
              name='tool_code',
              function=python_execution_safe_subset.arithmetic_eval,
              description='Evaluates python code.',
              example='tool_code("1 + 1") returns `2`.',
              color='magenta',
          )
      ]
  )


class ReactTest(parameterized.TestCase):

  def test_prompt(self):
    # Some typical ReAct prompt inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'
    exemplars = react.REACT_FEWSHOTS
    stop_prefix = ''
    stop_sequences = ['[Question]', '[Observe]']

    # Tool configuration.
    config = _get_environment_config_with_python()

    # Prompt configuration.
    prompt = react.ReActPromptJ2()

    # Current state of agent.
    state = react.ReActState(
        inputs=question,
        updates=[
            react.ReActStep(
                is_finished=False,
                thought=(
                    'First we need to find out the population of Tuebingen and'
                    ' Zuerich. We can use the Search tool for that.'
                ),
                action=llm_tool_use.FunctionCall(
                    function_name='Search',
                    args=('population of Tuebingen',),
                    kwargs={},
                ),
                observation=[
                    {
                        'County': 'Tübingen',
                        'Name': 'Tübingen',
                        'Population Estimate 2021-12-31': '91,877',
                        'index': 1,
                    },
                    {
                        'County': '',
                        'Name': (
                            'Tübingen 91,877 Population [2021] - Estimate 108.1'
                            ' km2 Area 850.2/km2 Population Density [2021] 1.0%'
                            ' Annual Population Change [2011 → 2021]'
                        ),
                        'Population Estimate 2021-12-31': '',
                        'index': 2,
                    },
                ],
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name='Search',
                    args=('population of Zuerich',),
                    kwargs={},
                ),
                observation='402,762 (2017)',
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    )

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_reply = (
        '[Thought]: Now we need to add the populations of Tuebingen and'
        ' Zuerich. We can use the Python tool for that.\n[Act]:'
        " `tool_code('91877 + 402762')`\nBla bla"
    )
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={'': [llm_reply]},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt contained the
    # expected content. (Although we don't verify all of the prompt formatting,
    # these assertions should be sufficient to catch many basic bugs where we
    # omitted a for-loop, or failed to include some of the fields due to a typo,
    # etc.)
    prompt_outputs, result = executing.run(
        prompt(
            tools=config.tools,
            exemplars=exemplars,
            stop_prefix=stop_prefix,
            stop_sequences=stop_sequences,
            state=state,
            force_finish=False,
        ),
        enable_tracing=True,
    )
    prefix = result.stages[0].outputs['prefix']

    with self.subTest('prompt_should_contain_the_tool_descriptions'):
      self.assertIn(config.tools[0].description, prefix)
      self.assertIn(config.tools[-1].description, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_inputs'):
      self.assertIn(exemplars[0].inputs, prefix)
      self.assertIn(exemplars[-1].inputs, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_steps'):
      self.assertIn(exemplars[0].updates[0].thought, prefix)
      self.assertIn(exemplars[-1].updates[-1].thought, prefix)

    with self.subTest('prompt_should_contain_the_actual_inputs'):
      self.assertIn(state.inputs, prefix)

    if state.updates:
      with self.subTest('prompt_should_contain_the_actual_thoughts_so_far'):
        self.assertIn(state.updates[0].thought, prefix)
        self.assertIn(state.updates[-1].thought, prefix)

      with self.subTest('prompt_should_contain_the_actual_actions_so_far'):
        if state.updates[0].action:
          self.assertIn(state.updates[0].action.function_name, prefix)
        if state.updates[-1].action:
          self.assertIn(state.updates[-1].action.function_name, prefix)

    with self.subTest('should_return_the_llm_reply'):
      self.assertEqual(llm_reply, prompt_outputs)

  def test_prompt_error(self):
    # Some typical inputs / configs. (As in the other test above.)
    question = 'What is the total population of Tuebingen and Zuerich?'
    exemplars = react.REACT_FEWSHOTS
    stop_prefix = ''
    stop_sequences = ['[Question]', '[Observe]']
    config = _get_environment_config_with_python()
    prompt = react.ReActPromptJ2()
    state = react.ReActState(inputs=question, updates=[])

    # We configure the LLM to raise an exception to verify how it is handled.
    error_message = 'Some error.'
    llm_backend = backends_test_utils.LLMForTest(
        default_reply=ValueError(error_message),
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt contained the
    # expected content. (Although we don't verify all of the prompt formatting,
    # these assertions should be sufficient to catch many basic bugs where we
    # omitted a for-loop, or failed to include some of the fields due to a typo,
    # etc.)
    prompt_outputs, result = executing.run(
        prompt(
            tools=config.tools,
            exemplars=exemplars,
            stop_prefix=stop_prefix,
            stop_sequences=stop_sequences,
            state=state,
            force_finish=False,
        ),
        enable_tracing=True,
    )
    prefix = result.stages[0].outputs['prefix']

    with self.subTest('prompt_should_contain_the_error_message'):
      self.assertIn(error_message, prefix)

    with self.subTest('should_return_the_llm_reply'):
      self.assertEqual(f'#ERROR#: {error_message}', prompt_outputs)

  @parameterized.named_parameters(
      (
          'call_arbitrary_function',
          "[Act]: some_tool('a', b=1)",
          {},
          react.ReActStep(
              is_finished=False,
              action=llm_tool_use.FunctionCall(
                  function_name='some_tool',
                  args=('a',),
                  kwargs={'b': 1},
              ),
              fmt=llm_tool_use.ArgumentFormat.PYTHON,
          ),
      ),
      (
          'call_finish_function_uppercase',
          "[Act]: Finish('final_answer')",
          {},
          react.ReActStep(
              is_finished=True,
              action=llm_tool_use.FunctionCall(
                  function_name='Finish',
                  args=('final_answer',),
              ),
              fmt=llm_tool_use.ArgumentFormat.PYTHON,
          ),
      ),
      (
          'call_finish_function_lowercase',
          "[Act]: finish('final_answer')",
          {},
          react.ReActStep(
              is_finished=True,
              action=llm_tool_use.FunctionCall(
                  function_name='finish',
                  args=('final_answer',),
              ),
              fmt=llm_tool_use.ArgumentFormat.PYTHON,
          ),
      ),
      (
          'finish_default_final_stop_sequence',
          '[Finish]: These are some answers:\n\n- Option one\n- Option two',
          {},
          react.ReActStep(
              is_finished=True, observation='These are some answers:'
          ),
      ),
      (
          'finish_no_final_stop_sequence',
          '[Finish]: These are some answers:\n\n- Option one\n- Option two',
          {'final_stop_sequence': None},
          react.ReActStep(
              is_finished=True,
              observation=(
                  'These are some answers:\n\n- Option one\n- Option two'
              ),
          ),
      ),
  )
  def test_react_parse(self, reply_text, parse_args, expected_result):
    agent = react.ReActAgent(
        parse=functools.partial(react.react_parse, **parse_args)
    )
    actual_result = agent.parse(reply_text)
    self.assertEqual(expected_result, actual_result, actual_result)

  def test_sample_next_step(self):
    # Some minimal agent configuration and inputs.
    question = 'What is larger, 10 or 15, and by how much?'
    config = _get_environment_config_with_python()
    prev_state = react.ReActState(inputs=question, updates=[])

    # We hard-code the LLM to return the same thought every time (assuming the
    # prompt is of the expected form).
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'What is larger, 10 or 15, and by how much\?$': (
                '[Thought]: We can use the Python tool to subtract one from '
                'another.\n[Act]: tool_code("10 - 15")\n[Observe]: Bla bla bla.'
            ),
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = react.ReActAgent(
        exemplars=react.REACT_FEWSHOTS,
        environment_config=config,
        max_steps=1,
        stop_prefix='',
    )

    num_candidates = 2
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      next_step_candidates = executing.run(
          agent.sample_next_step(
              state=prev_state, num_candidates=num_candidates, environment=env
          )
      )

    expected_next_step_candidate = react.ReActStep(
        is_finished=False,
        thought='We can use the Python tool to subtract one from another.',
        action=llm_tool_use.FunctionCall(
            function_name='tool_code', args=('10 - 15',), kwargs={}
        ),
        observation=-5,
        fmt=llm_tool_use.ArgumentFormat.PYTHON,
    )

    with self.subTest('should_return_the_expected_number_of_candidates'):
      self.assertLen(next_step_candidates, num_candidates)

    with self.subTest('should_return_the_expected_next_step_candidate'):
      self.assertEqual(expected_next_step_candidate, next_step_candidates[0])

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

  def test_execute(self):
    # Some minimal agent configuration and inputs.
    question = 'What is larger, 10 or 15, and by how much?'
    config = _get_environment_config_with_python()

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_reply_0 = (
        '[Thought]: We can use the Python tool to subtract one from '
        "another.\n[Act]: tool_code('10 - 15')\n[Observe]: Bla bla bla."
    )
    llm_reply_0_trimmed = (
        '[Thought]: We can use the Python tool to subtract one from '
        "another.\n[Act]: tool_code('10 - 15')\n"
    )
    llm_reply_1 = '[Finish]: 15 is larger than 10 by 5.\n'
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={'': [llm_reply_0, llm_reply_1]},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = react.ReActAgent(
        exemplars=react.REACT_FEWSHOTS,
        environment_config=config,
        max_steps=10,
        stop_prefix='',
    )

    (output, final_state), execution_result = executing.run(
        agent(inputs=question, return_final_state=True),
        enable_tracing=True,
    )
    leaf_results = execution_result.get_leaf_results()

    expected_final_state = react.ReActState(
        inputs=question,
        updates=[
            react.ReActStep(
                is_finished=False,
                thought=(
                    'We can use the Python tool to subtract one from another.'
                ),
                action=llm_tool_use.FunctionCall(
                    function_name='tool_code', args=('10 - 15',), kwargs={}
                ),
                observation=-5,
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                is_finished=True,
                observation='15 is larger than 10 by 5.',
            ),
        ],
    )

    with self.subTest('should_output_the_final_answer'):
      self.assertEqual('15 is larger than 10 by 5.', output)

    with self.subTest('should_return_the_expected_final_state'):
      self.assertEqual(
          expected_final_state,
          final_state,
          msg=(
              f'Expected final state:\n{expected_final_state}, '
              f'got:\n{final_state}'
          ),
      )

    with self.subTest('should_yield_one_leaf_result_per_llm_call_and_action'):
      self.assertLen(leaf_results, 3)

    with self.subTest('leaf_result_0_should_be_llm_call_for_step_1'):
      self.assertEqual('generate_text', leaf_results[0].stage_name)
      self.assertEqual(
          llm_reply_0_trimmed,
          leaf_results[0].outputs.get('output'),
          f'{leaf_results[0].outputs=}',
      )

    with self.subTest('leaf_result_1_should_be_tool_call_for_step_1'):
      self.assertEqual('run_tool', leaf_results[1].stage_name)
      self.assertEqual(
          {
              'tool_args': ('10 - 15',),
              'tool_kwargs': {},
              'tool_name': 'tool_code',
          },
          leaf_results[1].inputs,
      )
      self.assertEqual({'output': -5}, leaf_results[1].outputs)

    with self.subTest('leaf_result_2_should_be_llm_call_for_step_2'):
      self.assertEqual('generate_text', leaf_results[2].stage_name)
      self.assertEqual(
          llm_reply_1,
          leaf_results[2].outputs.get('output'),
          f'{leaf_results[2].outputs=}',
      )

  def test_execute_force_finish_due_to_max_steps(self):
    # Some minimal agent configuration and inputs.
    question = 'What is larger, 10 or 15, and by how much?'
    config = _get_environment_config_with_python()

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': [
                # LLM reply for step 1.
                (
                    '[Thought]: We can use the Python tool to subtract one from'
                    ' another.\n[Act]: tool_code("10 - 15")\n[Observe]: Bla bla'
                    ' bla.'
                ),
                # LLM reply for force-finish.
                '15 is larger than 10 by 5.\n\n[Question] Bla bla bla.',
            ]
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = react.ReActAgent(
        exemplars=react.REACT_FEWSHOTS,
        environment_config=config,
        max_steps=1,
        stop_prefix='',
    )

    (output, final_state), execution_result = executing.run(
        agent(inputs=question, return_final_state=True),
        enable_tracing=True,
    )
    leaf_results = execution_result.get_leaf_results()

    with self.subTest('should_output_the_stripped_final_llm_reply'):
      self.assertEqual('15 is larger than 10 by 5.', output)

    with self.subTest('should_yield_one_leaf_result_per_llm_call_and_action'):
      self.assertLen(leaf_results, 3)

    with self.subTest('leaf_result_0_should_be_llm_call_for_step_1'):
      self.assertEqual('generate_text', leaf_results[0].stage_name)

    with self.subTest('leaf_result_1_should_be_tool_call_for_step_1'):
      self.assertEqual('run_tool', leaf_results[1].stage_name)

    with self.subTest('leaf_result_2_should_be_llm_call_for_force_finish'):
      self.assertEqual('generate_text', leaf_results[2].stage_name)

    with self.subTest('should_end_in_finished_state'):
      self.assertTrue(final_state.updates[-1].is_finished)

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

  def test_execute_force_finish_due_to_empty_reply(self):
    # Some minimal agent configuration and inputs.
    question = 'What is larger, 10 or 15, and by how much?'
    config = _get_environment_config_with_python()

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': [
                # LLM reply for step 1.
                (
                    '[Thought]: We can use the Python tool to subtract one from'
                    ' another.\n[Act]: tool_code("10 - 15")\n[Observe]: Bla bla'
                    ' bla.'
                ),
                # LLM reply for step 2 (empty reply).
                '',
                # LLM reply for force-finish.
                '15 is larger than 10 by 5.\n\n[Question] Bla bla bla.',
            ]
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = react.ReActAgent(
        exemplars=react.REACT_FEWSHOTS,
        environment_config=config,
        max_steps=5,
        stop_prefix='',
    )

    (output, final_state), execution_result = executing.run(
        agent(inputs=question, return_final_state=True),
        enable_tracing=True,
    )
    leaf_results = execution_result.get_leaf_results()

    with self.subTest('should_output_the_stripped_final_llm_reply'):
      self.assertEqual('15 is larger than 10 by 5.', output)

    with self.subTest('should_yield_one_leaf_result_per_llm_call_and_action'):
      self.assertLen(leaf_results, 4)

    with self.subTest('leaf_result_0_should_be_llm_call_for_step_1'):
      self.assertEqual('generate_text', leaf_results[0].stage_name)

    with self.subTest('leaf_result_1_should_be_tool_call_for_step_1'):
      self.assertEqual('run_tool', leaf_results[1].stage_name)

    with self.subTest('leaf_result_2_should_be_llm_call_for_step_2'):
      self.assertEqual('generate_text', leaf_results[2].stage_name)

    with self.subTest('leaf_result_3_should_be_llm_call_for_force_finish'):
      self.assertEqual('generate_text', leaf_results[3].stage_name)

    with self.subTest('should_end_in_finished_state'):
      self.assertTrue(final_state.updates[-1].is_finished)

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)


if __name__ == '__main__':
  absltest.main()
