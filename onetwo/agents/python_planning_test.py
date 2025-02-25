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

import dataclasses
import logging
import re

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import python_planning
from onetwo.backends import backends_test_utils
from onetwo.builtins import formatting
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

_ExecutionStatus = python_execution.ExecutionStatus

# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


@dataclasses.dataclass
class MockSearch:
  """Functor that returns mock search results, for use in testing.

  Attributes:
    reply_by_query: Hard-coded replies to return for given queries.
  """

  reply_by_query: dict[str, str] = dataclasses.field(default_factory=dict)

  def __call__(self, query: str) -> str:
    return self.reply_by_query.get(query, f'Unknown query: {query}')


def firstnumber(x: str) -> float | str:
  """Returns the first number in a string, or an error message."""
  matches = re.search(r'^[\d]*([\d\.,]+).*', str(x).replace(',', ''))
  if matches:
    try:
      return float(matches.group(1))
    except Exception as e:  # pylint: disable=broad-except
      return f'Error: could not parse {x} as a number ({e})'
  else:
    return f'Error: could not parse {x} as a number'


def error_tool(x: str) -> str:
  """Returns an error message."""
  if x == 'show_key_error':
    raise KeyError('test key error')
  elif x == 'show_runtime_error':
    raise RuntimeError('test runtime error')
  elif x == 'show_file_permission_error':
    raise PermissionError('test file permission error')
  return x


def _get_environment_config_with_tools() -> (
    python_tool_use.PythonToolUseEnvironmentConfig
):
  """Returns an environment config with search and firstnumber tools."""
  # Set up some simple tools that don't involve RPCs.
  return python_tool_use.PythonToolUseEnvironmentConfig(
      tools=[
          llm_tool_use.Tool(
              name='search',
              function=MockSearch(
                  reply_by_query={
                      'population of Tuebingen': '91,877',
                      'population of Zuerich': '402,762',
                  }
              ),
              description='Google search engine.',
              example="search('capital of France')  # returns 'Paris'",
          ),
          llm_tool_use.Tool(
              name='firstnumber',
              function=firstnumber,
              description='Extracts the first number in a string.',
              example=(
                  "firstnumber('it is 1,203m high')  # return 1203.0 as a float"
              ),
          ),
      ]
  )


class PythonPlanningTest(parameterized.TestCase):

  @staticmethod
  def _build_params_for_prompt(tests):
    out_tests = []
    named_prompt_classes = {
        'j2': python_planning.PythonPlanningPromptJ2,
        'composable': python_planning.PythonPlanningPromptComposable,
    }
    for prompt_name, prompt_class in named_prompt_classes.items():
      for test_name, *test_args in tests:
        out_tests.append(
            (f'{test_name}_{prompt_name}', prompt_class, *test_args)
        )
    return out_tests

  @parameterized.named_parameters(
      _build_params_for_prompt((
          ('first_step', []),
          (
              'middle_step',
              [
                  python_planning.PythonPlanningStep(
                      is_finished=False,
                      code=(
                          '# First we need to find the populations of Tuebingen'
                          ' and Zuerich individually\npopulation1 ='
                          " search('population of Tuebingen')\npopulation2 ="
                          " search('population of Zuerich')\nprint(f'Tuebingen:"
                          " {population1}, Zuerich: {population2}')"
                      ),
                      result=(
                          "Tuebingen: [{'index': 1, 'Name': 'Tübingen',"
                          " 'County': 'Tübingen', 'Population Estimate"
                          " 2021-12-31': '91,877'}, {'index': 2, 'Name':"
                          " 'Tübingen 91,877 Population [2021] – Estimate 108.1"
                          ' km2 Area 850.2/km2 Population Density [2021] 1.0%'
                          " Annual Population Change [2011 → 2021]', 'County':"
                          " '', 'Population Estimate 2021-12-31': ''}],"
                          ' Zuerich: 402,762 (2017)\n'
                      ),
                      execution_status=_ExecutionStatus.SUCCESS,
                  ),
              ],
          ),
          (
              'last_step',
              [
                  python_planning.PythonPlanningStep(
                      is_finished=True,
                      code=(
                          '# First we need to find the populations of Tuebingen'
                          ' and Zuerich individually\npopulation1 ='
                          " search('population of Tuebingen')\npopulation2 ="
                          " search('population of Zuerich')\nprint(f'Tuebingen:"
                          " {population1}, Zuerich: {population2}')"
                      ),
                      result=(
                          "Tuebingen: [{'index': 1, 'Name': 'Tübingen',"
                          " 'County': 'Tübingen', 'Population Estimate"
                          " 2021-12-31': '91,877'}, {'index': 2, 'Name':"
                          " 'Tübingen 91,877 Population [2021] – Estimate 108.1"
                          ' km2 Area 850.2/km2 Population Density [2021] 1.0%'
                          " Annual Population Change [2011 → 2021]', 'County':"
                          " '', 'Population Estimate 2021-12-31': ''}],"
                          ' Zuerich: 402,762 (2017)\n'
                      ),
                      execution_status=_ExecutionStatus.SUCCESS,
                  ),
              ],
          ),
      ))
  )
  def test_prompt(self, prompt_class, updates):
    # Some typical Python Planning prompt inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'
    exemplars = python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS

    # Tool configuration.
    config = _get_environment_config_with_tools()

    # Prompt configuration.
    prompt = prompt_class()

    # Current state of agent.
    state = python_planning.PythonPlanningState(
        inputs=question,
        updates=updates,
    )

    # Here we define a test LLM to use in place of the actual LLM. To avoid the
    # need to hard-code here all of the expected requests, we will just specify
    # the sequence of replies to return, regardless of the request.
    llm_reply = """\
# Now we extract the numbers.
num1 = firstnumber(population1)
num2 = firstnumber(population2)
"""
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
        prompt(tools=config.tools, exemplars=exemplars, state=state),
        enable_tracing=True,
    )
    if prompt_class == python_planning.PythonPlanningPromptJ2:
      prefix = result.get_leaf_results()[0].inputs['request']
    elif prompt_class == python_planning.PythonPlanningPromptComposable:
      formatter = formatting.ConcatFormatter()
      prefix = formatter.format(result.get_leaf_results()[0].inputs['messages'])
      prefix = str(prefix)
    else:
      raise ValueError(f'Unsupported prompt class: {prompt_class}')

    logging.info('Prompt sent to the LLM:\n%s', prefix)

    with self.subTest(
        'prompt_should_start_with_instructions_followed_by_a_blank_line'
    ):
      self.assertStartsWith(
          prefix, python_planning.DEFAULT_PYTHON_PLANNING_INSTRUCTION + '\n\n'
      )
      self.assertNotIn(
          python_planning.DEFAULT_PYTHON_PLANNING_INSTRUCTION + '\n\n\n', prefix
      )

    with self.subTest('prompt_should_contain_the_tool_descriptions'):
      self.assertIn(config.tools[0].description, prefix)
      self.assertIn(config.tools[-1].description, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_inputs'):
      self.assertIn(exemplars[0].inputs, prefix)
      self.assertIn(exemplars[-1].inputs, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_steps'):
      self.assertIn(exemplars[0].updates[0].code, prefix)
      self.assertIn(exemplars[-1].updates[-1].code, prefix)

    with self.subTest('prompt_should_contain_the_actual_inputs'):
      self.assertIn(state.inputs, prefix)

    with self.subTest(
        'prompt_should_contain_question_followed_by_triple_backticks_on_next_line'
    ):
      self.assertIn(state.inputs + '\n```', prefix)
      self.assertNotIn(state.inputs + '\n\n```', prefix)

    if not state.updates or not state.updates[-1].is_finished:
      with self.subTest(
          'prompt_should_end_with_triple_backticks_on_its_own_line'
      ):
        self.assertEndsWith(prefix, '\n```')
        # Depending on whether the current state already contains a code block,
        # the prompt may or may not have a blank line before the closing fence,
        # but there should never be more than one blank line.
        self.assertNotEndsWith(prefix, '\n\n\n```')

    if state.updates:
      with self.subTest('prompt_should_contain_the_actual_code_so_far'):
        self.assertIn(state.updates[0].code, prefix)
        self.assertIn(state.updates[-1].code, prefix)

    with self.subTest('should_return_the_llm_reply'):
      self.assertEqual(llm_reply.strip(), prompt_outputs)

  @parameterized.named_parameters(
      ('j2', python_planning.PythonPlanningPromptJ2),
      ('composable', python_planning.PythonPlanningPromptComposable),
  )
  def test_prompt_error(self, prompt_class):
    # Some typical inputs / configs. (As in the other test above.)
    question = 'What is the total population of Tuebingen and Zuerich?'
    exemplars = python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS
    config = _get_environment_config_with_tools()
    prompt = prompt_class()

    state = python_planning.PythonPlanningState(inputs=question, updates=[])

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
        prompt(tools=config.tools, exemplars=exemplars, state=state),
        enable_tracing=True,
    )
    if prompt_class == python_planning.PythonPlanningPromptJ2:
      prefix = result.stages[0].outputs['prefix']

      with self.subTest('prompt_should_contain_the_error_message'):
        self.assertIn(error_message, prefix)

    with self.subTest('should_return_the_llm_reply'):
      self.assertEqual(f'#ERROR#: {error_message}', prompt_outputs)

  @parameterized.named_parameters(
      ('j2', python_planning.PythonPlanningPromptJ2),
      ('composable', python_planning.PythonPlanningPromptComposable),
  )
  def test_sample_next_step(self, prompt_class):
    # Some minimal agent configuration and inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'
    prev_state = python_planning.PythonPlanningState(inputs=question)

    # Set up some simple tools that don't involve RPCs.
    config = _get_environment_config_with_tools()

    # Mock LLM replies.
    llm_reply = """\
# First we need to find the populations of Tuebingen and Zuerich individually.
population1 = search('population of Tuebingen')
population2 = search('population of Zuerich')
print('Tuebingen: %s, Zuerich: %s' % (population1, population2))
"""
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'What is the total population of Tuebingen and Zuerich\?\n```$': (
                llm_reply
            ),
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        prompt=prompt_class(),
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=config,
        max_steps=1,
    )

    num_candidates = 2
    next_step_candidates = executing.run(
        agent.start_environment_and_sample_next_step(
            state=prev_state, num_candidates=num_candidates
        )
    )

    expected_next_step_candidate = python_planning.PythonPlanningStep(
        is_finished=False,
        code=llm_reply.strip(),
        result='Tuebingen: 91,877, Zuerich: 402,762\n',
        execution_status=_ExecutionStatus.SUCCESS,
    )

    with self.subTest('should_return_the_expected_number_of_candidates'):
      self.assertLen(next_step_candidates, num_candidates)

    with self.subTest('should_return_the_expected_next_step_candidate'):
      self.assertEqual(
          expected_next_step_candidate,
          next_step_candidates[0],
          msg=(
              f'Expected: {expected_next_step_candidate}, got:'
              f' {next_step_candidates[0]}'
          ),
      )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

  @parameterized.named_parameters(
      ('not_empty_key_error', 'show_key_error', ['KeyError'], True),
      (
          'not_empty_permission_error',
          'show_file_permission_error',
          ['PermissionError'],
          True,
      ),
      ('not_empty_runtime_error', 'show_runtime_error', ['RuntimeError'], True),
      ('empty', 'test', None, False),
  )
  def test_sample_next_step_with_irrecoverable_error(
      self, llm_reply_message, irrecoverable_error_types, expected_is_finished
  ):

    tool_with_irrecoverable_error_types = (
        llm_tool_use.Tool(
            name='error_tool',
            function=error_tool,
            description='Returns a KeyError exception.',
            example="error_tool('test')  # returns test as a string",
            irrecoverable_error_types=irrecoverable_error_types,
        ),
    )
    # Set up some simple tools that don't involve RPCs.
    config = python_tool_use.PythonToolUseEnvironmentConfig(
        tools=tool_with_irrecoverable_error_types
    )
    # Some minimal agent configuration and inputs.
    question = 'My test question'
    prev_state = python_planning.PythonPlanningState(inputs=question)

    # Mock LLM replies.
    llm_reply = f"""\
error_message = error_tool("{llm_reply_message}")
print(error_message)
"""
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'My test question\n```$': llm_reply,
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=config,
        max_steps=2,
    )

    result = executing.run(
        agent.start_environment_and_sample_next_step(
            state=prev_state, num_candidates=1
        )
    )
    self.assertEqual(result[0].is_finished, expected_is_finished)

  @parameterized.named_parameters(
      ('j2', python_planning.PythonPlanningPromptJ2),
      ('composable', python_planning.PythonPlanningPromptComposable),
  )
  def test_execute_with_max_steps(self, prompt_class):
    # Some minimal agent configuration and inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'

    # Set up some simple tools that don't involve RPCs.
    config = _get_environment_config_with_tools()

    # Mock LLM replies.
    llm_reply = """\
# First we need to find the populations of Tuebingen and Zuerich individually.
population1 = search('population of Tuebingen')
population2 = search('population of Zuerich')
print('Tuebingen: %s, Zuerich: %s' % (population1, population2))
"""
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'What is the total population of Tuebingen and Zuerich\?'
            + r'\n```$': llm_reply,
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        prompt=prompt_class(),
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=config,
        max_steps=1,
    )

    output = executing.run(agent(inputs=question))
    self.assertEqual('Tuebingen: 91,877, Zuerich: 402,762', output)

  def test_execute_with_exit(self):
    # Here we configure a large value of `max_steps` so as it verify that the
    # agent continues executing until the LLM outputs a code block containing a
    # call to the `exit()` function.

    question = 'What is 3 to the 4th power?'

    # Hard-coded sequence of LLM replies (culminating in `exit()`).
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_regex={
            '': [
                'x = 3 * 3',
                'x *= 3',
                'x *= 3\nprint(x)\nexit()',
                'print("Should not reach here.")',
            ]
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        max_steps=10,
    )

    output = executing.run(agent(inputs=question))

    with self.subTest('should_return_the_correct_output'):
      # We'll only get the correct output if the code blocks up through `exit()`
      # are executed in the correct order.
      self.assertEqual('81', output)

    with self.subTest('should_generate_only_the_expected_requests'):
      # This verifies that the agent actually stops when it sees `exit()`.
      self.assertEmpty(llm_backend.unexpected_prompts)

  def test_execute_with_multimodal_inputs(self):
    mock_image_of_zurich_bytes = b'<zurich_image>'
    question = content_lib.ChunkList([
        content_lib.Chunk(mock_image_of_zurich_bytes),
        content_lib.Chunk('What is the population of this city?'),
    ])

    # Set up some simple tools that don't involve RPCs.
    config = _get_environment_config_with_tools()

    # Mock LLM replies.
    llm_reply = """\
# This is a picture of Zuerich. We need to find its population.
population = search('population of Zuerich')
print('Zuerich population: %s' % population)
"""
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt_bytes_regex={
            b"[b'<zurich_image>', 'What is the population of this city?\n```$']": (
                llm_reply
            ),
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        prompt=python_planning.PythonPlanningPromptComposable(),
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=config,
        max_steps=1,
    )

    output = executing.run(agent(inputs=question))

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_correct_output'):
      self.assertEqual('Zuerich population: 402,762', output)

  @parameterized.named_parameters(
      ('no_fences', 'print("Hello!")', 'print("Hello!")'),
      (
          'no_fences_white_space',
          '\nprint("Hello!")\n',
          'print("Hello!")',
      ),
      (
          # This may occur, for example, when the starting fence is already
          # included in the prompt, and no stop sequence is specified.
          'closing_backticks_only',
          'print("Hello!")\n```',
          'print("Hello!")',
      ),
      (
          'generic_fence',
          '```\nprint("Hello!")\n```',
          'print("Hello!")',
      ),
      (
          'python_fence',
          '```python\nprint("Hello!")',
          'print("Hello!")',
      ),
      (
          'tool_code_fence',
          '```tool_code\nprint("Hello!")',
          'print("Hello!")',
      ),
      (
          'random_fence',
          '```random_fence\nprint("Hello!")',
          'print("Hello!")',
      ),
      (
          'first_line_is_the_word_python',
          'python\nprint("Hello!")',
          'print("Hello!")',
      ),
      (
          'first_line_is_the_word_tool_code',
          'tool_code\nprint("Hello!")',
          'print("Hello!")',
      ),
      (
          'multiline_python_fence',
          '```python\nprint("Hello!")\nprint("Bye!")',
          'print("Hello!")\nprint("Bye!")',
      ),
      (
          # The closing fence is treated like a "stop token" and everything
          # after it is discarded.
          'multiline_python_fence_text_beyond_trailing_backticks',
          '```python\nprint("Hello!")\nprint("Bye!")\n```\nunwanted_text',
          'print("Hello!")\nprint("Bye!")',
      ),
      (
          # The closing fence is treated like a "stop token" and everything
          # after it is discarded.
          'multiline_generic_fence_text_beyond_trailing_backticks',
          '```\nprint("Hello!")\nprint("Bye!")\n```\nunwanted_text',
          'print("Hello!")\nprint("Bye!")',
      ),
      (
          # Text before the first fence is always treated as code (even if it
          # was actually textual commentary or other unwanted text), and
          # anything after the fence is discarded, as it is generally
          # non-trivial to distinguish between `text_before_fence` (here) vs.
          # `multiple_code_blocks_no_start_fence` (below).
          'text_before_fence',
          'unwanted_pre\n```\nprint("Hello!")```',
          'unwanted_pre',
      ),
      (
          # Like above. Text before first fence is treated as first code block.
          # The first code block is kept. Others discarded.
          'text_before_and_after_fence',
          'unwanted_pre\n```python\nprint("Hello!")\n```\nunwanted_post',
          'unwanted_pre',
      ),
      (
          # The first code block is kept. Others discarded.
          'multiple_code_blocks_with_start_fence',
          '```\nprint("Hello!")```\ntext_between```\nprint("Bye!")```',
          'print("Hello!")',
      ),
      (
          # The first code block is kept. Others discarded.
          'multiple_code_blocks_with_start_fence_python',
          '```python\nprint("Hello!")```\ntext_between```\nprint("Bye!")```',
          'print("Hello!")',
      ),
      (
          # The first code block is kept. Others discarded.
          'multiple_code_blocks_no_start_fence',
          'print("Hello!")\n```\ntext_between```\nprint("Bye!")```',
          'print("Hello!")',
      ),
      (
          # The first code block is kept. Others discarded.
          'multiple_code_blocks_no_start_fence_ticks_in_first_line',
          'print("Hello!")```\ntext_between```\nprint("Bye!")```',
          'print("Hello!")',
      ),
      (
          # The first code block is kept. Others discarded.
          'multiple_code_blocks_no_start_fence_ticks_and_text_in_first_line',
          'print("Hello!")```text\ntext_between```\nprint("Bye!")```',
          'print("Hello!")',
      ),
  )
  def test_parse_llm_reply_code(self, llm_reply, expected):
    self.assertEqual(expected, python_planning._parse_llm_reply_code(llm_reply))


if __name__ == '__main__':
  absltest.main()
