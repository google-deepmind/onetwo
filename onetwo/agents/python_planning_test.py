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
from onetwo.backends import test_utils
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


class PythonPlanningTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('first_step', []),
      (
          'middle_step',
          [
              python_planning.PythonPlanningStep(
                  is_finished=False,
                  code=(
                      '# First we need to find the populations of Tuebingen and'
                      " Zuerich individually\npopulation1 = search('population"
                      " of Tuebingen')\npopulation2 = search('population of"
                      " Zuerich')\nprint(f'Tuebingen: {population1}, Zuerich:"
                      " {population2}')"
                  ),
                  result=(
                      "Tuebingen: [{'index': 1, 'Name': 'Tübingen', 'County':"
                      " 'Tübingen', 'Population Estimate 2021-12-31':"
                      " '91,877'}, {'index': 2, 'Name': 'Tübingen 91,877"
                      ' Population [2021] – Estimate 108.1 km2 Area 850.2/km2'
                      ' Population Density [2021] 1.0% Annual Population Change'
                      " [2011 → 2021]', 'County': '', 'Population Estimate"
                      " 2021-12-31': ''}], Zuerich: 402,762 (2017)\n"
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
                      '# First we need to find the populations of Tuebingen and'
                      " Zuerich individually\npopulation1 = search('population"
                      " of Tuebingen')\npopulation2 = search('population of"
                      " Zuerich')\nprint(f'Tuebingen: {population1}, Zuerich:"
                      " {population2}')"
                  ),
                  result=(
                      "Tuebingen: [{'index': 1, 'Name': 'Tübingen', 'County':"
                      " 'Tübingen', 'Population Estimate 2021-12-31':"
                      " '91,877'}, {'index': 2, 'Name': 'Tübingen 91,877"
                      ' Population [2021] – Estimate 108.1 km2 Area 850.2/km2'
                      ' Population Density [2021] 1.0% Annual Population Change'
                      " [2011 → 2021]', 'County': '', 'Population Estimate"
                      " 2021-12-31': ''}], Zuerich: 402,762 (2017)\n"
                  ),
                  execution_status=_ExecutionStatus.SUCCESS,
              ),
          ],
      ),
  )
  def test_prompt(self, updates):
    # Some typical Python Planning prompt inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'
    exemplars = python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS

    # Tool configuration.
    tools = [
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

    # Prompt configuration.
    prompt = python_planning.PythonPlanningPromptJ2()

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
    llm_backend = test_utils.LLMForTest(
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
        prompt(tools=tools, exemplars=exemplars, state=state),
        enable_tracing=True,
    )
    prefix = result.get_leaf_results()[0].inputs['request']

    logging.info('Prompt sent to the LLM:\n%s', prefix)

    with self.subTest('prompt_should_contain_the_tool_descriptions'):
      self.assertIn(tools[0].description, prefix)
      self.assertIn(tools[-1].description, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_inputs'):
      self.assertIn(exemplars[0].inputs, prefix)
      self.assertIn(exemplars[-1].inputs, prefix)

    with self.subTest('prompt_should_contain_the_exemplar_steps'):
      self.assertIn(exemplars[0].updates[0].code, prefix)
      self.assertIn(exemplars[-1].updates[-1].code, prefix)

    with self.subTest('prompt_should_contain_the_actual_inputs'):
      self.assertIn(state.inputs, prefix)

    with self.subTest('prompt_should_have_one_blank_line_after_the_question'):
      self.assertIn(state.inputs + '\n\n```', prefix)
      self.assertNotIn(state.inputs + '\n\n\n```', prefix)

    if not state.updates or not state.updates[-1].is_finished:
      with self.subTest(
          'prompt_should_end_with_triple_backticks_preceded_by_one_blank_line'
      ):
        # TODO: Would it be better to remove the trailing newline?
        self.assertEndsWith(prefix, '\n\n```\n')
        self.assertNotEndsWith(prefix, '\n\n\n```\n')

    if state.updates:
      with self.subTest('prompt_should_contain_the_actual_code_so_far'):
        self.assertIn(state.updates[0].code, prefix)
        self.assertIn(state.updates[-1].code, prefix)

    with self.subTest('should_return_the_llm_reply'):
      self.assertEqual(llm_reply.strip(), prompt_outputs)

  def test_sample_next_step(self):
    # Some minimal agent configuration and inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'
    prev_state = python_planning.PythonPlanningState(inputs=question)

    # Set up some simple tools that don't involve RPCs.
    tools = [
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

    # We hard-code the LLM to return the same thought every time (assuming the
    # prompt is of the expected form).
    llm_reply = """\
# First we need to find the populations of Tuebingen and Zuerich individually.
population1 = search('population of Tuebingen')
population2 = search('population of Zuerich')
print('Tuebingen: %s, Zuerich: %s' % (population1, population2))
"""
    llm_backend = test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'What is the total population of Tuebingen and Zuerich\?\n\n```$': (
                llm_reply
            ),
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
            tools=tools
        ),
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

  def test_execute_with_max_steps(self):
    # Some minimal agent configuration and inputs.
    question = 'What is the total population of Tuebingen and Zuerich?'

    # Set up some simple tools that don't involve RPCs.
    tools = [
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

    # We hard-code the LLM to return the same thought every time (assuming the
    # prompt is of the expected form).
    llm_reply = """\
# First we need to find the populations of Tuebingen and Zuerich individually.
population1 = search('population of Tuebingen')
population2 = search('population of Zuerich')
print('Tuebingen: %s, Zuerich: %s' % (population1, population2))
"""
    llm_backend = test_utils.LLMForTest(
        reply_by_prompt_regex={
            r'What is the total population of Tuebingen and Zuerich\?'
            + r'\n\n```\n$': llm_reply,
        },
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    agent = python_planning.PythonPlanningAgent(
        exemplars=python_planning.DEFAULT_PYTHON_PLANNING_EXEMPLARS,
        environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
            tools=tools,
        ),
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
    llm_backend = test_utils.LLMForTest(
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


if __name__ == '__main__':
  absltest.main()
