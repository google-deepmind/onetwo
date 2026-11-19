# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for the model-native function calling agent."""

from collections.abc import AsyncIterator, Sequence
import contextlib
import dataclasses
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import agent as agent_lib
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.core import results
from onetwo.stdlib.tool_use import llm_tool_use

_USER = fc_types.ConversationRole.USER
_MODEL = fc_types.ConversationRole.MODEL
_TOOL = fc_types.ConversationRole.TOOL

# What the agent labels each of its steps in the execution trace. Spelled out
# here rather than imported, because the label is a human-facing one that
# nothing outside the agent is meant to match on.
_STEP_STAGE_NAME = 'Agent.step'


def _get_weather(location: str) -> dict[str, Any]:
  """Gets the weather for a location.

  Args:
    location: City name.

  Returns:
    A dictionary of weather information.
  """
  return {'location': location, 'temp': 15}


def _get_time(city: str) -> str:
  """Gets the time in a city.

  Args:
    city: City name.

  Returns:
    The local time.
  """
  return f'12:00 in {city}'


def _explode(reason: str) -> str:
  """Always fails.

  Args:
    reason: Why it failed.

  Returns:
    Never returns.
  """
  raise ValueError(reason)


def _forbidden() -> str:
  """Always fails with an error the tool declares irrecoverable.

  Returns:
    Never returns.

  Raises:
    PermissionError: Always.
  """
  raise PermissionError('not allowed')


def _model_turn(
    content: str | None = None, calls: Sequence[fc_types.FunctionCall] = ()
) -> fc_types.ConversationTurn:
  return fc_types.ConversationTurn(role=_MODEL, content=content, calls=calls)


def _fake_source(
    tools: Sequence[Any], events: list[str]
) -> llm_tool_use.ToolSource:
  """Returns a source of tools whose opening and closing a test can observe.

  Stands in for anything whose tools outlive neither their connection nor the
  run: an MCP server, say.

  Args:
    tools: What it yields once open.
    events: Where `'open'` and `'close'` are recorded as they happen, so that a
      test can say when each did rather than only that it did.

  Returns:
    A source yielding `tools`.
  """

  @contextlib.asynccontextmanager
  async def opened() -> AsyncIterator[Sequence[Any]]:
    events.append('open')
    try:
      yield tools
    finally:
      events.append('close')

  return llm_tool_use.ToolSource(open=opened)


def _run(
    agent: agent_lib.Agent,
    inputs: str | content_lib.ChunkList | Sequence[fc_types.ConversationTurn],
    **kwargs,
) -> str:
  """Runs `agent` and returns its answer.

  `Agent.__call__` is declared to return a union covering `return_final_state`,
  so this narrows it back down for the benefit of the type checker.

  Args:
    agent: The agent to run.
    inputs: What to run it on.
    **kwargs: Forwarded to `Agent.__call__`, e.g. `max_steps`.

  Returns:
    The agent's answer, or '' if it gave none.
  """
  return cast(
      str, executing.run(agent(inputs, **kwargs))
  )


def _run_with_state(
    agent: agent_lib.Agent,
    inputs: str | content_lib.ChunkList | Sequence[fc_types.ConversationTurn],
    **kwargs,
) -> tuple[str, agent_lib.AgentState]:
  """Runs `agent` and returns its answer together with the final state.

  Args:
    agent: The agent to run.
    inputs: What to run it on.
    **kwargs: Forwarded to `Agent.__call__`, e.g. `max_steps`.

  Returns:
    The agent's answer ('' if it gave none) and the state it ended in.
  """
  return cast(
      tuple[str, agent_lib.AgentState],
      executing.run(agent(inputs, return_final_state=True, **kwargs)),
  )


@dataclasses.dataclass
class _ScriptedBackend:
  """A backend that replies with a fixed script, and records what it was sent.

  Stands in for a real backend: the agent under test only ever reaches a model
  through `llm.generate_with_tools`, so scripting that one builtin is enough to
  drive any exchange.
  """

  replies: list[fc_types.ConversationTurn] = dataclasses.field(
      default_factory=list
  )
  seen_turns: list[list[fc_types.ConversationTurn]] = dataclasses.field(
      default_factory=list
  )
  seen_tools: list[list[fc_types.ToolSpec]] = dataclasses.field(
      default_factory=list
  )

  def register(self) -> None:
    llm.generate_with_tools.configure(self.generate_with_tools)

  # `copy_self=False` so that the calls we record are visible to the test.
  @executing.make_executable(copy_self=False)
  async def generate_with_tools(
      self,
      turns: Sequence[fc_types.ConversationTurn],
      tools: Sequence[fc_types.ToolSpec],
      **kwargs,
  ) -> fc_types.ConversationTurn:
    del kwargs
    self.seen_turns.append(list(turns))
    self.seen_tools.append(list(tools))
    index = min(len(self.seen_turns) - 1, len(self.replies) - 1)
    return self.replies[index]


class FCStateTest(absltest.TestCase):

  def test_add_appends_turns_and_counts_steps(self):
    state = agent_lib.AgentState(
        turns=[fc_types.ConversationTurn(role=_USER, content='hi')]
    )
    step = agent_lib.AgentStep(
        model_turn=_model_turn(calls=[fc_types.FunctionCall('f')]),
        tool_turn=fc_types.ConversationTurn(
            role=_TOOL, responses=[fc_types.FunctionResponse('f', response=1)]
        ),
    )

    new_state = state + step

    with self.subTest('appends_both_turns'):
      self.assertLen(new_state.turns, 3)
    with self.subTest('counts_the_step'):
      self.assertEqual(new_state.num_steps, 1)
    with self.subTest('leaves_the_original_untouched'):
      self.assertLen(state.turns, 1)

  def test_add_without_tool_turn_appends_one_turn(self):
    state = agent_lib.AgentState()
    new_state = state + agent_lib.AgentStep(model_turn=_model_turn('done'))
    self.assertLen(new_state.turns, 1)

  def test_add_puts_the_user_turn_before_the_model_turn(self):
    state = agent_lib.AgentState()
    new_state = state + agent_lib.AgentStep(
        model_turn=_model_turn('done'),
        user_turn=fc_types.ConversationTurn(role=_USER, content='Answer now.'),
    )
    self.assertEqual([turn.role for turn in new_state.turns], [_USER, _MODEL])


class FCAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def test_replies_without_calls_finish_immediately(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    answer = _run(agent, 'Hi there')

    with self.subTest('returns_the_model_turn'):
      self.assertEqual(answer, 'Hello!')
    with self.subTest('calls_the_model_once'):
      self.assertLen(backend.seen_turns, 1)
    with self.subTest('declares_the_tool'):
      self.assertEqual(
          [spec.name for spec in backend.seen_tools[0]], ['_get_weather']
      )

  def test_declarations_carry_no_code(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    _run(agent, 'Hi there')

    spec = backend.seen_tools[0][0]
    with self.subTest('is_a_declaration_only_spec'):
      self.assertIsInstance(spec, fc_types.ToolSpec)
    with self.subTest('carries_the_json_schema'):
      self.assertIn('location', (spec.parameters or {})['properties'])

  def test_executes_the_requested_call_and_feeds_the_result_back(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}, id='c1'
                    )
                ]
            ),
            _model_turn('It is 15 degrees in London.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    answer, state = _run_with_state(agent, 'Weather in London?')

    with self.subTest('returns_the_final_answer'):
      self.assertEqual(answer, 'It is 15 degrees in London.')
    with self.subTest('sends_the_tool_result_back_to_the_model'):
      second_call_turns = backend.seen_turns[1]
      tool_turn = second_call_turns[-1]
      self.assertEqual(tool_turn.role, _TOOL)
      self.assertEqual(
          tool_turn.responses[0].response, {'location': 'London', 'temp': 15}
      )
    with self.subTest('matches_the_response_to_the_call_id'):
      self.assertEqual(tool_turn.responses[0].id, 'c1')
    with self.subTest('turns_are_user_model_tool_model'):
      self.assertEqual(
          [turn.role for turn in state.turns], [_USER, _MODEL, _TOOL, _MODEL]
      )

  @parameterized.named_parameters(
      ('parallel', True),
      ('sequential', False),
  )
  def test_answers_several_calls_in_the_order_requested(self, parallel):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    ),
                    fc_types.FunctionCall('_get_time', kwargs={'city': 'Oslo'}),
                ]
            ),
            _model_turn('Done.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather, _get_time], parallel_calls=parallel
    )

    _run(agent, 'Tell me about London and Oslo')

    responses = backend.seen_turns[1][-1].responses
    self.assertEqual(
        [response.function_name for response in responses],
        ['_get_weather', '_get_time'],
    )

  def test_unknown_tool_is_reported_so_the_model_can_correct_itself(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weathr', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    answer = _run(agent, 'Weather in London?')

    with self.subTest('tells_the_model_what_went_wrong'):
      error = backend.seen_turns[1][-1].responses[0].error or ''
      self.assertIn('Unknown tool', error)
      self.assertIn('_get_weather', error)
    with self.subTest('the_exchange_survives_and_completes'):
      self.assertEqual(answer, 'It is 15 degrees.')

  def test_failing_tool_is_reported_rather_than_raised(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_explode', kwargs={'reason': 'boom'})
                ]
            ),
            _model_turn('Sorry, that failed.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_explode])

    answer = _run(agent, 'Please explode')

    with self.subTest('reports_the_error_to_the_model'):
      self.assertEqual(
          backend.seen_turns[1][-1].responses[0].error, 'ValueError: boom'
      )
    with self.subTest('completes_the_exchange'):
      self.assertEqual(answer, 'Sorry, that failed.')

  def test_stop_on_tool_error_fails_fast(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_explode', kwargs={'reason': 'boom'})
                ]
            )
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_explode], stop_on_tool_error=True)

    with self.assertRaisesRegex(ValueError, 'boom'):
      _run(agent, 'Please explode')

  def test_irrecoverable_error_type_aborts(self):
    backend = _ScriptedBackend(
        replies=[_model_turn(calls=[fc_types.FunctionCall('forbidden')])]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[
            llm_tool_use.Tool(
                name='forbidden',
                function=_forbidden,
                irrecoverable_error_types=['PermissionError'],
            )
        ]
    )

    with self.assertRaises(PermissionError):
      _run(agent, 'Do the forbidden thing')

  def test_hand_written_tool_keeps_its_policy_and_gains_parameters(self):
    tool = llm_tool_use.Tool(
        name='weather',
        function=_get_weather,
        description='Custom description.',
        irrecoverable_error_types=['PermissionError'],
    )
    agent = agent_lib.Agent(tools=[tool])

    with self.subTest('derives_the_parameters'):
      parameters = agent._specs[0].parameters or {}  # pylint: disable=protected-access
      self.assertIn('location', parameters['properties'])
    with self.subTest('keeps_the_authors_description'):
      self.assertEqual(agent._specs[0].description, 'Custom description.')  # pylint: disable=protected-access
    with self.subTest('keeps_the_error_policy'):
      self.assertEqual(
          agent._tools_by_name['weather'].irrecoverable_error_types,  # pylint: disable=protected-access
          ['PermissionError'],
      )

  def test_stops_after_max_steps(self):
    # A model that never stops asking for tools.
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            )
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather], max_steps=3)

    answer, state = _run_with_state(agent, 'Weather in London?')

    with self.subTest('performs_exactly_max_steps_model_calls'):
      self.assertLen(backend.seen_turns, 3)
    with self.subTest('reports_that_it_ran_out_of_steps'):
      self.assertTrue(agent.steps_exhausted(state))
    with self.subTest('returns_no_answer'):
      self.assertEqual('', answer)

  def test_a_finished_exchange_is_not_reported_as_exhausted(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather], max_steps=3)

    _, state = _run_with_state(agent, 'Hi')

    self.assertFalse(agent.steps_exhausted(state))

  def test_final_instruction_gives_one_last_call_without_tools(self):
    call = fc_types.FunctionCall('_get_weather', kwargs={'location': 'London'})
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[call]),
            _model_turn(calls=[call]),
            _model_turn('15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather], max_steps=2, final_instruction='Answer now.'
    )

    answer, state = _run_with_state(agent, 'Weather in London?')

    with self.subTest('returns_the_final_answer'):
      self.assertEqual(answer, '15 degrees.')
    with self.subTest('calls_the_model_once_more_than_max_steps'):
      self.assertLen(backend.seen_turns, 3)
    with self.subTest('offers_no_tools_on_the_final_call'):
      self.assertEqual(backend.seen_tools[2], [])
    with self.subTest('sends_the_instruction_last'):
      last_turn_sent = backend.seen_turns[2][-1]
      self.assertEqual(
          (last_turn_sent.role, last_turn_sent.content), (_USER, 'Answer now.')
      )
    with self.subTest('records_the_instruction_in_the_turns'):
      self.assertEqual(
          [turn.role for turn in state.turns],
          [_USER, _MODEL, _TOOL, _MODEL, _TOOL, _USER, _MODEL],
      )
    with self.subTest('is_not_reported_as_exhausted'):
      self.assertFalse(agent.steps_exhausted(state))

  def test_calls_requested_on_the_final_call_are_not_executed(self):
    # A model that keeps asking for tools even once it has none.
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            )
        ]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather], max_steps=1, final_instruction='Answer now.'
    )

    answer, state = _run_with_state(agent, 'Weather in London?')

    with self.subTest('returns_no_answer'):
      self.assertEqual('', answer)
    with self.subTest('stops_after_the_final_call'):
      self.assertLen(backend.seen_turns, 2)
    with self.subTest('does_not_answer_the_final_calls'):
      self.assertEqual(
          [turn.role for turn in state.turns],
          [_USER, _MODEL, _TOOL, _USER, _MODEL],
      )
    with self.subTest('reports_that_it_ran_out_of_steps'):
      self.assertTrue(agent.steps_exhausted(state))

  def test_final_instruction_is_unused_when_the_model_answers_in_time(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather], max_steps=3, final_instruction='Answer now.'
    )

    _, state = _run_with_state(agent, 'Hi')

    with self.subTest('calls_the_model_once'):
      self.assertLen(backend.seen_turns, 1)
    with self.subTest('does_not_record_the_instruction'):
      self.assertEqual([turn.role for turn in state.turns], [_USER, _MODEL])

  def test_empty_reply_stops_by_default(self):
    backend = _ScriptedBackend(replies=[_model_turn()])
    backend.register()
    agent = agent_lib.Agent(max_steps=3)

    _, state = _run_with_state(agent, 'Hi')

    self.assertLen(backend.seen_turns, 1)
    self.assertFalse(agent.steps_exhausted(state))

  @parameterized.named_parameters(
      dict(
          testcase_name='answers_directly',
          empty_turn=fc_types.ConversationTurn(role=_MODEL, thought='Ready.'),
          followup_turns=[_model_turn('15 degrees.')],
          expected_roles=[_USER, _MODEL, _USER, _MODEL],
      ),
      dict(
          testcase_name='calls_tool_first',
          empty_turn=_model_turn('   '),
          followup_turns=[
              _model_turn(
                  calls=[
                      fc_types.FunctionCall(
                          '_get_weather', kwargs={'location': 'London'}
                      )
                  ]
              ),
              _model_turn('15 degrees.'),
          ],
          expected_roles=[_USER, _MODEL, _USER, _MODEL, _TOOL, _MODEL],
      ),
  )
  def test_empty_reply_is_retried_with_tools(
      self, empty_turn, followup_turns, expected_roles
  ):
    backend = _ScriptedBackend(replies=[empty_turn, *followup_turns])
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather],
        max_steps=5,
        empty_reply_instruction=agent_lib.DEFAULT_EMPTY_REPLY_INSTRUCTION,
    )

    answer, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(answer, '15 degrees.')
    self.assertEqual(
        [spec.name for spec in backend.seen_tools[1]], ['_get_weather']
    )
    self.assertEqual(
        backend.seen_turns[1][-1].content,
        agent_lib.DEFAULT_EMPTY_REPLY_INSTRUCTION,
    )
    self.assertEqual([turn.role for turn in state.turns], expected_roles)

  def test_persistent_empty_replies_stop_at_max_steps(self):
    backend = _ScriptedBackend(replies=[_model_turn()])
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather],
        max_steps=3,
        empty_reply_instruction='Please reply.',
    )

    answer, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(answer, '')
    self.assertLen(backend.seen_turns, 3)
    self.assertEqual(backend.seen_turns[1][-1].content, 'Please reply.')
    self.assertTrue(agent.steps_exhausted(state))

  def test_a_single_step_exposes_the_calls_without_running_them(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('Should not get here.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    answer, state = _run_with_state(agent, 'Weather in London?', max_steps=1)

    with self.subTest('returns_no_answer'):
      self.assertEqual('', answer)
    with self.subTest('keeps_the_requested_calls_in_the_final_state'):
      self.assertEqual(state.turns[1].calls[0].function_name, '_get_weather')
    with self.subTest('does_not_call_the_model_again'):
      self.assertLen(backend.seen_turns, 1)

  def test_accepts_an_existing_conversation(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])
    prior_turns = [
        fc_types.ConversationTurn(role=_USER, content='Capital of France?'),
        _model_turn('Paris is the capital.'),
        fc_types.ConversationTurn(role=_USER, content='Say it in one word.'),
    ]

    answer = _run(agent, prior_turns)

    with self.subTest('keeps_the_conversation'):
      self.assertLen(backend.seen_turns[0], 3)
    with self.subTest('answers_the_last_question'):
      self.assertEqual(answer, 'Paris.')

  def test_accepts_a_multimodal_question(self):
    backend = _ScriptedBackend(replies=[_model_turn('A puffin.')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])
    question = content_lib.ChunkList([
        content_lib.Chunk(b'<png bytes>', content_type='image/png'),
        content_lib.Chunk('What bird is this?'),
    ])

    answer = _run(agent, question)

    self.assertEqual(
        backend.seen_turns[0],
        [fc_types.ConversationTurn(role=_USER, content=question)],
    )
    self.assertEqual(answer, 'A puffin.')

  def test_returns_an_answer_given_as_chunks_as_text(self):
    reply = fc_types.ConversationTurn(
        role=_MODEL,
        content=content_lib.ChunkList(
            [content_lib.Chunk('It is '), content_lib.Chunk('15 degrees.')]
        ),
    )
    backend = _ScriptedBackend(replies=[reply])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    self.assertEqual('It is 15 degrees.', _run(agent, 'Weather in London?'))

  def test_streams_one_update_per_step(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    initial_state = agent_lib.AgentState(
        turns=[fc_types.ConversationTurn(role=_USER, content='Weather?')]
    )

    # `executing.stream` rather than `executing.run`, so that we see every
    # update rather than only the last one.
    updates = cast(
        list[agent_lib.AgentStep],
        list(
            executing.stream(agent.stream_updates(initial_state=initial_state))
        ),
    )

    with self.subTest('yields_one_update_per_model_call'):
      self.assertLen(updates, 2)
    with self.subTest('the_first_update_carries_the_tool_results'):
      first_step_turns = updates[0].turns
      self.assertEqual(first_step_turns[-1].role, _TOOL)
      self.assertEqual(
          first_step_turns[-1].responses[0].response,
          {'location': 'London', 'temp': 15},
      )
    with self.subTest('the_last_update_is_the_final_answer'):
      self.assertEqual(updates[1].model_turn.content, 'It is 15 degrees.')
      self.assertIsNone(updates[1].tool_turn)

  def test_no_tools_declares_nothing(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent()

    answer = _run(agent, 'Hi')

    with self.subTest('advertises_no_tools'):
      self.assertEmpty(backend.seen_tools[0])
    with self.subTest('still_answers'):
      self.assertEqual(answer, 'Hello!')


class FCAgentToolSourceTest(absltest.TestCase):
  """Tests the tools that are fetched per run rather than held on the agent.

  These are the tools that cannot be built once: those reached over a
  connection, which are callable only while it is open. Such a toolset is
  named among `tools` as a source, and the agent is expected to open it around
  each run and close it again, so that a caller writes no `async with` of
  their own and the agent survives to answer the next question.
  """

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def test_tools_from_a_source_are_declared_and_callable(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_get_time', kwargs={'city': 'Paris'})
                ]
            ),
            _model_turn('It is noon in Paris.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather, _fake_source([_get_time], events=[])]
    )

    answer = _run(agent, 'Time in Paris?')

    with self.subTest('declares_them_alongside_the_agents_own'):
      self.assertEqual(
          sorted(spec.name for spec in backend.seen_tools[0]),
          ['_get_time', '_get_weather'],
      )
    with self.subTest('executes_the_one_from_the_source'):
      tool_turn = backend.seen_turns[1][-1]
      self.assertEqual(tool_turn.responses[0].response, '12:00 in Paris')
    with self.subTest('answers'):
      self.assertEqual(answer, 'It is noon in Paris.')

  def test_a_source_is_open_for_exactly_the_length_of_the_run(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_get_time', kwargs={'city': 'Oslo'})
                ]
            ),
            _model_turn('It is noon in Oslo.'),
        ]
    )
    backend.register()
    events: list[str] = []
    agent = agent_lib.Agent(tools=[_fake_source([_get_time], events)])

    answer = _run(agent, 'Time in Oslo?')

    with self.subTest('opens_it_and_closes_it_once'):
      self.assertEqual(events, ['open', 'close'])
    with self.subTest('the_tool_worked_while_it_was_open'):
      self.assertEqual(
          backend.seen_turns[1][-1].responses[0].response, '12:00 in Oslo'
      )
    with self.subTest('answers'):
      self.assertEqual(answer, 'It is noon in Oslo.')

  def test_a_second_run_opens_the_source_again(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    events: list[str] = []
    agent = agent_lib.Agent(tools=[_fake_source([_get_time], events)])

    _run(agent, 'Hi')
    _run(agent, 'Hi again')

    with self.subTest('opens_and_closes_it_once_per_run'):
      self.assertEqual(events, ['open', 'close', 'open', 'close'])
    with self.subTest('declares_the_same_tools_both_times'):
      # The tools of a run are not kept on the agent, so the second run
      # declares what the second opening yielded and not that twice over.
      self.assertEqual(
          [spec.name for spec in backend.seen_tools[1]], ['_get_time']
      )

  def test_a_source_is_closed_even_when_the_run_fails(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_explode', kwargs={'reason': 'no'})
                ]
            )
        ]
    )
    backend.register()
    events: list[str] = []
    agent = agent_lib.Agent(
        tools=[_fake_source([_explode], events)], stop_on_tool_error=True
    )

    with self.subTest('the_failure_reaches_the_caller'):
      with self.assertRaises(ValueError):
        _run(agent, 'Explode please')
    with self.subTest('the_source_is_closed_anyway'):
      self.assertEqual(events, ['open', 'close'])

  def test_a_source_replaces_a_tool_of_the_same_name(self):
    def _fresher_get_time(city: str) -> str:
      """Tells the time in a city, an hour later than the other one does.

      Args:
        city: Where to look.

      Returns:
        The time there.
      """
      return f'13:00 in {city}'

    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_get_time', kwargs={'city': 'Rome'})
                ]
            ),
            _model_turn('It is one in Rome.'),
        ]
    )
    backend.register()
    fresher = llm_tool_use.Tool(name='_get_time', function=_fresher_get_time)
    agent = agent_lib.Agent(
        tools=[_get_time, _fake_source([fresher], events=[])]
    )

    _run(agent, 'Time in Rome?')

    with self.subTest('declares_it_once'):
      self.assertEqual(
          [spec.name for spec in backend.seen_tools[0]], ['_get_time']
      )
    with self.subTest('calls_the_one_from_the_source'):
      tool_turn = backend.seen_turns[1][-1]
      self.assertEqual(tool_turn.responses[0].response, '13:00 in Rome')

  def test_a_run_without_a_source_opens_nothing(self):
    backend = _ScriptedBackend(replies=[_model_turn('Hello!')])
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    answer = _run(agent, 'Hi')

    with self.subTest('declares_the_tools_it_was_built_with'):
      self.assertEqual(
          [spec.name for spec in backend.seen_tools[0]], ['_get_weather']
      )
    with self.subTest('answers'):
      self.assertEqual(answer, 'Hello!')


class FCAgentTracingTest(absltest.TestCase):
  """Tests what a run leaves behind in the execution trace.

  What the agent records is one account of a whole step -- the thought, every
  call it asked for and what each of them returned -- for whoever reads the
  trace afterwards. The evaluation metrics do not read it: they match the
  builtin stage nested inside, so that they see a model call however it was
  reached.
  """

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def _step_stages(
      self, agent: agent_lib.Agent, inputs: str
  ) -> list[results.ExecutionResult]:
    """Runs `agent` and returns the trace stage of each of its steps."""
    _, execution_result = executing.run(agent(inputs), enable_tracing=True)
    stages: list[results.ExecutionResult] = []

    def collect(node: results.ExecutionResult) -> None:
      if node.stage_name == _STEP_STAGE_NAME:
        stages.append(node)
      for child in node.stages:
        collect(child)

    collect(execution_result)
    return stages

  def test_records_each_call_and_what_it_returned(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}, id='c1'
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    stages = self._step_stages(agent, 'Weather in London?')

    with self.subTest('records_one_stage_per_model_call'):
      self.assertLen(stages, 2)
    with self.subTest('counts_the_calls_of_each_step'):
      self.assertEqual(
          [stage.info[fc_types.TOOL_CALLS_INFO_KEY] for stage in stages], [1, 0]
      )
    with self.subTest('describes_the_call'):
      self.assertEqual(
          stages[0].info[fc_types.TOOL_DETAILS_INFO_KEY],
          [{
              'type': 'function_call',
              'name': '_get_weather',
              'args': {'location': 'London'},
              'id': 'c1',
              'response': {'location': 'London', 'temp': 15},
          }],
      )
    with self.subTest('records_no_details_for_a_step_without_calls'):
      self.assertNotIn(fc_types.TOOL_DETAILS_INFO_KEY, stages[1].info)

  def test_records_a_failed_call_as_an_error(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall('_explode', kwargs={'reason': 'boom'})
                ]
            ),
            _model_turn('Sorry, that did not work.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_explode])

    stages = self._step_stages(agent, 'Please explode')

    detail = stages[0].info[fc_types.TOOL_DETAILS_INFO_KEY][0]
    with self.subTest('still_counts_the_call'):
      self.assertEqual(stages[0].info[fc_types.TOOL_CALLS_INFO_KEY], 1)
    with self.subTest('records_why_it_failed'):
      self.assertEqual(detail['response'], {'error': 'ValueError: boom'})
    with self.subTest('omits_the_absent_id'):
      self.assertNotIn('id', detail)

  def test_records_the_models_reasoning(self):
    backend = _ScriptedBackend(
        replies=[
            fc_types.ConversationTurn(
                role=_MODEL,
                thought='I should look that up.',
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'Paris'}
                    )
                ],
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    stages = self._step_stages(agent, 'Weather in Paris?')

    details = stages[0].info[fc_types.TOOL_DETAILS_INFO_KEY]
    with self.subTest('records_the_thought_before_the_call_it_led_to'):
      self.assertEqual(
          details[0], {'type': 'thought', 'text': 'I should look that up.'}
      )
    with self.subTest('records_the_call_too'):
      self.assertEqual(details[1]['name'], '_get_weather')

  def test_records_a_result_that_is_not_plain_data(self):
    class _Opaque:

      def __repr__(self) -> str:
        return '<opaque>'

    def _opaque_tool() -> Any:
      """Returns something that is not JSON.

      Returns:
        An object with no structured form.
      """
      return _Opaque()

    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[fc_types.FunctionCall('_opaque_tool')]),
            _model_turn('Done.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_opaque_tool])

    stages = self._step_stages(agent, 'Go')

    detail = stages[0].info[fc_types.TOOL_DETAILS_INFO_KEY][0]
    # Recorded as it stands, like everything else the trace holds. Reducing it
    # here would throw away what a notebook renderer could have shown, and
    # would be a second answer to a question `onetwo.core.tracing` has already
    # answered for a stage's inputs and outputs.
    self.assertIsInstance(detail['response'], _Opaque)

  def test_formats_the_thoughts_and_calls_of_a_run(self):
    backend = _ScriptedBackend(
        replies=[
            fc_types.ConversationTurn(
                role=_MODEL,
                thought='I should look that up.',
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    ),
                    fc_types.FunctionCall(
                        '_explode', kwargs={'reason': 'boom'}
                    ),
                ],
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather, _explode])

    _, execution_result = executing.run(
        agent('Weather in London?'), enable_tracing=True
    )

    self.assertEqual(
        agent_lib.format_tool_calls(execution_result),
        "thought: I should look that up.\n"
        "_get_weather({'location': 'London'})\n"
        "  -> {'location': 'London', 'temp': 15}\n"
        "_explode({'reason': 'boom'})\n"
        '  !! ValueError: boom',
    )

  def test_format_cuts_long_results(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    _, execution_result = executing.run(
        agent('Weather in London?'), enable_tracing=True
    )

    with self.subTest('keeps_the_ellipsis_within_max_chars'):
      self.assertEqual(
          agent_lib.format_tool_calls(execution_result, max_chars=10),
          "_get_weather({'location': 'London'})\n  -> {'locat...",
      )
    with self.subTest('cuts_without_an_ellipsis_when_it_does_not_fit'):
      self.assertEqual(
          agent_lib.format_tool_calls(execution_result, max_chars=2),
          "_get_weather({'location': 'London'})\n  -> {'",
      )

  def test_format_shows_every_kind_of_detail_a_backend_records(self):
    # As a backend that runs the tools itself records them: on the model call,
    # possibly with a response it could not match to any call.
    trace = results.ExecutionResult(
        stage_name='llm.generate_text',
        info={
            fc_types.TOOL_DETAILS_INFO_KEY: [
                fc_types.FunctionCallDetail(
                    name='now', response='12:00'
                ).to_dict(),
                fc_types.FunctionResponseDetail(
                    name='search', response='found'
                ).to_dict(),
            ]
        },
    )

    self.assertEqual(
        agent_lib.format_tool_calls(trace),
        'now()\n  -> 12:00\nsearch (no matching call)\n  -> found',
    )

  def test_format_rejects_a_detail_of_unknown_type(self):
    trace = results.ExecutionResult(
        stage_name='llm.generate_text',
        info={fc_types.TOOL_DETAILS_INFO_KEY: [{'type': 'something_new'}]},
    )

    with self.assertRaisesRegex(ValueError, 'something_new'):
      agent_lib.format_tool_calls(trace)


class FormatAgentStepsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def test_formats_each_call_and_the_answer(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    ),
                    fc_types.FunctionCall(
                        '_explode', kwargs={'reason': 'boom'}
                    ),
                ]
            ),
            _model_turn(
                calls=[fc_types.FunctionCall('_get_time', args=('Paris',))]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather, _explode, _get_time])

    _, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(
        agent_lib.format_agent_steps(state),
        "1. _get_weather('London') -> {'location': 'London', 'temp': 15}\n"
        "1. _explode('boom') -> error: ValueError: boom\n"
        "2. _get_time('Paris') -> 12:00 in Paris\n"
        '3. No tool call, the model answers: It is 15 degrees.',
    )

  def test_uses_the_given_result_formatter(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    _, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(
        agent_lib.format_agent_steps(
            state, format_result=lambda result: f"{result['temp']} degrees"
        ),
        "1. _get_weather('London') -> 15 degrees\n"
        '2. No tool call, the model answers: It is 15 degrees.',
    )

  def test_cuts_long_results_and_answers(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
            _model_turn('It is 15 degrees.'),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(tools=[_get_weather])

    _, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(
        agent_lib.format_agent_steps(state, max_chars=10),
        "1. _get_weather('London') -> {'locat...\n"
        '2. No tool call, the model answers: It is 1...',
    )

  def test_shows_calls_that_were_not_executed(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                calls=[
                    fc_types.FunctionCall(
                        '_get_weather', kwargs={'location': 'London'}
                    )
                ]
            ),
        ]
    )
    backend.register()
    agent = agent_lib.Agent(
        tools=[_get_weather], max_steps=1, final_instruction='Answer now.'
    )

    _, state = _run_with_state(agent, 'Weather in London?')

    self.assertEqual(
        agent_lib.format_agent_steps(state),
        "1. _get_weather('London') -> {'location': 'London', 'temp': 15}\n"
        "2. _get_weather('London') -> not executed",
    )


if __name__ == '__main__':
  absltest.main()
