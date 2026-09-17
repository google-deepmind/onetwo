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

"""Tests for the core function calling data types."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import content as content_lib
from onetwo.core import function_calling


class _Subclass(function_calling.FunctionCall):
  """Stands in for the compatibility subclass in `llm_tool_use`."""


@dataclasses.dataclass(frozen=True)
class _RetrievalResult:
  """Stands in for a rich (non-string) value returned by a local tool.

  Real examples live outside `onetwo.core` (e.g. `stdlib.qa.StrategyResult`),
  which this module must not depend on, so they are represented here by a
  minimal equivalent.
  """

  answer: str = ''
  doc_ids: tuple[str, ...] = ()


class FunctionCallTest(parameterized.TestCase):

  def test_defaults(self):
    call = function_calling.FunctionCall()

    self.assertEqual('', call.function_name)
    self.assertEqual((), call.args)
    self.assertEqual({}, call.kwargs)

  def test_positional_construction_matches_field_order(self):
    # Guards the field order, which positional callers rely on.
    call = function_calling.FunctionCall('Search', ('a',), {'b': 1})

    self.assertEqual(
        function_calling.FunctionCall(
            function_name='Search', args=('a',), kwargs={'b': 1}
        ),
        call,
    )

  def test_repr_is_unchanged_by_the_move(self):
    # Several callers serialise a `FunctionCall` by its repr and assert on the
    # resulting string, so the repr is effectively part of the public API.
    # Adding or reordering a field will break them; see b/561535278.
    self.assertEqual(
        "FunctionCall(function_name='Finish', args=('4',), kwargs={})",
        repr(function_calling.FunctionCall('Finish', ('4',), {})),
    )

  @parameterized.named_parameters(
      ('function_name', {'function_name': 'Other'}),
      ('args', {'args': ('z',)}),
      ('kwargs', {'kwargs': {'b': 2}}),
  )
  def test_not_equal_when_any_field_differs(self, overrides):
    call = function_calling.FunctionCall(
        function_name='Search', args=('a',), kwargs={'b': 1}
    )

    self.assertNotEqual(call, dataclasses.replace(call, **overrides))

  def test_equal_to_subclass_with_same_values(self):
    # The compatibility subclass in `llm_tool_use` must be interchangeable with
    # the core type while callers are being migrated, in both directions.
    base = function_calling.FunctionCall(function_name='Search', args=('a',))
    subclass = _Subclass(function_name='Search', args=('a',))

    self.assertEqual(base, subclass)
    self.assertEqual(subclass, base)

  def test_not_equal_to_unrelated_type(self):
    self.assertNotEqual(function_calling.FunctionCall(), 'Search()')

  def test_is_unhashable(self):
    # Matches the behaviour of the mutable dataclass this type replaces.
    with self.assertRaises(TypeError):
      hash(function_calling.FunctionCall())

  def test_is_mutable(self):
    call = function_calling.FunctionCall()

    call.function_name = 'Search'

    self.assertEqual('Search', call.function_name)


class FunctionResponseTest(parameterized.TestCase):

  def test_defaults(self):
    response = function_calling.FunctionResponse()

    self.assertEqual('', response.function_name)
    self.assertIsNone(response.response)
    self.assertIsNone(response.error)

  def test_equal_when_all_fields_match(self):
    self.assertEqual(
        function_calling.FunctionResponse(
            function_name='Search', response=['a']
        ),
        function_calling.FunctionResponse(
            function_name='Search', response=['a']
        ),
    )

  @parameterized.named_parameters(
      ('function_name', {'function_name': 'Other'}),
      ('response', {'response': ['z']}),
      ('error', {'error': 'boom'}),
  )
  def test_not_equal_when_any_field_differs(self, overrides):
    response = function_calling.FunctionResponse(
        function_name='Search', response=['a']
    )

    self.assertNotEqual(response, dataclasses.replace(response, **overrides))

  def test_error_is_represented_rather_than_raised(self):
    response = function_calling.FunctionResponse(
        function_name='Search', error='tool unavailable'
    )

    self.assertIsNone(response.response)
    self.assertEqual('tool unavailable', response.error)


class ToolSpecTest(parameterized.TestCase):

  def test_defaults(self):
    spec = function_calling.ToolSpec(name='search')

    self.assertEqual('search', spec.name)
    self.assertEqual('', spec.description)
    self.assertIsNone(spec.parameters)

  def test_custom_values(self):
    params = {'type': 'OBJECT', 'properties': {'query': {'type': 'STRING'}}}
    spec = function_calling.ToolSpec(
        name='search',
        description='Search documents.',
        parameters=params,
    )

    self.assertEqual('search', spec.name)
    self.assertEqual('Search documents.', spec.description)
    self.assertEqual(params, spec.parameters)

  def test_is_frozen(self):
    spec = function_calling.ToolSpec(name='search')
    with self.assertRaises(dataclasses.FrozenInstanceError):
      spec.name = 'other'  # pyrefly: ignore[read-only]


class ConversationRoleTest(parameterized.TestCase):

  def test_values(self):
    self.assertEqual('user', function_calling.ConversationRole.USER)
    self.assertEqual('model', function_calling.ConversationRole.MODEL)
    self.assertEqual('tool', function_calling.ConversationRole.TOOL)

  def test_string_comparison(self):
    self.assertEqual('user', function_calling.ConversationRole.USER.value)
    self.assertEqual(function_calling.ConversationRole.USER, 'user')


class ConversationTurnTest(parameterized.TestCase):

  def test_defaults(self):
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER
    )

    self.assertEqual(function_calling.ConversationRole.USER, turn.role)
    self.assertIsNone(turn.content)
    self.assertEqual((), turn.calls)
    self.assertEqual((), turn.responses)
    self.assertIsNone(turn.thought)

  def test_model_turn_with_calls_and_thought(self):
    call = function_calling.FunctionCall(
        function_name='search', kwargs={'q': 'ai'}
    )
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.MODEL,
        calls=(call,),
        thought='Need to search first.',
    )

    self.assertEqual(function_calling.ConversationRole.MODEL, turn.role)
    self.assertEqual((call,), turn.calls)
    self.assertEqual((), turn.responses)
    self.assertEqual('Need to search first.', turn.thought)
    self.assertIsNone(turn.content)

  def test_tool_turn_with_responses(self):
    response = function_calling.FunctionResponse(
        function_name='search', response=['doc1']
    )
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.TOOL,
        responses=(response,),
    )

    self.assertEqual(function_calling.ConversationRole.TOOL, turn.role)
    self.assertEqual((), turn.calls)
    self.assertEqual((response,), turn.responses)

  def test_user_turn_with_chunk_list(self):
    chunks = content_lib.ChunkList([content_lib.Chunk('hello')])
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER,
        content=chunks,
    )

    self.assertEqual(function_calling.ConversationRole.USER, turn.role)
    self.assertEqual(chunks, turn.content)

  def test_is_frozen(self):
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER
    )
    with self.assertRaises(dataclasses.FrozenInstanceError):
      turn.thought = 'new thought'  # pyrefly: ignore[read-only]


class ToolShapesTest(parameterized.TestCase):
  """Shows how the types express the tools we expect agents to be given.

  These tests are illustrative rather than exhaustive: each one transcribes a
  shape of tool that exists today (a knowledge-graph tool served over MCP, a
  web search tool, a Python sandbox) and checks that it is representable
  without extending the types.
  """

  def test_mcp_tool_with_flat_scalar_parameters(self):
    spec = function_calling.ToolSpec(
        name='find_paths_between_entities',
        description=(
            'Find multi-hop connection paths between two entities in the'
            ' knowledge graph up to max_depth hops.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'source_entity_id': {
                    'type': 'string',
                    'description': 'Starting entity ID.',
                },
                'target_entity_id': {
                    'type': 'string',
                    'description': 'Target entity ID.',
                },
                'max_depth': {
                    'type': 'integer',
                    'description': 'Maximum number of hops to explore.',
                    'default': 3,
                },
                'max_paths': {
                    'type': 'integer',
                    'description': 'Maximum number of paths to return.',
                    'default': 5,
                },
            },
            'required': ['source_entity_id', 'target_entity_id'],
        },
    )

    schema = spec.parameters or {}
    self.assertEqual(
        ['source_entity_id', 'target_entity_id'],
        schema['required'],
    )
    # Optional parameters carry their default in the schema rather than being
    # omitted, so the model can see the behaviour it gets by not passing them.
    self.assertEqual(3, schema['properties']['max_depth']['default'])

  def test_tool_with_a_single_string_parameter(self):
    # The shape shared by `search(query)` and the Python sandbox's
    # `tool_code(code)`: one required string, description carried by the
    # docstring of the underlying callable.
    spec = function_calling.ToolSpec(
        name='search',
        description='Searches the web and returns a relevant snippet.',
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query.',
                },
            },
            'required': ['query'],
        },
    )

    call = function_calling.FunctionCall(
        function_name='search', kwargs={'query': 'who won the 2022 world cup'}
    )

    schema = spec.parameters or {}
    self.assertEqual(['query'], list(schema['properties']))
    self.assertEqual(spec.name, call.function_name)

  def test_tool_with_no_parameters(self):
    # A nullary tool is expressed by omitting `parameters` entirely, rather
    # than by an empty schema, so that backends that reject an empty property
    # list can simply leave the field unset.
    spec = function_calling.ToolSpec(
        name='get_graph_statistics',
        description='Returns entity and relation counts for the graph.',
    )

    self.assertIsNone(spec.parameters)

  def test_tool_with_an_enumerated_parameter(self):
    # `get_entity_neighbors` accepts one of three directions. Declaring them as
    # a JSON Schema `enum` is what stops the model from inventing a fourth.
    spec = function_calling.ToolSpec(
        name='get_entity_neighbors',
        description='Retrieve 1-hop connected entities.',
        parameters={
            'type': 'object',
            'properties': {
                'entity_id': {'type': 'string'},
                'direction': {
                    'type': 'string',
                    'enum': ['both', 'outgoing', 'incoming'],
                    'default': 'both',
                },
                'limit': {'type': 'integer', 'default': 25},
            },
            'required': ['entity_id'],
        },
    )

    schema = spec.parameters or {}
    self.assertEqual(
        ['both', 'outgoing', 'incoming'],
        schema['properties']['direction']['enum'],
    )

  def test_model_turn_can_request_several_calls_in_parallel(self):
    # Models routinely emit independent calls in a single turn, including two
    # calls to the same tool, so `calls` is a sequence rather than a single
    # optional call.
    turn = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.MODEL,
        thought='Look up both entities before connecting them.',
        calls=(
            function_calling.FunctionCall(
                function_name='search_entities',
                kwargs={'query': 'Samuel Namara'},
            ),
            function_calling.FunctionCall(
                function_name='search_entities',
                kwargs={'query': 'Firuzabad'},
            ),
        ),
    )

    self.assertLen(turn.calls, 2)
    # Both calls name the same tool, so responses cannot be matched back to
    # calls by name alone; in follow-up CL, a function call ID will be added for
    # that purpose.
    self.assertEqual(
        {'search_entities'}, {call.function_name for call in turn.calls}
    )

  def test_response_can_hold_a_rich_python_value(self):
    # A locally executed tool may return an arbitrary Python object. Keeping
    # `response` untyped lets the value reach the caller intact; serialising it
    # for the model is the concern of whatever renders the turn.
    result = _RetrievalResult(answer='42', doc_ids=('doc1', 'doc2'))

    response = function_calling.FunctionResponse(
        function_name='answer_question', response=result
    )

    self.assertEqual(result, response.response)
    self.assertEqual(('doc1', 'doc2'), response.response.doc_ids)

  def test_failed_call_is_reported_back_to_the_model(self):
    # Tool failures are values, not exceptions, so that the loop can hand the
    # error to the model and let it retry just that call.
    response = function_calling.FunctionResponse(
        function_name='find_paths_between_entities',
        error="Source entity 'entity_unknown' not found in graph.",
    )

    self.assertIsNone(response.response)
    self.assertStartsWith(response.error, 'Source entity')

  def test_multi_hop_trajectory(self):
    # The end-to-end shape an agent accumulates: the question, the calls the
    # model made, what the tools returned, and the final answer, all in one
    # ordered sequence with nothing pushed into out-of-band state.
    trajectory = (
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER,
            content='How is Samuel Namara connected to Firuzabad?',
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.MODEL,
            thought='Resolve the entity id first.',
            calls=(
                function_calling.FunctionCall(
                    function_name='search_entities',
                    kwargs={'query': 'Samuel Namara', 'max_results': 1},
                ),
            ),
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.TOOL,
            responses=(
                function_calling.FunctionResponse(
                    function_name='search_entities',
                    response='[{"entity_id": "entity_samuel namara_person"}]',
                ),
            ),
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.MODEL,
            calls=(
                function_calling.FunctionCall(
                    function_name='find_paths_between_entities',
                    kwargs={
                        'source_entity_id': 'entity_samuel namara_person',
                        'target_entity_id': 'entity_firuzabad_geo',
                    },
                ),
            ),
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.TOOL,
            responses=(
                function_calling.FunctionResponse(
                    function_name='find_paths_between_entities',
                    response=(
                        '{"paths": ["(Samuel Namara) -[detained in]->'
                        ' (Firuzabad)"]}'
                    ),
                ),
            ),
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.MODEL,
            content='Samuel Namara was detained in Firuzabad.',
        ),
    )

    self.assertEqual(
        [
            function_calling.ConversationRole.USER,
            function_calling.ConversationRole.MODEL,
            function_calling.ConversationRole.TOOL,
            function_calling.ConversationRole.MODEL,
            function_calling.ConversationRole.TOOL,
            function_calling.ConversationRole.MODEL,
        ],
        [turn.role for turn in trajectory],
    )
    # The loop terminates on a model turn that requests no further calls.
    self.assertEmpty(trajectory[-1].calls)
    self.assertEqual(
        'Samuel Namara was detained in Firuzabad.', trajectory[-1].content
    )


if __name__ == '__main__':
  absltest.main()
