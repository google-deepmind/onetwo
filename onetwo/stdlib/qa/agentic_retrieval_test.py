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

"""Tests for the agentic QA strategy."""

from collections.abc import AsyncIterator, Callable, Sequence
import contextlib
import dataclasses
import functools
import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.stdlib.qa import agentic_retrieval
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.tool_use import llm_tool_use

_Document = retrieval_data_structures.Document

_USER = fc_types.ConversationRole.USER
_MODEL = fc_types.ConversationRole.MODEL

_DOCS = {
    'd1': _Document(
        doc_id='d1',
        title='France',
        content='Paris is the capital of France.',
    ),
    'd2': _Document(
        doc_id='d2',
        title='Germany',
        content='Berlin is the capital of Germany.',
    ),
    'd3': _Document(
        doc_id='d3', title='Spain', content='Madrid is the capital of Spain.'
    ),
}


def _corpus_tools(
    results_by_query: dict[str, list[_Document]],
    queries: list[str] | None = None,
) -> list[Callable[..., Any]]:
  """Returns tools over a fixed corpus, for the span of one question.

  The strategy supplies no tools of its own, so a test has to bring the kind
  of thing it is meant to be given. This is a stand-in for whatever a caller
  writes to expose their own corpus, and it is deliberately the awkward kind:
  `read` can only return what `search` has already found, which is what makes
  the pair worth rebuilding per question rather than sharing.

  Results are keyed by query rather than matched against content, so that a
  test can say exactly what the model's second search turns up without having
  to invent text that happens to match it.

  Args:
    results_by_query: What each query finds.
    queries: If given, every query is appended to it, so that a test can see how
      often the corpus was consulted across questions.

  Returns:
    The `search` and `read` tools, declared from the signatures below.
  """
  found: dict[str, _Document] = {}

  def search(query: str) -> list[dict[str, str]]:
    """Searches the corpus for passages relevant to a query.

    Args:
      query: What to search for.

    Returns:
      One entry per passage, each with its `doc_id`, its `title` and a summary.
    """
    if queries is not None:
      queries.append(query)
    hits = []
    for doc in results_by_query.get(query, []):
      found.setdefault(doc.doc_id, doc)
      hits.append(
          {'doc_id': doc.doc_id, 'title': doc.title, 'snippet': doc.text}
      )
    return hits

  def read(doc_id: str) -> str:
    """Returns the full text of a passage an earlier search found.

    Args:
      doc_id: The `doc_id` of a passage that `search` returned.

    Returns:
      The full text of that passage.

    Raises:
      ValueError: If no search has returned that passage.
    """
    doc = found.get(doc_id)
    if doc is None:
      known = ', '.join(found) or 'none yet'
      raise ValueError(
          f'No passage {doc_id!r} has been found. Found so far: {known}.'
      )
    return doc.text

  return [search, read]


def _source(
    make_tools: Callable[[], Sequence[Callable[..., Any]]],
    events: list[str] | None = None,
) -> llm_tool_use.ToolSource:
  """Returns tools built afresh for each question, let go once it is answered.

  Stands in for whatever a caller writes to expose a toolset that cannot be
  shared: one that remembers what it was asked, or one that holds a connection.

  Args:
    make_tools: Called once per question, for the tools of that question.
    events: If given, `'open'` and `'close'` are appended to it as they happen,
      so that a test can say when each did rather than only that it did.

  Returns:
    A source yielding the tools of one question.
  """

  @contextlib.asynccontextmanager
  async def opened() -> AsyncIterator[Sequence[Callable[..., Any]]]:
    if events is not None:
      events.append('open')
    try:
      yield make_tools()
    finally:
      if events is not None:
        events.append('close')

  return llm_tool_use.ToolSource(open=opened)


def _corpus_source(
    results_by_query: dict[str, list[_Document]],
    queries: list[str] | None = None,
) -> llm_tool_use.ToolSource:
  """Returns a source of the corpus tools, rebuilt for each question.

  Args:
    results_by_query: What each query finds.
    queries: If given, every query is appended to it, so that a test can see how
      often the corpus was consulted across questions.

  Returns:
    A source yielding the `search` and `read` pair of one question.
  """
  return _source(functools.partial(_corpus_tools, results_by_query, queries))


@dataclasses.dataclass
class _ScriptedBackend:
  """A backend that replies with a fixed script, and records what it was sent.

  The strategy reaches a model only through `llm.generate_with_tools`, so
  scripting that one builtin is enough to drive any exchange.
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


def _call(name: str, **kwargs) -> fc_types.FunctionCall:
  return fc_types.FunctionCall(function_name=name, kwargs=kwargs)


def _model_turn(
    content: str | None = None, calls: Sequence[fc_types.FunctionCall] = ()
) -> fc_types.ConversationTurn:
  return fc_types.ConversationTurn(role=_MODEL, content=content, calls=calls)


def _responses_in(
    conversation: Sequence[fc_types.ConversationTurn],
) -> list[fc_types.FunctionResponse]:
  """Returns every tool response in a conversation, in the order they occur."""
  responses = []
  for turn in conversation:
    responses.extend(turn.responses)
  return responses


def _add(left: int, right: int) -> int:
  """Adds two numbers.

  Args:
    left: The first number.
    right: The second number.

  Returns:
    Their sum.
  """
  return left + right


class AgenticRetrievalStrategyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def test_runs_with_tools_built_outside_the_strategy(self):
    # The corpus is wired up here, by the caller, and the strategy is handed
    # tools that know nothing of each other.
    queries = []
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='capital of France')]),
            _model_turn(calls=[_call('read', doc_id='d1')]),
            _model_turn('Paris.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'capital of France': [_DOCS['d1']]}, queries)],
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('returns_the_final_answer'):
      self.assertEqual(result.answer, 'Paris.')
    with self.subTest('searches_for_what_the_model_asked_for'):
      self.assertEqual(queries, ['capital of France'])
    with self.subTest('declares_the_tools_it_was_given'):
      self.assertEqual(
          sorted(spec.name for spec in backend.seen_tools[0]),
          ['read', 'search'],
      )
    with self.subTest('a_tool_result_reaches_the_model_unchanged'):
      self.assertEqual(
          _responses_in(backend.seen_turns[-1])[-1].response,
          _DOCS['d1'].text,
      )
    with self.subTest('puts_the_question_in_the_first_turn'):
      first_turn = backend.seen_turns[0][0]
      self.assertEqual(first_turn.role, _USER)
      self.assertIn('What is the capital of France?', str(first_turn.content))
    with self.subTest('sends_the_default_prompt'):
      self.assertEqual(
          first_turn.content,
          agentic_retrieval.DEFAULT_MCP_QUERY_TEMPLATE.format(
              question='What is the capital of France?',
              additional_instructions='',
          ),
      )

  def test_each_question_gets_its_own_tools(self):
    # `read` can only return what `search` found, so a second question that
    # reads the first question's document must be told it does not have it:
    # the two questions do not share what they retrieved.
    queries = []
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='france')]),
            _model_turn('Paris.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]}, queries)]
    )

    executing.run(strategy('What is the capital of France?'))
    backend.replies = [
        _model_turn(calls=[_call('read', doc_id='d1')]),
        _model_turn('I do not know.'),
    ]
    backend.seen_turns.clear()
    executing.run(strategy('And of Germany?'))

    with self.subTest('the_second_question_cannot_read_the_firsts_passage'):
      error = _responses_in(backend.seen_turns[-1])[-1].error or ''
      self.assertIn('d1', error)
    with self.subTest('the_corpus_itself_is_shared'):
      # One corpus served both questions; only the tools were rebuilt.
      self.assertEqual(queries, ['france'])

  def test_a_source_is_opened_for_the_question_and_closed_afterwards(self):
    # The shape a connection-backed toolset needs: opened for the question and
    # closed once it is answered.
    events = []
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('_add', left=2, right=2)]),
            _model_turn('Four.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_source(lambda: [_add], events)]
    )

    result = executing.run(strategy('What is two plus two?'))

    with self.subTest('the_session_supplies_the_tools'):
      self.assertEqual([spec.name for spec in backend.seen_tools[0]], ['_add'])
    with self.subTest('the_model_can_use_them'):
      self.assertEqual(result.answer, 'Four.')
    with self.subTest('the_session_is_opened_once_and_closed'):
      self.assertEqual(events, ['open', 'close'])

    executing.run(strategy('And again?'))

    with self.subTest('a_second_question_opens_its_own_session'):
      self.assertEqual(events, ['open', 'close', 'open', 'close'])

  def test_reports_no_documents_even_when_tools_return_them(self):
    # Deliberate, and worth pinning: a tool here can return anything, and the
    # strategy makes no attempt to recognise which of it was a document.
    # Recovering them by shape is easy and would be wrong -- see the module
    # docstring for what has to be settled before `docs` can be filled.
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='capital of France')]),
            _model_turn('Paris.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'capital of France': [_DOCS['d1']]})]
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('still_answers'):
      self.assertEqual(result.answer, 'Paris.')
    with self.subTest('the_tool_did_return_a_document'):
      returned = _responses_in(backend.seen_turns[-1])[-1].response
      self.assertEqual([hit['doc_id'] for hit in returned], ['d1'])
    with self.subTest('and_it_is_not_reported'):
      self.assertEmpty(result.docs)

  def test_reports_the_documents_found_in_the_answer_when_asked(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='capital of France')]),
            _model_turn('Paris [d1], not Berlin [d2].'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'capital of France': [_DOCS['d1']]})],
        doc_ids_from_answer=lambda answer: re.findall(r'\[(\w+)\]', answer),
    )

    result = executing.run(strategy('What is the capital of France?'))

    self.assertEqual([doc.doc_id for doc in result.docs], ['d1', 'd2'])

  def test_does_not_look_for_documents_in_an_empty_answer(self):
    backend = _ScriptedBackend(
        replies=[_model_turn(calls=[_call('search', query='france')])]
    )
    backend.register()
    answers_seen = []
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]})],
        max_steps=1,
        doc_ids_from_answer=lambda answer: answers_seen.append(answer) or [],
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('reports_no_documents'):
      self.assertEmpty(result.docs)
    with self.subTest('never_calls_the_parser'):
      self.assertEmpty(answers_seen)

  def test_mixes_shared_tools_with_per_question_ones(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_add, _corpus_source({})]
    )

    executing.run(strategy('What is the capital of France?'))

    self.assertEqual(
        sorted(spec.name for spec in backend.seen_tools[0]),
        ['_add', 'read', 'search'],
    )

  def test_a_shared_tool_is_callable_like_a_per_question_one(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('_add', left=2, right=2)]),
            _model_turn('Four.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(tools=[_add])

    result = executing.run(strategy('What is two plus two?'))

    self.assertEqual(result.answer, 'Four.')

  def test_a_strategy_with_no_tools_still_answers(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy()

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('returns_the_answer'):
      self.assertEqual(result.answer, 'Paris.')
    with self.subTest('declares_no_tools'):
      self.assertEmpty(backend.seen_tools[0])

  def test_the_model_may_search_as_many_times_as_it_wants(self):
    # The point of the loop: the second query is one the caller never wrote
    # and the model only arrived at after seeing what the first returned.
    queries = []
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='capitals')]),
            _model_turn(calls=[_call('search', query='more capitals')]),
            _model_turn('Paris, Berlin and Madrid.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[
            _corpus_source(
                {
                    'capitals': [_DOCS['d2'], _DOCS['d1']],
                    'more capitals': [_DOCS['d1'], _DOCS['d3']],
                },
                queries,
            )
        ]
    )

    result = executing.run(strategy('Name some capitals.'))

    with self.subTest('every_query_reaches_the_corpus'):
      self.assertEqual(queries, ['capitals', 'more capitals'])
    with self.subTest('and_the_last_word_is_the_answer'):
      self.assertEqual(result.answer, 'Paris, Berlin and Madrid.')

  def test_a_tool_error_is_reported_to_the_model(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='france')]),
            _model_turn(calls=[_call('read', doc_id='d9')]),
            _model_turn(calls=[_call('read', doc_id='d1')]),
            _model_turn('Paris.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]})]
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('the_exchange_survives_the_failure'):
      self.assertEqual(result.answer, 'Paris.')
    with self.subTest('the_model_is_told_what_went_wrong'):
      # The second step is the one that read a document that does not exist;
      # its error reaches the model in the conversation sent next.
      responses = _responses_in(backend.seen_turns[2])
      error = responses[-1].error or ''
      self.assertIn('d9', error)

  def test_a_question_that_needs_no_search_is_answered_directly(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({})]
    )

    result = executing.run(strategy('What is the capital of France?'))

    self.assertEqual(result.answer, 'Paris.')

  def test_running_out_of_steps_returns_no_answer(self):
    backend = _ScriptedBackend(
        replies=[_model_turn(calls=[_call('search', query='france')])]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]})],
        max_steps=2,
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('stops_at_the_bound'):
      self.assertLen(backend.seen_turns, 2)
    with self.subTest('has_no_answer_to_report'):
      self.assertEmpty(result.answer)

  def test_a_final_instruction_forces_an_answer_at_the_bound(self):
    backend = _ScriptedBackend(
        replies=[
            _model_turn(calls=[_call('search', query='france')]),
            _model_turn(calls=[_call('search', query='france')]),
            _model_turn('Paris.'),
        ]
    )
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]})],
        max_steps=2,
        final_instruction='Answer now.',
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('makes_one_more_call'):
      self.assertLen(backend.seen_turns, 3)
    with self.subTest('reports_the_forced_answer'):
      self.assertEqual(result.answer, 'Paris.')

  def test_an_empty_reply_instruction_retries_empty_turns(self):
    backend = _ScriptedBackend(replies=[_model_turn(), _model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        empty_reply_instruction='Please answer.',
    )

    result = executing.run(strategy('What is the capital of France?'))

    self.assertEqual(result.answer, 'Paris.')
    self.assertEqual(backend.seen_turns[1][-1].content, 'Please answer.')

  def test_text_alongside_a_pending_call_is_not_the_answer(self):
    # On reaching `max_steps` the last turn may still request a tool call,
    # with some thinking aloud next to it; that text is not a final answer.
    backend = _ScriptedBackend(
        replies=[
            _model_turn(
                'Let me search for [d1].',
                calls=[_call('search', query='france')],
            )
        ]
    )
    backend.register()
    answers_seen = []
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_corpus_source({'france': [_DOCS['d1']]})],
        max_steps=1,
        doc_ids_from_answer=lambda answer: answers_seen.append(answer) or [],
    )

    result = executing.run(strategy('What is the capital of France?'))

    with self.subTest('has_no_answer_to_report'):
      self.assertEmpty(result.answer)
    with self.subTest('reports_no_documents'):
      self.assertEmpty(result.docs)
    with self.subTest('never_calls_the_parser'):
      self.assertEmpty(answers_seen)

  def test_a_query_template_is_filled_in(self):
    template = 'Look it up.{additional_instructions}\nQ: {question} {{end}}'
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        query_template=template
    )

    executing.run(strategy('Capital of France?'))
    without_additional = backend.seen_turns[0][0].content
    backend.seen_turns.clear()
    executing.run(
        strategy('Capital of France?', additional_instructions='Be brief.')
    )
    with_additional = backend.seen_turns[0][0].content

    with self.subTest('fills_in_the_question'):
      self.assertEqual(
          without_additional, 'Look it up.\nQ: Capital of France? {end}'
      )
    with self.subTest('puts_additional_instructions_on_their_own_line'):
      self.assertEqual(
          with_additional, 'Look it up.\nBe brief.\nQ: Capital of France? {end}'
      )

  def test_additional_instructions_field_and_call_time_override(self):
    template = 'Look it up.{additional_instructions}\nQ: {question}'
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        query_template=template,
        additional_instructions='Use search first.',
    )

    executing.run(strategy('Capital of France?'))
    from_field = backend.seen_turns[0][0].content
    backend.seen_turns.clear()
    executing.run(
        strategy('Capital of France?', additional_instructions='Be brief.')
    )
    from_call = backend.seen_turns[0][0].content
    backend.seen_turns.clear()
    executing.run(strategy('Capital of France?', additional_instructions=''))
    cleared_by_call = backend.seen_turns[0][0].content

    with self.subTest('uses_strategy_field_by_default'):
      self.assertEqual(
          from_field, 'Look it up.\nUse search first.\nQ: Capital of France?'
      )
    with self.subTest('call_time_value_overrides_field'):
      self.assertEqual(
          from_call, 'Look it up.\nBe brief.\nQ: Capital of France?'
      )
    with self.subTest('empty_call_time_value_clears_field'):
      self.assertEqual(cleared_by_call, 'Look it up.\nQ: Capital of France?')

  def test_a_query_template_takes_the_text_of_a_chunk_list_question(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        query_template='{additional_instructions}Q: {question} {{end}}'
    )
    question = content_lib.ChunkList(['Capital of ', 'France?'])

    executing.run(strategy(question))

    self.assertEqual(
        backend.seen_turns[0][0].content, 'Q: Capital of France? {end}'
    )

  def test_a_query_template_without_a_question_is_rejected(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        query_template='{additional_instructions}'
    )

    with self.assertRaisesRegex(ValueError, 'question'):
      executing.run(strategy('Capital of France?'))

  def test_a_query_template_need_not_take_additional_instructions(self):
    backend = _ScriptedBackend(replies=[_model_turn('Paris.')])
    backend.register()
    strategy = agentic_retrieval.AgenticRetrievalStrategy(
        query_template='Answer: {question}'
    )

    executing.run(
        strategy('Capital of France?', additional_instructions='Be brief.')
    )

    self.assertEqual(
        backend.seen_turns[0][0].content, 'Answer: Capital of France?'
    )

  def test_is_available_from_the_strategy_registry(self):
    with self.subTest('can_be_built_by_name'):
      strategy = strategies_lib.get_strategy('agentic_retrieval')
      self.assertIsInstance(
          strategy, agentic_retrieval.AgenticRetrievalStrategy
      )
    with self.subTest('takes_its_tools_through_the_registry'):
      # The registry has no way to name a corpus, so tools are how a caller
      # says what this strategy may do.
      backend = _ScriptedBackend(replies=[_model_turn('Four.')])
      backend.register()
      strategy = strategies_lib.get_strategy('agentic_retrieval', tools=[_add])

      executing.run(strategy('What is two plus two?'))

      self.assertEqual([spec.name for spec in backend.seen_tools[0]], ['_add'])


class ExtractCitedDocIdsTest(parameterized.TestCase):
  """Tests for `extract_cited_doc_ids`."""

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_citations',
          answer='Evidence [123]. Also [456].',
          expected=['123', '456'],
      ),
      dict(
          testcase_name='deduplication_preserves_order',
          answer='See [456] and [123] and [456] again.',
          expected=['456', '123'],
      ),
      dict(
          testcase_name='no_citations',
          answer='No citations here.',
          expected=[],
      ),
      dict(
          testcase_name='empty_string',
          answer='',
          expected=[],
      ),
      dict(
          testcase_name='unclosed_bracket',
          answer='Broken [123 and [456].',
          expected=['456'],
      ),
      dict(
          testcase_name='unopened_bracket',
          answer='Broken 123] and [456].',
          expected=['456'],
      ),
      dict(
          testcase_name='nested_brackets',
          answer='Nested [[789]] here.',
          expected=['789'],
      ),
      dict(
          testcase_name='empty_brackets',
          answer='Empty [] here and [42].',
          expected=['42'],
      ),
      dict(
          testcase_name='non_numeric_content',
          answer='Text [abc] and [123] and [x1].',
          expected=['abc', '123', 'x1'],
      ),
      dict(
          testcase_name='adjacent_citations',
          answer='Back to back [1][2][3].',
          expected=['1', '2', '3'],
      ),
      dict(
          testcase_name='bracket_in_template_escapes',
          answer='Format {{like this}} and [99].',
          expected=['99'],
      ),
      dict(
          testcase_name='large_doc_id',
          answer='See [9999999].',
          expected=['9999999'],
      ),
      dict(
          testcase_name='comma_separated_citations',
          answer='They transferred [73758, 81636].',
          expected=['73758', '81636'],
      ),
      dict(
          testcase_name='comma_separated_with_dedup',
          answer='See [73758, 81636]. Also [73758, 81636].',
          expected=['73758', '81636'],
      ),
      dict(
          testcase_name='mixed_comma_and_single',
          answer='First [10]. Then [20, 30]. Then [10].',
          expected=['10', '20', '30'],
      ),
      dict(
          testcase_name='comma_separated_with_non_numeric',
          answer='Mixed [1243, 74325, fdghbdf].',
          expected=['1243', '74325', 'fdghbdf'],
      ),
  )
  def test_extract_cited_doc_ids(self, answer: str, expected: list[str]):
    self.assertEqual(agentic_retrieval.extract_cited_doc_ids(answer), expected)


if __name__ == '__main__':
  absltest.main()
