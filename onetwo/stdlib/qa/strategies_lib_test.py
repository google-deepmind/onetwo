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

from collections.abc import Iterable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.qa import mcp_strategies
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import qa_prompt_builder
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document
_Chunk = content_lib.Chunk
_ChunkList = content_lib.ChunkList
_StrategyResult = qa.StrategyResult


# =============================================================================
# Mock retriever
# =============================================================================


class MockRetriever(retrieval.Retriever[str, _Document]):
  """A simple mock retriever that returns a fixed set of documents."""

  def __init__(self, documents: list[_Document]):
    self._documents = documents

  @executing.make_executable
  async def retrieve(self, query: str, **kwargs) -> Iterable[_Document]:
    max_results = kwargs.get('max_results')
    if max_results is not None:
      return self._documents[:max_results]
    return self._documents


# =============================================================================
# Test helpers
# =============================================================================

_TEXT_DOCS = [
    _Document(doc_id='d1', content='Paris is the capital of France.'),
    _Document(doc_id='d2', content='Berlin is the capital of Germany.'),
    _Document(doc_id='d3', content='Madrid is the capital of Spain.'),
]

_MULTIMODAL_DOCS = [
    _Document(
        doc_id='img1',
        content=_ChunkList([
            _Chunk('A photo of the Eiffel Tower.'),
            _Chunk(b'\x89PNG_fake_image_bytes', content_type='image/png'),
        ]),
    ),
    _Document(
        doc_id='img2',
        content=_ChunkList([
            _Chunk('A diagram of the solar system.'),
        ]),
    ),
]

_TEXT_QUESTION = 'What is the capital of France?'
_MULTIMODAL_QUESTION = _ChunkList([
    _Chunk('Describe this image: '),
    _Chunk(b'\x89PNG_fake_query_image', content_type='image/png'),
])


# =============================================================================
# Registry tests
# =============================================================================


class RegistryTest(absltest.TestCase):
  """Tests for the strategy registry."""

  def test_builtin_strategies_registered(self):
    names = strategies_lib.list_strategies()
    self.assertIn('closed_book', names)
    self.assertIn('long_context', names)
    self.assertIn('rag', names)
    self.assertIn('retrieval_only', names)
    self.assertIn('mcp', names)

  def test_get_strategy_returns_correct_type(self):
    self.assertIsInstance(
        strategies_lib.get_strategy('closed_book'),
        strategies_lib.ClosedBookStrategy,
    )
    self.assertIsInstance(
        strategies_lib.get_strategy('long_context'),
        strategies_lib.LongContextStrategy,
    )
    self.assertIsInstance(
        strategies_lib.get_strategy('rag', retriever=MockRetriever([])),
        strategies_lib.RAGStrategy,
    )
    self.assertIsInstance(
        strategies_lib.get_strategy(
            'retrieval_only', retriever=MockRetriever([])
        ),
        strategies_lib.RetrievalOnlyStrategy,
    )
    self.assertIsInstance(
        strategies_lib.get_strategy(
            'mcp',
            mcp_server_urls='http://localhost:8000',
        ),
        mcp_strategies.GoogleGenAIMCPStrategy,
    )

  def test_get_unknown_strategy_raises(self):
    with self.assertRaises(KeyError):
      strategies_lib.get_strategy('nonexistent')

  def test_register_duplicate_raises(self):
    with self.assertRaises(ValueError):
      strategies_lib.register_strategy(
          'closed_book', strategies_lib.ClosedBookStrategy
      )

  def test_register_and_unregister_custom_strategy(self):
    strategies_lib.register_strategy(
        'custom', strategies_lib.ClosedBookStrategy
    )
    self.assertIn('custom', strategies_lib.list_strategies())
    strategies_lib.unregister_strategy('custom')
    self.assertNotIn('custom', strategies_lib.list_strategies())

  def test_unregister_unknown_raises(self):
    with self.assertRaises(KeyError):
      strategies_lib.unregister_strategy('nonexistent')


# =============================================================================
# Template tests
# =============================================================================


class DefaultPromptBuilderTest(parameterized.TestCase):
  """Tests for the default QADocumentPromptBuilder used by strategies."""

  @parameterized.named_parameters(
      dict(
          testcase_name='text_inputs',
          docs=[_Document(content='Some context.')],
          question='What?',
          expected_context='Some context.',
      ),
      dict(
          testcase_name='multimodal_question',
          docs=[_Document(content='Some context.')],
          question=_MULTIMODAL_QUESTION,
          expected_context='Some context.',
      ),
      dict(
          testcase_name='multimodal_context',
          docs=[
              _Document(
                  content=_ChunkList([
                      _Chunk('text part'),
                      _Chunk(b'\x89PNG_bytes', content_type='image/png'),
                  ])
              )
          ],
          question='What?',
          expected_context='text part<image/png>',
      ),
  )
  def test_template_produces_expected_content(
      self, docs, question, expected_context
  ):
    builder = qa_prompt_builder.QADocumentPromptBuilder()
    composable_prompt = ot.run(builder.compile_prompt(docs, question))
    content_str = str(composable_prompt)
    self.assertIn(expected_context, content_str)

  def test_template_text_content_structure(self):
    builder = qa_prompt_builder.QADocumentPromptBuilder()
    docs = [_Document(content='my context')]
    prompt_content = ot.run(builder.compile_prompt(docs, 'my question'))
    content_str = str(prompt_content)
    self.assertIn('Use the following pieces of context', content_str)
    self.assertIn('Question: my question', content_str)
    self.assertIn('Context:\nmy context', content_str)
    self.assertIn('Answer:', content_str)


# =============================================================================
# Closed-book strategy tests
# =============================================================================


class ClosedBookStrategyTest(parameterized.TestCase):
  """Tests for ClosedBookStrategy."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='mock answer',
    )
    self.llm_backend.register()

  @parameterized.named_parameters(
      dict(
          testcase_name='text_question',
          question=_TEXT_QUESTION,
      ),
      dict(
          testcase_name='multimodal_question',
          question=_MULTIMODAL_QUESTION,
      ),
  )
  def test_returns_answer_with_no_docs(self, question):
    strategy = strategies_lib.ClosedBookStrategy()
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, _StrategyResult)
    self.assertEqual(result.answer, 'mock answer')
    self.assertEmpty(result.docs)

  def test_prompt_contains_none_context(self):
    """The prompt sent to the LLM should contain 'None' as context."""
    strategy = strategies_lib.ClosedBookStrategy()
    _ = ot.run(strategy(question='What?'))
    # LLMForTest records the prompts it receives.
    self.assertLen(self.llm_backend.prompts, 1)
    self.assertIn('None', self.llm_backend.prompts[0])
    self.assertIn('QUERY:', self.llm_backend.prompts[0])
    self.assertIn('CONTEXT:', self.llm_backend.prompts[0])
    self.assertIn('Please answer the query.', self.llm_backend.prompts[0])


# =============================================================================
# Long-context strategy tests
# =============================================================================


class LongContextStrategyTest(parameterized.TestCase):
  """Tests for LongContextStrategy."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='long context answer',
    )
    self.llm_backend.register()

  @parameterized.named_parameters(
      dict(
          testcase_name='text_docs_text_question',
          documents=_TEXT_DOCS,
          question=_TEXT_QUESTION,
          expected_doc_ids=['d1', 'd2', 'd3'],
      ),
      dict(
          testcase_name='text_docs_multimodal_question',
          documents=_TEXT_DOCS,
          question=_MULTIMODAL_QUESTION,
          expected_doc_ids=['d1', 'd2', 'd3'],
      ),
      dict(
          testcase_name='multimodal_docs_text_question',
          documents=_MULTIMODAL_DOCS,
          question=_TEXT_QUESTION,
          expected_doc_ids=['img1', 'img2'],
      ),
  )
  def test_returns_answer_with_all_docs(
      self, documents, question, expected_doc_ids
  ):
    strategy = strategies_lib.LongContextStrategy(documents=documents)
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, _StrategyResult)
    self.assertEqual(result.answer, 'long context answer')
    self.assertEqual([doc.doc_id for doc in result.docs], expected_doc_ids)

  def test_empty_documents_returns_empty_docs(self):
    strategy = strategies_lib.LongContextStrategy(documents=[])
    result = ot.run(strategy(question='What?'))
    self.assertEmpty(result.docs)

  def test_prompt_contains_document_content(self):
    strategy = strategies_lib.LongContextStrategy(documents=_TEXT_DOCS[:1])
    _ = ot.run(strategy(question='What?'))
    self.assertLen(self.llm_backend.prompts, 1)
    self.assertIn('Paris is the capital of France', self.llm_backend.prompts[0])


# =============================================================================
# RAG strategy tests
# =============================================================================


class RAGStrategyTest(parameterized.TestCase):
  """Tests for RAGStrategy."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='rag answer',
    )
    self.llm_backend.register()

  @parameterized.named_parameters(
      dict(
          testcase_name='text_docs_text_question',
          documents=_TEXT_DOCS,
          question=_TEXT_QUESTION,
          max_results=2,
          expected_doc_ids=['d1', 'd2'],
      ),
      dict(
          testcase_name='text_docs_multimodal_question',
          documents=_TEXT_DOCS,
          question=_TEXT_QUESTION,
          max_results=None,
          expected_doc_ids=['d1', 'd2', 'd3'],
      ),
      dict(
          testcase_name='multimodal_docs_text_question',
          documents=_MULTIMODAL_DOCS,
          question=_TEXT_QUESTION,
          max_results=1,
          expected_doc_ids=['img1'],
      ),
  )
  def test_returns_answer_with_retrieved_docs(
      self, documents, question, max_results, expected_doc_ids
  ):
    retriever = MockRetriever(documents)
    strategy = strategies_lib.RAGStrategy(retriever=retriever)  # pyrefly: ignore[bad-argument-type]
    kwargs = {}
    if max_results is not None:
      kwargs['max_results'] = max_results
    result = ot.run(strategy(question=question, **kwargs))
    self.assertIsInstance(result, _StrategyResult)
    self.assertEqual(result.answer, 'rag answer')
    self.assertEqual([doc.doc_id for doc in result.docs], expected_doc_ids)

  def test_no_retriever_raises(self):
    strategy = strategies_lib.RAGStrategy(retriever=None)
    with self.assertRaises(ValueError):
      ot.run(strategy(question='What?'))

  def test_empty_retrieval_returns_empty_docs(self):
    retriever = MockRetriever([])
    strategy = strategies_lib.RAGStrategy(retriever=retriever)  # pyrefly: ignore[bad-argument-type]
    result = ot.run(strategy(question='What?'))
    self.assertEmpty(result.docs)

  def test_max_results_limits_retrieval(self):
    retriever = MockRetriever(_TEXT_DOCS)
    strategy = strategies_lib.RAGStrategy(retriever=retriever)  # pyrefly: ignore[bad-argument-type]
    result = ot.run(strategy(question='What?', max_results=1))
    self.assertLen(result.docs, 1)
    self.assertEqual(result.docs[0].doc_id, 'd1')


# =============================================================================
# Retrieval-only strategy tests
# =============================================================================


class RetrievalOnlyStrategyTest(parameterized.TestCase):
  """Tests for RetrievalOnlyStrategy."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='should not be called',
    )
    self.llm_backend.register()

  @parameterized.named_parameters(
      dict(
          testcase_name='text_docs_text_question',
          documents=_TEXT_DOCS,
          question=_TEXT_QUESTION,
          max_results=2,
          expected_doc_ids=['d1', 'd2'],
      ),
      dict(
          testcase_name='text_docs_multimodal_question',
          documents=_TEXT_DOCS,
          question=_TEXT_QUESTION,
          max_results=None,
          expected_doc_ids=['d1', 'd2', 'd3'],
      ),
      dict(
          testcase_name='multimodal_docs_text_question',
          documents=_MULTIMODAL_DOCS,
          question=_TEXT_QUESTION,
          max_results=1,
          expected_doc_ids=['img1'],
      ),
  )
  def test_returns_empty_answer_with_retrieved_docs(
      self, documents, question, max_results, expected_doc_ids
  ):
    retriever = MockRetriever(documents)
    strategy = strategies_lib.RetrievalOnlyStrategy(retriever=retriever)  # pyrefly: ignore[bad-argument-type]
    kwargs = {}
    if max_results is not None:
      kwargs['max_results'] = max_results
    result = ot.run(strategy(question=question, **kwargs))
    self.assertIsInstance(result, _StrategyResult)
    self.assertEqual(result.answer, '')
    self.assertEqual([doc.doc_id for doc in result.docs], expected_doc_ids)
    # Verify LLM was not called.
    self.assertEmpty(self.llm_backend.prompts)

  def test_no_retriever_raises(self):
    strategy = strategies_lib.RetrievalOnlyStrategy(retriever=None)
    with self.assertRaises(ValueError):
      ot.run(strategy(question='What?'))

  def test_empty_retrieval_returns_empty_docs(self):
    retriever = MockRetriever([])
    strategy = strategies_lib.RetrievalOnlyStrategy(retriever=retriever)  # pyrefly: ignore[bad-argument-type]
    result = ot.run(strategy(question='What?'))
    self.assertEmpty(result.docs)
    self.assertEqual(result.answer, '')
    self.assertEmpty(self.llm_backend.prompts)


# =============================================================================
# Cross-strategy parametrized tests
# =============================================================================


class CrossStrategyTest(parameterized.TestCase):
  """Tests shared properties across all strategy types."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='cross answer',
    )
    self.llm_backend.register()

    # Mock fastmcp Client and transport to avoid network requests.
    self.mock_client_class = self.enter_context(
        mock.patch.object(mcp_strategies.fastmcp, 'Client', autospec=True)
    )
    self.mock_client = self.mock_client_class.return_value
    self.mock_client.__aenter__ = mock.AsyncMock(return_value=self.mock_client)
    self.mock_client.session = mock.MagicMock()

  @parameterized.named_parameters(
      dict(
          testcase_name='closed_book',
          strategy=strategies_lib.ClosedBookStrategy(),
          question=_TEXT_QUESTION,
      ),
      dict(
          testcase_name='long_context',
          strategy=strategies_lib.LongContextStrategy(documents=_TEXT_DOCS),
          question=_TEXT_QUESTION,
      ),
      dict(
          testcase_name='rag',
          strategy=strategies_lib.RAGStrategy(
              retriever=MockRetriever(_TEXT_DOCS),  # pyrefly: ignore[bad-argument-type]
          ),
          question=_TEXT_QUESTION,
      ),
      dict(
          testcase_name='mcp',
          strategy=mcp_strategies.GoogleGenAIMCPStrategy(
              mcp_server_urls='http://localhost:8000/mcp',
          ),
          question=_TEXT_QUESTION,
      ),
  )
  def test_result_is_strategy_result(self, strategy, question):
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, _StrategyResult)
    self.assertIsInstance(result.answer, str)
    self.assertIsInstance(result.docs, list)

  @parameterized.named_parameters(
      dict(
          testcase_name='closed_book_multimodal',
          strategy=strategies_lib.ClosedBookStrategy(),
          question=_MULTIMODAL_QUESTION,
      ),
      dict(
          testcase_name='long_context_multimodal',
          strategy=strategies_lib.LongContextStrategy(
              documents=_MULTIMODAL_DOCS,
          ),
          question=_MULTIMODAL_QUESTION,
      ),
      dict(
          testcase_name='rag_multimodal',
          strategy=strategies_lib.RAGStrategy(
              retriever=MockRetriever(_MULTIMODAL_DOCS),  # pyrefly: ignore[bad-argument-type]
          ),
          question=_MULTIMODAL_QUESTION,
      ),
      dict(
          testcase_name='mcp_multimodal',
          strategy=mcp_strategies.GoogleGenAIMCPStrategy(
              mcp_server_urls='http://localhost:8000/mcp',
          ),
          question=_MULTIMODAL_QUESTION,
      ),
  )
  def test_multimodal_inputs_produce_valid_result(self, strategy, question):
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, _StrategyResult)
    self.assertEqual(result.answer, 'cross answer')


if __name__ == '__main__':
  absltest.main()
