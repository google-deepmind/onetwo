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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.stdlib.qa import mcp_strategies
from onetwo.stdlib.qa import qa

_Chunk = content_lib.Chunk
_ChunkList = content_lib.ChunkList
StrategyResult = qa.StrategyResult

_TEXT_QUESTION = 'What is the capital of France?'
_MULTIMODAL_QUESTION = _ChunkList([
    _Chunk('Describe this image: '),
    _Chunk(b'\x89PNG_fake_query_image', content_type='image/png'),
])


class MCPStrategyTest(parameterized.TestCase):
  """Tests for GoogleGenAIMCPStrategy."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='mcp answer',
    )
    self.llm_backend.register()

    # Mock fastmcp Client and transport to avoid network requests.
    self.mock_client_class = self.enter_context(
        mock.patch.object(mcp_strategies.fastmcp, 'Client', autospec=True)
    )
    self.mock_client = self.mock_client_class.return_value
    # Set up async context manager mock for __aenter__
    self.mock_client.__aenter__ = mock.AsyncMock(return_value=self.mock_client)
    self.mock_client.session = mock.MagicMock()

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
  def test_returns_answer(self, question):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, StrategyResult)
    self.assertEqual(result.answer, 'mcp answer')
    self.assertEmpty(result.doc_ids)

  def test_no_mcp_server_url_raises(self):
    with self.assertRaises(ValueError):
      mcp_strategies.GoogleGenAIMCPStrategy(mcp_server_url='')

  def test_query_template_formatting(self):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        query_template='Prompt: {question}{additional_instructions}',
    )
    result = ot.run(strategy(question='What?'))
    self.assertEqual(result.answer, 'mcp answer')
    self.assertLen(self.llm_backend.prompts, 1)
    self.assertEqual(self.llm_backend.prompts[0], 'Prompt: What?')

  def test_missing_question_placeholder_raises(self):
    with self.assertRaises(ValueError):
      mcp_strategies.GoogleGenAIMCPStrategy(
          mcp_server_url='http://localhost:8000/mcp',
          query_template='No placeholders here {additional_instructions}',
      )

  def test_missing_additional_instructions_placeholder_raises(self):
    with self.assertRaises(ValueError):
      mcp_strategies.GoogleGenAIMCPStrategy(
          mcp_server_url='http://localhost:8000/mcp',
          query_template='Only question: {question}',
      )

  def test_additional_instructions_str_question(self):
    """Additional instructions are injected into the template for str."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        query_template='Preamble. {additional_instructions}\nQ: {question}',
    )
    result = ot.run(
        strategy(
            question='What?',
            additional_instructions='Answer briefly.',
        )
    )
    self.assertEqual(result.answer, 'mcp answer')
    self.assertLen(self.llm_backend.prompts, 1)
    prompt = self.llm_backend.prompts[0]
    self.assertIn('Answer briefly.', prompt)
    self.assertIn('Q: What?', prompt)

  def test_no_additional_instructions_str_question(self):
    """Without additional_instructions the placeholder collapses to empty."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        query_template='Preamble. {additional_instructions}\nQ: {question}',
    )
    result = ot.run(strategy(question='What?'))
    self.assertEqual(result.answer, 'mcp answer')
    self.assertLen(self.llm_backend.prompts, 1)
    prompt = self.llm_backend.prompts[0]
    self.assertIn('Preamble. \nQ: What?', prompt)

  def test_additional_instructions_chunklist_question(self):
    """Additional instructions are injected for ChunkList questions."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        query_template='Preamble. {additional_instructions}\nQ: {question}',
    )
    result = ot.run(
        strategy(
            question=_MULTIMODAL_QUESTION,
            additional_instructions='Be concise.',
        )
    )
    self.assertEqual(result.answer, 'mcp answer')
    self.assertLen(self.llm_backend.prompts, 1)
    prompt = self.llm_backend.prompts[0]
    self.assertIn('Be concise.', prompt)

  def test_additional_instructions_default_template(self):
    """additional_instructions works with the real default_qa_template_mcp."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
    )
    result = ot.run(
        strategy(
            question='What is AI?',
            additional_instructions='Use only peer-reviewed sources.',
        )
    )
    self.assertEqual(result.answer, 'mcp answer')
    self.assertLen(self.llm_backend.prompts, 1)
    prompt = self.llm_backend.prompts[0]
    self.assertIn('Use only peer-reviewed sources.', prompt)
    self.assertIn('Question: What is AI?', prompt)

  @parameterized.named_parameters(
      dict(
          testcase_name='str_question_with_instructions',
          query_template='Prompt: {question}{additional_instructions}',
          question='What?',
          additional_instructions='Be quick.',
          expected='Prompt: What?\nBe quick.',
      ),
      dict(
          testcase_name='str_question_no_instructions',
          query_template='Prompt: {question}{additional_instructions}',
          question='What?',
          additional_instructions='',
          expected='Prompt: What?',
      ),
      dict(
          testcase_name='chunklist_question_with_instructions',
          query_template='Prompt: {question}{additional_instructions}',
          question=_ChunkList([
              _Chunk('What?'),
              _Chunk(b'image', content_type='image/png'),
          ]),
          additional_instructions='Be quick.',
          expected=_ChunkList([
              _Chunk('Prompt: '),
              _Chunk('What?'),
              _Chunk(b'image', content_type='image/png'),
              _Chunk('\nBe quick.'),
          ]),
      ),
      dict(
          testcase_name='chunklist_question_no_instructions',
          query_template='Prompt: {question}{additional_instructions}',
          question=_ChunkList([
              _Chunk('What?'),
              _Chunk(b'image', content_type='image/png'),
          ]),
          additional_instructions='',
          expected=_ChunkList([
              _Chunk('Prompt: '),
              _Chunk('What?'),
              _Chunk(b'image', content_type='image/png'),
              _Chunk(''),
          ]),
      ),
      dict(
          testcase_name='chunklist_multiple_question_placeholders',
          query_template=(
              'Prompt: {question} and {question}{additional_instructions}'
          ),
          question=_ChunkList([
              _Chunk('What?'),
          ]),
          additional_instructions='',
          expected=_ChunkList([
              _Chunk('What?'),
          ]),
      ),
  )
  def test_format_question(
      self, query_template, question, additional_instructions, expected
  ):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        query_template=query_template,
    )
    formatted = strategy._format_question(
        question=question, additional_instructions=additional_instructions
    )
    self.assertEqual(formatted, expected)

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
  def test_extract_cited_doc_ids(self, answer, expected):
    self.assertEqual(mcp_strategies._extract_cited_doc_ids(answer), expected)


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
          testcase_name='mcp',
          strategy=mcp_strategies.GoogleGenAIMCPStrategy(
              mcp_server_url='http://localhost:8000/mcp',
          ),
          question=_TEXT_QUESTION,
      ),
  )
  def test_result_is_strategy_result(self, strategy, question):
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, StrategyResult)
    self.assertIsInstance(result.answer, str)
    self.assertIsInstance(result.doc_ids, list)

  @parameterized.named_parameters(
      dict(
          testcase_name='mcp_multimodal',
          strategy=mcp_strategies.GoogleGenAIMCPStrategy(
              mcp_server_url='http://localhost:8000/mcp',
          ),
          question=_MULTIMODAL_QUESTION,
      ),
  )
  def test_multimodal_inputs_produce_valid_result(self, strategy, question):
    result = ot.run(strategy(question=question))
    self.assertIsInstance(result, StrategyResult)
    self.assertEqual(result.answer, 'cross answer')


if __name__ == '__main__':
  absltest.main()
