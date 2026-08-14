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

import asyncio
import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.genai import types as genai_types
from mcp import types as mcp_types
from mcp.client import streamable_http as mcp_streamable_http
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import routing
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
    self.assertEmpty(result.docs)

  def test_no_mcp_server_url_raises(self):
    with self.assertRaises(ValueError):
      mcp_strategies.GoogleGenAIMCPStrategy(mcp_server_url='')

  def test_safe_call_tool_handles_exceptions(self):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )
    mock_call_tool = mock.AsyncMock(
        side_effect=RuntimeError('HTTP 400 Bad Request')
    )
    self.mock_client.session.call_tool = mock_call_tool

    # Run strategy which will apply _safe_call_tool wrapper
    result = ot.run(strategy(question=_TEXT_QUESTION))
    self.assertEqual(result.answer, 'mcp answer')

    # Now verify that wrapped call_tool catches the exception and returns
    # error CallToolResult.
    wrapped_call_tool = self.mock_client.session.call_tool
    tool_res = ot.run(wrapped_call_tool('bad_tool', {'arg': 1}))
    self.assertTrue(tool_res.isError)
    self.assertIn('Error executing tool bad_tool', tool_res.content[0].text)

  def test_suppresses_fastmcp_shutdown_exception_when_answer_produced(self):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )
    self.mock_client.__aexit__.side_effect = RuntimeError(
        'Transport teardown error'
    )

    result = ot.run(strategy(question=_TEXT_QUESTION))
    self.assertEqual(result.answer, 'mcp answer')

  def test_resilient_transport_handles_http_400(self):
    transport = mcp_strategies._ResilientStreamableHTTPTransport(
        url='http://localhost:8000/mcp'
    )
    mock_request = mcp_types.JSONRPCRequest(
        jsonrpc='2.0',
        id='req-1',
        method='tools/call',
        params={'name': 'search_passages', 'arguments': {'bad_arg': 1}},
    )
    mock_msg = mcp_streamable_http.SessionMessage(
        mcp_types.JSONRPCMessage(mock_request)
    )
    mock_writer = mock.AsyncMock()
    mock_response = mock.AsyncMock()
    mock_response.status_code = 400
    mock_response.aread.return_value = b'{"detail": "Field required"}'

    class _MockClient:

      @contextlib.asynccontextmanager
      async def stream(self, *unused_args, **unused_kwargs):
        yield mock_response

    ctx = mcp_streamable_http.RequestContext(
        client=_MockClient(),  # pyrefly: ignore[bad-argument-type]
        session_message=mock_msg,
        read_stream_writer=mock_writer,
        session_id=None,
        metadata=None,
    )

    ot.run(transport._handle_post_request(ctx))

    mock_writer.send.assert_awaited_once()
    sent_msg = mock_writer.send.await_args[0][0]
    self.assertIsInstance(sent_msg.message.root, mcp_types.JSONRPCError)
    self.assertEqual(sent_msg.message.root.id, 'req-1')
    self.assertIn('HTTP 400', sent_msg.message.root.error.message)

  def test_resilient_transport_handles_http_404(self):
    transport = mcp_strategies._ResilientStreamableHTTPTransport(
        url='http://localhost:8000/mcp'
    )
    mock_request = mcp_types.JSONRPCRequest(
        jsonrpc='2.0',
        id='req-404',
        method='tools/call',
        params={'name': 'search_passages', 'arguments': {}},
    )
    mock_msg = mcp_streamable_http.SessionMessage(
        mcp_types.JSONRPCMessage(mock_request)
    )
    mock_writer = mock.AsyncMock()
    mock_response = mock.AsyncMock()
    mock_response.status_code = 404

    class _MockClient:

      @contextlib.asynccontextmanager
      async def stream(self, *unused_args, **unused_kwargs):
        yield mock_response

    ctx = mcp_streamable_http.RequestContext(
        client=_MockClient(),  # pyrefly: ignore[bad-argument-type]
        session_message=mock_msg,
        read_stream_writer=mock_writer,
        session_id=None,
        metadata=None,
    )

    ot.run(transport._handle_post_request(ctx))

    mock_writer.send.assert_awaited_once()
    sent_msg = mock_writer.send.await_args[0][0]
    self.assertIsInstance(sent_msg.message.root, mcp_types.JSONRPCError)
    self.assertEqual(sent_msg.message.root.id, 'req-404')
    self.assertEqual(sent_msg.message.root.error.code, 32600)

  def test_resilient_transport_handles_http_202(self):
    transport = mcp_strategies._ResilientStreamableHTTPTransport(
        url='http://localhost:8000/mcp'
    )
    mock_request = mcp_types.JSONRPCRequest(
        jsonrpc='2.0',
        id='req-202',
        method='notifications/initialized',
        params={},
    )
    mock_msg = mcp_streamable_http.SessionMessage(
        mcp_types.JSONRPCMessage(mock_request)
    )
    mock_writer = mock.AsyncMock()
    mock_response = mock.AsyncMock()
    mock_response.status_code = 202

    class _MockClient:

      @contextlib.asynccontextmanager
      async def stream(self, *unused_args, **unused_kwargs):
        yield mock_response

    ctx = mcp_streamable_http.RequestContext(
        client=_MockClient(),  # pyrefly: ignore[bad-argument-type]
        session_message=mock_msg,
        read_stream_writer=mock_writer,
        session_id=None,
        metadata=None,
    )

    ot.run(transport._handle_post_request(ctx))
    mock_writer.send.assert_not_awaited()

  def test_safe_call_tool_handles_per_tool_timeout(self):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        tool_timeout=0.01,
    )

    async def _hanging_call_tool(*unused_args, **unused_kwargs):
      await asyncio.sleep(1.0)

    self.mock_client.session.call_tool = _hanging_call_tool

    # Run strategy which will apply _safe_call_tool wrapper with
    # tool_timeout=0.01.
    result = ot.run(strategy(question=_TEXT_QUESTION))
    self.assertEqual(result.answer, 'mcp answer')

    wrapped_call_tool = self.mock_client.session.call_tool
    tool_res = ot.run(wrapped_call_tool('hanging_tool', {}))
    self.assertTrue(tool_res.isError)
    self.assertIn('Error executing tool hanging_tool', tool_res.content[0].text)

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


class MCPStrategyErrorResilienceTest(parameterized.TestCase):
  """Tests that the MCP strategy handles transient errors gracefully."""

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
    self.mock_client.__aenter__ = mock.AsyncMock(return_value=self.mock_client)
    self.mock_client.__aexit__ = mock.AsyncMock(return_value=None)
    self.mock_client.session = mock.MagicMock()

  def test_error_raises_instead_of_crashing_entire_eval(self):
    """A RuntimeError during llm.chat propagates so ot.evaluate can catch it."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )

    # First call succeeds.
    result = ot.run(strategy(question='What?'))
    self.assertEqual(result.answer, 'mcp answer')

    # Simulate an error on the second call.
    chat_wrapper = routing.function_registry[llm.chat.name]
    original_fn = chat_wrapper._registered_function

    def raising_fn(*_args, **_kwargs):
      raise RuntimeError('Client is not connected.')

    chat_wrapper._registered_function = raising_fn
    try:
      with self.assertRaises(RuntimeError):
        ot.run(strategy(question='What?'))
    finally:
      chat_wrapper._registered_function = original_fn

  def test_each_call_creates_a_fresh_client(self):
    """Each call creates a new MCP client, ensuring no stale session reuse."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )

    ot.run(strategy(question='First?'))
    self.assertEqual(self.mock_client_class.call_count, 1)

    ot.run(strategy(question='Second?'))
    self.assertEqual(self.mock_client_class.call_count, 2)

    ot.run(strategy(question='Third?'))
    self.assertEqual(self.mock_client_class.call_count, 3)

  def test_failed_call_does_not_affect_next_call(self):
    """A failed call's session is cleaned up; the next call gets a fresh one."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )

    # Simulate an error on the first call.
    chat_wrapper = routing.function_registry[llm.chat.name]
    original_fn = chat_wrapper._registered_function

    def raising_fn(*_args, **_kwargs):
      raise RuntimeError('Connection lost.')

    chat_wrapper._registered_function = raising_fn
    try:
      with self.assertRaises(RuntimeError):
        ot.run(strategy(question='Fails'))
    finally:
      chat_wrapper._registered_function = original_fn

    # The failed call's session should have been cleaned up via __aexit__.
    self.mock_client.__aexit__.assert_awaited()

    # Next call succeeds with a fresh client.
    result = ot.run(strategy(question='Succeeds'))
    self.assertEqual(result.answer, 'mcp answer')
    self.assertEqual(self.mock_client_class.call_count, 2)

  def test_session_cleaned_up_after_each_call(self):
    """__aexit__ is called after each call to clean up the session."""
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp'
    )

    ot.run(strategy(question='What?'))
    # __aexit__ should have been called once after the call completes.
    self.mock_client.__aexit__.assert_awaited_once()


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
    self.assertIsInstance(result.docs, list)

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


class ThinkingConfigMergingTest(parameterized.TestCase):
  """Tests that the MCP strategy merges thinking_config with existing defaults."""

  def setUp(self):
    super().setUp()
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply='thinking answer',
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
          testcase_name='no_existing_thinking_config',
          default_thinking_config=None,
          expected_include_thoughts=True,
          expected_thinking_budget=None,
          expected_thinking_level=None,
      ),
      dict(
          testcase_name='existing_config_with_budget_and_thoughts_disabled',
          default_thinking_config=genai_types.ThinkingConfig(
              include_thoughts=False,
              thinking_budget=1024,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=1024,
          expected_thinking_level=None,
      ),
      dict(
          testcase_name='existing_config_with_budget_only',
          default_thinking_config=genai_types.ThinkingConfig(
              thinking_budget=2048,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=2048,
          expected_thinking_level=None,
      ),
      dict(
          testcase_name='existing_config_with_zero_budget',
          default_thinking_config=genai_types.ThinkingConfig(
              include_thoughts=False,
              thinking_budget=0,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=0,
          expected_thinking_level=None,
      ),
      dict(
          testcase_name='existing_config_with_thinking_level',
          default_thinking_config=genai_types.ThinkingConfig(
              include_thoughts=False,
              thinking_level=genai_types.ThinkingLevel.LOW,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=None,
          expected_thinking_level=genai_types.ThinkingLevel.LOW,
      ),
      dict(
          testcase_name='existing_config_with_all_fields',
          default_thinking_config=genai_types.ThinkingConfig(
              include_thoughts=False,
              thinking_budget=4096,
              thinking_level=genai_types.ThinkingLevel.HIGH,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=4096,
          expected_thinking_level=genai_types.ThinkingLevel.HIGH,
      ),
      dict(
          testcase_name='existing_config_with_thoughts_already_enabled',
          default_thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=512,
          ),
          expected_include_thoughts=True,
          expected_thinking_budget=512,
          expected_thinking_level=None,
      ),
      dict(
          testcase_name='strategy_include_thoughts_false',
          default_thinking_config=genai_types.ThinkingConfig(
              thinking_budget=1024,
          ),
          strategy_include_thoughts=False,
          expected_include_thoughts=False,
          expected_thinking_budget=1024,
          expected_thinking_level=None,
      ),
  )
  def test_thinking_config_merging(
      self,
      default_thinking_config,
      expected_include_thoughts,
      expected_thinking_budget,
      expected_thinking_level,
      strategy_include_thoughts=True,
  ):
    # Set the default thinking_config in the backend's chat defaults.
    if default_thinking_config is not None:
      llm.chat.update(thinking_config=default_thinking_config)

    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        include_thoughts=strategy_include_thoughts,
    )

    # Capture the kwargs passed to llm.chat by temporarily swapping the
    # chat wrapper's _registered_function with a spy. This works because
    # _BuiltinWrapper.__call__ delegates to self._registered_function().
    captured_kwargs = {}
    chat_wrapper = routing.function_registry[llm.chat.name]
    original_fn = chat_wrapper._registered_function

    def capturing_fn(*args, **kwargs):
      captured_kwargs.update(kwargs)
      return original_fn(*args, **kwargs)

    chat_wrapper._registered_function = capturing_fn
    try:
      ot.run(strategy(question='Test question'))
    finally:
      chat_wrapper._registered_function = original_fn

    actual_config = captured_kwargs['thinking_config']
    with self.subTest('include_thoughts'):
      self.assertEqual(
          actual_config.include_thoughts, expected_include_thoughts
      )
    with self.subTest('thinking_budget'):
      self.assertEqual(actual_config.thinking_budget, expected_thinking_budget)
    with self.subTest('thinking_level'):
      self.assertEqual(actual_config.thinking_level, expected_thinking_level)

  def test_original_default_thinking_config_not_mutated(self):
    """Verifies the original default ThinkingConfig is not mutated in place."""
    original_config = genai_types.ThinkingConfig(
        include_thoughts=False,
        thinking_budget=1024,
    )
    llm.chat.update(thinking_config=original_config)

    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
    )
    ot.run(strategy(question='Test question'))

    # The original config should still have include_thoughts=False.
    self.assertFalse(original_config.include_thoughts)
    self.assertEqual(original_config.thinking_budget, 1024)

  def test_configurable_http_timeouts(self):
    strategy = mcp_strategies.GoogleGenAIMCPStrategy(
        mcp_server_url='http://localhost:8000/mcp',
        http_read_timeout=150.0,
        http_connect_timeout=45.0,
        http_write_timeout=30.0,
    )
    self.assertEqual(strategy.http_read_timeout, 150.0)
    self.assertEqual(strategy.http_connect_timeout, 45.0)
    self.assertEqual(strategy.http_write_timeout, 30.0)

    with mock.patch.object(
        mcp_strategies, '_ResilientStreamableHttpTransport'
    ) as mock_transport_cls:
      ot.run(strategy(question='Test question'))
      mock_transport_cls.assert_called_once_with(
          url='http://localhost:8000/mcp',
          http_read_timeout=150.0,
          http_connect_timeout=45.0,
          http_write_timeout=30.0,
      )


if __name__ == '__main__':
  absltest.main()
