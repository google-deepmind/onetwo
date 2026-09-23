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

"""Tests for ClaudeAPI engine."""

import base64
import collections
from typing import Any, Counter, Final, TypeAlias, cast

# We have to mock anthropic to make this test hermetic.
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import function_calling as fc_agents
from onetwo.backends import claude_api
from onetwo.backends import claude_mock
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.stdlib.tool_use import tool_schema
import PIL.Image

import anthropic

_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole

_ConversationRole: TypeAlias = fc_types.ConversationRole
_ConversationTurn: TypeAlias = fc_types.ConversationTurn
_FunctionCall: TypeAlias = fc_types.FunctionCall
_FunctionResponse: TypeAlias = fc_types.FunctionResponse
_ToolSpec: TypeAlias = fc_types.ToolSpec


_BATCH_SIZE: Final[int] = 4


def _get_and_register_backend(
    model_name: str | None = None,
) -> claude_api.ClaudeAPI:
  """Get an instance of ClaudeAPI and register its methods."""
  if model_name:
    backend = claude_api.ClaudeAPI(
        api_key='some_key', batch_size=_BATCH_SIZE, model_name=model_name
    )
  else:
    backend = claude_api.ClaudeAPI(api_key='some_key', batch_size=_BATCH_SIZE)
  backend.register()
  return backend


class ClaudeAPITest(parameterized.TestCase, core_test_utils.CounterAssertions):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enter_context(
        mock.patch.object(
            anthropic,
            'Anthropic',
            autospec=True,
            side_effect=claude_mock.Anthropic,
        )
    )

  def test_missing_api_key_raises(self):
    """Verifies that omitting api_key raises ValueError."""
    with self.assertRaisesRegex(
        ValueError, 'An API key is required for using the Claude API.'
    ):
      claude_api.ClaudeAPI()

  def test_instruct_should_use_api_formatter(self):
    _ = _get_and_register_backend()
    prompt = 'Task: Answer the question.\nQuestion:Why?\nAnswer:'
    try:
      _ = executing.run(llm.instruct(prompt=prompt))
    except ValueError as e:
      self.fail(f'Instruct raised an error. Maybe due to wrong formatter?\n{e}')

  def test_chat(self):
    """Verifies that chat works and calls the api directly."""
    backend = _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        llm.chat(
            messages=[_Message(content=prompt_text, role=_PredefinedRole.USER)],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_messages_create': 1,
        'chat_via_api_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_generate(self):
    """Verifies that repeated calls to the cached method behave as expected."""
    backend = _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            max_tokens=512,
        )
    )
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result_1'):
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'generate_text': 1,
        'generate_text_batches': 1,
        '_messages_create': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls_1'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    # One add and one miss for generate.
    expected_cache_counters = Counter(add_new=1, get_miss=1)
    with self.subTest('cache_behaves_as_expected_1'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # Same query to see if it has been cached.
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result_2'):
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)
    with self.subTest('sends_correct_number_of_api_calls_2'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1
    with self.subTest('cache_behaves_as_expected_2'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # Same query but with a deprecated parameter (temperature). Since
    # temperature is deprecated by Claude and dropped from the cache key,
    # this should still be a cache hit.
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            max_tokens=512,
            temperature=0.0,
        )
    )
    with self.subTest('returns_correct_result_3'):
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)
    with self.subTest('sends_correct_number_of_api_calls_3'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1  # Cache hit (temperature dropped).
    with self.subTest('cache_behaves_as_expected_3'):
      self.assertCounterEqual(
          handler._cache_data.counters, expected_cache_counters
      )

  def test_generate_with_details(self):
    """Verifies that generate_text with include_details returns score."""
    _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            max_tokens=512,
            include_details=True,
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res[0], claude_mock._DEFAULT_REPLY)
    with self.subTest('returns_correct_score'):
      # Claude does not provide logprobs, so score should be 0.0.
      self.assertEqual(res[1]['score'], 0.0)  # pyrefly: ignore[bad-index]

  def test_chat_with_system_message(self):
    """Verifies that system messages are extracted to the system param."""
    backend = _get_and_register_backend()
    res = executing.run(
        llm.chat(
            messages=[
                _Message(
                    content='You are helpful.',
                    role=_PredefinedRole.SYSTEM,
                ),
                _Message(
                    content='What is 2+2?',
                    role=_PredefinedRole.USER,
                ),
            ],
            max_tokens=50,
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_messages_create': 1,
        'chat_via_api_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  @parameterized.named_parameters(
      dict(
          testcase_name='image_only',
          system_content=content_lib.ChunkList(
              [content_lib.Chunk(PIL.Image.new('RGB', (2, 2), color='red'))]
          ),
          expect_error=True,
      ),
      dict(
          testcase_name='mixed_text_and_image',
          system_content=content_lib.ChunkList([
              content_lib.Chunk('You are helpful.'),
              content_lib.Chunk(PIL.Image.new('RGB', (2, 2), color='green')),
          ]),
          expect_error=True,
      ),
      dict(
          testcase_name='image_bytes',
          system_content=content_lib.ChunkList([
              content_lib.Chunk(
                  b'\x89PNG\r\n\x1a\nfake_png_data',
                  content_type='image/png',
              ),
          ]),
          expect_error=True,
      ),
      dict(
          testcase_name='text_only_string',
          system_content='You are helpful.',
          expect_error=False,
      ),
  )
  def test_chat_with_system_message_multimodal_handling(
      self, system_content, expect_error
  ):
    """Verifies that multimodal system messages raise, text ones succeed."""
    _get_and_register_backend()
    messages = [
        _Message(content=system_content, role=_PredefinedRole.SYSTEM),
        _Message(content='What is 2+2?', role=_PredefinedRole.USER),
    ]
    if expect_error:
      with self.assertRaisesRegex(
          ValueError,
          'Claude API does not support multimodal content in system messages',
      ):
        executing.run(llm.chat(messages=messages, max_tokens=50))
    else:
      res = executing.run(llm.chat(messages=messages, max_tokens=50))
      self.assertEqual(res, claude_mock._DEFAULT_REPLY)

  def test_generate_texts(self):
    """Verifies that generate_texts returns multiple samples."""
    _get_and_register_backend()
    res = executing.run(
        llm.generate_texts(
            prompt='Hello',
            samples=3,
            max_tokens=25,
        )
    )
    self.assertLen(res, 3)
    for sample in res:
      self.assertEqual(sample, claude_mock._DEFAULT_REPLY)

  def test_generate_with_thinking_enabled(self):
    """Verifies that thinking content is returned in details when enabled."""
    backend = claude_api.ClaudeAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
        generate_text_kwargs={
            'thinking': {'type': 'adaptive', 'budget_tokens': 1024},
        },
    )
    backend.register()
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt='Solve x^2 + 4x + 4 = 0',
            max_tokens=512,
            include_details=True,
        )
    )
    with self.subTest('returns_correct_text'):
      self.assertEqual(res[0], claude_mock._DEFAULT_REPLY)
    with self.subTest('returns_thinking'):
      self.assertIn('thinking_output', res[1])
      self.assertEqual(res[1]['thinking_output'], claude_mock._DEFAULT_THINKING)  # pyrefly: ignore[bad-index]

  def test_generate_without_thinking(self):
    """Verifies that thinking is absent from details when not enabled."""
    _get_and_register_backend()
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt='Hello',
            max_tokens=512,
            include_details=True,
        )
    )
    with self.subTest('returns_correct_text'):
      self.assertEqual(res[0], claude_mock._DEFAULT_REPLY)
    with self.subTest('no_thinking_in_details'):
      self.assertNotIn('thinking_output', res[1])

  def test_generate_texts_with_thinking(self):
    """Verifies that thinking is returned in generate_texts with details."""
    backend = claude_api.ClaudeAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
        generate_text_kwargs={
            'thinking': {'type': 'adaptive', 'budget_tokens': 1024},
        },
    )
    backend.register()
    res = executing.run(
        llm.generate_texts(  # pytype: disable=wrong-keyword-args
            prompt='Hello',
            samples=2,
            max_tokens=512,
            include_details=True,
        )
    )
    self.assertLen(res, 2)
    for sample in res:
      with self.subTest('returns_correct_text'):
        self.assertEqual(sample[0], claude_mock._DEFAULT_REPLY)
      with self.subTest('returns_thinking'):
        self.assertIn('thinking_output', sample[1])

  def test_generate_text_kwargs_applied_to_chat(self):
    """Verifies that generate_text_kwargs propagates thinking to chat."""
    backend = claude_api.ClaudeAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
        generate_text_kwargs={
            'thinking': {'type': 'adaptive', 'budget_tokens': 1024},
        },
    )
    backend.register()
    res = executing.run(
        llm.chat(  # pytype: disable=wrong-keyword-args
            messages=[
                _Message(content='Hello', role=_PredefinedRole.USER),
            ],
            max_tokens=512,
        )
    )
    self.assertEqual(res, claude_mock._DEFAULT_REPLY)


class ToolUseTest(parameterized.TestCase, core_test_utils.CounterAssertions):
  """Tests for Claude API tool use support."""

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enter_context(
        mock.patch.object(
            anthropic,
            'Anthropic',
            autospec=True,
            side_effect=claude_mock.Anthropic,
        )
    )

  def test_chat_with_tools_returns_text(self):
    """Verifies that chat with tools returns text (tool_use blocks dropped)."""
    backend = claude_api.ClaudeAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
        generate_text_kwargs={
            'tools': [{
                'name': 'web_search',
                'type': 'web_search_20260209',
            }],
        },
    )
    backend.register()
    res = executing.run(
        llm.chat(  # pytype: disable=wrong-keyword-args
            messages=[
                _Message(
                    content='Search for AAPL stock price',
                    role=_PredefinedRole.USER,
                ),
            ],
            max_tokens=512,
        )
    )
    # chat returns only the text portion; tool_use blocks are silently
    # dropped since tool orchestration is not yet supported.
    self.assertIsInstance(res, str)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_messages_create': 1,
        'chat_via_api_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_chat_with_per_call_tools(self):
    """Verifies that tools can be passed per-call via kwargs."""
    backend = claude_api.ClaudeAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
    )
    backend.register()
    res = executing.run(
        llm.chat(  # pytype: disable=wrong-keyword-args
            messages=[
                _Message(
                    content='What is the weather?',
                    role=_PredefinedRole.USER,
                ),
            ],
            max_tokens=512,
            tools=[{
                'name': 'get_weather',
                'description': 'Get weather for a location',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'location': {'type': 'string'},
                    },
                    'required': ['location'],
                },
            }],
        )
    )
    # Tool use blocks are silently dropped; only text is returned.
    self.assertIsInstance(res, str)

  def test_chat_without_tools_returns_text(self):
    """Verifies that chat without tools returns plain text as before."""
    _get_and_register_backend()
    res = executing.run(
        llm.chat(  # pytype: disable=wrong-keyword-args
            messages=[
                _Message(content='Hello', role=_PredefinedRole.USER),
            ],
            max_tokens=512,
        )
    )
    self.assertEqual(res, claude_mock._DEFAULT_REPLY)

  def test_extract_response_with_tool_use_blocks(self):
    """Verifies _extract_response correctly extracts tool_use blocks."""
    backend = claude_api.ClaudeAPI(api_key='some_key', batch_size=_BATCH_SIZE)

    # Build a mock response with both text and tool_use blocks.
    class MockResponse:
      pass

    response = MockResponse()
    response.content = [  # pyrefly: ignore[missing-attribute]
        claude_mock.TextBlock(text='Let me search for that.'),
        claude_mock.ToolUseBlock(
            id='toolu_123',
            name='web_search',
            input={'query': 'AAPL stock price'},
        ),
    ]
    response.stop_reason = 'tool_use'  # pyrefly: ignore[missing-attribute]

    text, thinking, tool_calls = backend._extract_response(response)
    with self.subTest('extracts_text'):
      self.assertEqual(text, 'Let me search for that.')
    with self.subTest('no_thinking'):
      self.assertIsNone(thinking)
    with self.subTest('extracts_tool_calls'):
      self.assertLen(tool_calls, 1)
      self.assertEqual(tool_calls[0]['id'], 'toolu_123')
      self.assertEqual(tool_calls[0]['name'], 'web_search')
      self.assertEqual(tool_calls[0]['input'], {'query': 'AAPL stock price'})

  def test_extract_response_with_thinking_and_tool_use(self):
    """Verifies _extract_response with thinking + tool_use."""
    backend = claude_api.ClaudeAPI(api_key='some_key', batch_size=_BATCH_SIZE)

    class MockResponse:
      pass

    response = MockResponse()
    response.content = [  # pyrefly: ignore[missing-attribute]
        claude_mock.ThinkingBlock(thinking='I need to search.'),
        claude_mock.ToolUseBlock(
            id='toolu_789',
            name='get_weather',
            input={'location': 'Paris'},
        ),
    ]
    response.stop_reason = 'tool_use'  # pyrefly: ignore[missing-attribute]

    text, thinking, tool_calls = backend._extract_response(response)
    with self.subTest('no_text'):
      self.assertEqual(text, '')
    with self.subTest('extracts_thinking'):
      self.assertEqual(thinking, 'I need to search.')
    with self.subTest('extracts_tool_calls'):
      self.assertLen(tool_calls, 1)
      self.assertEqual(tool_calls[0]['name'], 'get_weather')


class ConvertChunkToContentBlockTest(parameterized.TestCase):
  """Tests for _convert_chunk_to_claude_part."""

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_string',
          chunk=content_lib.Chunk('Hello, world!'),
          expected={'type': 'text', 'text': 'Hello, world!'},
      ),
      dict(
          testcase_name='empty_string',
          chunk=content_lib.Chunk(''),
          expected={'type': 'text', 'text': ''},
      ),
      dict(
          testcase_name='string_with_special_characters',
          chunk=content_lib.Chunk('Line 1\nLine 2\t"quoted"'),
          expected={'type': 'text', 'text': 'Line 1\nLine 2\t"quoted"'},
      ),
  )
  def test_string_chunks(self, chunk, expected):
    result = claude_api._convert_chunk_to_claude_part(chunk)
    self.assertEqual(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='jpeg_image_bytes',
          data=b'\xff\xd8\xff\xe0fake_jpeg_data',
          content_type='image/jpeg',
          expected_mime='image/jpeg',
      ),
      dict(
          testcase_name='png_image_bytes',
          data=b'\x89PNG\r\n\x1a\nfake_png_data',
          content_type='image/png',
          expected_mime='image/png',
      ),
  )
  def test_image_bytes_chunks(self, data, content_type, expected_mime):
    chunk = content_lib.Chunk(data, content_type=content_type)
    result = claude_api._convert_chunk_to_claude_part(chunk)
    self.assertEqual(result['type'], 'image')
    self.assertEqual(result['source']['type'], 'base64')
    self.assertEqual(result['source']['media_type'], expected_mime)
    expected_b64 = base64.b64encode(data).decode('ascii')
    self.assertEqual(result['source']['data'], expected_b64)

  def test_pil_image_chunk(self):
    image = PIL.Image.new('RGB', (2, 2), color='red')
    chunk = content_lib.Chunk(image)
    result = claude_api._convert_chunk_to_claude_part(chunk)
    self.assertEqual(result['type'], 'image')
    self.assertEqual(result['source']['type'], 'base64')
    self.assertEqual(result['source']['media_type'], 'image/jpeg')
    # Verify the base64 payload is valid and decodes to non-empty bytes.
    decoded = base64.b64decode(result['source']['data'])
    self.assertNotEmpty(decoded)

  @parameterized.named_parameters(
      dict(
          testcase_name='video_bytes',
          data=b'fake_video_data',
          content_type='video/mp4',
      ),
      dict(
          testcase_name='audio_bytes',
          data=b'fake_audio_data',
          content_type='audio/wav',
      ),
      dict(
          testcase_name='application_bytes',
          data=b'fake_pdf_data',
          content_type='application/pdf',
      ),
  )
  def test_unsupported_binary_content_type_raises_value_error(
      self, data, content_type
  ):
    chunk = content_lib.Chunk(data, content_type=content_type)
    with self.assertRaisesRegex(
        ValueError, 'Unsupported binary content_type for Claude API'
    ):
      claude_api._convert_chunk_to_claude_part(chunk)


class ConvertMessagesToClaude(parameterized.TestCase):
  """Tests for _convert_messages_to_claude_format."""

  def test_simple_user_message(self):
    messages = [
        _Message(content='Hello', role=_PredefinedRole.USER),
    ]
    api_messages, system_prompt = claude_api._convert_messages_to_claude_format(
        messages
    )
    self.assertLen(api_messages, 1)
    self.assertEqual(api_messages[0]['role'], 'user')
    self.assertEqual(api_messages[0]['content'], 'Hello')
    self.assertIsNone(system_prompt)

  def test_system_message_extraction(self):
    messages = [
        _Message(content='You are helpful.', role=_PredefinedRole.SYSTEM),
        _Message(content='Hello', role=_PredefinedRole.USER),
    ]
    api_messages, system_prompt = claude_api._convert_messages_to_claude_format(
        messages
    )
    self.assertLen(api_messages, 1)
    self.assertEqual(api_messages[0]['role'], 'user')
    self.assertEqual(system_prompt, 'You are helpful.')

  def test_multiple_system_messages_concatenated(self):
    messages = [
        _Message(content='Be helpful.', role=_PredefinedRole.SYSTEM),
        _Message(content='Be concise.', role=_PredefinedRole.SYSTEM),
        _Message(content='Hello', role=_PredefinedRole.USER),
    ]
    api_messages, system_prompt = claude_api._convert_messages_to_claude_format(
        messages
    )
    self.assertLen(api_messages, 1)
    self.assertEqual(system_prompt, 'Be helpful.\nBe concise.')

  def test_multi_turn_conversation(self):
    messages = [
        _Message(content='Hello', role=_PredefinedRole.USER),
        _Message(content='Hi there!', role=_PredefinedRole.MODEL),
        _Message(content='How are you?', role=_PredefinedRole.USER),
    ]
    api_messages, system_prompt = claude_api._convert_messages_to_claude_format(
        messages
    )
    self.assertLen(api_messages, 3)
    self.assertEqual(api_messages[0]['role'], 'user')
    self.assertEqual(api_messages[1]['role'], 'assistant')
    self.assertEqual(api_messages[2]['role'], 'user')
    self.assertIsNone(system_prompt)

  @parameterized.named_parameters(
      dict(
          testcase_name='pil_image_only',
          system_content=content_lib.ChunkList(
              [content_lib.Chunk(PIL.Image.new('RGB', (2, 2), color='blue'))]
          ),
          expect_error=True,
          expected_system_prompt=None,
      ),
      dict(
          testcase_name='mixed_text_and_pil_image',
          system_content=content_lib.ChunkList([
              content_lib.Chunk('You are helpful.'),
              content_lib.Chunk(PIL.Image.new('RGB', (2, 2), color='green')),
          ]),
          expect_error=True,
          expected_system_prompt=None,
      ),
      dict(
          testcase_name='raw_image_bytes',
          system_content=content_lib.ChunkList([
              content_lib.Chunk(
                  b'\x89PNG\r\n\x1a\nfake_png_data',
                  content_type='image/png',
              ),
          ]),
          expect_error=True,
          expected_system_prompt=None,
      ),
      dict(
          testcase_name='text_only_chunklist',
          system_content=content_lib.ChunkList([
              content_lib.Chunk('Be helpful. '),
              content_lib.Chunk('Be concise.'),
          ]),
          expect_error=False,
          expected_system_prompt='Be helpful. Be concise.',
      ),
      dict(
          testcase_name='plain_text_string',
          system_content='You are helpful.',
          expect_error=False,
          expected_system_prompt='You are helpful.',
      ),
  )
  def test_system_message_multimodal_handling(
      self, system_content, expect_error, expected_system_prompt
  ):
    """Verifies multimodal system messages raise, text ones succeed."""
    messages = [
        _Message(content=system_content, role=_PredefinedRole.SYSTEM),
        _Message(content='Hello', role=_PredefinedRole.USER),
    ]
    if expect_error:
      with self.assertRaisesRegex(
          ValueError,
          'Claude API does not support multimodal content in system messages',
      ):
        claude_api._convert_messages_to_claude_format(messages)
    else:
      api_messages, system_prompt = (
          claude_api._convert_messages_to_claude_format(messages)
      )
      self.assertLen(api_messages, 1)
      self.assertEqual(system_prompt, expected_system_prompt)


class CountTokensTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enter_context(
        mock.patch.object(
            anthropic,
            'Anthropic',
            autospec=True,
            side_effect=claude_mock.Anthropic,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='string',
          content='Hello world',
      ),
      dict(
          testcase_name='chunklist',
          content=content_lib.ChunkList([
              content_lib.Chunk('Hello, '),
              content_lib.Chunk('world!'),
          ]),
      ),
      dict(
          testcase_name='messages',
          content=[
              _Message(content='What is 2 + 2?', role=_PredefinedRole.USER),
              _Message(content='2 + 2 equals 4.', role=_PredefinedRole.MODEL),
              _Message(content='What about 3 + 3?', role=_PredefinedRole.USER),
          ],
      ),
      dict(
          testcase_name='multimodal_chunklist',
          content=content_lib.ChunkList([
              content_lib.Chunk('Describe this image:'),
              content_lib.Chunk(PIL.Image.new('RGB', (2, 2), color='red')),
          ]),
      ),
  )
  def test_count_tokens_returns_positive_int(self, content):
    """Verifies that count_tokens returns a positive int for various inputs."""
    backend = _get_and_register_backend()
    result = executing.run(
        llm.count_tokens(content=content)  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('returns_int'):
      self.assertIsInstance(result, int)
    with self.subTest('returns_positive'):
      self.assertGreater(result, 0)
    expected_backend_counters = collections.Counter({
        'count_tokens': 1,
        '_count_tokens_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_count_tokens_messages_more_than_string(self):
    """Verifies that multi-turn messages have more tokens than single-turn."""
    _get_and_register_backend()
    text = 'Hello world'
    single_message = [
        _Message(content=text, role=_PredefinedRole.USER),
    ]
    multi_messages = [
        _Message(content=text, role=_PredefinedRole.USER),
        _Message(content='Hi there!', role=_PredefinedRole.MODEL),
        _Message(content='How are you?', role=_PredefinedRole.USER),
    ]
    count_single = executing.run(
        llm.count_tokens(content=single_message)  # pytype: disable=wrong-keyword-args
    )
    count_multi = executing.run(
        llm.count_tokens(content=multi_messages)  # pytype: disable=wrong-keyword-args
    )
    # More messages should have more tokens (per-message overhead).
    self.assertGreater(count_multi, count_single)

  def test_count_tokens_caching(self):
    """Verifies that count_tokens results are cached."""
    backend = _get_and_register_backend()
    content_text = 'Cache me!'
    # First call.
    result1 = executing.run(
        llm.count_tokens(content=content_text)  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('first_call_hits_api'):
      self.assertEqual(backend._counters['count_tokens'], 1)
    # Second call with same content should be cached.
    result2 = executing.run(
        llm.count_tokens(content=content_text)  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('second_call_is_cached'):
      self.assertEqual(backend._counters['count_tokens'], 1)
    with self.subTest('results_are_equal'):
      self.assertEqual(result1, result2)


class StopSequenceFilteringTest(parameterized.TestCase):
  """Tests for whitespace-only stop sequence filtering in _messages_create."""

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enter_context(
        mock.patch.object(
            anthropic,
            'Anthropic',
            autospec=True,
            side_effect=claude_mock.Anthropic,
        )
    )

  def _make_backend(self) -> claude_api.ClaudeAPI:
    return claude_api.ClaudeAPI(api_key='some_key', batch_size=_BATCH_SIZE)

  def _make_messages(self) -> list[content_lib.Message]:
    return [
        _Message(content='Hello', role=_PredefinedRole.USER),
    ]

  @parameterized.named_parameters(
      dict(
          testcase_name='newline_only',
          stop=['\n'],
          expected_stop_sequences=None,  # All dropped → key absent.
      ),
      dict(
          testcase_name='whitespace_variants',
          stop=['\n', '\t', '  ', '\r\n'],
          expected_stop_sequences=None,
      ),
      dict(
          testcase_name='valid_sequences_pass_through',
          stop=['.', '!', 'END'],
          expected_stop_sequences=['.', '!', 'END'],
      ),
      dict(
          testcase_name='mixed_valid_and_whitespace',
          stop=['\n', '.', '\t', '!'],
          expected_stop_sequences=['.', '!'],
      ),
      dict(
          testcase_name='single_valid_sequence',
          stop=['STOP'],
          expected_stop_sequences=['STOP'],
      ),
      dict(
          testcase_name='empty_list',
          stop=[],
          expected_stop_sequences=None,  # Empty filtered list → key absent.
      ),
  )
  def test_stop_sequences_filtering(self, stop, expected_stop_sequences):
    backend = self._make_backend()
    messages = self._make_messages()

    from unittest import mock  # pylint: disable=g-import-not-at-top

    assert backend._client is not None
    with mock.patch.object(
        backend._client.messages,
        'create',
        wraps=backend._client.messages.create,
    ) as mock_create:
      backend._messages_create(messages=messages, stop=stop)  # pytype: disable=wrong-keyword-args
      mock_create.assert_called_once()
      call_kwargs = mock_create.call_args[1]
      if expected_stop_sequences is None:
        self.assertNotIn('stop_sequences', call_kwargs)
      else:
        self.assertEqual(call_kwargs['stop_sequences'], expected_stop_sequences)

  def test_stop_none_omits_stop_sequences(self):
    backend = self._make_backend()
    messages = self._make_messages()

    from unittest import mock  # pylint: disable=g-import-not-at-top

    assert backend._client is not None
    with mock.patch.object(
        backend._client.messages,
        'create',
        wraps=backend._client.messages.create,
    ) as mock_create:
      backend._messages_create(messages=messages, stop=None)  # pytype: disable=wrong-keyword-args
      mock_create.assert_called_once()
      call_kwargs = mock_create.call_args[1]
      self.assertNotIn('stop_sequences', call_kwargs)

  def test_whitespace_stop_logs_warning(self):
    backend = self._make_backend()
    messages = self._make_messages()

    with self.assertLogs(level='WARNING') as log_ctx:
      backend._messages_create(messages=messages, stop=['\n', '.'])  # pytype: disable=wrong-keyword-args

    self.assertTrue(
        any('whitespace-only stop sequence' in msg for msg in log_ctx.output),
        'Expected warning about whitespace stop sequences, got:'
        f' {log_ctx.output}',
    )

  def test_valid_stop_does_not_log_warning(self):
    backend = self._make_backend()
    messages = self._make_messages()
    # No warning should be raised — verify the call succeeds without error.
    backend._messages_create(messages=messages, stop=['.', '!'])  # pytype: disable=wrong-keyword-args

  def test_generate_text_with_whitespace_stop(self):
    _get_and_register_backend()
    # This was the original user error scenario.
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt='What is the capital of France?',
            max_tokens=50,
            stop=['\n'],
        )
    )
    self.assertEqual(res, claude_mock._DEFAULT_REPLY)

  def test_instruct_with_newline_stop(self):
    _get_and_register_backend()
    res = executing.run(
        llm.instruct(  # pytype: disable=wrong-arg-count
            'Question: What is 2+2?\nAnswer:',
            stop=['\n'],
        )
    )
    self.assertEqual(res, claude_mock._DEFAULT_REPLY)

  def test_generate_texts_with_whitespace_stop(self):
    _get_and_register_backend()
    res = executing.run(
        llm.generate_texts(  # pytype: disable=wrong-keyword-args
            prompt='Hello',
            samples=2,
            max_tokens=25,
            stop=['\n'],
        )
    )
    self.assertLen(res, 2)
    for sample in res:
      self.assertEqual(sample, claude_mock._DEFAULT_REPLY)


def _get_weather(location: str) -> dict[str, Any]:
  """Gets the weather for a specified location.

  Args:
    location: City name.

  Returns:
    A dictionary containing weather information.
  """
  return {'location': location, 'temp': 15, 'condition': 'Rainy'}


# Backends are handed declarations, never implementations, so this is what
# `_get_weather` looks like from a backend's point of view.
_WEATHER_SPEC = tool_schema.tool_from_callable(_get_weather).to_spec()


class _ScriptedMessages:
  """Stands in for `client.messages`, replying with what it was scripted to.

  Recording the requests is the point: most of what this backend does for
  function calling is deciding what to put on the wire, so that is what the
  tests look at.
  """

  def __init__(self, *replies: claude_mock.Message):
    self.replies = list(replies)
    self.requests: list[dict[str, Any]] = []

  def create(self, **api_kwargs) -> claude_mock.Message:
    self.requests.append(api_kwargs)
    # The last reply is repeated, so that a test which only cares about one
    # turn does not have to script every turn the model might take.
    return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class _ScriptedClient:
  """A client whose `messages` namespace is scripted."""

  def __init__(self, messages: _ScriptedMessages):
    self.messages = messages


class GenerateWithToolsTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):
  """Tests the single model call that `FCAgent` is built on."""

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

  def _script(self, *replies: claude_mock.Message) -> claude_api.ClaudeAPI:
    """Registers a backend whose client replies with `replies`, in order."""
    backend = _get_and_register_backend()
    self._messages = _ScriptedMessages(*replies)
    # A test double standing in for the client, which `_client` is not
    # annotated to accept.
    backend._client = _ScriptedClient(self._messages)  # pyrefly: ignore[bad-assignment]
    return backend

  def _last_request(self) -> dict[str, Any]:
    return self._messages.requests[-1]

  def _ask(self, *messages, **kwargs):
    return executing.run(
        llm.generate_with_tools(
            messages=list(messages), tools=[_WEATHER_SPEC], **kwargs
        )
    )

  def _question(self) -> _ConversationTurn:
    return _ConversationTurn(
        role=_ConversationRole.USER, content='What is the weather in London?'
    )

  def test_lifts_the_requested_call(self):
    self._script(
        claude_mock.Message(
            content=[
                claude_mock.ToolUseBlock(
                    id='toolu_1',
                    name='_get_weather',
                    input={'location': 'London'},
                )
            ],
            stop_reason='tool_use',
        )
    )

    turn = self._ask(self._question())

    with self.subTest('returns_a_single_model_turn'):
      self.assertEqual(turn.role, _ConversationRole.MODEL)
    with self.subTest('lifts_the_call'):
      self.assertLen(turn.calls, 1)
      self.assertEqual(turn.calls[0].function_name, '_get_weather')
      self.assertEqual(dict(turn.calls[0].kwargs), {'location': 'London'})
    with self.subTest('keeps_the_id_so_responses_can_be_matched'):
      self.assertEqual(turn.calls[0].id, 'toolu_1')

  def test_performs_exactly_one_model_call(self):
    # The model asks for a tool. A backend must not act on that: executing the
    # call and asking again is the agent's job, not the backend's.
    backend = self._script(
        claude_mock.Message(
            content=[
                claude_mock.ToolUseBlock(
                    id='toolu_1',
                    name='_get_weather',
                    input={'location': 'London'},
                )
            ],
            stop_reason='tool_use',
        )
    )

    turn = self._ask(self._question())

    with self.subTest('calls_the_model_once'):
      self.assertLen(self._messages.requests, 1)
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_with_tools=1, _generate_with_tools_batches=1),
      )
    with self.subTest('hands_the_call_back_unexecuted'):
      self.assertLen(turn.calls, 1)
      self.assertIsNone(turn.content)

  def test_declares_the_tools_as_json_schema(self):
    self._script(claude_mock.Message(content=[claude_mock.TextBlock('Hi.')]))

    self._ask(self._question())

    tools = self._last_request()['tools']
    self.assertLen(tools, 1)
    with self.subTest('declares_the_name_and_description'):
      self.assertEqual(tools[0]['name'], '_get_weather')
      self.assertStartsWith(tools[0]['description'], 'Gets the weather')
    with self.subTest('passes_the_json_schema_through_untouched'):
      self.assertEqual(
          tools[0]['input_schema'], dict(_WEATHER_SPEC.parameters or {})
      )

  def test_declares_an_empty_schema_for_a_tool_without_parameters(self):
    # Anthropic requires `input_schema` even when there is nothing to describe.
    self._script(claude_mock.Message(content=[claude_mock.TextBlock('Hi.')]))

    executing.run(
        llm.generate_with_tools(
            messages=[self._question()],
            tools=[_ToolSpec(name='ping', description='Pings.')],
        )
    )

    self.assertEqual(
        self._last_request()['tools'][0]['input_schema'],
        {'type': 'object', 'properties': {}},
    )

  def test_advertises_nothing_when_given_no_tools(self):
    self._script(claude_mock.Message(content=[claude_mock.TextBlock('Hi.')]))

    executing.run(
        llm.generate_with_tools(messages=[self._question()], tools=[])
    )

    # Advertising an empty list is not the same as advertising nothing.
    self.assertNotIn('tools', self._last_request())

  def test_sends_tool_results_back_as_user_content(self):
    self._script(
        claude_mock.Message(
            content=[claude_mock.TextBlock('It is 15 degrees in London.')]
        )
    )

    turn = self._ask(
        self._question(),
        _ConversationTurn(
            role=_ConversationRole.MODEL,
            calls=[
                _FunctionCall(
                    function_name='_get_weather',
                    kwargs={'location': 'London'},
                    id='toolu_1',
                )
            ],
        ),
        _ConversationTurn(
            role=_ConversationRole.TOOL,
            responses=[
                _FunctionResponse(
                    function_name='_get_weather',
                    response={'temp': 15},
                    id='toolu_1',
                )
            ],
        ),
    )

    messages = self._last_request()['messages']
    with self.subTest('sends_the_earlier_call_as_the_assistant'):
      self.assertEqual(messages[1]['role'], 'assistant')
      self.assertEqual(
          messages[1]['content'][0],
          {
              'type': 'tool_use',
              'id': 'toolu_1',
              'name': '_get_weather',
              'input': {'location': 'London'},
          },
      )
    with self.subTest('sends_the_result_as_the_user'):
      # Claude has no tool role, so results come back as user content.
      self.assertEqual(messages[2]['role'], 'user')
      self.assertEqual(
          messages[2]['content'][0],
          {
              'type': 'tool_result',
              'tool_use_id': 'toolu_1',
              'content': '{"temp": 15}',
          },
      )
    with self.subTest('returns_the_final_answer'):
      self.assertEqual(turn.content, 'It is 15 degrees in London.')

  def test_reports_tool_errors_to_the_model(self):
    self._script(
        claude_mock.Message(
            content=[claude_mock.TextBlock('Sorry, that failed.')]
        )
    )

    self._ask(
        _ConversationTurn(
            role=_ConversationRole.TOOL,
            responses=[
                _FunctionResponse(
                    function_name='_get_weather',
                    error='Unknown tool: _get_weathr.',
                    id='toolu_1',
                )
            ],
        )
    )

    self.assertEqual(
        self._last_request()['messages'][0]['content'][0],
        {
            'type': 'tool_result',
            'tool_use_id': 'toolu_1',
            'content': 'Unknown tool: _get_weathr.',
            'is_error': True,
        },
    )

  def test_round_trips_a_thought_signature(self):
    # Once thinking is on, Anthropic rejects a turn whose thinking blocks did
    # not come back with the tool results, so the signature has to survive the
    # trip through `ConversationTurn`.
    self._script(
        claude_mock.Message(
            content=[
                claude_mock.ThinkingBlock(
                    thinking='I should look up the weather.',
                    signature='opaque',
                ),
                claude_mock.TextBlock('Checking.'),
            ]
        )
    )

    question = self._question()
    first = self._ask(question)

    with self.subTest('keeps_the_reasoning_out_of_the_answer'):
      self.assertEqual(first.content, 'Checking.')
      self.assertIsNotNone(first.thought)

    # Feeding the turn straight back is how a caller continues a conversation,
    # so that is what has to preserve the signature.
    self._ask(question, first)

    resent = self._last_request()['messages'][1]['content'][0]
    with self.subTest('resends_the_thought_first_and_verbatim'):
      self.assertEqual(
          resent,
          {
              'type': 'thinking',
              'thinking': 'I should look up the weather.',
              'signature': 'opaque',
          },
      )

  def test_round_trips_a_signature_with_an_empty_body(self):
    # Adaptive thinking returns blocks whose body is empty but whose signature
    # is real. The `ChunkList` holding one of those is falsy, so a truthiness
    # test drops it and Anthropic then rejects the follow-up request. Found
    # against the live API, so it stays covered here.
    self._script(
        claude_mock.Message(
            content=[
                claude_mock.ThinkingBlock(thinking='', signature='opaque'),
                claude_mock.TextBlock('Checking.'),
            ]
        )
    )

    question = self._question()
    first = self._ask(question)
    self._ask(question, first)

    resent = self._last_request()['messages'][1]['content'][0]
    self.assertEqual(
        resent,
        {'type': 'thinking', 'thinking': '', 'signature': 'opaque'},
    )

  def test_runs_the_whole_loop_through_the_agent(self):
    # What all of the above is for: `FCAgent` asks, executes, and asks again,
    # and none of that is the backend's business beyond getting the wire
    # format right.
    self._script(
        claude_mock.Message(
            content=[
                claude_mock.ToolUseBlock(
                    id='toolu_1',
                    name='_get_weather',
                    input={'location': 'London'},
                )
            ],
            stop_reason='tool_use',
        ),
        claude_mock.Message(
            content=[claude_mock.TextBlock('It is 15 degrees in London.')]
        ),
    )

    agent = fc_agents.FCAgent(tools=[_get_weather])
    # `Agent.__call__` is declared to return a union that also covers
    # `return_final_state`, which this caller never asks for.
    turn = cast(
        _ConversationTurn,
        executing.run(agent('What is the weather in London?')),
    )

    with self.subTest('reaches_an_answer'):
      self.assertEqual(turn.content, 'It is 15 degrees in London.')
    with self.subTest('took_two_model_calls'):
      self.assertLen(self._messages.requests, 2)
    with self.subTest('fed_the_tool_result_back'):
      self.assertEqual(
          self._last_request()['messages'][2]['content'][0]['content'],
          '{"location": "London", "temp": 15, "condition": "Rainy"}',
      )


if __name__ == '__main__':
  absltest.main()
