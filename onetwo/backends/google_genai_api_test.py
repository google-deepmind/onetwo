# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for GoogleGenAIAPI engine."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Counter, Final, TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai
from google.genai import types as genai_types
from onetwo.backends import google_genai_api
from onetwo.builtins import llm as llm_lib
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import sampling
import pydantic

# For escaping the `pytype: disable=wrong-keyword-args`
llm: Any = llm_lib

_Message: TypeAlias = content_lib.Message
_ChunkList: TypeAlias = content_lib.ChunkList
_Chunk: TypeAlias = content_lib.Chunk
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_THREADPOOL_SIZE: Final[int] = 4

_MOCK_COUNT_TOKENS_RETURN: Final[int] = 5


def _mock_list_models() -> Sequence[genai_types.Model]:
  fake_base_params = {
      'version': '',
      'display_name': '',
      'description': '',
      'input_token_limit': 1,
      'output_token_limit': 1,
  }
  return [
      genai_types.Model(
          name=google_genai_api.DEFAULT_GENERATE_MODEL.gemini_api,
          supported_actions=['generateContent', 'countTokens'],
          **fake_base_params,
      ),
      genai_types.Model(
          name=google_genai_api.DEFAULT_MULTIMODAL_MODEL.gemini_api,
          supported_actions=['generateContent', 'countTokens'],
          **fake_base_params,
      ),
      genai_types.Model(
          name=google_genai_api.DEFAULT_EMBED_MODEL.gemini_api,
          supported_actions=['embedContent'],
          **fake_base_params,
      ),
  ]


def _get_and_register_backend(**kwargs) -> google_genai_api.GoogleGenAIAPI:
  """Get an instance of GoogleGenAIAPI and register its methods."""
  backend = google_genai_api.GoogleGenAIAPI(
      api_key='some_key', threadpool_size=_THREADPOOL_SIZE, **kwargs
  )
  backend.register()
  return backend


@dataclasses.dataclass
class RespondWithTextAndTemperature(
    Callable[[genai_types.GenerateContentConfig], Sequence[str]]
):
  """Returns a fixed reply text, with temperature appended if specified."""

  reply: str

  def __call__(
      self, config: genai_types.GenerateContentConfig
  ) -> Sequence[str]:
    candidate_count = config.candidate_count or 1
    if config.temperature:
      reply_text = f'{self.reply} {config.temperature:.1f}'
    else:
      reply_text = self.reply
    return [reply_text] * candidate_count


class GoogleGenaiApiTest(
    parameterized.TestCase,
    core_test_utils.CounterAssertions,
):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

    self._mock_genai_client = mock.create_autospec(genai.Client)
    self._mock_genai_client.models.generate_content.return_value = (
        genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        parts=[genai_types.Part(text='text')]
                    )
                )
            ]
        )
    )
    self.enter_context(
        mock.patch.object(genai, 'Client', return_value=self._mock_genai_client)
    )

    self._mock_genai_client.models.list.return_value = _mock_list_models()

  def test_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(llm.generate_text(prompt='Something1'))
    res2 = executing.run(llm.generate_text(prompt='Something2'))

    self.assertEqual(res, 'text')
    self.assertEqual(res2, 'text')
    self.assertEqual(
        self._mock_genai_client.models.generate_content.call_count, 2
    )
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=2),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=2, get_miss=2),
    )

  def test_generate_text_with_unsafe_content(self):
    backend = _get_and_register_backend()

    # Empty content is returned when the prompt is unsafe.
    self._mock_genai_client.models.generate_content.return_value = (
        genai_types.GenerateContentResponse(
            candidates=[genai_types.Candidate(content=None)]
        )
    )

    # Assert the appropriate error is raised.
    with self.assertRaises(
        ValueError,
        msg=(
            'GoogleGenAIAPI.generate_text returned no answers. This may be'
            ' caused by safety filters*'
        ),
    ):
      executing.run(llm.generate_text(prompt='Some unsafe prompt'))

    # Assert that the chat is called with the correct prompt.
    self._mock_genai_client.models.generate_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.generate_content.call_args
    print('#' * 80)
    print(mock_kwargs)
    print('#' * 80)
    self.assertEqual(mock_kwargs['contents'][0].text, 'Some unsafe prompt')
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=1),
    )

  def test_generate_text_with_model_string(self):
    backend = _get_and_register_backend(
        generate_model_name=google_genai_api.DEFAULT_GENERATE_MODEL.gemini_api
    )
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(llm.generate_text(prompt='Something1'))
    res2 = executing.run(llm.generate_text(prompt='Something2'))

    self.assertEqual(res, 'text')
    self.assertEqual(res2, 'text')
    self.assertEqual(
        self._mock_genai_client.models.generate_content.call_count, 2
    )
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=2),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=2, get_miss=2),
    )

  def test_generate_text_cached(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(llm.generate_text(prompt='Something'))
    res2 = executing.run(llm.generate_text(prompt='Something'))

    self.assertEqual(res, 'text')
    self.assertEqual(res2, 'text')
    self._mock_genai_client.models.generate_content.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1, get_hit=1),
    )

  def test_repeat_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=5,
            )
        )
    )
    self.assertEqual(res, ['text', 'text', 'text', 'text', 'text'])
    self.assertEqual(
        self._mock_genai_client.models.generate_content.call_count, 5
    )
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=5),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, add_new_sample=4, get_miss=1, get_hit_miss_sample=4),
    )

  def test_repeat_generate_text_cached(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=5,
            )
        )
    )
    res2 = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=6,
            )
        )
    )

    self.assertEqual(res, ['text', 'text', 'text', 'text', 'text'])
    self.assertEqual(res2, ['text', 'text', 'text', 'text', 'text', 'text'])
    self.assertEqual(
        self._mock_genai_client.models.generate_content.call_count, 6
    )
    self.assertCounterEqual(
        backend._counters,
        Counter(generate_text=6),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(
            add_new=1,  # The first generate_text from the first run.
            add_new_sample=5,  # 4 from the first run and 1 from the second run.
            get_miss=1,  # The first generate_text from the first run.
            get_hit=5,  # The second run hits the cache for the first 5 samples.
            get_hit_miss_sample=5,  # 4 from 1st run + 1 from 2nd run.
        ),
    )

  def test_batched_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(
        executing.par_iter([
            llm.generate_text(prompt='Something a'),
            llm.generate_text(prompt='Something b'),
            llm.generate_text(prompt='Something c'),
            llm.generate_text(prompt='Something d'),
            llm.generate_text(prompt='Something f'),
        ])
    )

    self.assertEqual(res, ['text', 'text', 'text', 'text', 'text'])
    self.assertEqual(
        self._mock_genai_client.models.generate_content.call_count, 5
    )
    self.assertCounterEqual(
        backend._counters,
        Counter(
            generate_text=5,
        ),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(
            add_new=5,
            get_miss=5,
        ),
    )

  def test_identical_batched_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(
        executing.par_iter([
            llm.generate_text(prompt='Something'),
            llm.generate_text(prompt='Something'),
            llm.generate_text(prompt='Something'),
            llm.generate_text(prompt='Something'),
            llm.generate_text(prompt='Something'),
        ])
    )

    self.assertEqual(res, ['text', 'text', 'text', 'text', 'text'])
    self._mock_genai_client.models.generate_content.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(
            generate_text=1,
        ),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(
            add_new=1,
            get_miss=1,
            get_hit=4,
        ),
    )

  def test_generate_texts(self):
    _ = _get_and_register_backend()

    results = executing.run(llm.generate_texts(prompt='prompt', samples=3))

    self.assertLen(results, 3)
    self.assertListEqual(list(results), ['text', 'text', 'text'])

  def test_generate_object_pydantic(self):
    backend = _get_and_register_backend()

    class Recipe(pydantic.BaseModel):
      recipe_name: str
      ingredients: list[str]

    expected_objects = [
        Recipe(
            recipe_name='Chocolate Chip Cookies',
            ingredients=['1 cup butter', '3/4 cup sugar'],
        ),
        Recipe(
            recipe_name='Sugar Cookies',
            ingredients=['1 cup sugar', '1/2 cup butter'],
        ),
    ]

    mock_response = mock.create_autospec(
        genai_types.GenerateContentResponse, instance=True
    )
    mock_response.candidates = [mock.MagicMock()]

    parsed_property_mock = mock.PropertyMock(return_value=expected_objects)
    type(mock_response).parsed = parsed_property_mock

    self._mock_genai_client.models.generate_content.return_value = mock_response

    prompt = 'List some cookie recipes'
    result = executing.run(llm.generate_object(prompt=prompt, cls=list[Recipe]))

    with self.subTest('ResultsStructuredCorrectly'):
      self.assertEqual(result, expected_objects)
      self.assertIsInstance(result, list)
      self.assertLen(result, 2)
      if result:
        self.assertIsInstance(result[0], Recipe)
        self.assertEqual(result[0].recipe_name, 'Chocolate Chip Cookies')
        self.assertEqual(
            result[0].ingredients, ['1 cup butter', '3/4 cup sugar']
        )

    with self.subTest('GenerateContentCall'):
      self._mock_genai_client.models.generate_content.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.models.generate_content.call_args
      self.assertEqual(
          mock_kwargs['config'].response_mime_type, 'application/json'
      )
      self.assertEqual(mock_kwargs['config'].response_schema, list[Recipe])
      self.assertEqual(mock_kwargs['contents'][0].text, prompt)

    with self.subTest('Counters'):
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_object=1),
      )

    with self.subTest('ParsedProperty'):
      parsed_property_mock.assert_called_once()

  def test_chat(self):
    backend = _get_and_register_backend()
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    msg_model = content_lib.Message(
        role=content_lib.PredefinedRole.MODEL, content='Hello user'
    )

    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    mock_chat.send_message.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='Hello')]
                )
            )
        ]
    )

    # First call, no history.
    _ = executing.run(llm.chat(messages=[msg_user]))
    with self.subTest('ChatCreatedAndEmptyHistory'):
      self._mock_genai_client.chats.create.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.chats.create.call_args
      self.assertEmpty(mock_kwargs['history'])
    with self.subTest('FirstMessageSent'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')

    self._mock_genai_client.chats.create.reset_mock()
    mock_chat.send_message.reset_mock()

    # Second call, with one message in the history.
    _ = executing.run(llm.chat(messages=[msg_model, msg_user]))
    with self.subTest('ChatCreatedWithHistory'):
      self._mock_genai_client.chats.create.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.chats.create.call_args
      self.assertLen(mock_kwargs['history'], 1)
      self.assertEqual(mock_kwargs['history'][0].role, 'model')
      self.assertEqual(mock_kwargs['history'][0].parts[0].text, 'Hello user')
    with self.subTest('SecondMessageSent'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')

    self.assertCounterEqual(
        backend._counters,
        Counter(chat=2),
    )

  def test_chat_with_unsafe_content(self):
    backend = _get_and_register_backend()
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )

    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    # Empty content is returned when the message is unsafe.
    mock_chat.send_message.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=None)]
    )

    # Assert the appropriate error is raised.
    with self.assertRaises(
        ValueError,
        msg=(
            'GoogleGenAIAPI.generate_text returned no answers. This may be'
            ' caused by safety filters*'
        ),
    ):
      executing.run(llm.chat(messages=[msg_user]))

    # Assert that the chat is called with the correct message.
    mock_chat.send_message.assert_called_once()
    _, mock_kwargs = mock_chat.send_message.call_args
    self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')
    self.assertCounterEqual(
        backend._counters,
        Counter(chat=1),
    )

  def test_chat_with_system_instruction(self):
    backend = _get_and_register_backend()
    msg_system = content_lib.Message(
        role=content_lib.PredefinedRole.SYSTEM, content='System message'
    )
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )

    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    mock_chat.send_message.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='Hello')]
                )
            )
        ]
    )

    # First call, with system instruction.
    _ = executing.run(llm.chat(messages=[msg_system, msg_user]))
    with self.subTest('FirstChatCreatedAndEmptyHistory'):
      self._mock_genai_client.chats.create.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.chats.create.call_args
      self.assertEmpty(mock_kwargs['history'])
    with self.subTest('FirstMessageSentWithSystemInstruction'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')
      config = mock_kwargs['config']
      self.assertEqual(config.system_instruction, 'System message')

    self._mock_genai_client.chats.create.reset_mock()
    mock_chat.send_message.reset_mock()

    # Second call, with system instruction passed as a kwarg.
    _ = executing.run(
        llm.chat(messages=[msg_user], system_instruction='System message 2')
    )
    with self.subTest('SecondChatCreatedAndEmptyHistory'):
      self._mock_genai_client.chats.create.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.chats.create.call_args
      self.assertEmpty(mock_kwargs['history'])
    with self.subTest('SecondMessageSentWithSystemInstruction'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')
      config = mock_kwargs['config']
      self.assertEqual(config.system_instruction, 'System message 2')

    self.assertCounterEqual(
        backend._counters,
        Counter(chat=2),
    )

  def test_chat_replace_unsupported_roles(self):
    backend = _get_and_register_backend(replace_unsupported_roles=True)

    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    mock_chat.send_message.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='Hello')]
                )
            )
        ]
    )

    executing.run(
        llm.chat(
            messages=[
                content_lib.Message(
                    role=content_lib.PredefinedRole.SYSTEM,
                    content='System message',
                ),
                content_lib.Message(
                    role='system',
                    content='System message 2',
                ),
                content_lib.Message(
                    role=content_lib.PredefinedRole.CONTEXT,
                    content='Context message',
                ),
                content_lib.Message(
                    role='context',
                    content='Context message 2',
                ),
                content_lib.Message(
                    role=content_lib.PredefinedRole.USER,
                    content='Hello model',
                ),
            ]
        )
    )

    with self.subTest('ChatCreatedWithCorrectHistory'):
      self._mock_genai_client.chats.create.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.chats.create.call_args
      called_history = mock_kwargs['history']

      self.assertEqual(called_history[0].role, 'user')
      self.assertEqual(called_history[0].parts[0].text, 'System message')
      self.assertEqual(called_history[1].role, 'user')
      self.assertEqual(called_history[1].parts[0].text, 'System message 2')
      self.assertEqual(called_history[2].role, 'user')
      self.assertEqual(called_history[2].parts[0].text, 'Context message')
      self.assertEqual(called_history[3].role, 'user')
      self.assertEqual(called_history[3].parts[0].text, 'Context message 2')

    with self.subTest('CorrectMessageSent'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      called_message = mock_kwargs['message']
      self.assertEqual(called_message[0].text, 'Hello model')

    self.assertCounterEqual(
        backend._counters,
        Counter(chat=1),
    )

  @parameterized.named_parameters(
      (
          'str_prompt',
          'this is a prompt',
          [genai_types.Part(text='this is a prompt')],
      ),
      (
          'chunk_str_empty_content_type',
          _ChunkList(chunks=[_Chunk(content='this is a prompt')]),
          [genai_types.Part(text='this is a prompt')],
      ),
      (
          'chunk_str',
          _ChunkList(
              chunks=[_Chunk(content='this is a prompt', content_type='str')]
          ),
          [genai_types.Part(text='this is a prompt')],
      ),
      (
          'chunk_image_jpeg',
          _ChunkList(
              chunks=[
                  _Chunk(content=b'this is an image', content_type='image/jpeg')
              ]
          ),
          [
              genai_types.Part(
                  inline_data=genai_types.Blob(
                      mime_type='image/jpeg', data=b'this is an image'
                  )
              ),
          ],
      ),
      (
          'chunk_video_mp4',
          _ChunkList(
              chunks=[
                  _Chunk(content=b'this is a video', content_type='video/mp4')
              ]
          ),
          [
              genai_types.Part(
                  inline_data=genai_types.Blob(
                      mime_type='video/mp4', data=b'this is a video'
                  )
              ),
          ],
      ),
      (
          'chunk_mixed',
          _ChunkList(
              chunks=[
                  _Chunk(content='text part'),
                  _Chunk(content=b'image part', content_type='image/png'),
              ]
          ),
          [
              genai_types.Part(text='text part'),
              genai_types.Part(
                  inline_data=genai_types.Blob(
                      mime_type='image/png', data=b'image part'
                  )
              ),
          ],
      ),
  )
  def test_convert_chunk_list_to_part_list(self, prompt, exp_contents_type):
    self.assertEqual(
        google_genai_api._convert_chunk_list_to_part_list(prompt),
        exp_contents_type,
    )

  def test_count_tokens_string(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    _ = executing.run(llm.count_tokens(content='Something'))

    self._mock_genai_client.models.count_tokens.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(count_tokens=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )
    self.assertEqual(
        self._mock_genai_client.models.count_tokens.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )

  def test_count_tokens_chunk_list(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    _ = executing.run(
        llm.count_tokens(
            content=content_lib.ChunkList(
                chunks=[content_lib.Chunk(content='Something')]
            )
        )
    )

    self._mock_genai_client.models.count_tokens.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(count_tokens=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )
    self.assertEqual(
        self._mock_genai_client.models.count_tokens.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )

  def test_tokenize_string(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of compute_tokens to return a constant value.
    mock_tokens = [1, 2, 3, 4, 5]
    self._mock_genai_client.models.compute_tokens.return_value = (
        genai_types.ComputeTokensResponse(
            tokens_info=[genai_types.TokensInfo(token_ids=mock_tokens)]
        )
    )
    content = 'Something'
    res = executing.run(llm.tokenize(content=content))

    self.assertEqual(res, mock_tokens)
    self.assertEqual(
        self._mock_genai_client.models.compute_tokens.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )
    self._mock_genai_client.models.compute_tokens.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(tokenize=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )

  def test_tokenize_chunk_list(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of compute_tokens to return a constant value.
    mock_tokens = [1, 2, 3, 4, 5]
    self._mock_genai_client.models.compute_tokens.return_value = (
        genai_types.ComputeTokensResponse(
            tokens_info=[genai_types.TokensInfo(token_ids=mock_tokens)]
        )
    )
    content = content_lib.ChunkList(
        chunks=[content_lib.Chunk(content='Something')]
    )
    res = executing.run(llm.tokenize(content=content))

    self.assertEqual(res, mock_tokens)
    self.assertEqual(
        self._mock_genai_client.models.compute_tokens.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )
    self._mock_genai_client.models.compute_tokens.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(tokenize=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )

  def test_embed_string(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of embed_content to return a constant value.
    mock_embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=mock_embedding)]
        )
    )
    content = 'Something'
    res = executing.run(llm.embed(content=content))

    self.assertEqual(res, mock_embedding)
    self.assertEqual(
        self._mock_genai_client.models.embed_content.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )
    self._mock_genai_client.models.embed_content.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(embed=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )

  def test_embed_chunk_list(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of embed_content to return a constant value.
    mock_embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=mock_embedding)]
        )
    )
    content = content_lib.ChunkList(
        chunks=[content_lib.Chunk(content='Something')]
    )
    res = executing.run(llm.embed(content=content))

    self.assertEqual(res, mock_embedding)
    self.assertEqual(
        self._mock_genai_client.models.embed_content.call_args[1]['contents'],
        [genai_types.Part(text='Something')],
    )
    self._mock_genai_client.models.embed_content.assert_called_once()
    self.assertCounterEqual(
        backend._counters,
        Counter(embed=1),
    )
    self.assertCounterEqual(
        handler._cache_data.counters,
        Counter(add_new=1, get_miss=1),
    )


if __name__ == '__main__':
  absltest.main()
