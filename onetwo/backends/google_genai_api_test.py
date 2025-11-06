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
import io
import os
from typing import Any, Counter, Final, TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai
from google.genai import types as genai_types
import google.genai.errors as genai_errors
import httpx
from onetwo.backends import google_genai_api
from onetwo.builtins import llm as llm_lib
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import sampling
from PIL import Image
import pydantic


BaseModel = pydantic.BaseModel
# For escaping the `pytype: disable=wrong-keyword-args`
llm: Any = llm_lib


_Message: TypeAlias = content_lib.Message
_ChunkList: TypeAlias = content_lib.ChunkList
_Chunk: TypeAlias = content_lib.Chunk
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_THREADPOOL_SIZE: Final[int] = 4

_MOCK_COUNT_TOKENS_RETURN: Final[int] = 5


# For testing llm.generate_object with a Pydantic model.
class MyData(BaseModel):
  name: str
  value: int
  details: list[str]


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
    self._mock_genai_client.models.count_tokens.return_value = (
        genai_types.CountTokensResponse(total_tokens=_MOCK_COUNT_TOKENS_RETURN)
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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'text')
      self.assertEqual(res2, 'text')
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count, 2
      )
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_text=2),
      )
    with self.subTest('UpdatesCacheCounters'):
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
    with self.subTest('CallsApiWithCorrectPrompt'):
      self._mock_genai_client.models.generate_content.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.models.generate_content.call_args
      self.assertEqual(mock_kwargs['contents'][0].text, 'Some unsafe prompt')
    with self.subTest('UpdatesBackendCounters'):
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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'text')
      self.assertEqual(res2, 'text')
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count, 2
      )
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_text=2),
      )
    with self.subTest('UpdatesCacheCounters'):
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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'text')
      self.assertEqual(res2, 'text')
    with self.subTest('CallsApiOnceAndUpdatesCounters'):
      self._mock_genai_client.models.generate_content.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_text=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=1, get_miss=1, get_hit=1),
      )

  def test_generate_text_with_retry(self):
    self._mock_genai_client.models.generate_content.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.ReadError('test'),
        genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        parts=[genai_types.Part(text='text')]
                    )
                )
            ]
        ),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    res = executing.run(llm.generate_text(prompt='Something'))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'text')
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count,
          max_retries + 1,
      )
      self.assertCounterEqual(
          backend._counters, Counter(generate_text=max_retries + 1)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

  def test_generate_text_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried."""
    self._mock_genai_client.models.generate_content.side_effect = (
        genai_errors.ClientError(
            code=400, response_json={'error': {'message': 'test'}}
        )
    )
    backend = _get_and_register_backend(max_retries=2)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.assertRaises(ValueError):
      executing.run(llm.generate_text(prompt='Something'))

    with self.subTest('DoesNotRetry'):
      self._mock_genai_client.models.generate_content.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(generate_text=1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(handler._cache_data.counters, Counter(get_miss=1))

  def test_repeat_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    num_repeats = 5
    res = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=num_repeats,
            )
        )
    )
    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, ['text'] * num_repeats)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count,
          num_repeats,
      )
      self.assertCounterEqual(
          backend._counters, Counter(generate_text=num_repeats)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1, add_new_sample=4, get_miss=1, get_hit_miss_sample=4
          ),
      )

  def test_repeat_generate_text_cached(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    num_repeats_1 = 5
    res = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=num_repeats_1,
            )
        )
    )
    num_repeats_2 = 6
    res2 = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=num_repeats_2,
            )
        )
    )

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, ['text'] * num_repeats_1)
      self.assertEqual(res2, ['text'] * num_repeats_2)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count,
          num_repeats_2,
      )
      self.assertCounterEqual(
          backend._counters, Counter(generate_text=num_repeats_2)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              # The first generate_text from the first run.
              add_new=1,
              # 4 from the first run and 1 from the second run.
              add_new_sample=5,
              # The first generate_text from the first run.
              get_miss=1,
              # The second run hits the cache for the first 5 samples.
              get_hit=5,
              # 4 from 1st run + 1 from 2nd run.
              get_hit_miss_sample=5,
          ),
      )

  def test_batched_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    prompts = [
        'Something a',
        'Something b',
        'Something c',
        'Something d',
        'Something f',
    ]
    res = executing.run(
        executing.par_iter([llm.generate_text(prompt=p) for p in prompts])
    )

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, ['text'] * len(prompts))
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertLen(
          prompts,
          self._mock_genai_client.models.generate_content.call_count,
      )
      self.assertCounterEqual(
          backend._counters,
          Counter(
              generate_text=len(prompts),
          ),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=len(prompts),
              get_miss=len(prompts),
          ),
      )

  def test_identical_batched_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    num_calls = 5
    res = executing.run(
        executing.par_iter([llm.generate_text(prompt='Something')] * num_calls)
    )

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, ['text'] * num_calls)
    with self.subTest('CallsApiOnceAndUpdatesCounters'):
      self._mock_genai_client.models.generate_content.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(
              generate_text=1,
          ),
      )
    with self.subTest('UpdatesCacheCounters'):
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

    samples = 3
    results = executing.run(
        llm.generate_texts(prompt='prompt', samples=samples)
    )

    with self.subTest('ReturnsCorrectNumberOfSamples'):
      self.assertLen(results, samples)
      self.assertListEqual(list(results), ['text'] * samples)

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

  def test_generate_object_caching(self):

    cache_dir = self.create_tempdir()
    cache_filename = os.path.join(cache_dir.full_path, 'test_cache.json')

    # Instance 1: Populate cache
    cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    backend1 = google_genai_api.GoogleGenAIAPI(
        api_key='some_key',
        cache=cache,
    )
    backend1.register()

    mock_response = mock.create_autospec(
        genai_types.GenerateContentResponse, instance=True
    )
    mock_response.candidates = [mock.MagicMock()]
    test_object = MyData(name='Test', value=123, details=['a', 'b'])
    mock_response.parsed = test_object

    with self.subTest('PopulateCache'):
      with mock.patch.object(
          backend1, '_generate_content', return_value=mock_response
      ) as mock_generate_content:
        obj1 = executing.run(
            backend1.generate_object(prompt='some prompt', cls=MyData)
        )
        mock_generate_content.assert_called_once()
        self.assertEqual(obj1, test_object)
        self.assertIsInstance(obj1, MyData)
        self.assertEqual(backend1._counters['generate_object'], 1)

    with self.subTest('CacheHitSameBackend'):
      obj2 = executing.run(
          backend1.generate_object(prompt='some prompt', cls=MyData)
      )
      self.assertEqual(mock_generate_content.call_count, 1)
      self.assertEqual(obj2, test_object)
      self.assertIsInstance(obj2, MyData)
      self.assertEqual(backend1._counters['generate_object'], 1)

    cache.save()
    self.assertTrue(os.path.exists(cache_filename))

    # Instance 2: Load from cache
    cache2 = caching.SimpleFunctionCache(cache_filename=cache_filename)
    backend2 = google_genai_api.GoogleGenAIAPI(
        api_key='some_key',
        threadpool_size=_THREADPOOL_SIZE,
        cache=cache2,
    )
    backend2.register()
    cache2.load()

    with self.subTest('CacheHitNewBackend'):
      with mock.patch.object(
          backend2, '_generate_content', return_value=mock_response
      ) as mock_generate_content_new:
        # This call should hit the loaded cache
        obj3 = executing.run(
            backend2.generate_object(prompt='some prompt', cls=MyData)
        )
        mock_generate_content_new.assert_not_called()
        self.assertEqual(obj3, test_object)
        self.assertIsInstance(obj3, MyData)
        # generate_object on backend2 was never actually called beyond the cache
        self.assertEqual(backend2._counters['generate_object'], 0)

        self.assertEqual(
            cache2._cache_data.counters['get_hit'], 2
        )  # 1 hit in backend1, 1 in backend2

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
    with self.subTest('CallsApiWithCorrectMessage'):
      mock_chat.send_message.assert_called_once()
      _, mock_kwargs = mock_chat.send_message.call_args
      self.assertEqual(mock_kwargs['message'][0].text, 'Hello model')
    with self.subTest('UpdatesBackendCounters'):
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

  def test_chat_with_retry(self):
    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    mock_chat.send_message.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.TransportError('test'),
        genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        parts=[genai_types.Part(text='Hello')]
                    )
                )
            ]
        ),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    res = executing.run(llm.chat(messages=[msg_user]))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'Hello')
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(mock_chat.send_message.call_count, max_retries + 1)
      self.assertCounterEqual(backend._counters, Counter(chat=max_retries + 1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

  def test_chat_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried in chat."""
    mock_chat = mock.MagicMock()
    self._mock_genai_client.chats.create.return_value = mock_chat
    mock_chat.send_message.side_effect = genai_errors.ClientError(
        code=400, response_json={'error': {'message': 'test'}}
    )
    backend = _get_and_register_backend(max_retries=2)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    with self.assertRaises(ValueError):
      executing.run(llm.chat(messages=[msg_user]))

    with self.subTest('DoesNotRetry'):
      mock_chat.send_message.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(chat=1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(handler._cache_data.counters, Counter(get_miss=1))

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

  def test_convert_chunk_list_with_pil_image(self):
    """Tests conversion of ChunkList containing a PIL Image."""
    pil_image = Image.new('RGB', (1, 1), color='black')
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()

    prompt = _ChunkList(chunks=[_Chunk(content=pil_image)])
    expected_part = genai_types.Part(
        inline_data=genai_types.Blob(mime_type='image/png', data=image_bytes)
    )

    result = google_genai_api._convert_chunk_list_to_part_list(prompt)
    self.assertLen(result, 1)
    self.assertEqual(result[0], expected_part)

  def test_count_tokens_string(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    content = 'Something'
    res = executing.run(llm.count_tokens(content=content))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_COUNT_TOKENS_RETURN)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self._mock_genai_client.models.count_tokens.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(count_tokens=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=1, get_miss=1),
      )
    with self.subTest('CallsApiWithCorrectContent'):
      self.assertEqual(
          self._mock_genai_client.models.count_tokens.call_args[1]['contents'],
          [genai_types.Part(text=content)],
      )

  def test_count_tokens_chunk_list(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    content = content_lib.ChunkList(
        chunks=[content_lib.Chunk(content='Something')]
    )
    res = executing.run(llm.count_tokens(content=content))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_COUNT_TOKENS_RETURN)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self._mock_genai_client.models.count_tokens.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(count_tokens=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=1, get_miss=1),
      )
    with self.subTest('CallsApiWithCorrectContent'):
      self.assertEqual(
          self._mock_genai_client.models.count_tokens.call_args[1]['contents'],
          [genai_types.Part(text='Something')],
      )

  def test_count_tokens_with_retry(self):
    self._mock_genai_client.models.count_tokens.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.TransportError('test'),
        genai_types.CountTokensResponse(total_tokens=_MOCK_COUNT_TOKENS_RETURN),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    res = executing.run(llm.count_tokens(content='Something'))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_COUNT_TOKENS_RETURN)
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(
          self._mock_genai_client.models.count_tokens.call_count,
          max_retries + 1,
      )
      self.assertCounterEqual(
          backend._counters, Counter(count_tokens=max_retries + 1)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

  def test_count_tokens_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried in count_tokens."""
    self._mock_genai_client.models.count_tokens.side_effect = (
        genai_errors.ClientError(
            code=400, response_json={'error': {'message': 'test'}}
        )
    )
    backend = _get_and_register_backend(max_retries=2)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.assertRaises(ValueError):
      executing.run(llm.count_tokens(content='Something'))

    with self.subTest('DoesNotRetry'):
      self._mock_genai_client.models.count_tokens.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(count_tokens=1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(handler._cache_data.counters, Counter(get_miss=1))

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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_tokens)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.compute_tokens.call_args[1][
              'contents'
          ],
          [genai_types.Part(text='Something')],
      )
      self._mock_genai_client.models.compute_tokens.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(tokenize=1),
      )
    with self.subTest('UpdatesCacheCounters'):
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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_tokens)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.compute_tokens.call_args[1][
              'contents'
          ],
          [genai_types.Part(text='Something')],
      )
      self._mock_genai_client.models.compute_tokens.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(tokenize=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=1, get_miss=1),
      )

  def test_tokenize_with_retry(self):
    mock_tokens = [1, 2, 3, 4, 5]
    self._mock_genai_client.models.compute_tokens.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.TransportError('test'),
        genai_types.ComputeTokensResponse(
            tokens_info=[genai_types.TokensInfo(token_ids=mock_tokens)]
        ),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    res = executing.run(llm.tokenize(content='Something'))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_tokens)
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(
          self._mock_genai_client.models.compute_tokens.call_count,
          max_retries + 1,
      )
      self.assertCounterEqual(
          backend._counters, Counter(tokenize=max_retries + 1)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

  def test_tokenize_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried in tokenize."""
    self._mock_genai_client.models.compute_tokens.side_effect = (
        genai_errors.ClientError(
            code=400, response_json={'error': {'message': 'test'}}
        )
    )
    backend = _get_and_register_backend(max_retries=2)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.assertRaises(ValueError):
      executing.run(llm.tokenize(content='Something'))

    with self.subTest('DoesNotRetry'):
      self._mock_genai_client.models.compute_tokens.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(tokenize=1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(handler._cache_data.counters, Counter(get_miss=1))

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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_embedding)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.embed_content.call_args[1]['contents'],
          [genai_types.Part(text='Something')],
      )
      self._mock_genai_client.models.embed_content.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(embed=1),
      )
    with self.subTest('UpdatesCacheCounters'):
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

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_embedding)
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.embed_content.call_args[1]['contents'],
          [genai_types.Part(text='Something')],
      )
      self._mock_genai_client.models.embed_content.assert_called_once()
      self.assertCounterEqual(
          backend._counters,
          Counter(embed=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=1, get_miss=1),
      )

  def test_embed_with_retry(self):
    mock_embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
    self._mock_genai_client.models.embed_content.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.TransportError('test'),
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=mock_embedding)]
        ),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    res = executing.run(llm.embed(content='Something'))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, mock_embedding)
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(
          self._mock_genai_client.models.embed_content.call_count,
          max_retries + 1,
      )
      self.assertCounterEqual(backend._counters, Counter(embed=max_retries + 1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

  def test_embed_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried in embed."""
    self._mock_genai_client.models.embed_content.side_effect = (
        genai_errors.ClientError(
            code=400, response_json={'error': {'message': 'test'}}
        )
    )
    backend = _get_and_register_backend(max_retries=2)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    with self.assertRaises(ValueError):
      executing.run(llm.embed(content='Something'))

    with self.subTest('DoesNotRetry'):
      self._mock_genai_client.models.embed_content.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(embed=1))
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(handler._cache_data.counters, Counter(get_miss=1))

  def test_generate_content_simple(self):
    _ = _get_and_register_backend()
    results = executing.run(llm.generate_content(prompt='prompt'))
    self.assertEqual(results, _ChunkList([_Chunk('text')]))

  def test_generate_content_args(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='text2')]
                )
            )
        ]
    )
    res = executing.run(
        llm.generate_content(
            prompt='prompt1',
            temperature=0.7,
            max_tokens=3,
            stop=['halt', 'zoll'],
            top_k=4,
            top_p=0.9,
        )
    )
    self.assertEqual(str(res), 'text2')
    self.assertEqual(
        gc_mock.call_args.kwargs['contents'],
        [
            genai_types.Content(
                parts=[genai_types.Part(text='prompt1')], role='user'
            )
        ],
    )
    self.assertEqual(
        gc_mock.call_args.kwargs['config'],
        genai_types.GenerateContentConfig(
            candidate_count=1,
            max_output_tokens=3,
            stop_sequences=['halt', 'zoll'],
            temperature=0.7,
            top_k=4.0,
            top_p=0.9,
        ),
    )

  def test_generate_content_details(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text='I am', thought=True),
                        genai_types.Part(text=' thinking', thought=True),
                        genai_types.Part(text=' bananas'),
                        genai_types.Part(text=' but also', thought=True),
                        genai_types.Part(text=' apples'),
                    ],
                )
            )
        ]
    )
    res = executing.run(
        llm.generate_content(prompt='prompt1', include_details=True)
    )
    self.assertEqual(str(res[0]), ' bananas apples')
    self.assertEqual(str(res[1]['thoughts']), 'I am thinking but also')
    # TODO: do we want to also cleanup the candidate?

  def test_generate_content_prompt(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content

    image = Image.new(mode='RGB', size=(1, 1))
    image.load()[0, 0] = (255, 255, 255)

    def run0(prompt: llm_lib.Prompt) -> Sequence[genai_types.Content]:
      _ = executing.run(llm.generate_content(prompt=prompt))
      out = gc_mock.call_args.kwargs['contents']
      # Simplify the image bytes
      for candidate in out:
        for part in candidate.parts:
          if part.inline_data and part.inline_data.mime_type == 'image/png':
            part.inline_data.data = b'<PNG>'
      return out

    part_text = genai_types.Part(text='text')
    part_sql = genai_types.Part(
        inline_data=genai_types.Blob(
            data=b'SELECT bytes', mime_type='application/sql'
        )
    )
    part_img = genai_types.Part(
        inline_data=genai_types.Blob(data=b'<PNG>', mime_type='image/png')
    )

    # Case 1: str
    c0 = [genai_types.Content(parts=[part_text], role='user')]
    self.assertEqual(run0('text'), c0)
    self.assertEqual(run0(_ChunkList(['text'])), c0)
    # Case 2: Sequence[_Chunk]
    chunks = [
        _Chunk('text'),  # user if role unset
        _Chunk(b'SELECT bytes', 'application/sql', role='model'),
        _Chunk(image),  # user if role unset
    ]
    contents = [
        genai_types.Content(parts=[part_text], role='user'),
        genai_types.Content(parts=[part_sql], role='model'),
        genai_types.Content(parts=[part_img], role='user'),
    ]
    self.assertEqual(run0(chunks), contents)
    # Case 3: _ChunkList
    self.assertEqual(run0(_ChunkList(chunks=chunks)), contents)
    # Case 4: Sequence[_Message], role is forced to that of enclosing message
    c4 = run0([_Message(content=_ChunkList(chunks), role='user')])
    self.assertEqual(
        c4,
        [
            genai_types.Content(
                parts=[part_text, part_sql, part_img], role='user'
            )
        ],
    )

  def test_generate_content_turns_and_instructions(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    chunks = [
        _Chunk('system', role='system'),
        _Chunk('instructions', role='system'),
        _Chunk('hello', role='user'),
        _Chunk('world', role='model'),
        _Chunk('hola', role='user'),
    ]
    _ = executing.run(llm.generate_content(prompt=chunks))
    out = gc_mock.call_args.kwargs
    self.assertEqual(
        out['config'].system_instruction,
        genai_types.Content(
            parts=[
                genai_types.Part(text='system'),
                genai_types.Part(text='instructions'),
            ],
            role='system',
        ),
    )
    self.assertEqual(
        out['contents'],
        [
            genai_types.Content(
                parts=[genai_types.Part(text='hello')], role='user'
            ),
            genai_types.Content(
                parts=[genai_types.Part(text='world')], role='model'
            ),
            genai_types.Content(
                parts=[genai_types.Part(text='hola')], role='user'
            ),
        ],
    )

  def test_generate_content_heal_spaces(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    # Text is "broken spaces"; " " should come from the model, not the prompt.
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text=' spaces')]
                )
            )
        ]
    )
    res = executing.run(
        llm.generate_content(
            prompt='broken ',
            healing_option=llm_lib.TokenHealingOption.SPACE_HEALING,
        )
    )
    self.assertEqual(  # Model sees the fixed prompt
        gc_mock.call_args.kwargs['contents'][0].parts[0].text, 'broken'
    )
    self.assertEqual(str(res), 'spaces')  # Return has no double space

  def test_generate_content_samples(self):
    backend = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content

    def generate_contents(samples: int, **kwargs):
      """Demonstrates generating multiple candidates e.g. generate_texts()."""
      return executing.run(
          backend._generate_contents(samples=samples, **kwargs)
      )

    def candidate(parts: list[str]):
      ps = [genai_types.Part(text=p) for p in parts]
      return genai_types.Candidate(content=genai_types.Content(parts=ps))

    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            candidate(['c1', 'snap']),
            candidate(['c2', 'fu']),
            candidate(['c3', 'bar']),
        ]
    )
    res = generate_contents(prompt='many', samples=3)
    self.assertEqual(gc_mock.call_args.kwargs['config'].candidate_count, 3)
    self.assertEqual(
        res,
        [
            _ChunkList(chunks=['c1', 'snap']),
            _ChunkList(chunks=['c2', 'fu']),
            _ChunkList(chunks=['c3', 'bar']),
        ],
    )

  def test_generate_content_decode_obj(self):
    def gen_obj(decoding_constraint: type[Any], **kwargs):
      """Demonstrates how generate_object() can be reimplemented."""
      kwargs['response_mime_type'] = 'application/json'
      kwargs['response_schema'] = decoding_constraint
      result = executing.run(llm.generate_content(**kwargs))
      adapter = pydantic.TypeAdapter(decoding_constraint)
      return adapter.validate_json(str(result))

    _ = _get_and_register_backend()
    expected = [
        MyData(name='Chips', value=1, details=['a', 'b']),
        MyData(name='Cookie', value=2, details=['c', 'd']),
    ]
    json = pydantic.TypeAdapter(list[MyData]).dump_json(expected)
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(parts=[genai_types.Part(text=json)])
            )
        ]
    )
    self.assertEqual(
        gen_obj(prompt='prompt', decoding_constraint=list[MyData]), expected
    )

  def test_generate_content_decode_str(self):
    def with_regex(decoding_constraint: str, **kwargs):
      """Demonstrates how a regex constraint can be added."""
      kwargs['response_mime_type'] = 'application/json'
      kwargs['response_schema'] = {
          'type': 'STRING',
          'pattern': decoding_constraint,
      }
      content = executing.run(llm.generate_content(**kwargs))
      text = pydantic.TypeAdapter(str).validate_json(str(content))
      return _ChunkList(chunks=[text])

    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='"unicorn"')]
                )
            )
        ]
    )
    result = with_regex(prompt='unicorn', decoding_constraint='(poney|unicorn)')
    self.assertEqual(str(result), 'unicorn')


if __name__ == '__main__':
  absltest.main()
