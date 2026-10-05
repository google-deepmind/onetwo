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

"""Tests for OpenAIAPI engine."""

import base64
import collections
import math
import os
from typing import Any, Counter, Final, TypeAlias, cast
from unittest import mock

# We have to mock openai to make this test hermetic.
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.agents import agent as agent_lib
from onetwo.backends import openai_api
from onetwo.backends import openai_mock
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
import PIL.Image
import pydantic

import openai

_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole

_ConversationRole: TypeAlias = fc_types.ConversationRole
_ConversationTurn: TypeAlias = fc_types.ConversationTurn
_FunctionCall: TypeAlias = fc_types.FunctionCall
_FunctionResponse: TypeAlias = fc_types.FunctionResponse
_ToolSpec: TypeAlias = fc_types.ToolSpec


_BATCH_SIZE: Final[int] = 4


# Define a Pydantic BaseModel for testing structured output caching.
class CityInfoTestData(pydantic.BaseModel):
  city: str
  population: int
  landmarks: list[str]


def _get_and_register_backend(
    model_name: str | None = None,
) -> openai_api.OpenAIAPI:
  """Get an instance of OpenAIAPI and register its methods."""
  if model_name:
    backend = openai_api.OpenAIAPI(
        api_key='some_key', batch_size=_BATCH_SIZE, model_name=model_name
    )
  else:
    backend = openai_api.OpenAIAPI(api_key='some_key', batch_size=_BATCH_SIZE)
  backend.register()
  return backend


class OpenAIAPITest(parameterized.TestCase, core_test_utils.CounterAssertions):

  def setUp(self):
    super().setUp()
    # The `self.create_tempdir` method uses command line flag and such flags
    # are not marked as parsed by default when running with pytest. Marking as
    # parsed directly here to make the pytest run pass.
    flags.FLAGS.mark_as_parsed()
    # We use a mock to avoid making actual calls to the API.
    self.enter_context(
        mock.patch.object(
            openai,
            'OpenAI',
            autospec=True,
            side_effect=openai_mock.OpenAI,
        )
    )

  def test_instruct_should_use_api_formatter(self):
    _ = _get_and_register_backend()
    prompt = 'Task: Answer the question.\nQuestion:Why?\nAnswer:'
    try:
      _ = executing.run(llm.instruct(prompt=prompt))
    except ValueError as e:
      # If llm.instruct inadvertently uses the default formatter rather than the
      # API formatter, the above prompt will raise an error due to the phrases
      # 'Task:' and 'Answer:' being interpreted as control tokens.
      self.fail(f'Instruct raised an error. Maybe due to wrong formatter?\n{e}')

  def test_chat(self):
    """Verifies that chat works and  calls the api directly."""
    backend = _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        llm.chat(
            messages=[_Message(content=prompt_text, role=_PredefinedRole.USER)],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_responses_create': 1,
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
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'generate_text': 1,
        'generate_text_batches': 1,
        '_responses_create': 1,
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
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    with self.subTest('sends_correct_number_of_api_calls_2'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1
    with self.subTest('cache_behaves_as_expected_2'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # Same query but different parameters, run generate request again.
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            max_tokens=512,
            temperature=0.0,
        )
    )
    with self.subTest('returns_correct_result_3'):
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        '_responses_create': 2,
        'generate_text': 2,
        'generate_text_batches': 2,
    })
    with self.subTest('sends_correct_number_of_api_calls_3'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_miss'] += 1  # Could not find generate.
    expected_cache_counters['add_new'] += 1  # Cached generate.
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
      self.assertEqual(res[0], openai_mock._DEFAULT_REPLY)
    with self.subTest('returns_correct_score'):
      self.assertEqual(res[1]['score'], -0.5)

  def test_generate_text_prepends_prefix(self):
    """Verifies that generate_text prepends the completion prefix."""
    backend = _get_and_register_backend()
    prompt_text = 'Something'

    with mock.patch.object(
        backend,
        '_responses_create',
        autospec=True,
        return_value=mock.MagicMock(output_text='ignored'),
    ) as mock_create:
      _ = executing.run(
          llm.generate_text(
              prompt=prompt_text,
          )
      )

      self.assertTrue(mock_create.called)
      kwargs = mock_create.call_args.kwargs
      messages = kwargs.get('messages')
      self.assertLen(messages, 2)
      self.assertEqual(messages[0].role, _PredefinedRole.SYSTEM)  # pyrefly: ignore[unsupported-operation]
      self.assertEqual(
          messages[0].content, openai_api._COMPLETION_SYSTEM_PROMPT_PREFIX  # pyrefly: ignore[unsupported-operation]
      )
      self.assertEqual(messages[1].role, _PredefinedRole.USER)  # pyrefly: ignore[unsupported-operation]
      self.assertEqual(messages[1].content, prompt_text)  # pyrefly: ignore[unsupported-operation]

  def test_generate_object(self):
    """Verifies that generate_object works and calls the mock parse API."""
    backend = _get_and_register_backend()

    class MyTestSchema(pydantic.BaseModel):
      # Must match the mock data in openai_mock.ResponsesNamespace.parse.
      name: str = 'default'
      population: int = 0

    prompt_text = 'Create an object representing a city.'

    res = executing.run(
        llm.generate_object(
            prompt=prompt_text,
            cls=MyTestSchema,
        )
    )

    with self.subTest('returns_correct_object_type'):
      self.assertIsInstance(res, MyTestSchema)
    with self.subTest('object_has_mock_data'):
      self.assertEqual(res.name, 'Mock string')
      self.assertEqual(res.population, 42)

    expected_backend_counters = collections.Counter({
        'generate_object': 1,
        '_responses_parse': 1,
        '_generate_object_internal_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_generate_object_with_invalid_cls_raises_error(self):
    """Test generate_object raise ValueError if cls not a Pydantic BaseModel."""
    _ = _get_and_register_backend()

    with self.subTest('cls_is_not_pydantic_raises_error'):

      class NotPydantic:
        pass

      with self.assertRaisesRegex(
          ValueError,
          r'The `cls` argument must be a subclass of `pydantic.BaseModel`\.',
      ):
        executing.run(
            llm.generate_object(
                prompt='test prompt',
                cls=NotPydantic,
            )
        )

    with self.subTest('cls_is_pydantic_instance_raises_error'):

      class MyTestSchema(pydantic.BaseModel):
        test: str

      with self.assertRaisesRegex(
          ValueError,
          r'The `cls` argument must be a subclass of `pydantic.BaseModel`\.',
      ):
        executing.run(
            llm.generate_object(
                prompt='yet another prompt',
                cls=MyTestSchema(test='instance'),
            )
        )

  def test_generate_object_caching(self):
    cache_dir = self.create_tempdir()
    cache_filename = os.path.join(cache_dir.full_path, 'openai_cache.json')
    test_object = CityInfoTestData(
        city='Paris', population=2141000, landmarks=['Eiffel Tower', 'Louvre']
    )

    # Mock for the Responses API: returns a Response-like object with
    # output_text containing the JSON of the test object.
    mock_response = mock.MagicMock()
    mock_response.output_text = test_object.model_dump_json()

    backend1 = openai_api.OpenAIAPI(
        model_name='gpt-5.4-mini',
        api_key='test_key',
        cache=caching.SimpleFunctionCache(cache_filename=cache_filename),
    )
    backend1.register()

    with mock.patch.object(
        backend1,
        '_responses_parse',
        autospec=True,
        return_value=mock_response,
    ) as mock_responses_parse:

      with self.subTest('PopulateCache'):
        obj1 = executing.run(
            backend1.generate_object(
                prompt='Describe a city', cls=CityInfoTestData
            )
        )
        mock_responses_parse.assert_called_once()
        self.assertEqual(obj1, test_object)
        self.assertIsInstance(obj1, CityInfoTestData)
        self.assertEqual(backend1._counters['generate_object'], 1)

      with self.subTest('CacheHitSameBackend'):
        obj2 = executing.run(
            backend1.generate_object(
                prompt='Describe a city', cls=CityInfoTestData
            )
        )
        self.assertEqual(mock_responses_parse.call_count, 1)
        self.assertEqual(obj2, test_object)
        self.assertIsInstance(obj2, CityInfoTestData)
        self.assertEqual(backend1._counters['generate_object'], 1)

      assert isinstance(backend1.cache, caching.SimpleFunctionCache)
      backend1.cache.save()
      self.assertTrue(os.path.exists(cache_filename))

      backend2 = openai_api.OpenAIAPI(
          model_name='gpt-5.4-mini',
          api_key='test_key',
          cache=caching.SimpleFunctionCache(cache_filename=cache_filename),
      )
      backend2.register()
      assert isinstance(backend2.cache, caching.SimpleFunctionCache)
      backend2.cache.load()
      with self.subTest('CacheHitNewBackend'):
        with mock.patch.object(
            backend2,
            '_responses_parse',
            autospec=True,
            return_value=mock_response,
        ) as mock_responses_parse_new:
          obj3 = executing.run(
              backend2.generate_object(
                  prompt='Describe a city', cls=CityInfoTestData
              )
          )
          mock_responses_parse_new.assert_not_called()
          self.assertEqual(obj3, test_object)
          self.assertIsInstance(obj3, CityInfoTestData)
          self.assertEqual(backend2._counters['generate_object'], 0)


class ConvertChunkToContentPartTest(parameterized.TestCase):
  """Tests for _convert_chunk_to_content_part."""

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_string',
          chunk=content_lib.Chunk('Hello, world!'),
          expected={'type': 'input_text', 'text': 'Hello, world!'},
      ),
      dict(
          testcase_name='empty_string',
          chunk=content_lib.Chunk(''),
          expected={'type': 'input_text', 'text': ''},
      ),
      dict(
          testcase_name='string_with_special_characters',
          chunk=content_lib.Chunk('Line 1\nLine 2\t"quoted"'),
          expected={'type': 'input_text', 'text': 'Line 1\nLine 2\t"quoted"'},
      ),
      dict(
          testcase_name='control_token_string',
          chunk=content_lib.Chunk('<start_of_turn>', content_type='ctrl'),
          expected={'type': 'input_text', 'text': '<start_of_turn>'},
      ),
  )
  def test_string_chunks(self, chunk, expected):
    result = openai_api._convert_chunk_to_openai_part(chunk)
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
    result = openai_api._convert_chunk_to_openai_part(chunk)
    self.assertEqual(result['type'], 'input_image')
    expected_b64 = base64.b64encode(data).decode('ascii')
    expected_url = f'data:{expected_mime};base64,{expected_b64}'
    self.assertEqual(result['image_url'], expected_url)

  def test_pil_image_chunk(self):
    image = PIL.Image.new('RGB', (2, 2), color='red')
    chunk = content_lib.Chunk(image)
    result = openai_api._convert_chunk_to_openai_part(chunk)
    self.assertEqual(result['type'], 'input_image')
    self.assertTrue(
        result['image_url'].startswith('data:image/jpeg;base64,'),
        'Expected data URL to start with "data:image/jpeg;base64,", '
        f'got: {result["image_url"][:40]}...',
    )
    # Verify the base64 payload is valid and decodes to non-empty bytes.
    b64_payload = result['image_url'].split(',', 1)[1]
    decoded = base64.b64decode(b64_payload)
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
        ValueError, 'Unsupported binary content_type for OpenAI API'
    ):
      openai_api._convert_chunk_to_openai_part(chunk)


class EmbedTest(parameterized.TestCase, core_test_utils.CounterAssertions):

  def setUp(self):
    super().setUp()
    # The `self.create_tempdir` method uses command line flag and such flags
    # are not marked as parsed by default when running with pytest. Marking as
    # parsed directly here to make the pytest run pass.
    flags.FLAGS.mark_as_parsed()
    # We use a mock to avoid making actual calls to the API.
    self.enter_context(
        mock.patch.object(
            openai,
            'OpenAI',
            autospec=True,
            side_effect=openai_mock.OpenAI,
        )
    )

  def test_embed_basic(self):
    """Verifies that embed returns a list of floats."""
    backend = _get_and_register_backend()
    result = executing.run(
        llm.embed(content='Hello, world!')
    )
    with self.subTest('returns_list_of_floats'):
      self.assertIsInstance(result, (list, tuple))
      self.assertNotEmpty(result)
      for v in result:
        self.assertIsInstance(v, float)
    expected_backend_counters = collections.Counter({
        'embed': 1,
        '_cached_embed_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_embed_with_output_dimensionality(self):
    """Verifies that output_dimensionality controls the embedding size."""
    _ = _get_and_register_backend()
    dim = 5
    result = executing.run(
        llm.embed(content='Hello', output_dimensionality=dim)
    )
    self.assertLen(result, dim)

  def test_embed_normalizes_by_default(self):
    """Verifies that llm.embed returns a unit-norm vector by default."""
    _get_and_register_backend()
    result = executing.run(llm.embed(content='Something'))
    norm = math.sqrt(sum(v * v for v in result))
    self.assertAlmostEqual(norm, 1.0, places=6)

  def test_embed_without_normalize(self):
    """Verifies that with normalize=False, the raw backend output is returned."""
    _get_and_register_backend()
    result = executing.run(llm.embed(content='Something', normalize=False))
    norm = math.sqrt(sum(v * v for v in result))
    # The mock returns non-unit-norm vectors by design.
    self.assertNotAlmostEqual(norm, 1.0, places=2)

  def test_embed_with_chunklist(self):
    """Verifies that embed works with a ChunkList input."""
    _get_and_register_backend()
    chunks = content_lib.ChunkList([
        content_lib.Chunk('Hello, '),
        content_lib.Chunk('world!'),
    ])
    result = executing.run(
        llm.embed(content=chunks)
    )
    self.assertIsInstance(result, (list, tuple))
    self.assertNotEmpty(result)

  def test_embed_chunklist_with_non_text_raises_error(self):
    """Verifies that embed raises ValueError for non-text chunks."""
    _get_and_register_backend()
    chunks = content_lib.ChunkList([
        content_lib.Chunk(b'binary_data', content_type='image/png'),
    ])
    with self.assertRaisesRegex(
        ValueError, 'OpenAIAPI.embed only supports text content'
    ):
      executing.run(
          llm.embed(content=chunks)
      )

  def test_embed_caching(self):
    """Verifies that embed results are cached."""
    backend = _get_and_register_backend()
    content_text = 'Cache me!'
    # First call.
    result1 = executing.run(llm.embed(content=content_text))
    with self.subTest('first_call_hits_api'):
      self.assertEqual(backend._counters['embed'], 1)
    # Second call with same content should be cached.
    result2 = executing.run(llm.embed(content=content_text))
    with self.subTest('second_call_is_cached'):
      self.assertEqual(backend._counters['embed'], 1)
    with self.subTest('results_are_equal'):
      self.assertEqual(result1, result2)


class CountTokensTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):

  def setUp(self):
    super().setUp()
    # The `self.create_tempdir` method uses command line flag and such flags
    # are not marked as parsed by default when running with pytest. Marking as
    # parsed directly here to make the pytest run pass.
    flags.FLAGS.mark_as_parsed()
    # We use a mock to avoid making actual calls to the API.
    self.enter_context(
        mock.patch.object(
            openai,
            'OpenAI',
            autospec=True,
            side_effect=openai_mock.OpenAI,
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
        llm.count_tokens(content=content)
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
    """Verifies that messages have more tokens than plain text (overhead)."""
    _get_and_register_backend()
    text = 'Hello world'
    messages = [
        _Message(content=text, role=_PredefinedRole.USER),
    ]
    count_text = executing.run(
        llm.count_tokens(content=text)
    )
    count_messages = executing.run(
        llm.count_tokens(content=messages)
    )
    # Messages should have overhead (per-message + reply primer tokens).
    self.assertGreater(count_messages, count_text)

  def test_count_tokens_caching(self):
    """Verifies that count_tokens results are cached."""
    backend = _get_and_register_backend()
    content_text = 'Cache me!'
    # First call.
    result1 = executing.run(
        llm.count_tokens(content=content_text)
    )
    with self.subTest('first_call_hits_api'):
      self.assertEqual(backend._counters['count_tokens'], 1)
    # Second call with same content should be cached.
    result2 = executing.run(
        llm.count_tokens(content=content_text)
    )
    with self.subTest('second_call_is_cached'):
      self.assertEqual(backend._counters['count_tokens'], 1)
    with self.subTest('results_are_equal'):
      self.assertEqual(result1, result2)


def _get_weather(location: str) -> dict[str, Any]:
  """Gets the weather for a specified location.

  Args:
    location: City name.

  Returns:
    A dictionary containing weather information.
  """
  return {'location': location, 'temp': 15, 'unit': 'C'}


def _always_fails(location: str) -> dict[str, Any]:
  """Looks up the weather, badly.

  Args:
    location: City name.

  Returns:
    Never returns.

  Raises:
    ValueError: Always.
  """
  del location
  raise ValueError('The weather service is down.')


def _message(text: str) -> openai_mock.ResponseOutputMessage:
  """Returns a message item carrying `text`."""
  return openai_mock.ResponseOutputMessage(
      content=[openai_mock.OutputTextContent(text=text)]
  )


def _call(
    name: str, arguments: str, call_id: str = 'call_1'
) -> openai_mock.ResponseFunctionToolCall:
  """Returns a function_call item asking for `name`."""
  return openai_mock.ResponseFunctionToolCall(
      name=name, arguments=arguments, call_id=call_id
  )


def _reasoning(
    summary: str = 'Thinking.',
    item_id: str = 'rs_1',
    encrypted_content: str = 'opaque',
) -> openai_mock.ResponseReasoningItem:
  """Returns a reasoning item."""
  return openai_mock.ResponseReasoningItem(
      summary=[openai_mock.SummaryText(summary)],
      item_id=item_id,
      encrypted_content=encrypted_content,
  )


class _ScriptedResponses:
  """Replies with a prepared sequence, recording what it was asked."""

  def __init__(self, *replies: openai_mock.Response):
    self._replies = list(replies)
    self.requests: list[dict[str, Any]] = []

  def create(self, **kwargs) -> openai_mock.Response:
    self.requests.append(kwargs)
    if not self._replies:
      raise AssertionError('The model was called more often than scripted.')
    return self._replies.pop(0)


class _ScriptedClient:
  """An OpenAI client whose `responses` are scripted."""

  def __init__(self, responses: _ScriptedResponses):
    self.responses = responses


class GenerateWithToolsTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):
  """Tests the single model call that `Agent` is built on."""

  def setUp(self):
    super().setUp()
    llm.reset_defaults()
    # Externally `openai` is the real library, so without this the tests
    # below that do not script the client would call the API for real.
    flags.FLAGS.mark_as_parsed()
    self.enter_context(
        mock.patch.object(
            openai,
            'OpenAI',
            autospec=True,
            side_effect=openai_mock.OpenAI,
        )
    )

  def _script(self, *replies: openai_mock.Response) -> openai_api.OpenAIAPI:
    """Registers a backend whose client replies with `replies`, in order."""
    backend = _get_and_register_backend()
    self._responses = _ScriptedResponses(*replies)
    # A test double standing in for the client, which `_client` is not
    # annotated to accept.
    backend._client = _ScriptedClient(self._responses)  # pyrefly: ignore[bad-assignment]
    return backend

  def _last_request(self) -> dict[str, Any]:
    return self._responses.requests[-1]

  def _ask(self, *messages, **kwargs):
    return executing.run(
        llm.generate_with_tools(messages=list(messages), **kwargs)
    )

  # --- Tool declarations ----------------------------------------------------

  def test_tool_declaration_is_flat(self):
    """The Responses API states name and parameters at the top level."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    schema = {
        'type': 'object',
        'properties': {'location': {'type': 'string'}},
        'required': ['location'],
    }
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[
            _ToolSpec(
                name='get_weather',
                description='Gets the weather.',
                parameters=schema,
            )
        ],
    )
    self.assertEqual(
        self._last_request()['tools'],
        [{
            'type': 'function',
            'name': 'get_weather',
            'description': 'Gets the weather.',
            'parameters': schema,
        }],
    )

  def test_tool_without_parameters_declares_empty_object(self):
    """A tool that takes nothing still needs a schema."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Now?'),
        tools=[_ToolSpec(name='now')],
    )
    self.assertEqual(
        self._last_request()['tools'][0]['parameters'],
        {'type': 'object', 'properties': {}},
    )

  def test_no_tools_declares_nothing(self):
    """An empty sequence is not the same as an empty list of tools."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Hello'),
        tools=[],
    )
    self.assertNotIn('tools', self._last_request())

  def test_builtin_tools_on_the_instance_are_kept(self):
    """A tool OpenAI runs itself stays available to a function-calling run."""
    backend = openai_api.OpenAIAPI(
        api_key='some_key',
        batch_size=_BATCH_SIZE,
        tools=[{'type': 'web_search'}],
    )
    backend.register()
    responses = _ScriptedResponses(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    # A test double standing in for the client, which `_client` is not
    # annotated to accept.
    backend._client = _ScriptedClient(responses)  # pyrefly: ignore[bad-assignment]
    executing.run(
        llm.generate_with_tools(
            messages=[
                _ConversationTurn(role=_ConversationRole.USER, content='News?')
            ],
            tools=[_ToolSpec(name='now')],
        )
    )
    self.assertEqual(
        [tool['type'] for tool in responses.requests[-1]['tools']],
        ['web_search', 'function'],
    )

  # --- Conversation to input items ------------------------------------------

  def test_user_turn_becomes_a_message(self):
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    self.assertEqual(
        self._last_request()['input'],
        [{'role': 'user', 'content': 'Weather?'}],
    )

  def test_model_call_becomes_a_function_call_item(self):
    """A call is an item beside the messages, not a block inside one."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        _ConversationTurn(
            role=_ConversationRole.MODEL,
            calls=[
                _FunctionCall(
                    function_name='get_weather',
                    id='call_abc',
                    kwargs={'location': 'London'},
                )
            ],
        ),
        tools=[],
    )
    self.assertEqual(
        self._last_request()['input'][1],
        {
            'type': 'function_call',
            'call_id': 'call_abc',
            'name': 'get_weather',
            # The API takes the arguments as a JSON string, not an object.
            'arguments': '{"location": "London"}',
        },
    )

  def test_tool_turn_becomes_function_call_output_items(self):
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        _ConversationTurn(
            role=_ConversationRole.TOOL,
            responses=[
                _FunctionResponse(
                    function_name='get_weather',
                    id='call_abc',
                    response={'temp': 15},
                )
            ],
        ),
        tools=[],
    )
    self.assertEqual(
        self._last_request()['input'][1],
        {
            'type': 'function_call_output',
            'call_id': 'call_abc',
            'output': '{"temp": 15}',
        },
    )

  def test_failed_tool_is_reported_to_the_model(self):
    """The API has no error flag, so the error travels as the content."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        _ConversationTurn(
            role=_ConversationRole.TOOL,
            responses=[
                _FunctionResponse(
                    function_name='get_weather',
                    id='call_abc',
                    error='It broke.',
                )
            ],
        ),
        tools=[],
    )
    self.assertEqual(
        self._last_request()['input'][1]['output'], 'Error: It broke.'
    )

  def test_call_id_falls_back_to_the_tool_name(self):
    """A conversation written by hand has no ids, but both halves must match."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(
            role=_ConversationRole.MODEL,
            calls=[_FunctionCall(function_name='get_weather', kwargs={})],
        ),
        _ConversationTurn(
            role=_ConversationRole.TOOL,
            responses=[
                _FunctionResponse(function_name='get_weather', response='15C')
            ],
        ),
        tools=[],
    )
    items = self._last_request()['input']
    self.assertEqual(items[0]['call_id'], 'get_weather')
    self.assertEqual(items[1]['call_id'], 'get_weather')

  # --- Response to conversation turn ----------------------------------------

  def test_function_call_becomes_a_call(self):
    self._script(
        openai_mock.Response(
            output=[_call('get_weather', '{"location": "London"}')],
            output_text='',
        )
    )
    turn = self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    self.assertEqual(turn.role, _ConversationRole.MODEL)
    self.assertLen(turn.calls, 1)
    self.assertEqual(turn.calls[0].function_name, 'get_weather')
    self.assertEqual(turn.calls[0].kwargs, {'location': 'London'})
    self.assertEqual(turn.calls[0].id, 'call_1')

  def test_message_becomes_content(self):
    self._script(
        openai_mock.Response(output=[_message('It is 15C.')], output_text='x')
    )
    turn = self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    # Read from the output items rather than `output_text`, so that a reply
    # mixing text and calls keeps them in step.
    self.assertEqual(turn.content, 'It is 15C.')
    self.assertEmpty(turn.calls)

  def test_unparsable_arguments_are_handed_over_raw(self):
    """A malformed argument string should reach the tool, not vanish."""
    self._script(
        openai_mock.Response(
            output=[_call('get_weather', 'not json')], output_text=''
        )
    )
    turn = self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    self.assertEqual(turn.calls[0].kwargs, {'_raw_arguments': 'not json'})

  # --- Reasoning round-trip -------------------------------------------------

  def test_reasoning_is_kept_whole_and_echoed_back(self):
    """The API matches reasoning by id, so the item must survive untouched."""
    self._script(
        openai_mock.Response(
            output=[
                _reasoning('Looking it up.', 'rs_7', 'secret'),
                _call('get_weather', '{}'),
            ],
            output_text='',
        ),
        openai_mock.Response(output=[_message('15C')], output_text='15C'),
    )
    turn = self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    self.assertEqual(str(turn.thought), 'Looking it up.')

    # Replaying that turn should put the item back on the wire, before the
    # call it led to.
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        turn,
        tools=[],
    )
    echoed = self._last_request()['input'][1]
    self.assertEqual(echoed['type'], 'reasoning')
    self.assertEqual(echoed['id'], 'rs_7')
    self.assertEqual(echoed['encrypted_content'], 'secret')
    self.assertEqual(self._last_request()['input'][2]['type'], 'function_call')

  def test_reasoning_without_a_summary_still_round_trips(self):
    """An empty summary is common, and dropping it breaks the next request."""
    self._script(
        openai_mock.Response(
            output=[
                openai_mock.ResponseReasoningItem(
                    summary=[], item_id='rs_8', encrypted_content='secret'
                )
            ],
            output_text='',
        ),
        openai_mock.Response(output=[_message('ok')], output_text='ok'),
    )
    turn = self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        tools=[],
    )
    self._ask(
        _ConversationTurn(role=_ConversationRole.USER, content='Weather?'),
        turn,
        tools=[],
    )
    echoed = self._last_request()['input'][1]
    self.assertEqual(echoed['id'], 'rs_8')

  def test_reasoning_from_another_backend_is_not_sent(self):
    """Loose text has no item to replay, and the API has nowhere to put it."""
    self._script(
        openai_mock.Response(output=[_message('ok')], output_text='ok')
    )
    self._ask(
        _ConversationTurn(
            role=_ConversationRole.MODEL,
            content='Hello',
            thought='reasoning from somewhere else',
        ),
        tools=[],
    )
    self.assertEqual(
        self._last_request()['input'],
        [{'role': 'assistant', 'content': 'Hello'}],
    )

  # --- End to end -----------------------------------------------------------

  def test_agent_loop_runs_against_the_mock(self):
    """`Agent` should drive the whole exchange with no scripting."""
    _get_and_register_backend()
    agent = agent_lib.Agent(tools=[_get_weather])
    turn = cast(
        fc_types.ConversationTurn,
        executing.run(agent('What is the weather in London?')),
    )
    self.assertEqual(turn.content, 'Hello')
    self.assertEmpty(turn.calls)

  def test_agent_loop_reports_a_failing_tool(self):
    """A tool that raises is shown to the model rather than ending the run."""
    _get_and_register_backend()
    agent = agent_lib.Agent(tools=[_always_fails])
    turn = cast(
        fc_types.ConversationTurn,
        executing.run(agent('What is the weather in London?')),
    )
    self.assertEqual(turn.content, 'Hello')


if __name__ == '__main__':
  absltest.main()
