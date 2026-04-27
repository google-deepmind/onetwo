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
from typing import Counter, Final, TypeAlias
from unittest import mock

# We have to mock openai to make this test hermetic.
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import openai_api
from onetwo.backends import openai_mock
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
import PIL.Image
import pydantic

import openai

_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


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
      _ = executing.run(llm.instruct(prompt=prompt))  # pytype: disable=wrong-keyword-args
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
        llm.chat(  # pytype: disable=wrong-keyword-args
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
        llm.generate_text(  # pytype: disable=wrong-keyword-args
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
        llm.generate_text(  # pytype: disable=wrong-keyword-args
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
        llm.generate_text(  # pytype: disable=wrong-keyword-args
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
        llm.generate_text(  # pytype: disable=wrong-keyword-args
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
          llm.generate_text(  # pytype: disable=wrong-keyword-args
              prompt=prompt_text,
          )
      )

      self.assertTrue(mock_create.called)
      kwargs = mock_create.call_args.kwargs
      messages = kwargs.get('messages')
      self.assertLen(messages, 2)
      self.assertEqual(messages[0].role, _PredefinedRole.SYSTEM)
      self.assertEqual(
          messages[0].content, openai_api._COMPLETION_SYSTEM_PROMPT_PREFIX
      )
      self.assertEqual(messages[1].role, _PredefinedRole.USER)
      self.assertEqual(messages[1].content, prompt_text)

  def test_generate_object(self):
    """Verifies that generate_object works and calls the mock parse API."""
    backend = _get_and_register_backend()

    class MyTestSchema(pydantic.BaseModel):
      # Must match the mock data in openai_mock.ResponsesNamespace.parse.
      name: str = 'default'
      population: int = 0

    prompt_text = 'Create an object representing a city.'

    res = executing.run(
        llm.generate_object(  # pytype: disable=wrong-keyword-args
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
            llm.generate_object(  # pytype: disable=wrong-keyword-args
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
            llm.generate_object(  # pytype: disable=wrong-keyword-args
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
            backend1.generate_object(  # pytype: disable=wrong-keyword-args
                prompt='Describe a city', cls=CityInfoTestData
            )
        )
        mock_responses_parse.assert_called_once()
        self.assertEqual(obj1, test_object)
        self.assertIsInstance(obj1, CityInfoTestData)
        self.assertEqual(backend1._counters['generate_object'], 1)

      with self.subTest('CacheHitSameBackend'):
        obj2 = executing.run(
            backend1.generate_object(  # pytype: disable=wrong-keyword-args
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
              backend2.generate_object(  # pytype: disable=wrong-keyword-args
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

  def test_embed_basic(self):
    """Verifies that embed returns a list of floats."""
    backend = _get_and_register_backend()
    result = executing.run(
        llm.embed(content='Hello, world!')  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('returns_list_of_floats'):
      self.assertIsInstance(result, (list, tuple))
      self.assertNotEmpty(result)
      for v in result:
        self.assertIsInstance(v, float)
    expected_backend_counters = collections.Counter({
        'embed': 1,
        'embed_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_embed_with_output_dimensionality(self):
    """Verifies that output_dimensionality controls the embedding size."""
    _ = _get_and_register_backend()
    dim = 5
    result = executing.run(
        llm.embed(content='Hello', output_dimensionality=dim)  # pytype: disable=wrong-keyword-args
    )
    self.assertLen(result, dim)

  def test_embed_with_normalize(self):
    """Verifies that normalize=True produces a unit-norm vector."""
    _get_and_register_backend()
    result = executing.run(
        llm.embed(content='Something', normalize=True)  # pytype: disable=wrong-keyword-args
    )
    norm = math.sqrt(sum(v * v for v in result))
    self.assertAlmostEqual(norm, 1.0, places=6)

  def test_embed_without_normalize(self):
    """Verifies that without normalize, the vector is NOT unit-norm."""
    _get_and_register_backend()
    result = executing.run(
        llm.embed(content='Something')  # pytype: disable=wrong-keyword-args
    )
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
        llm.embed(content=chunks)  # pytype: disable=wrong-keyword-args
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
          llm.embed(content=chunks)  # pytype: disable=wrong-keyword-args
      )

  def test_embed_caching(self):
    """Verifies that embed results are cached."""
    backend = _get_and_register_backend()
    content_text = 'Cache me!'
    # First call.
    result1 = executing.run(llm.embed(content=content_text))  # pytype: disable=wrong-keyword-args
    with self.subTest('first_call_hits_api'):
      self.assertEqual(backend._counters['embed'], 1)
    # Second call with same content should be cached.
    result2 = executing.run(llm.embed(content=content_text))  # pytype: disable=wrong-keyword-args
    with self.subTest('second_call_is_cached'):
      self.assertEqual(backend._counters['embed'], 1)
    with self.subTest('results_are_equal'):
      self.assertEqual(result1, result2)


class CountTokensTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):

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
    """Verifies that messages have more tokens than plain text (overhead)."""
    _get_and_register_backend()
    text = 'Hello world'
    messages = [
        _Message(content=text, role=_PredefinedRole.USER),
    ]
    count_text = executing.run(
        llm.count_tokens(content=text)  # pytype: disable=wrong-keyword-args
    )
    count_messages = executing.run(
        llm.count_tokens(content=messages)  # pytype: disable=wrong-keyword-args
    )
    # Messages should have overhead (per-message + reply primer tokens).
    self.assertGreater(count_messages, count_text)

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


if __name__ == '__main__':
  absltest.main()
