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

from collections.abc import Callable, Sequence
import dataclasses
import io
import json
import os
import shutil
import tempfile
import time
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
from onetwo.core import function_calling
from onetwo.core import prefix_caching
from onetwo.core import sampling
from onetwo.core import utils
from onetwo.stdlib.tool_use import tool_schema
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

# A mock vector with unit norm.
_MOCK_EMBEDDING: Final[list[float]] = [
    1.0 / 11.0,
    2.0 / 11.0,
    2.0 / 11.0,
    4.0 / 11.0,
    4.0 / 11.0,
    4.0 / 11.0,
    8.0 / 11.0,
]


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


def genai_content(parts: Sequence[str] | str, role: str | None = None):
  content = parts if isinstance(parts, list) else [parts]
  return genai_types.Content(
      parts=[genai_types.Part(text=p) for p in content],  # pyrefly: ignore[bad-argument-type]
      role=role,
  )


def genai_candidate(parts: Sequence[str] | str, role: str | None = None):
  return genai_types.Candidate(content=genai_content(parts, role))


def _get_and_register_backend(**kwargs) -> google_genai_api.GoogleGenAIAPI:
  """Get an instance of GoogleGenAIAPI and register its methods."""
  backend = google_genai_api.GoogleGenAIAPI(
      api_key='some_key', threadpool_size=_THREADPOOL_SIZE, **kwargs
  )
  backend.register()
  return backend


@dataclasses.dataclass
class RespondWithTextAndTemperature(
    Callable[[genai_types.GenerateContentConfig], Sequence[str]]  # pyrefly: ignore[invalid-inheritance]
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
            candidates=[genai_candidate('text')]
        )
    )
    self._mock_genai_client.models.count_tokens.return_value = (
        genai_types.CountTokensResponse(total_tokens=_MOCK_COUNT_TOKENS_RETURN)
    )
    self.enter_context(
        mock.patch.object(genai, 'Client', return_value=self._mock_genai_client)
    )

    self._mock_genai_client.models.list.return_value = _mock_list_models()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def test_generate_text(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    res = executing.run(llm.generate_text(prompt='Something1'))
    res2 = executing.run(llm.generate_text(prompt='Something2'))

    _, mock_kwargs = self._mock_genai_client.models.generate_content.call_args
    self.assertEqual(mock_kwargs['contents'][0].role, 'user')

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'text')
      self.assertEqual(res2, 'text')
    with self.subTest('CallsApiAndUpdatesCounters'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count, 2
      )
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_text=2, generate_contents=2),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=2,
              get_miss=2,
          ),
      )

  def test_generate_text_updates_tracing_with_usage_metadata(self):
    _ = _get_and_register_backend(disable_caching=True)

    mock_response = mock.Mock()
    mock_response.candidates = [genai_candidate('text')]
    mock_response.usage_metadata = mock.Mock(
        prompt_token_count=10,
        cached_content_token_count=5,
        candidates_token_count=20,
        thoughts_token_count=2,
        tool_use_prompt_token_count=0,
    )
    mock_response.prompt_feedback = None
    mock_response.automatic_function_calling_history = None

    self._mock_genai_client.models.generate_content.return_value = mock_response

    _, trace = executing.run(
        llm.generate_text(prompt='Something'), enable_tracing=True
    )

    # update_info is called inside _generate_contents_inner, which runs within
    # the tracing context established by the @tracing.trace-decorated
    # _generate_contents. The trace structure is:
    #   trace (root)
    #   └── stages[0]: GoogleGenAIAPI.generate_text
    #       └── stages[0]: GoogleGenAIAPI.generate_contents  <- info is here
    generate_text_stage = trace.stages[0]
    self.assertEqual(
        generate_text_stage.stage_name,
        'GoogleGenAIAPI.generate_text',
    )
    generate_contents_stage = generate_text_stage.stages[0]
    self.assertEqual(
        generate_contents_stage.stage_name,
        'GoogleGenAIAPI.generate_contents',
    )
    info = generate_contents_stage.info
    self.assertEqual(info['llm.tokens.prompt'], 10)
    self.assertEqual(info['llm.tokens.cached_content'], 5)
    self.assertEqual(info['llm.tokens.candidates'], 20)
    self.assertEqual(info['llm.tokens.thoughts'], 2)
    self.assertEqual(info['llm.tokens.tool_use_prompt'], 0)
    self.assertEqual(info['llm.tools.calls'], 0)
    self.assertEqual(
        info['llm.model'],
        google_genai_api.DEFAULT_GENERATE_MODEL.gemini_api.rsplit('/')[-1],
    )
    self.assertNotIn('llm.gcp_project', info)

  def test_generate_text_updates_tracing_with_gcp_project(self):
    _ = _get_and_register_backend(disable_caching=True, project='test-project')

    mock_response = mock.Mock()
    mock_response.candidates = [genai_candidate('text')]
    mock_response.usage_metadata = mock.Mock(
        prompt_token_count=10,
        cached_content_token_count=5,
        candidates_token_count=20,
        thoughts_token_count=2,
        tool_use_prompt_token_count=0,
    )
    mock_response.prompt_feedback = None
    mock_response.automatic_function_calling_history = None

    self._mock_genai_client.models.generate_content.return_value = mock_response

    _, trace = executing.run(
        llm.generate_text(prompt='Something'), enable_tracing=True
    )

    generate_text_stage = trace.stages[0]
    generate_contents_stage = generate_text_stage.stages[0]
    info = generate_contents_stage.info
    self.assertEqual(info['llm.gcp_project'], 'test-project')

  @parameterized.named_parameters(
      dict(
          testcase_name='no_tool_calls',
          candidate_parts=[genai_types.Part(text='Hello')],
          afc_history=None,
          expected_tool_calls=0,
      ),
      dict(
          testcase_name='single_tool_call_in_candidates',
          candidate_parts=[
              genai_types.Part(
                  function_call=genai_types.FunctionCall(
                      name='get_weather', args={'location': 'London'}
                  )
              ),
          ],
          afc_history=None,
          expected_tool_calls=1,
      ),
      dict(
          testcase_name='multiple_tool_calls_in_candidates',
          candidate_parts=[
              genai_types.Part(
                  function_call=genai_types.FunctionCall(
                      name='get_weather', args={'location': 'London'}
                  )
              ),
              genai_types.Part(
                  function_call=genai_types.FunctionCall(
                      name='get_weather', args={'location': 'Paris'}
                  )
              ),
          ],
          afc_history=None,
          expected_tool_calls=2,
      ),
      dict(
          testcase_name='tool_calls_from_afc_history',
          # Final response has only text (MCP/auto-tool-call path).
          candidate_parts=[genai_types.Part(text='The result is 5.')],
          afc_history=[
              genai_types.Content(
                  parts=[
                      genai_types.Part(
                          function_call=genai_types.FunctionCall(
                              name='add', args={'a': 2, 'b': 3}
                          )
                      ),
                      genai_types.Part(
                          function_call=genai_types.FunctionCall(
                              name='multiply', args={'a': 4, 'b': 5}
                          )
                      ),
                  ],
                  role='model',
              ),
              genai_types.Content(
                  parts=[
                      genai_types.Part(
                          function_response=genai_types.FunctionResponse(
                              name='add', response={'result': 5}
                          )
                      ),
                      genai_types.Part(
                          function_response=genai_types.FunctionResponse(
                              name='multiply', response={'result': 20}
                          )
                      ),
                  ],
                  role='user',
              ),
          ],
          expected_tool_calls=2,
      ),
  )
  def test_tracing_tool_call_count(
      self, candidate_parts, afc_history, expected_tool_calls
  ):
    _ = _get_and_register_backend(disable_caching=True)

    content = genai_types.Content(parts=candidate_parts, role='model')
    candidate = genai_types.Candidate(content=content)

    mock_response = mock.Mock()
    mock_response.candidates = [candidate]
    mock_response.usage_metadata = mock.Mock(
        prompt_token_count=10,
        cached_content_token_count=0,
        candidates_token_count=15,
        thoughts_token_count=0,
        tool_use_prompt_token_count=0,
    )
    mock_response.prompt_feedback = None
    mock_response.automatic_function_calling_history = afc_history

    self._mock_genai_client.models.generate_content.return_value = mock_response

    _, trace = executing.run(
        llm.generate_text(prompt='test'), enable_tracing=True
    )

    generate_contents_stage = trace.stages[0].stages[0]
    info = generate_contents_stage.info
    self.assertEqual(info['llm.tools.calls'], expected_tool_calls)

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
      self.assertEqual(
          mock_kwargs['contents'][0].parts[0].text, 'Some unsafe prompt'
      )
    with self.subTest('UpdatesBackendCounters'):
      self.assertCounterEqual(
          backend._counters,
          Counter(generate_text=1, generate_contents=1),
      )

  def test_generate_text_blocked_by_safety_filters(self):
    _ = _get_and_register_backend()

    # Simulate a response blocked by safety filters.
    mock_prompt_feedback = genai_types.GenerateContentResponsePromptFeedback(
        block_reason=genai_types.BlockedReason.SAFETY,
        safety_ratings=[
            genai_types.SafetyRating(
                category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                probability=genai_types.HarmProbability.HIGH,
            )
        ],
    )
    self._mock_genai_client.models.generate_content.return_value = (
        genai_types.GenerateContentResponse(
            candidates=[],
            prompt_feedback=mock_prompt_feedback,
        )
    )

    with self.assertRaisesRegex(
        ValueError,
        r'(?s)GoogleGenAIAPI.generate_content request was blocked by safety'
        r' filters:.*block_reason.*SAFETY',
    ):
      executing.run(llm.generate_text(prompt='Unsafe prompt'))

    # Assert that the chat is called with the correct prompt.
    self._mock_genai_client.models.generate_content.assert_called_once()

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
          Counter(generate_text=2, generate_contents=2),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=2,
              get_miss=2,
          ),
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
          Counter(generate_text=2, generate_contents=1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              get_miss=1,
              get_hit=1,
          ),
      )

  def test_generate_text_with_retry(self):
    self._mock_genai_client.models.generate_content.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.ReadError('test'),
        genai_types.GenerateContentResponse(
            candidates=[genai_candidate('text')],
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
          backend._counters,
          Counter(generate_text=1, generate_contents=max_retries + 1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              get_miss=1,
          ),
      )

  def test_generate_text_with_server_error_retry(self):
    self._mock_genai_client.models.generate_content.side_effect = [
        genai_errors.ServerError(
            code=500, response_json={'error': {'message': 'test'}}
        ),
        genai_types.GenerateContentResponse(
            candidates=[genai_candidate('text')],
        ),
    ]
    max_retries = 1
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
          backend._counters,
          Counter(generate_text=1, generate_contents=max_retries + 1),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              get_miss=1,
          ),
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
      self.assertCounterEqual(
          backend._counters, Counter(generate_text=1, generate_contents=1)
      )
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
            ),
            # Although in real usage, we would prefer a higher number of
            # parallel executions, we use `chunk_size=1` here to force the
            # requests to be executed sequentially and consequently make the
            # cache hits/misses predictable. If the identical requests are
            # executed sequentially, we guarantee cache hits for the 2nd and
            # later ones; if simultaneous, they could potentially all be cache
            # misses.
            chunk_size=1,
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
          backend._counters,
          Counter(generate_text=num_repeats, generate_contents=num_repeats),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              add_new_sample=4,
              get_miss=1,
              get_hit_miss_sample=4,
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
            ),
            chunk_size=1,  # Sequential execution for predictable cache hits.
        )
    )

    num_repeats_2 = 6
    res2 = executing.run(
        executing.par_iter(
            sampling.repeat(
                executable=llm.generate_text(prompt='Something'),
                num_repeats=num_repeats_2,
            ),
            chunk_size=1,  # Sequential execution for predictable cache hits.
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
          backend._counters,
          Counter(
              generate_text=num_repeats_1 + num_repeats_2,
              generate_contents=num_repeats_2,
          ),
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              add_new_sample=5,
              get_miss=1,
              get_hit=5,
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
              generate_contents=len(prompts),
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
          Counter(generate_text=num_calls, generate_contents=1),
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

    json_str = pydantic.TypeAdapter(list[Recipe]).dump_json(expected_objects)
    mock_response = genai_types.GenerateContentResponse(
        candidates=[genai_candidate(json_str)]  # pyrefly: ignore[bad-argument-type]
    )
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = mock_response

    prompt = 'List some cookie recipes'
    result = executing.run(llm.generate_object(prompt=prompt, cls=list[Recipe]))

    with self.subTest('ResultsStructuredCorrectly'):
      self.assertEqual(result, expected_objects)
      self.assertIsInstance(result, list)
      if result:
        self.assertIsInstance(result[0], Recipe)

    with self.subTest('GenerateContentCall'):
      gc_mock.assert_called_once()
      _, mock_kwargs = gc_mock.call_args
      self.assertEqual(
          mock_kwargs['config'].response_mime_type, 'application/json'
      )
      self.assertEqual(mock_kwargs['config'].response_schema, list[Recipe])
      self.assertEqual(mock_kwargs['contents'][0].parts[0].text, prompt)

    with self.subTest('Counters'):
      self.assertCounterEqual(backend._counters, Counter(generate_contents=1))

  def test_generate_object_caching(self):
    cache_filename = os.path.join(self.temp_dir, 'test_cache.json')

    # Instance 1: Populate cache
    cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    backend1 = _get_and_register_backend(cache=cache)

    test_object = MyData(name='Test', value=123, details=['a', 'b'])
    json_str = pydantic.TypeAdapter(MyData).dump_json(test_object)

    mock_response = genai_types.GenerateContentResponse(
        candidates=[genai_candidate(json_str)]  # pyrefly: ignore[bad-argument-type]
    )

    mock_generate_content = self._mock_genai_client.models.generate_content
    mock_generate_content.return_value = mock_response

    with self.subTest('PopulateCache'):
      obj1 = executing.run(  # pyrefly: ignore[no-matching-overload]
          backend1.generate_object(prompt='some prompt', cls=MyData)  # pyrefly: ignore[bad-argument-type, unexpected-keyword]
      )
      mock_generate_content.assert_called_once()
      self.assertEqual(obj1, test_object)
      self.assertIsInstance(obj1, MyData)
      self.assertEqual(backend1._counters['generate_contents'], 1)

    with self.subTest('CacheHitSameBackend'):
      obj2 = executing.run(  # pyrefly: ignore[no-matching-overload]
          backend1.generate_object(prompt='some prompt', cls=MyData)  # pyrefly: ignore[bad-argument-type, unexpected-keyword]
      )
      self.assertEqual(mock_generate_content.call_count, 1)
      self.assertEqual(obj2, test_object)
      self.assertIsInstance(obj2, MyData)
      self.assertEqual(backend1._counters['generate_contents'], 1)

    cache.save()
    self.assertTrue(os.path.exists(cache_filename))

    # Instance 2: Load from cache
    cache2 = caching.SimpleFunctionCache(cache_filename=cache_filename)
    backend2 = _get_and_register_backend(cache=cache2)
    cache2.load()

    with self.subTest('CacheHitNewBackend'):
      # This call should hit the loaded cache
      obj3 = executing.run(  # pyrefly: ignore[no-matching-overload]
          backend2.generate_object(prompt='some prompt', cls=MyData)  # pyrefly: ignore[bad-argument-type, unexpected-keyword]
      )
      self.assertEqual(obj3, test_object)
      self.assertIsInstance(obj3, MyData)
      # generate_object on backend2 was never actually called beyond the cache
      self.assertEqual(backend2._counters['generate_contents'], 0)

    # 1 hit in backend1, 1 in backend2
    self.assertEqual(cache2._cache_data.counters['get_hit'], 2)

  def test_chat(self):
    backend = _get_and_register_backend()
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    msg_model = content_lib.Message(
        role=content_lib.PredefinedRole.MODEL, content='Hello user'
    )

    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('Hello')]
    )

    # First call, no history.
    _ = executing.run(llm.chat(messages=[msg_user]))
    with self.subTest('OneMessageSent'):
      gc_mock.assert_called_once()
      contents = gc_mock.call_args.kwargs['contents']
      self.assertEqual(contents, [genai_content('Hello model', role='user')])

    gc_mock.reset_mock()

    # Second call, with one message in the history.
    _ = executing.run(llm.chat(messages=[msg_model, msg_user]))
    with self.subTest('TwoMessagesSent'):
      gc_mock.assert_called_once()
      contents = gc_mock.call_args.kwargs['contents']
      self.assertEqual(
          contents,
          [
              genai_content('Hello user', role='model'),
              genai_content('Hello model', role='user'),
          ],
      )

    self.assertCounterEqual(
        backend._counters, Counter(chat=2, generate_contents=2)
    )

  def test_chat_with_unsafe_content(self):
    backend = _get_and_register_backend()
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    gc_mock = self._mock_genai_client.models.generate_content

    # Empty content is returned when the message is unsafe.
    gc_mock.return_value = genai_types.GenerateContentResponse(
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
      gc_mock.assert_called_once()
      args = gc_mock.call_args.kwargs
      self.assertEqual(args['contents'][0].parts[0].text, 'Hello model')
    with self.subTest('UpdatesBackendCounters'):
      self.assertCounterEqual(
          backend._counters, Counter(chat=1, generate_contents=1)
      )

  def test_chat_with_system_instruction(self):
    backend = _get_and_register_backend()
    msg_system = content_lib.Message(
        role=content_lib.PredefinedRole.SYSTEM, content='System message'
    )
    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )

    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('Hello')]
    )

    # First call, with system instruction.
    _ = executing.run(llm.chat(messages=[msg_system, msg_user]))
    with self.subTest('FirstChat'):
      gc_mock.assert_called_once()
      args = gc_mock.call_args.kwargs
      self.assertEqual(
          args['config'].system_instruction,
          genai_content('System message', 'system'),
      )
      self.assertEqual(
          args['contents'],
          [genai_content('Hello model', 'user')],
      )
      gc_mock.reset_mock()

    # Second call, with system instruction passed as a kwarg.
    _ = executing.run(
        llm.chat(messages=[msg_user], system_instruction='System message 2')
    )
    with self.subTest('SecondChatWithSystemInstruction'):
      gc_mock.assert_called_once()
      args = gc_mock.call_args.kwargs
      self.assertEqual(args['config'].system_instruction, 'System message 2')
      self.assertEqual(
          args['contents'],
          [genai_content('Hello model', 'user')],
      )
    self.assertCounterEqual(
        backend._counters, Counter(chat=2, generate_contents=2)
    )

  def test_chat_replace_unsupported_roles(self):
    backend = _get_and_register_backend(replace_unsupported_roles=True)
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('Hello')]
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
    with self.subTest('ChatCreatedCorrectHistory'):
      gc_mock.assert_called_once()
      args = gc_mock.call_args.kwargs
      system = args['config'].system_instruction
      self.assertEqual(
          system,
          genai_content(['System message', 'System message 2'], role='system'),
      )
      contents = args['contents']
      self.assertEqual(
          contents,
          [
              genai_content(
                  ['Context message', 'Context message 2'],
                  role='user',
              ),
              genai_content(['Hello model'], role='user'),
          ],
      )
    self.assertCounterEqual(
        backend._counters, Counter(chat=1, generate_contents=1)
    )

  def test_part_from_chunk_function_call(self):
    fc_call = genai_types.FunctionCall(
        name='get_weather', args={'location': 'London'}
    )
    chunk = content_lib.Chunk(content='[Function Call]')
    chunk.metadata = {google_genai_api.FUNCTION_CALL_METADATA_KEY: fc_call}

    part = google_genai_api._part_from_chunk(chunk)

    self.assertIsNotNone(part.function_call)
    self.assertEqual(part.function_call.name, 'get_weather')
    self.assertEqual(part.function_call.args, {'location': 'London'})

  def test_part_from_chunk_function_response(self):
    fr_resp = genai_types.FunctionResponse(
        name='get_weather', response={'output': '15C'}
    )
    chunk = content_lib.Chunk(content='[Function Response]')
    chunk.metadata = {google_genai_api.FUNCTION_RESPONSE_METADATA_KEY: fr_resp}

    part = google_genai_api._part_from_chunk(chunk)

    self.assertIsNotNone(part.function_response)
    self.assertEqual(part.function_response.name, 'get_weather')
    self.assertEqual(part.function_response.response, {'output': '15C'})

  def test_chunks_with_details_from_candidate_function_call(self):
    fc_call = genai_types.FunctionCall(
        name='get_weather', args={'location': 'London'}
    )
    part = genai_types.Part(function_call=fc_call)
    content = genai_types.Content(parts=[part], role='model')
    candidate = genai_types.Candidate(content=content)

    chunk_list, _ = google_genai_api._chunks_with_details_from_candidate(
        candidate
    )

    self.assertLen(chunk_list, 1)
    chunk = chunk_list.chunks[0]
    self.assertIsNotNone(chunk.metadata)
    self.assertIn(google_genai_api.FUNCTION_CALL_METADATA_KEY, chunk.metadata)
    self.assertEqual(
        chunk.metadata[google_genai_api.FUNCTION_CALL_METADATA_KEY].name,
        'get_weather',
    )
    self.assertEqual(
        chunk.metadata[google_genai_api.FUNCTION_CALL_METADATA_KEY].args,
        {'location': 'London'},
    )

  @parameterized.named_parameters(
      (
          'network_errors',
          [
              genai_errors.ClientError(
                  code=408, response_json={'error': {'message': 'test'}}
              ),
              httpx.TransportError('test'),
          ],
      ),
      (
          'safety_filter_error',
          [
              # This response causes the backend to raise the ValueError,
              # which should now be retried.
              genai_types.GenerateContentResponse(
                  candidates=[genai_types.Candidate(content=None)]
              )
          ],
      ),
  )
  def test_chat_with_retry(self, retriable_side_effects):
    gc_mock = self._mock_genai_client.models.generate_content

    # success response to return after the errors
    success_response = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('Hello')]
    )

    gc_mock.side_effect = retriable_side_effects + [success_response]

    # We set max_retries exactly to the number of errors to ensure we
    # don't loop forever if the error wasn't actually retriable.
    max_retries = len(retriable_side_effects)
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    msg_user = content_lib.Message(
        role=content_lib.PredefinedRole.USER, content='Hello model'
    )
    res = executing.run(llm.chat(messages=[msg_user]))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, 'Hello')
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(gc_mock.call_count, max_retries + 1)
      self.assertCounterEqual(
          backend._counters, Counter(chat=1, generate_contents=max_retries + 1)
      )
    with self.subTest('UpdatesCacheCounters'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(
              add_new=1,
              get_miss=1,
          ),
      )

  def test_chat_with_non_retriable_error(self):
    """Tests that non-retriable errors are not retried in chat."""
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.side_effect = genai_errors.ClientError(
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
      gc_mock.assert_called_once()
      self.assertCounterEqual(
          backend._counters, Counter(chat=1, generate_contents=1)
      )
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

  def test_convert_chunk_list_with_pil_image_raises_error(self):
    """Tests that PIL Image is not supported."""
    pil_image = Image.new('RGB', (1, 1), color='black')
    with self.assertRaises(TypeError):
      google_genai_api._convert_chunk_list_to_part_list(
          _ChunkList(chunks=[_Chunk(content=pil_image)])
      )

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

  @parameterized.named_parameters(
      dict(
          testcase_name='non_normalized_backend_output_normalize_true',
          backend_output=[v * 2.0 for v in _MOCK_EMBEDDING],
          expected_output=_MOCK_EMBEDDING,
          normalize=True,
      ),
      dict(
          testcase_name='non_normalized_backend_output_normalize_false',
          backend_output=[v * 2.0 for v in _MOCK_EMBEDDING],
          expected_output=[v * 2.0 for v in _MOCK_EMBEDDING],
          normalize=False,
      ),
      dict(
          testcase_name='normalized_backend_output_normalize_true',
          backend_output=_MOCK_EMBEDDING,
          expected_output=_MOCK_EMBEDDING,
          normalize=True,
      ),
      dict(
          testcase_name='normalized_backend_output_normalize_false',
          backend_output=_MOCK_EMBEDDING,
          expected_output=_MOCK_EMBEDDING,
          normalize=False,
      ),
  )
  def test_embed_string(self, backend_output, expected_output, normalize):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of embed_content to return a constant value.
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=backend_output)]
        )
    )
    content = 'Something'
    res = executing.run(llm.embed(content=content, normalize=normalize))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, expected_output)
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
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )
    content = content_lib.ChunkList(
        chunks=[content_lib.Chunk(content='Something')]
    )
    res = executing.run(llm.embed(content=content))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_EMBEDDING)
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

  def test_embed_caching_compatibility(self):
    def make_expected_hash(
        content: str, task_type: str | None, has_task_type: bool = True
    ) -> str:
      content_hash = utils.get_str_hash(content)
      key_dict = {
          '_destination': 'embed',
          'content': content_hash,
      }
      if has_task_type:
        key_dict['task_type'] = task_type  # pyrefly: ignore[bad-assignment]
      key_str = json.dumps(key_dict, sort_keys=True)
      return utils.get_str_hash(key_str)

    # 1. Test with v1 model (default: gemini-embedding-001)
    backend_v1 = _get_and_register_backend()
    handler_v1 = backend_v1.cache
    assert isinstance(handler_v1, caching.SimpleFunctionCache)
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    # Call with task_type=None (default)
    _ = executing.run(llm.embed(content='text_v1_none'))
    # Call with explicit task_type
    _ = executing.run(
        llm.embed(
            content='text_v1_query',
            task_type=llm_lib.EmbedTaskType.RETRIEVAL_QUERY,
        )
    )

    # We expect one key with task_type: None and one with task_type:
    # 'RETRIEVAL_QUERY'.
    expected_hash_none_v1 = make_expected_hash('text_v1_none', None)
    expected_hash_query_v1 = make_expected_hash(
        'text_v1_query', 'RETRIEVAL_QUERY'
    )

    self.assertIn(expected_hash_none_v1, handler_v1._cache_data.values_by_key)
    self.assertIn(expected_hash_query_v1, handler_v1._cache_data.values_by_key)

    # 2. Test with v2 model
    backend_v2 = _get_and_register_backend(
        embed_model_name='models/gemini-embedding-2'
    )
    handler_v2 = backend_v2.cache
    assert isinstance(handler_v2, caching.SimpleFunctionCache)
    # Reset mock for new backend calls
    self._mock_genai_client.models.embed_content.reset_mock()
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    # Call with task_type=None (default)
    _ = executing.run(llm.embed(content='text_v2_none'))
    # Call with explicit task_type (will modify content with prefix and pop
    # task_type)
    _ = executing.run(
        llm.embed(
            content='text_v2_query',
            task_type=llm_lib.EmbedTaskType.RETRIEVAL_QUERY,
        )
    )

    # For v2 with task_type=None, task_type should be present and None
    expected_hash_none_v2 = make_expected_hash('text_v2_none', None)
    # For v2 with explicit task_type, content is prefixed, task_type is None
    expected_hash_query_v2 = make_expected_hash(
        'task: search result | query: text_v2_query', None
    )

    self.assertIn(expected_hash_none_v2, handler_v2._cache_data.values_by_key)
    self.assertIn(expected_hash_query_v2, handler_v2._cache_data.values_by_key)

  def test_embed_updates_tracing_with_model_name(self):
    _ = _get_and_register_backend(disable_caching=True)

    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    _, trace = executing.run(
        llm.embed(content='Something'), enable_tracing=True
    )

    # The trace structure is:
    #   trace (root)
    #   └── stages[0]: GoogleGenAIAPI.embed  <- info is here
    embed_stage = trace.stages[0]
    self.assertEqual(embed_stage.stage_name, 'GoogleGenAIAPI.embed')
    info = embed_stage.info
    self.assertEqual(
        info['llm.model'],
        google_genai_api.DEFAULT_EMBED_MODEL.gemini_api.rsplit('/')[-1],
    )
    self.assertNotIn('llm.gcp_project', info)

  def test_embed_updates_tracing_with_project(self):
    _ = _get_and_register_backend(disable_caching=True, project='test-project')

    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    _, trace = executing.run(
        llm.embed(content='Something'), enable_tracing=True
    )

    embed_stage = trace.stages[0]
    self.assertEqual(embed_stage.stage_name, 'GoogleGenAIAPI.embed')
    info = embed_stage.info
    self.assertEqual(info['llm.gcp_project'], 'test-project')

  def test_embed_with_retry(self):
    self._mock_genai_client.models.embed_content.side_effect = [
        genai_errors.ClientError(
            code=408, response_json={'error': {'message': 'test'}}
        ),
        httpx.TransportError('test'),
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        ),
    ]
    max_retries = 2
    backend = _get_and_register_backend(max_retries=max_retries)
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')

    res = executing.run(llm.embed(content='Something'))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_EMBEDDING)
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

  @parameterized.named_parameters(
      dict(
          testcase_name='with_task_type',
          embed_kwargs={'task_type': 'RETRIEVAL_DOCUMENT'},
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_DOCUMENT'
          ),
      ),
      dict(
          testcase_name='with_task_type_and_other_kwargs',
          embed_kwargs={
              'task_type': 'RETRIEVAL_DOCUMENT',
              'output_dimensionality': 768,
          },
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_DOCUMENT', output_dimensionality=768
          ),
      ),
      dict(
          testcase_name='with_retrieval_document_and_title',
          embed_kwargs={
              'task_type': 'RETRIEVAL_DOCUMENT',
              'title': 'My Title',
          },
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_DOCUMENT', title='My Title'
          ),
      ),
      dict(
          testcase_name='with_title_only_defaults_to_retrieval_document',
          embed_kwargs={'title': 'My Title'},
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_DOCUMENT', title='My Title'
          ),
      ),
      dict(
          testcase_name='without_task_type',
          embed_kwargs={},
          expected_config=None,
      ),
  )
  def test_embed_with_kwargs(self, embed_kwargs, expected_config):
    backend = _get_and_register_backend()
    content = 'Some text to embed'
    # We mock the return value of embed_content to return a constant value.
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    res = executing.run(llm.embed(content=content, **embed_kwargs))

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(res, _MOCK_EMBEDDING)
    with self.subTest('CallsApiWithCorrectConfig'):
      self._mock_genai_client.models.embed_content.assert_called_once()
      _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args
      self.assertEqual(mock_kwargs['config'], expected_config)
      self.assertEqual(
          mock_kwargs['contents'],
          [genai_types.Part(text=content)],
      )
    with self.subTest('UpdatesBackendCounters'):
      self.assertCounterEqual(
          backend._counters,
          Counter(embed=1),
      )

  def test_embed_caching_with_task_type(self):
    backend = _get_and_register_backend()
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)

    # We mock the return value of embed_content to return a constant value.
    embed_mock = self._mock_genai_client.models.embed_content
    embed_mock.return_value = genai_types.EmbedContentResponse(
        embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
    )

    content = 'Some text to embed'

    # Call 1: No task_type
    executing.run(llm.embed(content=content))
    with self.subTest('Call 1: No task_type'):
      embed_mock.assert_called_once()
      self.assertCounterEqual(backend._counters, Counter(embed=1))
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=1, get_miss=1)
      )

    # Call 2: task_type = RETRIEVAL_DOCUMENT
    executing.run(llm.embed(content=content, task_type='RETRIEVAL_DOCUMENT'))
    with self.subTest('Call 2: RETRIEVAL_DOCUMENT'):
      self.assertEqual(embed_mock.call_count, 2)
      self.assertCounterEqual(backend._counters, Counter(embed=2))
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=2, get_miss=2)
      )
      _, mock_kwargs = embed_mock.call_args
      self.assertEqual(mock_kwargs['config'].task_type, 'RETRIEVAL_DOCUMENT')

    # Call 3: task_type = SEMANTIC_SIMILARITY
    executing.run(llm.embed(content=content, task_type='SEMANTIC_SIMILARITY'))
    with self.subTest('Call 3: SEMANTIC_SIMILARITY'):
      self.assertEqual(embed_mock.call_count, 3)
      self.assertCounterEqual(backend._counters, Counter(embed=3))
      self.assertCounterEqual(
          handler._cache_data.counters, Counter(add_new=3, get_miss=3)
      )
      _, mock_kwargs = embed_mock.call_args
      self.assertEqual(mock_kwargs['config'].task_type, 'SEMANTIC_SIMILARITY')

    # Call 4: task_type = RETRIEVAL_DOCUMENT (Cache Hit)
    executing.run(llm.embed(content=content, task_type='RETRIEVAL_DOCUMENT'))
    with self.subTest('Call 4: RETRIEVAL_DOCUMENT (Cache Hit)'):
      self.assertEqual(embed_mock.call_count, 3)  # Should not increase
      self.assertCounterEqual(
          backend._counters, Counter(embed=3)
      )  # Should not increase
      self.assertCounterEqual(
          handler._cache_data.counters,
          Counter(add_new=3, get_miss=3, get_hit=1),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='retrieval_query',
          task_type='RETRIEVAL_QUERY',
          title=None,
          expected_text='task: search result | query: hello',
      ),
      dict(
          testcase_name='retrieval_document_with_title',
          task_type='RETRIEVAL_DOCUMENT',
          title='My Title',
          expected_text='title: My Title | text: hello',
      ),
      dict(
          testcase_name='retrieval_document_no_title',
          task_type='RETRIEVAL_DOCUMENT',
          title=None,
          expected_text='title: none | text: hello',
      ),
      dict(
          testcase_name='title_only_defaults_to_document',
          task_type=None,
          title='My Title',
          expected_text='title: My Title | text: hello',
      ),
      dict(
          testcase_name='semantic_similarity',
          task_type='SEMANTIC_SIMILARITY',
          title=None,
          expected_text='task: sentence similarity | query: hello',
      ),
      dict(
          testcase_name='classification',
          task_type='CLASSIFICATION',
          title=None,
          expected_text='task: classification | query: hello',
      ),
  )
  def test_embed_v2_formatting(self, task_type, title, expected_text):
    _ = _get_and_register_backend(embed_model_name='models/gemini-embedding-2')
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    _ = executing.run(
        llm.embed(content='hello', title=title, task_type=task_type)
    )

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args

    # Verify the formatted text was sent
    self.assertEqual(
        mock_kwargs['contents'],
        [genai_types.Part(text=expected_text)],
    )
    # Verify no task_type was sent in the config
    self.assertIsNone(mock_kwargs.get('config'))

  def test_embed_v2_multimodal_formatting(self):
    _ = _get_and_register_backend(embed_model_name='models/gemini-embedding-2')
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk(content='hello'),
            content_lib.Chunk(content=b'image_data', content_type='image/png'),
        ]
    )

    _ = executing.run(
        llm.embed(
            content=content, title='My Title', task_type='RETRIEVAL_DOCUMENT'
        )
    )

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args

    # Verify the chunks were prepended with the prefix chunk
    self.assertEqual(
        mock_kwargs['contents'],
        [
            genai_types.Part(text='title: My Title | text: '),
            genai_types.Part(text='hello'),
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type='image/png', data=b'image_data'
                )
            ),
        ],
    )

  def test_embed_v1_title_formatting(self):
    _ = _get_and_register_backend(
        embed_model_name='models/gemini-embedding-001'
    )
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    _ = executing.run(
        llm.embed(
            content='hello', title='My Title', task_type='RETRIEVAL_DOCUMENT'
        )
    )

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args

    # Verify raw text was sent and title was passed via EmbedContentConfig
    self.assertEqual(
        mock_kwargs['contents'],
        [genai_types.Part(text='hello')],
    )
    self.assertEqual(
        mock_kwargs['config'],
        genai_types.EmbedContentConfig(
            task_type='RETRIEVAL_DOCUMENT', title='My Title'
        ),
    )

  def test_embed_v1_title_multimodal_formatting(self):
    _ = _get_and_register_backend(
        embed_model_name='models/gemini-embedding-001'
    )
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk(content='hello'),
            content_lib.Chunk(content=b'image_data', content_type='image/png'),
        ]
    )

    _ = executing.run(
        llm.embed(
            content=content, title='My Title', task_type='RETRIEVAL_DOCUMENT'
        )
    )

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args

    # Verify the chunks were sent unprefixed and title was passed via config
    self.assertEqual(
        mock_kwargs['contents'],
        [
            genai_types.Part(text='hello'),
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type='image/png', data=b'image_data'
                )
            ),
        ],
    )
    self.assertEqual(
        mock_kwargs['config'],
        genai_types.EmbedContentConfig(
            task_type='RETRIEVAL_DOCUMENT', title='My Title'
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='string_without_task_type',
          content='',
          embed_kwargs={},
          expected_config=None,
      ),
      dict(
          testcase_name='string_retrieval_document_with_title',
          content='',
          embed_kwargs={
              'task_type': 'RETRIEVAL_DOCUMENT',
              'title': 'My Title',
          },
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_DOCUMENT', title='My Title'
          ),
      ),
      dict(
          testcase_name='chunk_list_retrieval_query',
          content=content_lib.ChunkList(chunks=[]),
          embed_kwargs={'task_type': 'RETRIEVAL_QUERY'},
          expected_config=genai_types.EmbedContentConfig(
              task_type='RETRIEVAL_QUERY'
          ),
      ),
  )
  def test_embed_v1_empty_content(self, content, embed_kwargs, expected_config):
    _ = _get_and_register_backend(
        embed_model_name='models/gemini-embedding-001'
    )
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    _ = executing.run(llm.embed(content=content, **embed_kwargs))

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args
    # The API rejects empty content, so it is replaced with 'none'.
    self.assertEqual(mock_kwargs['contents'], [genai_types.Part(text='none')])
    self.assertEqual(mock_kwargs['config'], expected_config)

  def test_embed_v1_multi_chunk_text(self):
    _ = _get_and_register_backend(
        embed_model_name='models/gemini-embedding-001'
    )
    self._mock_genai_client.models.embed_content.return_value = (
        genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=_MOCK_EMBEDDING)]
        )
    )

    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk(content='hello '),
            content_lib.Chunk(content='world'),
        ]
    )

    _ = executing.run(llm.embed(content=content))

    self._mock_genai_client.models.embed_content.assert_called_once()
    _, mock_kwargs = self._mock_genai_client.models.embed_content.call_args
    # Multi-chunk text ChunkLists are joined into a single Part to prevent the
    # SDK from sending multiple instances to the Vertex AI predict API.
    self.assertEqual(
        mock_kwargs['contents'], [genai_types.Part(text='hello world')]
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

  def test_generate_content_with_mcp_session(self):
    _ = _get_and_register_backend()

    class FakeClientSession:
      pass

    mock_mcp = mock.MagicMock()
    mock_mcp.ClientSession = FakeClientSession
    self.enter_context(mock.patch.dict('sys.modules', {'mcp': mock_mcp}))

    aio_gc_mock = mock.AsyncMock()
    aio_gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('async_mcp_text')]
    )
    self._mock_genai_client.aio.models.generate_content = aio_gc_mock

    # Instantiate our fake tool which behaves as an McpClientSession.
    fake_tool = FakeClientSession()

    res = executing.run(
        llm.generate_content(
            prompt='mcp_prompt',
            tools=[fake_tool],
        )
    )

    self.assertEqual(str(res), 'async_mcp_text')
    aio_gc_mock.assert_called_once()
    self._mock_genai_client.models.generate_content.assert_not_called()

  def test_generate_content_with_mcp_session_retry(self):
    """Tests that MCP content generation retries on retriable errors."""
    max_retries = 2

    class FakeClientSession:
      pass

    mock_mcp = mock.MagicMock()
    mock_mcp.ClientSession = FakeClientSession
    self.enter_context(mock.patch.dict('sys.modules', {'mcp': mock_mcp}))

    aio_gc_mock = mock.AsyncMock()
    aio_gc_mock.side_effect = [
        genai_errors.ClientError(
            code=429, response_json={'error': {'message': 'rate limited'}}
        ),
        httpx.ReadError('transport error'),
        genai_types.GenerateContentResponse(
            candidates=[genai_candidate('mcp_retry_text')]
        ),
    ]
    self._mock_genai_client.aio.models.generate_content = aio_gc_mock

    _ = _get_and_register_backend(max_retries=max_retries)
    fake_tool = FakeClientSession()

    res = executing.run(
        llm.generate_content(
            prompt='mcp_prompt',
            tools=[fake_tool],
        )
    )

    with self.subTest('ReturnsCorrectResult'):
      self.assertEqual(str(res), 'mcp_retry_text')
    with self.subTest('RetriesExpectedNumberOfTimes'):
      self.assertEqual(aio_gc_mock.call_count, max_retries + 1)
    with self.subTest('DoesNotUseSync'):
      self._mock_genai_client.models.generate_content.assert_not_called()

  def test_generate_content_with_mcp_session_non_retriable_error(self):
    """Tests that MCP generation does not retry non-retriable errors."""

    class FakeClientSession:
      pass

    mock_mcp = mock.MagicMock()
    mock_mcp.ClientSession = FakeClientSession
    self.enter_context(mock.patch.dict('sys.modules', {'mcp': mock_mcp}))

    aio_gc_mock = mock.AsyncMock()
    aio_gc_mock.side_effect = genai_errors.ClientError(
        code=400, response_json={'error': {'message': 'bad request'}}
    )
    self._mock_genai_client.aio.models.generate_content = aio_gc_mock

    _ = _get_and_register_backend(max_retries=2)
    fake_tool = FakeClientSession()

    with self.assertRaises(ValueError):
      executing.run(
          llm.generate_content(
              prompt='mcp_prompt',
              tools=[fake_tool],
          )
      )

    with self.subTest('DoesNotRetry'):
      aio_gc_mock.assert_called_once()

  def test_generate_content_args(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('text2')]
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
        [genai_content('prompt1', role='user')],
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
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

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
        _Chunk('text', role='user'),
        _Chunk(b'SELECT bytes', 'application/sql', role='model'),
        _Chunk(image_bytes, 'image/png', role='user'),
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
        genai_content(['system', 'instructions'], role='system'),
    )
    self.assertEqual(
        out['contents'],
        [
            genai_content('hello', role='user'),
            genai_content('world', role='model'),
            genai_content('hola', role='user'),
        ],
    )

  def test_generate_content_samples(self):
    backend = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content

    def generate_contents(samples: int, **kwargs):
      """Demonstrates generating multiple candidates e.g. generate_texts()."""
      return executing.run(  # pyrefly: ignore[no-matching-overload]
          backend._generate_contents(samples=samples, **kwargs)  # pyrefly: ignore[bad-argument-type, unexpected-keyword]
      )

    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[
            genai_candidate(['c1', 'snap']),
            genai_candidate(['c2', 'fu']),
            genai_candidate(['c3', 'bar']),
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
    json_str = pydantic.TypeAdapter(list[MyData]).dump_json(expected)
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate(json_str)]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertEqual(
        gen_obj(prompt='prompt', decoding_constraint=list[MyData]), expected
    )

  def test_generate_content_decode_str(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('"unicorn"')]
    )
    result = executing.run(
        llm.generate_text(
            prompt='unicorn', decoding_constraint='(poney|unicorn)'
        )
    )
    self.assertEqual(result, 'unicorn')

  def test_generate_in_parallel(self):
    def side_effect(*args, **kwargs):
      del args, kwargs
      time.sleep(0.5)
      return genai_types.GenerateContentResponse(
          candidates=[genai_candidate('text')],
      )

    _ = _get_and_register_backend(disable_caching=True)
    gc_mock = self._mock_genai_client.models.generate_content
    gc_mock.side_effect = side_effect

    with self.subTest('generate_text'):
      t0 = time.time()
      calls = [llm.generate_text(prompt=f'Something {x}') for x in range(10)]
      executing.run(executing.par_iter(iter(calls)))
      t1 = time.time()
      self.assertLess(t1 - t0, 5.0)

    with self.subTest('chat'):
      t0 = time.time()
      msgs = [
          content_lib.Message(
              role=content_lib.PredefinedRole.MODEL, content='Hello user'
          )
      ]
      calls = [llm.chat(msgs) for _ in range(10)]
      executing.run(executing.par_iter(iter(calls)))
      t1 = time.time()
      self.assertLess(t1 - t0, 5.0)

  def test_generate_content_with_cached_chunk_list_new_cache(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    mock_cache = mock.MagicMock()
    mock_cache.name = 'cachedContents/xyz'
    caches_mock.create.return_value = mock_cache

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('prefix'), _Chunk('suffix')],
        cache_chunk_count=1,
        cache_ttl='300s',
    )

    executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('CreatesCache'):
      caches_mock.create.assert_called_once()
      _, kwargs = caches_mock.create.call_args
      self.assertEqual(kwargs['config'].contents[0].parts[0].text, 'prefix')
      self.assertEqual(kwargs['config'].ttl, '300s')

    with self.subTest('UpdatesPromptCacheName'):
      self.assertEqual(prompt.cache_name, 'cachedContents/xyz')

    with self.subTest('CallsGenerateWithCache'):
      gc_mock.assert_called_once()
      _, kwargs = gc_mock.call_args
      self.assertEqual(kwargs['contents'][0].parts[0].text, 'suffix')
      self.assertEqual(kwargs['config'].cached_content, 'cachedContents/xyz')

  def test_generate_content_with_cached_chunk_list_existing_cache(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('prefix'), _Chunk('suffix')],
        cache_chunk_count=1,
        cache_name='cachedContents/abc',
        cache_ttl='300s',
    )

    executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('UpdatesCacheTTL'):
      caches_mock.update.assert_called_once()
      _, kwargs = caches_mock.update.call_args
      self.assertEqual(kwargs['name'], 'cachedContents/abc')
      self.assertEqual(kwargs['config'].ttl, '300s')

    with self.subTest('DoesNotCreateCache'):
      caches_mock.create.assert_not_called()

    with self.subTest('CallsGenerateWithCache'):
      gc_mock.assert_called_once()
      _, kwargs = gc_mock.call_args
      self.assertEqual(kwargs['config'].cached_content, 'cachedContents/abc')

  def test_generate_content_with_cached_chunk_list_no_refresh(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('prefix'), _Chunk('suffix')],
        cache_chunk_count=1,
        cache_name='cachedContents/abc',
        cache_ttl='300s',
        refresh_cache=False,
    )

    executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('DoesNotUpdateOrCreateCache'):
      caches_mock.update.assert_not_called()
      caches_mock.create.assert_not_called()

    with self.subTest('CallsGenerateWithCache'):
      gc_mock.assert_called_once()
      _, kwargs = gc_mock.call_args
      self.assertEqual(kwargs['contents'][0].parts[0].text, 'suffix')
      self.assertEqual(kwargs['config'].cached_content, 'cachedContents/abc')

  def test_generate_content_with_cached_chunk_list_cache_expired(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    caches_mock.update.side_effect = genai_errors.ClientError(
        code=404, response_json={'error': {'message': 'cache abc not found'}}
    )

    mock_cache = mock.MagicMock()
    mock_cache.name = 'cachedContents/new'
    caches_mock.create.return_value = mock_cache

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('prefix'), _Chunk('suffix')],
        cache_chunk_count=1,
        cache_name='cachedContents/abc',
    )

    executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('TriesToUpdate'):
      caches_mock.update.assert_called_once()

    with self.subTest('CreatesNewCache'):
      caches_mock.create.assert_called_once()

    with self.subTest('CallsGenerateWithNewCache'):
      gc_mock.assert_called_once()
      self.assertEqual(
          gc_mock.call_args.kwargs['config'].cached_content,
          'cachedContents/new',
      )

  def test_generate_content_with_cached_chunk_list_splitting(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    mock_cache = mock.MagicMock()
    mock_cache.name = 'cachedContents/split'
    caches_mock.create.return_value = mock_cache

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('p1'), _Chunk('p2'), _Chunk('suffix')],
        cache_chunk_count=2,
    )

    executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('PrefixContainsTwoParts'):
      caches_mock.create.assert_called_once()
      contents = caches_mock.create.call_args.kwargs['config'].contents
      self.assertIn(len(contents), (1, 2))
      if len(contents) == 1:
        self.assertLen(contents[0].parts, 2)
        self.assertEqual(contents[0].parts[0].text, 'p1')
        self.assertEqual(contents[0].parts[1].text, 'p2')
      else:
        self.assertEqual(contents[0].parts[0].text, 'p1')
        self.assertEqual(contents[1].parts[0].text, 'p2')

    with self.subTest('SuffixContainsSuffix'):
      gc_mock.assert_called_once()
      contents = gc_mock.call_args.kwargs['contents']
      self.assertLen(contents, 1)
      self.assertEqual(contents[0].parts[0].text, 'suffix')

  def test_generate_content_with_cached_chunk_list_too_small_fallback(self):
    _ = _get_and_register_backend()
    gc_mock = self._mock_genai_client.models.generate_content
    caches_mock = self._mock_genai_client.caches

    caches_mock.create.side_effect = genai_errors.ClientError(
        code=400,
        response_json={
            'error': {
                'message': (
                    'The cached content is of 60 tokens. The minimum token'
                    ' count to start caching is 1024.'
                )
            }
        },
    )

    gc_mock.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_candidate('fallback_text')]
    )

    prompt = prefix_caching.CachedChunkList(
        chunks=[_Chunk('prefix'), _Chunk('suffix')],
        cache_chunk_count=1,
    )

    res = executing.run(llm.generate_content(prompt=prompt))

    with self.subTest('FallsBackToFullPrompt'):
      gc_mock.assert_called_once()
      _, kwargs = gc_mock.call_args
      self.assertEqual(kwargs['contents'][0].parts[0].text, 'prefix')
      self.assertEqual(kwargs['contents'][1].parts[0].text, 'suffix')
      self.assertIsNone(kwargs['config'].cached_content)

    self.assertEqual(str(res), 'fallback_text')

  def test_generate_content_validation_nested_caching_in_sequences(self):
    _ = _get_and_register_backend()
    cached_chunks = prefix_caching.CachedChunkList(
        chunks=['a'], cache_chunk_count=1
    )
    with self.assertRaisesRegex(
        ValueError, 'CachedChunkList can only be used as the outermost prompt'
    ):
      executing.run(llm.generate_content(prompt=[cached_chunks]))  # type: ignore[wrong-arg-types]

  @parameterized.named_parameters(
      dict(
          testcase_name='none_small_pool',
          threadpool_size=4,
          http_options_in=None,
          expected_http_options_out=genai_types.HttpOptions(
              client_args={
                  'limits': httpx.Limits(
                      max_connections=100, max_keepalive_connections=100
                  )
              },
          ),
      ),
      dict(
          testcase_name='dict_small_pool',
          threadpool_size=4,
          http_options_in={'timeout': 10},
          expected_http_options_out={
              'timeout': 10,
              'client_args': {
                  'limits': httpx.Limits(
                      max_connections=100, max_keepalive_connections=100
                  )
              },
          },
      ),
      dict(
          testcase_name='options_small_pool',
          threadpool_size=4,
          http_options_in=genai_types.HttpOptions(timeout=10),
          expected_http_options_out=genai_types.HttpOptions(
              timeout=10,
              client_args={
                  'limits': httpx.Limits(
                      max_connections=100, max_keepalive_connections=100
                  )
              },
          ),
      ),
      dict(
          testcase_name='none_large_pool',
          threadpool_size=101,
          http_options_in=None,
          expected_http_options_out=genai_types.HttpOptions(
              client_args={
                  'limits': httpx.Limits(
                      max_connections=101, max_keepalive_connections=101
                  )
              },
          ),
      ),
      dict(
          testcase_name='dict_large_pool',
          threadpool_size=101,
          http_options_in={'timeout': 10},
          expected_http_options_out={
              'timeout': 10,
              'client_args': {
                  'limits': httpx.Limits(
                      max_connections=101, max_keepalive_connections=101
                  )
              },
          },
      ),
      dict(
          testcase_name='options_large_pool',
          threadpool_size=101,
          http_options_in=genai_types.HttpOptions(timeout=10),
          expected_http_options_out=genai_types.HttpOptions(
              timeout=10,
              client_args={
                  'limits': httpx.Limits(
                      max_connections=101, max_keepalive_connections=101
                  )
              },
          ),
      ),
  )
  def test_configure_http_limits(
      self, threadpool_size, http_options_in, expected_http_options_out
  ):
    http_options_out = google_genai_api._configure_http_limits(
        threadpool_size, http_options_in
    )
    self.assertEqual(http_options_out, expected_http_options_out)


class EmbedFormattingHelpersTest(parameterized.TestCase):
  """Tests for the embed formatting helper functions."""

  # --- _apply_embed_formatting_v2 ---

  @parameterized.named_parameters(
      dict(
          testcase_name='string_content',
          content='hello',
          kwargs={'task_type': 'RETRIEVAL_QUERY'},
          expected_content='prefix: hello',
          expected_has_task_type=False,
      ),
      dict(
          testcase_name='chunk_list_content',
          content=content_lib.ChunkList(
              chunks=[content_lib.Chunk(content='hello')]
          ),
          kwargs={'task_type': 'CLASSIFICATION'},
          expected_content=content_lib.ChunkList(
              chunks=[
                  content_lib.Chunk(content='prefix: '),
                  content_lib.Chunk(content='hello'),
              ]
          ),
          expected_has_task_type=False,
      ),
      dict(
          testcase_name='no_task_type_in_kwargs',
          content='hello',
          kwargs={'other_key': 'value'},
          expected_content='prefix: hello',
          expected_has_task_type=False,
      ),
  )
  def test_apply_embed_formatting_v2(
      self, content, kwargs, expected_content, expected_has_task_type
  ):
    result_content, result_kwargs = google_genai_api._apply_embed_formatting_v2(
        'prefix: ', content, kwargs
    )
    self.assertEqual(result_content, expected_content)
    self.assertEqual('task_type' in result_kwargs, expected_has_task_type)

  # --- _apply_embed_formatting_v1 ---

  @parameterized.named_parameters(
      dict(
          testcase_name='retrieval_document_with_title_string',
          content='hello',
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='My Title',
          expected_content='hello',
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title='My Title',
      ),
      dict(
          testcase_name='retrieval_document_with_title_chunk_list',
          content=content_lib.ChunkList(
              chunks=[content_lib.Chunk(content='hello')]
          ),
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='My Title',
          expected_content=content_lib.ChunkList(
              chunks=[
                  content_lib.Chunk(content='hello'),
              ]
          ),
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title='My Title',
      ),
      dict(
          testcase_name='retrieval_document_no_title',
          content='hello',
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title=None,
          expected_content='hello',
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title=None,
      ),
      dict(
          testcase_name='title_only_defaults_to_document',
          content='hello',
          task_type=None,
          title='My Title',
          expected_content='hello',
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title='My Title',
      ),
      dict(
          testcase_name='retrieval_query',
          content='hello',
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_QUERY,
          title=None,
          expected_content='hello',
          expected_task_type='RETRIEVAL_QUERY',
          expected_title=None,
      ),
      dict(
          testcase_name='no_task_type_no_title',
          content='hello',
          task_type=None,
          title=None,
          expected_content='hello',
          expected_task_type=None,
          expected_title=None,
      ),
      dict(
          testcase_name='retrieval_document_empty_title_empty_content',
          content='',
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='',
          expected_content='none',
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title=None,
      ),
      dict(
          testcase_name='no_task_type_empty_content',
          content='',
          task_type=None,
          title=None,
          expected_content='none',
          expected_task_type=None,
          expected_title=None,
      ),
      dict(
          testcase_name='retrieval_query_empty_chunk_list',
          content=content_lib.ChunkList(chunks=[]),
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_QUERY,
          title=None,
          expected_content=content_lib.ChunkList(
              chunks=[content_lib.Chunk(content='none')]
          ),
          expected_task_type='RETRIEVAL_QUERY',
          expected_title=None,
      ),
      dict(
          testcase_name='retrieval_document_multi_chunk_text_joined',
          content=content_lib.ChunkList(
              chunks=[
                  content_lib.Chunk(content='hello '),
                  content_lib.Chunk(content='world'),
              ]
          ),
          task_type=llm_lib.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='My Title',
          expected_content=content_lib.ChunkList(
              chunks=[
                  content_lib.Chunk(content='hello world'),
              ]
          ),
          expected_task_type='RETRIEVAL_DOCUMENT',
          expected_title='My Title',
      ),
      dict(
          testcase_name='multi_chunk_empty_text_joined_to_none',
          content=content_lib.ChunkList(
              chunks=[
                  content_lib.Chunk(content=''),
                  content_lib.Chunk(content=''),
              ]
          ),
          task_type=None,
          title=None,
          expected_content=content_lib.ChunkList(
              chunks=[content_lib.Chunk(content='none')]
          ),
          expected_task_type=None,
          expected_title=None,
      ),
  )
  def test_apply_embed_formatting_v1(
      self,
      content,
      task_type,
      title,
      expected_content,
      expected_task_type,
      expected_title,
  ):
    kwargs = {}
    result_content, result_kwargs = google_genai_api._apply_embed_formatting_v1(
        content, task_type, title, kwargs
    )
    self.assertEqual(result_content, expected_content)
    # "task_type" is always set, for caching compatibility.
    self.assertIn('task_type', result_kwargs)
    self.assertEqual(result_kwargs['task_type'], expected_task_type)
    self.assertEqual(result_kwargs.get('title'), expected_title)


def _make_fc_part(name: str, args: dict[str, Any]):
  fc = mock.Mock()
  fc.name = name
  fc.args = args
  return mock.Mock(function_call=fc, thought=None, text=None)


class ExtractToolCallsFromPartsTest(parameterized.TestCase):
  """Tests for _extract_tool_calls_from_parts."""

  @parameterized.named_parameters(
      ('none_parts', None, 0, []),
      ('empty_parts', [], 0, []),
      (
          'function_call_only',
          [_make_fc_part('foo', {'a': 1})],
          1,
          [google_genai_api.FunctionCallDetail(name='foo', args={'a': 1})],
      ),
      (
          'thought_only',
          [mock.Mock(function_call=None, thought=True, text='Thinking...')],
          0,
          [google_genai_api.ThoughtDetail(text='Thinking...')],
      ),
      (
          'text_only',
          [mock.Mock(function_call=None, thought=False, text='Hello')],
          0,
          [],
      ),
      (
          'mixed_parts',
          [
              mock.Mock(function_call=None, thought=True, text='Thinking...'),
              _make_fc_part('bar', {'x': 2}),
              mock.Mock(function_call=None, thought=False, text='Final answer'),
          ],
          1,
          [
              google_genai_api.ThoughtDetail(text='Thinking...'),
              google_genai_api.FunctionCallDetail(name='bar', args={'x': 2}),
          ],
      ),
  )
  def test_extract_tool_calls_from_parts(
      self, parts, expected_count, expected_details
  ):
    count, details = google_genai_api._extract_tool_calls_from_parts(parts)
    self.assertEqual(count, expected_count)
    self.assertEqual(details, expected_details)


class UpdateTraceInfoFromUsageMetadataTest(absltest.TestCase):
  """Tests for _update_trace_info_from_usage_metadata."""

  def _make_usage_metadata(
      self,
      prompt_token_count=10,
      cached_content_token_count=2,
      candidates_token_count=5,
      thoughts_token_count=3,
      tool_use_prompt_token_count=1,
  ):
    return mock.Mock(
        prompt_token_count=prompt_token_count,
        cached_content_token_count=cached_content_token_count,
        candidates_token_count=candidates_token_count,
        thoughts_token_count=thoughts_token_count,
        tool_use_prompt_token_count=tool_use_prompt_token_count,
    )

  def _make_part(
      self,
      has_function_call=False,
      thought=None,
      text=None,
      function_response=None,
  ):
    part = mock.Mock()
    part.thought = thought
    part.text = text
    part.function_response = function_response
    if has_function_call:
      part.function_call = mock.Mock()
      part.function_call.name = 'test_func'
      part.function_call.args = {'arg': 'val'}
    else:
      part.function_call = None
    return part

  def _make_candidate(self, parts):
    candidate = mock.Mock()
    candidate.content = mock.Mock()
    candidate.content.parts = parts
    return candidate

  def _make_response(self, candidates=None, afc_history=None):
    response = mock.Mock()
    response.candidates = candidates or []
    if afc_history is not None:
      response.automatic_function_calling_history = afc_history
    else:
      # Simulate getattr returning None.
      del response.automatic_function_calling_history
    return response

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_basic_token_counts_no_tool_calls(self, mock_update_info):
    usage_metadata = self._make_usage_metadata()
    candidate = self._make_candidate([self._make_part()])
    response = self._make_response(candidates=[candidate])

    google_genai_api._update_trace_info_from_usage_metadata(
        usage_metadata, 'models/gemini-2.0-flash', response
    )

    mock_update_info.assert_called_once_with({
        'llm.model': 'gemini-2.0-flash',
        'llm.tokens.prompt': 10,
        'llm.tokens.cached_content': 2,
        'llm.tokens.candidates': 5,
        'llm.tokens.thoughts': 3,
        'llm.tokens.tool_use_prompt': 1,
        'llm.tools.calls': 0,
    })

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_from_candidates(self, mock_update_info):
    parts = [
        self._make_part(has_function_call=True),
        self._make_part(has_function_call=False),
        self._make_part(has_function_call=True),
    ]
    candidate = self._make_candidate(parts)
    response = self._make_response(candidates=[candidate])

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 2)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
            },
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_from_automatic_function_calling_history(
      self, mock_update_info
  ):
    # When afc_history is present, tool calls should be counted from it
    # instead of from candidates.
    afc_content_1 = mock.Mock()
    afc_content_1.parts = [
        self._make_part(has_function_call=True),
        self._make_part(has_function_call=True),
    ]
    afc_content_2 = mock.Mock()
    afc_content_2.parts = [
        self._make_part(has_function_call=True),
    ]
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[afc_content_1, afc_content_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 3)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
            },
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
            },
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_from_afc_history(
      self, mock_update_info
  ):
    afc_content_1 = mock.Mock()
    afc_content_1.parts = [
        self._make_part(has_function_call=True),
    ]
    afc_content_2 = mock.Mock()
    afc_content_2.parts = [
        self._make_part(
            function_response=genai_types.FunctionResponse(
                name='test_func', response={'result': 'ok'}
            )
        ),
    ]
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[afc_content_1, afc_content_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 1)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'test_func',
                'args': {'arg': 'val'},
                'response': {'result': 'ok'},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_matched_by_id(
      self, mock_update_info
  ):
    part_call_1 = self._make_part(has_function_call=True)
    part_call_1.function_call.id = 'call_1'
    part_call_1.function_call.name = 'search'

    part_call_2 = self._make_part(has_function_call=True)
    part_call_2.function_call.id = 'call_2'
    part_call_2.function_call.name = 'search'

    afc_content_1 = mock.Mock()
    afc_content_1.parts = [part_call_1, part_call_2]

    # Return responses in reverse order to test ID matching.
    resp_2 = genai_types.FunctionResponse(
        id='call_2', name='search', response={'result': 2}
    )
    resp_1 = genai_types.FunctionResponse(
        id='call_1', name='search', response={'result': 1}
    )
    afc_content_2 = mock.Mock()
    afc_content_2.parts = [
        self._make_part(function_response=resp_2),
        self._make_part(function_response=resp_1),
    ]

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[afc_content_1, afc_content_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 2)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'search',
                'id': 'call_1',
                'args': {'arg': 'val'},
                'response': {'result': 1},
            },
            {
                'type': 'function_call',
                'name': 'search',
                'id': 'call_2',
                'args': {'arg': 'val'},
                'response': {'result': 2},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_sequential_name_fallback(
      self, mock_update_info
  ):
    # Turn 1: call search(q1) without ID
    call_1 = self._make_part(has_function_call=True)
    call_1.function_call.name = 'search'
    call_1.function_call.args = {'q': 'q1'}
    turn_1 = mock.Mock(parts=[call_1])

    # Turn 2: response for search(q1) without ID
    resp_1 = genai_types.FunctionResponse(name='search', response={'res': 1})
    turn_2 = mock.Mock(parts=[self._make_part(function_response=resp_1)])

    # Turn 3: call search(q2) without ID
    call_2 = self._make_part(has_function_call=True)
    call_2.function_call.name = 'search'
    call_2.function_call.args = {'q': 'q2'}
    turn_3 = mock.Mock(parts=[call_2])

    # Turn 4: response for search(q2) without ID
    resp_2 = genai_types.FunctionResponse(name='search', response={'res': 2})
    turn_4 = mock.Mock(parts=[self._make_part(function_response=resp_2)])

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1, turn_2, turn_3, turn_4],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 2)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'search',
                'args': {'q': 'q1'},
                'response': {'res': 1},
            },
            {
                'type': 'function_call',
                'name': 'search',
                'args': {'q': 'q2'},
                'response': {'res': 2},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_orphan_safety_fallback(
      self, mock_update_info
  ):
    # Response arrives without any preceding function_call in history
    resp = genai_types.FunctionResponse(
        name='orphan_tool', response={'status': 'orphan'}
    )
    turn_1 = mock.Mock(parts=[self._make_part(function_response=resp)])

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 0)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_response',
                'name': 'orphan_tool',
                'response': {'status': 'orphan'},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_duplicate_or_extra_responses(
      self, mock_update_info
  ):
    # 1 function_call for 'search', but 2 function_responses arrive
    call_part = self._make_part(has_function_call=True)
    call_part.function_call.name = 'search'
    turn_1 = mock.Mock(parts=[call_part])

    resp_1 = genai_types.FunctionResponse(name='search', response={'first': 1})
    resp_2 = genai_types.FunctionResponse(name='search', response={'second': 2})
    turn_2 = mock.Mock(
        parts=[
            self._make_part(function_response=resp_1),
            self._make_part(function_response=resp_2),
        ]
    )

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1, turn_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 1)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'search',
                'args': {'arg': 'val'},
                'response': {'first': 1},
            },
            {
                'type': 'function_response',
                'name': 'search',
                'response': {'second': 2},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_mismatched_id_fallback(
      self, mock_update_info
  ):
    # Function call has id='call_1', response has id='call_999'
    call_part = self._make_part(has_function_call=True)
    call_part.function_call.id = 'call_1'
    call_part.function_call.name = 'search'
    turn_1 = mock.Mock(parts=[call_part])

    resp = genai_types.FunctionResponse(
        id='call_999', name='search', response={'unmatched': True}
    )
    turn_2 = mock.Mock(parts=[self._make_part(function_response=resp)])

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1, turn_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 1)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'search',
                'id': 'call_1',
                'args': {'arg': 'val'},
            },
            {
                'type': 'function_response',
                'name': 'search',
                'response': {'unmatched': True},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_mixed_id_and_no_id(
      self, mock_update_info
  ):
    # Call 1 has an explicit id, Call 2 has no id
    call_1 = self._make_part(has_function_call=True)
    call_1.function_call.id = 'id_1'
    call_1.function_call.name = 'tool_a'

    call_2 = self._make_part(has_function_call=True)
    call_2.function_call.name = 'tool_b'
    turn_1 = mock.Mock(parts=[call_1, call_2])

    resp_1 = genai_types.FunctionResponse(
        id='id_1', name='tool_a', response={'a': 'ok'}
    )
    resp_2 = genai_types.FunctionResponse(name='tool_b', response={'b': 'ok'})
    turn_2 = mock.Mock(
        parts=[
            self._make_part(function_response=resp_1),
            self._make_part(function_response=resp_2),
        ]
    )

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1, turn_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 2)
    self.assertEqual(
        call_args['llm.tools.details'],
        [
            {
                'type': 'function_call',
                'name': 'tool_a',
                'id': 'id_1',
                'args': {'arg': 'val'},
                'response': {'a': 'ok'},
            },
            {
                'type': 'function_call',
                'name': 'tool_b',
                'args': {'arg': 'val'},
                'response': {'b': 'ok'},
            },
        ],
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_tool_calls_with_function_responses_json_serializable(
      self, mock_update_info
  ):
    part_call = self._make_part(has_function_call=True)
    part_call.function_call.id = 'call_1'
    part_call.function_call.name = 'search'
    turn_1 = mock.Mock(parts=[part_call])

    mock_mcp_res = mock.Mock()
    mock_mcp_res.model_dump.return_value = {
        'content': 'hello mcp',
        'isError': False,
    }
    resp = genai_types.FunctionResponse(
        id='call_1', name='search', response={'result': mock_mcp_res}
    )
    turn_2 = mock.Mock(parts=[self._make_part(function_response=resp)])

    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[turn_1, turn_2],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 1)
    expected_details = [
        {
            'type': 'function_call',
            'name': 'search',
            'id': 'call_1',
            'args': {'arg': 'val'},
            'response': {'result': {'content': 'hello mcp', 'isError': False}},
        },
    ]
    self.assertEqual(call_args['llm.tools.details'], expected_details)
    # Ensure the serialized structure is 100% JSON serializable.
    serialized = json.dumps(call_args['llm.tools.details'])
    self.assertIn('hello mcp', serialized)

  def test_attach_function_response_to_tool_details_direct(self):
    # 1. Matching by ID
    call_1 = google_genai_api.FunctionCallDetail(
        name='search', id='c1', args={}
    )
    call_2 = google_genai_api.FunctionCallDetail(
        name='search', id='c2', args={}
    )
    current_calls: list[google_genai_api.ToolCallDetail] = [call_1, call_2]
    resp_1 = genai_types.FunctionResponse(
        id='c2', name='search', response={'val': 2}
    )
    google_genai_api._attach_function_response_to_tool_details(
        resp_1, current_turn_details=current_calls
    )
    self.assertEqual(call_2.response, {'val': 2})
    self.assertIsNone(call_1.response)

    # 2. Sequential fallback when ID is absent
    resp_2 = genai_types.FunctionResponse(name='search', response={'val': 1})
    google_genai_api._attach_function_response_to_tool_details(
        resp_2, current_turn_details=current_calls
    )
    self.assertEqual(call_1.response, {'val': 1})

    # 3. Cross-turn matching from accumulated_details
    fetch_call = google_genai_api.FunctionCallDetail(
        name='fetch', id='f1', args={}
    )
    accumulated: list[google_genai_api.ToolCallDetail] = [fetch_call]
    curr_turn: list[google_genai_api.ToolCallDetail] = []
    resp_3 = genai_types.FunctionResponse(
        id='f1', name='fetch', response={'doc': 'abc'}
    )
    google_genai_api._attach_function_response_to_tool_details(
        resp_3, current_turn_details=curr_turn, accumulated_details=accumulated
    )
    self.assertEqual(fetch_call.response, {'doc': 'abc'})
    self.assertEmpty(curr_turn)

    # 4. Orphan response appends to current_turn
    resp_orphan = genai_types.FunctionResponse(
        name='unknown', response={'err': 'not found'}
    )
    google_genai_api._attach_function_response_to_tool_details(
        resp_orphan, current_turn_details=curr_turn
    )
    self.assertEqual(
        curr_turn,
        [
            google_genai_api.FunctionResponseDetail(
                name='unknown', response={'err': 'not found'}
            )
        ],
    )

  def test_tool_call_detail_structured_dataclasses(self):
    fc = google_genai_api.FunctionCallDetail(
        name='search', args={'q': 'test'}, id='c1', response={'hits': 5}
    )
    self.assertEqual(
        fc.to_dict(),
        {
            'type': 'function_call',
            'name': 'search',
            'args': {'q': 'test'},
            'id': 'c1',
            'response': {'hits': 5},
        },
    )

    fr = google_genai_api.FunctionResponseDetail(
        name='search', response={'hits': 5}
    )
    self.assertEqual(
        fr.to_dict(),
        {
            'type': 'function_response',
            'name': 'search',
            'response': {'hits': 5},
        },
    )

    thought = google_genai_api.ThoughtDetail(text='Thinking...')
    self.assertEqual(
        thought.to_dict(),
        {'type': 'thought', 'text': 'Thinking...'},
    )

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_afc_history_with_no_parts(self, mock_update_info):
    afc_content = mock.Mock()
    afc_content.parts = None
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])],
        afc_history=[afc_content],
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 0)

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_model_name_extraction(self, mock_update_info):
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])]
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(),
        'publishers/google/models/gemini-pro',
        response,
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.model'], 'gemini-pro')

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_model_name_without_slash(self, mock_update_info):
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])]
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'gemini-pro', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.model'], 'gemini-pro')

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_multiple_candidates_with_tool_calls(self, mock_update_info):
    candidate1 = self._make_candidate([
        self._make_part(has_function_call=True),
    ])
    candidate2 = self._make_candidate([
        self._make_part(has_function_call=True),
        self._make_part(has_function_call=True),
    ])
    response = self._make_response(candidates=[candidate1, candidate2])

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(), 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.tools.calls'], 3)

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_none_token_counts_passed_through(self, mock_update_info):
    usage_metadata = self._make_usage_metadata(
        thoughts_token_count=None,
        tool_use_prompt_token_count=None,
    )
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])]
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        usage_metadata, 'models/gemini-2.0-flash', response
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertIsNone(call_args['llm.tokens.thoughts'])
    self.assertIsNone(call_args['llm.tokens.tool_use_prompt'])

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_gcp_project_included(self, mock_update_info):
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])]
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(),
        'gemini-pro',
        response,
        project='my-gcp-project',
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertEqual(call_args['llm.gcp_project'], 'my-gcp-project')

  @mock.patch.object(google_genai_api.tracing, 'update_info')
  def test_gcp_project_omitted_when_none(self, mock_update_info):
    response = self._make_response(
        candidates=[self._make_candidate([self._make_part()])]
    )

    google_genai_api._update_trace_info_from_usage_metadata(
        self._make_usage_metadata(),
        'gemini-pro',
        response,
        project=None,
    )

    call_args = mock_update_info.call_args[0][0]
    self.assertNotIn('llm.gcp_project', call_args)


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


class GenerateWithToolsTest(
    parameterized.TestCase,
    core_test_utils.CounterAssertions,
):
  """Tests the single model call that `Agent` is built on."""

  def setUp(self):
    super().setUp()
    llm.reset_defaults()

    self._mock_genai_client = mock.create_autospec(genai.Client)
    self._mock_genai_client.models.list.return_value = _mock_list_models()
    self.enter_context(
        mock.patch.object(genai, 'Client', return_value=self._mock_genai_client)
    )
    self._reply_with(genai_types.Part(text='text'))

  def _reply_with(self, *parts: genai_types.Part) -> None:
    """Makes the mocked client reply with a single candidate of `parts`."""
    self._mock_genai_client.models.generate_content.return_value = (
        genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(parts=parts, role='model')
                )
            ]
        )
    )

  def _last_call_kwargs(self) -> dict[str, Any]:
    _, kwargs = self._mock_genai_client.models.generate_content.call_args
    return kwargs

  def _ask(self, *messages, **kwargs):
    return executing.run(
        llm.generate_with_tools(
            messages=list(messages), tools=[_WEATHER_SPEC], **kwargs
        )
    )

  def test_lifts_the_requested_call(self):
    _get_and_register_backend()
    self._reply_with(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name='_get_weather', id='call_1', args={'location': 'London'}
            )
        )
    )

    turn = self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER,
            content='What is the weather in London?',
        )
    )

    with self.subTest('returns_a_single_model_turn'):
      self.assertEqual(turn.role, function_calling.ConversationRole.MODEL)
    with self.subTest('lifts_the_call'):
      self.assertLen(turn.calls, 1)
      self.assertEqual(turn.calls[0].function_name, '_get_weather')
      self.assertEqual(dict(turn.calls[0].kwargs), {'location': 'London'})
    with self.subTest('keeps_the_id_so_responses_can_be_matched'):
      self.assertEqual(turn.calls[0].id, 'call_1')

  def test_performs_exactly_one_model_call(self):
    # The model asks for a tool. A backend must not act on that: executing the
    # call and asking again is the agent's job, not the backend's.
    backend = _get_and_register_backend()
    self._reply_with(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name='_get_weather', id='c1', args={'location': 'London'}
            )
        )
    )

    turn = self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER,
            content='What is the weather in London?',
        )
    )

    with self.subTest('calls_the_model_once'):
      self.assertEqual(
          self._mock_genai_client.models.generate_content.call_count, 1
      )
      self.assertCounterEqual(backend._counters, Counter(generate_with_tools=1))
    with self.subTest('hands_the_call_back_unexecuted'):
      self.assertLen(turn.calls, 1)
      self.assertIsNone(turn.content)

  def test_disables_the_sdks_own_function_calling_loop(self):
    # Two loops would be silent and horrible to debug, so the SDK's one is
    # switched off and the only loop is the one in `Agent`.
    _get_and_register_backend()

    self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER, content='Hi.'
        )
    )

    config = self._last_call_kwargs()['config']
    self.assertTrue(config.automatic_function_calling.disable)

  def test_applies_the_backends_generate_with_tools_kwargs(self):
    thinking_config = genai_types.ThinkingConfig(thinking_budget=-1)
    # Backends often share one set of defaults across `chat` and
    # `generate_with_tools`, including an automatic function calling config
    # that only makes sense for `chat`.
    _get_and_register_backend(
        generate_with_tools_kwargs={
            'thinking_config': thinking_config,
            'automatic_function_calling': (
                genai_types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=3
                )
            ),
        }
    )

    self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER, content='Hi.'
        )
    )

    config = self._last_call_kwargs()['config']
    with self.subTest('forwards_the_defaults_to_the_request'):
      self.assertEqual(config.thinking_config, thinking_config)
    with self.subTest('keeps_the_sdks_own_loop_disabled'):
      self.assertEqual(
          config.automatic_function_calling,
          genai_types.AutomaticFunctionCallingConfig(disable=True),
      )

  def test_declares_the_tools_as_json_schema(self):
    _get_and_register_backend()

    self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER, content='Hi.'
        )
    )

    tools = self._last_call_kwargs()['config'].tools
    self.assertLen(tools, 1)
    declarations = tools[0].function_declarations
    self.assertLen(declarations, 1)
    with self.subTest('declares_the_name_and_description'):
      self.assertEqual(declarations[0].name, '_get_weather')
      self.assertStartsWith(declarations[0].description, 'Gets the weather')
    with self.subTest('passes_the_json_schema_through_untouched'):
      self.assertEqual(
          declarations[0].parameters_json_schema,
          dict(_WEATHER_SPEC.parameters or {}),
      )

  def test_advertises_nothing_when_given_no_tools(self):
    _get_and_register_backend()

    executing.run(
        llm.generate_with_tools(
            messages=[
                function_calling.ConversationTurn(
                    role=function_calling.ConversationRole.USER, content='Hi.'
                )
            ],
            tools=[],
        )
    )

    # An empty `Tool` is not the same as no tools at all.
    self.assertIsNone(self._last_call_kwargs()['config'].tools)

  def test_sends_tool_results_back(self):
    _get_and_register_backend()
    self._reply_with(genai_types.Part(text='It is 15 degrees in London.'))

    turn = self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.USER,
            content='What is the weather in London?',
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.MODEL,
            calls=[
                function_calling.FunctionCall(
                    function_name='_get_weather',
                    kwargs={'location': 'London'},
                    id='c1',
                )
            ],
        ),
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.TOOL,
            responses=[
                function_calling.FunctionResponse(
                    function_name='_get_weather',
                    response={'temp': 15},
                    id='c1',
                )
            ],
        ),
    )

    contents = self._last_call_kwargs()['contents']
    with self.subTest('sends_the_earlier_call_as_the_model'):
      self.assertEqual(contents[1].role, 'model')
      self.assertEqual(contents[1].parts[0].function_call.name, '_get_weather')
    with self.subTest('sends_the_response_as_the_user'):
      # The GenAI API has no tool role, so results come back as user content.
      self.assertEqual(contents[2].role, 'user')
      response = contents[2].parts[0].function_response
      self.assertEqual(response.id, 'c1')
      self.assertEqual(response.response, {'temp': 15})
    with self.subTest('returns_the_final_answer'):
      self.assertEqual(turn.content, 'It is 15 degrees in London.')

  def test_reports_tool_errors_to_the_model(self):
    _get_and_register_backend()
    self._reply_with(genai_types.Part(text='Sorry, that failed.'))

    self._ask(
        function_calling.ConversationTurn(
            role=function_calling.ConversationRole.TOOL,
            responses=[
                function_calling.FunctionResponse(
                    function_name='_get_weather',
                    error='Unknown tool: _get_weathr.',
                )
            ],
        )
    )

    contents = self._last_call_kwargs()['contents']
    self.assertEqual(
        contents[0].parts[0].function_response.response,
        {'error': 'Unknown tool: _get_weathr.'},
    )

  def test_round_trips_a_call_signature(self):
    # Gemini refuses a conversation in which an earlier function call comes
    # back without the signature it was issued with:
    # 'Function call is missing a thought_signature in functionCall parts.'
    _get_and_register_backend()
    self._reply_with(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name='_get_weather', id='c1', args={'location': 'London'}
            ),
            thought_signature=b'opaque',
        )
    )

    question = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER,
        content='Weather in London?',
    )
    first = self._ask(question)

    with self.subTest('lifts_the_signature'):
      self.assertEqual(first.calls[0].thought_signature, b'opaque')

    self._ask(question, first)

    resent = self._last_call_kwargs()['contents'][1].parts
    with self.subTest('resends_the_signature_verbatim'):
      self.assertLen(resent, 1)
      self.assertEqual(resent[0].thought_signature, b'opaque')

  def test_round_trips_a_thought_signature(self):
    # A thinking model expects its own reasoning, and the opaque signature
    # attached to it, to come back verbatim on the next turn.
    _get_and_register_backend()
    self._reply_with(
        genai_types.Part(
            text='I should look up the weather.',
            thought=True,
            thought_signature=b'opaque',
        ),
        genai_types.Part(text='Checking.'),
    )

    question = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER,
        content='Weather in London?',
    )
    first = self._ask(question)

    with self.subTest('keeps_the_reasoning_out_of_the_answer'):
      self.assertEqual(first.content, 'Checking.')
      self.assertIsNotNone(first.thought)

    # Feeding the turn straight back is how a caller continues a conversation,
    # so that is what has to preserve the signature.
    self._ask(question, first)

    resent = [
        p for p in self._last_call_kwargs()['contents'][1].parts if p.thought
    ]
    with self.subTest('resends_the_thought'):
      self.assertLen(resent, 1)
      self.assertEqual(resent[0].text, 'I should look up the weather.')
    with self.subTest('resends_the_signature_verbatim'):
      self.assertEqual(resent[0].thought_signature, b'opaque')

  def test_is_cached_across_save_and_load(self):
    backend = _get_and_register_backend()
    self._reply_with(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name='_get_weather', id='c1', args={'location': 'London'}
            ),
            thought_signature=b'opaque',
        )
    )
    question = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.USER,
        content='Weather in London?',
    )
    first = self._ask(question)
    again = self._ask(question)

    generate_content = self._mock_genai_client.models.generate_content
    with self.subTest('calls_the_model_once_for_the_same_request'):
      self.assertEqual(generate_content.call_count, 1)
      self.assertEqual(again, first)

    cache_filename = self.create_tempfile().full_path
    handler = backend.cache
    assert isinstance(handler, caching.SimpleFunctionCache)
    handler.save(cache_filename=cache_filename, overwrite=True)
    fresh_handler = _get_and_register_backend().cache
    assert isinstance(fresh_handler, caching.SimpleFunctionCache)
    fresh_handler.load(cache_filename=cache_filename)
    loaded = self._ask(question)

    with self.subTest('replays_the_reply_from_file'):
      self.assertEqual(generate_content.call_count, 1)
      self.assertEqual(loaded, first)
    with self.subTest('keeps_the_signature_through_the_file'):
      self.assertEqual(loaded.calls[0].thought_signature, b'opaque')


if __name__ == '__main__':
  absltest.main()
