# Copyright 2024 DeepMind Technologies Limited.
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

"""Tests for VertexAIAPI engine."""

import collections
import string
from typing import Any, Counter, Final, TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import vertexai
from google.cloud.aiplatform_v1beta1.types import (
    prediction_service as gapic_prediction_service_types,
)
from vertexai import generative_models
from vertexai import language_models
from onetwo.backends import vertexai_api
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import sampling


Message: TypeAlias = content_lib.Message
Chunk: TypeAlias = content_lib.Chunk
ChunkList: TypeAlias = content_lib.ChunkList
PredefinedRole: TypeAlias = content_lib.PredefinedRole


_BATCH_SIZE: Final[int] = 4

_MOCK_COUNT_TOKENS_RETURN: Final[int] = 5


def _get_and_register_backend() -> vertexai_api.VertexAIAPI:
  """Get an instance of VertexAIAPI and register its methods."""
  backend = vertexai_api.VertexAIAPI(
      project='some-project', location='some-location', batch_size=_BATCH_SIZE
  )
  backend.register()
  return backend


@executing.make_executable
async def check_length_and_generate(
    prompt_text: str,
    max_token_count: int = _MOCK_COUNT_TOKENS_RETURN + 1,
    **other_args,
) -> Any:
  token_count = await llm.count_tokens(content=prompt_text)
  if token_count > max_token_count:
    raise ValueError('Prompt has too many tokens!')
  res = await llm.generate_text(prompt=prompt_text, **other_args)
  return res

mock_generate_model = mock.MagicMock(spec=generative_models.GenerativeModel)
mock_embed_model = mock.MagicMock(
    spec=vertexai.language_models.TextEmbeddingModel
)


# Patch the necessary modules and classes.
@mock.patch.object(vertexai, 'init', return_value=None, autospec=True)
@mock.patch.object(
    generative_models,
    'GenerativeModel',
    return_value=mock_generate_model,
    autospec=True,
)
@mock.patch.object(
    language_models,
    'TextEmbeddingModel',
    return_value=mock_embed_model,
    autospec=True,
)
class VertexAIAPITest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):
  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

    def generate_content_mock(prompt, generation_config):
      candidate_count = getattr(generation_config, 'candidate_count', 1)

      if prompt.startswith('raise_exception'):
        raise ValueError(
            'VertexAI.Model.generate_content raised err:\nFake error\n'
            'for request:\nFake request.'
        )
      letters = [string.ascii_lowercase[i % 26] for i in range(candidate_count)]
      candidates = [
          {'content': {'parts': [{'text': letter * 10}]}} for letter in letters
      ]
      return generative_models.GenerationResponse.from_dict(
          {'candidates': candidates}
      )

    mock_generate_model.generate_content.side_effect = generate_content_mock
    mock_generate_model.count_tokens.return_value = (
        gapic_prediction_service_types.CountTokensResponse(
            total_tokens=_MOCK_COUNT_TOKENS_RETURN
        )
    )
    mock_embed_model.get_embeddings.return_value = [
        language_models.TextEmbedding(values=[0.0])
    ]

  def test_generate_and_count_tokens(
      self,
      *args,
      **kwargs,
  ):  # pylint: disable=g-doc-args
    """Verifies that repeated calls to the cached method behave as expected.

    When performing multiple calls to the generate and count_tokens methods of
    the same VertexAIAPI instance caching and batching should behave as
    expected.
    """
    backend = _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        check_length_and_generate(
            prompt_text=prompt_text,
            stop=['\n\n'],
            max_tokens=512,
        )
    )
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result_1'):
      self.assertEqual(res, 'a' * 10)
    expected_backend_counters = collections.Counter({
        'generate_text': 1,
        'generate_text_batches': 1,
        'count_tokens': 1,
        'count_tokens_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls_1'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    # One add and one miss for generate and count_tokens each.
    expected_cache_counters = Counter(add_new=2, get_miss=2)
    with self.subTest('cache_behaves_as_expected_1'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # Same query to see if it has been cached.
    res = executing.run(
        check_length_and_generate(
            prompt_text=prompt_text,
            stop=['\n\n'],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result_2'):
      self.assertEqual(res, 'a' * 10)
    with self.subTest('sends_correct_number_of_api_calls_2'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 2  # Generate and count_tokens.
    with self.subTest('cache_behaves_as_expected_2'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # Same query but different parameters, run generate request again.
    res = executing.run(
        check_length_and_generate(
            prompt_text=prompt_text,
            stop=['different'],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result_3'):
      self.assertEqual(res, 'a' * 10)
    # We only send generate call, not the count_tokens call.
    expected_backend_counters = collections.Counter({
        'generate_text': 2,
        'generate_text_batches': 2,
        'count_tokens': 1,
        'count_tokens_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls_3'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1  # Looked up count_tokens.
    expected_cache_counters['get_miss'] += 1  # Could not find generate.
    expected_cache_counters['add_new'] += 1  # Cached generate.
    with self.subTest('cache_behaves_as_expected_3'):
      self.assertCounterEqual(
          handler._cache_data.counters, expected_cache_counters
      )

  def test_repeat_generate_and_count_tokens(self, *args, **kwargs):
    backend = _get_and_register_backend()
    prompt_text = 'Something'
    exe = executing.par_iter(
        sampling.repeat(
            executable=check_length_and_generate(
                prompt_text=prompt_text,
                temperature=0.5,
                stop=['.'],
            ),
            num_repeats=5,
        )
    )
    res = executing.run(exe)
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, 5 * ['a' * 10])
    expected_backend_counters = collections.Counter({
        'generate_text': 5,
        'generate_text_batches': 2,  # Two batches: 4 + 1.
        'count_tokens': 1,  # Because all the prompts are the same.
        'count_tokens_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(
          backend._counters,
          expected_backend_counters,
      )
    expected_cache_counters = Counter(
        # First generate and count_tokens requests cache a new key.
        add_new=2,
        # Four remaining requests to generate add new samples.
        add_new_sample=4,
        # Five for generate and one for count.
        get_miss=6,
        # Four for count.
        get_hit=4,
    )
    with self.subTest('cache_behaves_as_expected'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )
    # One more run of repeat to hit the `get_hit_miss_sample` counter.
    exe = executing.par_iter(
        sampling.repeat(
            executable=check_length_and_generate(
                prompt_text=prompt_text,
                temperature=0.5,
                stop=['.'],
            ),
            num_repeats=6,  # 6th value will hit get_hit_miss_sample counter.
        )
    )
    _ = executing.run(exe)
    expected_cache_counters = Counter(
        # Same.
        add_new=2,
        # One new sample from `generate` compared to the previous call.
        add_new_sample=5,
        # Same.
        get_miss=6,
        # 6 new for `count`, 5 new for `generate`.
        get_hit=15,
        # 6th sample for `generate` is missing (but its key is in cache).
        get_hit_miss_sample=1,
    )
    with self.subTest('cache_behaves_as_expected'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

  def test_identical_batched_generate_and_count_tokens(self, *args, **kwargs):

    backend = _get_and_register_backend()
    exe = executing.par_iter([
        check_length_and_generate(prompt_text='Something'),
        check_length_and_generate(prompt_text='Something'),
        check_length_and_generate(prompt_text='Something'),
        check_length_and_generate(prompt_text='Something'),
        check_length_and_generate(prompt_text='Something'),
    ])
    res = executing.run(exe)
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, 5 * ['a' * 10])
    expected_backend_counters = collections.Counter({
        'generate_text': 1,
        'generate_text_batches': 1,
        'count_tokens': 1,
        'count_tokens_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(
          backend._counters,
          expected_backend_counters,
      )
    expected_cache_counters = Counter(add_new=2, get_miss=2, get_hit=8)
    with self.subTest('cache_behaves_as_expected'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

  def test_generate_raises_exception(self, *args, **kwargs):
    _ = _get_and_register_backend()
    exe = check_length_and_generate(
        prompt_text='Something',
        max_token_count=_MOCK_COUNT_TOKENS_RETURN - 1,
    )
    with self.subTest('executable_raises_value_error'):
      with self.assertRaisesRegex(ValueError, 'Prompt has too many tokens*'):
        _ = executing.run(exe)
    exe = check_length_and_generate(
        prompt_text='raise_exception',
    )
    with self.subTest('generate_raises_value_error'):
      with self.assertRaisesRegex(
          ValueError, 'VertexAI.Model.generate_content raised er*'
      ):
        _ = executing.run(exe)
    exe = executing.par_iter([
        check_length_and_generate(prompt_text='a'),
        check_length_and_generate(prompt_text='raise_exception b'),
        check_length_and_generate(
            prompt_text='c',
        ),
    ])
    with self.subTest('batched_generate_raises_value_error'):
      with self.assertRaisesRegex(
          ValueError, 'VertexAI.Model.generate_content raised er*'
      ):
        _ = executing.run(exe)

  def test_generate_texts(self, *args, **kwargs):
    _ = _get_and_register_backend()
    prompt = 'Something'
    results = executing.run(llm.generate_texts(prompt=prompt, samples=3))
    self.assertLen(results, 3)
    self.assertListEqual(list(results), ['a' * 10, 'a' * 10, 'a' * 10])

  def test_chat(self, *args, **kwargs):
    _ = _get_and_register_backend()
    msg = Message(role=PredefinedRole.USER, content='Hello')
    with self.assertRaises(NotImplementedError):
      executing.run(llm.chat(messages=[msg]))

  def test_truncation(self, *args, **kwargs):
    _ = _get_and_register_backend()
    result = executing.run(
        llm.generate_text(
            prompt='something', include_details=True, max_tokens=2
        )
    )
    self.assertEqual(result, ('a' * 6, {'text': 'a' * 10}))

  def test_embed_text(self, *args, **kwargs):
    _ = _get_and_register_backend()
    content = 'something'
    result = executing.run(llm.embed(content=content))
    self.assertEqual(result, [0.0])

  def test_embed_chunk_list(self, *args, **kwargs):
    _ = _get_and_register_backend()
    content = content_lib.ChunkList(['something'])
    result = executing.run(llm.embed(content=content))
    self.assertEqual(result, [0.0])


if __name__ == '__main__':
  absltest.main()
