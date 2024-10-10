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

"""Tests for GeminiAPI engine."""

import collections
import string
from typing import Any, Counter, Final, TypeAlias
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import generativeai
from google.ai import generativelanguage as glm
from google.generativeai import client
from google.generativeai.types import content_types
from google.generativeai.types import model_types
from onetwo.backends import gemini_api
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import sampling


_Message: TypeAlias = content_lib.Message
_ChunkList: TypeAlias = content_lib.ChunkList
_Chunk: TypeAlias = content_lib.Chunk
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_BATCH_SIZE: Final[int] = 4

_MOCK_COUNT_TOKENS_RETURN: Final[int] = 5


def _mock_configure(api_key: str) -> None:
  del api_key  # Not used.
  pass


def _mock_list_models() -> model_types.ModelsIterable:
  fake_base_params = {
      'base_model_id': '',
      'version': '',
      'display_name': '',
      'description': '',
      'input_token_limit': 1,
      'output_token_limit': 1,
  }
  return [
      model_types.Model(
          name=gemini_api.DEFAULT_GENERATE_MODEL,
          supported_generation_methods=['generateContent', 'countTokens'],
          **fake_base_params,
      ),
      model_types.Model(
          name=gemini_api.DEFAULT_MULTIMODAL_MODEL,
          supported_generation_methods=['generateContent', 'countTokens'],
          **fake_base_params,
      ),
      model_types.Model(
          name=gemini_api.DEFAULT_EMBED_MODEL,
          supported_generation_methods=['embedContent'],
          **fake_base_params,
      ),
  ]


def _get_and_register_backend() -> gemini_api.GeminiAPI:
  """Get an instance of GeminiAPI and register its methods."""
  backend = gemini_api.GeminiAPI(api_key='some_key', batch_size=_BATCH_SIZE)
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


class GeminiAPITest(parameterized.TestCase, core_test_utils.CounterAssertions):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

    # Mock configure.
    self.mock_configure = self.enter_context(
        mock.patch.object(
            generativeai,
            'configure',
            autospec=True,
            side_effect=_mock_configure,
        )
    )
    # Mock list_models.
    self.mock_list_models = self.enter_context(
        mock.patch.object(
            generativeai,
            'list_models',
            autospec=True,
            side_effect=_mock_list_models,
        )
    )

    self.client = unittest.mock.MagicMock()
    client._client_manager.clients['generative'] = self.client

    self.observed_generate_content_requests = []
    self.observed_count_tokens_requests = []

    self.generate_content_responses = []

    def add_client_method(f):
      name = f.__name__
      setattr(self.client, name, f)
      return f

    @add_client_method
    def generate_content(  # pylint: disable=unused-variable
        request: glm.GenerateContentRequest,
        **kwargs,
    ) -> glm.GenerateContentResponse:
      """Return contents of generate_content_responses or fake replies."""
      del kwargs  # Not used.
      self.observed_generate_content_requests.append(request)
      self.assertIsInstance(request, glm.GenerateContentRequest)
      prompt = request.contents
      candidate_count = request.generation_config.candidate_count
      if self.generate_content_responses:
        # Return specified replies.
        if len(self.generate_content_responses) < candidate_count:
          raise ValueError('Not enough replies in generate_content_responses.')
        replies = self.generate_content_responses[:candidate_count]
        self.generate_content_responses.clear()
      else:
        # Return fake replies consisting of 10 letters each.
        if prompt[0].parts[0].text.startswith('raise_exception'):
          raise ValueError(
              'GenAI.Model.generate_content raised err:\nFake error\n'
              'for request:\nFake request.'
          )
        letters = [
            string.ascii_lowercase[i % 26] for i in range(candidate_count)
        ]
        replies = [letter * 10 for letter in letters]
      candidates = [
          {'content': {'parts': [{'text': reply}]}} for reply in replies
      ]
      return glm.GenerateContentResponse({'candidates': candidates})

    @add_client_method
    def count_tokens(  # pylint: disable=unused-variable
        request: glm.CountTokensRequest,
        **kwargs,
    ) -> glm.CountTokensResponse:
      del kwargs  # Not used.
      self.observed_count_tokens_requests.append(request)
      self.assertIsInstance(request, glm.CountTokensRequest)
      return glm.CountTokensResponse(total_tokens=_MOCK_COUNT_TOKENS_RETURN)

  def test_generate_and_count_tokens(self):
    """Verifies that repeated calls to the cached method behave as expected.

    When performing multiple calls to the generate and count_tokens methods of
    the same GeminiAPI instance caching and batching should behave as
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

  def test_repeat_generate_and_count_tokens(self):
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

  def test_batched_generate_and_count_tokens(self):
    backend = _get_and_register_backend()
    exe = executing.par_iter([
        check_length_and_generate(prompt_text='Something a'),
        check_length_and_generate(prompt_text='Something b'),
        check_length_and_generate(prompt_text='Something c'),
        check_length_and_generate(prompt_text='Something d'),
        check_length_and_generate(prompt_text='Something f'),
    ])
    res = executing.run(exe)
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, 5 * ['a' * 10])
    expected_backend_counters = collections.Counter({
        'generate_text': 5,
        'generate_text_batches': 2,  # Two batches: 4 + 1.
        'count_tokens': 5,
        'count_tokens_batches': 2,  # Two batches: 4 + 1.
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(
          backend._counters,
          expected_backend_counters,
      )
    expected_cache_counters = Counter(add_new=10, get_miss=10)
    with self.subTest('cache_behaves_as_expected'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

  def test_identical_batched_generate_and_count_tokens(self):

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

  def test_generate_raises_exception(self):
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
          ValueError, 'GeminiAPI.generate_content raised er*'
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
          ValueError, 'GeminiAPI.generate_content raised er*'
      ):
        _ = executing.run(exe)

  def test_generate_texts(self):
    _ = _get_and_register_backend()
    prompt = 'Something'
    results = executing.run(llm.generate_texts(prompt=prompt, samples=3))
    self.assertLen(results, 3)
    self.assertListEqual(list(results), ['a' * 10, 'a' * 10, 'a' * 10])

  def test_chat(self):
    backend = _get_and_register_backend()
    msg_user = _Message(role=_PredefinedRole.USER, content='Hello model')
    msg_model = _Message(role=_PredefinedRole.MODEL, content='Hello user')
    result = executing.run(llm.chat(messages=[msg_user]))

    with self.subTest('single_msg_returns_correct_result'):
      self.assertEqual(result, 'a' * 10)

    expected_backend_counters = collections.Counter({
        'chat': 1,
        'chat_via_api_batches': 1,
    })

    with self.subTest('single_msg_sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

    result = executing.run(llm.chat(messages=[msg_model, msg_user]))

    with self.subTest('multiple_msg_returns_correct_result'):
      self.assertEqual(result, 'a' * 10)

    expected_backend_counters = collections.Counter({
        'chat': 2,
        'chat_via_api_batches': 2,
    })

    with self.subTest('multiple_msg_sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_truncation(self):
    _ = _get_and_register_backend()
    result = executing.run(
        llm.generate_text(
            prompt='something', include_details=True, max_tokens=2
        )
    )
    self.assertEqual(result, ('a' * 6, {'text': 'a' * 10}))

  @parameterized.named_parameters(
      (
          'no_healing',
          'prompt',
          'reply',
          llm.TokenHealingOption.NONE,
          'prompt',
          'reply',
      ),
      (
          'no_healing_ws',
          'prompt ',
          'reply',
          llm.TokenHealingOption.NONE,
          'prompt ',
          'reply',
      ),
      (
          'space_healing_prompt_wo_ws',
          'prompt',
          'reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          'reply',
      ),
      (
          'space_healing_prompt_wo_ws_reply_with_leading_ws',
          'prompt',
          '  reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          '  reply',
      ),
      (
          'space_healing_prompt_chunklist_wo_ws_reply_with_leading_ws',
          _ChunkList(chunks=['prompt']),
          '  reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          '  reply',
      ),
      (
          'space_healing_prompt_w_ws_reply_wo_ws',
          'prompt  ',
          'reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          'reply',
      ),
      (
          'space_healing_prompt_w_ws_reply_w_ws',
          'prompt  ',
          '   reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          'reply',
      ),
      (
          'space_healing_prompt_chunklist_w_ws_reply_wo_ws',
          _ChunkList(chunks=['prompt  ']),
          'reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          'reply',
      ),
      (
          'space_healing_prompt_chunklist_w_ws_reply_w_ws',
          _ChunkList(chunks=['prompt  ']),
          '   reply',
          llm.TokenHealingOption.SPACE_HEALING,
          'prompt',
          'reply',
      ),
  )
  def test_healing_with_generate_text_and_chat(
      self, prompt, reply, healing_option, exp_healed_prompt, exp_healed_reply
  ):
    """Verifies that healing works as expected for generate_text."""
    _ = _get_and_register_backend()

    # Test llm.generate_text.
    executable = llm.generate_text(
        prompt=prompt,
        healing_option=healing_option,
    )
    self.generate_content_responses.append(reply)
    result = executing.run(executable)

    with self.subTest('generate_text_prompt_healed_correctly'):
      prompt_sent = (
          self.observed_generate_content_requests[-1].contents[0].parts[0].text
      )
      self.assertEqual(prompt_sent, exp_healed_prompt)
    with self.subTest('generate_text_reply_healed_correctly'):
      self.assertEqual(result, exp_healed_reply)

    # Test llm.chat.

    # Default `llm.chat` for gemini api uses native `chat_via_api`, which in
    # turn calls `generate_content` under the hood.

    msg_user = _Message(role=_PredefinedRole.USER, content=prompt)
    msg_model = _Message(role=_PredefinedRole.MODEL, content='I am model!')
    executable = llm.chat(
        messages=[msg_model, msg_user],
        healing_option=healing_option,
    )
    self.generate_content_responses.append(reply)
    result = executing.run(executable)

    with self.subTest('chat_prompt_healed_correctly'):
      # We have 2 messages (model, then user) and they appear as
      # [content, content] in the request. We want to see the content of the
      # last (i.e., user) message, which is why we use `contents[-1]`.
      prompt_sent = (
          self.observed_generate_content_requests[-1].contents[-1].parts[0].text
      )
      self.assertEqual(prompt_sent, exp_healed_prompt)
    with self.subTest('chat_reply_healed_correctly'):
      self.assertEqual(result, exp_healed_reply)

  @parameterized.named_parameters(
      (
          'str_prompt',
          'this is a prompt',
          'this is a prompt',
      ),
      (
          'chunk_str',
          _ChunkList(chunks=['this is a prompt']),
          ['this is a prompt'],
      ),
      (
          'chunk_image_jpeg',
          _ChunkList(
              chunks=[
                  _Chunk(content=b'this is an image', content_type='image/jpeg')
              ]
          ),
          [
              content_types.BlobDict(
                  data=b'this is an image', mime_type='image/jpeg'
              )
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
              content_types.BlobDict(
                  data=b'this is a video', mime_type='video/mp4'
              )
          ],
      ),
  )
  def test_convert_chunk_list_to_contents_type(self, prompt, exp_contents_type):
    self.assertEqual(
        gemini_api._convert_chunk_list_to_contents_type(prompt),
        exp_contents_type,
    )


if __name__ == '__main__':
  absltest.main()
