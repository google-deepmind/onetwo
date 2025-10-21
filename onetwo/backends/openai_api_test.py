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

"""Tests for OpenAIAPI engine."""

import collections
from typing import Counter, Final, TypeAlias
# We have to mock openai to make this test hermetic.
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import openai_api
from onetwo.backends import openai_mock
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
import pydantic

import openai

_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_BATCH_SIZE: Final[int] = 4


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
            stop=['\n\n'],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_chat_completions': 1,
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
            stop=['\n\n'],
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
        '_chat_completions': 1,
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
            stop=['\n\n'],
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
            stop=['different'],
            max_tokens=512,
        )
    )
    with self.subTest('returns_correct_result_3'):
      self.assertEqual(res, openai_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        '_chat_completions': 2,
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

  def test_generate_object(self):
    """Verifies that generate_object works and calls the mock parse API."""
    backend = _get_and_register_backend()

    class MyTestSchema(pydantic.BaseModel):
      item_name: str = 'default_item'
      item_id: int = 123
      is_valid: bool = True

    prompt_text = 'Create an object representing a valid item.'

    expected_object = MyTestSchema.model_construct()

    res = executing.run(
        llm.generate_object(  # pytype: disable=wrong-keyword-args
            prompt=prompt_text,
            cls=MyTestSchema,
        )
    )

    with self.subTest('returns_correct_object_type'):
      self.assertIsInstance(res, MyTestSchema)
    with self.subTest('object_matches_mock_construct'):
      self.assertEqual(res.item_name, expected_object.item_name)
      self.assertEqual(res.item_id, expected_object.item_id)
      self.assertEqual(res.is_valid, expected_object.is_valid)

    expected_backend_counters = collections.Counter({
        'generate_object': 1,
        '_chat_completions_parse': 1,
        'generate_object_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_generate_object_legacy_model_raises_error(self):
    """Tests that generate_object raises ValueError for models < gpt-4o-mini."""
    _ = _get_and_register_backend(model_name='gpt-3.5-turbo')

    class MyTestSchema(pydantic.BaseModel):
      test: str

    with self.assertRaisesRegex(
        ValueError, '`gpt-3.5-turbo` does not support structured output'
    ):
      executing.run(
          llm.generate_object(  # pytype: disable=wrong-keyword-args
              prompt='test',
              cls=MyTestSchema,
          )
      )

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

    with self.subTest('cls_is_built_in_type_raises_error'):
      with self.assertRaisesRegex(
          ValueError,
          r'The `cls` argument must be a subclass of `pydantic.BaseModel`\.',
      ):
        executing.run(
            llm.generate_object(  # pytype: disable=wrong-keyword-args
                prompt='another prompt',
                cls=str,
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


if __name__ == '__main__':
  absltest.main()
