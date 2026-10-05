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

"""Tests for OllamaAPI engine."""

import collections
import io
import math
from typing import Counter, Final, TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import ollama_api
from onetwo.backends import ollama_mock
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
import PIL.Image

_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_BATCH_SIZE: Final[int] = 4

_DUMMY_PIL_IMAGE: Final[PIL.Image.Image] = PIL.Image.new(
    'RGB', (1, 1), color='red'
)
_DUMMY_IMAGE_BUF = io.BytesIO()
_DUMMY_PIL_IMAGE.save(_DUMMY_IMAGE_BUF, format='JPEG')
_DUMMY_IMAGE_JPEG_BYTES: Final[bytes] = _DUMMY_IMAGE_BUF.getvalue()


def _get_and_register_backend(**kwargs) -> ollama_api.OllamaAPI:
  """Get an instance of OllamaAPI and register its methods."""
  backend = ollama_api.OllamaAPI(
      host='http://localhost:11434', batch_size=_BATCH_SIZE, **kwargs
  )
  backend._client = ollama_mock.Client(host=backend.host)  # pyrefly: ignore[bad-assignment]
  backend.register()
  return backend


class OllamaAPITest(parameterized.TestCase, core_test_utils.CounterAssertions):

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
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, ollama_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_chat_create': 1,
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
        )
    )
    # pytype hint.
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    with self.subTest('returns_correct_result_1'):
      self.assertEqual(res, ollama_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'generate_text': 1,
        '_generate_text_with_details_batches': 1,
        '_chat_create': 1,
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
        )
    )
    with self.subTest('returns_correct_result_2'):
      self.assertEqual(res, ollama_mock._DEFAULT_REPLY)
    with self.subTest('sends_correct_number_of_api_calls_2'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1
    with self.subTest('cache_behaves_as_expected_2'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

  def test_generate_with_details(self):
    """Verifies that generate_text with include_details returns score."""
    _get_and_register_backend()
    prompt_text = 'Something'
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            include_details=True,
        )
    )
    self.assertIsInstance(res, tuple)
    if isinstance(res, tuple):
      with self.subTest('returns_correct_result'):
        self.assertEqual(res[0], ollama_mock._DEFAULT_REPLY)
      with self.subTest('returns_correct_score'):
        # Ollama does not provide logprobs, so score should be 0.0.
        self.assertEqual(res[1]['score'], 0.0)

  def test_generate_texts(self):
    """Verifies that generate_texts returns multiple samples."""
    _get_and_register_backend()
    res = executing.run(
        llm.generate_texts(
            prompt='Hello',
            samples=3,
        )
    )
    self.assertLen(res, 3)
    for sample in res:
      self.assertEqual(sample, ollama_mock._DEFAULT_REPLY)

  @parameterized.named_parameters(
      dict(
          testcase_name='string_prompt',
          prompt='Hello world',
      ),
      dict(
          testcase_name='chunklist_prompt',
          prompt=content_lib.ChunkList([
              content_lib.Chunk('Hello '),
              content_lib.Chunk('world'),
          ]),
      ),
  )
  def test_generate_with_different_prompts(self, prompt):
    """Verifies that generate_text works with different prompt types."""
    _get_and_register_backend()
    res = executing.run(
        llm.generate_text(
            prompt=prompt,
        )
    )
    self.assertEqual(res, ollama_mock._DEFAULT_REPLY)

  def test_chat_with_system_message(self):
    """Verifies that system messages are passed inline to Ollama."""
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
        )
    )
    with self.subTest('returns_correct_result'):
      self.assertEqual(res, ollama_mock._DEFAULT_REPLY)
    expected_backend_counters = collections.Counter({
        'chat': 1,
        '_chat_create': 1,
        'chat_via_api_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_generate_text_with_thinking(self):
    """Verifies that generate_text returns thoughts if include_details is True."""
    backend = _get_and_register_backend()
    backend._client._thinking = 'Thinking process'  # pyrefly: ignore[missing-attribute]
    backend._client._reply = 'Result text'  # pyrefly: ignore[missing-attribute]

    prompt_text = 'Something'
    res = executing.run(
        llm.generate_text(
            prompt=prompt_text,
            include_details=True,
        )
    )
    self.assertIsInstance(res, tuple)
    if isinstance(res, tuple):
      with self.subTest('returns_correct_result'):
        self.assertEqual(res[0], 'Result text')
      with self.subTest('returns_correct_thoughts'):
        self.assertEqual(
            res[1]['thoughts'],
            content_lib.ChunkList([content_lib.Chunk('Thinking process')]),
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='plain_text_user',
          messages=[
              _Message(role=_PredefinedRole.USER, content='Hello'),
          ],
          expected=[
              {'role': 'user', 'content': 'Hello'},
          ],
      ),
      dict(
          testcase_name='plain_text_model_and_system',
          messages=[
              _Message(role=_PredefinedRole.SYSTEM, content='System prompt'),
              _Message(role=_PredefinedRole.MODEL, content='Assistant reply'),
              _Message(role='custom_role', content='Custom message'),
          ],
          expected=[
              {'role': 'system', 'content': 'System prompt'},
              {'role': 'assistant', 'content': 'Assistant reply'},
              {'role': 'custom_role', 'content': 'Custom message'},
          ],
      ),
      dict(
          testcase_name='chunklist_text_only',
          messages=[
              _Message(
                  role=_PredefinedRole.USER,
                  content=content_lib.ChunkList([
                      content_lib.Chunk('Hello '),
                      content_lib.Chunk('world!'),
                  ]),
              ),
          ],
          expected=[
              {'role': 'user', 'content': 'Hello world!'},
          ],
      ),
      dict(
          testcase_name='chunklist_with_image_bytes',
          messages=[
              _Message(
                  role=_PredefinedRole.USER,
                  content=content_lib.ChunkList([
                      content_lib.Chunk('Look at this:'),
                      content_lib.Chunk(
                          b'dummy_image_data', content_type='image/png'
                      ),
                  ]),
              ),
          ],
          expected=[
              {
                  'role': 'user',
                  'content': 'Look at this:',
                  'images': [b'dummy_image_data'],
              },
          ],
      ),
      dict(
          testcase_name='chunklist_with_pil_image',
          messages=[
              _Message(
                  role=_PredefinedRole.USER,
                  content=content_lib.ChunkList([
                      content_lib.Chunk('Here is an image:'),
                      content_lib.Chunk(_DUMMY_PIL_IMAGE),
                  ]),
              ),
          ],
          expected=[
              {
                  'role': 'user',
                  'content': 'Here is an image:',
                  'images': [_DUMMY_IMAGE_JPEG_BYTES],
              },
          ],
      ),
      dict(
          testcase_name='chunklist_with_non_image_bytes',
          messages=[
              _Message(
                  role=_PredefinedRole.USER,
                  content=content_lib.ChunkList([
                      content_lib.Chunk('Some PDF data:'),
                      content_lib.Chunk(
                          b'pdf_data', content_type='application/pdf'
                      ),
                  ]),
              ),
          ],
          expected=[
              {
                  'role': 'user',
                  'content': 'Some PDF data:',
              },
          ],
      ),
  )
  def test_convert_messages_to_ollama_format(self, messages, expected):
    res = ollama_api._convert_messages_to_ollama_format(messages)
    self.assertEqual(res, expected)

  def test_embed(self):
    backend = _get_and_register_backend()
    res = executing.run(
        llm.embed(
            content='Some text',
            normalize=False,
        )
    )
    with self.subTest('returns_correct_result'):
      # ollama_mock.Client defaults to _embedding_dim=8, and returns
      # [0.1, 0.2, ...]
      self.assertSequenceAlmostEqual(
          res, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
      )
    expected_backend_counters = collections.Counter({
        'embed': 1,
        '_cached_embed_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_embed_with_chunklist(self):
    _ = _get_and_register_backend()
    # Content has text and non-text (which should trigger warning and be
    # skipped).
    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk('Hello '),
            content_lib.Chunk(b'some bytes', content_type='image/png'),
            content_lib.Chunk('world'),
        ]
    )
    res = executing.run(
        llm.embed(
            content=content,
        )
    )
    # The non-text chunk is skipped, text chunks concatenated to "Hello world".
    # Since mock returns deterministic values based on dimensionality, it still
    # returns same vector length.
    self.assertLen(res, 8)

  def test_embed_caching(self):
    backend = _get_and_register_backend()
    prompt_text = 'Some text'
    _ = executing.run(
        llm.embed(
            content=prompt_text,
        )
    )
    handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
    expected_backend_counters = collections.Counter({
        'embed': 1,
        '_cached_embed_batches': 1,
    })
    with self.subTest('sends_correct_number_of_api_calls_1'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters = Counter(add_new=1, get_miss=1)
    with self.subTest('cache_behaves_as_expected_1'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

    # Same query to see if it has been cached.
    _ = executing.run(
        llm.embed(
            content=prompt_text,
        )
    )
    with self.subTest('sends_correct_number_of_api_calls_2'):
      self.assertCounterEqual(backend._counters, expected_backend_counters)
    expected_cache_counters['get_hit'] += 1
    with self.subTest('cache_behaves_as_expected_2'):
      self.assertCounterEqual(
          handler._cache_data.counters,
          expected_cache_counters,
      )

  def test_embed_with_dimensions(self):
    _ = _get_and_register_backend()
    res = executing.run(
        llm.embed(
            content='Some text',
            output_dimensionality=5,
            normalize=False,
        )
    )
    self.assertLen(res, 5)
    self.assertSequenceAlmostEqual(res, [0.1, 0.2, 0.3, 0.4, 0.5])

  def test_embed_normalize(self):
    _ = _get_and_register_backend()
    res = executing.run(
        llm.embed(
            content='Some text',
            normalize=True,
        )
    )
    # L2 norm should be 1.0.
    norm = math.sqrt(sum(x * x for x in res))
    self.assertAlmostEqual(norm, 1.0, places=6)


if __name__ == '__main__':
  absltest.main()
