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

import collections
import contextlib
from typing import Iterator
from unittest import mock

from absl.testing import absltest
from onetwo.backends import sentencepiece_tokenizer
from onetwo.builtins import llm
from onetwo.core import core_test_utils
from onetwo.core import executing

SentencePieceTokenizer = sentencepiece_tokenizer.SentencePieceTokenizer

_MOCK_TOKENIZED = [1, 2, 3]
_MOCK_DETOKENIZED = 'hello world'


@contextlib.contextmanager
def mock_sentencepiece_tokenizer_backend(
    **kwargs,
) -> Iterator[SentencePieceTokenizer]:
  with mock.patch(
      'onetwo.backends.sentencepiece_tokenizer.SentencePieceTokenizer._load_model'
  ) as mock_model:
    mock_instance = mock.MagicMock()
    mock_model.return_value = mock_instance
    mock_instance.EncodeAsIds.return_value = _MOCK_TOKENIZED
    mock_instance.DecodeIds.return_value = _MOCK_DETOKENIZED

    backend = SentencePieceTokenizer(**kwargs)
    backend.register()
    yield backend


class SentencePieceTokenizerTest(
    absltest.TestCase, core_test_utils.CounterAssertions
):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()
    self._cachefile = self.create_tempfile()

  def test_tokenizer_backend(self):
    with mock_sentencepiece_tokenizer_backend(
        cache_filename=self._cachefile.full_path,
        batch_size=4,
        model_name='ignored',
    ) as backend:
      tokenize_executable = [
          llm.tokenize(f'hello world {idx}') for idx in range(10)  # pytype: disable=wrong-arg-count
      ]
      tokenize_result = executing.run(executing.parallel(*tokenize_executable))

      with self.subTest('returns_correct_tokenization'):
        self.assertEqual(tokenize_result, [_MOCK_TOKENIZED] * 10)

      detokenize_executable = [
          llm.detokenize([1, 2, 3, 4] + [idx]) for idx in range(10)  # pytype: disable=wrong-arg-count
      ]
      detokenize_result = executing.run(
          executing.parallel(*detokenize_executable)
      )

      with self.subTest('returns_correct_detokenization'):
        self.assertEqual(detokenize_result, [_MOCK_DETOKENIZED] * 10)

      expected_backend_counters = collections.Counter(
          {
              'tokenize': 10,
              '_batch_tokenize': 3,
              '_batch_detokenize': 3,
              'detokenize': 10,
          },
      )
      with self.subTest('sends_correct_number_of_api_calls'):
        self.assertCounterEqual(backend._counters, expected_backend_counters)


if __name__ == '__main__':
  absltest.main()
