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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import word_tokenizer
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing

WordTokenizer = word_tokenizer.WordTokenizer


def _expected_word_id(word: str, pad_width: int = 7) -> int:
  return int(''.join(f'{ord(c):0{pad_width}d}' for c in word))


class WordTokenizerTest(
    parameterized.TestCase, core_test_utils.CounterAssertions
):

  def setUp(self):
    super().setUp()
    llm.reset_defaults()
    self._cachefile = self.create_tempfile()

  @parameterized.named_parameters(
      (
          'simple_sentence',
          'hello world',
          'hello world',
      ),
      (
          'multiple_spaces',
          '  hello   world  how   are you ',
          'hello world how are you',
      ),
      (
          'single_word',
          'hello',
          'hello',
      ),
      (
          'empty_string',
          '',
          '',
      ),
      (
          'whitespace_only',
          '   \n\t  ',
          '',
      ),
      (
          'tabs_and_newlines',
          'hello\nworld\ttest',
          'hello world test',
      ),
      (
          'repeated_words',
          'hello world hello',
          'hello world hello',
      ),
      (
          'unicode_and_emojis',
          'café 🚀 你好',
          'café 🚀 你好',
      ),
  )
  def test_tokenize_and_detokenize(self, text: str, expected_detokenized: str):
    backend = WordTokenizer()
    backend.register()

    words = text.split()
    expected_tokens = [_expected_word_id(w) for w in words]

    # Direct synchronous backend call (zero async overhead)
    sync_tokens = backend.tokenize(text)
    self.assertEqual(sync_tokens, expected_tokens)
    self.assertEqual(backend.detokenize(sync_tokens), expected_detokenized)

    # Builtin async call
    tokens = executing.run(llm.tokenize(text))
    self.assertEqual(tokens, expected_tokens)

    # Re-tokenizing preserves token IDs (deterministic)
    tokens_again = executing.run(llm.tokenize(text))
    self.assertEqual(tokens_again, expected_tokens)

    detokenized = executing.run(llm.detokenize(tokens))
    self.assertEqual(detokenized, expected_detokenized)

  @parameterized.named_parameters(
      (
          'single_character_words',
          'a b c',
          7,
          [97, 98, 99],
          'a b c',
      ),
      (
          'user_example_pad_width_8',
          f'{chr(1)}{chr(56)}{chr(23524)}',
          8,
          [int('000000010000005600023524')],
          f'{chr(1)}{chr(56)}{chr(23524)}',
      ),
      (
          'user_example_pad_width_7',
          f'{chr(1)}{chr(56)}{chr(23524)}',
          7,
          [int('000000100000560023524')],
          f'{chr(1)}{chr(56)}{chr(23524)}',
      ),
      (
          'max_unicode_code_point_pad_width_7',
          f'{chr(0x10FFFF)}',
          7,
          [1114111],
          f'{chr(0x10FFFF)}',
      ),
      (
          'null_character_token_id_zero',
          chr(0),
          7,
          [0],
          chr(0),
      ),
      (
          'custom_pad_width_6',
          'ab cd',
          6,
          [int('000097000098'), int('000099000100')],
          'ab cd',
      ),
  )
  def test_ord_based_token_ids(
      self,
      text: str,
      pad_width: int,
      expected_tokens: list[int],
      expected_detokenized: str,
  ):
    backend = WordTokenizer(pad_width=pad_width)
    backend.register()

    tokens = executing.run(llm.tokenize(text))
    self.assertEqual(tokens, expected_tokens)

    detokenized = executing.run(llm.detokenize(tokens))
    self.assertEqual(detokenized, expected_detokenized)
    self.assertEqual(backend.detokenize(tokens), expected_detokenized)

  @parameterized.named_parameters(
      (
          'two_words',
          'hello world',
          2,
      ),
      (
          'extra_whitespace',
          '  hello   world  how   are you ',
          5,
      ),
      (
          'empty_string',
          '',
          0,
      ),
      (
          'whitespace_only',
          '   \n\t ',
          0,
      ),
      (
          'single_word',
          'onetwo',
          1,
      ),
      (
          'chunk_list',
          content_lib.ChunkList(['hello ', 'world']),
          2,
      ),
      (
          'chunk_list_empty',
          content_lib.ChunkList([]),
          0,
      ),
      (
          'chunk_list_multi',
          content_lib.ChunkList(['one two ', 'three ', 'four five']),
          5,
      ),
  )
  def test_count_tokens(
      self, content: str | content_lib.ChunkList, expected_count: int
  ):
    backend = WordTokenizer()
    backend.register()

    # Async llm builtin (uses llm's default count_tokens implementation)
    llm_count = executing.run(llm.count_tokens(content))
    self.assertEqual(llm_count, expected_count)

  @parameterized.named_parameters(
      (
          'five_requests',
          5,
      ),
      (
          'twenty_requests',
          20,
      ),
  )
  def test_parallel_execution_and_counters(self, num_requests: int):
    backend = WordTokenizer(cache_filename=self._cachefile.full_path)
    backend.register()

    tokenize_executable = [
        llm.tokenize(f'hello world {idx}') for idx in range(num_requests)
    ]
    tokenize_result = executing.run(executing.parallel(*tokenize_executable))

    self.assertLen(tokenize_result, num_requests)
    for tokens in tokenize_result:
      self.assertLen(tokens, 3)

    detokenize_executable = [
        llm.detokenize(tokens) for tokens in tokenize_result
    ]
    detokenize_result = executing.run(
        executing.parallel(*detokenize_executable)
    )

    expected_texts = [f'hello world {idx}' for idx in range(num_requests)]
    self.assertEqual(detokenize_result, expected_texts)

    expected_backend_counters = collections.Counter(
        {
            'tokenize': num_requests,
            'detokenize': num_requests,
        },
    )
    self.assertCounterEqual(backend._counters, expected_backend_counters)

  @parameterized.named_parameters(
      (
          'comma_and_semicolon',
          r'[,;]+',
          'apple,banana;cherry',
          3,
          'apple banana cherry',
      ),
      (
          'dash_delimiter',
          r'-+',
          'one-two--three',
          3,
          'one two three',
      ),
      (
          'pipe_delimiter',
          r'\|+',
          'a|b||c',
          3,
          'a b c',
      ),
  )
  def test_custom_split_regex(
      self,
      split_regex: str,
      text: str,
      expected_token_count: int,
      expected_detokenized: str,
  ):
    backend = WordTokenizer(split_regex=split_regex)
    backend.register()

    tokens = executing.run(llm.tokenize(text))
    self.assertLen(tokens, expected_token_count)

    detok = executing.run(llm.detokenize(tokens))
    self.assertEqual(detok, expected_detokenized)
    self.assertEqual(
        executing.run(llm.count_tokens(text)), expected_token_count
    )

  @parameterized.named_parameters(
      (
          'tokenize_int',
          'tokenize',
          123,
      ),
      (
          'tokenize_list',
          'tokenize',
          [1, 2],
      ),
      (
          'detokenize_str_element',
          'detokenize',
          ['not_an_int'],
      ),
      (
          'count_tokens_int',
          'count_tokens',
          123,
      ),
      (
          'count_tokens_dict',
          'count_tokens',
          {'a': 1},
      ),
  )
  def test_invalid_inputs(self, method_name: str, invalid_arg: Any):
    backend = WordTokenizer()
    backend.register()

    with self.assertRaises(ValueError):
      if method_name == 'tokenize':
        executing.run(llm.tokenize(invalid_arg))
      elif method_name == 'detokenize':
        executing.run(llm.detokenize(invalid_arg))
      elif method_name == 'count_tokens':
        executing.run(llm.count_tokens(invalid_arg))


if __name__ == '__main__':
  absltest.main()
