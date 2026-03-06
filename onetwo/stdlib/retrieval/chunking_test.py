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

from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import retrieval_data_structures


async def mock_count_tokens_fn(text: str) -> int:
  return len(text)


async def mock_tokenize_fn(text: str) -> Sequence[int]:
  return [ord(c) for c in text]


@executing.make_executable  # pytype: disable=wrong-arg-types
async def mock_detokenize_fn(tokens: Sequence[int]) -> str:
  return ''.join([chr(token) for token in tokens])


class ChunkingTest(parameterized.TestCase):

  def testNoChunkingDefaultFormat(self):
    document = retrieval_data_structures.Document(
        content=content_lib.ChunkList([content_lib.Chunk('original text')]),
        title='Test Title',
        doc_id='test_doc_123',
        metadata={'key1': 'value1', 'key2': 'value2'},
    )
    chunker = chunking.NoChunking()
    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
    self.assertEqual(chunks[0].text, 'original text')

  def testNoChunkingFormat(self):
    document = retrieval_data_structures.Document(
        content='original text',
        title='Test Title',
        doc_id='test_doc_123',
        metadata={'key1': 'value1', 'key2': 'value2'},
    )
    chunker = chunking.NoChunking()
    chunker.document_format = 'DocFormat: {text} (Title: {title})'
    chunker.chunk_format = 'ChunkFormat: {text} [ID: {doc_id}, Meta: {key1}]'

    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count

    expected_final_text = (
        'ChunkFormat: DocFormat: original text (Title: Test Title) '
        '[ID: test_doc_123, Meta: value1]'
    )

    with self.subTest('correct_number_of_chunks'):
      self.assertLen(chunks, 1)

    with self.subTest('correct_chunk_text'):
      output_chunk = chunks[0]
      self.assertEqual(output_chunk.text, expected_final_text)

    with self.subTest('correct_chunk_attributes'):
      self.assertEqual(output_chunk.title, document.title)
      self.assertEqual(output_chunk.doc_id, document.doc_id)
      self.assertEqual(output_chunk.metadata, document.metadata)

  def testTruncateStringToMaxTokens(self):
    with (
        mock.patch.object(
            llm, 'count_tokens', autospec=True
        ) as mock_count_tokens,
        mock.patch.object(llm, 'detokenize', autospec=True) as mock_detokenize,
        mock.patch.object(llm, 'tokenize', autospec=True) as mock_tokenize,
    ):
      mock_count_tokens.side_effect = mock_count_tokens_fn
      mock_tokenize.side_effect = mock_tokenize_fn
      mock_detokenize.side_effect = mock_detokenize_fn
      text = 'hello world'
      with self.subTest('truncate_to_6_tokens'):
        truncated_text = ot.run(chunking.truncate_string_to_max_tokens(text, 6))
        # Trimming is done by the chunking method, not the truncation method.
        self.assertEqual(truncated_text, 'hello ')

      with self.subTest('truncate_to_60_tokens'):
        truncated_text = ot.run(
            chunking.truncate_string_to_max_tokens(text, 60)
        )
        self.assertEqual(truncated_text, 'hello world')

  def testFilterByMaxTokens(self):
    with (
        mock.patch.object(
            llm, 'count_tokens', autospec=True
        ) as mock_count_tokens,
        mock.patch.object(llm, 'detokenize', autospec=True) as mock_detokenize,
        mock.patch.object(llm, 'tokenize', autospec=True) as mock_tokenize,
    ):
      mock_count_tokens.side_effect = mock_count_tokens_fn
      mock_tokenize.side_effect = mock_tokenize_fn
      mock_detokenize.side_effect = mock_detokenize_fn
      texts = ['one', 'two', 'three', 'four']
      filtered_texts = ot.run(chunking.filter_by_max_tokens(texts, 10))
      self.assertEqual(filtered_texts, ['one', 'two', 'thre'])

  @parameterized.named_parameters(
      ('strip_whitespace_true', True),
      ('strip_whitespace_false', False),
  )
  @mock.patch.object(llm, 'count_tokens', autospec=True)
  @mock.patch.object(llm, 'detokenize', autospec=True)
  @mock.patch.object(llm, 'tokenize', autospec=True)
  def testChunkByMaxTokens(
      self, strip_whitespace, mock_tokenize, mock_detokenize, mock_count_tokens
  ):
    mock_count_tokens.side_effect = mock_count_tokens_fn
    mock_tokenize.side_effect = mock_tokenize_fn
    mock_detokenize.side_effect = mock_detokenize_fn
    document = retrieval_data_structures.Document(content='hello world')
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=6, strip_whitespace=strip_whitespace
    )
    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
    expected_chunks = [
        retrieval_data_structures.Document(
            content='hello' if strip_whitespace else 'hello '
        ),
        retrieval_data_structures.Document(content='world'),
    ]
    # Note that in this case, tokenization and concatenation are commutative,
    # so the fast chunker will produce the same chunks as the regular chunker.
    with self.subTest('chunker_is_equivalent'):
      self.assertEqual(
          chunks,
          expected_chunks,
      )

  @parameterized.named_parameters([
      {
          'testcase_name': 'no_overlap',
          'overlap_window': 0,
          'expected_chunks': [
              retrieval_data_structures.Document(content='hello'),
              retrieval_data_structures.Document(content='world'),
          ],
      },
      {
          'testcase_name': 'overlap_2',
          'overlap_window': 2,
          'expected_chunks': [
              retrieval_data_structures.Document(content='hello'),
              retrieval_data_structures.Document(content='o worl'),
              retrieval_data_structures.Document(content='rld'),
          ],
      },
  ])
  def testChunkByMaxTokensWithOverlap(
      self,
      overlap_window,
      expected_chunks,
  ):
    with (
        mock.patch.object(
            llm, 'count_tokens', autospec=True
        ) as mock_count_tokens,
        mock.patch.object(llm, 'detokenize', autospec=True) as mock_detokenize,
        mock.patch.object(llm, 'tokenize', autospec=True) as mock_tokenize,
    ):
      mock_count_tokens.side_effect = mock_count_tokens_fn
      mock_tokenize.side_effect = mock_tokenize_fn
      mock_detokenize.side_effect = mock_detokenize_fn
      document = retrieval_data_structures.Document(content='hello world')
      chunker = chunking.ChunkByMaxTokens(
          max_tokens_per_chunk=6, overlap_window=overlap_window
      )
      chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
      # Note that in this case, tokenization and concatenation are commutative,
      # so the fast chunker will produce the same chunks as the regular chunker.
      with self.subTest('chunker_is_equivalent'):
        self.assertEqual(
            chunks,
            expected_chunks,
        )


if __name__ == '__main__':
  absltest.main()
