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

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import retrieval_data_structures


class ChunkingTest(parameterized.TestCase):

  def setUp(self):
    """Register LLMForTest backend."""
    super().setUp()
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)

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
    text = 'hello world'
    with self.subTest('truncate_to_6_tokens'):
      truncated_text = ot.run(chunking.truncate_string_to_max_tokens(text, 6))
      # Trimming is done by the chunking method, not the truncation method.
      self.assertEqual(truncated_text, 'hello ')

    with self.subTest('truncate_to_60_tokens'):
      truncated_text = ot.run(chunking.truncate_string_to_max_tokens(text, 60))
      self.assertEqual(truncated_text, 'hello world')

  def testFilterByMaxTokens(self):
    texts = ['one', 'two', 'three', 'four']
    filtered_texts = ot.run(chunking.filter_by_max_tokens(texts, 10))
    self.assertEqual(filtered_texts, ['one', 'two', 'thre'])

  @parameterized.named_parameters(
      ('strip_whitespace_true', True),
      ('strip_whitespace_false', False),
  )
  def testChunkByMaxTokens(self, strip_whitespace):
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
    document = retrieval_data_structures.Document(content='hello world')
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=6, overlap_window=overlap_window
    )
    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
    with self.subTest('chunker_is_equivalent'):
      self.assertEqual(
          chunks,
          expected_chunks,
      )


if __name__ == '__main__':
  absltest.main()
