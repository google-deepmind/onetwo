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

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)


class ChunkingTest(parameterized.TestCase):

  def setUp(self):
    """Register LLMForTest backend."""
    super().setUp()
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)

  @parameterized.named_parameters([
      {
          'testcase_name': 'string_content',
          'document': retrieval_data_structures.Document(
              content='original text',
              title='Test Title',
              doc_id='test_doc_123',
              metadata={'key1': 'value1', 'key2': 'value2'},
          ),
      },
      {
          'testcase_name': 'chunk_list_content',
          'document': retrieval_data_structures.Document(
              content=content_lib.ChunkList(
                  [content_lib.Chunk('original text')]
              ),
              title='Test Title',
              doc_id='test_doc_123',
              metadata={'key1': 'value1', 'key2': 'value2'},
          ),
      },
      {
          'testcase_name': 'chunk_list_multimodal_content',
          'document': retrieval_data_structures.Document(
              content=content_lib.ChunkList([
                  content_lib.Chunk('original text'),
                  content_lib.Chunk(
                      b'mock_image_data', content_type='image/jpeg'
                  ),
                  content_lib.Chunk('more text'),
              ]),
              title='Test Title',
              doc_id='test_doc_123',
              metadata={'key1': 'value1', 'key2': 'value2'},
          ),
      },
  ])
  def testNoChunking(self, document):
    chunker = chunking.NoChunking()
    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
    with self.subTest('correct_number_of_chunks'):
      self.assertLen(chunks, 1)

    output_chunk = chunks[0]

    with self.subTest('correct_chunk_content'):
      self.assertEqual(output_chunk.content, document.content)

    with self.subTest('correct_chunk_attributes'):
      self.assertEqual(output_chunk.title, document.title)
      self.assertEqual(output_chunk.doc_id, document.doc_id)
      self.assertEqual(output_chunk.metadata, document.metadata)

    with self.subTest('same_document_in_and_out'):
      self.assertEqual(output_chunk, document)

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

  @parameterized.named_parameters([
      {
          'testcase_name': 'strip_whitespace_true_no_format',
          'strip_whitespace': True,
          'document_format': """{text}""",
          'chunk_format': """{text}""",
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='hello',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='world',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
      {
          'testcase_name': 'strip_whitespace_false_no_format',
          'strip_whitespace': False,
          'document_format': """{text}""",
          'chunk_format': """{text}""",
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='hello ',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='world',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
      {
          'testcase_name': 'strip_whitespace_true_with_chunk_format',
          'strip_whitespace': True,
          'document_format': """{text}""",
          'chunk_format': """Title: {title}\nContent: {text}""",
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='Title: Test Title\nContent: hello',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='Title: Test Title\nContent: world',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
      {
          'testcase_name': (
              'strip_whitespace_true_with_chunk_format_include_metadata'
          ),
          'strip_whitespace': True,
          'document_format': """{text}""",
          'chunk_format': (
              """Title: {title}\nContent: {text}\nChunk Rank: {chunk_number}/{total_number_of_chunks}"""
          ),
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='Title: Test Title\nContent: hello\nChunk Rank: 1/2',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='Title: Test Title\nContent: world\nChunk Rank: 2/2',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
      {
          'testcase_name': 'strip_whitespace_true_with_document_format',
          'strip_whitespace': True,
          'document_format': """Title: {title}\nContent: {text}""",
          'chunk_format': """{text}""",
          'max_tokens_per_chunk': 19,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='Title: Test Title\nC',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='ontent: hello world',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
  ])
  def testChunkByMaxTokens(
      self,
      strip_whitespace,
      document_format,
      chunk_format,
      max_tokens_per_chunk,
      expected_chunks,
  ):
    document = retrieval_data_structures.Document(
        title='Test Title', content='hello world', doc_id='1'
    )
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=max_tokens_per_chunk,
        strip_whitespace=strip_whitespace,
        document_format=document_format,
        chunk_format=chunk_format,
    )
    chunks = ot.run(chunker(document))  # pytype: disable=wrong-arg-count
    with self.subTest('chunker_is_equivalent'):
      self.assertEqual(
          chunks,
          expected_chunks,
      )

  @parameterized.named_parameters([
      {
          'testcase_name': 'no_overlap',
          'overlap_window': 0,
          'document': retrieval_data_structures.Document(
              content='hello world', doc_id='1'
          ),
          'expected_chunks': [
              retrieval_data_structures.Document(
                  content='hello',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  content='world',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
          ],
      },
      {
          'testcase_name': 'overlap_2',
          'overlap_window': 2,
          'document': retrieval_data_structures.Document(
              content='hello world', doc_id='1'
          ),
          'expected_chunks': [
              retrieval_data_structures.Document(
                  content='hello',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                  },
              ),
              retrieval_data_structures.Document(
                  content='o worl',
                  doc_id='1_2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                  },
              ),
              retrieval_data_structures.Document(
                  content='rld',
                  doc_id='1_3',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 3,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                  },
              ),
          ],
      },
  ])
  def testChunkByMaxTokensWithOverlap(
      self,
      overlap_window,
      document,
      expected_chunks,
  ):
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
