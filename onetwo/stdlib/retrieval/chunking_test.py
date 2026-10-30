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

from collections.abc import Iterable, Sequence
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import retrieval_data_structures

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)
_SPAN_WITHIN_ORIGINAL_DOC = (
    retrieval_data_structures.METADATA_FIELD_SPAN_WITHIN_ORIGINAL_DOC
)


@dataclasses.dataclass
class _CountingTokenizer(backends_test_utils.LLMForTest):
  """Test backend that counts its `tokenize` calls."""

  num_tokenize_calls: int = 0

  def tokenize(self, content: str | content_lib.ChunkList) -> Sequence[int]:
    self.num_tokenize_calls += 1
    return super().tokenize(content)


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
    chunks = list(ot.run(chunker(document)))
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
          'testcase_name': 'strip_whitespace_true',
          'strip_whitespace': True,
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='hello',
                  doc_id='1_chunk1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (0, 5),
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='world',
                  doc_id='1_chunk2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (6, 11),
                  },
              ),
          ],
      },
      {
          'testcase_name': 'strip_whitespace_false',
          'strip_whitespace': False,
          'max_tokens_per_chunk': 6,
          'expected_chunks': [
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='hello ',
                  doc_id='1_chunk1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (0, 6),
                  },
              ),
              retrieval_data_structures.Document(
                  title='Test Title',
                  content='world',
                  doc_id='1_chunk2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (6, 11),
                  },
              ),
          ],
      },
  ])
  def testChunkByMaxTokens(
      self,
      strip_whitespace,
      max_tokens_per_chunk,
      expected_chunks,
  ):
    document = retrieval_data_structures.Document(
        title='Test Title', content='hello world', doc_id='1'
    )
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=max_tokens_per_chunk,
        strip_whitespace=strip_whitespace,
    )
    chunks = ot.run(chunker(document))
    with self.subTest('chunker_is_equivalent'):
      self.assertEqual(
          chunks,
          expected_chunks,
      )
    with self.subTest('spans_correspond_to_substrings'):
      for chunk in chunks:
        start, end = chunk.metadata[_SPAN_WITHIN_ORIGINAL_DOC]
        self.assertEqual(document.text[start:end], chunk.content)

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
                  doc_id='1_chunk1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (0, 5),
                  },
              ),
              retrieval_data_structures.Document(
                  content='world',
                  doc_id='1_chunk2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                      _SPAN_WITHIN_ORIGINAL_DOC: (6, 11),
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
                  doc_id='1_chunk1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                      _SPAN_WITHIN_ORIGINAL_DOC: (0, 5),
                  },
              ),
              retrieval_data_structures.Document(
                  content='o worl',
                  doc_id='1_chunk2',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 2,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                      _SPAN_WITHIN_ORIGINAL_DOC: (4, 10),
                  },
              ),
              retrieval_data_structures.Document(
                  content='rld',
                  doc_id='1_chunk3',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 3,
                      _TOTAL_NUMBER_OF_CHUNKS: 3,
                      _SPAN_WITHIN_ORIGINAL_DOC: (8, 11),
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
    chunks = ot.run(chunker(document))
    with self.subTest('chunker_is_equivalent'):
      self.assertEqual(
          chunks,
          expected_chunks,
      )
    with self.subTest('spans_correspond_to_substrings'):
      for chunk in chunks:
        start, end = chunk.metadata[_SPAN_WITHIN_ORIGINAL_DOC]
        self.assertEqual(document.text[start:end], chunk.content)

  def testChunkByMaxTokensOverlapEqualsMaxTokensRaisesError(self):
    document = retrieval_data_structures.Document(
        content='hello world', doc_id='1'
    )
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=6, overlap_window=6
    )
    with self.assertRaises(ValueError):
      ot.run(chunker(document))

  def testChunkByMaxTokensCreatesTokenizerLazilyAndLocally(self):
    # `setUp` registers a global tokenizer. The configured tokenizer tokenizes
    # the same way, but we count its calls to tell them apart.
    tokenizer = _CountingTokenizer()
    tokenizer_factory = mock.MagicMock(return_value=tokenizer)
    self.enter_context(
        mock.patch.dict(
            backend_registry.BACKEND_REGISTRY,
            {'TokenizerForTest': tokenizer_factory},
        )
    )
    chunker = chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=6,
        tokenize_backend_config=backend_registry.TokenizeBackendConfig(
            backend_class_name='TokenizerForTest'
        ),
    )
    document = retrieval_data_structures.Document(
        content='hello world', doc_id='1'
    )

    with self.subTest('tokenizer_is_not_created_on_construction'):
      tokenizer_factory.assert_not_called()

    chunks = ot.run(chunker(document))
    _ = ot.run(chunker(document))

    with self.subTest('chunks_with_the_configured_tokenizer'):
      self.assertEqual([chunk.content for chunk in chunks], ['hello', 'world'])
      self.assertEqual(tokenizer.num_tokenize_calls, 2)

    with self.subTest('tokenizer_is_created_once'):
      tokenizer_factory.assert_called_once()

    with self.subTest('tokenizer_is_not_registered_globally'):
      _ = ot.run(llm.tokenize('hello'))
      self.assertEqual(tokenizer.num_tokenize_calls, 2)

  def testTextChunkerSpans(self):
    class ConcreteTextChunker(chunking.TextChunker):

      async def _chunk_text(self, text: str) -> Iterable[str]:  # pyrefly: ignore[bad-override]
        # Return two chunks: front and back half
        half = len(text) // 2
        return [
            text[:half],
            text[half:],
        ]

    document = retrieval_data_structures.Document(
        title='Test Title', content='abcdefgh', doc_id='test1'
    )
    chunker = ConcreteTextChunker()
    chunks = ot.run(chunker(document))

    self.assertLen(chunks, 2)

    self.assertEqual(chunks[0].content, 'abcd')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[0].doc_id, 'test1_chunk1')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[0].metadata[_ORIGINAL_DOC_ID], 'test1')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[0].metadata[_CHUNK_NUMBER], 1)  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[0].metadata[_TOTAL_NUMBER_OF_CHUNKS], 2)  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[0].metadata[_SPAN_WITHIN_ORIGINAL_DOC], (0, 4))  # pyrefly: ignore[bad-index]

    self.assertEqual(chunks[1].content, 'efgh')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[1].doc_id, 'test1_chunk2')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[1].metadata[_ORIGINAL_DOC_ID], 'test1')  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[1].metadata[_CHUNK_NUMBER], 2)  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[1].metadata[_TOTAL_NUMBER_OF_CHUNKS], 2)  # pyrefly: ignore[bad-index]
    self.assertEqual(chunks[1].metadata[_SPAN_WITHIN_ORIGINAL_DOC], (4, 8))  # pyrefly: ignore[bad-index]

  @parameterized.named_parameters([
      {
          'testcase_name': 'found_at_start',
          'full_text': 'hello world',
          'substring': 'hello',
          'expected': (0, 5),
      },
      {
          'testcase_name': 'found_in_middle',
          'full_text': 'hello world',
          'substring': 'o wo',
          'expected': (4, 8),
      },
      {
          'testcase_name': 'found_at_end',
          'full_text': 'hello world',
          'substring': 'world',
          'expected': (6, 11),
      },
      {
          'testcase_name': 'not_found',
          'full_text': 'hello world',
          'substring': 'abc',
          'expected': None,
      },
      {
          'testcase_name': 'empty_substring',
          'full_text': 'hello world',
          'substring': '',
          'expected': None,
      },
      {
          'testcase_name': 'empty_full_text_and_empty_substring',
          'full_text': '',
          'substring': '',
          'expected': None,
      },
      {
          'testcase_name': 'empty_full_text_and_non_empty_substring',
          'full_text': '',
          'substring': 'hello',
          'expected': None,
      },
  ])
  def testGetSubstringIndices(self, full_text, substring, expected):
    self.assertEqual(
        chunking.get_substring_indices(full_text, substring),
        expected,
    )

  @parameterized.named_parameters([
      {
          'testcase_name': 'basic_no_overlap',
          'content': 'one two three four five six seven eight nine ten',
          'max_words_per_chunk': 4,
          'overlap_window': 0,
          'expected_chunk_contents': [
              'one two three four',
              'five six seven eight',
              'nine ten',
          ],
      },
      {
          'testcase_name': 'basic_with_overlap',
          'content': 'one two three four five six seven eight nine ten',
          'max_words_per_chunk': 4,
          'overlap_window': 2,
          'expected_chunk_contents': [
              'one two three four',
              'three four five six',
              'five six seven eight',
              'seven eight nine ten',
              'nine ten',
          ],
      },
      {
          'testcase_name': 'punctuation_newlines_and_tabs',
          'content': (
              'Hello, world!\nThis is a test...\n\nDoes\tit chunk'
              ' properly?\nYes!'
          ),
          'max_words_per_chunk': 4,
          'overlap_window': 0,
          'expected_chunk_contents': [
              'Hello, world! This is',
              'a test... Does it',
              'chunk properly? Yes!',
          ],
      },
      {
          'testcase_name': 'punctuation_newlines_with_overlap',
          'content': 'First line.\nSecond line!\tThird line?\nFourth line.',
          'max_words_per_chunk': 3,
          'overlap_window': 1,
          'expected_chunk_contents': [
              'First line. Second',
              'Second line! Third',
              'Third line? Fourth',
              'Fourth line.',
          ],
      },
      {
          'testcase_name': 'single_word',
          'content': 'Hello!',
          'max_words_per_chunk': 5,
          'overlap_window': 0,
          'expected_chunk_contents': ['Hello!'],
      },
      {
          'testcase_name': 'empty_document',
          'content': '',
          'max_words_per_chunk': 5,
          'overlap_window': 0,
          'expected_chunk_contents': [],
      },
      {
          'testcase_name': 'whitespace_only',
          'content': '   \n\t   \n  ',
          'max_words_per_chunk': 5,
          'overlap_window': 0,
          'expected_chunk_contents': [],
      },
  ])
  def testChunkByMaxWords(
      self,
      content: str,
      max_words_per_chunk: int,
      overlap_window: int,
      expected_chunk_contents: list[str],
  ):
    document = retrieval_data_structures.Document(
        title='Test Title',
        content=content,
        doc_id='1',
    )
    chunker = chunking.ChunkByMaxWords(
        max_words_per_chunk=max_words_per_chunk,
        overlap_window=overlap_window,
    )
    chunks = list(ot.run(chunker(document)))
    chunk_contents = [chunk.content for chunk in chunks]
    self.assertEqual(chunk_contents, expected_chunk_contents)


class MakeChunkDocumentsTest(absltest.TestCase):

  def test_builds_chunk_documents(self):
    document = retrieval_data_structures.Document(
        doc_id='doc',
        title='Title',
        url='http://example.com',
        content='abcdef',
        metadata={'key': 'value'},
    )

    chunks = chunking.make_chunk_documents(
        document, ['abc', 'def'], spans=[(0, 3), (3, 6)]
    )

    self.assertEqual(
        chunks,
        [
            retrieval_data_structures.Document(
                doc_id='doc_chunk1',
                title='Title',
                url='http://example.com',
                content='abc',
                metadata={
                    'key': 'value',
                    _ORIGINAL_DOC_ID: 'doc',
                    _CHUNK_NUMBER: 1,
                    _TOTAL_NUMBER_OF_CHUNKS: 2,
                    _SPAN_WITHIN_ORIGINAL_DOC: (0, 3),
                },
            ),
            retrieval_data_structures.Document(
                doc_id='doc_chunk2',
                title='Title',
                url='http://example.com',
                content='def',
                metadata={
                    'key': 'value',
                    _ORIGINAL_DOC_ID: 'doc',
                    _CHUNK_NUMBER: 2,
                    _TOTAL_NUMBER_OF_CHUNKS: 2,
                    _SPAN_WITHIN_ORIGINAL_DOC: (3, 6),
                },
            ),
        ],
    )
    with self.subTest('original_document_is_unchanged'):
      self.assertEqual(document.doc_id, 'doc')
      self.assertEqual(document.content, 'abcdef')
      self.assertEqual(document.metadata, {'key': 'value'})

  def test_without_spans_does_not_set_span(self):
    document = retrieval_data_structures.Document(doc_id='doc', content='ab')

    chunks = chunking.make_chunk_documents(document, ['a', 'b'])

    self.assertNotIn(_SPAN_WITHIN_ORIGINAL_DOC, chunks[0].metadata)


if __name__ == '__main__':
  absltest.main()
