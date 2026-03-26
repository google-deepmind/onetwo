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
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import retrieval_data_structures

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)

DOCS = [
    retrieval_data_structures.Document(
        content='original text',
        title='Text-only doc',
        doc_id='1',
        metadata={'key1': 'value1', 'key2': 'value2'},
    ),
    retrieval_data_structures.Document(
        content=content_lib.ChunkList([content_lib.Chunk('original text')]),
        title='ChunkList doc',
        doc_id='2',
        metadata={'key1': 'value1', 'key2': 'value2'},
    ),
    retrieval_data_structures.Document(
        content=content_lib.ChunkList([
            content_lib.Chunk('original text'),
            content_lib.Chunk(b'mock_image_data', content_type='image/jpeg'),
            content_lib.Chunk('more text'),
        ]),
        title='Multimodal doc',
        doc_id='3',
        metadata={'key1': 'value1', 'key2': 'value2'},
    ),
]

DOC1_CHUNKS_5TOKENS = [
    retrieval_data_structures.Document(
        title='Text-only doc',
        content='origi',
        doc_id='1_1',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '1',
            _CHUNK_NUMBER: 1,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
    retrieval_data_structures.Document(
        title='Text-only doc',
        content='nal t',
        doc_id='1_2',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '1',
            _CHUNK_NUMBER: 2,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
    retrieval_data_structures.Document(
        title='Text-only doc',
        content='ext',
        doc_id='1_3',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '1',
            _CHUNK_NUMBER: 3,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
]

DOC2_CHUNKS_5TOKENS = [
    retrieval_data_structures.Document(
        title='ChunkList doc',
        content='origi',
        doc_id='2_1',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '2',
            _CHUNK_NUMBER: 1,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
    retrieval_data_structures.Document(
        title='ChunkList doc',
        content='nal t',
        doc_id='2_2',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '2',
            _CHUNK_NUMBER: 2,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
    retrieval_data_structures.Document(
        title='ChunkList doc',
        content='ext',
        doc_id='2_3',
        metadata={
            'key1': 'value1',
            'key2': 'value2',
            _ORIGINAL_DOC_ID: '2',
            _CHUNK_NUMBER: 3,
            _TOTAL_NUMBER_OF_CHUNKS: 3,
        },
    ),
]


class CorpusRewritingTest(parameterized.TestCase):

  def setUp(self):
    """Register LLMForTest backend."""
    super().setUp()
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)

  @parameterized.named_parameters([
      {
          'testcase_name': 'string_content',
          'corpus': [DOCS[0]],
      },
      {
          'testcase_name': 'chunk_list_content',
          'corpus': [DOCS[1]],
      },
      {
          'testcase_name': 'multimodal_content',
          'corpus': [DOCS[2]],
      },
      {
          'testcase_name': 'all_docs',
          'corpus': DOCS,
      },
  ])
  def testNoRewriting(self, corpus):
    chunker = corpus_rewriting.NoRewriting()
    rewritten_corpus = ot.run(chunker(corpus))  # pytype: disable=wrong-arg-count

    with self.subTest('same_corpus_in_and_out'):
      self.assertEqual(rewritten_corpus, corpus)

  @parameterized.named_parameters([
      {
          'testcase_name': 'NoChunking_one_doc',
          'chunker': chunking.NoChunking(),
          'corpus': [DOCS[0]],
          'expected_corpus': [DOCS[0]],
      },
      {
          'testcase_name': 'NoChunking_chunk_list_doc',
          'chunker': chunking.NoChunking(),
          'corpus': [DOCS[1]],
          'expected_corpus': [DOCS[1]],
      },
      {
          'testcase_name': 'NoChunking_multimodal_doc',
          'chunker': chunking.NoChunking(),
          'corpus': [DOCS[2]],
          'expected_corpus': [DOCS[2]],
      },
      {
          'testcase_name': 'NoChunking_all_docs',
          'chunker': chunking.NoChunking(),
          'corpus': DOCS,
          'expected_corpus': DOCS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_one_doc',
          'chunker': chunking.ChunkByMaxTokens(max_tokens_per_chunk=5),
          'corpus': [DOCS[0]],
          'expected_corpus': DOC1_CHUNKS_5TOKENS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_chunk_list_doc',
          'chunker': chunking.ChunkByMaxTokens(max_tokens_per_chunk=5),
          'corpus': [DOCS[1]],
          'expected_corpus': DOC2_CHUNKS_5TOKENS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_first_two_docs',
          'chunker': chunking.ChunkByMaxTokens(max_tokens_per_chunk=5),
          'corpus': [DOCS[0], DOCS[1]],
          'expected_corpus': DOC1_CHUNKS_5TOKENS + DOC2_CHUNKS_5TOKENS,
      },
  ])
  def testChunkingCorpusRewriter(self, chunker, corpus, expected_corpus):
    chunker = corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker)
    rewritten_corpus = ot.run(chunker(corpus))  # pytype: disable=wrong-arg-count

    with self.subTest('out_corpus_as_expected'):
      self.assertEqual(rewritten_corpus, expected_corpus)

  @parameterized.named_parameters([
      {
          'testcase_name': 'NoFormatting_one_doc',
          'formatter': document_formatting.NoFormatting(),
          'corpus': [DOCS[0]],
          'expected_corpus': [DOCS[0]],
      },
      {
          'testcase_name': 'NoFormatting_chunk_list_doc',
          'formatter': document_formatting.NoFormatting(),
          'corpus': [DOCS[1]],
          'expected_corpus': [DOCS[1]],
      },
      {
          'testcase_name': 'NoFormatting_multimodal_doc',
          'formatter': document_formatting.NoFormatting(),
          'corpus': [DOCS[2]],
          'expected_corpus': [DOCS[2]],
      },
      {
          'testcase_name': 'NoFormatting_all_docs',
          'formatter': document_formatting.NoFormatting(),
          'corpus': DOCS,
          'expected_corpus': DOCS,
      },
      {
          'testcase_name': 'TextDocumentFormatter_identity_one_doc',
          'formatter': document_formatting.TextDocumentFormatter(),
          'corpus': [DOCS[0]],
          'expected_corpus': [DOCS[0]],
      },
      {
          'testcase_name': 'TextDocumentFormatter_with_title_one_doc',
          'formatter': document_formatting.TextDocumentFormatter(
              format_str='Title: {title}\nContent: {text}',
          ),
          'corpus': [DOCS[0]],
          'expected_corpus': [
              retrieval_data_structures.Document(
                  content='Title: Text-only doc\nContent: original text',
                  title='Text-only doc',
                  doc_id='1',
                  metadata={'key1': 'value1', 'key2': 'value2'},
              ),
          ],
      },
      {
          'testcase_name': 'TextDocumentFormatter_with_title_two_docs',
          'formatter': document_formatting.TextDocumentFormatter(
              format_str='Title: {title}\nContent: {text}',
          ),
          'corpus': [DOCS[0], DOCS[1]],
          'expected_corpus': [
              retrieval_data_structures.Document(
                  content='Title: Text-only doc\nContent: original text',
                  title='Text-only doc',
                  doc_id='1',
                  metadata={'key1': 'value1', 'key2': 'value2'},
              ),
              retrieval_data_structures.Document(
                  content='Title: ChunkList doc\nContent: original text',
                  title='ChunkList doc',
                  doc_id='2',
                  metadata={'key1': 'value1', 'key2': 'value2'},
              ),
          ],
      },
      {
          'testcase_name': 'TextDocumentFormatter_with_metadata',
          'formatter': document_formatting.TextDocumentFormatter(
              format_str='{text} (key1={key1})',
          ),
          'corpus': [DOCS[0]],
          'expected_corpus': [
              retrieval_data_structures.Document(
                  content='original text (key1=value1)',
                  title='Text-only doc',
                  doc_id='1',
                  metadata={'key1': 'value1', 'key2': 'value2'},
              ),
          ],
      },
  ])
  def testFormattingCorpusRewriter(self, formatter, corpus, expected_corpus):
    rewriter = corpus_rewriting.FormattingCorpusRewriter(formatter=formatter)
    rewritten_corpus = ot.run(rewriter(corpus))  # pytype: disable=wrong-arg-count

    with self.subTest('out_corpus_as_expected'):
      self.assertEqual(rewritten_corpus, expected_corpus)

  @parameterized.named_parameters([
      {
          'testcase_name': 'empty_rewriters',
          'rewriters': [],
          'corpus': DOCS,
          'expected_corpus': DOCS,
      },
      {
          'testcase_name': 'NoChunking_once_all_docs',
          'rewriters': [
              corpus_rewriting.ChunkingCorpusRewriter(chunking.NoChunking())
          ],
          'corpus': DOCS,
          'expected_corpus': DOCS,
      },
      {
          'testcase_name': 'NoChunking_five_times_all_docs',
          'rewriters': (
              [corpus_rewriting.ChunkingCorpusRewriter(chunking.NoChunking())]
              * 5
          ),
          'corpus': DOCS,
          'expected_corpus': DOCS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_first_two_docs',
          'rewriters': [
              corpus_rewriting.ChunkingCorpusRewriter(
                  chunking.ChunkByMaxTokens(max_tokens_per_chunk=5)
              )
          ],
          'corpus': [DOCS[0], DOCS[1]],
          'expected_corpus': DOC1_CHUNKS_5TOKENS + DOC2_CHUNKS_5TOKENS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_many_tokens_five_times',
          'rewriters': (
              [
                  corpus_rewriting.ChunkingCorpusRewriter(
                      chunking.ChunkByMaxTokens(max_tokens_per_chunk=1000)
                  )
              ]
              * 5
          ),
          'corpus': [DOCS[0], DOCS[1]],
          'expected_corpus': [
              retrieval_data_structures.Document(
                  content='original text',
                  title='Text-only doc',
                  doc_id='1_1_1_1_1_1',
                  metadata={
                      'key1': 'value1',
                      'key2': 'value2',
                      _ORIGINAL_DOC_ID: '1_1_1_1_1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 1,
                  },
              ),
              retrieval_data_structures.Document(
                  content='original text',
                  title='ChunkList doc',
                  doc_id='2_1_1_1_1_1',
                  metadata={
                      'key1': 'value1',
                      'key2': 'value2',
                      _ORIGINAL_DOC_ID: '2_1_1_1_1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 1,
                  },
              ),
          ],
      },
      {
          'testcase_name': 'ChunkByMaxTokens_then_NoRewriting_first_two_docs',
          'rewriters': [
              corpus_rewriting.ChunkingCorpusRewriter(
                  chunking.ChunkByMaxTokens(max_tokens_per_chunk=5)
              ),
              corpus_rewriting.NoRewriting(),
          ],
          'corpus': [DOCS[0], DOCS[1]],
          'expected_corpus': DOC1_CHUNKS_5TOKENS + DOC2_CHUNKS_5TOKENS,
      },
      {
          'testcase_name': 'ChunkByMaxTokens_then_formatting',
          'rewriters': [
              corpus_rewriting.ChunkingCorpusRewriter(
                  chunking.ChunkByMaxTokens(max_tokens_per_chunk=6)
              ),
              corpus_rewriting.FormattingCorpusRewriter(
                  formatter=document_formatting.TextDocumentFormatter(
                      format_str='Title: {title}\nContent: {text}',
                  ),
              ),
          ],
          'corpus': [
              retrieval_data_structures.Document(
                  content='hello world',
                  title='Test Title',
                  doc_id='1',
              ),
          ],
          'expected_corpus': [
              retrieval_data_structures.Document(
                  content='Title: Test Title\nContent: hello',
                  title='Test Title',
                  doc_id='1_1',
                  metadata={
                      _ORIGINAL_DOC_ID: '1',
                      _CHUNK_NUMBER: 1,
                      _TOTAL_NUMBER_OF_CHUNKS: 2,
                  },
              ),
              retrieval_data_structures.Document(
                  content='Title: Test Title\nContent: world',
                  title='Test Title',
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
  def testSequentialCorpusRewriter(self, rewriters, corpus, expected_corpus):
    rewriter = corpus_rewriting.SequentialCorpusRewriter(rewriters=rewriters)
    rewritten_corpus = ot.run(rewriter(corpus))  # pytype: disable=wrong-arg-count

    self.assertEqual(rewritten_corpus, expected_corpus)


if __name__ == '__main__':
  absltest.main()
