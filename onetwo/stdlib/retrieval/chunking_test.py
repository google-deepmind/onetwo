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
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import retrieval_data_structures


class ChunkingTest(parameterized.TestCase):

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


if __name__ == '__main__':
  absltest.main()
