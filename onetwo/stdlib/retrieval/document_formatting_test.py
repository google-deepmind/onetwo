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
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import retrieval_data_structures


class DocumentFormattingTest(parameterized.TestCase):

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
  def testNoFormatting(self, document):
    formatter = document_formatting.NoFormatting()
    result = ot.run(formatter(document))  # pytype: disable=wrong-arg-count
    with self.subTest('correct_content'):
      self.assertEqual(result.content, document.content)

    with self.subTest('correct_attributes'):
      self.assertEqual(result.title, document.title)
      self.assertEqual(result.doc_id, document.doc_id)
      self.assertEqual(result.metadata, document.metadata)

    with self.subTest('same_document_in_and_out'):
      self.assertEqual(result, document)

  @parameterized.named_parameters([
      {
          'testcase_name': 'no_format',
          'format_str': '{text}',
          'expected_content': 'hello world',
      },
      {
          'testcase_name': 'with_title',
          'format_str': 'Title: {title}\nContent: {text}',
          'expected_content': 'Title: Test Title\nContent: hello world',
      },
      {
          'testcase_name': 'with_doc_id',
          'format_str': '[{doc_id}] {text}',
          'expected_content': '[1] hello world',
      },
      {
          'testcase_name': 'with_metadata',
          'format_str': '{text} (Author: {author})',
          'expected_content': 'hello world (Author: Alice)',
      },
      {
          'testcase_name': 'with_title_and_metadata',
          'format_str': 'Title: {title}\nAuthor: {author}\nContent: {text}',
          'expected_content': (
              'Title: Test Title\nAuthor: Alice\nContent: hello world'
          ),
      },
  ])
  def testTextDocumentFormatter(self, format_str, expected_content):
    document = retrieval_data_structures.Document(
        title='Test Title',
        content='hello world',
        doc_id='1',
        metadata={'author': 'Alice'},
    )
    formatter = document_formatting.TextDocumentFormatter(
        format_str=format_str,
    )
    result = ot.run(formatter(document))  # pytype: disable=wrong-arg-count

    with self.subTest('correct_formatted_content'):
      self.assertEqual(result.content, expected_content)

    with self.subTest('preserves_title'):
      self.assertEqual(result.title, document.title)

    with self.subTest('preserves_doc_id'):
      self.assertEqual(result.doc_id, document.doc_id)

    with self.subTest('preserves_metadata'):
      self.assertEqual(result.metadata, document.metadata)

  def testTextDocumentFormatterDefaultFormatIsIdentity(self):
    """The default format_str '{text}' should act as identity on content."""
    document = retrieval_data_structures.Document(
        title='Test Title',
        content='hello world',
        doc_id='1',
    )
    formatter = document_formatting.TextDocumentFormatter()
    result = ot.run(formatter(document))  # pytype: disable=wrong-arg-count
    self.assertEqual(result.content, 'hello world')

  def testTextDocumentFormatterDoesNotMutateInput(self):
    """Formatting should not mutate the original document."""
    document = retrieval_data_structures.Document(
        title='Test Title',
        content='hello world',
        doc_id='1',
        metadata={'key': 'value'},
    )
    formatter = document_formatting.TextDocumentFormatter(
        format_str='Title: {title}\nContent: {text}',
    )
    ot.run(formatter(document))  # pytype: disable=wrong-arg-count
    self.assertEqual(document.content, 'hello world')


if __name__ == '__main__':
  absltest.main()
