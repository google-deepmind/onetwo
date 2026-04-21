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
from onetwo.stdlib.qa import qa_prompt_builder
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document


class QAPromptBuilderTest(parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': 'default_template',
          'docs': [_Document(content='doc1'), _Document(content='doc2')],
          'question': 'my question',
          'expected_prompt': (
              """Use the following pieces of context to answer the question at
    the end. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.

    Context:
doc1

===========

doc2

Question: my question
Answer:"""
          ),
      },
      {
          'testcase_name': 'multimodal_content',
          'docs': [
              _Document(content='doc1'),
              _Document(
                  content=content_lib.ChunkList([
                      content_lib.Chunk('text before image'),
                      content_lib.Chunk(
                          b'fake_image_data', content_type='image/png'
                      ),
                      content_lib.Chunk('text after image'),
                  ])
              ),
          ],
          'question': 'my question',
          'expected_prompt': (
              """Use the following pieces of context to answer the question at
    the end. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.

    Context:
doc1

===========

text before image<image/png>text after image

Question: my question
Answer:"""
          ),
      },
  ])
  def test_qa_prompt_builder_with_document(
      self, docs, question, expected_prompt
  ):
    async def mock_formatter(doc: _Document) -> str:
      return doc.text

    builder = qa_prompt_builder.QAPromptBuilder[_Document, str](
        doc_to_content=mock_formatter,
        delimiter='\n\n===========\n\n'
    )
    composable_prompt = ot.run(builder.compile_prompt(docs, question))  # pytype: disable=wrong-arg-count
    self.assertEqual(str(composable_prompt), expected_prompt)

  def test_qa_document_prompt_builder(self):
    docs = [
        _Document(content='content1', title='Title 1'),
        _Document(content='content2', title='Title 2'),
    ]
    question = 'my question'

    builder = qa_prompt_builder.QADocumentPromptBuilder()
    composable_prompt = ot.run(builder.compile_prompt(docs, question))  # pytype: disable=wrong-arg-count

    expected_prompt = """Use the following pieces of context to answer the question at
    the end. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.

    Context:
content1

content2

Question: my question
Answer:"""

    self.assertEqual(str(composable_prompt), expected_prompt)

  def test_qa_document_prompt_builder_callable(self):
    docs = [
        _Document(content='content1', title='Title 1'),
        _Document(content='content2', title='Title 2'),
    ]
    question = 'my question'

    builder = qa_prompt_builder.QADocumentPromptBuilder()
    composable_prompt = ot.run(builder(docs, question))  # pytype: disable=wrong-arg-count

    expected_prompt = """Use the following pieces of context to answer the question at
    the end. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.

    Context:
content1

content2

Question: my question
Answer:"""

    self.assertEqual(str(composable_prompt), expected_prompt)


if __name__ == '__main__':
  absltest.main()
