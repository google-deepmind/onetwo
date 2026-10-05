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

import json

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
import PIL.Image
import PIL.ImageDraw


def _create_swiss_flag_image() -> PIL.Image.Image:
  """Creates an image of the Swiss flag."""
  img_flag = PIL.Image.new('RGB', (32, 32), 'red')
  draw = PIL.ImageDraw.Draw(img_flag)
  draw.rectangle((13, 6, 18, 25), fill='white')
  draw.rectangle((6, 13, 25, 18), fill='white')
  return img_flag


class RetrievalDataStructuresTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', retrieval_data_structures.Document(content=''), ''),
      (
          'text_only',
          retrieval_data_structures.Document(content='Hello world.'),
          'Hello world.',
      ),
      (
          'image_only',
          retrieval_data_structures.Document(
              content=content_lib.ChunkList(
                  [content_lib.Chunk(_create_swiss_flag_image())]
              )
          ),
          '<image/jpeg>',
      ),
      (
          'text_and_image',
          retrieval_data_structures.Document(
              content=content_lib.ChunkList([
                  content_lib.Chunk('Hello world.'),
                  content_lib.Chunk(_create_swiss_flag_image()),
              ])
          ),
          'Hello world.<image/jpeg>',
      ),
  )
  def test_document_content_to_text(self, document, expected_text):
    with self.subTest('text_property'):
      self.assertEqual(expected_text, document.text)

    with self.subTest('explicit_str_conversion'):
      self.assertEqual(expected_text, str(document.content))

    with self.subTest('implicit_str_conversion'):
      self.assertEqual(expected_text, f'{document.content}')

  def test_document_to_json_str_content(self):
    doc = retrieval_data_structures.Document(
        doc_id='id1',
        title='Title',
        content='Pure text content',
        url='https://example.com',
        metadata={'key': 'val'},
    )
    json_str = doc.to_json()
    self.assertEqual(
        json.loads(json_str),
        {
            'doc_id': 'id1',
            'title': 'Title',
            'content': 'Pure text content',
            'url': 'https://example.com',
            'metadata': {'key': 'val'},
        },
    )
    self.assertEqual(
        retrieval_data_structures.Document.from_json(json_str), doc
    )

  def test_document_to_json_multimodal_succeeds(self):
    multimodal_doc = retrieval_data_structures.Document(
        doc_id='m2',
        title='Multimodal ChunkList',
        content=content_lib.ChunkList([
            content_lib.Chunk(content=b'img_bytes', role='USER'),
            content_lib.Chunk(content='Text prompt', role='USER'),
        ]),
        metadata={'keys': ['a', 'b']},
    )
    json_str = multimodal_doc.to_json()
    self.assertIsInstance(json_str, str)

    restored = retrieval_data_structures.Document.from_json(json_str)
    self.assertEqual(restored.doc_id, multimodal_doc.doc_id)
    self.assertEqual(restored.title, multimodal_doc.title)
    self.assertIsInstance(restored.content, content_lib.ChunkList)
    assert isinstance(restored.content, content_lib.ChunkList)
    self.assertLen(restored.content.chunks, 2)
    assert isinstance(restored.content.chunks[0], content_lib.Chunk)
    assert isinstance(restored.content.chunks[1], content_lib.Chunk)
    self.assertEqual(restored.content.chunks[0].content, b'img_bytes')
    self.assertEqual(restored.content.chunks[1].content, 'Text prompt')
    self.assertEqual(restored.metadata['keys'], ['a', 'b'])

  def test_metadata_fields(self):
    self.assertEqual(
        retrieval_data_structures.METADATA_FIELD_SPAN_WITHIN_ORIGINAL_DOC,
        'span_within_original_doc',
    )


if __name__ == '__main__':
  absltest.main()
