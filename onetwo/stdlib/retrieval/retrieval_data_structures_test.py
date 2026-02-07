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


if __name__ == '__main__':
  absltest.main()
