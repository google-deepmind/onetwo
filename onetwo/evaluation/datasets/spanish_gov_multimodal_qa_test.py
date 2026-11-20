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

"""Tests for spanish_gov_multimodal_qa module."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from onetwo import ot
from onetwo.core import content as content_lib
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import spanish_gov_multimodal_qa
from PIL import Image as PILImage


def _make_spanish_government_questions_row(
    question_en: str = 'Who is the Prime Minister?',
    question_es: str = 'Quien es el Presidente del Gobierno?',
    answer: str = 'Pedro Sanchez',
) -> dict[str, Any]:
  """Creates a fake SpanishGovernmentMultimodalQA questions row."""
  return {
      'question_en': question_en.encode('utf-8'),
      'answer': answer.encode('utf-8'),
      'question_es': question_es.encode('utf-8'),
      'split': b'validation',
  }


def _make_spanish_government_corpus_row(
    full_name: str = 'Pedro Sanchez',
    id_person: str = 'pedro_sanchez',
    cv_text: str = 'Presidente del Gobierno de Espana.',
    image_shape: tuple[int, int, int] = (10, 10, 3),
) -> dict[str, Any]:
  """Creates a fake SpanishGovernmentMultimodalQA corpus row."""
  return {
      'full_name': full_name.encode('utf-8'),
      'id_person': id_person.encode('utf-8'),
      'cv_text': cv_text.encode('utf-8'),
      'image': np.zeros(image_shape, dtype=np.uint8),
      'last_updated': b'2023-01-01',
      'split': b'validation',
  }


class SpanishGovernmentMultimodalQADatasetLoaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tfds = self.enter_context(
        mock.patch(
            'onetwo.evaluation.datasets.spanish_gov_multimodal_qa.tfds',
            autospec=True,
        )
    )
    self.mock_tfds.HF_MERLION_PLACER_DIR = '/tmp/test/dir'
    # Mock PILImage to avoid creating real image objects from numpy arrays.
    self.mock_pil_image = self.enter_context(
        mock.patch(
            'onetwo.evaluation.datasets.spanish_gov_multimodal_qa.PILImage',
            autospec=True,
        )
    )
    default_img = PILImage.new('RGB', (2, 2))
    default_img.putdata([
        (0, 0, 0),
        (85, 85, 85),
        (170, 170, 170),
        (255, 255, 255),
    ])
    self.mock_pil_image.fromarray.return_value = default_img

  def test_registered_in_dataset_registry(self):
    loader = datasets.get_dataset_loader(
        'spanish_gov_multimodal_qa', max_number_of_documents=3
    )

    self.assertIsInstance(
        loader,
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader,
    )
    self.assertEqual(loader.max_number_of_documents, 3)
    # E.g. used to name the directory of indexes of this dataset.
    self.assertEqual(loader.dataset_name, 'spanish_gov_multimodal_qa')

  def test_load_examples_single_row(self):
    rows = [
        _make_spanish_government_questions_row(question_en='Q1', answer='A1')
    ]
    self.mock_tfds.as_numpy.side_effect = [iter(rows), iter([])]

    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader()
    )
    examples = list(ot.run(loader.load_examples()))

    self.assertLen(examples, 1)
    self.assertEqual(examples[0]['question'], 'Q1')
    self.assertEqual(examples[0]['answer'], 'A1')
    self.mock_tfds.load.assert_has_calls([
        mock.call(
            'huggingface:megaelius__spanishgovernmentmultimodalqa/questions',
            split='test',
        ),
        mock.call(
            'huggingface:megaelius__spanishgovernmentmultimodalqa/corpus',
            split='test',
        ),
    ])

  @parameterized.named_parameters(
      dict(
          testcase_name='english_question',
          lang=spanish_gov_multimodal_qa.QuestionLanguage.EN,
          expected_question='English Question',
      ),
      dict(
          testcase_name='spanish_question',
          lang=spanish_gov_multimodal_qa.QuestionLanguage.ES,
          expected_question='Spanish Question',
      ),
  )
  def test_load_examples_with_question_language(
      self, lang, expected_question
  ):
    rows = [
        _make_spanish_government_questions_row(
            question_en='English Question',
            question_es='Spanish Question',
            answer='A1',
        )
    ]
    self.mock_tfds.as_numpy.side_effect = [iter(rows), iter([])]

    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader(
            question_language=lang
        )
    )
    examples = list(ot.run(loader.load_examples()))

    self.assertLen(examples, 1)
    self.assertEqual(examples[0]['question'], expected_question)

  def test_load_examples_multiple_rows(self):
    rows = [
        _make_spanish_government_questions_row(question_en='Q1', answer='A1'),
        _make_spanish_government_questions_row(question_en='Q2', answer='A2'),
    ]
    self.mock_tfds.as_numpy.side_effect = [iter(rows), iter([])]

    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader()
    )
    examples = list(ot.run(loader.load_examples()))

    self.assertLen(examples, 2)

  def test_load_documents_single_row(self):
    """A single corpus row should produce 1 document (image and text)."""
    rows = [
        _make_spanish_government_corpus_row(
            id_person='p1', full_name='Person 1', cv_text='CV text 1.'
        )
    ]
    self.mock_tfds.as_numpy.return_value = iter(rows)

    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader()
    )
    documents = list(ot.run(loader.load_documents()))

    self.assertLen(documents, 1)
    doc = documents[0]
    self.assertEqual(doc.doc_id, 'p1')
    self.mock_tfds.load.assert_called_once_with(
        'huggingface:megaelius__spanishgovernmentmultimodalqa/corpus',
        split='test',
    )

    # Check document content.
    self.assertEqual(doc.title, 'Person 1')
    self.assertIsInstance(doc.content, content_lib.ChunkList)
    self.assertLen(doc.content, 2)
    # Check image chunk.
    self.assertIsInstance(doc.content[0], content_lib.Chunk)
    self.assertIsInstance(doc.content[0].content, bytes)
    self.assertEqual(doc.content[0].content_type, 'image/png')
    self.mock_pil_image.fromarray.assert_called_once()
    # Check text chunk.
    self.assertIsInstance(doc.content[1], content_lib.Chunk)
    self.assertEqual(doc.content[1].content_type, 'str')
    self.assertEqual(doc.content[1].content, 'CV text 1.')

  @parameterized.named_parameters(
      dict(testcase_name='limit_1_example', max_n=1, expected_examples=1),
      dict(testcase_name='no_limit_examples', max_n=None, expected_examples=2),
  )
  def test_load_examples_max_number(self, max_n, expected_examples):
    rows = [
        _make_spanish_government_questions_row(question_en='Q1', answer='A1'),
        _make_spanish_government_questions_row(question_en='Q2', answer='A2'),
    ]
    self.mock_tfds.as_numpy.side_effect = [iter(rows), iter([])]
    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader(
            max_number_of_examples=max_n
        )
    )
    examples = list(ot.run(loader.load_examples()))
    self.assertLen(examples, expected_examples)

  def test_load_documents_ignores_max_number_of_examples(self):
    rows = [
        _make_spanish_government_corpus_row(id_person='p1'),
        _make_spanish_government_corpus_row(id_person='p2'),
    ]
    self.mock_tfds.as_numpy.return_value = iter(rows)
    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader(
            max_number_of_examples=1
        )
    )
    documents = list(ot.run(loader.load_documents()))
    self.assertLen(documents, 2)

  def test_loader_normalizes_images_and_populates_golden_doc_ids(self):
    corpus_rows = [
        _make_spanish_government_corpus_row(
            id_person='10',
            full_name='María García López',
            cv_text='Worked at the Ministry.',
        ),
        _make_spanish_government_corpus_row(
            id_person='20',
            full_name='Carlos Pérez',
            cv_text='Worked at the Treasury.',
        ),
    ]
    question_rows = [
        _make_spanish_government_questions_row(
            question_en='Who worked at the Ministry of Science?',
            answer='María García López',
        ),
    ]

    self.mock_tfds.as_numpy.side_effect = [
        iter(corpus_rows),
        iter(question_rows),
        iter(corpus_rows),
    ]
    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader(
            max_number_of_documents=1,
            max_number_of_examples=1,
        )
    )
    docs = list(ot.run(loader.load_documents()))
    examples = list(ot.run(loader.load_examples()))

    self.assertLen(docs, 1)
    self.assertIsInstance(docs[0].content, content_lib.ChunkList)
    image_chunk = docs[0].content.chunks[0]
    self.assertIsInstance(image_chunk.content, bytes)
    self.assertEqual(image_chunk.content_type, 'image/png')

    self.assertLen(examples, 1)
    self.assertEqual(examples[0]['metadata']['golden_doc_ids'], ['10'])

  @parameterized.named_parameters(
      dict(
          testcase_name='solid_black_image',
          bg_color=(0, 0, 0),
          num_non_bg_pixels=0,
      ),
      dict(
          testcase_name='solid_white_image',
          bg_color=(255, 255, 255),
          num_non_bg_pixels=0,
      ),
      dict(
          testcase_name='mostly_white_placeholder_with_small_mark',
          bg_color=(255, 255, 255),
          num_non_bg_pixels=4,  # 96/100 = 96% white >= 95% threshold
      ),
  )
  def test_load_documents_skips_blank_or_placeholder_image(
      self, bg_color, num_non_bg_pixels
  ):
    rows = [
        _make_spanish_government_corpus_row(
            id_person='p1', full_name='Person 1', cv_text='CV text 1.'
        )
    ]
    self.mock_tfds.as_numpy.return_value = iter(rows)
    img = PILImage.new('RGB', (10, 10), color=bg_color)
    pixels = [bg_color] * 100
    for i in range(num_non_bg_pixels):
      pixels[i] = (128, 0, 0)
    img.putdata(pixels)
    self.mock_pil_image.fromarray.return_value = img

    loader = (
        spanish_gov_multimodal_qa.SpanishGovernmentMultimodalQADatasetLoader()
    )
    documents = list(ot.run(loader.load_documents()))

    self.assertLen(documents, 1)
    self.assertIsInstance(documents[0].content, content_lib.ChunkList)
    self.assertLen(documents[0].content.chunks, 1)
    self.assertEqual(documents[0].content.chunks[0].content, 'CV text 1.')
    self.assertEqual(documents[0].content.chunks[0].content_type, 'str')


if __name__ == '__main__':
  absltest.main()
