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

"""Tests for narrativeqa module."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from onetwo import ot
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import narrativeqa


def _make_narrativeqa_row(
    question: str = 'What is the capital of France?',
    answer: str = 'Paris',
    doc_id: str = 'test_doc_id_1',
    doc_text: str = 'France is a country in Europe. Its capital is Paris.',
    url: str = 'http://example.com/france',
    title: str = 'France Document',
    summary: str = 'Summary of France',
    kind: str = 'story',
) -> dict[str, Any]:
  """Creates a fake NarrativeQA row mimicking TFDS numpy output."""
  return {
      'document': {
          'id': doc_id.encode('utf-8'),
          'text': doc_text.encode('utf-8'),
          'url': url.encode('utf-8'),
          'summary': {
              'title': title.encode('utf-8'),
              'text': summary.encode('utf-8'),
          },
          'kind': kind.encode('utf-8'),
      },
      'question': {
          'text': question.encode('utf-8'),
      },
      'answers': {
          'text': np.array([answer.encode('utf-8')]),
      },
  }


class NarrativeQADatasetLoaderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_row',
          rows_kwargs=[
              dict(
                  question='Q1',
                  answer='A1',
              ),
          ],
          expected_example_count=1,
      ),
      dict(
          testcase_name='multiple_rows',
          rows_kwargs=[
              dict(
                  question='Q1',
                  answer='A1',
              ),
              dict(
                  question='Q2',
                  answer='A2',
              ),
          ],
          expected_example_count=2,
      ),
      dict(
          testcase_name='empty',
          rows_kwargs=[],
          expected_example_count=0,
      ),
  )
  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_load_returns_correct_example_counts(
      self, mock_tfds, rows_kwargs, expected_example_count
  ):
    rows = [_make_narrativeqa_row(**kw) for kw in rows_kwargs]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(split='train')
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, expected_example_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_row',
          rows_kwargs=[
              dict(doc_id='D1'),
          ],
          expected_doc_count=1,
      ),
      dict(
          testcase_name='multiple_rows_same_doc',
          rows_kwargs=[
              dict(doc_id='D1'),
              dict(doc_id='D1'),
          ],
          expected_doc_count=1,
      ),
      dict(
          testcase_name='multiple_rows_different_docs',
          rows_kwargs=[
              dict(doc_id='D1'),
              dict(doc_id='D2'),
          ],
          expected_doc_count=2,
      ),
  )
  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_load_documents_returns_correct_counts(
      self, mock_tfds, rows_kwargs, expected_doc_count
  ):
    rows = [_make_narrativeqa_row(**kw) for kw in rows_kwargs]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader()
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, expected_doc_count)

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_max_examples_limits_output(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(question='Q1', answer='A1'),
        _make_narrativeqa_row(question='Q2', answer='A2'),
        _make_narrativeqa_row(question='Q3', answer='A3'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(max_number_of_examples=2)
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)
    self.assertEqual(
        examples[0]['question'], 'In France Document, q1'  # pyrefly: ignore[bad-index]
    )
    self.assertEqual(
        examples[1]['question'], 'In France Document, q2'  # pyrefly: ignore[bad-index]
    )

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_max_examples_limits_documents_output(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(doc_id='D1'),
        _make_narrativeqa_row(doc_id='D2'),
        _make_narrativeqa_row(doc_id='D3'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(max_number_of_examples=2)
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 2)
    doc_ids = [doc.doc_id for doc in documents]
    self.assertIn('D1', doc_ids)
    self.assertIn('D2', doc_ids)
    self.assertNotIn('D3', doc_ids)

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_doc_ids_filter_examples(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(doc_id='D1', question='Q1'),
        _make_narrativeqa_row(doc_id='D2', question='Q2'),
        _make_narrativeqa_row(doc_id='D3', question='Q3'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(doc_ids=['D1', 'D3'])
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)
    self.assertEqual(
        examples[0]['question'], 'In France Document, q1'  # pyrefly: ignore[bad-index]
    )
    self.assertEqual(
        examples[1]['question'], 'In France Document, q3'  # pyrefly: ignore[bad-index]
    )

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_doc_ids_filter_documents(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(doc_id='D1'),
        _make_narrativeqa_row(doc_id='D2'),
        _make_narrativeqa_row(doc_id='D3'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(doc_ids=['D1', 'D3'])
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 2)
    doc_ids = [doc.doc_id for doc in documents]
    self.assertIn('D1', doc_ids)
    self.assertNotIn('D2', doc_ids)
    self.assertIn('D3', doc_ids)

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_clean_html_true(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(
            doc_text='<p>Hello <b>world</b>!</p>\n\n<pre>Code block</pre>'
        ),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(clean_html=True)
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 1)
    # The _clean_text logic extracts from <pre> if present, or BeautifulSoup
    # text otherwise.
    self.assertEqual(documents[0].content, 'Code block')  # pyrefly: ignore[bad-index]

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_clean_html_false(self, mock_tfds):
    raw_html = '<p>Hello <b>world</b>!</p>\n\n<pre>Code block</pre>'
    rows = [
        _make_narrativeqa_row(doc_text=raw_html),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader(clean_html=False)
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 1)
    self.assertEqual(documents[0].content, raw_html)  # pyrefly: ignore[bad-index]

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_tfds_load_arguments(self, mock_tfds):
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter([])

    loader = narrativeqa.NarrativeQADatasetLoader(split='test')
    _ = ot.run(loader.load_examples())

    mock_tfds.load.assert_called_with(
        'huggingface:narrativeqa:1.0.0', split='test'
    )

  def test_dataclass_defaults(self):
    loader = narrativeqa.NarrativeQADatasetLoader()
    self.assertEqual(loader.split, 'train')
    self.assertIsNone(loader.max_number_of_examples)
    self.assertTrue(loader.clean_html)
    self.assertIsNone(loader.doc_ids)

  def test_dataclass_custom_values(self):
    loader = narrativeqa.NarrativeQADatasetLoader(
        split='validation',
        max_number_of_examples=5,
        clean_html=False,
        doc_ids=['D1'],
    )
    self.assertEqual(loader.split, 'validation')
    self.assertEqual(loader.max_number_of_examples, 5)
    self.assertFalse(loader.clean_html)
    self.assertEqual(loader.doc_ids, ['D1'])

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_question_prepended_with_title(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(
            question='What happened to the hero?',
            title='Percival Keene',
        ),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader()
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 1)
    self.assertEqual(
        examples[0]['question'],  # pyrefly: ignore[bad-index]
        'In Percival Keene, what happened to the hero?',
    )

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_metadata_uses_golden_doc_ids(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(doc_id='doc_123'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader()
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 1)
    metadata = examples[0]['metadata']  # pyrefly: ignore[bad-index]
    self.assertNotIn('document_id', metadata)
    self.assertEqual(
        metadata[datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS],
        ['doc_123'],
    )

  @mock.patch('onetwo.evaluation.datasets.narrativeqa.tfds')
  def test_document_title_from_summary(self, mock_tfds):
    rows = [
        _make_narrativeqa_row(
            doc_id='D1',
            title='The Great Novel',
            url='http://example.com/novel',
        ),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = narrativeqa.NarrativeQADatasetLoader()
    documents = list(ot.run(loader.load_documents()))

    self.assertLen(documents, 1)
    self.assertEqual(documents[0].title, 'The Great Novel')
    self.assertEqual(documents[0].metadata['url'], 'http://example.com/novel')

  def test_registered_loaders(self):
    for split in ['train', 'validation', 'test']:
      loader = datasets_lib.get_dataset_loader(f'narrativeqa_{split}')
      self.assertIsInstance(loader, narrativeqa.NarrativeQADatasetLoader)
      self.assertEqual(loader.split, split)


if __name__ == '__main__':

  absltest.main()
