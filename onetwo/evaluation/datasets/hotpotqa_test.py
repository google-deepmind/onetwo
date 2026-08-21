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

"""Tests for hotpotqa module."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from onetwo import ot
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import hotpotqa


def _make_hotpotqa_row(
    question: str = 'What is the capital of France?',
    answer: str = 'Paris',
    row_id: str = 'test_id_1',
    row_type: str = 'bridge',
    level: str = 'easy',
    context_titles: list[str] | None = None,
    context_sentences: list[list[str]] | None = None,
) -> dict[str, Any]:
  """Creates a fake HotpotQA row mimicking TFDS numpy output."""
  if context_titles is None:
    context_titles = ['France', 'Paris']
  if context_sentences is None:
    context_sentences = [
        ['France is a country in Europe.', 'Its capital is Paris.'],
        [
            'Paris is the capital of France.',
            'It is known for the Eiffel Tower.',
        ],
    ]
  return {
      'id': row_id.encode('utf-8'),
      'question': question.encode('utf-8'),
      'answer': answer.encode('utf-8'),
      'type': row_type.encode('utf-8'),
      'level': level.encode('utf-8'),
      'supporting_facts': {
          'title': np.array([t.encode('utf-8') for t in ['France']]),
          'sent_id': np.array([1]),
      },
      'context': {
          'title': np.array([t.encode('utf-8') for t in context_titles]),
          'sentences': [
              np.array([s.encode('utf-8') for s in sents])
              for sents in context_sentences
          ],
      },
  }


class BuildDocumentsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.loader = hotpotqa.HotpotQADatasetLoader()

  @parameterized.named_parameters(
      dict(
          testcase_name='single_paragraph',
          titles=[b'Doc A'],
          sentences_list=[[b'Sentence 1.']],
          expected_count=1,
      ),
      dict(
          testcase_name='multiple_paragraphs',
          titles=[b'Doc A', b'Doc B', b'Doc C'],
          sentences_list=[
              [b'Sentence 1a.', b'Sentence 2a.'],
              [b'Sentence 1b.'],
              [b'Sentence 1c.', b'Sentence 2c.', b'Sentence 3c.'],
          ],
          expected_count=3,
      ),
      dict(
          testcase_name='empty_context',
          titles=[],
          sentences_list=[],
          expected_count=0,
      ),
  )
  def test_document_count(self, titles, sentences_list, expected_count):
    context = {
        'title': np.array(titles),
        'sentences': [np.array(sents) for sents in sentences_list],
    }
    documents = self.loader._build_documents_from_hotpotqa_context(context)
    self.assertLen(documents, expected_count)

  def test_document_fields(self):
    context = {
        'title': np.array([b'My Title']),
        'sentences': [np.array([b'First sentence.', b'Second sentence.'])],
    }
    documents = self.loader._build_documents_from_hotpotqa_context(context)
    self.assertLen(documents, 1)
    doc = documents[0]
    self.assertEqual(doc.doc_id, 'My Title')
    self.assertEqual(doc.title, 'My Title')
    self.assertEqual(doc.content, 'First sentence. Second sentence.')


class BuildExampleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.loader = hotpotqa.HotpotQADatasetLoader()

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          question='What is the capital of France?',
          answer='Paris',
      ),
      dict(
          testcase_name='unicode',
          question='What is café?',
          answer='A coffee shop',
      ),
  )
  def test_example_has_required_keys(self, question, answer):
    row = _make_hotpotqa_row(question=question, answer=answer)
    example = self.loader._build_example_from_hotpotqa_row(row)
    self.assertIn(datasets_lib.EXAMPLE_FIELD_QUESTION, example)
    self.assertIn(datasets_lib.EXAMPLE_FIELD_ANSWER, example)
    self.assertEqual(example[datasets_lib.EXAMPLE_FIELD_QUESTION], question)
    self.assertEqual(example[datasets_lib.EXAMPLE_FIELD_ANSWER], answer)

  @parameterized.named_parameters(
      dict(
          testcase_name='bridge_easy',
          row_id='abc123',
          row_type='bridge',
          level='easy',
      ),
      dict(
          testcase_name='comparison_hard',
          row_id='xyz789',
          row_type='comparison',
          level='hard',
      ),
  )
  def test_example_metadata(self, row_id, row_type, level):
    row = _make_hotpotqa_row(row_id=row_id, row_type=row_type, level=level)
    example = self.loader._build_example_from_hotpotqa_row(row)
    metadata = example[datasets_lib.EXAMPLE_FIELD_METADATA]
    self.assertEqual(metadata['id'], row_id)
    self.assertEqual(metadata['type'], row_type)
    self.assertEqual(metadata['level'], level)
    self.assertIn(datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS, metadata)


class HotpotQADatasetLoaderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_row',
          rows_kwargs=[
              dict(
                  question='Q1',
                  answer='A1',
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
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
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
              ),
              dict(
                  question='Q2',
                  answer='A2',
                  context_titles=['T2'],
                  context_sentences=[['S2.']],
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
  @mock.patch('onetwo.evaluation.datasets.hotpotqa.tfds')
  def test_load_returns_correct_example_counts(
      self, mock_tfds, rows_kwargs, expected_example_count
  ):
    rows = [_make_hotpotqa_row(**kw) for kw in rows_kwargs]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = hotpotqa.HotpotQADatasetLoader(name='distractor', split='train')
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, expected_example_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_row',
          rows_kwargs=[
              dict(
                  question='Q1',
                  answer='A1',
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
              ),
          ],
          expected_doc_count=1,
      ),
      dict(
          testcase_name='multiple_rows_same_doc',
          rows_kwargs=[
              dict(
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
              ),
              dict(
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
              ),
          ],
          expected_doc_count=1,
      ),
      dict(
          testcase_name='multiple_rows_different_docs',
          rows_kwargs=[
              dict(
                  context_titles=['T1'],
                  context_sentences=[['S1.']],
              ),
              dict(
                  context_titles=['T2'],
                  context_sentences=[['S2.']],
              ),
          ],
          expected_doc_count=2,
      ),
      dict(
          testcase_name='row_with_multiple_docs',
          rows_kwargs=[
              dict(
                  context_titles=['T1', 'T2', 'T3'],
                  context_sentences=[['S1.'], ['S2.'], ['S3.']],
              ),
          ],
          expected_doc_count=3,
      ),
  )
  @mock.patch('onetwo.evaluation.datasets.hotpotqa.tfds')
  def test_load_documents_returns_correct_counts(
      self, mock_tfds, rows_kwargs, expected_doc_count
  ):
    rows = [_make_hotpotqa_row(**kw) for kw in rows_kwargs]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = hotpotqa.HotpotQADatasetLoader()
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, expected_doc_count)

  @mock.patch('onetwo.evaluation.datasets.hotpotqa.tfds')
  def test_max_number_of_examples_limits_output(self, mock_tfds):
    rows = [
        _make_hotpotqa_row(question='Q1', answer='A1'),
        _make_hotpotqa_row(question='Q2', answer='A2'),
        _make_hotpotqa_row(question='Q3', answer='A3'),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = hotpotqa.HotpotQADatasetLoader(max_number_of_examples=2)
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)
    self.assertEqual(examples[0]['question'], 'Q1')  # pyrefly: ignore[bad-index]
    self.assertEqual(examples[1]['question'], 'Q2')  # pyrefly: ignore[bad-index]

  @mock.patch('onetwo.evaluation.datasets.hotpotqa.tfds')
  def test_max_number_of_documents_limits_documents_output(self, mock_tfds):
    rows = [
        _make_hotpotqa_row(context_titles=['T1'], context_sentences=[['S1.']]),
        _make_hotpotqa_row(context_titles=['T2'], context_sentences=[['S2.']]),
        _make_hotpotqa_row(context_titles=['T3'], context_sentences=[['S3.']]),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = hotpotqa.HotpotQADatasetLoader(
        split='train', max_number_of_documents=2
    )
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 2)
    doc_ids = [doc.doc_id for doc in documents]
    self.assertIn('T1', doc_ids)
    self.assertIn('T2', doc_ids)
    self.assertNotIn('T3', doc_ids)

  @mock.patch('onetwo.evaluation.datasets.hotpotqa.tfds')
  def test_tfds_load_arguments(self, mock_tfds):
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter([])

    loader = hotpotqa.HotpotQADatasetLoader(name='distractor', split='test')
    _ = ot.run(loader.load_examples())

    mock_tfds.load.assert_called_with(
        'huggingface:hotpot_qa/distractor', split='test'
    )

  def test_dataclass_defaults(self):
    loader = hotpotqa.HotpotQADatasetLoader()
    self.assertEqual(loader.name, 'fullwiki')
    self.assertEqual(loader.split, 'validation')
    self.assertIsNone(loader.max_number_of_examples)

  def test_dataclass_custom_values(self):
    loader = hotpotqa.HotpotQADatasetLoader(
        name='distractor', split='train', max_number_of_examples=10
    )
    self.assertEqual(loader.name, 'distractor')
    self.assertEqual(loader.split, 'train')
    self.assertEqual(loader.max_number_of_examples, 10)

  def test_registered_loaders(self):
    for split in ['train', 'validation', 'test']:
      loader = datasets_lib.get_dataset_loader(f'hotpotqa_{split}')
      self.assertIsInstance(loader, hotpotqa.HotpotQADatasetLoader)
      self.assertEqual(loader.split, split)


if __name__ == '__main__':
  absltest.main()
