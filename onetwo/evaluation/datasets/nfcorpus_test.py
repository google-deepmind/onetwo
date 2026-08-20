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

"""Tests for nfcorpus module."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import nfcorpus
from onetwo.stdlib.retrieval import retrieval_data_structures


def _make_nfcorpus_query_row(
    query_id: str = 'q_001',
    text: str = 'What is the treatment for hypertension?',
    title: str = '',
) -> dict[str, Any]:
  """Creates a fake NFCorpus query row mimicking TFDS numpy output."""
  row: dict[str, Any] = {
      '_id': query_id.encode('utf-8'),
      'text': text.encode('utf-8'),
      'title': title.encode('utf-8'),
  }
  return row


def _make_nfcorpus_qrels_row(
    query_id: str = 'q1',
    corpus_id: str = 'doc1',
    score: int = 1,
) -> dict[str, Any]:
  """Creates a fake NFCorpus qrels row mimicking TFDS numpy output."""
  return {
      'query-id': query_id.encode('utf-8'),
      'corpus-id': corpus_id.encode('utf-8'),
      'score': score,
  }


def _make_nfcorpus_corpus_row(
    corpus_id: str = 'doc_001',
    title: str = 'Hypertension Overview',
    text: str = 'Hypertension is high blood pressure.',
) -> dict[str, Any]:
  """Creates a fake NFCorpus corpus row mimicking TFDS numpy output."""
  return {
      '_id': corpus_id.encode('utf-8'),
      'title': title.encode('utf-8'),
      'text': text.encode('utf-8'),
  }


class BuildExampleFromRowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.loader = nfcorpus.NFCorpusDatasetLoader(split='test')

  def test_build_example_from_row(self):
    row = _make_nfcorpus_query_row(query_id='q1', text='Query 1')
    example = self.loader._build_example_from_row(
        row, golden_doc_scores={'MED-1': 1}
    )
    self.assertEqual(example[datasets_lib.EXAMPLE_FIELD_QUESTION], 'Query 1')
    metadata = example[datasets_lib.EXAMPLE_FIELD_METADATA]
    self.assertEqual(metadata['id'], 'q1')
    self.assertEqual(metadata['split'], 'test')
    self.assertEqual(
        metadata[datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_SCORES],
        {'MED-1': 1},
    )
    self.assertEqual(
        metadata[datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS],
        ['MED-1'],
    )


class BuildDocumentFromCorpusRowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.loader = nfcorpus.NFCorpusDatasetLoader(split='test')

  def test_build_document_from_corpus_row(self):
    row = _make_nfcorpus_corpus_row(
        corpus_id='doc1', title='Title 1', text='Text 1'
    )
    doc = self.loader._build_document_from_corpus_row(row)
    self.assertIsInstance(doc, retrieval_data_structures.Document)
    self.assertEqual(doc.doc_id, 'doc1')
    self.assertEqual(doc.title, 'Title 1')
    self.assertEqual(doc.content, 'Text 1')


class NFCorpusDatasetLoaderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_row',
          queries_kwargs=[dict(query_id='q1', text='Query 1')],
          qrels_kwargs=[dict(query_id='q1', corpus_id='doc1', score=1)],
          expected_example_count=1,
      ),
      dict(
          testcase_name='multiple_rows',
          queries_kwargs=[
              dict(query_id='q1', text='Query 1'),
              dict(query_id='q2', text='Query 2'),
          ],
          qrels_kwargs=[
              dict(query_id='q1', corpus_id='doc1', score=1),
              dict(query_id='q1', corpus_id='doc2', score=2),
              dict(query_id='q2', corpus_id='doc3', score=1),
              dict(query_id='q2', corpus_id='doc4', score=2),
          ],
          expected_example_count=2,
      ),
      dict(
          testcase_name='empty',
          queries_kwargs=[],
          qrels_kwargs=[],
          expected_example_count=0,
      ),
      dict(
          testcase_name='unmatched_qrels_filtered',
          queries_kwargs=[dict(query_id='q1', text='Query 1')],
          qrels_kwargs=[],
          expected_example_count=0,
      ),
  )
  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_load_returns_correct_example_counts(
      self, mock_tfds, queries_kwargs, qrels_kwargs, expected_example_count
  ):
    queries_rows = [_make_nfcorpus_query_row(**kw) for kw in queries_kwargs]
    qrels_rows = [_make_nfcorpus_qrels_row(**kw) for kw in qrels_kwargs]

    def mock_load(name, split, *args, **kwargs):
      del split, args, kwargs  # Unused.
      mock_ds = mock.MagicMock()
      mock_ds.name = name
      return mock_ds

    def mock_as_numpy(ds):
      if 'qrels' in getattr(ds, 'name', ''):
        return iter(qrels_rows)
      return iter(queries_rows)

    mock_tfds.load.side_effect = mock_load
    mock_tfds.as_numpy.side_effect = mock_as_numpy

    loader = nfcorpus.NFCorpusDatasetLoader(split='test')
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, expected_example_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_doc',
          rows_kwargs=[
              dict(corpus_id='doc1', title='Title 1', text='Text 1'),
          ],
          expected_doc_count=1,
      ),
      dict(
          testcase_name='multiple_docs',
          rows_kwargs=[
              dict(corpus_id='doc1', title='Title 1', text='Text 1'),
              dict(corpus_id='doc2', title='Title 2', text='Text 2'),
          ],
          expected_doc_count=2,
      ),
      dict(
          testcase_name='empty',
          rows_kwargs=[],
          expected_doc_count=0,
      ),
  )
  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_load_documents_returns_correct_counts(
      self, mock_tfds, rows_kwargs, expected_doc_count
  ):
    rows = [_make_nfcorpus_corpus_row(**kw) for kw in rows_kwargs]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = nfcorpus.NFCorpusDatasetLoader(split='test')
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, expected_doc_count)

  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_examples_are_evaluation_compatible(self, mock_tfds):
    fake_queries_rows = [
        _make_nfcorpus_query_row(query_id='q1', text='Query 1'),
        _make_nfcorpus_query_row(query_id='q2', text='Query 2'),
    ]
    fake_qrels_rows = [
        _make_nfcorpus_qrels_row(query_id='q1', corpus_id='doc1', score=1),
        _make_nfcorpus_qrels_row(query_id='q1', corpus_id='doc2', score=2),
    ]

    def mock_load(name, split, *args, **kwargs):
      del split, args, kwargs  # Unused.
      mock_ds = mock.MagicMock()
      mock_ds.name = name
      return mock_ds

    def mock_as_numpy(ds):
      if 'qrels' in getattr(ds, 'name', ''):
        return iter(fake_qrels_rows)
      return iter(fake_queries_rows)

    mock_tfds.load.side_effect = mock_load
    mock_tfds.as_numpy.side_effect = mock_as_numpy

    loader = nfcorpus.NFCorpusDatasetLoader(split='test')
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 1)
    example = list(examples)[0]
    self.assertEqual(example[datasets_lib.EXAMPLE_FIELD_QUESTION], 'Query 1')
    self.assertEqual(example[datasets_lib.EXAMPLE_FIELD_METADATA]['id'], 'q1')
    self.assertEqual(
        example[datasets_lib.EXAMPLE_FIELD_METADATA]['split'], 'test'
    )
    self.assertEqual(
        example[datasets_lib.EXAMPLE_FIELD_METADATA][
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_SCORES
        ],
        {'doc1': 1, 'doc2': 2},
    )
    self.assertEqual(
        example[datasets_lib.EXAMPLE_FIELD_METADATA][
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS
        ],
        ['doc1', 'doc2'],
    )

  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_documents_are_document_instances(self, mock_tfds):
    rows = [
        _make_nfcorpus_corpus_row(
            corpus_id='doc1', title='Title 1', text='Text 1'
        ),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = nfcorpus.NFCorpusDatasetLoader(split='test')
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 1)
    doc = list(documents)[0]
    self.assertIsInstance(doc, retrieval_data_structures.Document)
    self.assertEqual(doc.doc_id, 'doc1')
    self.assertEqual(doc.title, 'Title 1')
    self.assertEqual(doc.content, 'Text 1')

  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_deduplicates_documents(self, mock_tfds):
    rows = [
        _make_nfcorpus_corpus_row(
            corpus_id='doc1', title='Title 1', text='Text 1'
        ),
        _make_nfcorpus_corpus_row(
            corpus_id='doc1', title='Title 1 Dup', text='Text 1 Dup'
        ),
        _make_nfcorpus_corpus_row(
            corpus_id='doc2', title='Title 2', text='Text 2'
        ),
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(rows)

    loader = nfcorpus.NFCorpusDatasetLoader(split='test')
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 2)
    self.assertEqual([d.doc_id for d in documents], ['doc1', 'doc2'])

  @parameterized.named_parameters(
      dict(testcase_name='limit_1', max_n=1, expected_examples=1),
      dict(testcase_name='limit_2', max_n=2, expected_examples=2),
      dict(testcase_name='no_limit', max_n=None, expected_examples=3),
  )
  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_load_max_number_of_examples(
      self, mock_tfds, max_n, expected_examples
  ):
    fake_queries = [
        _make_nfcorpus_query_row(query_id=f'q_{i}') for i in range(3)
    ]
    fake_qrels = [_make_nfcorpus_qrels_row(query_id=f'q_{i}') for i in range(3)]

    def mock_load(name, split, *args, **kwargs):
      del split, args, kwargs  # Unused.
      mock_ds = mock.MagicMock()
      mock_ds.name = name
      return mock_ds

    def mock_as_numpy(ds):
      if 'qrels' in getattr(ds, 'name', ''):
        return iter(fake_qrels)
      return iter(fake_queries)

    mock_tfds.load.side_effect = mock_load
    mock_tfds.as_numpy.side_effect = mock_as_numpy

    loader = nfcorpus.NFCorpusDatasetLoader(
        split='test', max_number_of_examples=max_n
    )
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, expected_examples)

  @parameterized.named_parameters(
      dict(testcase_name='limit_1', max_n=1, expected_docs=1),
      dict(testcase_name='limit_2', max_n=2, expected_docs=2),
      dict(testcase_name='no_limit', max_n=None, expected_docs=3),
  )
  @mock.patch('onetwo.evaluation.datasets.nfcorpus.tfds')
  def test_load_max_number_of_documents(self, mock_tfds, max_n, expected_docs):
    fake_rows = [
        _make_nfcorpus_corpus_row(corpus_id=f'doc_{i}') for i in range(3)
    ]
    mock_tfds.load.return_value = mock.MagicMock()
    mock_tfds.as_numpy.return_value = iter(fake_rows)

    loader = nfcorpus.NFCorpusDatasetLoader(
        split='test', max_number_of_documents=max_n
    )
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, expected_docs)

  def test_dataclass_defaults(self):
    loader = nfcorpus.NFCorpusDatasetLoader()
    self.assertEqual(loader.split, 'test')
    self.assertIsNone(loader.max_number_of_examples)
    self.assertIsNone(loader.max_number_of_documents)

  def test_dataclass_custom_values(self):
    loader = nfcorpus.NFCorpusDatasetLoader(
        split='train',
        max_number_of_examples=5,
        max_number_of_documents=10,
    )
    self.assertEqual(loader.split, 'train')
    self.assertEqual(loader.max_number_of_examples, 5)
    self.assertEqual(loader.max_number_of_documents, 10)

  @parameterized.named_parameters(
      ('test', 'nfcorpus_test', 'test'),
      ('train', 'nfcorpus_train', 'train'),
      ('validation', 'nfcorpus_validation', 'validation'),
  )
  def test_registered_loaders(self, loader_name: str, expected_split: str):
    loader = datasets_lib.get_dataset_loader(loader_name)
    self.assertIsInstance(loader, nfcorpus.NFCorpusDatasetLoader)
    self.assertEqual(loader.split, expected_split)


if __name__ == '__main__':
  absltest.main()
