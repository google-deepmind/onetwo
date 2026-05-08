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

"""Tests for constrained indexing."""

import collections
import pickle
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import constrained_indexing
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document


# Mock llm.embed to be async
@executing.make_executable(copy_self=False)  # pytype: disable=wrong-arg-count
async def mock_embed_fn(text: str, task_type: str | None = None) -> np.ndarray:
  """Mock async embed function."""
  del text, task_type
  embedding = mock_embed_fn.embeddings[mock_embed_fn.call_count]
  mock_embed_fn.call_count += 1
  return embedding


class ConstrainedEmbeddingBasedIndexTest(absltest.TestCase):
  """Tests for ConstrainedEmbeddingBasedIndex."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_retrieve_constrained_duplicate_doc_ids(self):
    """Tests pre-filtering with duplicate doc_ids and ranking."""
    docs = [
        Document(
            doc_id='doc1', title='Geography', content='This is document one.'
        ),
        Document(
            doc_id='doc2',
            title='Geography',
            content='This is document two with same title.',
        ),
        Document(
            doc_id='doc3', title='History', content='This is document three.'
        ),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.8, 0.2])
    doc3_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    # Order: doc1, doc2, doc3 (during add_docs), then query (during retrieve).
    # Then query one more time during retrieve with min_score.
    mock_embed_fn.embeddings = [
        doc1_embed,
        doc2_embed,
        doc3_embed,
        query_embed,
        query_embed,
    ]

    index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
        discrete_field_extractors={'title': lambda doc: doc.title}
    )
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    constraints = constrained_retrieval.RetrievalConstraints(
        constraints=[
            constrained_retrieval.RetrievalConstraint(
                field_name='title',
                constraint_type=constrained_retrieval.RetrievalConstraintType.EQUALS,
                value='Geography',
            )
        ]
    )
    query = 'test query'

    with self.subTest('retrieve_constrained_documents_without_min_score'):
      original_search_batched = index.searcher.search_batched
      with mock.patch.object(
          index.searcher, 'search_batched', wraps=original_search_batched
      ) as wrapped_search_batched:
        retrieved_docs = ot.run(
            index(query, constraints=constraints, max_results=3)  # pytype: disable=wrong-arg-count
        )

        wrapped_search_batched.assert_called_once()
        _, mock_kwargs = wrapped_search_batched.call_args
        self.assertIn('doc_indices', mock_kwargs)
        self.assertCountEqual(mock_kwargs['doc_indices'], [0, 1])

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 2)
      # Check that the two documents with doc_id='doc1' are retrieved
      self.assertEqual(retrieved_list[0].content, 'This is document one.')
      self.assertEqual(
          retrieved_list[1].content,
          'This is document two with same title.',
      )

    with self.subTest('retrieve_constrained_documents_with_min_score'):
      original_search_batched = index.searcher.search_batched
      with mock.patch.object(
          index.searcher, 'search_batched', wraps=original_search_batched
      ) as wrapped_search_batched:
        retrieved_docs = ot.run(
            index(query, constraints=constraints, max_results=3, min_score=0.85)  # pytype: disable=wrong-arg-count
        )

        wrapped_search_batched.assert_called_once()
        _, mock_kwargs = wrapped_search_batched.call_args
        self.assertIn('doc_indices', mock_kwargs)
        self.assertCountEqual(mock_kwargs['doc_indices'], [0, 1])

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 1)
      self.assertCountEqual([d.doc_id for d in retrieved_list], ['doc1'])

  def test_retrieve_constrained_list_contains_any(self):
    """Tests pre-filtering with the LIST_CONTAINS_ANY constraint."""
    docs = [
        Document(
            doc_id='doc1', content='Doc one', metadata={'external_keys': ['A']}
        ),
        Document(
            doc_id='doc2', content='Doc two', metadata={'external_keys': ['B']}
        ),
        Document(
            doc_id='doc3',
            content='Doc three',
            metadata={'external_keys': ['C']},
        ),
        Document(
            doc_id='doc4',
            content='Doc four',
            metadata={'external_keys': ['A', 'C']},
        ),
    ]

    # Mock embeddings
    doc_embeds = [
        np.array([0.1, 0.9]),
        np.array([0.2, 0.8]),
        np.array([0.3, 0.7]),
        np.array([0.4, 0.6]),
    ]
    query_embed = np.array([0.15, 0.85])
    mock_embed_fn.embeddings = doc_embeds + [query_embed]

    def extract_external_keys(doc):
      return doc.metadata.get('external_keys', [])

    index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
        discrete_field_extractors={'external_keys': extract_external_keys}
    )
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    # Constraint: external_keys must contain 'A' or 'C'
    constraints = constrained_retrieval.RetrievalConstraints(
        constraints=[
            constrained_retrieval.RetrievalConstraint(
                field_name='external_keys',
                constraint_type=constrained_retrieval.RetrievalConstraintType.LIST_CONTAINS_ANY,
                value=['A', 'C'],
            )
        ]
    )
    query = 'test query'

    original_search_batched = index.searcher.search_batched
    with mock.patch.object(
        index.searcher, 'search_batched', wraps=original_search_batched
    ) as wrapped_search_batched:
      retrieved_docs = ot.run(
          index(query, constraints=constraints, max_results=4)  # pytype: disable=wrong-arg-count
      )

      wrapped_search_batched.assert_called_once()
      _, mock_kwargs = wrapped_search_batched.call_args
      self.assertIn('doc_indices', mock_kwargs)
      # Should search only indices 0, 2, 3 (doc1, doc3, doc4)
      self.assertCountEqual(mock_kwargs['doc_indices'], [0, 2, 3])

    retrieved_list = list(retrieved_docs)
    retrieved_ids = sorted([d.doc_id for d in retrieved_list])
    self.assertEqual(retrieved_ids, ['doc1', 'doc3', 'doc4'])

  def test_retrieve_constrained_list_contains_any_multiple_lists(self):
    """Tests pre-filtering with multiple LIST_CONTAINS_ANY for AND logic."""
    docs = [
        Document(
            doc_id='doc1', content='Doc one', metadata={'external_keys': ['A']}
        ),
        Document(
            doc_id='doc2',
            content='Doc two',
            metadata={'external_keys': ['B', 'C']},
        ),
        Document(
            doc_id='doc3',
            content='Doc three',
            metadata={'external_keys': ['C']},
        ),
        Document(
            doc_id='doc4',
            content='Doc four',
            metadata={'external_keys': ['A', 'C']},
        ),
    ]

    # Mock embeddings
    doc_embeds = [
        np.array([0.1, 0.9]),
        np.array([0.2, 0.8]),
        np.array([0.3, 0.7]),
        np.array([0.4, 0.6]),
    ]
    query_embed = np.array([1.0, 0.0])
    mock_embed_fn.embeddings = doc_embeds + 2 * [query_embed]

    def extract_external_keys(doc):
      return doc.metadata.get('external_keys', [])

    with self.subTest('build_index_with_external_keys'):
      index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
          discrete_field_extractors={'external_keys': extract_external_keys}
      )
      ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    # Constraint: external_keys must contain 'A' and 'C'
    constraints = constrained_retrieval.RetrievalConstraints(
        constraints=[
            constrained_retrieval.RetrievalConstraint(
                field_name='external_keys',
                constraint_type=constrained_retrieval.RetrievalConstraintType.LIST_CONTAINS_ANY,
                value=['A'],
            ),
            constrained_retrieval.RetrievalConstraint(
                field_name='external_keys',
                constraint_type=constrained_retrieval.RetrievalConstraintType.LIST_CONTAINS_ANY,
                value=['C'],
            ),
        ]
    )
    query = 'test query'

    with self.subTest('retrieve_constrained_documents_with_and_logic'):
      original_search_batched = index.searcher.search_batched
      with mock.patch.object(
          index.searcher, 'search_batched', wraps=original_search_batched
      ) as wrapped_search_batched:
        retrieved_docs = ot.run(
            index(query, constraints=constraints, max_results=4)  # pytype: disable=wrong-arg-count
        )

        wrapped_search_batched.assert_called_once()
        _, mock_kwargs = wrapped_search_batched.call_args
        self.assertIn('doc_indices', mock_kwargs)
        # Should search only index 3 (doc4) which contains both 'A' and 'C'
        self.assertCountEqual(mock_kwargs['doc_indices'], [3])

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 1)
      self.assertEqual(retrieved_list[0].doc_id, 'doc4')

    with self.subTest('retrieve_constrained_documents_with_renamed_key'):
      index.rename_discrete_field_key(
          field_name='external_keys', old_key='B', new_key='A'
      )
      original_search_batched = index.searcher.search_batched
      with mock.patch.object(
          index.searcher, 'search_batched', wraps=original_search_batched
      ) as wrapped_search_batched:
        retrieved_docs = ot.run(
            index(query, constraints=constraints, max_results=4)  # pytype: disable=wrong-arg-count
        )

        wrapped_search_batched.assert_called_once()
        _, mock_kwargs = wrapped_search_batched.call_args
        self.assertIn('doc_indices', mock_kwargs)
        # Should now search doc2 and doc4 as 'B' is renamed to 'A'
        self.assertCountEqual(mock_kwargs['doc_indices'], [1, 3])

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 2)
      self.assertEqual(retrieved_list[0].doc_id, 'doc4')
      self.assertEqual(retrieved_list[1].doc_id, 'doc2')


class ConstrainedRewritingEmbeddingBasedIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_delegates_to_inner_index(self):
    docs = [
        Document(
            doc_id='doc1', title='Geography', content='This is document one.'
        ),
        Document(
            doc_id='doc2',
            title='History',
            content='This is document two.',
        ),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    mock_embed_fn.embeddings = [
        doc1_embed,
        doc2_embed,
        query_embed,
        query_embed,
        query_embed,
    ]

    inner_index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
        discrete_field_extractors={'title': lambda doc: doc.title}
    )
    # Using ChunkingCorpusRewriter with default chunker to mirror ChunkingIndex
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = constrained_indexing.RewritingConstrainedEmbeddingBasedIndex(
        inner_index=inner_index, rewriter=rewriter
    )
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    with self.subTest('retrieve_doc_indices_and_scores'):
      original_search = inner_index.retrieve_doc_indices_and_scores
      with mock.patch.object(
          inner_index, 'retrieve_doc_indices_and_scores', wraps=original_search
      ) as wrapped_search:
        results = ot.run(index.retrieve_doc_indices_and_scores('query'))  # pytype: disable=wrong-arg-count
        wrapped_search.assert_called_once()
        self.assertLen(results, 2)

    with self.subTest('rename_discrete_field_key'):
      original_rename = inner_index.rename_discrete_field_key
      with mock.patch.object(
          inner_index, 'rename_discrete_field_key', wraps=original_rename
      ) as wrapped_rename:
        index.rename_discrete_field_key('title', 'Geography', 'Science')
        wrapped_rename.assert_called_once_with('title', 'Geography', 'Science')

    with self.subTest('retrieve_doc_score'):
      original_score = inner_index.retrieve_doc_score
      with mock.patch.object(
          inner_index, 'retrieve_doc_score', wraps=original_score
      ) as wrapped_score:
        score = ot.run(index.retrieve_doc_score('query', 0))  # pytype: disable=wrong-arg-count
        wrapped_score.assert_called_once_with('query', 0)
        np.testing.assert_allclose(score, 0.9)


class RewritingConstrainedEmbeddingBasedDocumentIndexSaveLoadTest(
    absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def test_save_load_roundtrip_full(self):
    """Tests the main save and load methods for chunks and embeds."""
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = constrained_indexing.RewritingConstrainedEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=constrained_indexing.ConstrainedEmbeddingBasedDocumentIndex(
            discrete_field_extractors={
                'external_keys': (
                    lambda doc: doc.metadata.get('external_keys', [])
                    if doc.metadata
                    else []
                )
            }
        ),
    )
    docs = [
        Document(doc_id='a', content='A', metadata={'external_keys': ['a']}),
        Document(doc_id='b', content='B', metadata={'external_keys': ['b']}),
    ]
    mock_embed_fn.embeddings = [
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
    ]
    executing.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    with self.subTest('save_index'):
      index.save(self.temp_dir)

    with self.subTest('load_index'):
      new_index = constrained_indexing.RewritingConstrainedEmbeddingBasedDocumentIndex(
          rewriter=rewriter,
          inner_index=constrained_indexing.ConstrainedEmbeddingBasedDocumentIndex(
              discrete_field_extractors={
                  'external_keys': (
                      lambda doc: doc.metadata.get('external_keys', [])
                      if doc.metadata
                      else []
                  )
              }
          ),
      )
      executing.run(new_index.load(self.temp_dir))  # pytype: disable=wrong-arg-count

    with self.subTest('verify_correct_attributes'):
      self.assertEqual(new_index.inner_index._docs, docs)
      self.assertEqual(
          new_index.inner_index._doc_indices_by_discrete_value,  # pytype: disable=attribute-error
          {'external_keys': {'a': [0], 'b': [1]}},
      )

    with self.subTest('individual_attributes_match'):
      self.assertEqual(
          new_index.inner_index._doc_indices_by_discrete_value,  # pytype: disable=attribute-error
          index.inner_index._doc_indices_by_discrete_value,  # pytype: disable=attribute-error
          '_doc_indices_by_discrete_value mismatch',
      )

      self.assertEqual(
          new_index.inner_index._docs, index.inner_index._docs, 'Docs mismatch'
      )
      for i in range(len(index.inner_index._doc_embeddings)):
        np.testing.assert_array_equal(
            new_index.inner_index._doc_embeddings[i],
            index.inner_index._doc_embeddings[i],
            f'Embedding at index {i} mismatch',
        )

  def test_rename_discrete_field_key_type_handling(self):
    """Tests the behavior of rename_discrete_field_key based on index type."""
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = constrained_indexing.RewritingConstrainedEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=constrained_indexing.ConstrainedEmbeddingBasedDocumentIndex(
            discrete_field_extractors={
                'external_keys': (
                    lambda doc: doc.metadata.get('external_keys', [])
                    if doc.metadata
                    else []
                )
            }
        ),
    )
    docs = [
        Document(doc_id='a', content='A', metadata={'external_keys': ['old']}),
    ]
    mock_embed_fn.embeddings = [np.array([0.1, 0.2])]
    executing.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count
    index.save(self.temp_dir)

    new_index = constrained_indexing.RewritingConstrainedEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=constrained_indexing.ConstrainedEmbeddingBasedDocumentIndex(
            discrete_field_extractors={
                'external_keys': (
                    lambda doc: doc.metadata.get('external_keys', [])
                    if doc.metadata
                    else []
                )
            }
        ),
    )
    executing.run(new_index.load(self.temp_dir))  # pytype: disable=wrong-arg-count

    with self.subTest('rename_with_defaultdict_works'):
      new_index.rename_discrete_field_key('external_keys', 'old', 'new')
      index_for_field = new_index.inner_index._doc_indices_by_discrete_value[  # pytype: disable=attribute-error
          'external_keys'
      ]
      self.assertNotIn('old', index_for_field)
      self.assertIn('new', index_for_field)
      self.assertEqual(index_for_field['new'], [0])

      new_index.rename_discrete_field_key('external_keys', 'new', 'another')
      new_index.rename_discrete_field_key('external_keys', 'another', 'new')
      self.assertIn('new', index_for_field)
      self.assertEqual(index_for_field['new'], [0])


class ConstrainedPickleabilityTest(absltest.TestCase):
  """Tests that ConstrainedEmbeddingBasedIndex is pickle-able."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_default_list_map_is_pickleable(self):
    """Verifies the _default_list_map helper pickles correctly."""
    d = constrained_indexing._default_list_map()
    d['key'].append(42)
    restored = pickle.loads(pickle.dumps(d))
    self.assertEqual(restored['key'], [42])
    self.assertIsInstance(restored, collections.defaultdict)
    self.assertEqual(restored['nonexistent'], [])

  def test_default_discrete_value_map_is_pickleable(self):
    """Verifies the _default_discrete_value_map helper pickles correctly."""
    d = constrained_indexing._default_discrete_value_map()
    d['field']['value'].append(0)
    restored = pickle.loads(pickle.dumps(d))
    self.assertEqual(restored['field']['value'], [0])
    self.assertIsInstance(restored, collections.defaultdict)
    self.assertIsInstance(restored['new_field'], collections.defaultdict)
    self.assertEqual(restored['new_field']['new_value'], [])

  def test_empty_embedding_based_index_is_pickleable(self):
    """An empty EmbeddingBasedIndex should survive a pickle roundtrip."""
    index = constrained_indexing.ConstrainedEmbeddingBasedIndex()
    restored = pickle.loads(pickle.dumps(index))
    self.assertEqual(restored.num_docs, 0)
    self.assertIsInstance(
        restored._doc_indices_by_discrete_value, collections.defaultdict
    )

  def test_doc_indices_by_discrete_value_is_pickleable(self):
    """The _doc_indices_by_discrete_value defaultdict survives pickling."""
    index = constrained_indexing.ConstrainedEmbeddingBasedIndex()
    index._doc_indices_by_discrete_value['title']['Geography'].append(0)
    index._doc_indices_by_discrete_value['title']['History'].append(1)

    restored = pickle.loads(pickle.dumps(index))
    self.assertIn('title', restored._doc_indices_by_discrete_value)
    self.assertEqual(
        restored._doc_indices_by_discrete_value['title']['Geography'], [0]
    )
    self.assertEqual(
        restored._doc_indices_by_discrete_value['title']['History'], [1]
    )


if __name__ == '__main__':
  absltest.main()
