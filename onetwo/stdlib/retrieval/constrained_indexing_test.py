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

import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import constrained_indexing
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document


# Mock llm.embed to be async
@executing.make_executable(copy_self=False)
async def mock_embed_fn(
    text: str, task_type: str | None = None, **kwargs
) -> np.ndarray:
  """Mock async embed function."""
  del text, task_type, kwargs
  embedding = mock_embed_fn.embeddings[mock_embed_fn.call_count]  # pyrefly: ignore[missing-attribute]
  mock_embed_fn.call_count += 1  # pyrefly: ignore[missing-attribute]
  return embedding


class ConstrainedEmbeddingBasedIndexTest(absltest.TestCase):
  """Tests for ConstrainedEmbeddingBasedIndex."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        doc1_embed,
        doc2_embed,
        doc3_embed,
        query_embed,
        query_embed,
    ]

    index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
        discrete_field_extractors={'title': lambda doc: doc.title}
    )
    ot.run(index.add_docs(docs))

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
            index(query, constraints=constraints, max_results=3)
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
            index(query, constraints=constraints, max_results=3, min_score=0.85)
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
    mock_embed_fn.embeddings = doc_embeds + [query_embed]  # pyrefly: ignore[missing-attribute]

    def extract_external_keys(doc):
      return doc.metadata.get('external_keys', [])

    index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
        discrete_field_extractors={'external_keys': extract_external_keys}
    )
    ot.run(index.add_docs(docs))

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
          index(query, constraints=constraints, max_results=4)
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
    mock_embed_fn.embeddings = doc_embeds + 2 * [query_embed]  # pyrefly: ignore[missing-attribute]

    def extract_external_keys(doc):
      return doc.metadata.get('external_keys', [])

    with self.subTest('build_index_with_external_keys'):
      index = constrained_indexing.ConstrainedEmbeddingBasedIndex(
          discrete_field_extractors={'external_keys': extract_external_keys}
      )
      ot.run(index.add_docs(docs))

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
            index(query, constraints=constraints, max_results=4)
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
            index(query, constraints=constraints, max_results=4)
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
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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

    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
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
    ot.run(index.add_docs(docs))

    with self.subTest('retrieve_doc_indices_and_scores'):
      original_search = inner_index.retrieve_doc_indices_and_scores
      with mock.patch.object(
          inner_index, 'retrieve_doc_indices_and_scores', wraps=original_search
      ) as wrapped_search:
        results = ot.run(index.retrieve_doc_indices_and_scores('query'))
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
        score = ot.run(index.retrieve_doc_score('query', 0))
        wrapped_score.assert_called_once_with('query', 0)
        np.testing.assert_allclose(score, 0.9)


class RewritingConstrainedEmbeddingBasedDocumentIndexSaveLoadTest(
    absltest.TestCase
):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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

    class FakeEmbedderBackend(backends_test_utils.LLMForTest):

      def __init__(self, **kwargs):
        self.kwargs = kwargs

    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = (
        FakeEmbedderBackend
    )

    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = constrained_indexing.RewritingConstrainedEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=constrained_indexing.ConstrainedEmbeddingBasedDocumentIndex(
            embed_backend_config=backend_registry.EmbedBackendConfig(
                backend_class_name='FakeEmbedderBackend',
                model_name='MyModel',
                kwargs={'api_key': 'secret'},
            ),
            discrete_field_extractors={
                'external_keys': (
                    lambda doc: doc.metadata.get('external_keys', [])
                    if doc.metadata
                    else []
                )
            },
        ),
    )
    docs = [
        Document(doc_id='a', content='A', metadata={'external_keys': ['a']}),
        Document(doc_id='b', content='B', metadata={'external_keys': ['b']}),
    ]
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
    ]
    executing.run(index.add_docs(docs))

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
      executing.run(new_index.load(self.temp_dir))

    with self.subTest('verify_correct_attributes'):
      self.assertEqual(new_index.inner_index._docs, docs)
      self.assertEqual(
          new_index.inner_index._doc_indices_by_discrete_value,
          {'external_keys': {'a': [0], 'b': [1]}},
      )

    with self.subTest('get_doc_after_load'):
      self.assertEqual(new_index.get_doc('b'), docs[1])

    with self.subTest('individual_attributes_match'):
      self.assertEqual(
          new_index.inner_index._doc_indices_by_discrete_value,
          index.inner_index._doc_indices_by_discrete_value,
          '_doc_indices_by_discrete_value mismatch',
      )

      self.assertIsNotNone(new_index.inner_index.embed_backend_config)
      self.assertEqual(
          new_index.inner_index.embed_backend_config.backend_class_name,
          'FakeEmbedderBackend',
      )
      self.assertEqual(
          new_index.inner_index.embed_backend_config.model_name, 'MyModel'
      )
      self.assertEqual(
          new_index.inner_index.embed_backend_config.kwargs,
          {'api_key': 'secret'},
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
    mock_embed_fn.embeddings = [np.array([0.1, 0.2])]  # pyrefly: ignore[missing-attribute]
    executing.run(index.add_docs(docs))
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
    executing.run(new_index.load(self.temp_dir))

    with self.subTest('rename_with_defaultdict_works'):
      new_index.rename_discrete_field_key('external_keys', 'old', 'new')
      index_for_field = new_index.inner_index._doc_indices_by_discrete_value[
          'external_keys'
      ]
      self.assertNotIn('old', index_for_field)
      self.assertIn('new', index_for_field)
      self.assertEqual(index_for_field['new'], [0])

      new_index.rename_discrete_field_key('external_keys', 'new', 'another')
      new_index.rename_discrete_field_key('external_keys', 'another', 'new')
      self.assertIn('new', index_for_field)
      self.assertEqual(index_for_field['new'], [0])


if __name__ == '__main__':
  absltest.main()
