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

from typing import Sequence
from unittest import mock

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.core import executing
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import indexing
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


async def mock_tokenize_fn(text: str) -> Sequence[int]:
  return [ord(c) for c in text]


@executing.make_executable  # pytype: disable=wrong-arg-types
async def mock_detokenize_fn(tokens: Sequence[int]) -> str:
  return ''.join([chr(token) for token in tokens])


mock_embed_fn.embeddings = []
mock_embed_fn.call_count = 0


class NaiveIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def testAddingAndRetrievingDocuments(self):
    index = indexing.NaiveIndex()

    with self.subTest('CreateInitialIndex'):
      docs = [
          Document(doc_id='doc1', content='This is document one.'),
          Document(doc_id='doc2', content='This is document two.'),
          Document(doc_id='doc3', content='This is document three.'),
      ]
      ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count
      self.assertEqual(index.num_docs, 3)

    with self.subTest('AddMoreDocuments'):
      more_docs = [
          Document(doc_id='doc4', content='This is document four.'),
          Document(doc_id='doc5', content='This is document five.'),
      ]
      ot.run(index.add_docs(more_docs))  # pytype: disable=wrong-arg-count
      self.assertEqual(index.num_docs, 5)

    with self.subTest('RetrieveDocuments'):
      query = 'find document one'
      retrieved_docs = ot.run(index(query, max_results=2))  # pytype: disable=wrong-arg-count

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 2)

      self.assertEqual(retrieved_list[0].doc_id, 'doc1')
      self.assertEqual(retrieved_list[1].doc_id, 'doc2')

      index.destroy_index()

  def testRetrieveDocumentsWithShuffle(self):
    index = indexing.NaiveIndex(shuffle_results=True)
    docs = [
        Document(doc_id='doc1', content='content1'),
        Document(doc_id='doc2', content='content2'),
        Document(doc_id='doc3', content='content3'),
    ]
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    with self.subTest('DeterministicShuffle'):
      query = 'some query'
      # We expect the same order for the same query.
      res1 = list(ot.run(index(query)))  # pytype: disable=wrong-arg-count
      res2 = list(ot.run(index(query)))  # pytype: disable=wrong-arg-count
      self.assertEqual(res1, res2)
      self.assertCountEqual(res1, docs)

    with self.subTest('MaxResultsShuffle'):
      # Check max_results behavior: takes first K then shuffles.
      # Since we added doc1, doc2, doc3 in order, max_results=2 should return
      # doc1 and doc2 (shuffled), but never doc3.
      res_limited = list(ot.run(index('query', max_results=2)))  # pytype: disable=wrong-arg-count
      self.assertLen(res_limited, 2)
      expected_subset = {'doc1', 'doc2'}
      self.assertEqual({d.doc_id for d in res_limited}, expected_subset)


class EmbeddingBasedIndexTest(absltest.TestCase):

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

    index = indexing.EmbeddingBasedIndex(
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
        Document(doc_id='doc1', content='Doc one', external_keys=['A']),
        Document(doc_id='doc2', content='Doc two', external_keys=['B']),
        Document(doc_id='doc3', content='Doc three', external_keys=['C']),
        Document(doc_id='doc4', content='Doc four', external_keys=['A', 'C']),
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
      return doc.external_keys

    index = indexing.EmbeddingBasedIndex(
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
        Document(doc_id='doc1', content='Doc one', external_keys=['A']),
        Document(doc_id='doc2', content='Doc two', external_keys=['B', 'C']),
        Document(doc_id='doc3', content='Doc three', external_keys=['C']),
        Document(doc_id='doc4', content='Doc four', external_keys=['A', 'C']),
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
      return doc.external_keys

    with self.subTest('build_index_with_external_keys'):
      index = indexing.EmbeddingBasedIndex(
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


class ChunkingEmbeddingBasedIndexTest(absltest.TestCase):

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
    ]

    inner_index = indexing.EmbeddingBasedIndex(
        discrete_field_extractors={'title': lambda doc: doc.title}
    )
    index = indexing.ChunkingEmbeddingBasedIndex(inner_index=inner_index)
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

  def test_custom_chunker(self):
    class HelloChunker:

      @executing.make_executable(copy_self=False)
      async def __call__(self, doc):
        return [
            Document(
                doc_id=f'{doc.doc_id}_chunk', content=f'Hello {doc.content}'
            )
        ]

    docs = [
        Document(doc_id='doc1', content='world'),
        Document(doc_id='doc2', content='there'),
    ]

    mock_embed_fn.embeddings = [
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
    ]

    inner_index = indexing.EmbeddingBasedIndex()
    index = indexing.ChunkingEmbeddingBasedIndex(
        inner_index=inner_index, chunker=HelloChunker()
    )
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    chunks = list(index.get_docs())
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0].content, 'Hello world')
    self.assertEqual(chunks[1].content, 'Hello there')
    self.assertEqual(chunks[0].doc_id, 'doc1_chunk')
    self.assertEqual(chunks[1].doc_id, 'doc2_chunk')


if __name__ == '__main__':
  absltest.main()
