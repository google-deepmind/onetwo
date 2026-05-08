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

import pickle
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
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
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)
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
  """Tests for EmbeddingBasedIndex (non-constrained)."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_retrieve_basic(self):
    """Tests basic vector search without constraints."""
    docs = [
        Document(doc_id='doc1', content='hello'),
        Document(doc_id='doc2', content='world'),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    mock_embed_fn.embeddings = [doc1_embed, doc2_embed, query_embed]

    index = indexing.EmbeddingBasedIndex()
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    query = 'hello'
    results = list(ot.run(index(query, max_results=1)))  # pytype: disable=wrong-arg-count
    self.assertLen(results, 1)
    self.assertEqual(results[0].doc_id, 'doc1')


class EmbeddingBasedDocumentIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def testAddingAndRetrievingDocuments(self):
    index = indexing.EmbeddingBasedDocumentIndex()
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
        Document(doc_id='doc3', content='This is document three.'),
    ]

    # Query embedding should be closest to doc1 and doc2.
    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.8, 0.2])
    doc3_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    # Order: doc1, doc2, doc3 (during add_docs), then query (during retrieve).
    # Then query three more times to compute distances.
    # Then query one more time during retrieve_doc_scores.
    mock_embed_fn.embeddings = [
        doc1_embed,
        doc2_embed,
        doc3_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
    ]

    # Add documents (triggers _build and the first 3 embed calls).
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    # Retrieve the top 2 documents (this triggers the 4th embed call).
    query = 'find document one'
    retrieved_docs = ot.run(index(query, max_results=2))  # pytype: disable=wrong-arg-count

    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)

    self.assertEqual(retrieved_list[0].doc_id, 'doc1')
    self.assertEqual(retrieved_list[1].doc_id, 'doc2')

    self.assertEqual(mock_embed_fn.call_count, 4)

    # Each of these will reembed the query
    score_zero = ot.run(index.retrieve_doc_score(query, 0))  # pytype: disable=wrong-arg-count
    score_one = ot.run(index.retrieve_doc_score(query, 1))  # pytype: disable=wrong-arg-count
    score_two = ot.run(index.retrieve_doc_score(query, 2))  # pytype: disable=wrong-arg-count

    self.assertEqual(score_zero, 0.9)
    self.assertEqual(score_one, 0.8)
    self.assertEqual(score_two, 0.1)

    # Retrieve docs with scores
    docs_and_scores = ot.run(
        index.retrieve_with_scores(query, max_results=2, min_score=0.9)  # pytype: disable=wrong-arg-count
    )  # pytype: disable=wrong-arg-count
    self.assertLen(docs_and_scores, 1)
    self.assertEqual(docs_and_scores[0][0].doc_id, 'doc1')
    self.assertEqual(docs_and_scores[0][1], 0.9)

    index.destroy_index()


# Tests replicating the behavior of the Chunking-based Indexes but using
# Rewriting-based Indexes.
class RewritingEmbeddingBasedIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_custom_rewriter(self):
    class HelloRewriter(corpus_rewriting.CorpusRewriter):

      @executing.make_executable(copy_self=False)
      async def __call__(self, corpus):
        results = []
        for doc in corpus:
          results.append(
              Document(
                  doc_id=f'{doc.doc_id}_chunk', content=f'Hello {doc.content}'
              )
          )
        return results

    docs = [
        Document(doc_id='doc1', content='world'),
        Document(doc_id='doc2', content='there'),
    ]

    mock_embed_fn.embeddings = [
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
    ]

    inner_index = indexing.EmbeddingBasedIndex()
    index = indexing.RewritingEmbeddingBasedIndex(
        inner_index=inner_index, rewriter=HelloRewriter()
    )
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    chunks = list(index.get_docs())
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0].content, 'Hello world')
    self.assertEqual(chunks[1].content, 'Hello there')
    self.assertEqual(chunks[0].doc_id, 'doc1_chunk')
    self.assertEqual(chunks[1].doc_id, 'doc2_chunk')


class RewritingEmbeddingBasedDocumentIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)
    self.addCleanup(mock.patch.stopall)
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def testAddingAndRetrievingDocuments(self):
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
        Document(doc_id='doc3', content='This is document three.'),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.8, 0.2])
    doc3_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    mock_embed_fn.embeddings = [
        doc1_embed,
        doc2_embed,
        doc3_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
    ]

    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    query = 'find document one'
    retrieved_docs = ot.run(index(query, max_results=2))  # pytype: disable=wrong-arg-count

    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)

    self.assertEqual(retrieved_list[0].doc_id, 'doc1')
    self.assertEqual(retrieved_list[1].doc_id, 'doc2')

    self.assertEqual(mock_embed_fn.call_count, 4)

    score_zero = ot.run(index.retrieve_doc_score(query, 0))  # pytype: disable=wrong-arg-count
    score_one = ot.run(index.retrieve_doc_score(query, 1))  # pytype: disable=wrong-arg-count
    score_two = ot.run(index.retrieve_doc_score(query, 2))  # pytype: disable=wrong-arg-count

    self.assertEqual(score_zero, 0.9)
    self.assertEqual(score_one, 0.8)
    self.assertEqual(score_two, 0.1)

    docs_and_scores = ot.run(
        index.retrieve_with_scores(query, max_results=2, min_score=0.9)  # pytype: disable=wrong-arg-count
    )
    self.assertLen(docs_and_scores, 1)
    self.assertEqual(docs_and_scores[0][0].doc_id, 'doc1')
    self.assertEqual(docs_and_scores[0][1], 0.9)

    index.destroy_index()

  def test_create_index_with_rewriter(self):
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
        Document(doc_id='doc3', content='This is document three.'),
    ]
    mock_embed_fn.embeddings = [np.array([0.9, 0.1]) for _ in range(20)]
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.ChunkByMaxTokens(max_tokens_per_chunk=5)
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(index.create_index(corpus_name='test_corpus', docs=docs))  # pytype: disable=wrong-keyword-args
    num_docs = len(list(index.get_docs()))
    retrieved_docs = ot.run(
        index('find document one', max_results=None)  # pytype: disable=wrong-arg-count
    )
    with self.subTest('num_docs_and_retrieved_docs_match'):
      self.assertLen(retrieved_docs, num_docs)

  def test_save_load_roundtrip(self):
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
    ]

    mock_embed_fn.embeddings = [
        np.array([0.9, 0.1]),
        np.array([0.8, 0.2]),
    ]

    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    index.save(self.temp_dir)
    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(new_index.load(self.temp_dir))  # pytype: disable=wrong-arg-count
    self.assertEqual(new_index.inner_index._docs, index.inner_index._docs)
    self.assertEqual(
        len(new_index.inner_index._doc_embeddings),
        len(index.inner_index._doc_embeddings),
    )

  def test_retrieve_after_load(self):
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.8, 0.2])
    query_embed = np.array([1.0, 0.0])

    mock_embed_fn.embeddings = [doc1_embed, doc2_embed]
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    index.save(self.temp_dir)

    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(new_index.load(self.temp_dir))  # pytype: disable=wrong-arg-count

    mock_embed_fn.embeddings = [query_embed]
    mock_embed_fn.call_count = 0
    retrieved_docs = ot.run(new_index('find document one', max_results=2))  # pytype: disable=wrong-arg-count
    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)
    self.assertEqual(retrieved_list[0].doc_id, 'doc1')


class RewritingEmbeddingBasedDocumentIndexSaveLoadTest(absltest.TestCase):
  """Tests the main save and load methods for chunks and embeds without constraints."""

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

  def test_save_load_roundtrip_basic(self):
    """Tests the main save and load methods for chunks and embeds without constraints."""
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=indexing.EmbeddingBasedDocumentIndex(),
    )
    docs = [
        Document(doc_id='a', content='A'),
        Document(doc_id='b', content='B'),
    ]
    mock_embed_fn.embeddings = [
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
    ]
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count
    index.save(self.temp_dir)

    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=indexing.EmbeddingBasedDocumentIndex(),
    )
    ot.run(new_index.load(self.temp_dir))  # pytype: disable=wrong-arg-count

    self.assertEqual(new_index.inner_index._docs, docs)
    self.assertLen(new_index.inner_index._doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_index.inner_index._doc_embeddings[0],
        index.inner_index._doc_embeddings[0],
    )


class PickleabilityTest(absltest.TestCase):
  """Tests that EmbeddingBasedIndex is pickle-able for distributed pipelines."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_populated_embedding_based_index_is_pickleable(self):
    """A populated EmbeddingBasedIndex should survive a pickle roundtrip."""
    docs = [
        Document(doc_id='doc1', content='hello'),
        Document(doc_id='doc2', content='world'),
    ]
    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.1, 0.9])
    mock_embed_fn.embeddings = [doc1_embed, doc2_embed]

    index = indexing.EmbeddingBasedIndex()
    ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count

    restored = pickle.loads(pickle.dumps(index))
    self.assertEqual(restored.num_docs, 2)
    restored_docs = list(restored.get_docs())
    self.assertEqual(restored_docs[0].doc_id, 'doc1')
    self.assertEqual(restored_docs[1].doc_id, 'doc2')

  def test_embedding_based_document_index_is_pickleable(self):
    """EmbeddingBasedDocumentIndex should also be pickle-able."""
    index = indexing.EmbeddingBasedDocumentIndex()
    restored = pickle.loads(pickle.dumps(index))
    self.assertEqual(restored.num_docs, 0)


if __name__ == '__main__':
  absltest.main()
