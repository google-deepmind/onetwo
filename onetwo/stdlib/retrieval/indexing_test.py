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

import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import searchers

Document = retrieval_data_structures.Document


# Mock llm.embed to be async
@executing.make_executable(copy_self=False)
async def mock_embed_fn(
    content: str | content_lib.ChunkList,
    task_type: str | None = None,
    title: str | None = None,
    **kwargs,
) -> np.ndarray:
  """Mock async embed function."""
  del content, task_type, title, kwargs
  embedding = mock_embed_fn.embeddings[mock_embed_fn.call_count]  # pyrefly: ignore[missing-attribute]
  mock_embed_fn.call_count += 1  # pyrefly: ignore[missing-attribute]
  return embedding


mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]


class NaiveIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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
      ot.run(index.add_docs(docs))
      self.assertEqual(index.num_docs, 3)

    with self.subTest('AddMoreDocuments'):
      more_docs = [
          Document(doc_id='doc4', content='This is document four.'),
          Document(doc_id='doc5', content='This is document five.'),
      ]
      ot.run(index.add_docs(more_docs))
      self.assertEqual(index.num_docs, 5)

    with self.subTest('RetrieveDocuments'):
      query = 'find document one'
      retrieved_docs = ot.run(index(query, max_results=2))

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
    ot.run(index.add_docs(docs))

    with self.subTest('DeterministicShuffle'):
      query = 'some query'
      # We expect the same order for the same query.
      res1 = list(ot.run(index(query)))
      res2 = list(ot.run(index(query)))
      self.assertEqual(res1, res2)
      self.assertCountEqual(res1, docs)

    with self.subTest('MaxResultsShuffle'):
      # Check max_results behavior: takes first K then shuffles.
      # Since we added doc1, doc2, doc3 in order, max_results=2 should return
      # doc1 and doc2 (shuffled), but never doc3.
      res_limited = list(ot.run(index('query', max_results=2)))
      self.assertLen(res_limited, 2)
      expected_subset = {'doc1', 'doc2'}
      self.assertEqual({d.doc_id for d in res_limited}, expected_subset)


class EmbeddingBasedIndexTest(absltest.TestCase):
  """Tests for EmbeddingBasedIndex (non-constrained)."""

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
    self.patcher = mock.patch('onetwo.builtins.llm.embed', new=mock_embed_fn)
    self.mock_llm_embed = self.patcher.start()
    self.addCleanup(self.patcher.stop)

  def test_retrieve_basic(self):
    """Tests basic vector search without constraints."""
    docs = [
        Document(doc_id='doc1', content='hello'),
        Document(doc_id='doc2', content='world'),
    ]

    doc1_embed = np.array([0.9, 0.1])
    doc2_embed = np.array([0.1, 0.9])
    query_embed = np.array([1.0, 0.0])

    mock_embed_fn.embeddings = [doc1_embed, doc2_embed, query_embed]  # pyrefly: ignore[missing-attribute]

    index = indexing.EmbeddingBasedIndex()
    ot.run(index.add_docs(docs))

    query = 'hello'
    results = list(ot.run(index(query, max_results=1)))
    self.assertLen(results, 1)
    self.assertEqual(results[0].doc_id, 'doc1')

  def test_retrieve_doc_score_matches_searcher_scores(self):
    """Tests that retrieve_doc_score scores like the searcher does."""
    docs = [Document(doc_id='doc1', content='hello')]
    doc_embed = np.array([3.0, 4.0])
    query_embed = np.array([1.0, 0.0])
    for normalize_embeds, expected_score in ((False, 3.0), (True, 0.6)):
      with self.subTest(f'normalize_embeds={normalize_embeds}'):
        # Order: doc (add_docs), query (retrieve), query (retrieve_doc_score).
        mock_embed_fn.embeddings = [doc_embed, query_embed, query_embed]  # pyrefly: ignore[missing-attribute]
        mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
        index = indexing.EmbeddingBasedIndex(
            searcher=searchers.BruteForceSearcher(
                normalize_embeds=normalize_embeds
            )
        )
        ot.run(index.add_docs(docs))

        [(_, searcher_score)] = ot.run(
            index.retrieve_doc_indices_and_scores('hello')
        )
        score = ot.run(index.retrieve_doc_score('hello', 0))

        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected_score)
        self.assertAlmostEqual(score, searcher_score)

  def test_get_doc_and_get_doc_embedding(self):
    docs = [
        Document(doc_id='doc1', content='hello'),
        Document(doc_id='doc2', content='world'),
    ]
    more_docs = [Document(doc_id='doc3', content='again')]
    doc_embeds = [np.array([0.9, 0.1]), np.array([0.1, 0.9]), np.array([1, 1])]
    mock_embed_fn.embeddings = doc_embeds  # pyrefly: ignore[missing-attribute]
    index = indexing.EmbeddingBasedIndex()
    ot.run(index.add_docs(docs))
    ot.run(index.add_docs(more_docs))

    with self.subTest('get_doc'):
      for doc in docs + more_docs:
        self.assertIs(index.get_doc(doc.doc_id), doc)

    with self.subTest('get_docs_by_ids_preserves_order'):
      self.assertEqual(
          index.get_docs_by_ids(['doc3', 'doc1']), [more_docs[0], docs[0]]
      )

    with self.subTest('get_doc_embedding'):
      for doc, embed in zip(docs + more_docs, doc_embeds):
        np.testing.assert_array_equal(
            index.get_doc_embedding(doc.doc_id), embed
        )

    with self.subTest('missing_id_raises_key_error'):
      with self.assertRaises(KeyError):
        index.get_doc('missing')
      with self.assertRaises(KeyError):
        index.get_doc_embedding('missing')

    with self.subTest('destroy_index_clears_lookup'):
      index.destroy_index()
      with self.assertRaises(KeyError):
        index.get_doc('doc1')

  def test_get_doc_with_duplicate_ids_returns_first(self):
    """Tests that docs sharing a doc_id are allowed; lookups return the first."""
    embeds = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    mock_embed_fn.embeddings = embeds  # pyrefly: ignore[missing-attribute]
    index = indexing.EmbeddingBasedIndex()
    first = Document(doc_id='doc1', content='first')
    within_batch = Document(doc_id='doc1', content='within batch')
    across_batches = Document(doc_id='doc1', content='across batches')
    ot.run(index.add_docs([first, within_batch]))
    ot.run(index.add_docs([across_batches]))

    self.assertEqual(index.num_docs, 3)
    for _ in range(2):  # Repeated lookups are stable.
      self.assertIs(index.get_doc('doc1'), first)
      np.testing.assert_array_equal(index.get_doc_embedding('doc1'), embeds[0])

  def test_add_docs_with_duplicate_ids_logs_warning(self):
    """Tests that docs with an already used doc_id are indexed with a warning."""
    mock_embed_fn.embeddings = [np.array([1.0, 0.0])] * 6  # pyrefly: ignore[missing-attribute]
    index = indexing.EmbeddingBasedIndex()

    with self.subTest('unique_ids_no_warning'):
      with self.assertNoLogs(level='WARNING'):
        ot.run(
            index.add_docs([
                Document(doc_id='doc1', content='a'),
                Document(doc_id='doc2', content='b'),
            ])
        )

    with self.subTest('duplicates_within_batch_and_of_existing_docs'):
      with self.assertLogs(level='WARNING') as logs:
        ot.run(
            index.add_docs([
                Document(doc_id='doc1', content='c'),  # Existing doc_id.
                Document(doc_id='doc3', content='d'),
                Document(doc_id='doc3', content='e'),  # Within batch.
                Document(doc_id='doc4', content='f'),
            ])
        )
      self.assertLen(logs.records, 1)
      self.assertIn('Added 2 documents', logs.output[0])
      self.assertIn("['doc1', 'doc3']", logs.output[0])

    with self.subTest('all_docs_indexed'):
      self.assertEqual(index.num_docs, 6)
      self.assertEqual(index.get_doc('doc1').content, 'a')
      self.assertEqual(index.get_doc('doc3').content, 'd')

  def test_get_doc_does_not_update_row_by_id(self):
    """Tests that lookups only read `_row_by_id`, which `add_docs` keeps up to date."""
    mock_embed_fn.embeddings = [np.array([1.0, 0.0])]  # pyrefly: ignore[missing-attribute]
    index = indexing.EmbeddingBasedIndex()
    doc1 = Document(doc_id='doc1', content='hello')
    ot.run(index.add_docs([doc1]))
    self.assertIs(index.get_doc('doc1'), doc1)

    # Docs appended to `_docs` without `add_docs` / `load` are not looked up.
    doc2 = Document(doc_id='doc2', content='world')
    index._docs.append(doc2)  # pylint: disable=protected-access
    with self.assertRaises(KeyError):
      index.get_doc('doc2')
    index._update_row_by_id()  # pylint: disable=protected-access
    self.assertIs(index.get_doc('doc2'), doc2)

  def test_localized_backend_initialization(self):
    """Tests that localized backend arguments correctly lazily configure the backend."""

    class FakeEmbedderBackend(backends_test_utils.LLMForTest):

      def __init__(self, embed_model_name: str | None = None, **kwargs):
        super().__init__()
        self.embed_model_name = embed_model_name
        self.kwargs = kwargs

    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = (
        FakeEmbedderBackend
    )

    index = indexing.EmbeddingBasedIndex(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend',
            model_name='fake_model',
            kwargs={'some_param': 'some_value'},
        )
    )

    backend = index.embedder_backend
    self.assertIsInstance(backend, FakeEmbedderBackend)
    self.assertEqual(backend.embed_model_name, 'fake_model')
    self.assertEqual(backend.kwargs, {'some_param': 'some_value'})

  def test_localized_backend_execution(self):
    """Tests that localized backend correctly intercepts embed calls."""
    self.patcher.stop()

    class FakeEmbedderBackend(backends_test_utils.EmbedderForTest):

      def __init__(self, model_name: str | None = None, **kwargs):
        super().__init__()
        self.called_embed = False

      @executing.make_executable(copy_self=False)
      async def embed(self, content: str | content_lib.ChunkList, **kwargs):
        self.called_embed = True
        return np.array([0.5, 0.5])

    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend2'] = (
        FakeEmbedderBackend
    )

    index = indexing.EmbeddingBasedIndex(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend2',
        )
    )

    docs = [Document(doc_id='doc1', content='test')]
    ot.run(index.add_docs(docs))

    backend = index.embedder_backend
    self.assertIsInstance(backend, FakeEmbedderBackend)
    self.assertTrue(backend.called_embed)
    self.patcher.start()


class EmbeddingBasedDocumentIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
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
    ot.run(index.add_docs(docs))

    # Retrieve the top 2 documents (this triggers the 4th embed call).
    query = 'find document one'
    retrieved_docs = ot.run(index(query, max_results=2))

    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)

    self.assertEqual(retrieved_list[0].doc_id, 'doc1')
    self.assertEqual(retrieved_list[1].doc_id, 'doc2')

    self.assertEqual(mock_embed_fn.call_count, 4)  # pyrefly: ignore[missing-attribute]

    # Each of these will reembed the query
    score_zero = ot.run(index.retrieve_doc_score(query, 0))
    score_one = ot.run(index.retrieve_doc_score(query, 1))
    score_two = ot.run(index.retrieve_doc_score(query, 2))

    self.assertEqual(score_zero, 0.9)
    self.assertEqual(score_one, 0.8)
    self.assertEqual(score_two, 0.1)

    # Retrieve docs with scores
    docs_and_scores = list(
        ot.run(index.retrieve_with_scores(query, max_results=2, min_score=0.9))
    )
    self.assertLen(docs_and_scores, 1)
    self.assertEqual(docs_and_scores[0][0].doc_id, 'doc1')
    self.assertEqual(docs_and_scores[0][1], 0.9)

    index.destroy_index()


# Tests replicating the behavior of the Chunking-based Indexes but using
# Rewriting-based Indexes.
class RewritingEmbeddingBasedIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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

    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
    ]

    inner_index = indexing.EmbeddingBasedIndex()
    index = indexing.RewritingEmbeddingBasedIndex(
        inner_index=inner_index, rewriter=HelloRewriter()
    )
    ot.run(index.add_docs(docs))

    chunks = list(index.get_docs())
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0].content, 'Hello world')
    self.assertEqual(chunks[1].content, 'Hello there')
    self.assertEqual(chunks[0].doc_id, 'doc1_chunk')
    self.assertEqual(chunks[1].doc_id, 'doc2_chunk')


class RewritingEmbeddingBasedDocumentIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
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

    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        doc1_embed,
        doc2_embed,
        doc3_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
        query_embed,
    ]

    ot.run(index.add_docs(docs))

    query = 'find document one'
    retrieved_docs = ot.run(index(query, max_results=2))

    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)

    self.assertEqual(retrieved_list[0].doc_id, 'doc1')
    self.assertEqual(retrieved_list[1].doc_id, 'doc2')

    self.assertEqual(mock_embed_fn.call_count, 4)  # pyrefly: ignore[missing-attribute]

    score_zero = ot.run(index.retrieve_doc_score(query, 0))
    score_one = ot.run(index.retrieve_doc_score(query, 1))
    score_two = ot.run(index.retrieve_doc_score(query, 2))

    self.assertEqual(score_zero, 0.9)
    self.assertEqual(score_one, 0.8)
    self.assertEqual(score_two, 0.1)

    docs_and_scores = list(
        ot.run(index.retrieve_with_scores(query, max_results=2, min_score=0.9))
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
    mock_embed_fn.embeddings = [np.array([0.9, 0.1]) for _ in range(20)]  # pyrefly: ignore[missing-attribute]
    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.ChunkByMaxTokens(max_tokens_per_chunk=5)
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(index.create_index(corpus_name='test_corpus', docs=docs))
    num_docs = len(list(index.get_docs()))
    retrieved_docs = ot.run(index('find document one', max_results=None))
    with self.subTest('num_docs_and_retrieved_docs_match'):
      self.assertLen(retrieved_docs, num_docs)

  def test_save_load_roundtrip(self):
    class FakeEmbedderBackend(backends_test_utils.LLMForTest):

      def __init__(self, embed_model_name: str | None = None, **kwargs):
        super().__init__()
        self.embed_model_name = embed_model_name
        self.kwargs = kwargs

    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = (
        FakeEmbedderBackend
    )
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )

    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=chunking.NoChunking()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=indexing.EmbeddingBasedDocumentIndex(
            embed_backend_config=backend_registry.EmbedBackendConfig(
                backend_class_name='FakeEmbedderBackend',
                model_name='MyModel',
                kwargs={'reply': 'val1'},
            )
        ),
    )
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
    ]

    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.9, 0.1]),
        np.array([0.8, 0.2]),
    ]

    ot.run(index.add_docs(docs))

    index.save(self.temp_dir)
    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(new_index.load(self.temp_dir))
    self.assertEqual(new_index.inner_index._docs, index.inner_index._docs)
    self.assertEqual(
        len(new_index.inner_index._doc_embeddings),
        len(index.inner_index._doc_embeddings),
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
        new_index.inner_index.embed_backend_config.kwargs, {'reply': 'val1'}
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

    mock_embed_fn.embeddings = [doc1_embed, doc2_embed]  # pyrefly: ignore[missing-attribute]
    ot.run(index.add_docs(docs))

    index.save(self.temp_dir)

    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=rewriter)
    ot.run(new_index.load(self.temp_dir))

    mock_embed_fn.embeddings = [query_embed]  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
    retrieved_docs = ot.run(new_index('find document one', max_results=2))
    retrieved_list = list(retrieved_docs)
    self.assertLen(retrieved_list, 2)
    self.assertEqual(retrieved_list[0].doc_id, 'doc1')

  def test_retrieve_original_documents(self):
    class SemicolonChunker(chunking.Chunker):

      @executing.make_executable(copy_self=False)
      async def __call__(self, document: Document) -> list[Document]:  # pyrefly: ignore[bad-override]
        parts = [
            p.strip() for p in str(document.content).split(';') if p.strip()
        ]
        chunks = []
        for i, part in enumerate(parts):
          chunks.append(
              Document(
                  doc_id=f'{document.doc_id}_chunk_{i}',
                  title=document.title,
                  content=part,
                  metadata={
                      retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID:
                      (
                          document.doc_id
                      ),
                      retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER: (
                          i + 1
                      ),
                      retrieval_data_structures.
                      METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS: len(
                          parts
                      ),
                  },
              )
          )
        return chunks

    rewriter = corpus_rewriting.ChunkingCorpusRewriter(
        chunker=SemicolonChunker()
    )
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        preserve_original_docs=True,
        retrieval_target=indexing.RetrievalTarget.ORIGINAL,
    )

    original_doc1 = Document(
        doc_id='doc1',
        title='Doc 1 Title',
        content='Doc 1 part A; Doc 1 part B; Doc 1 part C',
    )
    original_doc2 = Document(
        doc_id='doc2',
        title='Doc 2 Title',
        content='Doc 2 part A; Doc 2 part B',
    )
    original_doc3 = Document(
        doc_id='doc3',
        title='Doc 3 Title',
        content='Doc 3 part A; Doc 3 part B',
    )

    # 3 chunks for doc1, 2 for doc2, 2 for doc3 = 7 chunks total
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.95, 0.05]),
        np.array([0.90, 0.10]),
        np.array([0.85, 0.15]),
        np.array([0.80, 0.20]),
        np.array([0.75, 0.25]),
        np.array([0.10, 0.90]),
        np.array([0.05, 0.95]),
        # Query embeddings:
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]

    ot.run(index.add_docs([original_doc1, original_doc2, original_doc3]))
    self.assertEqual(index.num_docs, 7)  # 7 chunks in inner index
    self.assertLen(index._original_docs, 3)

    query = 'part A'
    # With max_results=4, retrieve top 4 chunks (3 from doc1, 1 from doc2) -> 2
    # unique original docs
    retrieved_originals = list(
        ot.run(
            index.retrieve(
                query,
                max_results=4,
            )
        )
    )
    self.assertLen(retrieved_originals, 2)
    self.assertEqual(retrieved_originals[0].doc_id, 'doc1')
    self.assertEqual(retrieved_originals[1].doc_id, 'doc2')
    self.assertEqual(
        retrieved_originals[0].content,
        'Doc 1 part A; Doc 1 part B; Doc 1 part C',
    )

    # With max_results=2, retrieve top 2 chunks (both from doc1) -> 1 unique
    # original doc
    retrieved_single = list(
        ot.run(
            index.retrieve(
                query,
                max_results=2,
            )
        )
    )
    self.assertLen(retrieved_single, 1)
    self.assertEqual(retrieved_single[0].doc_id, 'doc1')

  def _build_semicolon_chunking_index(
      self, retrieval_target: indexing.RetrievalTarget
  ) -> indexing.RewritingEmbeddingBasedDocumentIndex:
    """Returns an index over doc1 ('1A; 1B') and doc2 ('2A'), chunked on ';'."""

    class SemicolonChunker(chunking.Chunker):

      @executing.make_executable(copy_self=False)
      async def __call__(self, document: Document) -> list[Document]:  # pyrefly: ignore[bad-override]
        parts = [
            p.strip() for p in str(document.content).split(';') if p.strip()
        ]
        return [
            Document(
                doc_id=f'{document.doc_id}_chunk_{i}',
                content=part,
                metadata={
                    retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID: (
                        document.doc_id
                    ),
                },
            )
            for i, part in enumerate(parts)
        ]

    # Chunk embeddings (in add_docs order), followed by query embeddings.
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.9, 0.1]),
        np.array([0.8, 0.2]),
        np.array([0.1, 0.9]),
    ] + [np.array([1.0, 0.0])] * 2
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(
            chunker=SemicolonChunker()
        ),
        retrieval_target=retrieval_target,
        preserve_original_docs=True,
    )
    ot.run(
        index.add_docs([
            Document(doc_id='doc1', content='1A; 1B'),
            Document(doc_id='doc2', content='2A'),
        ])
    )
    return index

  def test_get_doc_rewritten_target_uses_rewritten_ids(self):
    index = self._build_semicolon_chunking_index(
        indexing.RetrievalTarget.REWRITTEN
    )
    retrieved = list(ot.run(index.retrieve('q', max_results=3)))
    self.assertEqual(
        [d.doc_id for d in retrieved],
        ['doc1_chunk_0', 'doc1_chunk_1', 'doc2_chunk_0'],
    )

    with self.subTest('round_trip'):
      for doc in retrieved:
        self.assertEqual(index.get_doc(doc.doc_id), doc)

    with self.subTest('get_docs_by_ids_preserves_order'):
      self.assertEqual(
          index.get_docs_by_ids(['doc2_chunk_0', 'doc1_chunk_0']),
          [retrieved[2], retrieved[0]],
      )

    with self.subTest('original_id_raises_key_error'):
      with self.assertRaises(KeyError):
        index.get_doc('doc1')

    with self.subTest('retrieval_target_override'):
      original = indexing.RetrievalTarget.ORIGINAL
      self.assertEqual(
          index.get_doc('doc1', retrieval_target=original).doc_id, 'doc1'
      )
      self.assertEqual(
          [
              d.doc_id
              for d in index.get_docs_by_ids(
                  ['doc2', 'doc1'], retrieval_target=original
              )
          ],
          ['doc2', 'doc1'],
      )

    with self.subTest('get_doc_embedding'):
      np.testing.assert_array_equal(
          index.get_doc_embedding('doc1_chunk_1'), np.array([0.8, 0.2])
      )

  def test_get_doc_original_target_uses_original_ids(self):
    index = self._build_semicolon_chunking_index(
        indexing.RetrievalTarget.ORIGINAL
    )
    retrieved = list(ot.run(index.retrieve('q', max_results=3)))
    self.assertEqual([d.doc_id for d in retrieved], ['doc1', 'doc2'])

    with self.subTest('round_trip'):
      for doc in retrieved:
        self.assertIs(index.get_doc(doc.doc_id), doc)

    with self.subTest('get_docs_by_ids_preserves_order'):
      self.assertEqual(
          index.get_docs_by_ids(['doc2', 'doc1']),
          [retrieved[1], retrieved[0]],
      )

    with self.subTest('rewritten_id_raises_key_error'):
      with self.assertRaises(KeyError):
        index.get_doc('doc1_chunk_0')

    with self.subTest('retrieval_target_override'):
      rewritten = indexing.RetrievalTarget.REWRITTEN
      self.assertEqual(
          index.get_doc('doc1_chunk_0', retrieval_target=rewritten).content,
          '1A',
      )
      self.assertEqual(
          [
              d.doc_id
              for d in index.get_docs_by_ids(
                  ['doc2_chunk_0', 'doc1_chunk_1'], retrieval_target=rewritten
              )
          ],
          ['doc2_chunk_0', 'doc1_chunk_1'],
      )

    with self.subTest('get_doc_embedding_still_uses_rewritten_ids'):
      np.testing.assert_array_equal(
          index.get_doc_embedding('doc1_chunk_1'), np.array([0.8, 0.2])
      )

  def test_get_doc_original_target_requires_preserved_docs(self):
    mock_embed_fn.embeddings = [np.array([1.0, 0.0])]  # pyrefly: ignore[missing-attribute]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        retrieval_target=indexing.RetrievalTarget.ORIGINAL,
        preserve_original_docs=False,
    )
    ot.run(index.add_docs([Document(doc_id='doc1', content='hello')]))
    with self.assertRaisesRegex(ValueError, 'preserve_original_docs'):
      index.get_doc('doc1')
    # The rewritten ID space is still available via `retrieval_target`.
    self.assertEqual(
        index.get_doc(
            'doc1', retrieval_target=indexing.RetrievalTarget.REWRITTEN
        ).doc_id,
        'doc1',
    )


class RewritingEmbeddingBasedDocumentIndexSaveLoadTest(absltest.TestCase):
  """Tests the main save and load methods for chunks and embeds without constraints."""

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
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
    ]
    ot.run(index.add_docs(docs))
    index.save(self.temp_dir)

    new_index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=indexing.EmbeddingBasedDocumentIndex(),
    )
    ot.run(new_index.load(self.temp_dir))

    self.assertEqual(new_index.inner_index._docs, docs)
    self.assertLen(new_index.inner_index._doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_index.inner_index._doc_embeddings[0],
        index.inner_index._doc_embeddings[0],
    )

    with self.subTest('get_doc_after_load'):
      self.assertEqual(new_index.get_doc('b'), docs[1])
      np.testing.assert_array_equal(
          new_index.inner_index.get_doc_embedding('b'),
          index.inner_index.get_doc_embedding('b'),
      )

  def _build_and_save_index(
      self, *, preserve_original_docs: bool
  ) -> list[Document]:
    """Builds and saves a 2-doc index to `self.temp_dir`; returns its docs."""
    docs = [
        Document(doc_id='a', title='Title A', content='A'),
        Document(doc_id='b', title='Title B', content='B'),
    ]
    mock_embed_fn.embeddings = [  # pyrefly: ignore[missing-attribute]
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(
            chunker=chunking.NoChunking()
        ),
        preserve_original_docs=preserve_original_docs,
    )
    ot.run(index.add_docs(docs))
    index.save(self.temp_dir)
    return docs

  def _load_index(
      self, *, preserve_original_docs: bool
  ) -> indexing.RewritingEmbeddingBasedDocumentIndex:
    """Loads the index saved in `self.temp_dir`, targeting original docs."""
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(
            chunker=chunking.NoChunking()
        ),
        preserve_original_docs=preserve_original_docs,
        retrieval_target=indexing.RetrievalTarget.ORIGINAL,
    )
    ot.run(index.load(self.temp_dir))
    return index

  def test_save_load_original_docs_roundtrip(self):
    docs = self._build_and_save_index(preserve_original_docs=True)
    original_docs_path = epath.Path(self.temp_dir) / (
        indexing.ORIGINAL_DOCS_JSONL_NAME
    )

    with self.subTest('original_docs_file_written'):
      self.assertTrue(original_docs_path.exists())
      self.assertLen(original_docs_path.read_text().splitlines(), 2)

    new_index = self._load_index(preserve_original_docs=True)

    with self.subTest('original_docs_loaded'):
      self.assertEqual(
          new_index._original_docs,  # pylint: disable=protected-access
          {'a': docs[0], 'b': docs[1]},
      )

    with self.subTest('get_doc_original'):
      self.assertEqual(new_index.get_doc('b'), docs[1])

    with self.subTest('retrieve_original'):
      mock_embed_fn.embeddings = [np.array([0.0, 1.0])]  # pyrefly: ignore[missing-attribute]
      mock_embed_fn.call_count = 0  # pyrefly: ignore[missing-attribute]
      retrieved = list(ot.run(new_index.retrieve('B', max_results=1)))
      self.assertEqual(retrieved, [docs[1]])

  def test_save_without_preserve_original_docs_writes_no_file(self):
    self._build_and_save_index(preserve_original_docs=False)
    self.assertFalse(
        (epath.Path(self.temp_dir) / indexing.ORIGINAL_DOCS_JSONL_NAME).exists()
    )

  def test_load_without_original_docs_file(self):
    """Tests that indices saved without original docs still load."""
    self._build_and_save_index(preserve_original_docs=False)
    new_index = self._load_index(preserve_original_docs=True)
    self.assertEqual(new_index.inner_index.num_docs, 2)
    self.assertEmpty(new_index._original_docs)  # pylint: disable=protected-access

  def test_load_without_preserve_original_docs_still_loads_file(self):
    """Tests that an existing original docs file is loaded and enables them."""
    docs = self._build_and_save_index(preserve_original_docs=True)
    new_index = self._load_index(preserve_original_docs=False)
    self.assertEqual(new_index.inner_index.num_docs, 2)
    self.assertEqual(
        new_index._original_docs,  # pylint: disable=protected-access
        {'a': docs[0], 'b': docs[1]},
    )
    self.assertTrue(new_index.preserve_original_docs)
    self.assertEqual(new_index.get_doc('a'), docs[0])

  def test_unsupported_serializer_type_raises_error(self):
    index = indexing.EmbeddingBasedDocumentIndex(
        serializer_type='nonexistent_type'
    )
    docs = [Document(doc_id='a', content='A')]
    mock_embed_fn.embeddings = [np.array([0.1, 0.2])]  # pyrefly: ignore[missing-attribute]
    ot.run(index.add_docs(docs))
    with self.assertRaisesRegex(
        ValueError, 'Unsupported serializer type: nonexistent_type'
    ):
      index.save(self.temp_dir)


class RuntimeKwargsTest(absltest.TestCase):
  """Tests that `runtime_kwargs` (credentials) stay out of repr and equality."""

  def test_runtime_kwargs_not_in_repr(self):
    index = indexing.EmbeddingBasedDocumentIndex(
        runtime_kwargs={'api_key': 'SECRET_VALUE_123'}
    )
    self.assertNotIn('SECRET_VALUE_123', repr(index))

  def test_indexes_differing_only_in_runtime_kwargs_are_equal(self):
    self.assertEqual(
        indexing.EmbeddingBasedDocumentIndex(runtime_kwargs={'api_key': 'a'}),
        indexing.EmbeddingBasedDocumentIndex(runtime_kwargs={'api_key': 'b'}),
    )


if __name__ == '__main__':
  absltest.main()
