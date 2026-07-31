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

# Copyright 2026 Google LLC
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

from typing import cast
from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import searchers

Document = retrieval_data_structures.Document


@batching.add_batching
class FakeEmbedder(backends_test_utils.EmbedderForTest):

  def __init__(self, embed_model_name: str | None = None, **kwargs):
    default_embedding = kwargs.pop('default_embedding', [])
    super().__init__(default_embedding=default_embedding)
    self.embed_model_name = embed_model_name
    self.extra_kwargs = kwargs


class LocalIndexRunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Register LLMForTest to handle tokenize/detokenize globally
    self.llm_backend = backends_test_utils.LLMForTest()
    self.llm_backend.register(register_tokenize=True)

    # Register a fake embedder backend
    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )
    self.embed_backend_config = backend_registry.EmbedBackendConfig(
        backend_class_name='FakeEmbedderBackend',
        model_name='gemini-embedding-2',
        kwargs={'default_embedding': [0.1, 0.2]},
    )

  @parameterized.named_parameters(
      (
          'standard_spec',
          index_build_interface.EmbeddingBasedDocumentIndexSpec,
          {},
          indexing.EmbeddingBasedDocumentIndex,
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          searchers.BruteForceSearcher,
          False,
          2,
      ),
      (
          'simple_rewriting_spec',
          index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_tokens_per_chunk': 100,
              'format_str': 'Title: {title}\nFacts: {text}',
          },
          indexing.RewritingEmbeddingBasedDocumentIndex,
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          searchers.BruteForceSearcher,
          False,
          2,
      ),
      (
          'simple_rewriting_spec_no_chunking',
          index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_tokens_per_chunk': None,
              'format_str': 'Title: {title}\nFacts: {text}',
          },
          indexing.RewritingEmbeddingBasedDocumentIndex,
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          searchers.BruteForceSearcher,
          False,
          2,
      ),
      (
          'custom_params',
          index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_tokens_per_chunk': 100,
              'format_str': 'Title: {title}\nFacts: {text}',
              'query_task_type': 'CUSTOM_QUERY_TASK',
              'document_task_type': 'CUSTOM_DOC_TASK',
              'searcher_config': index_build_interface.SearcherConfig(
                  class_name='BruteForceSearcher',
                  kwargs={'normalize_embeds': True},
              ),
              'runtime_kwargs': {'custom_key': 'custom_val'},
          },
          indexing.RewritingEmbeddingBasedDocumentIndex,
          'CUSTOM_QUERY_TASK',
          'CUSTOM_DOC_TASK',
          searchers.BruteForceSearcher,
          True,
          2,
      ),
  )
  def test_build_index(
      self,
      spec_cls,
      spec_kwargs,
      expected_index_cls,
      expected_query_task,
      expected_doc_task,
      expected_searcher_cls,
      expected_normalize_embeds,
      expected_num_docs,
  ):
    # Add embed_backend_config to spec_kwargs
    spec = spec_cls(
        embed_backend_config=self.embed_backend_config, **spec_kwargs
    )

    docs = [
        Document(
            doc_id='doc1',
            title='Title 1',
            content='This is document one.',
        ),
        Document(
            doc_id='doc2',
            title='Title 2',
            content='This is document two.',
        ),
    ]

    runner = local_index_runner.LocalIndexRunner()
    index = ot.run(runner.build_index(spec=spec, docs=docs))

    self.assertIsInstance(index, expected_index_cls)

    if hasattr(index, 'inner_index'):
      inner_index = getattr(index, 'inner_index')
    else:
      inner_index = index

    self.assertEqual(inner_index.query_task_type, expected_query_task)
    self.assertEqual(inner_index.document_task_type, expected_doc_task)
    self.assertIsInstance(inner_index.searcher, expected_searcher_cls)

    searcher = cast(searchers.BruteForceSearcher, inner_index.searcher)
    self.assertEqual(searcher.normalize_embeds, expected_normalize_embeds)
    self.assertEqual(index.num_docs, expected_num_docs)

    expected_runtime_kwargs = spec_kwargs.get('runtime_kwargs', {})
    self.assertEqual(inner_index.runtime_kwargs, expected_runtime_kwargs)

  def test_build_index_path_error(self):
    spec = index_build_interface.EmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
    )
    runner = local_index_runner.LocalIndexRunner()
    with self.assertRaisesRegex(
        ValueError, 'does not support loading documents from a path'
    ):
      ot.run(runner.build_index(spec=spec, docs='/path/to/docs'))

  def test_index_spec_serialization(self):
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        max_tokens_per_chunk=100,
        format_str='Title: {title}\nFacts: {text}',
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
            kwargs={'normalize_embeds': True},
        ),
    )

    json_str = spec.to_json()
    loaded_spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec.from_json(
        json_str
    )

    self.assertEqual(spec, loaded_spec)

  def test_index_spec_str_representation(self):
    spec = index_build_interface.EmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
        ),
    )
    self.assertEqual(
        str(spec),
        'EmbeddingBasedDocumentIndex_gemini-embedding-2__hash_fc6f82d9',
    )

  def test_simple_rewriting_index_spec_str_representation(self):
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
        ),
        max_tokens_per_chunk=512,
        overlap_window=50,
        format_str='{text}',
    )
    self.assertEqual(
        str(spec),
        'EmbeddingBasedDocumentIndex_chunk_size_512_overlap_window_50_gemini-embedding-2__hash_d95428be',
    )

  def test_index_spec_hash_stability(self):
    spec1 = index_build_interface.EmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        runtime_kwargs={'api_key': 'key1', 'threadpool_size': 10},
    )
    spec2 = index_build_interface.EmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        runtime_kwargs={'api_key': 'key2', 'threadpool_size': 20},
    )
    self.assertEqual(spec1.get_stable_hash(), spec2.get_stable_hash())


if __name__ == '__main__':
  absltest.main()
