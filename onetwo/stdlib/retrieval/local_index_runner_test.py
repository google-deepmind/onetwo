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

import os
import tempfile
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.backends import sentencepiece_tokenizer
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


@batching.add_batching
class FakeLLM(backends_test_utils.LLMForTest):

  def __init__(self, model_name: str | None = None, **kwargs):
    kwargs.setdefault('default_reply', "{'entities': [], 'triples': []}")
    super().__init__(**kwargs)
    self.model_name = model_name


class LocalIndexRunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_sp_processor = mock.MagicMock()
    self.mock_sp_processor.EncodeAsIds.side_effect = lambda s: [1] * len(
        s.split()
    )
    self.mock_sp_processor.DecodeIds.side_effect = lambda ids: ' '.join(
        ['word'] * len(ids)
    )
    self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=self.mock_sp_processor,
        )
    )

    # Register LLMForTest to handle tokenize/detokenize globally
    self.llm_backend = backends_test_utils.LLMForTest(
        default_reply="{'entities': [], 'triples': []}"
    )
    self.llm_backend.register(register_tokenize=True)

    # Register a fake embedder and LLM backend
    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    backend_registry.BACKEND_REGISTRY['FakeLLMBackend'] = FakeLLM
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeLLMBackend', None
    )
    self.embed_backend_config = backend_registry.EmbedBackendConfig(
        backend_class_name='FakeEmbedderBackend',
        model_name='gemini-embedding-2',
        kwargs={'default_embedding': [0.1, 0.2]},
    )
    self.generate_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='FakeLLMBackend',
        model_name='test-model',
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
      (
          'hipporag_spec',
          index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec,
          {
              'max_tokens_per_chunk': 100,
              'similarity_threshold': 0.8,
          },
          indexing.RewritingEmbeddingBasedDocumentIndex,
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          searchers.BruteForceSearcher,
          False,
          2,
      ),
      (
          'hipporag_spec_no_chunking',
          index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec,
          {
              'max_tokens_per_chunk': None,
              'format_str': 'Title: {title}\nFacts: {text}',
              'similarity_threshold': 0.8,
          },
          indexing.RewritingEmbeddingBasedDocumentIndex,
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          searchers.BruteForceSearcher,
          False,
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
    kwargs = {'embed_backend_config': self.embed_backend_config}
    if issubclass(spec_cls, index_build_interface.HippoRAGDocumentIndexSpec):
      kwargs['generate_backend_config'] = self.generate_backend_config
    kwargs.update(spec_kwargs)
    spec = spec_cls(**kwargs)

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

    if isinstance(inner_index, graphrag_indexing.HippoRAGDocumentIndex):
      if inner_index._entity_index is not None:
        self.assertTrue(hasattr(inner_index._entity_index, 'searcher'))
        self.assertIsInstance(
            inner_index._entity_index.searcher, expected_searcher_cls
        )
    else:
      self.assertTrue(hasattr(inner_index, 'searcher'))
      self.assertEqual(inner_index.query_task_type, expected_query_task)
      self.assertEqual(inner_index.document_task_type, expected_doc_task)
      self.assertIsInstance(inner_index.searcher, expected_searcher_cls)
      searcher = cast(searchers.BruteForceSearcher, inner_index.searcher)
      self.assertEqual(searcher.normalize_embeds, expected_normalize_embeds)

    self.assertEqual(index.num_docs, expected_num_docs)

    expected_runtime_kwargs = spec_kwargs.get('runtime_kwargs', {})
    self.assertEqual(inner_index.runtime_kwargs, expected_runtime_kwargs)

  def test_build_index_from_jsonl_path(self):
    spec = index_build_interface.EmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
    )
    runner = local_index_runner.LocalIndexRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
      docs_path = os.path.join(temp_dir, 'docs.jsonl')
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
      with open(docs_path, 'w') as f:
        for doc in docs:
          f.write(doc.to_json() + '\n')

      index = ot.run(runner.build_index(spec=spec, docs=docs_path))
      self.assertEqual(index.num_docs, 2)

  def test_get_tokenizer_backend(self):
    runner = local_index_runner.LocalIndexRunner()
    tokenizer = runner._get_tokenizer_backend()
    self.assertIsInstance(
        tokenizer, sentencepiece_tokenizer.SentencePieceTokenizer
    )
    self.assertEqual(tokenizer.model_name, 'GEMINI3_G145')

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
        'EmbeddingBasedDocumentIndex_chunk_strategy_words_chunk_size_512_overlap_window_50_gemini-embedding-2__hash_30cbd990',
    )

  def test_simple_rewriting_index_spec_str_representation_tokens(self):
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
        ),
        max_tokens_per_chunk=512,
        chunking_strategy='tokens',
        overlap_window=50,
        format_str='{text}',
    )
    self.assertEqual(
        str(spec),
        'EmbeddingBasedDocumentIndex_chunk_strategy_tokens_chunk_size_512_overlap_window_50_gemini-embedding-2__hash_691921ed',
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

  def test_hipporag_index_spec_serialization(self):
    spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        max_tokens_per_chunk=100,
        similarity_threshold=0.8,
    )
    json_str = spec.to_json()
    loaded_spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec.from_json(
        json_str
    )
    self.assertEqual(spec, loaded_spec)

  def test_raw_hipporag_index_spec_serialization(self):
    spec = index_build_interface.HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=0.85,
    )
    json_str = spec.to_json()
    loaded_spec = index_build_interface.HippoRAGDocumentIndexSpec.from_json(
        json_str
    )
    self.assertEqual(spec, loaded_spec)

  def test_raw_hipporag_index_spec_str_representation(self):
    spec = index_build_interface.HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=0.85,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_embed_gemini-embedding-2_gen_test-model_sim_0.85__hash_',
        str(spec),
    )

  def test_hipporag_index_spec_str_representation(self):
    spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        max_tokens_per_chunk=512,
        overlap_window=50,
        similarity_threshold=0.85,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_chunk_512_overlap_50_embed_gemini-embedding-2_gen_test-model_sim_0.85__hash_',
        str(spec),
    )

  def test_hipporag_index_spec_str_representation_fallback(self):
    embed_config = backend_registry.EmbedBackendConfig(
        backend_class_name='CustomEmbedClass',
        model_name='',
    )
    gen_config = backend_registry.GenerateBackendConfig(
        backend_class_name='CustomGenClass',
        model_name='',
    )
    raw_spec = index_build_interface.HippoRAGDocumentIndexSpec(
        embed_backend_config=embed_config,
        generate_backend_config=gen_config,
        similarity_threshold=0.75,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_embed_CustomEmbedClass_gen_CustomGenClass_sim_0.75__hash_',
        str(raw_spec),
    )

    rewriting_spec = (
        index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec(
            embed_backend_config=embed_config,
            generate_backend_config=gen_config,
            max_tokens_per_chunk=256,
            overlap_window=10,
            similarity_threshold=0.75,
        )
    )
    self.assertIn(
        'HippoRAGDocumentIndex_chunk_256_overlap_10_embed_CustomEmbedClass_gen_CustomGenClass_sim_0.75__hash_',
        str(rewriting_spec),
    )


if __name__ == '__main__':
  absltest.main()
