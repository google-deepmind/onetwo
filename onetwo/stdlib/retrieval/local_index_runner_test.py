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
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building  # pylint: disable=unused-import
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
          embedding_index_specs.EmbeddingBasedDocumentIndexSpec,
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
          embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_chunk_size': 100,
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
          embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_chunk_size': None,
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
          embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
          {
              'max_chunk_size': 100,
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
    kwargs = {'embed_backend_config': self.embed_backend_config}
    kwargs.update(spec_kwargs)
    runtime_kwargs = kwargs.pop('runtime_kwargs', {})
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
    index = ot.run(
        runner.build_index(spec=spec, docs=docs, runtime_kwargs=runtime_kwargs)
    )

    self.assertIsInstance(index, expected_index_cls)

    if hasattr(index, 'inner_index'):
      inner_index = getattr(index, 'inner_index')
    else:
      inner_index = index

    self.assertTrue(hasattr(inner_index, 'searcher'))
    self.assertEqual(inner_index.query_task_type, expected_query_task)
    self.assertEqual(inner_index.document_task_type, expected_doc_task)
    self.assertIsInstance(inner_index.searcher, expected_searcher_cls)
    searcher = cast(searchers.BruteForceSearcher, inner_index.searcher)
    self.assertEqual(searcher.normalize_embeds, expected_normalize_embeds)

    self.assertEqual(index.num_docs, expected_num_docs)

    self.assertEqual(inner_index.runtime_kwargs, runtime_kwargs)

  def test_build_index_from_jsonl_path(self):
    spec = embedding_index_specs.EmbeddingBasedDocumentIndexSpec(
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

  def test_index_spec_serialization(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        max_chunk_size=100,
        format_str='Title: {title}\nFacts: {text}',
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
            kwargs={'normalize_embeds': True},
        ),
    )

    json_str = spec.to_json()
    loaded_spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec.from_json(
        json_str
    )

    self.assertEqual(spec, loaded_spec)

  def test_index_spec_str_representation(self):
    spec = embedding_index_specs.EmbeddingBasedDocumentIndexSpec(
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
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
        ),
        max_chunk_size=512,
        overlap_window=50,
        format_str='{text}',
    )
    self.assertEqual(
        str(spec),
        'EmbeddingBasedDocumentIndex_chunk_strategy_words_chunk_size_512_overlap_window_50_gemini-embedding-2__hash_056842d8',
    )

  def test_simple_rewriting_index_spec_str_representation_tokens(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        query_task_type='CUSTOM_QUERY_TASK',
        document_task_type='CUSTOM_DOC_TASK',
        searcher_config=index_build_interface.SearcherConfig(
            class_name='BruteForceSearcher',
        ),
        max_chunk_size=512,
        chunking_strategy='tokens',
        overlap_window=50,
        format_str='{text}',
    )
    self.assertEqual(
        str(spec),
        'EmbeddingBasedDocumentIndex_chunk_strategy_tokens_chunk_size_512_overlap_window_50_gemini-embedding-2__hash_cf421895',
    )


if __name__ == '__main__':
  absltest.main()
