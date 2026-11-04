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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.backends import sentencepiece_tokenizer
from onetwo.core import batching
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import searchers
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import graphrag_prompts
from onetwo.stdlib.retrieval.graphrag import hipporag_index_specs
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing

Document = retrieval_data_structures.Document
HippoRAGDocumentIndexSpec = hipporag_index_specs.HippoRAGDocumentIndexSpec
SimpleRewritingHippoRAGDocumentIndexSpec = (
    hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec
)


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


class HippoRAGIndexSpecsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )
    llm_backend = backends_test_utils.LLMForTest(
        default_reply="{'entities': [], 'triples': []}"
    )
    llm_backend.register(register_tokenize=True)

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

  def test_spec_is_registered(self):
    self.assertIn(
        'simple_rewriting_hipporag', index_spec_registry.list_index_specs()
    )

  def test_spec_can_be_created_without_arguments(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec()
    self.assertEqual(
        spec.embed_backend_config, rag_defaults.default_embed_backend_config()
    )
    self.assertEqual(
        spec.generate_backend_config,
        rag_defaults.default_index_generate_backend_config(),
    )
    self.assertEqual(
        spec.chunker_config,
        chunker_registry.ChunkerConfig(),
    )

  @parameterized.named_parameters(
      (
          'chunking',
          {
              'chunker_config': chunker_registry.ChunkerConfig(
                  chunker_name='words',
                  kwargs={'max_words_per_chunk': 100},
              ),
          },
      ),
      (
          'no_chunking',
          {
              'chunker_config': chunker_registry.ChunkerConfig(
                  chunker_name='none'
              ),
              'format_str': 'Title: {title}\n{text}',
          },
      ),
  )
  def test_build_index(self, spec_kwargs):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=0.8,
        **spec_kwargs,
    )
    docs = [
        Document(doc_id='doc1', title='Title 1', content='Document one.'),
        Document(doc_id='doc2', title='Title 2', content='Document two.'),
    ]

    runner = local_index_runner.LocalIndexRunner()
    index = executing.run(runner.build_index(spec=spec, docs=docs))

    self.assertIsInstance(index, indexing.RewritingDocumentIndex)
    inner_index = index.inner_index
    self.assertIsInstance(inner_index, graphrag_indexing.HippoRAGDocumentIndex)
    if inner_index._entity_index is not None:
      self.assertIsInstance(
          inner_index._entity_index.searcher, searchers.BruteForceSearcher
      )
    self.assertEqual(index.num_docs, 2)

  def test_build_index_raises_for_bare_hipporag_spec(self):
    runner = local_index_runner.LocalIndexRunner()
    spec = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
    )
    with self.assertRaisesRegex(KeyError, 'HippoRAGDocumentIndexSpec'):
      executing.run(runner.build_index(spec=spec, docs=[]))

  def test_hipporag_index_round_trips_through_load_index(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 100},
        ),
    )
    docs = [
        Document(doc_id='doc1', content='This is document one.'),
        Document(doc_id='doc2', content='This is document two.'),
    ]
    out_dir = self.create_tempdir().full_path
    executing.run(
        index_building.build_index(spec=spec, docs=docs, output_dir=out_dir)
    )

    index = index_building.load_index(out_dir)

    self.assertIsInstance(index, indexing.RewritingDocumentIndex)
    self.assertIsInstance(
        index.inner_index, graphrag_indexing.HippoRAGDocumentIndex
    )
    self.assertEqual(index.num_docs, 2)

  def test_make_index_builds_default_extractor_from_settings(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
    )

    index = spec.make_index(runtime_kwargs={'api_key': 'key'})

    inner_index = index.inner_index
    assert isinstance(inner_index, graphrag_indexing.HippoRAGDocumentIndex)
    extractor = inner_index.entity_and_triples_extractor
    assert isinstance(extractor, graph_lib.GraphRAGEntityAndTriplesExtractor)
    # Same extractor as the one `make_index` used to build and inject.
    self.assertEqual(
        extractor.extraction_prompt,
        graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT.format(
            entity_types=', '.join(graphrag_prompts.DEFAULT_ENTITY_TYPES)
        ),
    )
    self.assertEqual(
        extractor.generate_backend_config, self.generate_backend_config
    )
    self.assertEqual(extractor.runtime_kwargs, {'api_key': 'key'})
    # The documents are passed as-is, so that multimodal content (e.g. PDFs)
    # reaches the model.
    self.assertIsInstance(
        extractor.document_formatter,
        document_formatting.NoFormatting,
    )

  def test_default_format_str_is_no_formatting(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
    )

    self.assertIsNone(spec.format_str)
    rewriters = getattr(
        spec.make_index(runtime_kwargs={}).rewriter, 'rewriters'
    )
    self.assertIsInstance(
        getattr(rewriters[1], 'formatter'), document_formatting.NoFormatting
    )

  def test_local_index_runner_forwards_ignore_flags(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        ignore_triples=False,
        ignore_similarity_edges=False,
    )
    docs = [Document(doc_id='doc1', title='Title 1', content='Content 1')]

    runner = local_index_runner.LocalIndexRunner()
    index = executing.run(runner.build_index(spec=spec, docs=docs))

    inner_index = getattr(index, 'inner_index')
    self.assertIsInstance(inner_index, graphrag_indexing.HippoRAGDocumentIndex)
    self.assertFalse(inner_index.ignore_triples)
    self.assertFalse(inner_index.ignore_similarity_edges)

  def test_hipporag_index_spec_serialization(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 100},
        ),
        similarity_threshold=0.8,
    )
    loaded_spec = SimpleRewritingHippoRAGDocumentIndexSpec.from_json(
        spec.to_json()
    )
    self.assertEqual(spec, loaded_spec)

  def test_raw_hipporag_index_spec_serialization(self):
    spec = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=0.85,
    )
    loaded_spec = HippoRAGDocumentIndexSpec.from_json(spec.to_json())
    self.assertEqual(spec, loaded_spec)

  def test_raw_hipporag_index_spec_str_representation(self):
    spec = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=0.85,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_embed_gemini-embedding-2_gen_test-model_sim_0.85__hash_',
        str(spec),
    )

  def test_hipporag_index_spec_str_representation(self):
    spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 512, 'overlap_window': 50},
        ),
        similarity_threshold=0.85,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_chunker_words-max_words_per_chunk=512-overlap_window=50_embed_gemini-embedding-2_gen_test-model_sim_0.85__hash_',
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
    raw_spec = HippoRAGDocumentIndexSpec(
        embed_backend_config=embed_config,
        generate_backend_config=gen_config,
        similarity_threshold=0.75,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_embed_CustomEmbedClass_gen_CustomGenClass_sim_0.75__hash_',
        str(raw_spec),
    )

    rewriting_spec = SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=embed_config,
        generate_backend_config=gen_config,
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 256, 'overlap_window': 10},
        ),
        similarity_threshold=0.75,
    )
    self.assertIn(
        'HippoRAGDocumentIndex_chunker_words-max_words_per_chunk=256-overlap_window=10_embed_CustomEmbedClass_gen_CustomGenClass_sim_0.75__hash_',
        str(rewriting_spec),
    )

  def test_hipporag_index_spec_ignore_flags_and_hash_differentiation(self):
    spec_default = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
    )
    spec_include_triples = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        ignore_triples=False,
    )
    spec_include_sim = HippoRAGDocumentIndexSpec(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        ignore_similarity_edges=False,
    )

    self.assertTrue(spec_default.ignore_triples)
    self.assertTrue(spec_default.ignore_similarity_edges)
    self.assertFalse(spec_include_triples.ignore_triples)
    self.assertFalse(spec_include_sim.ignore_similarity_edges)
    self.assertNotEqual(
        spec_default.get_stable_hash(), spec_include_triples.get_stable_hash()
    )
    self.assertNotEqual(
        spec_default.get_stable_hash(), spec_include_sim.get_stable_hash()
    )

    loaded_spec = HippoRAGDocumentIndexSpec.from_json(
        spec_include_triples.to_json()
    )
    self.assertEqual(spec_include_triples, loaded_spec)
    self.assertFalse(loaded_spec.ignore_triples)


if __name__ == '__main__':
  absltest.main()
