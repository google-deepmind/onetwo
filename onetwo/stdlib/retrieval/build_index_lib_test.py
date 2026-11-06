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

from collections.abc import Iterable
import dataclasses
import json
import os
import re
import types
from unittest import mock
import wave

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.backends import sentencepiece_tokenizer
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import routing_chunking  # pylint: disable=unused-import  # Registers 'routing'.
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval.graphrag import hipporag_index_specs
import pypdf

KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec

_DATASET_NAME = 'build_index_lib_test_dataset'
_FAKE_EMBEDDER = backend_registry.EmbedBackendConfig(
    backend_class_name='FakeEmbedderBackend', model_name='fake-model'
)


@batching.add_batching
class FakeEmbedder(backends_test_utils.EmbedderForTest):

  def __init__(self, embed_model_name: str | None = None, **kwargs):
    del kwargs  # Runtime kwargs (e.g. max_retries) are not used.
    super().__init__(default_embedding=[0.1, 0.2])
    self.embed_model_name = embed_model_name

  def _get_embed_reply(self, content):
    # Always returns the default embedding. The base implementation looks the
    # content up in a dict, which fails for (unhashable) multimodal ChunkLists.
    self.contents.append(content)
    return self.default_embedding


@batching.add_batching
class _KeywordEmbedder(backends_test_utils.EmbedderForTest):
  """Embeds text as the counts of a few keywords in it."""

  def __init__(self, **kwargs):
    del kwargs  # Unused (e.g. credentials passed as runtime kwargs).
    super().__init__()

  def _get_embed_reply(self, content):
    words = re.findall('[a-z]+', str(content).lower())
    return [words.count(w) for w in ('paris', 'madrid')] + [0.1]


@registry.register('build_index_lib_test.SentenceChunker')
@dataclasses.dataclass(kw_only=True)
class _SentenceChunker(chunking.TextChunker):
  """Splits text into sentences."""

  @ot_executing.make_executable(copy_self=False)
  async def _chunk_text(self, text: str) -> Iterable[str]:
    return [f'{s.strip()}.' for s in text.split('.') if s.strip()]


@dataclasses.dataclass(kw_only=True)
class _UnregisteredChunker(chunking.NoChunking):
  pass


def _custom_index(
    chunker: chunking.Chunker,
) -> indexing.RewritingEmbeddingBasedDocumentIndex:
  return indexing.RewritingEmbeddingBasedDocumentIndex(
      rewriter=corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
      inner_index=indexing.EmbeddingBasedDocumentIndex(
          embed_backend_config=backend_registry.EmbedBackendConfig(
              backend_class_name='KeywordEmbedderForTest'
          )
      ),
  )


class _CustomLoader(datasets_lib.DatasetLoader):
  """An unregistered dataset loader."""

  @ot_executing.make_executable
  async def load_examples(self):
    return []

  @ot_executing.make_executable
  async def load_documents(self):
    return [
        retrieval_data_structures.Document(doc_id=f'c{i}', content=f'doc {i}')
        for i in range(1, 4)
    ]


class BuildIndexLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    corpus_path = os.path.join(self.create_tempdir().full_path, 'corpus.jsonl')
    with open(corpus_path, 'w') as f:
      f.write(json.dumps({'doc_id': 'd1', 'content': 'Paris is in France.'}))
      f.write('\n')
      f.write(json.dumps({'doc_id': 'd2', 'content': 'Madrid is in Spain.'}))
      f.write('\n')
    datasets_lib.register_dataset_loader(
        _DATASET_NAME,
        lambda **kwargs: datasets_lib.JsonlDatasetLoader(
            corpus_path=corpus_path, **kwargs
        ),
    )
    self.addCleanup(datasets_lib.unregister_dataset_loader, _DATASET_NAME)

    self.enter_context(
        keyword_index_for_test.registered_keyword_index_spec(name='keyword')
    )

    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )
    self.output_dir = self.create_tempdir().full_path

  def test_build_index_object(self):
    self.enter_context(
        mock.patch.dict(
            backend_registry.BACKEND_REGISTRY,
            {'KeywordEmbedderForTest': _KeywordEmbedder},
        )
    )

    out_dir = build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        _custom_index(_SentenceChunker()),
        output=self.output_dir,
        runtime_kwargs={'api_key': 'secret'},
    )
    loaded = index_building.load_index(
        out_dir, runtime_kwargs={'api_key': 'secret'}
    )

    with self.subTest('saves_index_config_without_credentials'):
      with open(
          os.path.join(out_dir, config_serialization.INDEX_CONFIG_FILENAME)
      ) as f:
        self.assertNotIn('secret', f.read())
    with self.subTest('loads_the_index'):
      [doc] = ot_executing.run(loaded.retrieve('madrid', max_results=1))
      self.assertEqual(doc.text, 'Madrid is in Spain.')
    with self.subTest('unregistered_class_raises_before_loading_corpus'):
      corpus = mock.MagicMock()
      with self.assertRaisesRegex(TypeError, 'register its class'):
        build_index_lib.build_index(
            corpus,
            _custom_index(_UnregisteredChunker()),
            output=self.output_dir,
        )
      corpus.load_documents.assert_not_called()
    with self.subTest('runtime_kwargs_on_the_index_raise'):
      index = _custom_index(_SentenceChunker())
      index.inner_index.runtime_kwargs = {'api_key': 'secret'}
      with self.assertRaisesRegex(ValueError, 'Pass runtime kwargs'):
        build_index_lib.build_index(
            datasets.get_dataset_loader(_DATASET_NAME),
            index,
            output=self.output_dir,
        )

  def test_build_custom_spec_saves_index_and_spec(self):
    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(lowercase=False),
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertEqual(
        index_building.read_index_spec(out_dir),
        KeywordDocumentIndexSpec(lowercase=False),
    )
    with open(os.path.join(out_dir, index_building.INDEX_SPEC_FILENAME)) as f:
      self.assertEqual(json.load(f)['spec_type'], 'keyword')
    self.assertTrue(os.path.exists(os.path.join(out_dir, 'keyword_docs.jsonl')))

  def test_build_from_spec_passes_runtime_kwargs(self):
    build_spy = self.enter_context(
        mock.patch.object(
            index_building, 'build_index', wraps=index_building.build_index
        )
    )

    build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
        runtime_kwargs={'api_key': 'key'},
    )

    self.assertEqual(
        build_spy.call_args.kwargs['runtime_kwargs'], {'api_key': 'key'}
    )

  def test_build_from_spec_loads_documents_from_folder(self):
    corpus_dir = self.create_tempdir().mkdir('my_corpus')
    corpus_dir.create_file('paris.txt', content='Paris is in France.')
    corpus_dir.create_file('sub/madrid.txt', content='Madrid is in Spain.')

    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        corpus=corpus_dir.full_path,
        output=self.output_dir,
    )

    with open(os.path.join(out_dir, 'keyword_docs.jsonl')) as f:
      doc_ids = [json.loads(line)['doc_id'] for line in f]
    self.assertEqual(doc_ids, ['paris.txt', 'sub/madrid.txt'])
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), 'my_corpus')

  def test_registered_loader_corpus(self):
    build_mock = self._mock_index_building_build_index()

    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertEqual(
        [doc.doc_id for doc in build_mock.call_args.kwargs['docs']],
        ['d1', 'd2'],
    )
    self.assertEqual(build_mock.call_args.kwargs['output_dir'], out_dir)
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), _DATASET_NAME)

  def test_custom_loader_corpus(self):
    build_mock = self._mock_index_building_build_index()

    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        corpus=_CustomLoader(),
        output=self.output_dir,
    )

    self.assertEqual(
        [doc.doc_id for doc in build_mock.call_args.kwargs['docs']],
        ['c1', 'c2', 'c3'],
    )
    self.assertEqual(
        os.path.basename(os.path.dirname(out_dir)), '_CustomLoader'
    )

  def test_custom_loader_corpus_with_dataset_name(self):
    loader = _CustomLoader()
    loader.dataset_name = 'my_corpus'

    self.assertEqual(build_index_lib._corpus_name(loader), 'my_corpus')

  def test_path_corpus_loads_with_multimodal_loader(self):
    build_mock = self._mock_index_building_build_index()

    @ot_executing.make_executable
    async def fake_load_documents():
      return [retrieval_data_structures.Document(doc_id='d1', content='doc')]

    loader_cls = self.enter_context(
        mock.patch.object(multimodal_loader, 'MultimodalDatasetLoader')
    )
    loader_cls.return_value.load_documents = fake_load_documents

    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        corpus='/some/path/my_corpus/',
        output=self.output_dir,
    )

    loader_cls.assert_called_once_with(path='/some/path/my_corpus/')
    self.assertLen(build_mock.call_args.kwargs['docs'], 1)
    self.assertEqual(build_mock.call_args.kwargs['output_dir'], out_dir)
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), 'my_corpus')

  def test_corpus_name(self):
    for corpus, expected_name in (
        (datasets.get_dataset_loader(_DATASET_NAME), _DATASET_NAME),
        (_CustomLoader(), '_CustomLoader'),
        ('/some/path/my_corpus', 'my_corpus'),
        ('/some/path/my_corpus/', 'my_corpus'),
        ('relative/my_corpus/', 'my_corpus'),
    ):
      with self.subTest(expected_name + ':' + str(corpus)):
        self.assertEqual(build_index_lib._corpus_name(corpus), expected_name)

  def test_corpus_name_of_loaders_of_public_datasets(self):
    # Loaders are only constructed, no data is loaded.
    for name in (
        'browsecomp_plus',
        'hotpotqa_validation',
        'narrativeqa_validation',
        'nfcorpus_test',
    ):
      with self.subTest(name):
        loader = datasets.get_dataset_loader(name, max_number_of_documents=3)
        self.assertEqual(build_index_lib._corpus_name(loader), name)

  def test_build_from_spec_simple_rewriting_embedding_index(self):
    out_dir = build_index_lib.build_index_from_spec(
        spec=embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            embed_backend_config=_FAKE_EMBEDDER,
            chunker_config=chunker_registry.ChunkerConfig(
                chunker_name='words',
                kwargs={'max_words_per_chunk': 16, 'overlap_window': 0},
            ),
        ),
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    spec = index_building.read_index_spec(out_dir)
    self.assertIsInstance(
        spec,
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    )
    self.assertEqual(
        spec.chunker_config,
        chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 16, 'overlap_window': 0},
        ),
    )
    self.assertEqual(spec.embed_backend_config.model_name, 'fake-model')

  def test_build_from_spec_pdf_pages_chunker(self):
    input_dir = self.create_tempdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(os.path.join(input_dir.full_path, 'two_pages.pdf'), 'wb') as f:
      writer.write(f)

    out_dir = build_index_lib.build_index_from_spec(
        spec=embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            embed_backend_config=_FAKE_EMBEDDER,
            chunker_config=chunker_registry.ChunkerConfig(
                chunker_name='pdf_pages', kwargs={'pages_per_chunk': 1}
            ),
        ),
        corpus=input_dir.full_path,
        output=self.output_dir,
    )

    spec = index_building.read_index_spec(out_dir)
    self.assertIsInstance(
        spec,
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    )
    self.assertEqual(
        spec.chunker_config,
        chunker_registry.ChunkerConfig(
            chunker_name='pdf_pages', kwargs={'pages_per_chunk': 1}
        ),
    )

  def test_build_from_spec_audio_seconds_chunker(self):
    input_dir = self.create_tempdir()
    # A 2-second silent WAV file.
    with wave.open(os.path.join(input_dir.full_path, 'two_s.wav'), 'wb') as f:
      f.setnchannels(1)
      f.setsampwidth(2)
      f.setframerate(8000)
      f.writeframes(b'\x00\x00' * 16000)

    out_dir = build_index_lib.build_index_from_spec(
        spec=embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            embed_backend_config=_FAKE_EMBEDDER,
            chunker_config=chunker_registry.ChunkerConfig(
                chunker_name='audio_seconds', kwargs={'seconds_per_chunk': 1.0}
            ),
        ),
        corpus=input_dir.full_path,
        output=self.output_dir,
    )

    spec = index_building.read_index_spec(out_dir)
    assert isinstance(
        spec,
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    )
    self.assertEqual(
        spec.chunker_config,
        chunker_registry.ChunkerConfig(
            chunker_name='audio_seconds', kwargs={'seconds_per_chunk': 1.0}
        ),
    )

  def test_build_from_spec_routing_chunker(self):
    # The default text chunker of `RoutingChunker` loads and registers the
    # default SentencePiece model, which is not available in tests.
    llm.reset_defaults()
    self.addCleanup(llm.reset_defaults)
    self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )
    input_dir = self.create_tempdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(os.path.join(input_dir.full_path, 'two_pages.pdf'), 'wb') as f:
      writer.write(f)

    out_dir = build_index_lib.build_index_from_spec(
        spec=embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            embed_backend_config=_FAKE_EMBEDDER,
            chunker_config=chunker_registry.ChunkerConfig(
                chunker_name='routing'
            ),
        ),
        corpus=input_dir.full_path,
        output=self.output_dir,
    )

    spec = index_building.read_index_spec(out_dir)
    assert isinstance(
        spec,
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    )
    with self.subTest('uses_default_routing_chunker'):
      self.assertEqual(
          spec.chunker_config,
          chunker_registry.ChunkerConfig(chunker_name='routing'),
      )

    with self.subTest('output_directory_is_path_friendly'):
      self.assertIn('_chunker_routing_', os.path.basename(out_dir.rstrip('/')))

  def _mock_index_building_build_index(self) -> mock.MagicMock:
    """Replaces `index_building.build_index` with a no-op; returns the mock."""

    @ot_executing.make_executable
    async def fake_build_index(**kwargs):
      del kwargs

    return self.enter_context(
        mock.patch.object(
            index_building,
            'build_index',
            autospec=True,
            side_effect=fake_build_index,
        )
    )

  def test_hipporag_preset_uses_spec_defaults(self):
    build_mock = self._mock_index_building_build_index()

    build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        build_index_lib.IndexType.HIPPORAG,
        output=self.output_dir,
    )

    spec = build_mock.call_args.kwargs['spec']
    self.assertIsInstance(
        spec, hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec
    )
    self.assertEqual(
        spec.generate_backend_config,
        rag_defaults.default_index_generate_backend_config(),
    )
    default_spec = (
        hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec()
    )
    self.assertEqual(
        spec.similarity_threshold, default_spec.similarity_threshold
    )
    self.assertEqual(spec.ignore_triples, default_spec.ignore_triples)
    self.assertEqual(
        spec.ignore_similarity_edges, default_spec.ignore_similarity_edges
    )

  def test_presets_use_default_embedder(self):
    build_mock = self._mock_index_building_build_index()

    for index_type in build_index_lib.IndexType:
      with self.subTest(index_type):
        build_index_lib.build_index(
            datasets.get_dataset_loader(_DATASET_NAME),
            index_type,
            output=self.output_dir,
        )

        self.assertEqual(
            build_mock.call_args.kwargs['spec'].embed_backend_config,
            rag_defaults.default_embed_backend_config(),
        )

  def test_presets_do_not_chunk(self):
    build_mock = self._mock_index_building_build_index()

    for index_type in build_index_lib.IndexType:
      with self.subTest(index_type):
        build_index_lib.build_index(
            datasets.get_dataset_loader(_DATASET_NAME),
            index_type,
            output=self.output_dir,
        )

        self.assertEqual(
            build_mock.call_args.kwargs['spec'].chunker_config,
            chunker_registry.ChunkerConfig(chunker_name='none'),
        )

  def test_build_from_spec_uses_hipporag_spec_settings(self):
    # Non-default HippoRAG settings are set on the spec, which is passed to
    # `build_index_from_spec`.
    build_mock = self._mock_index_building_build_index()
    spec = hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec(
        similarity_threshold=0.5,
        ignore_triples=False,
        ignore_similarity_edges=False,
    )

    build_index_lib.build_index_from_spec(
        spec=spec,
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertIs(build_mock.call_args.kwargs['spec'], spec)

  def test_build_from_spec_uses_hipporag_spec_generate_backend_config(self):
    # A non-default HippoRAG generation model is set on the spec, which is
    # passed to `build_index_from_spec`.
    build_mock = self._mock_index_building_build_index()
    generate_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='FakeGenerateBackend',
        model_name='fake-generate-model',
    )
    spec = hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec(
        generate_backend_config=generate_backend_config,
    )

    build_index_lib.build_index_from_spec(
        spec=spec,
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertEqual(
        build_mock.call_args.kwargs['spec'].generate_backend_config,
        backend_registry.GenerateBackendConfig(
            backend_class_name='FakeGenerateBackend',
            model_name='fake-generate-model',
        ),
    )

  def test_runtime_kwargs_defaults_to_request_defaults(self):
    build_mock = self._mock_index_building_build_index()

    build_index_lib.build_index(
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertEqual(
        build_mock.call_args.kwargs['runtime_kwargs'],
        {
            'max_retries': rag_defaults.DEFAULT_MAX_RETRIES,
            'http_options': {'timeout': rag_defaults.DEFAULT_HTTP_TIMEOUT_MS},
        },
    )

  def test_mapping_runtime_kwargs_get_request_defaults(self):
    build_mock = self._mock_index_building_build_index()
    runtime_kwargs = types.MappingProxyType({'api_key': 'key'})

    build_index_lib.build_index(
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
        runtime_kwargs=runtime_kwargs,
    )

    self.assertEqual(
        build_mock.call_args.kwargs['runtime_kwargs'],
        {
            'api_key': 'key',
            'max_retries': rag_defaults.DEFAULT_MAX_RETRIES,
            'http_options': {'timeout': rag_defaults.DEFAULT_HTTP_TIMEOUT_MS},
        },
    )
    self.assertEqual(runtime_kwargs, {'api_key': 'key'})

  def test_runtime_kwargs_override_request_defaults(self):
    build_mock = self._mock_index_building_build_index()

    build_index_lib.build_index(
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
        runtime_kwargs={'max_retries': 2, 'http_options': {'timeout': 10}},
    )

    self.assertEqual(
        build_mock.call_args.kwargs['runtime_kwargs'],
        {'max_retries': 2, 'http_options': {'timeout': 10}},
    )

  def test_with_request_defaults(self):
    default_timeout = rag_defaults.DEFAULT_HTTP_TIMEOUT_MS
    for name, runtime_kwargs, expected in (
        (
            'none',
            None,
            {
                'max_retries': rag_defaults.DEFAULT_MAX_RETRIES,
                'http_options': {'timeout': default_timeout},
            },
        ),
        (
            'set_values_are_kept',
            {'max_retries': 0, 'http_options': {'timeout': 10}},
            {'max_retries': 0, 'http_options': {'timeout': 10}},
        ),
        (
            'timeout_is_merged_into_http_options',
            {'max_retries': 1, 'http_options': {'base_url': 'url'}},
            {
                'max_retries': 1,
                'http_options': {'base_url': 'url', 'timeout': default_timeout},
            },
        ),
        (
            'non_dict_http_options_is_kept',
            {'max_retries': 1, 'http_options': None},
            {'max_retries': 1, 'http_options': None},
        ),
    ):
      with self.subTest(name):
        self.assertEqual(
            build_index_lib.get_default_runtime_kwargs(runtime_kwargs), expected
        )

  def test_with_request_defaults_custom_defaults(self):
    self.assertEqual(
        build_index_lib.get_default_runtime_kwargs(
            {'http_options': {}}, max_retries=3, http_timeout_ms=100
        ),
        {'max_retries': 3, 'http_options': {'timeout': 100}},
    )
    # Values set in `runtime_kwargs` take precedence over custom defaults.
    self.assertEqual(
        build_index_lib.get_default_runtime_kwargs(
            {'max_retries': 1, 'http_options': {'timeout': 10}},
            max_retries=3,
            http_timeout_ms=100,
        ),
        {'max_retries': 1, 'http_options': {'timeout': 10}},
    )

  def test_with_request_defaults_does_not_modify_input(self):
    http_options = {'base_url': 'url'}
    runtime_kwargs = {'http_options': http_options}

    build_index_lib.get_default_runtime_kwargs(runtime_kwargs)

    self.assertEqual(runtime_kwargs, {'http_options': {'base_url': 'url'}})
    self.assertEqual(http_options, {'base_url': 'url'})

  def test_storage_format_reaches_index_building(self):
    build_mock = self._mock_index_building_build_index()

    for storage_format in (
        serialization.StorageFormat.JSONL_NPY,
    ):
      with self.subTest(storage_format.name):
        build_index_lib.build_index(
            corpus=datasets.get_dataset_loader(_DATASET_NAME),
            output=self.output_dir,
            storage_format=storage_format,
        )

        self.assertEqual(
            build_mock.call_args.kwargs['storage_format'], storage_format
        )

  def test_hipporag_accepts_default_storage_format(self):
    build_mock = self._mock_index_building_build_index()

    build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        build_index_lib.IndexType.HIPPORAG,
        output=self.output_dir,
        storage_format=serialization.StorageFormat.JSONL_NPY,
    )

    self.assertIsInstance(
        build_mock.call_args.kwargs['spec'],
        hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec,
    )

  def test_default_storage_format(self):
    build_mock = self._mock_index_building_build_index()

    build_index_lib.build_index(
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
    )

    self.assertEqual(
        build_mock.call_args.kwargs['storage_format'],
        serialization.StorageFormat.JSONL_NPY,
    )

  def test_storage_formats_are_registered_serializers(self):
    # The values are the keys of `SERIALIZER_REGISTRY` and the
    # `serializer_type` saved in `manifest.json`, so that saved indexes can be
    # loaded. The internal serializers are registered when `build_index_lib`
    # is imported.
    for storage_format in serialization.StorageFormat:
      with self.subTest(storage_format.name):
        self.assertIn(storage_format.value, serialization.SERIALIZER_REGISTRY)
    self.assertEqual(serialization.StorageFormat.JSONL_NPY.value, 'simple')

  def test_saves_manifest_with_storage_format(self):
    out_dir = build_index_lib.build_index_from_spec(
        spec=embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            embed_backend_config=_FAKE_EMBEDDER
        ),
        corpus=datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
        storage_format=serialization.StorageFormat.JSONL_NPY,
    )

    with open(os.path.join(out_dir, serialization.MANIFEST_JSON_NAME)) as f:
      self.assertEqual(
          json.load(f)['serializer_type'],
          serialization.StorageFormat.JSONL_NPY.value,
      )


if __name__ == '__main__':
  absltest.main()
