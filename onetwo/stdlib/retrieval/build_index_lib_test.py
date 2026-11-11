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
from unittest import mock
import wave

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import audio_chunking
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import pdf_chunking
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing
import pypdf

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


def _embedding_index() -> indexing.EmbeddingBasedDocumentIndex:
  return indexing.EmbeddingBasedDocumentIndex(
      embed_backend_config=_FAKE_EMBEDDER
  )


def _chunking_index(
    chunker: chunking.Chunker,
) -> indexing.RewritingEmbeddingBasedDocumentIndex:
  return indexing.RewritingEmbeddingBasedDocumentIndex(
      rewriter=corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
      inner_index=_embedding_index(),
  )


def _doc_ids(index_directory: str) -> list[str]:
  return sorted(
      doc.doc_id
      for doc in index_building.load_index(index_directory).get_docs()
  )


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


@dataclasses.dataclass
class _NoEntitiesExtractor:
  """Entity and triples extractor that finds none."""

  @ot_executing.make_executable(copy_self=False)
  async def __call__(self, doc):
    del doc
    return [], []


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

  def test_loads_documents_from_folder(self):
    corpus_dir = self.create_tempdir().mkdir('my_corpus')
    corpus_dir.create_file('paris.txt', content='Paris is in France.')
    corpus_dir.create_file('sub/madrid.txt', content='Madrid is in Spain.')

    out_dir = build_index_lib.build_index(
        corpus_dir.full_path, _embedding_index(), output=self.output_dir
    )

    self.assertEqual(_doc_ids(out_dir), ['paris.txt', 'sub/madrid.txt'])
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), 'my_corpus')

  def test_registered_loader_corpus(self):
    out_dir = build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        _embedding_index(),
        output=self.output_dir,
    )

    self.assertEqual(_doc_ids(out_dir), ['d1', 'd2'])
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), _DATASET_NAME)

  def test_custom_loader_corpus(self):
    out_dir = build_index_lib.build_index(
        _CustomLoader(), _embedding_index(), output=self.output_dir
    )

    self.assertEqual(_doc_ids(out_dir), ['c1', 'c2', 'c3'])
    self.assertEqual(
        os.path.basename(os.path.dirname(out_dir)), '_CustomLoader'
    )

  def test_custom_loader_corpus_with_dataset_name(self):
    loader = _CustomLoader()
    loader.dataset_name = 'my_corpus'

    self.assertEqual(build_index_lib._corpus_name(loader), 'my_corpus')

  def test_path_corpus_loads_with_multimodal_loader(self):
    @ot_executing.make_executable
    async def fake_load_documents():
      return [retrieval_data_structures.Document(doc_id='d1', content='doc')]

    loader_cls = self.enter_context(
        mock.patch.object(multimodal_loader, 'MultimodalDatasetLoader')
    )
    loader_cls.return_value.load_documents = fake_load_documents

    out_dir = build_index_lib.build_index(
        '/some/path/my_corpus/', _embedding_index(), output=self.output_dir
    )

    loader_cls.assert_called_once_with(path='/some/path/my_corpus/')
    self.assertEqual(_doc_ids(out_dir), ['d1'])
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

  def test_pdf_pages_chunker(self):
    input_dir = self.create_tempdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(os.path.join(input_dir.full_path, 'two_pages.pdf'), 'wb') as f:
      writer.write(f)

    out_dir = build_index_lib.build_index(
        input_dir.full_path,
        _chunking_index(pdf_chunking.ChunkByPdfPages(pages_per_chunk=1)),
        output=self.output_dir,
    )

    self.assertEqual(index_building.load_index(out_dir).num_docs, 2)

  def test_audio_seconds_chunker(self):
    input_dir = self.create_tempdir()
    # A 2-second silent WAV file.
    with wave.open(os.path.join(input_dir.full_path, 'two_s.wav'), 'wb') as f:
      f.setnchannels(1)
      f.setsampwidth(2)
      f.setframerate(8000)
      f.writeframes(b'\x00\x00' * 16000)

    out_dir = build_index_lib.build_index(
        input_dir.full_path,
        _chunking_index(
            audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=1.0)
        ),
        output=self.output_dir,
    )

    self.assertEqual(index_building.load_index(out_dir).num_docs, 2)

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

  def _patch_preset_backends(self) -> None:
    """Makes the ready-made indexes use fake embedding and extraction."""
    self.enter_context(
        mock.patch.object(
            rag_defaults,
            'default_embed_backend_config',
            return_value=_FAKE_EMBEDDER,
        )
    )
    self.enter_context(
        mock.patch.object(
            graphrag_indexing.HippoRAGDocumentIndex,
            '_make_extractor',
            return_value=_NoEntitiesExtractor(),
        )
    )
    # The default text chunker of `RoutingChunker` tokenizes with the default
    # tokenizer: replace it with one that tokenizes each character as a token.
    self.enter_context(
        mock.patch.dict(
            backend_registry.BACKEND_REGISTRY,
            {
                rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME: (
                    lambda **_: backends_test_utils.LLMForTest()
                )
            },
        )
    )

  def test_presets_are_saved_with_index_config(self):
    self._patch_preset_backends()
    for index_type, index_class in (
        (
            build_index_lib.IndexType.EMBEDDING,
            'RewritingEmbeddingBasedDocumentIndex',
        ),
        (build_index_lib.IndexType.GRAPHRAG, 'RewritingDocumentIndex'),
    ):
      with self.subTest(index_type):
        out_dir = build_index_lib.build_index(
            datasets.get_dataset_loader(_DATASET_NAME),
            index_type,
            output=self.output_dir,
        )

        self.assertRegex(
            os.path.basename(out_dir),
            f'^{index_class}_chunker_RoutingChunker_fake-model'
            '__hash_[0-9a-f]{8}$',
        )
        files = os.listdir(out_dir)
        self.assertIn(config_serialization.INDEX_CONFIG_FILENAME, files)
        self.assertEqual(index_building.load_index(out_dir).num_docs, 2)

  def test_preset_gets_default_runtime_kwargs(self):
    self._patch_preset_backends()
    from_config = self.enter_context(
        mock.patch.object(
            config_serialization,
            'from_config',
            wraps=config_serialization.from_config,
        )
    )

    build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        output=self.output_dir,
        runtime_kwargs={'api_key': 'key'},
    )

    self.assertEqual(
        from_config.call_args.args[1],
        {
            'api_key': 'key',
            'max_retries': rag_defaults.DEFAULT_MAX_RETRIES,
            'http_options': {'timeout': rag_defaults.DEFAULT_HTTP_TIMEOUT_MS},
        },
    )

  def test_preset_storage_format_reaches_save(self):
    self._patch_preset_backends()
    save = self.enter_context(
        mock.patch.object(
            indexing.RewritingEmbeddingBasedDocumentIndex, 'save', autospec=True
        )
    )

    for storage_format in (
        serialization.StorageFormat.JSONL_NPY,
    ):
      with self.subTest(storage_format.name):
        out_dir = build_index_lib.build_index(
            datasets.get_dataset_loader(_DATASET_NAME),
            output=self.output_dir,
            storage_format=storage_format,
        )

        save.assert_called_with(mock.ANY, out_dir, storage_format)

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
    out_dir = build_index_lib.build_index(
        datasets.get_dataset_loader(_DATASET_NAME),
        _embedding_index(),
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
