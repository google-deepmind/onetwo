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

import json
import os
from unittest import mock

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import pdf_chunking  # pylint: disable=unused-import  # Registers 'pdf_pages'.
import pypdf

KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec

_DATASET_NAME = 'build_index_lib_test_dataset'


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

  def test_build_custom_spec_saves_index_and_spec(self):
    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(lowercase=False),
        dataset_name=_DATASET_NAME,
        output_directory=self.output_dir,
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
        dataset_name=_DATASET_NAME,
        output_directory=self.output_dir,
        runtime_kwargs={'api_key': 'key'},
    )

    self.assertEqual(
        build_spy.call_args.kwargs['runtime_kwargs'], {'api_key': 'key'}
    )

  def test_build_from_spec_loads_documents_from_input(self):
    input_dir = self.create_tempdir()
    input_dir.create_file('paris.txt', content='Paris is in France.')
    input_dir.create_file('sub/madrid.txt', content='Madrid is in Spain.')

    out_dir = build_index_lib.build_index_from_spec(
        spec=KeywordDocumentIndexSpec(),
        dataset_name='my_corpus',
        output_directory=self.output_dir,
        input_dir=input_dir.full_path,
    )

    with open(os.path.join(out_dir, 'keyword_docs.jsonl')) as f:
      doc_ids = [json.loads(line)['doc_id'] for line in f]
    self.assertEqual(doc_ids, ['paris.txt', 'sub/madrid.txt'])
    self.assertEqual(os.path.basename(os.path.dirname(out_dir)), 'my_corpus')

  def test_build_from_spec_input_with_registered_dataset_name_raises(self):
    with self.assertRaisesRegex(ValueError, 'is a registered dataset'):
      build_index_lib.build_index_from_spec(
          spec=KeywordDocumentIndexSpec(),
          dataset_name=_DATASET_NAME,
          output_directory=self.output_dir,
          input_dir=self.create_tempdir().full_path,
      )

  def test_flat_kwargs_build_simple_rewriting_embedding_index(self):
    out_dir = build_index_lib.build_index(
        dataset_name=_DATASET_NAME,
        output_directory=self.output_dir,
        runtime_kwargs={},
        embedding_backend_class_name='FakeEmbedderBackend',
        embedding_model_name='fake-model',
        chunk_size=16,
        chunking_strategy='words',
        overlap_window=0,
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

    with self.assertRaisesRegex(ValueError, 'Unsupported chunking strategy'):
      build_index_lib.build_index(
          dataset_name=_DATASET_NAME,
          output_directory=self.output_dir,
          runtime_kwargs={},
          embedding_backend_class_name='FakeEmbedderBackend',
          embedding_model_name='fake-model',
          chunk_size=16,
          chunking_strategy='unknown_strategy',
      )

  def test_flat_kwargs_pdf_pages_chunking_strategy(self):
    input_dir = self.create_tempdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(os.path.join(input_dir.full_path, 'two_pages.pdf'), 'wb') as f:
      writer.write(f)

    out_dir = build_index_lib.build_index(
        dataset_name='my_pdfs',
        input_dir=input_dir.full_path,
        output_directory=self.output_dir,
        runtime_kwargs={},
        embedding_backend_class_name='FakeEmbedderBackend',
        embedding_model_name='fake-model',
        chunk_size=1,
        chunking_strategy='pdf_pages',
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

    with self.assertRaisesRegex(ValueError, 'overlap_window is not supported'):
      build_index_lib.build_index(
          dataset_name=_DATASET_NAME,
          output_directory=self.output_dir,
          runtime_kwargs={},
          embedding_backend_class_name='FakeEmbedderBackend',
          embedding_model_name='fake-model',
          chunk_size=1,
          chunking_strategy='pdf_pages',
          overlap_window=1,
      )


if __name__ == '__main__':
  absltest.main()
