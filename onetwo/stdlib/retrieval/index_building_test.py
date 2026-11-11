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

import os
import re

from absl.testing import absltest
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import routing_chunking

Document = retrieval_data_structures.Document

_DOCS = [
    Document(doc_id='d1', content='Paris is in France.'),
    Document(doc_id='d2', content='Madrid is in Spain.'),
]


@batching.add_batching
class FakeEmbedder(backends_test_utils.EmbedderForTest):

  def __init__(self, **kwargs):
    del kwargs  # Runtime kwargs are not used.
    super().__init__(default_embedding=[0.1, 0.2])


class IndexBuildingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )

  def _embedding_index(self) -> indexing.EmbeddingBasedDocumentIndex:
    return indexing.EmbeddingBasedDocumentIndex(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        )
    )

  def test_build_index_saves_index_config(self):
    out_dir = self.create_tempdir().full_path

    ot.run(
        index_building.build_index(
            index=self._embedding_index(), docs=_DOCS, output_dir=out_dir
        )
    )
    loaded = index_building.load_index(out_dir, runtime_kwargs={'k': 'v'})

    self.assertTrue(
        os.path.exists(
            os.path.join(out_dir, config_serialization.INDEX_CONFIG_FILENAME)
        )
    )
    assert isinstance(loaded, indexing.EmbeddingBasedDocumentIndex)
    self.assertEqual(loaded.runtime_kwargs, {'k': 'v'})
    self.assertEqual(loaded.num_docs, 2)

  def test_build_index_without_output_dir_returns_index(self):
    index = ot.run(
        index_building.build_index(index=self._embedding_index(), docs=_DOCS)
    )

    self.assertEqual(index.num_docs, 2)

  def test_build_index_unknown_runner_raises(self):
    with self.assertRaisesRegex(ValueError, 'Unknown runner'):
      ot.run(
          index_building.build_index(
              index=self._embedding_index(), docs=_DOCS, runner='unknown'
          )
      )

  def test_load_index_without_config_loads_embedding_index(self):
    out_dir = self.create_tempdir().full_path
    index = self._embedding_index()
    ot.run(index.add_docs(_DOCS))
    index.save(out_dir)

    loaded = index_building.load_index(out_dir, runtime_kwargs={'k': 'v'})

    assert isinstance(loaded, indexing.EmbeddingBasedDocumentIndex)
    self.assertEqual(loaded.runtime_kwargs, {'k': 'v'})
    self.assertEqual(loaded.num_docs, 2)

  def test_index_dir_name(self):
    def chunking_index(chunker):
      return indexing.RewritingEmbeddingBasedDocumentIndex(
          rewriter=corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
          inner_index=indexing.EmbeddingBasedDocumentIndex(
              embed_backend_config=backend_registry.EmbedBackendConfig(
                  backend_class_name='FakeEmbedderBackend',
                  model_name='fake-model',
              )
          ),
      )

    for index, expected_prefix in (
        (
            chunking_index(chunking.ChunkByMaxTokens(max_tokens_per_chunk=512)),
            (
                'RewritingEmbeddingBasedDocumentIndex_chunker_ChunkByMaxTokens-'
                'max_tokens_per_chunk=512-overlap_window=0-strip_whitespace=True_'
                'fake-model'
            ),
        ),
        (
            chunking_index(routing_chunking.RoutingChunker()),
            (
                'RewritingEmbeddingBasedDocumentIndex_chunker_RoutingChunker_'
                'fake-model'
            ),
        ),
        (
            self._embedding_index(),
            'EmbeddingBasedDocumentIndex_chunker_none_FakeEmbedderBackend',
        ),
    ):
      with self.subTest(expected_prefix):
        self.assertRegex(
            index_building.index_dir_name(index),
            f'^{re.escape(expected_prefix)}__hash_[0-9a-f]{{8}}$',
        )


if __name__ == '__main__':
  absltest.main()
