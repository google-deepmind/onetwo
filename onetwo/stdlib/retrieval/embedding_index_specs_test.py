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
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import indexing


class EmbeddingIndexSpecsTest(absltest.TestCase):

  def test_make_index_passes_runtime_kwargs_to_inner_index(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
    )

    index = spec.make_index({'api_key': 'key'})

    self.assertEqual(index.inner_index.runtime_kwargs, {'api_key': 'key'})

  def test_custom_chunker_strategy(self):
    custom_chunker = chunking.NoChunking()
    factory = mock.Mock(return_value=custom_chunker)
    chunker_registry.register_chunker('custom_strategy', factory)
    self.addCleanup(
        chunker_registry.CHUNKER_REGISTRY.pop, 'custom_strategy', None
    )

    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='custom_strategy',
            kwargs={'custom_arg': 42},
        ),
    )

    index = spec.make_index({'from_arg': 2})

    factory.assert_called_once_with(custom_arg=42)
    rewriters = getattr(index.rewriter, 'rewriters')
    self.assertIs(getattr(rewriters[0], 'chunker'), custom_chunker)

  def test_preserve_original_docs_and_retrieval_target(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
        preserve_original_docs=True,
        retrieval_target=indexing.RetrievalTarget.ORIGINAL,
    )
    index = spec.make_index({})
    self.assertTrue(index.preserve_original_docs)
    self.assertEqual(index.retrieval_target, indexing.RetrievalTarget.ORIGINAL)


if __name__ == '__main__':
  absltest.main()
