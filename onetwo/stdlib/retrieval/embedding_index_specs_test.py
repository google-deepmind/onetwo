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

import dataclasses

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_spec_registry
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
    @dataclasses.dataclass(kw_only=True)
    class CustomChunker(chunking.NoChunking):
      custom_arg: int = 0

    chunker_registry.register_chunker('custom_strategy', CustomChunker)
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

    rewriters = getattr(index.rewriter, 'rewriters')
    chunker = getattr(rewriters[0], 'chunker')
    self.assertIsInstance(chunker, CustomChunker)
    self.assertEqual(chunker.custom_arg, 42)

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

  def test_default_format_str_is_no_formatting(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
    )

    self.assertIsNone(spec.format_str)
    rewriters = getattr(spec.make_index({}).rewriter, 'rewriters')
    self.assertIsInstance(
        getattr(rewriters[1], 'formatter'), document_formatting.NoFormatting
    )

  def test_format_str_uses_text_document_formatter(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
        format_str='{title}: {text}',
    )

    rewriters = getattr(spec.make_index({}).rewriter, 'rewriters')
    formatter = getattr(rewriters[1], 'formatter')
    self.assertIsInstance(formatter, document_formatting.TextDocumentFormatter)
    self.assertEqual(formatter.format_str, '{title}: {text}')

  def test_saved_spec_keeps_its_format_str(self):
    # Saved specs store `format_str` explicitly, so specs saved with the
    # previous default ('{text}') are restored with it.
    spec_dict = index_spec_registry.spec_to_dict(
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
            format_str='{text}'
        )
    )
    self.assertEqual(spec_dict['spec']['format_str'], '{text}')
    self.assertEqual(
        index_spec_registry.spec_from_dict(spec_dict).format_str, '{text}'
    )
    default_spec_dict = index_spec_registry.spec_to_dict(
        embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec()
    )
    self.assertIsNone(default_spec_dict['spec']['format_str'])


if __name__ == '__main__':
  absltest.main()
