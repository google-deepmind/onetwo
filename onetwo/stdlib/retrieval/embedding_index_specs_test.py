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
from onetwo.backends import sentencepiece_tokenizer
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import embedding_index_specs


class EmbeddingIndexSpecsTest(absltest.TestCase):

  def test_tokens_chunker_uses_default_tokenizer(self):
    self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        max_chunk_size=10, chunking_strategy='tokens'
    )

    chunker = embedding_index_specs._get_chunker(spec)

    self.assertIsInstance(chunker, chunking.ChunkByMaxTokens)
    assert chunker.tokenize_backend_config is not None
    self.assertEqual(
        chunker.tokenize_backend_config.backend_class_name,
        'SentencePieceTokenizer',
    )
    self.assertEqual(chunker.tokenize_backend_config.model_name, 'GEMINI3_G145')

  def test_make_index_passes_runtime_kwargs_to_inner_index(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
    )

    index = spec.make_index({'api_key': 'key'})

    self.assertEqual(index.inner_index.runtime_kwargs, {'api_key': 'key'})


if __name__ == '__main__':
  absltest.main()
