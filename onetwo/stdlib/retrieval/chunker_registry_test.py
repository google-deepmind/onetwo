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
from onetwo.builtins import llm
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking


class ChunkerRegistryTest(absltest.TestCase):

  def test_builtin_chunkers_are_registered(self):
    self.assertEqual(
        chunker_registry.list_chunkers(),
        ['none', 'tokens', 'words'],
    )

  def test_get_none_chunker(self):
    config = chunker_registry.ChunkerConfig(chunker_name='none')
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, chunking.NoChunking)

  def test_get_words_chunker(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='words',
        kwargs={'max_words_per_chunk': 50, 'overlap_window': 5},
    )
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, chunking.ChunkByMaxWords)
    self.assertEqual(chunker.max_words_per_chunk, 50)
    self.assertEqual(chunker.overlap_window, 5)

  def test_get_tokens_chunker(self):
    llm.reset_defaults()
    self.addCleanup(llm.reset_defaults)
    mock_load = self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )
    tokenize_config = backend_registry.TokenizeBackendConfig(
        backend_class_name='SentencePieceTokenizer',
        model_name='GEMINI3_G145',
    )
    config = chunker_registry.ChunkerConfig(
        chunker_name='tokens',
        kwargs={
            'max_tokens_per_chunk': 64,
            'overlap_window': 8,
            'tokenize_backend_config': tokenize_config,
        },
    )
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, chunking.ChunkByMaxTokens)
    self.assertEqual(chunker.max_tokens_per_chunk, 64)
    self.assertEqual(chunker.overlap_window, 8)
    self.assertEqual(chunker.tokenize_backend_config, tokenize_config)
    self.assertEqual(
        str(config),
        'tokens-max_tokens_per_chunk=64-overlap_window=8-'
        'tokenize_backend_config=SentencePieceTokenizer-GEMINI3_G145',
    )
    mock_load.assert_called_once()
    self.assertTrue(llm.tokenize.is_configured())

  def test_chunker_config_str_and_json_round_trip(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='words',
        kwargs={'max_words_per_chunk': 100, 'overlap_window': 10},
    )
    self.assertEqual(
        str(config), 'words-max_words_per_chunk=100-overlap_window=10'
    )
    loaded = chunker_registry.ChunkerConfig.from_json(config.to_json())
    self.assertEqual(config, loaded)

  def test_register_and_get_custom_chunker(self):
    custom_chunker = chunking.NoChunking()
    captured = {}

    def _factory(**kwargs):
      captured.update(kwargs)
      return custom_chunker

    chunker_registry.register_chunker('custom_for_test', _factory)
    self.addCleanup(
        chunker_registry.CHUNKER_REGISTRY.pop, 'custom_for_test', None
    )

    self.assertIn('custom_for_test', chunker_registry.list_chunkers())
    config = chunker_registry.ChunkerConfig(
        chunker_name='custom_for_test',
        kwargs={'k': 'v'},
    )
    result = chunker_registry.get_chunker(config)
    self.assertIs(result, custom_chunker)
    self.assertEqual(captured, {'k': 'v'})

    with self.assertRaisesRegex(ValueError, 'already registered'):
      chunker_registry.register_chunker('custom_for_test', _factory)

  def test_get_chunker_unknown_raises_key_error(self):
    config = chunker_registry.ChunkerConfig(chunker_name='nonexistent_strategy')
    with self.assertRaisesRegex(KeyError, 'Unknown chunking strategy'):
      chunker_registry.get_chunker(config)


if __name__ == '__main__':
  absltest.main()
