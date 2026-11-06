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

"""Round trips of indexes through `index_config.json`."""

from collections.abc import Iterable
import dataclasses
import json
import re
from typing import Any
from unittest import mock

from absl.testing import absltest
from etils import epath
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document

_VOCABULARY = ('cat', 'dog', 'fish')
_EMBEDDER = backend_registry.EmbedBackendConfig(
    backend_class_name='KeywordEmbedderForTest'
)
_DOCS = [
    Document(
        doc_id='pets', title='Pets', content='The cat sleeps. The dog barks.'
    ),
    Document(doc_id='sea', title='Sea', content='A fish swims.'),
]


@batching.add_batching
class _KeywordEmbedder(backends_test_utils.EmbedderForTest):
  """Embeds text as the counts of the words of `_VOCABULARY` in it."""

  def __init__(self, **kwargs):
    del kwargs  # Unused (e.g. credentials passed as runtime kwargs).
    super().__init__()

  def _get_embed_reply(self, content):
    words = re.findall('[a-z]+', str(content).lower())
    return [words.count(w) for w in _VOCABULARY] + [0.1]


@registry.register('test.SentenceChunker')
@dataclasses.dataclass(kw_only=True)
class _SentenceChunker(chunking.TextChunker):
  """Splits text into chunks of `max_sentences` sentences."""

  max_sentences: int = 1

  @executing.make_executable(copy_self=False)
  async def _chunk_text(self, text: str) -> Iterable[str]:
    sentences = [f'{s.strip()}.' for s in text.split('.') if s.strip()]
    return [
        ' '.join(sentences[i : i + self.max_sentences])
        for i in range(0, len(sentences), self.max_sentences)
    ]


@dataclasses.dataclass(kw_only=True)
class _UnregisteredChunker(chunking.NoChunking):
  pass


@registry.register('test.WithMapping')
@dataclasses.dataclass(kw_only=True)
class _WithMapping:
  values: dict[Any, str] = dataclasses.field(default_factory=dict)


class ConfigSerializationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(
            backend_registry.BACKEND_REGISTRY,
            {'KeywordEmbedderForTest': _KeywordEmbedder},
        )
    )

  def test_custom_index_round_trip(self):
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.SequentialCorpusRewriter(
            rewriters=[
                corpus_rewriting.ChunkingCorpusRewriter(
                    chunker=_SentenceChunker(max_sentences=1)
                ),
                corpus_rewriting.FormattingCorpusRewriter(
                    formatter=document_formatting.TextDocumentFormatter(
                        format_str='{title}: {text}'
                    )
                ),
            ]
        ),
        inner_index=indexing.EmbeddingBasedDocumentIndex(
            embed_backend_config=_EMBEDDER, runtime_kwargs={'api_key': 'secret'}
        ),
    )
    executing.run(index.create_index(corpus_name='pets', docs=_DOCS))
    index_dir = self.create_tempdir().full_path

    index.save(index_dir)
    config_serialization.save_index_config(index, index_dir)
    loaded = index_building.load_index(
        index_dir, runtime_kwargs={'api_key': 'secret'}
    )

    assert isinstance(loaded, indexing.RewritingEmbeddingBasedDocumentIndex)
    with self.subTest('loads_the_same_config'):
      self.assertEqual(
          config_serialization.get_config(loaded),
          config_serialization.get_config(index),
      )
    with self.subTest('saves_enums_by_value_and_loads_them_as_enums'):
      self.assertIs(
          type(
              config_serialization.get_config(index)['config'][
                  'retrieval_target'
              ]
          ),
          str,
      )
      self.assertIs(type(loaded.retrieval_target), indexing.RetrievalTarget)
    with self.subTest('loads_lists_of_objects'):
      assert isinstance(
          loaded.rewriter, corpus_rewriting.SequentialCorpusRewriter
      )
      self.assertIsInstance(
          loaded.rewriter.rewriters[0], corpus_rewriting.ChunkingCorpusRewriter
      )
    with self.subTest('retrieves_the_same_chunks'):
      retrieved = list(executing.run(loaded.retrieve('dog', max_results=1)))
      self.assertEqual(retrieved[0].text, 'Pets: The dog barks.')
    with self.subTest('does_not_save_credentials'):
      saved = (
          epath.Path(index_dir) / config_serialization.INDEX_CONFIG_FILENAME
      ).read_text()
      self.assertNotIn('secret', saved)
      self.assertEqual(json.loads(saved)['format_version'], '1.0')
    with self.subTest('saves_backend_configs_as_registered_objects'):
      inner_config = json.loads(saved)['index']['config']['inner_index']
      self.assertEqual(
          inner_config['config']['embed_backend_config']['class'],
          'EmbedBackendConfig',
      )
    with self.subTest('passes_credentials_to_the_loaded_index'):
      self.assertEqual(loaded.inner_index.runtime_kwargs, {'api_key': 'secret'})

  def test_errors(self):
    with self.subTest('dict_with_non_str_keys'):
      with self.assertRaisesRegex(
          TypeError, r'Cannot save test\.WithMapping\.values'
      ):
        config_serialization.get_config(_WithMapping(values={1: 'a'}))

    index_dir = self.create_tempdir().full_path
    config_path = (
        epath.Path(index_dir) / config_serialization.INDEX_CONFIG_FILENAME
    )
    with self.subTest('unsupported_format_version'):
      config_path.write_text(json.dumps({'format_version': '2.0', 'index': {}}))
      with self.assertRaisesRegex(ValueError, "format_version '2.0'"):
        config_serialization.make_index_from_config_file(index_dir)
    with self.subTest('not_a_document_index'):
      config_path.write_text(
          json.dumps({
              'format_version': config_serialization.FORMAT_VERSION,
              'index': config_serialization.get_config(_WithMapping()),
          })
      )
      with self.assertRaisesRegex(TypeError, 'not a DocumentIndex'):
        config_serialization.make_index_from_config_file(index_dir)

  def test_unregistered_class_raises(self):
    index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(
            chunker=_UnregisteredChunker()
        ),
        inner_index=indexing.EmbeddingBasedDocumentIndex(),
    )

    with self.assertRaisesRegex(
        TypeError,
        'Cannot save ChunkingCorpusRewriter.chunker of type'
        ' _UnregisteredChunker: register its class',
    ):
      config_serialization.get_config(index)


if __name__ == '__main__':
  absltest.main()
