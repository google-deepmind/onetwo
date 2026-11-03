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

from absl.testing import absltest
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document
KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec

_DOCS = [
    Document(doc_id='d1', content='Paris is in France.'),
    Document(doc_id='d2', content='Madrid is in Spain.'),
]


@batching.add_batching
class FakeEmbedder(backends_test_utils.EmbedderForTest):

  def __init__(self, embed_model_name: str | None = None, **kwargs):
    del kwargs  # Runtime kwargs are not used.
    super().__init__(default_embedding=[0.1, 0.2])
    self.embed_model_name = embed_model_name


class IndexBuildingTest(absltest.TestCase):

  def _build_embedding_index(self) -> str:
    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        ),
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 16},
        ),
    )
    out_dir = self.create_tempdir().full_path
    ot.run(
        index_building.build_index(spec=spec, docs=_DOCS, output_dir=out_dir)
    )
    return out_dir

  def test_build_index_writes_spec_for_registered_spec(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    out_dir = self.create_tempdir().full_path
    spec = KeywordDocumentIndexSpec(lowercase=False)

    ot.run(
        index_building.build_index(spec=spec, docs=_DOCS, output_dir=out_dir)
    )

    self.assertEqual(index_building.read_index_spec(out_dir), spec)

  def test_build_index_without_output_dir_returns_index(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())

    index = ot.run(
        index_building.build_index(spec=KeywordDocumentIndexSpec(), docs=_DOCS)
    )

    self.assertEqual(index.num_docs, 2)

  def test_read_index_spec_returns_none_without_spec_file(self):
    self.assertIsNone(
        index_building.read_index_spec(self.create_tempdir().full_path)
    )

  def test_read_index_spec_unregistered_type_raises(self):
    out_dir = self.create_tempdir().full_path
    spec_path = os.path.join(out_dir, index_building.INDEX_SPEC_FILENAME)
    with open(spec_path, 'w') as f:
      f.write('{"spec_type": "not_registered", "spec": {}}')

    with self.assertRaisesRegex(KeyError, 'not_registered'):
      index_building.read_index_spec(out_dir)

  def test_load_index_restores_custom_index(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    out_dir = self.create_tempdir().full_path
    ot.run(
        index_building.build_index(
            spec=KeywordDocumentIndexSpec(lowercase=False),
            docs=_DOCS,
            output_dir=out_dir,
        )
    )

    index = index_building.load_index(out_dir)

    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertFalse(index.lowercase)
    results = ot.run(index.retrieve('Where is Madrid?', max_results=1))
    self.assertEqual([doc.doc_id for doc in results], ['d2'])

  def test_load_index_unregistered_custom_spec_raises(self):
    out_dir = self.create_tempdir().full_path
    with keyword_index_for_test.registered_keyword_index_spec():
      ot.run(
          index_building.build_index(
              spec=KeywordDocumentIndexSpec(), docs=_DOCS, output_dir=out_dir
          )
      )

    with self.assertRaisesRegex(KeyError, 'KeywordDocumentIndexSpec'):
      index_building.load_index(out_dir)

  def test_load_index_builtin_spec_matches_legacy_load(self):
    out_dir = self._build_embedding_index()
    legacy_index = indexing.EmbeddingBasedDocumentIndex()
    ot.run(legacy_index.load(out_dir))

    index = index_building.load_index(out_dir, runtime_kwargs={'k': 'v'})

    self.assertIsInstance(index, indexing.RewritingEmbeddingBasedDocumentIndex)
    self.assertEqual(index.inner_index.runtime_kwargs, {'k': 'v'})
    query = 'Where is Madrid?'
    self.assertEqual(
        ot.run(index.retrieve(query, max_results=2)),
        ot.run(legacy_index.retrieve(query, max_results=2)),
    )

  def test_load_index_without_spec_file_loads_embedding_index(self):
    out_dir = self._build_embedding_index()
    os.remove(os.path.join(out_dir, index_building.INDEX_SPEC_FILENAME))

    index = index_building.load_index(out_dir, runtime_kwargs={'k': 'v'})

    self.assertIsInstance(index, indexing.EmbeddingBasedDocumentIndex)
    self.assertNotIsInstance(
        index, indexing.RewritingEmbeddingBasedDocumentIndex
    )
    self.assertEqual(index.runtime_kwargs, {'k': 'v'})
    self.assertEqual(index.num_docs, 2)

  def test_load_index_unloadable_index_raises(self):
    self.enter_context(
        keyword_index_for_test.registered_unsaveable_index_spec()
    )
    out_dir = self.create_tempdir().full_path
    spec_path = os.path.join(out_dir, index_building.INDEX_SPEC_FILENAME)
    with open(spec_path, 'w') as f:
      f.write('{"spec_type": "UnsaveableDocumentIndexSpec", "spec": {}}')

    with self.assertRaisesRegex(
        TypeError,
        'UnsaveableDocumentIndex \\(built for UnsaveableDocumentIndexSpec\\)'
        ' does not implement load',
    ):
      index_building.load_index(out_dir)


if __name__ == '__main__':
  absltest.main()
