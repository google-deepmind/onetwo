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
import tempfile

from absl.testing import absltest
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import batching
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document

_DOCS = [
    Document(doc_id='doc1', title='Title 1', content='This is document one.'),
    Document(doc_id='doc2', title='Title 2', content='This is document two.'),
]


@batching.add_batching
class FakeEmbedder(backends_test_utils.EmbedderForTest):

  def __init__(self, **kwargs):
    del kwargs  # Runtime kwargs are not used.
    super().__init__(default_embedding=[0.1, 0.2])


class LocalIndexRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    backend_registry.BACKEND_REGISTRY['FakeEmbedderBackend'] = FakeEmbedder
    self.addCleanup(
        backend_registry.BACKEND_REGISTRY.pop, 'FakeEmbedderBackend', None
    )
    self.runner = local_index_runner.LocalIndexRunner()

  def _embedding_index(self) -> indexing.EmbeddingBasedDocumentIndex:
    return indexing.EmbeddingBasedDocumentIndex(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='FakeEmbedderBackend'
        )
    )

  def test_build_index_with_runtime_kwargs_and_output_dir(self):
    index = self._embedding_index()
    output_dir = self.create_tempdir().full_path

    built = ot.run(
        self.runner.build_index(
            index=index,
            docs=_DOCS,
            output_dir=output_dir,
            runtime_kwargs={'k': 'v'},
        )
    )

    assert isinstance(built, indexing.EmbeddingBasedDocumentIndex)
    with self.subTest('builds_a_copy_with_runtime_kwargs'):
      self.assertEqual(built.runtime_kwargs, {'k': 'v'})
      self.assertEqual(built.num_docs, 2)
      self.assertEqual(index.num_docs, 0)
    with self.subTest('saves_the_index'):
      self.assertNotEmpty(os.listdir(output_dir))

  def test_build_index_from_jsonl_path(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      docs_path = os.path.join(temp_dir, 'docs.jsonl')
      with open(docs_path, 'w') as f:
        for doc in _DOCS:
          f.write(doc.to_json() + '\n')  # pyrefly: ignore[missing-attribute]

      index = ot.run(
          self.runner.build_index(index=self._embedding_index(), docs=docs_path)
      )

    self.assertEqual(index.num_docs, 2)

  def test_build_index_unsaveable_index_with_output_dir_raises_before_add(
      self,
  ):
    index = indexing.NaiveIndex()
    output_dir = self.create_tempdir().full_path

    with self.assertRaisesRegex(
        TypeError, 'NaiveIndex does not implement save'
    ):
      ot.run(
          self.runner.build_index(
              index=index, docs=_DOCS, output_dir=output_dir
          )
      )

    self.assertEqual(index.num_docs, 0)
    self.assertEmpty(os.listdir(output_dir))

  def test_build_index_unsaveable_index_without_output_dir_builds(self):
    index = ot.run(
        self.runner.build_index(index=indexing.NaiveIndex(), docs=_DOCS)
    )

    self.assertEqual(index.num_docs, 2)


if __name__ == '__main__':
  absltest.main()
