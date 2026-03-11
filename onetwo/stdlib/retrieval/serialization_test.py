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
import shutil
import tempfile

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization

_Document = retrieval_data_structures.Document


class SerializationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_simple_serialization_roundtrip(self):
    serializer = serialization.SimpleEmbeddingBasedDocumentIndexSerializer(
        base_path=self.test_dir
    )

    doc1 = _Document(
        doc_id='id1', content='content1', metadata={'key1': 'val1'}
    )
    doc2 = _Document(
        doc_id='id2', content='content2', metadata={'key2': 'val2'}
    )
    state = index_state.DocumentIndexState(
        docs=[doc1, doc2],
        doc_embeddings=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        doc_indices_by_discrete_value={
            'external_keys': {'key1': [0], 'key2': [1]}
        },
    )
    serializer.save(state)

    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'docs.jsonl')))
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'embeddings.npy'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'discrete_indices.json'))
    )

    new_state = ot.run(serializer.load())
    self.assertLen(new_state.docs, 2)
    self.assertEqual(new_state.docs[0].doc_id, 'id1')
    self.assertEqual(new_state.docs[1].doc_id, 'id2')
    self.assertEqual(new_state.docs[0].metadata, {'key1': 'val1'})

    self.assertLen(new_state.doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_state.doc_embeddings[0], state.doc_embeddings[0]
    )
    np.testing.assert_array_equal(
        new_state.doc_embeddings[1], state.doc_embeddings[1]
    )

    self.assertEqual(
        new_state.doc_indices_by_discrete_value,
        state.doc_indices_by_discrete_value,
    )


if __name__ == '__main__':
  absltest.main()
