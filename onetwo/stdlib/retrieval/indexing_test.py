# Copyright 2025 DeepMind Technologies Limited.
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
import numpy as np
from onetwo import ot
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document


# Mock llm.embed to be async
@executing.make_executable(copy_self=False)  # pytype: disable=wrong-arg-count
async def mock_embed_fn(text: str) -> np.ndarray:
  """Mock async embed function."""
  del text
  embedding = mock_embed_fn.embeddings[mock_embed_fn.call_count]
  mock_embed_fn.call_count += 1
  return embedding


mock_embed_fn.embeddings = []
mock_embed_fn.call_count = 0


class NaiveIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_embed_fn.embeddings = []
    mock_embed_fn.call_count = 0
    self.mock_llm_embed = mock.patch(
        'onetwo.builtins.llm.embed', new=mock_embed_fn
    ).start()
    self.addCleanup(mock.patch.stopall)

  def testAddingAndRetrievingDocuments(self):
    index = indexing.NaiveIndex()

    with self.subTest('CreateInitialIndex'):
      docs = [
          Document(doc_id='doc1', content='This is document one.'),
          Document(doc_id='doc2', content='This is document two.'),
          Document(doc_id='doc3', content='This is document three.'),
      ]
      ot.run(index.add_docs(docs))  # pytype: disable=wrong-arg-count
      self.assertEqual(index.num_docs, 3)

    with self.subTest('AddMoreDocuments'):
      more_docs = [
          Document(doc_id='doc4', content='This is document four.'),
          Document(doc_id='doc5', content='This is document five.'),
      ]
      ot.run(index.add_docs(more_docs))  # pytype: disable=wrong-arg-count
      self.assertEqual(index.num_docs, 5)

    with self.subTest('RetrieveDocuments'):
      query = 'find document one'
      retrieved_docs = ot.run(index(query, max_results=2))  # pytype: disable=wrong-arg-count

      retrieved_list = list(retrieved_docs)
      self.assertLen(retrieved_list, 2)

      self.assertEqual(retrieved_list[0].doc_id, 'doc1')
      self.assertEqual(retrieved_list[1].doc_id, 'doc2')

      index.destroy_index()


if __name__ == '__main__':
  absltest.main()
