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
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document
KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec

_DOCS = [
    Document(doc_id='d1', content='Paris is in France.'),
    Document(doc_id='d2', content='Madrid is in Spain.'),
]


class IndexBuildingTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
