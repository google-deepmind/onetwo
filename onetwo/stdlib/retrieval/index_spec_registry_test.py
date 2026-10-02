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
import json
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
import dataclasses_json
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import builtin_index_specs  # pylint: disable=unused-import
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document
KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class _SubclassedKeywordSpec(KeywordDocumentIndexSpec):
  extra_field: int = 1


class IndexSpecRegistryTest(absltest.TestCase):

  def test_builtin_specs_are_registered(self):
    self.assertContainsSubset(
        ['embedding', 'simple_rewriting_embedding'],
        index_spec_registry.list_index_specs(),
    )

  def test_register_and_list(self):
    with keyword_index_for_test.registered_keyword_index_spec():
      self.assertIn(
          'KeywordDocumentIndexSpec', index_spec_registry.list_index_specs()
      )
    self.assertNotIn(
        'KeywordDocumentIndexSpec', index_spec_registry.list_index_specs()
    )

  def test_register_duplicate_raises(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(ValueError, 'already registered'):
      index_spec_registry.register_index_spec(
          KeywordDocumentIndexSpec, keyword_index_for_test.make_keyword_index
      )

  def test_register_non_spec_raises(self):
    with self.assertRaisesRegex(TypeError, 'not a subclass of IndexSpec'):
      index_spec_registry.register_index_spec(
          cast(Any, str), keyword_index_for_test.make_keyword_index
      )

  def test_make_index_unregistered_raises_with_registered_names(self):
    with self.assertRaisesRegex(
        KeyError, 'No index factory registered for KeywordDocumentIndexSpec'
    ) as cm:
      index_spec_registry.make_index(KeywordDocumentIndexSpec())
    self.assertIn("'embedding'", str(cm.exception))

  def test_unregistered_subclass_does_not_fall_back_to_parent(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(
        KeyError, 'No index factory registered for _SubclassedKeywordSpec'
    ):
      index_spec_registry.make_index(_SubclassedKeywordSpec())

  def test_make_index_returns_configured_empty_index(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    index = index_spec_registry.make_index(
        KeywordDocumentIndexSpec(lowercase=False)
    )
    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertFalse(index.lowercase)
    self.assertEqual(index.num_docs, 0)

  def test_make_index_passes_runtime_kwargs_to_factory(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    factory = mock.Mock(
        return_value=keyword_index_for_test.KeywordDocumentIndex()
    )
    index_spec_registry.register_index_spec(_SubclassedKeywordSpec, factory)
    spec = _SubclassedKeywordSpec()

    index_spec_registry.make_index(spec, {'api_key': 'key'})

    factory.assert_called_once_with(spec, {'api_key': 'key'})

  def test_custom_spec_builds_via_index_building_without_runner_changes(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    docs = [
        Document(doc_id='d1', content='The capital of France is Paris.'),
        Document(doc_id='d2', content='Madrid is the capital of Spain.'),
    ]

    index = ot.run(
        index_building.build_index(spec=KeywordDocumentIndexSpec(), docs=docs)
    )

    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertEqual(index.num_docs, 2)
    results = ot.run(index.retrieve('What is in Spain?', max_results=1))
    self.assertEqual([doc.doc_id for doc in results], ['d2'])

  def test_register_with_name(self):
    self.enter_context(
        keyword_index_for_test.registered_keyword_index_spec(name='keyword')
    )

    self.assertIn('keyword', index_spec_registry.list_index_specs())
    self.assertNotIn(
        'KeywordDocumentIndexSpec', index_spec_registry.list_index_specs()
    )
    self.assertIsInstance(
        index_spec_registry.make_index(KeywordDocumentIndexSpec()),
        keyword_index_for_test.KeywordDocumentIndex,
    )

  def test_register_duplicate_name_raises(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(ValueError, 'name .* already registered'):
      index_spec_registry.register_index_spec(
          _SubclassedKeywordSpec,
          keyword_index_for_test.make_keyword_index,
          name='KeywordDocumentIndexSpec',
      )

  def test_spec_dict_round_trip(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    spec = KeywordDocumentIndexSpec(lowercase=False)

    spec_dict = index_spec_registry.spec_to_dict(spec)

    self.assertEqual(
        spec_dict,
        {
            'spec_type': 'KeywordDocumentIndexSpec',
            'spec': {'lowercase': False},
        },
    )
    self.assertEqual(index_spec_registry.spec_from_dict(spec_dict), spec)

  def test_builtin_spec_dict_round_trip_excludes_runtime_kwargs(self):
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='SomeEmbedder', model_name='some-model'
        ),
        max_chunk_size=128,
        runtime_kwargs={'api_key': 'secret'},
    )

    spec_dict = index_spec_registry.spec_to_dict(spec)

    self.assertEqual(spec_dict['spec_type'], 'simple_rewriting_embedding')
    self.assertNotIn('secret', json.dumps(spec_dict))
    self.assertEqual(
        index_spec_registry.spec_from_dict(spec_dict),
        dataclasses.replace(spec, runtime_kwargs={}),
    )

  def test_spec_to_dict_unregistered_raises(self):
    with self.assertRaisesRegex(KeyError, 'KeywordDocumentIndexSpec'):
      index_spec_registry.spec_to_dict(KeywordDocumentIndexSpec())

  def test_default_str_is_path_friendly(self):
    spec = KeywordDocumentIndexSpec()
    self.assertEqual(
        str(spec), f'KeywordDocumentIndexSpec__hash_{spec.get_stable_hash()}'
    )


if __name__ == '__main__':
  absltest.main()
