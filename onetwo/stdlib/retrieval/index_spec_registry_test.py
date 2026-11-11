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
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
import dataclasses_json
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import keyword_index_for_test
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document
KeywordDocumentIndexSpec = keyword_index_for_test.KeywordDocumentIndexSpec


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class _SubclassedKeywordSpec(KeywordDocumentIndexSpec):
  extra_field: int = 1


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class _SpecWithoutMakeIndex(index_build_interface.IndexSpec):
  param: int = 1


class IndexSpecRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Restore the registry after each test, including decorator registrations.
    # pylint: disable=protected-access
    self.enter_context(
        mock.patch.dict(index_spec_registry._REGISTRATIONS_BY_NAME)
    )
    self.enter_context(
        mock.patch.dict(index_spec_registry._REGISTRATIONS_BY_TYPE)
    )
    # pylint: enable=protected-access

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

  def test_decorator_returns_the_class(self):
    decorated = index_spec_registry.register_index_spec('keyword')(
        KeywordDocumentIndexSpec
    )

    self.assertIs(decorated, KeywordDocumentIndexSpec)

  def test_decorator_on_class_definition(self):
    @index_spec_registry.register_index_spec('decorated')
    @dataclasses_json.dataclass_json
    @dataclasses.dataclass(kw_only=True)
    class DecoratedSpec(KeywordDocumentIndexSpec):
      pass

    index = index_spec_registry.make_index(DecoratedSpec(lowercase=False))

    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertFalse(getattr(index, 'lowercase'))
    self.assertEqual(
        index_spec_registry.spec_to_dict(DecoratedSpec())['spec_type'],
        'decorated',
    )

  def test_register_with_name(self):
    self.enter_context(
        keyword_index_for_test.registered_keyword_index_spec(name='keyword')
    )

    self.assertIn('keyword', index_spec_registry.list_index_specs())
    self.assertNotIn(
        'KeywordDocumentIndexSpec', index_spec_registry.list_index_specs()
    )

  def test_register_duplicate_raises(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(ValueError, 'already registered'):
      index_spec_registry.register_index_spec('other_name')(
          KeywordDocumentIndexSpec
      )

  def test_register_duplicate_name_raises(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(ValueError, 'name .* already registered'):
      index_spec_registry.register_index_spec('KeywordDocumentIndexSpec')(
          _SubclassedKeywordSpec
      )

  def test_register_non_spec_raises(self):
    with self.assertRaisesRegex(TypeError, 'not a subclass of IndexSpec'):
      index_spec_registry.register_index_spec('not_a_spec')(cast(Any, str))

  def test_register_without_make_index_raises(self):
    with self.assertRaisesRegex(TypeError, 'must implement make_index'):
      index_spec_registry.register_index_spec('no_make_index')(
          _SpecWithoutMakeIndex
      )
    self.assertNotIn('no_make_index', index_spec_registry.list_index_specs())

  def test_base_make_index_raises(self):
    with self.assertRaisesRegex(NotImplementedError, 'make_index'):
      _SpecWithoutMakeIndex().make_index({})

  def test_make_index_unregistered_raises_with_registered_names(self):
    with self.assertRaisesRegex(
        KeyError, 'Index spec KeywordDocumentIndexSpec is not registered'
    ) as cm:
      index_spec_registry.make_index(KeywordDocumentIndexSpec())
    self.assertIn("'embedding'", str(cm.exception))

  def test_unregistered_subclass_does_not_fall_back_to_parent(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    with self.assertRaisesRegex(
        KeyError, 'Index spec _SubclassedKeywordSpec is not registered'
    ):
      index_spec_registry.make_index(_SubclassedKeywordSpec())

  def test_registered_subclass_reuses_parent_make_index(self):
    index_spec_registry.register_index_spec()(_SubclassedKeywordSpec)

    index = index_spec_registry.make_index(
        _SubclassedKeywordSpec(lowercase=False)
    )

    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertFalse(getattr(index, 'lowercase'))

  def test_make_index_returns_configured_empty_index(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    index = index_spec_registry.make_index(
        KeywordDocumentIndexSpec(lowercase=False)
    )
    self.assertIsInstance(index, keyword_index_for_test.KeywordDocumentIndex)
    self.assertFalse(getattr(index, 'lowercase'))
    self.assertEqual(index.num_docs, 0)

  def test_make_index_passes_runtime_kwargs_to_make_index(self):
    self.enter_context(keyword_index_for_test.registered_keyword_index_spec())
    mock_make_index = self.enter_context(
        mock.patch.object(
            KeywordDocumentIndexSpec,
            'make_index',
            autospec=True,
            return_value=keyword_index_for_test.KeywordDocumentIndex(),
        )
    )
    spec = KeywordDocumentIndexSpec()

    index_spec_registry.make_index(spec, {'api_key': 'key'})

    mock_make_index.assert_called_once_with(spec, {'api_key': 'key'})

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

  def test_builtin_spec_dict_round_trip(self):
    spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='SomeEmbedder', model_name='some-model'
        ),
        chunker_config=chunker_registry.ChunkerConfig(
            chunker_name='words',
            kwargs={'max_words_per_chunk': 128},
        ),
    )

    spec_dict = index_spec_registry.spec_to_dict(spec)

    self.assertEqual(spec_dict['spec_type'], 'simple_rewriting_embedding')
    self.assertEqual(index_spec_registry.spec_from_dict(spec_dict), spec)

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
