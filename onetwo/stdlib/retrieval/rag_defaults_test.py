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

import inspect

from absl.testing import absltest
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import rag_defaults


class RagDefaultsTest(absltest.TestCase):

  def test_default_embed_backend_config(self):
    config = rag_defaults.default_embed_backend_config()
    self.assertEqual(
        config.backend_class_name, rag_defaults.DEFAULT_EMBED_BACKEND_CLASS_NAME
    )
    self.assertEqual(config.model_name, rag_defaults.DEFAULT_EMBED_MODEL_NAME)
    # Each call returns a new config, so that their `kwargs` are not shared.
    self.assertIsNot(config, rag_defaults.default_embed_backend_config())

  def test_default_index_generate_backend_config(self):
    config = rag_defaults.default_index_generate_backend_config()
    self.assertEqual(
        config.backend_class_name,
        rag_defaults.DEFAULT_INDEX_GENERATE_BACKEND_CLASS_NAME,
    )
    self.assertEqual(
        config.model_name, rag_defaults.DEFAULT_INDEX_GENERATE_MODEL_NAME
    )
    self.assertIsNot(
        config, rag_defaults.default_index_generate_backend_config()
    )

  def test_embedding_specs_can_be_created_without_arguments(self):
    for spec_cls in (
        index_build_interface.EmbeddingBasedDocumentIndexSpec,
        index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    ):
      with self.subTest(spec_cls.__name__):
        spec = spec_cls()
        self.assertEqual(
            spec.embed_backend_config,
            rag_defaults.default_embed_backend_config(),
        )

  def test_hipporag_spec_can_be_created_without_arguments(self):
    spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec()
    self.assertEqual(
        spec.embed_backend_config, rag_defaults.default_embed_backend_config()
    )
    self.assertEqual(
        spec.generate_backend_config,
        rag_defaults.default_index_generate_backend_config(),
    )

  def test_spec_instances_do_not_share_backend_configs(self):
    spec_1 = index_build_interface.EmbeddingBasedDocumentIndexSpec()
    spec_2 = index_build_interface.EmbeddingBasedDocumentIndexSpec()
    self.assertIsNot(spec_1.embed_backend_config, spec_2.embed_backend_config)

  def test_build_index_uses_defaults(self):
    params = inspect.signature(build_index_lib.build_index).parameters
    self.assertEqual(
        params['embedding_backend_class_name'].default,
        rag_defaults.DEFAULT_EMBED_BACKEND_CLASS_NAME,
    )
    self.assertEqual(
        params['embedding_model_name'].default,
        rag_defaults.DEFAULT_EMBED_MODEL_NAME,
    )
    self.assertEqual(
        params['generation_backend_class_name'].default,
        rag_defaults.DEFAULT_INDEX_GENERATE_BACKEND_CLASS_NAME,
    )
    self.assertEqual(
        params['generation_model_name'].default,
        rag_defaults.DEFAULT_INDEX_GENERATE_MODEL_NAME,
    )


if __name__ == '__main__':
  absltest.main()
