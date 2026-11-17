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
    low_thinking = {
        'generate_object_kwargs': {
            'thinking_config': {'thinking_level': 'LOW'}
        }
    }
    self.assertEqual(config.kwargs, low_thinking)
    self.assertIsNot(
        config, rag_defaults.default_index_generate_backend_config()
    )

  def test_qa_generate_backend_kwargs(self):
    medium_thinking = {'thinking_config': {'thinking_level': 'MEDIUM'}}
    self.assertEqual(
        rag_defaults.qa_generate_backend_kwargs(
            rag_defaults.DEFAULT_QA_GENERATE_MODEL_NAME
        ),
        {
            'chat_kwargs': medium_thinking,
            'generate_object_kwargs': medium_thinking,
            'generate_with_tools_kwargs': medium_thinking,
        },
    )
    # Other models may not support the same thinking levels.
    self.assertEqual(
        rag_defaults.qa_generate_backend_kwargs('some-other-model'), {}
    )

  def test_default_tokenize_backend_config(self):
    config = rag_defaults.default_tokenize_backend_config()
    self.assertEqual(
        config.backend_class_name,
        rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
    )
    self.assertEqual(
        config.model_name, rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME
    )
    self.assertIsNot(config, rag_defaults.default_tokenize_backend_config())

  def test_build_index_default_index_uses_defaults(self):
    params = inspect.signature(build_index_lib.build_index).parameters
    # The default `IndexType.EMBEDDING` uses
    # `rag_defaults.default_embed_backend_config()`, see
    # `build_index_lib_test.test_flat_kwargs_default_embedder`.
    self.assertEqual(
        params['index'].default, build_index_lib.IndexType.EMBEDDING
    )


if __name__ == '__main__':
  absltest.main()
