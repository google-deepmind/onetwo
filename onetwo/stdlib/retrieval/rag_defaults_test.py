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
from unittest import mock

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import rag_defaults


class RagDefaultsTest(absltest.TestCase):

  def test_google_genai_api_defaults_are_registered(self):
    medium_thinking = {'thinking_config': {'thinking_level': 'MEDIUM'}}
    generate_request_kwargs = {
        'max_retries': 5,
        'http_options': {'timeout': 300_000},
        'threadpool_size': 10,
    }
    self.assertEqual(
        rag_defaults.default_embed_backend_config(),
        backend_registry.EmbedBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name='gemini-embedding-2',
            kwargs={
                'max_retries': 5,
                'http_options': {'timeout': 60_000},
                'threadpool_size': 20,
            },
        ),
    )
    self.assertEqual(
        rag_defaults.default_index_generate_backend_config(),
        backend_registry.GenerateBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name='gemini-4-hydrogen',
            kwargs={
                'generate_object_kwargs': {
                    'thinking_config': {'thinking_level': 'LOW'}
                },
                **generate_request_kwargs,
            },
        ),
    )
    self.assertEqual(
        rag_defaults.default_qa_generate_backend_config('GoogleGenAIAPI'),
        backend_registry.GenerateBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name='gemini-4-hydrogen',
            kwargs={
                'chat_kwargs': medium_thinking,
                'generate_object_kwargs': medium_thinking,
                'generate_with_tools_kwargs': medium_thinking,
                **generate_request_kwargs,
            },
        ),
    )
    self.assertEqual(
        rag_defaults.default_ai_rater_backend_config(),
        backend_registry.GenerateBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name='gemini-3.6-flash',
            kwargs=generate_request_kwargs,
        ),
    )

  def test_getters_return_copies(self):
    config = rag_defaults.default_qa_generate_backend_config()
    config.kwargs['chat_kwargs']['thinking_config']['thinking_level'] = 'HIGH'

    self.assertEqual(
        rag_defaults.default_qa_generate_backend_config().kwargs['chat_kwargs'],
        {'thinking_config': {'thinking_level': 'MEDIUM'}},
    )

  def test_has_backend_defaults(self):
    self.assertTrue(rag_defaults.has_backend_defaults('GoogleGenAIAPI'))
    self.assertFalse(rag_defaults.has_backend_defaults('UnknownAPI'))

  def test_unknown_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "'Unknown'.*'GoogleGenAIAPI'"):
      rag_defaults.default_embed_backend_config('Unknown')

  def test_register_backend_defaults(self):
    self.enter_context(mock.patch.dict(rag_defaults._BACKEND_DEFAULTS))
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='FakeAPI', model_name='fake-model'
    )
    rag_defaults.register_backend_defaults(
        'FakeAPI', rag_defaults.BackendDefaults(qa_generate=config)
    )

    self.assertEqual(
        rag_defaults.default_qa_generate_backend_config('FakeAPI'), config
    )
    with self.assertRaisesRegex(ValueError, "'FakeAPI' has no default embed"):
      rag_defaults.default_embed_backend_config('FakeAPI')

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


class QaGenerateBackendConfigTest(absltest.TestCase):

  def test_default_model_keeps_default_kwargs(self):
    default = rag_defaults.default_qa_generate_backend_config()

    for model_name in (None, default.model_name):
      with self.subTest(str(model_name)):
        self.assertEqual(
            rag_defaults.qa_generate_backend_config(
                rag_defaults.DEFAULT_BACKEND_CLASS_NAME, model_name
            ),
            default,
        )

  def test_other_model_keeps_only_request_kwargs(self):
    config = rag_defaults.qa_generate_backend_config(
        'GoogleGenAIAPI', 'other-model'
    )

    self.assertEqual(config.model_name, 'other-model')
    self.assertEqual(
        config.kwargs,
        {
            'max_retries': 5,
            'http_options': {'timeout': 300_000},
            'threadpool_size': 10,
        },
    )

  def test_backend_without_defaults_with_explicit_model(self):
    config = rag_defaults.qa_generate_backend_config(
        'BackendWithoutDefaults', 'some-model'
    )

    self.assertEqual(config.backend_class_name, 'BackendWithoutDefaults')
    self.assertEqual(config.model_name, 'some-model')
    self.assertEqual(config.kwargs, {})


if __name__ == '__main__':
  absltest.main()
