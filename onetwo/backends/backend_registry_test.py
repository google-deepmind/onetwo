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
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backend_registry
from onetwo.backends import backends_base
from onetwo.backends import claude_api
from onetwo.backends import google_genai_api
from onetwo.backends import ollama_api
from onetwo.backends import openai_api
from onetwo.backends import sentencepiece_tokenizer
from onetwo.core import routing


class BackendRegistryTest(absltest.TestCase):

  def test_registry_contains_expected_backends(self):
    self.assertIn('GoogleGenAIAPI', backend_registry.BACKEND_REGISTRY)
    self.assertIn('OllamaAPI', backend_registry.BACKEND_REGISTRY)
    self.assertIn('ClaudeAPI', backend_registry.BACKEND_REGISTRY)
    self.assertIn('OpenAIAPI', backend_registry.BACKEND_REGISTRY)
    self.assertIs(
        backend_registry.BACKEND_REGISTRY['GoogleGenAIAPI'],
        google_genai_api.GoogleGenAIAPI,
    )
    self.assertIs(
        backend_registry.BACKEND_REGISTRY['OllamaAPI'],
        ollama_api.OllamaAPI,
    )
    self.assertIs(
        backend_registry.BACKEND_REGISTRY['ClaudeAPI'],
        claude_api.ClaudeAPI,
    )
    self.assertIs(
        backend_registry.BACKEND_REGISTRY['OpenAIAPI'],
        openai_api.OpenAIAPI,
    )

  def test_instantiate_backend(self):
    backend = backend_registry.instantiate_backend(
        'GoogleGenAIAPI', api_key='test'
    )
    self.assertIsInstance(backend, google_genai_api.GoogleGenAIAPI)

  def test_get_backend_google_genai(self):
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='gemini-1.5-pro',
        kwargs={'temperature': 0.5},
    )
    backend = backend_registry.get_backend(config, api_key='test_key')
    self.assertIsInstance(backend, google_genai_api.GoogleGenAIAPI)
    self.assertEqual(backend.model_name, 'gemini-1.5-pro')
    self.assertEqual(backend.generate_model_name, 'gemini-1.5-pro')
    self.assertEqual(backend.chat_model_name, 'gemini-1.5-pro')
    self.assertEqual(backend.temperature, 0.5)

  def test_get_backend_google_genai_embedding(self):
    config = backend_registry.EmbedBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='text-embedding-004',
    )
    backend = backend_registry.get_backend(config, api_key='test_key')
    self.assertIsInstance(backend, google_genai_api.GoogleGenAIAPI)
    self.assertEqual(backend.embed_model_name, 'text-embedding-004')

  def test_get_backend_other_backend(self):
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='OllamaAPI',
        model_name='llama2',
        kwargs={'max_tokens': 100},
    )
    backend = backend_registry.get_backend(config)
    self.assertIsInstance(backend, ollama_api.OllamaAPI)
    self.assertEqual(backend.model_name, 'llama2')
    self.assertEqual(backend.max_tokens, 100)

  def test_get_backend_tokenizer(self):
    self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )
    config = backend_registry.TokenizeBackendConfig(
        backend_class_name='SentencePieceTokenizer', model_name='GEMINI3_G145'
    )
    backend = backend_registry.get_backend(config)
    self.assertIsInstance(
        backend, sentencepiece_tokenizer.SentencePieceTokenizer
    )
    self.assertEqual(backend.model_name, 'GEMINI3_G145')

  def test_get_backend_conflict(self):
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        kwargs={'temperature': 0.5},
    )
    with self.assertRaisesRegex(
        ValueError, 'Conflict between BackendConfig and runtime arguments'
    ):
      backend_registry.get_backend(config, temperature=0.8)

  def test_backend_config_serialization(self):
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='gemini-1.5-pro',
        kwargs={'temperature': 0.5},
    )
    json_str = config.to_json()
    reconstructed = backend_registry.GenerateBackendConfig.from_json(json_str)
    self.assertEqual(config, reconstructed)

  def test_backend_config_str(self):
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='gemini-1.5-pro',
        kwargs={'temperature': 0.5},
    )
    self.assertEqual(
        str(config), 'GoogleGenAIAPI-gemini-1.5-pro-temperature=0.5'
    )


@dataclasses.dataclass
class _FakeBackend(backends_base.Backend):
  label: str

  def register(self, name: str | None = None):
    del name
    routing.function_registry['op'] = lambda: self.label


@dataclasses.dataclass
class _MixinConsumer(backend_registry.BackendMixin):
  tag: str = 'default'

  def read_op(self) -> str:
    with self._backend_context():
      return routing.function_registry['op']()


class BackendMixinTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'generate',
          backend_registry.GenerateBackendConfig,
          'model_name',
      ),
      (
          'embed',
          backend_registry.EmbedBackendConfig,
          'embed_model_name',
      ),
      (
          'tokenize',
          backend_registry.TokenizeBackendConfig,
          'model_name',
      ),
  )
  def test_caches_backend_once_and_forwards_runtime_kwargs(
      self, config_cls, expected_model_kwarg
  ):
    seen_kwargs: list[dict[str, Any]] = []

    def _make_backend(**kwargs):
      seen_kwargs.append(kwargs)
      return _FakeBackend(label=f'owned:{kwargs[expected_model_kwarg]}')

    config = config_cls(
        backend_class_name='Fake',
        model_name='test-model',
        kwargs={'extra': 1},
    )
    consumer = _MixinConsumer(
        'pos-arg',
        backend_config=config,
        runtime_kwargs=types.MappingProxyType({'api_key': 'secret-key'}),
    )

    with mock.patch.dict(
        backend_registry.BACKEND_REGISTRY, {'Fake': _make_backend}
    ):
      with routing.RegistryContext():
        first = consumer.read_op()
        second = consumer.read_op()

    self.assertEqual((first, second), ('owned:test-model', 'owned:test-model'))
    self.assertEqual(
        seen_kwargs,
        [{
            expected_model_kwarg: 'test-model',
            'extra': 1,
            'api_key': 'secret-key',
        }],
    )
    self.assertNotIn('secret-key', repr(consumer))
    self.assertEqual(
        consumer,
        _MixinConsumer(
            'pos-arg',
            backend_config=config,
            runtime_kwargs={'api_key': 'different-key'},
        ),
    )

    # dataclasses.replace drops the cached _backend so the new config applies.
    replaced = dataclasses.replace(
        consumer,
        backend_config=config_cls(
            backend_class_name='Fake', model_name='other-model'
        ),
    )
    with mock.patch.dict(
        backend_registry.BACKEND_REGISTRY, {'Fake': _make_backend}
    ):
      with routing.RegistryContext():
        self.assertEqual(replaced.read_op(), 'owned:other-model')
    self.assertLen(seen_kwargs, 2)

  def test_restores_ambient_registry_and_noop_when_config_is_none(self):
    factory = mock.MagicMock(return_value=_FakeBackend(label='owned'))
    config = backend_registry.GenerateBackendConfig(
        backend_class_name='Fake', model_name='m'
    )
    owned = _MixinConsumer(backend_config=config)
    ambient_only = _MixinConsumer()

    with mock.patch.dict(
        backend_registry.BACKEND_REGISTRY, {'Fake': factory}
    ):
      with routing.RegistryContext():
        routing.function_registry['op'] = lambda: 'ambient'
        self.assertEqual(ambient_only.read_op(), 'ambient')
        factory.assert_not_called()
        self.assertEqual(owned.read_op(), 'owned')
        self.assertEqual(routing.function_registry['op'](), 'ambient')
    factory.assert_called_once()


if __name__ == '__main__':
  absltest.main()
