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

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.backends import claude_api
from onetwo.backends import google_genai_api
from onetwo.backends import ollama_api
from onetwo.backends import openai_api


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


if __name__ == '__main__':
  absltest.main()
