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

"""Registry for OneTwo backend classes."""

from typing import Any

from onetwo.backends import backends_base
from onetwo.backends import claude_api
from onetwo.backends import google_genai_api
from onetwo.backends import ollama_api
from onetwo.backends import openai_api


BACKEND_REGISTRY: dict[str, type[backends_base.Backend]] = {
    'GoogleGenAIAPI': google_genai_api.GoogleGenAIAPI,
    'OllamaAPI': ollama_api.OllamaAPI,
    'ClaudeAPI': claude_api.ClaudeAPI,
    'OpenAIAPI': openai_api.OpenAIAPI,
}


def instantiate_backend(
    class_name: str, **kwargs: Any
) -> backends_base.Backend:
  """Instantiates a backend from the registry.

  Args:
    class_name: The name of the backend class to instantiate (e.g.
      'GoogleGenAIAPI').
    **kwargs: Arguments to pass to the backend constructor.

  Returns:
    An instance of the requested backend.

  Raises:
    ValueError: If the class_name is not found in the registry.
  """
  if class_name not in BACKEND_REGISTRY:
    raise ValueError(f'Unknown backend class: {class_name}')
  return BACKEND_REGISTRY[class_name](**kwargs)
