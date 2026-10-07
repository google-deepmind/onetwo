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

import dataclasses
from typing import Any

import dataclasses_json
from onetwo.backends import backends_base
from onetwo.backends import claude_api
from onetwo.backends import google_genai_api
from onetwo.backends import ollama_api
from onetwo.backends import openai_api
from onetwo.backends import sentencepiece_tokenizer


BACKEND_REGISTRY: dict[str, type[backends_base.Backend]] = {
    'GoogleGenAIAPI': google_genai_api.GoogleGenAIAPI,
    'OllamaAPI': ollama_api.OllamaAPI,
    'ClaudeAPI': claude_api.ClaudeAPI,
    'OpenAIAPI': openai_api.OpenAIAPI,
    'SentencePieceTokenizer': sentencepiece_tokenizer.SentencePieceTokenizer,
}


@dataclasses.dataclass(frozen=True)
class BackendConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for a backend.

  Attributes:
    backend_class_name: The name of the backend class to instantiate (e.g.
      'GoogleGenAIAPI').
    model_name: The name of the model to use.
    kwargs: Additional arguments to unpack into the backend constructor. This
      dictionary should contain model hyperparameters (e.g., `temperature`,
      `max_tokens`, `output_dimensionality`). Note that sensitive credentials
      (`api_key`, `credentials`) and environment-specific deployment settings
      (`threadpool_size`, `project`, `location`, `http_options`) are to be
      omitted here and must be provided dynamically at runtime instead.
  """

  backend_class_name: str | None = None
  model_name: str | None = None
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __str__(self) -> str:
    """Returns a compact, human-readable string representation."""
    parts = []
    if self.backend_class_name:
      parts.append(self.backend_class_name)
    if self.model_name:
      parts.append(self.model_name)
    for k, v in sorted(self.kwargs.items()):
      parts.append(f'{k}={v}')
    return '-'.join(parts)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class GenerateBackendConfig(BackendConfig):
  """Configuration specific to generation backends."""
  pass


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class EmbedBackendConfig(BackendConfig):
  """Configuration specific to embedding backends."""
  pass


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class TokenizeBackendConfig(BackendConfig):
  """Configuration specific to tokenizer backends (e.g. for chunking)."""
  pass


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


def get_backend(
    config: BackendConfig,
    **runtime_kwargs: Any,
) -> backends_base.Backend:
  """Instantiates a backend from a configuration.

  Args:
    config: The serialized configuration containing model hyperparameters (e.g.,
      `temperature`, `max_tokens`, `output_dimensionality`).
    **runtime_kwargs: Environment-specific deployment settings (e.g.,
      `threadpool_size`, `project`) and credentials (`api_key`, `credentials`)
      that are not stored in the serializable configuration.

  Returns:
    An instance of the requested backend.
  """
  if config.backend_class_name is None:
    raise ValueError('backend_class_name must be specified.')

  kwargs = dict(config.kwargs)

  if config.model_name:
    if isinstance(config, GenerateBackendConfig):
      # TODO: Update GoogleGenAIAPI to use model_name directly.
      if config.backend_class_name == 'GoogleGenAIAPI':
        kwargs['generate_model_name'] = config.model_name
        kwargs['chat_model_name'] = config.model_name
      else:
        kwargs['model_name'] = config.model_name
    elif isinstance(config, TokenizeBackendConfig):
      kwargs['model_name'] = config.model_name
    else:
      kwargs['embed_model_name'] = config.model_name

  if runtime_kwargs:
    # Raise an error if we are overriding something from the BackendConfig.
    overlapping_keys = set(kwargs.keys()).intersection(runtime_kwargs.keys())
    if overlapping_keys:
      raise ValueError(
          f'Conflict between BackendConfig and runtime arguments. The following'
          f' keys should not be in both: {overlapping_keys}'
      )

    kwargs.update(runtime_kwargs)

  return instantiate_backend(class_name=config.backend_class_name, **kwargs)
