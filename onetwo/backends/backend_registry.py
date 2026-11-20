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

from collections.abc import Iterator, Mapping
import contextlib
import copy
import dataclasses
from typing import Any

from absl import logging
import dataclasses_json
from onetwo.backends import backends_base
from onetwo.backends import claude_api
from onetwo.backends import google_genai_api
from onetwo.backends import ollama_api
from onetwo.backends import openai_api
from onetwo.backends import sentencepiece_tokenizer
from onetwo.core import routing


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
    kwargs: Additional arguments to unpack into the backend constructor (e.g.,
      `temperature`, `max_tokens`, `output_dimensionality`, `max_retries`).
      Sensitive credentials (`api_key`, `credentials`) must not be stored here;
      pass them at runtime to `get_backend` instead.
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
    **runtime_kwargs: Arguments that are not stored in the serializable
      configuration, such as credentials (`api_key`, `credentials`) and
      environment-specific settings (e.g., `project`). They override the
      same-named top-level keys of `config.kwargs`; nested values (e.g.,
      `http_options`) are replaced, not merged.

  Returns:
    An instance of the requested backend.
  """
  if config.backend_class_name is None:
    raise ValueError('backend_class_name must be specified.')

  # Deep copy, as backends may modify nested values (e.g., `http_options`).
  kwargs = copy.deepcopy(config.kwargs)

  if config.model_name:
    if isinstance(config, (GenerateBackendConfig, TokenizeBackendConfig)):
      kwargs['model_name'] = config.model_name
    else:
      kwargs['embed_model_name'] = config.model_name

  overridden_keys = sorted(set(kwargs) & set(runtime_kwargs))
  if overridden_keys:
    logging.info(
        'Runtime arguments override backend config kwargs: %s',
        overridden_keys,
    )
  kwargs.update(runtime_kwargs)

  return instantiate_backend(class_name=config.backend_class_name, **kwargs)


@dataclasses.dataclass
class BackendMixin:
  """Mixin for dataclasses that own an optional backend.

  Attributes:
    backend_config: Configuration of the backend (generation, embedding,
      tokenizer, etc.), or None to use whichever backend is currently
      registered.
    runtime_kwargs: Environment settings and credentials for the backend
      (excluded from `repr` and equality, and never serialized).
  """

  backend_config: BackendConfig | None = dataclasses.field(
      default=None, kw_only=True
  )
  runtime_kwargs: Mapping[str, Any] | None = dataclasses.field(
      default=None, repr=False, compare=False, kw_only=True
  )
  _backend: backends_base.Backend | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  @contextlib.contextmanager
  def _backend_context(self) -> Iterator[None]:
    """Registers this object's backend in a local registry context."""
    if self.backend_config is None:
      yield
      return
    if self._backend is None:
      self._backend = get_backend(
          self.backend_config, **(self.runtime_kwargs or {})
      )
    with routing.RegistryContext():
      self._backend.register()
      yield
