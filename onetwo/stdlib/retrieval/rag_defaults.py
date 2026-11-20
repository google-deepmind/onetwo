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

"""Default backends, models and settings of the OneTwo RAG API.

All the default backends and models of `rag.build_index` and `rag.evaluate`
(and of their command-line tools) are defined here. The default configs of a
backend class are registered with `register_backend_defaults`: other backends
(e.g. internal ones) register theirs from their own modules. Currently only
GoogleGenAIAPI defaults are registered here. Defaults specific to an index or
chunker are the defaults of its class.

The defaults are updated as newer models become available. Saved indexes keep
the configs they were built with (in `index_config.json`), so they keep loading
with their models until those models are turned down. Evaluation defaults are
not saved: pass the models explicitly to compare evaluations across changes of
the defaults.

The default configs also set request settings (e.g. retries, HTTP timeout and
number of threads), so they are saved with the indexes too. Like the models,
these defaults may change: the only guarantee is that saved indexes keep
loading. Runtime arguments (e.g. `extra_backend_kwargs` of `rag.build_index`
and `rag.evaluate`) override the request settings of a config.
"""

import copy
import dataclasses
from typing import Any, TypeVar

from onetwo.backends import backend_registry

# Default backend class and models of `rag.build_index` and `rag.evaluate`
# (registered below for `GoogleGenAIAPI`).
DEFAULT_BACKEND_CLASS_NAME = 'GoogleGenAIAPI'
DEFAULT_GENERATE_MODEL = 'gemini-4-hydrogen'
DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-2'


@dataclasses.dataclass(frozen=True)
class BackendDefaults:
  """Default configs of one backend class, per role.

  A role that the backend does not support is None.

  Attributes:
    embed: Config to embed documents and queries.
    index_generate: Config to generate while building indexes (e.g. GraphRAG
      entity and triple extraction).
    qa_generate: Config to generate answers when evaluating QA strategies.
  """

  embed: backend_registry.EmbedBackendConfig | None = None
  index_generate: backend_registry.GenerateBackendConfig | None = None
  qa_generate: backend_registry.GenerateBackendConfig | None = None


_BACKEND_DEFAULTS: dict[str, BackendDefaults] = {}

_ConfigT = TypeVar('_ConfigT', bound=backend_registry.BackendConfig)


def register_backend_defaults(
    backend_class_name: str, defaults: BackendDefaults
) -> None:
  """Registers the default configs of a backend class.

  Args:
    backend_class_name: Name of the backend class.
    defaults: Default configs of the backend class.
  """
  _BACKEND_DEFAULTS[backend_class_name] = defaults


def has_backend_defaults(backend_class_name: str) -> bool:
  """Returns whether default configs are registered for a backend class."""
  return backend_class_name in _BACKEND_DEFAULTS


def _backend_defaults(backend_class_name: str) -> BackendDefaults:
  """Returns the registered defaults of a backend class."""
  if backend_class_name not in _BACKEND_DEFAULTS:
    raise ValueError(
        f'No defaults registered for backend class {backend_class_name!r}.'
        f' Registered: {sorted(_BACKEND_DEFAULTS)}. Import the module that'
        ' registers its defaults.'
    )
  return _BACKEND_DEFAULTS[backend_class_name]


def _copy(
    backend_class_name: str, config: _ConfigT | None, role: str
) -> _ConfigT:
  """Returns a copy of a default config, which must exist."""
  if config is None:
    raise ValueError(f'{backend_class_name!r} has no default {role} config.')
  return copy.deepcopy(config)


def default_embed_backend_config(
    backend_class_name: str = DEFAULT_BACKEND_CLASS_NAME,
) -> backend_registry.EmbedBackendConfig:
  """Returns a new copy of the default embedding config of a backend class.

  Args:
    backend_class_name: Name of the backend class.
  """
  return _copy(
      backend_class_name,
      _backend_defaults(backend_class_name).embed,
      'embedding',
  )


def default_index_generate_backend_config(
    backend_class_name: str = DEFAULT_BACKEND_CLASS_NAME,
) -> backend_registry.GenerateBackendConfig:
  """Returns a new copy of the default index-building generation config.

  Args:
    backend_class_name: Name of the backend class.
  """
  return _copy(
      backend_class_name,
      _backend_defaults(backend_class_name).index_generate,
      'index generation',
  )


def default_qa_generate_backend_config(
    backend_class_name: str = DEFAULT_BACKEND_CLASS_NAME,
) -> backend_registry.GenerateBackendConfig:
  """Returns a new copy of the default QA generation config.

  Args:
    backend_class_name: Name of the backend class.
  """
  return _copy(
      backend_class_name,
      _backend_defaults(backend_class_name).qa_generate,
      'QA generation',
  )


def qa_generate_backend_config(
    backend_class_name: str = DEFAULT_BACKEND_CLASS_NAME,
    model_name: str | None = None,
) -> backend_registry.GenerateBackendConfig:
  """Returns the QA generation config of a backend class and model.

  Transitional: will be removed when `rag.evaluate` takes strategies with their
  own backend config.

  Without a model, or with the default model of the backend class, this is its
  default QA config (e.g. with its thinking level). Any other model gets no
  model-specific kwargs (e.g. thinking), since it may not support them, but
  GoogleGenAIAPI models keep the default request settings. Any backend class
  can be used with an explicit model.

  Args:
    backend_class_name: Name of the generation backend class.
    model_name: Name of the generation model, or None for the default one.
  """
  if model_name is None:
    return default_qa_generate_backend_config(backend_class_name)
  defaults = _BACKEND_DEFAULTS.get(backend_class_name)
  if (
      defaults
      and defaults.qa_generate
      and (defaults.qa_generate.model_name == model_name)
  ):
    return copy.deepcopy(defaults.qa_generate)
  kwargs = {}
  if backend_class_name == 'GoogleGenAIAPI':
    kwargs = copy.deepcopy(_GENERATE_REQUEST_KWARGS)
  return backend_registry.GenerateBackendConfig(
      backend_class_name=backend_class_name,
      model_name=model_name,
      kwargs=kwargs,
  )


def _thinking(level: str) -> dict[str, Any]:
  return {'thinking_config': {'thinking_level': level}}


# Request settings of the GoogleGenAIAPI default configs: the maximum number of
# retries, the HTTP timeout in milliseconds and the number of threads. Embedding
# requests are short, generation requests (with thinking) can be long.
_EMBED_REQUEST_KWARGS: dict[str, Any] = {
    'max_retries': 5,
    'http_options': {'timeout': 60_000},
    'threadpool_size': 20,
}
_GENERATE_REQUEST_KWARGS: dict[str, Any] = {
    'max_retries': 5,
    'http_options': {'timeout': 300_000},
    'threadpool_size': 10,
}


register_backend_defaults(
    'GoogleGenAIAPI',
    BackendDefaults(
        embed=backend_registry.EmbedBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name=DEFAULT_EMBEDDING_MODEL,
            kwargs=copy.deepcopy(_EMBED_REQUEST_KWARGS),
        ),
        # GraphRAG extracts entities and triples with `llm.generate_object`:
        # LOW thinking keeps these many calls fast and cheap.
        index_generate=backend_registry.GenerateBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name=DEFAULT_GENERATE_MODEL,
            kwargs={
                'generate_object_kwargs': _thinking('LOW'),
                **copy.deepcopy(_GENERATE_REQUEST_KWARGS),
            },
        ),
        # MEDIUM thinking for every call of the QA strategies.
        qa_generate=backend_registry.GenerateBackendConfig(
            backend_class_name='GoogleGenAIAPI',
            model_name=DEFAULT_GENERATE_MODEL,
            kwargs={
                'chat_kwargs': _thinking('MEDIUM'),
                'generate_object_kwargs': _thinking('MEDIUM'),
                'generate_with_tools_kwargs': _thinking('MEDIUM'),
                **copy.deepcopy(_GENERATE_REQUEST_KWARGS),
            },
        ),
    ),
)

# Default tokenizer to chunk documents by tokens, used by
# `routing_chunking.RoutingChunker` and the MCP server.
DEFAULT_TOKENIZE_BACKEND_CLASS_NAME = 'SentencePieceTokenizer'
DEFAULT_TOKENIZE_MODEL_NAME = (
    'https://huggingface.co/t5-small/resolve/main/spiece.model'
)


def default_tokenize_backend_config() -> backend_registry.TokenizeBackendConfig:
  """Returns a new config for the default tokenizer."""
  return backend_registry.TokenizeBackendConfig(
      backend_class_name=DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
      model_name=DEFAULT_TOKENIZE_MODEL_NAME,
  )


# TODO: Handle the default AI rater model name in a similar way
# to the generation and embedding models.
DEFAULT_AI_RATER_GENERATE_MODEL_NAME = 'gemini-3.6-flash'


def default_ai_rater_backend_config() -> backend_registry.GenerateBackendConfig:
  """Returns a new config for the default AI rater of `rag.evaluate`."""
  return backend_registry.GenerateBackendConfig(
      backend_class_name='GoogleGenAIAPI',
      model_name=DEFAULT_AI_RATER_GENERATE_MODEL_NAME,
      kwargs=copy.deepcopy(_GENERATE_REQUEST_KWARGS),
  )


# Backend request settings applied by `rag.build_index` (and its command-line
# tool) unless set in its `extra_backend_kwargs`: the maximum number of retries
# (`extra_backend_kwargs['max_retries']`) and the HTTP timeout in milliseconds
# (`extra_backend_kwargs['http_options']['timeout']`).
DEFAULT_MAX_RETRIES = 5
DEFAULT_HTTP_TIMEOUT_MS = 60_000
