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

"""Default models and settings used by the OneTwo RAG API.

All default backends and models of `rag.build_index`, `rag.evaluate` and their
command-line tools, as well as the default backend request settings of
`rag.build_index`, are defined here, so that they can be found (and updated) in
one place.

Defaults specific to one index or chunker (e.g. chunk size or searcher) are the
defaults of its class.
"""

from typing import Any

from onetwo.backends import backend_registry

# Embedding model used to build and query embedding-based indexes.
DEFAULT_EMBED_BACKEND_CLASS_NAME = 'GoogleGenAIAPI'
DEFAULT_EMBED_MODEL_NAME = 'gemini-embedding-2'

# Tokenizer used to chunk documents by tokens (`chunking_strategy='tokens'`).
# The Gemini tokenizer is only available inside Google (see
# `sentencepiece_tokenizer.TOKENIZER_PATHS`), so the open-source release
# defaults to a publicly downloadable SentencePiece model.
DEFAULT_TOKENIZE_BACKEND_CLASS_NAME = 'SentencePieceTokenizer'
DEFAULT_TOKENIZE_MODEL_NAME = (
    'https://huggingface.co/t5-small/resolve/main/spiece.model'
)

# Generation model used while building indexes (e.g. HippoRAG extraction), with
# low thinking to keep the many extraction calls fast and cheap (LOW is the
# lowest level supported by gemini-4-hydrogen, which rejects MINIMAL).
DEFAULT_INDEX_GENERATE_BACKEND_CLASS_NAME = 'GoogleGenAIAPI'
DEFAULT_INDEX_GENERATE_MODEL_NAME = 'gemini-4-hydrogen'
# Constructor kwargs of the `GoogleGenAIAPI` index generation backend. HippoRAG
# extracts entities and triples with `llm.generate_object`.
DEFAULT_INDEX_GENERATE_BACKEND_KWARGS = {
    'generate_object_kwargs': {
        'thinking_config': {'thinking_level': 'LOW'}
    },
}

# Generation models used by `rag.evaluate`: to answer questions, and by the AI
# rater to grade the answers.
# TODO: Decide whether index building and evaluation should use
# the same default generation model.
DEFAULT_QA_GENERATE_BACKEND_CLASS_NAME = 'GoogleGenAIAPI'
DEFAULT_QA_GENERATE_MODEL_NAME = 'gemini-4-hydrogen'
# Constructor kwargs of the default `GoogleGenAIAPI` QA backend, with medium
# thinking for every call made by the QA strategies: `llm.chat` and
# `llm.generate_object` (e.g. `rag`), and `llm.generate_with_tools`
# (`agentic_retrieval`). Only applied with the default QA model, since other
# models may not support the same thinking levels.
_MEDIUM_THINKING = {'thinking_config': {'thinking_level': 'MEDIUM'}}
DEFAULT_QA_GENERATE_BACKEND_KWARGS = {
    'chat_kwargs': _MEDIUM_THINKING,
    'generate_object_kwargs': _MEDIUM_THINKING,
    'generate_with_tools_kwargs': _MEDIUM_THINKING,
}
DEFAULT_AI_RATER_GENERATE_MODEL_NAME = 'gemini-3.6-flash'

# Backend request settings applied by `rag.build_index` (and its command-line
# tool) unless set in its `extra_backend_kwargs`: the maximum number of retries
# (`extra_backend_kwargs['max_retries']`) and the HTTP timeout in milliseconds
# (`extra_backend_kwargs['http_options']['timeout']`).
DEFAULT_MAX_RETRIES = 5
DEFAULT_HTTP_TIMEOUT_MS = 60_000


def default_embed_backend_config() -> backend_registry.EmbedBackendConfig:
  """Returns a new config for the default embedding model."""
  return backend_registry.EmbedBackendConfig(
      backend_class_name=DEFAULT_EMBED_BACKEND_CLASS_NAME,
      model_name=DEFAULT_EMBED_MODEL_NAME,
  )


def default_tokenize_backend_config() -> (
    backend_registry.TokenizeBackendConfig
):
  """Returns a new config for the default tokenizer."""
  return backend_registry.TokenizeBackendConfig(
      backend_class_name=DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
      model_name=DEFAULT_TOKENIZE_MODEL_NAME,
  )


def default_index_generate_backend_config() -> (
    backend_registry.GenerateBackendConfig
):
  """Returns a new config for the default index-building generation model."""
  return backend_registry.GenerateBackendConfig(
      backend_class_name=DEFAULT_INDEX_GENERATE_BACKEND_CLASS_NAME,
      model_name=DEFAULT_INDEX_GENERATE_MODEL_NAME,
      kwargs=DEFAULT_INDEX_GENERATE_BACKEND_KWARGS,
  )


def qa_generate_backend_kwargs(model_name: str) -> dict[str, Any]:
  """Returns the default constructor kwargs of the QA backend of `model_name`.

  These are `DEFAULT_QA_GENERATE_BACKEND_KWARGS` (medium thinking) for the
  default QA model, and no kwargs for any other model.

  Args:
    model_name: Name of the QA generation model.
  """
  if model_name == DEFAULT_QA_GENERATE_MODEL_NAME:
    return dict(DEFAULT_QA_GENERATE_BACKEND_KWARGS)
  return {}
