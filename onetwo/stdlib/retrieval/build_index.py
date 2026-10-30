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

# Copyright 2026 Google LLC
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

r"""Binary to build an embedding-based document index.
"""

import json
from typing import Sequence

from absl import app
from absl import flags
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import routing_chunking  # pylint: disable=unused-import  # Registers 'routing'.

# Re-export build_index for backward compatibility.
build_index = build_index_lib.build_index

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    'browsecomp_plus',
    'Name of the dataset to load from the registry.',
    allow_override=True,
)

_INPUT = flags.DEFINE_string(
    'input',
    None,
    (
        'Path to a folder of files of any type (text, PDF, image, audio,'
        ' video) to index. If set, documents are loaded from this folder'
        ' instead of the dataset registry, and --dataset_name must be set to'
        ' a name for the corpus, which is used in the output directory path.'
    ),
)

_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    None,
    'Directory where the built index will be saved.',
    required=True,
)

_EMBEDDING_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'embedding_backend_class_name',
    rag_defaults.DEFAULT_EMBED_BACKEND_CLASS_NAME,
    'Name of the embedding backend class.',
)

_EMBEDDING_MODEL_NAME = flags.DEFINE_string(
    'embedding_model_name',
    rag_defaults.DEFAULT_EMBED_MODEL_NAME,
    'Name of the embedding model.',
)

_INDEX_TYPE = flags.DEFINE_enum(
    'index_type',
    'embedding',
    ['embedding', 'hipporag'],
    'Type of index to build ("embedding" or "hipporag").',
)

_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size',
    None,
    (
        'Maximum tokens, words, PDF pages or seconds of audio or video per'
        ' chunk (depending on --chunking_strategy). If None, no chunking is'
        ' performed (unless --chunking_strategy=routing). Not supported with'
        ' --chunking_strategy=routing.'
    ),
)

_CHUNKING_STRATEGY = flags.DEFINE_enum(
    'chunking_strategy',
    'routing',
    [
        'routing',
        'tokens',
        'words',
        'pdf_pages',
        'audio_seconds',
        'video_seconds',
    ],
    'Chunking strategy to use: "routing" (default) chunks each document'
    ' according to its content type (see `RoutingChunker` in'
    ' onetwo/stdlib/retrieval/routing_chunking.py); any other value is the name'
    ' of a single chunker registered in `chunker_registry` that is applied to'
    ' all documents.',
)

_OVERLAP_WINDOW = flags.DEFINE_integer(
    'overlap_window',
    0,
    'Number of tokens or words to overlap between chunks.',
)

_TOKENIZE_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'tokenize_backend_class_name',
    rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
    'Name of the tokenizer backend class (when --chunking_strategy=tokens).',
)

_TOKENIZE_MODEL_NAME = flags.DEFINE_string(
    'tokenize_model_name',
    rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME,
    'Name of the tokenizer model (when --chunking_strategy=tokens).',
)

_FORMAT = flags.DEFINE_string(
    'format',
    None,
    (
        'Format string for document formatting. If "None" (string) or empty,'
        ' no formatting is done.'
    ),
)

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    '{}',
    (
        'JSON string containing runtime arguments for the backend (e.g. API'
        ' key, project).'
    ),
    allow_override=True,
)

_MAX_DOCUMENTS = flags.DEFINE_integer(
    'max_documents',
    None,
    'Maximum number of documents to load and index (useful for testing).',
)

_MAX_RETRIES = flags.DEFINE_integer(
    'max_retries',
    5,
    'Maximum number of retries for backend requests.',
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    60000,
    'HTTP timeout in milliseconds for backend requests (default: 60s).',
)

_SERIALIZER_TYPE = flags.DEFINE_enum(
    'serializer_type',
    'simple',
    ['simple', 'internal'],
    'Serializer type to use for saving the index.',
)

_EMBED_KWARGS = flags.DEFINE_string(
    'embed_kwargs',
    '{}',
    (
        'JSON string containing keyword arguments for the embedding backend.'
    ),
)


def _embedder_from_flags() -> backend_registry.EmbedBackendConfig:
  """Returns the embedding backend config set by the embedding flags."""
  embed_kwargs = json.loads(_EMBED_KWARGS.value)
  backend_kwargs = {}
  if embed_kwargs:
    backend_kwargs['embed_kwargs'] = dict(embed_kwargs)
  return backend_registry.EmbedBackendConfig(
      backend_class_name=_EMBEDDING_BACKEND_CLASS_NAME.value,
      model_name=_EMBEDDING_MODEL_NAME.value,
      kwargs=backend_kwargs,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  build_index_lib.build_index(
      dataset_name=_DATASET_NAME.value,
      input_dir=_INPUT.value,
      max_documents=_MAX_DOCUMENTS.value,
      index_type=_INDEX_TYPE.value,
      output_directory=_OUTPUT_DIRECTORY.value,
      embedder=_embedder_from_flags(),
      chunk_size=_CHUNK_SIZE.value,
      chunking_strategy=_CHUNKING_STRATEGY.value,
      overlap_window=_OVERLAP_WINDOW.value,
      tokenize_backend_class_name=_TOKENIZE_BACKEND_CLASS_NAME.value,
      tokenize_model_name=_TOKENIZE_MODEL_NAME.value,
      format_str=_FORMAT.value,
      serializer_type=_SERIALIZER_TYPE.value,
      runtime_kwargs=_RUNTIME_KWARGS.value,
      max_retries=_MAX_RETRIES.value,
      http_timeout=_HTTP_TIMEOUT.value,
  )


if __name__ == '__main__':
  app.run(main)
