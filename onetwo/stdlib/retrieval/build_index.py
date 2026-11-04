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
from absl import logging
from onetwo.backends import backend_registry
from onetwo.evaluation import datasets
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import routing_chunking  # pylint: disable=unused-import  # Registers 'routing'.
from onetwo.stdlib.retrieval import serialization

# Re-export build_index for backward compatibility.
build_index = build_index_lib.build_index

_CORPUS = flags.DEFINE_string(
    'corpus',
    'browsecomp_plus',
    (
        'What to index: the name of a registered dataset (see'
        ' `datasets.list_dataset_loaders()`), or else a path to a folder of'
        ' files of any type (text, PDF, image, audio, video). A value that is'
        ' a registered dataset name is always loaded from the dataset'
        ' registry; any other value is passed to `build_index` as a path. The'
        ' dataset name or the folder basename is used in the output directory'
        ' path.'
    ),
    allow_override=True,
)

_OUTPUT = flags.DEFINE_string(
    'output',
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
        ' performed (unless --chunking_strategy=routing). Ignored with'
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
    (
        'Number of tokens or words to overlap between chunks. Ignored unless'
        ' --chunking_strategy is tokens or words.'
    ),
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
    (
        'Maximum number of documents to load and index (useful for testing).'
        ' Only applies when --corpus is a registered dataset, whose loader'
        ' loads at most this many documents; it is an error with a folder'
        ' --corpus.'
    ),
)

_MAX_RETRIES = flags.DEFINE_integer(
    'max_retries',
    rag_defaults.DEFAULT_MAX_RETRIES,
    (
        'Maximum number of retries for backend requests. Ignored if'
        ' --runtime_kwargs sets "max_retries".'
    ),
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    rag_defaults.DEFAULT_HTTP_TIMEOUT_MS,
    (
        'HTTP timeout in milliseconds for backend requests (default: 60s).'
        ' Ignored if --runtime_kwargs sets "http_options" with a "timeout" (or'
        ' an "http_options" that is not a dict).'
    ),
)

_STORAGE_FORMAT = flags.DEFINE_enum_class(
    'storage_format',
    serialization.StorageFormat.JSONL_NPY,
    serialization.StorageFormat,
    (
        'How the index is saved on disk (see `serialization.StorageFormat`).'
        ' Only used with --index_type=embedding: --index_type=hipporag'
        ' indexes are always saved in their own graph format.'
    ),
)

_EMBED_KWARGS = flags.DEFINE_string(
    'embed_kwargs',
    '{}',
    (
        'JSON string containing keyword arguments for the embedding backend.'
    ),
)


def _corpus_from_flag(
    value: str, max_documents: int | None
) -> str | datasets.DatasetLoader:
  """Returns the corpus to index for the value of --corpus.

  Args:
    value: The name of a registered dataset, or a path to a folder of files.
    max_documents: Maximum number of documents to load (--max_documents). Only
      supported for registered datasets.

  Returns:
    The loader of the dataset registered under `value` (loading at most
    `max_documents` documents) if any, else `value` (a path to a folder of
    files).

  Raises:
    app.UsageError: If `max_documents` is set and `value` is not a registered
      dataset.
  """
  registered = datasets.list_dataset_loaders()
  if value in registered:
    return datasets.get_dataset_loader(
        value, max_number_of_documents=max_documents
    )
  if max_documents is not None:
    raise app.UsageError(
        '--max_documents only applies to registered datasets (one of'
        f' {registered}); to limit a folder corpus, use the Python API with'
        ' datasets.MultimodalDatasetLoader(path=...,'
        ' max_number_of_documents=N).'
    )
  return value


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


def _chunker_from_flags() -> chunker_registry.ChunkerConfig:
  """Returns the chunker config set by the chunking flags.

  Flags that do not apply to the chunking strategy are ignored with a warning.
  """
  chunking_strategy = _CHUNKING_STRATEGY.value
  chunk_size = _CHUNK_SIZE.value
  overlap_window = _OVERLAP_WINDOW.value
  if chunking_strategy == 'routing':
    # TODO: Add flags to configure the chunk size
    # of each content type (e.g. tokens for text, pages for PDFs, seconds for
    # audio and video) with 'routing', instead of only using the defaults of
    # `RoutingChunker`. This also requires keeping the index output directory
    # path-friendly when the chunker config has kwargs.
    if chunk_size is not None or overlap_window:
      logging.warning(
          'Ignoring --chunk_size and --overlap_window, which are not supported'
          ' with --chunking_strategy=routing: the default chunker of each'
          ' content type is used (see `routing_chunking`).'
      )
    # The defaults of `RoutingChunker` are used, so that the index output
    # directory (which includes the chunker config) stays path-friendly.
    return chunker_registry.ChunkerConfig(chunker_name='routing')
  if chunk_size is None:
    return chunker_registry.ChunkerConfig(chunker_name='none')
  if chunking_strategy == 'words':
    return chunker_registry.ChunkerConfig(
        chunker_name='words',
        kwargs={
            'max_words_per_chunk': chunk_size,
            'overlap_window': overlap_window,
        },
    )
  if chunking_strategy == 'tokens':
    return chunker_registry.ChunkerConfig(
        chunker_name='tokens',
        kwargs={
            'max_tokens_per_chunk': chunk_size,
            'overlap_window': overlap_window,
            'tokenize_backend_config': backend_registry.TokenizeBackendConfig(
                backend_class_name=_TOKENIZE_BACKEND_CLASS_NAME.value,
                model_name=_TOKENIZE_MODEL_NAME.value,
            ),
        },
    )
  # 'pdf_pages', 'audio_seconds' or 'video_seconds', whose chunks cannot
  # overlap.
  if overlap_window:
    logging.warning(
        'Ignoring --overlap_window, which is not supported with'
        ' --chunking_strategy=%s.',
        chunking_strategy,
    )
  if chunking_strategy == 'pdf_pages':
    return chunker_registry.ChunkerConfig(
        chunker_name='pdf_pages',
        kwargs={'pages_per_chunk': chunk_size},
    )
  # 'audio_seconds' or 'video_seconds'.
  return chunker_registry.ChunkerConfig(
      chunker_name=chunking_strategy,
      kwargs={'seconds_per_chunk': float(chunk_size)},
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  build_index_lib.build_index(
      corpus=_corpus_from_flag(_CORPUS.value, _MAX_DOCUMENTS.value),
      index_type=_INDEX_TYPE.value,
      output=_OUTPUT.value,
      embedder=_embedder_from_flags(),
      chunker=_chunker_from_flags(),
      format_str=_FORMAT.value,
      storage_format=_STORAGE_FORMAT.value,
      runtime_kwargs=build_index_lib.get_default_runtime_kwargs(
          json.loads(_RUNTIME_KWARGS.value),
          max_retries=_MAX_RETRIES.value,
          http_timeout_ms=_HTTP_TIMEOUT.value,
      ),
  )


if __name__ == '__main__':
  app.run(main)
