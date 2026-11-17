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
from onetwo.stdlib.retrieval import audio_chunking
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import pdf_chunking
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import routing_chunking
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval import video_chunking
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing

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
    ['embedding', 'graphrag'],
    'Type of index to build ("embedding" or "graphrag").',
)

_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size',
    None,
    (
        'Maximum tokens, words, PDF pages or seconds of audio or video per'
        ' chunk (depending on --chunker_type). Required with --chunker_type'
        ' tokens, words, pdf_pages, audio_seconds and video_seconds; ignored'
        ' with routing and none.'
    ),
)

_CHUNKER_TYPE = flags.DEFINE_enum(
    'chunker_type',
    'routing',
    [
        'routing',
        'none',
        'tokens',
        'words',
        'pdf_pages',
        'audio_seconds',
        'video_seconds',
    ],
    'Type of chunker to use: "routing" (default, as in the ready-made'
    ' indexes) chunks each document according to its content type (see'
    ' `RoutingChunker` in onetwo/stdlib/retrieval/routing_chunking.py);'
    ' "none" indexes whole documents (`chunking.NoChunking`); any other value'
    ' chunks all documents into chunks of --chunk_size with a single chunker.',
)

_OVERLAP_WINDOW = flags.DEFINE_integer(
    'overlap_window',
    0,
    (
        'Number of tokens or words to overlap between chunks. Ignored unless'
        ' --chunker_type is tokens or words.'
    ),
)

_TOKENIZE_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'tokenize_backend_class_name',
    rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
    'Name of the tokenizer backend class (when --chunker_type=tokens).',
)

_TOKENIZE_MODEL_NAME = flags.DEFINE_string(
    'tokenize_model_name',
    rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME,
    'Name of the tokenizer model (when --chunker_type=tokens).',
)

_EXTRA_BACKEND_KWARGS = flags.DEFINE_string(
    'extra_backend_kwargs',
    '{}',
    (
        'JSON string of extra keyword arguments for the constructors of the'
        ' backends (e.g. an API key, threadpool_size). Not saved with the'
        ' index. Usually not needed: credentials are read from the environment'
        ' (e.g. GOOGLE_API_KEY).'
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
        ' --extra_backend_kwargs sets "max_retries".'
    ),
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    rag_defaults.DEFAULT_HTTP_TIMEOUT_MS,
    (
        'HTTP timeout in milliseconds for backend requests (default: 60s).'
        ' Ignored if --extra_backend_kwargs sets "http_options" with a'
        ' "timeout" (or an "http_options" that is not a dict).'
    ),
)

_STORAGE_FORMAT = flags.DEFINE_enum_class(
    'storage_format',
    serialization.StorageFormat.JSONL_NPY,
    serialization.StorageFormat,
    (
        'How the index is saved on disk (see `serialization.StorageFormat`).'
        ' With --index_type=graphrag, only its embedding indexes are saved in'
        ' this format.'
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


def _chunker_from_flags() -> chunking.Chunker:
  """Returns the chunker set by the chunking flags.

  Flags that do not apply to the chunker type are ignored with a warning.

  Raises:
    app.UsageError: If --chunk_size is missing for a chunker type that needs it.
  """
  chunker_type = _CHUNKER_TYPE.value
  chunk_size = _CHUNK_SIZE.value
  overlap_window = _OVERLAP_WINDOW.value
  if chunker_type in ('routing', 'none'):
    # TODO: Add flags to configure the chunk size
    # of each content type (e.g. tokens for text, pages for PDFs, seconds for
    # audio and video) with 'routing', instead of only using the defaults of
    # `RoutingChunker`.
    if chunk_size is not None or overlap_window:
      logging.warning(
          'Ignoring --chunk_size and --overlap_window, which are not used with'
          ' --chunker_type=%s.',
          chunker_type,
      )
    if chunker_type == 'none':
      return chunking.NoChunking()
    return routing_chunking.RoutingChunker()
  if chunk_size is None:
    raise app.UsageError(
        f'--chunker_type={chunker_type} requires --chunk_size. Use'
        ' --chunker_type=none to index whole documents.'
    )
  if chunker_type == 'words':
    return chunking.ChunkByMaxWords(
        max_words_per_chunk=chunk_size, overlap_window=overlap_window
    )
  if chunker_type == 'tokens':
    return chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=chunk_size,
        overlap_window=overlap_window,
        tokenize_backend_config=backend_registry.TokenizeBackendConfig(
            backend_class_name=_TOKENIZE_BACKEND_CLASS_NAME.value,
            model_name=_TOKENIZE_MODEL_NAME.value,
        ),
    )
  # 'pdf_pages', 'audio_seconds' or 'video_seconds', whose chunks cannot
  # overlap.
  if overlap_window:
    logging.warning(
        'Ignoring --overlap_window, which is not supported with'
        ' --chunker_type=%s.',
        chunker_type,
    )
  if chunker_type == 'pdf_pages':
    return pdf_chunking.ChunkByPdfPages(pages_per_chunk=chunk_size)
  if chunker_type == 'audio_seconds':
    return audio_chunking.ChunkByAudioSeconds(
        seconds_per_chunk=float(chunk_size)
    )
  return video_chunking.ChunkByVideoSeconds(seconds_per_chunk=float(chunk_size))


def _index_from_flags() -> indexing.DocumentIndex:
  """Returns the (empty) index set by the index, embedding and chunking flags.

  Without other flags than --index_type, this is the ready-made index of
  `build_index_lib.IndexType`.
  """
  embedder = _embedder_from_flags()
  chunker = _chunker_from_flags()
  if _INDEX_TYPE.value == 'graphrag':
    index = graphrag_indexing.HippoRAGDocumentIndex(
        embed_backend_config=embedder,
        generate_backend_config=(
            rag_defaults.default_index_generate_backend_config()
        ),
    )
    return indexing.RewritingDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
        inner_index=index,
    )
  index = indexing.EmbeddingBasedDocumentIndex(embed_backend_config=embedder)
  return indexing.RewritingEmbeddingBasedDocumentIndex(
      rewriter=corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
      inner_index=index,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  build_index_lib.build_index(
      _corpus_from_flag(_CORPUS.value, _MAX_DOCUMENTS.value),
      _index_from_flags(),
      output=_OUTPUT.value,
      storage_format=_STORAGE_FORMAT.value,
      extra_backend_kwargs=build_index_lib.get_default_extra_backend_kwargs(
          json.loads(_EXTRA_BACKEND_KWARGS.value),
          max_retries=_MAX_RETRIES.value,
          http_timeout_ms=_HTTP_TIMEOUT.value,
      ),
  )


if __name__ == '__main__':
  app.run(main)
