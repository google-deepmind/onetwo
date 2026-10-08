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

"""Embedding-based `IndexSpec`s and their index factories.

Importing this module registers the embedding specs in `index_spec_registry`.
The factories are public so that custom index factories can reuse them (e.g.
`make_embedding_index` to build an inner embedding index).
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import dataclasses_json
from onetwo.backends import backend_registry
from onetwo.backends import backends_base
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class EmbeddingBasedDocumentIndexSpec(index_build_interface.IndexSpec):
  """Specification for building an EmbeddingBasedDocumentIndex.

  Attributes:
    embed_backend_config: Configuration for the embedder backend. Defaults to
      `rag_defaults.default_embed_backend_config()`.
    query_task_type: Task type to use when embedding queries.
    document_task_type: Task type to use when embedding documents.
    searcher_config: Configuration for the searcher to be used by the index for
      retrieving documents (e.g., BruteForceSearcher).
  """

  embed_backend_config: backend_registry.EmbedBackendConfig = (
      dataclasses.field(
          default_factory=rag_defaults.default_embed_backend_config
      )
  )
  query_task_type: str | None = 'RETRIEVAL_QUERY'
  document_task_type: str | None = 'RETRIEVAL_DOCUMENT'
  searcher_config: index_build_interface.SearcherConfig = dataclasses.field(
      default_factory=lambda: index_build_interface.SearcherConfig(
          class_name='BruteForceSearcher'
      )
  )
  serializer_type: str = dataclasses.field(
      default='simple',
      metadata=dataclasses_json.config(exclude=lambda _: True),
  )

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    model_name = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return f'EmbeddingBasedDocumentIndex_{model_name}__hash_{spec_hash}'


# TODO: Standardize the spec-to-rewriter mapping and make it
# generic.
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SimpleRewritingEmbeddingBasedDocumentIndexSpec(
    EmbeddingBasedDocumentIndexSpec
):
  """Specification for building a simple RewritingEmbeddingBasedDocumentIndex.

  This spec is a convenience wrapper for users that want to use a combination of
  chunking and formatting for corpus rewriting.

  Attributes:
    max_chunk_size: Maximum tokens or words per chunk for chunking.
    chunking_strategy: Chunking strategy to use ('words' or 'tokens').
    format_str: Format string for document formatting.
    overlap_window: Number of tokens or words to overlap between chunks.
    tokenize_backend_config: Configuration for the tokenizer used when
      `chunking_strategy` is 'tokens'. Defaults to
      `rag_defaults.default_tokenize_backend_config()`.
  """

  max_chunk_size: int | None = None
  chunking_strategy: str = 'words'
  format_str: str | None = '{text}'
  overlap_window: int = 0
  tokenize_backend_config: backend_registry.TokenizeBackendConfig = (
      dataclasses.field(
          default_factory=rag_defaults.default_tokenize_backend_config
      )
  )

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    model_name = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return (
        'EmbeddingBasedDocumentIndex'
        f'_chunk_strategy_{self.chunking_strategy}'
        f'_chunk_size_{self.max_chunk_size}'
        f'_overlap_window_{self.overlap_window}'
        f'_{model_name}'
        f'__hash_{spec_hash}'
    )


def _get_tokenizer_backend(
    config: backend_registry.TokenizeBackendConfig,
) -> backends_base.Backend:
  tokenizer = backend_registry.get_backend(config)
  # ChunkByMaxTokens tokenizes with the registered `llm.tokenize`.
  tokenizer.register()
  return tokenizer


def _get_chunker(
    spec: SimpleRewritingEmbeddingBasedDocumentIndexSpec,
) -> chunking.Chunker:
  if spec.max_chunk_size is None:
    return chunking.NoChunking()
  if getattr(spec, 'chunking_strategy', 'words') == 'words':
    return chunking.ChunkByMaxWords(
        max_words_per_chunk=spec.max_chunk_size,
        overlap_window=spec.overlap_window,
    )
  return chunking.ChunkByMaxTokens(
      max_tokens_per_chunk=spec.max_chunk_size,
      overlap_window=spec.overlap_window,
      tokenizer_backend=_get_tokenizer_backend(spec.tokenize_backend_config),
  )


def make_rewriter(
    spec: SimpleRewritingEmbeddingBasedDocumentIndexSpec,
) -> corpus_rewriting.CorpusRewriter:
  """Returns the chunk-then-format rewriter described by `spec`."""
  if spec.format_str is None:
    formatter = document_formatting.NoFormatting()
  else:
    formatter = document_formatting.TextDocumentFormatter(
        format_str=spec.format_str
    )
  return corpus_rewriting.SequentialCorpusRewriter(
      rewriters=[
          corpus_rewriting.ChunkingCorpusRewriter(chunker=_get_chunker(spec)),
          corpus_rewriting.FormattingCorpusRewriter(formatter=formatter),
      ]
  )


def make_embedding_index(
    spec: EmbeddingBasedDocumentIndexSpec,
    runtime_kwargs: Mapping[str, Any],
) -> indexing.EmbeddingBasedDocumentIndex:
  """Returns an empty `EmbeddingBasedDocumentIndex` configured by `spec`."""
  return indexing.EmbeddingBasedDocumentIndex(
      embed_backend_config=spec.embed_backend_config,
      query_task_type=spec.query_task_type,
      document_task_type=spec.document_task_type,
      searcher=index_build_interface.make_searcher(spec.searcher_config),
      serializer_type=spec.serializer_type,
      runtime_kwargs=dict(runtime_kwargs),
  )


def make_simple_rewriting_embedding_index(
    spec: SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    runtime_kwargs: Mapping[str, Any],
) -> indexing.RewritingEmbeddingBasedDocumentIndex:
  """Returns an empty chunking/formatting embedding index for `spec`."""
  return indexing.RewritingEmbeddingBasedDocumentIndex(
      rewriter=make_rewriter(spec),
      inner_index=make_embedding_index(spec, runtime_kwargs),
      serializer_type=spec.serializer_type,
  )


index_spec_registry.register_index_spec(
    EmbeddingBasedDocumentIndexSpec, make_embedding_index, name='embedding'
)
index_spec_registry.register_index_spec(
    SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    make_simple_rewriting_embedding_index,
    name='simple_rewriting_embedding',
)
