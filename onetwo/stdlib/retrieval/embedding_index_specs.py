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

"""Embedding-based `IndexSpec`s.

Importing this module registers the embedding specs in `index_spec_registry`.
Custom specs can reuse them, e.g. by subclassing a spec and calling
`super().make_index(...)` to build an inner embedding index, or by calling
`make_rewriter` to build the chunk-then-format rewriter.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import dataclasses_json
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults


@index_spec_registry.register_index_spec('embedding')
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

  def make_index(
      self, runtime_kwargs: Mapping[str, Any]
  ) -> indexing.EmbeddingBasedDocumentIndex:
    """Returns an empty `EmbeddingBasedDocumentIndex` configured by this spec."""
    return indexing.EmbeddingBasedDocumentIndex(
        embed_backend_config=self.embed_backend_config,
        query_task_type=self.query_task_type,
        document_task_type=self.document_task_type,
        searcher=index_build_interface.make_searcher(self.searcher_config),
        serializer_type=self.serializer_type,
        runtime_kwargs=dict(runtime_kwargs),
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
@index_spec_registry.register_index_spec('simple_rewriting_embedding')
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SimpleRewritingEmbeddingBasedDocumentIndexSpec(
    EmbeddingBasedDocumentIndexSpec
):
  """Specification for building a simple RewritingEmbeddingBasedDocumentIndex.

  This spec is a convenience wrapper for users that want to use a combination of
  chunking and formatting for corpus rewriting.

  Attributes:
    chunker_config: Configuration for the chunker used for corpus rewriting.
    format_str: Format string for document formatting.
  """

  chunker_config: chunker_registry.ChunkerConfig = dataclasses.field(
      default_factory=chunker_registry.ChunkerConfig
  )
  format_str: str | None = '{text}'

  def make_index(
      self, runtime_kwargs: Mapping[str, Any]
  ) -> indexing.RewritingEmbeddingBasedDocumentIndex:
    """Returns an empty chunking/formatting embedding index for this spec."""
    return indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=make_rewriter(self),
        inner_index=super().make_index(runtime_kwargs),
        serializer_type=self.serializer_type,
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
        f'_chunker_{self.chunker_config}'
        f'_{model_name}'
        f'__hash_{spec_hash}'
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
          corpus_rewriting.ChunkingCorpusRewriter(
              chunker=chunker_registry.get_chunker(spec.chunker_config)
          ),
          corpus_rewriting.FormattingCorpusRewriter(formatter=formatter),
      ]
  )
