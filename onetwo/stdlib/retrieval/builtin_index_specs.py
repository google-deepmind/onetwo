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

"""Index factories for the built-in `IndexSpec`s.

Importing this module registers the built-in specs in `index_spec_registry`.
The factories are public so that custom index factories can reuse them (e.g.
`make_embedding_index` to build an inner embedding index).
"""

from collections.abc import Mapping
from typing import Any

from onetwo.backends import sentencepiece_tokenizer
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing

EmbeddingBasedDocumentIndexSpec = (
    index_build_interface.EmbeddingBasedDocumentIndexSpec
)
SimpleRewritingEmbeddingBasedDocumentIndexSpec = (
    index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec
)
SimpleRewritingHippoRAGDocumentIndexSpec = (
    index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec
)


def _merge_runtime_kwargs(
    spec: EmbeddingBasedDocumentIndexSpec,
    runtime_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
  """Returns `spec.runtime_kwargs` updated with `runtime_kwargs`."""
  # Built-in specs may carry their own `runtime_kwargs` (e.g. set by
  # `build_index_lib.build_index`); explicitly passed ones take precedence.
  return {**spec.runtime_kwargs, **runtime_kwargs}


def _get_tokenizer_backend() -> sentencepiece_tokenizer.SentencePieceTokenizer:
  tokenizer = sentencepiece_tokenizer.SentencePieceTokenizer(
      model_name='GEMINI3_G145'
  )
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
      tokenizer_backend=_get_tokenizer_backend(),
  )


def _get_rewriter(
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
      runtime_kwargs=_merge_runtime_kwargs(spec, runtime_kwargs),
  )


def make_simple_rewriting_embedding_index(
    spec: SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    runtime_kwargs: Mapping[str, Any],
) -> indexing.RewritingEmbeddingBasedDocumentIndex:
  """Returns an empty chunking/formatting embedding index for `spec`."""
  return indexing.RewritingEmbeddingBasedDocumentIndex(
      rewriter=_get_rewriter(spec),
      inner_index=make_embedding_index(spec, runtime_kwargs),
      serializer_type=spec.serializer_type,
  )


index_spec_registry.register_index_spec(
    EmbeddingBasedDocumentIndexSpec, make_embedding_index
)
index_spec_registry.register_index_spec(
    SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    make_simple_rewriting_embedding_index,
)
