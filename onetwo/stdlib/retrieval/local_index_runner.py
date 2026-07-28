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

"""In-memory index runner implementation."""

from typing import Iterable

from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures


class LocalIndexRunner(
    index_build_interface.IndexRunner[
        index_build_interface.IndexSpec,
        Iterable[retrieval_data_structures.Document] | str,
    ]
):
  """In-memory index runner."""

  @executing.make_executable(copy_self=False)
  async def build_index(
      self,
      spec: index_build_interface.IndexSpec,
      docs: Iterable[retrieval_data_structures.Document] | str,
      output_dir: str | None = None,
      **kwargs,
  ) -> indexing.DocumentIndex:
    """Builds a DocumentIndex in memory."""
    if isinstance(docs, str):
      raise ValueError(
          "LocalIndexRunner does not support loading documents from a path."
      )

    runtime_kwargs = {}
    if isinstance(spec, index_build_interface.EmbeddingBasedDocumentIndexSpec):
      runtime_kwargs = spec.runtime_kwargs

    if isinstance(
        spec,
        index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    ):
      embed_backend_config = spec.embed_backend_config
      if spec.max_tokens_per_chunk is None:
        chunker = chunking.NoChunking()
      else:
        chunker = chunking.ChunkByMaxTokens(
            max_tokens_per_chunk=spec.max_tokens_per_chunk,
            overlap_window=spec.overlap_window,
        )
      formatter = document_formatting.TextDocumentFormatter(
          format_str=spec.format_str
      )
      rewriter = corpus_rewriting.SequentialCorpusRewriter(
          rewriters=[
              corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
              corpus_rewriting.FormattingCorpusRewriter(formatter=formatter),
          ]
      )
      inner_index = indexing.EmbeddingBasedDocumentIndex(
          embed_backend_config=embed_backend_config,
          query_task_type=spec.query_task_type,
          document_task_type=spec.document_task_type,
          searcher=index_build_interface.make_searcher(spec.searcher_config),
          runtime_kwargs=runtime_kwargs,
      )
      index = indexing.RewritingEmbeddingBasedDocumentIndex(
          rewriter=rewriter,
          inner_index=inner_index,
      )
    elif isinstance(
        spec, index_build_interface.EmbeddingBasedDocumentIndexSpec
    ):
      index = indexing.EmbeddingBasedDocumentIndex(
          embed_backend_config=spec.embed_backend_config,
          query_task_type=spec.query_task_type,
          document_task_type=spec.document_task_type,
          searcher=index_build_interface.make_searcher(spec.searcher_config),
          runtime_kwargs=runtime_kwargs,
      )
    else:
      raise ValueError(f"Unsupported spec type: {type(spec)}")

    await index.add_docs(docs)

    if output_dir:
      index.save(output_dir)

    return index
