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

from onetwo.backends import sentencepiece_tokenizer
from onetwo.core import executing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures

from onetwo.stdlib.retrieval import serialization


class LocalIndexRunner(
    index_build_interface.IndexRunner[
        index_build_interface.IndexSpec,
        Iterable[retrieval_data_structures.Document] | str,
    ]
):
  """In-memory index runner."""

  def _get_tokenizer_backend(
      self,
  ) -> sentencepiece_tokenizer.SentencePieceTokenizer:
    tokenizer = sentencepiece_tokenizer.SentencePieceTokenizer(
        model_name="GEMINI3_G145"
    )
    tokenizer.register()
    return tokenizer

  def _get_chunker(
      self,
      spec: index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
  ) -> chunking.Chunker:
    if spec.max_chunk_size is None:
      return chunking.NoChunking()
    if getattr(spec, "chunking_strategy", "words") == "words":
      return chunking.ChunkByMaxWords(
          max_words_per_chunk=spec.max_chunk_size,
          overlap_window=spec.overlap_window,
      )
    return chunking.ChunkByMaxTokens(
        max_tokens_per_chunk=spec.max_chunk_size,
        overlap_window=spec.overlap_window,
        tokenizer_backend=self._get_tokenizer_backend(),
    )

  @executing.make_executable(copy_self=False)
  async def build_index(
      self,
      spec: index_build_interface.IndexSpec,
      docs: Iterable[retrieval_data_structures.Document] | str,
      output_dir: str | None = None,
      **kwargs,
  ) -> indexing.Index:
    """Builds a DocumentIndex in memory."""
    if isinstance(docs, str):
      docs = serialization.load_documents(
          docs, doc_class=retrieval_data_structures.Document
      )

    runtime_kwargs = {}
    if isinstance(spec, index_build_interface.EmbeddingBasedDocumentIndexSpec):
      runtime_kwargs = spec.runtime_kwargs

    if isinstance(
        spec, index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec
    ):
      chunker = self._get_chunker(spec)
      if spec.format_str is None:
        formatter = document_formatting.NoFormatting()
      else:
        formatter = document_formatting.TextDocumentFormatter(
            format_str=spec.format_str
        )
      rewriter = corpus_rewriting.SequentialCorpusRewriter(
          rewriters=[
              corpus_rewriting.ChunkingCorpusRewriter(chunker=chunker),
              corpus_rewriting.FormattingCorpusRewriter(formatter=formatter),
          ]
      )

      extractor = graph_lib.GraphRAGEntityAndTriplesExtractor(
          extraction_prompt=graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT.format(
              entity_types=", ".join(graphrag_prompts.DEFAULT_ENTITY_TYPES)
          ),
          generate_backend_config=spec.generate_backend_config,
          runtime_kwargs=runtime_kwargs,
      )
      inner_index = graphrag_indexing.HippoRAGDocumentIndex(
          entity_and_triples_extractor=extractor,
          embed_backend_config=spec.embed_backend_config,
          generate_backend_config=spec.generate_backend_config,
          similarity_threshold=spec.similarity_threshold,
          runtime_kwargs=runtime_kwargs,
      )
      index = indexing.RewritingEmbeddingBasedDocumentIndex(
          rewriter=rewriter,
          inner_index=inner_index,  # pyrefly: ignore[bad-argument-type]
      )
    elif isinstance(
        spec,
        index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
    ):
      embed_backend_config = spec.embed_backend_config
      chunker = self._get_chunker(spec)

      if spec.format_str is None:
        formatter = document_formatting.NoFormatting()
      else:
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
          serializer_type=spec.serializer_type,
          runtime_kwargs=runtime_kwargs,
      )
      index = indexing.RewritingEmbeddingBasedDocumentIndex(
          rewriter=rewriter,
          inner_index=inner_index,
          serializer_type=spec.serializer_type,
      )
    elif isinstance(
        spec, index_build_interface.EmbeddingBasedDocumentIndexSpec
    ):
      index = indexing.EmbeddingBasedDocumentIndex(
          embed_backend_config=spec.embed_backend_config,
          query_task_type=spec.query_task_type,
          document_task_type=spec.document_task_type,
          searcher=index_build_interface.make_searcher(spec.searcher_config),
          serializer_type=spec.serializer_type,
          runtime_kwargs=runtime_kwargs,
      )
    else:
      raise ValueError(f"Unsupported spec type: {type(spec)}")

    await index.add_docs(docs)

    if output_dir:
      index.save(output_dir)

    return index
