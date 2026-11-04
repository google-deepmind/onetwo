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

"""Index specs for HippoRAG indexes.

Importing this module registers `SimpleRewritingHippoRAGDocumentIndexSpec` in
`index_spec_registry` under the name 'simple_rewriting_hipporag'.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import dataclasses_json
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class HippoRAGDocumentIndexSpec(
    embedding_index_specs.EmbeddingBasedDocumentIndexSpec
):
  """Base specification holding the HippoRAGDocumentIndex fields.

  This spec is not directly buildable: it is not registered in
  `index_spec_registry`, so building an index from it raises a `KeyError`. Use
  `SimpleRewritingHippoRAGDocumentIndexSpec` to build a HippoRAG index.

  Attributes:
    generate_backend_config: Configuration for the generation backend (LLM).
      Defaults to `rag_defaults.default_index_generate_backend_config()`.
    similarity_threshold: Minimum cosine similarity required to add a semantic
      similarity edge between entities in the graph.
    ignore_triples: Whether to ignore indexing relational triples into the
      triples index. Defaults to True.
    ignore_similarity_edges: Whether to ignore semantic similarity edges between
      entities in the graph. Defaults to True.
  """

  similarity_threshold: float = 0.8
  generate_backend_config: backend_registry.GenerateBackendConfig = (
      dataclasses.field(
          default_factory=rag_defaults.default_index_generate_backend_config
      )
  )
  ignore_triples: bool = True
  ignore_similarity_edges: bool = True

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    embed_model = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    gen_model = (
        self.generate_backend_config.model_name
        or self.generate_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return (
        f"HippoRAGDocumentIndex_embed_{embed_model}_gen_{gen_model}"
        f"_sim_{self.similarity_threshold}"
        f"__hash_{spec_hash}"
    )


@index_spec_registry.register_index_spec("simple_rewriting_hipporag")
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SimpleRewritingHippoRAGDocumentIndexSpec(
    HippoRAGDocumentIndexSpec,
    embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec,
):
  """Specification for building a simple Rewriting HippoRAGDocumentIndex.

  This spec is a convenience wrapper combining HippoRAG index building with
  chunking and formatting for corpus rewriting.
  """

  def make_index(
      self, runtime_kwargs: Mapping[str, Any]
  ) -> indexing.RewritingDocumentIndex:
    """Returns an empty chunking/formatting HippoRAG index for this spec."""
    runtime_kwargs = dict(runtime_kwargs)
    # The index builds its entity and triples extractor from these settings,
    # with the default extraction prompt and entity types.
    inner_index = graphrag_indexing.HippoRAGDocumentIndex(
        embed_backend_config=self.embed_backend_config,
        generate_backend_config=self.generate_backend_config,
        similarity_threshold=self.similarity_threshold,
        ignore_triples=self.ignore_triples,
        ignore_similarity_edges=self.ignore_similarity_edges,
        runtime_kwargs=runtime_kwargs,
    )
    return indexing.RewritingDocumentIndex(
        rewriter=embedding_index_specs.make_rewriter(self),
        inner_index=inner_index,
        preserve_original_docs=self.preserve_original_docs,
        retrieval_target=self.retrieval_target,
    )

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    embed_model = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    gen_model = (
        self.generate_backend_config.model_name
        or self.generate_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return (
        "HippoRAGDocumentIndex"
        f"_chunker_{self.chunker_config}"
        f"_embed_{embed_model}_gen_{gen_model}"
        f"_sim_{self.similarity_threshold}"
        f"__hash_{spec_hash}"
    )
