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

"""Serialization utilities for GraphRAG indices."""

import dataclasses
import json
from typing import Any, ClassVar, Generic, Sequence
from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing as indexing_lib
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization as retrieval_serialization
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
import pydantic

DocT = retrieval.DocT
Document = retrieval_data_structures.Document
EmbedBackendConfig = backend_registry.EmbedBackendConfig
GenerateBackendConfig = backend_registry.GenerateBackendConfig


@dataclasses.dataclass
class GraphRAGIndexState(Generic[DocT]):  # pyrefly: ignore[invalid-type-var]
  """State DTO container representing a snapshot of a GraphRAGIndex.

  Attributes:
    docs: Sequence of indexed documents.
    graph: The underlying KnowledgeGraph container holding nodes and edges.
    entity_index: Optional `EmbeddingBasedDocumentIndex` storing indexed
      `Document` representations of graph entity nodes (`EntityNode`).
    passage_index: Optional `EmbeddingBasedDocumentIndex` storing indexed
      `Document` representations of source document passages (`PassageNode`).
    triples_index: Optional `EmbeddingBasedDocumentIndex` storing indexed
      `Document` representations of relational triples (`Edge`).
    generate_backend_config: The `GenerateBackendConfig` config used for graph
      construction.
    embed_backend_config: The `EmbedBackendConfig` config used for node
      embeddings.
  """

  docs: Sequence[DocT] | None = None
  graph: graph_lib.KnowledgeGraph | None = None
  entity_index: indexing_lib.EmbeddingBasedDocumentIndex | None = None
  passage_index: indexing_lib.EmbeddingBasedDocumentIndex | None = None
  triples_index: indexing_lib.EmbeddingBasedDocumentIndex | None = None
  generate_backend_config: GenerateBackendConfig | None = None
  embed_backend_config: EmbedBackendConfig | None = None


@dataclasses.dataclass
class SimpleGraphRAGIndexSerializer(Generic[DocT]):  # pyrefly: ignore[invalid-type-var]
  """Serializer for GraphRAG indices.

  This class saves and loads `GraphRAGIndexState` using `etils.epath` for
  hermetic, filesystem-agnostic operations. It saves the index state to the
  following files:
  1. `manifest.json`: Stores index metadata, counts, and backend configs.
  2. `docs.jsonl` and `graph.jsonl`: Stores the documents and
      graph (nodes and edges) as JSON lines.
  3. `entity_index/`: Directory containing serialized index for entity nodes.
  4. `passage_index/`: Directory containing serialized index for passage nodes.
  5. `triples_index/`: Directory containing serialized index for relational
  edges.

  Attributes:
    doc_class: The document class (`type[DocT]`) to use for serializing and
      deserializing documents. Must implement `to_json()` and `from_json()`.
    base_path: The base directory path for saving and loading.
    docs_jsonl_name: Filename for storing documents (`docs.jsonl`).
    manifest_json_name: Filename for storing index metadata (`manifest.json`).
    graph_jsonl_name: Filename for storing graph topology (`graph.jsonl`).
    entity_index_dir_name: Directory name for storing `entity_index`.
    passage_index_dir_name: Directory name for storing `passage_index`.
    triples_index_dir_name: Directory name for storing `triples_index`.
  """

  doc_class: type[DocT]
  base_path: str | None = None
  docs_jsonl_name: str = "docs.jsonl"
  manifest_json_name: str = "manifest.json"
  graph_jsonl_name: str = "graph.jsonl"
  entity_index_dir_name: str = "entity_index"
  passage_index_dir_name: str = "passage_index"
  triples_index_dir_name: str = "triples_index"
  _GRAPH_ADAPTER: ClassVar[pydantic.TypeAdapter[Any]] = pydantic.TypeAdapter(
      graph_lib.EntityNode | graph_lib.PassageNode | graph_lib.Edge
  )

  def __post_init__(self):
    """Validates document class interface at initialization.

    Raises:
      TypeError: If `doc_class` does not implement `to_json` and `from_json`.
    """
    for method in ("to_json", "from_json"):
      if not hasattr(self.doc_class, method):
        raise TypeError(f"{self.doc_class.__name__} must implement {method}")

  def save(
      self,
      index_state_dto: GraphRAGIndexState[DocT],
      base_path: str,
      storage_format: retrieval_serialization.StorageFormat | None = None,
  ) -> None:
    """Saves index state (docs, graph, and backend configs) to base_path.

    Args:
      index_state_dto: Container representing the GraphRAG index snapshot to
        save.
      base_path: Destination directory path.
      storage_format: How the subspace indices are saved. The documents and
        graph are always saved as JSON lines.

    Raises:
      ValueError: If `base_path` is not specified or set on the serializer.
    """

    if base_path:
      self.base_path = base_path
    if not self.base_path:
      raise ValueError("Please specify a base path to save the index to.")

    path = epath.Path(self.base_path)
    if not path.exists():
      path.mkdir(parents=True, exist_ok=True)

    embed_cfg_dict = None
    if index_state_dto.embed_backend_config is not None:
      embed_cfg_dict = (
          index_state_dto.embed_backend_config.to_dict()  # pyrefly: ignore[missing-attribute]
      )

    generate_cfg_dict = None
    if index_state_dto.generate_backend_config is not None:
      generate_cfg_dict = (
          index_state_dto.generate_backend_config.to_dict()  # pyrefly: ignore[missing-attribute]
      )

    num_docs = (
        len(index_state_dto.docs) if index_state_dto.docs is not None else 0
    )
    num_nodes = (
        len(getattr(index_state_dto.graph, "nodes", []))
        if index_state_dto.graph is not None
        else 0
    )
    num_edges = (
        len(getattr(index_state_dto.graph, "edges", []))
        if index_state_dto.graph is not None
        else 0
    )

    # Save manifest
    manifest = {
        "num_docs": num_docs,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "embed_backend_config": embed_cfg_dict,
        "generate_backend_config": generate_cfg_dict,
    }
    manifest_path = path / self.manifest_json_name
    with manifest_path.open("w") as f:
      json.dump(manifest, f)

    # Save documents to JSONL.
    if index_state_dto.docs is not None:
      docs_path = path / self.docs_jsonl_name
      with docs_path.open("w") as f:
        for doc in index_state_dto.docs:
          f.write(doc.to_json() + "\n")

    # Save graph to JSONL.
    if index_state_dto.graph is not None:
      nodes = getattr(index_state_dto.graph, "nodes", [])
      edges = getattr(index_state_dto.graph, "edges", [])
      graph_path = path / self.graph_jsonl_name
      with graph_path.open("wb") as f:
        for node in nodes:
          f.write(self._GRAPH_ADAPTER.dump_json(node) + b"\n")
        for edge in edges:
          f.write(self._GRAPH_ADAPTER.dump_json(edge) + b"\n")

    # Save subspace indices (`EmbeddingBasedDocumentIndex`).
    if index_state_dto.entity_index is not None:
      entity_dir = path / self.entity_index_dir_name
      index_state_dto.entity_index.save(str(entity_dir), storage_format)
    if index_state_dto.passage_index is not None:
      passage_dir = path / self.passage_index_dir_name
      index_state_dto.passage_index.save(str(passage_dir), storage_format)
    if index_state_dto.triples_index is not None:
      triples_dir = path / self.triples_index_dir_name
      index_state_dto.triples_index.save(str(triples_dir), storage_format)

  @executing.make_executable(copy_self=False)
  async def load(
      self,
      base_path: str | None = None,
      runtime_kwargs: dict[str, Any] | None = None,
  ) -> GraphRAGIndexState[DocT]:
    """Loads and reconstructs the `GraphRAGIndexState` from a directory.

    Reads `manifest.json`, `docs.jsonl`, `graph.jsonl`, and any existing
    subspace index subdirectories (`entity_index/`, `passage_index/`,
    `triples_index/`) to reconstruct the complete index snapshot.

    Args:
      base_path: The directory path to load from. If not provided, falls back to
        `self.base_path`.
      runtime_kwargs: Optional keyword arguments passed into the constructor of
        each subspace `EmbeddingBasedDocumentIndex` to construct their backends.
        It is used to supply runtime attributes like `api_key` or
        `thread_pool_size` etc. If not provided, defaults to `{}`.

    Returns:
      A populated `GraphRAGIndexState` container holding the reconstructed
      documents, graph structure, subspace vector indices, and backend configs.
    Raises:
      FileNotFoundError: If no valid index files (`manifest.json`, `docs.jsonl`,
        `graph.jsonl`, or subspace index directories) exist at `base_path`.
      ValueError: If `base_path` is neither provided nor set on the serializer.
    """

    if base_path:
      self.base_path = base_path
    if not self.base_path:
      raise ValueError("Please specify a base path to load the index from.")

    path = epath.Path(self.base_path)

    manifest_path = path / self.manifest_json_name
    docs_path = path / self.docs_jsonl_name
    graph_path = path / self.graph_jsonl_name
    entity_dir = path / self.entity_index_dir_name
    passage_dir = path / self.passage_index_dir_name
    triples_dir = path / self.triples_index_dir_name
    runtime_kwargs = runtime_kwargs or {}

    if not any((
        manifest_path.exists(),
        docs_path.exists(),
        graph_path.exists(),
        entity_dir.exists(),
        passage_dir.exists(),
        triples_dir.exists(),
    )):
      raise FileNotFoundError(
          "Missing index files (manifest.json, docs.jsonl, graph.jsonl, or"
          f" subspace index directories) at {path}"
      )

    # Load Documents from JSONL
    docs = None
    if docs_path.exists():
      docs = []
      with docs_path.open("r") as f:
        for line in f:
          if line.strip():
            docs.append(self.doc_class.from_json(line))

    # Load graph from JSONL.
    graph_obj = None
    if graph_path.exists():
      graph_obj = graph_lib.KnowledgeGraph()
      with graph_path.open("rb") as f:
        for line in f:
          if line.strip():
            item = self._GRAPH_ADAPTER.validate_json(line)
            if isinstance(item, graph_lib.Edge):
              graph_obj.upsert_edge(
                  source_id=item.source_node_id,
                  target_id=item.target_node_id,
                  weight=item.weight,
                  metadata=item.metadata,
                  edge_id=item.id,
              )
            else:
              graph_obj.upsert_node(item)

    # Reload subspace indices (`EmbeddingBasedDocumentIndex`).
    entity_index = None
    if entity_dir.exists():
      entity_index = indexing_lib.EmbeddingBasedDocumentIndex(
          runtime_kwargs=runtime_kwargs
      )
      await entity_index.load(str(entity_dir))

    passage_index = None
    if passage_dir.exists():
      passage_index = indexing_lib.EmbeddingBasedDocumentIndex(
          runtime_kwargs=runtime_kwargs
      )
      await passage_index.load(str(passage_dir))

    triples_index = None
    if triples_dir.exists():
      triples_index = indexing_lib.EmbeddingBasedDocumentIndex(
          runtime_kwargs=runtime_kwargs
      )
      await triples_index.load(str(triples_dir))

    embed_cfg = None
    generate_cfg = None
    if manifest_path.exists():
      with manifest_path.open("r") as f:
        manifest_data = json.load(f)
      embed_cfg_data = manifest_data.get("embed_backend_config")
      if embed_cfg_data is not None:
        embed_cfg = EmbedBackendConfig.from_dict(embed_cfg_data)

      generate_cfg_data = manifest_data.get("generate_backend_config")
      if generate_cfg_data is not None:
        generate_cfg = GenerateBackendConfig.from_dict(generate_cfg_data)

    return GraphRAGIndexState(
        docs=docs,
        graph=graph_obj,
        entity_index=entity_index,
        passage_index=passage_index,
        triples_index=triples_index,
        generate_backend_config=generate_cfg,
        embed_backend_config=embed_cfg,
    )
