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

"""Knowledge graph-based retrieval indices."""

import dataclasses
from typing import Any, Generic, Iterable, Sequence, TypeVar
from absl import logging
import numpy as np
from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization as retrieval_serialization
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import graphrag_prompts
from onetwo.stdlib.retrieval.graphrag import serialization

QueryT = TypeVar("QueryT")
RetrievalResultT = TypeVar("RetrievalResultT")
DocT = TypeVar("DocT")
EntityT = TypeVar("EntityT")
TripleT = TypeVar("TripleT")
Document = retrieval_data_structures.Document

_RETRIEVAL_NOT_SUPPORTED_MESSAGE = (
    "HippoRAGDocumentIndex only builds the knowledge graph and its entity,"
    " passage and triples indexes; it does not support direct retrieval via"
    " `retrieve`, `retrieve_with_scores` or `retrieve_doc_indices_and_scores`."
    " To serve the graph, expose it to an agent instead, e.g. through an MCP"
    " server or an agentic retrieval strategy that explores the graph with"
    " tools (using `graph`, `entity_index`, `passage_index` and"
    " `triples_index`)."
)


@dataclasses.dataclass(kw_only=True)
class BaseKnowledgeGraphIndex(
    Generic[QueryT, RetrievalResultT, DocT, EntityT, TripleT],
    indexing.Index[QueryT, RetrievalResultT, DocT],
):
  """Base class for knowledge graph-based retrieval indices.

  This class holds the core state and configuration for knowledge graph
  retrieval indices. It maintains the indexed documents, the underlying
  knowledge graph (nodes and edges), and the backend configurations required
  for generation and embedding. It also maintains specialized, vector retrieval
  indices (`EmbeddingBasedDocumentIndex`) for different graph subspaces
  (`entity_index`, `passage_index`, `triples_index`). It provides default
  `save` and `load` methods to serialize and restore the index state using
  `serialization.SimpleGraphRAGIndexSerializer`.

  Concrete subclasses are responsible for implementing index construction
  (e.g., `add_docs` or `create_index`) and, if supported, query retrieval
  (`retrieve` and `retrieve_with_scores`). Subclasses that extract entities and
  triples from documents implement `_make_extractor`, which builds the extractor
  from the index settings; the extractor is then available (built lazily, once)
  through `entity_and_triples_extractor`. It is not a constructor argument, so
  that the init fields of the index are plain settings.


  Attributes:
    generate_backend_config: Configuration for the generation LLM backend used
      during graph construction (e.g., for entity and triple extraction).
    embed_backend_config: Configuration for the embedder backend used to compute
      vector embeddings for graph nodes and queries.
    runtime_kwargs: Environment-specific deployment settings (e.g.,
      `threadpool_size`, `project`) and credentials (`api_key`, `credentials`)
      that are not stored in the serializable configuration. Since it may hold
      credentials, it is excluded from `repr` and equality, and never saved by
      the serializers.
  """

  generate_backend_config: backend_registry.GenerateBackendConfig | None = None
  embed_backend_config: backend_registry.EmbedBackendConfig | None = None
  # Environment settings and credentials: excluded from `repr` and equality,
  # and never saved by the serializers.
  runtime_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  # Optional `EmbeddingBasedDocumentIndex` storing indexed `Document`
  # representations of graph entity nodes (`EntityNode`).
  _entity_index: indexing.EmbeddingBasedDocumentIndex | None = (
      dataclasses.field(default=None, init=False, repr=False)
  )
  # Optional `EmbeddingBasedDocumentIndex` storing indexed `Document`
  # representations of source document passages (`PassageNode`).
  _passage_index: indexing.EmbeddingBasedDocumentIndex | None = (
      dataclasses.field(default=None, init=False, repr=False)
  )
  # Optional `EmbeddingBasedDocumentIndex` storing indexed `Document`
  # representations of relational triples (`Edge`).
  _triples_index: indexing.EmbeddingBasedDocumentIndex | None = (
      dataclasses.field(default=None, init=False, repr=False)
  )
  # Documents managed by the index. If `None`, the corpus has not been
  # initialized yet.
  _docs: list[DocT] | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  # Knowledge graph storing the nodes and edges constructed from `_docs`.
  _graph: graph_lib.KnowledgeGraph | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  # Extractor built lazily by `_make_extractor` from the index settings.
  _entity_and_triples_extractor: (
      graph_lib.EntityAndTripleExtractorProtocol[DocT, EntityT, TripleT] | None
  ) = dataclasses.field(default=None, init=False, repr=False, compare=False)

  def _make_extractor(
      self,
  ) -> graph_lib.EntityAndTripleExtractorProtocol[DocT, EntityT, TripleT]:
    """Returns a new entity and triples extractor built from the settings.

    Subclasses that build the knowledge graph from documents override this
    method. It is also the extension point for using a different extractor
    (any implementation of `EntityAndTripleExtractorProtocol`).

    Raises:
      NotImplementedError: If the subclass does not support extraction.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not define an entity and triples"
        " extractor; override `_make_extractor`."
    )

  @property
  def entity_and_triples_extractor(
      self,
  ) -> graph_lib.EntityAndTripleExtractorProtocol[DocT, EntityT, TripleT]:
    """Returns the extractor, building it with `_make_extractor` on first use."""
    if self._entity_and_triples_extractor is None:
      self._entity_and_triples_extractor = self._make_extractor()
    return self._entity_and_triples_extractor

  def save(
      self,
      base_path: str,
      storage_format: retrieval_serialization.StorageFormat | None = None,
  ) -> None:
    """Saves the entire index state to a directory.

    Args:
      base_path: The directory to save the index state to.
      storage_format: How the entity, passage and triples indexes are saved. The
        documents and graph are always saved as JSON lines.
    """
    state = serialization.GraphRAGIndexState(
        docs=self.docs,
        graph=self.graph,
        entity_index=self._entity_index,
        passage_index=self._passage_index,
        triples_index=self._triples_index,
        generate_backend_config=self.generate_backend_config,
        embed_backend_config=self.embed_backend_config,
    )
    serializer = serialization.SimpleGraphRAGIndexSerializer(doc_class=Document)
    serializer.save(state, base_path, storage_format)

  @executing.make_executable(copy_self=False)
  async def load(
      self, base_path: str, runtime_kwargs: dict[str, Any] | None = None
  ) -> None:
    """Loads the index state from disk and restores all underlying components.

    This method reconstructs the core corpus (`docs`), the graph topology
    (`graph`), backend configurations (`embed_backend_config` and
    `generate_backend_config`), and any populated vector sub-indices
    (`entity_index`, `passage_index`, `triples_index`).
    Args:
      base_path: The directory path from which to load the index state.
      runtime_kwargs: Optional dictionary of runtime keyword arguments (e.g.,
        `api_key`, `thread_pool_size` etc.) to forward to the underlying
        `EmbeddingBasedDocumentIndex` subspace loaders. If `None`, defaults to
        `self.runtime_kwargs`.

    Raises:
      FileNotFoundError: If the manifest or required index files cannot be found
        under `base_path`.
      ValueError: If any underlying subspace index is corrupted or fails to
      load.
    """
    serializer = serialization.SimpleGraphRAGIndexSerializer(doc_class=Document)
    runtime_kwargs = runtime_kwargs or self.runtime_kwargs
    state: serialization.GraphRAGIndexState = await serializer.load(
        base_path, runtime_kwargs=runtime_kwargs
    )
    self._docs = list(state.docs) if state.docs is not None else None
    self._graph = state.graph
    self._entity_index = state.entity_index
    self._passage_index = state.passage_index
    self._triples_index = state.triples_index
    self.generate_backend_config = state.generate_backend_config
    self.embed_backend_config = state.embed_backend_config
    # The extractor depends on `generate_backend_config`; rebuild it on use.
    self._entity_and_triples_extractor = None

  @property
  def docs(self) -> list[DocT] | None:
    """Returns the indexed documents, or `None` if no corpus was built yet."""
    return self._docs

  @property
  def graph(self) -> graph_lib.KnowledgeGraph | None:
    """Returns the knowledge graph, or `None` if no graph was built yet."""
    return self._graph

  @property
  def entity_index(self) -> indexing.EmbeddingBasedDocumentIndex | None:
    """Returns the internal entity retrieval index."""
    return self._entity_index

  @property
  def passage_index(self) -> indexing.EmbeddingBasedDocumentIndex | None:
    """Returns the internal passage retrieval index."""
    return self._passage_index

  @property
  def triples_index(self) -> indexing.EmbeddingBasedDocumentIndex | None:
    """Returns the internal triples retrieval index."""
    return self._triples_index

  @property
  def num_docs(self) -> int:
    """Returns the total number of documents in the index."""
    return len(self.docs) if self.docs is not None else 0

  def get_docs(self) -> Sequence[DocT] | Iterable[DocT]:
    """Returns the sequence of indexed documents."""
    return self.docs if self.docs is not None else []

  def destroy_index(self) -> None:
    """Destroys and clears all in-memory graph index state."""
    self._docs = None
    self._graph = None
    self._entity_index = None
    self._passage_index = None
    self._triples_index = None


@registry.register("HippoRAGDocumentIndex")
@dataclasses.dataclass(kw_only=True)
class HippoRAGDocumentIndex(
    BaseKnowledgeGraphIndex[
        str, Document, Document, graph_lib.Entity, graph_lib.Triple
    ],
):
  """Knowledge graph index for HippoRAG.

  Builds a knowledge graph from the documents (passage and entity nodes,
  mention edges, relational triples and, optionally, semantic similarity
  edges between entities), together with embedding indexes over its entities,
  passages and triples, following https://arxiv.org/pdf/2502.14802.

  This index does not implement retrieval: `retrieve`,
  `retrieve_with_scores` and `retrieve_doc_indices_and_scores` raise
  `NotImplementedError`. Serving the built graph to an agent (e.g. through an
  MCP server) works
  better.

  Entities and triples are extracted from each document with a
  `GraphRAGEntityAndTriplesExtractor` that the index builds from its plain
  settings (`extraction_prompt`, `entity_types`, `generate_backend_config` and
  `runtime_kwargs`). To customize the extraction beyond these settings (e.g.
  `extraction_prompt_template`), subclass the extractor,
  and override `_make_extractor` in a subclass of the index to return it.

  Attributes:
    extraction_prompt: Template of the instructions given to the LLM to extract
      entities and triples from a document, formatted with `str.format` with
      `entity_types` (so literal braces must be doubled). The document is
      appended to it by the extractor. Defaults to
      `graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT`.
    entity_types: Entity types to extract, joined with ", " into the
      `{entity_types}` placeholder of `extraction_prompt`. Defaults to
      `graphrag_prompts.DEFAULT_ENTITY_TYPES`.
    similarity_threshold: Minimum cosine similarity required to add a semantic
      similarity edge between two entity nodes. Defaults to 0.8, taken from
      Table 13 in https://arxiv.org/pdf/2502.14802.
    ignore_triples: Whether to ignore indexing relational triples into the
      triples index. Defaults to True.
    ignore_similarity_edges: Whether to ignore semantic similarity edges between
      entity nodes during graph construction. Defaults to True.
  """

  extraction_prompt: str = (
      graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT
  )
  entity_types: tuple[str, ...] = tuple(graphrag_prompts.DEFAULT_ENTITY_TYPES)
  similarity_threshold: float = 0.8
  ignore_triples: bool = True
  ignore_similarity_edges: bool = True

  def _make_extractor(self) -> graph_lib.GraphRAGEntityAndTriplesExtractor:
    """Returns the extractor configured with the index settings."""
    return graph_lib.GraphRAGEntityAndTriplesExtractor(
        extraction_prompt=self.extraction_prompt.format(
            entity_types=", ".join(self.entity_types)
        ),
        generate_backend_config=self.generate_backend_config,
        runtime_kwargs=self.runtime_kwargs,
    )

  def _validate_docs(self, docs: Sequence[Document]) -> None:
    """Validates that candidate documents have valid identifiers before indexing.

    Args:
      docs: Sequence of documents to validate.

    Raises:
      ValueError: If any document in `docs` evaluates to `None`, an empty
      string, or a whitespace-only string for its `doc_id` attribute.
    """
    docs_missing_ids = []
    for i, doc in enumerate(docs):
      doc_id = doc.doc_id
      if not doc_id or not doc_id.strip():
        docs_missing_ids.append(i)
    if docs_missing_ids:
      raise ValueError(
          f"Docs missing doc_id ({len(docs_missing_ids)} / {len(list(docs))}): "
          f"{docs_missing_ids[:10]} ..."
      )

  def _add_entities_to_graph(
      self, entities: Sequence[graph_lib.Entity]
  ) -> dict[str, str]:
    """Adds or merges canonical entity nodes in the knowledge graph.

    Each extracted entity is inserted with its canonical global ID
    (`entity_{normalized_name}_{normalized_type}`). If an entity node with the
    same ID already exists in the graph, it is merged with the existing node.

    Args:
      entities: Sequence of extracted entity objects to insert or merge.

    Returns:
      A dictionary mapping normalized entity names (`ent.name.strip().lower()`)
      to their corresponding canonical node IDs in `self.graph`.

    Raises:
      ValueError: If `self.graph` has not been initialized.
    """
    if self.graph is None:
      raise ValueError("KnowledgeGraph is not initialized.")

    entity_name_to_id: dict[str, str] = {}
    for entity in entities:
      norm_name = (entity.name or str(entity)).strip().lower()
      norm_type = (entity.type or "").strip().lower()
      canonical_id = f"entity_{norm_name}_{norm_type}"

      node = graph_lib.EntityNode(
          id=canonical_id,
          entity=entity,
      )
      self.graph.upsert_node(node)
      entity_name_to_id[norm_name] = canonical_id
    return entity_name_to_id

  def _add_passage_to_graph(
      self, doc: Document, doc_index: int
  ) -> graph_lib.PassageNode:
    """Adds a passage node representing a source document to the graph.

    To support fast matrix lookups during retrieval, the passage node's `doc_id`
    field
    stores the integer index (`doc_index`) of the document in the corpus
    (`self.docs`), while the original string ID (`doc.doc_id`)
    is preserved in the node's `metadata['doc_id']`.

    Args:
      doc: The document object being ingested.
      doc_index: The integer index of `doc` within the corpus (`self.docs`).

    Returns:
      The created `PassageNode` added to `self.graph`.

    Raises:
      ValueError: If `self.graph` has not been initialized.
    """
    if self.graph is None:
      raise ValueError("KnowledgeGraph is not initialized.")

    raw_doc_id = doc.doc_id or str(len(self.graph.nodes))
    passage_id = f"passage_{raw_doc_id}"
    corpus_doc_index = doc_index

    node = graph_lib.PassageNode(
        id=passage_id,
        doc_id=corpus_doc_index,
        metadata={"doc_id": raw_doc_id},
    )
    self.graph.upsert_node(node)
    return node

  def _add_mention_edges_to_graph(
      self,
      passage_node_id: str,
      entity_node_id: str,
  ) -> None:
    """Adds bidirectional mention edges connecting a passage and an entity.

    Creates both a forward `MENTIONS` edge from the passage node to the entity
    node and a reverse `MENTIONED_IN` edge from the entity node back to the
    passage node.

    Args:
      passage_node_id: The ID of the `PassageNode` mentioning the entity.
      entity_node_id: The ID of the `EntityNode` mentioned in the passage.

    Raises:
      ValueError: If `self.graph` has not been initialized.
    """
    if self.graph is None:
      raise ValueError("KnowledgeGraph is not initialized.")

    entity_name = self.graph.get_entity_name(entity_node_id)
    passage_name = passage_node_id

    self.graph.upsert_edge(
        source_id=passage_node_id,
        target_id=entity_node_id,
        metadata={"type": "mention"},
        triple=graph_lib.Triple(
            source=passage_name,
            relation=graph_lib.BUILTIN_RELATION_MENTIONS,
            target=entity_name,
        ),
    )
    self.graph.upsert_edge(
        source_id=entity_node_id,
        target_id=passage_node_id,
        metadata={"type": "mention"},
        triple=graph_lib.Triple(
            source=entity_name,
            relation=graph_lib.BUILTIN_RELATION_MENTIONED_IN,
            target=passage_name,
        ),
    )

  def _add_relational_triples_to_graph(
      self,
      triples: Sequence[graph_lib.Triple],
      local_entity_name_to_id: dict[str, str],
  ) -> None:
    """Adds relational edges from extracted triples to the knowledge graph.

    Uses the document's local entity name mapping (`local_entity_name_to_id`)
    to resolve the raw string names of each triple (`triple.source`,
    `triple.target`) to their
    canonical entity node IDs in `self.graph`.

    Args:
      triples: Sequence of extracted relational triples to insert as edges.
      local_entity_name_to_id: Mapping from normalized entity names to canonical
        node IDs for entities extracted from the same source document.

    Raises:
      ValueError: If `self.graph` has not been initialized.
    """
    if self.graph is None:
      raise ValueError("KnowledgeGraph is not initialized.")

    for triple in triples:
      src_norm = getattr(triple, "source", str(triple)).strip().lower()
      tgt_norm = getattr(triple, "target", str(triple)).strip().lower()

      src_id = local_entity_name_to_id.get(src_norm)
      tgt_id = local_entity_name_to_id.get(tgt_norm)

      if src_id is not None and tgt_id is not None:
        self.graph.upsert_edge(
            source_id=src_id,
            target_id=tgt_id,
            metadata={"triples": [triple]},
            triple=triple,
        )

  async def _get_index_and_existing_ids(
      self, index: indexing.EmbeddingBasedDocumentIndex | None
  ) -> tuple[indexing.EmbeddingBasedDocumentIndex, set[str]]:
    """Initializes or updates index configs and returns existing doc IDs."""
    if index is None:
      index = indexing.EmbeddingBasedDocumentIndex(
          embed_backend_config=self.embed_backend_config,
          runtime_kwargs=self.runtime_kwargs,
      )
    else:
      index.embed_backend_config = (
          index.embed_backend_config or self.embed_backend_config
      )
      index.runtime_kwargs = index.runtime_kwargs or self.runtime_kwargs
    return index, {doc.doc_id for doc in index.get_docs()}

  async def _add_entities_to_entity_index(
      self, entity_ids: Sequence[str]
  ) -> list[graph_lib.EntityNode]:
    """Adds entity nodes identified by ID to the entity index.

    Args:
      entity_ids: Sequence of canonical entity node IDs to consider for
        indexing.

    Returns:
      A list of `EntityNode`s that were newly added to `_entity_index`.

    Raises:
      ValueError: If `self.graph` has not been initialized.
    """
    if not entity_ids:
      return []

    if self.graph is None:
      raise ValueError("KnowledgeGraph is not initialized.")

    self._entity_index, existing_ids = await self._get_index_and_existing_ids(
        self._entity_index
    )

    nodes_to_index: list[graph_lib.EntityNode] = []
    for ent_id in set(entity_ids) - existing_ids:
      node = self.graph.get_node(ent_id)
      if isinstance(node, graph_lib.EntityNode):
        nodes_to_index.append(node)

    if nodes_to_index:
      docs_to_index = [node.to_document() for node in nodes_to_index]
      await self._entity_index.add_docs(docs_to_index)

    return nodes_to_index

  async def _add_passages_to_passage_index(
      self, passage_docs: Sequence[Document]
  ) -> None:
    """Adds passage documents to the passage index."""
    if not passage_docs:
      return

    self._passage_index, existing_ids = await self._get_index_and_existing_ids(
        self._passage_index
    )
    docs_to_index = [
        doc for doc in passage_docs if doc.doc_id not in existing_ids
    ]
    if docs_to_index:
      await self._passage_index.add_docs(docs_to_index)

  async def _add_triples_to_triples_index(
      self, edges: Sequence[graph_lib.Edge]
  ) -> None:
    """Adds relational triple edges to the triples index."""
    if not edges:
      return

    self._triples_index, existing_ids = await self._get_index_and_existing_ids(
        self._triples_index
    )
    triple_edges = [edge for edge in edges if edge.id not in existing_ids]
    if triple_edges:
      docs_to_index = [edge.to_document() for edge in triple_edges]
      await self._triples_index.add_docs(docs_to_index)

  async def _add_similarity_edges(
      self,
      new_entity_nodes: Sequence[graph_lib.EntityNode],
      similarity_threshold: float,
      query_batch_size: int = 256,
  ) -> None:
    """Updates semantic similarity edges for newly added entity nodes.

    Reuses the embeddings already stored in `self._entity_index` for the new
    entity nodes as queries (avoiding one embedding call per entity), and runs
    them through the entity index's searcher in batches. For each new node,
    every indexed entity with similarity `>= similarity_threshold` gets
    bidirectional `SEMANTICALLY_SIMILAR_TO` edges in `self.graph`.

    Args:
      new_entity_nodes: Newly added EntityNodes from add_docs. Each must already
        be indexed in `self._entity_index`.
      similarity_threshold: Minimum similarity score required to add an edge.
      query_batch_size: Number of query embeddings searched per batch. Each
        batch materializes a `query_batch_size x num_entities` score matrix, so
        this bounds peak memory.
    """
    if not new_entity_nodes or self.graph is None or self._entity_index is None:
      return

    entity_index = self._entity_index
    indexed_docs = list(entity_index.get_docs())
    # Embeddings are stored in the same order as `get_docs()`.
    doc_embeddings = entity_index._doc_embeddings  # pylint: disable=protected-access
    doc_id_to_pos = {doc.doc_id: i for i, doc in enumerate(indexed_docs)}

    query_nodes: list[graph_lib.EntityNode] = []
    query_embeddings: list[np.ndarray] = []
    for node in new_entity_nodes:
      pos = doc_id_to_pos.get(node.id)
      if pos is None:
        logging.warning("Entity %s not found in entity index.", node.id)
        continue
      query_nodes.append(node)
      query_embeddings.append(doc_embeddings[pos])

    for batch_start in range(0, len(query_nodes), query_batch_size):
      batch_nodes = query_nodes[batch_start : batch_start + query_batch_size]
      batch_embeddings = np.stack(
          query_embeddings[batch_start : batch_start + query_batch_size]
      )
      # Results are sorted by descending score for each query.
      ranked_ids, ranked_scores = await entity_index.searcher.search_batched(
          batch_embeddings, final_num_neighbors=None
      )

      for new_node, doc_indices, scores in zip(
          batch_nodes, ranked_ids, ranked_scores
      ):
        for doc_idx, score in zip(doc_indices, scores):
          if score < similarity_threshold:
            break
          matched_doc = indexed_docs[doc_idx]
          if matched_doc.doc_id == new_node.id:
            continue

          if self.graph.has_similarity_edge(new_node.id, matched_doc.doc_id):
            continue

          target_node = self.graph.get_node(matched_doc.doc_id)
          if not isinstance(target_node, graph_lib.EntityNode):
            continue

          target_name = target_node.entity.name

          self.graph.upsert_edge(
              source_id=new_node.id,
              target_id=matched_doc.doc_id,
              weight=float(score),
              metadata={"type": "similarity", "score": float(score)},
              triple=graph_lib.Triple(
                  source=new_node.entity.name,
                  relation=graph_lib.BUILTIN_RELATION_SEMANTICALLY_SIMILAR_TO,
                  target=target_name,
              ),
          )
          self.graph.upsert_edge(
              source_id=matched_doc.doc_id,
              target_id=new_node.id,
              weight=float(score),
              metadata={"type": "similarity", "score": float(score)},
              triple=graph_lib.Triple(
                  source=target_name,
                  relation=graph_lib.BUILTIN_RELATION_SEMANTICALLY_SIMILAR_TO,
                  target=new_node.entity.name,
              ),
          )

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Sequence[Document]) -> None:  # pyrefly: ignore[bad-override]
    """Ingests documents into the corpus and populates the knowledge graph.

    This executes the indexing pipeline:
    1. Validates input documents and appends them to the index corpus
       (`self.docs`).
    2. Concurrently invokes `entity_and_triples_extractor` (built by
       `_make_extractor`) across all documents.
    3. Populates `self.graph` with document passage nodes,
       canonical deduplicated entity nodes (`entity_{name}_{type}`),
       bidirectional mention edges connecting them, and relational triples.
    4. Converts newly discovered canonical entity nodes to documents, indexes
       them into `self._entity_index`, and constructs semantic similarity edges
       (`SEMANTICALLY_SIMILAR_TO`) unless `ignore_similarity_edges` is True.
    5. Indexes source document passages into `self._passage_index`.
    6. Indexes relational triples into `self._triples_index` unless
       `ignore_triples` is True.

    Args:
      docs: Sequence of documents to add to the index.
    """
    new_docs = list(docs)
    if not new_docs:
      return
    self._validate_docs(new_docs)

    if self._docs is None:
      self._docs = []
    start_doc_idx = len(self._docs)
    self._docs.extend(new_docs)

    if self._graph is None:
      self._graph = graph_lib.KnowledgeGraph()

    entity_and_triples_extractor = self.entity_and_triples_extractor
    extraction_executables = [
        entity_and_triples_extractor(doc) for doc in new_docs
    ]
    extraction_results: Sequence[
        tuple[Sequence[graph_lib.Entity], Sequence[graph_lib.Triple]]
    ] = await executing.par_iter(extraction_executables)

    new_entity_ids: set[str] = set()
    for i, (entities, triples) in enumerate(extraction_results):
      doc = new_docs[i]
      doc_idx = start_doc_idx + i
      passage_node = self._add_passage_to_graph(doc, doc_index=doc_idx)
      local_entity_name_to_id = self._add_entities_to_graph(entities)
      new_entity_ids.update(local_entity_name_to_id.values())
      for canonical_id in set(local_entity_name_to_id.values()):
        self._add_mention_edges_to_graph(
            passage_node_id=passage_node.id,
            entity_node_id=canonical_id,
        )

      self._add_relational_triples_to_graph(
          triples=triples,
          local_entity_name_to_id=local_entity_name_to_id,
      )

    indexed_nodes = await self._add_entities_to_entity_index(
        list(new_entity_ids)
    )
    if not self.ignore_similarity_edges:
      await self._add_similarity_edges(
          indexed_nodes,
          similarity_threshold=self.similarity_threshold,
      )

    await self._add_passages_to_passage_index(new_docs)
    if not self.ignore_triples and self.graph is not None:
      await self._add_triples_to_triples_index(self.graph.edges)

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_indices_and_scores(
      self,
      query: str,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[tuple[int, float]]:
    """Not supported; see the class docstring.

    Raises:
      NotImplementedError: Always.
    """
    raise NotImplementedError(_RETRIEVAL_NOT_SUPPORTED_MESSAGE)

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(  # pyrefly: ignore[bad-override]
      self,
      query: str,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[tuple[Document, float]]:
    """Not supported; see the class docstring.

    Raises:
      NotImplementedError: Always.
    """
    raise NotImplementedError(_RETRIEVAL_NOT_SUPPORTED_MESSAGE)

  @executing.make_executable(copy_self=False)
  async def retrieve(  # pyrefly: ignore[bad-override]
      self,
      query: str,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[Document]:
    """Not supported; see the class docstring.

    Raises:
      NotImplementedError: Always.
    """
    raise NotImplementedError(_RETRIEVAL_NOT_SUPPORTED_MESSAGE)
