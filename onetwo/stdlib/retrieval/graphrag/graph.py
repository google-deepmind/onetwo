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

"""Module defining the classes for knowledge graphs and triple extraction."""

import abc
from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
from typing import Any, Generic, Optional, Protocol, Tuple, TypeVar
from absl import logging
import networkx as nx
from onetwo.backends import backend_registry
from onetwo.backends import backends_base
from onetwo.builtins import composables
from onetwo.builtins import llm
from onetwo.core import composing
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import retrieval_data_structures
import pydantic

Document = retrieval_data_structures.Document
Composable = composing.Composable
DocT = TypeVar("DocT")
EntityT = TypeVar("EntityT")
TripleT = TypeVar("TripleT")

# Built-in structural graph relations connecting passage nodes and entity nodes.
BUILTIN_RELATION_MENTIONS = "MENTIONS"
BUILTIN_RELATION_MENTIONED_IN = "MENTIONED_IN"
BUILTIN_RELATION_SEMANTICALLY_SIMILAR_TO = "SEMANTICALLY_SIMILAR_TO"


class Entity(pydantic.BaseModel):
  """A distinct entity, concept, or proper noun extracted from text.

  Attributes:
    name: The canonical, normalized display name of the entity.
    type: The semantic category or ontological classification of the entity.
    description: A brief 1-2 sentence contextual summary of what this entity is
      and its specific role within the source document.
  """

  name: str = pydantic.Field(
      default="", description="Canonical display name of the entity."
  )
  type: str = pydantic.Field(
      default="",
      description="Ontological classification of the entity.",
  )
  description: Optional[str] = pydantic.Field(
      default=None,
      description="Brief 1-2 sentence summary of the entity in context.",
  )


class Triple(pydantic.BaseModel):
  """A directed relational assertion connecting two extracted entities.

  Attributes:
    source: The exact canonical name of the subject entity.
    relation: Predicate phrase connecting source and target.
    target: The exact canonical name of the object entity.
  """

  source: str = pydantic.Field(
      ..., description="Exact name of the subject entity."
  )
  relation: str = pydantic.Field(
      default="ASSOCIATED_WITH",
      description="Predicate phrase connecting source and target.",
  )
  target: str = pydantic.Field(
      ..., description="Exact name of the object entity."
  )


class TripleExtractionResult(pydantic.BaseModel):
  """Container for all extracted entities and triples from a document.

  Attributes:
    entities: List of entities extracted from the document.
    triples: List of relational triples extracted from the document.
  """

  entities: list[Entity] = pydantic.Field(default_factory=list)
  triples: list[Triple] = pydantic.Field(default_factory=list)

  def sanitize_entities(self) -> None:
    """Sanitizes canonical names and ontology types."""
    for entity in self.entities:
      entity.name = entity.name.strip().lower()
      entity.type = entity.type.strip().lower()
    self.entities[:] = [
        entity for entity in self.entities if entity.name and entity.type
    ]

  def sanitize_triples(self) -> None:
    """Sanitizes endpoints/predicates and enforces referential safety."""
    for triple in self.triples:
      triple.source = triple.source.strip().lower()
      triple.relation = triple.relation.strip()
      triple.target = triple.target.strip().lower()
    self.triples[:] = [
        triple
        for triple in self.triples
        if triple.source and triple.relation and triple.target
    ]

    existing_names = {entity.name for entity in self.entities}
    self.triples[:] = [
        triple
        for triple in self.triples
        if triple.source in existing_names and triple.target in existing_names
    ]


@dataclasses.dataclass(kw_only=True)
class BaseNode(metaclass=abc.ABCMeta):
  """Abstract base class for nodes in a graph.

  Attributes:
    id: Unique identifier for the node.
    metadata: Dictionary storing additional data associated with the node.
      Currently is used to node to document mapping, and can be extended for
      other attributes.
  """

  id: str = ""
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class Edge:
  """Class for edges in a graph.

  Attributes:
    id: Unique identifier for the edge.
    source_node_id: Identifier of the source node.
    target_node_id: Identifier of the target node.
    weight: Numeric weight or strength of the edge.
    metadata: Dictionary storing additional data associated with the edge (e.g.,
      relation predicate). One of the fields in metadata is `triples`, which is
      used to store the canonical list of triples (`subject relation object`)
      that the edge represents.
  """

  id: str = ""
  source_node_id: str = ""
  target_node_id: str = ""
  weight: float = 1.0
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  def to_document(self) -> Document:
    """Converts the edge into a `Document` object for dense vector indexing.

    For edge embedding generation, the document `title` is set to `""`.
    The `content` is set to newline-separated canonical triple strings formatted
    from `metadata['triples']`. Structural edge properties (`source_node_id`,
    `target_node_id`, `weight`) are preserved safely in the document
    `metadata`.

    Returns:
      A `Document` object whose `doc_id` matches the edge ID, whose `content`
      is formatted for dense vector indexing, and whose `metadata` dictionary is
      populated with `"type": "triple"`, `source_node_id`, `target_node_id`,
      and `weight`.

    Raises:
      ValueError: If the edge metadata contains an unsupported triple type.
    """

    def _format_triple(t: Any) -> str:
      if isinstance(t, Triple):
        return f"{t.source} {t.relation} {t.target}".strip()
      if isinstance(t, dict):
        return (
            f"{t.get('source', '')} {t.get('relation', '')}"
            f" {t.get('target', '')}".strip()
        )
      if isinstance(t, str):
        return t.strip()
      raise ValueError(f"Unsupported triple type: {type(t)}.")

    triples = self.metadata.get("triples", [])
    if isinstance(triples, list) and triples:
      formatted = [_format_triple(t) for t in triples]
      content = "\n".join(t for t in formatted if t)
    else:
      content = ""

    if not content:
      content = f"{self.source_node_id} {self.target_node_id}".strip()

    return Document(
        doc_id=self.id,
        title="",
        content=content,
        metadata={
            "type": "triple",
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "weight": self.weight,
        },
    )

  @classmethod
  def from_document(cls, doc: Document) -> "Edge":
    """Reconstructs an `Edge` object from a `Document`.

    Args:
      doc: The `Document` to reconstruct the edge from.

    Returns:
      An `Edge` populated with the ID, endpoints, and weight extracted from
      `doc.metadata`, along with metadata preserving any extracted triples.
    """
    meta = dict(doc.metadata)
    source = str(meta.pop("source_node_id", ""))
    target = str(meta.pop("target_node_id", ""))
    weight = float(meta.pop("weight", 1.0))
    meta.pop("type", None)
    if "triples" not in meta:
      meta["triples"] = [str(doc.content)] if doc.content else []
    return cls(
        id=doc.doc_id,
        source_node_id=source,
        target_node_id=target,
        weight=weight,
        metadata=meta,
    )


@dataclasses.dataclass(kw_only=True)
class EntityNode(BaseNode):
  """Node representing an entity.

  Attributes:
    entity: The underlying Entity object represented by this node.
  """

  entity: Entity = dataclasses.field(default_factory=Entity)

  def to_document(self) -> Document:
    """Converts the entity node into a `Document` object.

    To ensure better embedding quality, the document `title` is set to the
    canonical entity name (`self.entity.name`), and `content` is set to
    `self.entity.description`.
    The `ontology_type` is preserved safely in the document
    `metadata`.

    Returns:
      A `Document` object whose `doc_id` matches the node ID, whose `title`
      contains the canonical entity name (`self.entity.name`) and whose
      `content` is set to `self.entity.description`, and whose
      `metadata` dictionary is populated with `"type": "entity"`,
      `ontology_type`, and any custom attributes from
      `node.metadata`.
    """
    return Document(
        doc_id=self.id,
        title=self.entity.name,
        content=self.entity.description or "",
        metadata={
            "type": "entity",
            "ontology_type": self.entity.type,
            **self.metadata,
        },
    )

  @classmethod
  def from_document(cls, doc: Document) -> "EntityNode":
    """Reconstructs an `EntityNode` object from an indexed `Document`.

    Args:
      doc: The `Document` to reconstruct the entity node from.

    Returns:
      An `EntityNode` holding the reconstructed `Entity` object (with name from
      `doc.title`, and description from `doc.content`), along with any remaining
      custom attributes preserved in `metadata`.
    """
    meta = dict(doc.metadata)
    entity = Entity(
        name=str(doc.title),
        type=str(meta.pop("ontology_type", "")),
        description=str(doc.content),
    )
    meta.pop("type", None)
    return cls(id=doc.doc_id, entity=entity, metadata=meta)


@dataclasses.dataclass(kw_only=True)
class PassageNode(BaseNode):
  """Node representing a passage/document.

  Attributes:
    doc_id: Index of the document that this passage node represents from the
      corpus.
  """

  doc_id: int = dataclasses.field(default=-1)


@dataclasses.dataclass(kw_only=True)
class KnowledgeGraph(Generic[TripleT]):
  """Class representing a knowledge graph.

  Attributes:
    max_description_length: Maximum allowed length for entity descriptions when
      merging entity nodes in upsert_node.
  """

  max_description_length: int = 3000
  _nodes_by_id: dict[str, BaseNode] = dataclasses.field(
      default_factory=dict, init=False, repr=False
  )
  _edges_by_key: dict[tuple[str, str], Edge] = dataclasses.field(
      default_factory=dict, init=False, repr=False
  )
  _entity_name_by_id: dict[str, str] = dataclasses.field(
      default_factory=dict, init=False, repr=False
  )
  # Cached NetworkX directed graph representation of the knowledge graph nodes
  # and edges. Defaults to None to indicate that the cache is uninitialized or
  # has been invalidated. It is lazily constructed in `to_networkx()` and reset
  # to None whenever nodes or edges are added or modified (via `upsert_node()`
  # or `upsert_edge()`).
  _di_graph: nx.DiGraph | None = dataclasses.field(
      default=None, init=False, repr=False
  )

  @property
  def nodes(self) -> list[BaseNode]:
    """Returns the list of nodes in the graph."""
    return list(self._nodes_by_id.values())

  @property
  def edges(self) -> list[Edge]:
    """Returns the list of edges in the graph."""
    return list(self._edges_by_key.values())

  def upsert_node(
      self, node: BaseNode, merge_description: bool = True
  ) -> BaseNode:
    """Inserts a node or merges it with an existing node if ID is present.

    When consolidating canonical entities across documents, if `node.id` already
    exists in the graph and both the existing and incoming nodes are
    `EntityNode`s, this concatenates their descriptions (if `merge_description`
    is True) and merges their metadata dictionaries.

    Args:
      node: The incoming node to insert or merge.
      merge_description: Whether to concatenate entity descriptions when merging
        canonical entity nodes.

    Returns:
      The newly inserted or consolidated `BaseNode` in `self._nodes_by_id`.
    """
    self._di_graph = None
    if node.id not in self._nodes_by_id:
      self._nodes_by_id[node.id] = node
      if isinstance(node, EntityNode):
        self._entity_name_by_id[node.id] = node.entity.name
      return node

    existing_node = self._nodes_by_id[node.id]
    if not (
        isinstance(existing_node, EntityNode) and isinstance(node, EntityNode)
    ):
      raise ValueError(
          "Cannot merge non-entity nodes with same IDs. Existing node: "
          f"{existing_node}, incoming node: {node}"
      )

    if merge_description:
      old_desc = (existing_node.entity.description or "").strip()
      new_desc = (node.entity.description or "").strip()
      if new_desc and new_desc not in old_desc:
        # Only append if new_desc provides new information and length is within
        # budget.
        if len(old_desc) < self.max_description_length:
          existing_node.entity.description = (
              f"{old_desc}\n{new_desc}".strip() if old_desc else new_desc
          )

    if node.metadata:
      existing_node.metadata.update(node.metadata)

    return existing_node

  def get_node(self, node_id: str) -> BaseNode | None:
    """Returns the node for the given node ID, or None if not present."""
    return self._nodes_by_id.get(node_id)

  def has_similarity_edge(self, source_id: str, target_id: str) -> bool:
    """Returns True if a similarity edge or triple exists between source_id and target_id."""
    key = (source_id, target_id)
    edge = self._edges_by_key.get(key)
    if edge is None:
      return False
    triples = edge.metadata.get("triples", [])
    return any(
        getattr(t, "relation", None) == BUILTIN_RELATION_SEMANTICALLY_SIMILAR_TO
        for t in triples
    )

  def upsert_edge(
      self,
      source_id: str,
      target_id: str,
      weight: float = 1.0,
      metadata: dict[str, Any] | None = None,
      triple: TripleT | None = None,
      edge_id: str | None = None,
  ) -> Edge:
    """Adds a new edge or consolidates weight and triples on an existing edge."""
    self._di_graph = None
    key = (source_id, target_id)
    if key in self._edges_by_key:
      edge = self._edges_by_key[key]
      edge.weight += weight
      if triple is not None:
        triples = edge.metadata.setdefault("triples", [])
        if isinstance(triples, list) and triple not in triples:
          triples.append(triple)
      return edge

    meta = dict(metadata) if metadata is not None else {}
    if triple is not None:
      triples = meta.setdefault("triples", [])
      if isinstance(triples, list):
        if triple not in triples:
          triples.append(triple)
      else:
        meta["triples"] = [triple]

    edge = Edge(
        id=edge_id if edge_id is not None else f"e_{source_id}_{target_id}",
        source_node_id=source_id,
        target_node_id=target_id,
        weight=weight,
        metadata=meta,
    )
    self._edges_by_key[(edge.source_node_id, edge.target_node_id)] = edge
    return edge

  def get_entity_name(self, entity_id: str) -> str:
    """Returns the entity name for the given entity ID."""
    return self._entity_name_by_id[entity_id]

  def to_networkx(self) -> nx.DiGraph:
    """Converts the KnowledgeGraph to a NetworkX Directed Graph (DiGraph).

    Each node in `self.nodes` is added as a NetworkX node using `node.id`, with
    attributes `node=node` and all properties from `node.metadata`.
    Each edge in `self.edges` is added as a NetworkX directed edge using
    `(source_node_id, target_node_id)`, with attributes
    `weight=float(edge.weight)`,
    `edge=edge`, and all properties from `edge.metadata`.

    Returns:
      A `networkx.DiGraph` representing the underlying knowledge graph.
    """
    if self._di_graph is not None:
      return self._di_graph
    g = nx.DiGraph()

    for node in self.nodes:
      g.add_node(node.id, node=node, **node.metadata)
    for edge in self.edges:
      g.add_edge(
          edge.source_node_id,
          edge.target_node_id,
          weight=float(edge.weight),
          edge=edge,
          **edge.metadata,
      )
    self._di_graph = g
    return g


class EntityAndTripleExtractorProtocol(Protocol[DocT, EntityT, TripleT]):
  """Protocol for entity and triple extractors.

  Given a document(DocT), the entity and triple extractor should return a list
  of entities(EntityT), and a list of triples(TripleT).
  """

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @abc.abstractmethod
  async def __call__(
      self,
      doc: DocT,
  ) -> Tuple[Sequence[EntityT], Sequence[TripleT]]:
    """Extracts entities and triples from a document."""


@dataclasses.dataclass(kw_only=True)
class GraphRAGEntityAndTriplesExtractor(
    EntityAndTripleExtractorProtocol[Document, Entity, Triple]
):
  """Entity and triples extractor for GraphRAG with referential integrity and deduplication.

  Extracts `Entity`s and `Triple`s from `Document`s.

  Attributes:
    extraction_prompt: The system prompt or instructions defining the guidelines
      for extracting entities and relational triples. Can be a plain string or a
      pre-structured `Composable` object (e.g., tagged with a specific
      conversational role).
      Note: Any domain-specific configuration (such as target entity types or
        ontology definitions) should be pre-formatted into this prompt before
        initializing the extractor. Do not include document placeholders (such
        as `{input_text}`) in this prompt, as the document text is combined at
        execution time via `extraction_prompt_template`.
    document_formatter: A `DocumentFormatter` instance used to convert the input
      `Document` into its string or `Composable` content representation prior to
      prompt assembly.
    generate_backend_config: Configuration for the generation LLM backend used
      during entity and triple extraction.
    runtime_kwargs: Environment-specific deployment settings (e.g.,
      `threadpool_size`, `project`) and credentials (`api_key`, `credentials`)
      that are used when instantiating `generator_backend`.
  """

  extraction_prompt: str | Composable = (
      """Extract entities and triples from the document."""
  )
  document_formatter: document_formatting.DocumentFormatter = dataclasses.field(
      default_factory=document_formatting.TextDocumentFormatter
  )
  generate_backend_config: backend_registry.GenerateBackendConfig | None = None
  runtime_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  _generator_backend: backends_base.Backend | None = dataclasses.field(
      default=None, repr=False, compare=False
  )

  def extraction_prompt_template(
      self, extraction_prompt: Composable, content: Composable
  ) -> Composable:
    """Returns the final prompt assembled from the instructions and document.

    By default, concatenates them with a section delimiter. Override this method
    in a subclass to customize prompt structuring, such as separating
    instructions and document into distinct `system` and `user` chat turns or
    injecting few-shot examples without dealing with string placeholders.

    Args:
      extraction_prompt: The extraction instructions (`self.extraction_prompt`).
      content: The content of the formatted document.
    """
    return (
        extraction_prompt  # pyrefly: ignore[bad-return]
        + composables.c("\n\nDocument:\n")  # pyrefly: ignore[bad-argument-type]
        + content
    )

  @property
  def generator_backend(self) -> backends_base.Backend | None:
    """Lazily instantiates the generator backend from config."""
    if (
        self._generator_backend is None
        and self.generate_backend_config is not None
    ):
      self._generator_backend = backend_registry.get_backend(
          self.generate_backend_config, **self.runtime_kwargs
      )
    return self._generator_backend

  @generator_backend.setter
  def generator_backend(self, value: backends_base.Backend | None) -> None:
    self._generator_backend = value

  @contextlib.contextmanager
  def _backend_routing_context(self) -> Iterator[None]:
    """Yields a context with the generator backend registered if one is set."""
    if self.generator_backend is not None:
      with routing.RegistryContext():
        self.generator_backend.register()
        yield
    else:
      yield

  @executing.make_executable(copy_self=False)
  async def __call__(  # pyrefly: ignore[bad-override]
      self,
      doc: Document,
  ) -> tuple[
      Sequence[Entity],
      Sequence[Triple],
  ]:
    """Returns deduplicated entities and triples from document."""

    formatted_doc = await self.document_formatter(doc)
    doc_content = formatted_doc.content

    with self._backend_routing_context():
      try:
        result: TripleExtractionResult = await llm.generate_object(  # pyrefly: ignore[bad-assignment]
            self.extraction_prompt_template(
                composables.c(self.extraction_prompt),  # pyrefly: ignore[bad-argument-type]
                composables.c(doc_content),  # pyrefly: ignore[bad-argument-type]
            ),
            TripleExtractionResult,
        )
      except (pydantic.ValidationError, ValueError) as e:
        logging.warning(
            "Failed to generate TripleExtractionResult from document: %s",
            doc.doc_id,
            exc_info=e,
        )
        result = TripleExtractionResult()

      result.sanitize_entities()

      result.sanitize_triples()

      return (result.entities, result.triples)
