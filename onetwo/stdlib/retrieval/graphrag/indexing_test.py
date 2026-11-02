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

"""Tests for indexing.py and BaseKnowledgeGraphIndex."""

import asyncio
import dataclasses
import functools
import os
import types
from typing import Any, Iterable, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing as indexing_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import graphrag_prompts
from onetwo.stdlib.retrieval.graphrag import indexing

Document = retrieval_data_structures.Document


class FakeKnowledgeGraphIndex(
    indexing.BaseKnowledgeGraphIndex[
        str, Document, Document, graph_lib.Entity, graph_lib.Triple
    ]
):
  """Concrete subclass of BaseKnowledgeGraphIndex for testing."""

  @property
  def num_docs(self) -> int:
    return len(self.docs) if self.docs is not None else 0

  def get_docs(self) -> Iterable[Document]:
    return self.docs or []

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[Document]) -> None:
    if self._docs is None:
      self._docs = []
    self._docs.extend(docs)

  def destroy_index(self) -> None:
    self._docs = None
    self._graph = None
    self._entity_index = None
    self._passage_index = None
    self._triples_index = None

  @executing.make_executable(copy_self=False)
  async def retrieve(
      self, query: str, *, max_results: int | None = None, **kwargs
  ) -> Iterable[Document]:
    return []


class BaseKnowledgeGraphIndexTest(parameterized.TestCase):
  """Tests for BaseKnowledgeGraphIndex saving and loading."""

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path

  def test_base_index_save_and_load(self):
    """Tests round-trip saving and loading of BaseKnowledgeGraphIndex state."""
    # 1. Create original index
    index = FakeKnowledgeGraphIndex()
    index._docs = [
        Document(doc_id="doc_1", title="Title 1", content="Content 1")
    ]
    entity = graph_lib.EntityNode(
        id="ent_1",
        entity=graph_lib.Entity(name="apple", type="organization"),
    )
    edge = graph_lib.Edge(
        id="edge_1",
        source_node_id="ent_1",
        target_node_id="ent_2",
        metadata={"triples": ["apple acquired neXT"]},
    )
    index._graph = graph_lib.KnowledgeGraph()
    index._graph.upsert_node(entity)
    index._graph.upsert_edge(
        source_id=edge.source_node_id,
        target_id=edge.target_node_id,
        weight=edge.weight,
        metadata=edge.metadata,
        edge_id=edge.id,
    )

    # Populate _entity_index
    entity_idx = indexing_lib.EmbeddingBasedDocumentIndex()
    entity_idx._docs = [entity.to_document()]
    entity_idx._doc_embeddings = [np.array([1.0, 0.0], dtype=np.float32)]
    index._entity_index = entity_idx

    # Populate _triples_index
    triples_idx = indexing_lib.EmbeddingBasedDocumentIndex()
    triples_idx._docs = [edge.to_document()]
    triples_idx._doc_embeddings = [np.array([0.0, 1.0], dtype=np.float32)]
    index._triples_index = triples_idx

    # 2. Save index to disk
    index.save(self.test_dir)
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, "manifest.json"))
    )
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, "entity_index")))
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, "triples_index"))
    )

    # 3. Load index from disk into a fresh instance
    loaded_index = FakeKnowledgeGraphIndex()
    ot.run(loaded_index.load(self.test_dir))

    # 4. Verify restored state
    self.assertIsNotNone(loaded_index.docs)
    assert loaded_index.docs is not None
    self.assertLen(loaded_index.docs, 1)
    self.assertEqual(loaded_index.docs[0].doc_id, "doc_1")

    self.assertIsNotNone(loaded_index.graph)
    assert loaded_index.graph is not None
    self.assertLen(loaded_index.graph.nodes, 1)
    self.assertLen(loaded_index.graph.edges, 1)
    self.assertEqual(loaded_index.graph.nodes[0].id, "ent_1")
    self.assertEqual(loaded_index.graph.edges[0].id, "edge_1")

    self.assertIsNotNone(loaded_index._entity_index)
    assert loaded_index._entity_index is not None
    self.assertEqual(loaded_index._entity_index.num_docs, 1)
    entity_docs = list(loaded_index._entity_index.get_docs())
    self.assertEqual(entity_docs[0].doc_id, "ent_1")
    self.assertEqual(entity_docs[0].title, "apple")
    self.assertEqual(entity_docs[0].content, "")

    self.assertIsNotNone(loaded_index._triples_index)
    assert loaded_index._triples_index is not None
    self.assertEqual(loaded_index._triples_index.num_docs, 1)
    triple_docs = list(loaded_index._triples_index.get_docs())
    self.assertEqual(triple_docs[0].doc_id, "edge_1")
    self.assertEqual(triple_docs[0].content, "apple acquired neXT")

    self.assertIsNone(loaded_index._passage_index)

  def test_base_index_num_docs_get_docs_and_destroy_index(self):
    """Tests num_docs, get_docs, and destroy_index on BaseKnowledgeGraphIndex."""
    index = FakeKnowledgeGraphIndex()
    self.assertEqual(index.num_docs, 0)
    self.assertEqual(list(index.get_docs()), [])

    doc = Document(doc_id="doc_1", content="Hello world")
    index._docs = [doc]
    index._graph = graph_lib.KnowledgeGraph()
    index._entity_index = indexing_lib.EmbeddingBasedDocumentIndex()

    self.assertEqual(index.num_docs, 1)
    self.assertEqual(list(index.get_docs()), [doc])

    index.destroy_index()
    self.assertIsNone(index.docs)
    self.assertIsNone(index.graph)
    self.assertIsNone(index._entity_index)

  def test_docs_and_graph_are_not_constructor_arguments(self):
    """Tests that `docs` and `graph` are built state, not init fields."""
    for index_class in (
        indexing.BaseKnowledgeGraphIndex,
        indexing.HippoRAGDocumentIndex,
    ):
      with self.subTest(index_class.__name__):
        init_field_names = {
            f.name for f in dataclasses.fields(index_class) if f.init
        }
        self.assertNoCommonElements(
            init_field_names, {"docs", "graph", "_docs", "_graph"}
        )

    built_state_kwargs: dict[str, Any] = {
        "docs": [],
        "graph": graph_lib.KnowledgeGraph(),
    }
    for name, value in built_state_kwargs.items():
      with self.subTest(f"{name}_rejected_by_constructor"):
        with self.assertRaises(TypeError):
          indexing.HippoRAGDocumentIndex(**{name: value})

  def test_docs_and_graph_are_read_only(self):
    """Tests that `docs` and `graph` cannot be assigned from outside."""
    index = FakeKnowledgeGraphIndex()
    with self.assertRaises(AttributeError):
      setattr(index, "docs", [])
    with self.assertRaises(AttributeError):
      setattr(index, "graph", graph_lib.KnowledgeGraph())
    self.assertIsNone(index.docs)
    self.assertIsNone(index.graph)

  def test_docs_and_graph_excluded_from_repr(self):
    """Tests that built docs are not dumped into the index repr."""
    index = FakeKnowledgeGraphIndex()
    ot.run(
        index.add_docs(
            [Document(doc_id="doc_1", content="UNIQUE_DOC_TEXT_123")]
        )
    )
    self.assertNotIn("UNIQUE_DOC_TEXT_123", repr(index))

  def test_base_index_has_no_default_extractor(self):
    """Tests that the base class requires subclasses to define an extractor."""
    index = FakeKnowledgeGraphIndex()
    with self.assertRaisesRegex(NotImplementedError, "_make_extractor"):
      _ = index.entity_and_triples_extractor


class HippoRAGExtractorSettingsTest(parameterized.TestCase):
  """Tests for the extractor that HippoRAGDocumentIndex builds from settings."""

  def test_default_extractor_reproduces_previous_defaults(self):
    """Tests the default prompt, entity types and doc formatting."""
    index = indexing.HippoRAGDocumentIndex()

    extractor = index.entity_and_triples_extractor

    self.assertIsInstance(
        extractor, graph_lib.GraphRAGEntityAndTriplesExtractor
    )
    assert isinstance(extractor, graph_lib.GraphRAGEntityAndTriplesExtractor)
    self.assertEqual(
        index.entity_types, tuple(graphrag_prompts.DEFAULT_ENTITY_TYPES)
    )
    self.assertEqual(
        extractor.extraction_prompt,
        graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT.format(
            entity_types=", ".join(graphrag_prompts.DEFAULT_ENTITY_TYPES)
        ),
    )
    self.assertEqual(
        extractor.document_formatter,
        graph_lib.GraphRAGEntityAndTriplesExtractor().document_formatter,
    )

  def test_extractor_uses_index_settings(self):
    """Tests that the extractor gets the settings of the index."""
    config = backend_registry.GenerateBackendConfig(
        backend_class_name="GoogleGenAIAPI", model_name="gemini-2.5-flash"
    )
    index = indexing.HippoRAGDocumentIndex(
        generate_backend_config=config,
        extraction_prompt="Types: {entity_types}. Answer in {{json}}.",
        entity_types=("person", "city"),
        runtime_kwargs={"api_key": "key"},
    )

    extractor = index.entity_and_triples_extractor

    assert isinstance(extractor, graph_lib.GraphRAGEntityAndTriplesExtractor)
    self.assertEqual(
        extractor.extraction_prompt, "Types: person, city. Answer in {json}."
    )
    self.assertEqual(extractor.generate_backend_config, config)
    self.assertEqual(extractor.runtime_kwargs, {"api_key": "key"})

  def test_extractor_is_built_once(self):
    index = indexing.HippoRAGDocumentIndex()
    self.assertIs(
        index.entity_and_triples_extractor, index.entity_and_triples_extractor
    )

  def test_extractor_is_not_a_constructor_argument(self):
    init_field_names = {
        f.name
        for f in dataclasses.fields(indexing.HippoRAGDocumentIndex)
        if f.init
    }
    self.assertNoCommonElements(
        init_field_names,
        {"entity_and_triples_extractor", "_entity_and_triples_extractor"},
    )

  def test_no_dataclass_field_is_a_function(self):
    """Tests that the index's fields are settings, not functions."""
    index = indexing.HippoRAGDocumentIndex()
    for field in dataclasses.fields(index):
      with self.subTest(field.name):
        self.assertNotIsInstance(
            getattr(index, field.name),
            (types.FunctionType, types.MethodType, functools.partial),
        )


@dataclasses.dataclass
class FakeExtractor:
  """Fake entity and triples extractor for testing."""

  mapping: dict[str, tuple[list[graph_lib.Entity], list[graph_lib.Triple]]]

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, doc: Document
  ) -> tuple[Sequence[graph_lib.Entity], Sequence[graph_lib.Triple]]:
    return self.mapping.get(doc.doc_id, ([], []))


class FakeHippoRAGIndex(indexing.HippoRAGDocumentIndex):
  """Concrete subclass of HippoRAGDocumentIndex for testing."""

  def __init__(self, *args, extractor: FakeExtractor | None = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._fake_extractor = extractor
    self._entity_index = FakeEntityIndex()
    self._passage_index = FakeEntityIndex()
    self._triples_index = FakeEntityIndex()

  def _make_extractor(self):
    if self._fake_extractor is not None:
      return self._fake_extractor
    return super()._make_extractor()

  @property
  def num_docs(self) -> int:
    return len(self.docs) if self.docs is not None else 0

  def get_docs(self) -> Iterable[Document]:
    return self.docs or []

  def destroy_index(self) -> None:
    self._docs = None
    self._graph = None
    self._entity_index = None
    self._passage_index = None
    self._triples_index = None


class FakeEntityIndex(indexing_lib.EmbeddingBasedDocumentIndex):
  """Fake entity index with deterministic similarity scores for testing."""

  def __init__(
      self,
      scores_mapping: dict[str, list[tuple[str | int, float]]] | None = None,
      embeddings_mapping: dict[str, Sequence[float]] | None = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.scores_mapping = scores_mapping or {}
    self.embeddings_mapping = embeddings_mapping or {}

  @property
  def num_docs(self) -> int:
    return len(self._docs)

  def get_docs(self) -> Iterable[Document]:
    return self._docs

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[Document]) -> None:
    for doc in docs:
      self._docs.append(doc)
      if self.embeddings_mapping:
        self._doc_embeddings.append(
            np.asarray(self.embeddings_mapping[doc.doc_id], dtype=float)
        )
    if self._doc_embeddings:
      await self.searcher.build(np.stack(self._doc_embeddings))

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_indices_and_scores(
      self,
      query: str,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[tuple[int, float]]:
    doc_id_to_idx = {doc.doc_id: i for i, doc in enumerate(self._docs)}
    raw_matches = self.scores_mapping.get(query, [])
    if not raw_matches:
      query_name = query.split("\n")[0].strip().lower()
      for key in self.scores_mapping:
        if query_name in key.lower() or key.lower() in query_name:
          raw_matches = self.scores_mapping[key]
          break

    matches = []
    for target, score in raw_matches:
      if isinstance(target, int):
        matches.append((target, score))
      elif target in doc_id_to_idx:
        matches.append((doc_id_to_idx[target], score))

    if min_score is not None:
      matches = [(idx, score) for idx, score in matches if score >= min_score]
    if max_results is not None:
      matches = matches[:max_results]
    return matches

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: str,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Iterable[tuple[Document, float]]:
    indices_and_scores = await self.retrieve_doc_indices_and_scores(
        query, max_results=max_results, min_score=min_score, **kwargs
    )
    return [(self._docs[idx], score) for idx, score in indices_and_scores]


class HippoRAGIndexTest(parameterized.TestCase):
  """E2E robust tests for HippoRAGIndex Phase 1 graph construction."""

  def setUp(self):
    super().setUp()
    self.doc_1 = Document(
        doc_id="doc_1", title="Apple", content="Apple released iPhone."
    )
    self.doc_2 = Document(
        doc_id="doc_2", title="Google", content="Google released Pixel."
    )
    self.mapping = {
        "doc_1": (
            [
                graph_lib.Entity(name="Apple", type="company"),
                graph_lib.Entity(name="iPhone", type="product"),
            ],
            [
                graph_lib.Triple(
                    source="Apple", relation="released", target="iPhone"
                )
            ],
        ),
        "doc_2": (
            [
                graph_lib.Entity(name="Google", type="company"),
                graph_lib.Entity(name="Pixel", type="product"),
            ],
            [
                graph_lib.Triple(
                    source="Google", relation="released", target="Pixel"
                )
            ],
        ),
    }
    self.extractor = FakeExtractor(mapping=self.mapping)
    self.index = FakeHippoRAGIndex(extractor=self.extractor)
    self.index._entity_index = FakeEntityIndex()

  def test_save_and_load_restores_docs_and_graph_built_by_add_docs(self):
    """Tests that docs and graph built by `add_docs` survive save and load."""
    ot.run(self.index.add_docs([self.doc_1, self.doc_2]))
    # Only the docs and graph are checked here; the embedding sub-indexes are
    # fakes without embeddings.
    self.index._entity_index = None
    self.index._passage_index = None
    self.index._triples_index = None
    base_path = self.create_tempdir().full_path

    self.index.save(base_path)
    loaded_index = indexing.HippoRAGDocumentIndex()
    ot.run(loaded_index.load(base_path))

    assert self.index.docs is not None and loaded_index.docs is not None
    self.assertEqual(
        [d.doc_id for d in loaded_index.docs],
        [d.doc_id for d in self.index.docs],
    )
    assert self.index.graph is not None and loaded_index.graph is not None
    self.assertCountEqual(
        [n.id for n in loaded_index.graph.nodes],
        [n.id for n in self.index.graph.nodes],
    )
    self.assertCountEqual(
        [e.id for e in loaded_index.graph.edges],
        [e.id for e in self.index.graph.edges],
    )

  def test_add_docs_basic_flow_and_topology(self):
    """Tests Phase 1 ingestion: passage nodes, doc-scoped entities, and mention edges."""
    ot.run(self.index.add_docs([self.doc_1, self.doc_2]))

    # 1. Verify corpus storage
    self.assertIsNotNone(self.index.docs)
    assert self.index.docs is not None
    self.assertLen(self.index.docs, 2)
    self.assertEqual(self.index.docs[0].doc_id, "doc_1")
    self.assertEqual(self.index.docs[1].doc_id, "doc_2")

    # 2. Verify graph initialization and node counts
    self.assertIsNotNone(self.index.graph)
    assert self.index.graph is not None
    # 2 passage nodes + 4 canonical entity nodes = 6 total nodes
    self.assertLen(self.index.graph.nodes, 6)
    # 4 entities * 2 mention edges each = 8 mention edges
    # + 2 relational triples = 10 edges
    self.assertLen(self.index.graph.edges, 10)

    # 3. Verify passage nodes
    passage_1 = self.index.graph._nodes_by_id.get("passage_doc_1")
    self.assertIsInstance(passage_1, graph_lib.PassageNode)
    assert isinstance(passage_1, graph_lib.PassageNode)
    self.assertEqual(passage_1.doc_id, 0)
    self.assertEqual(passage_1.metadata["doc_id"], "doc_1")

    passage_2 = self.index.graph._nodes_by_id.get("passage_doc_2")
    self.assertIsInstance(passage_2, graph_lib.PassageNode)
    assert isinstance(passage_2, graph_lib.PassageNode)
    self.assertEqual(passage_2.doc_id, 1)
    self.assertEqual(passage_2.metadata["doc_id"], "doc_2")

    # 4. Verify canonical entity nodes (entity_{name}_{type})
    ent_apple = self.index.graph._nodes_by_id.get("entity_apple_company")
    self.assertIsInstance(ent_apple, graph_lib.EntityNode)
    assert isinstance(ent_apple, graph_lib.EntityNode)
    self.assertEqual(ent_apple.entity.name, "Apple")
    self.assertEqual(ent_apple.entity.type, "company")

    ent_iphone = self.index.graph._nodes_by_id.get("entity_iphone_product")
    self.assertIsNotNone(ent_iphone)

    ent_google = self.index.graph._nodes_by_id.get("entity_google_company")
    self.assertIsNotNone(ent_google)

    ent_pixel = self.index.graph._nodes_by_id.get("entity_pixel_product")
    self.assertIsNotNone(ent_pixel)

    # 5. Verify bidirectional mention edges
    # Forward: passage -> entity (MENTIONS)
    edge_fwd = self.index.graph._edges_by_key.get(
        ("passage_doc_1", "entity_apple_company")
    )
    self.assertIsNotNone(edge_fwd)
    assert edge_fwd is not None
    triples_fwd = edge_fwd.metadata.get("triples", [])
    self.assertEqual(
        triples_fwd[0].relation, graph_lib.BUILTIN_RELATION_MENTIONS
    )
    self.assertEqual(triples_fwd[0].target, "Apple")

    # Reverse: entity -> passage (MENTIONED_IN)
    edge_rev = self.index.graph._edges_by_key.get(
        ("entity_apple_company", "passage_doc_1")
    )
    self.assertIsNotNone(edge_rev)
    assert edge_rev is not None
    triples_rev = edge_rev.metadata.get("triples", [])
    self.assertLen(triples_rev, 1)
    self.assertEqual(triples_rev[0].source, "Apple")
    self.assertEqual(
        triples_rev[0].relation, graph_lib.BUILTIN_RELATION_MENTIONED_IN
    )
    self.assertEqual(triples_rev[0].target, "passage_doc_1")

    # 6. Verify relational triples
    edge_rel = self.index.graph._edges_by_key.get(
        ("entity_apple_company", "entity_iphone_product")
    )
    self.assertIsNotNone(edge_rel)
    assert edge_rel is not None
    triples_rel = edge_rel.metadata.get("triples", [])
    self.assertLen(triples_rel, 1)
    self.assertEqual(triples_rel[0].relation, "released")

  def test_add_docs_intra_doc_duplicate_entities_consolidate(self):
    """Tests that duplicate entity mentions within the same doc normalize to the same node."""
    doc_dup = Document(
        doc_id="doc_dup",
        title="Apple Dup",
        content="Apple and apple and APPLE.",
    )
    mapping = {
        "doc_dup": (
            [
                graph_lib.Entity(name="Apple", type="company"),
                graph_lib.Entity(name="apple", type="company"),
                graph_lib.Entity(name=" APPLE ", type="company"),
            ],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(extractor=extractor)
    index._entity_index = FakeEntityIndex()
    ot.run(index.add_docs([doc_dup]))

    assert index.graph is not None
    self.assertLen(index.graph.nodes, 2)  # 1 passage + 1 consolidated entity
    self.assertLen(
        index.graph.edges, 2
    )  # 1 forward mention + 1 reverse mention
    node = index.graph._nodes_by_id.get("entity_apple_company")
    self.assertIsNotNone(node)

  def test_add_docs_cross_doc_entity_deduplication(
      self,
  ):
    """Tests that entities across documents with same name/type merge descriptions."""
    doc_a = Document(
        doc_id="doc_a", title="A", content="Apple released iPhone."
    )
    doc_b = Document(doc_id="doc_b", title="B", content="Apple acquired NeXT.")
    mapping = {
        "doc_a": (
            [
                graph_lib.Entity(
                    name="Apple", type="company", description="Tech company."
                ),
            ],
            [],
        ),
        "doc_b": (
            [
                graph_lib.Entity(
                    name="Apple", type="company", description="Tech company."
                ),
            ],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(extractor=extractor)
    index._entity_index = FakeEntityIndex()
    ot.run(index.add_docs([doc_a, doc_b]))

    assert index.graph is not None
    self.assertLen(index.graph.nodes, 3)  # 2 passages + 1 merged entity
    ent_node = index.graph._nodes_by_id.get("entity_apple_company")
    self.assertIsInstance(ent_node, graph_lib.EntityNode)
    assert isinstance(ent_node, graph_lib.EntityNode)
    self.assertEqual(ent_node.entity.description, "Tech company.")

  def test_add_docs_missing_doc_id_raises_value_error(self):
    """Tests that objects with doc_id=None trigger validation failure."""
    doc_no_id = Document(content="No ID here.")
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Docs missing doc_id" in str(e)
    ):
      ot.run(self.index.add_docs([doc_no_id]))

  def test_add_docs_empty_and_whitespace_doc_id_raises_value_error(self):
    """Tests that documents with empty or whitespace-only doc_id strings trigger validation failure."""
    doc_empty_id = Document(doc_id="", content="Empty ID.")
    doc_whitespace_id = Document(doc_id="   ", content="Whitespace ID.")

    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Docs missing doc_id" in str(e)
    ):
      ot.run(self.index.add_docs([doc_empty_id]))

    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Docs missing doc_id" in str(e)
    ):
      ot.run(self.index.add_docs([doc_whitespace_id]))

  def test_add_docs_empty_list_no_op(self):
    """Tests that passing an empty document sequence is a safe no-op."""
    ot.run(self.index.add_docs([]))
    self.assertIsNone(self.index.docs)
    self.assertIsNone(self.index.graph)

  def test_overridden_make_extractor_is_used(self):
    """Tests that a subclass overriding `_make_extractor` controls extraction."""
    index = FakeHippoRAGIndex(extractor=self.extractor)
    self.assertIs(index.entity_and_triples_extractor, self.extractor)
    ot.run(index.add_docs([self.doc_1]))
    assert index.graph is not None
    self.assertIsNotNone(index.graph.get_node("entity_apple_company"))

  def test_add_docs_incremental_ingestion_preserves_corpus_indices(self):
    """Tests calling add_docs multiple times incrementally indexes corpus doc_id correctly."""
    ot.run(self.index.add_docs([self.doc_1]))
    assert self.index.graph is not None
    passage_1 = self.index.graph._nodes_by_id.get("passage_doc_1")
    assert isinstance(passage_1, graph_lib.PassageNode)
    self.assertEqual(passage_1.doc_id, 0)

    # Incrementally ingest doc_2
    ot.run(self.index.add_docs([self.doc_2]))
    assert self.index.docs is not None
    self.assertLen(self.index.docs, 2)
    passage_2 = self.index.graph._nodes_by_id.get("passage_doc_2")
    assert isinstance(passage_2, graph_lib.PassageNode)
    self.assertEqual(passage_2.doc_id, 1)
    self.assertLen(self.index.graph.nodes, 6)

  def test_add_docs_entity_index_population_and_1to1_mapping(self):
    """Tests that _entity_index stores exactly one Document per canonical entity (1:1 mapping)."""
    fake_entity_index = FakeEntityIndex()
    self.index._entity_index = fake_entity_index
    ot.run(self.index.add_docs([self.doc_1, self.doc_2]))

    self.assertEqual(fake_entity_index.num_docs, 4)
    indexed_ids = {d.doc_id for d in fake_entity_index.get_docs()}
    self.assertEqual(
        indexed_ids,
        {
            "entity_apple_company",
            "entity_iphone_product",
            "entity_google_company",
            "entity_pixel_product",
        },
    )

  def test_add_docs_cross_doc_entity_deduplication_without_description_merging(
      self,
  ):
    """Tests the behavior of entity deduplication across documents."""
    doc_a = Document(doc_id="doc_a", title="A", content="Tech company.")
    doc_b = Document(doc_id="doc_b", title="B", content="Tech company.")
    mapping = {
        "doc_a": (
            [
                graph_lib.Entity(
                    name="Apple",
                    type="company",
                    description="Tech company.",
                ),
            ],
            [],
        ),
        "doc_b": (
            [
                graph_lib.Entity(
                    name="Apple",
                    type="company",
                    description="Tech company.",
                ),
            ],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(extractor=extractor)
    fake_entity_index = FakeEntityIndex()
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a, doc_b]))

    assert index.graph is not None
    ent_node = index.graph.get_node("entity_apple_company")
    self.assertIsInstance(ent_node, graph_lib.EntityNode)
    assert isinstance(ent_node, graph_lib.EntityNode)
    self.assertEqual(ent_node.entity.description, "Tech company.")
    docs_in_index = {d.doc_id: d.content for d in fake_entity_index.get_docs()}
    self.assertEqual(
        docs_in_index["entity_apple_company"],
        "Tech company.",
    )

  def test_add_docs_semantic_similarity_edges_created_above_threshold(self):
    """Tests that bidirectional SEMANTICALLY_SIMILAR_TO edges are created when similarity >= threshold."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    doc_b = Document(doc_id="doc_b", title="B", content="Apple Inc.")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
        "doc_b": (
            [graph_lib.Entity(name="Apple Inc.", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=False,
    )
    index.similarity_threshold = 0.8
    # Unit vectors with a dot product of exactly 0.9.
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_apple inc._company": [0.9, np.sqrt(1.0 - 0.9**2)],
        }
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a, doc_b]))

    assert index.graph is not None
    similarity_edges = [
        e for e in index.graph.edges if e.metadata.get("type") == "similarity"
    ]
    self.assertLen(similarity_edges, 2)
    for edge in similarity_edges:
      self.assertEqual(edge.weight, 0.9)
      self.assertEqual(edge.metadata["score"], 0.9)

  def test_add_docs_semantic_similarity_edges_filtered_below_threshold(
      self,
  ):
    """Tests that semantic similarity edges are omitted when similarity < threshold."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    doc_b = Document(doc_id="doc_b", title="B", content="Google")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
        "doc_b": (
            [graph_lib.Entity(name="Google", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=False,
    )
    index.similarity_threshold = 0.8
    # Unit vectors with a dot product of 0.75.
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_google_company": [0.75, np.sqrt(1.0 - 0.75**2)],
        }
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a, doc_b]))

    assert index.graph is not None
    similarity_edges = [
        e for e in index.graph.edges if e.metadata.get("type") == "similarity"
    ]
    self.assertEmpty(similarity_edges)

  def test_add_docs_semantic_similarity_edges_no_self_loops(self):
    """Tests that similarity edge update never creates self-loop edges."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=False,
    )
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={"entity_apple_company": [1.0, 0.0]}
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a]))

    assert index.graph is not None
    for edge in index.graph.edges:
      self.assertNotEqual(edge.source_node_id, edge.target_node_id)

  def test_add_docs_semantic_similarity_edges_incremental_ingestion(self):
    """Tests that incrementally adding documents discovers similarity edges against existing entities."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    doc_b = Document(doc_id="doc_b", title="B", content="Apple Inc.")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
        "doc_b": (
            [graph_lib.Entity(name="Apple Inc.", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=False,
    )
    # Unit vectors with a dot product of exactly 0.85.
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_apple inc._company": [0.85, np.sqrt(1.0 - 0.85**2)],
        }
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a]))
    assert index.graph is not None
    self.assertEmpty(
        [e for e in index.graph.edges if e.metadata.get("type") == "similarity"]
    )

    ot.run(index.add_docs([doc_b]))
    similarity_edges = [
        e for e in index.graph.edges if e.metadata.get("type") == "similarity"
    ]
    self.assertLen(similarity_edges, 2)
    self.assertEqual(similarity_edges[0].weight, 0.85)

  def test_add_docs_semantic_similarity_edges_multiple_matches(self):
    """Tests that an entity matching multiple entities above threshold creates all similarity edges."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    doc_b = Document(doc_id="doc_b", title="B", content="Apple Inc.")
    doc_c = Document(doc_id="doc_c", title="C", content="Apple Computer")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
        "doc_b": (
            [graph_lib.Entity(name="Apple Inc.", type="company")],
            [],
        ),
        "doc_c": (
            [graph_lib.Entity(name="Apple Computer", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=False,
    )
    index.similarity_threshold = 0.8
    # Pairwise dot products: A-B = 0.9, A-C = 0.85, B-C ~= 0.99.
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_apple inc._company": [0.9, np.sqrt(1.0 - 0.9**2)],
            "entity_apple computer_company": [0.85, np.sqrt(1.0 - 0.85**2)],
        }
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a, doc_b, doc_c]))

    assert index.graph is not None
    similarity_edges = [
        e for e in index.graph.edges if e.metadata.get("type") == "similarity"
    ]
    # 3 entity pairs (A-B, A-C, B-C) * 2 directions each = 6 similarity edges
    self.assertLen(similarity_edges, 6)

  @parameterized.parameters(1, 2, 256)
  def test_add_similarity_edges_independent_of_query_batch_size(
      self, query_batch_size
  ):
    """Tests that batching queries yields the same edges for any batch size."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    mapping = {
        "doc_a": (
            [
                graph_lib.Entity(name="Apple", type="company"),
                graph_lib.Entity(name="Apple Inc.", type="company"),
                graph_lib.Entity(name="Google", type="company"),
            ],
            [],
        ),
    }
    index = FakeHippoRAGIndex(
        extractor=FakeExtractor(mapping=mapping),
        ignore_similarity_edges=True,
    )
    index._entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_apple inc._company": [0.9, np.sqrt(1.0 - 0.9**2)],
            "entity_google_company": [0.0, 1.0],
        }
    )
    ot.run(index.add_docs([doc_a]))
    assert index.graph is not None
    entity_nodes = [
        node
        for node in index.graph.nodes
        if isinstance(node, graph_lib.EntityNode)
    ]
    # Nodes missing from the entity index are skipped.
    missing_node = graph_lib.EntityNode(
        id="entity_missing_company",
        entity=graph_lib.Entity(name="Missing", type="company"),
    )

    asyncio.run(
        index._add_similarity_edges(
            entity_nodes + [missing_node],
            similarity_threshold=0.8,
            query_batch_size=query_batch_size,
        )
    )

    similarity_edges = {
        (e.source_node_id, e.target_node_id): e.weight
        for e in index.graph.edges
        if e.metadata.get("type") == "similarity"
    }
    self.assertEqual(
        similarity_edges,
        {
            ("entity_apple_company", "entity_apple inc._company"): 0.9,
            ("entity_apple inc._company", "entity_apple_company"): 0.9,
        },
    )

  def test_add_docs_passage_and_triples_index_population(self):
    """Tests that _passage_index is populated and _triples_index is omitted by default."""
    fake_passage_index = FakeEntityIndex()
    fake_triples_index = FakeEntityIndex()
    self.index._passage_index = fake_passage_index
    self.index._triples_index = fake_triples_index

    ot.run(self.index.add_docs([self.doc_1, self.doc_2]))

    # Verify passage index has doc_1 and doc_2
    self.assertEqual(fake_passage_index.num_docs, 2)
    passage_ids = {d.doc_id for d in fake_passage_index.get_docs()}
    self.assertEqual(passage_ids, {"doc_1", "doc_2"})

    # Verify triples index is not populated when ignore_triples=True (default)
    self.assertEqual(fake_triples_index.num_docs, 0)

  def test_add_docs_triples_index_populated_when_ignore_triples_false(self):
    """Tests that _triples_index is populated when ignore_triples=False."""
    index = FakeHippoRAGIndex(
        extractor=self.extractor,
        ignore_triples=False,
    )
    fake_passage_index = FakeEntityIndex()
    fake_triples_index = FakeEntityIndex()
    index._passage_index = fake_passage_index
    index._triples_index = fake_triples_index

    ot.run(index.add_docs([self.doc_1, self.doc_2]))

    self.assertEqual(fake_passage_index.num_docs, 2)
    self.assertEqual(fake_triples_index.num_docs, 10)

  def test_defaults_ignore_triples_and_similarity_edges_are_true(self):
    """Tests that ignore_triples and ignore_similarity_edges default to True."""
    index = FakeHippoRAGIndex(extractor=self.extractor)
    self.assertTrue(index.ignore_triples)
    self.assertTrue(index.ignore_similarity_edges)

  def test_add_docs_ignore_triples_true(self):
    """Tests that ignore_triples=True omits triples from triples index while keeping them in graph."""
    fake_triples_index = FakeEntityIndex()
    index = FakeHippoRAGIndex(
        extractor=self.extractor,
        ignore_triples=True,
    )
    index._triples_index = fake_triples_index

    ot.run(index.add_docs([self.doc_1, self.doc_2]))

    assert index.graph is not None
    # 2 passage nodes + 4 canonical entity nodes = 6 total nodes
    self.assertLen(index.graph.nodes, 6)
    # 4 entities * 2 mention edges each + 2 relational triples
    # = 10 edges in graph
    self.assertLen(index.graph.edges, 10)

    # Verify relational triple edges ARE in graph
    rel_edge = index.graph._edges_by_key.get(
        ("entity_apple_company", "entity_iphone_product")
    )
    self.assertIsNotNone(rel_edge)

    # Verify triples index was not populated
    self.assertEqual(fake_triples_index.num_docs, 0)

  def test_add_docs_ignore_similarity_edges_true(self):
    """Tests that ignore_similarity_edges=True skips similarity edge creation."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple")
    doc_b = Document(doc_id="doc_b", title="B", content="Apple Inc.")
    mapping = {
        "doc_a": (
            [graph_lib.Entity(name="Apple", type="company")],
            [],
        ),
        "doc_b": (
            [graph_lib.Entity(name="Apple Inc.", type="company")],
            [],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_similarity_edges=True,
        similarity_threshold=0.8,
    )
    # Unit vectors with a dot product of exactly 0.9.
    fake_entity_index = FakeEntityIndex(
        embeddings_mapping={
            "entity_apple_company": [1.0, 0.0],
            "entity_apple inc._company": [0.9, np.sqrt(1.0 - 0.9**2)],
        }
    )
    index._entity_index = fake_entity_index

    ot.run(index.add_docs([doc_a, doc_b]))

    assert index.graph is not None
    similarity_edges = [
        e for e in index.graph.edges if e.metadata.get("type") == "similarity"
    ]
    self.assertEmpty(similarity_edges)

  def test_add_docs_ignore_both_triples_and_similarity_edges_true(self):
    """Tests indexing when both ignore_triples and ignore_similarity_edges are True."""
    doc_a = Document(doc_id="doc_a", title="A", content="Apple iPhone")
    mapping = {
        "doc_a": (
            [
                graph_lib.Entity(name="Apple", type="company"),
                graph_lib.Entity(name="iPhone", type="product"),
            ],
            [
                graph_lib.Triple(
                    source="Apple", relation="released", target="iPhone"
                )
            ],
        ),
    }
    extractor = FakeExtractor(mapping=mapping)
    index = FakeHippoRAGIndex(
        extractor=extractor,
        ignore_triples=True,
        ignore_similarity_edges=True,
    )
    fake_triples_index = FakeEntityIndex()
    index._triples_index = fake_triples_index

    ot.run(index.add_docs([doc_a]))

    assert index.graph is not None
    # 1 passage node + 2 entity nodes = 3 nodes
    self.assertLen(index.graph.nodes, 3)
    # 2 entities * 2 mention edges = 4 mention edges +
    # 1 relational triple = 5 edges (no similarity edges)
    self.assertLen(index.graph.edges, 5)
    self.assertEqual(fake_triples_index.num_docs, 0)

  @parameterized.parameters(
      "retrieve", "retrieve_with_scores", "retrieve_doc_indices_and_scores"
  )
  def test_retrieval_methods_raise_not_implemented(self, method_name):
    """Tests that retrieval is not supported, even on a populated index."""
    ot.run(self.index.add_docs([self.doc_1, self.doc_2]))
    retrieval_method = getattr(self.index, method_name)

    with self.assertRaisesRegex(NotImplementedError, "MCP server"):
      ot.run(retrieval_method("Apple"))

  def test_retrieval_methods_raise_not_implemented_on_empty_index(self):
    """Tests that retrieval raises before checking whether the index is empty."""
    index = indexing.HippoRAGDocumentIndex()

    with self.assertRaisesRegex(NotImplementedError, "MCP server"):
      ot.run(index.retrieve("Apple"))


if __name__ == "__main__":
  absltest.main()
