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

"""Tests for serialization.py and GraphRAGIndexState subspace serialization."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import indexing as indexing_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import serialization

Document = retrieval_data_structures.Document
EmbedBackendConfig = backend_registry.EmbedBackendConfig
GenerateBackendConfig = backend_registry.GenerateBackendConfig


class SimpleGraphRAGIndexSerializerTest(parameterized.TestCase):
  """Tests for SimpleGraphRAGIndexSerializer with EmbeddingBasedDocumentIndex subspaces."""

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    self.serializer = serialization.SimpleGraphRAGIndexSerializer(
        doc_class=Document
    )

  def test_full_index_state_roundtrip(self):
    """Tests saving and loading all components: docs, graph, and all three subspace indices."""
    # 1. Create sample documents
    docs = [
        Document(doc_id="doc_1", title="Title 1", content="Content of doc 1."),
        Document(doc_id="doc_2", title="Title 2", content="Content of doc 2."),
    ]

    # 2. Create sample graph nodes and edges
    entity_node = graph_lib.EntityNode(
        id="ent_1",
        entity=graph_lib.Entity(
            name="steve jobs", type="person", description="CEO of Apple."
        ),
        metadata={"src": "doc_1"},
    )
    passage_node = graph_lib.PassageNode(
        id="pass_1",
        doc_id=0,
        metadata={"text": "Content of doc 1.", "title": "Title 1"},
    )
    edge = graph_lib.Edge(
        id="edge_1",
        source_node_id="ent_1",
        target_node_id="ent_2",
        weight=0.9,
        metadata={"triples": ["steve jobs CEO of apple"]},
    )
    graph_obj = graph_lib.KnowledgeGraph()
    graph_obj.upsert_node(entity_node)
    graph_obj.upsert_node(passage_node)
    graph_obj.upsert_edge(
        source_id=edge.source_node_id,
        target_id=edge.target_node_id,
        weight=edge.weight,
        metadata=edge.metadata,
        edge_id=edge.id,
    )

    # 3. Create subspace EmbeddingBasedDocumentIndex instances
    # and populate with documents and vectors
    entity_index = indexing_lib.EmbeddingBasedDocumentIndex()
    entity_index._docs = [entity_node.to_document()]
    entity_index._doc_embeddings = [np.array([1.0, 2.0], dtype=np.float32)]

    passage_doc = Document(
        doc_id="pass_1",
        title="Title 1",
        content="Content of doc 1.",
        metadata={"type": "passage", "corpus_doc_id": 0},
    )
    passage_index = indexing_lib.EmbeddingBasedDocumentIndex()
    passage_index._docs = [passage_doc]
    passage_index._doc_embeddings = [np.array([3.0, 4.0], dtype=np.float32)]

    triples_index = indexing_lib.EmbeddingBasedDocumentIndex()
    triples_index._docs = [edge.to_document()]
    triples_index._doc_embeddings = [np.array([5.0, 6.0], dtype=np.float32)]

    # 4. Create backend configs
    embed_cfg = EmbedBackendConfig(
        backend_class_name="MockEmbedder", model_name="mock-embed-v1"
    )
    generate_cfg = GenerateBackendConfig(
        backend_class_name="MockGenerator", model_name="mock-gen-v1"
    )

    state = serialization.GraphRAGIndexState(
        docs=docs,
        graph=graph_obj,
        entity_index=entity_index,
        passage_index=passage_index,
        triples_index=triples_index,
        embed_backend_config=embed_cfg,
        generate_backend_config=generate_cfg,
    )

    # 5. Save index state
    self.serializer.save(state, self.test_dir)

    # Verify directory structure created on disk
    path = epath.Path(self.test_dir)
    self.assertTrue((path / "manifest.json").exists())
    self.assertTrue((path / "docs.jsonl").exists())
    self.assertTrue((path / "graph.jsonl").exists())
    self.assertTrue((path / "entity_index").exists())
    self.assertTrue((path / "passage_index").exists())
    self.assertTrue((path / "triples_index").exists())

    # 6. Load index state
    loaded_state = ot.run(self.serializer.load(self.test_dir))

    # Verify loaded documents
    self.assertIsNotNone(loaded_state.docs)
    self.assertLen(loaded_state.docs, 2)
    self.assertEqual(loaded_state.docs[0].doc_id, "doc_1")
    self.assertEqual(loaded_state.docs[1].doc_id, "doc_2")

    # Verify loaded graph topology
    self.assertIsNotNone(loaded_state.graph)
    assert loaded_state.graph is not None
    self.assertLen(loaded_state.graph.nodes, 2)
    self.assertLen(loaded_state.graph.edges, 1)
    loaded_entity = next(
        n
        for n in loaded_state.graph.nodes
        if isinstance(n, graph_lib.EntityNode)
    )
    self.assertEqual(loaded_entity.id, "ent_1")
    self.assertEqual(loaded_entity.entity.name, "steve jobs")
    loaded_edge = loaded_state.graph.edges[0]
    self.assertEqual(loaded_edge.id, "edge_1")
    self.assertEqual(
        loaded_edge.metadata["triples"], ["steve jobs CEO of apple"]
    )

    # Verify loaded entity_index
    self.assertIsNotNone(loaded_state.entity_index)
    assert loaded_state.entity_index is not None
    self.assertEqual(loaded_state.entity_index.num_docs, 1)
    loaded_entity_docs = list(loaded_state.entity_index.get_docs())
    self.assertEqual(loaded_entity_docs[0].doc_id, "ent_1")
    self.assertEqual(loaded_entity_docs[0].title, "steve jobs")
    self.assertEqual(loaded_entity_docs[0].content, "CEO of Apple.")
    np.testing.assert_array_equal(
        loaded_state.entity_index._doc_embeddings[0],
        np.array([1.0, 2.0], dtype=np.float32),
    )

    # Verify loaded passage_index
    self.assertIsNotNone(loaded_state.passage_index)
    assert loaded_state.passage_index is not None
    self.assertEqual(loaded_state.passage_index.num_docs, 1)
    loaded_passage_docs = list(loaded_state.passage_index.get_docs())
    self.assertEqual(loaded_passage_docs[0].doc_id, "pass_1")

    # Verify loaded triples_index
    self.assertIsNotNone(loaded_state.triples_index)
    assert loaded_state.triples_index is not None
    self.assertEqual(loaded_state.triples_index.num_docs, 1)
    loaded_triple_docs = list(loaded_state.triples_index.get_docs())
    self.assertEqual(loaded_triple_docs[0].doc_id, "edge_1")
    self.assertEqual(loaded_triple_docs[0].content, "steve jobs CEO of apple")

    # Verify loaded configs
    self.assertIsNotNone(loaded_state.embed_backend_config)
    self.assertEqual(
        loaded_state.embed_backend_config.model_name, "mock-embed-v1"
    )
    self.assertIsNotNone(loaded_state.generate_backend_config)
    self.assertEqual(
        loaded_state.generate_backend_config.model_name, "mock-gen-v1"
    )

  def test_partial_subspace_index_roundtrip(self):
    """Tests saving and loading when only entity_index is populated while others are None."""
    entity_node = graph_lib.EntityNode(
        id="ent_1", entity=graph_lib.Entity(name="apple", type="org")
    )
    graph_obj = graph_lib.KnowledgeGraph()
    graph_obj.upsert_node(entity_node)

    entity_index = indexing_lib.EmbeddingBasedDocumentIndex()
    entity_index._docs = [entity_node.to_document()]
    entity_index._doc_embeddings = [np.array([10.0, 20.0], dtype=np.float32)]

    state = serialization.GraphRAGIndexState(
        docs=[],
        graph=graph_obj,
        entity_index=entity_index,
        passage_index=None,
        triples_index=None,
    )

    self.serializer.save(state, self.test_dir)
    loaded_state = ot.run(self.serializer.load(self.test_dir))

    self.assertIsNotNone(loaded_state.entity_index)
    assert loaded_state.entity_index is not None
    self.assertEqual(loaded_state.entity_index.num_docs, 1)
    self.assertIsNone(loaded_state.passage_index)
    self.assertIsNone(loaded_state.triples_index)

  def test_missing_index_files_raises_filenotfounderror(self):
    """Tests loading from an empty directory raises FileNotFoundError."""
    empty_dir = os.path.join(self.test_dir, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    with self.assertRaises(FileNotFoundError):
      ot.run(self.serializer.load(empty_dir))


if __name__ == "__main__":
  absltest.main()
