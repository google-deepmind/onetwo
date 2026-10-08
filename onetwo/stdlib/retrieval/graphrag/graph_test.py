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

"""Tests for graph.py and GraphRAGEntityAndTriplesExtractor."""

from absl.testing import absltest
from absl.testing import parameterized
import networkx as nx
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.builtins import composables
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph
from onetwo.stdlib.retrieval.graphrag import graphrag_prompts

DOCS = [
    retrieval_data_structures.Document(
        doc_id="1",
        title="Doc 1",
        content=(
            "Steve Jobs was CEO of Apple. Steve Jobs was born in San Francisco."
        ),
    ),
    retrieval_data_structures.Document(
        doc_id="2",
        title="Doc 2",
        content="No entities or relationships exist here.",
    ),
]

DEFAULT_REPLY = "{'entities': [], 'triples': []}"

MOCK_NORMAL_REPLY = """{
    'entities': [
        {
            'name': 'Steve Jobs',
            'type': 'person',
            'description': 'CEO of Apple.',
        },
        {
            'name': 'Apple',
            'type': 'organization',
            'description': 'Tech company.',
        },
        {
            'name': 'San Francisco',
            'type': 'location',
            'description': 'City where Steve Jobs was born.',
        },
    ],
    'triples': [
        {
            'source': 'Steve Jobs',
            'relation': 'CEO of',
            'target': 'Apple',
        },
        {
            'source': 'Steve Jobs',
            'relation': 'Born in',
            'target': 'San Francisco',
        },
        {
            'source': 'Steve Jobs',
            'relation': 'works with',
            'target': 'Unknown Person',
        },
    ],
}"""

MOCK_EMPTY_REPLY = """{'entities': [], 'triples': []}"""


def register_mock_llm_backend() -> None:
  """Registers a mock LLM backend returning TripleExtractionResult dictionaries."""
  llm_backend = backends_test_utils.LLMForTest(
      reply_by_prompt_regex={
          r"(?s).*Steve Jobs was CEO of Apple. Steve Jobs was born in San Francisco.*": (
              MOCK_NORMAL_REPLY
          ),
          r"(?s).*No entities or relationships exist here.*": MOCK_EMPTY_REPLY,
      },
      reply_by_prompt_bytes_regex={
          b"(?s).*Steve Jobs was CEO of Apple. Steve Jobs was born in San Francisco.*": (
              MOCK_NORMAL_REPLY
          ),
          b"(?s).*No entities or relationships exist here.*": MOCK_EMPTY_REPLY,
      },
      default_reply=DEFAULT_REPLY,
  )
  llm_backend.register()


class TripleExtractionResultTest(parameterized.TestCase):
  """Tests for TripleExtractionResult sanitization methods."""

  @parameterized.named_parameters(
      dict(
          testcase_name="sanitizes_names_and_types_and_removes_empty",
          entities=[
              graph.Entity(
                  name=" Steve Jobs ",
                  type=" person ",
                  description="CEO of Apple.",
              ),
              graph.Entity(
                  name="Apple",
                  type=" organization ",
                  description="Tech company.",
              ),
              graph.Entity(name="EmptyType", type="", description="No type."),
              graph.Entity(name="", type="PERSON", description="No name."),
          ],
          expected_entities=[
              graph.Entity(
                  name="steve jobs",
                  type="person",
                  description="CEO of Apple.",
              ),
              graph.Entity(
                  name="apple",
                  type="organization",
                  description="Tech company.",
              ),
          ],
      ),
  )
  def test_sanitize_entities(self, entities, expected_entities):
    """Tests sanitization of entity canonical names and ontology types."""
    result = graph.TripleExtractionResult(entities=entities)
    result.sanitize_entities()
    self.assertEqual(result.entities, expected_entities)

  @parameterized.named_parameters(
      dict(
          testcase_name="sanitizes_endpoints_and_enforces_referential_safety",
          entities=[
              graph.Entity(name="steve jobs", type="person"),
              graph.Entity(name="apple", type="organization"),
              graph.Entity(name="san francisco", type="location"),
          ],
          triples=[
              graph.Triple(
                  source=" Steve Jobs ",
                  relation=" CEO of ",
                  target=" Apple ",
              ),
              graph.Triple(
                  source="Steve Jobs",
                  relation="works with",
                  target="Unknown Person 1",
              ),
              graph.Triple(
                  source="Unknown Person 2",
                  relation="Born in",
                  target="San Francisco",
              ),
              graph.Triple(source="Apple", relation="", target="Apple"),
          ],
          expected_triples=[
              graph.Triple(
                  source="steve jobs",
                  relation="CEO of",
                  target="apple",
              ),
          ],
      ),
  )
  def test_sanitize_triples(self, entities, triples, expected_triples):
    """Tests triple sanitization (preserving relation case) and referential safety."""
    result = graph.TripleExtractionResult(entities=entities, triples=triples)
    result.sanitize_triples()
    self.assertEqual(result.triples, expected_triples)


class GraphRAGEntityAndTriplesExtractorTest(parameterized.TestCase):
  """Tests for GraphRAGEntityAndTriplesExtractor executable class."""

  def setUp(self):
    super().setUp()
    register_mock_llm_backend()

  @parameterized.named_parameters(
      dict(
          testcase_name="normal_extraction_with_referential_integrity",
          doc=DOCS[0],
          expected_entity_names={
              "steve jobs",
              "apple",
              "san francisco",
          },
          expected_entity_types={"person", "organization", "location"},
          expected_triples={
              ("steve jobs", "CEO of", "apple"),
              ("steve jobs", "Born in", "san francisco"),
          },
      ),
      dict(
          testcase_name="empty_extraction",
          doc=DOCS[1],
          expected_entity_names=set(),
          expected_entity_types=set(),
          expected_triples=set(),
      ),
  )
  def test_extractor(
      self, doc, expected_entity_names, expected_entity_types, expected_triples
  ):
    """Tests extraction, endpoint lowercasing with relation case preserved, and referential safety."""
    extractor = graph.GraphRAGEntityAndTriplesExtractor()
    (entities, triples), trace = ot.run(extractor(doc), enable_tracing=True)

    with self.subTest("entity_names_should_be_correct"):
      self.assertEqual(set(e.name for e in entities), expected_entity_names)
    with self.subTest("entity_types_should_be_correct"):
      self.assertEqual(set(e.type for e in entities), expected_entity_types)
    with self.subTest("triples_should_be_correct"):
      self.assertEqual(
          set((t.source, t.relation, t.target) for t in triples),
          expected_triples,
      )
    with self.subTest("no_unexpected_default_replies"):
      for leaf in trace.get_leaf_results():
        if "output" in leaf.outputs and isinstance(leaf.outputs["output"], str):
          self.assertNotIn(DEFAULT_REPLY, leaf.outputs["output"])

  def test_custom_document_formatter(self):
    """Tests extraction when using a custom document formatter."""
    custom_formatter = document_formatting.TextDocumentFormatter(
        format_str="Title: {title}\nContent: {text}"
    )
    extractor = graph.GraphRAGEntityAndTriplesExtractor(
        document_formatter=custom_formatter
    )
    entities, triples = ot.run(extractor(DOCS[0]))
    self.assertLen(entities, 3)
    self.assertLen(triples, 2)

  def test_custom_extraction_prompt_template(self):
    """Tests extraction when customizing extraction_prompt_template and using Composable."""
    custom_template = lambda prompt, doc: (
        composables.c(prompt, role="system")  # pyrefly: ignore[unexpected-keyword, unsupported-operation]
        + composables.c("\n\nContext Document:\n")  # pyrefly: ignore[bad-argument-type, unsupported-operation]
        + composables.c(doc, role="user")  # pyrefly: ignore[unexpected-keyword]
    )
    extractor = graph.GraphRAGEntityAndTriplesExtractor(
        extraction_prompt="Extract entities and triples from the document.",
        extraction_prompt_template=custom_template,
    )
    entities, triples = ot.run(extractor(DOCS[0]))
    self.assertLen(entities, 3)
    self.assertLen(triples, 2)


class NodeAndEdgeDocumentConversionTest(parameterized.TestCase):
  """Tests for to_document and from_document on EntityNode, PassageNode, and Edge."""

  def test_entity_node_conversion(self):
    """Tests exact name embedding formatting and round-trip for EntityNode."""
    entity = graph.Entity(
        name="steve jobs",
        type="person",
        description="Co-founder and CEO of Apple Inc.",
    )
    node = graph.EntityNode(
        id="ent_1",
        entity=entity,
        metadata={"source_doc": "doc_42"},
    )
    doc = node.to_document()
    self.assertEqual(doc.doc_id, "ent_1")
    self.assertEqual(doc.title, "steve jobs")
    self.assertEqual(doc.content, "Co-founder and CEO of Apple Inc.")
    self.assertEqual(doc.metadata["type"], "entity")
    self.assertEqual(doc.metadata["ontology_type"], "person")
    self.assertNotIn("description", doc.metadata)
    self.assertEqual(doc.metadata["source_doc"], "doc_42")

    reconstructed = graph.EntityNode.from_document(doc)
    self.assertEqual(reconstructed.id, "ent_1")
    self.assertEqual(reconstructed.entity.name, "steve jobs")
    self.assertEqual(reconstructed.entity.type, "person")
    self.assertEqual(
        reconstructed.entity.description, "Co-founder and CEO of Apple Inc."
    )
    self.assertEqual(reconstructed.metadata, {"source_doc": "doc_42"})

  def test_edge_conversion(self):
    """Tests triple conversion formatting and round-trip."""
    edge = graph.Edge(
        id="edge_1",
        source_node_id="node_10",
        target_node_id="node_20",
        weight=0.95,
        metadata={
            "triples": ["steve jobs CEO of apple"],
        },
    )
    doc = edge.to_document()

    self.assertEqual(doc.doc_id, "edge_1")
    self.assertEqual(doc.title, "")
    self.assertEqual(doc.content, "steve jobs CEO of apple")
    self.assertEqual(doc.metadata["type"], "triple")
    self.assertEqual(doc.metadata["source_node_id"], "node_10")
    self.assertEqual(doc.metadata["target_node_id"], "node_20")
    self.assertAlmostEqual(doc.metadata["weight"], 0.95)
    self.assertNotIn("triples", doc.metadata)

    reconstructed = graph.Edge.from_document(doc)
    self.assertEqual(reconstructed.id, "edge_1")
    self.assertEqual(reconstructed.source_node_id, "node_10")
    self.assertEqual(reconstructed.target_node_id, "node_20")
    self.assertAlmostEqual(reconstructed.weight, 0.95)
    self.assertEqual(
        reconstructed.metadata,
        {
            "triples": ["steve jobs CEO of apple"],
        },
    )

  def test_edge_conversion_with_triple_object(self):
    """Tests edge conversion when metadata['triples'] holds a Triple object."""
    triple_obj = graph.Triple(
        source="steve jobs", relation="CEO of", target="apple"
    )
    edge = graph.Edge(
        id="edge_2",
        source_node_id="node_10",
        target_node_id="node_20",
        metadata={"triples": [triple_obj]},
    )
    doc = edge.to_document()
    self.assertEqual(doc.content, "steve jobs CEO of apple")

  def test_edge_conversion_with_multiple_triples(self):
    """Tests edge conversion when metadata['triples'] holds multiple Triple objects."""
    t1 = graph.Triple(source="steve jobs", relation="CEO of", target="apple")
    t2 = graph.Triple(
        source="steve jobs", relation="founder of", target="apple"
    )
    edge = graph.Edge(
        id="edge_3",
        source_node_id="node_10",
        target_node_id="node_20",
        weight=2.0,
        metadata={"triples": [t1, t2]},
    )
    doc = edge.to_document()
    self.assertEqual(
        doc.content, "steve jobs CEO of apple\nsteve jobs founder of apple"
    )

  def test_edge_conversion_with_dict_triple(self):
    """Tests edge conversion when metadata['triples'] holds deserialized dict objects."""
    dict_triple = {
        "source": "steve jobs",
        "relation": "CEO of",
        "target": "apple",
    }
    edge = graph.Edge(
        id="edge_4",
        source_node_id="node_10",
        target_node_id="node_20",
        metadata={"triples": [dict_triple]},
    )
    doc = edge.to_document()
    self.assertEqual(doc.content, "steve jobs CEO of apple")


class KnowledgeGraphTest(parameterized.TestCase):
  """Tests for KnowledgeGraph helper and lookup methods."""

  def test_upsert_node_idempotency(self):
    """Tests that adding an existing node returns the original instance and avoids duplicate entries."""
    kg = graph.KnowledgeGraph()
    ent = graph.Entity(name="apple", type="organization")
    node1 = graph.EntityNode(id="apple", entity=ent)
    node2 = graph.EntityNode(id="apple", entity=ent)

    added1 = kg.upsert_node(node1)
    added2 = kg.upsert_node(node2)

    self.assertIs(added1, added2)
    self.assertLen(kg.nodes, 1)

  def test_upsert_edge_consolidation(self):
    """Tests that upsert_edge increments weight and consolidates triples for existing edges."""
    kg = graph.KnowledgeGraph()
    t1 = graph.Triple(source="steve jobs", relation="CEO of", target="apple")
    t2 = graph.Triple(
        source="steve jobs", relation="founder of", target="apple"
    )

    edge1 = kg.upsert_edge(
        source_id="steve jobs",
        target_id="apple",
        weight=1.0,
        triple=t1,
    )
    self.assertEqual(edge1.weight, 1.0)
    self.assertEqual(edge1.metadata["triples"], [t1])

    edge2 = kg.upsert_edge(
        source_id="steve jobs",
        target_id="apple",
        weight=0.5,
        triple=t2,
    )
    self.assertIs(edge1, edge2)
    self.assertAlmostEqual(edge2.weight, 1.5)
    self.assertEqual(edge2.metadata["triples"], [t1, t2])

  def test_upsert_node_consolidation(self):
    """Tests that upsert_node concatenates descriptions and merges metadata."""
    kg = graph.KnowledgeGraph()
    ent1 = graph.Entity(
        name="apple", type="organization", description="Released iPhone."
    )
    node1 = graph.EntityNode(
        id="entity_apple_organization", entity=ent1, metadata={"a": 1}
    )
    kg.upsert_node(node1)

    ent2 = graph.Entity(
        name="apple", type="organization", description="Acquired NeXT."
    )
    node2 = graph.EntityNode(
        id="entity_apple_organization", entity=ent2, metadata={"b": 2}
    )
    merged = kg.upsert_node(node2, merge_description=True)

    self.assertIs(merged, node1)
    self.assertLen(kg.nodes, 1)
    self.assertEqual(
        merged.entity.description, "Released iPhone.\nAcquired NeXT."
    )
    self.assertEqual(merged.metadata, {"a": 1, "b": 2})

  def test_upsert_node_consolidation_respects_max_description_length(self):
    """Tests that upsert_node respects max_description_length."""
    kg = graph.KnowledgeGraph(max_description_length=20)
    ent1 = graph.Entity(
        name="apple",
        type="organization",
        description="A" * 25,
    )
    node1 = graph.EntityNode(id="entity_apple", entity=ent1)
    kg.upsert_node(node1)

    ent2 = graph.Entity(
        name="apple",
        type="organization",
        description="B" * 10,
    )
    node2 = graph.EntityNode(id="entity_apple", entity=ent2)
    merged = kg.upsert_node(node2, merge_description=True)

    self.assertIsInstance(merged, graph.EntityNode)
    self.assertEqual(merged.entity.description, "A" * 25)

  def test_get_entity_name_success(self):
    """Tests that get_entity_name returns the entity name for an existing EntityNode."""
    kg = graph.KnowledgeGraph()
    ent = graph.Entity(name="Apple", type="company")
    node = graph.EntityNode(id="entity_apple_company", entity=ent)
    kg.upsert_node(node)

    self.assertEqual(kg.get_entity_name("entity_apple_company"), "Apple")

  def test_get_entity_name_not_found_raises_key_error(self):
    """Tests that get_entity_name raises KeyError for a non-existent entity ID."""
    kg = graph.KnowledgeGraph()
    with self.assertRaises(KeyError):
      kg.get_entity_name("nonexistent_id")

  def test_get_entity_name_non_entity_node_raises_key_error(self):
    """Tests that get_entity_name raises KeyError when called on a non-EntityNode ID."""
    kg = graph.KnowledgeGraph()
    passage_node = graph.PassageNode(id="passage_1", doc_id=0)
    kg.upsert_node(passage_node)

    with self.assertRaises(KeyError):
      kg.get_entity_name("passage_1")

  def test_to_networkx_conversion(self):
    """Tests that to_networkx returns a networkx.DiGraph preserving nodes and edges."""
    kg = graph.KnowledgeGraph()
    ent = graph.Entity(name="Apple", type="company")
    node1 = graph.EntityNode(id="entity_apple_company", entity=ent)
    node2 = graph.PassageNode(id="passage_1", doc_id=0)
    kg.upsert_node(node1)
    kg.upsert_node(node2)

    kg.upsert_edge("passage_1", "entity_apple_company", weight=1.5)

    g = kg.to_networkx()
    self.assertIsInstance(g, nx.DiGraph)
    self.assertLen(g.nodes, 2)
    self.assertLen(g.edges, 1)

    self.assertIn("entity_apple_company", g.nodes)
    self.assertIn("passage_1", g.nodes)

    self.assertEqual(
        g.edges[("passage_1", "entity_apple_company")]["weight"], 1.5
    )

  def test_to_networkx_invalidation_on_upsert_node_and_edge(self):
    """Tests that cached _di_graph is invalidated when nodes or edges are added."""
    kg = graph.KnowledgeGraph()
    node1 = graph.EntityNode(
        id="node_1", entity=graph.Entity(name="Apple", type="company")
    )
    kg.upsert_node(node1)

    g1 = kg.to_networkx()
    self.assertIn("node_1", g1.nodes)
    self.assertLen(g1.nodes, 1)

    # Calling to_networkx again without changes should
    # return the cached instance.
    self.assertIs(kg.to_networkx(), g1)

    # Adding a new node should invalidate the cached graph.
    node2 = graph.PassageNode(id="passage_1", doc_id=0)
    kg.upsert_node(node2)
    g2 = kg.to_networkx()
    self.assertIsNot(g2, g1)
    self.assertIn("passage_1", g2.nodes)
    self.assertLen(g2.nodes, 2)
    self.assertEmpty(g2.edges)

    # Calling to_networkx again without changes should return g2.
    self.assertIs(kg.to_networkx(), g2)

    # Adding an edge should invalidate the cached graph.
    kg.upsert_edge("passage_1", "node_1", weight=2.0)
    g3 = kg.to_networkx()
    self.assertIsNot(g3, g2)
    self.assertLen(g3.edges, 1)
    self.assertEqual(g3.edges[("passage_1", "node_1")]["weight"], 2.0)

  def test_extractor_generator_backend_property_and_setter(self):
    """Tests lazy instantiation and manual setter for generator_backend."""
    extractor = graph.GraphRAGEntityAndTriplesExtractor(
        runtime_kwargs={"api_key": "fake_api_key"}
    )
    self.assertIsNone(extractor.generator_backend)

    config = backend_registry.GenerateBackendConfig(
        backend_class_name="GoogleGenAIAPI", model_name="gemini-2.5-flash"
    )
    extractor.generate_backend_config = config
    backend = extractor.generator_backend
    self.assertIsNotNone(backend)

    custom_backend = backends_test_utils.LLMForTest()
    extractor.generator_backend = custom_backend
    self.assertIs(extractor.generator_backend, custom_backend)

  def test_graphrag_structured_extraction_prompt_formatting(self):
    """Tests formatting of GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT."""
    formatted = graphrag_prompts.GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT.format(
        entity_types=", ".join(graphrag_prompts.DEFAULT_ENTITY_TYPES)
    )
    self.assertIn("organization", formatted)
    self.assertIn("geo", formatted)


if __name__ == "__main__":
  absltest.main()
