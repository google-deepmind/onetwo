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

r"""GraphRAG MCP server for a HippoRAG knowledge graph index.

Exposes a pre-built `HippoRAGDocumentIndex` (see
`onetwo.stdlib.retrieval.graphrag.indexing`) as a suite of Model Context
Protocol (MCP) tools, so that an agent (e.g. the `agentic_retrieval` strategy in
`onetwo.stdlib.qa`) can explore the graph: semantic entity and passage search,
graph traversal, pathfinding and grounded document retrieval. This is the
recommended way to serve a HippoRAG index, which does not implement `retrieve`
itself.

Tools provided:
  1. search_entities: Semantic vector search over graph entity nodes.
  2. search_passages: Semantic vector search over source document passages.
  3. get_entity_info: Detailed entity info (name, type, description) and
     mentioning passage IDs.
  4. get_entity_neighbors: Retrieves 1-hop connected entities and relationship
     predicates (triples) connecting them.
  5. find_paths_between_entities: Multi-hop relation pathfinding between two
     entities showing the chain of intermediate entities and relationship
     predicates.
  6. get_entity_passages: Source document snippets directly mentioning an
     entity.
  7. get_document: Complete document text, URL, and metadata retrieval by
     doc_id.
"""

import json
from typing import Any, Literal, cast

from absl import logging
from mcp.server.fastmcp import server
import networkx as nx
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.builtins import llm
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import indexing

Document = retrieval_data_structures.Document


def load_index(
    index_path: str,
    index_runtime_kwargs: dict[str, Any] | str | None = None,
) -> indexing.HippoRAGDocumentIndex:
  """Loads a saved HippoRAG index and prepares its sub-indexes for querying.

  Args:
    index_path: Directory of the saved HippoRAG index.
    index_runtime_kwargs: Runtime kwargs (as a dict or JSON-encoded string) for
      the index and its embedding sub-indexes (e.g. API keys or backend-specific
      settings).

  Returns:
    The loaded index.
  """
  if isinstance(index_runtime_kwargs, str):
    runtime_kwargs = (
        json.loads(index_runtime_kwargs) if index_runtime_kwargs else {}
    )
  else:
    runtime_kwargs = index_runtime_kwargs or {}

  index = indexing.HippoRAGDocumentIndex(runtime_kwargs=runtime_kwargs)
  logging.info('Loading GraphRAG index from: %s', index_path)
  ot.run(index.load(index_path))
  logging.info(
      'Loaded graph with %d nodes, %d edges, %s entity index, and %s passage'
      ' index.',
      len(index.graph.nodes) if index.graph else 0,
      len(index.graph.edges) if index.graph else 0,
      'active' if index.entity_index is not None else 'no',
      'active' if index.passage_index is not None else 'no',
  )
  return index


def create_server(
    *,
    index_path: str,
    port: int = 8776,
    transport: str = 'stdio',
    host: str = 'localhost',
    snippet_max_tokens: int = 512,
    index_runtime_kwargs: dict[str, Any] | str | None = None,
    tokenize_backend_config: (
        backend_registry.TokenizeBackendConfig | dict[str, Any] | str | None
    ) = None,
) -> server.FastMCP:
  """Loads a saved HippoRAG index and creates the GraphRAG MCP server for it.

  Args:
    index_path: Directory of the saved HippoRAG index.
    port: Port to listen on for SSE/streamable-http.
    transport: Transport protocol ('stdio', 'sse', 'streamable-http').
    host: Host/interface to bind to (default: localhost).
    snippet_max_tokens: Maximum length of result snippets (-1 = no truncation).
      Measured in tokenizer tokens if `tokenize_backend_config` is set, and in
      whitespace-separated words otherwise.
    index_runtime_kwargs: Runtime kwargs (as a dict or JSON-encoded string)
      passed to the index loader (e.g. API keys or backend-specific settings).
    tokenize_backend_config: Optional tokenizer backend config (as a
      `TokenizeBackendConfig`, dict, or JSON-encoded string) used to truncate
      snippets.

  Returns:
    A configured FastMCP server instance.
  """
  return create_server_for_index(
      load_index(index_path, index_runtime_kwargs),
      port=port,
      transport=transport,
      host=host,
      snippet_max_tokens=snippet_max_tokens,
      tokenize_backend_config=tokenize_backend_config,
  )


def run_server(
    *,
    index_path: str,
    port: int = 8776,
    transport: str = 'stdio',
    host: str = 'localhost',
    snippet_max_tokens: int = 512,
    index_runtime_kwargs: dict[str, Any] | str | None = None,
    tokenize_backend_config: (
        backend_registry.TokenizeBackendConfig | dict[str, Any] | str | None
    ) = None,
) -> None:
  """Loads a saved HippoRAG index and runs the GraphRAG MCP server for it.

  Args:
    index_path: Directory of the saved HippoRAG index.
    port: Port to listen on for SSE/streamable-http.
    transport: Transport protocol ('stdio', 'sse', 'streamable-http').
    host: Host/interface to bind to (default: localhost).
    snippet_max_tokens: Maximum length of result snippets (-1 = no truncation).
      Measured in tokenizer tokens if `tokenize_backend_config` is set, and in
      whitespace-separated words otherwise.
    index_runtime_kwargs: Runtime kwargs (as a dict or JSON-encoded string)
      passed to the index loader (e.g. API keys or backend-specific settings).
    tokenize_backend_config: Optional tokenizer backend config (as a
      `TokenizeBackendConfig`, dict, or JSON-encoded string) used to truncate
      snippets.
  """
  mcp = create_server(
      index_path=index_path,
      port=port,
      transport=transport,
      host=host,
      snippet_max_tokens=snippet_max_tokens,
      index_runtime_kwargs=index_runtime_kwargs,
      tokenize_backend_config=tokenize_backend_config,
  )
  mcp.run(transport=cast(Literal['stdio', 'sse', 'streamable-http'], transport))


def create_server_for_index(
    index: indexing.HippoRAGDocumentIndex,
    *,
    port: int = 8776,
    transport: str = 'stdio',
    host: str = 'localhost',
    snippet_max_tokens: int = 512,
    tokenize_backend_config: (
        backend_registry.TokenizeBackendConfig | dict[str, Any] | str | None
    ) = None,
) -> server.FastMCP:
  """Creates the GraphRAG MCP server for an already loaded HippoRAG index.

  Args:
    index: The HippoRAG index to serve. Its `graph`, `docs`, `entity_index` and
      `passage_index` are used as they are.
    port: Port to listen on for SSE/streamable-http.
    transport: Transport protocol ('stdio', 'sse', 'streamable-http').
    host: Host/interface to bind to (default: localhost).
    snippet_max_tokens: Maximum length of result snippets (-1 = no truncation).
      Measured in tokenizer tokens if `tokenize_backend_config` is set, and in
      whitespace-separated words otherwise.
    tokenize_backend_config: Optional tokenizer backend config (as a
      `TokenizeBackendConfig`, dict, or JSON-encoded string) used to truncate
      snippets.

  Returns:
    A configured FastMCP server instance.
  """
  if transport == 'stdio':
    mcp = server.FastMCP('GraphRAG MCP Server')
  else:
    mcp = server.FastMCP('GraphRAG MCP Server', host=host, port=port)

  if isinstance(tokenize_backend_config, str):
    if tokenize_backend_config:
      tokenize_backend_config = (
          backend_registry.TokenizeBackendConfig.from_json(
              tokenize_backend_config
          )
      )
    else:
      tokenize_backend_config = None
  elif isinstance(tokenize_backend_config, dict):
    if tokenize_backend_config:
      tokenize_backend_config = (
          backend_registry.TokenizeBackendConfig.from_dict(
              tokenize_backend_config
          )
      )
    else:
      tokenize_backend_config = None

  # Register a tokenizer backend for snippet truncation only when needed.
  tokenizer = None
  if snippet_max_tokens >= 0 and tokenize_backend_config is not None:
    tokenizer = backend_registry.get_backend(tokenize_backend_config)
    tokenizer.register()

  graph = index.graph
  entity_index = index.entity_index
  passage_index = index.passage_index

  nx_graph: nx.DiGraph = graph.to_networkx() if graph else nx.DiGraph()

  # Pre-compute entity-only undirected subgraph for fast O(1) pathfinding.
  if graph:
    entity_node_ids = {
        n
        for n in nx_graph.nodes
        if isinstance(graph.get_node(n), graph_lib.EntityNode)
    }
    entity_subgraph_undirected = nx_graph.subgraph(
        entity_node_ids
    ).to_undirected(as_view=True)
  else:
    entity_subgraph_undirected = nx.Graph()

  # Pre-compute document lookup table by doc_id for O(1) passage retrieval.
  docs_by_id: dict[str, Document] = {}
  if index.docs:
    for doc in index.docs:
      docs_by_id[str(doc.doc_id)] = doc

  async def _truncate(text: str) -> str:
    if snippet_max_tokens < 0 or not text:
      return text
    if tokenizer is None:
      words = text.split()
      if len(words) <= snippet_max_tokens:
        return text
      return ' '.join(words[:snippet_max_tokens]) + ' [truncated]'
    token_ids = await llm.tokenize(text)
    if len(token_ids) <= snippet_max_tokens:
      return text
    truncated = str(await llm.detokenize(token_ids[:snippet_max_tokens]))
    if len(truncated) < len(text):
      return truncated + ' [truncated]'
    return text

  # ─── TOOL 1: Search Entities ──────────────────────────────────────────
  @mcp.tool(
      name='search_entities',
      description=(
          'Performs semantic vector search to find entity nodes in the'
          ' knowledge graph. Each entity was indexed using both its canonical'
          ' name (title) and a description of its attributes, activities, and'
          ' role in context (content).\n\nReturns max_results matching entities'
          ' with their entity_id, canonical name, type, description, and'
          ' semantic similarity score.'
      ),
  )
  async def search_entities(query: str, max_results: int = 5) -> str:
    """Semantic vector search over entities in the knowledge graph.

    Args:
      query: Natural language query describing the entity to search for.
      max_results: Number of matching entities to return.

    Returns:
      JSON list of matching entities with id, name, type, description, and
      score.
    """
    if entity_index is None:
      return json.dumps({'error': 'Entity index is not initialized.'})
    try:
      results = await entity_index.retrieve_with_scores(
          query, max_results=max_results
      )
      output = []
      for doc, score in results:
        node = graph.get_node(doc.doc_id) if graph else None
        entity_type = (
            node.entity.type
            if isinstance(node, graph_lib.EntityNode)
            else 'passage'
        )
        output.append({
            'entity_id': doc.doc_id,
            'name': doc.title,
            'type': entity_type,
            'description': doc.content,
            'score': round(float(score), 4),
        })
      return json.dumps(output, indent=2, ensure_ascii=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('search_entities failed for query: %s', query)
      return json.dumps({'error': str(e)})

  # ─── TOOL 2: Search Passages ──────────────────────────────────────────
  @mcp.tool(
      name='search_passages',
      description=(
          'Performs semantic vector search over the original source document'
          ' passages to find specific facts, claims, or context matching a'
          ' natural language query.\n\nReturns top matching passages with'
          ' their doc_id, title, snippet (truncated to max_tokens length, for'
          ' example 512), and similarity score.'
      ),
  )
  async def search_passages(query: str, max_results: int = 5) -> str:
    """Semantic vector search over passage documents in the index.

    Args:
      query: Natural language query to search specific facts or context in
        documents.
      max_results: Number of matching passages to return.

    Returns:
      JSON list of matching passages with doc_id, title, snippet, and score.
    """
    if passage_index is None:
      return json.dumps({
          'error': (
              'Passage index is not initialized or not found in the index'
              ' directory.'
          )
      })
    try:
      results = await passage_index.retrieve_with_scores(
          query, max_results=max_results
      )
      output = []
      for doc, score in results:
        text = doc.content
        snippet = await _truncate(text if isinstance(text, str) else str(text))
        output.append({
            'doc_id': str(doc.doc_id),
            'title': doc.title,
            'snippet': snippet,
            'score': round(float(score), 4),
        })
      return json.dumps(output, indent=2, ensure_ascii=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('search_passages failed for query: %s', query)
      return json.dumps({'error': str(e)})

  # ─── TOOL 3: Get Entity Details ───────────────────────────────────────
  @mcp.tool(
      name='get_entity_info',
      description=(
          'Retrieve detailed information about an entity (canonical name,'
          ' ontological type, full description, doc ids mentioning it) for a'
          ' specific entity by its entity_id.'
      ),
  )
  def get_entity_info(entity_id: str) -> str:
    """Retrieves full details of an entity node.

    Args:
      entity_id: Canonical entity ID (e.g., "entity_samuel namara_person").

    Returns:
      JSON object with entity name, type, description, mentioning_passage_ids,
      and metadata.
    """
    if not graph:
      return json.dumps({'error': 'Knowledge graph is not initialized.'})

    node = graph.get_node(entity_id)
    if not isinstance(node, graph_lib.EntityNode):
      return json.dumps({'error': f"Entity '{entity_id}' not found in graph."})

    mentioning_passage_ids = []
    if entity_id in nx_graph:
      for nbr_id in nx_graph.neighbors(entity_id):
        nbr_node = graph.get_node(nbr_id)
        if isinstance(nbr_node, graph_lib.PassageNode):
          mentioning_passage_ids.append(nbr_id)

    return json.dumps(
        {
            'entity_id': node.id,
            'name': node.entity.name,
            'type': node.entity.type,
            'description': node.entity.description,
            'mentioning_passage_ids': mentioning_passage_ids,
            'metadata': node.metadata,
        },
        indent=2,
        ensure_ascii=False,
    )

  def _extract_relation(edge_data: dict[str, Any]) -> str:
    """Extracts all distinct predicate strings from edge metadata (dict or Triple)."""
    triples = edge_data.get('triples', [])
    if not triples or not isinstance(triples, list):
      return str(edge_data.get('type', 'RELATED_TO'))

    relations = []
    for t in triples:
      rel = ''
      if isinstance(t, dict):
        rel = t.get('relation', '')
      elif hasattr(t, 'relation'):
        rel = getattr(t, 'relation') or ''
      elif isinstance(t, str):
        rel = t
      if rel and rel not in relations:
        relations.append(rel)

    return (
        ' / '.join(relations)
        if relations
        else str(edge_data.get('type', 'RELATED_TO'))
    )

  # ─── TOOL 4: Get Entity Neighbors ─────────────────────────────────────
  @mcp.tool(
      name='get_entity_neighbors',
      description=(
          'Retrieve 1-hop connected entities and the relationship predicates'
          ' connecting them to a given entity_id.'
      ),
  )
  def get_entity_neighbors(
      entity_id: str, direction: str = 'both', limit: int = 25
  ) -> str:
    """Gets adjacent entity nodes and relationship predicates.

    Args:
      entity_id: Canonical entity ID (e.g., "entity_samuel namara_person").
      direction: Direction of edges to inspect ("both", "outgoing", "incoming").
      limit: Maximum number of neighbors to return.

    Returns:
      JSON list of neighbor dicts containing neighbor_id, name, type, relation,
      direction, and weight.
    """
    if not graph or entity_id not in nx_graph:
      return json.dumps({'error': f"Entity '{entity_id}' not found in graph."})

    neighbors = []

    # Outgoing edges (entity_id -> target)
    if direction in ('both', 'outgoing'):
      for _, target_id, edge_data in nx_graph.out_edges(entity_id, data=True):
        target_node = graph.get_node(target_id)
        if isinstance(target_node, graph_lib.EntityNode):
          relation_str = _extract_relation(edge_data)
          neighbors.append({
              'neighbor_id': target_id,
              'name': target_node.entity.name,
              'type': target_node.entity.type,
              'relation': relation_str,
              'direction': 'outgoing',
              'weight': edge_data.get('weight', 1.0),
          })

    # Incoming edges (source -> entity_id)
    if direction in ('both', 'incoming'):
      for src_id, _, edge_data in nx_graph.in_edges(entity_id, data=True):
        src_node = graph.get_node(src_id)
        if isinstance(src_node, graph_lib.EntityNode):
          relation_str = _extract_relation(edge_data)
          neighbors.append({
              'neighbor_id': src_id,
              'name': src_node.entity.name,
              'type': src_node.entity.type,
              'relation': relation_str,
              'direction': 'incoming',
              'weight': edge_data.get('weight', 1.0),
          })

    return json.dumps(neighbors[:limit], indent=2, ensure_ascii=False)

  # ─── TOOL 5: Find Paths Between Entities ──────────────────────────────
  @mcp.tool(
      name='find_paths_between_entities',
      description=(
          'Find multi-hop connection paths between two entities in the'
          ' knowledge graph up to max_depth hops. Returns formatted relational'
          ' chains (e.g. "(Entity A) -[relation]-> (Entity B) -[relation]->'
          ' (Entity C)").'
      ),
  )
  def find_paths_between_entities(
      source_entity_id: str,
      target_entity_id: str,
      max_depth: int = 3,
      max_paths: int = 5,
  ) -> str:
    """Finds paths connecting two entities in the graph.

    Args:
      source_entity_id: Starting entity ID.
      target_entity_id: Target entity ID.
      max_depth: Maximum number of hops to explore.
      max_paths: Maximum number of paths to return.

    Returns:
      JSON object containing a list of formatted paths.
    """
    if not graph:
      return json.dumps({'error': 'Graph is not initialized.'})
    if source_entity_id not in nx_graph:
      return json.dumps(
          {'error': f"Source entity '{source_entity_id}' not found in graph."}
      )
    if target_entity_id not in nx_graph:
      return json.dumps(
          {'error': f"Target entity '{target_entity_id}' not found in graph."}
      )

    try:
      paths = []
      for p in nx.all_simple_paths(
          entity_subgraph_undirected,
          source_entity_id,
          target_entity_id,
          cutoff=max_depth,
      ):
        path_segments = []
        for i in range(len(p) - 1):
          u, v = p[i], p[i + 1]
          u_node = graph.get_node(u)
          v_node = graph.get_node(v)
          edge_data = (
              nx_graph.get_edge_data(u, v) or nx_graph.get_edge_data(v, u) or {}
          )
          rel = _extract_relation(edge_data)
          u_name = (
              u_node.entity.name
              if isinstance(u_node, graph_lib.EntityNode)
              else u
          )
          v_name = (
              v_node.entity.name
              if isinstance(v_node, graph_lib.EntityNode)
              else v
          )
          path_segments.append(f'({u_name}) -[{rel}]-> ({v_name})')
        paths.append(' -> '.join(path_segments))
        if len(paths) >= max_paths:
          break

      return json.dumps({'paths': paths}, indent=2, ensure_ascii=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'find_paths_between_entities failed between %s and %s',
          source_entity_id,
          target_entity_id,
      )
      return json.dumps({'error': str(e)})

  # ─── TOOL 6: Get Passages for Entity ──────────────────────────────────
  @mcp.tool(
      name='get_entity_passages',
      description=(
          'Retrieve source document snippets that directly mention a specific'
          ' entity by its entity_id. Use this to read the local textual'
          ' evidence surrounding an entity.'
      ),
  )
  async def get_entity_passages(entity_id: str, limit: int = 5) -> str:
    """Fetches passage snippets linked to an entity node.

    Args:
      entity_id: Canonical entity ID (e.g., "entity_samuel namara_person").
      limit: Maximum number of passage snippets to return.

    Returns:
      JSON list of passages with doc_id, title, and snippet.
    """
    if not graph or entity_id not in nx_graph:
      return json.dumps({'error': f"Entity '{entity_id}' not found in graph."})

    passages = []
    for nbr_id in nx_graph.neighbors(entity_id):
      node = graph.get_node(nbr_id)
      if isinstance(node, graph_lib.PassageNode):
        raw_doc_id = node.id.removeprefix('passage_')
        doc = docs_by_id.get(raw_doc_id)
        snippet = await _truncate(doc.text if doc else '')
        passages.append({
            'doc_id': raw_doc_id,
            'title': doc.title if doc else '',
            'snippet': snippet,
        })
        if len(passages) >= limit:
          break

    return json.dumps(passages, indent=2, ensure_ascii=False)

  # ─── TOOL 7: Get Full Document ────────────────────────────────────────
  @mcp.tool(
      name='get_document',
      description=(
          'Retrieve the complete text, URL, and metadata of a source document'
          ' by its doc_id. Use this when you need to inspect full details,'
          ' dates, numbers, or verify fine-grained factual claims.'
      ),
  )
  def get_document(doc_id: str) -> str:
    """Retrieves full text of a document for factual verification.

    Args:
      doc_id: Document ID to retrieve (e.g. "0" or "passage_0").

    Returns:
      JSON object containing doc_id, title, full text, and metadata.
    """
    clean_doc_id = str(doc_id).removeprefix('passage_')
    doc = docs_by_id.get(clean_doc_id)
    if doc is None:
      return json.dumps({'error': f"Document '{doc_id}' not found."})

    return json.dumps(
        {
            'doc_id': doc.doc_id,
            'title': doc.title,
            'text': doc.text,
            'url': doc.url,
            'metadata': doc.metadata,
        },
        indent=2,
        ensure_ascii=False,
    )

  return mcp
