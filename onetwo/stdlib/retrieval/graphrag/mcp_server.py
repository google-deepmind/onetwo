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

The tools that return source documents (`search_passages`,
`get_entity_passages` and `get_document`) return unstructured MCP tool results:
for each document, a JSON header (e.g. its doc_id, title and score) followed by
the document's content as MCP content blocks (see
`mcp_tools_utils.content_blocks`). Text documents are a single text block, and
multimodal documents (e.g. the single PDF pages of a `pdf_pages` index) a block
per chunk, so that e.g. PDF pages reach the model as PDFs rather than as text
placeholders (`<application/pdf>`).

Serve an index (a saved index directory or an index object) with `start_mcp`, or
build the server with `create_server` to run it yourself or to test it. See
``mcp_server_main.py`` in this directory for a launch script.
"""

from collections.abc import Mapping
import dataclasses
import json
from typing import Any, Literal

from absl import logging
from mcp import types as mcp_types
from mcp.server.fastmcp import server
import networkx as nx
from onetwo.backends import backend_registry
from onetwo.builtins import llm
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing as indexing_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import indexing
from onetwo.stdlib.tool_use import mcp_tools_utils

Document = retrieval_data_structures.Document


def _error_result(message: str) -> mcp_types.CallToolResult:
  """Returns a tool error as a `{"error": ...}` JSON block.

  Like the tools that return JSON, the error is returned as a regular result
  rather than with `isError=True`.

  Args:
    message: The error message.
  """
  return mcp_types.CallToolResult(
      content=[mcp_tools_utils.json_block({'error': message})]
  )


def unwrap_hipporag_index(
    index: indexing_lib.DocumentIndex,
) -> indexing.HippoRAGDocumentIndex:
  """Returns the HippoRAG index that `index` is or wraps.

  HippoRAG indexes may be wrapped in a `RewritingDocumentIndex`, which chunks
  and formats the documents (e.g. by the `build_index` command-line tool with a
  chunking strategy). This
  unwraps `index` through any `RewritingDocumentIndex`es and checks that the
  innermost index is a `HippoRAGDocumentIndex`, which holds the (rewritten)
  documents that the server serves.

  Args:
    index: The index, e.g. as loaded by `index_building.load_index`.

  Returns:
    `index` if it is a `HippoRAGDocumentIndex`, else the `HippoRAGDocumentIndex`
    that it wraps.

  Raises:
    TypeError: If `index` neither is nor wraps (through
      `RewritingDocumentIndex`es) a `HippoRAGDocumentIndex`.
  """
  inner_index = index
  while isinstance(inner_index, indexing_lib.RewritingDocumentIndex):
    inner_index = inner_index.inner_index
  if not isinstance(inner_index, indexing.HippoRAGDocumentIndex):
    raise TypeError(
        'The GraphRAG MCP server requires a HippoRAGDocumentIndex, but got a'
        f' {type(index).__name__}'
        + (
            f' wrapping a {type(inner_index).__name__}'
            if inner_index is not index
            else ''
        )
        + '.'
    )
  return inner_index


def _load_index(
    index_path: str,
    runtime_kwargs: Mapping[str, Any] | None,
) -> indexing.HippoRAGDocumentIndex:
  """Loads a saved HippoRAG index (see `create_server`).

  The index is loaded with `index_building.load_index`, and unwrapped with
  `unwrap_hipporag_index`.

  Args:
    index_path: Directory of the saved HippoRAG index.
    runtime_kwargs: Runtime kwargs for the index and its embedding sub-indexes
      (e.g. API keys or backend-specific settings).

  Returns:
    The loaded `HippoRAGDocumentIndex`, unwrapped.

  Raises:
    TypeError: If the saved index is not a `HippoRAGDocumentIndex`, possibly
      wrapped in `RewritingDocumentIndex`es.
  """
  logging.info('Loading GraphRAG index from: %s', index_path)
  return unwrap_hipporag_index(
      index_building.load_index(index_path, dict(runtime_kwargs or {}))
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ExtraMcpConfig:
  """The parsed `extra_mcp_config` of `create_server` (see there).

  Private on purpose: the extra settings are experimental, so they are not
  exposed as a public type. Callers pass a plain mapping, which is validated
  and parsed with `from_mapping`.

  Attributes:
    snippet_max_tokens: Maximum length of each snippet, or None to return the
      full text.
    tokenize_backend_config: Tokenizer used to truncate snippets, or None to
      truncate them by whitespace-separated words.
  """

  snippet_max_tokens: int | None = 512
  tokenize_backend_config: backend_registry.TokenizeBackendConfig | None = None

  @classmethod
  def from_mapping(
      cls, extra_mcp_config: Mapping[str, Any] | None
  ) -> '_ExtraMcpConfig':
    """Returns the config with the settings of `extra_mcp_config`.

    Args:
      extra_mcp_config: The `extra_mcp_config` of `create_server`, or None for
        the defaults.

    Raises:
      ValueError: If `extra_mcp_config` has an unsupported key.
    """
    settings = dict(extra_mcp_config or {})
    supported = {field.name for field in dataclasses.fields(cls)}
    unknown = settings.keys() - supported
    if unknown:
      raise ValueError(
          f'Unknown extra_mcp_config keys: {sorted(unknown)}. Supported keys:'
          f' {sorted(supported)}.'
      )
    tokenize_backend_config = settings.get('tokenize_backend_config')
    if isinstance(tokenize_backend_config, Mapping):
      settings['tokenize_backend_config'] = (
          backend_registry.TokenizeBackendConfig.from_dict(
              dict(tokenize_backend_config)
          )
      )
    return cls(**settings)


def create_server(
    index: str | indexing_lib.DocumentIndex,
    *,
    host: str = 'localhost',
    port: int = 8775,
    runtime_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> server.FastMCP:
  """Creates the GraphRAG MCP server for a HippoRAG index.

  To serve it, call its `run` method with a transport, or use `start_mcp`
  instead.

  Args:
    index: The index to serve: the directory of a saved index, or an index
      object. Either must be a `HippoRAGDocumentIndex`, possibly wrapped in
      `RewritingDocumentIndex`es (see `unwrap_hipporag_index`). Its `graph`,
      `docs`, `entity_index` and `passage_index` are used as they are.
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    runtime_kwargs: Runtime kwargs (e.g. API keys or backend-specific settings)
      with which to load the index when ``index`` is a directory, see
      ``index_building.load_index``. Must be None for an index object, which
      already has its own.
    extra_mcp_config: Extra, advanced settings of the MCP tools, or None to use
      the defaults. Unlike the other arguments, these settings are NOT part of
      the stable API. They are still evolving, so keys may be added, renamed or
      removed, and their meanings and defaults may change, in any release and
      without notice. Most users should not need to set them. Every key is
      optional (missing keys use their defaults), and unknown keys raise a
      ``ValueError`` (rather than being ignored) so that typos and removed keys
      fail loudly. Currently supported keys:
      - ``'snippet_max_tokens'`` (``int | None``, default 512): maximum length
        of each text block of a snippet, in tokenizer tokens if
        ``'tokenize_backend_config'`` is set and in whitespace-separated words
        otherwise, or None to return the full text. Media is never truncated,
        nor are ``get_document`` results.
      - ``'tokenize_backend_config'`` (``TokenizeBackendConfig``, its dict, or
        None): tokenizer used to truncate snippets. None (default) truncates
        them by words.

  Returns:
    A configured FastMCP server instance.

  Raises:
    ValueError: If ``extra_mcp_config`` has an unsupported key, or
      ``runtime_kwargs`` is set for an index object.
    TypeError: If the index is not a (possibly wrapped)
      ``HippoRAGDocumentIndex``.
  """
  config = _ExtraMcpConfig.from_mapping(extra_mcp_config)
  if isinstance(index, str):
    hipporag_index = _load_index(index, runtime_kwargs)
  elif runtime_kwargs is not None:
    raise ValueError(
        'runtime_kwargs are only used to load an index from a directory, not'
        f' with an index object ({type(index).__name__}).'
    )
  else:
    hipporag_index = unwrap_hipporag_index(index)
  logging.info(
      'Loaded graph with %d nodes, %d edges, %s entity index, and %s passage'
      ' index.',
      len(hipporag_index.graph.nodes) if hipporag_index.graph else 0,
      len(hipporag_index.graph.edges) if hipporag_index.graph else 0,
      'active' if hipporag_index.entity_index is not None else 'no',
      'active' if hipporag_index.passage_index is not None else 'no',
  )

  # `host` and `port` are only used with the streamable-http transport.
  mcp = server.FastMCP('GraphRAG MCP Server', host=host, port=port)

  # `_truncate` does not truncate with `snippet_max_tokens=-1`.
  snippet_max_tokens = (
      -1 if config.snippet_max_tokens is None else config.snippet_max_tokens
  )
  # Register a tokenizer backend for snippet truncation only when needed.
  tokenizer = None
  if (
      config.snippet_max_tokens is not None
      and config.tokenize_backend_config is not None
  ):
    tokenizer = backend_registry.get_backend(config.tokenize_backend_config)
    tokenizer.register()

  graph = hipporag_index.graph
  entity_index = hipporag_index.entity_index
  passage_index = hipporag_index.passage_index

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
  if hipporag_index.docs:
    for doc in hipporag_index.docs:
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

  async def _document_blocks(
      header: dict[str, Any], doc: Document | None, *, truncate: bool
  ) -> list[mcp_types.ContentBlock]:
    """Returns a document's JSON header block followed by its content blocks.

    Args:
      header: JSON metadata of the document, e.g. its doc_id, title and score.
      doc: The document, or None to return only the header (e.g. for a passage
        node without a document).
      truncate: Whether to truncate each text block to `snippet_max_tokens`
        (see `_truncate`). Media (e.g. PDFs) is always sent whole.
    """
    blocks: list[mcp_types.ContentBlock] = [mcp_tools_utils.json_block(header)]
    if doc is None:
      return blocks
    # Truncation is done here rather than by `content_blocks`, so that
    # snippets are truncated by words when no tokenizer is configured.
    for block in await mcp_tools_utils.content_blocks(
        doc.content, max_tokens=-1, resource_id=str(doc.doc_id)
    ):
      if truncate and isinstance(block, mcp_types.TextContent):
        block = mcp_types.TextContent(
            type='text', text=await _truncate(block.text)
        )
      blocks.append(block)
    return blocks

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
      structured_output=False,
  )
  async def search_passages(
      query: str, max_results: int = 5
  ) -> mcp_types.CallToolResult:
    """Semantic vector search over passage documents in the index.

    Args:
      query: Natural language query to search specific facts or context in
        documents.
      max_results: Number of matching passages to return.

    Returns:
      For each matching passage, a JSON block with its doc_id, title and score,
      followed by its content blocks (text truncated to `snippet_max_tokens`).
    """
    if passage_index is None:
      return _error_result(
          'Passage index is not initialized or not found in the index'
          ' directory.'
      )
    try:
      results = await passage_index.retrieve_with_scores(
          query, max_results=max_results
      )
      content: list[mcp_types.ContentBlock] = []
      for doc, score in results:
        header = {
            'doc_id': str(doc.doc_id),
            'title': doc.title,
            'score': round(float(score), 4),
        }
        content.extend(await _document_blocks(header, doc, truncate=True))
      return mcp_types.CallToolResult(content=content)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('search_passages failed for query: %s', query)
      return _error_result(str(e))

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
      structured_output=False,
  )
  async def get_entity_passages(
      entity_id: str, limit: int = 5
  ) -> mcp_types.CallToolResult:
    """Fetches passage snippets linked to an entity node.

    Args:
      entity_id: Canonical entity ID (e.g., "entity_samuel namara_person").
      limit: Maximum number of passage snippets to return.

    Returns:
      For each passage, a JSON block with its doc_id and title, followed by its
      content blocks (text truncated to `snippet_max_tokens`).
    """
    if not graph or entity_id not in nx_graph:
      return _error_result(f"Entity '{entity_id}' not found in graph.")

    content: list[mcp_types.ContentBlock] = []
    num_passages = 0
    for nbr_id in nx_graph.neighbors(entity_id):
      if num_passages >= limit:
        break
      node = graph.get_node(nbr_id)
      if not isinstance(node, graph_lib.PassageNode):
        continue
      raw_doc_id = node.id.removeprefix('passage_')
      doc = docs_by_id.get(raw_doc_id)
      header = {'doc_id': raw_doc_id, 'title': doc.title if doc else ''}
      content.extend(await _document_blocks(header, doc, truncate=True))
      num_passages += 1

    return mcp_types.CallToolResult(content=content)

  # ─── TOOL 7: Get Full Document ────────────────────────────────────────
  @mcp.tool(
      name='get_document',
      description=(
          'Retrieve the complete text, URL, and metadata of a source document'
          ' by its doc_id. Use this when you need to inspect full details,'
          ' dates, numbers, or verify fine-grained factual claims.'
      ),
      structured_output=False,
  )
  async def get_document(doc_id: str) -> mcp_types.CallToolResult:
    """Retrieves full text of a document for factual verification.

    Args:
      doc_id: Document ID to retrieve (e.g. "0" or "passage_0").

    Returns:
      A JSON block with the document's doc_id, title, url and metadata, followed
      by its untruncated content blocks.
    """
    clean_doc_id = str(doc_id).removeprefix('passage_')
    doc = docs_by_id.get(clean_doc_id)
    if doc is None:
      return _error_result(f"Document '{doc_id}' not found.")

    header = {
        'doc_id': doc.doc_id,
        'title': doc.title,
        'url': doc.url,
        'metadata': doc.metadata,
    }
    return mcp_types.CallToolResult(
        content=await _document_blocks(header, doc, truncate=False)
    )

  return mcp


def start_mcp(
    index: str | indexing_lib.DocumentIndex,
    *,
    transport: Literal['stdio', 'streamable-http'] = 'stdio',
    host: str = 'localhost',
    port: int = 8775,
    runtime_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> None:
  """Serves the GraphRAG MCP server until it is stopped.

  Builds the server with `create_server` (see there for the arguments other
  than `transport`) and runs it. This blocks: e.g. to evaluate a strategy with
  the server (`rag.evaluate` with `mcp_kwargs={'mcp_server_urls': ...}`), start
  it in another process.

  Args:
    index: The index to serve: the directory of a saved HippoRAG index or an
      index object, see `create_server`.
    transport: MCP transport: 'stdio' (e.g. when an MCP client launches the
      server as a subprocess) or 'streamable-http' (serves
      http://<host>:<port>/mcp).
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    runtime_kwargs: See `create_server`.
    extra_mcp_config: EXPERIMENTAL extra settings of the tools, see
      `create_server`.
  """
  mcp = create_server(
      index,
      host=host,
      port=port,
      runtime_kwargs=runtime_kwargs,
      extra_mcp_config=extra_mcp_config,
  )
  mcp.run(transport=transport)
