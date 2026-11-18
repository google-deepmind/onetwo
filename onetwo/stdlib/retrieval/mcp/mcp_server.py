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

r"""OneTwo Index MCP Server library.

Exposes a pre-built OneTwo document index (e.g. an EmbeddingBasedDocumentIndex)
as an MCP tool, allowing AI agents (e.g., Antigravity) to perform agentic
retrieval over an existing index.

This module provides ``start_mcp``, which serves an index until the server is
stopped, and ``create_server``, which only builds the server (e.g. to run it
yourself or to test it).  Extra keyword arguments for the index's backends
(e.g. the embedding backend's credentials or cache) are passed via the
``extra_backend_kwargs`` parameter. Advanced settings of the tools (e.g. the
number of search hits) are
passed via ``extra_mcp_config``, which is experimental and not part of the
stable API.

Typical workflow:
  1. Build an embedding index offline (e.g., via
     ``onetwo.stdlib.retrieval.build_index``).
  2. Serve it with ``start_mcp`` (e.g. ``rag.start_mcp``, or the
     ``mcp_server_main`` launch script).
  3. Agents call the ``search`` tool to retrieve relevant documents.
  4. Agents call the ``get_document`` tool to retrieve a full document by its
     docid.

The tools return unstructured MCP tool results, see
https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-result:
each hit is a JSON header with its docid and score, followed by the document's
content as MCP content blocks. Text documents are a single text block, and
multimodal documents (e.g. single PDF pages, whose content is an
``application/pdf`` chunk) are a block per chunk (text, image, audio, or an
embedded binary resource for anything else, such as PDFs), so that their
non-text content reaches the model as such rather than as text placeholders
(e.g. ``<application/pdf>``).

For example, a ``search`` whose hits are a text document and a single PDF page
returns (base64 data shortened)::

  {
    "content": [
      {"type": "text", "text": "{\"docid\": \"doc_1\", \"score\": 0.8123}"},
      {"type": "text", "text": "The quarterly report shows that ..."},
      {"type": "text", "text": "{\"docid\": \"page_3\", \"score\": 0.7456}"},
      {
        "type": "resource",
        "resource": {
          "uri": "onetwo-index://unused-uri/page_3",
          "mimeType": "application/pdf",
          "blob": "JVBERi0xLjcK..."
        }
      }
    ],
    "isError": false
  }

An image chunk would instead be sent as
``{"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo..."}``.

See ``mcp_server_main.py`` in this directory for a minimal launch script
example.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, Literal

from absl import logging
from mcp import types as mcp_types
from mcp.server.fastmcp import server
from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.tool_use import mcp_tools_utils


def _build_search_description(max_results: int) -> str:
  """Returns the tool description for the search tool."""
  return (
      f'Perform a search on a knowledge source. Returns top-{max_results}'
      ' hits with docid, score, and snippet. The snippet contains the'
      " document's contents (may be truncated based on token limits)."
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ExtraMcpConfig:
  """The parsed `extra_mcp_config` of `create_server` (see there).

  Private on purpose: the extra settings are experimental, so they are not
  exposed as a public type. Callers pass a plain mapping, which is validated
  and parsed with `from_mapping`.

  Attributes:
    max_results: Number of hits returned by the `search` tool.
    snippet_max_tokens: Maximum number of tokens of each text chunk of a hit,
      or None to return the full text.
    tokenize_backend_config: Tokenizer used to truncate the text of hits.
  """

  max_results: int = 5
  snippet_max_tokens: int | None = 512
  tokenize_backend_config: backend_registry.TokenizeBackendConfig = (
      dataclasses.field(
          default_factory=rag_defaults.default_tokenize_backend_config
      )
  )

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
    if tokenize_backend_config is None:
      settings.pop('tokenize_backend_config', None)
    elif isinstance(tokenize_backend_config, Mapping):
      settings['tokenize_backend_config'] = (
          backend_registry.TokenizeBackendConfig.from_dict(
              dict(tokenize_backend_config)
          )
      )
    return cls(**settings)


def create_server(
    index: str | indexing.DocumentIndex,
    *,
    host: str = 'localhost',
    port: int = 8775,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> server.FastMCP:
  """Creates and configures the OneTwo Index MCP server.

  The server has a `search` tool and a `get_document` tool. To serve it, call
  its `run` method with a transport, or use `start_mcp` instead.

  Args:
    index: The index to serve: the directory of a saved index (e.g. one saved
      by ``rag.build_index``), loaded with ``index_building.load_index``, or an
      index object (e.g. one returned by ``rag.build_index``).
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    extra_backend_kwargs: Extra keyword arguments for the constructors of the
      backends that this call creates (e.g. an API key) when ``index`` is a
      directory, see ``index_building.load_index``. Must be None for an index
      object, which already has its backends. Usually not needed: credentials
      are read from the environment (e.g. `GOOGLE_API_KEY`). Not a stable part
      of the API.
    extra_mcp_config: Extra, advanced settings of the MCP tools, or None to use
      the defaults. Unlike the other arguments, these settings are NOT part of
      the stable API. They are still evolving, so keys may be added, renamed or
      removed, and their meanings and defaults may change, in any release and
      without notice. Most users should not need to set them. Every key is
      optional (missing keys use their defaults), and unknown keys raise a
      ``ValueError`` (rather than being ignored) so that typos and removed keys
      fail loudly. Currently supported keys: - ``'max_results'`` (``int``,
      default 5): number of hits returned by ``search``. -
      ``'snippet_max_tokens'`` (``int | None``, default 512): maximum number of
      tokens of each text chunk of a ``search`` hit, or None to return the full
      text. Media is never truncated, nor are ``get_document`` results. -
      ``'tokenize_backend_config'`` (``TokenizeBackendConfig``, its dict, or
      None): tokenizer used to truncate snippets. None (default) uses
      ``rag_defaults.default_tokenize_backend_config()``.

  Returns:
    A configured FastMCP server instance.

  Raises:
    ValueError: If ``extra_mcp_config`` has an unsupported key, or
      ``extra_backend_kwargs`` is set for an index object.
  """
  config = _ExtraMcpConfig.from_mapping(extra_mcp_config)
  if not isinstance(index, str) and extra_backend_kwargs is not None:
    raise ValueError(
        'extra_backend_kwargs are only used to load an index from a directory,'
        ' not'
        f' with an index object ({type(index).__name__}).'
    )

  # `host` and `port` are only used with the streamable-http transport.
  mcp = server.FastMCP('OneTwo Index MCP Server', host=host, port=port)

  if isinstance(index, str):
    logging.info('Loading index from: %s', index)
    document_index = index_building.load_index(
        index, extra_backend_kwargs=extra_backend_kwargs
    )
    logging.info(
        'Index loaded successfully. %d documents in the index.',
        document_index.num_docs,
    )
  else:
    document_index = index
    logging.info(
        'Serving %s with %d documents.',
        type(document_index).__name__,
        document_index.num_docs,
    )

  # Register a tokenizer for snippet truncation only when needed. This
  # configures the global `llm.tokenize` and `llm.detokenize` builtins, which
  # `mcp_tools_utils.content_blocks` uses.
  if config.snippet_max_tokens is not None:
    backend_registry.get_backend(config.tokenize_backend_config).register()
  # `content_blocks` does not truncate with `max_tokens=-1`.
  snippet_max_tokens = (
      -1 if config.snippet_max_tokens is None else config.snippet_max_tokens
  )

  async def _search_content(query: str) -> list[mcp_types.ContentBlock]:
    """Returns the content of the `search` results for `query`."""
    results_with_scores = await document_index.retrieve_with_scores(
        query, max_results=config.max_results
    )
    content: list[mcp_types.ContentBlock] = []
    for doc, score in results_with_scores:
      content.append(
          mcp_tools_utils.json_block(
              {'docid': doc.doc_id, 'score': round(float(score), 4)}
          )
      )
      content.extend(
          await mcp_tools_utils.content_blocks(
              doc.content,
              max_tokens=snippet_max_tokens,
              resource_id=str(doc.doc_id),
          )
      )
    return content

  # ─── MCP Tools ──────────────────────────────────────────────────────
  # The tools declare no output schema (`structured_output=False`): they
  # return MCP content blocks, so that non-text content reaches the model.

  @mcp.tool(
      name='search',
      description=_build_search_description(config.max_results),
      structured_output=False,
  )
  def search(query: str) -> mcp_types.CallToolResult:
    """Search the index and return top-k hits.

    Args:
      query: Search query string.

    Returns:
      For each hit, a JSON text block with its docid and score, followed by the
      document's content blocks (see `mcp_tools_utils.content_blocks`). On
      failure, a single JSON text block with an `error` entry.
    """
    try:
      return mcp_types.CallToolResult(
          content=executing.run(_search_content(query))
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Search failed for query: %s', query)
      # TODO: mark failures with `isError=True` and the message
      # as text, as MCP specifies, rather than an `{"error": ...}` JSON block.
      return mcp_types.CallToolResult(
          content=[mcp_tools_utils.json_block({'error': str(e)})]
      )

  del search  # Registered via decorator; suppress unused variable warning.

  @mcp.tool(
      name='get_document',
      description='Retrieve a full document by its docid.',
      structured_output=False,
  )
  def get_document(docid: str) -> mcp_types.CallToolResult | None:
    """Retrieve a full document by its ID from the search index.

    Args:
      docid: Document ID to retrieve.

    Returns:
      A JSON text block with the document's docid, title, url and metadata,
      followed by its untruncated content blocks (see
      `mcp_tools_utils.content_blocks`), or None if not found.
    """
    try:
      doc = document_index.get_doc(docid)
    except KeyError:
      # TODO: return `isError=True` with an "Unknown docid"
      # message, as for `search` failures, so that the model is told what
      # went wrong rather than getting an empty result.
      return None
    header = mcp_tools_utils.json_block({
        'docid': doc.doc_id,
        'title': doc.title,
        'url': doc.url,
        'metadata': doc.metadata,
    })
    return mcp_types.CallToolResult(
        content=[
            header,
            *executing.run(
                mcp_tools_utils.content_blocks(
                    doc.content, max_tokens=-1, resource_id=str(doc.doc_id)
                )
            ),
        ]
    )

  del get_document  # Registered via decorator.

  return mcp


def start_mcp(
    index: str | indexing.DocumentIndex,
    *,
    transport: Literal['stdio', 'streamable-http'] = 'stdio',
    host: str = 'localhost',
    port: int = 8775,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> None:
  """Serves the OneTwo Index MCP server until it is stopped.

  Builds the server with `create_server` (see there for the arguments other
  than `transport`) and runs it. This blocks: e.g. to evaluate a strategy with
  the server (`rag.evaluate` with `mcp_kwargs={'mcp_server_urls': ...}`), start
  it in another process.

  Args:
    index: The index to serve: the directory of a saved index or an index
      object, see `create_server`.
    transport: MCP transport: 'stdio' (e.g. when an MCP client launches the
      server as a subprocess) or 'streamable-http' (serves
      http://<host>:<port>/mcp).
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    extra_backend_kwargs: See `create_server`.
    extra_mcp_config: EXPERIMENTAL extra settings of the tools, see
      `create_server`.
  """
  mcp = create_server(
      index,
      host=host,
      port=port,
      extra_backend_kwargs=extra_backend_kwargs,
      extra_mcp_config=extra_mcp_config,
  )
  mcp.run(transport=transport)
