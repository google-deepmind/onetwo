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

Exposes a pre-built OneTwo EmbeddingBasedDocumentIndex as an MCP tool, allowing
AI agents (e.g., Antigravity) to perform agentic retrieval over an existing
index.

This module provides the reusable ``create_server`` function that can be called
from any launch script.  Environment-specific settings (e.g. the embedding
backend's credentials or cache) are passed via the ``index_runtime_kwargs``
parameter.

Typical workflow:
  1. Build an embedding index offline (e.g., via
     ``onetwo.stdlib.retrieval.build_index``).
  2. Start a launch script that calls ``create_server`` with the right kwargs.
  3. Agents call the ``search`` tool to retrieve relevant documents.
  4. Optionally, agents call the ``get_document`` tool to retrieve a full
     document by its docid (when ``include_get_document`` is set).

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

from typing import Any

from absl import logging
from mcp import types as mcp_types
from mcp.server.fastmcp import server
from onetwo.backends import sentencepiece_tokenizer
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.tool_use import mcp_tools_utils


def _build_search_description(max_results: int) -> str:
  """Returns the tool description for the search tool."""
  return (
      f'Perform a search on a knowledge source. Returns top-{max_results}'
      ' hits with docid, score, and snippet. The snippet contains the'
      " document's contents (may be truncated based on token limits)."
  )


# TODO: Revisit this API to reduce the number of arguments
# (e.g. grouping `snippet_max_tokens` and `tokenizer_model_name`).
def create_server(
    *,
    index_path: str,
    max_results: int,
    snippet_max_tokens: int,
    port: int,
    transport: str,
    host: str = 'localhost',
    include_get_document: bool = False,
    index_runtime_kwargs: dict[str, Any] | None = None,
    tokenizer_model_name: str | None = None,
) -> server.FastMCP:
  """Creates and configures the OneTwo Index MCP server.

  Args:
    index_path: Path to the pre-built embedding index.
    max_results: Default number of top results to return.
    snippet_max_tokens: Maximum tokens per snippet (-1 = no truncation). Text
      only, per text chunk for multimodal documents. Media is never truncated.
    port: Port for SSE/streamable-http transports.
    transport: Transport type (stdio, sse, streamable-http).
    host: Host/interface to bind the server on (default: localhost).
    include_get_document: If True, register a get_document tool.
    index_runtime_kwargs: Optional runtime kwargs passed to the
      ``EmbeddingBasedDocumentIndex`` constructor (e.g. the embedding backend's
      credentials or cache).  When ``None``, an empty dict is used.
    tokenizer_model_name: Name of the SentencePiece tokenizer model to use for
      snippet truncation.  Must be a key in
      ``sentencepiece_tokenizer.TOKENIZER_PATHS`` or a direct file path.
      Required when ``snippet_max_tokens >= 0``.

  Returns:
    A configured FastMCP server instance.

  Raises:
    ValueError: If ``snippet_max_tokens >= 0`` but ``tokenizer_model_name``
      is not provided.
  """
  if snippet_max_tokens >= 0 and tokenizer_model_name is None:
    raise ValueError(
        'tokenizer_model_name must be provided when snippet_max_tokens >= 0.'
    )

  if transport == 'stdio':
    mcp = server.FastMCP('OneTwo Index MCP Server')
  else:
    mcp = server.FastMCP('OneTwo Index MCP Server', host=host, port=port)

  # TODO: Support other index types (e.g. any
  # `RewritingIndex`), not only `EmbeddingBasedDocumentIndex`.
  document_index = indexing.EmbeddingBasedDocumentIndex(
      runtime_kwargs=index_runtime_kwargs or {},
  )

  logging.info('Loading index from: %s', index_path)
  executing.run(document_index.load(index_path))
  logging.info(
      'Index loaded successfully. %d documents in the index.',
      document_index.num_docs,
  )

  # Register a tokenizer for snippet truncation only when needed. This
  # configures the global `llm.tokenize` and `llm.detokenize` builtins, which
  # `mcp_tools_utils.content_blocks` uses.
  if snippet_max_tokens >= 0:
    assert tokenizer_model_name is not None  # Validated above.
    sentencepiece_tokenizer.SentencePieceTokenizer(
        model_name=tokenizer_model_name
    ).register()

  async def _search_content(query: str) -> list[mcp_types.ContentBlock]:
    """Returns the content of the `search` results for `query`."""
    results_with_scores = await document_index.retrieve_with_scores(
        query, max_results=max_results
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
      description=_build_search_description(max_results),
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

  if include_get_document:

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
