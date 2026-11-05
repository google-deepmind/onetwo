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

"""Tests the documents returned by the GraphRAG MCP server's tools."""

import asyncio
import base64
from collections.abc import Sequence
import json
import types
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from mcp import types as mcp_types
from mcp.shared import memory
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.graphrag import graph as graph_lib
from onetwo.stdlib.retrieval.graphrag import indexing
from onetwo.stdlib.retrieval.graphrag import mcp_server

Document = retrieval_data_structures.Document

_PDF_BYTES = b'%PDF-1.7\nfake page'
_ENTITY_ID = 'entity_revenue_concept'

# A text document, as in an index of text chunks.
_TEXT_DOC = Document(
    doc_id='report_chunk_1',
    title='Annual report',
    content='Revenue grew by 10% in 2024.',
    url='https://example.com/report',
    metadata={'chunk': 1},
)
# A single PDF page, as in a `pdf_pages` index.
_PDF_DOC = Document(
    doc_id='report_p1',
    title='Annual report',
    content=content_lib.ChunkList(
        [content_lib.Chunk(content=_PDF_BYTES, content_type='application/pdf')]
    ),
    url='https://example.com/report.pdf',
    metadata={'page': 1},
)


class _FakePassageIndex:
  """Passage index whose `retrieve_with_scores` returns all its documents."""

  def __init__(self, docs: Sequence[Document]):
    self.docs = docs

    @executing.make_executable
    async def retrieve_with_scores(
        query: str, max_results: int = 5
    ) -> list[tuple[Document, float]]:
      del query
      return [(doc, 0.9) for doc in self.docs][:max_results]

    self.retrieve_with_scores = retrieve_with_scores


def _make_graph(docs: Sequence[Document]) -> graph_lib.KnowledgeGraph:
  """Returns a graph with one entity, mentioned by every passage of `docs`."""
  graph = graph_lib.KnowledgeGraph()
  graph.upsert_node(
      graph_lib.EntityNode(
          id=_ENTITY_ID,
          entity=graph_lib.Entity(name='Revenue', type='concept'),
      )
  )
  for doc_index, doc in enumerate(docs):
    passage_id = f'passage_{doc.doc_id}'
    graph.upsert_node(
        graph_lib.PassageNode(
            id=passage_id, doc_id=doc_index, metadata={'doc_id': doc.doc_id}
        )
    )
    graph.upsert_edge(_ENTITY_ID, passage_id)
  return graph


def _call_tool(
    docs: Sequence[Document],
    name: str,
    arguments: dict[str, Any],
    *,
    snippet_max_tokens: int = -1,
) -> mcp_types.CallToolResult:
  """Returns the result of calling a tool of a server over `docs`.

  Args:
    docs: The documents of the fake HippoRAG index served.
    name: Name of the tool.
    arguments: Arguments of the tool.
    snippet_max_tokens: `snippet_max_tokens` of the server. Without a
      tokenizer, snippets are truncated to this number of words.
  """
  # A fake HippoRAG index of `docs`, without an entity index.
  index = types.SimpleNamespace(
      docs=list(docs),
      graph=_make_graph(docs),
      entity_index=None,
      passage_index=_FakePassageIndex(docs),
  )
  server = mcp_server.create_server_for_index(
      cast(indexing.HippoRAGDocumentIndex, index),
      transport='streamable-http',
      port=0,
      snippet_max_tokens=snippet_max_tokens,
  )

  async def run() -> mcp_types.CallToolResult:
    async with memory.create_connected_server_and_client_session(
        server
    ) as client:
      return await client.call_tool(name, arguments)

  return asyncio.run(run())


def _pdf_block() -> mcp_types.EmbeddedResource:
  """Returns `_PDF_DOC`'s page as the server sends it."""
  return mcp_types.EmbeddedResource(
      type='resource',
      resource=mcp_types.BlobResourceContents.model_validate({
          'uri': 'onetwo-index://unused-uri/report_p1',
          'mimeType': 'application/pdf',
          'blob': base64.b64encode(_PDF_BYTES).decode('ascii'),
      }),
  )


def _json_text_block(value: Any) -> mcp_types.TextContent:
  """Returns a compact JSON header block, as `mcp_tools_utils` sends it."""
  return mcp_types.TextContent(type='text', text=json.dumps(value))


class McpServerDocumentsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'search_passages',
          'search_passages',
          {'query': 'revenue'},
          {'doc_id': 'report_p1', 'title': 'Annual report', 'score': 0.9},
      ),
      (
          'get_entity_passages',
          'get_entity_passages',
          {'entity_id': _ENTITY_ID},
          {'doc_id': 'report_p1', 'title': 'Annual report'},
      ),
      (
          'get_document',
          'get_document',
          {'doc_id': 'report_p1'},
          {
              'doc_id': 'report_p1',
              'title': 'Annual report',
              'url': 'https://example.com/report.pdf',
              'metadata': {'page': 1},
          },
      ),
  )
  def test_returns_pdf_pages_as_pdfs(
      self, tool_name: str, arguments: dict[str, Any], header: dict[str, Any]
  ):
    result = _call_tool([_PDF_DOC], tool_name, arguments)

    with self.subTest('is_not_error'):
      self.assertFalse(result.isError)
    with self.subTest('header_then_pdf'):
      self.assertEqual(result.content, [_json_text_block(header), _pdf_block()])
    with self.subTest('no_structured_content'):
      self.assertIsNone(result.structuredContent)

  @parameterized.named_parameters(
      (
          'search_passages',
          'search_passages',
          {'query': 'revenue'},
          {'doc_id': 'report_chunk_1', 'title': 'Annual report', 'score': 0.9},
      ),
      (
          'get_entity_passages',
          'get_entity_passages',
          {'entity_id': _ENTITY_ID},
          {'doc_id': 'report_chunk_1', 'title': 'Annual report'},
      ),
      (
          'get_document',
          'get_document',
          {'doc_id': 'report_chunk_1'},
          {
              'doc_id': 'report_chunk_1',
              'title': 'Annual report',
              'url': 'https://example.com/report',
              'metadata': {'chunk': 1},
          },
      ),
  )
  def test_returns_text_documents_as_text(
      self, tool_name: str, arguments: dict[str, Any], header: dict[str, Any]
  ):
    result = _call_tool([_TEXT_DOC], tool_name, arguments)

    with self.subTest('is_not_error'):
      self.assertFalse(result.isError)
    with self.subTest('header_then_text'):
      self.assertEqual(
          result.content,
          [
              _json_text_block(header),
              mcp_types.TextContent(
                  type='text', text='Revenue grew by 10% in 2024.'
              ),
          ],
      )
    with self.subTest('no_structured_content'):
      self.assertIsNone(result.structuredContent)

  @parameterized.named_parameters(
      (
          'search_passages_truncates',
          'search_passages',
          {'query': 'revenue'},
          'Revenue grew [truncated]',
      ),
      (
          'get_entity_passages_truncates',
          'get_entity_passages',
          {'entity_id': _ENTITY_ID},
          'Revenue grew [truncated]',
      ),
      (
          'get_document_does_not_truncate',
          'get_document',
          {'doc_id': 'report_chunk_1'},
          'Revenue grew by 10% in 2024.',
      ),
  )
  def test_truncates_snippets_but_not_documents(
      self, tool_name: str, arguments: dict[str, Any], text: str
  ):
    result = _call_tool(
        [_TEXT_DOC], tool_name, arguments, snippet_max_tokens=2
    )

    self.assertEqual(
        result.content[1:], [mcp_types.TextContent(type='text', text=text)]
    )

  @parameterized.named_parameters(
      (
          'unknown_document',
          'get_document',
          {'doc_id': 'unknown'},
          "Document 'unknown' not found.",
      ),
      (
          'unknown_entity',
          'get_entity_passages',
          {'entity_id': 'unknown'},
          "Entity 'unknown' not found in graph.",
      ),
  )
  def test_returns_error_block(
      self, tool_name: str, arguments: dict[str, Any], message: str
  ):
    result = _call_tool([_TEXT_DOC], tool_name, arguments)

    self.assertEqual(result.content, [_json_text_block({'error': message})])


if __name__ == '__main__':
  absltest.main()
