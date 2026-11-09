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

"""Tests the tools of the OneTwo index MCP server."""

import asyncio
import base64
from collections.abc import Iterable, Sequence
import dataclasses
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from mcp import types as mcp_types
from mcp.shared import memory
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval.mcp import mcp_server

# Re-use the Document dataclass from OneTwo for constructing fake data.
Document = retrieval_data_structures.Document

# ---------------------------------------------------------------------------
# Fake documents used across all tests.
# ---------------------------------------------------------------------------
_FAKE_DOCS: Sequence[Document] = (
    Document(
        doc_id='1001',
        title='League of Legends - Wikipedia',
        content=(
            'League of Legends (LoL) is a multiplayer online battle arena'
            ' video game developed and published by Riot Games.'
        ),
        url='https://en.wikipedia.org/wiki/League_of_Legends',
        metadata={'split': 'train'},
    ),
    Document(
        doc_id='1002',
        title='Dota 2 - Wikipedia',
        content=(
            'Dota 2 is a multiplayer online battle arena video game developed'
            ' and published by Valve.'
        ),
        url='https://en.wikipedia.org/wiki/Dota_2',
        metadata={'split': 'train'},
    ),
    Document(
        doc_id='1003',
        title='Video Game Composers Delighted by Grammys',
        content=(
            'The music of the gaming world is finally getting recognition of'
            ' its own from the Recording Academy.'
        ),
        url='https://variety.com/2022/music/news/grammys-video-game-soundtrack-award',
        metadata={'split': 'train'},
    ),
)

_PDF_BYTES = b'%PDF-1.7\nfake page'
_PNG_BYTES = b'\x89PNG\r\n\x1a\nfake'
_WAV_BYTES = b'RIFF\x00\x00\x00\x00WAVEfake'

# Multimodal documents, like the single PDF pages of a `pdf_pages` index, plus
# an audio recording.
_FAKE_MULTIMODAL_DOCS: Sequence[Document] = (
    Document(
        doc_id='report_p1',
        title='Annual report',
        content=content_lib.ChunkList([
            content_lib.Chunk(
                content=_PDF_BYTES, content_type='application/pdf'
            )
        ]),
        metadata={'page': 1},
    ),
    Document(
        doc_id='report/p2',
        title='Annual report',
        content=content_lib.ChunkList([
            content_lib.Chunk('Figure 2:'),
            content_lib.Chunk(content=_PNG_BYTES, content_type='image/png'),
        ]),
        metadata={'page': 2},
    ),
    Document(
        doc_id='earnings_call',
        title='Earnings call',
        content=content_lib.ChunkList([
            content_lib.Chunk(content=_WAV_BYTES, content_type='audio/wav'),
        ]),
    ),
)


@dataclasses.dataclass
class _FakeIndex(indexing.NaiveIndex):
  """`NaiveIndex` whose hits have the deterministic scores 0.9, 0.8, ..."""

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self, query: str, *, max_results: int | None = None, **kwargs
  ) -> Iterable[tuple[Document, float]]:
    """Overridden from base class (Retriever)."""
    results = await self.retrieve(query, max_results=max_results, **kwargs)
    return [(doc, 0.9 - 0.1 * i) for i, doc in enumerate(results)]


def _make_fake_index(docs: Sequence[Document]) -> _FakeIndex:
  """Returns a `_FakeIndex` with `docs`."""
  index = _FakeIndex()
  executing.run(index.add_docs(docs))
  return index


async def _fake_tokenize(content: str | content_lib.ChunkList) -> list[int]:
  """Returns one token per character of `content`, as ``llm.tokenize``.

  This makes encoding and decoding exactly invertible and keeps truncation
  boundaries easy to reason about in assertions.

  Args:
    content: The content to tokenize.

  Returns:
    A list of tokens, one per character of the input.
  """
  assert isinstance(content, str), content
  return [ord(character) for character in content]


async def _fake_detokenize(tokens: Sequence[int]) -> str:
  """Returns the characters of `tokens`, as ``llm.detokenize``."""
  return ''.join(chr(token_id) for token_id in tokens)


def _register_fake_tokenizer() -> None:
  """Stand-in for ``SentencePieceTokenizer.register``."""
  llm.tokenize.configure(_fake_tokenize)
  llm.detokenize.configure(_fake_detokenize)


def _text_of(block: mcp_types.ContentBlock) -> str:
  """Returns the text of a text content block."""
  assert isinstance(block, mcp_types.TextContent), block
  return block.text


def _create_test_server(
    *,
    include_get_document: bool = False,
    docs: Sequence[Document] = _FAKE_DOCS,
    snippet_max_tokens: int = 512,
) -> tuple['mcp_server.server.FastMCP', list[Any]]:
  """Creates a ``FastMCP`` server backed by fake data.

  Patches the index loading and tokenizer construction so no real CNS path or
  backend credentials are required.  ``index_building.load_index`` returns a
  ``_FakeIndex`` with ``docs``, and the tokenizer registers character-level
  fakes as ``llm.tokenize`` and ``llm.detokenize``.  The patches are returned
  as a list of started patchers that the caller must stop after the tests.

  Args:
    include_get_document: If True, also register the ``get_document`` tool.
    docs: The documents of the fake index.
    snippet_max_tokens: Maximum tokens (characters, with the fake tokenizer) per
      snippet.

  Returns:
    A ``(server, patchers)`` tuple.  Call ``p.stop()`` on each patcher when
    the test is done.
  """
  patchers: list[Any] = []

  def _start(patcher: Any) -> Any:
    patchers.append(patcher)
    return patcher.start()

  _start(
      mock.patch.object(
          mcp_server.index_building,
          'load_index',
          return_value=_make_fake_index(docs),
      )
  )
  _start(
      mock.patch.object(
          mcp_server.sentencepiece_tokenizer.SentencePieceTokenizer,
          '__init__',
          return_value=None,
      )
  )
  _start(
      mock.patch.object(
          mcp_server.sentencepiece_tokenizer.SentencePieceTokenizer,
          'register',
          side_effect=_register_fake_tokenizer,
      )
  )

  mcp = mcp_server.create_server(
      index_path='/fake/index/path',
      max_results=3,
      snippet_max_tokens=snippet_max_tokens,
      port=0,
      transport='streamable-http',
      include_get_document=include_get_document,
      tokenizer_model_name='GEMINI3_G145',
  )
  return mcp, patchers


class DefaultSetupSearchOnlyTest(absltest.TestCase):
  """Tests the default configuration: only the ``search`` tool is exposed."""

  def setUp(self):
    super().setUp()
    self.mcp_server, self._patchers = _create_test_server(
        include_get_document=False,
    )

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()
    super().tearDown()

  def test_tools_list_contains_only_search(self):
    """Verifies that ``tools/list`` returns exactly one tool: ``search``."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.list_tools()
        return result

    result = asyncio.run(run())

    tool_names = [tool.name for tool in result.tools]
    self.assertEqual(tool_names, ['search'])

  def test_search_tool_schema(self):
    """Verifies that the ``search`` tool has the expected input schema."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.list_tools()
        return result

    result = asyncio.run(run())

    search_tool = result.tools[0]
    self.assertEqual(search_tool.name, 'search')
    self.assertIn('query', search_tool.inputSchema.get('properties', {}))
    self.assertIn('query', search_tool.inputSchema.get('required', []))

  def test_search_returns_results(self):
    """Verifies that calling ``search`` returns the expected fake results."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.call_tool(
            'search', {'query': 'League of Legends'}
        )
        return result

    result = asyncio.run(run())

    self.assertFalse(result.isError)
    self.assertIsNone(result.structuredContent)
    # A JSON header and a text block per hit.
    self.assertLen(result.content, 6)

    header = json.loads(_text_of(result.content[0]))
    with self.subTest('first_result_docid'):
      self.assertEqual(header['docid'], '1001')
    with self.subTest('first_result_has_score'):
      self.assertAlmostEqual(header['score'], 0.9, places=4)
    with self.subTest('first_result_has_snippet'):
      self.assertIn('League of Legends', _text_of(result.content[1]))

  def test_search_result_headers(self):
    """Each hit's header must have exactly ``docid`` and ``score`` keys."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.call_tool('search', {'query': 'any'})
        return result

    result = asyncio.run(run())

    headers = [json.loads(_text_of(block)) for block in result.content[::2]]
    self.assertEqual(
        [
            {'docid': '1001', 'score': 0.9},
            {'docid': '1002', 'score': 0.8},
            {'docid': '1003', 'score': 0.7},
        ],
        headers,
    )


class SnippetTruncationTest(absltest.TestCase):
  """Tests the truncation of the hits to ``snippet_max_tokens``."""

  def setUp(self):
    super().setUp()
    self.mcp_server, self._patchers = _create_test_server(snippet_max_tokens=10)

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()
    super().tearDown()

  def test_search_truncates_snippets(self):
    """Snippets are truncated to ``snippet_max_tokens`` (characters here)."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        return await client.call_tool('search', {'query': 'any'})

    result = asyncio.run(run())

    self.assertEqual('League of ', _text_of(result.content[1]))


class MultimodalSnippetTruncationTest(absltest.TestCase):
  """Tests that only the text chunks of multimodal hits are truncated."""

  def setUp(self):
    super().setUp()
    self.mcp_server, self._patchers = _create_test_server(
        docs=(
            Document(
                doc_id='scan_p1',
                content=content_lib.ChunkList([
                    content_lib.Chunk('A long caption of the scanned page.'),
                    content_lib.Chunk(
                        content=_PDF_BYTES, content_type='application/pdf'
                    ),
                ]),
            ),
        ),
        snippet_max_tokens=10,
    )

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()
    super().tearDown()

  def test_search_truncates_text_chunks_and_passes_media_through(self):
    """Text chunks are truncated per chunk; the PDF blob is sent unchanged."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        return await client.call_tool('search', {'query': 'any'})

    result = asyncio.run(run())

    self.assertEqual(
        [
            mcp_types.TextContent(
                type='text',
                text=json.dumps({'docid': 'scan_p1', 'score': 0.9}),
            ),
            mcp_types.TextContent(type='text', text='A long cap'),
            mcp_types.EmbeddedResource(
                type='resource',
                resource=mcp_types.BlobResourceContents.model_validate({
                    'uri': 'onetwo-index://unused-uri/scan_p1',
                    'mimeType': 'application/pdf',
                    'blob': base64.b64encode(_PDF_BYTES).decode('ascii'),
                }),
            ),
        ],
        result.content,
    )


class WithGetDocumentTest(absltest.TestCase):
  """Tests the non-default setup with ``--include_get_document=True``."""

  def setUp(self):
    super().setUp()
    self.mcp_server, self._patchers = _create_test_server(
        include_get_document=True,
    )

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()
    super().tearDown()

  def test_tools_list_contains_search_and_get_document(self):
    """Verifies that both ``search`` and ``get_document`` tools are listed."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.list_tools()
        return result

    result = asyncio.run(run())

    tool_names = sorted(tool.name for tool in result.tools)
    self.assertEqual(tool_names, ['get_document', 'search'])

  def test_get_document_schema(self):
    """Verifies the ``get_document`` tool has the expected input schema."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.list_tools()
        return result

    result = asyncio.run(run())

    get_doc_tool = next(t for t in result.tools if t.name == 'get_document')
    self.assertIn('docid', get_doc_tool.inputSchema.get('properties', {}))
    self.assertIn('docid', get_doc_tool.inputSchema.get('required', []))

  def test_search_still_works(self):
    """``search`` should behave identically regardless of ``get_document``."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.call_tool(
            'search', {'query': 'League of Legends'}
        )
        return result

    result = asyncio.run(run())

    self.assertFalse(result.isError)
    self.assertLen(result.content, 6)
    self.assertEqual(json.loads(_text_of(result.content[0]))['docid'], '1001')

  def test_get_document_returns_full_document(self):
    """Verifies ``get_document`` returns all expected fields for a valid ID."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.call_tool('get_document', {'docid': '1003'})
        return result

    result = asyncio.run(run())

    self.assertFalse(result.isError)
    header_block, text_block = result.content
    header = json.loads(_text_of(header_block))
    with self.subTest('docid'):
      self.assertEqual(header['docid'], '1003')
    with self.subTest('title'):
      self.assertEqual(
          header['title'],
          'Video Game Composers Delighted by Grammys',
      )
    with self.subTest('text'):
      self.assertIn('Recording Academy', _text_of(text_block))
    with self.subTest('url'):
      self.assertIn('variety.com', header['url'])
    with self.subTest('metadata'):
      self.assertEqual(header['metadata'], {'split': 'train'})

  def test_get_document_returns_none_for_unknown_id(self):
    """``get_document`` returns None, i.e. no content, for an unknown docid."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        result = await client.call_tool(
            'get_document', {'docid': 'nonexistent'}
        )
        return result

    result = asyncio.run(run())

    with self.subTest('is_not_error'):
      self.assertFalse(result.isError)
    with self.subTest('no_content'):
      self.assertEmpty(result.content)


class MultimodalIndexTest(absltest.TestCase):
  """Tests an index with multimodal (PDF, image and audio) documents."""

  def setUp(self):
    super().setUp()
    self.mcp_server, self._patchers = _create_test_server(
        include_get_document=True, docs=_FAKE_MULTIMODAL_DOCS
    )

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()
    super().tearDown()

  def _call_tool(
      self, name: str, arguments: dict[str, Any]
  ) -> mcp_types.CallToolResult:
    """Calls a tool of the server through an in-memory MCP client."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        return await client.call_tool(name, arguments)

    return asyncio.run(run())

  def test_tools_declare_no_output_schema(self):
    """Multimodal results are content blocks, not structured content."""

    async def run():
      async with memory.create_connected_server_and_client_session(
          self.mcp_server
      ) as client:
        return await client.list_tools()

    result = asyncio.run(run())

    self.assertEqual(
        {'search': None, 'get_document': None},
        {tool.name: tool.outputSchema for tool in result.tools},
    )

  def test_search_returns_content_blocks(self):
    """Each hit is a JSON header followed by its content blocks."""
    result = self._call_tool('search', {'query': 'report'})

    self.assertFalse(result.isError)
    self.assertIsNone(result.structuredContent)
    self.assertEqual(
        [
            mcp_types.TextContent(
                type='text',
                text=json.dumps({'docid': 'report_p1', 'score': 0.9}),
            ),
            mcp_types.EmbeddedResource(
                type='resource',
                resource=mcp_types.BlobResourceContents.model_validate({
                    'uri': 'onetwo-index://unused-uri/report_p1',
                    'mimeType': 'application/pdf',
                    'blob': base64.b64encode(_PDF_BYTES).decode('ascii'),
                }),
            ),
            mcp_types.TextContent(
                type='text',
                text=json.dumps({'docid': 'report/p2', 'score': 0.8}),
            ),
            mcp_types.TextContent(type='text', text='Figure 2:'),
            mcp_types.ImageContent(
                type='image',
                data=base64.b64encode(_PNG_BYTES).decode('ascii'),
                mimeType='image/png',
            ),
            mcp_types.TextContent(
                type='text',
                text=json.dumps({'docid': 'earnings_call', 'score': 0.7}),
            ),
            mcp_types.AudioContent(
                type='audio',
                data=base64.b64encode(_WAV_BYTES).decode('ascii'),
                mimeType='audio/wav',
            ),
        ],
        result.content,
    )

  def test_get_document_returns_header_and_content(self):
    """``get_document`` returns the document's fields and its PDF."""
    result = self._call_tool('get_document', {'docid': 'report_p1'})

    self.assertFalse(result.isError)
    header, pdf = result.content
    with self.subTest('header'):
      assert isinstance(header, mcp_types.TextContent)
      self.assertEqual(
          {
              'docid': 'report_p1',
              'title': 'Annual report',
              'url': '',
              'metadata': {'page': 1},
          },
          json.loads(header.text),
      )
    with self.subTest('pdf'):
      assert isinstance(pdf, mcp_types.EmbeddedResource)
      assert isinstance(pdf.resource, mcp_types.BlobResourceContents)
      self.assertEqual('application/pdf', pdf.resource.mimeType)
      self.assertEqual(_PDF_BYTES, base64.b64decode(pdf.resource.blob))

  def test_get_document_returns_none_for_unknown_id(self):
    """``get_document`` returns None, i.e. no content, for an unknown docid."""
    result = self._call_tool('get_document', {'docid': 'nonexistent'})

    with self.subTest('is_not_error'):
      self.assertFalse(result.isError)
    with self.subTest('no_content'):
      self.assertEmpty(result.content)


class LoadIndexTest(absltest.TestCase):
  """Tests that the index is loaded with `index_building.load_index`."""

  def _create_server_and_get_load_index_call(self, **kwargs: Any) -> Any:
    """Creates a server and returns the call of `load_index`.

    Args:
      **kwargs: Extra arguments of `create_server`.

    Returns:
      The only call of the mocked `index_building.load_index`.
    """
    with mock.patch.object(
        mcp_server.index_building,
        'load_index',
        autospec=True,
        return_value=_make_fake_index(_FAKE_DOCS),
    ) as mock_load_index:
      mcp_server.create_server(
          index_path='/fake/index/path',
          max_results=3,
          snippet_max_tokens=-1,
          port=0,
          transport='streamable-http',
          **kwargs,
      )
    mock_load_index.assert_called_once()
    return mock_load_index.call_args

  def test_loads_the_index_with_its_runtime_kwargs(self):
    call = self._create_server_and_get_load_index_call(
        runtime_kwargs={'api_key': 'fake_key'}
    )

    self.assertEqual(
        call,
        mock.call('/fake/index/path', runtime_kwargs={'api_key': 'fake_key'}),
    )

  def test_runtime_kwargs_default_to_none(self):
    call = self._create_server_and_get_load_index_call()

    self.assertEqual(
        call, mock.call('/fake/index/path', runtime_kwargs=None)
    )


if __name__ == '__main__':
  absltest.main()
