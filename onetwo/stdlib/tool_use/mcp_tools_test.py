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

"""Tests for `mcp_tools`, against a real MCP server, in-process or over stdio."""

import asyncio
import base64
import os
import sys
from typing import Any

from absl.testing import absltest
from mcp import types as mcp_types
from mcp.server.fastmcp import server as fastmcp_server
from mcp.shared import memory
from onetwo.core import content as content_lib
from onetwo.stdlib.tool_use import mcp_tools

# Run this file with this argument to serve `_server()` over stdio instead.
_SERVE = '--serve_stdio'
_PNG_BYTES = b'\x89PNG\r\n\x1a\nfake'
_PDF_BYTES = b'%PDF-1.7\nfake'
_WAV_BYTES = b'RIFF\x00\x00\x00\x00WAVEfake'
# Content blocks of kinds that `mcp_tools` does not support, in wire format.
_UNSUPPORTED_BLOCKS = {
    'audio': {
        'type': 'audio',
        'data': base64.b64encode(_WAV_BYTES).decode('ascii'),
        'mimeType': 'audio/wav',
    },
    'text_resource': {
        'type': 'resource',
        'resource': {
            'uri': 'test://notes/n1',
            'mimeType': 'text/markdown',
            'text': '# Notes',
        },
    },
    'non_pdf_blob': {
        'type': 'resource',
        'resource': {
            'uri': 'test://blobs/b1',
            'mimeType': 'application/zip',
            'blob': base64.b64encode(b'PK fake').decode('ascii'),
        },
    },
    'untyped_blob': {
        'type': 'resource',
        'resource': {
            'uri': 'test://blobs/b2',
            'blob': base64.b64encode(_PDF_BYTES).decode('ascii'),
        },
    },
    'resource_link': {
        'type': 'resource_link',
        'name': 'appendix',
        'uri': 'test://appendices/a1',
    },
}


def _server() -> fastmcp_server.FastMCP:
  """Returns a server declaring one tool of each shape worth testing."""
  mcp = fastmcp_server.FastMCP('Test Server')

  @mcp.tool(name='lookup', description='Looks a document up by id.')
  def lookup(docid: str) -> dict[str, Any]:
    """Returns a document, which is already an object."""
    if docid != 'd1':
      raise ValueError(f'No such document: {docid}')
    return {'docid': docid, 'title': 'First'}

  @mcp.tool(name='search', description='Finds documents.')
  def search(query: str) -> list[dict[str, Any]]:
    """Returns several documents, which is not an object."""
    del query
    return [{'docid': 'd1'}, {'docid': 'd2'}]

  @mcp.tool(
      name='shout', description='Returns plain text.', structured_output=False
  )
  def shout(word: str) -> str:
    """Returns text, with no output schema declared."""
    return word.upper()

  @mcp.tool(
      name='inspect_portrait',
      description='Returns metadata text and an image.',
      structured_output=False,
  )
  def inspect_portrait(docid: str) -> mcp_types.CallToolResult:
    """Returns interleaved text and image content blocks."""
    return mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type='text', text=f'docid={docid}'),
            mcp_types.ImageContent(
                type='image',
                data=base64.b64encode(_PNG_BYTES).decode('ascii'),
                mimeType='image/png',
            ),
        ]
    )

  del lookup, search, shout, inspect_portrait  # Registered by the decorator.

  @mcp.tool(
      name='fetch_report',
      description='Returns a report as a PDF.',
      structured_output=False,
  )
  def fetch_report(docid: str) -> mcp_types.CallToolResult:
    """Returns text and a PDF embedded as a binary resource."""
    # Validated from the wire format, which is also how URIs are parsed.
    return mcp_types.CallToolResult.model_validate({
        'content': [
            {'type': 'text', 'text': f'docid={docid}'},
            {
                'type': 'resource',
                'resource': {
                    'uri': f'test://reports/{docid}',
                    'mimeType': 'application/pdf',
                    'blob': base64.b64encode(_PDF_BYTES).decode('ascii'),
                },
            },
        ]
    })

  @mcp.tool(
      name='fetch_unsupported',
      description='Returns a content block of an unsupported kind.',
      structured_output=False,
  )
  def fetch_unsupported(kind: str) -> mcp_types.CallToolResult:
    """Returns text and the unsupported block named by `kind`."""
    return mcp_types.CallToolResult.model_validate({
        'content': [
            {'type': 'text', 'text': f'kind={kind}'},
            _UNSUPPORTED_BLOCKS[kind],
        ]
    })

  del fetch_report, fetch_unsupported  # Registered by the decorator.
  return mcp


async def _with_tools(body):
  """Runs `body` with the tools of an in-process server, keyed by name."""
  async with memory.create_connected_server_and_client_session(
      _server()
  ) as session:
    declared = await session.list_tools()
    tools = {
        tool.name: mcp_tools.tool_from_declaration(session, tool)
        for tool in declared.tools
    }
    return await body(tools)


class McpToolsTest(absltest.TestCase):

  def test_declares_every_tool_the_server_declares(self):
    async def body(tools):
      return sorted(tools), tools['lookup'].description

    names, description = asyncio.run(_with_tools(body))

    self.assertEqual(
        [
            'fetch_report',
            'fetch_unsupported',
            'inspect_portrait',
            'lookup',
            'search',
            'shout',
        ],
        names,
    )
    self.assertEqual('Looks a document up by id.', description)

  def test_passes_the_servers_schema_through_untouched(self):
    async def body(tools):
      return tools['lookup'].parameters

    parameters = asyncio.run(_with_tools(body))

    # Exactly what the server sent, `title` included: a JSON Schema is what the
    # backends want, so there is nothing for this module to convert.
    self.assertEqual(
        {
            'properties': {'docid': {'title': 'Docid', 'type': 'string'}},
            'required': ['docid'],
            'title': 'lookupArguments',
            'type': 'object',
        },
        parameters,
    )

  def test_a_result_that_is_an_object_arrives_as_that_object(self):
    async def body(tools):
      return await tools['lookup'](docid='d1')

    self.assertEqual(
        {'docid': 'd1', 'title': 'First'}, asyncio.run(_with_tools(body))
    )

  def test_a_result_that_is_not_an_object_keeps_the_servers_envelope(self):
    async def body(tools):
      return await tools['search'](query='anything')

    # A list cannot be structured content on its own, so the server wraps it.
    # `result` is what FastMCP chose to call the envelope; this module does not
    # unwrap it, because the key is the server's choice rather than protocol.
    self.assertEqual(
        {'result': [{'docid': 'd1'}, {'docid': 'd2'}]},
        asyncio.run(_with_tools(body)),
    )

  def test_a_tool_that_declares_no_output_schema_returns_text(self):
    async def body(tools):
      return await tools['shout'](word='hello')

    self.assertEqual('HELLO', asyncio.run(_with_tools(body)))

  def test_a_failing_tool_raises_so_the_model_can_be_told(self):
    async def body(tools):
      try:
        await tools['lookup'](docid='nope')
      except RuntimeError as error:
        return str(error)
      return None

    message = asyncio.run(_with_tools(body))

    self.assertIsNotNone(message)
    self.assertIn('No such document: nope', message)

  def test_a_stdio_server_is_started_and_its_tools_work(self):
    # The server is this file, run again as a subprocess in serving mode. A
    # hermetic Python has no separate interpreter to run it with, but then the
    # file has been built into an executable beside it, which serves as well.
    if sys.executable:
      command = [sys.executable, __file__, _SERVE]
    else:
      command = [os.path.splitext(__file__)[0], _SERVE]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    async def body():
      async with mcp_tools.stdio_session(command, env=env) as tools:
        by_name = {tool.name: tool for tool in tools}
        return sorted(by_name), await by_name['shout'](word='hello')

    names, shouted = asyncio.run(body())

    self.assertEqual(
        [
            'fetch_report',
            'fetch_unsupported',
            'inspect_portrait',
            'lookup',
            'search',
            'shout',
        ],
        names,
    )
    self.assertEqual('HELLO', shouted)

  def test_multimodal_tool_result_decodes_to_chunk_list(self):
    async def body(tools):
      return await tools['inspect_portrait'](docid='CV_1')

    returned = asyncio.run(_with_tools(body))

    self.assertEqual(
        content_lib.ChunkList([
            content_lib.Chunk('docid=CV_1'),
            content_lib.Chunk(content=_PNG_BYTES, content_type='image/png'),
        ]),
        returned,
    )

  def test_pdf_tool_result_decodes_to_chunk_list(self):
    async def body(tools):
      return await tools['fetch_report'](docid='R_1')

    returned = asyncio.run(_with_tools(body))

    self.assertEqual(
        content_lib.ChunkList([
            content_lib.Chunk('docid=R_1'),
            content_lib.Chunk(
                content=_PDF_BYTES, content_type='application/pdf'
            ),
        ]),
        returned,
    )

  def test_unsupported_content_blocks_raise(self):
    for kind in _UNSUPPORTED_BLOCKS:
      with self.subTest(kind):

        async def body(tools, kind=kind):
          # Caught here, since the session's task group would wrap it.
          try:
            await tools['fetch_unsupported'](kind=kind)
          except ValueError as error:
            return str(error)
          return None

        message = asyncio.run(_with_tools(body))

        self.assertIsNotNone(message)
        self.assertIn('Unsupported MCP', message)

  def test_a_stdio_command_must_name_an_executable(self):
    async def body():
      async with mcp_tools.stdio_session([]):
        pass

    with self.assertRaises(ValueError):
      asyncio.run(body())


if __name__ == '__main__':
  if _SERVE in sys.argv:
    _server().run('stdio')
  else:
    absltest.main()
