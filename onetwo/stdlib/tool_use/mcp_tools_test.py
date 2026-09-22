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

"""Tests for `mcp_tools`, against a real MCP server running in-process."""

import asyncio
from typing import Any

from absl.testing import absltest
from mcp.server.fastmcp import server as fastmcp_server
from mcp.shared import memory
from onetwo.stdlib.tool_use import mcp_tools


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

  del lookup, search, shout  # Registered by the decorator.
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

    self.assertEqual(['lookup', 'search', 'shout'], names)
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


if __name__ == '__main__':
  absltest.main()
