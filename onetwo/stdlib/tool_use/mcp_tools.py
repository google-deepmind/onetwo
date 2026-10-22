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

"""Tools from an MCP server, as `Tool`s any backend can be given.

The Model Context Protocol is how a corpus, a service or a toolbox declares
what it can do without anyone writing an adapter for it. A server says which
tools it has and what arguments they take; a client discovers them at run time.

This module is the join between that and the rest of OneTwo. It turns what a
server declares into ordinary `llm_tool_use.Tool`s, which is the only thing the
backends know about:

```python
  agent = agent_lib.Agent(
      tools=[get_weather, mcp_tools.server('http://localhost:8777/mcp')]
  )
```

A server that speaks stdio instead, as many local ones do, is named by its
command line and started for each run: `mcp_tools.stdio_server(['node',
'index.js'])`.

Because the result is a plain `Tool`, every backend that can offer a tool to a
model can offer an MCP tool -- there is nothing MCP-specific left downstream of
this module, and nothing here that is specific to a backend, a strategy or an
agent.

A connection is one session: the tools are valid for as long as it is open, and
calling one after it closes is an error. `server` leaves that to whoever is
given it, which is how an agent comes to open one per run and close it again,
so that concurrent questions never share a connection. `mcp_session` is the
same thing with the lifetime in the caller's hands:

```python
  async with mcp_tools.mcp_session('http://localhost:8777/mcp') as tools:
    ...
```
"""

import base64
from collections.abc import AsyncIterator, Mapping, Sequence
import contextlib
import functools
from typing import Any

from mcp import types as mcp_types
from mcp.client import session as client_session
from mcp.client import stdio
from mcp.client import streamable_http
from onetwo.core import content as content_lib
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import mcp_http


# TODO: Extend to handle multimodal content.
def _text_of(result: mcp_types.CallToolResult) -> str:
  """Returns the text blocks of a result, joined.

  Args:
    result: What the server sent back.

  Returns:
    Every text block, newline-separated. Empty if the result carried none,
    which is what a server sends for a tool that returns nothing.
  """
  return '\n'.join(
      content.text
      for content in result.content
      if isinstance(content, mcp_types.TextContent)
  )


def _chunk_from_block(block: mcp_types.ContentBlock) -> content_lib.Chunk:
  """Converts an MCP text or image `ContentBlock` into a OneTwo `Chunk`."""
  if isinstance(block, mcp_types.TextContent):
    return content_lib.Chunk(content=block.text)
  if isinstance(block, mcp_types.ImageContent):
    return content_lib.Chunk(
        content=base64.b64decode(block.data), content_type=block.mimeType
    )
  raise TypeError(f'Unsupported MCP content block type: {type(block)}')


def _returned(result: mcp_types.CallToolResult) -> Any:
  """Returns what a tool call produced, as plain Python or a `ChunkList`.

  A server that declares an output schema must answer with `structuredContent`,
  and the client session checks that it did and that the content conforms -- so
  where there is structure, it is the server's own, not something recovered by
  guessing at the text. When no output schema is declared and `result.content`
  carries `ImageContent`, the blocks are decoded in order into a
  `content_lib.ChunkList`; otherwise only text was sent, and is joined.

  Note that the structure is whatever the server chose to send, envelope
  included: a tool whose result is not itself an object has to be wrapped in
  one to satisfy the schema, and servers built with FastMCP wrap it as
  `{'result': ...}`. Unwrapping that here would be guessing again, since the
  key is the server's choice and not part of the protocol.

  Args:
    result: What the server sent back.

  Returns:
    The structured content if there is any, a `ChunkList` when `result.content`
    contains image blocks, or the joined text.
  """
  if result.structuredContent is not None:
    return result.structuredContent
  if any(isinstance(b, mcp_types.ImageContent) for b in result.content):
    return content_lib.ChunkList(
        [_chunk_from_block(block) for block in result.content]
    )
  return _text_of(result)


def tool_from_declaration(
    session: client_session.ClientSession, declared: mcp_types.Tool
) -> llm_tool_use.Tool:
  """Returns one tool a server declared, as a `Tool`.

  Args:
    session: An open session to call the tool through. It must outlive the
      returned tool, which holds it.
    declared: What the server said about the tool.

  Returns:
    A tool that calls the server, declared as the server declared it.
  """

  async def call(**kwargs) -> Any:
    result = await session.call_tool(declared.name, arguments=kwargs)
    if result.isError:
      # Raised rather than returned, because an agent turns an exception from a
      # tool into a message to the model, which can then correct itself. The
      # text is what the server chose to say about the failure.
      raise RuntimeError(_text_of(result) or f'{declared.name} failed.')
    return _returned(result)

  return llm_tool_use.Tool(
      name=declared.name,
      description=declared.description or '',
      # Passed on as the server wrote it. A JSON Schema is what the backends
      # want and what MCP carries, so there is nothing to convert; keywords the
      # server includes that a model API does not use, such as the `title` that
      # FastMCP generates, are ignored by the API rather than rejected.
      parameters=declared.inputSchema,
      function=call,
  )


@contextlib.asynccontextmanager
async def _session_tools(
    read: Any, write: Any
) -> AsyncIterator[Sequence[llm_tool_use.Tool]]:
  """Opens a session over a transport and yields the tools it declares.

  Args:
    read: The transport's stream of messages from the server.
    write: The transport's stream of messages to the server.

  Yields:
    One tool per tool the server declares, valid until the session closes.
  """
  async with client_session.ClientSession(read, write) as session:
    await session.initialize()
    declared = await session.list_tools()
    yield [tool_from_declaration(session, tool) for tool in declared.tools]


@contextlib.asynccontextmanager
async def mcp_session(url: str) -> AsyncIterator[Sequence[llm_tool_use.Tool]]:
  """Connects to an MCP server and yields the tools it declares.

  Args:
    url: Streamable-HTTP endpoint of a running server.

  Yields:
    One tool per tool the server declares, valid until the session closes.
  """
  async with mcp_http.client() as http_client:
    async with streamable_http.streamable_http_client(
        url, http_client=http_client
    ) as (read, write, _):
      async with _session_tools(read, write) as tools:
        yield tools


@contextlib.asynccontextmanager
async def stdio_session(
    command: Sequence[str], env: Mapping[str, str] | None = None
) -> AsyncIterator[Sequence[llm_tool_use.Tool]]:
  """Starts an MCP server as a subprocess and yields the tools it declares.

  The server is spoken to over its stdin and stdout, and is stopped when the
  session closes.

  Args:
    command: The server's command line, executable first, as for `subprocess`.
    env: The server's environment. By default, only the variables the MCP SDK
      deems safe to inherit, such as `HOME` and `PATH`, are passed on.

  Yields:
    One tool per tool the server declares, valid until the session closes.
  """
  if not command:
    raise ValueError('An MCP server command must name an executable.')
  params = stdio.StdioServerParameters(
      command=command[0],
      args=list(command[1:]),
      env=dict(env) if env is not None else None,
  )
  async with stdio.stdio_client(params) as (read, write):
    async with _session_tools(read, write) as tools:
      yield tools


def server(url: str) -> llm_tool_use.ToolSource:
  """Returns a server named as a source of tools, to be listed among them.

  Nothing is connected to here. The server is asked what it has when a run
  begins, and the session is closed when the run ends, so the same value can be
  used for one question or a thousand:

  ```python
    agent = agent_lib.Agent(
        tools=[get_weather, mcp_tools.server('http://localhost:8777/mcp')]
    )
  ```

  Args:
    url: Streamable-HTTP endpoint of the server.

  Returns:
    A source that yields the server's tools for the length of one run.
  """
  return llm_tool_use.ToolSource(open=functools.partial(mcp_session, url))


def stdio_server(
    command: Sequence[str], env: Mapping[str, str] | None = None
) -> llm_tool_use.ToolSource:
  """Returns a local server named as a source of tools, like `server`.

  For servers that speak stdio rather than HTTP. Nothing is started here: the
  server is started when a run begins and stopped when it ends, so each run
  gets its own process:

  ```python
    agent = agent_lib.Agent(
        tools=[mcp_tools.stdio_server(['node', 'server/dist/index.js'])]
    )
  ```

  Args:
    command: The server's command line, executable first, as for `subprocess`.
    env: The server's environment. By default, only the variables the MCP SDK
      deems safe to inherit, such as `HOME` and `PATH`, are passed on.

  Returns:
    A source that yields the server's tools for the length of one run.
  """
  return llm_tool_use.ToolSource(
      open=functools.partial(stdio_session, command, env)
  )
