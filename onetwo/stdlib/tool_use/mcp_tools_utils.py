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

"""Utilities for MCP tools built on OneTwo.

Converts OneTwo content (a string or a `ChunkList`) into MCP content blocks, so
that MCP servers (e.g. over OneTwo indexes) return it the same way in
unstructured MCP tool results, see
https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-result:
text is a single text block, and multimodal content (e.g. a single PDF page,
whose content is an ``application/pdf`` chunk) is a block per chunk (text,
image, audio, or an embedded binary resource for anything else, such as PDFs),
so that its non-text content reaches the model as such rather than as text
placeholders (e.g. ``<application/pdf>``).

Also serves MCP servers in a background thread (`serve_in_background`), so that
a client can connect to them from the same process, e.g. in a colab.
"""

# TODO: Show that the OneTwo agent works with multimodal MCP
# servers in general (not only RAG ones), e.g. with a mock MCP server that
# returns these content blocks in the agent manual tests.

import base64
import dataclasses
import json
import threading
import time
from typing import Any
from urllib import parse

from mcp import types as mcp_types
from mcp.server.fastmcp import server
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import chunking
import uvicorn

# Prefix of the placeholder URIs of the embedded resources of the results, e.g.
# `onetwo-index://unused-uri/<docid>`. MCP requires a URI on every embedded
# resource, which is the only block type that can carry a PDF. The URI is not
# used: nothing resolves it, and it is unrelated to the URL the server is
# served on.
# TODO: b/578382212 - Make these URIs useful, e.g. by registering a resource
# template for them (`@mcp.resource('onetwo-index://documents/{docid}')`) so
# that clients can read a document by its URI.
_RESOURCE_URI_PREFIX = 'onetwo-index://unused-uri/'


def _media_block(
    *, data: bytes, mime_type: str, uri: str
) -> mcp_types.ContentBlock:
  """Returns non-text content as the MCP content block of its MIME type.

  Images and audio have their own block types. Any other binary content (e.g.
  `application/pdf`) is sent as an embedded binary resource.

  Args:
    data: The raw bytes of the content.
    mime_type: The MIME type of the content.
    uri: URI identifying the content, used for embedded resources.
  """
  encoded = base64.b64encode(data).decode('ascii')
  if mime_type.startswith('image/'):
    return mcp_types.ImageContent(
        type='image', data=encoded, mimeType=mime_type
    )
  if mime_type.startswith('audio/'):
    return mcp_types.AudioContent(
        type='audio', data=encoded, mimeType=mime_type
    )
  return mcp_types.EmbeddedResource(
      type='resource',
      resource=mcp_types.BlobResourceContents.model_validate(
          {'uri': uri, 'mimeType': mime_type, 'blob': encoded}
      ),
  )


async def content_blocks(
    content: str | content_lib.ChunkList,
    *,
    max_tokens: int,
    resource_id: str,
) -> list[mcp_types.ContentBlock]:
  """Returns `content` as MCP content blocks, in order.

  A string is a single text block. A `ChunkList` is a block per chunk.

  Only text is truncated, with the tokenizer registered as the `llm.tokenize`
  backend: media chunks (e.g. PDFs, images) are sent whole, so a tool may
  return whole files.

  Args:
    content: The content, e.g. the `content` of a `Document`.
    max_tokens: Maximum number of tokens of a string, or of each text chunk of
      a `ChunkList`. Negative disables truncation.
    resource_id: Identifier of the content (e.g. a docid), used in the URIs of
      the embedded resources.

  Raises:
    TypeError: If a chunk is neither text nor bytes (e.g. a PIL image).
  """
  # TODO: consider cropping multimodal content as well.
  if isinstance(content, str):
    content = content_lib.ChunkList([content_lib.Chunk(content)])
  uri = _RESOURCE_URI_PREFIX + parse.quote(resource_id, safe='')
  blocks: list[mcp_types.ContentBlock] = []
  for chunk in content:
    match chunk.content:
      case str() as text:
        # `truncate_string_to_max_tokens` slices `tokens[:max_tokens]`, so a
        # negative value would drop tokens from the end instead of disabling
        # truncation.
        if text and max_tokens >= 0:
          text = await chunking.truncate_string_to_max_tokens(text, max_tokens)
        if text:
          blocks.append(mcp_types.TextContent(type='text', text=text))
      case bytes() as data:
        blocks.append(
            _media_block(data=data, mime_type=chunk.content_type, uri=uri)
        )
      case _:
        raise TypeError(
            f'Unsupported content of {resource_id}: {type(chunk.content)}.'
        )
  return blocks


def json_block(value: Any) -> mcp_types.TextContent:
  """Returns `value` as a JSON text content block."""
  # TODO: Make the typing of the JSON headers of the documents
  # that MCP servers return more explicit (e.g. a dataclass with their doc_id,
  # title and score), instead of the free-form dicts passed here.
  return mcp_types.TextContent(type='text', text=json.dumps(value, default=str))


@dataclasses.dataclass(frozen=True)
class BackgroundMcpServer:
  """An MCP server running in a background thread of this process.

  Returned by `serve_in_background`.

  Attributes:
    url: The streamable-HTTP endpoint of the server, e.g.
      `http://localhost:8775/mcp`, to connect clients to (e.g.
      `mcp_tools.server(url)`).
  """

  url: str
  _server: uvicorn.Server = dataclasses.field(repr=False)
  _thread: threading.Thread = dataclasses.field(repr=False)

  def stop(self, timeout: float = 10.0) -> None:
    """Stops the server and frees its port.

    Args:
      timeout: How long to wait, in seconds, for the server to shut down.
    """
    self._server.should_exit = True
    self._thread.join(timeout)


def serve_in_background(
    mcp: server.FastMCP, *, startup_timeout: float = 30.0
) -> BackgroundMcpServer:
  """Serves `mcp` over streamable HTTP in a background thread.

  Unlike `mcp.run(transport='streamable-http')`, which blocks until the server
  is stopped, this returns as soon as the server accepts connections, so that
  a client can connect to it from the same process (e.g. an agent in a colab):

  ```python
    background_server = serve_in_background(mcp)
    agent = agent_lib.Agent(tools=[mcp_tools.server(background_server.url)])
    ...
    background_server.stop()
  ```

  The server is served on the host and port of `mcp.settings`, i.e. those it
  was created with.

  Args:
    mcp: The server to serve.
    startup_timeout: How long to wait, in seconds, for the server to start.

  Returns:
    The running server, with its URL and a way to stop it.

  Raises:
    RuntimeError: If the server fails to start, e.g. because its port is
      already in use.
    TimeoutError: If the server does not start within `startup_timeout`.
  """
  host, port = mcp.settings.host, mcp.settings.port
  # `loop='asyncio'`: the default, 'auto', uses uvloop if it can be imported,
  # which is not always usable (e.g. in some colab runtimes).
  uvicorn_server = uvicorn.Server(
      uvicorn.Config(
          mcp.streamable_http_app(),
          host=host,
          port=port,
          loop='asyncio',
          log_level=mcp.settings.log_level.lower(),
      )
  )
  thread = threading.Thread(
      target=uvicorn_server.run, name=f'mcp-server-{port}', daemon=True
  )
  thread.start()
  deadline = time.monotonic() + startup_timeout
  while not uvicorn_server.started:
    if not thread.is_alive():
      raise RuntimeError(
          f'The MCP server failed to start on {host}:{port}, e.g. because the'
          ' port is already in use. See the logs for details.'
      )
    if time.monotonic() > deadline:
      uvicorn_server.should_exit = True
      raise TimeoutError(
          f'The MCP server did not start on {host}:{port} within'
          f' {startup_timeout} seconds.'
      )
    time.sleep(0.05)
  return BackgroundMcpServer(
      url=f'http://{host}:{port}{mcp.settings.streamable_http_path}',
      _server=uvicorn_server,
      _thread=thread,
  )
