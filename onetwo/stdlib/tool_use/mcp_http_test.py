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

"""Tests for `mcp_http`, against a real MCP server running in-process."""

import asyncio
from collections.abc import Callable
import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import httpx
from mcp import types as mcp_types
from mcp.client import session as client_session
from mcp.client import streamable_http
from mcp.server.fastmcp import server as fastmcp_server
from mcp.shared import exceptions as mcp_exceptions
from onetwo.stdlib.tool_use import mcp_http

# Requests never leave the process; the host is one the server accepts.
_URL = 'http://127.0.0.1:8000/mcp'

# A message the server receives, as parsed JSON-RPC.
_Message = dict[str, Any]


def _server() -> fastmcp_server.FastMCP:
  """Returns a server with two tools that would both succeed if called."""
  # Answers with plain JSON, and keeps no session, so that every message is one
  # self-contained HTTP exchange: the in-process transport cannot stream.
  mcp = fastmcp_server.FastMCP(
      'Test Server', json_response=True, stateless_http=True
  )

  @mcp.tool(name='good', description='Always accepted.')
  def good(word: str) -> str:
    return word

  @mcp.tool(name='bad', description='Rejected with an HTTP error.')
  def bad(word: str) -> str:
    return word

  del good, bad  # Registered by the decorator.
  return mcp


def _rejecting(app: Any, rejects: Callable[[_Message], bool], status: int):
  """Returns `app`, answering the messages `rejects` picks with `status`.

  This is what a server or proxy that rejects requests at the HTTP level does,
  instead of answering with a JSON-RPC error.

  Args:
    app: The server's ASGI app.
    rejects: Whether a posted message is to be rejected.
    status: The HTTP status to reject it with.
  """

  async def middleware(scope, receive, send):
    if scope['type'] != 'http' or scope['method'] != 'POST':
      return await app(scope, receive, send)
    body, more = b'', True
    while more:
      part = await receive()
      body += part.get('body', b'')
      more = part.get('more_body', False)
    if rejects(json.loads(body)):
      await send({'type': 'http.response.start', 'status': status})
      await send({'type': 'http.response.body', 'body': b'Rejected'})
      return
    replayed = False

    async def replay():
      nonlocal replayed
      if replayed:
        return await receive()
      replayed = True
      return {'type': 'http.request', 'body': body}

    return await app(scope, replay, send)

  return middleware


def _calls_to(name: str) -> Callable[[_Message], bool]:
  return lambda message: (
      message.get('method') == 'tools/call'
      and message['params']['name'] == name
  )


async def _call_in_turn(
    rejects: Callable[[_Message], bool], status: int, names: list[str]
) -> list[Any]:
  """Calls tools in turn on one session, and returns each result or error."""
  mcp = _server()
  transport = httpx.ASGITransport(
      app=_rejecting(mcp.streamable_http_app(), rejects, status)
  )
  outcomes = []
  async with mcp.session_manager.run():
    async with mcp_http.client(transport) as http_client:
      async with streamable_http.streamable_http_client(
          _URL, http_client=http_client
      ) as (read, write, _):
        async with client_session.ClientSession(read, write) as session:
          await session.initialize()
          for name in names:
            try:
              await session.call_tool(name, {'word': name})
              outcomes.append('ok')
            except mcp_exceptions.McpError as error:
              outcomes.append(error.error)
  return outcomes


class McpHttpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bad_request', 400, mcp_types.INVALID_PARAMS),
      ('unprocessable', 422, mcp_types.INVALID_PARAMS),
      ('rate_limited', 429, mcp_types.INTERNAL_ERROR),
      ('bad_gateway', 502, mcp_types.INTERNAL_ERROR),
  )
  def test_a_rejected_call_fails_alone_and_the_session_carries_on(
      self, status, code
  ):
    with self.assertLogs(level='WARNING'):
      before, rejected, after = asyncio.run(
          _call_in_turn(_calls_to('bad'), status, ['good', 'bad', 'good'])
      )

    self.assertEqual('ok', before)
    self.assertEqual(
        mcp_types.ErrorData(code=code, message=f'HTTP {status}: Rejected'),
        rejected,
    )
    self.assertEqual('ok', after)

  @parameterized.parameters(401, 403)
  def test_a_call_the_session_is_not_allowed_still_ends_it(self, status):
    # Every later call would be refused too, so failing at once is kinder to
    # the caller than letting a model retry against a session that is dead.
    with self.assertRaises(BaseExceptionGroup) as raised:
      asyncio.run(_call_in_turn(_calls_to('bad'), status, ['good', 'bad']))

    self.assertIsNotNone(raised.exception.subgroup(httpx.HTTPStatusError))


if __name__ == '__main__':
  absltest.main()
