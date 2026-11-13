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

"""An HTTP client for MCP that survives a tool call failing over HTTP.

A server is meant to report a failed tool call inside a successful HTTP
response, as a JSON-RPC error. Some servers, or the proxies in front of them,
reject the call with an HTTP error status instead: 400 or 422 for arguments
they do not accept, 429 when rate limited, 502 from a failing gateway. The MCP
SDK's client does not handle that: it raises in a background task, which ends
the connection and cancels whoever opened it. An agent would lose its whole run
to one rejected call, without the model ever seeing why.

`client` returns the HTTP client the SDK would use, with one change: such a
response to a tool call is rewritten into the JSON-RPC error the server should
have sent, before the SDK reads it. The caller gets an ordinary `McpError`, and
the connection stays open for the calls after it. Every other response is left
alone, so a status that means the session itself is unusable, such as 401 or
403, still ends it at once. This module can be dropped once the SDK handles
HTTP errors itself.

```python
  async with mcp_http.client() as http_client:
    async with streamable_http.streamable_http_client(
        url, http_client=http_client
    ) as (read, write, _):
      ...
```
"""

import json
import logging

import httpx
from mcp import types as mcp_types

# The MCP SDK's own defaults: 30 seconds for most operations, and 5 minutes for
# reading, since results can arrive on a stream that stays open a long time.
# Connecting gets 3 minutes, so that a client process stalling (e.g. in garbage
# collection) does not fail connections to a local server.
_TIMEOUT = httpx.Timeout(30.0, connect=180.0, read=300.0)

# The statuses that say one call failed rather than the session: arguments the
# server rejects, rate limiting, and any failure on the server's side.
_CALL_REJECTED = frozenset({400, 422, 429})

# How much of an error body the model is shown. Proxies answer with whole pages.
_MAX_TEXT = 1000


def client(
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
  """Returns an HTTP client for an MCP session, to be used as a context manager.

  Args:
    transport: Where requests are sent. Defaults to the network, through any
      proxy the environment names; tests pass an in-process one.

  Returns:
    A client with the SDK's defaults, whose HTTP error responses to tool calls
    reach the session as JSON-RPC errors.
  """
  return _Client(follow_redirects=True, timeout=_TIMEOUT, transport=transport)


class _Client(httpx.AsyncClient):
  """Rewrites HTTP error responses to tool calls; passes the rest on.

  This overrides `send` rather than wrapping the transport so that everything
  httpx does below it is unchanged: proxies from the environment still apply,
  and authentication still sees the original status, to refresh a token on 401.
  """

  async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
    response = await super().send(request, **kwargs)
    status = response.status_code
    if status not in _CALL_REJECTED and status < 500:
      return response
    message = json.loads(request.content) if request.method == 'POST' else {}
    if message.get('method') != 'tools/call':
      return response
    try:
      await response.aread()
    finally:
      await response.aclose()
    logging.warning(
        'MCP tool call %s got HTTP %d; reporting it as an error.',
        message['params']['name'],
        status,
    )
    return _as_json_rpc(message['id'], request, response)


def _as_json_rpc(
    call_id: str | int, request: httpx.Request, response: httpx.Response
) -> httpx.Response:
  """Returns the JSON-RPC error the server should have answered a call with.

  Args:
    call_id: The id of the rejected tool call.
    request: The request that carried it.
    response: The server's HTTP error response, already read.
  """
  status = response.status_code
  error = mcp_types.JSONRPCError(
      jsonrpc='2.0',
      id=call_id,
      error=mcp_types.ErrorData(
          code=(
              mcp_types.INVALID_PARAMS
              if status in (400, 422)
              else mcp_types.INTERNAL_ERROR
          ),
          message=f'HTTP {status}: {response.text[:_MAX_TEXT]}',
      ),
  )
  return httpx.Response(
      200,
      json=error.model_dump(by_alias=True, exclude_none=True, mode='json'),
      request=request,
  )
