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

"""Tests serving MCP servers in the background with `serve_in_background`."""

import asyncio
import socket
from typing import Any

from absl.testing import absltest
from mcp.server.fastmcp import server
from onetwo.stdlib.tool_use import mcp_tools
from onetwo.stdlib.tool_use import mcp_tools_utils
import portpicker

_HOST = '127.0.0.1'


def _create_server(port: int) -> server.FastMCP:
  """Returns a server with a single `echo` tool, on `port`."""
  mcp = server.FastMCP('Test server', host=_HOST, port=port)

  @mcp.tool()
  def echo(text: str) -> str:
    """Returns `text`."""
    return text

  del echo  # Registered via decorator.
  return mcp


def _port_is_free(port: int) -> bool:
  """Returns whether `port` can be bound on `_HOST`."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
      sock.bind((_HOST, port))
    except OSError:
      return False
    return True


async def _call_echo(url: str, text: str) -> Any:
  """Calls the `echo` tool of the server at `url`."""
  async with mcp_tools.mcp_session(url) as tools:
    (echo,) = tools
    return await echo(text=text)


class ServeInBackgroundTest(absltest.TestCase):

  def test_serves_until_stopped(self):
    port = portpicker.pick_unused_port()

    background_server = mcp_tools_utils.serve_in_background(
        _create_server(port)
    )

    with self.subTest('has_the_url_of_the_server'):
      self.assertEqual(background_server.url, f'http://{_HOST}:{port}/mcp')
    with self.subTest('is_serving_when_returned'):
      # FastMCP wraps the `str` returned by `echo` as `{'result': ...}`.
      self.assertEqual(
          asyncio.run(_call_echo(background_server.url, 'hello')),
          {'result': 'hello'},
      )
    with self.subTest('stop_frees_the_port'):
      background_server.stop()
      self.assertTrue(_port_is_free(port))

  def test_port_in_use_raises(self):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind((_HOST, 0))
      sock.listen()
      port = sock.getsockname()[1]

      with self.assertRaisesRegex(RuntimeError, 'failed to start'):
        mcp_tools_utils.serve_in_background(_create_server(port))


if __name__ == '__main__':
  absltest.main()
