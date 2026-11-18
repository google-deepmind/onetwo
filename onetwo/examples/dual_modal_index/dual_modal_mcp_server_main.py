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

r"""Thin CLI launcher for the OneTwo Dual-Modal Index MCP Server.

Starts an MCP server exposing search_text_chunks, search_images, and
get_document over a saved DualModalDocumentIndex.

Example usage:
  python -m onetwo.examples.dual_modal_index.dual_modal_mcp_server_main \
      --index_path=/path/to/dual_modal_index \
      --transport=streamable-http \
      --port=8775

Clients then connect to http://localhost:8775/mcp.
"""

from collections.abc import Sequence
import json
from typing import Literal, cast

from absl import app
from absl import flags
from onetwo.examples.dual_modal_index import dual_modal_mcp_server

_INDEX_PATH = flags.DEFINE_string(
    'index_path',
    None,
    'Directory of a saved DualModalDocumentIndex.',
    required=True,
)

_MAX_RESULTS = flags.DEFINE_integer(
    'max_results',
    5,
    'Number of top results returned by each search tool.',
)

_SNIPPET_MAX_CHARS = flags.DEFINE_integer(
    'snippet_max_chars',
    1200,
    'Maximum characters per text snippet (~300 tokens). -1 means no'
    ' truncation.',
)

_HOST = flags.DEFINE_string(
    'host',
    'localhost',
    'Host/interface to bind the server on.',
)

_PORT = flags.DEFINE_integer(
    'port',
    8775,
    'Port to listen on.',
)

_TRANSPORT = flags.DEFINE_enum(
    'transport',
    'streamable-http',
    ['stdio', 'streamable-http'],
    'MCP transport protocol ("stdio" or "streamable-http").',
)

_EXTRA_BACKEND_KWARGS = flags.DEFINE_string(
    'extra_backend_kwargs',
    None,
    'Optional JSON-encoded dictionary of extra keyword arguments for the'
    ' backends of the loaded index (e.g. an API key). Usually not needed:'
    ' credentials are read from the environment (e.g. GOOGLE_API_KEY).',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected command line arguments: {argv[1:]}')

  dual_modal_mcp_server.start_mcp(
      _INDEX_PATH.value,
      transport=cast(Literal['stdio', 'streamable-http'], _TRANSPORT.value),
      host=_HOST.value,
      port=_PORT.value,
      extra_backend_kwargs=(
          json.loads(_EXTRA_BACKEND_KWARGS.value)
          if _EXTRA_BACKEND_KWARGS.value
          else None
      ),
      extra_mcp_config={
          'max_results': _MAX_RESULTS.value,
          # A negative --snippet_max_chars disables truncation.
          'snippet_max_chars': (
              _SNIPPET_MAX_CHARS.value
              if _SNIPPET_MAX_CHARS.value >= 0
              else None
          ),
      },
  )


if __name__ == '__main__':
  app.run(main)
