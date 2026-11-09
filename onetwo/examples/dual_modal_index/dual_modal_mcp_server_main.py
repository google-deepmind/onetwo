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

Starts an MCP server exposing `search_text_chunks`, `search_images`, and
`get_document` over pre-built OneTwo text and image indices saved with
`preserve_original_docs=True`.

Example usage:
  python -m onetwo.examples.dual_modal_index.dual_modal_mcp_server_main \
      --text_index_path=/path/to/text_index \
      --image_index_path=/path/to/image_index \
      --transport=streamable-http \
      --port=8775
"""

from collections.abc import Sequence
import json
from typing import Literal, cast

from absl import app
from absl import flags
from onetwo.examples.dual_modal_index import dual_modal_mcp_server

_TEXT_INDEX_PATH = flags.DEFINE_string(
    'text_index_path',
    '',
    'Path to the pre-built text embedding index.',
)

_IMAGE_INDEX_PATH = flags.DEFINE_string(
    'image_index_path',
    '',
    'Path to the pre-built image embedding index.',
)

_MAX_RESULTS = flags.DEFINE_integer(
    'max_results',
    5,
    'Number of top results returned by each search tool.',
)

_SNIPPET_MAX_CHARS = flags.DEFINE_integer(
    'snippet_max_chars',
    1200,
    'Maximum characters per text snippet (~300 tokens; -1 disables'
    ' truncation).',
)

_INCLUDE_GET_DOCUMENT = flags.DEFINE_boolean(
    'include_get_document',
    True,
    'If True, register get_document to retrieve full unchunked documents.',
)

_HOST = flags.DEFINE_string(
    'host',
    'localhost',
    'Host/interface to bind the HTTP/SSE server on.',
)

_PORT = flags.DEFINE_integer(
    'port',
    8775,
    'Port to listen on for HTTP/SSE transports.',
)

_TRANSPORT = flags.DEFINE_enum(
    'transport',
    'streamable-http',
    ['stdio', 'sse', 'streamable-http'],
    'MCP transport protocol ("stdio", "sse", or "streamable-http").',
)

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    '',
    'Optional JSON-encoded dictionary of runtime kwargs forwarded to'
    ' index_building.load_index.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected command line arguments: {argv[1:]}')
  if not _TEXT_INDEX_PATH.value and not _IMAGE_INDEX_PATH.value:
    raise app.UsageError(
        'At least one of --text_index_path or --image_index_path must be set.'
    )

  runtime_kwargs = (
      json.loads(_RUNTIME_KWARGS.value) if _RUNTIME_KWARGS.value else None
  )

  mcp = dual_modal_mcp_server.create_dual_modal_server(
      text_index_path=_TEXT_INDEX_PATH.value,
      image_index_path=_IMAGE_INDEX_PATH.value,
      max_results=_MAX_RESULTS.value,
      snippet_max_chars=_SNIPPET_MAX_CHARS.value,
      include_get_document=_INCLUDE_GET_DOCUMENT.value,
      host=_HOST.value,
      port=_PORT.value,
      transport=_TRANSPORT.value,
      index_runtime_kwargs=runtime_kwargs,
  )
  # _TRANSPORT.value is typed as str, while mcp.run expects a Literal.
  mcp.run(
      transport=cast(
          Literal['stdio', 'sse', 'streamable-http'], _TRANSPORT.value
      )
  )


if __name__ == '__main__':
  app.run(main)
