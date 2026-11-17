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

r"""Launch script for the GraphRAG MCP server.

Starts an MCP server exposing a pre-built HippoRAG graph index for semantic
entity search, graph traversal, pathfinding, and passage retrieval.

Example usage:
  python -m onetwo.stdlib.retrieval.graphrag.mcp_server_main \
      --index_path=/path/to/hipporag/index \
      --transport=streamable-http

Clients then connect to http://localhost:8775/mcp.
"""

from collections.abc import Sequence
import json
from typing import cast, Literal

from absl import app
from absl import flags
from onetwo.stdlib.retrieval.graphrag import mcp_server

_INDEX_PATH = flags.DEFINE_string(
    'index_path',
    None,
    'Path to the pre-built HippoRAG graph index directory.',
    required=True,
)

_PORT = flags.DEFINE_integer('port', 8775, 'Port to listen on.')

_SNIPPET_MAX_TOKENS = flags.DEFINE_integer(
    'snippet_max_tokens',
    512,
    'Maximum length of each result snippet, in tokenizer tokens if'
    ' --tokenize_backend_config is set and in words otherwise. -1 means no'
    ' truncation.',
)

_TOKENIZE_BACKEND_CONFIG = flags.DEFINE_string(
    'tokenize_backend_config',
    None,
    'Optional JSON-encoded `TokenizeBackendConfig` (e.g.'
    ' \'{"backend_class_name": "SentencePieceTokenizer",'
    ' "model_name": "GEMINI3_G145"}\') used to truncate snippets.',
)

_TRANSPORT = flags.DEFINE_enum(
    'transport',
    'stdio',
    ['stdio', 'streamable-http'],
    'MCP transport protocol ("stdio" or "streamable-http").',
)

_HOST = flags.DEFINE_string(
    'host',
    'localhost',
    'Host/interface to bind the server on.',
)

_EXTRA_BACKEND_KWARGS = flags.DEFINE_string(
    'extra_backend_kwargs',
    None,
    'Optional JSON-encoded dictionary of extra keyword arguments for the'
    ' backends of the loaded index (e.g. an API key). Usually not needed:'
    ' credentials are read from the environment (e.g. GOOGLE_API_KEY).',
)


def main(argv: Sequence[str]) -> None:
  del argv
  mcp_server.start_mcp(
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
          # A negative --snippet_max_tokens disables truncation.
          'snippet_max_tokens': (
              _SNIPPET_MAX_TOKENS.value
              if _SNIPPET_MAX_TOKENS.value >= 0
              else None
          ),
          'tokenize_backend_config': (
              json.loads(_TOKENIZE_BACKEND_CONFIG.value)
              if _TOKENIZE_BACKEND_CONFIG.value
              else None
          ),
      },
  )


if __name__ == '__main__':
  app.run(main)
