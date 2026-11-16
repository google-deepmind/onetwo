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

Clients then connect to http://localhost:8776/mcp.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from onetwo.stdlib.retrieval.graphrag import mcp_server

_INDEX_PATH = flags.DEFINE_string(
    'index_path',
    None,
    'Path to the pre-built HippoRAG graph index directory.',
    required=True,
)

_PORT = flags.DEFINE_integer('port', 8776, 'Port to listen on.')

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

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    None,
    'Optional JSON-encoded dictionary of runtime kwargs passed to the index'
    ' loader.',
)


def main(argv: Sequence[str]) -> None:
  del argv
  mcp_server.run_server(
      index_path=_INDEX_PATH.value,
      port=_PORT.value,
      transport=_TRANSPORT.value,
      host=_HOST.value,
      snippet_max_tokens=_SNIPPET_MAX_TOKENS.value,
      index_runtime_kwargs=_RUNTIME_KWARGS.value,
      tokenize_backend_config=_TOKENIZE_BACKEND_CONFIG.value,
  )


if __name__ == '__main__':
  app.run(main)
