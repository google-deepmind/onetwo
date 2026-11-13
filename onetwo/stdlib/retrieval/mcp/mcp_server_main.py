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

r"""Launch script for the OneTwo Index MCP server.

Starts an MCP server that exposes a pre-built OneTwo EmbeddingBasedDocumentIndex
for semantic search.  This is a minimal example -- see ``mcp_server.py`` in
this directory for the server implementation.

Example usage:
  python -m onetwo.stdlib.retrieval.mcp.mcp_server_main \
      --index_path=/path/to/index \
      --runtime_kwargs='{"api_key": "YOUR_API_KEY"}' \
      --transport=streamable-http

Clients then connect to http://localhost:8775/mcp.
"""

import json
from typing import cast, Literal, Sequence

from absl import app
from absl import flags
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval.mcp import mcp_server

_INDEX_PATH = flags.DEFINE_string(
    'index_path',
    None,
    'Path to the pre-built embedding index (e.g., a CNS path).',
    required=True,
)

_MAX_RESULTS = flags.DEFINE_integer(
    'max_results',
    5,
    'Default number of top results to return from the search tool.',
)

_SNIPPET_MAX_TOKENS = flags.DEFINE_integer(
    'snippet_max_tokens',
    512,
    (
        'Maximum number of tokens for each result snippet. -1 means no'
        ' truncation (return full text).'
    ),
)

_PORT = flags.DEFINE_integer('port', 8775, 'Port to listen on.')

_HOST = flags.DEFINE_string(
    'host', 'localhost', 'Host/interface to bind the server on.'
)

_TRANSPORT = flags.DEFINE_enum(
    'transport',
    'stdio',
    ['stdio', 'sse', 'streamable-http'],
    'MCP transport protocol.  "stdio" communicates over stdin/stdout and is'
    ' used when an MCP client (e.g. Jetski) launches the server as a'
    ' subprocess.  "streamable-http" starts a standalone HTTP endpoint that'
    ' clients connect to via a URL (e.g. http://localhost:8775/mcp).'
    '  "sse" is the legacy Server-Sent Events transport (deprecated by the'
    ' MCP spec; prefer streamable-http).',
)

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    None,
    (
        'JSON-encoded dictionary of runtime kwargs of the loaded index (e.g.'
        ' \'{"api_key": "YOUR_API_KEY"}\').'
    ),
)

_TOKENIZER_MODEL_NAME = flags.DEFINE_string(
    'tokenizer_model_name',
    rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME,
    (
        'SentencePiece tokenizer model used to truncate snippets: a key of'
        ' `sentencepiece_tokenizer.TOKENIZER_PATHS` or the path of a model'
        ' file. Unused when --snippet_max_tokens=-1.'
    ),
)


def main(argv: Sequence[str]) -> None:
  del argv
  transport = cast(Literal['stdio', 'sse', 'streamable-http'], _TRANSPORT.value)

  runtime_kwargs = None
  if _RUNTIME_KWARGS.value:
    runtime_kwargs = json.loads(_RUNTIME_KWARGS.value)

  mcp = mcp_server.create_server(
      index_path=_INDEX_PATH.value,
      port=_PORT.value,
      host=_HOST.value,
      transport=transport,
      runtime_kwargs=runtime_kwargs,
      extra_mcp_config={
          'max_results': _MAX_RESULTS.value,
          # A negative --snippet_max_tokens disables truncation.
          'snippet_max_tokens': (
              _SNIPPET_MAX_TOKENS.value
              if _SNIPPET_MAX_TOKENS.value >= 0
              else None
          ),
          'tokenize_backend_config': backend_registry.TokenizeBackendConfig(
              backend_class_name='SentencePieceTokenizer',
              model_name=_TOKENIZER_MODEL_NAME.value,
          ),
      },
  )
  mcp.run(transport=transport)


if __name__ == '__main__':
  app.run(main)
