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

# Copyright 2026 Google LLC
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

r"""Binary to build an embedding-based document index.
"""

from typing import Sequence

from absl import app
from absl import flags
from onetwo.stdlib.retrieval import build_index_lib
from onetwo.stdlib.retrieval import rag_defaults

# Re-export build_index for backward compatibility.
build_index = build_index_lib.build_index

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    'browsecomp_plus',
    'Name of the dataset to load from the registry.',
    allow_override=True,
)

_DOCS_PATH = flags.DEFINE_string(
    'docs_path',
    None,
    'Path to a file (SSTable or JSONL) containing the documents to index.',
)

_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    None,
    'Directory where the built index will be saved.',
    required=True,
)

_EMBEDDING_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'embedding_backend_class_name',
    rag_defaults.DEFAULT_EMBED_BACKEND_CLASS_NAME,
    'Name of the embedding backend class.',
)

_EMBEDDING_MODEL_NAME = flags.DEFINE_string(
    'embedding_model_name',
    rag_defaults.DEFAULT_EMBED_MODEL_NAME,
    'Name of the embedding model.',
)

_INDEX_TYPE = flags.DEFINE_enum(
    'index_type',
    'embedding',
    ['embedding', 'hipporag'],
    'Type of index to build ("embedding" or "hipporag").',
)

_GENERATION_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'generation_backend_class_name',
    rag_defaults.DEFAULT_INDEX_GENERATE_BACKEND_CLASS_NAME,
    'Name of the generation backend class (for GraphRAG/HippoRAG).',
)

_GENERATION_MODEL_NAME = flags.DEFINE_string(
    'generation_model_name',
    rag_defaults.DEFAULT_INDEX_GENERATE_MODEL_NAME,
    'Name of the generation model (for GraphRAG/HippoRAG).',
)

_SIMILARITY_THRESHOLD = flags.DEFINE_float(
    'similarity_threshold',
    0.8,
    'Minimum similarity threshold required for semantic similarity edges. This'
    ' is only used for HippoRAG (when --index_type=hipporag) to construct'
    ' similarity edges.',
)

_IGNORE_TRIPLES = flags.DEFINE_boolean(
    'ignore_triples',
    True,
    'Whether to ignore relational triples during HippoRAG graph construction'
    ' and indexing.',
)

_IGNORE_SIMILARITY_EDGES = flags.DEFINE_boolean(
    'ignore_similarity_edges',
    True,
    'Whether to ignore semantic similarity edges during HippoRAG graph'
    ' construction.',
)

_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size',
    None,
    'Maximum tokens or words per chunk. If None, no chunking is performed.',
)

_CHUNKING_STRATEGY = flags.DEFINE_enum(
    'chunking_strategy',
    'words',
    ['tokens', 'words'],
    'Chunking strategy to use ("words" for ChunkByMaxWords or "tokens" for'
    ' ChunkByMaxTokens).',
)

_OVERLAP_WINDOW = flags.DEFINE_integer(
    'overlap_window',
    0,
    'Number of tokens or words to overlap between chunks.',
)

_TOKENIZE_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'tokenize_backend_class_name',
    rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME,
    'Name of the tokenizer backend class (when --chunking_strategy=tokens).',
)

_TOKENIZE_MODEL_NAME = flags.DEFINE_string(
    'tokenize_model_name',
    rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME,
    'Name of the tokenizer model (when --chunking_strategy=tokens).',
)

_FORMAT = flags.DEFINE_string(
    'format',
    None,
    (
        'Format string for document formatting. If "None" (string) or empty,'
        ' no formatting is done.'
    ),
)

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    '{}',
    (
        'JSON string containing runtime arguments for the backend (e.g. API'
        ' key, project).'
    ),
    allow_override=True,
)

_MAX_DOCUMENTS = flags.DEFINE_integer(
    'max_documents',
    None,
    'Maximum number of documents to load and index (useful for testing).',
)

_MAX_RETRIES = flags.DEFINE_integer(
    'max_retries',
    5,
    'Maximum number of retries for backend requests.',
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    60000,
    'HTTP timeout in milliseconds for backend requests (default: 60s).',
)

_SERIALIZER_TYPE = flags.DEFINE_enum(
    'serializer_type',
    'simple',
    ['simple', 'internal'],
    'Serializer type to use for saving the index.',
)

_EMBED_KWARGS = flags.DEFINE_string(
    'embed_kwargs',
    '{}',
    (
        'JSON string containing keyword arguments for the embedding backend.'
    ),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  build_index_lib.build_index(
      dataset_name=_DATASET_NAME.value,
      max_documents=_MAX_DOCUMENTS.value,
      docs_path=_DOCS_PATH.value,
      index_type=_INDEX_TYPE.value,
      output_directory=_OUTPUT_DIRECTORY.value,
      embedding_backend_class_name=_EMBEDDING_BACKEND_CLASS_NAME.value,
      embedding_model_name=_EMBEDDING_MODEL_NAME.value,
      embed_kwargs=_EMBED_KWARGS.value,
      chunk_size=_CHUNK_SIZE.value,
      chunking_strategy=_CHUNKING_STRATEGY.value,
      overlap_window=_OVERLAP_WINDOW.value,
      tokenize_backend_class_name=_TOKENIZE_BACKEND_CLASS_NAME.value,
      tokenize_model_name=_TOKENIZE_MODEL_NAME.value,
      format_str=_FORMAT.value,
      generation_backend_class_name=_GENERATION_BACKEND_CLASS_NAME.value,
      generation_model_name=_GENERATION_MODEL_NAME.value,
      similarity_threshold=_SIMILARITY_THRESHOLD.value,
      ignore_triples=_IGNORE_TRIPLES.value,
      ignore_similarity_edges=_IGNORE_SIMILARITY_EDGES.value,
      serializer_type=_SERIALIZER_TYPE.value,
      runtime_kwargs=_RUNTIME_KWARGS.value,
      max_retries=_MAX_RETRIES.value,
      http_timeout=_HTTP_TIMEOUT.value,
  )


if __name__ == '__main__':
  app.run(main)
