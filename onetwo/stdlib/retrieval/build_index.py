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

import json
from typing import Any, Sequence

from absl import app
from absl import flags
from onetwo.backends import backend_registry
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building

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
    'GoogleGenAIAPI',
    'Name of the embedding backend class.',
)

_EMBEDDING_MODEL_NAME = flags.DEFINE_string(
    'embedding_model_name',
    'gemini-embedding-2',
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
    'GoogleGenAIAPI',
    'Name of the generation backend class (for GraphRAG/HippoRAG).',
)

_GENERATION_MODEL_NAME = flags.DEFINE_string(
    'generation_model_name',
    'gemini-3-flash-preview',
    'Name of the generation model (for GraphRAG/HippoRAG).',
)

_SIMILARITY_THRESHOLD = flags.DEFINE_float(
    'similarity_threshold',
    0.8,
    'Minimum similarity threshold required for semantic similarity edges. This'
    ' is only used for HippoRAG (when --index_type=hipporag) to construct'
    ' similarity edges.',
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

_EMBED_KWARGS = flags.DEFINE_string(
    'embed_kwargs',
    '{}',
    (
        'JSON string containing keyword arguments for the embedding backend.'
    ),
)


def build_index(
    dataset_name: str,
    output_directory: str,
    runtime_kwargs: dict[str, Any],
    max_documents: int | None = None,
    docs_path: str | None = None,
    index_type: str = 'embedding',
    embedding_backend_class_name: str = 'GoogleGenAIAPI',
    embedding_model_name: str = 'gemini-embedding-2',
    embed_kwargs: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunking_strategy: str = 'words',
    overlap_window: int = 0,
    format_str: str | None = None,
    generation_backend_class_name: str = 'GoogleGenAIAPI',
    generation_model_name: str = 'gemini-3-flash-preview',
    similarity_threshold: float = 0.8,
) -> str:
  """Builds a document index for the specified dataset.

  Args:
    dataset_name: Name of the dataset to load from the registry.
    output_directory: Directory pattern where the built index will be saved.
    runtime_kwargs: Arguments for backends (e.g. API key).
    max_documents: Max documents to index.
    docs_path: Optional path to documents.
    index_type: Type of index ('embedding' or 'hipporag').
    embedding_backend_class_name: Embedding backend class name.
    embedding_model_name: Embedding model name.
    embed_kwargs: Keyword arguments for the embedding backend.
    chunk_size: Max tokens or words per chunk.
    chunking_strategy: Chunking strategy ('tokens' or 'words').
    overlap_window: Tokens or words to overlap between chunks.
    format_str: Format string for document formatting.
    generation_backend_class_name: Generation backend class name for HippoRAG.
    generation_model_name: Generation model name for HippoRAG.
    similarity_threshold: Similarity threshold for HippoRAG.

  Returns:
    The output directory path where the index was saved.
  """
  if docs_path:
    if dataset_name not in docs_path:
      raise ValueError(
          f'Dataset name "{dataset_name}" not found in docs path "{docs_path}"'
      )
    docs = docs_path
    print(f'Using documents from path: {docs}')
  else:
    print(f'Loading dataset {dataset_name}...')
    docs = datasets.load_documents(
        dataset_name,
        max_documents=max_documents,
    )
    print(f'Loaded {len(docs)} documents.')

  # Construct spec
  backend_kwargs = {}
  if embed_kwargs:
    backend_kwargs['embed_kwargs'] = embed_kwargs
  embed_backend_config = backend_registry.EmbedBackendConfig(
      backend_class_name=embedding_backend_class_name,
      model_name=embedding_model_name,
      kwargs=backend_kwargs,
  )
  rt_kwargs = dict(runtime_kwargs) if runtime_kwargs else {}

  if index_type == 'hipporag':
    generate_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name=generation_backend_class_name,
        model_name=generation_model_name,
    )
    spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        generate_backend_config=generate_backend_config,
        max_chunk_size=chunk_size,
        chunking_strategy=chunking_strategy,
        overlap_window=overlap_window,
        format_str=format_str,
        similarity_threshold=similarity_threshold,
        runtime_kwargs=rt_kwargs,
    )
  else:
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        max_chunk_size=chunk_size,
        chunking_strategy=chunking_strategy,
        overlap_window=overlap_window,
        format_str=format_str,
        runtime_kwargs=rt_kwargs,
    )

  out_dir = index_building.resolve_index_output_directory(
      output_directory_pattern=output_directory,
      dataset_name=dataset_name,
      spec=spec,
  )
  print(f'Will save the index to output_directory: {out_dir}')

  # Build index
  print('Building index...')
  _ = ot_executing.run(
      index_building.build_index(
          spec=spec,
          docs=docs,
          output_dir=out_dir,
          runner='local',
      )
  )
  print(f'Index built and saved to {out_dir}.')
  return out_dir


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  format_str = None if _FORMAT.value in ('None', '') else _FORMAT.value
  runtime_kwargs = json.loads(_RUNTIME_KWARGS.value)
  embed_kwargs = json.loads(_EMBED_KWARGS.value)

  build_index(
      dataset_name=_DATASET_NAME.value,
      max_documents=_MAX_DOCUMENTS.value,
      docs_path=_DOCS_PATH.value,
      index_type=_INDEX_TYPE.value,
      output_directory=_OUTPUT_DIRECTORY.value,
      embedding_backend_class_name=_EMBEDDING_BACKEND_CLASS_NAME.value,
      embedding_model_name=_EMBEDDING_MODEL_NAME.value,
      embed_kwargs=embed_kwargs,
      chunk_size=_CHUNK_SIZE.value,
      chunking_strategy=_CHUNKING_STRATEGY.value,
      overlap_window=_OVERLAP_WINDOW.value,
      format_str=format_str,
      generation_backend_class_name=_GENERATION_BACKEND_CLASS_NAME.value,
      generation_model_name=_GENERATION_MODEL_NAME.value,
      similarity_threshold=_SIMILARITY_THRESHOLD.value,
      runtime_kwargs=runtime_kwargs,
  )


if __name__ == '__main__':
  app.run(main)
