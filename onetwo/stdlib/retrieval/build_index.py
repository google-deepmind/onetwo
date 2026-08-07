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

Run as:
bazel run //third_party/py/onetwo/stdlib/retrieval:build_index -- \
  --dataset_name=browsecomp_plus \
  --max_documents=5 \
  --embedding_backend_class_name=GoogleGenAIAPI \
  --embedding_model_name=gemini-embedding-2 \
  --runtime_kwargs='{"api_key": "YOUR_API_KEY", "threadpool_size": 20}'
"""

import json
from typing import Sequence

from absl import app
from absl import flags
from onetwo.backends import backend_registry
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    None,
    'Name of the dataset to load from the registry.',
    required=True,
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
    'Maximum tokens per chunk. If None, no chunking is performed.',
)

_OVERLAP_WINDOW = flags.DEFINE_integer(
    'overlap_window',
    0,
    'Number of tokens to overlap between chunks.',
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
)

_MAX_DOCUMENTS = flags.DEFINE_integer(
    'max_documents',
    None,
    'Maximum number of documents to load and index (useful for testing).',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f'Loading dataset {_DATASET_NAME.value}...')
  docs = datasets.load_documents(
      _DATASET_NAME.value,
      max_documents=_MAX_DOCUMENTS.value,
  )
  print(f'Loaded {len(docs)} documents.')

  # Construct spec
  embed_backend_config = backend_registry.EmbedBackendConfig(
      backend_class_name=_EMBEDDING_BACKEND_CLASS_NAME.value,
      model_name=_EMBEDDING_MODEL_NAME.value,
  )
  format_str = None if _FORMAT.value in ('None', '') else _FORMAT.value
  runtime_kwargs = json.loads(_RUNTIME_KWARGS.value)

  if _INDEX_TYPE.value == 'hipporag':
    generate_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name=_GENERATION_BACKEND_CLASS_NAME.value,
        model_name=_GENERATION_MODEL_NAME.value,
    )
    spec = index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        generate_backend_config=generate_backend_config,
        max_tokens_per_chunk=_CHUNK_SIZE.value,
        overlap_window=_OVERLAP_WINDOW.value,
        format_str=format_str,
        similarity_threshold=_SIMILARITY_THRESHOLD.value,
        runtime_kwargs=runtime_kwargs,
    )
  else:
    spec = index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        max_tokens_per_chunk=_CHUNK_SIZE.value,
        overlap_window=_OVERLAP_WINDOW.value,
        format_str=format_str,
        runtime_kwargs=runtime_kwargs,
    )

  output_directory = index_building.resolve_index_output_directory(
      output_directory_pattern=_OUTPUT_DIRECTORY.value,
      dataset_name=_DATASET_NAME.value,
      spec=spec,
  )
  print(f'Will save the index to output_directory: {output_directory}')

  # Build index
  print('Building index...')
  _ = ot_executing.run(
      index_building.build_index(
          spec=spec,
          docs=docs,
          output_dir=output_directory,
          runner='local',
      )
  )
  print(f'Index built and saved to {output_directory}.')


if __name__ == '__main__':
  app.run(main)
