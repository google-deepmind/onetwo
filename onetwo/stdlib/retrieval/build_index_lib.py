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

"""Library to build an embedding-based document index."""

from collections.abc import Mapping
import json
from typing import Any

from onetwo.backends import backend_registry
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval.graphrag import hipporag_index_specs


# TODO: Consider merging max_retries and http_timeout
# into runtime_kwargs.
def build_index(
    dataset_name: str,
    output_directory: str,
    runtime_kwargs: Mapping[str, Any] | str,
    max_documents: int | None = None,
    input_dir: str | None = None,
    index_type: str = 'embedding',
    embedding_backend_class_name: str = (
        rag_defaults.DEFAULT_EMBED_BACKEND_CLASS_NAME
    ),
    embedding_model_name: str = rag_defaults.DEFAULT_EMBED_MODEL_NAME,
    embed_kwargs: Mapping[str, Any] | str | None = None,
    chunk_size: int | None = None,
    chunking_strategy: str = 'words',
    overlap_window: int = 0,
    tokenize_backend_class_name: str = (
        rag_defaults.DEFAULT_TOKENIZE_BACKEND_CLASS_NAME
    ),
    tokenize_model_name: str = rag_defaults.DEFAULT_TOKENIZE_MODEL_NAME,
    format_str: str | None = None,
    serializer_type: str = 'simple',
    max_retries: int | None = 5,
    http_timeout: int | None = 60000,
) -> str:
  """Builds and saves a document index for the specified dataset.

  Args:
    dataset_name: Name of the dataset. Unless `input_dir` is set, documents are
      loaded from the dataset registry under this name. Also used in the output
      directory path.
    output_directory: Directory pattern where the built index will be saved.
    runtime_kwargs: Arguments for backends (e.g. API key).
    max_documents: Max documents to index.
    input_dir: Optional path to a folder of files of any type (text, PDF, image,
      audio, video) to index, loaded with `MultimodalDatasetLoader`.
    index_type: Type of index ('embedding' or 'hipporag'). 'hipporag' uses the
      default settings of `SimpleRewritingHippoRAGDocumentIndexSpec`, including
      its default generation model (`generate_backend_config`); to change them
      (e.g. `similarity_threshold` or the generation model), pass that spec to
      `build_index_from_spec` instead.
    embedding_backend_class_name: Embedding backend class name.
    embedding_model_name: Embedding model name.
    embed_kwargs: Keyword arguments for the embedding backend.
    chunk_size: Max tokens, words or PDF pages per chunk (depending on
      `chunking_strategy`).
    chunking_strategy: Chunking strategy ('tokens', 'words' or 'pdf_pages').
      'pdf_pages' requires all documents to be PDFs, does not support
      `overlap_window`, and requires `pdf_chunking` to be imported (which
      registers it), so that this library does not depend on PDF libraries.
    overlap_window: Tokens or words to overlap between chunks.
    tokenize_backend_class_name: Tokenizer backend class name (when
      `chunking_strategy` is 'tokens').
    tokenize_model_name: Tokenizer model name (when `chunking_strategy` is
      'tokens').
    format_str: Format string for document formatting.
    serializer_type: Serializer type to use ('simple' or 'internal').
    max_retries: Maximum number of retries for backend requests.
    http_timeout: HTTP timeout in milliseconds for backend requests.

  Returns:
    The output directory path where the index was saved.
  """
  # Construct spec
  if isinstance(embed_kwargs, str):
    embed_kwargs = json.loads(embed_kwargs)
  backend_kwargs = {}
  if embed_kwargs:
    backend_kwargs['embed_kwargs'] = dict(embed_kwargs)
  embed_backend_config = backend_registry.EmbedBackendConfig(
      backend_class_name=embedding_backend_class_name,
      model_name=embedding_model_name,
      kwargs=backend_kwargs,
  )
  if isinstance(runtime_kwargs, str):
    rt_kwargs = json.loads(runtime_kwargs)
  else:
    rt_kwargs = dict(runtime_kwargs) if runtime_kwargs else {}
  if 'max_retries' not in rt_kwargs and max_retries is not None:
    rt_kwargs['max_retries'] = max_retries
  if 'http_options' not in rt_kwargs and http_timeout is not None:
    rt_kwargs['http_options'] = {'timeout': http_timeout}
  elif 'http_options' in rt_kwargs and isinstance(
      rt_kwargs['http_options'], dict
  ):
    if 'timeout' not in rt_kwargs['http_options'] and http_timeout is not None:
      rt_kwargs['http_options']['timeout'] = http_timeout

  if format_str in ('None', ''):
    format_str = None

  if chunk_size is None:
    chunker_config = chunker_registry.ChunkerConfig(chunker_name='none')
  elif chunking_strategy == 'words':
    chunker_config = chunker_registry.ChunkerConfig(
        chunker_name='words',
        kwargs={
            'max_words_per_chunk': chunk_size,
            'overlap_window': overlap_window,
        },
    )
  elif chunking_strategy == 'tokens':
    chunker_config = chunker_registry.ChunkerConfig(
        chunker_name='tokens',
        kwargs={
            'max_tokens_per_chunk': chunk_size,
            'overlap_window': overlap_window,
            'tokenize_backend_config': backend_registry.TokenizeBackendConfig(
                backend_class_name=tokenize_backend_class_name,
                model_name=tokenize_model_name,
            ),
        },
    )
  elif chunking_strategy == 'pdf_pages':
    if overlap_window:
      raise ValueError(
          'overlap_window is not supported with chunking_strategy="pdf_pages".'
      )
    chunker_config = chunker_registry.ChunkerConfig(
        chunker_name='pdf_pages',
        kwargs={'pages_per_chunk': chunk_size},
    )
  else:
    raise ValueError(f'Unsupported chunking strategy: {chunking_strategy!r}')

  spec = embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
      embed_backend_config=embed_backend_config,
      chunker_config=chunker_config,
      format_str=format_str,
      serializer_type=serializer_type,
  )
  if index_type == 'hipporag':
    spec = hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        chunker_config=chunker_config,
        format_str=format_str,
    )

  return build_index_from_spec(
      spec=spec,
      dataset_name=dataset_name,
      output_directory=output_directory,
      runtime_kwargs=rt_kwargs,
      max_documents=max_documents,
      input_dir=input_dir,
  )


def build_index_from_spec(
    spec: index_build_interface.IndexSpec,
    dataset_name: str,
    output_directory: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
    max_documents: int | None = None,
    input_dir: str | None = None,
) -> str:
  """Builds and saves the index described by `spec` for the specified dataset.

  Use this for custom indexes: register the spec type with
  `rag.register_index_spec`, then pass an instance of the spec here. The spec
  is used as-is.

  Args:
    spec: Spec of the index to build.
    dataset_name: Name of the dataset. Unless `input_dir` is set, documents are
      loaded from the dataset registry under this name. Also used in the output
      directory path.
    output_directory: Directory pattern where the built index will be saved.
    runtime_kwargs: Arguments for the index's backends (e.g. API key). Passed to
      `spec.make_index`; not saved with the index.
    max_documents: Max documents to index.
    input_dir: Optional path to a folder of files of any type (text, PDF, image,
      audio, video) to index, loaded with `MultimodalDatasetLoader`. If set,
      `dataset_name` must be a name for this corpus (not a registered dataset).

  Returns:
    The output directory path where the index was saved.

  Raises:
    ValueError: If `input_dir` is set and `dataset_name` is a registered
    dataset.
  """
  # TODO: Remove this check once the output directory is passed
  # explicitly as `<cns_path_to_save>/<dataset_name>/` instead of being derived
  # from `dataset_name`.
  if input_dir and dataset_name in datasets.list_dataset_loaders():
    raise ValueError(
        f'dataset_name={dataset_name!r} is a registered dataset, but input_dir'
        ' is set: set dataset_name to a name for the corpus at input_dir,'
        ' which is used in the output directory path, so that the index is not'
        f' saved under the directory of the {dataset_name!r} dataset.'
    )
  if input_dir:
    print(f'Loading documents from input_dir {input_dir}...')
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=input_dir, max_number_of_documents=max_documents
    )
    docs = list(ot_executing.run(loader.load_documents()))
    print(f'Loaded {len(docs)} documents.')
  else:
    print(f'Loading dataset {dataset_name}...')
    docs = datasets.load_documents(
        dataset_name,
        max_number_of_documents=max_documents,
    )
    print(f'Loaded {len(docs)} documents.')

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
          runtime_kwargs=runtime_kwargs,
      )
  )
  print(f'Index built and saved to {out_dir}.')
  return out_dir
