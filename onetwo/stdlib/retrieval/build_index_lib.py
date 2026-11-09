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
import enum
import hashlib
import json
from typing import Any

from etils import epath
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval.graphrag import hipporag_index_specs


class IndexType(str, enum.Enum):
  """Ready-made indexes, with default settings.

  EMBEDDING: An embedding-based index of the documents, unchunked, with the
    default embedding model.
  HIPPORAG: A HippoRAG knowledge graph index, with the default embedding and
    generation models.
  """

  EMBEDDING = 'embedding'
  HIPPORAG = 'hipporag'


# TODO: Will be updated once the we move away from specs.
def _preset_spec(index_type: IndexType) -> index_build_interface.IndexSpec:
  """Returns the spec of the ready-made index `index_type`."""
  embed_backend_config = rag_defaults.default_embed_backend_config()
  chunker_config = chunker_registry.ChunkerConfig(chunker_name='none')
  if index_type == IndexType.HIPPORAG:
    return hipporag_index_specs.SimpleRewritingHippoRAGDocumentIndexSpec(
        embed_backend_config=embed_backend_config,
        chunker_config=chunker_config,
    )
  return embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec(
      embed_backend_config=embed_backend_config,
      chunker_config=chunker_config,
  )


def _corpus_name(corpus: str | datasets.DatasetLoader) -> str:
  """Returns the name of `corpus`, used in the index output directory path."""
  if isinstance(corpus, str):
    return epath.Path(corpus).name
  return corpus.dataset_name or type(corpus).__name__


def get_default_runtime_kwargs(
    runtime_kwargs: Mapping[str, Any] | None,
    *,
    max_retries: int = rag_defaults.DEFAULT_MAX_RETRIES,
    http_timeout_ms: int = rag_defaults.DEFAULT_HTTP_TIMEOUT_MS,
) -> dict[str, Any]:
  """Returns `runtime_kwargs` with default backend request settings.

  Settings already in `runtime_kwargs` are kept. The input is not modified.

  Args:
    runtime_kwargs: Arguments for backends (e.g. API key).
    max_retries: Maximum number of retries of backend requests, set as
      `'max_retries'` if absent.
    http_timeout_ms: HTTP timeout of backend requests in milliseconds, set as
      `'http_options': {'timeout': http_timeout_ms}` if `'http_options'` is
      absent, or as its `'timeout'` if `'http_options'` is a dict without one.

  Returns:
    A new dict of runtime kwargs with the defaults applied.
  """
  rt_kwargs = dict(runtime_kwargs) if runtime_kwargs else {}
  rt_kwargs.setdefault('max_retries', max_retries)
  http_options = rt_kwargs.get('http_options')
  if 'http_options' not in rt_kwargs:
    rt_kwargs['http_options'] = {'timeout': http_timeout_ms}
  elif isinstance(http_options, dict) and 'timeout' not in http_options:
    rt_kwargs['http_options'] = {**http_options, 'timeout': http_timeout_ms}
  return rt_kwargs


def build_index(
    corpus: str | datasets.DatasetLoader,
    index: IndexType | indexing.DocumentIndex = IndexType.EMBEDDING,
    *,
    output: str,
    storage_format: serialization.StorageFormat = (
        serialization.StorageFormat.JSONL_NPY
    ),
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> str:
  """Builds and saves a document index of the specified corpus.

  To customize the index (e.g. its embedding model or chunker), pass an index
  object instead of a ready-made `IndexType`, e.g.
  `indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=...,
  inner_index=...)`.
  Its config is saved as `index_config.json`, so it can be loaded from its
  directory alone (e.g. by `rag.evaluate(index_directory=...)`). The classes of
  a custom index and its components must be registered with
  `@registry.register(...)`, and the module that registers them must be
  imported before the index is loaded.

  Args:
    corpus: What to index: either a path to a folder of files of any type (text,
      PDF, image, audio, video), loaded with `MultimodalDatasetLoader`, or a
      `datasets.DatasetLoader` whose `load_documents` is used, e.g.
      `datasets.get_dataset_loader(name)` for a registered dataset, or a custom
      loader. The output directory path includes the name of the corpus: the
      basename of the folder for a path; for a loader, its `dataset_name` (set
      by `datasets.get_dataset_loader` to the registered name, or by a custom
      loader) if any, else its class name. To index only the first documents,
      pass a loader with a limit, e.g.
      `datasets.MultimodalDatasetLoader(path=..., max_number_of_documents=N)` or
      `datasets.get_dataset_loader(name, max_number_of_documents=N)`.
    index: The index to build: a ready-made `IndexType`, or an (empty)
      `DocumentIndex` object.
    output: Base directory (pattern) where the built index will be saved.
    storage_format: How the index is saved on disk (see
      `serialization.StorageFormat`).
    runtime_kwargs: Arguments for backends (e.g. API key), not saved with the
      index. Any `runtime_kwargs` set on an index object are ignored. Backend
      requests are retried up to `'max_retries'` times (default
      `rag_defaults.DEFAULT_MAX_RETRIES`) and time out after `'http_options':
      {'timeout': ...}` milliseconds (default
      `rag_defaults.DEFAULT_HTTP_TIMEOUT_MS`, also added to an `'http_options'`
      dict without `'timeout'`). See `get_default_runtime_kwargs`.

  Returns:
    The output directory path where the index was saved.

  Raises:
    KeyError, TypeError: If an index object cannot be saved, e.g. because the
      class of one of its components is not registered. Raised before the
      corpus is loaded.
  """
  rt_kwargs = get_default_runtime_kwargs(runtime_kwargs)
  if isinstance(index, str):
    return build_index_from_spec(
        spec=_preset_spec(IndexType(index)),
        corpus=corpus,
        output=output,
        runtime_kwargs=rt_kwargs,
        storage_format=storage_format,
    )

  # Fail fast, before loading the corpus.
  config = config_serialization.get_config(index)
  if not indexing.supports_save(index):
    raise TypeError(f'{type(index).__name__} does not support saving.')
  # Rebuilds the index with the runtime kwargs, without modifying `index`.
  # TODO: Pass runtime kwargs with `connect` instead.
  index = config_serialization.from_config(config, rt_kwargs)

  docs = _load_documents(corpus)
  config_hash = hashlib.sha256(
      json.dumps(config, sort_keys=True).encode()
  ).hexdigest()[:8]
  out_dir = index_building.resolve_index_output_directory(
      output_directory_pattern=output,
      dataset_name=_corpus_name(corpus),
      subdirectory=f'{config["class"]}__hash_{config_hash}',
  )
  print(f'Will save the index to output directory: {out_dir}')
  print('Building index...')
  ot_executing.run(
      index.create_index(corpus_name=_corpus_name(corpus), docs=docs)
  )
  index.save(out_dir, storage_format)
  config_serialization.save_index_config(index, out_dir)
  print(f'Index built and saved to {out_dir}.')
  return out_dir


def _load_documents(
    corpus: str | datasets.DatasetLoader,
) -> list[retrieval_data_structures.Document]:
  """Returns the documents of `corpus` (see `build_index`)."""
  if isinstance(corpus, str):
    print(f'Loading documents from {corpus}...')
    loader = multimodal_loader.MultimodalDatasetLoader(path=corpus)
    docs = list(ot_executing.run(loader.load_documents()))
  else:
    # `DatasetLoader` is a Protocol that is not runtime checkable: any other
    # corpus is used as a loader.
    print('Loading documents...')
    docs = list(ot_executing.run(corpus.load_documents()))
  print(f'Loaded {len(docs)} documents.')
  return docs


def build_index_from_spec(
    spec: index_build_interface.IndexSpec,
    corpus: str | datasets.DatasetLoader,
    output: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
    storage_format: serialization.StorageFormat = (
        serialization.StorageFormat.JSONL_NPY
    ),
) -> str:
  """Builds and saves the index described by `spec` of the specified corpus.

  Used for the ready-made indexes of `build_index` and by the `build_index`
  command-line tool. The spec is used as-is.

  Args:
    spec: Spec of the index to build.
    corpus: What to index: either a path to a folder of files of any type (text,
      PDF, image, audio, video), loaded with `MultimodalDatasetLoader`, or a
      `datasets.DatasetLoader` whose `load_documents` is used, e.g.
      `datasets.get_dataset_loader(name)` for a registered dataset, or a custom
      loader. The output directory path includes the name of the corpus: the
      basename of the folder for a path; for a loader, its `dataset_name` (set
      by `datasets.get_dataset_loader` to the registered name, or by a custom
      loader) if any, else its class name. To index only the first documents,
      pass a loader with a limit, e.g.
      `datasets.MultimodalDatasetLoader(path=..., max_number_of_documents=N)` or
      `datasets.get_dataset_loader(name, max_number_of_documents=N)`.
    output: Base directory (pattern) where the built index will be saved.
    runtime_kwargs: Arguments for the index's backends (e.g. API key). Passed to
      `spec.make_index`; not saved with the index.
    storage_format: How the index is saved on disk (see
      `serialization.StorageFormat`).

  Returns:
    The output directory path where the index was saved.
  """
  docs = _load_documents(corpus)
  out_dir = index_building.resolve_index_output_directory(
      output_directory_pattern=output,
      dataset_name=_corpus_name(corpus),
      subdirectory=str(spec),
  )
  print(f'Will save the index to output directory: {out_dir}')

  # Build index
  print('Building index...')
  _ = ot_executing.run(
      index_building.build_index(
          spec=spec,
          docs=docs,
          output_dir=out_dir,
          runner='local',
          runtime_kwargs=runtime_kwargs,
          storage_format=storage_format,
      )
  )
  print(f'Index built and saved to {out_dir}.')
  return out_dir
