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
from typing import Any

from etils import epath
from onetwo.core import executing as ot_executing
from onetwo.evaluation import datasets
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import routing_chunking
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing


class IndexType(str, enum.Enum):
  """Ready-made indexes, with default settings.

  Both chunk the documents by content type (with the defaults of
  `routing_chunking.RoutingChunker`).

  EMBEDDING: An embedding-based index of the chunks.
  GRAPHRAG: A graph-based index of the chunks, built as a modified HippoRAG
    implementation.
  """

  EMBEDDING = 'embedding'
  GRAPHRAG = 'graphrag'


def _preset_index(index_type: IndexType) -> indexing.DocumentIndex:
  """Returns a new, empty ready-made index of type `index_type`."""
  rewriter = corpus_rewriting.ChunkingCorpusRewriter(
      chunker=routing_chunking.RoutingChunker()
  )
  embed_backend_config = rag_defaults.default_embed_backend_config()
  if index_type == IndexType.GRAPHRAG:
    return indexing.RewritingDocumentIndex(
        rewriter=rewriter,
        inner_index=graphrag_indexing.HippoRAGDocumentIndex(
            embed_backend_config=embed_backend_config,
            generate_backend_config=(
                rag_defaults.default_index_generate_backend_config()
            ),
        ),
    )
  elif index_type == IndexType.EMBEDDING:
    return indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=rewriter,
        inner_index=indexing.EmbeddingBasedDocumentIndex(
            embed_backend_config=embed_backend_config
        ),
    )
  else:
    raise ValueError(f'Unknown index type: {index_type}')


def _corpus_name(corpus: str | datasets.DatasetLoader) -> str:
  """Returns the name of `corpus`, used in the index output directory path."""
  if isinstance(corpus, str):
    return epath.Path(corpus).name
  return corpus.dataset_name or type(corpus).__name__


def build_index(
    corpus: str | datasets.DatasetLoader,
    index: IndexType | indexing.DocumentIndex = IndexType.EMBEDDING,
    *,
    output: str,
    storage_format: serialization.StorageFormat = (
        serialization.StorageFormat.JSONL_NPY
    ),
    extra_backend_kwargs: Mapping[str, Any] | None = None,
) -> indexing.DocumentIndex:
  """Builds and saves a document index of the specified corpus.

  To customize the index (e.g. its embedding model or chunker), pass an index
  object instead of a ready-made `IndexType`, e.g.
  `indexing.RewritingEmbeddingBasedDocumentIndex(rewriter=...,
  inner_index=...)`.
  The config of the index (ready-made or not) is saved as `index_config.json`,
  so it can be loaded from its directory alone (e.g. by
  `rag.evaluate(..., index=...)`). The classes of
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
    output: Base directory (pattern) where the built index will be saved, in the
      subdirectory `<corpus name>/<index_building.index_dir_name(index)>`.
    storage_format: How the index is saved on disk (see
      `serialization.StorageFormat`).
    extra_backend_kwargs: Extra keyword arguments for the constructors of the
      backends that this call creates (e.g. an API key). Not saved with the
      index. Usually not needed: credentials are read from the environment (e.g.
      `GOOGLE_API_KEY`). Not a stable part of the API. Any `runtime_kwargs` set
      on an index object are ignored. They override the same keys of the kwargs
      of the backend configs (e.g. the request settings of the default configs,
      see `rag_defaults`).

  Returns:
    The built index, with the `extra_backend_kwargs`, ready to be used, e.g. by
    `rag.evaluate(..., index=...)` or `rag.start_mcp(...)`. It is also saved in
    `<output>/<corpus name>/<index_building.index_dir_name(index)>` (the
    directory is printed), from which it can be loaded later, e.g. by
    `index_building.load_index` or `rag.evaluate(..., index=<directory>)`.

  Raises:
    KeyError, TypeError: If an index object cannot be saved, e.g. because the
      class of one of its components is not registered. Raised before the
      corpus is loaded.
  """
  rt_kwargs = dict(extra_backend_kwargs) if extra_backend_kwargs else {}
  if isinstance(index, str):
    index = _preset_index(IndexType(index))

  # Fail fast, before loading the corpus.
  config = config_serialization.get_config(index)
  if not indexing.supports_save(index):
    raise TypeError(f'{type(index).__name__} does not support saving.')
  # Rebuilds the index with the runtime kwargs, without modifying `index`.
  # TODO: Pass runtime kwargs with `connect` instead.
  index = config_serialization.from_config(config, rt_kwargs)

  docs = _load_documents(corpus)
  out_dir = index_building.resolve_index_output_directory(
      output_directory_pattern=output,
      dataset_name=_corpus_name(corpus),
      subdirectory=index_building.index_dir_name(index),
  )
  print(f'Will save the index to output directory: {out_dir}')
  print('Building index...')
  ot_executing.run(
      index.create_index(corpus_name=_corpus_name(corpus), docs=docs)
  )
  index.save(out_dir, storage_format)
  config_serialization.save_index_config(index, out_dir)
  print(f'Index built and saved to {out_dir}.')
  return index


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
