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

from etils import epath
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


def _corpus_name(corpus: str | datasets.DatasetLoader) -> str:
  """Returns the name of `corpus`, used in the index output directory path."""
  if isinstance(corpus, str):
    return epath.Path(corpus).name
  return corpus.dataset_name or type(corpus).__name__


# TODO: Consider merging max_retries and http_timeout
# into runtime_kwargs.
def build_index(
    corpus: str | datasets.DatasetLoader,
    output: str,
    runtime_kwargs: Mapping[str, Any] | str,
    index_type: str = 'embedding',
    embedder: backend_registry.EmbedBackendConfig | None = None,
    chunker: chunker_registry.ChunkerConfig | None = None,
    format_str: str | None = None,
    serializer_type: str = 'simple',
    max_retries: int | None = 5,
    http_timeout: int | None = 60000,
) -> str:
  """Builds and saves a document index of the specified corpus.

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
    output: Base directory (pattern) where the built index will be saved.
    runtime_kwargs: Arguments for backends (e.g. API key).
    index_type: Type of index ('embedding' or 'hipporag'). 'hipporag' uses the
      default settings of `SimpleRewritingHippoRAGDocumentIndexSpec`, including
      its default generation model (`generate_backend_config`); to change them
      (e.g. `similarity_threshold` or the generation model), pass that spec to
      `build_index_from_spec` instead.
    embedder: Config of the embedding backend, used by both index types. If
      None, uses `rag_defaults.default_embed_backend_config()`. Its `kwargs` are
      passed to the backend constructor, so default arguments of the backend's
      `embed` calls go under the `'embed_kwargs'` key, e.g.
      `EmbedBackendConfig(backend_class_name='GoogleGenAIAPI',
      model_name='gemini-embedding-2', kwargs={'embed_kwargs':
      {'output_dimensionality': 768}})`. The config is saved with the index, so
      credentials (e.g. API key) go in `runtime_kwargs` instead.
    chunker: Config of the chunker that splits documents before they are
      indexed, e.g. `ChunkerConfig(chunker_name='routing')`. If None, documents
      are not chunked. `chunker_name` must be registered: see `chunker_registry`
      for the available chunkers and how to register them.
    format_str: Format string for document formatting.
    serializer_type: Serializer type to use ('simple' or 'internal').
    max_retries: Maximum number of retries for backend requests.
    http_timeout: HTTP timeout in milliseconds for backend requests.

  Returns:
    The output directory path where the index was saved.
  """
  # Construct spec
  embed_backend_config = (
      embedder
      if embedder is not None
      else rag_defaults.default_embed_backend_config()
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

  chunker_config = (
      chunker
      if chunker is not None
      else chunker_registry.ChunkerConfig(chunker_name='none')
  )
  if chunker_config.chunker_name not in chunker_registry.list_chunkers():
    raise ValueError(
        f'Unknown chunker: {chunker_config.chunker_name!r}. Registered'
        f' chunkers: {chunker_registry.list_chunkers()}. Chunkers with heavy'
        ' dependencies are registered by importing their module (e.g.'
        " `routing_chunking` for 'routing' or `pdf_chunking` for 'pdf_pages')."
    )

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
      corpus=corpus,
      output=output,
      runtime_kwargs=rt_kwargs,
  )


def build_index_from_spec(
    spec: index_build_interface.IndexSpec,
    corpus: str | datasets.DatasetLoader,
    output: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> str:
  """Builds and saves the index described by `spec` of the specified corpus.

  Use this for custom indexes: register the spec type with
  `rag.register_index_spec`, then pass an instance of the spec here. The spec
  is used as-is.

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

  Returns:
    The output directory path where the index was saved.
  """
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

  out_dir = index_building.resolve_index_output_directory(
      output_directory_pattern=output,
      dataset_name=_corpus_name(corpus),
      spec=spec,
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
      )
  )
  print(f'Index built and saved to {out_dir}.')
  return out_dir
