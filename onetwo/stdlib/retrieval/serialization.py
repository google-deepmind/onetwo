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

"""Serialization protocols and simple implementations for retrieval indices."""

import collections
import dataclasses
import io
import json
from typing import Generic, TypeVar

from etils import epath
import numpy as np
from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import retrieval_data_structures
import tqdm

DocT = TypeVar('DocT', bound=retrieval_data_structures.Document)


def load_documents(
    file_path: str | epath.Path,
    doc_class: type[DocT],
    show_progress: bool = True,
    total: int | None = None,
) -> list[DocT]:
  """Loads documents from a file path (supports single or sharded files).

  Args:
    file_path: Path to the documents file.
    doc_class: The class to use for deserializing documents.
    show_progress: Whether to display a tqdm progress bar during loading.
    total: Optional total number of documents for progress tracking.

  Returns:
    A list of deserialized documents.
  """
  file_path = epath.Path(file_path)
  parent = file_path.parent
  stem = file_path.stem
  suffix = file_path.suffix
  shards = sorted(parent.glob(f'{stem}-*-of-*{suffix}'))

  # Direct instantiation via doc_class(**json.loads(line)) bypasses runtime
  # reflection and schema inspection from dataclasses_json.from_json(),
  # speeding up loading by >15x on large datasets.
  def _read_single(p: epath.Path) -> list[DocT]:
    content = p.read_text()
    return [
        doc_class(**json.loads(line))
        for line in content.splitlines()
        if line.strip()
    ]

  if len(shards) > 1:
    docs = []
    shards_iter = (
        tqdm.tqdm(shards, desc='Loading document shards')
        if show_progress
        else shards
    )
    for s in shards_iter:
      docs.extend(_read_single(s))
    return docs
  else:
    content = file_path.read_text()
    lines = content.splitlines()
    iterator = lines
    if show_progress:
      iterator = tqdm.tqdm(
          iterator, total=total or len(lines), desc='Loading index documents'
      )
    return [doc_class(**json.loads(line)) for line in iterator if line.strip()]


@dataclasses.dataclass
class SimpleEmbeddingBasedIndexSerializer(Generic[DocT]):
  """Simple serializer for EmbeddingBasedIndex.

  Saves the list of documents to JSONL file(s) and embeddings to NPY file(s).

  Attributes:
    doc_class: The class to use for serializing and deserializing documents. It
      must have `to_json()` and `from_json(json_str)` methods.
    base_path: The default base path for saving and loading the index state.
    docs_jsonl_name: Name of the file for storing documents (JSON lines).
    embeds_npy_name: Name of the file for storing embeddings (NumPy).
    embed_backend_config_json_name: Name of the file for storing the embedding
      backend config (JSON).
    num_shards: Number of shards for parallel saving (1 for unsharded).
    show_progress: Whether to display a tqdm progress bar during loading.
  """

  doc_class: type[DocT]
  base_path: str | epath.Path | None = None
  docs_jsonl_name: str = 'docs.jsonl'
  embeds_npy_name: str = 'embeddings.npy'
  embed_backend_config_json_name: str = 'embed_backend_config.json'
  num_shards: int = 1
  show_progress: bool = True

  def __post_init__(self):
    """Validates interface at initialization."""
    for method in ['to_json', 'from_json']:
      if not hasattr(self.doc_class, method):
        raise TypeError(f'{self.doc_class.__name__} must implement {method}')

  def save(
      self,
      state: index_state.EmbeddingBasedIndexState[DocT],
      base_path: str | epath.Path | None = None,
  ) -> None:
    """Saves the index state to the given base path.

    Args:
      state: The index state to save.
      base_path: The directory to save the index to. If not provided, the
        default `base_path` will be used.

    Raises:
      ValueError: If no base_path is provided.
    """
    raw_path = base_path or self.base_path
    if not raw_path:
      raise ValueError('base_path is not set.')
    path = epath.Path(raw_path)

    path.mkdir(parents=True, exist_ok=True)
    # Save documents to JSONL (sharded if num_shards > 1).
    docs_path = path / self.docs_jsonl_name
    stem = docs_path.stem
    suffix = docs_path.suffix
    if self.num_shards > 1 and state.docs:
      shard_size = (len(state.docs) + self.num_shards - 1) // self.num_shards
      for s in range(self.num_shards):
        shard_path = path / f'{stem}-{s:05d}-of-{self.num_shards:05d}{suffix}'
        with shard_path.open('w') as f:
          for doc in state.docs[s * shard_size : (s + 1) * shard_size]:
            doc_document: retrieval_data_structures.Document = doc
            f.write(doc_document.to_json() + '\n')
    else:
      with docs_path.open('w') as f:
        for doc in state.docs:
          doc_document: retrieval_data_structures.Document = doc
          f.write(doc_document.to_json() + '\n')

    # Save embeddings to NPY (sharded if num_shards > 1).
    embeddings_path = path / self.embeds_npy_name
    embed_matrix = np.stack(state.doc_embeddings, axis=0)
    estem = embeddings_path.stem
    esuffix = embeddings_path.suffix
    if self.num_shards > 1:
      shard_size = (len(embed_matrix) + self.num_shards - 1) // self.num_shards
      for s in range(self.num_shards):
        shard_path = path / f'{estem}-{s:05d}-of-{self.num_shards:05d}{esuffix}'
        with shard_path.open('wb') as f:
          np.save(f, embed_matrix[s * shard_size : (s + 1) * shard_size])
    else:
      with embeddings_path.open('wb') as f:
        np.save(f, embed_matrix)

    # Save embed_backend_config to JSON if it exists.
    if state.embed_backend_config:
      config_path = path / self.embed_backend_config_json_name
      with config_path.open('w') as f:
        try:
          f.write(state.embed_backend_config.to_json())  # pyrefly: ignore[missing-attribute]
        except TypeError as e:
          raise TypeError(
              'Failed to serialize EmbedBackendConfig to JSON. Ensure all '
              f'kwargs are JSON-serializable. Original error: {e}'
          ) from e

  @executing.make_executable(copy_self=False)
  async def load(
      self,
      base_path: str | epath.Path | None = None,
  ) -> index_state.EmbeddingBasedIndexState[DocT]:
    """Loads the index state from the given base path.

    Args:
      base_path: The directory to load the index from. If not provided, the
        default `base_path` will be used.

    Returns:
      The loaded index state.

    Raises:
      ValueError: If no base_path is provided.
    """
    raw_path = base_path or self.base_path
    if not raw_path:
      raise ValueError('base_path is not set.')
    path = epath.Path(raw_path)

    # Load documents from JSONL.
    docs = load_documents(
        file_path=path / self.docs_jsonl_name,
        doc_class=self.doc_class,
        show_progress=self.show_progress,
    )

    # Load embeddings from NPY (sharded or single).
    def _load_single_npy(s: epath.Path) -> np.ndarray:
      content = s.read_bytes()
      return np.load(io.BytesIO(content))

    embeddings_path = path / self.embeds_npy_name
    estem = embeddings_path.stem
    esuffix = embeddings_path.suffix
    embed_shards = sorted(path.glob(f'{estem}-*-of-*{esuffix}'))

    if len(embed_shards) > 1:
      loaded_embed_shards = []
      shards_iter = (
          tqdm.tqdm(embed_shards, desc='Loading embedding shards')
          if self.show_progress
          else embed_shards
      )
      for s in shards_iter:
        loaded_embed_shards.append(_load_single_npy(s))
      doc_embeddings = list(np.concatenate(loaded_embed_shards, axis=0))
    elif embeddings_path.exists():
      doc_embeddings = list(_load_single_npy(embeddings_path))
    else:
      doc_embeddings = []

    # Load embed_backend_config if it exists.
    config_path = path / self.embed_backend_config_json_name
    embed_backend_config = None
    if config_path.exists():
      with config_path.open('r') as f:
        json_str = f.read()
      embed_backend_config = backend_registry.EmbedBackendConfig.from_json(  # pyrefly: ignore[missing-attribute]
          json_str
      )

    return index_state.EmbeddingBasedIndexState(
        docs=docs,
        doc_embeddings=doc_embeddings,
        embed_backend_config=embed_backend_config,
    )


@dataclasses.dataclass
class SimpleConstrainedEmbeddingBasedIndexSerializer(
    SimpleEmbeddingBasedIndexSerializer[DocT]
):
  """Serializer for ConstrainedEmbeddingBasedIndex.

  Inherits doc and embed saving/loading from SimpleEmbeddingBasedIndexSerializer
  and adds saving/loading of discrete field indices to a JSON file.

  Attributes:
    discrete_indices_json_name: Name of the file for storing discrete indices.
  """

  discrete_indices_json_name: str = 'discrete_indices.json'

  def save(
      self,
      state: index_state.EmbeddingBasedIndexState[DocT],
      base_path: str | epath.Path | None = None,
  ) -> None:
    """Saves the index state to the given base path."""
    super().save(state, base_path)

    if isinstance(state, index_state.ConstrainedEmbeddingBasedIndexState):
      raw_path = base_path or self.base_path
      if not raw_path:
        raise ValueError('base_path is not set.')
      path = epath.Path(raw_path)
      indices_path = path / self.discrete_indices_json_name
      with indices_path.open('w') as f:
        json.dump(state.doc_indices_by_discrete_value, f)

  @executing.make_executable(copy_self=False)
  async def load(  # pyrefly: ignore[bad-override]
      self,
      base_path: str | epath.Path | None = None,
  ) -> index_state.ConstrainedEmbeddingBasedIndexState[DocT]:
    """Loads the index state from the given base path."""
    base_state = await super().load(base_path)

    raw_path = base_path or self.base_path
    if not raw_path:
      raise ValueError('base_path is not set.')
    path = epath.Path(raw_path)
    indices_path = path / self.discrete_indices_json_name
    with indices_path.open('r') as f:
      loaded_indices = json.load(f)

    doc_indices_by_discrete_value = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for field_name, field_dict in loaded_indices.items():
      doc_indices_by_discrete_value[field_name] = collections.defaultdict(
          list, field_dict
      )

    return index_state.ConstrainedEmbeddingBasedIndexState(
        docs=base_state.docs,
        doc_embeddings=base_state.doc_embeddings,
        embed_backend_config=base_state.embed_backend_config,
        doc_indices_by_discrete_value=doc_indices_by_discrete_value,  # pyrefly: ignore[bad-argument-type]
    )
