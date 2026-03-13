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

import dataclasses
import json
import os
from typing import Generic, TypeVar

import numpy as np
from onetwo.core import executing
from onetwo.stdlib.retrieval import index_state

DocT = TypeVar('DocT')


@dataclasses.dataclass
class SimpleEmbeddingBasedIndexSerializer(Generic[DocT]):
  """Simple serializer for EmbeddingBasedIndex.

  Saves the list of documents to a JSONL file, the embeddings to a NPY file,
  and the discrete field indices to a JSON file.

  Attributes:
    doc_class: The class to use for serializing and deserializing documents. It
      must have `to_json()` and `from_json(json_str)` methods.
    base_path: The default base path for saving and loading the index state.
    docs_jsonl_name: Name of the file for storing documents (JSON lines).
    embeds_npy_name: Name of the file for storing embeddings (NumPy).
    discrete_indices_json_name: Name of the file for storing discrete indices.
  """

  doc_class: type[DocT]
  base_path: str | None = None
  docs_jsonl_name: str = 'docs.jsonl'
  embeds_npy_name: str = 'embeddings.npy'
  discrete_indices_json_name: str = 'discrete_indices.json'

  def __post_init__(self):
    """Validates interface at initialization."""
    for method in ['to_json', 'from_json']:
      if not hasattr(self.doc_class, method):
        raise TypeError(f'{self.doc_class.__name__} must implement {method}')

  def save(
      self,
      state: index_state.EmbeddingBasedIndexState[DocT],
      base_path: str | None = None,
  ) -> None:
    """Saves the index state to the given base path.

    Args:
      state: The index state to save.
      base_path: The directory to save the index to. If not provided, the
        default `base_path` will be used.

    Raises:
      ValueError: If no base_path is provided.
    """
    path = base_path or self.base_path
    if not path:
      raise ValueError('base_path is not set.')

    os.makedirs(path, exist_ok=True)
    # Save documents to JSONL.
    docs_path = os.path.join(path, self.docs_jsonl_name)
    with open(docs_path, 'w') as f:
      for doc in state.docs:
        f.write(doc.to_json() + '\n')  # pytype: disable=attribute-error

    # Save embeddings to NPY.
    embeddings_path = os.path.join(path, self.embeds_npy_name)
    np.save(embeddings_path, np.stack(state.doc_embeddings, axis=0))

    # Save discrete indices to JSON.
    indices_path = os.path.join(path, self.discrete_indices_json_name)
    with open(indices_path, 'w') as f:
      json.dump(state.doc_indices_by_discrete_value, f)

  @executing.make_executable(copy_self=False)
  async def load(
      self,
      base_path: str | None = None,
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
    path = base_path or self.base_path
    if not path:
      raise ValueError('base_path is not set.')

    # Load documents from JSONL.
    docs_path = os.path.join(path, self.docs_jsonl_name)
    docs = []
    with open(docs_path, 'r') as f:
      for line in f:
        if line.strip():
          docs.append(self.doc_class.from_json(line))  # pytype: disable=attribute-error

    # Load embeddings from NPY.
    embeddings_path = os.path.join(path, self.embeds_npy_name)
    doc_embeddings = list(np.load(embeddings_path))

    # Load discrete indices from JSON.
    indices_path = os.path.join(path, self.discrete_indices_json_name)
    with open(indices_path, 'r') as f:
      doc_indices_by_discrete_value = json.load(f)

    return index_state.EmbeddingBasedIndexState(
        docs=docs,
        doc_embeddings=doc_embeddings,
        doc_indices_by_discrete_value=doc_indices_by_discrete_value,
    )
