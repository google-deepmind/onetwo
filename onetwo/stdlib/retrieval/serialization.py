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

import json
import os
from typing import TypeVar

import numpy as np
from onetwo.core import executing
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document

DocT = TypeVar('DocT')
DocEmbeddingT = TypeVar('DocEmbeddingT')


class SimpleEmbeddingBasedDocumentIndexSerializer:
  """Simple serializer for EmbeddingBasedDocumentIndex.

  It saves documents and discrete field indices as JSON/JSONL and embeddings as
  a NumPy .npy file.

  File Layout under `base_path`:
    - docs.jsonl: Individual Document objects serialized per line.
    - embeddings.npy: A single NumPy binary file containing the vector matrix.
    - discrete_indices.json: A JSON map of field values to document offsets.

  Attributes:
    base_path: The directory where index files are stored. If the directory does
      not exist during save(), it will be created.
    docs_jsonl_name: Name of the file for storing documents (JSON lines).
    embeds_npy_name: Name of the file for storing embeddings (NumPy).
    discrete_indices_json_name: Name of the file for storing discrete indices.
  """

  base_path: str | None = None
  docs_jsonl_name = 'docs.jsonl'
  embeds_npy_name = 'embeddings.npy'
  discrete_indices_json_name = 'discrete_indices.json'

  def __init__(self, base_path: str | None = None):
    self.base_path = base_path

  def save(
      self,
      document_index_state: index_state.DocumentIndexState,
      base_path: str | None = None,
  ) -> None:
    """Saves the index state to the specified path.

    Args:
      document_index_state: The index state to save.
      base_path: The destination directory. If provided, overrides the
        instance's base_path. Files are saved directly inside this folder; if
        the folder does not exist, it is created automatically.

    Raises:
      ValueError: If no base_path is provided.
    """
    path = base_path or self.base_path
    if not path:
      raise ValueError('base_path is not set.')

    if not os.path.exists(path):
      os.makedirs(path)

    # Save documents.
    docs_path = os.path.join(path, self.docs_jsonl_name)
    with open(docs_path, 'w') as f:
      for doc in document_index_state.docs:
        f.write(doc.to_json() + '\n')

    # Save embeddings.
    embeds_path = os.path.join(path, self.embeds_npy_name)
    np.save(embeds_path, np.array(document_index_state.doc_embeddings))

    # Save discrete indices.
    indices_path = os.path.join(path, self.discrete_indices_json_name)
    with open(indices_path, 'w') as f:
      json.dump(document_index_state.doc_indices_by_discrete_value, f)

  @executing.make_executable(copy_self=False)
  async def load(
      self,
      base_path: str | None = None,
  ) -> index_state.DocumentIndexState:
    """Loads the index state from the specified path.

    Args:
      base_path: The directory containing the index files. If provided,
        overrides the instance's base_path.

    Returns:
      A DocumentIndexState DTO populated with data from the files.

    Raises:
      ValueError: If no base_path is provided.
      FileNotFoundError: If the required files are missing from the path.
    """
    path = base_path or self.base_path
    if not path:
      raise ValueError('base_path is not set.')

    # Load documents.
    docs_path = os.path.join(path, self.docs_jsonl_name)
    docs = []  # pylint: disable=protected-access
    with open(docs_path, 'r') as f:
      for line in f:
        if line.strip():
          docs.append(_Document.from_json(line))  # pylint: disable=protected-access

    # Load embeddings.
    embeds_path = os.path.join(path, self.embeds_npy_name)
    embeddings = np.load(embeds_path)
    doc_embeddings = [embeddings[i] for i in range(embeddings.shape[0])]  # pylint: disable=protected-access

    # Load discrete indices.
    indices_path = os.path.join(path, self.discrete_indices_json_name)
    with open(indices_path, 'r') as f:
      doc_indices_by_discrete_value = json.load(f)  # pylint: disable=protected-access

    return index_state.DocumentIndexState(
        docs=docs,
        doc_embeddings=doc_embeddings,
        doc_indices_by_discrete_value=doc_indices_by_discrete_value,
    )
