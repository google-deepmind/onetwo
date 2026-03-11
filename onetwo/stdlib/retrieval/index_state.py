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

"""Data Transfer Objects for Index Serialization.

This module defines the schema for data exchanged between Indexing structures
and Serialization handlers. By using these DTOs, we decouple the physical
storage logic (JSONL/NPY) from the in-memory index implementation.
"""

import dataclasses
from typing import Any, List
import numpy as np
from onetwo.stdlib.retrieval import retrieval_data_structures


@dataclasses.dataclass(frozen=True)
class DocumentIndexState:
  """A point-in-time snapshot of an EmbeddingBasedDocumentIndex state.

  Attributes:
    docs: The raw Document objects to be serialized.
    doc_embeddings: A list of numpy arrays representing the document embeddings.
    doc_indices_by_discrete_value: Internal reverse index for discrete field
      values, used for constrained retrieval. See the
      `EmbeddingBasedDocumentIndex` class for more details.
  """

  docs: List[retrieval_data_structures.Document]
  doc_embeddings: List[np.ndarray]
  doc_indices_by_discrete_value: dict[str, dict[Any, list[int]]]
