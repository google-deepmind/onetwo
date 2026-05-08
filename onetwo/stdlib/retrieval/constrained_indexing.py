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

"""Index classes for constrained retrieval."""

import collections
from collections.abc import Callable, Iterable, Sequence
import dataclasses
from typing import Any, Generic

import numpy as np

from onetwo.core import executing
from onetwo.core import tracing
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures

QueryT = indexing.QueryT
RetrievalResultT = indexing.RetrievalResultT
DocT = indexing.DocT
Document = retrieval_data_structures.Document


def _default_list_map() -> collections.defaultdict[Any, list[int]]:
  return collections.defaultdict(list)


def _default_discrete_value_map() -> (
    collections.defaultdict[str, collections.defaultdict[Any, list[int]]]
):
  return collections.defaultdict(_default_list_map)


@dataclasses.dataclass(kw_only=True)
class ConstrainedEmbeddingBasedIndex(
    indexing.EmbeddingBasedIndex[QueryT, RetrievalResultT, DocT],
    constrained_retrieval.ConstrainedRetriever[QueryT, RetrievalResultT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """Embedding-based index that supports constrained retrieval.

  Attributes:
    discrete_field_extractors: A dictionary where keys are string field names
      and values are functions. Each function accepts a document (DocT) and
      returns a discrete value (e.g., str, int) or a list of such values
      for that field. These extractors populate internal reverse indices,
      mapping field values to document indices. This allows for efficient
      application of `RetrievalConstraints` during the retrieval process, using
      the keys in this dictionary as `field_name` in the constraints.
      The 'discrete_' qualifier distinguishes these from potential future
      non-discrete fields (e.g., continuous values).
      Example:
      ```python
        discrete_field_extractors = {
          'author': lambda doc: doc.author,
          'tags': lambda doc: doc.tags
        }
        # This would allow for constraints like
        `RetrievalConstraint(field_name='author', value='Jane Doe')`
        `RetrievalConstraint(
            field_name='tags',
            constraint_type=LIST_CONTAINS_ANY,
            value=['python', 'java'])`.
      ```
  """

  discrete_field_extractors: dict[str, Callable[[DocT], Any]] = (
      dataclasses.field(default_factory=dict)
  )

  # Internal reverse index for discrete field values, used for constrained
  # retrieval. This is a mapping from field name to a mapping of field value to
  # a list of matching document indices. The name is qualified with 'discrete_'
  # to distinguish it from potential future fields whose values are not discrete
  # (e.g., continuous values for which constraints may be specified in terms of
  # a value range).
  _doc_indices_by_discrete_value: dict[str, dict[Any, list[int]]] = (
      dataclasses.field(default_factory=_default_discrete_value_map)
  )

  def _update_doc_indices_by_discrete_value(
      self,
      docs: Sequence[DocT],
      doc_offset: int,
  ) -> None:
    """Updates the field value to doc indices mapping."""
    for i, doc in enumerate(docs):
      doc_index = doc_offset + i
      for field_name, extractor in self.discrete_field_extractors.items():
        value = extractor(doc)
        if isinstance(value, list):
          for val in value:
            self._doc_indices_by_discrete_value[field_name][val].append(
                doc_index
            )
        else:
          self._doc_indices_by_discrete_value[field_name][value].append(
              doc_index
          )

  @executing.make_executable(copy_self=False)
  @tracing.trace('ConstrainedEmbeddingBasedIndex.add_docs')
  async def add_docs(self, docs: Sequence[DocT]) -> None:
    """Overridden from base class (Index)."""
    num_existing_docs = len(self._docs)
    await super().add_docs(docs)  # pytype: disable=wrong-arg-count
    self._update_doc_indices_by_discrete_value(docs, num_existing_docs)

  @executing.make_executable(copy_self=False)
  @tracing.trace(
      'ConstrainedEmbeddingBasedIndex.retrieve_doc_indices_and_scores'
  )
  async def retrieve_doc_indices_and_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
      **kwargs,
  ) -> Sequence[tuple[int, float]]:
    """Returns indices and scores of the most relevant documents."""
    del kwargs  # Unused.
    if constraints is None:
      return await super().retrieve_doc_indices_and_scores(  # pytype: disable=wrong-arg-count
          query, max_results=max_results, min_score=min_score
      )

    query_embedding = await self._calculate_query_embedding(query)  # pytype: disable=wrong-arg-count
    query_embeddings = np.array([query_embedding])

    candidate_indices_list = constrained_retrieval.get_candidate_indices(
        constraints, self._doc_indices_by_discrete_value
    )
    if not candidate_indices_list:
      return []

    ranked_doc_indices, ranked_doc_scores = (
        await self.searcher.search_batched(  # pytype: disable=wrong-keyword-args
            query_embeds=query_embeddings,
            final_num_neighbors=max_results,
            doc_indices=candidate_indices_list,
        )
    )
    if min_score is None:
      return list(zip(ranked_doc_indices[0], ranked_doc_scores[0]))
    else:
      result = []
      for index, score in zip(ranked_doc_indices[0], ranked_doc_scores[0]):
        if score < min_score:
          break
        result.append((index, score))
      return result

  @executing.make_executable(copy_self=False)
  @tracing.trace('ConstrainedEmbeddingBasedIndex.retrieve_with_scores')
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
      **kwargs,
  ) -> Iterable[tuple[DocT, float]]:
    """Overridden from base class (ConstrainedRetriever)."""
    doc_indices_and_scores = await self.retrieve_doc_indices_and_scores(  # pytype: disable=wrong-keyword-args
        query,
        max_results=max_results,
        min_score=min_score,
        constraints=constraints,
    )
    ranked_documents_and_scores = [
        (self._docs[i], score) for i, score in doc_indices_and_scores
    ]
    return ranked_documents_and_scores

  @executing.make_executable(copy_self=False)
  @tracing.trace('ConstrainedEmbeddingBasedIndex.retrieve', skip=['output'])
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
      **kwargs,
  ) -> Iterable[DocT]:
    """Overridden from base class (ConstrainedRetriever)."""
    doc_indices_and_scores = await self.retrieve_doc_indices_and_scores(  # pytype: disable=wrong-keyword-args
        query,
        max_results=max_results,
        min_score=min_score,
        constraints=constraints,
    )
    ranked_documents = [self._docs[i] for i, _ in doc_indices_and_scores]
    return ranked_documents

  def rename_discrete_field_key(
      self, field_name: str, old_key: str, new_key: str
  ):
    """Renames or merges a key in the index for a discrete field.

    This operation finds all document indices associated with `old_key`
    for the given `field_name` and re-associates them with `new_key`.
    If `new_key` already exists in the index for this field, the indices
    from `old_key` are added to `new_key`'s existing list of indices (merge).
    If `old_key` does not exist for `field_name`, this method does nothing.

    Args:
      field_name: The name of the discrete field to update (e.g.,
        'external_keys').
      old_key: The existing key whose document associations will be moved.
      new_key: The target key to associate the documents with.

    Raises:
      ValueError: If `field_name` is not a known discrete field index.
    """
    if field_name not in self._doc_indices_by_discrete_value:
      raise ValueError(
          f"Field '{field_name}' not found in discrete field indices."
      )

    index_for_field = self._doc_indices_by_discrete_value[field_name]
    if old_key in index_for_field:
      doc_indices = index_for_field.pop(old_key)
      index_for_field[new_key].extend(doc_indices)

  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    super().destroy_index()
    self._doc_indices_by_discrete_value = _default_discrete_value_map()


@dataclasses.dataclass(kw_only=True)
class RewritingConstrainedEmbeddingBasedIndex(
    indexing.RewritingEmbeddingBasedIndex[QueryT, RetrievalResultT, DocT],
    constrained_retrieval.ConstrainedRetriever[QueryT, RetrievalResultT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """Index that rewrites documents and provides constrained retrieval."""

  inner_index: ConstrainedEmbeddingBasedIndex[
      QueryT, RetrievalResultT, DocT
  ] = dataclasses.field(default_factory=ConstrainedEmbeddingBasedIndex)

  def rename_discrete_field_key(
      self, field_name: str, old_key: str, new_key: str
  ) -> None:
    """Delegates to the inner index."""
    self.inner_index.rename_discrete_field_key(  # pytype: disable=attribute-error
        field_name, old_key, new_key
    )


from onetwo.stdlib.retrieval import serialization
constrained_serializer = (
    serialization.SimpleConstrainedEmbeddingBasedIndexSerializer)


@dataclasses.dataclass(kw_only=True)
class ConstrainedEmbeddingBasedDocumentIndex(
    ConstrainedEmbeddingBasedIndex[str, Document, Document],
    indexing.DocumentIndex,
):
  """Embedding-based index for Document objects with support for constraints."""

  def save(self, base_path: str) -> None:
    """Saves the index state."""
    index_state_dto = index_state.ConstrainedEmbeddingBasedIndexState(
        docs=self._docs,
        doc_embeddings=self._doc_embeddings,
        doc_indices_by_discrete_value=self._doc_indices_by_discrete_value,
    )
    constrained_serializer(Document).save(index_state_dto, base_path)

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state."""
    index_state_dto = await constrained_serializer(Document).load(base_path)  # pytype: disable=wrong-arg-count
    self._docs = index_state_dto.docs
    self._doc_embeddings = index_state_dto.doc_embeddings
    if hasattr(index_state_dto, 'doc_indices_by_discrete_value'):
      self._doc_indices_by_discrete_value = (
          index_state_dto.doc_indices_by_discrete_value
      )
    else:
      self._doc_indices_by_discrete_value = {}
    # Build the searcher index.
    await self.searcher.build(np.stack(self._doc_embeddings, axis=0))  # pytype: disable=wrong-arg-count


@dataclasses.dataclass(kw_only=True)
class RewritingConstrainedEmbeddingBasedDocumentIndex(
    RewritingConstrainedEmbeddingBasedIndex[str, Document, Document],
    ConstrainedEmbeddingBasedDocumentIndex,
):
  """Embedding-based index with rewriting for Document objects with constraints."""

  inner_index: ConstrainedEmbeddingBasedDocumentIndex = dataclasses.field(
      default_factory=ConstrainedEmbeddingBasedDocumentIndex
  )

  def save(self, base_path: str) -> None:
    """Saves the index state."""
    self.inner_index.save(base_path)  # pytype: disable=attribute-error

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state."""
    await self.inner_index.load(base_path)  # pytype: disable=wrong-arg-count,attribute-error
