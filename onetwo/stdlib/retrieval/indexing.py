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

"""Index classes for retrieval and retrieval-augmented generation.

Design principles:

* To support uniform eval sweeps and modularity, we define all
indexing strategies of interest in the form of an inheritance
hierarchy, with `Index` at the top, followed by `DocumentIndex` and its various
concrete implementations below that.
* The top-level `Index` is generically parametrized by the specific choice of
query and document data structure.
* The `DocumentIndex` class specifies an opinionated choice of
document representation (the `Document` data structure defined in
`retrieval_data_structures.py`) and query representation (str) and is intended
to serve as the superclass of all indexing/retrieval strategies used.
"""

import abc
import collections
import dataclasses
import random
from typing import Any, Callable, Generic, Iterable, Sequence

import numpy as np
from onetwo.builtins import llm
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import searchers


QueryT = retrieval.QueryT
RetrievalResultT = retrieval.RetrievalResultT
DocT = retrieval.DocT
Document = retrieval_data_structures.Document


class Index(
    retrieval.Retriever[QueryT, RetrievalResultT],
    Generic[QueryT, RetrievalResultT, DocT],
    metaclass=abc.ABCMeta,
):
  """Generic interface for a container that organizes objects for retrieval.

  The objects managed by this class are referred to as "documents", but can be
  of an arbitrary type (determined by the subclass via the `DocT` type
  parameter) -- e.g., they could be either plain strings, or sequences of
  multimodal content chunks, or arbitrary structured data types.

  The objects returned at retrieval time may or may not be the same objects (or
  same type of objects) that are added to the index. The type of object that is
  returned is determined independently by the subclass via the
  `RetrievalResultT` type parameter. E.g., the subclass may index documents in
  the form of structured data, and then convert the structured data to plain
  text before returning the results to the caller at retrieval time. Similarly,
  even when dealing with objects of a uniform type, the subclass has the
  flexibility to decide whether to return some selection of the original
  documents at retrieval time, or to return some content derived from the
  original documents, e.g., a smaller chunk of text, or some other text that is
  derived in a non-trivial way from the original documents.

  The typical usage pattern is to call `create_index` once to populate the
  corpus and create the index, followed by one or more calls to the index
  instance itself (which invokes the `__call__` method) to perform retrieval.

  It is possible to add additional documents to the index by calling `add_docs`
  multiple times, but this should be done only after `create_index` has been
  called once. Subclasses determine whether to support interleaved calls to
  `add_docs` and retrieval. If not supported, for example, due to
  global cross-document index building, the subclass might implement a
  `finalized` flag. This flag would be set to True once one-time processing of
  the corpus is complete and the indexing is finalized (e.g., during the first
  retrieval call or via an explicit `finalize` method). Any relevant
  preprocessing or indexing of the documents can be done either greedily within
  the `add_docs` method or lazily during the first retrieval call.

  Attributes:
    corpus_name: Human-readable name of the corpus, which should ideally be
      descriptive of the scope of documents that are expected to be added.
      Depending on the implementation, this may be used either purely for
      debugging/logging purposes, or else may be used also for identifying the
      corpus within some shared corpus management infrastructure.
  """

  corpus_name: str = 'unnamed'

  @property
  @abc.abstractmethod
  def num_docs(self) -> int:
    """Returns the number of (possibly derived) documents in the corpus."""

  @abc.abstractmethod
  def get_docs(self) -> Iterable[DocT]:
    """Returns the (possibly derived) documents in the corpus."""

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Adds the documents to the corpus.

    Args:
      docs: The documents to add to the corpus.
    """

  @executing.make_executable(copy_self=False)
  @tracing.trace('Index.create_index')
  async def create_index(self, corpus_name: str, docs: Iterable[DocT]) -> None:
    """Creates the index with given documents. Typically called only once.

    Args:
      corpus_name: Human-readable name of the corpus.
      docs: The initial documents to add to the corpus.
    """
    if self.num_docs > 0:
      raise ValueError(
          'Attempting to call `create_index` on a corpus with existing'
          f' documents: old_name={self.corpus_name}, new_name={corpus_name},'
          f' num_existing_docs={self.num_docs},'
          f' num_docs_to_add={len(list(docs))}'
      )

    self.corpus_name = corpus_name
    await self.add_docs(docs)  # pytype: disable=wrong-arg-count

  @abc.abstractmethod
  def destroy_index(self) -> None:
    """Destroys the index and frees up related resources."""


@dataclasses.dataclass(kw_only=True)
class IndexWrapper(
    Index[QueryT, RetrievalResultT, DocT],
):
  """Index that wraps an inner index, delegating calls by default.

  Attributes:
    inner_index: The inner index to wrap.
  """

  inner_index: Index[QueryT, RetrievalResultT, DocT]

  @property
  def num_docs(self) -> int:
    """Overridden from base class (Index)."""
    return self.inner_index.num_docs

  def get_docs(self) -> Iterable[DocT]:
    return self.inner_index.get_docs()

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    await self.inner_index.add_docs(docs=docs)  # pytype: disable=wrong-keyword-args

  @executing.make_executable(copy_self=False)
  async def create_index(self, corpus_name: str, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    await self.inner_index.create_index(corpus_name=corpus_name, docs=docs)  # pytype: disable=wrong-keyword-args

  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self.inner_index.destroy_index()

  @executing.make_executable(copy_self=False)
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[RetrievalResultT]:
    """Overridden from base class (Retriever)."""
    return await self.inner_index.retrieve(query, max_results=max_results, **kwargs)  # pytype: disable=wrong-arg-count

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Overridden from base class (Retriever)."""
    return await self.inner_index.retrieve_with_scores(  # pytype: disable=wrong-arg-count
        query, max_results=max_results, **kwargs
    )


@dataclasses.dataclass(kw_only=True)
class ChunkingIndex(IndexWrapper[QueryT, RetrievalResultT, DocT]):
  """Index that chunks documents before adding them to an inner index.

  By default, this class treats the chunks as the primary units of retrieval.
  When `get_docs`, `retrieve` or `retrieve_with_scores` is called, the objects
  returned are the individual chunks (derived from the original documents) as
  stored in the inner index, rather than the original source documents.

  Future iterations could support alternative behaviors, such as:
  1.  **Parent-Document Retrieval**: Returning the original document associated
      with a retrieved chunk.
  2.  **Contextual Windowing**: Returning a chunk along with its neighboring
      surrounding context.

  Attributes:
    chunker: Chunker used to split the documents.
  """

  chunker: chunking.Chunker = dataclasses.field(
      default_factory=chunking.NoChunking
  )

  @executing.make_executable(copy_self=False)
  async def create_index(self, corpus_name: str, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    if self.inner_index.num_docs > 0:
      raise ValueError(
          'Attempting to call `create_index` on a corpus with existing'
          f' documents: old_name={self.inner_index.corpus_name},'
          f' new_name={corpus_name},'
          f' num_existing_docs={self.inner_index.num_docs},'
          f' num_docs_to_add={len(list(docs))}'
      )

    self.inner_index.corpus_name = corpus_name  # pytype: disable=attribute-error
    await self.add_docs(docs)  # pytype: disable=wrong-arg-count

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    chunks_by_doc = await executing.parallel(
        *[self.chunker(doc) for doc in docs]  # pytype: disable=wrong-arg-count
    )
    chunks = []
    for chunk_list in chunks_by_doc:
      chunks.extend(chunk_list)
    await self.inner_index.add_docs(docs=chunks)  # pytype: disable=wrong-keyword-args


@dataclasses.dataclass
class DocumentIndex(
    Index[str, Document, Document],
    metaclass=abc.ABCMeta,
):
  """All DocumentIndex classes should be subclasses of this type."""


@dataclasses.dataclass
class EmptyIndex(DocumentIndex):
  """Trivial empty index that always retrieves empty results."""

  @property
  def num_docs(self) -> int:
    """Overridden from base class (Index)."""
    return 0

  def get_docs(self) -> Iterable[Document]:
    """Overridden from base class (Index)."""
    return []

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmptyIndex.add_docs')
  async def add_docs(self, docs: Iterable[Document]) -> None:
    """Overridden from base class (Index)."""
    pass

  @tracing.trace('EmptyIndex.destroy_index')
  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmptyIndex.retrieve')
  async def retrieve(
      self,
      query: str,
      *,
      max_results: int | None = None,
  ) -> Iterable[Document]:
    """Overridden from base class (Retriever)."""
    del query, max_results
    return []


@dataclasses.dataclass
class NaiveIndex(DocumentIndex):
  """Index that stores the original document text as-is and retrieves naively.

  Simply returns the first K added documents each time, regardless of the query.

  Attributes:
    docs: The documents that were added to the corpus.
    shuffle_results: Whether to shuffle the resulting documents before returning
      them. Useful for synthetic datasets where distractor documents (i.e.,
      documents that are irrelevant to the query, used to challenge the
      retriever) are added in the same order as golden documents.
  """

  docs: list[Document] = dataclasses.field(default_factory=list)
  shuffle_results: bool = False

  @property
  def num_docs(self) -> int:
    """Overridden from base class (Index)."""
    return len(self.docs)

  def get_docs(self) -> Iterable[Document]:
    """Overridden from base class (Index)."""
    return self.docs

  @executing.make_executable(copy_self=False)
  @tracing.trace('NaiveIndex.add_docs')
  async def add_docs(self, docs: Iterable[Document]) -> None:
    """Overridden from base class (Index)."""
    self.docs.extend(docs)

  @tracing.trace('NaiveIndex.destroy_index')
  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self.docs.clear()

  @executing.make_executable(copy_self=False)
  @tracing.trace('NaiveIndex.retrieve')
  async def retrieve(
      self,
      query: str,
      *,
      max_results: int | None = None,
  ) -> Iterable[Document]:
    """Overridden from base class (Retriever)."""
    if max_results is None:
      max_results = len(self.docs)
    if not self.shuffle_results:
      return self.docs[:max_results]
    random.seed(query)
    result = self.docs[:max_results].copy()
    random.shuffle(result)
    return result


def default_prepare_query(query: Any) -> str:
  """Returns an appropriately formatted query string for embeddings."""
  return f'task: search result | query: {str(query)}'


# TODO: Determine the most effective default formatting strategy.
def default_prepare_document(doc: Any) -> str:
  """Returns an appropriately formatted doc string for embeddings."""
  if not isinstance(doc, retrieval_data_structures.Document):
    return str(doc)
  # TODO: Implement proper support for multimodal content.
  # Currently when formatting the document contents for passing to the
  # embedding model, we are implicitly converting it to a string, which
  # ignores any multimodal portions.
  title = doc.title or 'none'
  text = doc.content or 'none'
  return f'title: {title} | text: {text}'


# fmt: off
@dataclasses.dataclass(kw_only=True)
class AbstractEmbeddingBasedIndex(
    Index[QueryT, RetrievalResultT, DocT],
    constrained_retrieval.ConstrainedRetriever[QueryT, RetrievalResultT],
):
  r"""Abstract base class for embedding-based retrieval indices.

  This base class implements various functionality around processing of
  queries and documents prior to embedding that is relevant to be in-memory and
  distributed embedding-based indices. However, note that the child class is
  still responsible for providing concrete implementations for `add_docs`,
  `retrieve` and `retrieve_with_scores` along with the embedding functionality.

  Attributes:
    prepare_query: Callable that prepares the query for embedding.
    prepare_document: Callable that prepares the document for embedding.
    task_type: Task type to use when calling `llm.embed` (optional).
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
  # fmt: on
  prepare_query: Callable[[QueryT], str] = default_prepare_query
  prepare_document: Callable[[DocT], str] = default_prepare_document

  task_type: str | None = None
  discrete_field_extractors: dict[str, Callable[[DocT], Any]] = (
      dataclasses.field(default_factory=dict)
  )


def _validate_docs(docs: Sequence[Document]) -> None:
  """Validates that the given documents all have doc_id attributes.

  Args:
    docs: The documents to validate.

  Raises: ValueError if one or more are missing IDs.
  """
  docs_missing_ids = []
  for i, doc in enumerate(docs):
    if doc.doc_id is None:
      docs_missing_ids.append(i)
  if docs_missing_ids:
    raise ValueError(
        f'Docs missing doc_id ({len(docs_missing_ids)} / {len(list(docs))}): '
        f'{docs_missing_ids[:10]} ...'
    )


@dataclasses.dataclass(kw_only=True)
class EmbeddingBasedIndex(
    AbstractEmbeddingBasedIndex[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """In-memory index that retrieves documents based on embedding similarity.

  This class implements an index that stores document embeddings in memory and
  performs retrieval by computing the similarity between query and document
  embeddings. It uses the `llm.embed` action to generate embeddings.

  Attributes:
    searcher: Searcher used for nearest neighbor search between query and
      document embeddings.
  """

  searcher: searchers.Searcher = dataclasses.field(
      default_factory=searchers.ConstrainedBruteForceSearcher
  )
  # List of the documents that are added to the index.
  _docs: list[DocT] = dataclasses.field(default_factory=list)
  # List of the embeddings of the documents that are added to the index.
  _doc_embeddings: list[np.ndarray] = dataclasses.field(default_factory=list)

  # Internal reverse index for discrete field values, used for constrained
  # retrieval. This is a mapping from field name to a mapping of field value to
  # a list of matching document indices. The name is qualified with 'discrete_'
  # to distinguish it from potential future fields whose values are not discrete
  # (e.g., continuous values for which constraints may be specified in terms of
  # a value range).
  _doc_indices_by_discrete_value: dict[str, dict[Any, list[int]]] = (
      dataclasses.field(
          default_factory=lambda: collections.defaultdict(
              lambda: collections.defaultdict(list)
          )
      )
  )

  @executing.make_executable(copy_self=False)
  async def _calculate_query_embedding(self, query: QueryT) -> Sequence[float]:
    """Returns a newly-calculated embedding of the query."""
    query_text = self.prepare_query(query)
    query_embedding = await llm.embed(query_text, task_type=self.task_type)  # pytype: disable=wrong-arg-count
    return query_embedding

  @executing.make_executable(copy_self=False)
  async def _calculate_doc_embeddings(
      self, docs: Sequence[DocT]
  ) -> Sequence[Sequence[float]]:
    """Returns newly-calculated embeddings of the documents."""
    doc_texts = [self.prepare_document(doc) for doc in docs]
    embed_executables = [
        llm.embed(text, task_type=self.task_type) for text in doc_texts  # pytype: disable=wrong-arg-count
    ]
    embedded_results = await executing.parallel(*embed_executables)
    return embedded_results

  @property
  def num_docs(self) -> int:
    """Overridden from base class (Index)."""
    return len(self._docs)

  def get_docs(self) -> Iterable[DocT]:
    """Overridden from base class (Index)."""
    return self._docs

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
  @tracing.trace('EmbeddingBasedIndex.add_docs')
  async def add_docs(self, docs: Sequence[DocT]) -> None:
    """Overridden from base class (Index)."""
    _validate_docs(docs)
    num_existing_docs = len(self._docs)
    self._docs.extend(docs)
    self._update_doc_indices_by_discrete_value(docs, num_existing_docs)
    embedded_results = await self._calculate_doc_embeddings(docs)  # pytype: disable=wrong-arg-count
    self._doc_embeddings.extend([np.array(embed) for embed in embedded_results])
    # Build the searcher index.
    await self.searcher.build(np.stack(self._doc_embeddings, axis=0))  # pytype: disable=wrong-arg-count

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.retrieve_doc_indices_and_scores')
  async def retrieve_doc_indices_and_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
  ) -> Sequence[tuple[int, float]]:
    """Returns indices and scores of the most relevant documents.

    The indices returned are compatible with the list of documents returned
    by `get_docs()`.

    Args:
      query: The query used for retrieval.
      max_results: The maximum number of results to return.
      min_score: The minimum similarity score for a result to be included.
      constraints: Optional retrieval constraints to filter the documents.
    """
    query_embedding = await self._calculate_query_embedding(query)  # pytype: disable=wrong-arg-count

    # For compatibility with search_batched, which expects a batch of queries.
    query_embeddings = np.array([query_embedding])
    if constraints is not None:
      candidate_indices_list = constrained_retrieval.get_candidate_indices(
          constraints, self._doc_indices_by_discrete_value
      )
      if not candidate_indices_list:
        return []

      # Pass the candidate_indices_list to search_batched.
      # The searcher will handle searching only within these indices.
      ranked_doc_indices, ranked_doc_scores = (
          await self.searcher.search_batched(  # pytype: disable=wrong-keyword-args
              query_embeds=query_embeddings,
              final_num_neighbors=max_results,
              doc_indices=candidate_indices_list,
          )
      )
    else:
      ranked_doc_indices, ranked_doc_scores = await self.searcher.search_batched(  # pytype: disable=wrong-keyword-args
          query_embeds=query_embeddings, final_num_neighbors=max_results
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
  async def retrieve_doc_score(self, query: str, doc_index: int) -> float:
    """Returns similarity score of the document at `doc_index` for a query."""
    query_embedding = await self._calculate_query_embedding(query)  # pytype: disable=wrong-arg-count
    query_embedding = np.array([query_embedding])
    doc_embedding = self._doc_embeddings[doc_index]
    return np.dot(query_embedding, doc_embedding)

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.retrieve_with_scores')
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
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
  @tracing.trace('EmbeddingBasedIndex.retrieve')
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

  @tracing.trace('EmbeddingBasedIndex.destroy_index')
  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self._docs = []
    self._doc_embeddings = []
    self._doc_indices_by_discrete_value = {}
    self.searcher.destroy()


@dataclasses.dataclass(kw_only=True)
class ChunkingEmbeddingBasedIndex(
    ChunkingIndex[QueryT, RetrievalResultT, DocT],
    EmbeddingBasedIndex[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """Index that chunks documents and provides embedding-specific methods.

  This class combines the behavior of `ChunkingIndex` with the vector-search
  capabilities of `EmbeddingBasedIndex`. Note that the document access and
  retrieval methods (`get_docs`, `retrieve`, `retrieve_with_scores`, and
  `retrieve_doc_indices_and_scores`) currently return the specific chunks and
  their corresponding chunk-level embeddings/scores.

  As with `ChunkingIndex`, this behavior is currently "chunk-centric". Future
  updates may introduce the ability to map these results back to the original
  document IDs.

  Attributes:
    inner_index: The underlying `EmbeddingBasedIndex` used for vector storage
      and similarity search.
  """

  inner_index: EmbeddingBasedIndex[QueryT, RetrievalResultT, DocT]

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_indices_and_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      constraints: constrained_retrieval.RetrievalConstraints | None = None,
  ) -> Sequence[tuple[int, float]]:
    """Delegates to the inner index."""
    return await self.inner_index.retrieve_doc_indices_and_scores(  # pytype: disable=attribute-error,wrong-keyword-args,wrong-arg-count
        query,
        max_results=max_results,
        min_score=min_score,
        constraints=constraints,
    )

  def rename_discrete_field_key(
      self, field_name: str, old_key: str, new_key: str
  ) -> None:
    """Delegates to the inner index."""
    self.inner_index.rename_discrete_field_key(  # pytype: disable=attribute-error
        field_name, old_key, new_key
    )

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_score(self, query: str, doc_index: int) -> float:
    """Delegates to the inner index."""
    return await self.inner_index.retrieve_doc_score(  # pytype: disable=attribute-error,wrong-keyword-args,wrong-arg-count
        query, doc_index
    )

from onetwo.stdlib.retrieval import serialization
serializer = serialization.SimpleEmbeddingBasedIndexSerializer


@dataclasses.dataclass(kw_only=True)
class EmbeddingBasedDocumentIndex(
    EmbeddingBasedIndex[str, Document, Document],
    DocumentIndex,
):
  """Embedding-based index for Document objects."""

  def save(self, base_path: str) -> None:
    """Saves the index state."""
    index_state_dto = index_state.EmbeddingBasedIndexState(
        docs=self._docs,
        doc_embeddings=self._doc_embeddings,
        doc_indices_by_discrete_value=self._doc_indices_by_discrete_value,
    )
    serializer(Document).save(index_state_dto, base_path)

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state."""
    index_state_dto = await serializer(Document).load(base_path)  # pytype: disable=wrong-arg-count
    self._docs = index_state_dto.docs
    self._doc_embeddings = index_state_dto.doc_embeddings
    self._doc_indices_by_discrete_value = (
        index_state_dto.doc_indices_by_discrete_value
    )
    # Build the searcher index.
    await self.searcher.build(np.stack(self._doc_embeddings, axis=0))  # pytype: disable=wrong-arg-count


@dataclasses.dataclass(kw_only=True)
class ChunkingEmbeddingBasedDocumentIndex(
    ChunkingEmbeddingBasedIndex[str, Document, Document],
    EmbeddingBasedDocumentIndex,
):
  """Embedding-based index with chunking for Document objects."""

  inner_index: EmbeddingBasedDocumentIndex = dataclasses.field(
      default_factory=EmbeddingBasedDocumentIndex
  )

  def save(self, base_path: str) -> None:
    """Saves the index state."""
    self.inner_index.save(base_path)

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state."""
    await self.inner_index.load(base_path)  # pytype: disable=wrong-arg-count
