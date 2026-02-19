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
from collections.abc import Callable, Iterable
import dataclasses
import random
from typing import Any, Generic

from onetwo.core import executing
from onetwo.core import tracing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import constrained_retrieval
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures

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
class IndexWrapper(Index[QueryT, RetrievalResultT, DocT]):
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
    Generic[QueryT, RetrievalResultT, DocT],
    constrained_retrieval.ConstrainedRetriever[QueryT, RetrievalResultT],
):
  r"""Abstract base class for embedding-based retrieval indices.

  This base class implements various functionality around processing of
  queries and documents prior to embedding that is relevant to be in-memory and
  distributed embedding-based indices. However, note that the child class is
  still responsible for providing concrete implementations for `add_docs`,
  `retrieve` and `retrieve_with_scores` along with the embedding functionality.

  Attributes:
    chunker: Chunker used to split the documents before embedding.
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
  chunker: chunking.Chunker = dataclasses.field(
      default_factory=chunking.NoChunking
  )

  prepare_query: Callable[[QueryT], str] = default_prepare_query
  prepare_document: Callable[[DocT], str] = default_prepare_document

  task_type: str | None = None
  discrete_field_extractors: dict[str, Callable[[DocT], Any]] = (
      dataclasses.field(default_factory=dict)
  )
