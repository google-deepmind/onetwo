# Copyright 2025 DeepMind Technologies Limited.
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

"""Interfaces used in general retrieval QA strategies."""

import abc
from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

from onetwo.core import executing

# Types used to represent the objects (documents) that can be retrieved or added
# to a corpus, and the data type that can be used to query for them.
#
# The `_RetrievalResult` and `_Doc` types may or may not be the same (e.g.,
# could accept a structured document data structure for adding to the corpus,
# but then convert the relevant content to plain text at retrieval time).
# These are also not strictly limited to traditional "document"-like data
# structures, but could be an object that we want to index and retrieve.
#
# The `_Query` would most traditionally be a `str`, but it could also be
# something like a `ChunkList` (to support multimodal queries), or it could be
# identical to the `_RetrievalResult` and/or `_Doc` type (e.g., an `Entity` or
# `Passage`) in the case where we want to retrieve by object similarity.

QueryT = TypeVar('QueryT')
RetrievalResultT = TypeVar('RetrievalResultT')
DocT = TypeVar('DocT')


class Retriever(Protocol[QueryT, RetrievalResultT]):
  """Generic interface for a strategy that retrieves results given a query.

  The primary way to use a Retriever instance is to call it directly
  (e.g., `retriever_instance(query)`) which invokes the `__call__` method.

  Subclasses should implement the core retrieval logic in the `_retrieve`
  method. The `__call__` method is often a wrapper around `_retrieve` and serves
  as the main public entry point.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def _retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
  ) -> Iterable[RetrievalResultT]:
    """Core retrieval logic to be implemented by subclasses.

    This method should contain the actual implementation for fetching and
    ranking results based on the query. It is intended to be called internally,
    typically by `__call__` or `retrieve_with_scores`.

    The results will be sorted from most relevant to least relevant.

    Args:
      query: The query for which to retrieve results. Could either be a
        traditional search-style query, a natural language question, or a longer
        document or other object for which we wish to retrieve similar/related
        objects.
      max_results: The maximum number of results to return. If `None`, then all
        results will be returned (or some hard-coded limit determined by the
        subclass).
    """

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Returns the results that are most relevant to the query, with scores."""
    results = await self._retrieve(query=query, max_results=max_results)  # pytype: disable=wrong-keyword-args
    # By default, we assign a score that is inversely proportional to the
    # position of the result in the list. This is a reasonable default for
    # cases where the `retrieve` method itself is already returning the results
    # in order of relevance.
    return [(result, 1.0 / (ind + 1)) for ind, result in enumerate(results)]

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      query: str,
      *,
      max_results: int | None = None,
  ) -> Iterable[RetrievalResultT]:
    """Returns the documents (or passages) that are most relevant to the query.

    This is the primary method users should call to perform retrieval.
    Implementations of this method in subclasses usually delegate the core
    logic to the `_retrieve` method.

    The results will be sorted from most relevant to least relevant.

    Args:
      query: The query for which to retrieve results. Could either be a
        traditional search-style query, a natural language question, or a longer
        document for which we wish to retrieve similar/related documents.
      max_results: The maximum number of results to return. If `None`, then all
        results will be returned (or some hard-coded limit determined by the
        subclass).
    """
    return await self._retrieve(query=query, max_results=max_results)  # pytype: disable=wrong-keyword-args


class Index(
    Retriever[QueryT, RetrievalResultT], Generic[QueryT, RetrievalResultT, DocT]
):
  """Generic interface for a container that organizes objects for retrieval.

  The objects managed by this class are referred to as "documents", but can be
  of an arbitrary type (determined by the subclass via the `_Doc` type
  parameter) -- e.g., they could be either plain strings, or sequences of
  multimodal content chunks, or arbitrary structured data types.

  The objects returned at retrieval time may or may not be the same objects (or
  same type of objects) that are added to the index. The type of object that is
  returned is determined independently by the subclass via the
  `_RetrievalResult` type parameter. E.g., the subclass may index documents in
  the form of structured data, and then convert the structured data to plain
  text before returning the results to the caller at retrieval time. Similarly,
  even when dealing with objects of a uniform type, the subclass has the
  flexibility to decide whether to return some selection of the original
  documents at retrieval time, or to return some content derived from the
  original documents, e.g., a smaller chunk of text, or some other text that is
  derived in a non-trivial way from the original documents.

  The typical usage pattern is that `add_docs` will be called one or more times,
  followed by one or more calls to `retrieve`. Any relevant preprocessing or
  indexing of the documents can be done either greedily during `add_docs` or
  lazily during the first call to `retrieve`. It is up to the subclass whether
  or not to support interleaved calls to `add_docs` and `retrieve`. If this is
  not supported, e.g., due to some kind of global cross-document index building,
  the subclass may want to implement some kind of `finalized` flag that is set
  to True at the point where the one-time processing is completed and the corpus
  is finalized (e.g., during the first call to `retrieve`, or when some kind of
  `finalize` method is explicitly called).
  """

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

  def destroy(self) -> None:
    """Destroys the index and frees up related resources."""
