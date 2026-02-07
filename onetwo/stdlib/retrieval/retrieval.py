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

"""Retrieval-related interfaces for QA strategies."""

import abc
from collections.abc import Iterable
from typing import Protocol, TypeVar, final

from onetwo.core import executing

# Types used to represent the objects (documents) that can be retrieved or added
# to a corpus, and the data type that can be used to query for them.
#
# The `RetrievalResultT` and `DocT` types may or may not be the same (e.g.,
# could accept a structured document data structure for adding to the corpus,
# but then convert the relevant content to plain text at retrieval time).
# These are also not strictly limited to traditional "document"-like data
# structures, but could be an object that we want to index and retrieve.
#
# The `QueryT` would most traditionally be a `str`, but it could also be
# something like a `ChunkList` (to support multimodal queries), or it could be
# identical to the `RetrievalResultT` and/or `DocT` type in the case where we
# want to retrieve by object similarity.

QueryT = TypeVar('QueryT')
RetrievalResultT = TypeVar('RetrievalResultT')
DocT = TypeVar('DocT')


class Retriever(Protocol[QueryT, RetrievalResultT]):
  """Generic interface for a strategy that retrieves results given a query.

  The primary way to use a Retriever instance is to call it directly
  (e.g., `retriever_instance(query)`) which invokes the `__call__` method.

  Subclasses should implement the core retrieval logic in the `retrieve`
  method. The `__call__` method is final and a wrapper around `retrieve` and
  serves to provide a more convenient API.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
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
      **kwargs: Additional keyword arguments to pass to the `retrieve` method.
    """

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Returns the results that are most relevant to the query, with scores."""
    results = await self.retrieve(query=query, max_results=max_results, **kwargs)  # pytype: disable=wrong-keyword-args
    # By default, we assign a score that is inversely proportional to the
    # position of the result in the list. This is a reasonable default for
    # cases where the `retrieve` method itself is already returning the results
    # in order of relevance.
    return [(result, 1.0 / (ind + 1)) for ind, result in enumerate(results)]

  @executing.make_executable(copy_self=False)
  @final
  async def __call__(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[RetrievalResultT]:
    """Returns the documents (or passages) that are most relevant to the query.

    The `__call__` method is final and a wrapper around `retrieve` and
    serves to provide a more convenient API. Subclasses should implement the
    core retrieval logic in the `retrieve` method.

    The results will be sorted from most relevant to least relevant.

    Args:
      query: The query for which to retrieve results. Could either be a
        traditional search-style query, a natural language question, or a longer
        document for which we wish to retrieve similar/related documents.
      max_results: The maximum number of results to return. If `None`, then all
        results will be returned (or some hard-coded limit determined by the
        subclass).
      **kwargs: Additional keyword arguments to pass to the `retrieve` method.
    """
    return await self.retrieve(query=query, max_results=max_results, **kwargs)  # pytype: disable=wrong-keyword-args
