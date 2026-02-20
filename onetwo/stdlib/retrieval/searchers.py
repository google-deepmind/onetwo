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

"""Searchers for nearest neighbor search between query and documents."""

import abc
import dataclasses
from typing import Sequence
import numpy as np
from onetwo.core import executing


class Searcher(metaclass=abc.ABCMeta):
  """Generic interface for a class that implements nearest neighbor search."""

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def build(self, doc_embeds: np.ndarray) -> None:
    """Builds an index of the documents so that they can be searched later.

    Normally `build` is called only once, but if it is called a second time, the
    specified `doc_embeds` will replace whatever was provided before.

    Arguments:
      doc_embeds: Two dimensional array representing the embeddings of the
        documents that are to be made available for retrieval. Note that the
        order matters here, as the return value of `search_batched` will refer
        to documents in terms of their index within this sequence.
    """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def search_batched(
      self,
      query_embeds: np.ndarray,
      *,
      final_num_neighbors: int | None = None,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns indices and scores of the most relevant docs for the query.

    Arguments:
      query_embeds: Two dimension array representing the embeddings of one or
        more queries to search for, where `query_embeds[i]` represents the
        embeddings of the ith query.
      final_num_neighbors: The number of neighbors to return for each query.
      **kwargs: Optional additional arguments (implementation-specific).

    Returns:
      A tuple of (top_ids, top_scores), representing the nearest neighbors,
      starting from the nearest. Each of `top_ids` and `top_scores` is a
      two-dimensional array, whose first dimension is the length of
      `query_embeds` and whose second dimension is the lesser of
      `final_num_neighbors` and the total number of documents in the index.
    """
    pass  # pytype: disable=bad-return-type

  @abc.abstractmethod
  def destroy(self) -> None:
    """Destroys the searcher and frees up related resources."""


class ConstrainedSearcher(Searcher):
  """Searcher that applies constraints to the retrieval process."""

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def search_batched(
      self,
      query_embeds: np.ndarray,
      *,
      final_num_neighbors: int | None = None,
      doc_indices: Sequence[int] | None = None,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns indices and scores of the most relevant docs for the query.

    Arguments:
      query_embeds: Two dimension array representing the embeddings of one or
        more queries to search for, where `query_embeds[i]` represents the
        embeddings of the ith query.
      final_num_neighbors: The number of neighbors to return for each query.
      doc_indices: The indices of the documents to search over. If None or an
        empty sequence, then all documents will be considered.
      **kwargs: Optional additional arguments (implementation-specific).

    Returns:
      A tuple of (top_ids, top_scores), representing the nearest neighbors,
      starting from the nearest. Each of `top_ids` and `top_scores` is a
      two-dimensional array, whose first dimension is the length of
      `query_embeds` and whose second dimension is the lesser of
      `final_num_neighbors` and the total number of documents in the index.
    """
    pass  # pytype: disable=bad-return-type


@dataclasses.dataclass
class BruteForceSearcher(Searcher):
  """Searcher that determines nearest neighbors via dot product similarity.

  This searcher supports two modes of operation based on the `normalize_embeds`
  flag. By default, it performs a raw dot product, which is the industry
  standard for performance when embeddings are already unit-normalized. If the
  embeddings are not normalized (e.g., after Matryoshka truncation), setting
  `normalize_embeds` to True effectively converts the operation to
  cosine similarity.

  Attributes:
    normalize_embeds: If True, L2-normalizes both document and query embeddings
      to unit length during the `build` and `search_batched` phases. This
      ensures the resulting scores represent cosine similarity (range [-1, 1]).
      Note: Input arrays are not modified in-place; a normalized copy is stored.
  """

  normalize_embeds: bool = False
  _doc_embeds: np.ndarray | None = None

  @executing.make_executable(copy_self=False)
  async def build(self, doc_embeds: np.ndarray) -> None:
    """Builds an index of the documents so that they can be searched later."""
    if self.normalize_embeds:
      norms = np.linalg.norm(doc_embeds, axis=1, keepdims=True)
      normalized_embeds = doc_embeds / (norms + 1e-10)
      self._doc_embeds = normalized_embeds
    else:
      self._doc_embeds = doc_embeds

  @executing.make_executable(copy_self=False)
  async def search_batched(
      self,
      query_embeds: np.ndarray,
      *,
      final_num_neighbors: int | None = None,
      **kwargs,
  ) -> tuple[Sequence[int], Sequence[float]]:
    """Returns ranked indices of the most relevant docs for the query."""
    if self._doc_embeds is None:
      raise ValueError('Need to call `build()` before `search_batched()`.')
    if self.normalize_embeds:
      q_norms = np.linalg.norm(query_embeds, axis=1, keepdims=True)
      query_embeds = query_embeds / (q_norms + 1e-10)
    scores = query_embeds.dot(self._doc_embeds.T)  # Q x D
    if final_num_neighbors is None:
      top_ids = scores.argsort(axis=1)[:, ::-1]
    else:
      k = min(final_num_neighbors, scores.shape[1])
      # Get top_k scores, but not in sorted order.
      # This is faster than sorting all scores (O(D) vs O(D*log(D))).
      top_k_indices_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]
      top_k_scores_unsorted = np.take_along_axis(
          scores, top_k_indices_unsorted, axis=1
      )
      # Sort to get top_k scores in decreasing order.
      sorted_order_in_top_k = np.argsort(top_k_scores_unsorted, axis=1)[:, ::-1]
      top_ids = np.take_along_axis(
          top_k_indices_unsorted, sorted_order_in_top_k, axis=1
      )
    top_scores = np.take_along_axis(scores, top_ids, axis=1)
    return top_ids, top_scores  # (Q x top_k, Q x top_k)

  def destroy(self) -> None:
    """Overridden from base class (Searcher)."""
    self._doc_embeds = None

  def __eq__(self, other):
    if not isinstance(other, BruteForceSearcher):
      return NotImplemented
    if self.normalize_embeds != other.normalize_embeds:
      return False
    if self._doc_embeds is None or other._doc_embeds is None:
      return self._doc_embeds is None and other._doc_embeds is None
    return np.array_equal(self._doc_embeds, other._doc_embeds)


class ConstrainedBruteForceSearcher(BruteForceSearcher, ConstrainedSearcher):
  """Brute force searcher that applies constraints to the retrieval process."""

  @executing.make_executable(copy_self=False)
  async def search_batched(
      self,
      query_embeds: np.ndarray,
      *,
      final_num_neighbors: int | None = None,
      doc_indices: Sequence[int] | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Overridden from base class (ConstrainedSearcher)."""

    if self._doc_embeds is None:
      raise ValueError('Need to call `build()` before `search_batched()`.')

    if self.normalize_embeds:
      q_norms = np.linalg.norm(query_embeds, axis=1, keepdims=True)
      query_embeds = query_embeds / (q_norms + 1e-10)

    if doc_indices is None or not doc_indices:
      search_embeds = self._doc_embeds
      original_indices_map = np.arange(self._doc_embeds.shape[0])
    else:
      if max(doc_indices) >= self._doc_embeds.shape[0] or min(doc_indices) < 0:
        raise ValueError('`doc_indices` are out of bounds.')
      search_embeds = self._doc_embeds[doc_indices]
      original_indices_map = np.array(doc_indices)

    scores = np.dot(query_embeds, search_embeds.T)  # Q x D_search
    if final_num_neighbors is None:
      top_ids_local = scores.argsort(axis=1)[:, ::-1]
    else:
      k = min(final_num_neighbors, scores.shape[1])
      # Get top_k scores, but not in sorted order.
      # This is faster than sorting all scores (O(D) vs O(D*log(D))).
      top_k_indices_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]
      top_k_scores_unsorted = np.take_along_axis(
          scores, top_k_indices_unsorted, axis=1
      )
      # Sort to get top_k scores in decreasing order.
      sorted_order_in_top_k = np.argsort(top_k_scores_unsorted, axis=1)[:, ::-1]
      top_ids_local = np.take_along_axis(
          top_k_indices_unsorted, sorted_order_in_top_k, axis=1
      )

    # Map local indices back to the original index space.
    top_ids = original_indices_map[top_ids_local]
    top_scores = np.take_along_axis(scores, top_ids_local, axis=1)
    return top_ids, top_scores  # (Q x top_k, Q x top_k)
