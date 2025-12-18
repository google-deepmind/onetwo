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

"""Index classes for retrieval and retrieval-augmented generation."""

import abc
from collections.abc import Iterable
from typing import Generic

from onetwo.core import executing
from onetwo.stdlib.retrieval import retrieval

QueryT = retrieval.QueryT
RetrievalResultT = retrieval.RetrievalResultT
DocT = retrieval.DocT
Retriever = retrieval.Retriever


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
