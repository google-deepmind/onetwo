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

"""Utilities for rewriting corpora."""

from collections.abc import Iterable
import dataclasses
from typing import Protocol, Sequence, TypeVar

from onetwo import ot
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import chunking

DocT = TypeVar('DocT')


class CorpusRewriter(Protocol[DocT]):
  """Generic interface for a strategy that rewrites a corpus.

  A corpus is an iterable of documents. The corpus rewriter
  may modify the documents in any way. For example, it may chunk all the
  documents and put the chunks together in a new corpus, format each
  document into a specific format, or any other transformation on the corpus.
  """

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      corpus: Iterable[DocT],
  ) -> Iterable[DocT]:
    """Returns a rewritten corpus.

    Args:
      corpus: The corpus to rewrite.

    Returns:
      The rewritten corpus represented as an iterable of documents.
    """


class NoRewriting(CorpusRewriter[DocT]):
  """Rewriter that returns the corpus as is."""

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      corpus: Iterable[DocT],
  ) -> Iterable[DocT]:
    """Returns the corpus as is.

    Args:
      corpus: The corpus to rewrite.

    Returns:
      The corpus as is.
    """
    return corpus


@dataclasses.dataclass(frozen=True)
class ChunkingCorpusRewriter(CorpusRewriter[DocT]):
  """Rewriter that chunks documents individually.

  A chunker generates one or more documents from a given document, usually
  excerpts of the original document. This CorpusRewriter chunks each document
  in parallel and returns a new corpus with all the chunks. For more details on
  chunkers, see `onetwo.stdlib.retrieval.chunking`.

  Attributes:
    chunker: The chunker to use for chunking documents.
  """

  chunker: chunking.Chunker

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      corpus: Iterable[DocT],
  ) -> Iterable[DocT]:
    """Overridden from base class (CorpusRewriter)."""
    chunked_corpus = []
    for chunks in await ot.parallel(
        *[self.chunker(document) for document in corpus]  # pytype: disable=wrong-arg-count
    ):
      chunked_corpus.extend(chunks)
    return chunked_corpus


@dataclasses.dataclass(frozen=True)
class SequentialCorpusRewriter(CorpusRewriter[DocT]):
  """Rewriter that applies a sequence of corpus rewriters in order.

  This CorpusRewriter can be useful for applying multiple transformations to a
  corpus, for example chunking and then formatting each chunk. This allows
  creating a configurable pipeline of transformations that are applied to a
  corpus.

  # TODO: Add example of a rewriter applying chunking and
  # formatting once the formatting interface is implemented.

  Attributes:
    rewriters: The sequence of corpus rewriters to apply.
  """

  rewriters: Sequence[CorpusRewriter[DocT]]

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      corpus: Iterable[DocT],
  ) -> Iterable[DocT]:
    """Overridden from base class (CorpusRewriter)."""
    rewritten_corpus = corpus
    for rewriter in self.rewriters:
      rewritten_corpus = await rewriter(rewritten_corpus)  # pytype: disable=wrong-arg-count
    return rewritten_corpus
