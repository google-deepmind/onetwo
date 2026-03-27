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

r"""Utilities for transforming and pre-processing document collections (corpora).

In a RAG system, raw data is rarely ready for indexing as-is. A **Corpus
Rewriter** is a tool that transforms a collection of documents into a
more useful format before they are indexed. These transformations are critical
for ensuring that the indexed documents are mathematically optimized for the
searcher, and contextually grounded for the user.

We provide implementations of the following common rewriting patterns:

* Chunking: Converting large documents into many smaller, semantically
cohesive pieces. This is typically a $1 \to N$ relationship.
* Formatting: Standardizing how information is presented (e.g., prepending
titles to text) to help the LLM understand the context of each document or
chunk. This is typically a $1 \to 1$ relationship.

We provide a `SequentialCorpusRewriter` that can be used to combine multiple
transformations in a specific order (e.g., first chunk the text, then format
each chunk) before indexing.
"""

from collections.abc import Iterable
import dataclasses
from typing import Protocol, Sequence, TypeVar

from onetwo import ot
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import document_formatting

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
    """Processes the input corpus and returns the transformed version.

    Args:
      corpus: An iterable collection of documents to be processed.

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
  """Rewriter that fragments documents into smaller pieces (chunks).

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
class FormattingCorpusRewriter(CorpusRewriter[DocT]):
  """Rewriter that applies a consistent structure to every document.

  While chunking changes the **quantity** of documents, formatting
  changes the **content structure** of each document (a 1 to 1 mapping).
  A document formatter maps exactly one input document to exactly one output
  document, usually by reformatting its text content (e.g., adding
  titles, applying templates). This CorpusRewriter applies the formatter to
  each document in the corpus in parallel. For more details on document
  formatters, see `onetwo.stdlib.retrieval.document_formatting`.

  Attributes:
    formatter: The document formatter to apply to each document.
  """

  formatter: document_formatting.DocumentFormatter

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      corpus: Iterable[DocT],
  ) -> Iterable[DocT]:
    """Overridden from base class (CorpusRewriter)."""
    return await ot.parallel(
        *[self.formatter(document) for document in corpus]  # pytype: disable=wrong-arg-count
    )


@dataclasses.dataclass(frozen=True)
class SequentialCorpusRewriter(CorpusRewriter[DocT]):
  r"""Rewriter that applies a sequence of corpus rewriters in order.

  This CorpusRewriter can be useful for applying multiple transformations to a
  corpus, for example chunking and then formatting each chunk. This allows
  creating a configurable pipeline of transformations that are applied to a
  corpus.

  Example: A rewriter that chunks documents and then formats each chunk::

    SequentialCorpusRewriter(rewriters=[
        ChunkingCorpusRewriter(chunker=chunking.ChunkByMaxTokens()),
        FormattingCorpusRewriter(
            formatter=document_formatting.TextDocumentFormatter(
                format_str='Title: {title}\nContent: {text}',
            ),
        ),
    ])

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
