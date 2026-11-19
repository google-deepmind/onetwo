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

An **Index** is a specialized container that stores documents and organizes
them for fast retrieval. In a RAG pipeline, the `Index` is the 'Knowledge Base'
that the system queries to find relevant context.

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

Key Index Types:

* RewritingIndex: Acts as a transformation layer. Before documents are
stored, they are passed through a 'Rewriter' (e.g., a Chunker). This is
essential for formatting documents and breaking large documents into smaller,
semantically dense fragments to improve the quality of retrieval.

* EmbeddingBasedIndex: Implements Semantic Search. Instead of traditional
keyword matching, it uses LLM backends to convert documents into
high-dimensional embeddings. Retrieval is performed by finding embeddings that
are mathematically closest to the query embedding.
"""

import abc
import collections
from collections.abc import Container
import contextlib
import dataclasses
import enum
import random
from typing import Any, ClassVar, Generic, Iterable, Iterator, Sequence, TypeVar, cast

from absl import logging
from etils import epath
import numpy as np
from onetwo.backends import backend_registry
from onetwo.backends import backends_base
from onetwo.builtins import llm
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import tracing
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import searchers
from onetwo.stdlib.retrieval import serialization

QueryT = TypeVar('QueryT')
RetrievalResultT = TypeVar('RetrievalResultT')
DocT = TypeVar('DocT')
Document = retrieval_data_structures.Document

# TODO: Revisit this default and consider making it default in
# all public APIs.
DEFAULT_INDEXING_THREADPOOL_SIZE = 30

# Name of the file (in the index directory, next to the inner index's files)
# that a `RewritingIndex` saves its original documents to.
ORIGINAL_DOCS_JSONL_NAME = 'original_docs.jsonl'


def _validate_docs(docs: Sequence[Any]) -> None:
  """Validates that the given documents all have doc_id attributes.

  Args:
    docs: The documents to validate.

  Raises: ValueError if one or more are missing IDs.
  """
  docs_missing_ids = []
  for i, doc in enumerate(docs):
    if getattr(doc, 'doc_id', None) is None:
      docs_missing_ids.append(i)
  if docs_missing_ids:
    raise ValueError(
        f'Docs missing doc_id ({len(docs_missing_ids)} / {len(docs)}): '
        f'{docs_missing_ids[:10]} ...'
    )


def _find_duplicate_doc_ids(
    docs: Sequence[Any], existing_doc_ids: Container[str]
) -> list[str]:
  """Returns the doc_ids of `docs` that are already used.

  A doc_id counts as already used if it is in `existing_doc_ids` or belongs to
  an earlier document in `docs`. Such documents are counted once each.

  Args:
    docs: The documents to check. Must all have doc_ids.
    existing_doc_ids: doc_ids of the documents already in the index.
  """
  duplicate_doc_ids = []
  seen_doc_ids = set()
  for doc in docs:
    doc_id = getattr(doc, 'doc_id')
    if doc_id in existing_doc_ids or doc_id in seen_doc_ids:
      duplicate_doc_ids.append(doc_id)
    seen_doc_ids.add(doc_id)
  return duplicate_doc_ids


class RetrievalTarget(str, enum.Enum):
  """Target objects to return during retrieval.

  REWRITTEN: Returns the rewritten documents (e.g. chunks) as stored in the
    inner index.
  ORIGINAL: Returns the original documents (e.g. parent documents) associated
    with the retrieved rewritten documents.
  """

  REWRITTEN = 'rewritten'
  ORIGINAL = 'original'


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

  def get_doc(self, doc_id: str) -> DocT:
    """Returns the document with the given ID."""
    for doc in self.get_docs():
      if getattr(doc, 'doc_id', None) == doc_id:
        return doc
    raise KeyError(doc_id)

  def get_docs_by_ids(self, doc_ids: Iterable[str]) -> list[DocT]:
    """Returns the documents with the given IDs, in the order of the IDs."""
    return [self.get_doc(doc_id) for doc_id in doc_ids]

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
    await self.add_docs(docs)

  @abc.abstractmethod
  def destroy_index(self) -> None:
    """Destroys the index and frees up related resources."""

  def save(
      self,
      base_path: str,
      storage_format: serialization.StorageFormat | None = None,
  ) -> None:
    """Saves the index state to the directory `base_path`.

    Not supported by default. Subclasses that support saving override both
    `save` and `load` (see `supports_save`).

    Args:
      base_path: The directory to save the index state to.
      storage_format: How to save the index state, or None for the index's
        default format. `load` reads it from the saved files.

    Raises:
      NotImplementedError: If the subclass does not override `save`.
    """
    raise NotImplementedError(
        f'{type(self).__name__} does not support saving; override save() and'
        ' load().'
    )

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state from the directory `base_path`.

    Not supported by default. Subclasses that support saving override both
    `save` and `load` (see `supports_save`).

    Args:
      base_path: The directory that the index state was saved to by `save`.

    Raises:
      NotImplementedError: If the subclass does not override `load`.
    """
    raise NotImplementedError(
        f'{type(self).__name__} does not support loading; override save() and'
        ' load().'
    )


def supports_save(index: Index) -> bool:
  """Returns whether `index` can be saved with `save` and loaded with `load`.

  An index supports saving if its class overrides both `Index.save` and
  `Index.load`. A `RewritingDocumentIndex` also requires that its inner index
  supports saving.

  Args:
    index: The index to check.
  """
  index_type = type(index)
  if index_type.save is Index.save or index_type.load is Index.load:
    return False
  if isinstance(index, RewritingDocumentIndex):
    return supports_save(index.inner_index)
  return True


@dataclasses.dataclass(kw_only=True)
class IndexWrapper(
    Index[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
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

  def get_doc(self, doc_id: str) -> DocT:
    """Overridden from base class (Index)."""
    return self.inner_index.get_doc(doc_id)

  def get_docs_by_ids(self, doc_ids: Iterable[str]) -> list[DocT]:
    """Overridden from base class (Index)."""
    return self.inner_index.get_docs_by_ids(doc_ids)

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    await self.inner_index.add_docs(docs=docs)

  @executing.make_executable(copy_self=False)
  async def create_index(self, corpus_name: str, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    await self.inner_index.create_index(corpus_name=corpus_name, docs=docs)

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
    return await self.inner_index.retrieve(
        query, max_results=max_results, **kwargs
    )

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Overridden from base class (Retriever)."""
    return await self.inner_index.retrieve_with_scores(
        query, max_results=max_results, **kwargs
    )


@dataclasses.dataclass(kw_only=True)
class RewritingIndex(
    IndexWrapper[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """Index that rewrites a corpus before adding it to an inner index.

  By default, this class treats the rewritten documents as the primary units of
  retrieval. When `get_docs`, `retrieve` or `retrieve_with_scores` is called,
  the objects returned are the individual rewritten documents (derived from the
  original documents) as stored in the inner index, rather than the original
  source documents.

  When `retrieval_target` is set to `RetrievalTarget.ORIGINAL` (or passed to
  `retrieve`/`retrieve_with_scores`), the index will instead resolve and
  return the original documents associated with the retrieved rewritten
  documents.

  `get_doc` and `get_docs_by_ids` look up IDs in the same ID space as the
  documents returned by `retrieve`: rewritten document IDs (e.g. chunk IDs)
  for `RetrievalTarget.REWRITTEN`, and original document IDs for
  `RetrievalTarget.ORIGINAL`. Thus `index.get_doc(doc.doc_id)` succeeds for
  any `doc` returned by `index.retrieve(...)`. Both also accept a
  `retrieval_target` argument to look up IDs in the other ID space.

  One specific usecase is to use this class to chunk documents before adding
  them to an inner index. In this case, the `rewriter` would be a
  `ChunkingRewriter`:

  Example:

    chunking_index = RewritingIndex(
      rewriter = corpus_rewriting.ChunkingRewriter(
        chunker = chunking.ChunkByMaxTokens(
          max_tokens_per_chunk = 128,
          overlap_tokens = 16,
        )
      )
    )

  Attributes:
    rewriter: CorpusRewriter used to rewrite the documents.
    retrieval_target: Default target objects to return upon retrieval (rewritten
      or original).
    preserve_original_docs: Whether to preserve original documents in memory
      when `add_docs` is called. Note that setting this to True will save the
      documents during index serialization.
  """

  rewriter: corpus_rewriting.CorpusRewriter = dataclasses.field(
      default_factory=corpus_rewriting.NoRewriting
  )
  retrieval_target: RetrievalTarget = RetrievalTarget.REWRITTEN
  preserve_original_docs: bool = False
  _original_docs: dict[str, DocT] = dataclasses.field(
      default_factory=dict, init=False, repr=False
  )

  def get_original_doc(self, chunk: DocT) -> DocT | None:
    """Resolves the original document for a given chunk or document."""
    # 1. Try metadata (standard chunking case)
    if metadata := getattr(chunk, 'metadata', None):
      if isinstance(metadata, dict) and (
          original_id := metadata.get(
              retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
          )
      ):
        return self._original_docs.get(original_id)

    # 2. Try direct doc_id (fallback for NoRewriting/1-to-1 rewriting)
    if chunk_id := getattr(chunk, 'doc_id', None):
      return self._original_docs.get(chunk_id)
    return None

  def get_doc(
      self,
      doc_id: str,
      *,
      retrieval_target: RetrievalTarget | None = None,
  ) -> DocT:
    """Overridden from base class (Index).

    Args:
      doc_id: The ID of the document to return, in the ID space of the
        retrieval target (see class docstring).
      retrieval_target: The ID space to look up `doc_id` in. Defaults to
        `self.retrieval_target`.

    Raises:
      KeyError: If no document with the given ID is in the index.
      ValueError: If the retrieval target is `RetrievalTarget.ORIGINAL` but
        `preserve_original_docs` is False, as original documents are then not
        available.

    Returns:
      The document with the given ID.
    """
    target = retrieval_target or self.retrieval_target
    if target == RetrievalTarget.REWRITTEN:
      return self.inner_index.get_doc(doc_id)
    if not self.preserve_original_docs:
      raise ValueError(
          'Looking up original documents by ID requires'
          ' `preserve_original_docs=True` during index building.'
      )
    return self._original_docs[doc_id]

  def get_docs_by_ids(
      self,
      doc_ids: Iterable[str],
      *,
      retrieval_target: RetrievalTarget | None = None,
  ) -> list[DocT]:
    """Overridden from base class (Index).

    Args:
      doc_ids: The IDs of the documents to return, in the ID space of the
        retrieval target (see class docstring).
      retrieval_target: The ID space to look up `doc_ids` in. Defaults to
        `self.retrieval_target`.

    Raises:
      KeyError: If any of the IDs is not in the index.
      ValueError: See `get_doc`.

    Returns:
      The documents with the given IDs.
    """
    target = retrieval_target or self.retrieval_target
    if target == RetrievalTarget.REWRITTEN:
      return self.inner_index.get_docs_by_ids(doc_ids)
    return [
        self.get_doc(doc_id, retrieval_target=target) for doc_id in doc_ids
    ]

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

    self.inner_index.corpus_name = corpus_name
    await self.add_docs(docs)

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    doc_list = list(docs)
    if not doc_list:
      return
    _validate_docs(doc_list)
    if self.preserve_original_docs:
      for doc in doc_list:
        self._original_docs[getattr(doc, 'doc_id')] = doc
    rewritten_docs = await self.rewriter(doc_list)
    await self.inner_index.add_docs(docs=rewritten_docs)

  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self._original_docs.clear()
    self.inner_index.destroy_index()

  # TODO: Original documents are always saved as JSONL,
  # regardless of the storage format. Fix in follow-up CL.
  def _save_original_docs(self, base_path: str) -> None:
    """Saves the original documents to `ORIGINAL_DOCS_JSONL_NAME`.

    The file is written to `base_path`, next to the inner index's files, with
    one JSON-serialized document per line (in the order they were added).

    Args:
      base_path: The directory the index is saved to.
    """
    path = epath.Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    serialization.save_documents(
        cast(
            Sequence[retrieval_data_structures.Document],
            list(self._original_docs.values()),
        ),
        path / ORIGINAL_DOCS_JSONL_NAME,
    )

  def _load_original_docs(
      self,
      base_path: str,
      doc_class: type[retrieval_data_structures.Document],
  ) -> None:
    """Loads the original documents saved by `_save_original_docs`.

    Does nothing if there is no `ORIGINAL_DOCS_JSONL_NAME` file in `base_path`
    (e.g. for indices saved without original documents). Otherwise, also sets
    `preserve_original_docs`, so that the loaded original documents can be
    looked up, and are kept up to date and saved again by later calls to
    `add_docs` and `save`.

    Args:
      base_path: The directory the index was saved to.
      doc_class: The class to deserialize the documents as.
    """
    path = epath.Path(base_path) / ORIGINAL_DOCS_JSONL_NAME
    if not path.exists():
      return
    docs = serialization.load_documents(path, doc_class, show_progress=False)
    self._original_docs = {doc.doc_id: cast(DocT, doc) for doc in docs}
    if not self.preserve_original_docs:
      logging.info('Found %s; setting preserve_original_docs=True.', path)
      self.preserve_original_docs = True

  @executing.make_executable(copy_self=False)
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[RetrievalResultT]:
    """Overridden from base class (Retriever)."""
    docs_and_scores = await self.retrieve_with_scores(
        query, max_results=max_results, **kwargs
    )
    return [doc for doc, _ in docs_and_scores]

  @executing.make_executable(copy_self=False)
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[tuple[RetrievalResultT, float]]:
    """Overridden from base class (Retriever)."""
    if self.retrieval_target == RetrievalTarget.REWRITTEN:
      return await self.inner_index.retrieve_with_scores(
          query, max_results=max_results, **kwargs
      )

    rewritten_docs_and_scores = list(
        await self.inner_index.retrieve_with_scores(
            query, max_results=max_results, **kwargs
        )
    )

    original_docs_and_scores: list[tuple[RetrievalResultT, float]] = []
    seen_original_ids = set()
    for doc, score in rewritten_docs_and_scores:
      original = self.get_original_doc(cast(DocT, doc))
      if original is not None:
        original_id = getattr(original, 'doc_id')
        if original_id not in seen_original_ids:
          original_docs_and_scores.append(
              (cast(RetrievalResultT, original), score)
          )
          seen_original_ids.add(original_id)
          if (
              max_results is not None
              and len(original_docs_and_scores) >= max_results
          ):
            break
    return original_docs_and_scores


@dataclasses.dataclass
class DocumentIndex(
    Index[str, Document, Document],
    metaclass=abc.ABCMeta,
):
  """All DocumentIndex classes should be subclasses of this type."""


@registry.register('RewritingDocumentIndex')
@dataclasses.dataclass(kw_only=True)
class RewritingDocumentIndex(
    RewritingIndex[str, Document, Document],
    DocumentIndex,
):
  """`RewritingIndex` for `Document` objects, wrapping any such index.

  Supports `save` and `load` if the inner index does: the inner index is saved
  to the same directory, next to the original documents (if preserved).

  Attributes:
    inner_index: The index that the rewritten documents are added to.
  """

  inner_index: Index[str, Document, Document]

  def save(
      self,
      base_path: str,
      storage_format: serialization.StorageFormat | None = None,
  ) -> None:
    """Saves the index state."""
    self.inner_index.save(base_path, storage_format)
    if self.preserve_original_docs:
      self._save_original_docs(base_path)

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state."""
    await self.inner_index.load(base_path)
    self._load_original_docs(base_path, Document)


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
      **kwargs,
  ) -> Iterable[Document]:
    """Overridden from base class (Retriever)."""
    del query, max_results, kwargs
    return []


@registry.register('NaiveIndex')
@dataclasses.dataclass
class NaiveIndex(DocumentIndex):
  """Index that stores the original document text as-is and retrieves naively.

  Simply returns the first K added documents each time, regardless of the query.

  Attributes:
    shuffle_results: Whether to shuffle the resulting documents before returning
      them. Useful for synthetic datasets where distractor documents (i.e.,
      documents that are irrelevant to the query, used to challenge the
      retriever) are added in the same order as golden documents.
  """

  shuffle_results: bool = False
  # Documents added with `add_docs`.
  _docs: list[Document] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )

  @property
  def docs(self) -> list[Document]:
    """Returns the documents that were added to the index."""
    return self._docs

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
    self._docs.extend(docs)

  @tracing.trace('NaiveIndex.destroy_index')
  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self._docs.clear()

  @executing.make_executable(copy_self=False)
  @tracing.trace('NaiveIndex.retrieve')
  async def retrieve(
      self,
      query: str,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[Document]:
    """Overridden from base class (Retriever)."""
    del kwargs
    if max_results is None:
      max_results = len(self.docs)
    if not self.shuffle_results:
      return self.docs[:max_results]
    random.seed(query)
    result = self.docs[:max_results].copy()
    random.shuffle(result)
    return result


# fmt: off
@dataclasses.dataclass(kw_only=True)
class AbstractEmbeddingBasedIndex(
    Index[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  r"""Abstract base class for embedding-based retrieval indices.

  This base class implements various functionality around processing of
  queries and documents prior to embedding that is relevant to be in-memory and
  distributed embedding-based indices. However, note that the child class is
  still responsible for providing concrete implementations for `add_docs`,
  `retrieve` and `retrieve_with_scores` along with the embedding functionality.

  Attributes:
    query_task_type: Task type to use when embedding queries via `llm.embed`
      (e.g. 'SEMANTIC_SIMILARITY', 'RETRIEVAL_QUERY'). The backend handles
      translation for both v1 and v2 embedding models.
    document_task_type: Task type to use when embedding documents via
      `llm.embed` (e.g. 'SEMANTIC_SIMILARITY', 'RETRIEVAL_DOCUMENT'). The
      backend handles translation for both v1 and v2 embedding models.
    embed_backend_config: Configuration for the embedder backend, which is used
      for calculating embeddings. If None, the pre-existing `llm.embed` backend
      is used.
    runtime_kwargs: Environment-specific deployment settings (e.g.,
      `threadpool_size`, `project`) and credentials (`api_key`, `credentials`)
      that are not stored in the serializable configuration. Used in conjunction
      with `embed_backend_config` to lazily instantiate the embedder backend
      upon initialization or when the index is loaded. Since it may hold
      credentials, it is excluded from `repr` and equality, and never saved by
      the serializers.
  """
  query_task_type: str | None = 'RETRIEVAL_QUERY'
  document_task_type: str | None = 'RETRIEVAL_DOCUMENT'

  # Configuration for the embedder backend.
  embed_backend_config: backend_registry.EmbedBackendConfig | None = None
  # Environment settings and credentials: excluded from `repr` and equality,
  # and never saved by the serializers.
  runtime_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  # Localized embedder backend to use for calculating embeddings. This is
  # lazily instantiated from `embed_backend_config` (which is saved with the
  # index and restored when it is loaded).
  _embedder_backend: backends_base.Backend | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  @property
  def embedder_backend(self):
    """Lazily instantiates the backend from `embed_backend_config` if unset."""
    if (self._embedder_backend is None and
        self.embed_backend_config is not None):
      self._embedder_backend = backend_registry.get_backend(
          self.embed_backend_config,
          **self.runtime_kwargs,
      )
    return self._embedder_backend

  @embedder_backend.setter
  def embedder_backend(self, value):
    """Manually overrides the embedder backend (for custom injection only)."""
    self._embedder_backend = value

  @contextlib.contextmanager
  def _backend_routing_context(self) -> Iterator[None]:
    """Yields a context with the embedder backend registered if one is set."""
    if self.embedder_backend is not None:
      with routing.RegistryContext():
        self.embedder_backend.register()
        yield
    else:
      yield


def _default_list_map():
  """Returns a defaultdict(list)."""
  return collections.defaultdict(list)


def _default_discrete_value_map():
  """Returns a nested defaultdict for discrete value indexing."""
  return collections.defaultdict(_default_list_map)


# TODO: Combine EmbeddingBasedIndex and
# EmbeddingBasedDocumentIndex as EmbeddingBasedIndex requires `QueryT` to be
# Either ChunkList or str and DocT to have `.title` and `.content`, essentially
# a `Document`.
@dataclasses.dataclass(kw_only=True)
class EmbeddingBasedIndex(
    AbstractEmbeddingBasedIndex[QueryT, DocT, DocT],
    Generic[QueryT, DocT],
):
  """In-memory index that retrieves documents based on embedding similarity.

  This class implements an index that stores document embeddings in memory and
  performs retrieval by computing the similarity between query and document
  embeddings. It uses the `llm.embed` action to generate embeddings.

  Attributes:
    searcher: Searcher used for nearest neighbor search between query and
      document embeddings. It holds the built search state of the index
      (`init=False` fields), so each index needs its own searcher instance: do
      not share one between indexes.
  """

  searcher: searchers.Searcher = dataclasses.field(
      default_factory=searchers.BruteForceSearcher
  )
  # List of the documents that are added to the index.
  _docs: list[DocT] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )
  # List of the embeddings of the documents that are added to the index.
  _doc_embeddings: list[np.ndarray] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )
  # Mapping from doc ID to the row of that document in `_docs` (and, equally,
  # in `_doc_embeddings` and the searcher). `add_docs` requires doc_ids to be
  # unique, but indices saved before that was enforced may contain duplicate
  # doc_ids; for these, this maps to the first document with each doc_id.
  _row_by_id: dict[str, int] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )

  @executing.make_executable(copy_self=False)
  async def _calculate_query_embedding(self, query: QueryT) -> Sequence[float]:
    """Returns a newly-calculated embedding of the query."""
    with self._backend_routing_context():
      query_embedding = await llm.embed(query, task_type=self.query_task_type)

    return query_embedding

  @executing.make_executable(copy_self=False)
  async def _calculate_doc_embeddings(
      self, docs: Sequence[DocT]
  ) -> Sequence[Sequence[float]]:
    """Returns newly-calculated embeddings of the documents."""
    threadpool_size = self.runtime_kwargs.get(
        'threadpool_size', DEFAULT_INDEXING_THREADPOOL_SIZE
    )
    with self._backend_routing_context():
      embed_executables = (
          llm.embed(
              doc.content,  # pyrefly: ignore[missing-attribute]
              task_type=self.document_task_type,
              title=doc.title,  # pyrefly: ignore[missing-attribute]
          )
          for doc in docs
      )
      return await executing.par_iter(
          embed_executables,
          chunk_size=threadpool_size,
          show_progress=True,
          progress_desc='Embedding docs',
          total=len(docs),
      )

  @property
  def num_docs(self) -> int:
    """Overridden from base class (Index)."""
    return len(self._docs)

  def get_docs(self) -> Iterable[DocT]:
    """Overridden from base class (Index)."""
    return self._docs

  def _update_row_by_id(self) -> None:
    """Adds the rows of `_docs` that are not yet in `_row_by_id`."""
    # If a loaded index has duplicate doc_ids, `_row_by_id` has fewer entries
    # than `_docs`, so some rows get re-visited (harmlessly, due to
    # `setdefault`) on every call.
    for row in range(len(self._row_by_id), len(self._docs)):
      self._row_by_id.setdefault(getattr(self._docs[row], 'doc_id'), row)

  def get_doc(self, doc_id: str) -> DocT:
    """Overridden from base class (Index)."""
    return self._docs[self._row_by_id[doc_id]]

  def get_doc_embedding(self, doc_id: str) -> np.ndarray:
    """Returns the embedding of the document with the given ID."""
    return self._doc_embeddings[self._row_by_id[doc_id]]

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.add_docs')
  async def add_docs(self, docs: Iterable[DocT]) -> None:
    """Overridden from base class (Index)."""
    docs = list(docs)
    if not docs:
      return
    _validate_docs(docs)
    if duplicate_doc_ids := _find_duplicate_doc_ids(docs, self._row_by_id):
      raise ValueError(
          f'{len(duplicate_doc_ids)} documents have a doc_id that is already'
          f' used by another document (e.g. {duplicate_doc_ids[:10]}). doc_ids'
          ' must be unique within an index.'
      )
    self._docs.extend(docs)
    self._update_row_by_id()
    embedded_results = await self._calculate_doc_embeddings(docs)
    self._doc_embeddings.extend([np.array(embed) for embed in embedded_results])
    # Build the searcher index.
    await self.searcher.build(np.stack(self._doc_embeddings, axis=0))

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.retrieve_doc_indices_and_scores')
  async def retrieve_doc_indices_and_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[tuple[int, float]]:
    """Returns indices and scores of the most relevant documents.

    The indices returned are compatible with the list of documents returned
    by `get_docs()`.

    Args:
      query: The query used for retrieval.
      max_results: The maximum number of results to return.
      min_score: The minimum similarity score for a result to be included.
      **kwargs: Additional keyword arguments.
    """
    del kwargs  # Unused.
    if not self.num_docs:
      return []
    query_embedding = await self._calculate_query_embedding(query)

    if max_results is None:
      max_results = self.num_docs

    # For compatibility with search_batched, which expects a batch of queries.
    query_embeddings = np.array([query_embedding])
    ranked_doc_indices, ranked_doc_scores = await self.searcher.search_batched(
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
  async def retrieve_doc_score(self, query: QueryT, doc_index: int) -> float:
    """Returns similarity score of the document at `doc_index` for a query.

    The score is computed the same way as in `retrieve_doc_indices_and_scores`:
    a dot product, with both embeddings L2-normalized first if the searcher
    has `normalize_embeds` set.

    Args:
      query: The query used for scoring.
      doc_index: Index of the document, compatible with `get_docs()`.
    """
    if doc_index < 0 or doc_index >= len(self._doc_embeddings):
      return 0.0
    query_embedding = np.asarray(await self._calculate_query_embedding(query))
    doc_embedding = self._doc_embeddings[doc_index]
    if getattr(self.searcher, 'normalize_embeds', False):
      # Same epsilon as `BruteForceSearcher`.
      query_embedding = query_embedding / (
          np.linalg.norm(query_embedding) + 1e-10
      )
      doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-10)
    return float(np.dot(query_embedding, doc_embedding))

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.retrieve_with_scores')
  async def retrieve_with_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Iterable[tuple[DocT, float]]:
    """Overridden from base class (Retriever)."""
    doc_indices_and_scores = await self.retrieve_doc_indices_and_scores(
        query,
        max_results=max_results,
        min_score=min_score,
    )
    ranked_documents_and_scores = [
        (self._docs[i], score) for i, score in doc_indices_and_scores
    ]
    return ranked_documents_and_scores

  @executing.make_executable(copy_self=False)
  @tracing.trace('EmbeddingBasedIndex.retrieve', skip=['output'])
  async def retrieve(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Iterable[DocT]:
    """Overridden from base class (Retriever)."""
    doc_indices_and_scores = await self.retrieve_doc_indices_and_scores(
        query,
        max_results=max_results,
        min_score=min_score,
    )
    ranked_documents = [self._docs[i] for i, _ in doc_indices_and_scores]
    return ranked_documents

  @tracing.trace('EmbeddingBasedIndex.destroy_index')
  def destroy_index(self) -> None:
    """Overridden from base class (Index)."""
    self._docs = []
    self._doc_embeddings = []
    self._row_by_id = {}
    self.searcher.destroy()


@dataclasses.dataclass(kw_only=True)
class RewritingEmbeddingBasedIndex(
    RewritingIndex[QueryT, RetrievalResultT, DocT],
    Generic[QueryT, RetrievalResultT, DocT],
):
  """Index that rewrites documents and provides embedding-specific methods.

  This class combines the behavior of `RewritingIndex` with the vector-search
  capabilities of `EmbeddingBasedIndex`. Note that the document access and
  retrieval methods (`get_docs`, `retrieve`, `retrieve_with_scores`, and
  `retrieve_doc_indices_and_scores`) currently return the specific rewritten
  documents and their corresponding level embeddings/scores.

  Attributes:
    inner_index: The underlying `EmbeddingBasedIndex` used for vector storage
      and similarity search.
  """

  inner_index: EmbeddingBasedIndex[QueryT, RetrievalResultT]

  def get_doc_embedding(self, doc_id: str) -> np.ndarray:
    """Delegates to the inner index."""
    return self.inner_index.get_doc_embedding(doc_id)

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_score(self, query: QueryT, doc_index: int) -> float:
    """Delegates to the inner index."""
    return await self.inner_index.retrieve_doc_score(
        query, doc_index
    )

  @executing.make_executable(copy_self=False)
  async def retrieve_doc_indices_and_scores(
      self,
      query: QueryT,
      *,
      max_results: int | None = None,
      min_score: float | None = None,
      **kwargs,
  ) -> Sequence[tuple[int, float]]:
    """Delegates to the inner index."""
    return await self.inner_index.retrieve_doc_indices_and_scores(
        query,
        max_results=max_results,
        min_score=min_score,
        **kwargs,
    )


@registry.register('EmbeddingBasedDocumentIndex')
@dataclasses.dataclass(kw_only=True)
class EmbeddingBasedDocumentIndex(
    EmbeddingBasedIndex[str, Document],
    DocumentIndex,
):
  """Embedding-based index for Document objects."""

  # How the index is saved by default, and how `load` reads a directory without
  # a `manifest.json` (which records the format otherwise). A class constant,
  # not part of the index config.
  default_storage_format: ClassVar[serialization.StorageFormat] = (
      serialization.StorageFormat.JSONL_NPY
  )

  def save(
      self,
      base_path: str,
      storage_format: serialization.StorageFormat | None = None,
  ) -> None:
    """Saves the index state."""
    index_state_dto = index_state.EmbeddingBasedIndexState(
        docs=self._docs,
        doc_embeddings=self._doc_embeddings,
        embed_backend_config=self.embed_backend_config,
    )
    serializer = serialization.get_serializer(
        storage_format or self.default_storage_format, Document
    )
    serializer.save(index_state_dto, base_path)

  # TODO: Consider passing runtime_kwargs to the load method.
  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    """Loads the index state, in the storage format it was saved in."""
    storage_format = serialization.get_storage_format(
        base_path, default=self.default_storage_format
    )
    serializer = serialization.get_serializer(storage_format, Document)
    index_state_dto = await serializer.load(base_path)
    if duplicate_doc_ids := _find_duplicate_doc_ids(index_state_dto.docs, {}):
      # Only `add_docs` enforces unique doc_ids, so that indices saved before
      # it did can still be loaded.
      logging.warning(
          'Loaded %d documents whose doc_id is already used by another document'
          ' in the index (e.g. %s). `get_doc` returns the first document for'
          ' each doc_id.',
          len(duplicate_doc_ids),
          duplicate_doc_ids[:10],
      )
    self._docs = index_state_dto.docs
    self._row_by_id = {}
    self._update_row_by_id()
    self._doc_embeddings = index_state_dto.doc_embeddings
    if index_state_dto.embed_backend_config is not None:
      self.embed_backend_config = index_state_dto.embed_backend_config
      self._embedder_backend = None
    # Build the searcher index.
    await self.searcher.build(np.stack(self._doc_embeddings, axis=0))


@registry.register('RewritingEmbeddingBasedDocumentIndex')
@dataclasses.dataclass(kw_only=True)
class RewritingEmbeddingBasedDocumentIndex(
    RewritingEmbeddingBasedIndex[str, Document, Document],
    RewritingDocumentIndex,
):
  """Embedding-based index with rewriting for Document objects.

  The embedding settings (e.g. `embed_backend_config`) are those of
  `inner_index`.
  """

  inner_index: EmbeddingBasedDocumentIndex = dataclasses.field(
      default_factory=EmbeddingBasedDocumentIndex
  )
