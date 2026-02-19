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

"""Utilities for chunking text into smaller pieces."""

import abc
from collections.abc import Iterable
import copy
import dataclasses
from typing import Protocol
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import retrieval_data_structures


class Chunker(Protocol):
  """Generic interface for a strategy that converts documents to chunks.

  Depending on the strategy, the chunks may either be discrete or overlapping,
  and may either be simple substrings of the original document or content that
  is derived in a non-trivial way from the original document.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> Iterable[retrieval_data_structures.Document]:
    """Returns one or more chunks suitable for indexing, based on the document.

    Args:
      document: The document to chunk.
    """


# TODO: Implement proper support for multimodal content.
@dataclasses.dataclass(kw_only=True)
class TextChunker(Chunker):
  """Chunks the document based on its text content only.

  Preprocesses the document text prior to chunking, and postprocesses the
  chunked text afterwards, based on configurable format strings.
  All other document attributes are passed through to the output chunks.
  The method `_format_text` can be overridden to provide additional custom
  logic for transforming the text of the document (and only the text). Any other
  attributes of the document (e.g., title, metadata) will be passed through to
  the output chunks as-is, and can be used to modify the text of the document
  before chunking or after chunking.

  Attributes:
    document_format: A string template that can be used to format the text of
      the document before chunking. The template may refer to the following
      fields: `text`, `title`, `doc_id`, and any other fields in the `metadata`
      dictionary of the document. Example: '{text} (Title: {title})'.
    chunk_format: A string template that can be used to format the text of the
      chunk after chunking. The template may refer to the following fields:
      `text`, `title`, `doc_id`, and any other fields in the `metadata` field of
      the document. Example: '{text} (Title: {title})'.
  """

  document_format: str = '{text}'
  chunk_format: str = '{text}'

  def _format_text(
      self, document: retrieval_data_structures.Document, format_str: str
  ) -> str:
    """Formats the chunk text."""
    return format_str.format(
        text=document.content,
        title=document.title,
        doc_id=document.doc_id,
        **document.metadata,
    )

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def _chunk_text(self, text: str) -> Iterable[str]:
    """Returns one or more chunks suitable for indexing, based on the text.

    Args:
      text: The text to chunk.
    """  #  pytype: disable=bad-return-type

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> Iterable[retrieval_data_structures.Document]:
    """Returns one or more chunks suitable for indexing, based on the document.

    Args:
      document: The document to chunk.
    """
    document_text = self._format_text(document, self.document_format)
    chunks = []
    for chunk_text in await self._chunk_text(document_text):  # pytype: disable=wrong-arg-count
      chunk = copy.deepcopy(document)
      chunk.content = chunk_text
      chunk.content = self._format_text(chunk, self.chunk_format)
      chunks.append(chunk)
    return chunks


@dataclasses.dataclass(kw_only=True)
class NoChunking(TextChunker):
  """Trivial chunker that returns the original document as-is."""

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def _chunk_text(self, text: str) -> Iterable[str]:
    """Returns `document` as-is."""
    return [text]
