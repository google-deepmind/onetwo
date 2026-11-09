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

"""Utilities for formatting and structuring documents for indexing.

In a RAG pipeline, the quality of an LLM's answer depends heavily on how
information is presented in its prompt. A **Document Formatter** ensures
that raw document content is augmented with critical context (like titles,
authors, or source URLs) before it is indexed.

A document formatter always maintains a 1 to 1 relationship: one input document
results in exactly one output document.
"""

import copy
import dataclasses
from typing import Protocol
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures


class DocumentFormatter(Protocol):
  """Generic interface for a strategy that formats a document.

  A document formatter maps exactly one input document to exactly one output
  document (1:1). This is intended for transformations that do not change the
  number of documents, such as reformatting text, adding titles, or applying
  templates.
  """

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> retrieval_data_structures.Document:
    """Returns a formatted version of the document.

    Args:
      document: The document to format.

    Returns:
      A formatted document.
    """


@registry.register('NoFormatting')
@dataclasses.dataclass(frozen=True)
class NoFormatting(DocumentFormatter):
  """Formatter that returns the document as-is."""

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> retrieval_data_structures.Document:
    """Returns `document` as-is."""
    return document


@registry.register('TextDocumentFormatter')
@dataclasses.dataclass(kw_only=True)
class TextDocumentFormatter(DocumentFormatter):
  """Formats the document text based on a configurable format string.

  The method `_format_text` can be overridden to provide additional custom
  logic for transforming the text of the document (and only the text). Any other
  attributes of the document (e.g., title, metadata) will be passed through to
  the output document as-is.

  Attributes:
    format_str: A string template that can be used to format the text of the
      document. The template may refer to the following fields: `text`, `title`,
      `doc_id`, and any other fields in the `metadata` dictionary of the
      document. Example: '{text} (Title: {title})'.
  """

  format_str: str = '{text}'

  def _format_text(
      self, document: retrieval_data_structures.Document, format_str: str
  ) -> str:
    """Formats the document text."""
    return format_str.format(
        text=document.content,
        title=document.title,
        doc_id=document.doc_id,
        **document.metadata,
    )

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> retrieval_data_structures.Document:
    """Formats the document text using the format string.

    Args:
      document: The document to format.

    Returns:
      A new document with the formatted text.
    """
    formatted_document = copy.deepcopy(document)
    formatted_document.content = self._format_text(document, self.format_str)
    return formatted_document


@registry.register('KeepTextFormatter')
@dataclasses.dataclass(frozen=True)
class KeepTextFormatter(DocumentFormatter):
  """Keeps only text chunks, dropping all non-text content of each document.

  If the document contains no text content, content is set to an empty
  ChunkList.
  """

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> retrieval_data_structures.Document:
    """Returns `document` with only text content kept."""
    if isinstance(document.content, str):
      return document
    content = content_lib.ChunkList([
        chunk
        for chunk in document.content
        if content_lib.is_str_content_type(chunk.content_type)
    ])
    return dataclasses.replace(document, content=content)


@registry.register('KeepImageFormatter')
@dataclasses.dataclass(frozen=True, kw_only=True)
class KeepImageFormatter(DocumentFormatter):
  """Keeps only image chunks, dropping all other content of each document.

  If the document contains no image chunks, content is set to an empty
  ChunkList.

  Attributes:
    keep_title: Whether to preserve the document title. Defaults to False,
      clearing the title to '' so that no textual fields remain.
  """

  keep_title: bool = False

  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> retrieval_data_structures.Document:
    """Returns `document` with only image chunks kept."""
    if isinstance(document.content, str):
      image_chunks = []
    else:
      image_chunks = [
          chunk
          for chunk in document.content
          if chunk.content_type.startswith('image/')
      ]
    return dataclasses.replace(
        document,
        title=document.title if self.keep_title else '',
        content=content_lib.ChunkList(image_chunks),
    )
