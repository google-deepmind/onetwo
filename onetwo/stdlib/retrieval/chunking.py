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
from typing import Any, Protocol, Sequence
from onetwo import ot
from onetwo.builtins import llm
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


@executing.make_executable  # pytype: disable=wrong-arg-types
@tracing.trace(skip=['tokenizer_backend'])
async def truncate_string_to_max_tokens(
    text: str,
    max_tokens: int,
    tokenizer_backend: Any | None = None,
) -> str:
  """Returns the longest prefix of `text` that is within `max_tokens`.

  Args:
    text: The string to truncate.
    max_tokens: The maximum number of tokens to return.
    tokenizer_backend: The LLM backend to use for tokenization. If not
      specified, then will default to the currently-registered backend.
  """

  with ot.RegistryContext():
    if tokenizer_backend is not None:
      tokenizer_backend.register()
    return await llm.detokenize((await llm.tokenize(text))[:max_tokens])  # pytype: disable=wrong-arg-count


@executing.make_executable  # pytype: disable=wrong-arg-types
@tracing.trace(skip=['tokenizer_backend'])
async def filter_by_max_tokens(
    texts: Iterable[str],
    max_tokens: int,
    *,
    include_partial: bool = True,
    tokenizer_backend: Any | None = None,
) -> Sequence[str]:
  """Returns the input stream truncated once max_tokens is reached.

  Args:
    texts: The input stream to filter.
    max_tokens: The maximum total number of tokens to return.
    include_partial: If True, then may include a partial text at the end, that
      has been truncated so that the total number of tokens returns is exactly
      `max_tokens`. If False, then only full texts will be returned.
    tokenizer_backend: The LLM backend to use for tokenization. If not
      specified, then will default to the currently-registered backend.
  """
  with ot.RegistryContext():
    if tokenizer_backend is not None:
      tokenizer_backend.register()
    total_tokens = 0
    filtered_texts = []
    for text in texts:
      text_tokens = await llm.count_tokens(text)  # pytype: disable=wrong-arg-count
      total_tokens += text_tokens
      if total_tokens < max_tokens:
        # Room to spare.
        filtered_texts.append(text)
        continue
      elif total_tokens == max_tokens:
        # Exactly hit limit.
        filtered_texts.append(text)
        break
      else:
        # Exceeded limit.
        if include_partial:
          # At this point total_tokens > max_tokens.
          # So max_tokens - total_tokens < 0.
          # This value is how much we need to "save"
          # So text_tokens reduced by what we need to save is
          # the max length we can still afford, so we truncate to that.
          max_text_tokens = text_tokens + (max_tokens - total_tokens)
          truncated = await truncate_string_to_max_tokens(text, max_text_tokens)
          filtered_texts.append(truncated)
        break
    return filtered_texts


@dataclasses.dataclass(kw_only=True)
class ChunkByMaxTokens(TextChunker):
  """Chunks text by max tokens.

  The algorithm flows as follows:
  1. Tokenize the text.
  2. Split into chunks of size `max_tokens_per_chunk`.
  3. Detokenize the chunks.

  The differences in tokenization are due to non-commutativity of tokenization
  with concatenation, e.g. Tokenize("un" + "happy") != Tokenize("un") +
  Tokenize("happy").

  Attributes:
    max_tokens_per_chunk: The maximum number of tokens per chunk. Note that this
      is only partially enforced since chunk_format is applied after chunking.
    tokenizer_backend: The LLM backend to use for tokenization. If not
      specified, then will default to the currently-registered backend.
    strip_whitespace: Whether to strip whitespace from the ends of each chunk.
    overlap_window: The number of tokens to overlap between chunks. If this is
      set to 0, then the chunks will not overlap.
  """

  # 150 tokens is rather small, but corresponds roughly to the 100-word chunks
  # used in early research on retrieval-augmented generation, such as
  # https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
  max_tokens_per_chunk: int = 150
  tokenizer_backend: Any | None = None
  strip_whitespace: bool = True
  overlap_window: int = 0

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def _chunk_text(self, text: str) -> Iterable[str]:
    """Returns `text` broken into chunks of up to `max_tokens_per_chunk`.

    Args:
      text: The text to chunk.
    """
    if self.max_tokens_per_chunk <= 0:
      raise ValueError(
          'max_tokens_per_chunk must be positive, got'
          f' {self.max_tokens_per_chunk}'
      )
    if self.overlap_window < 0:
      raise ValueError(
          f'overlap_window must be non-negative, got {self.overlap_window}'
      )
    if self.overlap_window > self.max_tokens_per_chunk:
      raise ValueError(
          'overlap_window must be smaller than max_tokens_per_chunk, got'
          f' {self.overlap_window} > {self.max_tokens_per_chunk}'
      )
    tokenized_chunks = []
    remaining_tokens = await llm.tokenize(text)  # pytype: disable=wrong-arg-count
    while remaining_tokens:
      if self.overlap_window > 0:
        truncated_tokens = remaining_tokens[: self.max_tokens_per_chunk]
        truncated_non_overlap_tokens = remaining_tokens[
            : self.max_tokens_per_chunk - self.overlap_window
        ]
        remaining_tokens = remaining_tokens[len(truncated_non_overlap_tokens) :]
      else:
        truncated_tokens = remaining_tokens[: self.max_tokens_per_chunk]
        remaining_tokens = remaining_tokens[len(truncated_tokens) :]
      tokenized_chunks.append(truncated_tokens)
    chunks = await ot.parallel(*[
        llm.detokenize(tokenized_chunk)  # pytype: disable=wrong-arg-count
        for tokenized_chunk in tokenized_chunks
    ])
    if self.strip_whitespace:
      return [chunk.strip() for chunk in chunks]
    else:
      return chunks
