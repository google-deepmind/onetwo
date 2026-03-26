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

"""Data structures for Retrieval-Augmented Generation (RAG).

This module defines the fundamental data units used to store, manage, and track
information within a retrieval system. In a typical RAG workflow, external data
(corpora) is processed into structured 'Documents' which are then indexed for
efficient lookup.

A core concept to understand in this module is 'Chunking'—the process of
breaking down large-scale information (like a book or a technical manual) into
smaller, semantically cohesive units. These units are more digestible for LLMs
and allow for more precise retrieval as well as attribution to the source
document.

The structures here ensure that even after a document is fragmented into
multiple chunks, its original context and origin (metadata) are preserved.
"""

import dataclasses
from typing import Any

import dataclasses_json
from onetwo.core import content as content_lib

# ============================================================================
# Metadata constants for document tracking and attribution.
# ============================================================================

# These constants define standard keys for the metadata dictionary in a
# Document. They are primarily used by 'Chunkers' to maintain a link between a
# fragmented piece of content (the chunk) and its source material
# (the original document).

# **chunk_number**: The sequential position of the fragment (chunk) within the
# source document. This is 1-indexed (e.g., the first paragraph of a PDF would
# be chunk 1).
METADATA_FIELD_CHUNK_NUMBER = 'chunk_number'

# **total_number_of_chunks**: The total count of fragments (chunks) derived from
# the original Document. This allows the system to determine the relative
# coverage of a specific chunk within its source document.
METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS = 'total_number_of_chunks'

# **original_doc_id**: The unique identifier of the parent Document from which
# this chunk was extracted. Essential for cross-referencing and de-duplication
# during retrieval.
METADATA_FIELD_ORIGINAL_DOC_ID = 'original_doc_id'


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class Document:
  """The primary data unit for storage and retrieval.

  A `Document` encapsulates both the raw content (text or multimodal) and the
  contextual information (metadata) required for an LLM to utilize it as a
  source of truth. In an agentic workflow, this object represents the 'answer
  source' found during a retrieval operation.

  Attributes:
    doc_id: Optional identifier by which the document can be referred to for
      attribution purposes. If not specified at indexing time, then an arbitrary
      identifier will be populated automatically, so at least by retrieval time
      there should always be a unique identifier.
    title: Optional title of the document.
    content: The primary payload of the document. This can be a simple string
      for text-only data or a `content_lib.ChunkList` for multimodal content
      (e.g., text interleaved with images or video references).
    url: The URL from which the document was originally fetched. If specified,
      then `self.content` is expected to exactly match the contents of the file
      at the given URL (modulo, at most, some trivial formatting adjustments),
      as of the time the URL was accessed.
    metadata: A flexible dictionary for storing arbitrary key-value pairs.
      Commonly used for the chunking fields defined above or domain-specific
      data like 'author', 'timestamp', or 'category'.
    text: A view on the document's `content`, represented as a plain text string
      (with just placeholders like '<image/jpeg>' for any multimodal content).
  """

  doc_id: str = ''
  title: str = ''
  content: str | content_lib.ChunkList = ''
  url: str = ''
  # TODO: Add a field to track the timestamp at which the content was
  # fetched from the given URL?
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def text(self) -> str:
    """Returns just the text, with placeholders for multimodal content."""
    # ChunkList's existing '__str__' method already gives us a reasonable text
    # representation, so we just delegate to that for now.
    return str(self.content)
