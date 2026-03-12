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

"""Basic data structures used in general retrieval QA strategies."""

import dataclasses
from typing import Any

import dataclasses_json
from onetwo.core import content as content_lib


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class Document:
  """A document that can be added to a corpus based retriever.

  Attributes:
    doc_id: Optional identifier by which the document can be referred to for
      attribution purposes. If not specified at indexing time, then an arbitrary
      identifier will be populated automatically, so at least by retrieval time
      there should always be a unique identifier.
    title: Optional title of the document.
    content: The content of the document (text or arbitrary multimodal content).
      This is the minimum required for indexing.
    url: The URL from which the document was originally fetched. If specified,
      then `self.content` is expected to exactly match the contents of the file
      at the given URL (modulo, at most, some trivial formatting adjustments),
      as of the time the URL was accessed.
    metadata: Optional metadata about the document (or other custom content).
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
