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

"""Chunkers for PDF documents.

These chunkers operate on documents whose content is a single PDF, represented
as a `content_lib.ChunkList` containing one `application/pdf` chunk. Each output
chunk is itself a (smaller) PDF.
"""

from collections.abc import Sequence
import dataclasses
import io
from typing import ClassVar

from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import registry

PDF_CONTENT_TYPE = 'application/pdf'


@registry.register('ChunkByPdfPages')
@dataclasses.dataclass(kw_only=True)
class ChunkByPdfPages(chunking.SingleMediaChunker):
  """Chunks a PDF document into PDFs of up to `pages_per_chunk` pages each.

  The input document's content must be a `ChunkList` containing a single
  `application/pdf` chunk. Each output chunk is a copy of the input document
  whose content is a `ChunkList` with a single `application/pdf` chunk holding
  a contiguous range of pages. The last chunk may contain fewer pages. The
  metadata of each chunk additionally includes `original_doc_id`,
  `chunk_number` and `total_number_of_chunks`.

  Attributes:
    pages_per_chunk: The maximum number of pages per chunk.
  """

  media_description: ClassVar[str] = repr(PDF_CONTENT_TYPE)

  pages_per_chunk: int = 1

  def _accepts(self, content_type: str) -> bool:
    """Overridden from base class (SingleMediaChunker)."""
    return content_type == PDF_CONTENT_TYPE

  def _split(
      self, media: bytes, content_type: str
  ) -> Sequence[tuple[bytes, str]]:
    """Overridden from base class (SingleMediaChunker).

    Args:
      media: The bytes of the media file to split.
      content_type: The content type of `media`.

    Returns:
      The pieces of `media`, as (bytes, content type), in order.

    Raises:
      ValueError: If `pages_per_chunk` is not positive.
    """
    if self.pages_per_chunk <= 0:
      raise ValueError(
          f'pages_per_chunk must be positive, got {self.pages_per_chunk}'
      )
    # Imported here so that importing this module (and registering the
    # chunker) does not load the PDF library.
    import pypdf  # pylint: disable=g-import-not-at-top

    reader = pypdf.PdfReader(io.BytesIO(media))
    pieces = []
    for start in range(0, len(reader.pages), self.pages_per_chunk):
      writer = pypdf.PdfWriter()
      for page in reader.pages[start : start + self.pages_per_chunk]:
        writer.add_page(page)
      buffer = io.BytesIO()
      writer.write(buffer)
      pieces.append((buffer.getvalue(), content_type))
    return pieces
