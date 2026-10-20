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

from collections.abc import Iterable
import copy
import dataclasses
import io

from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import retrieval_data_structures
import pypdf

PDF_CONTENT_TYPE = 'application/pdf'

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)


def _get_pdf_bytes(document: retrieval_data_structures.Document) -> bytes:
  """Returns the PDF bytes of `document`.

  Args:
    document: A document whose content is a `ChunkList` with exactly one chunk
      of content type `application/pdf`.

  Raises:
    ValueError: If the document content is not a single PDF chunk.
  """
  content = document.content
  if (
      not isinstance(content, content_lib.ChunkList)
      or len(content) != 1
      or content.chunks[0].content_type != PDF_CONTENT_TYPE
  ):
    raise ValueError(
        f'Expected document {document.doc_id!r} to contain a single'
        f' {PDF_CONTENT_TYPE!r} chunk, got: {content!r}'
    )
  pdf_bytes = content.chunks[0].content
  if not isinstance(pdf_bytes, bytes):
    raise ValueError(
        f'Expected PDF content to be bytes, got {type(pdf_bytes)}.'
    )
  return pdf_bytes


@dataclasses.dataclass(kw_only=True)
class ChunkByPdfPages(chunking.Chunker):
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

  pages_per_chunk: int = 1

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> Iterable[retrieval_data_structures.Document]:
    """Returns `document` split into PDFs of up to `pages_per_chunk` pages.

    Args:
      document: The document to chunk.

    Raises:
      ValueError: If `pages_per_chunk` is not positive or if the document
        content is not a single PDF chunk.
    """
    if self.pages_per_chunk <= 0:
      raise ValueError(
          f'pages_per_chunk must be positive, got {self.pages_per_chunk}'
      )
    reader = pypdf.PdfReader(io.BytesIO(_get_pdf_bytes(document)))
    num_pages = len(reader.pages)
    page_starts = range(0, num_pages, self.pages_per_chunk)
    total_number_of_chunks = len(page_starts)
    chunks = []
    for i, start in enumerate(page_starts):
      writer = pypdf.PdfWriter()
      for page in reader.pages[start : start + self.pages_per_chunk]:
        writer.add_page(page)
      buffer = io.BytesIO()
      writer.write(buffer)

      # We avoid deep-copying the whole document so as not to copy the full
      # original PDF bytes once per chunk.
      chunk = dataclasses.replace(
          document,
          # Modify the doc_id of every chunk to produce a unique id per chunk.
          doc_id=f'{document.doc_id}_chunk{i+1}',
          content=content_lib.ChunkList([
              content_lib.Chunk(
                  buffer.getvalue(), content_type=PDF_CONTENT_TYPE
              )
          ]),
          metadata=copy.deepcopy(document.metadata),
      )
      chunk.metadata[_ORIGINAL_DOC_ID] = document.doc_id
      chunk.metadata[_CHUNK_NUMBER] = i + 1
      chunk.metadata[_TOTAL_NUMBER_OF_CHUNKS] = total_number_of_chunks
      chunks.append(chunk)
    return chunks
