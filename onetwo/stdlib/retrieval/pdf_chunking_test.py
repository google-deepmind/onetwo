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

import io

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import pdf_chunking
from onetwo.stdlib.retrieval import retrieval_data_structures
import pypdf
from pypdf import generic

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)


def _make_pdf(num_pages: int) -> bytes:
  """Returns the bytes of a PDF whose page i (1-indexed) reads 'Page i'."""
  writer = pypdf.PdfWriter()
  font = generic.DictionaryObject({
      generic.NameObject('/Type'): generic.NameObject('/Font'),
      generic.NameObject('/Subtype'): generic.NameObject('/Type1'),
      generic.NameObject('/BaseFont'): generic.NameObject('/Helvetica'),
  })
  for i in range(1, num_pages + 1):
    page = writer.add_blank_page(width=612, height=792)
    page[generic.NameObject('/Resources')] = generic.DictionaryObject({
        generic.NameObject('/Font'): generic.DictionaryObject(
            {generic.NameObject('/F1'): font}
        )
    })
    contents = generic.ContentStream(None, writer)
    contents.set_data(f'BT /F1 24 Tf 72 720 Td (Page {i}) Tj ET'.encode())
    page.replace_contents(contents)
  buffer = io.BytesIO()
  writer.write(buffer)
  return buffer.getvalue()


def _get_page_texts(pdf_bytes: bytes) -> list[str]:
  """Returns the extracted text of each page of the PDF."""
  reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
  return [page.extract_text().strip() for page in reader.pages]


def _get_document_page_texts(
    document: retrieval_data_structures.Document,
) -> list[str]:
  """Returns the page texts of a document containing a single PDF chunk."""
  assert isinstance(document.content, content_lib.ChunkList)
  pdf_bytes = document.content.chunks[0].content
  assert isinstance(pdf_bytes, bytes)
  return _get_page_texts(pdf_bytes)


def _make_pdf_document(num_pages: int) -> retrieval_data_structures.Document:
  return retrieval_data_structures.Document(
      doc_id='doc',
      title='Test Title',
      content=content_lib.ChunkList([
          content_lib.Chunk(
              _make_pdf(num_pages),
              content_type=pdf_chunking.PDF_CONTENT_TYPE,
          )
      ]),
      metadata={'key': 'value'},
  )


class ChunkByPdfPagesTest(parameterized.TestCase):

  def test_registered_in_chunker_registry(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='pdf_pages', kwargs={'pages_per_chunk': 3}
    )
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, pdf_chunking.ChunkByPdfPages)
    self.assertEqual(chunker.pages_per_chunk, 3)

  def test_mock_pdf_has_expected_pages(self):
    self.assertEqual(
        _get_page_texts(_make_pdf(3)), ['Page 1', 'Page 2', 'Page 3']
    )

  @parameterized.named_parameters(
      ('1_page_1_per_chunk', 1, 1, 1),
      ('1_page_3_per_chunk', 1, 3, 1),
      ('2_pages_1_per_chunk', 2, 1, 2),
      ('2_pages_2_per_chunk', 2, 2, 1),
      ('5_pages_1_per_chunk', 5, 1, 5),
      ('5_pages_2_per_chunk', 5, 2, 3),
      ('5_pages_3_per_chunk', 5, 3, 2),
      ('5_pages_10_per_chunk', 5, 10, 1),
      ('10_pages_1_per_chunk', 10, 1, 10),
      ('10_pages_2_per_chunk', 10, 2, 5),
      ('10_pages_3_per_chunk', 10, 3, 4),
      ('10_pages_10_per_chunk', 10, 10, 1),
      ('10_pages_15_per_chunk', 10, 15, 1),
  )
  def test_chunks_pdf_by_pages(
      self, num_pages, pages_per_chunk, expected_num_chunks
  ):
    document = _make_pdf_document(num_pages)
    chunker = pdf_chunking.ChunkByPdfPages(pages_per_chunk=pages_per_chunk)

    chunks = list(ot.run(chunker(document)))

    with self.subTest('correct_number_of_chunks'):
      self.assertLen(chunks, expected_num_chunks)

    # The pages (1-indexed) expected in each chunk.
    expected_page_texts = [
        [
            f'Page {p}'
            for p in range(
                start + 1, min(start + pages_per_chunk, num_pages) + 1
            )
        ]
        for start in range(0, num_pages, pages_per_chunk)
    ]

    for i, chunk in enumerate(chunks):
      with self.subTest(f'chunk_{i+1}_is_single_pdf'):
        self.assertIsInstance(chunk.content, content_lib.ChunkList)
        self.assertLen(chunk.content, 1)
        self.assertEqual(
            chunk.content.chunks[0].content_type,
            pdf_chunking.PDF_CONTENT_TYPE,
        )

      with self.subTest(f'chunk_{i+1}_has_correct_pages'):
        self.assertEqual(
            _get_document_page_texts(chunk),
            expected_page_texts[i],
        )

      with self.subTest(f'chunk_{i+1}_has_correct_attributes'):
        self.assertEqual(chunk.doc_id, f'doc_chunk{i+1}')
        self.assertEqual(chunk.title, document.title)
        self.assertEqual(
            chunk.metadata,
            {
                'key': 'value',
                _ORIGINAL_DOC_ID: 'doc',
                _CHUNK_NUMBER: i + 1,
                _TOTAL_NUMBER_OF_CHUNKS: expected_num_chunks,
            },
        )

    with self.subTest('original_document_is_unchanged'):
      self.assertEqual(document.doc_id, 'doc')
      self.assertEqual(document.metadata, {'key': 'value'})
      self.assertEqual(
          _get_document_page_texts(document),
          [f'Page {p}' for p in range(1, num_pages + 1)],
      )

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  def test_non_positive_pages_per_chunk_raises(self, pages_per_chunk):
    chunker = pdf_chunking.ChunkByPdfPages(pages_per_chunk=pages_per_chunk)
    with self.assertRaisesRegex(ValueError, 'pages_per_chunk must be positive'):
      ot.run(chunker(_make_pdf_document(2)))

  @parameterized.named_parameters(
      ('string_content', 'some text'),
      (
          'non_pdf_chunk',
          content_lib.ChunkList(
              [content_lib.Chunk(b'img', content_type='image/jpeg')]
          ),
      ),
      (
          'multiple_chunks',
          content_lib.ChunkList([
              content_lib.Chunk(
                  _make_pdf(1), content_type=pdf_chunking.PDF_CONTENT_TYPE
              ),
              content_lib.Chunk('some text'),
          ]),
      ),
  )
  def test_non_pdf_document_raises(self, content):
    document = retrieval_data_structures.Document(doc_id='doc', content=content)
    chunker = pdf_chunking.ChunkByPdfPages(pages_per_chunk=1)
    with self.assertRaisesRegex(ValueError, 'single .application/pdf. chunk'):
      ot.run(chunker(document))


if __name__ == '__main__':
  absltest.main()
