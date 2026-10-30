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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.backends import backends_test_utils
from onetwo.backends import sentencepiece_tokenizer
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import audio_chunking
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import pdf_chunking
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import routing_chunking
from onetwo.stdlib.retrieval import video_chunking
import pypdf


def _document(
    content: str | content_lib.ChunkList,
) -> retrieval_data_structures.Document:
  return retrieval_data_structures.Document(doc_id='doc', content=content)


def _media_document(
    content_type: str, media: bytes = b'media'
) -> retrieval_data_structures.Document:
  return _document(
      content_lib.ChunkList(
          [content_lib.Chunk(media, content_type=content_type)]
      )
  )


def _mixed_document() -> retrieval_data_structures.Document:
  return _document(
      content_lib.ChunkList([
          content_lib.Chunk('caption'),
          content_lib.Chunk(b'img', content_type='image/jpeg'),
      ])
  )


def _make_pdf(num_pages: int) -> bytes:
  writer = pypdf.PdfWriter()
  for _ in range(num_pages):
    writer.add_blank_page(width=612, height=792)
  buffer = io.BytesIO()
  writer.write(buffer)
  return buffer.getvalue()


class GetContentAndMediaTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('str', _document('some text'), ('text/plain', 'text')),
      (
          'single_str_chunk',
          _document(content_lib.ChunkList([content_lib.Chunk('some text')])),
          ('text/plain', 'text'),
      ),
      (
          'markdown_chunks',
          _document(
              content_lib.ChunkList([
                  content_lib.Chunk('# a', content_type='text/markdown'),
                  content_lib.Chunk('# b', content_type='text/markdown'),
              ])
          ),
          ('text/plain', 'text'),
      ),
      (
          'mixed_text_chunks',
          _document(
              content_lib.ChunkList([
                  content_lib.Chunk('# a', content_type='text/markdown'),
                  content_lib.Chunk('b', content_type='ctrl'),
              ])
          ),
          ('text/plain', 'text'),
      ),
      (
          'pdf',
          _media_document('application/pdf'),
          ('application/pdf', 'application'),
      ),
      ('audio', _media_document('audio/mpeg'), ('audio/mpeg', 'audio')),
      ('video', _media_document('video/mp4'), ('video/mp4', 'video')),
      ('image', _media_document('image/png'), ('image/png', 'image')),
      (
          'text_and_media',
          _mixed_document(),
          ('multipart/mixed', 'multipart'),
      ),
      (
          'multiple_media',
          _document(
              content_lib.ChunkList([
                  content_lib.Chunk(b'img1', content_type='image/jpeg'),
                  content_lib.Chunk(b'img2', content_type='image/png'),
              ])
          ),
          ('multipart/mixed', 'multipart'),
      ),
  )
  def test_get_content_and_media_type(self, document, expected_types):
    self.assertEqual(
        routing_chunking.get_content_and_media_type(document), expected_types
    )

  @parameterized.named_parameters(
      (
          'empty_chunk_list',
          _document(content_lib.ChunkList([])),
          "Document 'doc' has no content",
      ),
      (
          'unsupported_content_type',
          _media_document('application/zip'),
          "Unsupported content type 'application/zip' of document 'doc'",
      ),
  )
  def test_get_content_and_media_type_raises(self, document, error_regex):
    with self.assertRaisesRegex(ValueError, error_regex):
      routing_chunking.get_content_and_media_type(document)


class RoutingChunkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The default text chunker loads and registers the default SentencePiece
    # model, which is not available in tests.
    llm.reset_defaults()
    self.addCleanup(llm.reset_defaults)
    self.mock_load = self.enter_context(
        mock.patch.object(
            sentencepiece_tokenizer.SentencePieceTokenizer,
            '_load_model',
            return_value=mock.MagicMock(),
        )
    )

  def test_registered_in_chunker_registry(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='routing', kwargs={'pdf': {'pages_per_chunk': 2}}
    )

    chunker = chunker_registry.get_chunker(config)

    assert isinstance(chunker, routing_chunking.RoutingChunker)
    self.assertEqual(
        chunker.pdf, pdf_chunking.ChunkByPdfPages(pages_per_chunk=2)
    )

  def test_defaults(self):
    chunker = routing_chunking.RoutingChunker()

    self.assertEqual(
        chunker.text,
        chunking.ChunkByMaxTokens(
            max_tokens_per_chunk=512,
            overlap_window=0,
            tokenize_backend_config=(
                rag_defaults.default_tokenize_backend_config()
            ),
        ),
    )
    self.assertEqual(
        chunker.pdf, pdf_chunking.ChunkByPdfPages(pages_per_chunk=1)
    )
    self.assertEqual(
        chunker.audio,
        audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=30.0),
    )
    self.assertEqual(
        chunker.video,
        video_chunking.ChunkByVideoSeconds(seconds_per_chunk=30.0),
    )
    self.assertEqual(chunker.image, chunking.NoChunking())
    self.assertEqual(chunker.multipart, chunking.NoChunking())

  @parameterized.named_parameters(
      ('text', _document('some text'), 'text'),
      ('pdf', _media_document('application/pdf'), 'pdf'),
      ('audio', _media_document('audio/mpeg'), 'audio'),
      ('video', _media_document('video/webm'), 'video'),
      ('image', _media_document('image/png'), 'image'),
      ('mixed_content', _mixed_document(), 'multipart'),
  )
  def test_routes_by_content_type(self, document, expected_field):
    chunker = routing_chunking.RoutingChunker()

    self.assertIs(
        chunker.get_chunker(document), getattr(chunker, expected_field)
    )

  @parameterized.named_parameters(
      ('image', _media_document('image/png')),
      ('mixed_content', _mixed_document()),
  )
  def test_leaves_images_and_mixed_content_as_is(self, document):
    chunker = routing_chunking.RoutingChunker()

    with self.subTest('uses_no_chunking'):
      self.assertIsInstance(chunker.get_chunker(document), chunking.NoChunking)

    with self.subTest('returns_document_as_is'):
      self.assertEqual(list(ot.run(chunker(document))), [document])

  def test_unsupported_content_type_raises(self):
    chunker = routing_chunking.RoutingChunker()

    with self.assertRaisesRegex(ValueError, 'Unsupported content type'):
      ot.run(chunker(_media_document('application/zip')))

  def test_chunks_with_default_chunkers(self):
    chunker = routing_chunking.RoutingChunker()

    with self.subTest('loads_default_tokenizer_on_creation'):
      self.mock_load.assert_called_once()

    with self.subTest('text_is_chunked_by_512_tokens'):
      # The default tokenizer was registered when the chunker was created.
      # Replace it with one that tokenizes each character as one token.
      backends_test_utils.LLMForTest().register(register_tokenize=True)
      chunks = list(ot.run(chunker(_document('a' * 1200))))
      self.assertEqual([len(chunk.text) for chunk in chunks], [512, 512, 176])

    with self.subTest('pdf_is_chunked_by_page'):
      chunks = list(
          ot.run(
              chunker(
                  _media_document(pdf_chunking.PDF_CONTENT_TYPE, _make_pdf(3))
              )
          )
      )
      self.assertEqual(
          [chunk.doc_id for chunk in chunks],
          ['doc_chunk1', 'doc_chunk2', 'doc_chunk3'],
      )

  def test_chunks_with_custom_chunker(self):
    chunker = routing_chunking.RoutingChunker(
        pdf=pdf_chunking.ChunkByPdfPages(pages_per_chunk=2)
    )

    chunks = list(
        ot.run(
            chunker(
                _media_document(pdf_chunking.PDF_CONTENT_TYPE, _make_pdf(3))
            )
        )
    )

    self.assertLen(chunks, 2)

  def test_json_round_trip(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='routing',
        kwargs={
            'pdf': pdf_chunking.ChunkByPdfPages(pages_per_chunk=2),
            'audio': audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=5.0),
        },
    )

    reloaded_config = chunker_registry.ChunkerConfig.from_json(config.to_json())
    reloaded_chunker = chunker_registry.get_chunker(reloaded_config)

    self.assertEqual(reloaded_chunker, chunker_registry.get_chunker(config))
    assert isinstance(reloaded_chunker, routing_chunking.RoutingChunker)
    self.assertEqual(
        reloaded_chunker.pdf, pdf_chunking.ChunkByPdfPages(pages_per_chunk=2)
    )
    self.assertEqual(
        reloaded_chunker.audio,
        audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=5.0),
    )


if __name__ == '__main__':
  absltest.main()
