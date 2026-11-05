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

"""Chunker that routes each document to a sub-chunker based on its content type.

This lets a single index chunk a mixed corpus (e.g. text, PDFs, audio and video
files loaded with `MultimodalDatasetLoader`) with a different strategy per
media type: tokens for text, pages for PDFs and seconds for audio and video,
while images and mixed content are left as-is.

Importing this module registers `RoutingChunker` in `chunker_registry` under the
name 'routing' (and, through its imports, the PDF, audio and video chunkers).
"""

from collections.abc import Iterable
import dataclasses
from typing import Any

from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.stdlib.retrieval import audio_chunking
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import pdf_chunking
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import registry
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import video_chunking

# Content type under which documents with text content (a `str`, or a
# `ChunkList` with only string chunks) are routed, whatever the content type of
# their chunks (e.g. 'text/markdown' or 'text/html'), so that all text is
# chunked the same way.
_TEXT_CONTENT_TYPE = 'text/plain'

# Content type under which documents whose content is a `ChunkList` mixing
# several chunks of different kinds (e.g. text and images) are routed. This is
# the standard MIME type for content made of several parts of different types.
_MIXED_CONTENT_TYPE = 'multipart/mixed'

# Media types (the part of a content type before '/') of the single-chunk
# documents that can be routed, besides text and PDFs.
_AUDIO_MEDIA_TYPE = 'audio'
_VIDEO_MEDIA_TYPE = 'video'
_IMAGE_MEDIA_TYPE = 'image'

# Keyword arguments of the default sub-chunker of each field of
# `RoutingChunker`. Text is tokenized with the default SentencePiece tokenizer
# of the RAG API (see `rag_defaults`). Images and mixed content are left as-is
# (`chunking.NoChunking`, which takes no arguments).
DEFAULT_CHUNKER_KWARGS: dict[str, dict[str, Any]] = {
    'text': {
        'max_tokens_per_chunk': 512,
        'overlap_window': 0,
        'tokenize_backend_config': (
            rag_defaults.default_tokenize_backend_config()
        ),
    },
    'pdf': {'pages_per_chunk': 1},
    'audio': {'seconds_per_chunk': 30.0},
    'video': {'seconds_per_chunk': 30.0},
    'image': {},
    'multipart': {},
}


def _media_type(content_type: str) -> str:
  """Returns the media type of `content_type` (the part before '/')."""
  return content_type.partition('/')[0]


def get_content_and_media_type(
    document: retrieval_data_structures.Document,
) -> tuple[str, str]:
  """Returns the content type and media type under which `document` is routed.

  Args:
    document: The document to route.

  Returns:
    A tuple `(content_type, media_type)`, where `media_type` is the part of
    `content_type` before '/' (e.g. 'audio' for 'audio/mpeg'), and
    `content_type` is:
    - For text content (a `str`, or a `ChunkList` whose chunks all have a
      string content type, see `content_lib.is_str_content_type`):
      'text/plain', whatever the content type of the chunks.
    - For a `ChunkList` with a single PDF, audio, video or image chunk: the
      content type of that chunk.
    - For a `ChunkList` with several chunks, not all of which are text (e.g.
      text and images): 'multipart/mixed'.

  Raises:
    ValueError: If the content of `document` is an empty `ChunkList`, or a
      single chunk of an unsupported content type (i.e. not text, PDF, audio,
      video or image).
  """
  content = document.content
  if isinstance(content, str):
    return _TEXT_CONTENT_TYPE, _media_type(_TEXT_CONTENT_TYPE)
  chunks = content.chunks
  if not chunks:
    raise ValueError(f'Document {document.doc_id!r} has no content.')
  if all(
      content_lib.is_str_content_type(chunk.content_type) for chunk in chunks
  ):
    return _TEXT_CONTENT_TYPE, _media_type(_TEXT_CONTENT_TYPE)
  if len(chunks) > 1:
    return _MIXED_CONTENT_TYPE, _media_type(_MIXED_CONTENT_TYPE)
  content_type = chunks[0].content_type
  media_type = _media_type(content_type)
  if content_type != pdf_chunking.PDF_CONTENT_TYPE and media_type not in (
      _AUDIO_MEDIA_TYPE,
      _VIDEO_MEDIA_TYPE,
      _IMAGE_MEDIA_TYPE,
  ):
    raise ValueError(
        f'Unsupported content type {content_type!r} of document'
        f' {document.doc_id!r}: expected text, a PDF, audio, video or an'
        ' image.'
    )
  return content_type, media_type


@registry.register('RoutingChunker')
@dataclasses.dataclass(kw_only=True)
class RoutingChunker(chunking.Chunker):
  """Chunks each document with a sub-chunker chosen by its content type.

  The content type of a document is determined by `get_content_and_media_type`,
  and the document is chunked with:
  - `text` for text (of any 'text/*' content type);
  - `pdf` for a PDF file;
  - `audio` for an audio file;
  - `video` for a video file;
  - `image` for an image;
  - `multipart` for content that mixes several kinds of chunks (e.g. text and
    images).
  Documents of any other content type (or without content) are rejected with a
  ValueError.

  Each sub-chunker can be configured (e.g. `RoutingChunker(text=
  chunking.ChunkByMaxTokens(max_tokens_per_chunk=256))`), but its class is
  fixed: to chunk documents differently, write a custom `Chunker`. By default,
  the sub-chunkers are created with `DEFAULT_CHUNKER_KWARGS`. The default
  tokenizer of `text` is only loaded when the first text document is chunked.

  The chunks are those returned by the sub-chunker, so they carry its metadata
  (e.g. `original_doc_id` and `chunk_number`).

  Attributes:
    text: Chunker for text documents.
    pdf: Chunker for PDF documents.
    audio: Chunker for audio documents.
    video: Chunker for video documents.
    image: Chunker for image documents. Images are small enough to be embedded
      whole, so they are left as-is.
    multipart: Chunker for documents mixing several kinds of chunks. There is
      no generic way of splitting such content, so it is left as-is.
  """

  text: chunking.ChunkByMaxTokens = dataclasses.field(
      default_factory=lambda: chunking.ChunkByMaxTokens(
          **DEFAULT_CHUNKER_KWARGS['text']
      )
  )
  pdf: pdf_chunking.ChunkByPdfPages = dataclasses.field(
      default_factory=lambda: pdf_chunking.ChunkByPdfPages(
          **DEFAULT_CHUNKER_KWARGS['pdf']
      )
  )
  audio: audio_chunking.ChunkByAudioSeconds = dataclasses.field(
      default_factory=lambda: audio_chunking.ChunkByAudioSeconds(
          **DEFAULT_CHUNKER_KWARGS['audio']
      )
  )
  video: video_chunking.ChunkByVideoSeconds = dataclasses.field(
      default_factory=lambda: video_chunking.ChunkByVideoSeconds(
          **DEFAULT_CHUNKER_KWARGS['video']
      )
  )
  image: chunking.NoChunking = dataclasses.field(
      default_factory=lambda: chunking.NoChunking(
          **DEFAULT_CHUNKER_KWARGS['image']
      )
  )
  multipart: chunking.NoChunking = dataclasses.field(
      default_factory=lambda: chunking.NoChunking(
          **DEFAULT_CHUNKER_KWARGS['multipart']
      )
  )

  def get_chunker(
      self, document: retrieval_data_structures.Document
  ) -> chunking.Chunker:
    """Returns the sub-chunker for `document`.

    Args:
      document: The document to chunk.

    Raises:
      ValueError: If the content type of `document` is not supported (see
        `get_content_and_media_type`).
    """
    content_type, media_type = get_content_and_media_type(document)
    if content_type == _TEXT_CONTENT_TYPE:
      return self.text
    if content_type == pdf_chunking.PDF_CONTENT_TYPE:
      return self.pdf
    if media_type == _AUDIO_MEDIA_TYPE:
      return self.audio
    if media_type == _VIDEO_MEDIA_TYPE:
      return self.video
    if media_type == _IMAGE_MEDIA_TYPE:
      return self.image
    if content_type == _MIXED_CONTENT_TYPE:
      return self.multipart
    raise ValueError(
        f'No chunker for content type {content_type!r} of document'
        f' {document.doc_id!r}.'
    )

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      document: retrieval_data_structures.Document,
  ) -> Iterable[retrieval_data_structures.Document]:
    """Returns the chunks of `document` from its content type's sub-chunker.

    Args:
      document: The document to chunk.

    Raises:
      ValueError: If the content type of `document` is not supported (see
        `get_content_and_media_type`).
    """
    return await self.get_chunker(document)(document)


chunker_registry.register_chunker('routing', RoutingChunker)
