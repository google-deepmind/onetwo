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

"""Chunkers for video documents.

These chunkers operate on documents whose content is a single video file,
represented as a `content_lib.ChunkList` containing one `video/*` chunk. Each
output chunk is itself a (shorter) video file.

Importing this module registers `ChunkByVideoSeconds` in `chunker_registry`
under the name 'video_seconds'.
"""

from collections.abc import Sequence
import dataclasses
import math
import os
import tempfile
from typing import ClassVar

from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking

# File extensions of the supported video content types. Chunks are encoded in
# the same container format (and with the same content type) as the input.
_EXTENSION_BY_CONTENT_TYPE = {
    'video/mp4': '.mp4',
    'video/mpeg': '.mpg',
    'video/ogg': '.ogv',
    'video/quicktime': '.mov',
    'video/webm': '.webm',
    'video/x-matroska': '.mkv',
    'video/x-msvideo': '.avi',
}

# Number of decimals to which `duration / seconds_per_chunk` is rounded before
# computing the number of chunks, so that e.g. a 10.0004s video split into 5s
# chunks yields 2 chunks rather than 2 chunks plus a near-empty third one.
_NUM_CHUNKS_ROUNDING_DECIMALS = 3


@dataclasses.dataclass(kw_only=True)
class ChunkByVideoSeconds(chunking.SingleMediaChunker):
  """Chunks a video document into clips of up to `seconds_per_chunk` each.

  The input document's content must be a `ChunkList` containing a single
  `video/*` chunk. Each output chunk is a copy of the input document whose
  content is a `ChunkList` with a single video chunk holding a contiguous clip
  of the video (re-encoded, so that clips start and end at the requested times
  rather than at keyframes). The last chunk may be shorter. Only video in a
  known container format (MP4, MPEG, Ogg, QuickTime, WebM, Matroska, AVI) is
  supported, and each chunk keeps its format and content type. The metadata of
  each chunk additionally includes `original_doc_id`, `chunk_number` and
  `total_number_of_chunks`.

  Attributes:
    seconds_per_chunk: The maximum duration of each chunk, in seconds.
  """

  media_description: ClassVar[str] = 'video'

  seconds_per_chunk: float = 30.0

  def _accepts(self, content_type: str) -> bool:
    """Overridden from base class (SingleMediaChunker)."""
    return content_type.startswith('video/')

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
      ValueError: If `seconds_per_chunk` is not positive, or if `content_type`
        is not a supported video content type.
    """
    if self.seconds_per_chunk <= 0:
      raise ValueError(
          f'seconds_per_chunk must be positive, got {self.seconds_per_chunk}'
      )
    if content_type not in _EXTENSION_BY_CONTENT_TYPE:
      raise ValueError(
          f'Unsupported video content type: {content_type!r}. Supported'
          f' content types: {sorted(_EXTENSION_BY_CONTENT_TYPE)}.'
      )
    extension = _EXTENSION_BY_CONTENT_TYPE[content_type]
    # Imported here so that importing this module (and registering the
    # chunker) does not load the video library.
    import ffmpeg  # pylint: disable=g-import-not-at-top

    # ffmpeg reads and writes files, so we go through a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
      input_path = os.path.join(temp_dir, f'input{extension}')
      with open(input_path, 'wb') as f:
        f.write(media)
      duration = float(ffmpeg.probe(input_path)['format']['duration'])
      num_chunks = math.ceil(
          round(
              duration / self.seconds_per_chunk, _NUM_CHUNKS_ROUNDING_DECIMALS
          )
      )
      pieces = []
      for i in range(num_chunks):
        output_path = os.path.join(temp_dir, f'chunk{i+1}{extension}')
        (
            ffmpeg.input(input_path, ss=i * self.seconds_per_chunk)
            .output(output_path, t=self.seconds_per_chunk)
            .run(quiet=True, overwrite_output=True)
        )
        with open(output_path, 'rb') as f:
          pieces.append((f.read(), content_type))
    return pieces


chunker_registry.register_chunker('video_seconds', ChunkByVideoSeconds)
