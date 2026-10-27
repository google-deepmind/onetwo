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

"""Chunkers for audio documents.

These chunkers operate on documents whose content is a single audio file,
represented as a `content_lib.ChunkList` containing one `audio/*` chunk. Each
output chunk is itself a (shorter) audio file.

Importing this module registers `ChunkByAudioSeconds` in `chunker_registry`
under the name 'audio_seconds'.
"""

from collections.abc import Sequence
import dataclasses
import io
from typing import ClassVar

from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import chunking
import pydub

# ffmpeg format names of the audio content types that are decoded and encoded
# in their own format. Audio in other formats is decoded by letting ffmpeg probe
# the content, and chunks are encoded as WAV.
_FORMAT_BY_CONTENT_TYPE = {
    'audio/flac': 'flac',
    'audio/x-flac': 'flac',
    'audio/mp3': 'mp3',
    'audio/mpeg': 'mp3',
    'audio/ogg': 'ogg',
    'audio/vnd.wave': 'wav',
    'audio/wav': 'wav',
    'audio/wave': 'wav',
    'audio/x-wav': 'wav',
}
_FALLBACK_FORMAT = 'wav'
_FALLBACK_CONTENT_TYPE = 'audio/wav'


@dataclasses.dataclass(kw_only=True)
class ChunkByAudioSeconds(chunking.SingleMediaChunker):
  """Chunks an audio document into clips of up to `seconds_per_chunk` each.

  The input document's content must be a `ChunkList` containing a single
  `audio/*` chunk. Each output chunk is a copy of the input document whose
  content is a `ChunkList` with a single audio chunk holding a contiguous clip
  of the audio. The last chunk may be shorter. Audio in a known format (WAV,
  MP3, OGG, FLAC) keeps its format and content type; other audio is encoded as
  WAV. The metadata of each chunk additionally includes `original_doc_id`,
  `chunk_number` and `total_number_of_chunks`.

  Attributes:
    seconds_per_chunk: The maximum duration of each chunk, in seconds.
  """

  media_description: ClassVar[str] = 'audio'

  seconds_per_chunk: float = 30.0

  def _accepts(self, content_type: str) -> bool:
    """Overridden from base class (SingleMediaChunker)."""
    return content_type.startswith('audio/')

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
      ValueError: If `seconds_per_chunk` is not positive.
    """
    if self.seconds_per_chunk <= 0:
      raise ValueError(
          f'seconds_per_chunk must be positive, got {self.seconds_per_chunk}'
      )
    audio_format = _FORMAT_BY_CONTENT_TYPE.get(content_type)
    audio = pydub.AudioSegment.from_file(io.BytesIO(media), format=audio_format)
    if audio_format is None:
      audio_format = _FALLBACK_FORMAT
      content_type = _FALLBACK_CONTENT_TYPE

    # pydub indexes audio in milliseconds.
    milliseconds_per_chunk = max(1, round(self.seconds_per_chunk * 1000))
    pieces = []
    for start in range(0, len(audio), milliseconds_per_chunk):
      buffer = io.BytesIO()
      audio[start : start + milliseconds_per_chunk].export(
          buffer, format=audio_format
      )
      pieces.append((buffer.getvalue(), content_type))
    return pieces


chunker_registry.register_chunker('audio_seconds', ChunkByAudioSeconds)
