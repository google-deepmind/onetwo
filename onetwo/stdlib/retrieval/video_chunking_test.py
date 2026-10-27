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

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import ffmpeg
from onetwo import ot
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import video_chunking

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)

# Clip durations are only accurate up to about one frame (0.1s at 10 fps).
_DURATION_TOLERANCE_SECONDS = 0.15


def _make_video(duration_seconds: float, extension: str = '.mp4') -> bytes:
  """Returns the bytes of a 10 fps test-pattern video of the given duration."""
  with tempfile.TemporaryDirectory() as temp_dir:
    path = os.path.join(temp_dir, f'video{extension}')
    (
        ffmpeg.input(
            f'testsrc=duration={duration_seconds}:size=64x48:rate=10',
            f='lavfi',
        )
        .output(path)
        .run(quiet=True, overwrite_output=True)
    )
    with open(path, 'rb') as f:
      return f.read()


def _get_duration_seconds(video_bytes: bytes, extension: str) -> float:
  with tempfile.TemporaryDirectory() as temp_dir:
    path = os.path.join(temp_dir, f'video{extension}')
    with open(path, 'wb') as f:
      f.write(video_bytes)
    return float(ffmpeg.probe(path)['format']['duration'])


def _make_video_document(
    video: bytes, content_type: str = 'video/mp4'
) -> retrieval_data_structures.Document:
  return retrieval_data_structures.Document(
      doc_id='doc',
      title='Test Title',
      content=content_lib.ChunkList(
          [content_lib.Chunk(video, content_type=content_type)]
      ),
      metadata={'key': 'value'},
  )


def _get_video_chunk(
    document: retrieval_data_structures.Document,
) -> tuple[bytes, str]:
  """Returns the bytes and content type of a document's single chunk."""
  assert isinstance(document.content, content_lib.ChunkList)
  assert len(document.content) == 1
  chunk = document.content.chunks[0]
  assert isinstance(chunk.content, bytes)
  return chunk.content, chunk.content_type


class ChunkByVideoSecondsTest(parameterized.TestCase):

  def test_registered_in_chunker_registry(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='video_seconds', kwargs={'seconds_per_chunk': 10.0}
    )
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, video_chunking.ChunkByVideoSeconds)
    self.assertEqual(chunker.seconds_per_chunk, 10.0)

  @parameterized.named_parameters(
      ('1s_video_1s_per_chunk', 1.0, 1.0, [1.0]),
      ('1s_video_10s_per_chunk', 1.0, 10.0, [1.0]),
      ('2s_video_1s_per_chunk', 2.0, 1.0, [1.0, 1.0]),
      ('2500ms_video_1s_per_chunk', 2.5, 1.0, [1.0, 1.0, 0.5]),
      ('5s_video_2s_per_chunk', 5.0, 2.0, [2.0, 2.0, 1.0]),
      ('5s_video_5s_per_chunk', 5.0, 5.0, [5.0]),
  )
  def test_chunks_mp4_by_seconds(
      self, duration_seconds, seconds_per_chunk, expected_chunk_durations
  ):
    document = _make_video_document(_make_video(duration_seconds))
    chunker = video_chunking.ChunkByVideoSeconds(
        seconds_per_chunk=seconds_per_chunk
    )

    chunks = list(ot.run(chunker(document)))

    expected_num_chunks = len(expected_chunk_durations)
    with self.subTest('correct_number_of_chunks'):
      self.assertLen(chunks, expected_num_chunks)

    for i, chunk in enumerate(chunks):
      chunk_bytes, content_type = _get_video_chunk(chunk)
      with self.subTest(f'chunk_{i+1}_is_mp4'):
        self.assertEqual(content_type, 'video/mp4')

      with self.subTest(f'chunk_{i+1}_has_correct_duration'):
        self.assertAlmostEqual(
            _get_duration_seconds(chunk_bytes, '.mp4'),
            expected_chunk_durations[i],
            delta=_DURATION_TOLERANCE_SECONDS,
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

  def test_chunks_webm_keeps_format(self):
    document = _make_video_document(
        _make_video(3.0, extension='.webm'), content_type='video/webm'
    )
    chunker = video_chunking.ChunkByVideoSeconds(seconds_per_chunk=1.0)

    chunks = list(ot.run(chunker(document)))

    self.assertLen(chunks, 3)
    for chunk in chunks:
      chunk_bytes, content_type = _get_video_chunk(chunk)
      self.assertEqual(content_type, 'video/webm')
      self.assertAlmostEqual(
          _get_duration_seconds(chunk_bytes, '.webm'),
          1.0,
          delta=_DURATION_TOLERANCE_SECONDS,
      )

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  def test_non_positive_seconds_per_chunk_raises(self, seconds_per_chunk):
    chunker = video_chunking.ChunkByVideoSeconds(
        seconds_per_chunk=seconds_per_chunk
    )
    with self.assertRaisesRegex(
        ValueError, 'seconds_per_chunk must be positive'
    ):
      ot.run(chunker(_make_video_document(b'unused')))

  @parameterized.named_parameters(
      ('flv', 'video/x-flv'),
      ('3gpp', 'video/3gpp'),
  )
  def test_unsupported_content_type_raises(self, content_type):
    document = _make_video_document(b'unused', content_type=content_type)
    chunker = video_chunking.ChunkByVideoSeconds(seconds_per_chunk=1.0)
    with self.assertRaisesRegex(
        ValueError, f'Unsupported video content type: {content_type!r}'
    ):
      ot.run(chunker(document))

  @parameterized.named_parameters(
      ('string_content', 'some text'),
      (
          'non_video_chunk',
          content_lib.ChunkList(
              [content_lib.Chunk(b'img', content_type='image/jpeg')]
          ),
      ),
      (
          'multiple_chunks',
          content_lib.ChunkList([
              content_lib.Chunk(b'video', content_type='video/mp4'),
              content_lib.Chunk('some text'),
          ]),
      ),
  )
  def test_non_video_document_raises(self, content):
    document = retrieval_data_structures.Document(doc_id='doc', content=content)
    chunker = video_chunking.ChunkByVideoSeconds(seconds_per_chunk=1.0)
    with self.assertRaisesRegex(ValueError, 'single video chunk'):
      ot.run(chunker(document))


if __name__ == '__main__':
  absltest.main()
