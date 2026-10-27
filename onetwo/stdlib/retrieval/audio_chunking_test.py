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
from onetwo.stdlib.retrieval import audio_chunking
from onetwo.stdlib.retrieval import chunker_registry
from onetwo.stdlib.retrieval import retrieval_data_structures
import pydub
from pydub import generators

_ORIGINAL_DOC_ID = retrieval_data_structures.METADATA_FIELD_ORIGINAL_DOC_ID
_CHUNK_NUMBER = retrieval_data_structures.METADATA_FIELD_CHUNK_NUMBER
_TOTAL_NUMBER_OF_CHUNKS = (
    retrieval_data_structures.METADATA_FIELD_TOTAL_NUMBER_OF_CHUNKS
)


def _make_audio(duration_ms: int, audio_format: str = 'wav') -> bytes:
  """Returns the bytes of a 440 Hz tone of `duration_ms` milliseconds."""
  tone = generators.Sine(440).to_audio_segment(duration=duration_ms)
  buffer = io.BytesIO()
  tone.export(buffer, format=audio_format)
  return buffer.getvalue()


def _make_audio_document(
    audio: bytes, content_type: str = 'audio/wav'
) -> retrieval_data_structures.Document:
  return retrieval_data_structures.Document(
      doc_id='doc',
      title='Test Title',
      content=content_lib.ChunkList(
          [content_lib.Chunk(audio, content_type=content_type)]
      ),
      metadata={'key': 'value'},
  )


def _get_audio_chunk(
    document: retrieval_data_structures.Document,
) -> content_lib.Chunk:
  """Returns the single chunk of a document's `ChunkList` content."""
  assert isinstance(document.content, content_lib.ChunkList)
  assert len(document.content) == 1
  return document.content.chunks[0]


def _get_duration_ms(chunk: content_lib.Chunk, audio_format: str) -> int:
  assert isinstance(chunk.content, bytes)
  return len(
      pydub.AudioSegment.from_file(
          io.BytesIO(chunk.content), format=audio_format
      )
  )


class ChunkByAudioSecondsTest(parameterized.TestCase):

  def test_registered_in_chunker_registry(self):
    config = chunker_registry.ChunkerConfig(
        chunker_name='audio_seconds', kwargs={'seconds_per_chunk': 2.5}
    )
    chunker = chunker_registry.get_chunker(config)
    self.assertIsInstance(chunker, audio_chunking.ChunkByAudioSeconds)
    self.assertEqual(chunker.seconds_per_chunk, 2.5)

  @parameterized.named_parameters(
      ('1s_audio_1s_per_chunk', 1000, 1.0, [1000]),
      ('1s_audio_10s_per_chunk', 1000, 10.0, [1000]),
      ('2s_audio_1s_per_chunk', 2000, 1.0, [1000, 1000]),
      ('2500ms_audio_1s_per_chunk', 2500, 1.0, [1000, 1000, 500]),
      ('5s_audio_2s_per_chunk', 5000, 2.0, [2000, 2000, 1000]),
      ('5s_audio_half_second_per_chunk', 5000, 0.5, [500] * 10),
      ('5s_audio_5s_per_chunk', 5000, 5.0, [5000]),
  )
  def test_chunks_wav_by_seconds(
      self, duration_ms, seconds_per_chunk, expected_chunk_durations_ms
  ):
    document = _make_audio_document(_make_audio(duration_ms))
    chunker = audio_chunking.ChunkByAudioSeconds(
        seconds_per_chunk=seconds_per_chunk
    )

    chunks = list(ot.run(chunker(document)))

    expected_num_chunks = len(expected_chunk_durations_ms)
    with self.subTest('correct_number_of_chunks'):
      self.assertLen(chunks, expected_num_chunks)

    for i, chunk in enumerate(chunks):
      audio_chunk = _get_audio_chunk(chunk)
      with self.subTest(f'chunk_{i+1}_is_wav'):
        self.assertEqual(audio_chunk.content_type, 'audio/wav')

      with self.subTest(f'chunk_{i+1}_has_correct_duration'):
        self.assertEqual(
            _get_duration_ms(audio_chunk, 'wav'),
            expected_chunk_durations_ms[i],
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
          _get_duration_ms(_get_audio_chunk(document), 'wav'), duration_ms
      )

  def test_chunks_mp3_keeps_format(self):
    document = _make_audio_document(
        _make_audio(3000, audio_format='mp3'), content_type='audio/mpeg'
    )
    chunker = audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=1.0)

    chunks = list(ot.run(chunker(document)))

    self.assertLen(chunks, 3)
    for chunk in chunks:
      audio_chunk = _get_audio_chunk(chunk)
      self.assertEqual(audio_chunk.content_type, 'audio/mpeg')
      # MP3 encoding pads the audio slightly, so the duration is approximate.
      self.assertAlmostEqual(
          _get_duration_ms(audio_chunk, 'mp3'), 1000, delta=100
      )

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  def test_non_positive_seconds_per_chunk_raises(self, seconds_per_chunk):
    chunker = audio_chunking.ChunkByAudioSeconds(
        seconds_per_chunk=seconds_per_chunk
    )
    with self.assertRaisesRegex(
        ValueError, 'seconds_per_chunk must be positive'
    ):
      ot.run(chunker(_make_audio_document(_make_audio(1000))))

  @parameterized.named_parameters(
      ('string_content', 'some text'),
      (
          'non_audio_chunk',
          content_lib.ChunkList(
              [content_lib.Chunk(b'img', content_type='image/jpeg')]
          ),
      ),
      (
          'multiple_chunks',
          content_lib.ChunkList([
              content_lib.Chunk(_make_audio(1000), content_type='audio/wav'),
              content_lib.Chunk('some text'),
          ]),
      ),
  )
  def test_non_audio_document_raises(self, content):
    document = retrieval_data_structures.Document(doc_id='doc', content=content)
    chunker = audio_chunking.ChunkByAudioSeconds(seconds_per_chunk=1.0)
    with self.assertRaisesRegex(ValueError, 'single audio chunk'):
      ot.run(chunker(document))


if __name__ == '__main__':
  absltest.main()
