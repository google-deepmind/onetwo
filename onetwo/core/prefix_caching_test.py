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

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import content as content_lib
from onetwo.core import prefix_caching

_Chunk = content_lib.Chunk
_Message = content_lib.Message


class PrefixCachingTest(parameterized.TestCase):

  def test_cached_chunk_list_creation_succeeds(self):
    chunks = [_Chunk('a'), _Chunk('b')]
    cached = prefix_caching.CachedChunkList(chunks, cache_chunk_count=1)
    self.assertEqual(cached.chunks, chunks)
    self.assertEqual(cached.cache_chunk_count, 1)
    self.assertIsNone(cached.cache_name)
    self.assertIsNone(cached.cache_ttl)

  @parameterized.named_parameters(
      dict(
          testcase_name='zero_count',
          cache_chunk_count=0,
          expected_error=(
              'Invalid cache_chunk_count: 0. Must be between 1 and the total'
              ' number of chunks'
          ),
      ),
      dict(
          testcase_name='too_large_count',
          cache_chunk_count=3,
          expected_error=(
              'Invalid cache_chunk_count: 3. Must be between 1 and the total'
              ' number of chunks'
          ),
      ),
  )
  def test_cached_chunk_list_creation_raises_error(
      self, cache_chunk_count, expected_error
  ):
    chunks = [_Chunk('a'), _Chunk('b')]
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = prefix_caching.CachedChunkList(
          chunks=chunks, cache_chunk_count=cache_chunk_count
      )

  def test_cached_chunk_list_rstrip_raises_error(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a '), _Chunk('b ')],
        cache_chunk_count=2,
        cache_name='cachedContents/xyz',
    )
    with self.assertRaisesRegex(
        ValueError, 'Cannot rstrip CachedChunkList'
    ):
      _ = cached.rstrip()

  @parameterized.named_parameters(
      dict(
          testcase_name='string_prompt',
          prompt='hello',
      ),
      dict(
          testcase_name='flat_chunk_list',
          prompt=content_lib.ChunkList([_Chunk('hello')]),
      ),
      dict(
          testcase_name='outermost_cached_chunk_list',
          prompt=prefix_caching.CachedChunkList(
              chunks=[_Chunk('hello')], cache_chunk_count=1
          ),
      ),
      dict(
          testcase_name='sequence_of_messages_no_cache',
          prompt=[_Message(role='user', content='hello')],
      ),
      dict(
          testcase_name='sequence_of_chunks_no_cache',
          prompt=[_Chunk('hello')],
      ),
  )
  def test_validate_cached_prompt_succeeds(self, prompt):
    prefix_caching.validate_cached_prompt(prompt)

  def test_validate_cached_prompt_nested_in_sequence_of_chunks_raises_error(
      self,
  ):
    cached_chunks = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')], cache_chunk_count=1
    )
    prompt = [_Chunk('prefix'), cached_chunks]

    with self.assertRaisesRegex(
        ValueError,
        'CachedChunkList can only be used as the outermost prompt, not nested'
        ' inside a sequence',
    ):
      prefix_caching.validate_cached_prompt(prompt)

  def test_validate_cached_prompt_nested_in_message_content_raises_error(self):
    cached_chunks = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')], cache_chunk_count=1
    )
    msg = _Message(role='user', content=cached_chunks)
    prompt = [msg]

    with self.assertRaisesRegex(
        ValueError,
        'CachedChunkList cannot be used as the content of a Message',
    ):
      prefix_caching.validate_cached_prompt(prompt)

  def test_cached_chunk_list_equality(self):
    chunks = [_Chunk('a'), _Chunk('b')]
    plain_list = content_lib.ChunkList(chunks)

    cached1 = prefix_caching.CachedChunkList(
        chunks,
        cache_chunk_count=1,
        cache_name='cache1',
        cache_ttl='300s',
    )
    cached2 = prefix_caching.CachedChunkList(
        chunks,
        cache_chunk_count=1,
        cache_name='cache1',
        cache_ttl='300s',
    )
    cached3 = prefix_caching.CachedChunkList(
        chunks,
        cache_chunk_count=2,  # Different
        cache_name='cache1',
        cache_ttl='300s',
    )

    self.assertEqual(cached1, cached2)
    self.assertNotEqual(cached1, cached3)

    # Verify symmetry of inequality (cross-type comparison must be False both
    # ways).
    self.assertNotEqual(plain_list, cached1)
    self.assertNotEqual(cached1, plain_list)

  def test_cached_chunk_list_repr(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')],
        cache_chunk_count=1,
        cache_name='cache1',
        cache_ttl='300s',
    )
    expected_repr = (
        "CachedChunkList(chunks=[Chunk(a, role=None)], "
        "cache_chunk_count=1, cache_name='cache1', cache_ttl='300s')"
    )
    self.assertEqual(repr(cached), expected_repr)

  def test_cached_chunk_list_add_preserves_metadata(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')],
        cache_chunk_count=1,
        cache_name='cache1',
        cache_ttl='300s',
    )
    # Add a string
    combined = cached + 'b'
    self.assertIsInstance(combined, prefix_caching.CachedChunkList)
    self.assertEqual(combined.chunks, [_Chunk('a'), _Chunk('b')])
    self.assertEqual(combined.cache_chunk_count, 1)
    self.assertEqual(combined.cache_name, 'cache1')

    # Add a Chunk
    combined2 = cached + _Chunk('b')
    self.assertEqual(combined2.chunks, [_Chunk('a'), _Chunk('b')])
    self.assertEqual(combined2.cache_chunk_count, 1)

    # Add a ChunkList
    combined3 = cached + content_lib.ChunkList([_Chunk('b')])
    self.assertEqual(combined3.chunks, [_Chunk('a'), _Chunk('b')])
    self.assertEqual(combined3.cache_chunk_count, 1)

  def test_cached_chunk_list_add_ignores_other_metadata(self):
    cached1 = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')],
        cache_chunk_count=1,
        cache_name='cache1',
    )
    cached2 = prefix_caching.CachedChunkList(
        chunks=[_Chunk('b')],
        cache_chunk_count=1,
        cache_name='cache2',
    )
    combined = cached1 + cached2
    self.assertIsInstance(combined, prefix_caching.CachedChunkList)
    self.assertEqual(combined.chunks, [_Chunk('a'), _Chunk('b')])
    self.assertEqual(combined.cache_chunk_count, 1)  # From cached1
    self.assertEqual(combined.cache_name, 'cache1')  # From cached1

  def test_cached_chunk_list_radd_raises_error(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('b')],
        cache_chunk_count=1,
        cache_name='cache1',
    )
    # Prepend a string should raise ValueError
    with self.assertRaisesRegex(
        ValueError, 'Cannot prepend str to CachedChunkList'
    ):
      _ = 'a' + cached

  def test_cached_chunk_list_lstrip_raises_error(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('  a')],
        cache_chunk_count=1,
        cache_name='cache1',
    )
    with self.assertRaisesRegex(
        ValueError, 'Cannot lstrip CachedChunkList'
    ):
      _ = cached.lstrip()

  def test_cached_chunk_list_strip_raises_error(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('  a  ')],
        cache_chunk_count=1,
        cache_name='cache1',
    )
    with self.assertRaisesRegex(
        ValueError, 'Cannot strip CachedChunkList'
    ):
      _ = cached.strip()

  def test_cached_chunk_list_iadd_preserves_metadata(self):
    cached = prefix_caching.CachedChunkList(
        chunks=[_Chunk('a')],
        cache_chunk_count=1,
        cache_name='cache1',
    )
    cached += 'b'
    self.assertIsInstance(cached, prefix_caching.CachedChunkList)
    self.assertEqual(cached.chunks, [_Chunk('a'), _Chunk('b')])
    self.assertEqual(cached.cache_chunk_count, 1)
    self.assertEqual(cached.cache_name, 'cache1')


if __name__ == '__main__':
  absltest.main()
