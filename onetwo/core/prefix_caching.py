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

"""Server-side prefix caching for OneTwo.

Server-side prefix caching (also known as context caching) is a feature
supported by certain LLM providers (like Gemini) that allows caching a prefix of
a prompt on the provider's servers. Subsequent calls that share the exact same
prefix can reuse the cached content, significantly reducing latency and token
costs.

This is particularly useful for scenarios with:
- Large system instructions.
- Long context documents (e.g., books, codebases, databases) injected into
  prompts.
- Many few-shot examples that remain fixed across queries.

OneTwo provides the `CachedChunkList` class to represent flat prompts (Sequence
of Chunks) with caching.

To keep the design clean, predictable, and to avoid complex cache-name tracking,
OneTwo enforces strict rules on how prefix caching can be structured:
- **Outermost Level Only**: Caching metadata must be defined at the very
  outermost level of the prompt.
- **No Nesting**: You cannot nest `CachedChunkList` inside a `Message.content`,
  nor can you nest `CachedChunkList` inside other sequences.
- **Contiguous Prefix**: Caching always applies to a contiguous prefix of the
  prompt starting from the very beginning.

We provide the `validate_cached_prompt` function to enforce these rules.
"""

from __future__ import annotations

from collections.abc import Sequence

from onetwo.builtins import llm
from onetwo.core import content as content_lib


class CachedChunkList(content_lib.ChunkList):
  """A ChunkList with server-side prefix caching metadata.

  This class represents a flat prompt (sequence of Chunks) where a prefix
  containing the first `cache_chunk_count` chunks is intended to be cached
  on the server.

  Example:
    ```python
    prompt = CachedChunkList(
        chunks=[
            Chunk("System instruction..."),
            Chunk("Large context..."),
            Chunk("Query..."),
        ],
        cache_chunk_count=2,  # Cache system instruction + large context
        cache_ttl="300s",
    )
    ```

  Attributes:
    cache_chunk_count: The number of chunks (from the beginning of the list) to
      be cached. Must be between 1 and the total number of chunks.
    cache_name: The resource name of the cache returned by the server (e.g.,
      'cachedContents/xyz') or None if the cache has not been created yet. The
      backend will automatically populate this field.
    cache_ttl: Optional TTL (Time-to-Live) string for the cache (e.g., "300s").
      If None, the server's default TTL will be used.
  """

  def __init__(
      self,
      chunks: (
          Sequence[content_lib.Chunk | content_lib.ContentType] | None
      ) = None,
      *,
      cache_chunk_count: int,
      cache_name: str | None = None,
      cache_ttl: str | None = None,
  ):
    super().__init__(chunks)
    self.cache_chunk_count = cache_chunk_count
    self.cache_name = cache_name
    self.cache_ttl = cache_ttl

    if self.cache_chunk_count < 1 or self.cache_chunk_count > len(self.chunks):
      raise ValueError(
          f'Invalid cache_chunk_count: {self.cache_chunk_count}. Must be'
          f' between 1 and the total number of chunks ({len(self.chunks)}).'
      )

  def __eq__(self, other: object) -> bool:
    if type(self) is not type(other):
      return False
    return (
        super().__eq__(other)
        and self.cache_chunk_count == other.cache_chunk_count
        and self.cache_name == other.cache_name
        and self.cache_ttl == other.cache_ttl
    )

  def __repr__(self) -> str:
    """Overridden from base class (ChunkList)."""
    chunks_str = '[' + ', '.join([repr(c) for c in self.chunks]) + ']'
    return (
        f'CachedChunkList(chunks={chunks_str}, '
        f'cache_chunk_count={self.cache_chunk_count!r}, '
        f'cache_name={self.cache_name!r}, '
        f'cache_ttl={self.cache_ttl!r})'
    )

  def __add__(
      self,
      other: (
          content_lib.ContentType | content_lib.Chunk | content_lib.ChunkList
      ),
  ) -> CachedChunkList:
    """Overridden from base class (ChunkList)."""
    combined: content_lib.ChunkList = super().__add__(other)
    return CachedChunkList(
        chunks=combined.chunks,
        cache_chunk_count=self.cache_chunk_count,
        cache_name=self.cache_name,
        cache_ttl=self.cache_ttl,
    )

  def __radd__(
      self, other: content_lib.ContentType | content_lib.Chunk
  ) -> content_lib.ChunkList:
    """Overridden from base class (ChunkList)."""
    raise ValueError(
        f'Cannot prepend {type(other).__name__} to CachedChunkList. '
        'Caching metadata is only valid for a contiguous prefix starting '
        'from the very beginning of the prompt.'
    )

  def __iadd__(
      self,
      other: (
          content_lib.ContentType | content_lib.Chunk | content_lib.ChunkList
      ),
  ) -> CachedChunkList:
    """Overridden from base class (ChunkList)."""
    _ = super().__iadd__(other)
    return self

  def lstrip(
      self, chars: str | None = None, /
  ) -> content_lib.ChunkList:
    """Overridden from base class (ChunkList)."""
    raise ValueError(
        'Cannot lstrip CachedChunkList. Modifying the beginning of the prompt '
        'invalidates the prefix cache. Perform the strip operation before '
        'creating the CachedChunkList, or convert it to a plain ChunkList '
        'first.'
    )

  def strip(
      self, chars: str | None = None, /
  ) -> content_lib.ChunkList:
    """Overridden from base class (ChunkList)."""
    raise ValueError(
        'Cannot strip CachedChunkList. Modifying the beginning of the prompt '
        'invalidates the prefix cache. Perform the strip operation before '
        'creating the CachedChunkList, or convert it to a plain ChunkList '
        'first.'
    )

  def rstrip(
      self, chars: str | None = None, /
  ) -> content_lib.ChunkList:
    """Overridden from base class (ChunkList)."""
    raise ValueError(
        'Cannot rstrip CachedChunkList. Stripping a cached prompt can silently '
        'alter the prompt length and invalidate the intended '
        'cache_chunk_count. Perform the strip operation before creating '
        'the CachedChunkList, or convert it to a plain ChunkList first.'
    )


def validate_cached_prompt(prompt: llm.Prompt) -> None:
  """Validates that caching classes are used correctly in the prompt.

  This function enforces OneTwo's outermost-only caching design rules:
  1. `CachedChunkList` is only allowed at the outermost level.
  2. It cannot be nested inside a `Message` content or inside other sequences.

  Args:
    prompt: The prompt to validate.

  Raises:
    ValueError: If CachedChunkList is used in a nested/invalid structure.
  """
  if isinstance(prompt, CachedChunkList):
    return

  if isinstance(prompt, str) or isinstance(prompt, content_lib.ChunkList):
    return

  if isinstance(prompt, Sequence):
    for item in prompt:
      if isinstance(item, CachedChunkList):
        raise ValueError(
            f'{type(item).__name__} can only be used as the outermost prompt, '
            'not nested inside a sequence.'
        )
      if isinstance(item, content_lib.Message):
        if isinstance(item.content, CachedChunkList):
          raise ValueError(
              f'{type(item.content).__name__} cannot be used as the content of'
              ' a Message. Caching must be defined at the outermost prompt'
              ' level.'
          )
