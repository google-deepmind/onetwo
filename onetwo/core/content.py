# Copyright 2024 DeepMind Technologies Limited.
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

"""Definitions of built-in functions for the OneTwo Language Models API.

These built-in functions are mostly interfaces that need to be implemented.
This is done by calling their `configure` method.

Important: None is treated as a value that can be overridden by the default
parameter values that have been set with `configure`. So calling
generate('...', temperature=None) will use the default temperature (if one
was provided), and would fail otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import enum
from typing import Final, NamedTuple, TypeAlias, Union, cast
import immutabledict
import PIL.Image

ContentType: TypeAlias = Union[str, bytes, PIL.Image.Image]

_CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE: Final[Mapping[str, list[str]]] = (
    immutabledict.immutabledict({
        'str': ['str',],
        'bytes': ['bytes', 'image/'],
        'pil_image': ['image/'],
    })
)


class PredefinedRole(enum.Enum):
  """Predefined roles for a message.

  We define a number of specific roles that may be used in specific ways by the
  formatters. The string value corresponding to these roles is not directly used
  in the prompt, but instead the formatters use these roles as keys in the
  `role_map` property. See the builtins/formatting.py module for more details.
  """
  MODEL = 'model'
  USER = 'user'
  SYSTEM = 'system'
  CONTEXT = 'context'


@dataclasses.dataclass
class Chunk:
  """Dataclass to represent a chunk in a multimodal prompt.

  If the content_type is not specified, it will be inferred.
  If it is specified, it should correspond to the type of the content.
  For example, if content_type is 'image/jpeg', the content should be bytes.
  See _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE for accepted types.
  """
  content: ContentType
  content_type: str = dataclasses.field(default_factory=str)

  def __post_init__(self):
    # Check that the content is of one of the accepted types and set the
    # content_type if not set.
    match self.content:
      case str():
        if not self.content_type:
          self.content_type = 'str'
        python_type = 'str'
      case bytes():
        if not self.content_type:
          self.content_type = 'bytes'
        python_type = 'bytes'
      case PIL.Image.Image():
        if not self.content_type:
          self.content_type = 'image/jpeg'
        python_type = 'pil_image'
      case _:
        raise ValueError(
            f'Creating a Chunk with content type {type(self.content)} which'
            f' does not match supported types: {ContentType}'
        )
    # Check that the content type and the provided content_type argument are
    # compatible.
    prefixes = _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE[python_type]
    if not any(self.content_type.startswith(prefix) for prefix in prefixes):
      raise ValueError(
          f'Creating a Chunk with content of type {python_type} but'
          f' content_type is set to {self.content_type} which is not'
          f' compatible (accepted prefixes are {prefixes}).'
      )

  def __bool__(self):
    return self.__nonzero__()

  def __nonzero__(self):
    return bool(self.content)

  def __str__(self) -> str:
    # This function is lossless on string and lossy on byte content.
    if self.content_type == 'str':
      return cast(str, self.content)
    else:
      return f'<{self.content_type}>'

  def lstrip(self, chars: str | None = None, /) -> Chunk:
    """Apply `lstrip` to the chunk if it is a string.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `lstrip()`.

    Returns:
      Copy of Chunk, where `lstrip` is applied to its content if it is a string.
    """
    updated_content = self.content
    if self.content_type == 'str':
      updated_content = cast(str, self.content).lstrip(chars)
    return Chunk(content=updated_content, content_type=self.content_type)

  def rstrip(self, chars: str | None = None, /) -> Chunk:
    """Apply `rstrip` to the chunk if it is a string.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `rstrip()`.

    Returns:
      Copy of Chunk, where `rstrip` is applied to its content if it is a string.
    """
    updated_content = self.content
    if self.content_type == 'str':
      updated_content = cast(str, self.content).rstrip(chars)
    return Chunk(content=updated_content, content_type=self.content_type)

  def startswith(
      self,
      prefix: str,
      start: int = 0,
      end: int | None = None,
      /,
  ) -> bool:
    """Return `startswith` applied to the string representation of self.

    Args:
      prefix: Prefix string to match.
      start: Where to start matching the prefix.
      end: Where to end matching the prefix.

    Returns:
      True if the string representation of the object matches with the prefix.
    """
    rendered = str(self)
    if end is None:
      end = len(rendered)
    return rendered.startswith(prefix, start, end)


@dataclasses.dataclass
class ChunkList:
  """Dataclass to represent a list of chunks in a multimodal prompt."""
  chunks: list[Chunk] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if not isinstance(self.chunks, list):
      raise ValueError(
          f'Creating a ChunkList with content type {type(self.chunks)} which'
          f' does not match expected type "list".'
      )
    # Convenience best-effort casting of the elements as Chunks.
    for i, chunk in enumerate(self.chunks):
      if not isinstance(chunk, Chunk):
        self.chunks[i] = Chunk(chunk)

  def __bool__(self):
    return self.__nonzero__()

  def __nonzero__(self):
    return any(bool(chunk) for chunk in self.chunks)

  def __len__(self) -> int:
    return len(self.chunks)

  def __getitem__(self, index: int | slice, /) -> Chunk | ChunkList:
    if isinstance(index, slice):
      return ChunkList(self.chunks[index])
    else:
      return self.chunks[index]

  def __iter__(self) -> Iterator[Chunk]:
    return iter(self.chunks)

  def __str__(self) -> str:  # pylint: disable=invalid-str-returned
    # This function is lossless on string and lossy on byte chunks.
    return ''.join([str(chunk) for chunk in self.chunks])

  def __repr__(self) -> str:  # pylint: disable=invalid-str-returned
    # For debug purposes.
    return (
        'ChunkList('
        + ', '.join([f"'{str(chunk)}'" for chunk in self.chunks])
        + ')'
    )

  def __iadd__(self, other: ContentType | Chunk | ChunkList) -> ChunkList:
    match other:
      case Chunk():
        self.chunks.append(other)
      case ChunkList():
        self.chunks += other.chunks
      case _:
        self.chunks.append(Chunk(other))
    return self

  def __add__(self, other: ContentType | Chunk | ChunkList) -> ChunkList:
    match other:
      case Chunk():
        return ChunkList(self.chunks + [other])
      case ChunkList():
        return ChunkList(self.chunks + other.chunks)
      case _:
        return ChunkList(self.chunks + [Chunk(other)])

  def __radd__(self, other: ContentType | Chunk) -> ChunkList:
    match other:
      case Chunk():
        return ChunkList([other] + self.chunks)
      case ChunkList():
        return ChunkList(other.chunks + self.chunks)
      case _:
        return ChunkList([Chunk(other)] + self.chunks)

  def lstrip(self, chars: str | None = None, /) -> ChunkList:
    """Apply `lstrip` to the first chunk of the list.

    TODO. Should we propagate
      lstrip to the first nonempty chunk? Refer to subtest called
      `chunk_list_lstrip_does_not_skip_empty_chunks`.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `lstrip()`.

    Returns:
      Copy of ChunkList, where `lstrip` is applied to its first chunk.
    """
    if self.chunks:
      return ChunkList([self.chunks[0].lstrip(chars)] + self.chunks[1:])
    return ChunkList()

  def rstrip(self, chars: str | None = None, /) -> ChunkList:
    """Apply `rstrip` to the last chunk of the list.

    TODO. Should we propagate
      rstrip to the last nonempty chunk?

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `rstrip()`.

    Returns:
      Copy of ChunkList, where `rstrip` is applied to its last chunk.
    """
    if self.chunks:
      return ChunkList(self.chunks[:-1] + [self.chunks[-1].rstrip(chars)])
    return ChunkList()

  def startswith(
      self,
      prefix: str,
      start: int = 0,
      end: int | None = None,
      /,
  ) -> bool:
    """Return `startswith` applied to the string representation of self.

    Args:
      prefix: Prefix string to match.
      start: Where to start matching the prefix.
      end: Where to end matching the prefix.

    Returns:
      True if the string representation of the object matches with the prefix.
    """
    rendered = str(self)
    if end is None:
      end = len(rendered)
    return rendered.startswith(prefix, start, end)

  def to_simple_string(self) -> str:
    """Converts the chunk list to a string whithout the multimodal elements."""
    return ''.join(
        [str(chunk) for chunk in self.chunks if chunk.content_type == 'str']
    )


class Message(NamedTuple):
  """NamedTuple to represent a message in a chat conversation."""

  role: str | PredefinedRole
  content: str | ChunkList
