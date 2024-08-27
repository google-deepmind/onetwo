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

"""Basic types used in the builtins.

Here we define `Chunk` and `ChunkList`, which are the basic types used in the
builtins to describe the content of a (multimodal) prompt. `Message` is a
named tuple to represent a message in a chat conversation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import enum
from typing import Final, NamedTuple, TypeAlias, Union, cast

import immutabledict
import PIL.Image

ContentType: TypeAlias = Union[str, bytes, PIL.Image.Image]

# Mapping from Python type to the content type prefixes that are accepted.
# For example, if the content is a string, the content_type should start with
# 'str' or 'ctrl'.
# Special cases:
# - 'ctrl' is used to indicate that the content is a string that should be
# treated as a special control token (this is used for example if the
# underlying model has been trained or fine-tuned with special control tokens
# and its API expects these tokens to be sent not as strings).
# - 'image/' is used to indicate that the content is the bytes content of an
# image.
_CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE: Final[Mapping[str, list[str]]] = (
    immutabledict.immutabledict({
        'str': ['str', 'ctrl'],
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


RoleType: TypeAlias = Union[str, PredefinedRole, None]


@dataclasses.dataclass
class Chunk:
  """Dataclass to represent a chunk in a multimodal prompt.

  Attributes:
    content: The content of the chunk.
    content_type: The type of the content. If not specified, it will be
      inferred. If it is specified, it should correspond to the type of the
      content. For example, if content_type is 'image/jpeg', the content should
      be bytes. See _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE for accepted types.
    role: The role to assign to the chunk when grouping chunks into chat
      messages. If not specified, it will be treated as USER. Ignored when the
      chunk is used in a non-chat operation.
  """

  content: ContentType
  content_type: str = dataclasses.field(default_factory=str)
  role: RoleType = None

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
    if self.content_type in _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE['str']:
      return cast(str, self.content)
    else:
      return f'<{self.content_type}>'

  def is_empty(self) -> bool:
    """Returns True if the chunk is empty."""
    match self.content:
      case str() | bytes():
        if self.content:
          return False
      case PIL.Image.Image():
        if cast(PIL.Image.Image, self.content).size != (0, 0):
          return False
    return True

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
      # Note that we do not apply lstrip to control tokens (content_type
      # 'ctrl').
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
      # Note that we do not apply rstrip to control tokens (content_type
      # 'ctrl').
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
    return rendered.startswith(prefix, start, end)

  def endswith(
      self,
      suffix: str,
      start: int = 0,
      end: int | None = None,
      /,
  ) -> bool:
    """Return `endswith` applied to the string representation of self.

    Args:
      suffix: Suffix string to match.
      start: Where to start matching the suffix.
      end: Where to end matching the suffix.

    Returns:
      True if the string representation of the object matches with the suffix.
    """
    rendered = str(self)
    return rendered.endswith(suffix, start, end)


class ChunkList:
  """Class to represent a list of chunks in a multimodal prompt.

  Attributes:
    chunks: List of Chunks.
  """

  def __init__(self, chunks: list[Chunk | ContentType] | None = None):
    if chunks is not None and not isinstance(chunks, list):
      # In case typing did not catch this.
      raise ValueError(
          f'Creating a ChunkList with chunks type {type(chunks)} which'
          ' does not match expected type "list".'
      )
    self.chunks: list[Chunk] = []
    if chunks is None:
      chunks = []
    # Convenience best-effort casting of the elements as Chunks.
    for chunk in chunks:
      if isinstance(chunk, Chunk):
        self.chunks.append(chunk)
      else:
        self.chunks.append(Chunk(chunk))

  def __eq__(self, other):
    return self.chunks == other.chunks

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
    """Remove leading empty chunks, apply `lstrip` to the first non-empty one.

    If the entire first non-empty chunk gets stripped, this implementation does
    not propagate to the next non-empty chunk, i.e.
    ChunkList(['', '', 'abc', 'abcd']).lstrip('abc') results in
    ChunkList(['abcd']).

    TODO
      in case the entire chunk gets stripped.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `lstrip()`.

    Returns:
      Copy of ChunkList, where `lstrip` is applied to its first chunk.
      Leading empty chunks are removed.
    """

    # Find the first non-empty chunk.
    first_nonempty_id = 0
    while first_nonempty_id < len(self.chunks):
      if not self.chunks[first_nonempty_id].is_empty():
        break
      # Empty chunk. Shift right.
      first_nonempty_id += 1

    if first_nonempty_id == len(self.chunks):
      # Chunk list is empty or all chunks are empty.
      return ChunkList()

    # Strip the first non-empty chunk and delete the leading empty chunks.
    result = [self.chunks[first_nonempty_id].lstrip(chars)] + self.chunks[
        first_nonempty_id + 1 :
    ]
    if result[0].is_empty():
      # If we stripped the entire first chunk, remove it.
      del result[0]
    return ChunkList(result)

  def rstrip(self, chars: str | None = None, /) -> ChunkList:
    """Remove trailing empty chunks, apply `rstrip` to the last non-empty one.

    If the entire last non-empty chunk gets stripped, this implementation
    propagates to the previous non-empty chunk, i.e.
    ChunkList(['dabc', 'abc', '', '']).rstrip('abc') results in
    ChunkList(['d']).

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `rstrip()`.

    Returns:
      Copy of ChunkList, where `rstrip` is applied to its last non-empty chunk.
      Trailing empty chunks are removed. If the last non-empty chunk gets
      entirely stripped, the previous non-empty chunk is stripped and so on.
    """
    if not self.chunks:
      return ChunkList()

    last_nonempty_id = len(self.chunks) - 1

    while last_nonempty_id >= 0:
      if not self.chunks[last_nonempty_id].rstrip(chars).is_empty():
        break
      # Last chunk is empty or we stripped it entirely. Shift left.
      last_nonempty_id -= 1

    if last_nonempty_id == -1:
      # We stripped all chunks.
      return ChunkList()

    return ChunkList(
        self.chunks[:last_nonempty_id]
        + [self.chunks[last_nonempty_id].rstrip(chars)]
    )

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
    return rendered.startswith(prefix, start, end)

  def endswith(
      self,
      suffix: str,
      start: int = 0,
      end: int | None = None,
      /,
  ) -> bool:
    """Return `endswith` applied to the string representation of self.

    Args:
      suffix: Suffix string to match.
      start: Where to start matching the suffix.
      end: Where to end matching the suffix.

    Returns:
      True if the string representation of the object matches with the suffix.
    """
    rendered = str(self)
    return rendered.endswith(suffix, start, end)

  def to_simple_string(self) -> str:
    """Converts the chunk list to a string whithout the multimodal elements.

    This includes the strings and the control tokens (content_type 'ctrl').

    Returns:
      A string containing all the string elements of the chunk list.
    """
    return ''.join([
        str(chunk)
        for chunk in self.chunks
        if chunk.content_type in _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE['str']
    ])


class Message(NamedTuple):
  """NamedTuple to represent a message in a chat conversation."""

  role: str | PredefinedRole
  content: str | ChunkList
