# Copyright 2025 DeepMind Technologies Limited.
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

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import enum
import logging
from typing import Any, Final, TypeAlias, Union, cast

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
        'bytes': [
            'bytes',
            'image/',
            'video/',
            'audio/',
            'application/',
            'vision/',
        ],
        'pil_image': ['image/'],
    })
)

# Mapping from content type prefix to the Python types that are accepted.
# For example, if the content type prefix is 'image/', the Python type can be
# either 'pil_image' or 'bytes'.
_PYTHON_TYPE_BY_CONTENT_TYPE_PREFIX: Final[Mapping[str, list[str]]] = (
    immutabledict.immutabledict({
        'str': ['str'],
        'ctrl': ['str'],
        'image/': ['pil_image', 'bytes'],
        'bytes': ['bytes'],
        'video/': ['bytes'],
        'audio/': ['bytes'],
        'application/': ['bytes'],
        'vision/': ['bytes'],
    })
)


def _get_python_and_content_type_from_content(
    content: ContentType,
) -> tuple[str, str]:
  """Returns the python_type and the default content_type based on the content.

  The python type shows the type of the content.
  The default content_type is used when the content_type is not specified.

  Args:
    content: The content to get the python type and the default content_type
      for.

  Returns:
    A tuple of (python_type, default_content_type).

  Raises:
    ValueError: If the content is not one of the accepted types (str, bytes,
    PIL.Image.Image).
  """
  match content:
    case str():
      return ('str', 'str')
    case bytes():
      return 'bytes', 'bytes'
    case PIL.Image.Image():
      image: PIL.Image.Image = content
      fmt = image.format  # e.g., 'PNG', 'JPEG', None
      if fmt:
        # Use the actual format if available.
        content_type = f'image/{fmt.lower()}'
      else:
        # Default for images created in memory (format=None).
        content_type = 'image/jpeg'
      return 'pil_image', content_type
    case _:
      raise ValueError(
          f'Creating a Chunk with content of type {type(content)} which is not'
          ' one of the accepted types (str, bytes, PIL.Image.Image).'
      )


def _get_content_type_prefix(content_type: str) -> str:
  """Returns the content type prefix for the given content_type.

  Args:
    content_type: The content_type to get the prefix for.

  Returns:
    The content type prefix for the given content_type. If the content_type is
    not one of the known types, returns 'Unknown'.
    Ex: 'image/jpeg' -> 'image/'
        'text/csv' -> 'Unknown'
  """
  for prefix in _PYTHON_TYPE_BY_CONTENT_TYPE_PREFIX.keys():
    # If the content_type is one of the known types, return the prefix.
    if content_type.startswith(prefix):
      return prefix
  # If the content_type is not one of the known types, return Unknown.
  return 'Unknown'


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
    metadata: Metadata information to be supplied to the downstream request
      object.
  """

  content: ContentType
  content_type: str = dataclasses.field(default_factory=str)
  role: RoleType = None
  metadata: Any = None

  def __post_init__(self):
    # Get the python type and the default content_type based on the content.
    python_type, default_content_type = (
        _get_python_and_content_type_from_content(self.content)
    )

    # If the content_type is not set, set it to the default content_type.
    if not self.content_type:
      self.content_type = default_content_type
      return

    content_type_prefix = _get_content_type_prefix(self.content_type)

    # If the content_type is unknown, log a warning.
    if content_type_prefix == 'Unknown':
      logging.warning(
          'Creating a Chunk with unknown content_type: %s. This might cause'
          ' errors if the type of the content is not compatible with the'
          ' provided content_type.',
          self.content_type,
      )
      return
    else:
      if (
          python_type
          not in _PYTHON_TYPE_BY_CONTENT_TYPE_PREFIX[content_type_prefix]
      ):
        raise ValueError(
            f'Creating a Chunk with content of type {python_type} but'
            f' content_type is set to {self.content_type} which is not'
            ' compatible (accepted prefixes are'
            f' {_PYTHON_TYPE_BY_CONTENT_TYPE_PREFIX[content_type_prefix]}).'
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

  def __repr__(self) -> str:
    # For debug purposes. This function is lossless on string and lossy on any
    # other content. To idea is to have a concise representation of the chunks
    # that can be used for debugging.
    if self.content_type in _CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE['str']:
      return f'Chunk({self.content}, role={self.role})'
    else:
      return f'Chunk(<{self.content_type}>, role={self.role})'

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

  def strip(self, chars: str | None = None, /) -> Chunk:
    """Apply `strip` to the chunk if it is a string.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `strip()`.

    Returns:
      Copy of Chunk, where `strip` is applied to its content if it is a string.
    """
    updated_content = self.content
    if self.content_type == 'str':
      # Note that we do not apply strip to control tokens (content_type
      # 'ctrl').
      updated_content = cast(str, self.content).strip(chars)
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

  def __init__(self, chunks: Sequence[Chunk | ContentType] | None = None):
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
    if isinstance(other, ChunkList):
      return self.chunks == other.chunks
    else:
      return False

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
        + ', '.join([f'{repr(chunk)}' for chunk in self.chunks])
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

  def strip(self, chars: str | None = None, /) -> ChunkList:
    """Apply both `lstrip` and `rstrip`.

    Args:
      chars: String specifying the set of characters to be removed. When chars
        is None (default) corresponds to the default of builtin `strip()`.

    Returns:
      Copy of ChunkList, where `lstrip` is applied to its first non-empty
      chunk and `rstrip` is applied to its last non-empty chunk.
      Leading and trailing empty chunks are removed.
    """
    return self.lstrip(chars).rstrip(chars)

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


@dataclasses.dataclass
class Message:
  """A message in a chat conversation.

  Attributes:
    role: The role of the message.
    content: The content of the message.
  """

  role: str | PredefinedRole
  content: str | ChunkList

  @classmethod
  def create_normalized(
      cls,
      role: str | PredefinedRole,
      content: str | ChunkList | Sequence[Chunk],
  ) -> 'Message':
    """Returns a message with the given role and normalized content."""
    if isinstance(content, str):
      normalized_content = content
    else:
      if isinstance(content, ChunkList):
        chunks = content.chunks
      else:
        chunks = content

      if (
          len(chunks) == 1
          and chunks[0].content_type == 'str'
          and not chunks[0].metadata
      ):
        # In the case of a single string with no metadata, the same information
        # could be represented either a standalone string or as a ChunkList with
        # a single chunk. We normalize to the simpler representation.
        normalized_content = chunks[0].content
      else:
        normalized_content = ChunkList(chunks)

    return cls(role=role, content=normalized_content)

  def get_chunk_list(self) -> ChunkList:
    """Returns the message contents in the form of a ChunkList."""
    if isinstance(self.content, str):
      return ChunkList([self.content])
    else:
      return self.content
