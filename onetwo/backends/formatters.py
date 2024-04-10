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

"""Implementation of various formatters for different models."""

from collections.abc import Sequence
from typing import Final, TypeAlias

import aenum
from onetwo.builtins import formatting
from onetwo.core import content as content_lib


_PredefinedRole = content_lib.PredefinedRole
_Message: TypeAlias = content_lib.Message
_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList
_FormatterName = formatting.FormatterName
_Formatter = formatting.Formatter


_START_OF_TURN: Final[str] = '<start_of_turn>'
_END_OF_TURN: Final[str] = '<end_of_turn>'


class GemmaFormatter(_Formatter):
  """Gemma formatter for instruction tuned models."""

  @property
  def role_map(self) -> dict[str | _PredefinedRole, str]:
    """See parent class."""
    return {
        _PredefinedRole.USER: 'user',
        _PredefinedRole.MODEL: 'model'
    }

  def is_already_formatted(self, content: Sequence[_Message]) -> bool:
    """See parent class."""
    if not content:
      return False
    return any(
        _START_OF_TURN in str(message.content)
        for message in content
    )

  def extra_stop_sequences(self) -> list[str]:
    """See parent class."""
    return [_START_OF_TURN]

  def _format(
      self,
      content: Sequence[_Message],
  ) -> _ChunkList:
    """See parent class."""
    res = []
    for i, message in enumerate(content):
      role_name = self.role_map[message.role]
      if i == len(content) - 1 and message.role == _PredefinedRole.MODEL:
        if (str(message.content)):
          res.append(f'{_START_OF_TURN}{role_name}\n{message.content}')
        else:
          res.append(f'{_START_OF_TURN}{role_name}')
      else:
        res.append(
            f'{_START_OF_TURN}{role_name}\n{message.content}{_END_OF_TURN}'
        )
    res = '\n'.join(res)
    return _ChunkList([_Chunk(res)])


aenum.extend_enum(formatting.FormatterName, 'GEMMA', 'gemma')
formatting.FORMATTER_CLASS_BY_NAME[_FormatterName.GEMMA] = GemmaFormatter
