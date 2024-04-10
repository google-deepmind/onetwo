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

"""Formatting tools for instruction tuned models."""

import abc
from collections.abc import Mapping, Sequence
import copy
import pprint
from typing import Any, Final, final, TypeAlias

import aenum
from onetwo.core import content as content_lib


_Message: TypeAlias = content_lib.Message
_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


_INSTRUCTION_QUESTION_PREFIX: Final[str] = 'Task:'
_INSTRUCTION_ANSWER_PREFIX: Final[str] = 'Answer:'
_INSTRUCTION_SEPARATOR: Final[str] = '\n'
_MULTITURN_ROLE_PATTERN: Final[str] = '**{}**:'
_MULTITURN_TURN_SEPARATOR: Final[str] = '\n'


@aenum.unique
class FormatterName(aenum.Enum):
  """Possible formatter names.

  This is an extensible Enum which will be extended in other modules to add
  backend-specific formatters.
  """

  NONE = 'none'  # No formatting, just concatenate all messages.
  DEFAULT = 'default'  # Best effort formatting to produce a useful prompt.
  API = 'api'  # Attempts to use the model API to pass messages directly.


class Formatter(metaclass=abc.ABCMeta):
  """Abstract class for formatters.

  A formatter is a class that is able to convert structured content into
  a prompt.
  For example it is used for `instruct` and `chat`.
  Formatting means two things:
  - Adding structure to the sequence of messages, for example adding some
    specification of the roles of the messages and separators between them.
  - Adding model-specific control tokens to the prompt that correspond to how
    the model was trained.

  There are two kinds of calls that will use formatting:
  - The `instruct` method which is typically used to provide a prompt in the
    form of instructions to an instruction-tuned model.
  - The `chat` method which is used to provide a prompt in the form of a
    chat conversation to a model that was trained on multi-turn conversations.

  We use the same formatter for both cases, and the default implementation
  of `instruct` will simply call `chat` with a single turn, and the
  implementation of `chat` may have some specific treatment of the case where
  there is a single turn.
  """

  def __init__(self, kwargs: Mapping[str, Any] | None = None):
    """Initializes the formatter with an optional dict of arguments."""
    if kwargs is None:
      kwargs = {}
    self._kwargs = kwargs

  @property
  def role_map(self) -> dict[str | _PredefinedRole, str]:
    """Returns a mapping from role to string representation.

    This serves two purposes:
    - It specifies which roles are supported by the formatter. Any role that is
      not in this map will be ignored or raise an error if
      `raise_error_if_unsupported_roles=True` in the call to `format`.
    - It allows to specify how the role name is represented in the prompt.
    """
    return {_PredefinedRole.USER: 'user'}

  @abc.abstractmethod
  def is_already_formatted(self, content: Sequence[_Message]) -> bool:
    """Returns whether the content is already formatted."""

  @abc.abstractmethod
  def _format(
      self,
      content: Sequence[_Message],
  ) -> _ChunkList:
    """Converts messages to a chunk list that can be used as a prompt.

    This method (which should be overridden by subclasses) is where the actual
    formatting happens. It is called by the public `format` method which
    performs a few checks and then calls this method.

    Args:
      content: The content to format in the form of a sequence of Messages.

    Returns:
      A ChunkList with the formatted content, ready to be used as a prompt in
      a call to `generate_text` for example.
    """

  def extra_stop_sequences(self) -> list[str]:
    """Returns optional extra stop sequences to use during generation.

    This is used for example to add extra stop sequences that correspond to
    separators between roles. For example if the formatting is enclosing the
    user messages with "**User**:" "**End User**" then we can add "**User**:"
    as a stop sequence so that when the model generates its response it does
    not continue by generating a simulated user message.
    """
    return []

  @final
  def format(
      self,
      content: str | _ChunkList | _Message | Sequence[_Message],
      raise_error_if_already_formatted: bool = True,
      raise_error_if_unsupported_roles: bool = False,
  ) -> _ChunkList:
    """Returns formatted ChunkList."""
    if isinstance(content, str) or isinstance(content, _ChunkList):
      content = [
          _Message(role=_PredefinedRole.USER, content=content)
      ]
    if isinstance(content, _Message):
      # Form a sequence of messages.
      content = [content]
    if not content:
      # Return empty ChunkList if the content is empty.
      return _ChunkList()
    already_formatted = self.is_already_formatted(content)
    if already_formatted:
      if raise_error_if_already_formatted:
        raise ValueError(
            'Content already contains some of the formatting control'
            f' sequences:\n{pprint.pformat(content)}'
        )
      else:
        # Return the content as is (converting message content to chunks).
        return _ChunkList([_Chunk(c.content) for c in content])
    # Remove messages with unsupported roles.
    filtered_content = [
        message for message in content if message.role in self.role_map.keys()
    ]
    if raise_error_if_unsupported_roles and len(filtered_content) != len(
        content
    ):
      raise ValueError(
          'Content contains unsupported roles for this formatter. Supported'
          f' roles: {self.role_map.keys()}.'
          f' Messages:\n{pprint.pformat(content)}'
      )
    return self._format(filtered_content)


class DefaultFormatter(Formatter):
  """Default formatter for instruction tuned models."""

  _DEFAULT_FEWSHOT: Final[list[tuple[str, str]]] = [
      ('Write me a palindrome.', 'Level')
  ]

  @property
  def role_map(self) -> dict[str | _PredefinedRole, str]:
    return {
        _PredefinedRole.USER: 'User',
        _PredefinedRole.MODEL: 'Model',
        _PredefinedRole.SYSTEM: 'System',
        _PredefinedRole.CONTEXT: 'Context',
    }

  def is_already_formatted(self, content: Sequence[_Message]) -> bool:
    concat = str(content)
    matches = [_INSTRUCTION_QUESTION_PREFIX, _INSTRUCTION_ANSWER_PREFIX]
    return any(substr in concat for substr in matches)

  def extra_stop_sequences(self) -> list[str]:
    return [
        f'{_INSTRUCTION_SEPARATOR}{_INSTRUCTION_QUESTION_PREFIX}',
        _MULTITURN_TURN_SEPARATOR
        + _MULTITURN_ROLE_PATTERN.format(self.role_map[_PredefinedRole.USER]),
    ]

  def _format(
      self,
      content: Sequence[_Message],
  ) -> _ChunkList:
    prompt = _ChunkList()
    role_list = [m.role for m in content]
    # We first determine if we are in the `instruct` situation where there is
    # a single turn, with a user message and an assistant prefix.
    if role_list == [_PredefinedRole.USER] or role_list == [
        _PredefinedRole.USER,
        _PredefinedRole.MODEL,
    ]:
      prompt += content[0].content
      assistant_prefix = ''
      if len(content) == 2:
        assistant_prefix = content[1].content

      instruct_prompt = content_lib.ChunkList()
      if self._kwargs.get('use_fewshots', False):
        for prompt, answer in self._DEFAULT_FEWSHOT:
          instruct_prompt += (
              _INSTRUCTION_QUESTION_PREFIX
              + ' '
              + prompt
              + _INSTRUCTION_SEPARATOR
              + _INSTRUCTION_ANSWER_PREFIX
              + ' '
              + answer
              + '\n'
          )
      prompt = prompt.lstrip(' ')
      if prompt:
        instruct_prompt += (
            _INSTRUCTION_QUESTION_PREFIX
            + ' '
            + prompt
            + _INSTRUCTION_SEPARATOR
            + _INSTRUCTION_ANSWER_PREFIX
        )
      else:
        instruct_prompt += (
            _INSTRUCTION_QUESTION_PREFIX
            + _INSTRUCTION_SEPARATOR
            + _INSTRUCTION_ANSWER_PREFIX
        )
      if assistant_prefix is not None:
        assistant_prefix = assistant_prefix.lstrip(' ')
        if assistant_prefix:
          instruct_prompt += ' ' + assistant_prefix
      return instruct_prompt
    else:
      # We are in the `chat` situation where there are possibly multiple turns,
      # or a single turn but with context and/or system instructions.
      messages = copy.copy(list(content))
      if messages[-1].role != _PredefinedRole.MODEL:
        # Include empty assistant message in the end.
        messages.append(_Message(role=_PredefinedRole.MODEL, content=''))
      for msg in messages[:-1]:
        if msg.role == _PredefinedRole.SYSTEM:
          if msg.content:
            prompt += _MULTITURN_ROLE_PATTERN.format(
                self.role_map[_PredefinedRole.SYSTEM]
            ) + (
                f' Actor "{self.role_map[_PredefinedRole.MODEL]}" needs to obey'
                ' the following rules when generating the messages below:\n'
            )
            prompt += msg.content + '\n'
        else:
          prompt += (
              _MULTITURN_ROLE_PATTERN.format(self.role_map[msg.role])
              + f' {msg.content}\n'
          )
      # Last message has assistant role. Don't include end of turn tag.
      last_assistant_content = messages[-1].content
      last_assistant_content = last_assistant_content.lstrip(' ')
      if last_assistant_content:
        prompt += (
            _MULTITURN_ROLE_PATTERN.format(self.role_map[_PredefinedRole.MODEL])
            + f' {last_assistant_content}\n'
        )
      else:
        prompt += _MULTITURN_ROLE_PATTERN.format(
            self.role_map[_PredefinedRole.MODEL]
        )
      return prompt


FORMATTER_CLASS_BY_NAME: dict[FormatterName, type[Formatter]] = {
    FormatterName.DEFAULT: DefaultFormatter,
}
