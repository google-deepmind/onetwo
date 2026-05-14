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

"""A module that mocks the Anthropic library, for testing purposes.

This document explains how to use the messages method:
https://docs.anthropic.com/en/api/messages

This document explains how to use the count tokens method:
https://docs.anthropic.com/en/api/messages-count-tokens
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

_DEFAULT_REPLY: Final[str] = 'Hello'
_DEFAULT_THINKING: Final[str] = 'Let me think about this.'


class TextBlock:
  """Mock for an Anthropic text content block."""

  def __init__(self, text: str):
    self.type = 'text'
    self.text = text


class ThinkingBlock:
  """Mock for an Anthropic thinking content block."""

  def __init__(self, thinking: str):
    self.type = 'thinking'
    self.thinking = thinking


class Usage:
  """Mock for the usage object in a Messages API response."""

  def __init__(self, input_tokens: int, output_tokens: int):
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens


class Message:
  """Mock for the Anthropic Messages API response object."""

  def __init__(
      self,
      content: Sequence[TextBlock],
      role: str = 'assistant',
      stop_reason: str = 'end_turn',
      usage: Usage | None = None,
  ):
    self.id = 'msg_mock_123'
    self.type = 'message'
    self.role = role
    self.content = content
    self.model = 'claude-sonnet-4-20250514'
    self.stop_reason = stop_reason
    self.stop_sequence = None
    self.usage = usage or Usage(input_tokens=10, output_tokens=5)


class TokenCountResponse:
  """Mock for the messages.count_tokens() response."""

  def __init__(self, input_tokens: int):
    self.input_tokens = input_tokens


class MessagesNamespace:
  """Mock for client.messages namespace."""

  def __init__(self, reply: str = _DEFAULT_REPLY):
    self._reply = reply

  def create(
      self,
      *,
      model: str,
      max_tokens: int,
      messages: Sequence[Mapping[str, Any]],
      system: str | Sequence[Mapping[str, Any]] | None = None,
      temperature: float | None = None,
      top_p: float | None = None,
      top_k: int | None = None,
      stop_sequences: Sequence[str] | None = None,
      thinking: Mapping[str, Any] | None = None,
      output_config: Mapping[str, Any] | None = None,
      **kwargs,
  ) -> Message:
    """Mocks client.messages.create for the Messages API."""
    if not messages:
      raise ValueError(
          "Error code: 400 - {'type': 'error', 'error': {'type':"
          " 'invalid_request_error', 'message': 'messages: must not be"
          " empty'}}"
      )
    del (
        model,
        max_tokens,
        messages,
        system,
        temperature,
        top_p,
        top_k,
        stop_sequences,
        output_config,
        kwargs,
    )
    content: list[TextBlock | ThinkingBlock] = []
    # When thinking is enabled, prepend a thinking block.
    if thinking and thinking.get('type') in ('enabled', 'adaptive'):
      content.append(ThinkingBlock(thinking=_DEFAULT_THINKING))
    content.append(TextBlock(text=self._reply))
    return Message(content=content)

  def count_tokens(
      self,
      *,
      model: str,
      messages: Sequence[Mapping[str, Any]],
      system: str | Sequence[Mapping[str, Any]] | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      **kwargs,
  ) -> TokenCountResponse:
    """Mocks client.messages.count_tokens for token counting.

    Estimates tokens by splitting text on whitespace. Adds per-message
    overhead (3 tokens per message + 3 tokens for the reply primer) to
    approximate the real API's behavior.

    Args:
      model: The model name.
      messages: The messages to count tokens for.
      system: The system prompt.
      tools: The tools for the model.
      **kwargs: Additional arguments.

    Returns:
      TokenCountResponse: The number of tokens.
    """
    del model, tools, kwargs
    total = 0
    for msg in messages:
      total += 3  # Per-message overhead.
      content = msg.get('content', '')
      if isinstance(content, str):
        total += len(content.split())
      elif isinstance(content, list):
        for part in content:
          if isinstance(part, dict) and part.get('type') == 'text':
            total += len(part.get('text', '').split())
          elif isinstance(part, dict) and part.get('type') == 'image':
            total += 85  # Approximate image token overhead.
    total += 3  # Reply primer overhead.
    if system:
      if isinstance(system, str):
        total += len(system.split())
      elif isinstance(system, list):
        for block in system:
          if isinstance(block, dict) and block.get('type') == 'text':
            total += len(block.get('text', '').split())
    return TokenCountResponse(input_tokens=total)


class Anthropic:
  """A mock of the Anthropic class that provides messages.create/count_tokens.

  See
  https://github.com/anthropics/anthropic-sdk-python
  """

  def __init__(self, api_key: str | None = None):
    del api_key
    self._reply = _DEFAULT_REPLY
    self._messages_namespace = MessagesNamespace(reply=self._reply)

  @property
  def messages(self):
    return self._messages_namespace
