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

"""A module that mocks the OpenAI library, for testing purposes.

This document explains how to use the responses method:
https://platform.openai.com/docs/api-reference/responses/create
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeVar, get_args, get_origin

import pydantic

_DEFAULT_REPLY: Final[str] = 'Hello'
_T = TypeVar('_T', bound=pydantic.BaseModel)


# Responses API mock types.
class OutputTextContent:
  """Mock for response output_text content item."""

  def __init__(self, text: str, logprobs: Sequence[Any] | None = None):
    self.type = 'output_text'
    self.text = text
    self.annotations = []
    self.logprobs = logprobs or []


class ResponseOutputMessage:
  """Mock for a message item in the Responses API output."""

  def __init__(
      self,
      content: Sequence[OutputTextContent],
      role: str = 'assistant',
      status: str = 'completed',
  ):
    self.type = 'message'
    self.role = role
    self.status = status
    self.content = content


class Response:
  """Mock for the Responses API response object."""

  def __init__(
      self,
      output: Sequence[ResponseOutputMessage],
      output_text: str,
  ):
    self.output = output
    self.output_text = output_text


class ResponsesNamespace:
  """Mock for client.responses namespace."""

  def __init__(self, reply: str = _DEFAULT_REPLY):
    self._reply = reply

  def create(
      self,
      *,
      model: str,
      input: str | Sequence[Mapping[str, str]],  # pylint: disable=redefined-builtin
      instructions: str | None = None,
      reasoning: Mapping[str, Any] | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      **kwargs,
  ) -> Response:
    """Mocks client.responses.create for the Responses API."""
    if not input and not any(
        k in kwargs
        for k in ['previous_response_id', 'prompt', 'conversation_id']
    ):
      raise ValueError(
          "Error code: 400 - {'error': {'message': 'One of \"input\" or"
          " \"previous_response_id\"or \\'prompt\\'or \\'conversation_id\\'"
          " must be provided.', 'type': 'invalid_request_error', 'param':"
          " None, 'code': 'missing_required_parameter'}}"
      )
    include = kwargs.get('include', [])
    logprobs = []
    if 'message.output_text.logprobs' in include:
      logprobs = [{'token': self._reply, 'logprob': -0.5}]
    del model, input, instructions, reasoning, tools, kwargs
    content = OutputTextContent(text=self._reply, logprobs=logprobs)
    message = ResponseOutputMessage(content=[content])
    return Response(output=[message], output_text=self._reply)

  def parse(
      self,
      *,
      model: str,
      input: str | Sequence[Mapping[str, str]],  # pylint: disable=redefined-builtin
      instructions: str | None = None,
      text_format: type[_T] | None = None,
      reasoning: Mapping[str, Any] | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      **kwargs,
  ) -> Response:
    """Mocks client.responses.parse for structured outputs."""
    if not input and not any(
        k in kwargs
        for k in ['previous_response_id', 'prompt', 'conversation_id']
    ):
      raise ValueError(
          "Error code: 400 - {'error': {'message': 'One of \"input\" or"
          " \"previous_response_id\"or \\'prompt\\'or \\'conversation_id\\'"
          " must be provided.', 'type': 'invalid_request_error', 'param':"
          " None, 'code': 'missing_required_parameter'}}"
      )
    del model, input, instructions, reasoning, tools, kwargs

    if (
        text_format is not None
        and isinstance(text_format, type)
        and issubclass(text_format, pydantic.BaseModel)
    ):
      try:
        mock_data = {}
        for field_name, field_info in text_format.model_fields.items():
          field_type = field_info.annotation
          if field_type is int:
            mock_data[field_name] = 42
          elif field_type is str:
            mock_data[field_name] = 'Mock string'
          elif field_type is float:
            mock_data[field_name] = 3.14
          elif field_type is bool:
            mock_data[field_name] = True
          elif get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            if isinstance(item_type, type) and issubclass(
                item_type, pydantic.BaseModel
            ):
              item_mock_data = {}
              for f_name, f_info in item_type.model_fields.items():
                f_type = f_info.annotation
                if f_type is int:
                  item_mock_data[f_name] = 42
                elif f_type is str:
                  item_mock_data[f_name] = 'Mock string'
                elif f_type is float:
                  item_mock_data[f_name] = 3.14
                elif f_type is bool:
                  item_mock_data[f_name] = True
                else:
                  item_mock_data[f_name] = None
              mock_item = item_type(**item_mock_data)
              mock_data[field_name] = [mock_item]
            else:
              if item_type is int:
                mock_data[field_name] = [42]
              elif item_type is str:
                mock_data[field_name] = ['Mock string']
              else:
                mock_data[field_name] = []
          else:
            mock_data[field_name] = None

        mock_obj = text_format(**mock_data)
        content_text = mock_obj.model_dump_json()
      except Exception as e:  # pylint: disable=broad-except
        content_text = f'Mock failed to construct {text_format}: {e}'
    else:
      content_text = self._reply

    content = OutputTextContent(text=content_text)
    message = ResponseOutputMessage(content=[content])
    return Response(output=[message], output_text=content_text)


class OpenAI:
  """A mock of the OpenAI class that provides responses.create/parse methods.

  See
  https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/_client.py#L49
  """

  def __init__(self, api_key: str | None = None):
    del api_key
    self._reply = _DEFAULT_REPLY
    self._responses_namespace = ResponsesNamespace(reply=self._reply)

  @property
  def responses(self):
    return self._responses_namespace
