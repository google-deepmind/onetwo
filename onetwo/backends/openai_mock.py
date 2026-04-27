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

This document explains how to use the embeddings method:
https://developers.openai.com/api/reference/resources/embeddings/methods/create
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


class InputTokensCountResponse:
  """Mock for the responses.input_tokens.count() response."""

  def __init__(self, input_tokens: int):
    self.input_tokens = input_tokens
    self.object = 'response.input_tokens'


class InputTokensNamespace:
  """Mock for client.responses.input_tokens namespace."""

  def count(
      self,
      *,
      model: str,
      input: str | Sequence[Mapping[str, str]],  # pylint: disable=redefined-builtin
      instructions: str | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      **kwargs,
  ) -> InputTokensCountResponse:
    """Mocks client.responses.input_tokens.count for token counting.

    Estimates tokens by splitting text on whitespace. Adds per-message
    overhead (3 tokens per message + 3 tokens for the reply primer) to
    approximate the real API's behavior.

    Args:
      model: The model name.
      input: The input to the model.
      instructions: The instructions for the model.
      tools: The tools for the model.
      **kwargs: Additional arguments.

    Returns:
      InputTokensCountResponse: The number of tokens.
    """
    del model, tools, kwargs
    total = 0
    if isinstance(input, str):
      total += len(input.split())
    else:
      for msg in input:
        total += 3  # Per-message overhead.
        content = msg.get('content', '')
        if isinstance(content, str):
          total += len(content.split())
        elif isinstance(content, list):
          for part in content:
            if isinstance(part, dict) and part.get('type') == 'input_text':
              total += len(part.get('text', '').split())
            elif isinstance(part, dict) and part.get('type') == 'input_image':
              total += 85  # Approximate image token overhead.
      total += 3  # Reply primer overhead.
    if instructions:
      total += len(instructions.split())
    return InputTokensCountResponse(input_tokens=total)


class ResponsesNamespace:
  """Mock for client.responses namespace."""

  def __init__(self, reply: str = _DEFAULT_REPLY):
    self._reply = reply
    self._input_tokens_namespace = InputTokensNamespace()

  @property
  def input_tokens(self):
    return self._input_tokens_namespace

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


class EmbeddingData:
  """Mock for a single embedding object in the Embeddings API response."""

  def __init__(self, embedding: Sequence[float], index: int = 0):
    self.object = 'embedding'
    self.embedding = embedding
    self.index = index


class EmbeddingsResponse:
  """Mock for the Embeddings API response object."""

  def __init__(self, data: Sequence[EmbeddingData]):
    self.object = 'list'
    self.data = data


class EmbeddingsNamespace:
  """Mock for client.embeddings namespace."""

  def __init__(self, embedding_dim: int = 10):
    self._embedding_dim = embedding_dim

  def create(
      self,
      *,
      model: str,
      input: str | Sequence[str],  # pylint: disable=redefined-builtin
      dimensions: int | None = None,
      **kwargs,
  ) -> EmbeddingsResponse:
    """Mocks client.embeddings.create for the Embeddings API."""
    del model, kwargs
    dim = dimensions if dimensions is not None else self._embedding_dim
    # Generate a deterministic non-unit-norm mock embedding based on input.
    if isinstance(input, str):
      inputs = [input]
    else:
      inputs = list(input)
    data = []
    for i, text in enumerate(inputs):
      seed = sum(ord(c) for c in text) if text else 0
      raw = [(seed + j) * 0.1 for j in range(dim)]
      # Do NOT normalize, so tests can verify the `normalize` kwarg.
      data.append(EmbeddingData(embedding=raw, index=i))
    return EmbeddingsResponse(data=data)


class OpenAI:
  """A mock of the OpenAI class that provides responses.create/parse methods.

  See
  https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/_client.py#L49
  """

  def __init__(self, api_key: str | None = None):
    del api_key
    self._reply = _DEFAULT_REPLY
    self._responses_namespace = ResponsesNamespace(reply=self._reply)
    self._embeddings_namespace = EmbeddingsNamespace()

  @property
  def responses(self):
    return self._responses_namespace

  @property
  def embeddings(self):
    return self._embeddings_namespace
