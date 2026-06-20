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

"""A module that mocks the Ollama library, for testing purposes.

This document explains how to use the chat method:
https://github.com/ollama/ollama-python

This document explains how to use the Ollama REST API:
https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from typing import Any, Final, Type, TypeVar, Union
import pydantic

_DEFAULT_REPLY: Final[str] = 'Hello'
_T = TypeVar('_T')


class Message(pydantic.BaseModel):
  """Mock for an Ollama Message object."""

  role: str = 'assistant'
  content: str = ''
  images: list[bytes] | None = None
  thinking: str | None = None


class Options(pydantic.BaseModel):
  """Mock for the Ollama Options object."""

  model_config = pydantic.ConfigDict(extra='allow')


class ChatResponse(pydantic.BaseModel):
  """Mock for the Ollama ChatResponse object."""

  message: Message
  model: str = 'llama3.2'
  done: bool = True
  done_reason: str = 'stop'
  total_duration: int = 1000000
  prompt_eval_count: int = 10
  eval_count: int = 5


class DummyResponse:
  """Dummy response object to mock HTTP response."""

  def __init__(self, data: dict[str, Any]):
    self._data = data

  def json(self) -> dict[str, Any]:
    return self._data


class EmbedResponse(pydantic.BaseModel):
  """Mock for the Ollama EmbedResponse object."""

  model: str = 'embeddinggemma'
  embeddings: list[list[float]] = pydantic.Field(default_factory=list)


class Client:
  """A mock of the Ollama Client class that provides chat functionality."""

  def __init__(self, host: str | None = None):
    del host
    self._reply = _DEFAULT_REPLY
    self._thinking: str | None = None
    self._embedding_dim: int = 8
    self._embedding: list[float] | None = None

  def _request_raw(self, *args, **kwargs) -> DummyResponse:
    """Return a DummyResponse simulating an Ollama HTTP response."""
    del args
    json_data = kwargs.get('json', {})
    messages = json_data.get('messages', [])

    images = None
    if messages:
      last_msg = messages[-1]
      if isinstance(last_msg, dict):
        images = last_msg.get('images')
      elif hasattr(last_msg, 'images'):
        images = last_msg.images

    message_data = {
        'role': 'assistant',
        'content': self._reply,
    }
    if images is not None:
      message_data['images'] = images
    if self._thinking is not None:
      message_data['thinking'] = self._thinking

    return DummyResponse({
        'message': message_data,
        'model': json_data.get('model', 'llama3.2'),
        'done': True,
        'done_reason': 'stop',
        'total_duration': 1000000,
        'prompt_eval_count': 10,
        'eval_count': 5,
    })

  def _request(
      self,
      cls: Type[_T],
      *args,
      stream: bool = False,
      **kwargs,
  ) -> Union[_T, Any]:
    if stream:
      raise NotImplementedError('Streaming not supported in mock')
    raw_json = self._request_raw(*args, **kwargs).json()
    return cls(**raw_json)

  def chat(self, *args, **kwargs) -> ChatResponse:
    del args
    return self._request(ChatResponse, json=kwargs)

  def embed(self, **kwargs) -> EmbedResponse:
    """Mock for the Ollama embed endpoint."""
    model = kwargs.get('model', 'embeddinggemma')
    if self._embedding is not None:
      embeddings = [self._embedding]
    else:
      dim = kwargs.get('dimensions', self._embedding_dim)
      embeddings = [[0.1 * (i + 1) for i in range(dim)]]
    return EmbedResponse(model=model, embeddings=embeddings)
