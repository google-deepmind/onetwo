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

"""OneTwo connector for the Ollama API.

All Ollama models (llama3.2, mistral, gemma, etc.) are supported through the
[Chat
API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion).
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import io
import logging
import pprint
from typing import Any, Final, Type, TypeVar, Union

import ollama
from onetwo.backends import backends_base
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
import PIL.Image

# See https://ollama.com/library.
DEFAULT_MODEL: Final[str] = 'gemma4:e2b'

# Supported roles in the chat method.
# Ollama uses 'user', 'assistant', and 'system' roles for messages.
_OLLAMA_ROLE_BY_PREDEFINED_ROLE = {
    content_lib.PredefinedRole.USER: 'user',
    content_lib.PredefinedRole.MODEL: 'assistant',
    content_lib.PredefinedRole.SYSTEM: 'system',
}

_COMPLETION_SYSTEM_PROMPT_PREFIX: Final[str] = 'Continue the sentence: '


def _convert_content_to_ollama_format(
    content: str | content_lib.ChunkList,
) -> tuple[str, list[bytes]]:
  """Converts OneTwo message content to Ollama API format.

  Args:
    content: Either a plain string or a OneTwo ChunkList.

  Returns:
    A tuple of (text_content, images) where:
      - text_content is the concatenated text from all text chunks.
      - images is a list of image bytes.
  """
  if isinstance(content, str):
    return content, []

  text_parts = []
  images = []
  for chunk in content.chunks:
    match chunk.content:
      case str() as text:
        text_parts.append(text)
      case bytes() as data:
        if chunk.content_type.startswith('image/'):
          images.append(data)
        else:
          logging.warning('Skipping non-image bytes chunk.')
      case PIL.Image.Image() as image:
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        images.append(buf.getvalue())
      case _:
        logging.warning(
            'Skipping unsupported chunk type: %s', type(chunk.content)
        )

  return ''.join(text_parts), images


def _convert_messages_to_ollama_format(
    messages: Sequence[content_lib.Message],
) -> list[dict[str, Any]]:
  """Converts OneTwo messages to the Ollama Chat API format.

  The Ollama Chat API accepts messages as:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

  Unlike Claude, Ollama supports system messages inline in the messages list.

  Args:
    messages: Sequence of OneTwo Message objects.

  Returns:
    A list of dicts suitable for the `messages` parameter.
  """
  api_messages = []
  for message in messages:
    role = _OLLAMA_ROLE_BY_PREDEFINED_ROLE.get(message.role, str(message.role))
    content, images = _convert_content_to_ollama_format(message.content)
    api_message = {
        'role': role,
        'content': content,
    }
    if images:
      api_message['images'] = images
    api_messages.append(api_message)
  return api_messages


_T = TypeVar('_T')


class CustomMessage(ollama.Message):
  thinking: str | None = None


class CustomChatResponse(ollama.ChatResponse):
  message: CustomMessage


class CustomOllamaClient(ollama.Client):
  """Subclass of ollama.Client that supports extracting the thinking field."""

  def _request(
      self,
      cls: Type[_T],
      *args,
      stream: bool = False,
      **kwargs,
  ) -> Union[_T, Any]:
    if stream:
      return super()._request(cls, *args, stream=stream, **kwargs)
    if cls is ollama.ChatResponse:
      cls = CustomChatResponse
    raw_json = self._request_raw(*args, **kwargs).json()
    return cls(**raw_json)


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class OllamaAPI(
    caching.CacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Ollama API.

  Uses the Chat API (client.chat) for all models.

  Note: Ollama does not provide tokenization or token counting
  APIs. These builtins are not registered by this backend.

  TODO: Implement `.embed` method.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    batch_size: Number of requests (generate_text or chat) that is grouped
      together when sending them to the API. Ollama API does not explicitly
      support batching so we send multiple requests from separate threads.
    host: Ollama server host URL (default: http://localhost:11434).
    model_name: Model to use for completion and chat requests. See
      https://ollama.com/library.
    max_qps: Maximum queries per second for the backend (if None, no rate
      limiting is applied).
    max_retries: Maximum number of times to retry a request in case of an
      exception.
    max_tokens: Maximum number of tokens to generate. Maps to `num_predict` in
      Ollama options (can be set as a default and can be overridden per
      request).
    stop: Stop sequences (as a list of strings) for LLM text generation (can be
      set as a default and can be overridden per request).
    generate_text_kwargs: Additional default parameter values to apply in calls
      to `llm.generate_text`.
  """

  batch_size: int = 1
  host: str | None = None
  model_name: str = DEFAULT_MODEL
  max_qps: float | None = None
  max_retries: int = 0

  # Generation parameters.
  max_tokens: int | None = None
  stop: Sequence[str] | None = None

  # Additional optional parameters for use during backend registration.
  generate_text_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Attributes not set by constructor.
  _client: ollama.Client | None = dataclasses.field(init=False, default=None)

  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def register(
      self,
      name: str | None = None,
      register_generate: bool = True,
  ) -> None:
    """See parent class."""
    del name
    # Reset all the defaults in case some other backend was already registered.
    llm.reset_defaults(
        reset_generate=register_generate,
        reset_tokenize=False,  # Ollama does not provide a tokenization API.
        reset_embed=False,  # Ollama does not provide an embeddings API.
    )
    if register_generate:
      llm.generate_text.configure(
          self.generate_text,
          max_tokens=self.max_tokens,
          stop=self.stop,
          **self.generate_text_kwargs,
      )

      llm.chat.configure(self.chat, formatter=formatting.FormatterName.API)
      llm.instruct.configure(
          llm.default_instruct, formatter=formatting.FormatterName.API
      )

  def __post_init__(self) -> None:
    if self.cache is None:
      self.cache = caching.SimpleFunctionCache()
    if self._client is None:
      self._client = CustomOllamaClient(host=self.host)

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _chat_create(
      self,
      messages: Sequence[content_lib.Message],
      *,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      **kwargs,
  ) -> Any:
    """Generate content via the Chat API (client.chat).

    Args:
      messages: Sequence of messages to send to the model.
      max_tokens: Maximum number of tokens to generate.
      stop: Stop sequences for LLM text generation.
      **kwargs: Optional Ollama specific arguments, see
        https://github.com/ollama/ollama/blob/main/docs/api.md

    Returns:
      A response from the Chat API.
    """
    self._counters['_chat_create'] += 1

    api_messages = _convert_messages_to_ollama_format(messages)

    # Build options dict from supported parameters.
    options: dict[str, Any] = {}
    effective_max_tokens = max_tokens or self.max_tokens
    if effective_max_tokens is not None:
      options['num_predict'] = effective_max_tokens
    effective_stop = stop if stop is not None else self.stop
    if effective_stop is not None:
      options['stop'] = list(effective_stop)

    # Map additional kwargs to options.
    temperature = kwargs.pop('temperature', None)
    if temperature is not None:
      options['temperature'] = temperature
    top_k = kwargs.pop('top_k', None)
    if top_k is not None:
      options['top_k'] = top_k
    top_p = kwargs.pop('top_p', None)
    if top_p is not None:
      options['top_p'] = top_p

    api_kwargs: dict[str, Any] = {
        'model': self.model_name,
        'messages': api_messages,
    }

    if options:
      api_kwargs['options'] = options

    api_kwargs.update(kwargs)

    assert self._client is not None
    try:
      response = self._client.chat(**api_kwargs)
      logging.info('Ollama API request: %s', api_kwargs)
      logging.info('Ollama API response: %s', response)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'ollama.client.chat raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(api_messages)[:100]}'
      ) from err
    return response

  def _complete_prompt(
      self,
      *,
      prompt: str | content_lib.ChunkList,
      **kwargs,
  ):
    """Generate a string completion using the Chat API."""
    messages = [
        content_lib.Message(
            content=_COMPLETION_SYSTEM_PROMPT_PREFIX,
            role=content_lib.PredefinedRole.SYSTEM,
        ),
        content_lib.Message(
            content=prompt,
            role=content_lib.PredefinedRole.USER,
        ),
    ]
    return self._chat_create(  # pytype: disable=wrong-keyword-args
        messages=messages,
        **kwargs,
    )

  def _extract_text_from_response(self, response: Any) -> str:
    """Extracts text from an Ollama Chat API response.

    Args:
      response: The response object from the Chat API.

    Returns:
      The text content from the response message.
    """
    return response.message.content

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='OllamaAPI.generate_text')
  async def generate_text(
      self,
      prompt: str | content_lib.ChunkList,
      *,
      include_details: bool = False,
      **kwargs,
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    text, details = await self._generate_text_with_details(  # pytype: disable=wrong-arg-count
        prompt=prompt, **kwargs
    )
    if include_details:
      return text, details
    return text

  @caching.cache_method(  # Cache this method.
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def _generate_text_with_details(
      self,
      prompt: str | content_lib.ChunkList,
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      decoding_constraint: str | None = None,
      top_logprobs: int | None = None,
      healing_option: Any = None,
  ) -> tuple[str, Mapping[str, Any]]:
    """Internal method that always returns (text, details) and is cached."""
    self._counters['generate_text'] += 1
    del decoding_constraint, top_logprobs, healing_option
    kwargs = {}
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    if temperature is not None:
      kwargs['temperature'] = temperature
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p

    response = self._complete_prompt(
        prompt=prompt,
        **kwargs,
    )
    text = self._extract_text_from_response(response)
    details = {'text': text, 'score': 0.0}
    thinking = getattr(response.message, 'thinking', None)
    if thinking:
      details[llm.THOUGHTS] = content_lib.ChunkList(
          [content_lib.Chunk(thinking)]
      )
    return text, details

  @tracing.trace(name='OllamaAPI.chat')
  async def chat(
      self,
      messages: Sequence[content_lib.Message],
      formatter: formatting.FormatterName = formatting.FormatterName.API,
      **kwargs,
  ) -> str:
    """See builtins.llm.chat."""
    if formatter != formatting.FormatterName.API:
      raise ValueError(
          'This llm.chat supports only the API formatter. '
          f'Got: {formatter.value}'
      )
    # Remove kwargs that are not supported by the Ollama API.
    kwargs.pop('formatter_kwargs', None)
    # Use native Chat API via client.chat.
    return await self.chat_via_api(messages, **kwargs)

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @caching.cache_method(  # Cache this stochastic method.
      name='chat',
      is_sampled=True,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['messages']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def chat_via_api(
      self,
      messages: Sequence[content_lib.Message],
      **kwargs,
  ) -> str:
    """See builtins.llm.chat."""
    self._counters['chat'] += 1
    response = self._chat_create(messages=messages, **kwargs)  # pytype: disable=wrong-keyword-args,wrong-arg-count
    return self._extract_text_from_response(response)
