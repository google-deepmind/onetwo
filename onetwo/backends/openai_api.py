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

"""OneTwo connector for the OpenAI API.

All models from gpt-3.5-turbo to gpt-5.4 are supported through the [Responses
API](https://developers.openai.com/api/reference/resources/$shared#(resource)%20%24shared%20%3E%20(model)%20responses_model%20%3E%20(schema)%20%3E%20(variant)%201)

"""

import base64
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import io
import math
import pprint
from typing import Any, Final, TypeAlias, TypeVar, get_origin

from onetwo.backends import backends_base
import openai

from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
import PIL.Image
import pydantic

TypeAdapter = pydantic.TypeAdapter
_T = TypeVar('_T')

# Type alias for OpenAI message content: plain text or a list of content parts.
OpenAIContent: TypeAlias = str | list[dict[str, Any]]

# See https://platform.openai.com/docs/models/overview.
DEFAULT_MODEL: Final[str] = 'gpt-5.4-mini'

# See https://developers.openai.com/api/docs/guides/embeddings.
DEFAULT_EMBED_MODEL: Final[str] = 'text-embedding-3-small'

# Supported roles in the responses.create method are listed at
# https://platform.openai.com/docs/api-reference/responses/create
# If a role is not supported, it will be passed as a string (i.e. if one
# calls `chat` with `Message(role='tool')`, then 'tool' will be used directly).
# TODO: Once we add a PredefinedRole.TOOL, we can add it to this map.
_OPENAI_ROLE_BY_PREDEFINED_ROLE = {
    content_lib.PredefinedRole.USER: 'user',
    content_lib.PredefinedRole.MODEL: 'assistant',
    content_lib.PredefinedRole.SYSTEM: 'system',
}

_COMPLETION_SYSTEM_PROMPT_PREFIX: Final[str] = 'Continue the sentence: '


def _convert_chunk_to_openai_part(
    chunk: content_lib.Chunk,
) -> dict[str, Any]:
  """Converts a single OneTwo Chunk into an OpenAI content part dict.

  See https://platform.openai.com/docs/api-reference/responses/create for the
  expected format of content parts.

  Args:
    chunk: A OneTwo Chunk object to convert.

  Returns:
    A dict representing an OpenAI content part (e.g. input_text or
    input_image).

  Raises:
    ValueError: If the chunk has an unsupported binary content type.
    TypeError: If the chunk content is an unsupported Python type.
  """
  match chunk.content:
    case str() as text:
      return {'type': 'input_text', 'text': text}
    case bytes() as data:
      if chunk.content_type.startswith('image/'):
        b64 = base64.b64encode(data).decode('ascii')
        data_url = f'data:{chunk.content_type};base64,{b64}'
        return {'type': 'input_image', 'image_url': data_url}
      else:
        raise ValueError(
            'Unsupported binary content_type for OpenAI API: '
            f'{chunk.content_type}'
        )
    case PIL.Image.Image() as image:
      buf = io.BytesIO()
      image.save(buf, format='JPEG')
      b64 = base64.b64encode(buf.getvalue()).decode('ascii')
      data_url = 'data:image/jpeg;base64,' + b64
      return {'type': 'input_image', 'image_url': data_url}


def _convert_content_to_openai_format(
    content: str | content_lib.ChunkList,
) -> OpenAIContent:
  """Converts OneTwo message content to OpenAI API format.

  For pure text, returns a string. For multimodal content (e.g. images),
  returns a list of content part dicts following the OpenAI Responses API
  format.

  Args:
    content: Either a plain string or a OneTwo ChunkList.

  Returns:
    A string for text-only content, or a list of content part dicts for
    multimodal content.
  """
  if isinstance(content, str):
    return content
  # ChunkList: if all chunks are text, concatenate into a single string.
  if all(isinstance(c.content, str) for c in content.chunks):
    return content.to_simple_string()
  # Mixed/multimodal content — convert each chunk to a content part.
  return [_convert_chunk_to_openai_part(c) for c in content.chunks]


def _convert_messages_to_input(
    messages: Sequence[content_lib.Message],
) -> list[dict[str, OpenAIContent]]:
  """Converts OneTwo messages to the Responses API input format.

  The Responses API accepts the same message dict format as Chat Completions:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

  For multimodal messages, the content is a list of content parts:
  [{"role": "user", "content": [{"type": "input_text", "text": "..."},
                                {"type": "input_image", "image_url": "..."}]}]

  Args:
    messages: Sequence of OneTwo Message objects.

  Returns:
    A list of dicts suitable for the `input` parameter of responses.create.
  """
  return [
      {
          'role': _OPENAI_ROLE_BY_PREDEFINED_ROLE.get(
              message.role, str(message.role)
          ),
          'content': _convert_content_to_openai_format(message.content),
      }
      for message in messages
  ]


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class OpenAIAPI(
    caching.CacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """OpenAI API.

  Uses the Responses API (client.responses.create) for all models.
  Uses the Embeddings API (client.embeddings.create) for embeddings.

  TODO: Implement streaming for generate_text and chat.
  TODO: Implement tokenize.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    batch_size: Number of requests (generate_text or chat or generate_embedding)
      that is grouped together when sending them to GenAI API. GenAI API does
      not explicitly support batching (i.e. multiple requests can't be passed
      via arguments). Instead we send multiple requests from separate threads.
    api_key: OpenAI API key string (if provided directly, otherwise it will be
      read from the environment variable OPENAI_API_KEY by the OpenAI library
      https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/_client.py#L294
        ).
    model_name: Model to use for completion and chat requests. See
      https://platform.openai.com/docs/models/overview.
    enable_streaming: Whether to enable streaming replies from generate_text.
    max_qps: Maximum queries per second for the backend (if None, no rate
      limiting is applied).
    max_retries: Maximum number of times to retry a request in case of an
      exception.
    temperature: Temperature parameter (float) for LLM generation (can be set as
      a default and can be overridden per request).
    max_tokens: Maximum number of tokens to generate (can be set as a default
      and can be overridden per request).
    stop: Stop sequences (as a list of strings) for LLM text generation (can be
      set as a default and can be overridden per request).
    top_p: Top-p parameter (float) for LLM text generation (can be set as a
      default and can be overridden per request).
    reasoning: Optional reasoning configuration dict for o-series models. E.g.,
      {"effort": "low"} or {"effort": "high", "summary": "auto"}. See
      https://platform.openai.com/docs/api-reference/responses/create.
    tools: Optional list of tool definitions for function calling. Each tool
      should be a dict following the Responses API tool schema, e.g., [{"type":
      "function", "name": "get_weather", "parameters": {...}}]. See
      https://platform.openai.com/docs/api-reference/responses/create.
    embed_model_name: Name of the model to use for embeddings. See
      https://platform.openai.com/docs/models/overview.
    embed_kwargs: Optional keyword arguments to pass to the embed method.
  """

  batch_size: int = 1
  api_key: str | None = None
  model_name: str = DEFAULT_MODEL
  enable_streaming: bool = False
  max_qps: float | None = None
  max_retries: int = 0

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None

  # Responses API specific parameters
  reasoning: dict[str, Any] | None = None
  tools: list[dict[str, Any]] | None = None

  # Embed parameters
  embed_model_name: str = DEFAULT_EMBED_MODEL

  # Additional optional parameters for use during backend registration
  embed_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Attributes not set by constructor.
  _client: openai.OpenAI | None = dataclasses.field(init=False, default=None)

  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def register(
      self,
      name: str | None = None,
      register_generate: bool = True,
      register_embed: bool = True,
  ) -> None:
    """See parent class."""
    del name
    # Reset all the defaults in case some other backend was already registered.
    # Indeed, we rely on certain builtins configured with OneTwo defaults.
    llm.reset_defaults(
        reset_generate=register_generate, reset_embed=register_embed
    )
    if register_generate:
      llm.generate_text.configure(
          self.generate_text,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          stop=self.stop,
          top_p=self.top_p,
      )

      llm.generate_texts.configure(
          self.generate_texts,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          stop=self.stop,
          top_p=self.top_p,
      )

      llm.generate_object.configure(
          self.generate_object,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          stop=self.stop,
          top_p=self.top_p,
      )

      llm.chat.configure(self.chat, formatter=formatting.FormatterName.API)
      llm.instruct.configure(
          llm.default_instruct, formatter=formatting.FormatterName.API
      )
    if register_embed:
      llm.embed.configure(self.embed, **self.embed_kwargs)

  def __post_init__(self) -> None:
    if self.cache is None:
      self.cache = caching.SimpleFunctionCache()
    if self.api_key:
      self._client = openai.OpenAI(api_key=self.api_key)
    else:
      self._client = openai.OpenAI()

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _responses_create(
      self,
      messages: Sequence[content_lib.Message],
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      top_p: float | None = None,
      reasoning: dict[str, Any] | None = None,
      tools: list[dict[str, Any]] | None = None,
      **kwargs,
  ) -> Any:
    """Generate content via the Responses API (client.responses.create).

    Args:
      messages: Sequence of messages to send to the model.
      temperature: Temperature parameter (float) for LLM generation.
      max_tokens: Maximum number of tokens to generate.
      top_p: Top-p parameter (float) for LLM text generation.
      reasoning: Optional reasoning config dict for o-series models, e.g.,
        {"effort": "low"} or {"effort": "high", "summary": "auto"}.
      tools: Optional list of tool definitions for function calling, e.g.,
        [{"type": "function", "name": "get_weather", "parameters": {...}}].
      **kwargs: Optional OpenAI specific arguments, see
        https://platform.openai.com/docs/api-reference/responses/create

    Returns:
      A response from the Responses API.
    """
    self._counters['_responses_create'] += 1

    # Responses API does not support stop or top_k.
    kwargs.pop('stop', None)
    kwargs.pop('top_k', None)
    kwargs.pop('decoding_constraint', None)

    converted_input = _convert_messages_to_input(messages)

    # Extract system/instructions messages to use the `instructions` param.
    instructions = None
    input_messages = []
    for msg in converted_input:
      if msg['role'] == 'system':
        # Concatenate multiple system messages into instructions.
        if instructions is None:
          instructions = msg['content']
        else:
          instructions += '\n' + msg['content']
      else:
        input_messages.append(msg)

    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_output_tokens'] = max_tokens
    if top_p is not None:
      kwargs['top_p'] = top_p

    # Apply instance-level defaults for reasoning and tools, then override
    # with any per-call values.
    effective_reasoning = reasoning if reasoning is not None else self.reasoning
    effective_tools = tools if tools is not None else self.tools
    if effective_reasoning is not None:
      kwargs['reasoning'] = effective_reasoning
    if effective_tools is not None:
      kwargs['tools'] = effective_tools

    api_kwargs = {
        'model': self.model_name,
    }
    if input_messages:
      api_kwargs['input'] = input_messages
      if instructions is not None:
        api_kwargs['instructions'] = instructions
    elif instructions is not None:
      api_kwargs['input'] = instructions
    else:
      api_kwargs['input'] = ''
    api_kwargs.update(kwargs)

    try:
      # TODO: Trace this external API call.
      response = self._client.responses.create(**api_kwargs)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'openai.client.responses.create raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(converted_input)[:100]}'
      ) from err
    return response

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _responses_parse(
      self,
      messages: Sequence[content_lib.Message],
      cls: Any,
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      top_p: float | None = None,
      reasoning: dict[str, Any] | None = None,
      tools: list[dict[str, Any]] | None = None,
      **kwargs,
  ) -> Any:
    """Generate structured content via client.responses.parse.

    This method is specifically for generating structured outputs conforming
    to a Pydantic schema, leveraging the OpenAI SDK's built-in parsing.

    Args:
      messages: Sequence of messages to send to the model.
      cls: The Pydantic BaseModel subclass. The SDK uses this to build the
        response schema and parse the output.
      temperature: Temperature parameter (float) for LLM generation.
      max_tokens: Maximum number of tokens to generate.
      top_p: Top-p parameter (float) for LLM text generation.
      reasoning: Optional reasoning config dict for o-series models.
      tools: Optional list of tool definitions for function calling.
      **kwargs: Additional keyword arguments to pass to the underlying
        `client.responses.parse` call.

    Returns:
      A response from OpenAI's API.
    """
    self._counters['_responses_parse'] += 1

    if not isinstance(cls, type) or not issubclass(cls, pydantic.BaseModel):
      raise ValueError(
          'The `cls` argument must be a subclass of `pydantic.BaseModel`.'
      )

    converted_input = _convert_messages_to_input(messages)

    # Extract system/instructions messages.
    instructions = None
    input_messages = []
    for msg in converted_input:
      if msg['role'] == 'system':
        if instructions is None:
          instructions = msg['content']
        else:
          instructions += '\n' + msg['content']
      else:
        input_messages.append(msg)

    api_kwargs = {
        'model': self.model_name,
        'text_format': cls,
    }
    if input_messages:
      api_kwargs['input'] = input_messages
      if instructions is not None:
        api_kwargs['instructions'] = instructions
    elif instructions is not None:
      api_kwargs['input'] = instructions
    else:
      api_kwargs['input'] = ''
    if temperature is not None:
      api_kwargs['temperature'] = temperature
    if max_tokens is not None:
      api_kwargs['max_output_tokens'] = max_tokens
    if top_p is not None:
      api_kwargs['top_p'] = top_p

    # Apply instance-level defaults for reasoning and tools, then override
    # with any per-call values.
    effective_reasoning = reasoning if reasoning is not None else self.reasoning
    effective_tools = tools if tools is not None else self.tools
    if effective_reasoning is not None:
      api_kwargs['reasoning'] = effective_reasoning
    if effective_tools is not None:
      api_kwargs['tools'] = effective_tools
    api_kwargs.update(kwargs)

    try:
      response = self._client.responses.parse(**api_kwargs)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'OpenAI API client.responses.parse raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(converted_input)[:100]}'
      ) from err

    return response

  def _complete_prompt(
      self,
      *,
      prompt: str | content_lib.ChunkList,
      **kwargs,
  ):
    """Generate a string completion using the Responses API."""
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
    return self._responses_create(  # pytype: disable=wrong-keyword-args
        messages=messages,
        **kwargs,
    )

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='OpenAIAPI.generate_text')
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
  def generate_text(
      self,
      prompt: str | content_lib.ChunkList,
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      decoding_constraint: str | None = None,
      include_details: bool = False,
      top_logprobs: int | None = None,
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1
    # Responses API does not support stop or top_k.
    del decoding_constraint, stop, top_k
    kwargs = {}
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if top_p is not None:
      kwargs['top_p'] = top_p
    if top_logprobs is not None:
      kwargs['top_logprobs'] = top_logprobs
      kwargs['include'] = ['message.output_text.logprobs']
    elif include_details:
      kwargs['include'] = ['message.output_text.logprobs']

    response = self._complete_prompt(
        prompt=prompt,
        **kwargs,
    )
    text = response.output_text
    score = 0.0
    if include_details:
      try:
        logprobs_data = response.output[0].content[0].logprobs
        score = sum(
            item.logprob
            if hasattr(item, 'logprob')
            else item.get('logprob', 0.0)
            for item in logprobs_data
        )
      except (AttributeError, IndexError):
        pass
    return (text, {'text': text, 'score': score}) if include_details else text

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='OpenAIAPI.generate_texts')
  @caching.cache_method(  # Cache this method.
      name='generate_texts',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def generate_texts(
      self,
      prompt: str | content_lib.ChunkList,
      samples: int = 1,
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      decoding_constraint: str | None = None,
      include_details: bool = False,
      top_logprobs: int | None = None,
  ) -> Sequence[str | tuple[str, Mapping[str, Any]]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_texts'] += 1
    # Responses API does not support stop or top_k.
    del decoding_constraint, stop, top_k
    kwargs = {}
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if top_p is not None:
      kwargs['top_p'] = top_p
    if top_logprobs is not None:
      kwargs['top_logprobs'] = top_logprobs
      kwargs['include'] = ['message.output_text.logprobs']
    elif include_details:
      kwargs['include'] = ['message.output_text.logprobs']
    # Responses API does not support `n` parameter. Make separate calls.
    results = []
    for _ in range(samples):
      response = self._complete_prompt(
          prompt=prompt,
          **kwargs,
      )
      text = response.output_text
      score = 0.0
      if include_details:
        try:
          logprobs_data = response.output[0].content[0].logprobs
          score = sum(
              item.logprob
              if hasattr(item, 'logprob')
              else item.get('logprob', 0.0)
              for item in logprobs_data
          )
        except (AttributeError, IndexError):
          pass
      results.append(
          (text, {'text': text, 'score': score}) if include_details else text
      )

    return results

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @caching.cache_method(
      name='generate_object',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def _generate_object_internal(
      self,
      prompt: str | content_lib.ChunkList | Sequence[content_lib.Message],
      cls: type[Any],
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      top_p: float | None = None,
  ) -> dict[str, Any]:
    """Internal helper for generate_object that is cached."""
    self._counters['generate_object'] += 1

    if isinstance(prompt, Sequence) and isinstance(
        prompt[0], content_lib.Message
    ):
      messages = prompt
    else:
      messages = [
          content_lib.Message(
              content=prompt,
              role=content_lib.PredefinedRole.USER,
          )
      ]

    kwargs = {}
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if top_p is not None:
      kwargs['top_p'] = top_p

    response = self._responses_parse(  # pytype: disable=wrong-keyword-args
        messages=messages,
        cls=cls,
        **kwargs,
    )

    # The Responses API returns the parsed JSON as output_text. We parse it
    # back through pydantic to get a validated, serializable dict.
    raw_text = response.output_text
    adapter = TypeAdapter(cls)
    parsed_object = adapter.validate_json(raw_text)
    return adapter.dump_python(parsed_object, mode='json')

  @tracing.trace(name='OpenAIAPI.generate_object')
  async def generate_object(
      self,
      prompt: str | content_lib.ChunkList | Sequence[content_lib.Message],
      cls: type[_T],
      *,
      temperature: float | None = None,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
  ) -> _T:
    """Generates a structured object via OpenAI's Responses API.

    Suitable for registering as implementation of builtins.llm.generate_object.


    This method leverages OpenAI's native structured output support via the
    Responses API and requires `cls` to be a subclass of `pydantic.BaseModel`.
    The OpenAI SDK automatically converts the Pydantic schema, sends it to the
    API, and returns the structured JSON in the response.

    To support caching, this method calls the internal helper
    `_generate_object_internal`, which executes the API call and returns a
    serialized dictionary. This method then deserializes that dictionary
    back into an instance of the specified class `cls` using Pydantic for
    validation.

    Args:
        prompt: The input prompt to the model.
        cls: The Pydantic BaseModel subclass defining the structure of the
          desired output. The OpenAI SDK uses this to constrain the model's
          output and automatically parse the result.
        temperature: See builtins.llm.generate_text.
        max_tokens: See builtins.llm.generate_text.
        stop: See builtins.llm.generate_text.
        top_k: See builtins.llm.generate_text.
        top_p: See builtins.llm.generate_text.

    Returns:
        An instance of the type specified by the 'cls' argument.

    Raises:
        ValueError: If the model does not support structured output, or if the
          OpenAI API returns an unexpected response type.
    """
    # Responses API does not support stop or top_k.
    del stop, top_k

    is_primitive = cls in (int, float, str, bool)
    is_list = get_origin(cls) is list

    if is_primitive or is_list:
      target_cls = pydantic.create_model('Wrapper', value=(cls, ...))
      is_wrapped = True
    else:
      target_cls = cls
      is_wrapped = False

    serializable_result = await self._generate_object_internal(
        prompt=prompt,
        cls=target_cls,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    value = serializable_result['value'] if is_wrapped else serializable_result
    adapter = TypeAdapter(cls)
    return adapter.validate_python(value)

  @tracing.trace(name='OpenAIAPI.chat')
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
    # Remove kwargs that are not supported by the OpenAI API.
    kwargs.pop('formatter_kwargs', None)
    # Use native Responses API via client.responses.create.
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
    response = self._responses_create(messages=messages, **kwargs)  # pytype: disable=wrong-keyword-args,wrong-arg-count
    return response.output_text

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='OpenAIAPI.embed')
  @caching.cache_method(  # Cache this deterministic method.
      name='embed',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def embed(
      self,
      content: str | content_lib.ChunkList,
      **kwargs,
  ) -> Sequence[float]:
    """Generate an embedding vector for the given content.

    Uses the OpenAI Embeddings API (client.embeddings.create).
    See
    https://developers.openai.com/api/reference/resources/embeddings/methods/create

    Args:
      content: The text content to embed (string or ChunkList).
      **kwargs: Optional keyword arguments (see below).

    Optional keyword arguments:
      output_dimensionality: Maximum number of dimensions for the output
        embedding vector. Mapped to the OpenAI ``dimensions`` parameter. Only
        supported by text-embedding-3 and later models.
      normalize: If True, re-normalize the embedding vector to unit norm. Useful
        when using ``output_dimensionality`` with models that return truncated
        non-unit-norm vectors.

    Returns:
      A sequence of floats representing the embedding vector.
    """
    self._counters['embed'] += 1

    # Convert ChunkList to plain text. Only text-based embeddings are supported.
    if isinstance(content, content_lib.ChunkList):
      text_parts = []
      for chunk in content.chunks:
        if isinstance(chunk.content, str):
          text_parts.append(chunk.content)
        else:
          raise ValueError(
              'OpenAIAPI.embed only supports text content. '
              f'Got chunk with type: {type(chunk.content)}'
          )
      text = ''.join(text_parts)
    else:
      text = content

    # Build API kwargs.
    api_kwargs = {
        'model': self.embed_model_name,
        'input': text,
    }
    if 'output_dimensionality' in kwargs:
      api_kwargs['dimensions'] = kwargs['output_dimensionality']

    try:
      response = self._client.embeddings.create(**api_kwargs)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'openai.client.embeddings.create raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(text)[:100]}'
      ) from err

    if not response.data:
      raise ValueError(
          'OpenAIAPI.embed returned no embeddings for request:\n'
          f'{pprint.pformat(text)[:100]}'
      )
    values = response.data[0].embedding
    if not values:
      raise ValueError(
          'OpenAIAPI.embed returned empty embedding vector for request:\n'
          f'{pprint.pformat(text)[:100]}'
      )

    # Optionally re-normalize to unit norm (useful for truncated embeddings).
    if kwargs.get('normalize', False):
      norm = math.sqrt(sum(v * v for v in values))
      if norm > 0 and not math.isclose(norm, 1.0, rel_tol=1e-6):
        values = [v / norm for v in values]

    return values
