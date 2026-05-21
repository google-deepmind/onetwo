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

"""OneTwo connector for the Anthropic Claude API.

All Claude models (claude-sonnet-4-20250514, claude-opus-4-20250514, etc.) are
supported through the
[Messages API](https://docs.anthropic.com/en/api/messages).

"""

import base64
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import io
import logging
import pprint
from typing import Any, Final, TypeAlias

from onetwo.backends import backends_base
import anthropic

from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
import PIL.Image

# Type alias for Claude message content: plain text or a list of content blocks.
ClaudeContent: TypeAlias = str | list[dict[str, Any]]

# See https://docs.anthropic.com/en/docs/about-claude/models.
DEFAULT_MODEL: Final[str] = 'claude-opus-4-7'

# Default max_tokens for Claude API (required parameter).
DEFAULT_MAX_TOKENS: Final[int] = 4096

# Supported roles in the messages.create method are listed at
# https://docs.anthropic.com/en/api/messages.
# Claude uses 'user' and 'assistant' roles for messages, and 'system' is
# handled as a separate top-level parameter.
_CLAUDE_ROLE_BY_PREDEFINED_ROLE = {
    content_lib.PredefinedRole.USER: 'user',
    content_lib.PredefinedRole.MODEL: 'assistant',
    # System messages are extracted and passed via the `system` parameter.
    content_lib.PredefinedRole.SYSTEM: 'system',
}

_TOOL_RESULT_ROLE: Final[str] = 'user'

_COMPLETION_SYSTEM_PROMPT_PREFIX: Final[str] = 'Continue the sentence: '

_DEPRECATED_PARAMS: Final[tuple[str, ...]] = ('temperature', 'top_k', 'top_p')


def _convert_chunk_to_claude_part(
    chunk: content_lib.Chunk,
) -> dict[str, Any]:
  """Converts a single OneTwo Chunk into a Claude content block dict.

  See https://docs.anthropic.com/en/api/messages for the
  expected format of content blocks.

  Args:
    chunk: A OneTwo Chunk object to convert.

  Returns:
    A dict representing a Claude content block (e.g. text or image).

  Raises:
    ValueError: If the chunk has an unsupported binary content type.
    TypeError: If the chunk content is an unsupported Python type.
  """
  match chunk.content:
    case str() as text:
      return {'type': 'text', 'text': text}
    case bytes() as data:
      if chunk.content_type.startswith('image/'):
        b64 = base64.b64encode(data).decode('ascii')
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': chunk.content_type,
                'data': b64,
            },
        }
      else:
        raise ValueError(
            'Unsupported binary content_type for Claude API: '
            f'{chunk.content_type}'
        )
    case PIL.Image.Image() as image:
      buf = io.BytesIO()
      image.save(buf, format='JPEG')
      b64 = base64.b64encode(buf.getvalue()).decode('ascii')
      return {
          'type': 'image',
          'source': {
              'type': 'base64',
              'media_type': 'image/jpeg',
              'data': b64,
          },
      }


def _convert_content_to_claude_format(
    content: str | content_lib.ChunkList,
) -> ClaudeContent:
  """Converts OneTwo message content to Claude API format.

  For pure text, returns a string. For multimodal content (e.g. images),
  returns a list of content block dicts following the Claude Messages API
  format.

  Args:
    content: Either a plain string or a OneTwo ChunkList.

  Returns:
    A string for text-only content, or a list of content block dicts for
    multimodal content.
  """
  if isinstance(content, str):
    return content
  # ChunkList: if all chunks are text, concatenate into a single string.
  if all(isinstance(c.content, str) for c in content.chunks):
    return content.to_simple_string()
  # Mixed/multimodal content — convert each chunk to a content block.
  return [_convert_chunk_to_claude_part(c) for c in content.chunks]


def _convert_messages_to_claude_format(
    messages: Sequence[content_lib.Message],
) -> tuple[list[dict[str, ClaudeContent]], str | None]:
  """Converts OneTwo messages to the Claude Messages API format.

  The Claude Messages API accepts messages as:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

  System messages are extracted and returned separately, as Claude requires
  them to be passed via the top-level `system` parameter rather than inline.

  For multimodal messages, the content is a list of content blocks:
  [{"role": "user", "content": [{"type": "text", "text": "..."},
                                {"type": "image", "source": {...}}]}]

  Args:
    messages: Sequence of OneTwo Message objects.

  Returns:
    A tuple of (messages_list, system_prompt) where messages_list is a list
    of dicts suitable for the `messages` parameter and system_prompt is the
    concatenated system messages (or None if there are no system messages).
  """
  system_parts = []
  api_messages = []
  for message in messages:
    role = _CLAUDE_ROLE_BY_PREDEFINED_ROLE.get(message.role, str(message.role))
    if role == 'system':
      # Claude handles system messages as a separate top-level parameter.
      content = _convert_content_to_claude_format(message.content)
      if isinstance(content, str):
        system_parts.append(content)
      else:
        # Claude's `system` parameter is text-only.
        for part in content:
          if part.get('type') == 'text':
            system_parts.append(part['text'])
          else:
            raise ValueError(
                'Claude API does not support multimodal content in system '
                f'messages. Found content of type: {part.get("type")}'
            )
    else:
      api_messages.append({
          'role': role,
          'content': _convert_content_to_claude_format(message.content),
      })

  system_prompt = '\n'.join(system_parts) if system_parts else None
  return api_messages, system_prompt


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class ClaudeAPI(
    caching.CacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Anthropic Claude API.

  Uses the Messages API (client.messages.create) for all models.
  Uses the Token Count API (client.messages.count_tokens) for token counting.

  Note: Anthropic does not provide an embeddings API. Tokenize and detokenize
  are not available through the Anthropic API either.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    batch_size: Number of requests (generate_text or chat) that is grouped
      together when sending them to the API. Anthropic API does not explicitly
      support batching so we send multiple requests from separate threads.
    api_key: Anthropic API key string (if provided directly, otherwise it will
      be read from the environment variable ANTHROPIC_API_KEY by the Anthropic
      library).
    model_name: Model to use for completion and chat requests. See
      https://docs.anthropic.com/en/docs/about-claude/models.
    max_qps: Maximum queries per second for the backend (if None, no rate
      limiting is applied).
    max_retries: Maximum number of times to retry a request in case of an
      exception.
    max_tokens: Maximum number of tokens to generate. This is a *required*
      parameter for the Claude API (can be set as a default and can be
      overridden per request).
    stop: Stop sequences (as a list of strings) for LLM text generation (can be
      set as a default and can be overridden per request).
    generate_text_kwargs: Additional default parameter values to apply in calls
      to `llm.generate_text`, `llm.generate_texts`, and `llm.chat`. Any kwargs
      set here are passed directly to the Anthropic Messages API. This is the
      place to set parameters like `thinking` for extended reasoning, `effort`
      for simpler reasoning control, or `tools`/`tool_choice` for tool use.

  Key `generate_text_kwargs` parameters:
    thinking: Thinking configuration for the model. This is a dictionary
      containing the field "type" (one of "enabled", "disabled", "adaptive")
      and other configuration values depending on the specified type. See
      https://platform.claude.com/docs/en/api/messages/create#create.thinking
    output_config: Output configuration for the model. This is a dictionary
      containing the optional fields "effort" (one of "low", "medium", "high",
      "xhigh", "max") and "format". "effort" controls how long the model
      spends thinking, while "format" controls the format of the output. See
      https://platform.claude.com/docs/en/api/messages/create#create.output_config
    tools: List of tool definitions for the Messages API. Each tool should
      follow the Anthropic tool schema. Supports both custom function tools and
      built-in tools (e.g. web_search). See
      https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
    tool_choice: Tool choice configuration. Controls how the model selects
      tools. E.g., {"type": "auto"}, {"type": "any"}, {"type": "tool",
      "name": "tool_name"}.

    Note: `thinking`, `output_config`, `tools`, and `tool_choice` cannot be
    passed per-call through the `llm.generate_text` and `llm.generate_texts`
    builtins (they reject unknown kwargs). Instead, set them at registration
    time via `generate_text_kwargs`. For per-call control, use `llm.chat()`
    which accepts `**kwargs`.

  Example usage:

    Basic backend (default thinking):

    ```python
      backend = ClaudeAPI(
          api_key='sk-ant-...',
          model_name='claude-opus-4-7',
      )
      backend.register()
    ```

    Claude Sonnet 4.6 — thinking with explicit budget:

    ```python
      backend = ClaudeAPI(
          api_key='sk-ant-...',
          model_name='claude-sonnet-4-6',
          generate_text_kwargs={
              'thinking': {'type': 'enabled', 'budget_tokens': 4096},
          },
      )
      backend.register()
    ```

    Claude Opus 4.7 — effort-based reasoning control:

    ```python
      backend = ClaudeAPI(
          api_key='sk-ant-...',
          model_name='claude-opus-4-7',
          generate_text_kwargs={
              'output_config': {'effort': 'high'},
          },
      )
      backend.register()
    ```

    Claude Opus 4.7 with web search tool:

    ```python
      backend = ClaudeAPI(
          api_key='sk-ant-...',
          model_name='claude-opus-4-7',
          generate_text_kwargs={
              'tools': [{'type': 'web_search_20260209', 'name': 'web_search'}],
          },
      )
      backend.register()
    ```
  """

  batch_size: int = 1
  api_key: str | None = None
  model_name: str = DEFAULT_MODEL
  max_qps: float | None = None
  max_retries: int = 0

  # Generation parameters.
  max_tokens: int | None = DEFAULT_MAX_TOKENS
  stop: Sequence[str] | None = None

  # Additional optional parameters for use during backend registration.
  generate_text_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Attributes not set by constructor.
  _client: anthropic.Anthropic | None = dataclasses.field(
      init=False, default=None
  )

  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def register(
      self,
      name: str | None = None,
      register_generate: bool = True,
      register_tokenize: bool = True,
  ) -> None:
    """See parent class."""
    del name
    # Reset all the defaults in case some other backend was already registered.
    llm.reset_defaults(
        reset_generate=register_generate,
        reset_tokenize=register_tokenize,
        reset_embed=False,  # Claude does not provide an embeddings API.
    )
    if register_generate:
      llm.generate_text.configure(
          self.generate_text,
          max_tokens=self.max_tokens,
          stop=self.stop,
          **self.generate_text_kwargs,
      )

      llm.generate_texts.configure(
          self.generate_texts,
          max_tokens=self.max_tokens,
          stop=self.stop,
          **self.generate_text_kwargs,
      )

      llm.chat.configure(
          self.chat,
          formatter=formatting.FormatterName.API,
          **self.generate_text_kwargs,
      )
      llm.instruct.configure(
          llm.default_instruct, formatter=formatting.FormatterName.API
      )
    if register_tokenize:
      llm.count_tokens.configure(self.count_tokens)

  def __post_init__(self) -> None:
    if self.cache is None:
      self.cache = caching.SimpleFunctionCache()
    if self.api_key:
      self._client = anthropic.Anthropic(api_key=self.api_key)
    else:
      raise ValueError('An API key is required for using the Claude API.')

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _messages_create(
      self,
      messages: Sequence[content_lib.Message],
      *,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      **kwargs,
  ) -> Any:
    """Generate content via the Messages API (client.messages.create).

    Args:
      messages: Sequence of messages to send to the model.
      max_tokens: Maximum number of tokens to generate.
      stop: Stop sequences for LLM text generation.
      **kwargs: Optional Anthropic specific arguments, see
        https://docs.anthropic.com/en/api/messages

    Returns:
      A response from the Messages API.
    """
    self._counters['_messages_create'] += 1

    # Pop unsupported kwargs.
    kwargs.pop('decoding_constraint', None)

    api_messages, system_prompt = _convert_messages_to_claude_format(messages)

    api_kwargs: dict[str, Any] = {
        'model': self.model_name,
        'max_tokens': max_tokens or self.max_tokens or DEFAULT_MAX_TOKENS,
        'messages': api_messages,
    }

    if system_prompt is not None:
      api_kwargs['system'] = system_prompt
    if stop is not None:
      api_kwargs['stop_sequences'] = list(stop)

    api_kwargs.update(kwargs)

    assert self._client is not None
    try:
      response = self._client.messages.create(**api_kwargs)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'anthropic.client.messages.create raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(api_messages)[:100]}'
      ) from err
    return response

  def _complete_prompt(
      self,
      *,
      prompt: str | content_lib.ChunkList,
      **kwargs,
  ):
    """Generate a string completion using the Messages API."""
    # Pop deprecated sampling parameters (still accepted by the API but
    # deprecated). Log a warning so callers know these are ignored.
    for param in _DEPRECATED_PARAMS:
      if kwargs.pop(param, None) is not None:
        logging.warning(
            'Claude API parameter %r is deprecated and will be ignored.',
            param,
        )
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
    return self._messages_create(  # pytype: disable=wrong-keyword-args
        messages=messages,
        **kwargs,
    )

  def _extract_response(
      self, response: Any
  ) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Extracts text, thinking_output, and tool_use blocks from a response.

    Args:
      response: The response object from the Messages API.

    Returns:
      A tuple of (text, thinking_output, tool_calls) where:
        - text is the concatenated text from all text content blocks.
        - thinking_output is the concatenated thinking content (or None).
        - tool_calls is a list of dicts for any tool_use blocks, each with
          keys 'id', 'name', and 'input'.
    """
    text_parts = []
    thinking_parts = []
    tool_calls = []
    for block in response.content:
      if hasattr(block, 'thinking'):
        thinking_parts.append(block.thinking)
      elif block.type == 'tool_use':
        tool_calls.append({
            'id': block.id,
            'name': block.name,
            'input': block.input,
        })
      elif hasattr(block, 'text'):
        text_parts.append(block.text)
    text = ''.join(text_parts)
    thinking_output = ''.join(thinking_parts) if thinking_parts else None
    return text, thinking_output, tool_calls

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='ClaudeAPI.generate_text')
  @caching.cache_method(  # Cache this method.
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(
          hashed=['prompt'],
          dropped=['include_details', 'temperature', 'top_k', 'top_p'],
      ),
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
      thinking: Mapping[str, Any] | None = None,
      output_config: Mapping[str, Any] | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      tool_choice: Mapping[str, Any] | None = None,
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1
    del decoding_constraint, top_logprobs
    kwargs = {}
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    # temperature, top_k, top_p are deprecated by Claude and will be
    # stripped with a warning in _complete_prompt.
    if temperature is not None:
      kwargs['temperature'] = temperature
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p

    if thinking is not None:
      kwargs['thinking'] = thinking
    if output_config is not None:
      kwargs['output_config'] = output_config
    if tools is not None:
      kwargs['tools'] = tools
    if tool_choice is not None:
      kwargs['tool_choice'] = tool_choice

    response = self._complete_prompt(
        prompt=prompt,
        **kwargs,
    )
    # TODO: Allow including tool calls in the output if
    # `include_details` is `True`.
    text, thinking_output, _ = self._extract_response(response)
    score = 0.0
    if include_details:
      details: dict[str, Any] = {'text': text, 'score': score}
      if thinking_output is not None:
        details['thinking_output'] = thinking_output
      return (text, details)
    return text

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='ClaudeAPI.generate_texts')
  @caching.cache_method(  # Cache this method.
      name='generate_texts',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(
          hashed=['prompt'],
          dropped=['include_details', 'temperature', 'top_k', 'top_p'],
      ),
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
      thinking: Mapping[str, Any] | None = None,
      output_config: Mapping[str, Any] | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      tool_choice: Mapping[str, Any] | None = None,
  ) -> Sequence[str | tuple[str, Mapping[str, Any]]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_texts'] += 1
    del decoding_constraint, top_logprobs
    kwargs = {}
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    # temperature, top_k, top_p are deprecated by Claude and will be
    # stripped with a warning in _complete_prompt.
    if temperature is not None:
      kwargs['temperature'] = temperature
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p
    if thinking is not None:
      kwargs['thinking'] = thinking
    if output_config is not None:
      kwargs['output_config'] = output_config
    if tools is not None:
      kwargs['tools'] = tools
    if tool_choice is not None:
      kwargs['tool_choice'] = tool_choice
    # Claude Messages API does not support `n` parameter. Make separate calls.
    results = []
    for _ in range(samples):
      response = self._complete_prompt(
          prompt=prompt,
          **kwargs,
      )
      text, thinking_output, _ = self._extract_response(response)
      score = 0.0
      if include_details:
        details: dict[str, Any] = {'text': text, 'score': score}
        if thinking_output is not None:
          details['thinking_output'] = thinking_output
        results.append((text, details))
      else:
        results.append(text)

    return results

  @tracing.trace(name='ClaudeAPI.chat')
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
    # Remove kwargs that are not supported by the Claude API.
    kwargs.pop('formatter_kwargs', None)
    # Use native Messages API via client.messages.create.
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
    # Claude models do not support assistant message prefill (the conversation
    # must end with a user message). If the last message is an assistant
    # message, strip it and log a warning.
    messages = list(messages)
    if messages and messages[-1].role == content_lib.PredefinedRole.MODEL:
      logging.warning(
          'Claude does not support assistant message prefill. Dropping'
          ' trailing assistant message: %s',
          str(messages[-1].content)[:100],
      )
      messages.pop()
    response = self._messages_create(messages=messages, **kwargs)  # pytype: disable=wrong-keyword-args,wrong-arg-count
    text, _, _ = self._extract_response(response)
    return text

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='ClaudeAPI.count_tokens')
  @caching.cache_method(  # Cache this deterministic method.
      name='count_tokens',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  async def count_tokens(
      self, content: str | content_lib.ChunkList | Sequence[content_lib.Message]
  ) -> int:
    """Count input tokens using the Claude Messages API.

    Uses the token counting endpoint (client.messages.count_tokens) which
    provides accurate token counts for text, images, tools, and other content
    types without relying on local tokenizer estimation.

    See https://docs.anthropic.com/en/api/messages-count-tokens

    Args:
      content: The content to count tokens for. Can be a plain string, a
        ChunkList (text-only or multimodal), or a list of Message objects.

    Returns:
      The number of input tokens.
    """
    return await self._count_tokens(content)  # pytype: disable=wrong-arg-count

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def _count_tokens(
      self, content: str | content_lib.ChunkList | Sequence[content_lib.Message]
  ) -> int:
    """Inner method for llm.count_tokens."""
    self._counters['count_tokens'] += 1

    # Convert content to the Messages API format.
    if isinstance(content, str):
      api_messages = [{'role': 'user', 'content': content}]
      system_prompt = None
    elif isinstance(content, content_lib.ChunkList):
      converted = _convert_content_to_claude_format(content)
      api_messages = [{'role': 'user', 'content': converted}]
      system_prompt = None
    elif isinstance(content, Sequence) and all(
        isinstance(m, content_lib.Message) for m in content
    ):
      api_messages, system_prompt = _convert_messages_to_claude_format(content)
    else:
      raise TypeError(
          'count_tokens expects str, ChunkList, or Sequence[Message], '
          f'got {type(content)}'
      )

    api_kwargs: dict[str, Any] = {
        'model': self.model_name,
        'messages': api_messages,
    }
    if system_prompt is not None:
      api_kwargs['system'] = system_prompt

    assert self._client is not None
    try:
      response = self._client.messages.count_tokens(**api_kwargs)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'anthropic.client.messages.count_tokens raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err

    if not response.input_tokens:
      raise ValueError(
          'ClaudeAPI.count_tokens returned no token count for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    return response.input_tokens
