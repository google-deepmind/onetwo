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

"""OneTwo connector for the OpenAI API.

"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import pprint
from typing import Any, Final

from onetwo.backends import backends_base
import openai

from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import utils


DEFAULT_GENERATE_MODEL: Final[str] = 'gpt-3.5-turbo'

# Supported roles in the chat.completions.create method are listed at
# https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/types/chat/chat_completion_role.py
# If a role is not supported, it will be passed as a string (i.e. if one
# calls `chat` with `Message(role='tool')`, then 'tool' will be used directly).
# TODO: Once we add a PredefinedRole.TOOL, we can add it to this map.
_OPENAI_ROLE_BY_PREDEFINED_ROLE = {
    content_lib.PredefinedRole.USER: 'user',
    content_lib.PredefinedRole.MODEL: 'assistant',
    content_lib.PredefinedRole.SYSTEM: 'system',
}


# Note that we don't specify the type of logprobs (it is defined as
# ChoiceLogprobs in the openai library) because we don't want to explicitly
# import the module where it is defined.
def _get_score(logprobs: Any) -> float:
  """Returns the score of a candidate."""
  if logprobs is None:
    return 0.0
  score = 0.0
  for c in logprobs.content:
    score += c.logprob
  return score


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class OpenAIAPI(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """OpenAI API.

  TODO: Implement streaming for generate_text and chat.
  TODO: Implement embed and tokenize.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled)
    batch_size: Number of requests (generate_text or chat or generate_embedding)
      that is grouped together when sending them to GenAI API. GenAI API does
      not explicitly support batching (i.e. multiple requests can't be passed
      via arguments). Instead we send multiple requests from separate threads.
    api_key: OpenAI API key string (if provided directly, otherwise it will be
      read from the environment variable OPENAI_API_KEY by the OpenAI library
      https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/_client.py#L294
      ).
    model_name: Name of the model to use for `generate` requests. Model names
      are listed at https://platform.openai.com/docs/models/overview.
    enable_streaming: Whether to enable streaming replies from generate_text.
    max_qps: Maximum queries per second for the backend (if None, no rate
      limiting is applied).
    temperature: Temperature parameter (float) for LLM generation (can be set as
      a default and can be overridden per request).
    max_tokens: Maximum number of tokens to generate (can be set as a default
      and can be overridden per request).
    stop: Stop sequences (as a list of strings) for LLM text generation (can be
      set as a default and can be overridden per request).
    top_p: Top-p parameter (float) for LLM text generation (can be set as a
      default and can be overridden per request).
    top_k: Top-k parameter (int) for LLM text generation (can be set as a
      default and can be overridden per request).
  """

  batch_size: int = 1
  api_key: str | None = None
  model_name: str = DEFAULT_GENERATE_MODEL
  enable_streaming: bool = False
  max_qps: float | None = None

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None
  top_k: int | None = None

  # Attributes not set by constructor.
  _client: openai.OpenAI | None = dataclasses.field(init=False, default=None)

  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def register(self, name: str | None = None) -> None:
    """See parent class."""
    del name
    llm.generate_text.configure(
        self.generate_text,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        stop=self.stop,
        top_p=self.top_p,
        top_k=self.top_k,
    )
    llm.generate_texts.configure(
        self.generate_texts,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        stop=self.stop,
        top_p=self.top_p,
        top_k=self.top_k,
    )
    llm.chat.configure(self.chat, formatter=formatting.FormatterName.API)

  def __post_init__(self) -> None:
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    if self.api_key is not None:
      self._client = openai.OpenAI(api_key=self.api_key)
    else:
      self._client = openai.OpenAI()

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _chat_completion(
      self,
      messages: Sequence[content_lib.Message],
      *,
      samples: int = 1,
      temperature: float | None = None,
      max_tokens: int | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      **kwargs,
  ) -> Any:
    """Generate content via the chat.completions.create method.

    Args:
      messages: Sequence of messages to send to the model.
      samples: Number of samples to generate.
      temperature: Temperature parameter (float) for LLM generation.
      max_tokens: Maximum number of tokens to generate.
      stop: Stop sequences (as a list of strings) for LLM text generation.
      top_k: Top-k parameter (int) for LLM text generation.
      top_p: Top-p parameter (float) for LLM text generation.
      **kwargs: Optional OpenAI specific arguments, see
        https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/resources/chat/completions.py#L616C5-L616C16

    Returns:
      A response from the OpenAI API (as a Choice object, see
      https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/types/chat/chat_completion.py#L19
      ).
    """
    self._counters['_chat_completion'] += 1
    converted_messages = [
        {
            'role': _OPENAI_ROLE_BY_PREDEFINED_ROLE.get(
                message.role, str(message.role)
            ),
            'content': str(message.content),
        }
        for message in messages
    ]
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p
    if samples > 1:
      kwargs['n'] = samples
    try:
      response = self._client.chat.completions.create(
          model=self.model_name,
          messages=converted_messages,
          **kwargs,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'OpenAI API chat.completions.create raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(converted_messages)[:100]}'
      ) from err
    empty = True
    for candidate in response.choices:
      if candidate and candidate.message.content:
        empty = False
    if empty:
      response_msg = pprint.pformat(response.choices)
      raise ValueError(
          'GeminiAPI.generate_text returned no answers. This may be caused '
          f'by safety filters:\n{response_msg}'
      )
    return response

  @caching.cache_method(  # Cache this method.
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
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
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1
    del decoding_constraint
    kwargs = {}
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p
    if include_details:
      kwargs['logprobs'] = True
    messages = [
        content_lib.Message(
            content=prompt, role=content_lib.PredefinedRole.USER
        )
    ]
    response = self._chat_completion(messages=messages, **kwargs)
    candidate = response.choices[0]
    score = _get_score(candidate.logprobs)
    text = candidate.message.content
    return (
        (text, {'text': text, 'score': score})
        if include_details
        else text
    )

  @caching.cache_method(  # Cache this method.
      name='generate_texts',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
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
  ) -> Sequence[str | tuple[str, Mapping[str, Any]]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_texts'] += 1
    del decoding_constraint
    kwargs = {}
    if temperature is not None:
      kwargs['temperature'] = temperature
    if max_tokens is not None:
      kwargs['max_tokens'] = max_tokens
    if stop is not None:
      kwargs['stop'] = stop
    if top_k is not None:
      kwargs['top_k'] = top_k
    if top_p is not None:
      kwargs['top_p'] = top_p
    if include_details:
      kwargs['logprobs'] = True
    messages = [
        content_lib.Message(
            content=prompt, role=content_lib.PredefinedRole.USER
        )
    ]
    response = self._chat_completion(
        messages=messages, samples=samples, **kwargs
    )
    results = []
    for candidate in response.choices:
      if candidate and candidate.message.content:
        score = _get_score(candidate.logprobs)
        text = candidate.message.content
        results.append(
            (text, {'text': text, 'score': score})
            if include_details
            else text
        )
    return results

  async def chat(
      self,
      messages: Sequence[content_lib.Message],
      formatter: formatting.FormatterName = formatting.FormatterName.API,
      **kwargs,
  ) -> str:
    """See builtins.llm.chat."""
    if formatter == formatting.FormatterName.API:
      return await self.chat_via_api(messages, **kwargs)
    else:
      return await llm.default_chat(messages, formatter, **kwargs)

  @executing.make_executable
  @caching.cache_method(  # Cache this stochastic method.
      name='chat',
      is_sampled=True,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['messages']),
  )
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
    response = self._chat_completion(messages, **kwargs)
    return response.choices[0].message.content
