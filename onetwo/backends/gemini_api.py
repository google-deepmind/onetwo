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

"""OneTwo connector for the Google GenAI API.

See https://ai.google.dev/api/python/google/generativeai.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import pprint
from typing import Any, cast, Final, TypeAlias

from absl import logging
import google.generativeai as genai
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import immutabledict
from onetwo.backends import backends_base
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.builtins import llm_utils
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import utils





_ChunkList: TypeAlias = content_lib.ChunkList
_TokenHealingOption: TypeAlias = llm.TokenHealingOption

# Available models are listed at https://ai.google.dev/models/gemini.
# input_token_limit=30720, output_token_limit=2048.
DEFAULT_GENERATE_MODEL: Final[str] = 'models/gemini-pro'
# input_token_limit=30720, output_token_limit=2048.
DEFAULT_MULTIMODAL_MODEL: Final[str] = 'models/gemini-pro-vision'
# input_token_limit=2048.
DEFAULT_EMBED_MODEL: Final[str] = 'models/embedding-001'

# Refer to
# https://ai.google.dev/api/python/google/ai/generativelanguage/HarmCategory.
SAFETY_DISABLED: Final[Mapping[int, int]] = immutabledict.immutabledict({
    safety_types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        safety_types.HarmBlockThreshold.BLOCK_NONE
    ),
    safety_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        safety_types.HarmBlockThreshold.BLOCK_NONE
    ),
    safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        safety_types.HarmBlockThreshold.BLOCK_NONE
    ),
    safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        safety_types.HarmBlockThreshold.BLOCK_NONE
    ),
})


def _convert_chunk_list_to_contents_type(
    prompt: str | _ChunkList
) -> content_types.ContentsType:
  """Convert ChunkList to the type compatible with Gemini API's ContentsType."""
  if isinstance(prompt, content_lib.ChunkList):
    converted = []
    for c in prompt:
      match c.content_type:
        case 'str':
          converted.append(c.content)
        case 'bytes' | 'image/jpeg':
          # If we have bytes we assume the image is in jpeg format.
          # TODO: Support other formats.
          converted.append(
              content_types.BlobDict(
                  mime_type='image/jpeg', data=cast(bytes, c.content)
              )
          )
        case _:
          converted.append(c.content)
    prompt = converted
  return prompt


def _truncate(text: str, max_tokens: int | None = None) -> str:
  """Truncates text to the given number of tokens."""
  # Unfortunately, when setting a max_output_tokens value in the API that is
  # smaller than what the model would naturally generate, the response is
  # empty with a finish_reason of "MAX_TOKENS". So we need to do post-hoc
  # truncation.
  # However we don't want to tokenize the answer in order to know its exact
  # token length, so instead we approximately truncate by counting characters.
  if max_tokens is None:
    return text
  else:
    return text[:max_tokens * 3]


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class GeminiAPI(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Google GenAI API.

  TODO: Implement streaming for generate_text and chat.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled)
    batch_size: Number of requests (generate_text or chat or generate_embedding)
      that is grouped together when sending them to GenAI API. GenAI API does
      not explicitly support batching (i.e. multiple requests can't be passed
      via arguments). Instead we send multiple requests from separate threads.
    api_key: GenAI API key string.
    api_key_file: Full quialified path to a file that contains GenAI API key on
      its first line. Only one of api_key or api_key_file can be provided. If
      neither of them is set, it is searched in the GOOGLE_API_KEY environment
      variable.
    generate_model_name: Name of the model to use for `generate` requests.
    chat_model_name: Name of the model to use for `chat` requests.
    embed_model_name: Name of the model to use for `embed` requests. replies.
      Default is 1.
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
    top_k: Top-k parameter (int) for LLM text generation (can be set as a
      default and can be overridden per request).
  """

  batch_size: int = 1
  api_key: str | None = None
  api_key_file: str | None = None
  generate_model_name: str = DEFAULT_GENERATE_MODEL
  chat_model_name: str = DEFAULT_GENERATE_MODEL
  embed_model_name: str = DEFAULT_EMBED_MODEL
  enable_streaming: bool = False
  max_qps: float | None = None
  max_retries: int = 0

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None
  top_k: int | None = None

  # Attributes not set by constructor.
  _generate_model: genai.GenerativeModel | None = dataclasses.field(
      init=False, default=None
  )
  _chat_model: genai.GenerativeModel | None = dataclasses.field(
      init=False, default=None
  )
  _embed_model: genai.GenerativeModel | None = dataclasses.field(
      init=False, default=None
  )
  _available_models: dict[str, Any] = dataclasses.field(
      init=False, default_factory=dict
  )
  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )

  def register(self, name: str | None = None) -> None:
    """See parent class."""
    del name
    # Reset all the defaults in case some other backend was already registered.
    # Indeed, we rely on certain builtins configured with OneTwo defaults.
    llm.reset_defaults()
    llm.generate_text.configure(
        self.generate_text,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        stop=self.stop,
        top_p=self.top_p,
        top_k=self.top_k,
    )
    llm.embed.configure(self.embed)
    llm.chat.configure(
        self.chat, formatter=formatting.FormatterName.API
    )
    llm.count_tokens.configure(self.count_tokens)

  def _get_api_key(self) -> str | None:
    """Retrieve GenAI API key.

    If one of the attributes api_key_file or api_key_file are provided we
    retrieve the key from there. Otherwise we return `None`.

    Returns:
      API key if it was located, otherwise None.
    """
    if self.api_key_file is not None and self.api_key:
      raise ValueError('Cannot use both api_key_file and api_key.')
    if self.api_key:
      return self.api_key
    if self.api_key_file is not None:
      if not os.path.exists(self.api_key_file):
        raise ValueError(f'File {self.api_key_file} does not exist.')
      with open(self.api_key_file, 'r') as f:
        return f.readline().strip()
    return None

  def _verify_available_models(self):
    """Verify that specified models are available and support all methods."""
    available_models = {m.name: m for m in genai.list_models()}
    logging.info('Available models:')
    for model_name, model in available_models.items():
      logging.info('Model: %s', model_name)
      logging.info('%s', pprint.pformat(model))
    self._available_models = available_models
    # TODO: Consider checking availability only for the models that
    # the user is planning to use. Revisit after the "API CL".
    if self.generate_model_name not in available_models:
      raise ValueError(f'Model {self.generate_model_name} not available.')
    if self.chat_model_name not in available_models:
      raise ValueError(f'Model {self.chat_model_name} not available.')
    if self.embed_model_name not in available_models:
      raise ValueError(f'Model {self.embed_model_name} not available.')
    if 'generateContent' not in (
        available_models[self.generate_model_name].supported_generation_methods
    ):
      raise ValueError(
          f'Model {self.generate_model_name} does not support generateContent.'
      )
    if 'countTokens' not in (
        available_models[self.generate_model_name].supported_generation_methods
    ):
      raise ValueError(
          f'Model {self.generate_model_name} does not support countTokens.'
      )
    if 'embedContent' not in (
        available_models[self.embed_model_name].supported_generation_methods
    ):
      raise ValueError(
          f'Model {self.embed_model_name} does not support embedContent.'
      )
    generation_config = genai.GenerationConfig(
        candidate_count=1,  # Using 1 as the default.
        stop_sequences=self.stop,
        # No default set as we handle truncation (the API truncation returns
        # an empty response).
        max_output_tokens=None,
        temperature=self.temperature,
        top_p=self.top_p,
        top_k=self.top_k,
    )
    self._generate_model = genai.GenerativeModel(
        self.generate_model_name, generation_config=generation_config
    )
    self._chat_model = genai.GenerativeModel(
        self.chat_model_name, generation_config=generation_config
    )
    self._embed_model = genai.GenerativeModel(
        self.embed_model_name
    )
    logging.info(
        'Registered models:\n'
        'Default for generate/count_tokens: %s\n'
        'Default for chat: %s\n'
        'Default for embed: %s',
        self.generate_model_name,
        self.chat_model_name,
        self.embed_model_name,
    )

  def __post_init__(self) -> None:
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    # Register GenAI API key.
    api_key = self._get_api_key()
    genai.configure(api_key=api_key)
    # Check available models.
    self._verify_available_models()

  @utils.rate_limit_method(qps=utils.FromInstance('max_qps'))
  def _generate_content(
      self,
      *,
      prompt: str | content_lib.ChunkList,
      samples: int = 1,
      temperature: float | None = None,
      stop: Sequence[str] | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      **kwargs,  # Optional genai specific arguments.
  ) -> generation_types.GenerateContentResponse:
    """Generate content."""
    prompt = _convert_chunk_list_to_contents_type(prompt)

    generation_config = genai.GenerationConfig(
        candidate_count=samples,
        stop_sequences=stop,
        max_output_tokens=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    try:
      # TODO: Trace this external API call.
      response = cast(
          genai.GenerativeModel, self._generate_model
      ).generate_content(
          prompt,
          generation_config=generation_config,
          **kwargs,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GeminiAPI.generate_content raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(prompt)[:100]}'
      ) from err
    empty = True
    for candidate in response.candidates:
      if candidate and candidate.content.parts:
        empty = False
    if empty:
      response_msg = pprint.pformat(response.candidates)
      raise ValueError(
          'GeminiAPI.generate_text returned no answers. This may be caused '
          f'by safety filters:\n{response_msg}'
      )
    return response

  @executing.make_executable
  @caching.cache_method(  # Cache this method.
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
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
      healing_option: _TokenHealingOption = _TokenHealingOption.NONE,
      **kwargs,  # Optional genai specific arguments.
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1
    del decoding_constraint
    healed_prompt: _ChunkList = llm_utils.maybe_heal_prompt(
        original_prompt=prompt, healing_option=healing_option
    )
    response = self._generate_content(
        prompt=healed_prompt,
        samples=1,
        temperature=temperature,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    raw = response.text
    truncated = _truncate(raw, max_tokens)
    reply = llm_utils.maybe_heal_reply(
        reply_text=truncated,
        original_prompt=prompt,
        healing_option=healing_option,
    )
    return (reply, {'text': raw}) if include_details else reply

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

    # TODO: The send_message method does not support parameters
    # like temperature, top_k, top_p, etc. So they are just ignored. We should
    # issue a warning to the user if they are set.
    healing_option = kwargs.pop('healing_option', _TokenHealingOption.NONE)

    if len(messages) == 1:
      if messages[0].role != content_lib.PredefinedRole.USER:
        raise ValueError(
            'When using the native Gemini API chat interface and passing a '
            'single message, it must be a user message.'
        )

    history = []
    for msg in messages:
      if msg.role == content_lib.PredefinedRole.SYSTEM:
        continue
      match msg.role:
        case content_lib.PredefinedRole.USER:
          role = 'user'
        case content_lib.PredefinedRole.MODEL:
          role = 'model'
        case _:
          raise ValueError(f'Unsupported role: {msg.role}')
      history.append(
          content_types.to_content(
              {
                  'role': role,
                  'parts': _convert_chunk_list_to_contents_type(msg.content),
              }
          )
      )
    generation_config = genai.GenerationConfig(
        candidate_count=1,
    )
    healed_content: _ChunkList = llm_utils.maybe_heal_prompt(
        original_prompt=messages[-1].content,
        healing_option=healing_option,
    )
    healed_content = _convert_chunk_list_to_contents_type(healed_content)
    # TODO: Trace this external API call.
    chat = cast(
        genai.GenerativeModel, self._chat_model
    ).start_chat(history=history[:-1])
    response = chat.send_message(
        content={
            'role': history[-1].role,
            'parts': healed_content,
        },
        generation_config=generation_config,
    )
    reply = llm_utils.maybe_heal_reply(
        reply_text=response.text,
        original_prompt=messages[-1].content,
        healing_option=healing_option,
    )
    return reply

  @executing.make_executable
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
  def embed(self, content: str | content_lib.ChunkList) -> Sequence[float]:
    """See builtins.llm.embed."""
    self._counters['embed'] += 1

    # TODO: Trace this external API call.
    return genai.embed_content(
        model=self.embed_model_name,
        content=content,
    )

  @executing.make_executable
  @caching.cache_method(  # Cache this method.
      name='count_tokens',
      is_sampled=False,  # Method is deterministic.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def count_tokens(self, content: str | content_lib.ChunkList) -> int:
    """See builtins.llm.count_tokens."""
    self._counters['count_tokens'] += 1

    try:
      # TODO: Trace this external API call.
      response = cast(
          genai.GenerativeModel, self._generate_model
      ).count_tokens(content)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GeminiAPI.count_tokens raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err
    return response.total_tokens
