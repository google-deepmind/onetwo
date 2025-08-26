# Copyright 2025 DeepMind Technologies Limited.
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

See https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import pprint
from typing import Any, Final, TypeAlias, Union, cast

from absl import logging
from google import genai
import google.auth.credentials
from google.genai import client
from google.genai import types as genai_types
from onetwo.backends import backends_base
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.builtins import llm_utils
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils





_ChunkList: TypeAlias = content_lib.ChunkList
_TokenHealingOption: TypeAlias = llm.TokenHealingOption


@dataclasses.dataclass(frozen=True)
class ModelName:
  vertex_ai: str
  gemini_api: str


# Available models are listed at
# Vertex AI:
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
# Gemini API: https://ai.google.dev/gemini-api/docs/models

# input_token_limit=1,048,576, output_token_limit=65,535.
DEFAULT_GENERATE_MODEL: Final[ModelName] = ModelName(
    vertex_ai='publishers/google/models/gemini-2.5-flash',
    gemini_api='models/gemini-2.5-flash',
)
DEFAULT_MULTIMODAL_MODEL: Final[ModelName] = DEFAULT_GENERATE_MODEL
# input_token_limit=2048.
DEFAULT_EMBED_MODEL: Final[ModelName] = ModelName(
    vertex_ai='publishers/google/models/gemini-embedding-001',
    gemini_api='models/gemini-embedding-001',
)

# Refer to
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
SAFETY_DISABLED: Final[Sequence[genai_types.SafetySetting]] = [
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def _convert_chunk_list_to_content_list(
    prompt: str | _ChunkList,
) -> list[genai_types.Content]:
  """Converts ChunkList to the type compatible with SDK's chat history."""
  if isinstance(prompt, str):
    return [genai_types.Part(text=prompt)]

  contents = []
  for c in prompt:
    match c.content_type:
      case 'str':
        contents.append(genai_types.Part(text=c.content))
      case 'bytes' | 'image/jpeg':
        contents.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type='image/jpeg', data=cast(bytes, c.content)
                )
            )
        )
      case 'video/mp4':
        contents.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type='video/mp4', data=cast(bytes, c.content)
                )
            )
        )
      case _:
        contents.append(genai_types.Part(text=str(c.content)))
  return contents


def _replace_if_unsupported_role(
    message: content_lib.Message,
) -> content_lib.Message:
  """Replaces unsupported roles with 'user'."""
  replace_roles = {
      content_lib.PredefinedRole.CONTEXT,
      content_lib.PredefinedRole.SYSTEM,
  }
  if message.role in (replace_roles | {role.value for role in replace_roles}):
    message.role = content_lib.PredefinedRole.USER
  return message


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class GoogleGenAIAPI(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Google GenAI API.

  This class supports both Vertex AI and Gemini API.
  The `tokenize` method is only supported using Vertex AI.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled)
    batch_size: Number of requests (generate_text or chat or generate_embedding)
      that is grouped together when sending them to GenAI API. GenAI API does
      not explicitly support batching (i.e. multiple requests can't be passed
      via arguments). Instead we send multiple requests from separate threads.
    vertexai: Whether to use the Vertex AI API.
    api_key: GenAI API key string.
    api_key_file: Full quialified path to a file that contains GenAI API key on
      its first line. Only one of api_key or api_key_file can be provided. If
      neither of them is set, it is searched in the GOOGLE_API_KEY environment
      variable.
    credentials: Google Auth credentials.
    project: Google Cloud project ID.
    location: Google Cloud project location.
    debug_config: Debug config for the API.
    http_options: HTTP options for the API.
    generate_model_name: Name of the model to use for `generate` requests.
    chat_model_name: Name of the model to use for `chat` requests.
    embed_model_name: Name of the model to use for `embed` requests.
    enable_streaming: Whether to enable streaming replies from generate_text.
    max_qps: Maximum queries per second for the backend (if None, no rate
      limiting is applied).
    max_retries: Maximum number of times to retry a request in case of an
      exception.
    initial_base_delay: Initial delay in seconds for retrying a request in case
      of an exception.
    max_base_delay: Maximum delay in seconds for retrying a request in case of
      an exception.
    replace_unsupported_roles: Whether to replace roles `system` and `context`
      with role 'user' in chat requests.
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
    generate_text_kwargs: Additional default parameter values to apply in calls
      to `llm.generate_text`.
    chat_kwargs: Additional default parameter values to apply in calls to
      `llm.chat`.
  """

  batch_size: int = 1
  vertexai: bool | None = None
  api_key: str | None = None
  api_key_file: str | None = None
  credentials: google.auth.credentials.Credentials | None = None
  project: str | None = None
  location: str | None = None
  debug_config: client.DebugConfig | None = None
  http_options: (
      Union[genai_types.HttpOptions, genai_types.HttpOptionsDict] | None
  ) = None
  generate_model_name: str | ModelName = DEFAULT_GENERATE_MODEL
  chat_model_name: str | ModelName = DEFAULT_GENERATE_MODEL
  embed_model_name: str | ModelName = DEFAULT_EMBED_MODEL
  enable_streaming: bool = False
  max_qps: float | None = None
  max_retries: int = 0
  initial_base_delay: int = 1
  max_base_delay: int = 32
  replace_unsupported_roles: bool = False

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None
  top_k: int | None = None

  # Additional optional parameters for use during backend registration
  generate_text_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  chat_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Attributes not set by constructor.
  _genai_client: genai.Client = dataclasses.field(init=False)
  _available_models: dict[str, Any] = dataclasses.field(
      init=False, default_factory=dict
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
      register_embed: bool = True,
  ) -> None:
    """See parent class.

    Args:
      name: Name to use for registration.
      register_generate: Whether to register the generation related functions,
        including generate_text, chat, and instruct.
      register_tokenize: Whether to register the tokenize function.
      register_embed: Whether to register the embed function.
    """
    del name
    # Reset the defaults for the functions being registered.
    llm.reset_defaults(
        reset_generate=register_generate,
        reset_tokenize=register_tokenize,
        reset_embed=register_embed,
    )

    if register_generate:
      llm.generate_text.configure(
          self.generate_text,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          stop=self.stop,
          top_p=self.top_p,
          top_k=self.top_k,
          **self.generate_text_kwargs,
      )
    if register_embed:
      llm.embed.configure(self.embed)
    if register_generate:
      llm.chat.configure(  # pytype: disable=wrong-arg-types
          self.chat,
          formatter=formatting.FormatterName.API,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          stop=self.stop,
          top_p=self.top_p,
          top_k=self.top_k,
          **self.chat_kwargs,
      )
    if register_tokenize:
      llm.tokenize.configure(self.tokenize)
      llm.count_tokens.configure(self.count_tokens)
    if register_generate:
      llm.instruct.configure(
          llm.default_instruct, formatter=formatting.FormatterName.API
      )

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

  def _generate_model_name(self) -> str:
    if isinstance(self.generate_model_name, ModelName):
      if self.vertexai:
        return self.generate_model_name.vertex_ai
      else:
        return self.generate_model_name.gemini_api
    return self.generate_model_name

  def _chat_model_name(self) -> str:
    if isinstance(self.chat_model_name, ModelName):
      if self.vertexai:
        return self.chat_model_name.vertex_ai
      else:
        return self.chat_model_name.gemini_api
    return self.chat_model_name

  def _embed_model_name(self) -> str:
    if isinstance(self.embed_model_name, ModelName):
      if self.vertexai:
        return self.embed_model_name.vertex_ai
      else:
        return self.embed_model_name.gemini_api
    return self.embed_model_name

  def _verify_available_models(self):
    """Verify that specified models are available and support all methods."""
    available_models = {m.name: m for m in self._genai_client.models.list()}
    logging.info('Available models:')
    for model_name, model in available_models.items():
      logging.info('Model: %s', model_name)
      logging.info('%s', pprint.pformat(model))
    self._available_models = available_models
    if self._generate_model_name() not in available_models:
      raise ValueError(f'Model {self._generate_model_name()} not available.')
    if self._chat_model_name() not in available_models:
      raise ValueError(f'Model {self._chat_model_name()} not available.')
    # Embed models are not listed in the available models for Vertex AI API.
    if not self.vertexai and self._embed_model_name() not in available_models:
      raise ValueError(f'Model {self._embed_model_name()} not available.')

  def __post_init__(self) -> None:
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    # Configure GenAI client.
    self._genai_client = genai.Client(
        vertexai=self.vertexai,
        api_key=self._get_api_key(),
        credentials=self.credentials,
        project=self.project,
        location=self.location,
        debug_config=self.debug_config,
        http_options=self.http_options,
    )
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
  ) -> genai_types.GenerateContentResponse:
    """Generate content using the configured generation model."""
    prompt = _convert_chunk_list_to_content_list(prompt)
    generation_config = genai_types.GenerateContentConfig(
        candidate_count=samples,
        temperature=temperature,
        stop_sequences=stop,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    try:
      response = self._genai_client.models.generate_content(
          model=self._generate_model_name(),
          contents=prompt,
          config=generation_config,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GoogleGenAIAPI.generate_content raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(prompt)[:100]}'
      ) from err
    empty = True
    for candidate in response.candidates:
      if candidate and candidate.content.parts:
        empty = False
    if empty:
      response_msg = pprint.pformat(response.candidates)
      raise ValueError(
          'GoogleGenAIAPI.generate_text returned no answers. This may be'
          f' caused by safety filters:\n{response_msg}'
      )
    return response

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='GoogleGenAIAPI.generate_text')
  @caching.cache_method(  # Cache this method.
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  @utils.with_retry(
      max_retries=utils.FromInstance('max_retries'),
      initial_base_delay=utils.FromInstance('initial_base_delay'),
      max_base_delay=utils.FromInstance('max_base_delay'),
  )  # pytype: disable=wrong-arg-types
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

    response = self._generate_content(  # pytype: disable=wrong-keyword-args
        prompt=healed_prompt,
        samples=1,
        max_output_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    raw = response.text
    reply = llm_utils.maybe_heal_reply(
        reply_text=raw,
        original_prompt=prompt,
        healing_option=healing_option,
    )
    return (reply, {'text': raw}) if include_details else reply

  @tracing.trace(name='GoogleGenAIAPI.chat')
  async def chat(
      self,
      messages: Sequence[content_lib.Message],
      formatter: formatting.FormatterName = formatting.FormatterName.API,
      **kwargs,
  ) -> str:
    """See builtins.llm.chat."""
    if self.replace_unsupported_roles:
      messages = [_replace_if_unsupported_role(msg) for msg in messages]
    if formatter == formatting.FormatterName.API:
      return await self.chat_via_api(messages, **kwargs)
    else:
      return await llm.default_chat(messages, formatter, **kwargs)

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @caching.cache_method(  # Cache this stochastic method.
      name='chat',
      is_sampled=True,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['messages']),
  )
  @utils.with_retry(
      max_retries=utils.FromInstance('max_retries'),
      initial_base_delay=utils.FromInstance('initial_base_delay'),
      max_base_delay=utils.FromInstance('max_base_delay'),
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

    healing_option = kwargs.pop('healing_option', _TokenHealingOption.NONE)
    max_tokens = kwargs.pop('max_tokens', None)
    temperature = kwargs.pop('temperature', None)
    stop = kwargs.pop('stop', None)
    top_k = kwargs.pop('top_k', None)
    top_p = kwargs.pop('top_p', None)
    system_instruction = kwargs.pop('system_instruction', None)

    # In case the caller does not specify a system instruction, we accept the
    # first message as a system instruction if it has a system role.
    if (
        system_instruction is None
        and len(messages) > 1
        and messages[0].role
        in (
            content_lib.PredefinedRole.SYSTEM,
            content_lib.PredefinedRole.SYSTEM.value,
        )
    ):
      system_instruction = messages[0].content
      messages = messages[1:]

    history = [
        genai_types.Content(
            role=msg.role.value
            if isinstance(msg.role, content_lib.PredefinedRole)
            else msg.role,
            parts=_convert_chunk_list_to_content_list(msg.content),
        )
        for msg in messages
    ]

    generation_config = genai_types.GenerateContentConfig(
        candidate_count=1,
        stop_sequences=stop,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        system_instruction=system_instruction,
        **kwargs,
    )
    healed_content: _ChunkList = llm_utils.maybe_heal_prompt(
        original_prompt=messages[-1].content,
        healing_option=healing_option,
    )
    chat = self._genai_client.chats.create(
        model=self._chat_model_name(),
        history=history[:-1],
        config=generation_config,
    )
    response = chat.send_message(
        message=_convert_chunk_list_to_content_list(healed_content),
        config=generation_config,
    )
    reply = llm_utils.maybe_heal_reply(
        reply_text=response.text,
        original_prompt=messages[-1].content,
        healing_option=healing_option,
    )
    return reply

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='GoogleGenAIAPI.embed')
  @caching.cache_method(  # Cache this deterministic method.
      name='embed',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @utils.with_retry(
      max_retries=utils.FromInstance('max_retries'),
      initial_base_delay=utils.FromInstance('initial_base_delay'),
      max_base_delay=utils.FromInstance('max_base_delay'),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def embed(self, content: str | content_lib.ChunkList) -> Sequence[float]:
    """See builtins.llm.embed."""
    self._counters['embed'] += 1
    content = _convert_chunk_list_to_content_list(content)
    try:
      response = self._genai_client.models.embed_content(
          model=self._embed_model_name(),
          contents=content,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GoogleGenAIAPI.embed raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err
    if not response.embeddings:
      raise ValueError(
          'GoogleGenAIAPI.embed returned no embeddings for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    if len(response.embeddings) != 1:
      raise ValueError(
          'GoogleGenAIAPI.embed returned more than one embedding for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    if not response.embeddings[0].values:
      raise ValueError(
          'GoogleGenAIAPI.embed returned no embedding values for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    return response.embeddings[0].values

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='GoogleGenAIAPI.count_tokens')
  @caching.cache_method(  # Cache this method.
      name='count_tokens',
      is_sampled=False,  # Method is deterministic.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @utils.with_retry(
      max_retries=utils.FromInstance('max_retries'),
      initial_base_delay=utils.FromInstance('initial_base_delay'),
      max_base_delay=utils.FromInstance('max_base_delay'),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def count_tokens(self, content: str | content_lib.ChunkList) -> int:
    """See builtins.llm.count_tokens."""
    self._counters['count_tokens'] += 1
    content = _convert_chunk_list_to_content_list(content)
    try:
      response = self._genai_client.models.count_tokens(
          model=self._generate_model_name(),
          contents=content,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GoogleGenAIAPI.count_tokens raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err
    if not response.total_tokens:
      raise ValueError(
          'GoogleGenAIAPI.count_tokens returned no total tokens for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    return response.total_tokens

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='GoogleGenAIAPI.tokenize')
  @caching.cache_method(  # Cache this deterministic method.
      name='tokenize',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @utils.with_retry(
      max_retries=utils.FromInstance('max_retries'),
      initial_base_delay=utils.FromInstance('initial_base_delay'),
      max_base_delay=utils.FromInstance('max_base_delay'),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def tokenize(self, content: str | content_lib.ChunkList) -> Sequence[int]:
    """See builtins.llm.tokenize."""
    self._counters['tokenize'] += 1
    try:
      response = self._genai_client.models.compute_tokens(
          model=self._generate_model_name(),
          contents=_convert_chunk_list_to_content_list(content),
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'GoogleGenAIAPI.tokenize raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err

    tokens_info = response.tokens_info
    if not tokens_info or len(tokens_info) != 1:
      raise ValueError(
          'GoogleGenAIAPI.tokenize returned no tokens or more than one tokens'
          f' info for request:\n{pprint.pformat(content)[:100]}'
      )
    token_ids = tokens_info[0].token_ids
    if not token_ids:
      raise ValueError(
          'GoogleGenAIAPI.tokenize returned no token IDs for request:\n'
          f'{pprint.pformat(content)[:100]}'
      )
    return token_ids
