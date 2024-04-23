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

"""OneTwo connector for the Google Cloud Vertex AI API.

See https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import pprint
from typing import Any, cast, Final, Union

from absl import logging
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerationConfig, GenerateContentResponse, GenerativeModel, HarmCategory, HarmBlockThreshold, Image
from vertexai.vision_models import MultimodalEmbeddingInput, MultimodalEmbeddingModel
import immutabledict
from onetwo.backends import backends_base
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import utils





# Available models are listed at https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini.
# input_token_limit=32760, output_token_limit=8192.
DEFAULT_GENERATE_MODEL: Final[str] = 'gemini-1.0-pro'
# input_token_limit=16384, output_token_limit=2048.
DEFAULT_MULTIMODAL_MODEL: Final[str] = 'gemini-1.0-pro-vision'
# input_token_limit=3072.
DEFAULT_EMBED_MODEL: Final[str] = 'textembedding-gecko@003'

# Refer to
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#request_body.
SAFETY_DISABLED: Final[Mapping[int, int]] = immutabledict.immutabledict({
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
})

EmbeddingModel = Union[TextEmbeddingModel, MultimodalEmbeddingModel]


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
class VertexAIAPI(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Google Cloud Vertex AI API.

  TODO: Implement streaming for generate_text and chat.
  TODO: Implement embed.
  TODO: Add rate limiting.

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
  project: str | None = None
  location: str | None = None
  generate_model_name: str = DEFAULT_GENERATE_MODEL
  chat_model_name: str = DEFAULT_GENERATE_MODEL
  embed_model_name: str = DEFAULT_EMBED_MODEL
  enable_streaming: bool = False
  max_qps: float | None = None

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None
  top_k: int | None = None

  # Attributes not set by constructor.
  _generate_model: GenerativeModel | None = dataclasses.field(
      init=False, default=None
  )
  _chat_model: GenerativeModel | None = dataclasses.field(
      init=False, default=None
  )
  _embed_model: EmbeddingModel | None = dataclasses.field(
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
    # TODO: Disabling generate_texts so that we fall back to the
    # default implemetation calling multiple times generate_text. Indeed it
    # seems that there is a limitation to num_candidates=1 in generate_content.
    # llm.generate_texts.configure(
    #     self.generate_texts,
    #     temperature=self.temperature,
    #     max_tokens=self.max_tokens,
    #     stop=self.stop,
    #     top_p=self.top_p,
    #     top_k=self.top_k,
    # )
    llm.embed.configure(self.embed)
    llm.chat.configure(
        self.chat, formatter=formatting.FormatterName.API
    )
    llm.count_tokens.configure(self.count_tokens)

  def _verify_available_models(self):
    """Verify that specified models are available and support all methods."""
    generation_config = GenerationConfig(
        candidate_count=1,  # Using 1 as the default.
        stop_sequences=self.stop,
        # No default set as we handle truncation (the API truncation returns
        # an empty response).
        max_output_tokens=None,
        temperature=self.temperature,
        top_p=self.top_p,
        top_k=self.top_k,
    )
    self._generate_model = GenerativeModel(
        self.generate_model_name, generation_config=generation_config
    )
    self._chat_model = GenerativeModel(
        self.chat_model_name, generation_config=generation_config
    )
    self._embed_model = EmbeddingModel(
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
    # Initialize Vertex AI
    vertexai.init(
        project=self.project,
        location=self.location,
    )
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
  ) -> GenerateContentResponse:
    """Generate content."""
    if isinstance(prompt, content_lib.ChunkList):
      converted = []
      for c in prompt:
        match c.content_type:
          case 'str':
            converted.append(c.content)
          case 'bytes' | 'image/jpeg':
            # If we have bytes we assume the image is in jpeg format.
            # TODO: support other formats.
            converted.append(
                Image.from_bytes(cast(bytes, c.content))
            )
          case _:
            converted.append(c.content)
      prompt = converted

    generation_config = GenerationConfig(
        candidate_count=samples,
        stop_sequences=stop,
        max_output_tokens=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    try:
      # TODO: Trace this external API call.
      response = self._generate_model.generate_content(
          prompt,
          generation_config=generation_config,
          **kwargs,
      )
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'VertexAIAPI.generate_content raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(prompt)[:100]}'
      ) from err
    empty = True
    for candidate in response.candidates:
      if candidate and candidate.content.parts:
        empty = False
    if empty:
      response_msg = pprint.pformat(response.candidates)
      raise ValueError(
          'VertexAIAPI.generate_content returned no answers. This may be caused '
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
      include_details: bool = False,
      **kwargs,  # Optional genai specific arguments.
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1
    response = self._generate_content(
        prompt=prompt,
        samples=1,
        temperature=temperature,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    raw = response.text
    truncated = _truncate(raw, max_tokens)
    return (truncated, {'text': raw}) if include_details else truncated

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
      include_details: bool = False,
      **kwargs,  # Optional genai specific arguments.
  ) -> Sequence[str | tuple[str, Mapping[str, Any]]]:
    """See builtins.llm.generate_texts."""
    self._counters['generate_texts'] += 1
    response = self._generate_content(
        prompt=prompt,
        samples=samples,
        temperature=temperature,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    results = []
    for candidate in response.candidates:
      if candidate and candidate.content.parts:
        raw = candidate.content.parts[0].text
        truncated = _truncate(raw, max_tokens)
        results.append(
            truncated if not include_details else (truncated, {'text': raw})
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
    raise NotImplementedError("chat_via_api is not implemented.")


  @caching.cache_method(  # Cache this deterministic method.
      name='embed',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def embed(self, content: str | content_lib.ChunkList) -> Sequence[float]:
    """See builtins.llm.embed."""
    self._counters['embed'] += 1

    # TODO: Trace this external API call.
    if type(content) == str:
      content = content_lib.ChunkList([content])
    return self._embed_model.get_embeddings(content)

  @caching.cache_method(  # Cache this method.
      name='count_tokens',
      is_sampled=False,  # Method is deterministic.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['content']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def count_tokens(self, content: str | content_lib.ChunkList) -> int:
    """See builtins.llm.count_tokens."""
    self._counters['count_tokens'] += 1

    try:
      # TODO: Trace this external API call.
      response = self._generate_model.count_tokens(content)
    except Exception as err:  # pylint: disable=broad-except
      raise ValueError(
          f'VertexAIAPI.count_tokens raised err:\n{err}\n'
          f'for request:\n{pprint.pformat(content)[:100]}'
      ) from err
    return response.total_tokens
