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

"""Google OneTwo API connector.

A Backend that connects to a OneTwo model server and exposes its
functionalities.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any
from onetwo.backends import backends_base
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import utils
import requests






@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class OneTwoAPI(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Google OneTwo API.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled)
    endpoint: The address to connect to (typically some http endpoint).
    batch_size: Number of requests (generate_text or chat or generate_embedding)
      that is grouped together when sending them to OneTwo API. OneTwo API does
      not explicitly support batching (i.e. multiple requests can't be passed
      via arguments). Instead we send multiple requests from separate threads.
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
  endpoint: str = dataclasses.field(init=True, default_factory=str)
  batch_size: int = 1
  enable_streaming: bool = False
  max_qps: float | None = None

  # Generation parameters
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None
  top_p: float | None = None
  top_k: int | None = None

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
    llm.count_tokens.configure(self.count_tokens)
    llm.tokenize.configure(self.tokenize)

  def __post_init__(self) -> None:
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    # Check the health status of the endpoint.
    try:
      response = requests.get(self.endpoint + '/health')
      if response.status_code != requests.codes.ok:
        raise ValueError(f'OneTwoAPI endpoint unhealthy: {response.text}')
    except Exception as err:
      raise ValueError(f'OneTwoAPI connection failed: {err}') from err

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
      **kwargs,  # Optional server-specific arguments.
  ) -> str | tuple[str, Mapping[str, Any]]:
    """See builtins.llm.generate_text."""
    self._counters['generate_text'] += 1

    if isinstance(prompt, content_lib.ChunkList):
      prompt = str(prompt)

    args = {
        'prompt': prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stop': stop,
        'top_k': top_k,
        'top_p': top_p,
        'include_details': include_details,
    }
    args.update(kwargs)
    # TODO: Trace this external API call.
    response = requests.post(
        self.endpoint + '/generate_text',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(args),
    )
    if response.status_code != requests.codes.ok:
      raise ValueError(f'OneTwoAPI /generate_text failed: {response.text}')
    response = json.loads(response.text)
    return (response if include_details else response[0])

  @caching.cache_method(  # Cache this method.
      name='tokenize',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def tokenize(
      self,
      content: str | content_lib.ChunkList,
  ) -> list[int]:
    """See builtins.llm.tokenize."""
    self._counters['tokenize'] += 1

    if isinstance(content, content_lib.ChunkList):
      content = str(content)

    # TODO: Trace this external API call.
    response = requests.post(
        self.endpoint + '/tokenize',
        json={
            'content': content,
        },
    )
    if response.status_code != requests.codes.ok:
      raise ValueError(f'OneTwoAPI /tokenize failed: {response.text}')
    response = json.loads(response.text)
    return response['result']

  @caching.cache_method(  # Cache this method.
      name='count_tokens',
      is_sampled=False,
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt']),
  )
  @batching.batch_method_with_threadpool(
      batch_size=utils.FromInstance('batch_size'),
      wrapper=batching.add_logging,
  )
  def count_tokens(
      self,
      content: str | content_lib.ChunkList,
  ) -> int:
    """See builtins.llm.tokenize."""
    self._counters['count_tokens'] += 1

    if isinstance(content, content_lib.ChunkList):
      content = str(content)

    # TODO: Trace this external API call.
    response = requests.post(
        url=self.endpoint + '/count_tokens',
        json={'content': content},
    )
    if response.status_code != requests.codes.ok:
      raise ValueError(
          f'OneTwoAPI /count_tokens failed: {response.text}'
      )
    response = json.loads(response.text)
    return response['result']
