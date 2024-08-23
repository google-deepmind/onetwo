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

"""OneTwo connector for a locally-loaded Gemma model.

This will load and run the Gemma model in process. It can use a GPU/TPU if
one is available.
"""

import collections
from collections.abc import Sequence
import dataclasses
from typing import Any, Final

from absl import logging
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
from onetwo.backends import backends_base
# Necessary for the FormatterName enum to be populated.
from onetwo.backends import formatters  # pylint: disable=unused-import
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import tracing
from onetwo.core import utils

import sentencepiece as spm


# Parameters for the sampler.
# TODO: Figure out the right settings.
# For now we use the values in gemma/colabs/sampling_tutorial.ipynb
_N_UNUSED_TOKENS: Final[int] = 128
_CACHE_SIZE: Final[int] = 1024
# Default value for max_tokens when none is provided at construction time.
_MAX_TOKENS: Final[int] = 1024

# Reply fields.
REPLY_TEXT: Final[str] = 'reply'
REPLY_SCORE: Final[str] = 'score'


@batching.add_batching  # Methods of this class are batched.
@dataclasses.dataclass
class Gemma(
    caching.FileCacheEnabled,  # Methods of this class are cached.
    backends_base.Backend,
):
  """Google Gemma API wrapper.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled)
    batch_size: Number of requests that can be sent simultaneously over
      multiple threads. Currently batch_size > 1 is NOT SUPPORTED.
    cache_filename: Some methods of this class will be cached. This attribute
      is any user-defined string that is used as a name of the file when storing
      cache on disk.
    checkpoint_path: Path to the checkpoint.
    vocab_path: Path to the vocabulary.
    temperature: Temperature for sampling.
    max_tokens: Maximum number of tokens to generate.
    stop: List of strings to stop the generation.
  """

  # Only use batch_size = 1 for now!
  # We use a single instance of the sampler which is not thread-safe.
  # TODO: Make it possible to use batch_size > 1.
  batch_size: int = 1
  cache_filename: str = 'gemma_cache'
  checkpoint_path: str | None = None
  vocab_path: str | None = None

  # Sampling parameters that can be set as defaults.
  temperature: float | None = None
  max_tokens: int | None = None
  stop: Sequence[str] | None = None

  # Used for logging by the batching.add_logging wrapper function in
  # batching.batch_method_with_threadpool decorator.
  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )
  _sampler: sampler_lib.Sampler | None = dataclasses.field(
      init=False, default=None
  )

  def register(self, name: str | None = None) -> None:
    """See parent class."""
    del name
    # Reset all the defaults in case some other backend was already registered.
    # Indeed, we rely on certain builtins configured with OneTwo defaults.
    llm.reset_defaults()
    # Configure the generate_text method with the default parameters set in
    # the constructor (or __post_init__).
    llm.generate_text.configure(
        self.generate_text,
        max_tokens=self.max_tokens,
        temperature=self.temperature,
        stop=self.stop,
    )
    if self.checkpoint_path is not None and self.checkpoint_path.endswith(
        '-it'
    ):
      llm.chat.configure(
          llm.default_chat, formatter=formatting.FormatterName.GEMMA
      )
    else:
      llm.chat.configure(
          llm.default_chat, formatter=formatting.FormatterName.DEFAULT
      )

  def _load_model(
      self,
  ) -> tuple[
      transformer_lib.TransformerConfig,
      spm.SentencePieceProcessor,
      params_lib.Params,
  ]:
    """Load the model parameters."""
    logging.info('Loading parameters from %s', self.checkpoint_path)
    params = params_lib.load_and_format_params(self.checkpoint_path)
    vocab = spm.SentencePieceProcessor()
    logging.info('Loading vocab from %s', self.vocab_path)
    vocab.Load(self.vocab_path)
    logging.info('Creating transformer config')
    transformer_config = transformer_lib.TransformerConfig.from_params(
        params=params, cache_size=_CACHE_SIZE
    )
    return transformer_config, vocab, params

  def __post_init__(self) -> None:
    if self.batch_size > 1:
      raise NotImplementedError('batch_size > 1 is not supported.')
    # Create cache.
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    transformer_config, vocab, nested_params = self._load_model()
    # Create a sampler with the right param shapes.
    logging.info('Creating sampler')
    if self.max_tokens is None:
      # Set the default to some large number.
      self.max_tokens = _MAX_TOKENS
    transformer = transformer_lib.Transformer(transformer_config)
    self._sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        params=nested_params['transformer'],
    )
    logging.info('Sampler ready')

  @tracing.trace(name='Gemma.generate_text')
  @caching.cache_method(
      name='generate_text',
      is_sampled=True,  # Two calls with same args may return different replies.
      cache_key_maker=lambda: caching.CacheKeyMaker(hashed=['prompt'])
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
      include_details: bool = False,
  ) -> str | tuple[str, dict[str, Any]]:
    """See builtins.llm.generate_text."""
    # TODO: Temperature is not used by the sampler, we should add
    # temperature sampling (and other top_p/top_k versions).
    del temperature

    self._counters['generate_text'] += 1
    if isinstance(prompt, content_lib.ChunkList):
      # For now we just make a string out of the ChunkList by skipping the non
      # string chunks and joining the string ones.
      prompt = prompt.to_simple_string()

    logging.info('Start sampling')
    # TODO: We use a single instance of the sampler and send one
    # single prompt. In order to support batch_size>1 we should create several
    # instances and possibly send a list of prompts to each of the instances,
    # but we need to figure out what the sampler uses as batch_size underneath.
    # TODO: Trace this external API call.
    response = self._sampler(
        input_strings=[prompt], total_generation_steps=max_tokens
    )
    logging.info('Done sampling')
    response_text = response.text[0]
    stop_sequences = [] if stop is None else list(stop)
    truncated_reply = backends_base.truncate_reply(
        response_text, stop_sequences
    )
    if include_details:
      return truncated_reply, {
          REPLY_TEXT: response_text,
          REPLY_SCORE: sum(response.logits[0]),
      }
    else:
      return truncated_reply
