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

"""SentencePiece tokenizer backend for OneTwo."""

import collections
import dataclasses
import os
from typing import Final, Mapping, Sequence
import urllib.request

import immutabledict
from onetwo.backends import backends_base
from onetwo.builtins import llm
from onetwo.core import batching
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils

from typing_extensions import override

import sentencepiece as sp

TOKENIZER_PATHS: Final[Mapping[str, str]] = immutabledict.immutabledict({})


@batching.add_batching
@dataclasses.dataclass
class SentencePieceTokenizer(caching.FileCacheEnabled, backends_base.Backend):
  """SentencePiece tokenizer backend for OneTwo.

  This class provides a local tokenizer using the `sentencepiece` library.

  Attributes:
    disable_caching: Whether caching is enabled for this object (inherited from
      CacheEnabled).
    cache_filename: Name of the file (full path) where the cache is stored
      (inherited from FileCacheEnabled).
    max_retries: Maximum number of times to retry a request in case of an
      exception.
    model_name: Name of the model to use (can be a key in TOKENIZER_PATHS or a
      direct file path).
    batch_size: Batch size for batching the tokenization requests.
  """

  max_retries: int = 0
  model_name: str = ''
  batch_size: int = 64

  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )
  _model: sp.SentencePieceProcessor = dataclasses.field(init=False)

  @override
  def register(self, name: str | None = None):
    """See parent class."""
    llm.tokenize.configure(self.tokenize)
    llm.detokenize.configure(self.detokenize)

  def _load_model(self) -> sp.SentencePieceProcessor:
    if self.model_name in TOKENIZER_PATHS:
      checkpoint_path = TOKENIZER_PATHS[self.model_name]
    elif self.model_name.startswith(('http://', 'https://')):
      cache_dir = os.path.expanduser('~/.cache/onetwo/models')
      os.makedirs(cache_dir, exist_ok=True)
      filename = os.path.basename(self.model_name)
      checkpoint_path = os.path.join(cache_dir, filename)

      if not os.path.exists(checkpoint_path):
        try:
          print(
              f'Downloading model from {self.model_name} to'
              f' {checkpoint_path}...'
          )
          urllib.request.urlretrieve(self.model_name, checkpoint_path)
          print('Download complete.')
        except Exception as e:
          raise ValueError(
              f'Failed to download model from {self.model_name}. Error: {e}'
          ) from e
    else:
      checkpoint_path = self.model_name

    sp_processor = sp.SentencePieceProcessor()

    try:
      with open(checkpoint_path, 'rb') as f:
        model_content = f.read()
      sp_processor.LoadFromSerializedProto(model_content)
    except Exception as e:  # pylint: disable=broad-except
      try:
        sp_processor.Load(checkpoint_path)
      except Exception as e2:  # pylint: disable=broad-except
        raise ValueError(
            f'Failed to load model from {checkpoint_path}. '
            f'Errors: [GFile: {e}], [Load: {e2}]'
        ) from e

    return sp_processor

  def __post_init__(self) -> None:
    self._cache_handler = caching.SimpleFunctionCache(
        cache_filename=self.cache_filename,
    )
    if not self.model_name:
      raise ValueError('model_name must be specified.')
    self._model = self._load_model()

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='SentencePieceTokenizer.tokenize')
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))  # pytype: disable=wrong-arg-types
  async def tokenize(
      self, content: str | content_lib.ChunkList
  ) -> Sequence[int]:
    """See builtins.llm.tokenize."""
    self._counters['tokenize'] += 1
    if not isinstance(content, str):
      raise ValueError('Tokenizing non-string content is not yet supported.')
    return await self._batchable_tokenize(content)  # pytype: disable=bad-return-type

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name='SentencePieceTokenizer.detokenize')
  @utils.with_retry(max_retries=utils.FromInstance('max_retries'))  # pytype: disable=wrong-arg-types
  async def detokenize(self, tokens: Sequence[int]) -> Sequence[str]:
    """See builtins.llm.detokenize."""
    self._counters['detokenize'] += 1
    if not all(isinstance(item, int) for item in tokens):
      raise ValueError('Input is not a sequence of integers.')
    return await self._batchable_detokenize(tokens)  # pytype: disable=bad-return-type

  @batching.to_thread_pool_method(
      num_workers=utils.FromInstance('threadpool_size')
  )
  def _tokenize_with_threadpool(self, content: str) -> Sequence[Sequence[int]]:
    return self._batchable_tokenize(content)  # pytype: disable=bad-return-type

  @batching.batchable_method(
      implementation=utils.FromInstance('_batch_tokenize'),
  )
  def _batchable_tokenize(self, content: str) -> Sequence[int]:
    pass  # pytype: disable=bad-return-type

  @batching.batch_method(batch_size=utils.FromInstance('batch_size'))
  def _batch_tokenize(
      self, requests: batching.ParametersBatch
  ) -> Sequence[Sequence[int]]:
    """Batches the tokenization requests."""
    self._counters['_batch_tokenize'] += 1
    contents = [c['content'] for c in requests]
    tokenized_result = [self._model.EncodeAsIds(c) for c in contents]
    return tokenized_result

  @batching.to_thread_pool_method(
      num_workers=utils.FromInstance('threadpool_size')
  )
  def _detokenize_with_threadpool(self, tokens: Sequence[int]) -> Sequence[str]:
    return self._batchable_detokenize(tokens)  # pytype: disable=bad-return-type

  @batching.batchable_method(
      implementation=utils.FromInstance('_batch_detokenize'),
  )
  def _batchable_detokenize(self, tokens: Sequence[int]) -> Sequence[str]:
    pass  # pytype: disable=bad-return-type

  @batching.batch_method(batch_size=utils.FromInstance('batch_size'))
  def _batch_detokenize(
      self, requests: batching.ParametersBatch
  ) -> Sequence[str]:
    """Batches the tokenization requests."""
    self._counters['_batch_detokenize'] += 1
    tokens = [t['tokens'] for t in requests]
    detokenized_result = [self._model.DecodeIds(t) for t in tokens]
    return detokenized_result
