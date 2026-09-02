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

"""Word-based tokenizer backend for OneTwo."""

import collections
import dataclasses
import re
from typing import Sequence

from onetwo.backends import backends_base
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from typing_extensions import override


@dataclasses.dataclass
class WordTokenizer(backends_base.Backend):
  """Word-based tokenizer backend for OneTwo.

  This class provides a fast, in-memory word-based tokenizer that splits text
  into words. Each word is converted into an integer token ID by treating each
  character's unicode code point (ord) as a digit in base 10**pad_width.

  Since the maximum unicode code point in Python is sys.maxunicode = 0x10FFFF
  (1,114,111 in decimal, which has 7 digits), a pad_width of 7 is the exact
  minimum width needed to represent any character.

  Unlike remote LLM backends, this tokenizer executes synchronously and
  in-memory without batch queues, network retry wrappers, or deepcopy overhead.

  Attributes:
    model_name: Identifier name for the tokenizer.
    pad_width: Number of digits to zero-pad each character code point. Defaults
      to 7 (covers all unicode code points up to sys.maxunicode = 1,114,111).
    split_regex: Optional regex pattern used to split text into words. If None,
      standard whitespace splitting (str.split) is used.
    batch_size: Retained for interface compatibility; in-memory tokenization
      runs directly without queue delay.
    cache_filename: Retained for interface compatibility.
    max_retries: Retained for interface compatibility.
  """

  model_name: str = 'word'
  pad_width: int = 7
  split_regex: str | None = None
  batch_size: int = 64
  cache_filename: str | None = None
  max_retries: int = 0

  _counters: collections.Counter[str] = dataclasses.field(
      init=False, default_factory=collections.Counter
  )
  _base: int = dataclasses.field(init=False)
  _compiled_split_regex: re.Pattern[str] | None = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    self._base = 10**self.pad_width
    if self.split_regex is not None:
      self._compiled_split_regex = re.compile(self.split_regex)
    else:
      self._compiled_split_regex = None

  @override
  def register(self, name: str | None = None):
    """See parent class."""
    llm.tokenize.configure(self.tokenize)
    llm.detokenize.configure(self.detokenize)

  def _split_words(self, text: str) -> list[str]:
    """Splits text into words."""
    if not text:
      return []
    if self._compiled_split_regex is not None:
      return [w for w in self._compiled_split_regex.split(text) if w]
    return text.split()

  def _encode_word(self, word: str) -> int:
    """Encodes a single word into an integer token ID."""
    res = 0
    base = self._base
    for c in word:
      res = res * base + ord(c)
    return res

  def _decode_word(self, token_id: int) -> str:
    """Decodes a single integer token ID into a word."""
    if not isinstance(token_id, int):
      raise ValueError(f'Unsupported token type: {type(token_id)}')
    if token_id == 0:
      return chr(0)
    base = self._base
    chars = []
    while token_id > 0:
      token_id, rem = divmod(token_id, base)
      chars.append(chr(rem))
    chars.reverse()
    return ''.join(chars)

  def _encode(self, text: str) -> list[int]:
    """Encodes a string into a list of integer token IDs."""
    words = self._split_words(text)
    return [self._encode_word(w) for w in words]

  def _decode(self, tokens: Sequence[int]) -> str:
    """Decodes a sequence of integer token IDs into text."""
    words = [self._decode_word(t) for t in tokens]
    return ' '.join(words)

  def tokenize(self, content: str | content_lib.ChunkList) -> Sequence[int]:
    """See builtins.llm.tokenize."""
    self._counters['tokenize'] += 1
    if isinstance(content, content_lib.ChunkList):
      content = str(content)
    elif not isinstance(content, str):
      raise ValueError('Tokenizing non-string content is not yet supported.')
    return self._encode(content)

  def detokenize(self, tokens: Sequence[int]) -> str:
    """See builtins.llm.detokenize."""
    self._counters['detokenize'] += 1
    return self._decode(tokens)
