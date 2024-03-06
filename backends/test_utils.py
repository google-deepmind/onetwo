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

"""Utilities for testing backends."""

import ast
import collections
from collections.abc import AsyncIterator, Mapping, Sequence
import dataclasses
import re
from typing import Any, Type, TypeVar

from onetwo.backends import base as backends_base
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing


_T = TypeVar('_T')


@dataclasses.dataclass
class LLMForTest(backends_base.Backend):
  """Mock LLM backend.

  Note that there are multiple ways of controlling the replies that are returned
  by the backend (using `reply_by_prompt`, or `reply_by_sequence`, etc.).
  Usually just one of these attributes is specified, but it is also possible to
  specify multiple of them, in which case we will apply them in the following
  order of preference until one is found that defines a reply.

  For `generate_text`:
  1. reply_by_prompt
  2. reply_by_prompt_regex
  3. default_reply

  For `score_text`:
  1. reply_by_prompt_target
  2. default_score

  For the `reply_by_*` mappings, if the value is a sequence of replies, then the
  replies will be consumed in the order in which requests are received, and once
  they are used up, any further requests will be treated as if the matching
  entry were not present in the mapping (typically this means falling back to
  `default_reply` or `default_score`).

  Attributes:
    reply_by_prompt: Mapping from prompt to reply, or to a sequence of replies
      in case we want to simulate multiple samples.
    reply_by_prompt_regex: Mapping of prompt regex to the reply that should be
      returned in response to a complete request whose prompt text matches that
      regex  (or to a sequence of replies in case we want to simulate multiple
      samples). Follows the behavior of `re.search`, which means that you can
      also simply specify a substring of the prompt to match any prompt
      containing that substring. E.g., if you want to simply specify a single
      sequence of replies to return regardless of the request, you can do so by
      specifying an empty regex (''). If multiple regexes match the same prompt,
      then will return the reply associated with whichever of the regexes it
      happened to evaluate first.
    reply_by_prompt_target: Mapping from prompt+target to score.
    default_reply: Default reply if not found in reply_by_prompt.
    default_score: Default score to return if not found in
      reply_by_prompt_target.
    iterable_replies: Whether to return the reply to generate_text in a
      streaming fashion.
    prompts: All prompts that were received as `generate_text` or `score_text`
      requests, in the order received. (Same format as `unexpected_prompts`
      below.)
    unexpected_prompts: Prompts that were not found in the corresponding
      mappings (i.e., prompts for which we ended up falling back to returning
      the `default_reply` or `default_score`).
  """

  # Attributes for controlling the replies to be returned.
  reply_by_prompt: Mapping[str, str | Sequence[str]] = dataclasses.field(
      default_factory=dict
  )
  reply_by_prompt_regex: Mapping[str, str | Sequence[str]] = dataclasses.field(
      default_factory=dict
  )
  reply_by_prompt_target: Mapping[str, float] = dataclasses.field(
      default_factory=dict
  )
  default_reply: str = dataclasses.field(default_factory=str)
  default_score: float = dataclasses.field(default_factory=float)
  iterable_replies: bool = False

  # Attributes used for tracking the actual requests / replies (for assertions).
  prompts: list[str] = dataclasses.field(init=False, default_factory=list)
  unexpected_prompts: list[str] = dataclasses.field(
      init=False, default_factory=list
  )

  # Attributes used for tracking the actual requests / replies (internal).
  _num_generate_text_requests_by_prompt: collections.Counter[str] = (
      dataclasses.field(init=False, default_factory=collections.Counter)
  )
  _num_generate_text_requests_by_regex: collections.Counter[str] = (
      dataclasses.field(init=False, default_factory=collections.Counter)
  )

  def _get_generate_text_reply(self, prompt: str) -> str:
    """Returns the reply for the given prompt, while updating counters."""
    self.prompts.append(prompt)

    # By prompt.
    if prompt in self.reply_by_prompt:
      reply = self.reply_by_prompt[prompt]
      if isinstance(reply, str):
        # Single reply specified. Always return it.
        return reply
      else:
        # Sequence of replies specified. Return the next (until we run out).
        reply_index = self._num_generate_text_requests_by_prompt[prompt]
        self._num_generate_text_requests_by_prompt[prompt] += 1
        if reply_index < len(reply):
          return reply[reply_index]

    # By prompt regex.
    for regex, reply in self.reply_by_prompt_regex.items():
      if not re.search(regex, prompt):
        continue
      if isinstance(reply, str):
        # Single reply specified. Always return it.
        return reply
      else:
        # Sequence of replies specified. Return the next (until we run out).
        reply_index = self._num_generate_text_requests_by_regex[regex]
        self._num_generate_text_requests_by_regex[regex] += 1
        if reply_index < len(reply):
          return reply[reply_index]

    # Default.
    self.unexpected_prompts.append(prompt)
    return self.default_reply

  @executing.make_executable
  def generate_text(
      self,
      prompt: str | content_lib.ChunkList,
      stop: Sequence[str] | None = None,
      include_details: bool = False,
  ) -> str | tuple[str, Mapping[str, Any]]:
    if isinstance(prompt, content_lib.ChunkList):
      prompt = prompt.to_simple_string()

    reply = self._get_generate_text_reply(prompt)

    def produce_reply(reply: str) -> str | tuple[str, Mapping[str, Any]]:
      truncated_reply = (
          reply
          if stop is None
          else backends_base.truncate_reply(reply, list(stop))
      )
      if include_details:
        return (
            truncated_reply,
            {'text': reply, 'score': self.default_score},
        )
      else:
        return truncated_reply

    if not self.iterable_replies:
      return produce_reply(reply)
    else:

      @executing.make_executable
      async def iterate(
          reply,
      ) -> AsyncIterator[str | tuple[str, Mapping[str, Any]]]:
        for i in range(1, len(reply) + 1):
          yield produce_reply(reply[:i])

      return iterate(reply)

  def score_text(
      self, prompt: str | content_lib.ChunkList, targets: Sequence[str]
  ) -> list[float]:
    if isinstance(prompt, content_lib.ChunkList):
      prompt = prompt.to_simple_string()
    scores = []
    for target in targets:
      pt = f'{prompt} {target}'
      self.prompts.append(pt)
      if pt not in self.reply_by_prompt_target:
        self.unexpected_prompts.append(pt)
      scores.append(self.reply_by_prompt_target.get(pt, self.default_score))
    return scores

  async def generate_object(
      self, prompt: str | content_lib.ChunkList, cls: Type[_T]
  ) -> _T:
    reply = await self.generate_text(prompt, include_details=False)
    result = ast.literal_eval(reply)
    if isinstance(result, dict):
      return cls(**result)
    else:
      return result

  def register(self, name: str | None = None):
    del name
    llm.generate_text.configure(self.generate_text)
    llm.score_text.configure(self.score_text)
    llm.generate_object.configure(self.generate_object)
