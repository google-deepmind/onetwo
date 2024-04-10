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

"""A module that mocks the OpenAI library, for testing purposes.

This document explains how to use the chat completion method:
https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
"""


from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

_DEFAULT_REPLY: Final[str] = 'Hello'
_DEFAULT_SCORE: Final[float] = -0.1


class OpenAIMessage(NamedTuple):
  role: str
  content: str


class ChatCompletionTokenLogprob(NamedTuple):
  token: str
  logprob: float


class ChoiceLogProbs(NamedTuple):
  content: Sequence[ChatCompletionTokenLogprob]


class Choice(NamedTuple):
  index: int
  message: OpenAIMessage
  logprobs: ChoiceLogProbs
  finish_reason: str


class ChatCompletion(NamedTuple):
  choices: Sequence[Choice]


class OpenAI:
  """A mock of the OpenAI class that provides a chat.completions.create method.

  See
  https://github.com/openai/openai-python/blob/f0bdef04611a24ed150d19c4d180aacab3052704/src/openai/_client.py#L49
  """

  def __init__(self, api_key: str | None = None):
    del api_key
    self._reply = _DEFAULT_REPLY

  @property
  def chat(self):
    return self

  @property
  def completions(self):
    return self

  def create(
      self,
      model: str,
      messages: Sequence[Mapping[str, str]],
      **kwargs,
  ) -> ChatCompletion:
    del model, messages
    samples = 1
    if 'n' in kwargs:
      samples = kwargs['n']
    return ChatCompletion(
        choices=[
            Choice(
                index=i,
                message=OpenAIMessage(role='assistant', content=self._reply),
                logprobs=ChoiceLogProbs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token='a', logprob=_DEFAULT_SCORE
                        )
                    ]
                ),
                finish_reason='stop',
            )
            for i in range(samples)
        ]
    )
