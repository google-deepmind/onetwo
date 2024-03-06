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

"""Definitions of built-in functions for the OneTwo Language Models API.

These built-in functions are mostly interfaces that need to be implemented.
This is done by calling their `configure` method.

Important: None is treated as a value that can be overridden by the default
parameter values that have been set with `configure`. So calling
generate('...', temperature=None) will use the default temperature (if one
was provided), and would fail otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import logging
import random
import textwrap
from typing import Any, Final, NamedTuple, TypeVar, cast

from onetwo.builtins import base
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling


ROLE_MODEL: Final[str] = 'model'
ROLE_INSTRUCTIONS: Final[str] = 'instructions'
ROLE_USER: Final[str] = 'user'

DEFAULT_INSTRUCT_FEWSHOT: Final[str] = textwrap.dedent("""\
    Task: Write me a palindrome.
    Answer: Level
    """)

_T = TypeVar('_T')


@base.Builtin[str]
def generate_text_from_chunks(
    chunk_list: content_lib.ChunkList,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    include_details: bool = False,
) -> str | tuple[str, Mapping[str, Any]]:
  """Interface of the multimodal generate_text built-in function.

  This function should be configured by the backend, but the generate_text
  function can be used as the main entry point, as it also takes care of the
  parsing of a string prompt (provided `string_to_chunk_list` is also
  configured).

  Args:
    chunk_list: A content_lib.ChunkList representing the input prompt.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    include_details: If True, the result will be a Sequence of tuples instead of
      a sequence of strings (see include_details in generate_text).

  Returns:
    The answer from the calling the MMM text generation function
    on the provided chunks.
  """
  del chunk_list, temperature, max_tokens, stop, top_k, top_p, include_details
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@base.Builtin[str | tuple[str, Mapping[str, Any]]]
async def generate_text(
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
  """Interface of the generate_text built-in function.

  Complete the provided prompt with the LLM and return the completion. This is
  intended to be the "purest" form of completion, where "what you specify is
  what the LLM will see", i.e., little to no modification is applied to your
  prompt before it is sent to the model. See `instruct` and `chat` for the
  higher level builtin functions.

  Args:
    prompt: A string (which will have to be parsed into a content_lib.ChunkList)
      or a content_lib.ChunkList.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    decoding_constraint: Optional decoding constraint regex (str).
    include_details: If True, the result will be a tuple with a string and a
      Mapping containing additional information (backend-specific).

  Returns:
    The answer from the calling the LLM text generation function
    on the provided chunks.
  """
  del (
      prompt,
      temperature,
      max_tokens,
      stop,
      top_k,
      top_p,
      decoding_constraint,
      include_details,
  )
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


async def _default_generate_text_from_string(
    prompt: str | content_lib.ChunkList, **kwargs
) -> str | tuple[str, Mapping[str, Any]]:
  """Default implementation of generate_text treats the string as a chunk."""

  if isinstance(prompt, str):
    prompt = content_lib.ChunkList([prompt])  # pytype: disable=wrong-arg-types
  return await generate_text_from_chunks(prompt, **kwargs)


generate_text.configure(_default_generate_text_from_string)


@base.Builtin
def generate_texts(
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
  """Interface of the generate_texts built-in function.

  Complete the provided prompt with the LLM and return multiple completions.
  This is intended to be the "purest" form of completion, where "what you
  specify is what the LLM will see", i.e., little to no modification is applied
  to your prompt before it is sent to the model. See `instruct` and `chat` for
  the higher level builtin functions.

  Args:
    prompt: A string (which will have to be parsed into a content_lib.ChunkList)
      or a content_lib.ChunkList.
    samples: Number of samples to generate for this prompt.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    decoding_constraint: Optional decoding constraint regex (str).
    include_details: If True, the result will be a Sequence of tuples instead of
      a sequence of strings (see include_details in generate_text).

  Returns:
    A Sequence of samples from the LLM.
  """
  del (
      prompt,
      samples,
      temperature,
      max_tokens,
      stop,
      top_k,
      top_p,
      decoding_constraint,
      include_details,
  )
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@executing.make_executable
def _default_generate_texts(
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
  """Default implementation via repeatedly calling generate_text(_with_details).

  Args:
    prompt: The text prompt as a string.
    samples: Number of samples to generate for this prompt.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    decoding_constraint: Optional decoding constraint regex (str).
    include_details: If True, the result will be a Sequence of tuples instead of
      a sequence of strings (see include_details in generate_text).

  Returns:
    A Sequence of samples from the LLM.
  """
  executable = generate_text(
      prompt,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
      top_k=top_k,
      top_p=top_p,
      decoding_constraint=decoding_constraint,
      include_details=include_details,
  )
  if not isinstance(executable, executing.Executable):
    raise ValueError(
        'The implementation of generate should return an Executable.'
    )
  executables = sampling.repeat(executable, samples)
  # Note that we return an executable and thus the make_executable decorator
  # will ensure that the returned value can be executed or streamed
  # transparently.
  # In particular we disable the bad-return-type check because of this
  # conversion done dynamically.
  return executing.par_iter(executables)  # pytype: disable=bad-return-type


generate_texts.configure(_default_generate_texts)


@base.Builtin
def tokenize(content: str | content_lib.ChunkList) -> Sequence[int]:
  """Interface of the tokenize_text built-in function.

  Args:
    content: The content (string or ChunkList) to be tokenized.

  Returns:
    A Sequence of token ids.
  """
  del content
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@base.Builtin
def count_tokens(content: str | content_lib.ChunkList) -> int:
  """Interface of the count_tokens built-in function.

  Args:
    content: The content (string or ChunkList) to be tokenized.

  Returns:
    The length of the tokenized string (number of tokens).
  """
  del content
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@executing.make_executable
async def _default_count_tokens(
    content: str | content_lib.ChunkList,
) -> int:
  tokens = await tokenize(content)
  return len(tokens)


count_tokens.configure(_default_count_tokens)


@base.Builtin
def embed(content: str | content_lib.ChunkList) -> Sequence[float]:
  """Interface of the embed built-in function.

  Args:
    content: The content (string or ChunkList) to be embedded.

  Returns:
    A Sequence of floats.
  """
  del content
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@base.Builtin
def score_text(
    prompt: str | content_lib.ChunkList, targets: Sequence[str]
) -> Sequence[float]:
  """Interface of the score_text built-in function.

  Args:
    prompt: The prefix with which to compute the scores of the targets.
    targets: Sequence of possible completions of the prompt to be scored.

  Returns:
    A Sequence of floats, one for each target.
  """
  del prompt, targets
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


class Message(NamedTuple):
  """NamedTuple to represent a message in a chat conversation."""

  role: str
  content: str | content_lib.ChunkList


@base.Builtin
def instruct(
    prompt: str | content_lib.ChunkList,
    assistant_prefix: str | content_lib.ChunkList | None = None,
    **kwargs,
) -> str:
  """Interface of the instruct built-in function.

  Ask LLM assistant to execute the instructions provided in the prompt. This is
  different from "completion" (see `generate_text`), although it is quite
  difficult to draw a line between the two. Instruction tuning (IT) can be
  viewed as a process that takes pre-trained (PT) LLMs capable of completing the
  sequences and enables them to execute instructions. In other words, this
  builtin function can be implemented by sending requests (properly formatted
  with control tokens) to IT models. On the other hand, it can be also
  implemented by sending specially designed prompts to the PT models for
  completion, e.g.,
  ```
    Task: <prompt>
    Answer: <assistant_prefix>
  ```

  Args:
    prompt: Instructions to be executed. E.g., "Write me a short story!".
    assistant_prefix: Optional beginning of the assistant's reply. If provided,
      assistant's response will start from this string. E.g., "This story
      happened on a Planet called".
    **kwargs: Optional arguments.

  Returns:
    The string returned from LLM.
  """
  del prompt, assistant_prefix, kwargs
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
async def default_instruct(
    prompt: str | content_lib.ChunkList,
    assistant_prefix: str | content_lib.ChunkList | None = None,
    **kwargs,
) -> str:
  """Instruction prompting implementation of instruct via generate_text."""
  if assistant_prefix is not None and assistant_prefix.startswith(' '):
    logging.warning(
        'assistant_prefix starts with space. This may potentially '
        'cause bad LLM outputs.'
    )
  concat = f'{prompt}\n{assistant_prefix}'
  matches = ['Task:', 'Answer:']
  if any(substr in concat for substr in matches):
    logging.warning(
        'Looks like provided prompt or assistant_prefix already contain '
        'some formatting, including "Task:" and "Answer:" strings. This may '
        'cause difficulties and confuse LLM.'
    )
  # We use few shot prompting by default.

  use_fewshot = True
  if 'use_fewshot' in kwargs:
    value = kwargs['use_fewshot']
    if isinstance(value, bool):
      use_fewshot = value
    del kwargs['use_fewshot']
  instruct_prompt = content_lib.ChunkList()
  if isinstance(use_fewshot, bool) and use_fewshot:
    instruct_prompt += DEFAULT_INSTRUCT_FEWSHOT
  prompt = prompt.lstrip(' ')
  if prompt:
    instruct_prompt += 'Task: ' + prompt + '\nAnswer:'
  else:
    instruct_prompt += 'Task:\nAnswer:'
  if assistant_prefix is not None:
    assistant_prefix = assistant_prefix.lstrip(' ')
    if assistant_prefix:
      instruct_prompt += ' ' + assistant_prefix
  copied_args = copy.deepcopy(kwargs)
  if 'stop' not in copied_args:
    copied_args['stop'] = []
  copied_args['stop'].append('\nTask:')
  content = await generate_text(instruct_prompt, **copied_args)
  content = cast(str, content)
  return content


instruct.configure(default_instruct)


@base.Builtin
def chat(messages: Sequence[Message], **kwargs) -> str:
  """Interface of the chat built-in function.

  Ask LLM to generate a new reply or complete existing reply in the chat. The
  generated replies always correspond to assistant role. If the last message of
  messages has a role different from assistant, the turn will be switched to
  assistant and LLM will generate assistant's reply. Otherwise LLM will complete
  the last message.

  Args:
    messages: A Sequence of Message(s). Messages can have any roles. Among them
      ROLE_MODEL and ROLE_INSTRUCTIONS roles have special semantics. Instructuns
      role can be used to describe the rules that assistant is expected to
      follow when generating (or completing) the messages, e.g., "Start every
      answer by quoting Shakespeare". The first non-empty message with role
      ROLE_INSTRUCTIONS will be used to form the rules, the others will be
      ignored.
    **kwargs: Optional arguments.

  Returns:
    The string returned from LLM. The response always corresponds to the
      assistant role.
  """
  del messages, kwargs
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
async def default_chat(messages: Sequence[Message], **kwargs) -> str:
  """Default implementation of chat via prompting and generate_text."""
  prompt = content_lib.ChunkList()
  # If there is a message with instructions role, append the rules.
  for msg in messages:
    if msg.role == ROLE_INSTRUCTIONS and msg.content:
      prompt += (
          f'Actor "{ROLE_MODEL}" needs to obey the following rules when '
          'generating the messages below:\n'
      )
      prompt += msg.content + '\n\n'
      break
  # Remove any messages with instructions roles.
  messages = [msg for msg in messages if msg.role != ROLE_INSTRUCTIONS]
  if messages[-1].role != ROLE_MODEL:
    # Include empty assistant message in the end.
    messages.append(Message(role=ROLE_MODEL, content=''))
  for message in messages[:-1]:
    # Process all but the last message.
    prompt += f'**{message.role}**: ' + message.content + '\n'
  # Last message has assistant role. Don't include end of turn tag.
  last_assistant_content = messages[-1].content
  last_assistant_content = last_assistant_content.lstrip(' ')
  if last_assistant_content:
    prompt += f'**{ROLE_MODEL}**: ' + last_assistant_content
  else:
    prompt += f'**{ROLE_MODEL}**:'
  copied_args = copy.deepcopy(kwargs)
  if 'stop' not in copied_args:
    copied_args['stop'] = []
  # Stop generation when LLM finishes the assistant turn.
  copied_args['stop'].append('\n**')
  content = await generate_text(prompt, **copied_args)
  content = cast(str, content)
  return content


chat.configure(default_chat)


@base.Builtin
def select(
    prompt: str | content_lib.ChunkList,
    options: Sequence[str | content_lib.ChunkList],
    include_details: bool = False,
) -> str | tuple[str, int, Sequence[float]]:
  """Interface of the choose built-in function.

  Args:
    prompt: The prompt asking for a choice among options.
    options: Possible completions to choose from.
    include_details: If False, return only the selected option, otherwise return
      a tuple (see returned value below).

  Returns:
    The selected option if include_details=False (default), or a tuple of
    (selected option, index of the selected option, sequence of scores for all
    options).
  """
  del prompt, options, include_details
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@base.Builtin
def rank(
    prompt: str | content_lib.ChunkList,
    options: Sequence[str | content_lib.ChunkList],
    top_k: int = 1,
    include_details: bool = False,
) -> Sequence[str] | tuple[Sequence[str], Sequence[float]]:
  """Interface of the choose built-in function.

  Args:
    prompt: The prompt asking for a choice among options.
    options: Possible completions to choose from.
    top_k: The maximum number of options to return (ranked). If top_k=0,
      all options will be ranked and returned.
    include_details: If False, return only the top k options, otherwise return a
      tuple (see returned value below).

  Returns:
    The top k options if include_details=False (default), or a tuple of
    (sequence of top k options, sequence of scores for all options).
  """
  del prompt, options, top_k, include_details
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
async def _default_select_via_score(
    prompt: str | content_lib.ChunkList,
    options: Sequence[str | content_lib.ChunkList],
    include_details: bool = False,
) -> str | tuple[str, int, Sequence[float]]:
  """Default implementation of select via score."""
  # Convert the options into strings.
  options = [str(option) for option in options]
  # Compute the scores.
  scores = await score_text(prompt, options)
  max_value = max(scores)
  max_indices = [i for i, score in enumerate(scores) if score == max_value]
  best_index = random.choice(max_indices)
  if include_details:
    return options[best_index], best_index, scores
  else:
    return options[best_index]


select.configure(_default_select_via_score)


@executing.make_executable
async def _default_rank_via_score(
    prompt: str | content_lib.ChunkList,
    options: Sequence[str | content_lib.ChunkList],
    top_k: int = 1,
    include_details: bool = False,
) -> Sequence[str] | tuple[Sequence[str], Sequence[float]]:
  """Default implementation of rank via score."""
  # Convert the options into strings.
  options = [str(option) for option in options]
  # Compute the scores.
  scores = await score_text(prompt, options)
  results = sorted(zip(options, scores), key=lambda x: x[1], reverse=True)

  if top_k > 0:
    top_results = results[:top_k]
  else:
    top_results = results
  top_results = [r[0] for r in top_results]

  if include_details:
    return (top_results, scores)
  else:
    return top_results


rank.configure(_default_rank_via_score)


@base.Builtin
async def generate_object(
    prompt: str | content_lib.ChunkList,
    cls: type[_T],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    ) -> _T:
  """Interface of the generate_object builtin function.

  Args:
    prompt: The text prompt as a string.
    cls: The object type to generate.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).

  Returns:
    An object decoded from the LLM's constrained decoding.
  """
  del prompt, cls, temperature, max_tokens, top_k, top_p
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )
