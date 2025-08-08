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
import random
from typing import Any, TypeAlias, TypeVar

from onetwo.builtins import builtins_base
from onetwo.builtins import formatting
from onetwo.builtins import llm_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import sampling
from onetwo.core import tracing


_T = TypeVar('_T')

_ChunkList: TypeAlias = content_lib.ChunkList
_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole
TokenHealingOption: TypeAlias = llm_utils.TokenHealingOption


@builtins_base.Builtin[str | tuple[str, Mapping[str, Any]]]
async def generate_text(
    prompt: str | _ChunkList,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> str | tuple[str, Mapping[str, Any]]:
  """Interface of the generate_text built-in function.

  Complete the provided prompt with the LLM and return the completion. This is
  intended to be the "purest" form of completion, where "what you specify is
  what the LLM will see", i.e., little to no modification is applied to your
  prompt before it is sent to the model. See `instruct` and `chat` for the
  higher level builtin functions.

  Args:
    prompt: A string (which will have to be parsed into a _ChunkList) or a
      _ChunkList.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    decoding_constraint: Optional decoding constraint regex (str).
    include_details: If True, the result will be a tuple with a string and a
      Mapping containing additional information (backend-specific).
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

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
      healing_option,
  )
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@executing.make_executable
def echo_generate_text(
    prompt: str | _ChunkList, **kwargs
) -> str | tuple[str, Mapping[str, Any]]:
  """Implementation of generate_text that simply returns the prompt."""
  if isinstance(prompt, str):
    return prompt
  return '', {'prompt': prompt, 'kwargs': kwargs}


@builtins_base.Builtin
def generate_texts(
    prompt: str | _ChunkList,
    samples: int = 1,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> Sequence[str | tuple[str, Mapping[str, Any]]]:
  """Interface of the generate_texts built-in function.

  Version of `generate_text` that generates multiple samples.

  Args:
    prompt: A string (which will have to be parsed into a _ChunkList) or a
      _ChunkList.
    samples: Number of samples to generate for this prompt.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    decoding_constraint: Optional decoding constraint regex (str).
    include_details: If True, the result will be a Sequence of tuples instead of
      a sequence of strings (see include_details in generate_text).
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

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
      healing_option,
  )
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@executing.make_executable
def _default_generate_texts(
    prompt: str | _ChunkList,
    samples: int = 1,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
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
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

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
      healing_option=healing_option,
  )
  if not isinstance(executable, executing.Executable):
    raise ValueError(
        'The implementation of generate_text should return an Executable.'
    )
  executables = sampling.repeat(executable, samples)
  # Note that we return an executable and thus the make_executable decorator
  # will ensure that the returned value can be executed or streamed
  # transparently.
  # In particular we disable the bad-return-type check because of this
  # conversion done dynamically.
  return executing.par_iter(executables)  # pytype: disable=bad-return-type


@builtins_base.Builtin
def tokenize(content: str | _ChunkList) -> Sequence[int]:
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


@builtins_base.Builtin
def count_tokens(content: str | _ChunkList) -> int:
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
    content: str | _ChunkList,
) -> int:
  tokens = await tokenize(content)
  return len(tokens)


@builtins_base.Builtin
def embed(content: str | _ChunkList) -> Sequence[float]:
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


@builtins_base.Builtin
def score_text(
    prompt: str | _ChunkList,
    targets: Sequence[str],
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> Sequence[float]:
  """Interface of the score_text built-in function.

  Args:
    prompt: The prefix with which to compute the scores of the targets.
    targets: Sequence of possible completions of the prompt to be scored.
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

  Returns:
    A Sequence of floats, one for each target.
  """
  del prompt, targets, healing_option
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@builtins_base.Builtin
def instruct(
    prompt: str | _ChunkList,
    assistant_prefix: str | _ChunkList | None = None,
    *,
    formatter: formatting.FormatterName = formatting.FormatterName.DEFAULT,
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
    formatter: The formatter to use (see `formatting.FormatterName`).
    **kwargs: Optional arguments to be passed to the generate_text function or
      to the formatter. We assume that formatter-specific arguments (if present)
      are all gathered in kwargs['formatter_kwargs'] as a dict.

  Returns:
    The string returned from LLM.
  """
  del prompt, assistant_prefix, formatter, kwargs
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
@tracing.trace(name='llm.instruct')
async def default_instruct(
    prompt: str | _ChunkList,
    assistant_prefix: str | _ChunkList | None = None,
    formatter: formatting.FormatterName = formatting.FormatterName.DEFAULT,
    **kwargs,
) -> str:
  """Default implementation of instruct via chat."""
  messages = [_Message(role=_PredefinedRole.USER, content=prompt)]
  if assistant_prefix:
    messages.append(
        _Message(role=_PredefinedRole.MODEL, content=assistant_prefix)
    )
  return await chat(messages, formatter=formatter, **kwargs)


@builtins_base.Builtin
def chat(
    messages: Sequence[_Message],
    *,
    formatter: formatting.FormatterName = formatting.FormatterName.DEFAULT,
    **kwargs,
) -> str:
  """Interface of the chat built-in function.

  Ask LLM to generate a new reply or complete existing reply in the chat. The
  generated replies always correspond to assistant role. If the last message of
  messages has a role different from assistant, the turn will be switched to
  assistant and LLM will generate assistant's reply. Otherwise LLM will complete
  the last message.

  Args:
    messages: A Sequence of Message(s). Supported roles of the messages depend
      on the backend used.
    formatter: The formatter to use (see `formatting.FormatterName`).
    **kwargs: Optional arguments to be passed to the generate_text function or
      to the formatter. We assume that formatter-specific arguments (if present)
      are all gathered in kwargs['formatter_kwargs'] as a dict.

  Returns:
    The string returned from LLM. The response always corresponds to the
      assistant role.
  """
  del messages, formatter, kwargs
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
@tracing.trace(name='llm.chat')
async def default_chat(
    messages: Sequence[_Message],
    formatter: formatting.FormatterName = formatting.FormatterName.DEFAULT,
    **kwargs,
) -> str:
  """Default implementation of chat via prompting and generate_text."""

  formatter_kwargs = {}
  if 'formatter_kwargs' in kwargs:
    # We store formatter-specific arguments under `formatter_kwargs` key. If it
    # is there we store it and remove it from `kwargs`, as these will be used to
    # call `generate_text`.
    formatter_kwargs = kwargs['formatter_kwargs']
    del kwargs['formatter_kwargs']

  if formatter == formatting.FormatterName.API:
    raise NotImplementedError(
        'API formatting is not supported in this implementation of `chat`'
        ' builtin.'
    )
  elif formatter == formatting.FormatterName.NONE:
    # Concatenate all messages into a single ChunkList.
    chunk_list = _ChunkList()
    for msg in messages:
      chunk_list += msg.content
    return await generate_text(chunk_list, **kwargs)
  else:
    formatter_class = formatting.FORMATTER_CLASS_BY_NAME.get(formatter, None)
    if formatter_class is None:
      raise ValueError(f'Formatter {formatter.value} is not supported.')
    formatter_instance = formatter_class(formatter_kwargs)  # pytype: disable=not-instantiable
    prompt = formatter_instance.format(messages)
    stop_sequences = formatter_instance.extra_stop_sequences()
    defaults = routing.function_registry[generate_text.name].defaults
    if 'stop' in defaults:
      stop_sequences += defaults['stop'] or []
    if 'stop' in kwargs:
      stop_sequences += kwargs['stop'] or []
    if stop_sequences:
      kwargs['stop'] = list(set(stop_sequences))
    return await generate_text(prompt, **kwargs)


@builtins_base.Builtin
def select(
    prompt: str | _ChunkList,
    options: Sequence[str | _ChunkList],
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> str | tuple[str, int, Sequence[float]]:
  """Interface of the choose built-in function.

  Args:
    prompt: The prompt asking for a choice among options.
    options: Possible completions to choose from.
    include_details: If False, return only the selected option, otherwise return
      a tuple (see returned value below).
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

  Returns:
    The selected option if include_details=False (default), or a tuple of
    (selected option, index of the selected option, sequence of scores for all
    options).
  """
  del prompt, options, include_details, healing_option
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@builtins_base.Builtin
def rank(
    prompt: str | _ChunkList,
    options: Sequence[str | _ChunkList],
    top_k: int = 1,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> Sequence[str] | tuple[Sequence[str], Sequence[float]]:
  """Interface of the choose built-in function.

  Args:
    prompt: The prompt asking for a choice among options.
    options: Possible completions to choose from.
    top_k: The maximum number of options to return (ranked). If top_k=0, all
      options will be ranked and returned.
    include_details: If False, return only the top k options, otherwise return a
      tuple (see returned value below).
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

  Returns:
    The top k options if include_details=False (default), or a tuple of
    (sequence of top k options, sequence of scores for all options).
  """
  del prompt, options, top_k, include_details, healing_option
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


@executing.make_executable
@tracing.trace(name='llm.select')
async def _default_select_via_score(
    prompt: str | _ChunkList,
    options: Sequence[str | _ChunkList],
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> str | tuple[str, int, Sequence[float]]:
  """Default implementation of select via score."""
  # Convert the options into strings.
  options = [str(option) for option in options]
  # Compute the scores.
  scores = await score_text(
      prompt=prompt,
      targets=options,
      healing_option=healing_option,
  )
  max_value = max(scores)
  max_indices = [i for i, score in enumerate(scores) if score == max_value]
  best_index = random.choice(max_indices)
  if include_details:
    return options[best_index], best_index, scores
  else:
    return options[best_index]


@executing.make_executable
@tracing.trace(name='llm.rank')
async def _default_rank_via_score(
    prompt: str | _ChunkList,
    options: Sequence[str | _ChunkList],
    top_k: int = 1,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> Sequence[str] | tuple[Sequence[str], Sequence[float]]:
  """Default implementation of rank via score."""
  # Convert the options into strings.
  options = [str(option) for option in options]
  # Compute the scores.
  scores = await score_text(
      prompt=prompt,
      targets=options,
      healing_option=healing_option,
  )
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


@builtins_base.Builtin
async def generate_object(
    prompt: str | _ChunkList,
    cls: type[_T],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> _T:
  """Interface of the generate_object builtin function.

  Args:
    prompt: The text prompt as a string.
    cls: The object type to generate.
    temperature: Optional temperature parameter (float).
    max_tokens: Optional maximum number of tokens to generate (int).
    top_k: Optional top_k parameter (int).
    top_p: Optional top_p parameter (float).
    healing_option: Type of token healing applied to the prompt (`NONE` by
      default).

  Returns:
    An object decoded from the LLM's constrained decoding.
  """
  del prompt, cls, temperature, max_tokens, top_k, top_p, healing_option
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`.'
      ' This function cannot be called directly.'
  )


def reset_defaults():
  """Resets default implementations for all builtins in this file."""
  # Keep all module level `some_builtin.configure(...)` commands in this method.
  # Default `generate_texts` uses `generate_text`.
  generate_texts.configure(_default_generate_texts)
  # Default `count_tokens` uses `tokenize`.
  count_tokens.configure(_default_count_tokens)
  # Default `instruct` uses `chat`.
  instruct.configure(default_instruct)
  # Default `chat` uses `generate_text`.
  chat.configure(default_chat)
  # Default `select` uses `score_text`.
  select.configure(_default_select_via_score)
  # Default `rank` uses `score_text`.
  rank.configure(_default_rank_via_score)


reset_defaults()
