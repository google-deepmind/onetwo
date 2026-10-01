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
import enum
import random
from typing import Any, Final, TypeAlias, TypeVar

from onetwo.builtins import builtins_base
from onetwo.builtins import formatting
from onetwo.builtins import legacy_llm_builtins
from onetwo.builtins import llm_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling
from onetwo.core import routing
from onetwo.core import tracing

# For backward compatibility.
# New code should use llm.chat or llm.generate_content.
generate_texts = legacy_llm_builtins.generate_texts
generate_text = legacy_llm_builtins.generate_text

_T = TypeVar('_T')

_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList
_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole
TokenHealingOption: TypeAlias = llm_utils.TokenHealingOption


class EmbedTaskType(enum.StrEnum):
  RETRIEVAL_QUERY = 'RETRIEVAL_QUERY'
  RETRIEVAL_DOCUMENT = 'RETRIEVAL_DOCUMENT'
  SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY'
  CLASSIFICATION = 'CLASSIFICATION'
  CLUSTERING = 'CLUSTERING'
  CODE_RETRIEVAL_QUERY = 'CODE_RETRIEVAL_QUERY'
  QUESTION_ANSWERING = 'QUESTION_ANSWERING'
  FACT_VERIFICATION = 'FACT_VERIFICATION'


# Static instruction prefixes that the Gemini embedding models expect for each
# task type. `RETRIEVAL_DOCUMENT` is excluded because its prefix depends on the
# runtime `title` value, see `prefix_for_task_type`.
PREFIX_FROM_TASK_TYPE: Final[Mapping[EmbedTaskType | str, str]] = {
    EmbedTaskType.RETRIEVAL_QUERY: 'task: search result | query: ',
    EmbedTaskType.SEMANTIC_SIMILARITY: 'task: sentence similarity | query: ',
    EmbedTaskType.CLASSIFICATION: 'task: classification | query: ',
    EmbedTaskType.CLUSTERING: 'task: clustering | query: ',
    EmbedTaskType.CODE_RETRIEVAL_QUERY: 'task: code retrieval | query: ',
    EmbedTaskType.QUESTION_ANSWERING: 'task: question answering | query: ',
    EmbedTaskType.FACT_VERIFICATION: 'task: fact checking | query: ',
}


def prefix_for_task_type(
    task_type: EmbedTaskType | str | None,
    title: str | None = None,
) -> str | None:
  """Returns the embedding instruction prefix for a task type, if any.

  Args:
    task_type: The task type for the embedding, or None if unspecified. Passing
      a `title` without a `task_type` is treated as `RETRIEVAL_DOCUMENT`.
    title: Optional title of the content, only used for `RETRIEVAL_DOCUMENT`.

  Returns:
    The prefix to prepend to the content, or None if no prefix applies.
  """
  if task_type == EmbedTaskType.RETRIEVAL_DOCUMENT or (
      task_type is None and title is not None
  ):
    return f'title: {title or "none"} | text: '
  if task_type is None:
    return None
  return PREFIX_FROM_TASK_TYPE.get(task_type)


_UNIMPLEMENTED_ERROR = NotImplementedError(
    'The implementation should be provided at runtime by calling `configure`'
    ' or `get_variant`. This function cannot be called directly.'
)

# A key for the `include_details` where thoughts can be found
THOUGHTS = 'thoughts'


Prompt = str | Sequence[_Chunk] | _ChunkList | Sequence[_Message]
Content = _ChunkList | tuple[_ChunkList, Mapping[str, Any]]


@builtins_base.Builtin[Content]
async def generate_content(
    prompt: Prompt,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> Content:
  """Interface of the generate_content built-in function.

  Args:
    prompt: The model prompt (as string, chunks or multi-turn messages).
    temperature: Optional temperature parameter.
    max_tokens: Optional maximum number of tokens to generate.
    stop: Optional Sequence of strings on which to stop the generation.
    top_k: Optional top_k parameter.
    top_p: Optional top_p parameter.
    include_details: If True, the result will be a tuple with a Mapping
      containing additional (backend-specific) information.
    healing_option: Type of token healing applied to the prompt.

  Returns:
    A sequence of answers. Each answer is a string (text-only) or a list of
    Chunks (if multimodal), as pairs (answer, details) if include_details=True.
  """
  del (prompt, temperature, max_tokens, stop, top_k, top_p)
  del (include_details, healing_option)
  raise _UNIMPLEMENTED_ERROR


@executing.make_executable
def echo_generate_text(
    prompt: str | _ChunkList, **kwargs
) -> str | tuple[str, Mapping[str, Any]]:
  """Implementation of generate_text that simply returns the prompt."""
  if isinstance(prompt, str):
    return prompt
  return '', {'prompt': prompt, 'kwargs': kwargs}


@builtins_base.Builtin
def tokenize(content: str | _ChunkList) -> Sequence[int]:
  """Interface of the tokenize_text built-in function.

  Args:
    content: The content (string or ChunkList) to be tokenized.

  Returns:
    A Sequence of token ids.
  """
  del content
  raise _UNIMPLEMENTED_ERROR


@builtins_base.Builtin
def detokenize(tokens: Sequence[int]) -> str | _ChunkList:
  """Interface of the detokenize built-in function.

  Args:
    tokens: The tokens (Sequence[int]) to be detokenized.

  Returns:
    The detokenized string or ChunkList.
  """
  del tokens
  raise _UNIMPLEMENTED_ERROR


@builtins_base.Builtin
def count_tokens(content: str | _ChunkList) -> int:
  """Interface of the count_tokens built-in function.

  Args:
    content: The content (string or ChunkList) to be tokenized.

  Returns:
    The length of the tokenized string (number of tokens).
  """
  del content
  raise _UNIMPLEMENTED_ERROR


@executing.make_executable
async def _default_count_tokens(
    content: str | _ChunkList,
) -> int:
  tokens = await tokenize(content)
  return len(tokens)


@builtins_base.Builtin
def embed(
    content: str | _ChunkList,
    task_type: EmbedTaskType | str | None = None,
    title: str | None = None,
    *,
    normalize: bool = True,
    **kwargs,
) -> Sequence[float]:
  """Interface of the embed built-in function.

  Implementations must return an L2-normalized (unit norm) embedding unless
  `normalize` is False, so that dot products between embeddings are cosine
  similarities. Since the defaults of this signature are not passed to the
  implementations, implementations must declare `normalize: bool = True`
  themselves. They should normalize with `utils.l2_normalize_embedding` (from
  `onetwo.core`) after any caching, so that caches hold the raw model output.

  Args:
    content: The content (string or ChunkList) to be embedded.
    task_type: The standard task type for the embedding.
    title: Optional title of the content. Titles are useful for certain task
      types. For example the gemini-embedding-2 model formats the title and
      content jointly into "title: {title} | text: {content}" if the task type
      RETRIEVAL_DOCUMENT is passed, see
      https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2.
    normalize: If True (default), the returned embedding is L2-normalized. Set
      to False to get the raw output of the model.
    **kwargs: Optional arguments to be passed to the embed function.

  Returns:
    A Sequence of floats, with unit L2 norm if `normalize` is True.
  """
  del content, task_type, title, normalize, kwargs
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


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
    formatter_instance = formatter_class(formatter_kwargs)
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


# TODO: Considering renaming this built-in function to
# `chat_with_tools`.
@builtins_base.Builtin
def generate_with_tools(
    messages: Sequence[function_calling.ConversationTurn],
    tools: Sequence[function_calling.ToolSpec],
    **kwargs,
) -> function_calling.ConversationTurn:
  """Interface of the generate_with_tools built-in function.

  Performs a single model call in which the model may request tool calls. This
  is deliberately one step rather than a loop: the loop that executes the
  requested calls and feeds the results back lives in
  `onetwo.agents.agent.Agent`, so that it is written once rather
  than once per backend.

  Implementations must:
    * return exactly one turn, with `role=MODEL`;
    * not execute any tool, which is why `tools` are `ToolSpec`s and carry no
      code;
    * accept a history that contains `TOOL` turns, since that is how the
      results of previous calls are fed back;
    * populate `FunctionCall.id` whenever the underlying API supplies one, so
      that parallel calls can be matched to their responses;
    * round-trip `ConversationTurn.thought` and
      `FunctionCall.thought_signature`, since a thinking model needs the
      signatures it issued back verbatim, and Gemini rejects a turn whose
      earlier calls have lost theirs.

  Args:
    messages: The conversation so far. The last turn is typically a USER turn or
      a TOOL turn carrying the results of the calls the model just made.
    tools: Declarations of the tools the model may call. An empty sequence means
      that no tools are advertised to the model.
    **kwargs: Generation parameters (e.g. `temperature`, `max_tokens`, `stop`).

  Returns:
    A single MODEL turn, which may contain text content, a thought, and/or one
    or more function calls.
  """
  del messages, tools, kwargs
  raise _UNIMPLEMENTED_ERROR


@executing.make_executable
async def _default_generate_with_tools(
    messages: Sequence[function_calling.ConversationTurn],
    tools: Sequence[function_calling.ToolSpec],
    **kwargs,
) -> function_calling.ConversationTurn:
  del messages, tools, kwargs
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


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
  raise _UNIMPLEMENTED_ERROR


def reset_defaults(
    reset_generate: bool = True,
    reset_tokenize: bool = True,
    reset_embed: bool = True,
):
  """Resets default implementations for all builtins in this file.

  Args:
    reset_generate: Whether to reset the default implementations of all
      generation related builtins (e.g., `generate_text`, `generate_texts`,
      `chat`, `instruct`, `select`, `rank`).
    reset_tokenize: Whether to reset the default implementation of tokenization
      related builtins (e.g., `count_tokens`).
    reset_embed: Whether to reset the default implementation of embed related
      builtins (e.g., `embed`).
  """
  # Keep all module level `some_builtin.configure(...)` commands in this method.
  if reset_generate:
    legacy_llm_builtins.reset_defaults()
    # Default `instruct` uses `chat`.
    instruct.configure(default_instruct)
    # Default `chat` uses `generate_text`.
    chat.configure(default_chat)
    generate_with_tools.configure(_default_generate_with_tools)
    # Default `select` uses `score_text`.
    select.configure(_default_select_via_score)
    # Default `rank` uses `score_text`.
    rank.configure(_default_rank_via_score)

  if reset_tokenize:
    # Default `count_tokens` uses `tokenize`.
    count_tokens.configure(_default_count_tokens)

  if reset_embed:
    # No default implementation for embed.
    pass


reset_defaults()
