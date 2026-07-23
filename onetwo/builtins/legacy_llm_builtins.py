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

# Copyright 2024 Google LLC
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

"""Legacy LLM builtins (generate_text and generate_texts).

These builtins are kept for backward compatibility. New code should prefer
using llm.chat or llm.generate_content.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, cast, overload

from onetwo.builtins import builtins_base
from onetwo.builtins import llm_utils
from onetwo.core import composing
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling

_ChunkList: TypeAlias = content_lib.ChunkList
_Chunk: TypeAlias = content_lib.Chunk
TokenHealingOption: TypeAlias = llm_utils.TokenHealingOption
_PromptType: TypeAlias = (
    str | _ChunkList | Sequence[_Chunk] | composing.Composable
)

_UNIMPLEMENTED_ERROR = NotImplementedError(
    'Implementation should be provided by configuring the builtin at runtime'
    ' or `get_variant`. This function cannot be called directly.'
)


class GenerateTextCallable(object):
  """Type representation for generate_text builtin (for overloads)."""

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      *,
      include_details: Literal[False] = False,
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[str]: ...

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      *,
      include_details: Literal[True],
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[tuple[str, dict[str, Any]]]: ...

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      *,
      include_details: bool,
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[str | tuple[str, dict[str, Any]]]: ...

  def __call__(
      self,
      prompt: _PromptType,
      *,
      include_details: bool = False,
      **kwargs: Any,
  ) -> Any:
    return cast(Any, None)

  @property
  def name(self) -> str:
    return cast(Any, None)

  @property
  def _name(self) -> str:
    return cast(Any, None)

  def is_configured(self) -> bool:
    return cast(Any, None)

  def configure(
      self,
      implementation: Callable[..., Any],
      *args: Any,
      **kwargs: Any,
  ) -> None:
    pass

  def update(self, **kwargs: Any) -> None:
    pass


class GenerateTextsCallable(object):
  """Type representation for generate_texts builtin (for overloads)."""

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      samples: int = 1,
      *,
      include_details: Literal[False] = False,
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[Sequence[str]]: ...

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      samples: int = 1,
      *,
      include_details: Literal[True],
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[
      Sequence[tuple[str, dict[str, Any]]]
  ]: ...

  @overload
  def __call__(
      self,
      prompt: _PromptType,
      samples: int = 1,
      *,
      include_details: bool,
      **kwargs: Any,
  ) -> executing.FunctionExecWrapper[
      Sequence[str | tuple[str, dict[str, Any]]]
  ]: ...

  def __call__(
      self,
      prompt: _PromptType,
      samples: int = 1,
      *,
      include_details: bool = False,
      **kwargs: Any,
  ) -> Any:
    return cast(Any, None)

  @property
  def name(self) -> str:
    return cast(Any, None)

  @property
  def _name(self) -> str:
    return cast(Any, None)

  def is_configured(self) -> bool:
    return cast(Any, None)

  def configure(
      self,
      implementation: Callable[..., Any],
      *args: Any,
      **kwargs: Any,
  ) -> None:
    pass

  def update(self, **kwargs: Any) -> None:
    pass


@builtins_base.Builtin[str | tuple[str, dict[str, Any]]]
async def _generate_text(
    prompt: str | _ChunkList | Sequence[_Chunk],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    include_details: bool = False,
    healing_option: TokenHealingOption = TokenHealingOption.NONE,
) -> str | tuple[str, dict[str, Any]]:
  """Interface of the generate_text built-in function."""
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
  raise _UNIMPLEMENTED_ERROR


@builtins_base.Builtin
def _generate_texts(
    prompt: str | _ChunkList | Sequence[_Chunk],
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
) -> Sequence[str | tuple[str, dict[str, Any]]]:
  """Interface of the generate_texts built-in function."""
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
  raise _UNIMPLEMENTED_ERROR


generate_text = cast(GenerateTextCallable, _generate_text)
generate_texts = cast(GenerateTextsCallable, _generate_texts)

# pytype: disable=protected-access
_generate_text._name = 'onetwo.builtins.llm.generate_text'  # pylint: disable=protected-access
_generate_texts._name = 'onetwo.builtins.llm.generate_texts'  # pylint: disable=protected-access
# pytype: enable=protected-access


@executing.make_executable
def _default_generate_texts(
    prompt: str | _ChunkList | Sequence[_Chunk],
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
) -> Sequence[str | tuple[str, dict[str, Any]]]:
  """Default implementation via repeatedly calling generate_text.

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
  return executing.par_iter(executables)  # pytype: disable=bad-return-type


def reset_defaults() -> None:
  _generate_texts.configure(_default_generate_texts)
