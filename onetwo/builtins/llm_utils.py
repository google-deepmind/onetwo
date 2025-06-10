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

"""Technical details and utilities for llm builtins."""

from __future__ import annotations


from collections.abc import Mapping, Sequence
import enum
from typing import cast, Any, TypeAlias

from onetwo.core import content as content_lib

_ChunkList: TypeAlias = content_lib.ChunkList


class TokenHealingOption(enum.Enum):
  """Options for token healing.

  Token healing is a technique that can often significantly improve generation
  results (or rather help to avoid erroneous results). Consider prompts ending
  with a whitespace, e.g., "Theory of ". Depending on the model and the
  tokenizer, this string is likely to be encoded into a token sequence where the
  last element is a whitespace token (WT) corresponding to a single whitespace "
  ". The model will then generate next tokens, following WT. An issue with this
  scenario is that words are often tokenized together with the leading
  whitespace. Likely, there is " relativity" token (RT) and for the majority of
  sentenses where the word "relativity" (case-sensitive!) occured, it was
  encoded into RT. So the model may be very "familiar" with RT, but not with
  "relativity" token (no space). Unfortunately, a separate whitespace token
  already appears in the end of the sequence, so the model can't generate RT
  (indeed, two whitespaces in a row is very uncommon, so the model won't do it)
  and instead generates something else (that often looks odd). This phenomenon
  is more general and can occur with other tokens in the end of the prompt as
  well. For more details refer to https://github.com/guidance-ai/guidance.
  """
  NONE = 'NONE'
  # In token healing we replace the last token of the prompt and send the prompt
  # to completion. In the ideal scenario where constrained decoding is available
  # we make sure that the string representation of the first token that we
  # generate contains the string representation of the removed token as a
  # prefix. This way we guarantee that the completion matches user's
  # expectations and we avoid token boundary artifacts at the same time. If the
  # constrained decoding is not available for the model we sample completions
  # in hope that the constraint will be satisfied and roll back to vanilla
  # completion in case of failure.
  TOKEN_HEALING = 'TOKEN_HEALING'
  # In space healing we remove trailing whitespaces from the end of the prompt
  # and send the prompt to completion. In case the prompt has trailing
  # whitespaces we also remove any leading whitespaces from the generated text.
  # This may be considered as a less general ad-hoc implementation for the
  # token healing.
  SPACE_HEALING = 'SPACE_HEALING'


def space_heal_reply(
    reply: str | tuple[str, Mapping[str, Any]]
) -> str | tuple[str, Mapping[str, Any]]:
  """Possibly remove spaces from the beginning of the reply.

  Args:
    reply: Reply that we want to space heal. Reply is either a string or a tuple
      of a string and a dictionary of additional information.

  Returns:
    Reply with leading whitespaces removed. In case of a tuple, the additional
    information is preserved.
  """
  if isinstance(reply, str):
    reply: str = reply.lstrip(' ')
  elif isinstance(reply, tuple):
    reply, details = reply
    reply = reply.lstrip(' ')
    reply = (reply, details)
  else:
    raise ValueError(f'Unexpected type of the reply:{type(reply)}')
  return reply


def maybe_heal_prompt(
    *,
    original_prompt: str | _ChunkList,
    healing_option: TokenHealingOption,
) -> _ChunkList:
  """Maybe heal the prompt for further generation.

  Args:
    original_prompt: Prompt that we may want to heal.
    healing_option: Healing option that we want to use.

  Returns:
    In case we use space healing, we remove trailing whitespaces from the
    prompt. The token healing is not supported yet. Otherwise, we return the
    prompt as is.
  """
  if isinstance(original_prompt, str):
    original_prompt = _ChunkList([original_prompt])
  healed_prompt: _ChunkList = original_prompt
  if healing_option == TokenHealingOption.SPACE_HEALING:
    # Remove trailing whitespaces.
    healed_prompt = healed_prompt.rstrip(' ')
  elif healing_option == TokenHealingOption.TOKEN_HEALING:
    # TODO: Support token healing.
    raise NotImplementedError('Token healing is not supported yet.')
  return healed_prompt


def maybe_heal_reply(
    *,
    reply_text: str,
    original_prompt: str | _ChunkList,
    healing_option: TokenHealingOption,
) -> str:
  """Maybe heal the reply.

  Args:
    reply_text: Reply text that we may want to heal.
    original_prompt: The prompt that was used to generate the reply before any
      type of healing was applied to it.
    healing_option: Healing option that was used to generate the reply.

  Returns:
    In case we used space healing and the original prompt ended with a
    whitespace, we return the reply with leading whitespaces removed. The token
    healing is not supported yet. Otherwise, we return the reply as is.
  """
  if (
      healing_option == TokenHealingOption.SPACE_HEALING
      and original_prompt.endswith(' ')
  ):
    # We need to remove spaces from the beginning of the replies.
    return cast(str, space_heal_reply(reply_text))
  elif healing_option == TokenHealingOption.TOKEN_HEALING:
    # TODO: Support token healing.
    raise NotImplementedError('Token healing is not supported yet.')
  return reply_text


def maybe_heal_prompt_and_targets(
    *,
    original_prompt: str | _ChunkList,
    original_targets: Sequence[str],
    healing_option: TokenHealingOption,
) -> tuple[_ChunkList, Sequence[str]]:
  """Maybe heal the prompt and targets for further scoring.

  Args:
    original_prompt: Prompt that we may want to heal.
    original_targets: Targets that we may want to heal.
    healing_option: Healing option that we want to use.

  Returns:
    In case we use space healing, we remove trailing whitespaces from the
    prompt and add leading whitespaces to the targets when necessary. The token
    healing is not supported yet. Otherwise, we return the prompt and targets as
    is.
  """
  if isinstance(original_prompt, str):
    original_prompt = _ChunkList([original_prompt])
  healed_prompt: _ChunkList = original_prompt
  healed_targets = list(original_targets)
  if healing_option == TokenHealingOption.SPACE_HEALING:
    if healed_prompt.endswith(' '):
      # Remove trailing whitespaces.
      healed_prompt = healed_prompt.rstrip(' ')
      # Add leading whitespaces to the targets if necessary.
      for i, target in enumerate(healed_targets):
        if not target.startswith(' '):
          healed_targets[i] = ' ' + target
  elif healing_option == TokenHealingOption.TOKEN_HEALING:
    # TODO: Support token healing.
    raise NotImplementedError('Token healing is not supported yet.')
  return healed_prompt, healed_targets
