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

"""Composable versions of builtin functions."""

from collections.abc import Sequence
from typing import Any, TypeVar

from onetwo.builtins import llm
from onetwo.builtins import prompt_templating
from onetwo.core import composing
from onetwo.core import content as content_lib


_T = TypeVar('_T')

store = composing.store
section_start = composing.section_start
section_end = composing.section_end


@composing.make_composable
def f(
    context: composing.Context, content: str, role: content_lib.RoleType = None
) -> content_lib.Chunk:
  """Composable version of the string.format function that uses context vars."""
  try:
    content = content.format(**context.variables)
  except KeyError as e:
    raise ValueError(
        f'Could not format {content} with context'
        f' {context.variables}, some variables were not found.'
    ) from e
  return content_lib.Chunk(content, role=role)


@composing.make_composable
def c(
    context: composing.Context, content: Any, role: content_lib.RoleType = None
) -> content_lib.ChunkList:
  """Composable chunk created from content."""
  del context
  return content_lib.ChunkList([content_lib.Chunk(content, role=role)])


@composing.make_composable
async def j(
    context: composing.Context,
    template: str,
    name: str = 'JinjaTemplate',
    role: content_lib.RoleType = None,
) -> content_lib.Chunk:
  """Composable version of a jinja formatted prompt using context vars."""
  result = await prompt_templating.JinjaTemplateWithCallbacks(
      name=name, text=template
  ).render(**context.variables)
  # We extract the output variables from the template and store them into the
  # context.
  for key, value in result.items():
    if key != prompt_templating.PROMPT_PREFIX:
      context[key] = value
  return content_lib.Chunk(result[prompt_templating.PROMPT_PREFIX], role=role)


@composing.make_composable
def generate_text(context: composing.Context, **kwargs) -> ...:
  """Composable version of the llm.generate_text function."""
  return llm.generate_text(context.prefix, **kwargs)


@composing.make_composable
async def chat(
    context: composing.Context,
    **kwargs,
) -> content_lib.ChunkList:
  """Composable version of the llm.chat function."""
  return content_lib.ChunkList([
      content_lib.Chunk(
          await llm.chat(context.to_messages(), **kwargs),
          role=content_lib.PredefinedRole.MODEL,
      )
  ])


@composing.make_join_composable  # pytype: disable=wrong-arg-types
async def select(
    context: composing.Context,
    options: Sequence[tuple[str, composing.Context]],
) -> tuple[str, composing.Context]:
  """Composable select function.

  Args:
    context: Context of execution.
    options: Sequence of options. Each option is a pair (text, context).

  Returns:
    The pair (text, context) that is selected among the possible options. For
    example, this could be the one with the highest score.
  """
  common_prefix = context.prefix
  text_options = [text for text, _ in options]
  value, index, _ = await llm.select(
      common_prefix, text_options, include_details=True
  )
  return value, options[index][1]


@composing.make_composable
def generate_object(
    context: composing.Context, cls: type[Any], **kwargs
) -> ...:
  """Composable version of the llm.generate_object function."""
  return llm.generate_object(context.prefix, cls, **kwargs)


@composing.make_composable
async def instruct(
    context: composing.Context,
    assistant_prefix: str | None = None,
    **kwargs
) -> ...:
  """Composable version of the llm.instruct function."""
  # TODO: We are awaiting llm.instruct which means we cannot
  # iterate through the result if we are doing streaming. We should instead
  # wrap this into an ExecutableWithPostprocessing which adds the
  # assistant_prefix, or change the prefix directly and call instruct with
  # the unchanged prefix (requires to deep-copy the prefix).
  result = await llm.instruct(context.prefix, assistant_prefix, **kwargs)
  if assistant_prefix is None:
    return result
  else:
    return assistant_prefix + result
