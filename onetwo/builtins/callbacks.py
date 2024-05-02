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

"""Callback versions of the builtins to use in jinja prompt templates."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

import immutabledict
from onetwo.builtins import llm
from onetwo.core import constants
from onetwo.core import executing
from onetwo.core import results
from onetwo.core import templating
from onetwo.core import tracing


_DRY_RUN_PREFIX_VAR = templating._DRY_RUN_PREFIX_VAR  # pylint: disable=protected-access
PROMPT_PREFIX = templating.PROMPT_PREFIX
PromptTemplateContext = templating.PromptTemplateContext


_T = TypeVar('_T')
_MockType = TypeVar('_MockType')
_EMPTY_MAP = immutabledict.immutabledict({})
CHOICES_VAR = constants.CHOICES_VAR
SCORES_VAR = constants.SCORES_VAR


def _heal_reply(reply):
  """Possibly remove spaces from the beginning of the replies."""
  if isinstance(reply, str):
    reply: str = reply.lstrip(' ')
  elif isinstance(reply, tuple):
    reply, details = reply
    reply = reply.lstrip(' ')
    reply = (reply, details)
  elif isinstance(reply, Sequence):
    # We have a Sequence.
    reply = [_heal_reply(r) for r in reply]
  else:
    raise ValueError(f'Unexpected type of the reply:{type(reply)}')
  return reply


@templating.add_mock_value
@tracing.trace(name='llm', skip=['context', results.MAIN_OUTPUT])
async def llm_callback(
    context: templating.PromptTemplateContext,
    samples: int | None = None,
    space_healing: bool = True,
    mock_value: str | None = None,
    **engine_param_overrides,
) -> str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]]:
  """Callback to call a language model on the current prefix.

  When used in a template, the syntax is equivalent to a function that would
  be defined with only varname and stop as arguments (context is not passed).

  Args:
    context: Context information passed automatically to the function.
    samples: Number of samples to get from the LLM. If None, a single sample is
      returned, if samples is an integer, a list of samples is returned.
    space_healing: If True (default), we apply the "space healing" technique,
      which may resolve the issues related to inadequate generation due to
      spaces in the end of the prompt. When sending the prefix to completion
      space healing removes whitespaces from the end of the prefix. In case the
      spaces are removed and the generated text starts with spaces, it also
      removes all the spaces from the beginning of the generated text.
    mock_value: Mocks the LLM call with mock_value if given. If None (default),
      calls the LLM normally. Note that the mock_value does not result in a
      traced stage to facilitate the use of zip with in-context examples.
      See test_j2_llm_mock_value_with_zipped_examples for an example.
    **engine_param_overrides: list of keyword=value pairs to be passed to the
      engine (they are passed to the constructor of an EngineParams object).

  Returns:
    The content of the Reply.reply field which can be a string or a mapping
    when multiple values are returned (e.g. some language models might return
    both the text output and some additional object with side information
    such as a score).
  """
  # The mock_value is added by the add_mock_value decorator to prevent it
  # from being traced. It is part of the function signature for documentation
  # purposes only (to satisfy pylint).
  del mock_value

  prompt = context.prefix
  if space_healing:
    # Remove the spaces from the end of the prompt.
    prompt = prompt.rstrip(' ')
  # We manually add the prefix as an input even though it is not technically
  # a parameter of the llm function.
  tracing.execution_context.get().inputs['request'] = prompt

  if samples is None:
    executable = llm.generate_text(
        prompt, include_details=True, **engine_param_overrides
    )
    assert isinstance(executable, executing.FunctionExecWrapper), (
        f'Unexpected type of the executable:{type(executable)}. This means the'
        ' generate_text builtin was not properly configured (it'
        ' should be wrapped into make_executable)'
    )
    reply = await executable.pre_execute()

    if isinstance(reply, executing.Executable):
      context.iterable_reply = reply
      # Once we set the _iterable_reply to something different from None, it
      # will be picked up by the _execute_reply or _iterate_through_reply
      # methods to be processed, so we just have to yield control and wait for
      # _iterable_reply to be set back to None (indicating that the processing
      # is finished) before continuing.
      while context.iterable_reply is not None:
        await asyncio.sleep(0)  # Yield control to the reply iteration process.
      # Hint for the type checker.
      assert context.iterated_reply is not None
      # The processing of the _iterable_reply is finished, we expect the final
      # result to be stored in _iterated_reply from which we pick it up.
      reply = context.iterated_reply

  else:
    replies = await llm.generate_texts(
        prompt, samples, include_details=True, **engine_param_overrides,
    )
    replies = list(replies)
    # In the repeat mode we don't stream the requests back so we execute them
    # fully before returning.
    for index, reply in enumerate(replies):
      if isinstance(reply, executing.Executable):
        replies[index] = await reply
    reply = replies

  if space_healing and context.prefix.endswith(' '):
    # We may need to remove spaces from the beginning of the replies.
    reply = _heal_reply(reply)
  details = reply[1] if samples is None else [r[1] for r in reply]
  reply = reply[0] if samples is None else [r[0] for r in reply]

  if isinstance(details, Mapping):
    tracing.execution_context.get().outputs.update(details)
  else:  # We have a str or Sequence.
    tracing.execution_context.get().outputs[results.MAIN_OUTPUT] = details

  return reply


@templating.add_mock_value
@tracing.trace(name='generate_text', skip=['context', results.MAIN_OUTPUT])
async def generate_text(
    context: templating.PromptTemplateContext,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    mock_value: str | None = None,
) -> str:
  """See llm.generate_text."""
  # The mock_value is added by the add_mock_value decorator to prevent it
  # from being traced. It is part of the function signature for documentation
  # purposes only (to satisfy pylint).
  del mock_value

  prefix = context.prefix
  # We manually add the prefix as an input even though it is not technically
  # a parameter of the choose function.
  tracing.execution_context.get().inputs['request'] = prefix

  reply = await llm.generate_text(
      prefix,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
      top_k=top_k,
      top_p=top_p,
      decoding_constraint=decoding_constraint,
      include_details=False,  # We just want the string, no details.
  )
  reply = cast(str, reply)
  tracing.execution_context.get().outputs[results.MAIN_OUTPUT] = reply
  return reply


@templating.add_mock_value
@tracing.trace(name='generate_texts', skip=['context', results.MAIN_OUTPUT])
async def generate_texts(
    context: templating.PromptTemplateContext,
    samples: int = 1,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
    mock_value: str | None = None,
) -> Sequence[str]:
  """See llm.generate_texts."""
  # The mock_value is added by the add_mock_value decorator to prevent it
  # from being traced. It is part of the function signature for documentation
  # purposes only (to satisfy pylint).
  del mock_value

  prefix = context.prefix
  # We manually add the prefix as an input even though it is not technically
  # a parameter of the choose function.
  tracing.execution_context.get().inputs['request'] = prefix

  reply = await llm.generate_texts(
      prefix,
      samples=samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
      top_k=top_k,
      top_p=top_p,
      decoding_constraint=decoding_constraint,
      include_details=False,  # We just want the string, no details.
  )
  tracing.execution_context.get().outputs[results.MAIN_OUTPUT] = reply
  return reply  # pytype: disable=bad-return-type


# TODO: Add tracing for cls using a serializable type
@templating.add_mock_value
@tracing.trace(name='generate_object',
               skip=['context', 'cls', results.MAIN_OUTPUT])
async def generate_object(
    context: templating.PromptTemplateContext,
    cls: type[_T],
    space_healing: bool = True,
    mock_value: _T | None = None,
    **engine_param_overrides,
) -> _T:
  """Callback to call a language model on the current prefix and generate an object from the response.

  Similar to `_llm_callback` callback, but uses constrained decoding to
  constrain the output to the json description of `cls`.
  Useful when trying to generate an instance of an object to continue the
  flow of a python project.

  Args:
    context: Context information passed automatically to the function.
    cls: The type of object to generate. See `llm.generate_object`() for
      supported types.
    space_healing: If True (default), we apply the "space healing" technique,
      which may resolve the issues related to inadequate generation due to
      spaces in the end of the prompt. When sending the prefix to completion
      space healing removes whitespaces from the end of the prefix. In case the
      spaces are removed and the generated text starts with spaces, it also
      removes all the spaces from the beginning of the generated text.
    mock_value: Mocks the LLM call with mock_value if given. If None (default),
      calls the LLM normally. Note that the mock_value does not result in a
      traced stage to facilitate the use of zip with in-context examples. See
      test_j2_llm_mock_value_with_zipped_examples for an example.
    **engine_param_overrides: list of keyword=value pairs to be passed to the
      engine (they are passed to the constructor of an EngineParams object).

  Returns:
    The content of the Reply.reply field which can be a string or a mapping
    when multiple values are returned (e.g. some language models might return
    both the text output and some additional object with side information
    such as a score).
  """
  # The mock_value is added by the add_mock_value decorator to prevent it
  # from being traced. It is part of the function signature for documentation
  # purposes only (to satisfy pylint).
  del mock_value

  prompt = context.prefix
  if space_healing:
    # Remove the spaces from the end of the prompt.
    prompt = prompt.rstrip(' ')
  # We manually add the prefix as an input even though it is not technically
  # a parameter of the llm function.
  tracing.execution_context.get().inputs['request'] = prompt
  reply = await llm.generate_object(prompt, cls, **engine_param_overrides)

  tracing.execution_context.get().outputs[results.MAIN_OUTPUT] = reply
  return reply


@templating.add_mock_value
@tracing.trace(name='choose', skip=['context', results.MAIN_OUTPUT])
async def choose(
    context: templating.PromptTemplateContext,
    candidates: Sequence[str],
    top_k: int = 1,
    space_healing: bool = True,
    mock_value: str | None = None,
) -> str:
  """Callback to choose from a set of completions given the context.

  Similar to `_llm_callback` callback, but using `score` instead of `complete`
  method. Useful when we know the options we want to choose from. Reduces the
  generation problem to the classification problem.

  Updates context variables with a dictionary `('choices': choices, 'scores':
  scores)`, where `choices` is a list containing `top_k` highest scored elements
  among `candidates` sorted in descreasing order and `scores` is a list of the
  same length as `candidates`, where `scores[j]` contains the score of
  `candidates[j]`. For example:
    _choose_callback(context, ['yes', 'no', 'abstain'], 1)
  may return:
    {'choices': ['yes'], 'scores': [-0.356, -1.609, -2.302]},
  while
    _choose_callback(context, ['red', 'green', 'blue', 'orange'], 2)
  may return
    {'choices': ['orange', 'green'],
      'scores': [-2.995, -1.609, -1.897, -0.510]}.

  Args:
    context: Context information passed automatically to the function.
    candidates: List of candidate completions we are choosing from.
    top_k: We score all `candidates`, sort them according to the
      log-probabilities and return `top_k` top scoring options.
    space_healing: refer to _llm_callback.
    mock_value: Mocks the LLM call with mock_value if given. If None (default),
      calls the LLM normally. Note that the mock_value does not result in a
      traced stage to facilitate the use of zip with in-context examples. See
      test_j2_choose_callback_mock_value_with_zipped_examples for an example.

  Returns:
    The highest scoring option among `candidates`.
  """
  # The mock_value is added by the add_mock_value decorator to prevent it
  # from being traced. It is part of the function signature for documentation
  # purposes only (to satisfy pylint).
  del mock_value

  prefix = context.prefix
  if space_healing:
    # Remove the spaces from the end of the prompt.
    prefix = prefix.rstrip(' ')
  # We manually add the prefix as an input even though it is not technically
  # a parameter of the choose function.
  tracing.execution_context.get().inputs['request'] = prefix

  choices, scores = await llm.rank(
      prefix, candidates, top_k=top_k, include_details=True
  )

  context_variables = {CHOICES_VAR: choices, SCORES_VAR: scores}
  context.context_variables.update(context_variables)
  tracing.execution_context.get().outputs.update(context_variables)
  return choices[0]


def mock_llm_callback(
    context: templating.PromptTemplateContext,
    default_reply: str,
    **engine_param_overrides
) -> str:
  """Mock `llm` operation."""
  # We don't apply space healing to mocked callbacks.
  del engine_param_overrides
  if _DRY_RUN_PREFIX_VAR not in context.output_variables:
    context.output_variables[_DRY_RUN_PREFIX_VAR] = {}
  if 'llm' not in context.output_variables[_DRY_RUN_PREFIX_VAR]:
    context.output_variables[_DRY_RUN_PREFIX_VAR]['llm'] = []
  # Store the prefix in outputs.
  context.output_variables[_DRY_RUN_PREFIX_VAR]['llm'].append(context.prefix)
  return default_reply


def mock_generate_object_callback(
    context: templating.PromptTemplateContext,
    cls: type[_T],
    default_type_to_object_map: Mapping[
        type[_MockType], _MockType
    ] = _EMPTY_MAP,
    **engine_param_overrides,
) -> _T:
  """Mock `generate_object` operation."""
  # We don't apply space healing to mocked callbacks.
  del engine_param_overrides
  if _DRY_RUN_PREFIX_VAR not in context.output_variables:
    context.output_variables[_DRY_RUN_PREFIX_VAR] = {}
  if 'generate_object' not in context.output_variables[_DRY_RUN_PREFIX_VAR]:
    context.output_variables[_DRY_RUN_PREFIX_VAR]['generate_object'] = []
  # Store the prefix in outputs.
  context.output_variables[_DRY_RUN_PREFIX_VAR]['generate_object'].append(
      context.prefix
  )
  assert (
      cls in default_type_to_object_map
  ), f'No default instance was supplied for object type: {cls}'
  return default_type_to_object_map[cls]


def mock_choose_callback(
    context: templating.PromptTemplateContext,
    candidates: Sequence[str],
    top_k: int = 1,
) -> str:
  """Mock `choose` operation."""
  # We don't apply space healing to mocked callbacks.
  if _DRY_RUN_PREFIX_VAR not in context.output_variables:
    context.output_variables[_DRY_RUN_PREFIX_VAR] = {}
  if 'choose' not in context.output_variables[_DRY_RUN_PREFIX_VAR]:
    context.output_variables[_DRY_RUN_PREFIX_VAR]['choose'] = []
  # Store the prefix in outputs.
  context.output_variables[_DRY_RUN_PREFIX_VAR]['choose'].append(context.prefix)
  # Populate "choices" and "scores" context variables.
  num_candidates = len(candidates)
  choices = candidates[:top_k]
  context.context_variables.update({
      CHOICES_VAR: choices,
      SCORES_VAR: num_candidates * [0.0]
  })
  return choices[0]
