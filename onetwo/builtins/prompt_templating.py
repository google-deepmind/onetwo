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

"""Library for executing a prompt template."""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import functools
from typing import Any, TypeVar

import immutabledict
from onetwo.builtins import callbacks
from onetwo.core import executing
from onetwo.core import templating


_DRY_RUN_PREFIX_VAR = templating._DRY_RUN_PREFIX_VAR  # pylint: disable=protected-access
PROMPT_PREFIX = templating.PROMPT_PREFIX

_T = TypeVar('_T')
_EMPTY_MAP = immutabledict.immutabledict({})


@dataclasses.dataclass
class JinjaTemplateWithCallbacks(templating.JinjaTemplate):
  """A Jinja2 template augmented with additional LLM-specific callbacks."""

  def __post_init__(self):
    """See parent class."""
    super().__post_init__()
    self.register_callback('llm', callbacks.llm_callback, pass_context=True)
    # Also registering the llm callback as 'generate_text' for compatibility
    # with builtins and composables.
    # TODO: deprecate the llm version and make generate_text support
    # dry run.
    self.register_callback(
        'generate_text', callbacks.generate_text, pass_context=True
    )
    self.register_callback(
        'generate_texts', callbacks.generate_texts, pass_context=True
    )
    self.register_callback('choose', callbacks.choose, pass_context=True)
    self.register_callback(
        'generate_object', callbacks.generate_object, pass_context=True
    )

  def _postprocess_iterable_reply(self, iterable_reply: Any) -> str:
    if isinstance(iterable_reply, tuple) and len(iterable_reply) == 2:
      # This is the case where we got a reply with details. We just use the
      # first part.
      # Note that this is specific to the generate_text reply so it may
      # not work for arbitrary streaming callbacks.
      return iterable_reply[0]
    else:
      return super()._postprocess_iterable_reply(iterable_reply)

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def dry_run(
      self,
      inputs: Mapping[str, Any],
      llm_default_reply: str = 'default',
      generate_object_default_reply_map: Mapping[type[_T], _T] = _EMPTY_MAP,
      mock_callbacks: (
          list[tuple[str, Callable[[Any], Any], bool]] | None
      ) = None,
  ) -> Mapping[str, Any]:
    """Dry runs the prompt and returns prefixes sent to the LLM.

    This method renders the jinja2 prompt and gets the prefixes that
    are sent to the language model without actually executing the LLM requests.
    By default this method mocks the `llm` and `choose` operations.
    The mocked `llm` simply returns the `llm_default_reply` provided by the
    user, while the mocked `choose` chooses the first `top_k` provided options
    and populates scores with all `0.0`.

    The user can also mock other callbacks. This can be useful when there is a
    callback that sends LLM requests within itself, eg. by defining and
    executing its own PromptTemplate instance. In this case the callback
    functions provided by the user can store prefixes in the
    `_DRY_RUN_PREFIX_VAR` output variable.

    Args:
      inputs: Inputs to the prompt template.
      llm_default_reply: String that is used as a reply from `llm` operation.
      generate_object_default_reply_map: map from type to instance for default
        values calling generate_object
      mock_callbacks: List of (callback_name, callback_fn, pass_context) for
        additional callbacks to be mocked.

    Returns:
      A dict with at least one key `PROMPT_PREFIX` that contains the final
      rendered string of the entire prompt. If the template has `llm()` calls
      the dict contains an `llm` key that stores a list of string-valued
      rendered prefixes sent to the `llm()` operation. If template has
      `choose()` calls the dict contains a `choose` key that stores a list of
      string-valued rendered prefixes sent to the `choose()` operation.
      The output may contain other keys if the user provides additional
      mock callbacks that write to the `_DRY_RUN_PREFIX_VAR` output variable.
    """
    # To mock some of the callbacks we modify the `_callbacks` attribute
    # of the PromptTemplateJ2 instance. To avoid subtle issues (for example:
    # parallel async execution of multiple dry_run coroutines of the same
    # PromptTemplateJ2 instance) we create a copy of the instance.
    mock_prompt = copy.deepcopy(self)
    # By default we mock `choose` and `llm` operations.
    mock_prompt.register_callback(
        'llm',
        functools.partial(
            callbacks.mock_llm_callback, default_reply=llm_default_reply
        ),
        pass_context=True,
    )
    mock_prompt.register_callback(
        'choose', callbacks.mock_choose_callback, pass_context=True
    )
    mock_prompt.register_callback(
        'generate_object',
        functools.partial(
            callbacks.mock_generate_object_callback,
            default_type_to_object_map=generate_object_default_reply_map,
        ),
        pass_context=True,
    )
    # We also mock any other callbacks provided by the user.
    if mock_callbacks is not None:
      for callback_name, callback_fn, callback_pass_context in mock_callbacks:
        mock_prompt.register_callback(
            name=callback_name,
            function=callback_fn,
            pass_context=callback_pass_context,
        )
    outputs = await mock_prompt.render(**inputs)
    try:
      result = outputs[_DRY_RUN_PREFIX_VAR]
    except KeyError as exc:
      raise ValueError(
          'No DRY_RUN found in outputs. Please make sure that the template has'
          ' a llm() call.'
      ) from exc
    result[PROMPT_PREFIX] = outputs[PROMPT_PREFIX]
    return result
