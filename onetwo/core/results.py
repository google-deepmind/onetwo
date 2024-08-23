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

"""Data structures for storing results of prompting and evaluation."""

from __future__ import annotations

import collections
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime
import html
import itertools
import logging
import pprint
import textwrap
import timeit
from typing import Any, Protocol

import dataclasses_json
import termcolor


################################################################################
# Constants used as keys in the `inputs` and `outputs` mappings.
# Note that these are not intended to be an exhaustive list and are included
# here only for convenience, to avoid repeating the same raw string in multiple
# places in the code.
# TODO: Revisit how the results of individual calls to the
# language model should be represented, including multi-modal language models
# and `ScoreRequest` in addition to `CompleteRequest`.
# ################################################################################

# The request text to be sent to the engine.
INPUT_KEY_REQUEST = 'request'

# Constants used in legacy request/reply data structures.
OUTPUT_KEY_REPLY_TEXT = 'reply_text'
OUTPUT_KEY_REPLY_TEXT_STRIPPED = 'reply_text_stripped'
OUTPUT_KEY_REPLY_OBJECT = 'reply_object'

# Main output from a tool/callback/chain. This is used to indicate for example
# which output value to pass as inputs to further steps of computation.
MAIN_OUTPUT = 'output'
# Output field storing the repeated values to compute list metrics.
VALUES_FOR_LIST_METRICS = 'values_for_list_metrics'

# Constants for standard counter names.
COUNTER_TOTAL_EXAMPLES = 'total_count'
COUNTER_ERRORS = 'error_count'


def _exclude_empty(x: Any) -> bool:
  """Excludes empty values (when outputting dataclass_json.to_dict)."""
  return bool(not x)


def get_metric_counter_name(metric_name: str) -> str:
  """Returns the name of the counter tracking # data points for the metric."""
  return f'{metric_name}_count'


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ExecutionResult:
  """Full results of prompt or chain execution, including debug details.

  Note that this is a nested data structure and is designed such that the same
  data structure can be used to represent the results of executing a prompting
  strategy as a whole (e.g., for one given example from a dataset), or the
  result of sending a single request to an underlying language model, or
  anything in between.

  Attributes:
    stage_name: Name of the corresponding prompt stage, in the case where the
      execution results are for one of the stages of an outer prompt chain.
    inputs: The inputs to the current prompting stage -- e.g., the contents of
      the parsed dataset record (in the case of a top-level ExecutionResult) or
      the output variables of a preceding chain (in the case of an intermediate
      result). For a leaf-level ExecutionResult corresponding to a single
      `CompleteRequest` sent to a text-only language model, this would contain a
      key called 'request', along with any input variables that were used in
      constructing that request.
    outputs: The outputs of the current prompting stage. For a leaf-level
      ExecutionResult corresponding to a single `CompleteRequest` sent to a
      text-only language model, this would contain a few hard-coded keys called
      'reply_object', 'reply_text', and 'reply_text_stripped', along with a
      value for the engine reply placeholder. For a multi-step prompt template
      bundled with reply parsing logic, this would contain the result of parsing
      each of the engine replies received in the course of the prompt template
      execution, with the result of later replies overwriting those of previous
      replies, in the case of name clashes. For a prompt chain, the structure of
      the outputs would be determined by the chain implementation.
    stages: In the case of a prompt chain (or of a multi-step prompt template),
      this contains the execution results of each of the steps in the chain, in
      order of execution.
    error: Error message in the case where an error occurred.
    info: Contains identifying information for the given data point -- e.g.,
      record_id, exemplar_list_id, exemplar_list_size, sample_id, sample_size.
      For now, this is populated only on top-level ExecutionResults.
    start_time: The monotonic clock time at which the execution started.
    end_time: The monotonic clock time at which the execution ended.
  """
  # Disabling type checking due to a wrong type annotation in dataclasses_json:
  # https://github.com/lidatong/dataclasses-json/issues/336
  # pytype: disable=wrong-arg-types

  # Fields relevant only for the sub-stages of a PromptChain.
  stage_name: str = dataclasses.field(
      default='', metadata=dataclasses_json.config(exclude=_exclude_empty)
  )

  # Fields relevant at arbitrary levels of nesting.
  inputs: dict[str, Any] = dataclasses.field(default_factory=dict)
  outputs: dict[str, Any] = dataclasses.field(default_factory=dict)
  stages: list['ExecutionResult'] = dataclasses.field(
      default_factory=list,
      metadata=dataclasses_json.config(exclude=_exclude_empty),
  )
  error: str = dataclasses.field(
      default='', metadata=dataclasses_json.config(exclude=_exclude_empty)
  )

  # Fields currently relevant only at the top level.
  # TODO: We should revisit whether it makes sense to use
  # an info data structure like this in lower levels of the execution hierarchy
  # as well; if not, it may be cleaner to move this into `EvaluationResult`.
  info: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata=dataclasses_json.config(exclude=_exclude_empty),
  )

  start_time: float = 0.0
  end_time: float = 0.0

  # pytype: enable=wrong-arg-types

  def get_leaf_results(self) -> list['ExecutionResult']:
    """Returns references to the leaves of the result hierarchy."""
    leaf_results = []
    if self.stages:
      for stage in self.stages:
        leaf_results.extend(stage.get_leaf_results())
    else:
      leaf_results.append(self)
    return leaf_results

  def format(self, color: bool = True) -> str:
    """Returns a pretty-formatted version of the result hierarchy.

    Args:
      color: If True, then will return a string that is annotated with
        `termcolor` to apply colors and boldfacing when the text is printed to
        the terminal or in a colab. If False, then will return just plain text.
    """
    lines = []
    if self.stages:
      # Non-leaf result.
      for stage in self.stages:
        if stage.stage_name:
          if color:
            lines.append(
                termcolor.colored(stage.stage_name, attrs=['bold', 'underline'])
            )
          else:
            lines.append(stage.stage_name)
          lines.append(textwrap.indent(stage.format(color=color), '  '))
        else:
          lines.append(stage.format(color=color))
      for key, value in self.outputs.items():
        if color:
          lines.append(termcolor.colored(f'Parsed {key}:', attrs=['bold']))
          lines.append(termcolor.colored(str(value), 'magenta'))
        else:
          lines.append(f'* Parsed {key}')
          lines.append(str(value))
    else:
      # Leaf result.
      show_reply_on_new_line = True
      request = str(self.inputs.get(INPUT_KEY_REQUEST, None))
      if OUTPUT_KEY_REPLY_TEXT in self.outputs:
        # Show text replies from LLMs as a continuation of the request prompt.
        reply = str(self.outputs.get(OUTPUT_KEY_REPLY_TEXT, None))
        show_reply_on_new_line = False
      elif 'reply' in self.outputs:
        # For a BaseReply received from a tool, show the reply object.
        reply = str(self.outputs.get('reply', None))
      else:
        # Otherwise, fall back to showing the whole outputs data structure.
        reply = str(self.outputs)
      if color:
        lines.append(termcolor.colored('Request/Reply', attrs=['bold']))
        formatted_reply = termcolor.colored(reply, 'blue')
      else:
        lines.append('* Request/Reply')
        formatted_reply = f'<<{reply}>>'
      if show_reply_on_new_line:
        lines.append(f'{request}\n{formatted_reply}')
      else:
        lines.append(f'{request}{formatted_reply}')

    formatted_text = '\n'.join(lines)
    return formatted_text

  def get_reply_summary(self) -> str:
    """Returns a summary of replies useful for logging."""
    # We abbreviate nested input fields like 'exemplar' and nested output
    # fields like 'reply_object', as these are too bulky to be readable.
    def _pformat_abbreviated(original: dict[Any, Any]) -> str:
      abbreviated = original.copy()
      for k, v in abbreviated.items():
        if isinstance(v, dict) or isinstance(v, list):
          abbreviated[k] = '...'
      return pprint.pformat(abbreviated)

    result = (
        '\n\n=======================================================\n'
        f'Inputs: {_pformat_abbreviated(self.inputs)}\n'
        '--------------\n'
        f'Outputs: {_pformat_abbreviated(self.outputs)}'
    )
    if self.error:
      result += f'--------------\nError: {self.error}\n'
    return result

  def update_start_time(self) -> None:
    """Sets start time to now."""
    self.start_time = timeit.default_timer()

  def update_end_time(self) -> None:
    """Sets end time to now."""
    self.end_time = timeit.default_timer()

  def get_elapsed_time(self) -> float:
    """Returns the elapsed time between start and end time in seconds."""
    if self.end_time == 0.0:
      return 0.0
    return self.end_time - self.start_time


def format_result(
    result: ExecutionResult | Sequence[ExecutionResult],
    color: bool = True,
) -> str:
  """Returns a pretty-formatted version of the result hierarchy.

  See `ExecutionResult.format` for details. This function does this same
  thing, but also support formatting of a sequence of results.

  Args:
    result: The result or sequence of results to format.
    color: See `ExecutionResult.format`.
  """
  if isinstance(result, Sequence):
    texts = []
    for i, res in enumerate(result):
      if color:
        preamble = termcolor.colored(
            f'Sample {i + 1}/{len(result)}\n',
            'green',
            attrs=['bold', 'underline'],
        )
      else:
        preamble = f'* Sample {i + 1}/{len(result)}\n'
      texts.append(preamble + res.format(color=color))
    return '\n\n'.join(texts)
  else:
    return result.format(color=color)


def apply_formatting(
    result: ExecutionResult,
    function: Callable[[ExecutionResult], str]
) -> str:
  """Formats the ExecutionResult by applying a function at each node.

  Args:
    result: ExecutionResult to be formatted.
    function: A function taking an ExecutionResult and returning a string
      representation.

  Returns:
    A tree representation of the ExecutionResult where each node is represented
    using the function.
  """
  res = function(result)
  subres = ''
  for stage in result.stages:
    subres += apply_formatting(stage, function)
  return res + textwrap.indent(subres, '  ')


def get_name_tree(result: ExecutionResult) -> str:
  """Returns a tree representation with the stage names for easy inspection."""
  return apply_formatting(
      result, lambda s: f'- {s.stage_name}\n' if s.stage_name else '-\n'
  )


def _trim_key(key: str) -> str:
  if len(key) < 30:
    return key
  else:
    return key[:27] + '...'


def _trim_value(value: str) -> str:
  if len(value) < 50:
    return value
  else:
    return value[:23] + '[...]' + value[-23:]


def get_name_keys_tree(result: ExecutionResult) -> str:
  """Returns a tree with the stage names and input/output keys."""

  def formatting(result: ExecutionResult) -> str:
    inputs = list(map(_trim_key, result.inputs.keys()))
    outputs = list(map(_trim_key, result.outputs.keys()))
    return f'- {result.stage_name}: {inputs} -> {outputs}\n'

  return apply_formatting(result, formatting)


def get_short_values_tree(result: ExecutionResult) -> str:
  """Returns a tree with the values trimmed to a single line."""

  def render_dict(d: Mapping[str, Any]) -> str:
    trimmed = {
        _trim_key(k): _trim_value(repr(v)) for k, v in d.items()
    }
    if len(d.keys()) <= 1:
      return str(d)
    return (
        '{\n'
        + '\n'.join([f'    {k}: {v}' for k, v in trimmed.items()])
        + '\n  }'
    )

  def formatting(result: ExecutionResult) -> str:
    return (
        f'- {result.stage_name}:\n  inputs: {render_dict(result.inputs)}\n '
        f' outputs: {render_dict(result.outputs)}\n'
    )

  return apply_formatting(result, formatting)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EvaluationResult(ExecutionResult):
  """Full results of an evaluation run on a given example, with metrics.

  Corresponds more or less one-to-one to the contents of a single record of the
  'results_debug.json' file that is output at the end of each evaluation run.
  The 'results.json' file  contains the same content, but with the 'stages'
  field and a few keys of the other mappings omitted.

  Attributes:
    targets: Target values (i.e., "golden" outputs), against which to compare
      the outputs when calculating metrics.
    metrics: Evaluation metrics, such as accuracy.
  """
  # Disabling type checking due to a wrong type annotation in dataclasses_json:
  # https://github.com/lidatong/dataclasses-json/issues/336
  # pytype: disable=wrong-arg-types

  targets: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata=dataclasses_json.config(exclude=_exclude_empty),
  )
  metrics: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata=dataclasses_json.config(exclude=_exclude_empty),
  )
  # pytype: enable=wrong-arg-types

  def to_compact_record(self) -> 'EvaluationResult':
    """Returns a compact version of self for writing to `results.json`."""
    record_compact = copy.deepcopy(self)
    record_compact.stages = []

    # We enumerate here any keys that we prefer to omit from the compact
    # representation of the input and output formats, due to the content being
    # too bulky (as in the case of `exemplar` or `VALUES_FOR_LIST_METRICS`) or
    # repetitive content that is included in the intenral results data structure
    # only for legacy reasons (as in the case of `original` and `record_id`).
    input_fields_to_omit = {'exemplar', 'original', 'record_id'}
    output_fields_to_omit = {VALUES_FOR_LIST_METRICS}

    for key in input_fields_to_omit:
      if key in record_compact.inputs:
        del record_compact.inputs[key]
    for key in output_fields_to_omit:
      if key in record_compact.outputs:
        del record_compact.outputs[key]
    return record_compact

  @classmethod
  def from_execution_result(
      cls, execution_result: ExecutionResult
  ) -> 'EvaluationResult':
    """Returns an EvaluationResult with the same content as execution_result."""
    evaluation_result = EvaluationResult()
    for field in dataclasses.fields(ExecutionResult):
      setattr(
          evaluation_result, field.name, getattr(execution_result, field.name)
      )
    return evaluation_result


# Deprecated alias, kept temporarily for backwards compatibility.
ExperimentResult = EvaluationResult


def execution_result_from_dict(data: dict[str, Any]) -> ExecutionResult:
  """Returns an ExecutionResult restored from a structure created by to_dict."""
  result = ExecutionResult.from_dict(data)
  # Theoretically `ExecutionResult.from_dict` alone should be sufficient, but
  # for some reason when we try to do this, the nested stage results fail to get
  # converted from dicts back into ExecutionResult objects. Might be a
  # limitation of dataclasses_json with respect to self-referential data
  # structures.
  result.stages = list(execution_result_from_dict(s) for s in result.stages)
  return result


def evaluation_result_from_dict(data: dict[str, Any]) -> EvaluationResult:
  """Returns an EvaluationResult restored from structure created by to_dict."""
  result = EvaluationResult.from_dict(data)
  # See note on `execution_result_from_dict` above for why this is needed.
  result.stages = list(execution_result_from_dict(s) for s in result.stages)
  return result


@dataclasses.dataclass
class EvaluationTiming:
  """Timing information for an evaluation run.

  Attributes:
    start_time: The start time of the evaluation.
    end_time: The end time of the evaluation.
    time_elapsed: The time elapsed from beginning to end of the evaluation.
  """
  start_time: datetime.datetime = datetime.datetime.now()
  end_time: datetime.datetime = datetime.datetime.now()

  @property
  def time_elapsed(self) -> datetime.timedelta:
    return self.end_time - self.start_time


@dataclasses.dataclass
class EvaluationSummary:
  """Summary of the results of an evaluation run.

  Attributes:
    timing: Evaluation timing information.
    metrics: Mapping of metric name to metric value.
    counters: Mapping of counter name to counter value.
    info: Mapping that can be used for storing arbitrary aggregated information
      about the evaluation results, beyond what is stored in `metrics` and
      `counters`. Unlike `metrics` and `counters`, which are averaged or summmed
      in a standard way when adding `EvaluationSummary` objects, the `info`
      field is not updated automatically during addition of `EvaluationSummary`
      objects; its management is instead left entirely up to the code that
      constructs the `EvaluationSummary` objects.
    example_keys: Mapping of example index to example key.
    results: Mapping of example key to evaluation result (w/o detailed trace).
    results_debug: Mapping of example key to evaluation result (w/ detailed
      trace).
    final_states: Mapping of example key to final state of agent. Only relevant
      when the strategy is a subclass of `Agent`.
  """
  timing: EvaluationTiming = dataclasses.field(default_factory=EvaluationTiming)
  metrics: dict[str, float] = dataclasses.field(default_factory=dict)
  counters: collections.Counter[str] = dataclasses.field(
      default_factory=collections.Counter)
  info: dict[str, Any] = dataclasses.field(default_factory=dict)
  example_keys: dict[int, str|int] = dataclasses.field(default_factory=dict)
  results: dict[str|int, EvaluationResult] = dataclasses.field(
      default_factory=dict)
  results_debug: dict[str|int, EvaluationResult] = dataclasses.field(
      default_factory=dict)
  final_states: dict[str|int, Any] = dataclasses.field(default_factory=dict)
  # TODO: Support storing traces output by custom tracers.

  def replace_example_index_and_key(
      self, example_index: int, example_key: str|int) -> None:
    """Replaces the example index and keys in the summary with the given ones.

    This is intended to be used only in the case where the EvaluationSummary
    contains the results of just a single example.

    Args:
      example_index: The new example index to use.
      example_key: The new example key to use.
    """
    current_example_keys = (
        set(self.example_keys.values())
        | set(self.results.keys())
        | set(self.results_debug.keys())
        | set(self.final_states.keys())
    )
    if (len(current_example_keys) > 1):
      raise ValueError(
          'Cannot replace example index and key in EvaluationSummary with '
          'multiple examples.')
    self.example_keys = {example_index: example_key}
    if self.results:
      self.results[example_key] = self.results.pop(list(self.results.keys())[0])
    if self.results_debug:
      self.results_debug[example_key] = self.results_debug.pop(
          list(self.results_debug.keys())[0])
    if self.final_states:
      self.final_states[example_key] = self.final_states.pop(
          list(self.final_states.keys())[0])

  def __iadd__(self, other: EvaluationSummary) -> EvaluationSummary:
    """Adds the contents of the given evaluation summary to the current one.

    Updates all attributes in-place.

    Note that we intentionally do not provide an implementation for ordinary
    `__add__`, as normally when adding evaluation summary objects the typical
    pattern is to have one summary representing the evaluation run as a whole
    and then to add it summaries representing the incremental results from
    evaluation of individual examples in the evaluation run. In such cases, it
    is much more efficient to use `__iadd__`, so as to avoid unnecessary copying
    of the increasingly large summary object of the full evaluation run.

    Args:
      other: The other evaluation summary to add.

    Returns:
      Self (after updating with the contents of `other`).
    """
    self.timing.start_time = min(
        self.timing.start_time, other.timing.start_time)
    self.timing.end_time = max(
        self.timing.end_time, other.timing.end_time)

    # Update the metrics as weighted averages based on number of examples.
    for metric_name in self.metrics.keys() | other.metrics.keys():
      self_value = self.metrics.get(metric_name, 0.0)
      other_value = other.metrics.get(metric_name, 0.0)
      # If the metric value is explicitly defined in an EvaluationSummary, then
      # we expect the corresponding counter value to also be explicitly defined.
      # The preferred approach is to define a separate counter for each metric,
      # so that we can decide individually for each metric whether to include a
      # given example in the computation of the metric or not. The alternative
      # approach, which is simpler but less flexible, is to use
      # COUNTER_TOTAL_EXAMPLES as the counter for all metrics.
      metric_counter_name = get_metric_counter_name(metric_name)
      if metric_name in self.metrics:
        self_count = self.counters.get(
            metric_counter_name, self.counters[COUNTER_TOTAL_EXAMPLES]
        )
      else:
        self_count = self.counters.get(metric_counter_name, 0)
      if metric_name in other.metrics:
        other_count = other.counters.get(
            metric_counter_name, other.counters[COUNTER_TOTAL_EXAMPLES]
        )
      else:
        other_count = other.counters.get(metric_counter_name, 0)

      numerator = self_value * self_count + other_value * other_count
      denominator = self_count + other_count
      if denominator:
        self.metrics[metric_name] = numerator / denominator
      else:
        logging.warning(
            'Attempting to add EvaluationSummary objects that have values'
            ' defined for metric %r, but no corresponding counter: metrics1=%s,'
            ' metrics2=%s, counters1=%s, counters2=%s',
            metric_name,
            self.metrics,
            other.metrics,
            self.counters,
            other.counters,
        )

    self.counters.update(other.counters)
    self.example_keys.update(other.example_keys)
    self.results.update(other.results)
    self.results_debug.update(other.results_debug)
    self.final_states.update(other.final_states)
    return self


@dataclasses.dataclass
class HTMLObjectRendering:
  """Reply from an HTMLObjectRenderer.

  Attributes:
    html: The full HTML rendering of the object. In cases where the object is to
      be rendered in a collapsible form, this is the rendering that is to be
      used when it is in expanded state.
    collapsed_html: The HTML rendering of the object to use when it is in
      collapsed state. If `None`, then it will fall back to the default. Only
      relevant when `collapsible` is True.
    title: The title to be shown in boldface contually (both when collapsed
      and when expanded), in cases where the object is rendered in a collapsible
      form. If `None`, then falls back to the default.
    expanded: Whether the object should be rendered in expanded state by
      default. Only relevant when `collapsible` is True.
    collapsible: Whether to render the object in a collapsible form (in cases
      where such an option is available).
  """
  html: str = ''
  collapsed_html: str | None = dataclasses.field(default=None, kw_only=True)
  title: str | None = dataclasses.field(default=None, kw_only=True)
  expanded: bool = dataclasses.field(default=False, kw_only=True)
  collapsible: bool = dataclasses.field(default=True, kw_only=True)


class HTMLObjectRenderer(Protocol):
  """Callable for rendering a specific type of object as HTML."""

  def __call__(
      self,
      renderer: HTMLRenderer,
      object_to_render: Any,
      *,
      element_id: str,
      levels_to_expand: int,
  ) -> HTMLObjectRendering | None:
    """Returns an HTML rendering of the given object, or `None` to punt.

    The typical use case is for registering a list of custom renderers, each
    of which is intended to handle a specific type of object. The renderers are
    called in order of registration, and the first renderer to return a non-None
    value is the one that is used to render the object.

    Args:
      renderer: The original renderer that is calling this custom renderer. The
        custom renderer can make recursive calls to `renderer.render_object` to
        delegate the rendering of child nodes to the original renderer
        (including the full list of registered custom renderers).
      object_to_render: The object to render.
      element_id: The id to be used for the outermost HTML element returned by
        this function. Ids for inner elements are expected to be constructed by
        appending suffixes to this id.
      levels_to_expand: The number of levels in the hierarchy below this one to
        expand by default, for elements that are capable of
        expanding/collapsing. When making recursive calls for rendering child
        nodes, levels_to_expand should be reduced by 1. If levels_to_expand
        <= 0, then the element should be rendered in collapsed state, and no
        calls should be made for rendering child nodes.
    """
    ...


def _is_update_list_state(object_to_render: Any) -> bool:
  """Returns whether the given object is an `UpdateListState` object."""
  # We detect this case by inspecting the object's attributes rather than
  # via `instanceof` to avoid a circular dependency on `agents_base.py`.
  # TODO: Come up with a cleaner way of configuring a default set of
  # agent-specific renderers, without introducing dependencies from the core
  # code to the individual agent modules.
  return hasattr(object_to_render, 'inputs') and hasattr(
      object_to_render, 'updates'
  )


def _as_string(object_to_render: Any) -> str:
  """Returns an appropriate string representation for rendering in HTML."""
  return html.escape(repr(object_to_render))


@dataclasses.dataclass
class HTMLRenderer:
  """Renders Execution[Result|Summary] and related objects in HTML format.

  Attributes:
    levels_to_expand: Number of levels in the hierarchy to expand by default,
      when rendering a single result or list of results. (When rendering an
      entire EvaluationSummary, we display only the top level initially.)
    custom_renderers: List of custom renderers to use for rendering specific
      types of objects. The function should return a string containing the
      rendered HTML (if the object is of the type that the renderer is intended
      to handle), or None (to defer to  the next renderer in the list).
  """

  # Public attributes.
  levels_to_expand: int = 3
  custom_renderers: list[HTMLObjectRenderer] = dataclasses.field(
      default_factory=list
  )

  # =========================================================================
  # Private attributes. (Not exposing them publicly yet, as they may change.)
  # =========================================================================

  # Number of additional levels (beyond `levels_to_expand`) to render upfront
  # in HTML (in collapsed state) so that they can be expanded later. Setting
  # some limit here is useful, among other things, in helping avoid the risk of
  # infinite recursion in the case of a bug in the rendering code.
  _additional_levels_to_make_expandable: int = 20

  # Maximum length of the single-line representation of an object to show when
  # it is in collapsed state. If longer than this, it will be truncated, with
  # '...'.
  _max_collapsed_rendering_length: int = 160

  # Maximum length for displaying a single value (e.g., input or output) of dict
  # or list type on a single line. If the value's string representation is
  # within this length, then it will be shown on a single line; if longer, then
  # it will be expanded into a separate line for each element in the list or
  # dict.
  _max_single_line_value_string_length: int = 80

  def _string_representation_can_fit_on_single_line(
      self, object_to_render: Any
  ) -> bool:
    object_as_string = _as_string(object_to_render)
    return len(object_as_string) <= self._max_single_line_value_string_length

  def _render_llm_request_reply(
      self, request: str, reply: str, element_id: str, levels_to_expand: int
  ) -> str:
    """Returns HTML rendering the request/reply as multi-line colored text."""
    # TODO: Support roles and multi-modal requests/replies.
    del element_id, levels_to_expand
    updated_request = request.replace('\n', '<br>')
    updated_reply = reply.replace('\n', '<br>')
    return (
        '<p style="color:black;background-color:white">'
        f'<span style="color:black">{updated_request}</span>'
        f'<span style="color:blue">{updated_reply}</span></p>'
    )

  def _render_dict_like_object(
      self, object_to_render: Any, *, element_id: str, levels_to_expand: int
  ) -> HTMLObjectRendering | None:
    """Returns a rendering of a dict as text or in ...<ul>...</ul> form."""
    if isinstance(object_to_render, dict):
      object_as_dict = object_to_render
      title = f'{type(object_to_render).__name__}({len(object_to_render)})'
    else:
      try:
        object_as_dict = vars(object_to_render)
        title = f'{type(object_to_render).__name__}'
      except TypeError:
        # Type cannot be converted to a dict.
        return None

    if self._string_representation_can_fit_on_single_line(object_to_render):
      return None

    lines = []
    lines.append(title)
    lines.append('<ul>')
    for i, (key, value) in enumerate(object_as_dict.items()):
      lines.append(
          self.render_object_as_collapsible_list_element(
              value,
              title=f'{key}:',
              element_id=f'{element_id}-{i}',
              levels_to_expand=levels_to_expand-1,
          )
      )
    lines.append('</ul>')

    return HTMLObjectRendering(
        html='\n'.join(lines),
        expanded=(levels_to_expand > 0),
    )

  def _render_result(
      self, object_to_render: Any, *, element_id: str, levels_to_expand: int
  ) -> HTMLObjectRendering | None:
    """Returns a rendering of an ExecutionResult in ...<ul>...</ul> form."""
    if not isinstance(object_to_render, ExecutionResult):
      return None
    result = object_to_render

    # Skip over redundant ExecutionResult objects that are simply empty wrappers
    # for a single ExecutionResult stage that contains the actual content. This
    # situation commonly occurs in the outermost layer of the ExecutionResult
    # that is returned when `executing.run` is called with
    # `enabled_tracing=True`.
    # TODO: Make the redundancy check more robust -- e.g., consider
    # whether the result may be an `EvaluationResult` with metrics or targets
    # populated.
    if (
        not result.stage_name
        and not result.inputs
        and not result.outputs
        and len(result.stages) == 1
    ):
      return self.render_object(
          result.stages[0],
          # We still specify a different element_id, so that we can detect in
          # the unit tests whether we skipped over the outer element or not.
          element_id=f'{element_id}-0',
          levels_to_expand=levels_to_expand,
      )

    # TODO: Remove the special treatment of the special outputs key
    # 'output', once we support storing non-dict outputs directly in the
    # ExecutionResult.
    if isinstance(result.outputs, dict) and set(result.outputs.keys()) == {
        'output'
    }:
      outputs = result.outputs.get('output')
    else:
      outputs = result.outputs

    inputs_string = _as_string(result.inputs)
    outputs_string = _as_string(outputs)
    collapsed_html = f'{inputs_string} <b>&rArr;</b> {outputs_string}'

    # TODO: Remove the special treatment of the special outputs key
    # 'target', once we support storing non-dict targets directly in the
    # EvaluationResult.
    if isinstance(result, EvaluationResult):
      if isinstance(result.targets, dict) and set(result.targets.keys()) == {
          'target'
      }:
        targets = result.targets.get('target')
      else:
        targets = result.targets
    else:
      targets = {}

    # Special formatting for leaf-level LLM requests.
    if result.stage_name == 'generate_text' and not result.stages:
      request = result.inputs.get('request', 'MISSING_REQUEST_FIELD')
      reply = result.outputs.get('output', 'MISSING_OUTPUT_FIELD')
      request_reply_string = self._render_llm_request_reply(
          request=request,
          reply=reply,
          element_id=f'{element_id}rr',
          levels_to_expand=levels_to_expand - 1,
      )
      return HTMLObjectRendering(
          html=request_reply_string,
          collapsed_html=collapsed_html,
          expanded=(levels_to_expand > 0),
      )

    # Default formatting for arbitrary prompting stages.
    lines = []
    lines.append('<ul>')
    lines.append(
        self.render_object_as_collapsible_list_element(
            result.inputs,
            title='inputs:',
            element_id=f'{element_id}i',
            levels_to_expand=levels_to_expand - 1,
        )
    )
    lines.append(
        self.render_object_as_collapsible_list_element(
            outputs,
            title='outputs:',
            element_id=f'{element_id}o',
            levels_to_expand=levels_to_expand - 1,
        )
    )
    if result.error:
      lines.append(
          self.render_object_as_collapsible_list_element(
              result.error,
              title='error:',
              element_id=f'{element_id}-error',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if result.info:
      lines.append(
          self.render_object_as_collapsible_list_element(
              result.info,
              title='info:',
              element_id=f'{element_id}-info',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if isinstance(result, EvaluationResult):
      if targets:
        lines.append(
            self.render_object_as_collapsible_list_element(
                targets,
                title='targets:',
                element_id=f'{element_id}t',
                levels_to_expand=levels_to_expand - 1,
            )
        )
      if result.metrics:
        lines.append(
            self.render_object_as_collapsible_list_element(
                result.metrics,
                title='metrics:',
                element_id=f'{element_id}m',
                levels_to_expand=levels_to_expand - 1,
            )
        )
    if result.stages:
      for i, stage in enumerate(result.stages):
        lines.append(
            self.render_object_as_collapsible_list_element(
                stage,
                element_id=f'{element_id}-{i}',
                index_within_list=i if len(result.stages) > 1 else None,
                levels_to_expand=levels_to_expand - 1,
            )
        )
    lines.append('</ul>')

    return HTMLObjectRendering(
        html='\n'.join(lines),
        collapsed_html=collapsed_html,
        title=f'<u>{result.stage_name}</u>' if result.stage_name else None,
        expanded=(levels_to_expand > 0),
    )

  def _render_agent_state(
      self, object_to_render: Any, *, element_id: str, levels_to_expand: int = 0
  ) -> HTMLObjectRendering | None:
    """Returns a rendering of the state as text or in ...<ul>...</ul> form."""
    if not _is_update_list_state(object_to_render):
      return None
    if self._string_representation_can_fit_on_single_line(object_to_render):
      return None
    state = object_to_render

    lines = []
    lines.append('<ul>')
    lines.append(
        self.render_object_as_collapsible_list_element(
            state.inputs,
            title='inputs:',
            element_id=f'{element_id}i',
            # Note that we purposefully don't decrement levels_to_expand here,
            # as it is usually more intuitive to either fully expand the inputs
            # and updates, or to not expand them at all (rather than halfway).
            levels_to_expand=levels_to_expand,
        )
    )
    lines.append(
        self.render_object_as_collapsible_list_element(
            state.updates,
            title='updates:',
            element_id=f'{element_id}u',
            # Note that we purposefully don't decrement levels_to_expand here,
            # as it is usually more intuitive to either fully expand the inputs
            # and updates, or to not expand them at all (rather than halfway).
            levels_to_expand=levels_to_expand,
        )
    )
    lines.append('</ul>')

    inputs_string = _as_string(state.inputs)
    return HTMLObjectRendering(
        html='\n'.join(lines),

        collapsed_html=(
            f'{inputs_string} <b>&rArr;</b> {len(state.updates)} updates'
        ),
        expanded=(levels_to_expand > 0),
    )

  def _render_list(
      self, object_to_render: Any, *, element_id: str, levels_to_expand: int = 0
  ) -> HTMLObjectRendering | None:
    """Returns a rendering of a seqeuence of elements in <ul>...</ul> form."""
    if not isinstance(object_to_render, list) and not isinstance(
        object_to_render, tuple
    ):
      return None
    if self._string_representation_can_fit_on_single_line(object_to_render):
      return None

    lines = []
    lines.append(f'{type(object_to_render).__name__}({len(object_to_render)})')
    lines.append('<ul>')
    for i, element in enumerate(object_to_render):
      lines.append(
          self.render_object_as_collapsible_list_element(
              element,
              element_id=f'{element_id}-{i}',
              index_within_list=i,
              levels_to_expand=levels_to_expand-1,
          )
      )
    lines.append('</ul>')

    return HTMLObjectRendering(
        html='\n'.join(lines),
        collapsed_html=(
            f'{type(object_to_render).__name__}({len(object_to_render)})'
        ),
        expanded=(levels_to_expand > 0),
    )

  def _render_evaluation_summary(
      self, object_to_render: Any, *, element_id: str, levels_to_expand: int
  ) -> HTMLObjectRendering | None:
    """Returns a rendering of an EvaluationSummary in <ul>...</ul>... form."""
    if not isinstance(object_to_render, EvaluationSummary):
      return None
    summary = object_to_render

    lines = []
    lines.append('<ul>')
    if summary.timing:
      lines.append(
          self.render_object_as_collapsible_list_element(
              summary.timing,
              title='timing:',
              element_id=f'{element_id}-timing',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if summary.metrics:
      lines.append(
          self.render_object_as_collapsible_list_element(
              summary.metrics,
              title='metrics:',
              element_id=f'{element_id}-metrics',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if summary.counters:
      lines.append(
          self.render_object_as_collapsible_list_element(
              summary.counters,
              title='counters:',
              element_id=f'{element_id}-counters',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if summary.results:
      if len(summary.results) != len(summary.example_keys):
        raise ValueError(
            'Number of results and example keys must match, but got'
            f' {len(summary.results)} vs {len(summary.example_keys)}.'
        )
      results_list = [
          summary.results[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      lines.append(
          self.render_object_as_collapsible_list_element(
              results_list,
              title='results:',
              element_id=f'{element_id}-results',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    if summary.results_debug:
      if len(summary.results_debug) != len(summary.example_keys):
        raise ValueError(
            'Number of results_debug and example keys must match, but got'
            f' {len(summary.results_debug)} vs {len(summary.example_keys)}.'
        )
      results_debug_list = [
          summary.results_debug[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      # Note that we ignore `self.levels_to_expand` here, and instead always
      # render the individual sections initially in collapsed state when
      # rendering an entire `EvaluationSummary`, as it tends to look rather
      # overwhelming to have many different detailed EvaluationResult traces
      # expanded all at once.
      lines.append(
          self.render_object_as_collapsible_list_element(
              results_debug_list,
              title='results_debug:',
              element_id=f'{element_id}-results_debug',
              levels_to_expand=0,
          )
      )
    if summary.final_states:
      if len(summary.final_states) != len(summary.example_keys):
        raise ValueError(
            'Number of final_states and example keys must match, but got'
            f' {len(summary.final_states)} vs {len(summary.example_keys)}.'
        )
      final_state_list = [
          summary.final_states[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      lines.append(
          self.render_object_as_collapsible_list_element(
              final_state_list,
              title='final_states:',
              element_id=f'{element_id}-final_states',
              levels_to_expand=levels_to_expand - 1,
          )
      )
    lines.append('</ul>')
    return HTMLObjectRendering(
        html='\n'.join(lines),
        expanded=(levels_to_expand > 0),
    )

  def _get_default_renderers(self) -> Sequence[HTMLObjectRenderer]:
    """Returns a list of default renderers for various object types."""
    return [
        HTMLRenderer._render_result,
        HTMLRenderer._render_evaluation_summary,
        HTMLRenderer._render_agent_state,
        HTMLRenderer._render_list,
        HTMLRenderer._render_dict_like_object,
    ]

  def render_object(
      self,
      object_to_render: Any,
      *,
      element_id: str,
      levels_to_expand: int,
  ) -> HTMLObjectRendering:
    """Returns an HTML rendering of the given object (without JavaScript, etc.).

    Can be called recursively to render nested objects.

    Args:
      object_to_render: The object to render.
      element_id: The id to be used for the outermost HTML element returned by
        this function. Ids for inner elements are expected to be constructed
        by appending suffixes to this id.
      levels_to_expand: The number of levels in the hierarchy below this one to
        expand by default, for elements that are capable of
        expanding/collapsing. When making recursive calls for rendering child
        nodes, levels_to_expand is expected to be reduced by 1.
    """
    if levels_to_expand < -self._additional_levels_to_make_expandable:
      # Too deep in the hierarchy to expand. At this point, we simply render
      # the object as a string, without any additional HTML formatting, to
      # avoid any risk of infinite recursion.
      return HTMLObjectRendering(html=_as_string(object_to_render))

    all_renderers = itertools.chain(
        self.custom_renderers, self._get_default_renderers()
    )
    for renderer in all_renderers:
      rendering = renderer(
          self,
          object_to_render,
          element_id=element_id,
          levels_to_expand=levels_to_expand,
      )
      if rendering is not None:
        return rendering

    return HTMLObjectRendering(
        html=_as_string(object_to_render), collapsible=False
    )

  def render_object_as_collapsible_list_element(
      self,
      object_to_render: Any,
      *,
      title: str | None = None,
      element_id: str,
      levels_to_expand: int,
      index_within_list: int | None = None,
  ) -> str:
    """Returns a rendering of an object as HTML string in <li>...</li>... form.

    The returned HTML string is suitable for repeated insertion within a
    <ul>...</ul> block.

    Delegates most of the object-specific rendering logic to `render_object`.
    If the `HTMLObjectRendering` returned by `render_object` has
    `collapsible == True`, then (with the exception of a few trivial cases) the
    object will be rendered as a collapsible line, in which the user can toggle
    between the longer `html` representation and the shorter `collapsed_html`
    representation by clicking on the "title" portion of the line.

    Args:
      object_to_render: The object to render.
      title: The title to be shown in boldface contually (both when collapsed
        and when expanded). If `None`, then falls back to the title from the
        rendering returned by `render_object`.
      element_id: The id to be used for the outermost HTML element returned by
        this function. Ids for inner elements are expected to be constructed by
        appending suffixes to this id.
      levels_to_expand: The number of levels in the hierarchy below this one to
        expand by default, for elements that are capable of
        expanding/collapsing. When making recursive calls for rendering child
        nodes, levels_to_expand is expected to be reduced by 1.
      index_within_list: The index of the element within the list, if the
        element is to be displayed as part of a numbered list.
    """
    object_rendering = self.render_object(
        object_to_render,
        element_id=element_id,
        levels_to_expand=levels_to_expand,
    )

    # State of expansion.
    # Note that a visible `span` is style `display:inline`, while a visible
    # `div` is style `display:block`.
    if object_rendering.expanded:
      collapsed_style = 'display:none'
      expanded_style = 'display:inline'
    else:
      collapsed_style = 'display:inline'
      expanded_style = 'display:none'

    lines = []

    # Title to show both when collapsed and when expanded.
    if title is None:
      title = object_rendering.title
    if title is None:
      title = f'{type(object_to_render).__name__}'

    # Optional display of index number at the beginning of the title.
    title_pieces = []
    if index_within_list is not None:
      title_pieces.append(f'[{index_within_list+1}]')
    if title:
      title_pieces.append(title)
    full_title = ' '.join(title_pieces)
    if full_title:
      full_title = f'<b>{full_title}</b> '

    # Content to show when collapsed.
    collapsed_html = object_rendering.collapsed_html
    if collapsed_html is None:
      object_as_string = _as_string(object_to_render)
      if len(object_as_string) > self._max_collapsed_rendering_length:
        object_as_string = (
            object_as_string[: self._max_collapsed_rendering_length] + '...'
        )
      collapsed_html = object_as_string

    # If the entire content can be shown in collapsed state, then don't
    # bother with the expandable behavior.
    collapsible = object_rendering.collapsible and (
        collapsed_html != object_rendering.html
    )
    if collapsible:
      # Collapsible display.
      lines.append(
          '<li><span onClick="toggleElements('
          f"['{element_id}', '{element_id}c'])\">"
          f'{full_title}</span>'
          f'<span id="{element_id}c" style="{collapsed_style}">'
          f'{collapsed_html}</span>'
      )
      lines.append(
          f'<span id="{element_id}"'
          f' style="{expanded_style}">{object_rendering.html}</span>'
      )
      lines.append('</li>')
    else:
      # Simple display.
      lines.append(f'<li>{full_title}{object_rendering.html}</li>')

    return'\n'.join(lines)

  def render(
      self,
      object_to_render: ExecutionResult | Sequence[ExecutionResult],
      *,
      element_id: str = '0',
  ) -> str:
    """Returns a full HTML + JavaScript rendering of the given result(s).

    Should not be called recursively, as this would lead to repetition of the
    JavaScript code and other outer boilerplate. For recursive calls, use
    `render_object` instead.

    Args:
      object_to_render: The object to render.
      element_id: The id to be used for the outermost HTML element returned by
        this function. Ids for inner elements will be generated automatically by
        appending suffixes to this id.

    Returns:
      An HTML block suitable for passing to `IPython.display.HTML(...)` for
      displaying in colab.
    """
    javascript_string = textwrap.dedent("""\
      <script>
      function toggleElement(element_id) {
        var element = document.getElementById(element_id);
        if (element.style.display == 'none') {
          if (element.tagName.toUpperCase() == 'SPAN') {
            element.style.display = 'inline';
          } else {
            element.style.display = 'block';
          }
        } else {
          element.style.display = 'none';
        }
      }
      function toggleElements(element_ids) {
        for (element_id of element_ids) {
          toggleElement(element_id);
        }
      }
      </script>
      """)
    if isinstance(object_to_render, ExecutionResult):
      # We display `ExecutionResult` objects as a list element, so as to ensure
      # that the top-level stage name is displayed.
      object_html = self.render_object_as_collapsible_list_element(
          object_to_render,
          element_id=element_id,
          levels_to_expand=self.levels_to_expand,
      )
      rendered_html = f"""\
{javascript_string}
<div>
  <ul>
{textwrap.indent(object_html, '    ')}
  </ul>
</div>
"""
    else:
      # For objects other than `ExecutionResult`, we display them directly, so
      # as to avoid any redundant extra bulletpoint at the top.
      object_rendering = self.render_object(
          object_to_render,
          element_id=element_id,
          levels_to_expand=self.levels_to_expand,
      )
      rendered_html = f'{javascript_string}\n{object_rendering.html}'
    return rendered_html
