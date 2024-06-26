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

"""Data structures for storing results of prompting and experiment execution."""

from __future__ import annotations

import collections
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime
import pprint
import textwrap
from typing import Any

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

# Reply text of the final reply in the exact form returned by the engine. Used
# in the result of sending a `BaseRequest` to a `LanguageModelEngine`.
OUTPUT_KEY_REPLY_TEXT = 'reply_text'
# Final reply text, with whitespace stripped. Used in the result of sending
# a `BaseRequest` to a `LanguageModelEngine`.
OUTPUT_KEY_REPLY_TEXT_STRIPPED = 'reply_text_stripped'
# Raw value returned by the underlying engine for the final reply, from which
# the reply text was extracted.
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
  # as well; if not, it may be cleaner to move this into `ExperimentResult`.
  info: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata=dataclasses_json.config(exclude=_exclude_empty),
  )

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
    trimmed = {_trim_key(k): _trim_value(repr(v)) for k, v in d.items()}
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
class ExperimentResult(ExecutionResult):
  """Full results of an experiment run on a given example, with metrics.

  Corresponds more or less one-to-one to the contents of a single record of the
  'results_debug.json' file that is output at the end of each experiment run.
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

  def to_compact_record(self) -> 'ExperimentResult':
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
  ) -> 'ExperimentResult':
    """Returns an ExperimentResult with the same content as execution_result."""
    experiment_result = ExperimentResult()
    for field in dataclasses.fields(ExecutionResult):
      setattr(
          experiment_result, field.name, getattr(execution_result, field.name)
      )
    return experiment_result


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


def experiment_result_from_dict(data: dict[str, Any]) -> ExperimentResult:
  """Returns an ExperimentResult restored from structure created by to_dict."""
  result = ExperimentResult.from_dict(data)
  # See note on `execution_result_from_dict` above for why this is needed.
  result.stages = list(execution_result_from_dict(s) for s in result.stages)
  return result


@dataclasses.dataclass
class ExperimentTiming:
  """Timing information for an experiment.

  Attributes:
    start_time: The start time of the experiment.
    end_time: The start time of the experiment.
    time_elapsed: The time elapsed from beginning to end of the experiment.
  """
  start_time: datetime.datetime = datetime.datetime.now()
  end_time: datetime.datetime = datetime.datetime.now()

  @property
  def time_elapsed(self) -> datetime.timedelta:
    return self.end_time - self.start_time


@dataclasses.dataclass
class ExperimentSummary:
  """Summary of the results of an experiment.

  Attributes:
    timing: Experiment timing information.
    metrics: Mapping of metric name to metric value.
    counters: Mapping of counter name to counter value.
    example_keys: Mapping of example index to example key.
    results: Mapping of example key to experiment result (w/o detailed trace).
    results_debug: Mapping of example key to experiment result (w/ detailed
      trace).
    final_states: Mapping of example key to final state of agent. Only relevant
      when the strategy is a subclass of `Agent`.
  """
  timing: ExperimentTiming = dataclasses.field(default_factory=ExperimentTiming)
  metrics: dict[str, float] = dataclasses.field(default_factory=dict)
  counters: collections.Counter[str] = dataclasses.field(
      default_factory=collections.Counter)
  example_keys: dict[int, str|int] = dataclasses.field(default_factory=dict)
  results: dict[str|int, ExperimentResult] = dataclasses.field(
      default_factory=dict)
  results_debug: dict[str|int, ExperimentResult] = dataclasses.field(
      default_factory=dict)
  final_states: dict[str|int, Any] = dataclasses.field(default_factory=dict)

  def replace_example_index_and_key(
      self, example_index: int, example_key: str|int) -> None:
    """Replaces the example index and keys in the summary with the given ones.

    This is intended to be used only in the case where the ExperimentSummary
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
          'Cannot replace example index and key in ExperimentSummary with '
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

  def __iadd__(self, other: ExperimentSummary) -> ExperimentSummary:
    """Adds the contents of the given experiment summary to the current one.

    Updates all attributes in-place.

    Note that we intentionally do not provide an implementation for ordinary
    `__add__`, as normally when adding experiment summary objects the typical
    pattern is to have one summary representing the experiment as a whole and
    then to add it summaries representing the incremental results from
    evaluation of individual examples in the experiment. In such cases, it is
    much more efficient to use `__iadd__`, so as to avoid unnecessary copying
    of the increasingly large summary object of the full experiment.

    Args:
      other: The other experiment summary to add.

    Returns:
      Self (after updating with the contents of `other`).
    """
    self.timing.start_time = min(
        self.timing.start_time, other.timing.start_time)
    self.timing.end_time = max(
        self.timing.end_time, other.timing.end_time)

    # Update the metrics as weighted averages based on number of examples.
    self_count = self.counters[COUNTER_TOTAL_EXAMPLES]
    other_count = other.counters[COUNTER_TOTAL_EXAMPLES]
    if self_count and other_count:
      if self.metrics.keys() != other.metrics.keys():
        raise ValueError(
            'Cannot add ExperimentSummary objects with inconsistent metric '
            f'keys ({self.metrics.keys()}) != ({other.metrics.keys()})')
    for metric_name in self.metrics.keys() | other.metrics.keys():
      self_value = self.metrics.get(metric_name, 0)
      other_value = other.metrics.get(metric_name, 0)
      numerator = self_value * self_count + other_value * other_count
      denominator = self_count + other_count
      self.metrics[metric_name] = numerator / denominator

    self.counters.update(other.counters)
    self.example_keys.update(other.example_keys)
    self.results.update(other.results)
    self.results_debug.update(other.results_debug)
    self.final_states.update(other.final_states)
    return self


@dataclasses.dataclass
class HTMLRenderer:
  """Renders ExecutionResult(s) or ExperimentSummary in HTML format.

  Attributes:
    levels_to_expand: Number of levels in the hierarchy to expand by default,
      when rendering a single result or list of results. (When rendering an
      entire ExperimentSummary, we display only the top level initially.)
  """

  # Public attributes.
  levels_to_expand: int = 3

  # =========================================================================
  # Private attributes. (Not exposing them publicly yet, as they may change.)
  # =========================================================================

  # Maximum length of the single-line representation of a stage's content string
  # to show when the stage is in collapsed state. If longer than this, it will
  # be truncated, with '...'.
  _max_stage_content_summary_length: int = 160

  # Maximum length for displaying a single value (e.g., input or output) of dict
  # or list type on a single line. If the value's string representation is
  # within this length, then it will be shown on a single line; if longer, then
  # it will be expanded into a separate line for each element in the list or
  # dict.
  _max_single_line_value_string_length: int = 80

  def _render_llm_request_reply(
      self, request: str, reply: str, element_id: str | None = None
  ) -> str:
    """Returns HTML rendering the request/reply as multi-line colored text."""
    # TODO: Support roles and multi-modal requests/replies.
    updated_request = request.replace('\n', '<br>')
    updated_reply = reply.replace('\n', '<br>')
    id_string = f' id="{element_id}"' if element_id is not None else ''
    return (
        f'<p{id_string} style="color:black;background-color:white"><span'
        f' style="color:black">{updated_request}</span><span'
        f' style="color:blue">{updated_reply}</span></p>'
    )

  def _render_single_value(
      self, value: Any, *, element_id: str | None = None
  ) -> str:
    """Returns a rendering of the value as text or in ...<ul>...</ul> form."""
    if not value:
      return repr(value)

    if (
        isinstance(value, list)
        and len(repr(value)) > self._max_single_line_value_string_length
    ):
      lines = []
      lines.append(f'{type(value).__name__}({len(value)})')
      lines.append('<ul id="{id}">')
      for i, element in enumerate(value):
        element_html = self._render_single_value(
            element, element_id=f'{element_id}-{i}'
        )
        lines.append(f'<li>{element_html}</li>')
      lines.append('</ul>')
      return '\n'.join(lines)

    if (
        isinstance(value, dict)
        and len(repr(value)) > self._max_single_line_value_string_length
    ):
      lines = []
      lines.append(f'{type(value).__name__}({len(value)})')
      lines.append(f'<ul id="{element_id}">')
      for i, (key, value) in enumerate(value.items()):
        element_html = self._render_single_value(
            value, element_id=f'{element_id}-{i}'
        )
        lines.append(f'<li><b>{key}</b>: {element_html}</li>')
      lines.append('</ul>')
      return '\n'.join(lines)

    # Default handling.
    return repr(value)

  def _render_result(
      self,
      result: ExecutionResult,
      *,
      element_id: str,
      stage_number: int | None = None,
      levels_to_expand: int = 0,
  ) -> str:
    """Returns a rendering of the ExecutionResult in <li>...</li>... form."""
    # Skip over redundant ExecutionResult objects that are simply empty wrappers
    # for a single ExecutionResult stage that contains the actual content. This
    # situation commonly occurs in the outermost layer of the ExecutionResult
    # that is returned when `executing.run` is called with
    # `enabled_tracing=True`.
    if (
        not result.stage_name
        and not result.inputs
        and not result.outputs
        and len(result.stages) == 1
    ):
      return self._render_result(
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
    # TODO: Remove the special treatment of the special outputs key
    # 'target', once we support storing non-dict targets directly in the
    # ExperimentResult.
    if isinstance(result, ExperimentResult):
      if isinstance(result.targets, dict) and set(result.targets.keys()) == {
          'target'
      }:
        targets = result.targets.get('target')
      else:
        targets = result.targets
    else:
      targets = {}

    stage_number_str = (
        f'[{stage_number+1}] ' if stage_number is not None else ''
    )
    stage_content_str = f'{result.inputs!r} <b>&rArr;</b> {outputs!r}'
    if len(stage_content_str) > self._max_stage_content_summary_length:
      stage_content_str = (
          stage_content_str[:self._max_stage_content_summary_length] + '...'
      )
    expanded = levels_to_expand > 0
    if expanded:
      content_string_style = 'display:none'
      content_block_style = 'display:block'
    else:
      content_string_style = 'display:inline'
      content_block_style = 'display:none'
    lines = []
    lines.append(
        f"<li onClick=\"toggleElements(['{element_id}', '{element_id}c'])\">"
        f'<b>{stage_number_str}<u>{result.stage_name}</u></b>'
        f' <div id="{element_id}c"'
        f' style="{content_string_style}">{stage_content_str}</div></li>'
    )
    if result.stage_name == 'generate_text' and not result.stages:
      # Special formatting for leaf-level LLM requests.
      request = result.inputs.get('request', 'MISSING_REQUEST_FIELD')
      reply = result.outputs.get('output', 'MISSING_OUTPUT_FIELD')
      request_reply_string = self._render_llm_request_reply(
          request=request, reply=reply
      )
      lines.append(
          f'<div id="{element_id}"'
          f' style="{content_block_style}">{request_reply_string}</div>'
      )
    else:
      # Default formatting for arbitrary prompting stages.
      lines.append(f'<ul id="{element_id}" style="{content_block_style}">')
      lines.append(f'<!--<li><b>stage_name</b>: {result.stage_name}</li>-->')
      inputs_string = self._render_single_value(
          result.inputs, element_id=f'{element_id}i'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'{element_id}i\')"><b>inputs</b>'
          f'</span>: {inputs_string}</li>'
      )
      outputs_string = self._render_single_value(
          outputs, element_id=f'{element_id}o'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'{element_id}o\')">'
          f'<b>outputs</b></span>: {outputs_string}</li>'
      )
      if result.error:
        lines.append(f'<li><b>error</b>: {result.error}</li>')
      if result.info:
        lines.append(f'<li><b>info</b>: {result.info}</li>')
      if isinstance(result, ExperimentResult):
        if targets:
          targets_string = self._render_single_value(
              targets, element_id=f'{element_id}t'
          )
          lines.append(
              '<li><span onClick="toggleElement(\'{element_id}t\')">'
              f'<b>targets</b></span>: {targets_string}</li>'
          )
        if result.metrics:
          metrics_string = self._render_single_value(
              result.metrics, element_id=f'{element_id}m'
          )
          lines.append(
              '<li><span onClick="toggleElement(\'{element_id}m\')">'
              f'<b>metrics</b></span>: {metrics_string}</li>'
          )
      if result.stages:
        for i, stage in enumerate(result.stages):
          lines.append(
              self._render_result(
                  stage,
                  element_id=f'{element_id}-{i}',
                  stage_number=i if len(result.stages) > 1 else None,
                  levels_to_expand=levels_to_expand - 1,
              )
          )
      lines.append('</ul>')
    return '\n'.join(lines)

  def _render_result_list(
      self,
      result_list: Sequence[ExecutionResult],
      *,
      element_id: str,
      levels_to_expand: int = 0,
  ) -> str:
    """Returns a rendering of the ExecutionResults in <ul>...</ul> form."""
    lines = []
    lines.append(f'<ul id="{element_id}">')
    for i, result in enumerate(result_list):
      lines.append(
          self._render_result(
              result,
              element_id=f'{element_id}-{i}',
              stage_number=i,
              levels_to_expand=levels_to_expand,
          )
      )
    lines.append('</ul>')
    return '\n'.join(lines)

  def _render_agent_state(
      self,
      state: Any,
      *,
      element_id: str,
      stage_number: int | None = None,
      levels_to_expand: int = 0,
  ) -> str:
    """Returns a rendering of the state as text or in ...<ul>...</ul> form."""
    if stage_number is None:
      stage_number_str = ''
    else:
      stage_number_str = f'[{stage_number+1}] '

    type_string = f'{type(state).__name__}'
    try:
      # Special handling for the common case of `UpdateListState`.
      # We detect this case via `try...except` because `UpdateListState` is a
      # generic and does not `instanceof`.
      content_str = (
          f'{state.inputs!r} <b>&rArr;</b> {len(state.updates)} updates'
      )
      # If we get this far, then `state` must be an instance of
      # `UpdateListState`, or at least something that looks and acts like one.
      is_update_list_state = True
    except TypeError:
      # Default handling where `state` is not an `UpdateListState`.
      content_str = repr(state)
      is_update_list_state = False

    expanded = (levels_to_expand > 0)
    if expanded:
      content_string_style = 'display:none'
      content_block_style = 'display:block'
    else:
      content_string_style = 'display:inline'
      content_block_style = 'display:none'

    if len(content_str) > self._max_stage_content_summary_length:
      content_str = content_str[:self._max_stage_content_summary_length] + '...'

    lines = []
    lines.append(
        f"<li onClick=\"toggleElements(['{element_id}', '{element_id}c'])\">"
        f'<b>{stage_number_str}<u>{type_string}</u></b> '
        f'<div id="{element_id}c" style="{content_string_style}">{content_str}'
        f'</div></li>'
    )
    if is_update_list_state:
      # Special handling for the common case of `UpdateListState`.
      lines.append(f'<ul id="{element_id}" style="{content_block_style}">')
      inputs_string = self._render_single_value(
          state.inputs, element_id=f'{element_id}i'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'{element_id}i\')"><b>inputs</b>'
          f'</span>: {inputs_string}</li>'
      )
      update_string = self._render_single_value(
          state.updates, element_id=f'{element_id}u'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'{element_id}u\')"><b>updates</b>'
          f'</span>: {update_string}</li>'
      )
      lines.append('</ul>')
    else:
      # Default handling for arbitrary `state` types.
      lines.append(
          f'<div id="{element_id}" style="{content_block_style}">'
          f'{repr(state)}</div>'
      )

    return '\n'.join(lines)

  def _render_agent_state_list(
      self, states: list[Any], *, element_id: str, levels_to_expand: int = 0
  ) -> str:
    """Returns a rendering of the states in <ul>...</ul> form."""
    lines = []
    lines.append(f'<ul id="{element_id}">')
    for i, state in enumerate(states):
      lines.append(
          self._render_agent_state(
              state,
              element_id=f'{element_id}-{i}',
              stage_number=i,
              levels_to_expand=levels_to_expand,
          )
      )
    lines.append('</ul>')
    return '\n'.join(lines)

  def _render_experiment_summary(self, summary: ExperimentSummary) -> str:
    """Returns a rendering of the ExperimentSummary in <li>...</li>... form."""
    # TODO: Shall we consider `self.levels_to_expand` here? For now
    # we're ignoring it, and instead always rendering the individual sections
    # initially in collapsed state when rendering an entire `ExperimentSummary`,
    # as it tends to look rather overwhelming to have many different
    # ExperimentResult objects expanded all at once.
    lines = []
    if summary.timing:
      timing_string = self._render_single_value(
          summary.timing, element_id='timing'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'timing\')"><b>timing</b></span>:'
          f' {timing_string}</li>'
      )
    if summary.metrics:
      metrics_string = self._render_single_value(
          summary.metrics, element_id='metrics'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'metrics\')"><b>metrics</b>'
          f'</span>: {metrics_string}</li>'
      )
    if summary.counters:
      counters_string = self._render_single_value(
          summary.counters, element_id='counters'
      )
      lines.append(
          '<li><span onClick="toggleElement(\'counters\')"><b>counters</b>'
          f'</span>: {counters_string}</li>'
      )
    if summary.results:
      assert len(summary.results) == len(summary.example_keys)
      results_list = [
          summary.results[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      lines.append(
          '<li><span onClick="toggleElement(\'results\')"><b>results</b>'
          f'</span>: {len(results_list)}</li>'
      )
      lines.append(
          self._render_result_list(
              results_list, element_id='results', levels_to_expand=0
          )
      )
    if summary.results_debug:
      assert len(summary.results_debug) == len(summary.example_keys)
      results_debug_list = [
          summary.results_debug[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      lines.append(
          '<li><span onClick="toggleElement(\'results_debug\')">'
          f'<b>results_debug</b></span>: {len(results_debug_list)}</li>'
      )
      lines.append(
          self._render_result_list(
              results_debug_list, element_id='results_debug', levels_to_expand=0
          )
      )
    if summary.final_states:
      assert len(summary.final_states) == len(summary.example_keys)
      final_state_list = [
          summary.final_states[summary.example_keys[i]]
          for i in sorted(summary.example_keys)
      ]
      lines.append(
          '<li><span onClick="toggleElement(\'final_states\')">'
          f'<b>final_states</b></span>: {len(final_state_list)}</li>'
      )
      lines.append(
          self._render_agent_state_list(
              final_state_list, element_id='final_states', levels_to_expand=0
          )
      )
    return '\n'.join(lines)

  def render(
      self,
      object_to_render: ExecutionResult | Sequence[ExecutionResult],
      *,
      element_id: str = '0',
  ) -> str:
    """Returns a full HTML + JavaScript rendering of the given result(s).

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
        if (element.style.display === 'none') {
          if (element_id.endsWith('c')) {
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
      content_string = self._render_result(
          object_to_render,
          element_id=element_id,
          levels_to_expand=self.levels_to_expand,
      )
    elif isinstance(object_to_render, Sequence):
      content_string = self._render_result_list(
          object_to_render,
          element_id=element_id,
          levels_to_expand=self.levels_to_expand,
      )
    elif isinstance(object_to_render, ExperimentSummary):
      content_string = self._render_experiment_summary(object_to_render)
    else:
      raise ValueError(
          f'Unsupported type {type(object_to_render)}): {object_to_render}'
      )

    rendered_html = f"""\
{javascript_string}
<div>
  <ul>
{textwrap.indent(content_string, '    ')}
  </ul>
</div>
"""
    return rendered_html
