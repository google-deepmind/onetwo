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

"""Library for evaluating agents and other prompting strategies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
import copy
import datetime
import functools
import pprint
import traceback
from typing import Any, ParamSpec, Protocol, TypeAlias, TypeVar

from onetwo.agents import agents_base
from onetwo.core import executing
from onetwo.core import results
from onetwo.core import tracing
from onetwo.core import updating
from onetwo.core import utils
import tqdm

_Args = ParamSpec('_Args')
_I = TypeVar('_I')
_O = TypeVar('_O')

Example: TypeAlias = Mapping[str, Any]


MetricValue: TypeAlias = float | None
MetricValueWithExtraInfo: TypeAlias = tuple[MetricValue, Mapping[str, Any]]


EvaluationMetricPossibleReturnTypes: TypeAlias = (
    # Option 1: Just return a float (or None).
    MetricValue | Awaitable[MetricValue] | executing.Executable[MetricValue] |
    # Option 2: Return a float (or None), paired with a dict of extra info.
    MetricValueWithExtraInfo | Awaitable[MetricValueWithExtraInfo] |
    executing.Executable[MetricValueWithExtraInfo]
)


class MetricFunctionWithExampleArg(Protocol[_O]):
  """Function to compute an evaluation metric for a given example.

  The most basic notion of a metric function is that it takes a target and a
  prediction, and returns a float representing how good the prediction is, in
  comparison to the target.

  A number of variations, however, are also supported:
  * The function can take the example as an additional argument, which can be
    used to retrieve additional information about the example that is not
    otherwise available from the target or prediction.
  * The function can be async, or decorated with `@executing.make_executable`,
    e.g., for a metric function that is backed by an AI rater.
  * The function can return `None`, in which case no value for the given metric
    will be recorded for that example, and that example will not be included
    when calculating the aggregated value of that metric over the dataset.
  * The function can return along with the float value a dict of extra
    information, which can be used to store arbitrary additional information
    about the example that is not necessarily represented as a float (e.g., in
    the case of an AI rater that internally uses chain-of-thought prompting, the
    extra info could include the AI rater's rationale). If provided, the extra
    info will be included in the `info` field of the `ExperimentResult` and
    `ExperimentSummary` for the example.
  """

  def __call__(
      self,
      target: _O,
      prediction: _O,
      example: Example | None = None,
  ) -> EvaluationMetricPossibleReturnTypes:
    """Returns the value of the metric (+extra info?) for a given example."""
    ...


class MetricFunctionWithoutExampleArg(Protocol[_O]):
  """Metric function that does not take the example as an argument."""

  def __call__(
      self,
      target: _O,
      prediction: _O,
  ) -> EvaluationMetricPossibleReturnTypes:
    """Returns the value of the metric (+extra info?) for a given example."""
    ...


MetricFunction: TypeAlias = (
    MetricFunctionWithExampleArg | MetricFunctionWithoutExampleArg
)


class AggregationFunction(Protocol):
  """Function for performing custom aggregation of experiment result info."""

  def __call__(
      self,
      aggregate_summary: results.ExperimentSummary,
      example_summary: results.ExperimentSummary,
  ) -> None:
    """Updates `aggregate_summary.info` based on `example_summary.info`.

    Args:
      aggregate_summary: Aggregated results for the evaluation run so far,
        including the current example (i.e, after updating metrics, counters,
        etc., in the standard way, and possibly after applying some other custom
        aggregation functions).
      example_summary: Results for the current example, which are to be merged
        into the aggregate results.
    """
    ...


@executing.make_executable
async def _execute_with_tracing(
    strategy: Callable[_Args, _O] | Callable[_Args, executing.Executable[_O]],
    *args: _Args.args,
    **kwargs: _Args.kwargs,
) -> tuple[_O, results.ExecutionResult]:
  """Returns the result of executing the strategy, along with a trace.

  Unlike `executing.run(..., enable_tracing=True)`, which can only be applied
  once at the outermost level of a given evaluation run, this function can be
  applied multiple places within an evaluation strategy. The canonical use case
  is to individually wrap the evaluation of each example within a run over a
  larger dataset, so that we can get a trace associated with that specific
  example and stream it together with the example's results.

  The intention would be to eventually allow calls to `_execute_with_tracing` to
  be nested at arbitrary levels of the prompting strategy, without interference.
  At that point, it could make sense to move this function into a more central
  location such as the `executing` module for wider reuse. Currently, however,
  there are some caveats, in that using `_execute_with_tracing` essentially
  hijacks the tracing mechanism, which means that it cannot be safely nested and
  cannot be used meaningfully with `executing.run(..., enable_tracing=True)`

  Args:
    strategy: The strategy to execute. Can be an arbitrary callable (async or
      ordinary function, decorated with `@executing.make_executable or not,
      agent, other callable, etc.).
    *args: Positional arguments to pass to the strategy.
    **kwargs: Keyword arguments to pass to the strategy.
  """

  # TODO: We are wrapping the whole call to the strategy in a fresh
  # ExecutionResult object so as to ensure that we can subsequently extract a
  # single ExecutionResult object containing precisely the trace of what
  # occurred during the call to the strategy. What is the best solution in the
  # long run? Should we just throw this temporary ExecutionResult away once we
  # have extracted the child ExecutionResult from it? Or will we ever need to
  # keep it (e.g., if for whatever reason it ends up having multiple child
  # stages)? What should we do about the existing value of the context variable
  # (if any)? Do we need to attach the new ExecutionResult object(s) as stage(s)
  # of it?
  parent_trace = results.ExecutionResult(stage_name='_execute_with_tracing')
  tracing.execution_context.set(parent_trace)

  @tracing.trace(skip=['return_final_state'])
  @functools.wraps(strategy)
  async def wrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> _O:
    return await utils.call_and_maybe_await(strategy, *args, **kwargs)

  prediction = await wrapper(*args, **kwargs)

  # Harvest the trace that was defined in the wrapper function above.
  trace = copy.deepcopy(tracing.execution_context.get(None))
  if len(trace.stages) != 1:
    raise ValueError(
        'Expected exactly one stage in the trace:'
        f' {pprint.pformat(trace, width=160)}'
    )
  trace = trace.stages[0]

  # If the underlying function was already being traced, then the outer trace
  # layer defined via the wrapper will be redundant, and we can omit it.
  is_outer_trace_redundant = len(trace.stages) == 1 and (
      isinstance(strategy, agents_base.Agent)
      or (
          trace.stages[0].stage_name == trace.stage_name
          and trace.stages[0].inputs == trace.inputs
          and trace.stages[0].outputs == trace.outputs
      )
  )
  if is_outer_trace_redundant:
    trace = trace.stages[0]

  # TODO: Make sure that if an outer tracer is defined, the traces
  # from `function` and its sub-stages still show up as stages of the outer
  # trace object, the same as if `function` were called directly.
  # if tracer is not None:
  #   tracing.execution_tracer.reset(tracer_token)

  return prediction, trace


@executing.make_executable
async def _evaluate_example(
    strategy: Callable[[_I], _O] | Callable[[_I], executing.Executable[_O]],
    example: Example,
    *,
    inputs_extractor: Callable[[Example], _I] = lambda x: x['question'],
    target_extractor: Callable[[Example], _O] = lambda x: x['answer'],
    metric_functions: dict[str, MetricFunction] | None = None,
    output_final_states: bool = False,
) -> tuple[Example, results.ExperimentSummary]:
  """Evaluates the given strategy on the given example.

  Args:
    strategy: The prompting strategy to evaluate.
    example: The example to evaluate the strategy on..
    inputs_extractor: Function that given an example returns a tuple of args and
      kwargs to pass as inputs to the strategy.
    target_extractor: Function that given an example returns the target value
      (if any) to which the prediction can be compared for determining accuracy.
    metric_functions: Mapping of metric name to a function that calculates the
      value of that metric for a single example. For each entry in this mapping,
      a corresponding entry will be added in `ExperimentSummary.metrics`,
      containing the average of that metric's values across all examples.
    output_final_states: Whether to populate `final_states` field in
      `ExperimentSummary`. Only applicable to agent strategies.

  Returns:
    A tuple of the example and an experiment summary containing the results for
    just that example.
  """
  summary = results.ExperimentSummary(
      timing=results.ExperimentTiming(
          start_time=datetime.datetime.now(), end_time=datetime.datetime.now()
      ),
  )
  args, kwargs = inputs_extractor(example)
  target = target_extractor(example)

  prediction = None
  execution_result = None
  final_state = None
  error = None

  try:
    if isinstance(strategy, agents_base.Agent):
      (prediction, final_state), execution_result = await _execute_with_tracing(
          strategy, *args, **kwargs, return_final_state=True
      )
    else:
      prediction, execution_result = await _execute_with_tracing(
          strategy, *args, **kwargs
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    traceback.print_exc()
    error = e

  # Counters
  summary.counters[results.COUNTER_TOTAL_EXAMPLES] = 1
  summary.counters[results.COUNTER_ERRORS] = 1 if error else 0

  # Metrics (and extra info)
  extra_info = {}
  if metric_functions:
    for metric_name, metric_function in metric_functions.items():
      try:
        metric_value = await utils.call_and_maybe_await(
            metric_function,
            target=target,
            prediction=prediction,
            example=example,
        )
      except TypeError:
        # Since not all metric functions require access to the `example` dict,
        # we fall back to trying calling the metric function with just the
        # `target` and `prediction` arguments.
        metric_value = await utils.call_and_maybe_await(
            metric_function, target=target, prediction=prediction
        )
      if isinstance(metric_value, tuple):
        metric_value, extra_info = metric_value
      if metric_value is not None:
        summary.metrics[metric_name] = metric_value
        metric_counter_name = results.get_metric_counter_name(metric_name)
        summary.counters[metric_counter_name] = 1

  # The below placeholders will be replaced by an actual example key calculated
  # on the caller side.
  example_index = 0
  example_key = 'placeholder'

  # Example keys
  summary.example_keys[example_index] = example_key

  # Results and results debug
  if execution_result is None:
    # Error occurred during strategy execution. Output just a minimal summary.
    experiment_debug = results.ExperimentResult(
        stage_name=strategy.__name__,
        inputs={'args': args, 'kwargs': kwargs},
        error=error,
    )
  else:
    # Strategy executed without error. Output a full summary.
    experiment_debug = results.ExperimentResult.from_execution_result(
        execution_result
    )

  if final_state:
    # For agents, we are setting `return_final_state=True` internally as a
    # means of retrieving the final_state. This has the side effect that the
    # `outputs` of the agent show up as a tuple of `(answer, final_state)`,
    # rather than simply `answer`. We clean this up manually here, so that
    # final_state appears only in `summary.final_states`, not in the `outputs`
    # of `summary.results` or `summary.results_debug`.
    if not isinstance(experiment_debug.outputs, dict) or set(
        experiment_debug.outputs.keys()
    ) != {'output'}:
      raise ValueError(
          'Expected agent outputs to be a dict with a single key `output`,'
          f' but got: {pprint.pformat(experiment_debug.outputs)}'
      )
    output = experiment_debug.outputs['output']
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or output[1] != final_state
    ):
      raise ValueError(
          'Expected agent output to be a tuple of length 2, where the 2nd'
          f' element is the final_state, but got: {pprint.pformat(output)}'
      )
    experiment_debug.outputs['output'] = output[0]

  if error and not experiment_debug.error:
    experiment_debug.error = error
  experiment_debug.counters = summary.counters
  # TODO: Modify `ExperimentResult.targets` to accept `Any`, and
  # then store the target value there directly rather than wrapping as a dict.
  experiment_debug.targets = {'target': target}
  # TODO: Modify `ExperimentResult` to separate `counters` from
  # `metrics`.
  experiment_debug.metrics = {}
  experiment_debug.metrics.update(summary.counters)
  experiment_debug.metrics.update(summary.metrics)
  if extra_info:
    experiment_debug.info.update(extra_info)
  experiment_result = copy.deepcopy(experiment_debug)
  experiment_result.stages = []
  summary.results[example_key] = experiment_result
  summary.results_debug[example_key] = experiment_debug

  # Final states
  if output_final_states and final_state is not None:
    summary.final_states[example_key] = final_state

  summary.timing.end_time = datetime.datetime.now()

  return example, summary


def evaluate(
    strategy: Callable[[_I], _O] | Callable[[_I], executing.Executable[_O]],
    examples: Iterable[Example],
    *,
    inputs_extractor: Callable[[Example], _I] = lambda x: ([x['question']], {}),
    target_extractor: Callable[[Example], _O] = lambda x: x['answer'],
    metric_functions: Mapping[str, MetricFunction] | None = None,
    aggregation_functions: Sequence[AggregationFunction] | None = None,
    examples_total_num: int | None = None,
    output_results: bool = False,
    output_results_debug: bool = False,
    output_final_states: bool = False,
    output_filter: (
        Callable[[Example, results.ExperimentResult], bool] | None
    ) = None,
    example_key_function: Callable[[int, Example], str | int] | None = None,
) -> results.ExperimentSummary:
  """Evaluates the given strategy on the given examples.

  Args:
    strategy: Prompting strategy to evaluate. Can be an arbitrary callable
      (async or ordinary function, decorated with `@executing.make_executable`
      or not, agent, other callable, etc.).
    examples: Examples on which to evaluate the strategy.
    inputs_extractor: Function that given an example returns a tuple of args and
      kwargs to pass as inputs to the strategy.
    target_extractor: Function that given an example returns the target value
      (if any) to which the prediction can be compared for determining accuracy.
    metric_functions: Mapping of metric name to a function that calculates the
      value of that metric for a single example. For each entry in this mapping,
      a corresponding entry will be added in `ExperimentSummary.metrics`,
      containing the average of that metric's value across all examples.
    aggregation_functions: Functions for performing custom aggregation of
      example-level information across a dataset.
    examples_total_num: Even if examples object has no implementation of __len__
      (examples could be a generator function with `yield`) user may still know
      (and provide) its exact length. This value is used only for logging the
      progress of evaluation.
    output_results: Whether to populate `results` field in `ExperimentSummary`.
    output_results_debug: Whether to populate `results_debug` field in
      `ExperimentSummary`.
    output_final_states: Whether to populate `final_states` field in
      `ExperimentSummary`. Only applicable to agent strategies.
    output_filter: Function to indicate whether a given example should have its
      details included in the results/traces/final_states. If not specified,
      then all examples will be included.
    example_key_function: Function to determine the key for a given example, for
      use in the results/traces/final_states mappings. If not specified, then
      will use the example index as the key.

  Returns:
    ExperimentSummary containing the evaluation results.
  """
  if hasattr(examples, '__len__'):
    examples_len = len(examples)
  elif examples_total_num is not None:
    examples_len = examples_total_num
  else:
    # In this case tqdm will only print progress without progressbar and ETA.
    examples_len = None

  experiment_summary = results.ExperimentSummary(
      timing=results.ExperimentTiming(
          start_time=datetime.datetime.now(), end_time=datetime.datetime.now()
      ),
  )
  eval_executable = executing.par_iter(
      _evaluate_example(
          strategy=strategy,
          example=example,
          inputs_extractor=inputs_extractor,
          target_extractor=target_extractor,
          metric_functions=metric_functions,
          output_final_states=output_final_states,
      )
      for example in examples
  )

  with executing.safe_stream(eval_executable, iteration_depth=1) as iterator:
    pbar = tqdm.tqdm(iterator, total=examples_len)
    update: updating.ListUpdate  # ListUpdate because of par_iter.
    for update in pbar:
      if len(update.payload) != 1:
        raise ValueError(
            'ListUpdate returned by stream_updates is expected to have '
            f'exactly one element in its payload. Got {pprint.pformat(update)}'
        )
      # Payload contains a single element of the form
      # (critic_result, example_id).
      (example, example_experiment_summary), example_index = update.payload[0]
      if example_key_function:
        example_key = example_key_function(example_index, example)
      else:
        example_key = example_index
      example_experiment_summary.replace_example_index_and_key(
          example_index=example_index, example_key=example_key
      )

      if len(example_experiment_summary.results) != 1:
        raise ValueError(
            'Expected exactly one result in ExperimentSummary.results when'
            ' evaluating on a single example. Got'
            f' {pprint.pformat(example_experiment_summary.results)}.\nFull'
            f' summary: {pprint.pformat(example_experiment_summary)}'
        )
      experiment_result = list(example_experiment_summary.results.values())[0]
      example_experiment_summary.info = experiment_result.info
      # TODO: Support storing traces output by custom tracers.

      if output_filter:
        include_outputs_for_example = output_filter(example, experiment_result)
      else:
        include_outputs_for_example = True

      if not include_outputs_for_example:
        example_experiment_summary.example_keys = {}
      if not output_results or not include_outputs_for_example:
        example_experiment_summary.results = {}
      if not output_results_debug or not include_outputs_for_example:
        example_experiment_summary.results_debug = {}
      if not output_final_states or not include_outputs_for_example:
        example_experiment_summary.final_states = {}

      experiment_summary += example_experiment_summary
      if aggregation_functions:
        for aggregation_function in aggregation_functions:
          aggregation_function(experiment_summary, example_experiment_summary)

  # TODO: Capture the backend and tool caches and store them in the
  # experiment summary too.

  return experiment_summary
