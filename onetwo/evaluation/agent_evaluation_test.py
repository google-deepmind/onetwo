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

import collections
from collections.abc import Iterator
import datetime
import os
import pprint
import random
import string
import time
from typing import Any, Final, TypeAlias, cast

from absl.testing import absltest
from absl.testing import parameterized
import freezegun
import immutabledict
from onetwo.agents import agents_base
from onetwo.agents import agents_test_utils
from onetwo.backends import backends_test_utils
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import core_test_utils
from onetwo.core import executing
from onetwo.core import iterating
from onetwo.core import results
from onetwo.core import tracing
from onetwo.evaluation import agent_evaluation





_Example: TypeAlias = agent_evaluation.Example
_ExecutionResult: TypeAlias = results.ExecutionResult
_EvaluationResult: TypeAlias = results.EvaluationResult
_EvaluationSummary: TypeAlias = results.EvaluationSummary
_EvaluationTiming: TypeAlias = results.EvaluationTiming
_UpdateListState: TypeAlias = agents_base.UpdateListState


_EXAMPLES_WITH_ONE_OPTION: Final[tuple[_Example, ...]] = tuple([
    immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        'answer': f'bla_generated option={num}',
    })
    for num in range(5)
])
_EXAMPLES_WITH_TWO_OPTIONS: Final[tuple[_Example, ...]] = tuple([
    immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        'another_option': 'something',
        'answer': f'bla_generated option={num} another_option=something',
    })
    for num in range(5)
])


def _reset_times(eval_summary: _EvaluationSummary):
  for er in eval_summary.results.values():
    core_test_utils.reset_times(er)
  for er in eval_summary.results_debug.values():
    core_test_utils.reset_times(er)


def _get_iterator_of_examples(total_num: int) -> Iterator[_Example]:
  for num in range(total_num):
    yield immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        'answer': f'bla_generated option={num}',
    })


def _accuracy_function_that_returns_example_option(
    target: str, prediction: str, example: _Example
) -> tuple[float, dict[str, Any]]:
  correct_val = float(target == prediction)
  extra_info = {'some_key': example.get('option', None)}
  return correct_val, extra_info


def _aggregate_via_dict_update(
    aggregate_summary: _EvaluationSummary, example_summary: _EvaluationSummary
) -> None:
  """Aggregation function matching default behavior of `evaluation.evaluate`."""
  aggregate_summary.info.update(example_summary.info)


def ordinary_identity_function(x):
  return x


async def async_identity_function(x):
  return x


@executing.make_executable  # pytype: disable=wrong-arg-types
def executable_identity_function(x):
  return x


@tracing.trace  # pytype: disable=wrong-arg-types
def traced_ordinary_identity_function(x):
  return x


@executing.make_executable
@tracing.trace  # pytype: disable=wrong-arg-types
async def executable_traced_identity_function(x):
  return x


def ordinary_accuracy_function(target: str, prediction: str):
  return float(target == prediction)


async def async_accuracy_function(target: str, prediction: str):
  return float(target == prediction)


@executing.make_executable  # pytype: disable=wrong-arg-types
def executable_accuracy_function(target: str, prediction: str):
  return float(target == prediction)


@executing.make_executable  # pytype: disable=wrong-arg-types
def executable_accuracy_function_taking_example_as_arg(
    target: str, prediction: str, example: _Example
):
  del example
  return float(target == prediction)


@executing.make_executable  # pytype: disable=wrong-arg-types
async def strategy_taking_question_and_option_and_calling_llm(
    question: str, option: int, **kwargs
) -> str:
  """Simple strategy that requires `question` and `option` args."""
  result = await llm.generate_text(prompt=question)  # pytype: disable=wrong-keyword-args
  another_option = kwargs.get('another_option', None)
  if another_option is not None:
    return f'{result} option={option} another_option={another_option}'
  return f'{result} {option=}'


class OrdinaryAccuracyCallable:

  def __call__(self, target: str, prediction: str):
    return float(target == prediction)


class AsyncAccuracyCallable:

  async def __call__(self, target: str, prediction: str):
    return float(target == prediction)


class ExecutableAccuracyCallable:

  @executing.make_executable  # pytype: disable=wrong-arg-types
  def __call__(self, target: str, prediction: str):
    return float(target == prediction)


class ExecutableAccuracyCallableTakingExampleAsArg:

  @executing.make_executable  # pytype: disable=wrong-arg-types
  def __call__(self, target: str, prediction: str, example: _Example):
    del example
    return float(target == prediction)


class AgentEvaluationTest(parameterized.TestCase):

  def assertFileExists(self, filename: str):
    self.assertTrue(
        os.path.exists(filename),
        f'File expected to exist but does not: {filename}',
    )

  @parameterized.named_parameters(
      (
          'ordinary_function',
          ordinary_identity_function,
          'ordinary_identity_function',
      ),
      ('async_function', async_identity_function, 'async_identity_function'),
      (
          'executable_function',
          executable_identity_function,
          'executable_identity_function',
      ),
      (
          'traced_ordinary_function',
          traced_ordinary_identity_function,
          'traced_ordinary_identity_function',
      ),
      (
          'executable_traced_function',
          executable_traced_identity_function,
          'executable_traced_identity_function',
      ),
  )
  def test_evaluate_simple_strategies(self, strategy, expected_stage_name):
    examples = [
        {'question': 'a', 'answer': 'b'},
        {'question': 'c', 'answer': 'c'},
    ]
    now = datetime.datetime(2024, 5, 9, 12, 0, 0)
    with freezegun.freeze_time(now):
      summary = agent_evaluation.evaluate(
          strategy=strategy,
          examples=examples,
          metric_functions={'accuracy': ordinary_accuracy_function},
          output_results=True,
          output_results_debug=True,
          output_final_states=True,
      )
      _reset_times(summary)

    expected_timing = _EvaluationTiming(start_time=now, end_time=now)
    expected_results = {
        0: _EvaluationResult(
            stage_name=expected_stage_name,
            inputs={'x': 'a'},
            outputs={'output': 'a'},
            targets={'target': 'b'},
            metrics={
                'total_count': 1,
                'error_count': 0,
                'accuracy': 0.0,
                'accuracy_count': 1,
            },
        ),
        1: _EvaluationResult(
            stage_name=expected_stage_name,
            inputs={'x': 'c'},
            outputs={'output': 'c'},
            targets={'target': 'c'},
            metrics={
                'total_count': 1,
                'error_count': 0,
                'accuracy': 1.0,
                'accuracy_count': 1,
            },
        ),
    }
    expected_summary = _EvaluationSummary(
        metrics={'accuracy': 0.5},
        counters=collections.Counter(
            {'total_count': 2, 'error_count': 0, 'accuracy_count': 2}
        ),
        example_keys={0: 0, 1: 1},
        results=expected_results,
        # In this case, the results_debug should be the same as the results
        # because the strategy consists of just a single step.
        results_debug=expected_results,
    )

    # We separate out the timing field because it requires some special handling
    # for comparing the FakeDatetime objects returned by freezegun.
    actual_timing = summary.timing
    summary.timing = expected_summary.timing = None

    with self.subTest('timing'):
      self.assertEqual(expected_timing, actual_timing)

    with self.subTest('summary_except_timing'):
      self.assertMultiLineEqual(
          pprint.pformat(expected_summary, width=160),
          pprint.pformat(summary, width=160),
          f'Incorrect EvaluationSummary contents:\n{summary}\n----\nDiff',
      )

  def test_evaluate_agent(self):
    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [
        {'question': 'a', 'answer': 'b'},
        {'question': 'c', 'answer': 'c'},
    ]
    now = datetime.datetime(2024, 5, 9, 12, 0, 0)
    with freezegun.freeze_time(now):
      summary = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
          strategy=strategy,
          examples=examples,
          metric_functions={'accuracy': ordinary_accuracy_function},
          output_results=True,
          output_results_debug=True,
          output_final_states=True,
      )
      _reset_times(summary)

    expected_timing = _EvaluationTiming(start_time=now, end_time=now)
    expected_summary = _EvaluationSummary(
        metrics={'accuracy': 0.0},
        counters=collections.Counter(
            {'total_count': 2, 'error_count': 0, 'accuracy_count': 2}
        ),
        example_keys={0: 0, 1: 1},
        results={
            0: _EvaluationResult(
                stage_name='StringAgent',
                inputs={
                    'inputs': 'a',
                    'initial_state': None,
                    'max_steps': None,
                    'stop_condition': None,
                    'return_final_state': True,
                },
                outputs={'output': 'a b c'},
                targets={'target': 'b'},
                metrics={
                    'total_count': 1,
                    'error_count': 0,
                    'accuracy': 0.0,
                    'accuracy_count': 1,
                },
            ),
            1: _EvaluationResult(
                stage_name='StringAgent',
                inputs={
                    'inputs': 'c',
                    'initial_state': None,
                    'max_steps': None,
                    'stop_condition': None,
                    'return_final_state': True,
                },
                outputs={'output': 'd e f'},
                targets={'target': 'c'},
                metrics={
                    'total_count': 1,
                    'error_count': 0,
                    'accuracy': 0.0,
                    'accuracy_count': 1,
                },
            ),
        },
        results_debug={
            0: _EvaluationResult(
                stage_name='StringAgent',
                inputs={
                    'inputs': 'a',
                    'initial_state': None,
                    'max_steps': None,
                    'stop_condition': None,
                    'return_final_state': True,
                },
                outputs={'output': 'a b c'},
                stages=[
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(inputs='a', updates=[]),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['a']},
                    ),
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(
                                inputs='a', updates=['a']
                            ),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['b']},
                    ),
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(
                                inputs='a', updates=['a', 'b']
                            ),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['c']},
                    ),
                ],
                targets={'target': 'b'},
                metrics={
                    'total_count': 1,
                    'error_count': 0,
                    'accuracy': 0.0,
                    'accuracy_count': 1,
                },
            ),
            1: _EvaluationResult(
                stage_name='StringAgent',
                inputs={
                    'inputs': 'c',
                    'initial_state': None,
                    'max_steps': None,
                    'stop_condition': None,
                    'return_final_state': True,
                },
                outputs={'output': 'd e f'},
                stages=[
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(inputs='c', updates=[]),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['d']},
                    ),
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(
                                inputs='c', updates=['d']
                            ),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['e']},
                    ),
                    _ExecutionResult(
                        stage_name='SingleSampleAgent.sample_next_step',
                        inputs={
                            'state': _UpdateListState(
                                inputs='c', updates=['d', 'e']
                            ),
                            'num_candidates': 1,
                        },
                        outputs={'output': ['f']},
                    ),
                ],
                targets={'target': 'c'},
                metrics={
                    'total_count': 1,
                    'error_count': 0,
                    'accuracy': 0.0,
                    'accuracy_count': 1,
                },
            ),
        },
        final_states={
            0: _UpdateListState(inputs='a', updates=['a', 'b', 'c']),
            1: _UpdateListState(inputs='c', updates=['d', 'e', 'f']),
        },
    )

    # We separate out the timing field because it requires some special handling
    # for comparing the FakeDatetime objects returned by freezegun.
    actual_timing = summary.timing
    summary.timing = expected_summary.timing = None

    with self.subTest('timing'):
      self.assertEqual(expected_timing, actual_timing)

    with self.subTest('summary_except_timing'):
      self.assertMultiLineEqual(
          pprint.pformat(expected_summary, width=160),
          pprint.pformat(summary, width=160),
          f'Incorrect EvaluationSummary contents:\n{summary}\n----\nDiff',
      )

  def test_inputs_and_target_extractors(self):
    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [{'q': 'q1', 'context': 'context1', 'golden': 'b'}]
    summary = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
        strategy=strategy,
        examples=examples,
        inputs_extractor=lambda x: ([x['context'] + x['q']], {}),
        target_extractor=lambda x: x['golden'],
        output_results=True,
    )
    _reset_times(summary)

    with self.subTest('inputs_extractor'):
      self.assertEqual('context1q1', summary.results[0].inputs.get('inputs'))

    with self.subTest('target_extractor'):
      self.assertEqual('b', summary.results[0].targets.get('target'))

  @parameterized.named_parameters(
      ('output_results', True, False, False),
      ('output_results_debug', False, True, False),
      ('output_final_states', False, False, True),
  )
  def test_output_flags(
      self, output_results, output_results_debug, output_final_states
  ):
    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [{'question': 'q1', 'answer': 'b'}]
    summary = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
        strategy=strategy,
        examples=examples,
        output_results=output_results,
        output_results_debug=output_results_debug,
        output_final_states=output_final_states,
    )
    _reset_times(summary)

    with self.subTest('output_results'):
      self.assertEqual(output_results, bool(summary.results))
    with self.subTest('output_results_debug'):
      self.assertEqual(output_results_debug, bool(summary.results_debug))
    with self.subTest('output_final_states'):
      self.assertEqual(output_final_states, bool(summary.final_states))

  @parameterized.named_parameters(
      (
          'none',
          None,
          {0, 1},
      ),
      (
          'based_on_example',
          lambda example, result: example['question'] == 'a',
          {0},
      ),
      (
          'based_on_evaluation_result',
          lambda example, result: result.metrics['accuracy'] > 0.0,
          {1},
      ),
  )
  def test_output_filter(self, output_filter, expected_example_keys):
    examples = [
        {'question': 'a', 'answer': 'b'},  # Identity function will be wrong.
        {'question': 'c', 'answer': 'c'},  # Identity function will be correct.
    ]
    summary = agent_evaluation.evaluate(
        strategy=ordinary_identity_function,
        examples=examples,
        metric_functions={'accuracy': ordinary_accuracy_function},
        output_filter=output_filter,
        output_results=True,
        output_results_debug=True,
        output_final_states=True,
    )
    _reset_times(summary)

    with self.subTest('example_keys'):
      self.assertSameElements(
          expected_example_keys,
          summary.example_keys.values(),
          str(summary.example_keys),
      )

    with self.subTest('results_keys'):
      self.assertSameElements(
          expected_example_keys, summary.results.keys(), str(summary.results)
      )

    with self.subTest('results_debug_keys'):
      self.assertSameElements(
          expected_example_keys,
          summary.results_debug.keys(),
          str(summary.results_debug),
      )

    with self.subTest('metrics_should_be_unaffected_by_the_filter'):
      self.assertDictEqual({'accuracy': 0.5}, summary.metrics)

  def test_example_key_function(self):
    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [
        {'question': 'q1', 'answer': 'b'},
        {'question': 'q2', 'answer': 'c'},
    ]
    summary = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
        strategy=strategy,
        examples=examples,
        example_key_function=lambda index, example: str(example['question']),
        output_results=True,
        output_results_debug=True,
        output_final_states=True,
    )
    _reset_times(summary)

    with self.subTest('example_keys'):
      self.assertEqual({0: 'q1', 1: 'q2'}, summary.example_keys)
    with self.subTest('results_keys'):
      self.assertSameElements({'q1', 'q2'}, summary.results.keys())
    with self.subTest('results_debug_keys'):
      self.assertSameElements({'q1', 'q2'}, summary.results_debug.keys())
    with self.subTest('final_states_keys'):
      self.assertSameElements({'q1', 'q2'}, summary.final_states.keys())

  @parameterized.named_parameters(
      (
          'ordinary_function',
          ordinary_accuracy_function,
      ),
      (
          'async_function',
          async_accuracy_function,
      ),
      (
          'executable_function',
          executable_accuracy_function,
      ),
      (
          'function_taking_example_as_arg',
          executable_accuracy_function_taking_example_as_arg,
      ),
      ('ordinary_callable', OrdinaryAccuracyCallable()),
      ('async_callable', AsyncAccuracyCallable()),
      ('executable_callable', ExecutableAccuracyCallable()),
      (
          'callable_taking_example_as_arg',
          ExecutableAccuracyCallableTakingExampleAsArg(),
      ),
  )
  def test_metric_function_types(self, metric_function):
    examples = [
        {'question': 'a', 'answer': 'b'},  # Identity function will be wrong.
        {'question': 'c', 'answer': 'c'},  # Identity function will be correct.
    ]
    summary = agent_evaluation.evaluate(
        strategy=ordinary_identity_function,
        examples=examples,
        metric_functions={'accuracy': metric_function},
    )
    _reset_times(summary)
    self.assertDictEqual({'accuracy': 0.5}, summary.metrics)

  def test_metric_function_returning_extra_info(self):
    metric_function = lambda target, prediction, example: (
        float(target == prediction),
        {'accuracy_rationale': 'because', 'group': example['group']},
    )
    examples = [
        # Identity function will be wrong.
        {'question': 'a', 'answer': 'b', 'group': 'group1'},
        # Identity function will be correct.
        {'question': 'c', 'answer': 'c', 'group': 'group2'},
    ]
    summary = agent_evaluation.evaluate(
        strategy=ordinary_identity_function,
        examples=examples,
        metric_functions={'accuracy': metric_function},
        output_results=True,
    )
    _reset_times(summary)

    with self.subTest('metrics'):
      self.assertDictEqual({'accuracy': 0.5}, summary.metrics)

    with self.subTest('results_info'):
      self.assertDictEqual(
          {'accuracy_rationale': 'because', 'group': 'group1'},
          summary.results[0].info,
      )

  def test_strategy_arg_mismatch(self):
    summary = agent_evaluation.evaluate(
        strategy=strategy_taking_question_and_option_and_calling_llm,
        examples=[{'question': 'Something'}],
        metric_functions={'accuracy': ordinary_accuracy_function},
        # Note that we only extract `question`, not `option` (arg mismatch).
        inputs_extractor=lambda x: ([x['question']], {}),
        target_extractor=lambda x: '',
        output_results=True,
    )

    expected_result = _EvaluationResult(
        stage_name='strategy_taking_question_and_option_and_calling_llm',
        inputs={'args': ['Something'], 'kwargs': {}},
        error=TypeError(
            'strategy_taking_question_and_option_and_calling_llm()'
            " missing 1 required positional argument: 'option'"
        ),
        targets={'target': ''},
        metrics={
            'accuracy': 0.0,
            'accuracy_count': 1,
            'total_count': 1,
            'error_count': 1,
        },
    )

    with self.subTest('counters_should_reflect_error'):
      self.assertDictEqual(
          {'total_count': 1, 'error_count': 1, 'accuracy_count': 1},
          summary.counters,
      )

    with self.subTest('metrics_should_still_be_calculated'):
      self.assertDictEqual({'accuracy': 0.0}, summary.metrics)

    with self.subTest('result_should_include_inputs_and_error_message'):
      self.assertMultiLineEqual(
          pprint.pformat(expected_result, width=160),
          pprint.pformat(summary.results[0], width=160),
          f'Incorrect EvaluationResult contents:\n{summary}\n----\nDiff',
      )

  @parameterized.named_parameters(
      ('list_examples', _EXAMPLES_WITH_ONE_OPTION + _EXAMPLES_WITH_TWO_OPTIONS),
      ('generator_examples', _get_iterator_of_examples(5)),
  )
  def test_example_formats(self, examples):
    """Verifies evaluation of a simple strategy produces expected results."""
    def _fake_generate_text(prompt: str | content_lib.ChunkList) -> str:
      return f'{prompt}_generated'

    llm.generate_text.configure(_fake_generate_text)

    extract_all_example_fields_except_target = lambda x: (
        [],
        {k: v for k, v in x.items() if k != 'answer'},
    )

    summary = agent_evaluation.evaluate(
        strategy=strategy_taking_question_and_option_and_calling_llm,
        examples=examples,
        inputs_extractor=extract_all_example_fields_except_target,
        metric_functions={
            'accuracy': _accuracy_function_that_returns_example_option
        },
        aggregation_functions=[_aggregate_via_dict_update],
        output_results=True,
    )
    _reset_times(summary)
    timing: _EvaluationTiming = summary.timing

    with self.subTest('produces_plausible_timedelta'):
      self.assertGreater(timing.time_elapsed.total_seconds(), 0.0)

    with self.subTest('produces_correct_avg_metric'):
      # We expect all answers to be correct by design.
      self.assertEqual(
          1.0, summary.metrics['accuracy'], pprint.pformat(summary, width=160)
      )

    with self.subTest('produces_correct_aggregate_metric_info'):
      # The aggregation function we specified simply calls `update` on the
      # info dict.
      self.assertDictEqual({'some_key': 4}, summary.info)

  def test_llm_with_batching(self):
    """Verifies correct execution when running with batched LLMs."""
    examples = _EXAMPLES_WITH_ONE_OPTION
    inputs_extractor = lambda x: ([x['question']], {'option': x['option']})

    # First, sequential slow execution.
    fake_llm_model = backends_test_utils.LLMForTest(
        batch_size=1,
        wait_time_before_reply=datetime.timedelta(seconds=1.0),
        default_reply=lambda x: f'{x}_generated',
    )
    fake_llm_model.register()  # Configure llm.generate_text.
    # Runs in ~ 5sec.
    summary_slow = agent_evaluation.evaluate(
        strategy=strategy_taking_question_and_option_and_calling_llm,
        examples=examples,
        inputs_extractor=inputs_extractor,
        metric_functions={
            'accuracy': _accuracy_function_that_returns_example_option
        },
    )
    _reset_times(summary_slow)
    timing_slow: _EvaluationTiming = summary_slow.timing

    with self.subTest('runs_correctly_with_batchsize_eq_1'):
      # We expect all answers to be correct by design.
      self.assertEqual(1.0, summary_slow.metrics['accuracy'])

    # Second, parallel fast execution.
    fake_llm_model = backends_test_utils.LLMForTest(
        batch_size=len(examples),
        wait_time_before_reply=datetime.timedelta(seconds=1.0),
        default_reply=lambda x: f'{x}_generated',
    )
    fake_llm_model.register()  # Configure llm.generate_text.
    # Runs in ~ 1sec.
    summary_fast = agent_evaluation.evaluate(
        strategy=strategy_taking_question_and_option_and_calling_llm,
        examples=examples,
        inputs_extractor=inputs_extractor,
        metric_functions={
            'accuracy': _accuracy_function_that_returns_example_option
        },
    )
    _reset_times(summary_fast)
    timing_fast: _EvaluationTiming = summary_fast.timing

    with self.subTest('runs_correctly_with_batchsize_gt_1'):
      # We expect all answers to be correct by design.
      self.assertEqual(1.0, summary_fast.metrics['accuracy'])

    with self.subTest('runs_efficiently_with_batching'):
      self.assertBetween(timing_slow.time_elapsed.total_seconds(), 5.0, 6.0)
      self.assertBetween(timing_fast.time_elapsed.total_seconds(), 1.0, 2.0)

  def test_random_execution_order(self):
    """Verifies arbitrary execution order is handled correctly."""
    examples = [{'example_id': num, 'answer': str(num)} for num in range(10)]
    metric_function = lambda target, prediction: (
        float(target == prediction),
        {'processed_id': prediction},
    )

    @executing.make_executable  # pytype: disable=wrong-arg-types
    async def _strategy(example_id: int, **kwargs) -> str:
      del kwargs  # Not used.
      # Receives f'{example_id}_generated'.
      result = await llm.generate_text(prompt=str(example_id))  # pytype: disable=wrong-keyword-args
      return cast(str, result).replace('_generated', '')

    def _fake_generate_text_with_random_wait(
        prompt: str | content_lib.ChunkList,
    ) -> str:
      seconds_to_sleep = random.random()
      time.sleep(seconds_to_sleep)
      return f'{prompt}_generated'

    def _keep_track_of_execution_order(
        aggregate_summary: _EvaluationSummary,
        example_summary: _EvaluationSummary,
    ) -> None:
      if 'execution_order' not in aggregate_summary.info:
        aggregate_summary.info['execution_order'] = []
      aggregate_summary.info['execution_order'].append(
          example_summary.info['processed_id']
      )

    llm.generate_text.configure(
        iterating.to_thread(_fake_generate_text_with_random_wait),
    )
    summary = agent_evaluation.evaluate(
        strategy=_strategy,
        examples=examples,
        inputs_extractor=lambda example: ([example['example_id']], {}),
        metric_functions={'accuracy': metric_function},
        aggregation_functions=[_keep_track_of_execution_order],
    )
    _reset_times(summary)
    with self.subTest('examples_are_iterated_not_in_order'):
      # Chance of this subtest failing is one over 10 factorial, i.e. small.
      self.assertNotEqual(list(range(10)), summary.info['execution_order'])
    with self.subTest('evaluation_metrics_are_correct'):
      self.assertEqual(1.0, summary.metrics['accuracy'])

  def test_callback(self):
    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [
        {'question': 'a', 'answer': 'wrong'},  # This will be wrong.
        {'question': 'c', 'answer': 'd e f'},  # This will be correct.
    ]

    # Here we illustrate how to use a callback to do some kind of custom
    # processing of the results (in this case, just logging).
    log_messages = []
    def callback(
        example_index: int, example: _Example, summary: _EvaluationSummary
    ) -> None:
      log_messages.append(
          f"{example_index}: {example['question']} =>"
          f" {summary.metrics['accuracy']}"
      )

    _ = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
        strategy=strategy,
        examples=examples,
        metric_functions={'accuracy': ordinary_accuracy_function},
        callback=callback,
    )
    log_messages.sort()

    with self.subTest('callback_was_called_once_per_example'):
      self.assertLen(log_messages, 2)

    with self.subTest('log_messages_from_callback_reflect_correct_args'):
      self.assertSequenceEqual(['0: a => 0.0', '1: c => 1.0'], log_messages)

  def test_write_evaluation_summary_as_json(self):
    tmp_dir = self.create_tempdir().full_path
    output_dir = os.path.join(tmp_dir, 'test_write_evaluation_summary_as_json')
    os.makedirs(output_dir, exist_ok=True)

    strategy = agents_test_utils.StringAgent(
        max_length=3, sequence=list(string.ascii_lowercase)
    )
    examples = [
        {'question': 'a', 'answer': 'a b c'},
        {'question': 'c', 'answer': 'd e f'},
    ]
    now = datetime.datetime(2024, 5, 9, 12, 0, 0)
    with freezegun.freeze_time(now):
      summary = agent_evaluation.evaluate(  # pytype: disable=wrong-arg-types
          strategy=strategy,
          examples=examples,
          metric_functions={'accuracy': ordinary_accuracy_function},
          output_results=True,
          output_results_debug=True,
          output_final_states=True,
      )
      _reset_times(summary)

    agent_evaluation.write_evaluation_summary_as_json(
        summary=summary, output_dir=output_dir
    )

    # Assert that files were output successfully.
    counters_path = os.path.join(output_dir, 'counters.json')
    counters_json = core_test_utils.maybe_read_json(counters_path)
    counters = counters_json if counters_json else {}
    metrics_path = os.path.join(output_dir, 'metrics.json')
    metrics_json = core_test_utils.maybe_read_json(metrics_path)
    metrics = metrics_json if metrics_json else {}

    with self.subTest('should_output_metrics'):
      self.assertFileExists(metrics_path)

    with self.subTest('should_output_results'):
      self.assertFileExists(os.path.join(output_dir, 'results.json'))

    with self.subTest('should_output_results_debug'):
      self.assertFileExists(
          os.path.join(output_dir, 'results_debug.json')
      )

    with self.subTest('should_output_final_states'):
      self.assertFileExists(
          os.path.join(output_dir, 'final_states.json')
      )

    with self.subTest('counters_should_be_valid_json'):
      self.assertIsNotNone(counters_json)

    with self.subTest('counters_should_reflect_expected_results'):
      self.assertEqual(0, counters.get('error_count', None))
      self.assertEqual(2, counters.get('total_count', None))

    with self.subTest('metrics_should_be_valid_json'):
      self.assertIsNotNone(metrics_json)

    with self.subTest('metrics_should_reflect_expected_results'):
      self.assertEqual(1.0, metrics.get('accuracy', None))


if __name__ == '__main__':
  absltest.main()
