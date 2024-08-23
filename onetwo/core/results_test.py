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

import collections
from collections.abc import Sequence
import copy
import dataclasses
import datetime
import logging
import pprint
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import html5lib
from onetwo.agents import agents_base
from onetwo.builtins import formatting
from onetwo.core import content
from onetwo.core import results
import termcolor

STAGE_DECOMP = 'decomp'
STAGE_TRANSLATE = 'translate'


def _get_html_parse_error_string(
    parse_errors: Sequence[tuple[tuple[int, int], str, dict[str, Any]]],
    html: str,
) -> str:
  """Returns a readable representation of the html5lib.HTMLParser errors."""
  expanded_html_lines = html.splitlines()
  result_lines = []
  for pos, error_name, datavars in parse_errors:
    del datavars
    error_row = pos[0] - 1
    error_col = pos[1] - 1
    # Print the error message.
    result_lines.append(error_name)
    # Print the problematic line.
    result_lines.append(expanded_html_lines[error_row])
    result_lines.append(' ' * error_col + '^')
  return '\n'.join(result_lines)


class EvaluationResultTest(absltest.TestCase):

  def test_get_leaf_results(self):
    leaf1 = results.ExecutionResult(
        inputs={'request': 'Q: q A:'},
        outputs={'reply_text': 'To answer "q", we need: "q1".'},
    )
    leaf2 = results.ExecutionResult(
        inputs={'request': 'Q: q1 A:'},
        outputs={'reply_text': ' r1.'},
    )
    leaf3 = results.ExecutionResult(
        inputs={'request': 'Q: q1 A: r1.\nQ: q A:'},
        outputs={'reply_text': ' Wrong answer.'},
    )
    execution_result = results.ExecutionResult(
        inputs={'question': 'q'},
        outputs={'answer': 'Wrong answer'},
        stages=[
            results.ExecutionResult(
                stage_name=STAGE_DECOMP,
                inputs={'question': 'q'},
                outputs={
                    'decomposition': ['q1', 'q'],
                },
                stages=[leaf1],
            ),
            results.ExecutionResult(
                stage_name=STAGE_TRANSLATE,
                inputs={'decomposition': ['q1', 'q']},
                outputs={'answer': 'Wrong answer'},
                stages=[
                    leaf2,
                    leaf3,
                ],
            ),
        ],
    )
    expected_leaf_results = [leaf1, leaf2, leaf3]
    self.assertEqual(expected_leaf_results, execution_result.get_leaf_results())

  def test_format(self):
    execution_result = results.EvaluationResult(
        inputs={'question': 'q'},
        outputs={'answer': 'Wrong answer'},
        stages=[
            results.ExecutionResult(
                stage_name=STAGE_DECOMP,
                inputs={'question': 'q'},
                outputs={'decomposition': ['q1', 'q'],},
                stages=[
                    results.ExecutionResult(
                        inputs={'request': 'Q: q A:'},
                        outputs={
                            'reply_text': 'To answer "q", we need: "q1".'
                        },
                    ),
                ],
            ),
            results.ExecutionResult(
                stage_name=STAGE_TRANSLATE,
                inputs={'decomposition': ['q1', 'q']},
                outputs={'answer': 'Wrong answer'},
                stages=[
                    results.ExecutionResult(
                        inputs={'request': 'Q: q1 A:'},
                        outputs={
                            'reply_text': ' r1.'
                        },
                    ),
                    results.ExecutionResult(
                        inputs={'request': 'Q: q1 A: r1.\nQ: q A:'},
                        outputs={
                            'reply_text': ' Wrong answer.'
                        },
                    ),
                ],
            ),
        ],
    )

    expected_no_color = textwrap.dedent("""\
      decomp
        * Request/Reply
        Q: q A:<<To answer "q", we need: "q1".>>
        * Parsed decomposition
        ['q1', 'q']
      translate
        * Request/Reply
        Q: q1 A:<< r1.>>
        * Request/Reply
        Q: q1 A: r1.
        Q: q A:<< Wrong answer.>>
        * Parsed answer
        Wrong answer
      * Parsed answer
      Wrong answer""")

    expected_with_color = (
        termcolor.colored('decomp', attrs=['bold', 'underline']) + '\n'
        + '  ' + termcolor.colored('Request/Reply', attrs=['bold']) + '\n'
        + '  Q: q A:'
        + termcolor.colored('To answer "q", we need: "q1".', 'blue')
        + '\n'
        + '  ' + termcolor.colored('Parsed decomposition:', attrs=['bold'])
        + '\n'
        + '  ' + termcolor.colored("['q1', 'q']", 'magenta') + '\n'
        + termcolor.colored('translate', attrs=['bold', 'underline']) + '\n'
        + '  ' + termcolor.colored('Request/Reply', attrs=['bold']) + '\n'
        + '  Q: q1 A:'
        + termcolor.colored(' r1.', 'blue') + '\n'
        + '  ' + termcolor.colored('Request/Reply', attrs=['bold']) + '\n'
        + '  Q: q1 A: r1.\n'
        + '  Q: q A:'
        + termcolor.colored(' Wrong answer.', 'blue') + '\n'
        + '  ' + termcolor.colored('Parsed answer:', attrs=['bold']) + '\n'
        + '  ' + termcolor.colored('Wrong answer', 'magenta') + '\n'
        + termcolor.colored('Parsed answer:', attrs=['bold']) + '\n'
        + termcolor.colored('Wrong answer', 'magenta')
    )

    actual_no_color = execution_result.format(color=False)
    actual_with_color = execution_result.format(color=True)

    with self.subTest('without_color'):
      self.assertEqual(
          expected_no_color,
          actual_no_color,
          f'\n\nExpected:\n{expected_no_color}\nActual:\n\n{actual_no_color}',
      )

    with self.subTest('with_color'):
      self.assertEqual(
          expected_with_color,
          actual_with_color,
          f'\n\nExpected:\n{expected_with_color}\n\nActual:\n{actual_with_color}',
      )

  def test_format_non_string(self):
    execution_result = results.EvaluationResult(
        inputs={'request': '1+1'},
        outputs={'reply': '2'},
        stages=[],
    )

    expected_no_color = (
        '* Request/Reply'
        + '\n'
        + '1+1'
        + '\n'
        + '<<2>>'
    )

    expected_with_color = (
        termcolor.colored('Request/Reply', attrs=['bold'])
        + '\n'
        + '1+1'
        + '\n'
        + termcolor.colored('2', 'blue')
    )

    actual_no_color = execution_result.format(color=False)
    actual_with_color = execution_result.format(color=True)

    with self.subTest('without_color'):
      self.assertEqual(
          expected_no_color,
          actual_no_color,
          f'\n\nExpected:\n{expected_no_color}\nActual:\n\n{actual_no_color}',
      )

    with self.subTest('with_color'):
      self.assertEqual(
          expected_with_color,
          actual_with_color,
          f'\n\nExpected:\n{expected_with_color}'
          + f'\n\nActual:\n{actual_with_color}',
      )

  def test_to_dict_and_from_dict(self):
    evaluation_result = results.EvaluationResult(
        inputs={'question': 'q'},
        outputs={'answer': 'Wrong answer'},
        stages=[
            results.ExecutionResult(
                stage_name=STAGE_DECOMP,
                inputs={'question': 'q'},
                outputs={'decomposition': ['q1', 'q'],},
                stages=[
                    results.ExecutionResult(
                        inputs={'request': 'Q: q A:'},
                        outputs={
                            'reply_text': 'To answer "q", we need: "q1".'
                        },
                    ),
                ],
            ),
            results.ExecutionResult(
                stage_name=STAGE_TRANSLATE,
                inputs={'decomposition': ['q1', 'q']},
                outputs={'answer': 'Wrong answer'},
                stages=[
                    results.ExecutionResult(
                        inputs={'request': 'Q: q1 A:'},
                        outputs={
                            'reply_text': ' r1.'
                        },
                    ),
                    results.ExecutionResult(
                        inputs={'request': 'Q: q1 A: r1.\nQ: q A:'},
                        outputs={
                            'reply_text': ' Wrong answer.'
                        },
                    ),
                ],
            ),
        ],
        info={
            'record_id': 0,
            'sample_id': 0,
            'sample_size': 1,
        },
        targets={'answer': 'Gold answer'},
        metrics={'strict_accuracy': 0},
    )

    results_dict = evaluation_result.to_dict()
    expected_results_dict = {
        'inputs': {'question': 'q'},
        'outputs': {'answer': 'Wrong answer'},
        'stages': [
            {
                'stage_name': 'decomp',
                'inputs': {'question': 'q'},
                'outputs': {'decomposition': ['q1', 'q']},
                'stages': [{
                    'inputs': {'request': 'Q: q A:'},
                    'outputs': {'reply_text': 'To answer "q", we need: "q1".'},
                    'start_time': 0.0,
                    'end_time': 0.0,
                }],
                'start_time': 0.0,
                'end_time': 0.0,
            },
            {
                'stage_name': 'translate',
                'inputs': {'decomposition': ['q1', 'q']},
                'outputs': {'answer': 'Wrong answer'},
                'stages': [
                    {
                        'inputs': {'request': 'Q: q1 A:'},
                        'outputs': {'reply_text': ' r1.'},
                        'start_time': 0.0,
                        'end_time': 0.0,
                    },
                    {
                        'inputs': {'request': 'Q: q1 A: r1.\nQ: q A:'},
                        'outputs': {'reply_text': ' Wrong answer.'},
                        'start_time': 0.0,
                        'end_time': 0.0,
                    },
                ],
                'start_time': 0.0,
                'end_time': 0.0,
            },
        ],
        'info': {'record_id': 0, 'sample_id': 0, 'sample_size': 1},
        'targets': {'answer': 'Gold answer'},
        'metrics': {'strict_accuracy': 0},
        'start_time': 0.0,
        'end_time': 0.0,
    }

    with self.subTest('to_dict_excludes_empty_fields'):
      self.assertDictEqual(expected_results_dict, results_dict)

    recovered_results = results.evaluation_result_from_dict(results_dict)

    with self.subTest('roundtrip_recovers_original_contents'):
      self.assertEqual(evaluation_result, recovered_results)

  def test_to_compact_record(self):
    evaluation_result = results.EvaluationResult(
        inputs={
            'question': 'q',
            'original': 'INPUT: q OUTPUT: a',
            'record_id': 0,
            'exemplar': [{'question': 'q1', 'answer': 'a1'}],
        },
        outputs={'answer': 'a1', 'values_for_list_metrics': {'answer': ['a1']}},
        error='error',
        stages=[
            results.ExecutionResult(
                stage_name='stage1',
                inputs={'request': 'q1'},
                outputs={'reply': 'r1'},
            )
        ],
        info={'record_id': 0},
        targets={'answer': 'a'},
        metrics={'accuracy': 0.0},
    )

    original_evaluation_result = copy.deepcopy(evaluation_result)

    compact_record = evaluation_result.to_compact_record()

    expected_compact_record = results.EvaluationResult(
        inputs={'question': 'q'},
        outputs={'answer': 'a1'},
        error='error',
        stages=[],
        info={'record_id': 0},
        targets={'answer': 'a'},
        metrics={'accuracy': 0.0},
    )

    with self.subTest('should_omit_stages'):
      self.assertEmpty(compact_record.stages)

    with self.subTest('should_omit_exemplars'):
      self.assertNotIn('exemplar', compact_record.inputs)

    with self.subTest('should_omit_record_id'):
      self.assertNotIn('record_id', compact_record.inputs)

    with self.subTest('should_omit_values_for_list_metrics'):
      self.assertNotIn(
          results.VALUES_FOR_LIST_METRICS, compact_record.outputs
      )

    with self.subTest('should_have_same_content_otherwise'):
      self.assertEqual(expected_compact_record, compact_record)

    with self.subTest('original_should_remain_unchanged'):
      self.assertEqual(original_evaluation_result, evaluation_result)

  def test_from_execution_result(self):
    # First we create an ExecutionResult with some nested structure.
    execution_result = results.ExecutionResult(
        inputs={'question': 'q'},
        outputs={'answer': 'a1'},
        error='error',
        stages=[
            results.ExecutionResult(
                stage_name='stage1',
                inputs={'request': 'q1'},
                outputs={'reply': 'r1'},
            )
        ],
    )

    # Now we convert that to an EvaluationResult.
    evaluation_result = results.EvaluationResult.from_execution_result(
        execution_result)

    # Once we have converted it to an EvaluationResult, we can populate the
    # evaluation-specific fields.
    evaluation_result.info = {'record_id': 0}
    evaluation_result.targets = {'answer': 'a'}
    evaluation_result.metrics = {'accuracy': 0.0}

    # The resulting EvaluationResult should have all the contents of the
    # original ExecutionResult, plus the additional fields.
    expected_evaluation_result = results.EvaluationResult(
        inputs={'question': 'q'},
        outputs={'answer': 'a1'},
        error='error',
        stages=[
            results.ExecutionResult(
                stage_name='stage1',
                inputs={'request': 'q1'},
                outputs={'reply': 'r1'},
            )
        ],
        info={'record_id': 0},
        targets={'answer': 'a'},
        metrics={'accuracy': 0.0},
    )
    self.assertEqual(
        expected_evaluation_result,
        evaluation_result,
        f'Actual result: {pprint.pformat(evaluation_result)}',
    )

  def test_apply_formatting(self):
    res = results.ExecutionResult(
        stage_name='stage1',
        inputs={'question': 'q'},
        outputs={
            'answer': 'a1',
            'some super ...................... long key': (
                'some \n super long........................ \n value\n'
            ),
        },
        stages=[
            results.ExecutionResult(stage_name='stage2'),
            results.ExecutionResult(
                stage_name='stage3',
                stages=[results.ExecutionResult(stage_name='')],
            ),
            results.ExecutionResult(stage_name='', inputs={'a': 0, 'b': 1}),
        ],
    )
    expected_name_tree = textwrap.dedent("""\
        - stage1
          - stage2
          - stage3
            -
          -
        """)
    expected_name_keys_tree = textwrap.dedent("""\
        - stage1: ['question'] -> ['answer', 'some super ...................']
          - stage2: [] -> []
          - stage3: [] -> []
            - : [] -> []
          - : ['a', 'b'] -> []
        """)
    expected_short_values_tree = textwrap.dedent("""\
        - stage1:
          inputs: {'question': 'q'}
          outputs: {
            answer: 'a1'
            some super ...................: 'some \\n super long....[...]........... \\n value\\n'
          }
          - stage2:
            inputs: {}
            outputs: {}
          - stage3:
            inputs: {}
            outputs: {}
            - :
              inputs: {}
              outputs: {}
          - :
            inputs: {
              a: 0
              b: 1
            }
            outputs: {}
        """)

    with self.subTest('should_produce_name_tree'):
      self.assertEqual(expected_name_tree, results.get_name_tree(res))

    with self.subTest('should_produce_name_keys_tree'):
      self.assertEqual(expected_name_keys_tree, results.get_name_keys_tree(res))

    with self.subTest('should_produce_short_values_tree'):
      self.assertEqual(
          expected_short_values_tree,
          results.get_short_values_tree(res),
          results.get_short_values_tree(res),
      )


class EvaluationTimingTest(absltest.TestCase):

  def test_time_elapsed(self):
    start_time = datetime.datetime(2024, 5, 9, 12, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=3)
    timing = results.EvaluationTiming(start_time=start_time, end_time=end_time)
    self.assertEqual(datetime.timedelta(seconds=3), timing.time_elapsed)


class EvaluationSummaryTest(parameterized.TestCase):

  def test_iadd(self):
    start_time = datetime.datetime(2024, 5, 9, 12, 0, 0)
    summary1 = results.EvaluationSummary(
        timing=results.EvaluationTiming(
            start_time=start_time,
            end_time=start_time + datetime.timedelta(seconds=3),
        ),
        metrics={'accuracy': 0.5},
        counters=collections.Counter({
            results.COUNTER_TOTAL_EXAMPLES: 1,
            'accuracy_count': 1,
            'a': 1,
            'b': 1,
        }),
        info={'arbitrary_key1': ['arbitrary_value1']},
        example_keys={1: 'example1'},
        results={
            'example1': results.EvaluationResult(inputs={'input': '1'}),
        },
        results_debug={
            'example1': results.EvaluationResult(
                inputs={'input': '1'},
                stages=[results.ExecutionResult(stage_name='stage1_1')],
            ),
        },
        final_states={
            'example1': agents_base.UpdateListState(inputs='1', updates=['1'])
        },
    )
    summary2 = results.EvaluationSummary(
        timing=results.EvaluationTiming(
            start_time=start_time + datetime.timedelta(seconds=6),
            end_time=start_time + datetime.timedelta(seconds=9),
        ),
        metrics={'accuracy': 0.8},
        counters=collections.Counter({
            results.COUNTER_TOTAL_EXAMPLES: 2,
            'accuracy_count': 2,
            'b': 1,
            'c': 1,
        }),
        info={'arbitrary_key2': 'arbitrary_value2'},
        example_keys={2: 'example2', 3: 'example3'},
        results={
            'example2': results.EvaluationResult(inputs={'input': '2'}),
            'example3': results.EvaluationResult(inputs={'input': '3'}),
        },
        results_debug={
            'example2': results.EvaluationResult(
                inputs={'input': '2'},
                stages=[results.ExecutionResult(stage_name='stage2_1')],
            ),
            'example3': results.EvaluationResult(
                inputs={'input': '3'},
                stages=[results.ExecutionResult(stage_name='stage3_1')],
            ),
        },
        final_states={
            'example2': agents_base.UpdateListState(inputs='2', updates=['2']),
            'example3': agents_base.UpdateListState(inputs='3', updates=['3']),
        },
    )

    expected_sum = results.EvaluationSummary(
        timing=results.EvaluationTiming(
            start_time=start_time,
            end_time=start_time + datetime.timedelta(seconds=9),
        ),
        metrics={'accuracy': 0.7},
        counters=collections.Counter({
            results.COUNTER_TOTAL_EXAMPLES: 3,
            'accuracy_count': 3,
            'a': 1,
            'b': 2,
            'c': 1,
        }),
        # The `info` field is expected to be left unchanged during addition.
        info={'arbitrary_key1': ['arbitrary_value1']},
        example_keys={1: 'example1', 2: 'example2', 3: 'example3'},
        results={
            'example1': results.EvaluationResult(inputs={'input': '1'}),
            'example2': results.EvaluationResult(inputs={'input': '2'}),
            'example3': results.EvaluationResult(inputs={'input': '3'}),
        },
        results_debug={
            'example1': results.EvaluationResult(
                inputs={'input': '1'},
                stages=[results.ExecutionResult(stage_name='stage1_1')],
            ),
            'example2': results.EvaluationResult(
                inputs={'input': '2'},
                stages=[results.ExecutionResult(stage_name='stage2_1')],
            ),
            'example3': results.EvaluationResult(
                inputs={'input': '3'},
                stages=[results.ExecutionResult(stage_name='stage3_1')],
            ),
        },
        final_states={
            'example1': agents_base.UpdateListState(inputs='1', updates=['1']),
            'example2': agents_base.UpdateListState(inputs='2', updates=['2']),
            'example3': agents_base.UpdateListState(inputs='3', updates=['3']),
        },
    )

    summary1 += summary2
    # Adjusting for potential double precision error.
    for metric_name in summary1.metrics:
      summary1.metrics[metric_name] = round(summary1.metrics[metric_name], 8)

    with self.subTest('should_update_result_in_place'):
      self.assertMultiLineEqual(
          pprint.pformat(expected_sum, width=160),
          pprint.pformat(summary1, width=160),
          f'Incorrect EvaluationSummary contents:\n{summary1}\n----\nDiff'
      )

  @parameterized.named_parameters(
      (
          'empty',
          results.EvaluationSummary(),
          results.EvaluationSummary(),
          results.EvaluationSummary(),
      ),
      (
          'single_counter_shared_by_all_metrics',
          results.EvaluationSummary(
              metrics={'accuracy': 0.5, 'bleu': 0.7},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 1.0, 'bleu': 0.9},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 4}),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 0.9, 'bleu': 0.86},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 5}),
          ),
      ),
      (
          'separate_counters_per_metric',
          results.EvaluationSummary(
              metrics={'accuracy': 0.5, 'bleu': 0.7},
              counters=collections.Counter({
                  results.COUNTER_TOTAL_EXAMPLES: 1,
                  'accuracy_count': 1,
                  'bleu_count': 1,
              }),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 1.0, 'bleu': 0.9},
              counters=collections.Counter({
                  results.COUNTER_TOTAL_EXAMPLES: 4,
                  'accuracy_count': 1,
                  'bleu_count': 3,
              }),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 0.75, 'bleu': 0.85},
              counters=collections.Counter({
                  results.COUNTER_TOTAL_EXAMPLES: 5,
                  'accuracy_count': 2,
                  'bleu_count': 4,
              }),
          ),
      ),
      (
          'missing_value_not_counted',
          results.EvaluationSummary(
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 1.0},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 4}),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 1.0},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 5}),
          ),
      ),
      (
          'missing_counter_treated_as_zero',
          results.EvaluationSummary(
              metrics={'accuracy': 0.5},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 1.0},
          ),
          results.EvaluationSummary(
              metrics={'accuracy': 0.5},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
      ),
  )
  def test_iadd_metrics(self, summary1, summary2, expected_sum):
    summary1 += summary2
    # Adjusting for potential double precision error.
    for metric_name in summary1.metrics:
      summary1.metrics[metric_name] = round(summary1.metrics[metric_name], 8)
    self.assertMultiLineEqual(
        pprint.pformat(expected_sum, width=160),
        pprint.pformat(summary1, width=160),
        f'Incorrect EvaluationSummary contents:\n{summary1}\n----\nDiff'
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'fully_populated_result',
          'original': results.EvaluationSummary(
              metrics={'accuracy': 0.5},
              counters=collections.Counter(
                  {results.COUNTER_TOTAL_EXAMPLES: 1, 'a': 1, 'b': 1}
              ),
              example_keys={0: 'placeholder'},
              results={
                  'placeholder': results.EvaluationResult(
                      inputs={'input': '1'}
                  ),
              },
              results_debug={
                  'placeholder': results.EvaluationResult(
                      inputs={'input': '1'},
                      stages=[results.ExecutionResult(stage_name='stage1_1')],
                  ),
              },
              final_states={
                  'placeholder': agents_base.UpdateListState(
                      inputs='1', updates=['1']
                  )
              },
          ),
          'expected': results.EvaluationSummary(
              # Fields like metrics and counters are left as-is.
              metrics={'accuracy': 0.5},
              counters=collections.Counter(
                  {results.COUNTER_TOTAL_EXAMPLES: 1, 'a': 1, 'b': 1}
              ),
              # Example_keys is replaced with the new index and key.
              example_keys={1: 'example1'},
              # Results, results_debug, and final_states have their keys
              # replaced with the new key, but their values left as-is.
              results={
                  'example1': results.EvaluationResult(inputs={'input': '1'}),
              },
              results_debug={
                  'example1': results.EvaluationResult(
                      inputs={'input': '1'},
                      stages=[results.ExecutionResult(stage_name='stage1_1')],
                  ),
              },
              final_states={
                  'example1': agents_base.UpdateListState(
                      inputs='1', updates=['1']
                  )
              },
          ),
      },
      {
          'testcase_name': 'minimal_result',
          'original': results.EvaluationSummary(
              example_keys={0: 'placeholder'},
              results={
                  'placeholder': results.EvaluationResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'expected': results.EvaluationSummary(
              example_keys={1: 'example1'},
              results={
                  'example1': results.EvaluationResult(inputs={'input': '1'}),
              },
          ),
      },
      {
          'testcase_name': 'empty_summary',
          'original': results.EvaluationSummary(),
          'expected': results.EvaluationSummary(example_keys={1: 'example1'}),
      },
      {
          'testcase_name': 'example_keys_not_present',
          'original': results.EvaluationSummary(
              results={
                  'placeholder': results.EvaluationResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'expected': results.EvaluationSummary(
              example_keys={1: 'example1'},
              results={
                  'example1': results.EvaluationResult(inputs={'input': '1'}),
              },
          ),
      },
  )
  def test_replace_example_index_and_key_success_cases(
      self,
      original: results.EvaluationSummary,
      expected: results.EvaluationSummary,
  ):
    original.replace_example_index_and_key(1, 'example1')
    self.assertEqual(expected, original, original)

  @parameterized.named_parameters(
      (
          'multiple_keys',
          results.EvaluationSummary(
              example_keys={1: 'example1', 2: 'example2'},
              results={
                  'example1': results.EvaluationResult(inputs={'input': '1'}),
                  'example2': results.EvaluationResult(inputs={'input': '2'}),
              },
          ),
          'Cannot replace example index and key .* with multiple examples.',
      ),
      (
          'key_mismatch',
          results.EvaluationSummary(
              example_keys={0: 'placeholder'},
              results={
                  'some_other_key': results.EvaluationResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'Cannot replace example index and key .* with multiple examples.',
      ),
  )
  def test_replace_example_index_and_key_error_cases(
      self, summary: results.EvaluationSummary, expected_error_pattern: str
  ):
    with self.assertRaisesRegex(ValueError, expected_error_pattern):
      summary.replace_example_index_and_key(3, 'example3')


class HTMLRendererTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='int',
          object_to_render=123,
          expected_html='123',
      ),
      dict(
          testcase_name='string',
          object_to_render='ab',
          expected_html='&#x27;ab&#x27;',
      ),
      dict(
          testcase_name='list',
          object_to_render=['a', 'b'],
          expected_html='[&#x27;a&#x27;, &#x27;b&#x27;]',
      ),
      dict(
          testcase_name='dict',
          object_to_render={'a': 1, 'b': 2},
          expected_html='{&#x27;a&#x27;: 1, &#x27;b&#x27;: 2}',
      ),
      dict(
          testcase_name='aenum',
          object_to_render=content.PredefinedRole.USER,
          # Note: It's important that the <> signs are escaped here, or else
          # they will interfere with the HTML parsing.
          expected_html='&lt;PredefinedRole.USER: &#x27;user&#x27;&gt;',
      ),
  )
  def test_render_object(self, object_to_render, expected_html):
    renderer = results.HTMLRenderer()
    rendering = renderer.render_object(
        object_to_render, element_id='0', levels_to_expand=1
    )
    with self.subTest('html'):
      self.assertEqual(expected_html, rendering.html)

  @parameterized.named_parameters(
      ('empty_result', results.ExecutionResult()),
      (
          'result_with_single_element_in_each_container',
          results.EvaluationResult(
              inputs={'i1': 'i_v1'},
              outputs={'o1': 'o_v1'},
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
              ],
              targets={'t1': 't_v1'},
              metrics={'m1': 0.1},
          ),
      ),
      (
          'result_with_multiple_elements_in_each_container',
          results.EvaluationResult(
              inputs={'i1': 'i_v1', 'i2': 'i_v2'},
              outputs={'o1': 'o_v1', 'o2': 'o_v2'},
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
                  results.ExecutionResult(stage_name='stage2'),
              ],
              targets={'t1': 't_v1', 't2': 't_v2'},
              metrics={'m1': 0.1, 'm2': 0.2},
          ),
      ),
      ('empty_list', []),
      ('list_with_single_result', [results.ExecutionResult()]),
      (
          'list_with_multiple_results',
          [
              results.ExecutionResult(stage_name='result1'),
              results.EvaluationResult(stage_name='result2'),
          ],
      ),
      ('empty_evaluation summary', results.EvaluationSummary()),
      (
          'evaluation summary_with_single_element_in_each_container',
          results.EvaluationSummary(
              timing=results.EvaluationTiming(
                  start_time=datetime.datetime(2024, 5, 9, 12, 0, 0),
                  end_time=datetime.datetime(2024, 5, 9, 12, 0, 3),
              ),
              metrics={'m1': 0.1},
              counters=collections.Counter({'c1': 0.1}),
              example_keys={1: 'example1'},
              results={
                  'example1': results.EvaluationResult(
                      inputs={'i1': 'i_v1'},
                      outputs={'o1': 'o_v1'},
                      stages=[],
                      targets={'t1': 't_v1'},
                      metrics={'m1': 0.1},
                  ),
              },
              results_debug={
                  'example1': results.EvaluationResult(
                      inputs={'i1': 'i_v1'},
                      outputs={'o1': 'o_v1'},
                      stages=[
                          results.ExecutionResult(stage_name='stage1'),
                      ],
                      targets={'t1': 't_v1'},
                      metrics={'m1': 0.1},
                  ),
              },
              final_states={
                  'example1': agents_base.UpdateListState(
                      inputs='a', updates=['a', 'b', 'c']
                  )
              },
          ),
      ),
      ('list_with_single_summary', [results.EvaluationSummary()]),
      (
          'chat_result',
          # This is an example of a result that previously caused problems
          # when we were failing to escape special characters in the HTML.
          results.ExecutionResult(
              stage_name='llm.chat',
              inputs={
                  'messages': [
                      content.Message(
                          role=content.PredefinedRole.USER,
                          content=(
                              'This is a very very very very very very long'
                              ' question.'
                          ),
                      )
                  ],
                  'formatter': formatting.FormatterName.API,
              },
              outputs={'output': 'This is the answer.'},
          ),
      ),
  )
  def test_render_returns_valid_html(self, object_to_render: Any):
    renderer = results.HTMLRenderer()
    rendered_html = renderer.render(object_to_render)
    logging.info('Rendered HTML: %s', rendered_html)

    # Here we create a minimal full HTML page containing the rendered content
    # and verify that it is fully valid HTML (no mismatched HTML tags, etc.).
    expanded_html = f'<!DOCTYPE html><html>{rendered_html}</html>'
    parser = html5lib.HTMLParser()
    parser.parse(expanded_html)
    self.assertEmpty(
        parser.errors,
        _get_html_parse_error_string(parser.errors, expanded_html),
    )

  @parameterized.named_parameters(
      (
          'should_skip_empty_outer_result_with_single_stage',
          results.ExecutionResult(
              stages=[
                  results.ExecutionResult(
                      stage_name='stage1',
                      inputs={'i1': 'i_v1'},
                      outputs={'output': 'y'},
                  )
              ]
          ),
          1,
      ),
      (
          'should_not_skip_empty_outer_result_with_multiple_stages',
          results.ExecutionResult(
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
                  results.ExecutionResult(stage_name='stage2'),
              ]
          ),
          3,
      ),
      (
          'should_not_skip_outer_result_with_non_empty_name',
          results.ExecutionResult(
              stage_name='empty_outer_stage',
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
              ],
          ),
          2,
      ),
      (
          'should_not_skip_outer_result_with_non_empty_inputs',
          results.ExecutionResult(
              inputs={'i1': 'i_v1'},
              stages=[results.ExecutionResult(stage_name='stage1')],
          ),
          2,
      ),
      (
          'should_not_skip_outer_result_with_non_empty_outputs',
          results.ExecutionResult(
              outputs={'o1': 'o_v1'},
              stages=[results.ExecutionResult(stage_name='stage1')],
          ),
          2,
      ),
  )
  def test_render_skips_empty_outer_result_placeholder(
      self, object_to_render: Any, expected_number_of_result_blocks: bool
  ):
    # Note that we wrap the `object_to_render` in a list, as this will cause
    # the renderer to attempt to render the object in a collapsible form (via
    # referencing to the element_id), which allows us to more easily verify
    # whether the outer result element was skipped or not.
    renderer = results.HTMLRenderer()
    rendered_html = renderer.render([object_to_render], element_id='0')
    logging.info('Rendered HTML: %s', rendered_html)

    # We expect there to be exactly one `toggle_element` directive for each
    # result object that was actually displayed.
    self.assertEqual(
        expected_number_of_result_blocks,
        rendered_html.count('onClick="toggleElements'),
        f'Full html:\n{rendered_html}',
    )

  def test_render_generates_correct_content(self):
    @dataclasses.dataclass
    class MyDataClass:
      field1: str
      field2: str

    object_to_render = results.EvaluationResult(
        stage_name='MyStrategy',
        inputs={'i1': 'i_v1'},
        outputs={
            'o_very_long_key1': 'o_very_long_value1',
            'o_very_long_key2': 'o_very_long_value2',
            'o_very_long_key3': MyDataClass(
                field1='o_very_long_value3_1',
                field2='o_very_long_value3_2',
            ),
        },
        stages=[
            results.ExecutionResult(
                stage_name='stage1',
                inputs={'i2': 'i_v2'},
                outputs={'o2': 'o_v2'},
            ),
        ],
        targets={'t1': 't_v1'},
        metrics={'m1': 0.1},
    )

    renderer = results.HTMLRenderer()
    html = renderer.render(object_to_render)
    logging.info('Rendered HTML: %s', html)

    with self.subTest('javascript_appears_exactly_once'):
      self.assertEqual(
          1,
          html.count('function toggleElement(element_id)'),
          f'\nFull html:\n{html}',
      )

    with self.subTest('element_ids_are_correctly_paired_with_toggle_commands'):
      self.assertEqual(
          1,
          html.count("toggleElements(['0o', '0oc'])"),
          f'\nFull html:\n{html}',
      )
      self.assertEqual(1, html.count('id="0o"'), f'\nFull html:\n{html}')
      self.assertEqual(1, html.count('id="0oc"'), f'\nFull html:\n{html}')

    with self.subTest('top_level_stage_name'):
      self.assertIn(
          '<b><u>MyStrategy</u></b> ', html, f'\nFull html:\n{html}'
      )

    with self.subTest('inner_stage_names'):
      self.assertIn('<b><u>stage1</u></b> ', html, f'\nFull html:\n{html}')

    with self.subTest('result_keys'):
      self.assertIn('<b>inputs:</b> ', html, f'\nFull html:\n{html}')

    with self.subTest('result_collapsed_html'):
      self.assertIn(
          '{&#x27;i2&#x27;: &#x27;i_v2&#x27;} <b>&rArr;</b> {&#x27;o2&#x27;:'
          ' &#x27;o_v2&#x27;}',
          html,
          f'\nFull html:\n{html}',
      )

    with self.subTest('long_dicts_displayed_in_multiple_lines'):
      self.assertIn('<b>o_very_long_key1:</b> ', html, f'\nFull html:\n{html}')

    with self.subTest('long_dicts_include_data_type_and_size_on_first_line'):
      self.assertIn('dict(3)', html, f'\nFull html:\n{html}')

    with self.subTest('short_dicts_displayed_inline'):
      self.assertIn(
          '<li><b>inputs:</b> {&#x27;i1&#x27;: &#x27;i_v1&#x27;}</li>',
          html,
          f'\nFull html:\n{html}',
      )
      self.assertNotIn('<b>i1:</b> ', html, f'\nFull html:\n{html}')

    with self.subTest('dataclasses_display_like_dicts'):
      self.assertIn(
          '<span id="0o-2" style="display:inline">MyDataClass',
          html,
          f'\nFull html:\n{html}',
      )
      self.assertIn(
          '<li><b>field1:</b> &#x27;o_very_long_value3_1&#x27;</li>',
          html,
          f'\nFull html:\n{html}',
      )

  def test_custom_renderer(self):
    def dict_renderer(
        renderer: results.HTMLRenderer,
        object_to_render: Any,
        *,
        element_id: str,
        levels_to_expand: int,
    ) -> results.HTMLObjectRendering | None:
      """Renders a dict as a string of the form 'dict_renderer(keys=[...])'."""
      if not isinstance(object_to_render, dict):
        return None
      rendered_keys = [
          renderer.render_object(
              k,
              element_id=f'{element_id}-{i}',
              levels_to_expand=levels_to_expand - 1,
          ).html
          for i, k in enumerate(object_to_render.keys())
      ]
      return results.HTMLObjectRendering(
          html=f'dict_renderer(keys={rendered_keys})', collapsible=False
      )

    renderer = results.HTMLRenderer(custom_renderers=[dict_renderer])

    object_to_render = results.EvaluationResult(
        stage_name='MyStrategy',
        inputs={'i1': 'i_v1'},
        outputs={'o1': 'o_v1'},
        stages=[
            results.ExecutionResult(
                stage_name='stage1',
                inputs={'i2': 'i_v2'},
                outputs={'o2': 'o_v2'},
            ),
        ],
        targets={'t1': 't_v1'},
        metrics={'m1': 0.1},
    )

    html = renderer.render(object_to_render)
    logging.info('Rendered HTML: %s', html)

    with self.subTest('should_render_custom_content_for_dict'):
      self.assertIn(
          "<b>inputs:</b> dict_renderer(keys=['&#x27;i1&#x27;'])",
          html,
          f'\nFull html:\n{html}',
      )
      self.assertIn(
          "<b>outputs:</b> dict_renderer(keys=['&#x27;o1&#x27;'])",
          html,
          f'\nFull html:\n{html}',
      )

    with self.subTest('should_render_standard_content_for_other_types'):
      self.assertIn(
          '{&#x27;i2&#x27;: &#x27;i_v2&#x27;} <b>&rArr;</b> {&#x27;o2&#x27;:'
          ' &#x27;o_v2&#x27;}',
          html,
          f'\nFull html:\n{html}',
      )

if __name__ == '__main__':
  absltest.main()
