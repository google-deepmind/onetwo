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
import copy
import datetime
import logging
import pprint
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import html5lib
from onetwo.agents import agents_base
from onetwo.core import results
import termcolor

STAGE_DECOMP = 'decomp'
STAGE_TRANSLATE = 'translate'


class ExperimentResultsTest(absltest.TestCase):

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
    execution_result = results.ExperimentResult(
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
    execution_result = results.ExperimentResult(
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
    experiment_result = results.ExperimentResult(
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

    results_dict = experiment_result.to_dict()
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
                }],
            },
            {
                'stage_name': 'translate',
                'inputs': {'decomposition': ['q1', 'q']},
                'outputs': {'answer': 'Wrong answer'},
                'stages': [
                    {
                        'inputs': {'request': 'Q: q1 A:'},
                        'outputs': {'reply_text': ' r1.'},
                    },
                    {
                        'inputs': {'request': 'Q: q1 A: r1.\nQ: q A:'},
                        'outputs': {'reply_text': ' Wrong answer.'},
                    },
                ],
            },
        ],
        'info': {'record_id': 0, 'sample_id': 0, 'sample_size': 1},
        'targets': {'answer': 'Gold answer'},
        'metrics': {'strict_accuracy': 0},
    }

    with self.subTest('to_dict_excludes_empty_fields'):
      self.assertDictEqual(expected_results_dict, results_dict)

    recovered_results = results.experiment_result_from_dict(results_dict)

    with self.subTest('roundtrip_recovers_original_contents'):
      self.assertEqual(experiment_result, recovered_results)

  def test_to_compact_record(self):
    experiment_result = results.ExperimentResult(
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

    original_experiment_result = copy.deepcopy(experiment_result)

    compact_record = experiment_result.to_compact_record()

    expected_compact_record = results.ExperimentResult(
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
      self.assertEqual(original_experiment_result, experiment_result)

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

    # Now we convert that to an ExperimentResult.
    experiment_result = results.ExperimentResult.from_execution_result(
        execution_result)

    # Once we have converted it to an ExperimentResult, we can populate the
    # experiment-specific fields.
    experiment_result.info = {'record_id': 0}
    experiment_result.targets = {'answer': 'a'}
    experiment_result.metrics = {'accuracy': 0.0}

    # The resulting ExperimentResult should have all the contents of the
    # original ExecutionResult, plus the additional fields.
    expected_experiment_result = results.ExperimentResult(
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
        expected_experiment_result,
        experiment_result,
        f'Actual result: {pprint.pformat(experiment_result)}',
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


class ExperimentTimingTest(absltest.TestCase):

  def test_time_elapsed(self):
    start_time = datetime.datetime(2024, 5, 9, 12, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=3)
    timing = results.ExperimentTiming(start_time=start_time, end_time=end_time)
    self.assertEqual(datetime.timedelta(seconds=3), timing.time_elapsed)


class ExperimentSummaryTest(parameterized.TestCase):

  def test_iadd(self):
    start_time = datetime.datetime(2024, 5, 9, 12, 0, 0)
    summary1 = results.ExperimentSummary(
        timing=results.ExperimentTiming(
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
            'example1': results.ExperimentResult(inputs={'input': '1'}),
        },
        results_debug={
            'example1': results.ExperimentResult(
                inputs={'input': '1'},
                stages=[results.ExecutionResult(stage_name='stage1_1')],
            ),
        },
        final_states={
            'example1': agents_base.UpdateListState(inputs='1', updates=['1'])
        },
    )
    summary2 = results.ExperimentSummary(
        timing=results.ExperimentTiming(
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
            'example2': results.ExperimentResult(inputs={'input': '2'}),
            'example3': results.ExperimentResult(inputs={'input': '3'}),
        },
        results_debug={
            'example2': results.ExperimentResult(
                inputs={'input': '2'},
                stages=[results.ExecutionResult(stage_name='stage2_1')],
            ),
            'example3': results.ExperimentResult(
                inputs={'input': '3'},
                stages=[results.ExecutionResult(stage_name='stage3_1')],
            ),
        },
        final_states={
            'example2': agents_base.UpdateListState(inputs='2', updates=['2']),
            'example3': agents_base.UpdateListState(inputs='3', updates=['3']),
        },
    )

    expected_sum = results.ExperimentSummary(
        timing=results.ExperimentTiming(
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
            'example1': results.ExperimentResult(inputs={'input': '1'}),
            'example2': results.ExperimentResult(inputs={'input': '2'}),
            'example3': results.ExperimentResult(inputs={'input': '3'}),
        },
        results_debug={
            'example1': results.ExperimentResult(
                inputs={'input': '1'},
                stages=[results.ExecutionResult(stage_name='stage1_1')],
            ),
            'example2': results.ExperimentResult(
                inputs={'input': '2'},
                stages=[results.ExecutionResult(stage_name='stage2_1')],
            ),
            'example3': results.ExperimentResult(
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
          f'Incorrect ExperimentSummary contents:\n{summary1}\n----\nDiff'
      )

  @parameterized.named_parameters(
      (
          'empty',
          results.ExperimentSummary(),
          results.ExperimentSummary(),
          results.ExperimentSummary(),
      ),
      (
          'single_counter_shared_by_all_metrics',
          results.ExperimentSummary(
              metrics={'accuracy': 0.5, 'bleu': 0.7},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 1.0, 'bleu': 0.9},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 4}),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 0.9, 'bleu': 0.86},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 5}),
          ),
      ),
      (
          'separate_counters_per_metric',
          results.ExperimentSummary(
              metrics={'accuracy': 0.5, 'bleu': 0.7},
              counters=collections.Counter({
                  results.COUNTER_TOTAL_EXAMPLES: 1,
                  'accuracy_count': 1,
                  'bleu_count': 1,
              }),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 1.0, 'bleu': 0.9},
              counters=collections.Counter({
                  results.COUNTER_TOTAL_EXAMPLES: 4,
                  'accuracy_count': 1,
                  'bleu_count': 3,
              }),
          ),
          results.ExperimentSummary(
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
          results.ExperimentSummary(
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 1.0},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 4}),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 1.0},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 5}),
          ),
      ),
      (
          'missing_counter_treated_as_zero',
          results.ExperimentSummary(
              metrics={'accuracy': 0.5},
              counters=collections.Counter({results.COUNTER_TOTAL_EXAMPLES: 1}),
          ),
          results.ExperimentSummary(
              metrics={'accuracy': 1.0},
          ),
          results.ExperimentSummary(
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
        f'Incorrect ExperimentSummary contents:\n{summary1}\n----\nDiff'
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'fully_populated_result',
          'original': results.ExperimentSummary(
              metrics={'accuracy': 0.5},
              counters=collections.Counter(
                  {results.COUNTER_TOTAL_EXAMPLES: 1, 'a': 1, 'b': 1}
              ),
              example_keys={0: 'placeholder'},
              results={
                  'placeholder': results.ExperimentResult(
                      inputs={'input': '1'}
                  ),
              },
              results_debug={
                  'placeholder': results.ExperimentResult(
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
          'expected': results.ExperimentSummary(
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
                  'example1': results.ExperimentResult(inputs={'input': '1'}),
              },
              results_debug={
                  'example1': results.ExperimentResult(
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
          'original': results.ExperimentSummary(
              example_keys={0: 'placeholder'},
              results={
                  'placeholder': results.ExperimentResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'expected': results.ExperimentSummary(
              example_keys={1: 'example1'},
              results={
                  'example1': results.ExperimentResult(inputs={'input': '1'}),
              },
          ),
      },
      {
          'testcase_name': 'empty_summary',
          'original': results.ExperimentSummary(),
          'expected': results.ExperimentSummary(example_keys={1: 'example1'}),
      },
      {
          'testcase_name': 'example_keys_not_present',
          'original': results.ExperimentSummary(
              results={
                  'placeholder': results.ExperimentResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'expected': results.ExperimentSummary(
              example_keys={1: 'example1'},
              results={
                  'example1': results.ExperimentResult(inputs={'input': '1'}),
              },
          ),
      },
  )
  def test_replace_example_index_and_key_success_cases(
      self,
      original: results.ExperimentSummary,
      expected: results.ExperimentSummary,
  ):
    original.replace_example_index_and_key(1, 'example1')
    self.assertEqual(expected, original, original)

  @parameterized.named_parameters(
      (
          'multiple_keys',
          results.ExperimentSummary(
              example_keys={1: 'example1', 2: 'example2'},
              results={
                  'example1': results.ExperimentResult(inputs={'input': '1'}),
                  'example2': results.ExperimentResult(inputs={'input': '2'}),
              },
          ),
          'Cannot replace example index and key .* with multiple examples.',
      ),
      (
          'key_mismatch',
          results.ExperimentSummary(
              example_keys={0: 'placeholder'},
              results={
                  'some_other_key': results.ExperimentResult(
                      inputs={'input': '1'}
                  ),
              },
          ),
          'Cannot replace example index and key .* with multiple examples.',
      ),
  )
  def test_replace_example_index_and_key_error_cases(
      self, summary: results.ExperimentSummary, expected_error_pattern: str
  ):
    with self.assertRaisesRegex(ValueError, expected_error_pattern):
      summary.replace_example_index_and_key(3, 'example3')


class HTMLRendererTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_result', results.ExecutionResult()),
      (
          'result_with_single_element_in_each_container',
          results.ExperimentResult(
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
          results.ExperimentResult(
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
              results.ExperimentResult(stage_name='result2'),
          ],
      ),
      ('empty_experiment_summary', results.ExperimentSummary()),
      (
          'experiment_summary_with_single_element_in_each_container',
          results.ExperimentSummary(
              timing=results.ExperimentTiming(
                  start_time=datetime.datetime(2024, 5, 9, 12, 0, 0),
                  end_time=datetime.datetime(2024, 5, 9, 12, 0, 3),
              ),
              metrics={'m1': 0.1},
              counters=collections.Counter({'c1': 0.1}),
              example_keys={1: 'example1'},
              results={
                  'example1': results.ExperimentResult(
                      inputs={'i1': 'i_v1'},
                      outputs={'o1': 'o_v1'},
                      stages=[],
                      targets={'t1': 't_v1'},
                      metrics={'m1': 0.1},
                  ),
              },
              results_debug={
                  'example1': results.ExperimentResult(
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
  )
  def test_render_returns_valid_html(self, object_to_render: Any):
    renderer = results.HTMLRenderer()
    html = renderer.render(object_to_render)
    logging.info('Rendered HTML: %s', html)

    # Here we create a minimal full HTML page containing the rendered content
    # and verify that it is fully valid HTML (no mismatched HTML tags, etc.).
    expanded_html = f'<!DOCTYPE html><html>{html}</html>'
    parser = html5lib.HTMLParser()
    parser.parse(expanded_html)
    self.assertEmpty(parser.errors)

  @parameterized.named_parameters(
      (
          'empty_outer_result_with_single_stage',
          results.ExecutionResult(
              stages=[results.ExecutionResult(stage_name='stage1')]
          ),
          True,
      ),
      (
          'empty_outer_result_with_multiple_stages',
          results.ExecutionResult(
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
                  results.ExecutionResult(stage_name='stage2'),
              ]
          ),
          False,
      ),
      (
          'outer_result_has_non_empty_name',
          results.ExecutionResult(
              stage_name='empty_outer_stage',
              stages=[
                  results.ExecutionResult(stage_name='stage1'),
              ]
          ),
          False,
      ),
      (
          'outer_result_has_non_empty_inputs',
          results.ExecutionResult(
              inputs={'i1': 'i_v1'},
              stages=[results.ExecutionResult(stage_name='stage1')],
          ),
          False,
      ),
      (
          'outer_result_has_non_empty_outputs',
          results.ExecutionResult(
              outputs={'o1': 'o_v1'},
              stages=[results.ExecutionResult(stage_name='stage1')]
          ),
          False,
      ),
  )
  def test_render_skips_empty_outer_result_placeholder(
      self, object_to_render: Any, should_skip_outer_result: bool
  ):
    renderer = results.HTMLRenderer()
    html = renderer.render(object_to_render, element_id='0')
    logging.info('Rendered HTML: %s', html)

    skipped_outer_result = 'id="0"' not in html
    self.assertEqual(should_skip_outer_result, skipped_outer_result)

if __name__ == '__main__':
  absltest.main()
