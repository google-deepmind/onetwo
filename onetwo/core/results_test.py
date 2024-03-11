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

import copy
import pprint
import textwrap

from absl.testing import absltest
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


if __name__ == '__main__':
  absltest.main()
