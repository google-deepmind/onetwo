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

"""Tests for evaluation code."""

from collections.abc import Iterator
import datetime
import functools
import random
import textwrap
import time
from typing import Any, Final, cast

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
from onetwo.backends import backends_test_utils
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import iterating
from onetwo.evaluation import evaluation


_GROUND_TRUTH_KEY: Final[str] = 'answer'
_GENERATE_TEXT_MAX_WAIT_SECS: Final[datetime.timedelta] = (
    datetime.timedelta(seconds=1.0)
)
_EXAMPLES_WITH_ONE_OPTION: Final[tuple[evaluation.Example, ...]] = tuple([
    immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        _GROUND_TRUTH_KEY: f'bla_generated option={num}',
    })
    for num in range(5)
])
_EXAMPLES_WITH_TWO_OPTIONS: Final[tuple[evaluation.Example, ...]] = tuple([
    immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        'another_option': 'something',
        _GROUND_TRUTH_KEY: (
            f'bla_generated option={num} another_option=something'
        ),
    })
    for num in range(5)
])


def _get_iterator_of_examples(total_num: int) -> Iterator[evaluation.Example]:
  for num in range(total_num):
    yield immutabledict.immutabledict({
        'question': 'bla',
        'option': num,
        _GROUND_TRUTH_KEY: f'bla_generated option={num}',
    })


def _fake_generate_text(prompt: str | content_lib.ChunkList) -> str:
  return f'{prompt}_generated'


def _fake_generate_text_returns_anything(
    prompt: str | content_lib.ChunkList,
    return_value: str,
) -> str:
  del prompt
  return return_value


def _fake_generate_text_with_random_wait(
    prompt: str | content_lib.ChunkList,
) -> str:
  time_to_sleep = _GENERATE_TEXT_MAX_WAIT_SECS.total_seconds()
  time_to_sleep *= random.random()
  time.sleep(time_to_sleep)
  return f'{prompt}_generated'


def _metric_fn(
    answer: str,
    example: evaluation.Example,
) -> evaluation.MetricResult:
  correct_val = float(answer == example[_GROUND_TRUTH_KEY])
  extra_info = {'some_key': example.get('option', None)}
  return correct_val, extra_info


@executing.make_executable()
async def simple_strategy(question: str, option: int, **kwargs) -> str:
  """Simple strategy that requires `question` and `option` args."""
  if _GROUND_TRUTH_KEY in kwargs:
    del kwargs[_GROUND_TRUTH_KEY]  # Never look at ground truth when evalutaing!
  result = await llm.generate_text(prompt=question)
  another_option = kwargs.get('another_option', None)
  if another_option is not None:
    return f'{result} option={option} another_option={another_option}'
  return f'{result} {option=}'


class EvaluateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

  def test_strategy_arg_mismatch(self):
    executables = evaluation.compile_strategies(
        strategies=[simple_strategy],
        examples=[{'question': 'Something'}],
    )
    with self.subTest('raises_value_error_when_missing_args'):
      with self.assertRaisesRegex(
          TypeError,
          'required positional argument: \'option\'',
      ):
        for ex in executables:
          _ = executing.run(ex[0])

  @parameterized.named_parameters(
      ('list_examples', _EXAMPLES_WITH_ONE_OPTION + _EXAMPLES_WITH_TWO_OPTIONS),
      ('generator_examples', _get_iterator_of_examples(5)),
  )
  def test_simple_strategy(self, examples):
    """Verifies evaluation of a simple strategy produces expected results."""
    llm.generate_text.configure(_fake_generate_text)

    time_elapsed, avg_metric, metric_info = evaluation.evaluate(
        strategy=simple_strategy,
        examples=examples,
        critic=_metric_fn,
    )

    with self.subTest('produces_plausible_timedelta'):
      self.assertGreater(time_elapsed.total_seconds(), 0.0)

    with self.subTest('produces_correct_avg_metric'):
      # We expect all answers to be correct by design.
      self.assertEqual(avg_metric, 1.0)

    with self.subTest('produces_correct_aggregate_metric_info'):
      # By default the metric info dict gets simply updated.
      self.assertEqual(metric_info, {'some_key': 4})

  def test_llm_with_batching(self):
    """Verifies correct execution when running with batched LLMs."""
    examples = _EXAMPLES_WITH_ONE_OPTION
    # First, sequential slow execution.
    fake_llm_model = backends_test_utils.LLMForTest(
        batch_size=1,
        wait_time_before_reply=datetime.timedelta(seconds=1.0),
        default_reply=lambda x: f'{x}_generated',
    )
    fake_llm_model.register()  # Configure llm.generate_text.
    # Runs in ~ 5sec.
    time_elapsed_slow, avg_metric, _ = evaluation.evaluate(
        strategy=simple_strategy,
        examples=examples,
        critic=_metric_fn,
    )

    with self.subTest('runs_correctly_with_batchsize_eq_1'):
      # We expect all answers to be correct by design.
      self.assertEqual(avg_metric, 1.0)

    # Second, parallel fast execution.
    fake_llm_model = backends_test_utils.LLMForTest(
        batch_size=len(examples),
        wait_time_before_reply=datetime.timedelta(seconds=1.0),
        default_reply=lambda x: f'{x}_generated',
    )
    fake_llm_model.register()  # Configure llm.generate_text.
    # Runs in ~ 1sec.
    time_elapsed_fast, avg_metric, _ = evaluation.evaluate(
        strategy=simple_strategy,
        examples=examples,
        critic=_metric_fn,
    )
    with self.subTest('runs_correctly_with_batchsize_gt_1'):
      # We expect all answers to be correct by design.
      self.assertEqual(avg_metric, 1.0)

    with self.subTest('runs_efficiently_with_batching'):
      self.assertBetween(time_elapsed_slow.total_seconds(), 5., 6.)
      self.assertBetween(time_elapsed_fast.total_seconds(), 1., 2.)

  def test_random_execution_order(self):
    """Verifies arbitrary execution order is handled correctly."""
    examples = [
        {
            'example_id': num,
            _GROUND_TRUTH_KEY: str(num),
        }
        for num in range(10)
    ]

    @executing.make_executable()
    async def _strategy(example_id: int, **kwargs) -> str:
      del kwargs  # Not used.
      # Receives f'{example_id}_generated'.
      result = await llm.generate_text(prompt=str(example_id))
      return cast(str, result).replace('_generated', '')

    def _keep_track_of_execution_order(
        aggr_info: dict[str, Any],
        new_info: evaluation.ExtraInfo,
    ) -> None:
      if 'execution_order' not in aggr_info:
        aggr_info['execution_order'] = []
      aggr_info['execution_order'].append(new_info['processed_id'])

    llm.generate_text.configure(
        iterating.to_thread(_fake_generate_text_with_random_wait),
    )
    _, avg_metric, aggr_info = evaluation.evaluate(
        strategy=_strategy,
        examples=examples,
        critic=lambda answer, example: (
            float(answer == example[_GROUND_TRUTH_KEY]),
            {'processed_id': answer},
        ),
        update_extra_info_fn=_keep_track_of_execution_order,
    )

    with self.subTest('examples_are_iterated_not_in_order'):
      # Chance of this subtest failing is one over 10 factorial, i.e. small.
      self.assertNotEqual(aggr_info['execution_order'], list(range(10)))
    with self.subTest('evaluation_metrics_are_correct'):
      self.assertEqual(avg_metric, 1.0)

  @parameterized.named_parameters(
      (
          'no_question_key',
          {'bla': 'bla', 'golden_answer': 'bla'},
          'Example does not contain the question key',
      ),
      (
          'no_golden_answer_key',
          {'question': 'bla'},
          'Example does not contain the golden answer key',
      ),
  )
  def test_naive_evaluation_critic_example_error_handling(
      self, example, expected_error_message
  ):
    with self.assertRaisesRegex(ValueError, expected_error_message):
      _ = executing.run(
          evaluation.naive_evaluation_critic('answer 1', example=example)
      )

  @parameterized.named_parameters(
      (
          'critic_beginning_unexpected',
          ' bla',
          'Critic is expected to start its answer from " yes" or " no" .',
      ),
  )
  def test_naive_evaluation_critic_llm_reply_error_handling(
      self, critic_llm_reply, expected_error_message
  ):
    llm.generate_text.configure(
        _fake_generate_text_returns_anything, return_value=critic_llm_reply
    )
    with self.assertRaisesRegex(ValueError, expected_error_message):
      _ = executing.run(
          evaluation.naive_evaluation_critic(
              'answer 1',
              example={'question': 'bla', 'golden_answer': 'bla'},
          )
      )

  @parameterized.named_parameters(
      ('yes_with_reason_lowercase', 'yes\nreason: because\n\n', 1.0, 'because'),
      ('yes_with_reason_uppercase', 'Yes\nReason: because\n\n', 1.0, 'because'),
      ('no_with_reason_lowercase', 'no\nReason: because\n\n', 0.0, 'because'),
      ('no_with_reason_uppercase', 'no\nReason: because\n\n', 0.0, 'because'),
      ('yes_without_reason', 'yes', 1.0, ''),
      ('no_without_reason', 'no', 0.0, ''),
      (
          'yes_with_reason_bold',
          '**yes**\n**Reason:** because\n\n',
          1.0,
          'because',
      ),
      (
          'no_with_reason_bold',
          '**no**\n**Reason:** because\n\n',
          0.0,
          'because',
      ),
      (
          'should_not_be_confused_by_no_in_reason',
          'yes\nReason: because no\n\n',
          1.0,
          'because no',
      ),
      (
          'should_not_be_confused_by_yes_in_reason',
          'no\nReason: because yes\n\n',
          0.0,
          'because yes',
      ),
  )
  def test_naive_evaluation_critic_produces_correct_output(
      self, critic_llm_reply, expected_value, expected_reason
  ):
    question = 'some question'
    llm.generate_text.configure(
        _fake_generate_text_returns_anything,
        return_value=critic_llm_reply,
    )
    res = executing.run(
        evaluation.naive_evaluation_critic(
            'answer 1',
            example={'question': question, 'golden_answer': 'bla'},
        )
    )
    with self.subTest('should_return_correct_numeric_metric_value'):
      self.assertEqual(expected_value, res[0])
    with self.subTest('extra_info_should_be_keyed_by_question'):
      self.assertIn(question, res[1])
    with self.subTest('should_return_reason_in_extra_info'):
      self.assertEqual(expected_reason, res[1][question]['reason'])


class CompareWithCriticTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

  def test_apply_critic_to_answers(self):

    fake_example = {'fake_example_key': 'fake_example_value'}

    @executing.make_executable
    async def some_executable(a: int, b: int):
      return f'{a}_{b}', fake_example

    @executing.make_executable
    async def some_comparison_critic(answers, example):
      del example
      return '+'.join(answers)

    @executing.make_executable
    async def some_evaluation_critic(answer, example):
      del example
      return answer

    def exec_seq_generator(num_examples: int = 10, num_execs=3):
      for example in range(num_examples):
        executables = []
        for executable in range(num_execs):
          executables.append(some_executable(example, executable))
        yield executables

    critic_values_generator = evaluation.apply_critic_to_answers(
        stream_of_exec_seq=exec_seq_generator(),
        critic=some_comparison_critic,
    )
    with self.subTest('comparison_produces_correct_output'):
      for el_id, el in enumerate(critic_values_generator):
        expected_value = (f'{el_id}_0+{el_id}_1+{el_id}_2', fake_example)
        self.assertEqual(executing.run(el), expected_value)

    critic_values_generator = evaluation.apply_critic_to_answers(
        stream_of_exec_seq=exec_seq_generator(num_execs=1),
        critic=some_evaluation_critic,
        critic_takes_single_answer=True,
    )
    with self.subTest('evaluation_produces_correct_output'):
      for el_id, el in enumerate(critic_values_generator):
        expected_value = (f'{el_id}_0', fake_example)
        self.assertEqual(executing.run(el), expected_value)

    critic_values_generator = evaluation.apply_critic_to_answers(
        stream_of_exec_seq=exec_seq_generator(num_execs=3),
        critic=some_evaluation_critic,
        critic_takes_single_answer=True,
    )
    with self.subTest('raises_error_when_evaluation_critic_gets_many_execs'):
      with self.assertRaisesRegex(
          ValueError,
          'Critic function is expected to take only one answer',
      ):
        for el in critic_values_generator:
          _ = executing.run(el)

  def test_compare_two_simple_strategies(self):
    """Verifies comparison of simple strategies produces expected results."""
    def _generate_text(prompt: str | content_lib.ChunkList) -> str:
      if (
          (isinstance(prompt, str) or isinstance(prompt, content_lib.ChunkList))
          and str(prompt).endswith('Answer')
      ):  # Prompt of naive_comparison_critic ends with 'Answer'.
        return '1'
      else:
        return 'generated'

    # This way default critic always chooses the first answer.
    llm.generate_text.configure(_generate_text)

    @executing.make_executable
    async def strategy_a(question, **unused_kwargs):
      del unused_kwargs
      return await llm.generate_text(prompt=f'{question}_a')

    @executing.make_executable
    async def strategy_b(question, **unused_kwargs):
      del unused_kwargs
      return await llm.generate_text(prompt=f'{question}_b')

    questions = ['question 1', 'question 2', 'question 3', 'question 4']
    reference_answers = ['answer 1', 'answer 2', 'answer 3', 'answer 4']

    _, total_votes, _ = evaluation.compare_with_critic(
        strategies=[strategy_a, strategy_b],
        examples=[{'question': q} for q in questions],
        critic=functools.partial(
            evaluation.naive_comparison_critic,
            shuffle_answers=False,
        ),
    )

    with self.subTest('produces_correct_total_votes'):
      self.assertEqual(total_votes, [4, 0])

    _, total_votes, _ = evaluation.compare_with_critic(
        strategies=[strategy_a, strategy_b],
        examples=[
            {'question': q, 'reference_answer': ra}
            for (q, ra) in zip(questions, reference_answers)
        ],
        critic=functools.partial(
            evaluation.naive_comparison_critic,
            use_reference_answer=True,
            shuffle_answers=False,
        ),
    )

    with self.subTest('produces_correct_total_votes_when_ran_with_reference'):
      self.assertEqual(total_votes, [4, 0])

  def test_naive_comparison_critic(self):
    with self.subTest('raises_error_if_no_question_key'):
      with self.assertRaisesRegex(
          ValueError,
          'Example does not contain the question key',
      ):
        _ = executing.run(
            evaluation.naive_comparison_critic(
                answers=['answer 1', 'answer 2'],
                example={'bla': 'bla'},
            )
        )

    with self.subTest('raises_error_if_less_than_two_answers'):
      with self.assertRaisesRegex(
          ValueError,
          'Comparison critic is meant to compare multiple',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            answers=[],
            example={'question': 'some question'},
        ))
      with self.assertRaisesRegex(
          ValueError,
          'Comparison critic is meant to compare multiple',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            answers=['answer 1'],
            example={'question': 'some question'},
        ))

    with self.subTest('raises_error_if_no_reference_answer_key'):
      with self.assertRaisesRegex(
          ValueError,
          'Example does not contain the reference answer key',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            'answer 1',
            'answer 2',
            example={'question': 'some question'},
            use_reference_answer=True,
        ))

    # naive_critic is based on llm.generate_text.
    llm.generate_text.configure(_fake_generate_text)
    with self.subTest('raises_error_if_critic_returns_not_int_str'):
      with self.assertRaisesRegex(
          ValueError,
          'Critic is expected to return a string that contains an int and that',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            answers=['answer 1', 'answer 2'],
            example={'question': 'some question'},
        ))

    with self.subTest('raises_error_if_critic_returns_unexpected_int'):
      llm.generate_text.configure(
          _fake_generate_text_returns_anything,
          return_value='3',
      )
      with self.assertRaisesRegex(
          ValueError,
          ' which is not a valid answer number.',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            answers=['answer 1', 'answer 2'],
            example={'question': 'some question'},
        ))
      llm.generate_text.configure(
          _fake_generate_text_returns_anything,
          return_value='0',
      )
      with self.assertRaisesRegex(
          ValueError,
          ' which is not a valid answer number.',
      ):
        _ = executing.run(evaluation.naive_comparison_critic(
            answers=['answer 1', 'answer 2'],
            example={'question': 'some question'},
        ))

    llm.generate_text.configure(
        _fake_generate_text_returns_anything,
        return_value='2',
    )
    with self.subTest('produces_correct_output'):
      res = executing.run(
          evaluation.naive_comparison_critic(
              answers=['answer 1', 'answer 2'],
              example={'question': 'bla'},
              shuffle_answers=False,
          )
      )
      # Stringify the prompt (ChunkList -> str).
      res[1]['bla']['critic_prompt'] = str(res[1]['bla']['critic_prompt'])
      self.assertEqual(res, (
          1,
          {
              'bla': {
                  'best_answer': 'answer 2',
                  'answers': ['answer 1', 'answer 2'],
                  'critic_prompt': textwrap.dedent("""\
                      Here are 2 different answers:
                      Answer 1. answer 1
                      Answer 2. answer 2

                      Here is a question (or task description):
                      bla

                      The best answer for the question above is Answer"""),
              }
          },
      ))

    with self.subTest('produces_correct_output_when_using_reference_answers'):
      res = executing.run(
          evaluation.naive_comparison_critic(
              answers=['answer 1', 'answer 2'],
              example={'question': 'bla', 'reference_answer': 'haha'},
              use_reference_answer=True,
              shuffle_answers=False,
          )
      )
      # Stringify the prompt (ChunkList -> str).
      res[1]['bla']['critic_prompt'] = str(res[1]['bla']['critic_prompt'])
      self.assertEqual(res, (
          1,
          {
              'bla': {
                  'best_answer': 'answer 2',
                  'answers': ['answer 1', 'answer 2'],
                  'critic_prompt': textwrap.dedent("""\
                      Here are 2 different answers:
                      Answer 1. answer 1
                      Answer 2. answer 2

                      Here is a question (or task description):
                      bla

                      A good answer could look something like this:
                      haha

                      The best answer for the question above is Answer"""),
              }
          },
      ))


if __name__ == '__main__':
  absltest.main()
