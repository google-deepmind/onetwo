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

"""Various evaluation routines.

All functions are meant as minimal templates that demonstrate how to run
custom prompt strategies on multiple examples in an efficient way and evaluate
the results.

We encourage our users to fork this file and modify it according to their needs.

We cover a few types of evaluation scenarios:
1. Compare strategy's answer with the ground truth answer in a programmatic way.
  E.g., `float(answer == example['golden_answer']` (see `evaluate`);
2. Compare strategy's answer with the ground truth answer using LLM critic.
  E.g., if we want both "2pi cm" and "6.28 centimeters" to count as correct
  answers (see `evaluate`).
3. Compare answers of multiple strategies using LLM critic
  (see `compare_with_critic`);
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import datetime
import pprint
import random
import textwrap
import time
from typing import cast, Any, Final, Protocol, TypeAlias, TypeVar

from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import updating
import tqdm


_QUESTION_KEY: Final[str] = 'question'
_REFERENCE_ANSWER_KEY: Final[str] = 'reference_answer'
_GOLDEN_ANSWER_KEY: Final[str] = 'golden_answer'

_Result = TypeVar('_Result')
_CriticResult = TypeVar('_CriticResult')
_ChunkList: TypeAlias = content_lib.ChunkList
Example: TypeAlias = Mapping[str, Any]
SingleMetricValue: TypeAlias = float
ExtraInfo: TypeAlias = Mapping[str, Any]
MetricResult: TypeAlias = tuple[SingleMetricValue, ExtraInfo]
# Tuple of [id_of_best_answer, extra_info], where id_of_best_answer is 0-based.
ComparisonResult: TypeAlias = tuple[int, ExtraInfo]


class _EvaluationCritic(Protocol[_Result]):
  def __call__(
      self,
      answer: _Result,
      example: Example,
      **kwargs: Any,
  ) -> executing.Executable[MetricResult] | MetricResult: ...


class _ComparisonCritic(Protocol[_Result]):
  def __call__(
      self,
      answers: Sequence[_Result],
      example: Example,
      **kwargs: Any,
  ) -> executing.Executable[ComparisonResult] | ComparisonResult: ...


def _compile_strategies(
    strategies: Sequence[Callable[..., executing.Executable[_Result]]],
    examples: Iterable[Example],
) -> Iterator[Sequence[executing.Executable[tuple[_Result, Example]]]]:
  """Compiles multiple strategies into executables on a sequence of examples.

  Args:
    strategies: Sequence of prompting strategies. We assume that examples
      contain all the arguments required for every strategy, i.e., that
      `strategy(**example)` returns a valid Executable for any example in
      examples and strategy in strategies.
    examples: Iterable of examples on which to evaluate the strategy.

  Yields:
    For each example in examples yields Sequence[Executable], where every
      elements correspond to different elements in strategies. Running each of
      these executables corresponds to applying the corresponding prompting
      strategy on the example.

  Raises:
    ValueError: If `strategy(**example)` raises a TypeError for any example in
      examples or strategy in strategies. This likely means that signature of
      `strategy` is not compatible with fields stored in the example, e.g., if
      example did not provide all the required arguments for `strategy`
      function.
  """

  @executing.make_executable
  async def _execute_and_add_example(
      executable: executing.Executable,
      example: Example,
  ):
    # Note: arg `executable` will be executed before entering the function.
    return executable, example

  for example in examples:
    executables = []
    for strategy in strategies:
      try:
        executable = strategy(**example)
      except TypeError as err:
        raise ValueError(
            'Error occurred when calling `strategy(**example)` for '
            f'example={pprint.pformat(example)}.'
        ) from err
      executables.append(executable)
    # We wrap the user defined executable to also return an example that was
    # used to define it. This is done to make sure that later we have an
    # access to that example in case we may need it.
    yield [_execute_and_add_example(exec, example) for exec in executables]


@executing.make_executable
async def naive_comparison_critic(
    answers: Sequence[str | content_lib.ChunkList],
    example: Example,
    question_key: str = _QUESTION_KEY,
    reference_answer_key: str = _REFERENCE_ANSWER_KEY,
    use_reference_answer: bool = False,
    shuffle_answers: bool = True,
) -> ComparisonResult:
  """Naive implementation of a comparison critic strategy.

  Args:
    answers: Sequence of different answers for the question contained in the
      example.
    example: Example that contains the question.
    question_key: Key of the element in the example dictionary that contains the
      question.
    reference_answer_key: Key of the element in the example dictionary that
      contains the reference answer (if present).
    use_reference_answer: Use the reference answer when comparing different
      answers.
    shuffle_answers: If True (default) answers will be shuffled before
      comparison is performed (to break any order bias).

  Returns:
    A tuple (best_answer_id, extra_info), where best_answer_id is 0-based.

  Raises:
    ValueError:
      If example does not contain the question key. Or use_reference_answer is
      True and example doed not contain the reference answer key. Or if critic
      returned a string that does not consist of a single integer, i.e., can not
      be used with int().
  """
  if question_key not in example:
    raise ValueError(
        f'Example does not contain the question key {question_key}:\n'
        f'{pprint.pformat(example)}'
    )

  if len(answers) < 2:
    raise ValueError(
        'Comparison critic is meant to compare multiple (more than 1) answers '
        f'but received only {len(answers)}:\n{answers}'
    )

  # Possibly shuffle the answers for comparison.
  possibly_shuffled_answers = list(answers)
  answer_ids = list(range(len(possibly_shuffled_answers)))
  if shuffle_answers:
    random.shuffle(answer_ids)  # Shuffle ids randomly.
    # Shuffle the answers in the same order.
    possibly_shuffled_answers = [
        possibly_shuffled_answers[el_id] for el_id in answer_ids
    ]

  # Create a prompt for comparison.
  num_answers = len(possibly_shuffled_answers)
  list_of_answers = content_lib.ChunkList()
  for answer_id, answer in enumerate(possibly_shuffled_answers):
    list_of_answers += f'Answer {answer_id + 1}. ' + answer.lstrip().rstrip()
    list_of_answers += '\n'
  critic_prompt = content_lib.ChunkList()
  critic_prompt += f'Here are {num_answers} different answers:\n'
  critic_prompt += list_of_answers + '\n'
  critic_prompt += 'Here is a question (or task description):\n'
  critic_prompt += example[question_key].lstrip().rstrip() + '\n\n'
  if use_reference_answer:
    if reference_answer_key not in example:
      raise ValueError(
          f'Example does not contain the reference answer key {question_key}:\n'
          f'{pprint.pformat(example)}'
      )
    critic_prompt += 'A good answer could look something like this:\n'
    critic_prompt += example[reference_answer_key].lstrip().rstrip() + '\n\n'
  critic_prompt += 'The best answer for the question above is Answer'
  res = await llm.generate_text(
      prompt=critic_prompt,
      max_tokens=3,
      stop=['.', '\n'],
  )
  try:
    res = int(res)
  except ValueError as err:
    raise ValueError(
        'Critic is expected to return a string that contains an int and that '
        f'can be used with int() function. Instead got:\n{res}'
    ) from err
  if res not in range(1, num_answers + 1):
    raise ValueError(
        f'Critic returned "{res}" which is not a valid answer number.'
    )
  best_answer_id = res - 1
  best_answer = possibly_shuffled_answers[best_answer_id]
  if shuffle_answers:
    # Map critic's choice back to the original ids.
    best_answer_id = answer_ids[best_answer_id]
  return best_answer_id, {
      example[question_key]: {
          'best_answer': best_answer,
          # Highlights that the order is irrelevant:
          'answers': list(answers),
          'critic_prompt': critic_prompt,
      }
  }


def _apply_critic_to_answers(
    stream_of_exec_seq: Iterator[
        Sequence[executing.Executable[tuple[_Result, Example]]]
    ],
    critic: _EvaluationCritic[_Result] | _ComparisonCritic[_Result],
    critic_takes_single_answer: bool = False,
) -> Iterator[executing.Executable[tuple[_CriticResult, Example]]]:
  """Create a stream of executables with critic answers.

  Converts stream of (non-executed) strategy tuples into stream of
  (non-executed) critic values.

  Args:
    stream_of_exec_seq: Iterator where i-th element is [s_1_i, ..., s_K_i].
      s_k_i is an answer of `k`-th strategy (out of K) on the i-th example. When
      executed, s_k_i returns a tuple (answer, example).
    critic: A function that follows either _EvaluationCritic or
      _ComparisonCritic protocol. Returns Executable[_CriticResult]
      (or _CriticResult).
    critic_takes_single_answer: Whether the critic function is expected to match
      _EvaluationCritic or _ComparisonCritic protocol. For the first case use
      True, for the latter use False (default).

  Yields:
    Stream of critic's values on the stream of answers.
  """
  @executing.make_executable
  async def _apply_critic(
      executables: Sequence[executing.Executable[tuple[_Result, Example]]],
  ) -> tuple[_CriticResult, Example]:
    """Call a critic on unpacked answers and examples."""
    # While `make_executable` decorator does execute all the Executable
    # arguments, in our case `executables` is a Sequence of Executables, so
    # `make_executable` won't do it for us. We need to execute them first.
    answers_and_examples = []
    for el in executables:
      res = await el
      answers_and_examples.append(res)
    # All examples should be the same, but we don't check it.
    example = answers_and_examples[0][1]
    answers_arg = [answer for (answer, _) in answers_and_examples]
    if critic_takes_single_answer:
      if len(answers_arg) > 1:
        raise ValueError(
            'Critic function is expected to take only one answer, because'
            '`critic_takes_single_answer` flag is set to True. However, critic '
            f'received {len(answers_arg)} answers to process:\n{answers_arg}'
        )
      critic_result = cast(_EvaluationCritic, critic)(
          answer=answers_arg[0],
          example=example,
      )
    else:
      critic_result = cast(_ComparisonCritic, critic)(
          answers=answers_arg,
          example=example,
      )
    if isinstance(critic_result, executing.Executable):
      # Critic function returned Executable[_CriticResult]. Need to await.
      critic_result = await critic_result
    return critic_result, example

  for exec_seq in stream_of_exec_seq:
    yield _apply_critic(executables=exec_seq)


def compare_with_critic(
    *,
    strategies: Sequence[Callable[..., executing.Executable[_Result]]],
    examples: Iterable[Example],
    critic: _ComparisonCritic[_Result] = naive_comparison_critic,
    examples_total_num: int | None = None,
    update_extra_info_fn: Callable[
        [dict[str, Any], ExtraInfo], None
    ] = lambda aggr_info, new_info: aggr_info.update(new_info),
    print_debug: bool = False,
) -> tuple[
    datetime.timedelta,
    Sequence[int],
    ExtraInfo,
]:
  """Compares multiple prompting strategies on a sequence of examples.

  Often it is difficult to evaluate model's (strategy's) answer on a single
  example, e.g., for open ended questions like "compose a short poem" or "write
  a funny joke about rabbits that rhymes". In the same time often it is much
  easier to compare answers of two (or more) different models (strategies) on
  the same example. Doing so may provide a good signal for improving the
  strategies and hill climbing.

  It may be a good idea to shuffle the answers before showing them to the
  comparison critic in order to avoid an order bias. Shuffling should be
  performed by the critic.

  In onetwo we often represent a prompting strategy as a function that takes in
  any arguments and returns an Executable. This Executable produces an answer
  when awaited or ran (with onetwo.run). Formally, a "strategy" is an object
  of type `Callable[..., executing.Executable[ReturnType]]`.

  Args:
    strategies: Sequence of K prompting strategies that we want to compare. We
      assume that examples contain all the arguments required for every strategy
      in strategies, i.e., that `strategy(**example)` returns a valid Executable
      for any example in examples and every strategy in strategies.
    examples: Iterable of examples on which to compare the K strategies.
    critic: A strategy that takes (i) answers produced by all the different
      strategies that we want to compare and (ii) an example on which these
      answers were produced and returns an Executable. Upon execution it needs
      to return a tuple (best_answer_id, extra_info), where best_answer_id is a
      0-based id of the best answer. Signature must follow _ComparisonCritic
      protocol.
    examples_total_num: Even if examples object has no implementation of __len__
      (examples could be a generator function with `yield`) user may still know
      (and provide) its exact length. This value is used only for logging the
      progress of evaluation.
    update_extra_info_fn: Function that aggregates additional metric
      information, returned by _ComparisonCritic. We aggregate additional
      information in-place with `update_extra_info_fn(aggr_info, new_info)`.
      By default we use dict's `update` method.
    print_debug: Print per example debug information.

  Returns:
    A tuple of comparison duration, total votes per strategy, and additional
    comparison information.

  Raises:
    RuntimeError: If `stream_updates` returns unexpected results.
  """

  num_strategies = len(strategies)

  if hasattr(examples, '__len__'):
    examples_len = len(examples)
  elif examples_total_num is not None:
    examples_len = examples_total_num
  else:
    # In this case tqdm will only print progress without progressbar and ETA.
    examples_len = None

  start_time = time.monotonic()

  # -> Iterable[Sequence[Executable[tuple[_Result, Example]]]].
  strategies_exec_per_example = _compile_strategies(strategies, examples)
  # -> Iterable[Executable[tuple[ComparisonResult, Example]]].
  critics_exec_per_example = _apply_critic_to_answers(
      stream_of_exec_seq=strategies_exec_per_example,
      critic=critic,
  )
  eval_executable = executing.par_iter(critics_exec_per_example)
  votes_by_strategy = [0 for _ in range(num_strategies)]
  aggr_comparison_info = {}
  num_examples = 0
  with executing.stream_updates(eval_executable, iteration_depth=1) as iterator:
    pbar = tqdm.tqdm(iterator, total=examples_len)
    update: updating.ListUpdate  # ListUpdate because of par_iter.
    for update in pbar:
      if len(update.payload) != 1:
        raise RuntimeError(
            'ListUpdate returned by stream_updates is expected to have '
            f'exactly one element in its payload. Got {pprint.pformat(update)}'
        )
      # Payload contains a single element of the form
      # (critic_result, example_id).
      critic_result, _ = update.payload[0]
      (best_strategy_id, extra_comparison_info), example = critic_result
      votes_by_strategy[best_strategy_id] += 1
      num_examples += 1
      if print_debug:
        msg = (
            f"example='{pprint.pformat(example)}', "
            f'critic_choice={best_strategy_id + 1}/{num_strategies}, '
            f'votes={votes_by_strategy}'
        )
        tqdm.tqdm.write(msg)
      update_extra_info_fn(aggr_comparison_info, dict(extra_comparison_info))
      msg = f'votes={votes_by_strategy}'
      pbar.set_description(msg)

  duration_secs = time.monotonic() - start_time
  normalized_votes = [
      round(votes / num_examples, 3) for votes in votes_by_strategy
  ]
  print(
      '\nCompare result: votes: %s, processed examples: %d, duration=%.2f secs.'
      % (str(normalized_votes), num_examples, duration_secs)
  )
  return (
      datetime.timedelta(seconds=duration_secs),
      votes_by_strategy,
      aggr_comparison_info,
  )


@executing.make_executable
async def naive_evaluation_critic(
    answer: str | _ChunkList,
    example: Example,
    question_key: str = _QUESTION_KEY,
    golden_answer_key: str = _GOLDEN_ANSWER_KEY,
) -> MetricResult:
  """Naive implementation of an evaluation critic strategy.

  Given a question, a golden answer, and a candidate answer, this critic should
  decide whether the answers are equivalent. E.g., "2pi cm" and "6.28
  centimeters" could be both considered correct for the question "What is
  circumference of a circle with radius 1cm?".

  Args:
    answer: Candidate answer for the question (contained in the example) that
      a critic needs to evaluate.
    example: Example that contains the question and the golden answer.
    question_key: Key of the element in the example dictionary that contains the
      question.
    golden_answer_key: Key of the element in the example dictionary that
      contains the golden answer.

  Returns:
    A tuple (SingleMetricValue, ExtraInfo).

  Raises:
    ValueError:
      If example does not contain the question key. Or use_reference_answer is
      True and example doed not contain the reference answer key. Or if critic
      returned a string that does not consist of a single integer, i.e., can not
      be used with int().
  """
  if question_key not in example:
    raise ValueError(
        f'Example does not contain the question key {question_key}:\n'
        f'{pprint.pformat(example)}'
    )
  if golden_answer_key not in example:
    raise ValueError(
        f'Example does not contain the golden answer key {golden_answer_key}:\n'
        f'{pprint.pformat(example)}'
    )
  critic_prompt = _ChunkList()
  critic_prompt += textwrap.dedent("""\
      Question: Circumference of a circle with radius 1cm?
      Answer 1 (correct): 2pi cm
      Answer 2: 6.28 centimenter
      Is Answer 2 also correct? (yes/no): yes
      Reason: pi is ~3.141 and 2pi is ~6.282. 6.28 is an accurate enough answer.

      Question: Spell first 5 digits of pi.
      Answer 1 (correct): 3.1415
      Answer 2: 3.14
      Is Answer 2 also correct? (yes/no): no
      Reason: Answer 2 provides only 3 digits, while 5 were required.

      Question: """)
  critic_prompt += example[question_key] + '\n'
  critic_prompt += 'Answer 1 (correct): ' + example[golden_answer_key] + '\n'
  critic_prompt += 'Answer 2: ' + answer + '\n'
  critic_prompt += 'Is Answer 2 also correct? (yes/no):'
  res = await llm.generate_text(
      prompt=critic_prompt,
      stop=['Question:'],
  )
  # TODO: Implement the following as a builtin. It's fairly common.
  if res.startswith(' yes'):
    is_correct = 1.
  elif res.startswith(' no'):
    is_correct = 0.
  else:
    raise ValueError(
        'Critic is expected to start its answer from " yes" or " no" . Instead '
        'it generated an unexpected result:\n{res}'
    )
  reason = ''
  if '\nReason: ' in res:
    reason = res.split('\nReason: ', 1)[1].split('\n\n')[0].strip()
  extra_evaluation_info = {
      example[question_key]: {
          'golden_answer': example[golden_answer_key],
          'candidate_answer': answer,
          'answer_is_correct': res.startswith(' yes'),
          'reason': reason,
          'critic_prompt': critic_prompt,
      }
  }
  return (is_correct, extra_evaluation_info)


def evaluate(
    *,
    strategy: Callable[..., executing.Executable[_Result]],
    examples: Iterable[Example],
    critic: _EvaluationCritic[_Result] = naive_evaluation_critic,
    examples_total_num: int | None = None,
    update_extra_info_fn: Callable[
        [dict[str, Any], ExtraInfo], None
    ] = lambda aggr_info, new_info: aggr_info.update(new_info),
    print_debug: bool = False,
) -> tuple[
    datetime.timedelta,
    SingleMetricValue,
    ExtraInfo,
]:
  """Evaluates a prompting strategy on a sequence of examples possibly with LLM.

  When programmatically comparing strategy's answer with the ground truth answer
  is hard one may use LLM "critic" to evaluate the answer. For example, we may
  want to count both "2pi cm" and "6.28 centimeter" as correct answers. In this
  case `evaluate` can be used with LLM based critic. Otherwise, if
  comparison can be performed programmatically, this function can be also used
  with `critic` that is a normal function (i.e., function that returns
  `MetricResult` instead of `Executable[MetricResult]`).

  In onetwo we often represent a prompting strategy as a function that takes in
  any arguments and returns an Executable. Formally, a strategy is an object of
  type `Callable[..., executing.Executable[ReturnType]]`.

  Args:
    strategy: Prompting strategy to evaluate. We assume that examples contain
      all the arguments required for the strategy, i.e., that
      `strategy(**example)` returns a valid Executable for any example in
      examples.
    examples: Iterable of examples on which to evaluate the strategy.
    critic: A strategy (or a normal function) that takes (i) an answer
      produced by the strategy that we want to evaluate and (ii) an example on
      which the answer was produced (that contains the question and the golden
      answer), and returns an Executable[MetricResult] (or simply MetricResult).
      Must follow the _EvaluationCritic protocol.
    examples_total_num: Even if examples object has no implementation of __len__
      (examples could be a generator function with `yield`) user may still know
      (and provide) its exact length. This value is used only for logging the
      progress of evaluation.
    update_extra_info_fn: Function that aggregates additional evaluation
      information, returned by `metric_fn`. We aggregate additional information
      in-place with `update_extra_info_fn(aggr_info, new_info)`. By
      default we use dict's `update` method.
    print_debug: Print per example debug information.

  Returns:
    A tuple of evaluation duration, average metric value, and additional metric
      information.

  Raises:
    RuntimeError: If `stream_updates` returns unexpected results.
  """

  if hasattr(examples, '__len__'):
    examples_len = len(examples)
  elif examples_total_num is not None:
    examples_len = examples_total_num
  else:
    # In this case tqdm will only print progress without progressbar and ETA.
    examples_len = None

  start_time = time.monotonic()

  # -> Iterable[Sequence[Executable[tuple[_Result, Example]]]].
  strategy_exec_per_example = _compile_strategies([strategy], examples)
  # -> Iterable[Executable[tuple[MetricResult, Example]]].
  critics_exec_per_example = _apply_critic_to_answers(
      stream_of_exec_seq=strategy_exec_per_example,
      critic=critic,
      critic_takes_single_answer=True,  # Because we use _EvaluationCritic.
  )
  eval_executable = executing.par_iter(critics_exec_per_example)
  aggr_evaluation_info = {}
  metric_values, avg_metric = [], 0.0
  with executing.stream_updates(eval_executable, iteration_depth=1) as iterator:
    pbar = tqdm.tqdm(iterator, total=examples_len)
    update: updating.ListUpdate  # ListUpdate because of par_iter.
    for update in pbar:
      if len(update.payload) != 1:
        raise RuntimeError(
            'ListUpdate returned by stream_updates is expected to have '
            f'exactly one element in its payload. Got {pprint.pformat(update)}'
        )
      # Payload contains a single element of the form
      # (critic_result, example_id).
      critic_result, _ = update.payload[0]
      (metric, extra_evaluation_info), example = critic_result
      if print_debug:
        msg = (
            f"example='{pprint.pformat(example)}', "
            f'metric={metric} '
            f'extra_info={extra_evaluation_info}'
        )
        tqdm.tqdm.write(msg)
      update_extra_info_fn(
          aggr_evaluation_info,
          dict(extra_evaluation_info),
      )
      metric_values.append(metric)
      avg_metric = sum(metric_values) / len(metric_values)
      msg = f'Avg metric={avg_metric:.4f}'
      pbar.set_description(msg)

  duration_secs = time.monotonic() - start_time
  print(
      '\nEval result: Avg metric=%.4f, processed examples=%d, duration=%.2f'
      ' secs.' % (avg_metric, len(metric_values), duration_secs)
  )
  return (
      datetime.timedelta(seconds=duration_secs),
      avg_metric,
      aggr_evaluation_info,
  )
