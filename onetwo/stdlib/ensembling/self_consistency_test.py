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

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import caching
from onetwo.core import executing
from onetwo.core import sampling
from onetwo.stdlib.ensembling import self_consistency
from onetwo.stdlib.reasoning import chain_of_thought


# "Chain-of-thought" strategy that simply returns its two arguments as the
# reasoning and answer, with the sampling key appended to the answer to show
# how many times it was called. Here we show that SelfConsistency can be
# used to wrap strategies with arbitrary numbers of arguments.
@executing.make_executable  # pytype: disable=wrong-arg-types
async def two_arg_cot(arg1: str, arg2: str):
  key = caching.context_sampling_key.get()
  return chain_of_thought.CoTReply(reasoning=arg1, answer=f'{arg2}-{key}')


# Variant of `two_arg_cot` that returns None for the first sample.
@executing.make_executable  # pytype: disable=wrong-arg-types
async def two_arg_cot_first_sample_none(arg1: str, arg2: str):
  key = caching.context_sampling_key.get()
  answer = f'{arg2}-{key}' if key else None
  return chain_of_thought.CoTReply(reasoning=arg1, answer=answer)


# Simple filter function to filter out None values.
def filter_none_sync(output: chain_of_thought.CoTReply):
  return output.answer is not None


async def filter_none_async(output: chain_of_thought.CoTReply):
  return filter_none_sync(output)


@executing.make_executable  # pytype: disable=wrong-arg-types
def filter_none_executable(output: chain_of_thought.CoTReply):
  return filter_none_sync(output)


# Simple custom scorer that uses the answer's numeric suffix as the score
# (e.g., if the answer is 'b-3', the score is 3).
def score_by_trailing_int_sync(args, kwargs, output):
  del args, kwargs
  return int(output.answer.split('-')[-1] or 0)


async def score_by_trailing_int_async(args, kwargs, output):
  return score_by_trailing_int_sync(args, kwargs, output)


@executing.make_executable  # pytype: disable=wrong-arg-types
def score_by_trailing_int_executable(args, kwargs, output):
  return score_by_trailing_int_sync(args, kwargs, output)


# Simple custom bucketizer that uses the answer's numeric suffix mod 2
# (e.g., if the answer is 'b-3', the bucket is 1).
def bucketize_by_trailing_int_mod_2_sync(output):
  return int(output.answer.split('-')[-1] or 0) % 2


async def bucketize_by_trailing_int_mod_2_async(output):
  return bucketize_by_trailing_int_mod_2_sync(output)


@executing.make_executable  # pytype: disable=wrong-arg-types
def bucketize_by_trailing_int_mod_2_executable(output):
  return bucketize_by_trailing_int_mod_2_sync(output)


# Simple custom representative output selector that selects the answer with the
# largest numeric suffix (e.g., out of 'b-1' and 'b-3', it selects 'b-3').
def select_highest_trailing_int_sync(
    bucket: Any,
    predictions_with_scores: Sequence[tuple[chain_of_thought.CoTReply, float]],
) -> chain_of_thought.CoTReply | None:
  del bucket
  if not predictions_with_scores:
    return None
  sorted_predictions_with_scores = sorted(
      predictions_with_scores,
      key=lambda x: int(x[0].answer.split('-')[-1] or 0),
      reverse=True,
  )
  return sorted_predictions_with_scores[0][0]


async def select_highest_trailing_int_async(
    bucket: Any,
    predictions_with_scores: Sequence[tuple[chain_of_thought.CoTReply, float]],
) -> chain_of_thought.CoTReply | None:
  return select_highest_trailing_int_sync(bucket, predictions_with_scores)


@executing.make_executable  # pytype: disable=wrong-arg-types
def select_highest_trailing_int_executable(
    bucket: Any,
    predictions_with_scores: Sequence[tuple[chain_of_thought.CoTReply, float]],
) -> chain_of_thought.CoTReply | None:
  return select_highest_trailing_int_sync(bucket, predictions_with_scores)


class SelfConsistencyTest(parameterized.TestCase):

  def test_self_consistency_basics(self):
    # Underlying strategy that returns its two args as the reasoning and answer.
    cot = two_arg_cot

    # Self-consistency variant of the above strategy.
    cot_sc = self_consistency.SelfConsistency(
        sampler=sampling.Repeated(cot),
        num_samples=4,
        bucketizer=lambda x: x.answer,
    )

    # We first call the underlying strategy directly for comparison.
    res_cot = executing.run(cot('a', 'b'))
    expected_res_cot = chain_of_thought.CoTReply(reasoning='a', answer='b-')
    with self.subTest('underlying_strategy_returns_single_answer'):
      self.assertEqual(expected_res_cot, res_cot, res_cot)

    # Notice that the self-consistency variant takes exactly the same arguments
    # as the underlying strategy, but returns a distribution over answers.
    res_cot_sc = executing.run(cot_sc('a', 'b'))  # pytype: disable=wrong-arg-count
    expected_res_cot_sc = [
        (chain_of_thought.CoTReply(reasoning='a', answer='b-'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-1'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-2'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-3'), 0.25),
    ]
    with self.subTest('self_consistency_returns_distribution_over_answers'):
      self.assertEqual(expected_res_cot_sc, res_cot_sc, res_cot_sc)

    # If we want to return just the consensus answer, we can wrap the
    # SelfConsistency strategy with ExtractConsensus. This makes the interface
    # the same as the original strategy, so it can be used interchangeably.
    cot_sc_consensus = self_consistency.ExtractConsensus(inner=cot_sc)
    res_cot_sc_consensus = executing.run(cot_sc_consensus('a', 'b'))  # pytype: disable=wrong-arg-count
    # The consensus answer is the answer with the highest score. By default,
    # in case of a tie (like in this case), the first answer is selected.
    expected_res_cot_sc_consensus = chain_of_thought.CoTReply(
        reasoning='a', answer='b-'
    )
    with self.subTest('extract_consensus_returns_consensus_answer'):
      self.assertEqual(
          expected_res_cot_sc_consensus,
          res_cot_sc_consensus,
          res_cot_sc_consensus,
      )

  @parameterized.named_parameters(
      ('sync', filter_none_sync),
      ('async', filter_none_async),
      ('executable', filter_none_executable),
  )
  def test_self_consistency_output_filter(self, output_filter):
    # In each case, we use a custom filter that filters out None values.
    # (Specifically, we set it up so that sample key 0 returns answer None.)
    # Note that the scores of the filtered-out answers are omitted from the
    # denominator when normalizing the scores to calculate the probabilities.
    sc_strategy = self_consistency.SelfConsistency(
        sampler=sampling.Repeated(two_arg_cot_first_sample_none),
        num_samples=5,
        output_filter=output_filter,
        bucketizer=lambda x: x.answer,
    )
    expected_sc_result = [
        (chain_of_thought.CoTReply(reasoning='a', answer='b-1'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-2'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-3'), 0.25),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-4'), 0.25),
    ]

    sc_result = executing.run(sc_strategy('a', 'b'))  # pytype: disable=wrong-arg-count
    self.assertEqual(expected_sc_result, sc_result, sc_result)

  @parameterized.named_parameters(
      ('sync', score_by_trailing_int_sync),
      ('async', score_by_trailing_int_async),
      ('executable', score_by_trailing_int_executable),
  )
  def test_self_consistency_scorer(self, scorer):
    # In each case, we use a custom scorer that looks at the answer's numeric
    # suffix (e.g., if the answer is 'b-3', the score is 3).
    sc_strategy = self_consistency.SelfConsistency(
        sampler=sampling.Repeated(two_arg_cot),
        num_samples=5,
        bucketizer=lambda x: x.answer,
        scorer=scorer,
    )
    expected_sc_result = [
        (chain_of_thought.CoTReply(reasoning='a', answer='b-4'), 0.4),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-3'), 0.3),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-2'), 0.2),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-1'), 0.1),
        (chain_of_thought.CoTReply(reasoning='a', answer='b-'), 0.0),
    ]

    sc_result = executing.run(sc_strategy('a', 'b'))  # pytype: disable=wrong-arg-count
    with self.subTest('self_consistency_distribution_based_on_custom_scorer'):
      self.assertEqual(expected_sc_result, sc_result, sc_result)

    consensus_strategy = self_consistency.ExtractConsensus(inner=sc_strategy)
    expected_consensus_result = chain_of_thought.CoTReply(
        reasoning='a', answer='b-4'
    )
    consensus_result = executing.run(consensus_strategy('a', 'b'))  # pytype: disable=wrong-arg-count
    with self.subTest('consensus_answer_is_the_one_with_highest_score'):
      self.assertEqual(
          expected_consensus_result,
          consensus_result,
          consensus_result,
      )

  @parameterized.named_parameters(
      ('sync', bucketize_by_trailing_int_mod_2_sync),
      ('async', bucketize_by_trailing_int_mod_2_async),
      ('executable', bucketize_by_trailing_int_mod_2_executable),
  )
  def test_self_consistency_bucketizer(self, bucketizer):
    # In each case, we use a custom scorer that looks at the answer's numeric
    # suffix (e.g., if the answer is 'b-3', the score is 3).
    sc_strategy = self_consistency.SelfConsistency(
        sampler=sampling.Repeated(two_arg_cot),
        num_samples=5,
        bucketizer=bucketizer,
        scorer=score_by_trailing_int_sync,
    )
    expected_sc_result = [
        # Answer with sample key 0, 2, and 4 are in bucket 0.
        # The sum of their scores is 6 (normalized to probability 0.6).
        # Within the bucket, the output with the highest score is chosen by
        # default as the representative output.
        (chain_of_thought.CoTReply(reasoning='a', answer='b-4'), 0.6),
        # Answer with sample key 1 and 3 are in bucket 1.
        # The sum of their scores is 4 (normalized to probability 0.4).
        (chain_of_thought.CoTReply(reasoning='a', answer='b-3'), 0.4),
    ]

    sc_result = executing.run(sc_strategy('a', 'b'))  # pytype: disable=wrong-arg-count
    self.assertEqual(expected_sc_result, sc_result, sc_result)

  @parameterized.named_parameters(
      ('sync', select_highest_trailing_int_sync),
      ('async', select_highest_trailing_int_async),
      ('executable', select_highest_trailing_int_executable),
  )
  def test_self_consistency_representative_output_selector(self, selector):
    # In each case, we use a custom scorer that selects the output with the
    # highest trailing integer in its answer (i.e., with the largest sample key)
    # as the representative for each bucket.
    sc_strategy = self_consistency.SelfConsistency(
        sampler=sampling.Repeated(two_arg_cot),
        num_samples=5,
        bucketizer=bucketize_by_trailing_int_mod_2_sync,
        representative_output_selector=selector,
    )
    expected_sc_result = [
        # Answer with sample key 0, 2, and 4 are in bucket 0.
        # The sum of their scores is 3 (normalized to probability 0.6).
        # Within the bucket, the output with the highest score is chosen by
        # default as the representative output.
        (chain_of_thought.CoTReply(reasoning='a', answer='b-4'), 0.6),
        # Answer with sample key 1 and 3 are in bucket 1.
        # The sum of their scores is 2 (normalized to probability 0.4).
        (chain_of_thought.CoTReply(reasoning='a', answer='b-3'), 0.4),
    ]

    sc_result = executing.run(sc_strategy('a', 'b'))  # pytype: disable=wrong-arg-count
    self.assertEqual(expected_sc_result, sc_result, sc_result)


if __name__ == '__main__':
  absltest.main()
