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

"""Implementation of variations of the self-consistency strategy.

Generalized from the ideas of the self-consistency paper:
https://arxiv.org/pdf/2203.11171

The basic idea of self-consistency as presented in the original paper is to
generate multiple samples using chain-of-thought or some other strategy
involving diverse reasoning paths; extract a normalized answer from each sample;
and then do majority voting over the answers.

Here we reformulate self-consistency as a meta-strategy that wraps some
underlying strategy that outputs a single answer (typically via some kind of
reasoning path or other intermediate steps) and converts it into a strategy that
outputs a marginal distribution over possible answers (marginalizing over the
intermediate steps). The marginal distribution is estimated via repeated
sampling from the underlying strategy.

Supported variations include:
* Self-consistency over chain-of-thought (like in the original paper).
* Self-consistency over a multi-step prompting strategy (e.g., ReAct).
* Self-consistency over a multi-arg strategy (e.g., Retrieval QA).
* Self-consistency over diverse parameterizations of the underlying strategy
  (e.g., with samples taken using different choices of few-shot exemplars,
   as in https://arxiv.org/pdf/2206.02336).
* Self-consistency over diverse underlying strategies.
* Self-consistency with answer normalization applied during bucketization.
* Self-consistency with weighted voting
  (as in https://arxiv.org/pdf/2206.02336).
* Evaluation based on the consensus answer alone.
* Evaluation based on the full answer distribution (e.g., accuracy@k).
* Evaluation taking into account a representative reasoning path.
"""

import collections
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
from typing import Any, Generic, ParamSpec, Protocol, TypeAlias, TypeVar

from onetwo.core import executing
from onetwo.core import sampling
from onetwo.core import tracing
from onetwo.core import utils

# Type representing a strategy's arguments.
_Args = ParamSpec('_Args')

# Type representing a strategy's output.
_O = TypeVar('_O')

# Type representing the buckets used in a self-consistency strategy.
_Bucket = TypeVar('_Bucket')

# Type aliases.
_PredictionsWithScores: TypeAlias = Sequence[tuple[_O, float]]


@tracing.trace  # pytype: disable=wrong-arg-types
def select_first_highest_scoring_output(
    bucket: Any, predictions_with_scores: _PredictionsWithScores
) -> _O | None:
  """Returns the output with highest score, or as tie-breaker, the first.

  Intended for use as a `representative_output_selector`.

  Args:
    bucket: The bucket for which we are selecting a representative output.
    predictions_with_scores: The outputs and their scores to choose from.
  """
  del bucket
  if not predictions_with_scores:
    return None
  return sorted(predictions_with_scores, key=lambda x: x[1], reverse=True)[0][0]


class Scorer(Protocol[_O]):
  """Function to score a pair of inputs + candidate output.

  Suitable for use as the `scorer` in a self-consistency strategy, or for use
  in filtering of candidate outputs (when combined with a score threshold).

  The callable can be an ordinary function or can be async or decorated with
  `@executing.make_executable` (e.g., for a scoring function that is backed by
  an AI rater).
  """

  def __call__(
      self,
      args: _Args.args,
      kwargs: _Args.kwargs,
      output: _O,
  ) -> float | Awaitable[float] | executing.Executable[float]:
    """Returns the score for the given inputs + candidate output.

    Bigger score is assumed to be better.

    Args:
      args: Positional arguments portion of the inputs.
      kwargs: Keyword arguments portion of the inputs.
      output: The candidate output to score.
    """
    ...


@dataclasses.dataclass
class SelfConsistency(Generic[_O, _Bucket]):
  """Self-consistency strategy, which can wrap an arbitrary inner strategy.

  If the inner strategy maps inputs (_Args) to output (_O), the self-consistency
  variant of it will take inputs (_Args) and return a distribution over possible
  outputs (i.e., `list[tuple[_O, float]]`). The caller can then either use the
  full distribution directly (e.g., for calculating a metric like accuracy@k),
  or take the top-weighted candidate output as the single "consensus answer".

  The float values in the output represent arbitrary scores (not necessarily
  probabilities), but can be converted into probabilities by dividing by the
  sum of the scores.

  The implementation is parameterized by several components: `sampler`,
  `output_filter`, `scorer`, `bucketizer`, etc. The only required one is
  `sampler`, as the rest all have default implementations, which are sufficient
  for performing the basic self-consistency strategy, as described in the
  original paper. Each of the components can be provided in the form of either
  an ordinary function or an async function.

  Attributes:
    sampler: The inner strategy that is used for generating each prediction.
    num_samples: Number of candidate predictions to sample before voting.
    output_filter: A function that takes a candidate output and returns True if
      it should be included in the distribution. Could be used, for example, to
      filter out `None` values. By default, no filtering is applied.
    scorer: A function that takes a pair of inputs + candidate output and
      returns a score. By default, it just gives a vote of 1.0 for each
      candidate.
    bucketizer: A function that takes a candidate output and returns a bucket.
      Candidates placed in the same bucket are considered to be "equivalent" for
      purposes of voting and of determining the distribution of outputs.
    representative_output_selector: A function that selects a single
      representative out of all of the candidates within a given bucket. The sum
      of the scores of all the candidates in that bucket will be considered the
      total score for that single representative output.
  """

  sampler: sampling.Sampler[_O]
  num_samples: int = 1
  output_filter: Callable[[_O], bool] | Callable[[_O], Awaitable[bool]] = (
      lambda x: True
  )
  scorer: Scorer[_O] = lambda args, kwargs, output: 1.0
  bucketizer: Callable[[_O], _Bucket] | Callable[[_O], Awaitable[_Bucket]] = (
      lambda x: x
  )
  representative_output_selector: (
      Callable[[_Bucket, _PredictionsWithScores], _O | None]
      | Callable[[_Bucket, _PredictionsWithScores], Awaitable[_O | None]]
  ) = select_first_highest_scoring_output

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self, *args: _Args.args, **kwargs: _Args.kwargs
  ) -> list[tuple[_O, float]]:
    # Generate samples.
    result_objects = await self.sampler(
        *args, num_samples=self.num_samples, **kwargs
    )

    # Create mapping of bucket to list of (output, score) tuples.
    scored_predictions_by_bucket = collections.defaultdict(list)
    for result in result_objects:
      should_include = await utils.call_and_maybe_await(
          self.output_filter, result
      )
      if should_include:
        score = await utils.call_and_maybe_await(
            self.scorer, args=args, kwargs=kwargs, output=result
        )
        bucket = await utils.call_and_maybe_await(self.bucketizer, result)
        if score is not None:
          scored_predictions_by_bucket[bucket].append((result, score))

    # Calculate score and representative output for each bucket.
    output_distribution = []
    for bucket, predictions_with_scores in scored_predictions_by_bucket.items():
      representative_output = await utils.call_and_maybe_await(
          self.representative_output_selector, bucket, predictions_with_scores
      )
      bucket_score = sum(score for _, score in predictions_with_scores)
      output_distribution.append((representative_output, bucket_score))
    output_distribution.sort(key=lambda x: x[1], reverse=True)

    # Normalize the scores to form a probability distribution.
    sum_of_scores = sum(score for _, score in output_distribution)
    if sum_of_scores > 0:
      output_distribution = [
          (output, score / sum_of_scores)
          for output, score in output_distribution
      ]

    return output_distribution


@dataclasses.dataclass
class ExtractConsensus(Generic[_O]):
  """Wraps a SelfConsistency strategy to return just the consensus output.

  Attributes:
    inner: The SelfConsistency strategy to wrap.
  """

  inner: Callable[_Args, Awaitable[list[tuple[_O, float]]]]

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self, *args: _Args.args, **kwargs: _Args.kwargs
  ) -> _O | None:
    """Returns the consensus output, or None if the distribution is empty."""
    predicted_distribution = await self.inner(*args, **kwargs)
    if predicted_distribution:
      return predicted_distribution[0][0]
    else:
      return None
