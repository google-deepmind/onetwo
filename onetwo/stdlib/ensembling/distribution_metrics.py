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

"""Metric functions taking a distribution of model predictions as input.

These can be used, for example, for evaluating the outputs of a SelfConsistency
strategy.
"""

from collections.abc import Sequence
import dataclasses
from typing import Generic, TypeVar
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils
from onetwo.evaluation import agent_evaluation

# Type representing a strategy's output.
_O = TypeVar('_O')


@dataclasses.dataclass
class AccuracyAtK(Generic[_O]):
  """Returns the maximum accuracy among the first k predictions.

  Attributes:
    k: Number of candidate answers to consider for accuracy. If None, then will
      consider all predictions (even those with probability 0). If non-None,
      then will only consider the first k.
    base_metric: Metric to use for calculating the accuracy of an individual
      prediction vs. the target. Defaults to exact-match accuracy. Bigger is
      assumed to mean better.
  """

  k: int | None = 1
  base_metric: agent_evaluation.MetricFunction = lambda t, p: 1.0 * (t == p)

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self, target: _O, prediction: Sequence[tuple[_O, float]]
  ) -> float:
    """Returns accuracy at k (value from 0.0 to 1.0).

    Args:
      target: The target (single answer).
      prediction: The predicted distribution over possible answers.
    """
    k = self.k if self.k is not None else len(prediction)
    if k < 0:
      raise ValueError(f'k must be non-negative, got {k}')

    # For simplicity, we are currently calling `self.base_metric` sequentially
    # for each of the predictions. In the case where `self.base_metric` is an
    # `async` function, though, this could potentially be optimized in the
    # future by wrapping the calls to the base metric with `executing.parallel`.
    base_metric_values = [
        await utils.call_and_maybe_await(self.base_metric, target, x)
        for x, _ in prediction[:k]
    ]
    return max(base_metric_values, default=0.0)
