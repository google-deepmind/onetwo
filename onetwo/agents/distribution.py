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

"""Agent that can return a distribution over the next state updates."""

import abc
import dataclasses
import random
from typing import Generic, TypeVar, final

import numpy as np
from onetwo.agents import agents_base
from onetwo.core import executing
from onetwo.core import tracing


def logsumexp(vector: np.ndarray) -> float:
  max_value = np.max(vector)
  return max_value + np.log(np.sum(np.exp(vector - max_value)))


def lognormalize(vector: np.ndarray) -> np.ndarray:
  return vector - logsumexp(vector)


def softmax(vector: np.ndarray) -> np.ndarray:
  return np.exp(lognormalize(vector))


# Type that `Agent.run` takes as input.
_I = TypeVar('_I')

# Type that `Agent.run` returns as output.
_O = TypeVar('_O')

# Type used to represent the state of the agent.
_S = TypeVar('_S')

# Type used to represent a scored incremental update of the agent state.
_SU = TypeVar('_SU', bound=agents_base.ScoredUpdate)

# Type used to represent the environment of the agent.
_E = TypeVar('_E')


@dataclasses.dataclass
class DistributionAgent(
    Generic[_I, _O, _S, _SU, _E],
    agents_base.Agent[_I, _O, _S, _SU, _E],
):
  """Agent that can return a distribution over the next state updates.

  Its updates have a score attached to them, which can be used to represent
  a probability. So they are of type agents_base.ScoredUpdate.
  The sample_next_step method is implemented by actually sampling from the
  distribution which is defined by the get_next_step_distribution method.
  """

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(
      'DistributionAgent.get_next_step_distribution', skip=['environment']
  )
  @abc.abstractmethod
  async def get_next_step_distribution(
      self, state: _S, environment: _E | None = None
  ) -> list[_SU]:
    """Returns a distribution over possible updates from this state.

    Instead of sampling candidates, this returns a full distribution over
    the possible updates with scores that represent probabilities.

    Args:
      state: Current state of the agent.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).

    Returns:
      A list of tuples (update, score) where the scores represent probabilities.
    """

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(
      'DistributionAgent.sample_next_step', skip=['environment']
  )
  @final
  async def sample_next_step(
      self, *, state: _S, num_candidates: int, environment: _E | None = None
  ) -> list[_SU]:
    """Returns next step samples with their probabilities.

    Overridden from base class (Agent).

    The candidates are sampled with replacement, which means that the same
    candidate next step may appear multiple times. The sampling is done using
    the distribution returned by `get_next_step_distribution`.

    Args:
      state: Current state of the agent.
      num_candidates: Number of possible next steps to generate.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).

    Returns:
      An incremental update to the agent state that would occur as a result of
      performing each of the possible next steps, along with a score for that
      next step. Note that if a given update is sampled twice it will have the
      same score attached to it, so even though the scores represent
      probabilities of the corresponding update, they cannot be summed over the
      sampled candidates.
    """
    distribution = await self.get_next_step_distribution(state, environment)
    values = [d.update for d in distribution]
    probabilities = [d.score for d in distribution]
    # This requires the updates to be hashable.
    dist_dict = {d.update: d.score for d in distribution}
    samples = random.choices(values, weights=probabilities, k=num_candidates)
    return [
        agents_base.ScoredUpdate(update=s, score=dist_dict[s])
        for s in samples
    ]


@dataclasses.dataclass
class ReweightedDistributionAgent(
    Generic[_I, _O, _S, _SU, _E],
    DistributionAgent[_I, _O, _S, _SU, _E],
    agents_base.AgentWrapper[_I, _O, _S, _SU, _E],
):
  """DistributionAgent that reweights the distribution according to parameters.

  In particular this implements temperature reweighting, top_k reweighting
  (zeroing out all entries after the top k ones), and top_p reweighting
  (zeroing out all entries after the first ones whose cumulative sum exceeds p).
  All these parameters can be set together, and at least one of them must be
  set.

  Attributes:
    inner_agent: The original DistributionAgent whose distribution has to be
      reweighted (inherited from `AgentWrapper`).
    temperature: The temperature parameter (float).
    top_p: The p-threshold parameter (float).
    top_k: The k parameter (int).
  """
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None

  def __post_init__(self):
    if self.temperature is None and self.top_p is None and self.top_k is None:
      raise ValueError('Either temperature or top_p or top_k must be provided.')
    if not isinstance(self.inner_agent, DistributionAgent):
      raise ValueError('inner_agent must be a DistributionAgent.')

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def get_next_step_distribution(
      self, state: _S, environment: _E | None = None
  ) -> list[_SU]:
    # Type hint.
    assert isinstance(self.inner_agent, DistributionAgent)
    original = await self.inner_agent.get_next_step_distribution(
        state, environment
    )
    indices = range(len(original))
    values = np.array([p.score for p in original], dtype=np.float32)
    indexed_distribution = zip(indices, values)
    # Use temperature to update the distribution.
    if self.temperature is not None:
      if self.temperature <= 1e-10:
        # When temperature is (very close to) zero, we have a hard-max
        # distribution. Hence we have a uniform distribution over the values
        # that have maximal probability under the original distribution.
        max_value = np.max(values)
        max_indices = np.where(values == max_value)[0]
        values = np.zeros(len(original))
        values[max_indices] = 1.0
        values /= len(max_indices)
      else:
        logits = np.log(values)
        logits /= self.temperature
        values = softmax(logits)
      indexed_distribution = zip(indices, values)
    # Use top_k
    if self.top_k is not None or self.top_p is not None:
      sorted_distribution = sorted(
          indexed_distribution, key=lambda x: x[1], reverse=True
      )
      sorted_values = np.array(
          [p for _, p in sorted_distribution], dtype=np.float32
      )
      cum_values = np.cumsum(sorted_values)
      if self.top_k is not None:
        cutoff = min(self.top_k, len(cum_values))
      else:
        cutoff = len(cum_values)
      if self.top_p is not None:
        initial_cutoff = cutoff
        while cutoff and cum_values[cutoff - 1] >= self.top_p:
          cutoff -= 1
        # We are either at 0 or at a value that is < self.top_p, so we increase
        # by 1 to exceed self.top_p
        cutoff += 1
        # Make sure we apply the more restrictive of top_k and top_p in the case
        # where both were specified.
        cutoff = min(cutoff, initial_cutoff)
      sorted_values[cutoff:] = 0.0
      sorted_values /= np.sum(sorted_values)
      indexed_distribution = [
          (index, sorted_values[i])
          for i, (index, _) in enumerate(sorted_distribution)
      ]
    updated_distribution = [
        # (update, new_score, original_index)
        (original[i].update, p, i) for (i, p) in indexed_distribution
    ]
    # We restore the original order.
    updated_distribution = sorted(updated_distribution, key=lambda x: x[2])
    return [
        agents_base.ScoredUpdate(update=u, score=p)
        for (u, p, _) in updated_distribution
    ]
