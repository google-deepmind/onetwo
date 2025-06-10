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

"""Optimization agents."""

from __future__ import annotations

import collections
from collections.abc import AsyncIterator, Sequence
import contextlib
import copy
import dataclasses
from typing import Generic, TypeAlias, TypeVar, final
from onetwo.agents import agents_base
from onetwo.agents import critics
from onetwo.core import executing
from onetwo.core import tracing


# Type that the base agent takes as input.
_I = TypeVar('_I')

# Type that the base agent returns as output.
_O = TypeVar('_O')

# Type used to represent the state of the base agent.
_S = TypeVar('_S')

# Type used to represent an incremental update of the base agent state.
_U = TypeVar('_U')

# Type used to represent the environment of the base agent.
_E = TypeVar('_E')


@dataclasses.dataclass
class ResamplingAgent(
    Generic[_I, _O, _S, _U, _E],
    agents_base.AgentWrapper[_I, _O, _S, _U, _E],
):
  """Simple resampler possibly using a critic to guide the search.

  At each step, the next step samples are produced as follows:
  - For each expected sample, extra_sampling_factor updates are sampled from the
    inner_agent.
  - These candidates are ranked using the critic.
  - We only keep the top num_candidates samples and discard the others.

  Attributes:
    inner_agent: The agent to sample from (inherited from AgentWrapper).
    critic: A critic to rank the updates.
    extra_sampling_factor: How many extra updates to sample before applying the
      critic.
  """

  critic: critics.RankingFunction[_S, _U]
  extra_sampling_factor: int = 1

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace(
      'ResamplingAgent.sample_next_step', skip=['environment']
  )
  @final
  async def sample_next_step(
      self, *, state: _S, num_candidates: int, environment: _E | None = None
  ) -> list[_U]:
    """See base class (Agent)."""
    next_steps = await self.inner_agent.sample_next_step(  # pytype: disable=wrong-keyword-args
        state=state,
        num_candidates=self.extra_sampling_factor * num_candidates,
        environment=environment,
    )
    ranked_updates_indices = await self.critic(  # pytype: disable=wrong-arg-count
        [(state, next_step) for next_step in next_steps]
    )
    sorted_next_steps = [next_steps[i] for i in ranked_updates_indices]
    return sorted_next_steps[:num_candidates]


# The BeamSearch agent keeps a list of states of size beam_size.
# After each update the list is re-sorted according to the critic, but an
# incremental update to that state is represented as another list of states
# that will replace the current one.
_BSUpdate: TypeAlias = Sequence[_S]


@dataclasses.dataclass
class BeamSearchState(Generic[_S, _U]):
  """Beam search state representation.

  Attributes:
    states: List of states of the underlying agent.
  """
  states: list[_S]

  def __add__(self, update: _BSUpdate) -> BeamSearchState[_S, _U]:
    """Required overload for supporting accumulation of updates."""
    new_states = copy.deepcopy(list(update))
    return BeamSearchState(new_states)

  def __iadd__(self, update: _BSUpdate) -> BeamSearchState[_S, _U]:
    """Required overload for supporting accumulation of updates."""
    self.states = list(update)
    return self


_BSState: TypeAlias = BeamSearchState[_S, _U]


@dataclasses.dataclass
class BeamSearch(
    Generic[_I, _O, _S, _U, _E],
    agents_base.Agent[
        _I, _O, _BSState, _BSUpdate, _E
    ],
):
  """Beam search agent.

  Maintains a sorted list of states of size beam_size.

  Attributes:
    inner_agent: The agent to perform beam search on.
    critic: A critic to rank the updates.
    beam_size: The number of active states to maintain.
    max_candidates: How many candidates to sample before applying the critic.
      If None we will use the beam_size.
    diversify_beam: If True, then the states in the beam will be compared
      (hence they need to be hashable) so that we minimize duplication and
      increase exploration. This is useful in particular when the sampling
      of the underlying agent is deterministic.
  """

  inner_agent: agents_base.Agent[_I, _O, _S, _U, _E]
  critic: critics.RankingFunction[_S, _U]
  beam_size: int = 2
  max_candidates: int | None = None  # If None we will use the beam_size.
  diversify_beam: bool = False

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @final
  async def initialize_state(
      self, inputs: _I, environment: _E | None = None
  ) -> _BSState:
    """See base class (Agent)."""
    initial_state = await self.inner_agent.initialize_state(inputs, environment)  # pytype: disable=wrong-arg-count
    return BeamSearchState([initial_state] * self.beam_size)

  @contextlib.asynccontextmanager
  async def start_environment(self) -> AsyncIterator[_E]:
    """See base class (Agent)."""
    async with self.inner_agent.start_environment() as env:
      yield env

  @final
  def is_finished(self, state: _BSState) -> bool:
    """See base class (Agent)."""
    # We are done as soon as the best state is done.
    return self.inner_agent.is_finished(state.states[0])

  @final
  def extract_output(self, state: _BSState) -> _O | None:
    """See base class (Agent)."""
    return self.inner_agent.extract_output(state.states[0])

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace('BeamSearch.sample_next_step', skip=['environment'])
  @final
  async def sample_next_step(
      self,
      *,
      state: _BSState,
      num_candidates: int,
      environment: _E | None = None,
  ) -> list[_BSUpdate]:
    """See base class (Agent)."""
    if num_candidates > 1:
      raise NotImplementedError('num_candidates must be 1 for beam search.')

    if self.diversify_beam:
      # As an optimization, if we have multiple times the same state in the beam
      # we pool together the updates.
      multiplicities = collections.Counter(state.states)
      unique_states = list(multiplicities.keys())
      executables = [
          self.inner_agent.sample_next_step(  # pytype: disable=wrong-keyword-args
              state=state,
              num_candidates=multiplicities[state]
              * (self.max_candidates or self.beam_size),
              environment=environment,
          )
          for state in unique_states
      ]
      next_steps_lists = await executing.par_iter(executables)
      next_inner_steps = []
      for inner_state, next_steps in zip(unique_states, next_steps_lists):
        # We add the new state/update pairs to the next_inner_steps list, but
        # try and deduplicate the elements to get a diverse list.
        # But we need at least beam_size elements to choose from at the end so
        # we ensure that each beam contributes one element.
        added_candidates = 0
        first_candidate = None
        for i, candidate in enumerate(next_steps):
          if first_candidate is None:
            first_candidate = candidate
          if i < len(next_steps) - multiplicities[inner_state]:
            if (inner_state, candidate) not in next_inner_steps:
              next_inner_steps.append((inner_state, candidate))
              added_candidates += 1
          else:
            next_inner_steps.append((inner_state, candidate))
            added_candidates += 1
        if added_candidates < multiplicities[inner_state]:
          # We need to add more candidates to fill the beam, so we duplicate
          # the first that we added.
          while added_candidates < multiplicities[inner_state]:
            next_inner_steps.append((inner_state, first_candidate))
            added_candidates += 1
      # At this point we have at least beam_size elements in next_inner_steps
      # since we added at least multiplicities[inner_state] elements at each
      # iteration. so we get a total which exceeds the number of elements we
      # started with, which is the beam size.
    else:
      executables = [
          self.inner_agent.sample_next_step(  # pytype: disable=wrong-keyword-args
              state=inner_state,
              num_candidates=self.max_candidates or self.beam_size,
              environment=environment,
          )
          for inner_state in state.states
      ]
      next_steps_lists = await executing.par_iter(executables)
      next_inner_steps = []
      for inner_state, next_steps in zip(state.states, next_steps_lists):
        next_inner_steps.extend([(inner_state, step) for step in next_steps])
    # We sort the candidates using the critic.
    sorted_states_indices = await self.critic(next_inner_steps)  # pytype: disable=wrong-arg-count
    # And we pick the top beam_size ones.
    best_states_and_updates = [
        next_inner_steps[sorted_states_indices[i]]
        for i in range(self.beam_size)
    ]
    # The update we return is a new BeamSearch state, i.e. a list of new states
    # for the underlying agent.
    return [[s + u for s, u in best_states_and_updates]]
