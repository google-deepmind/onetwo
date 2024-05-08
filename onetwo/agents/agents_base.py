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

""""Core interfaces for agents."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Callable
import contextlib
import copy
import dataclasses
from typing import Generic, TypeVar, final

from onetwo.core import executing
from onetwo.core import sampling
from onetwo.core import tracing
from onetwo.core import utils


# Type that `Agent.run` takes as input.
_I = TypeVar('_I')

# Type that `Agent.run` returns as output.
_O = TypeVar('_O')

# Type used to represent the state of the agent.
_S = TypeVar('_S')

# Type used to represent an incremental update of the agent state.
_U = TypeVar('_U')

# Type used to represent the environment of the agent.
_E = TypeVar('_E')


@dataclasses.dataclass
class Agent(Generic[_I, _O, _S, _U, _E], metaclass=abc.ABCMeta):
  """Generic interface for an agent that performs repeated steps toward a goal.

  Should be stateless, in the sense that the attributes of the Agent should
  serve purely as configuration parameters, which are independent of the inputs
  that are passed in to any given invocation of the agent. To the extent needed,
  the state of execution of any given set of inputs will always be passed
  explicitly in the arguments and return values of the agent's methods.

  Aside from the arbitrary input type (I) and output type (O), the Agent is
  parameterized by both a state type (S) and state update type (U). These types
  are expected to be related in the following ways:
  * At the beginning of execution, the state (S) will be initialized from the
    inputs (I), and from then on, only the state (S) will be maintained. This
    means that the state (S) should subsume all of the information from the
    inputs that is relevant to performance of the task. The mapping of inputs
    to state is controlled by `Agent.initialize_state`.
  * The state (S) should support addition with a state update (U) to yield a
    new state (S), i.e., via `new_state = state + update` or `state += update`.
  * The state (S) should contain all of the information needed for generating
    the outputs (O). The mapping of state to outputs is controlled by
    `Agent.extract_output`.

  The last type that the Agent in parameterized by is that of the
  environment (E). The environment is optional (in the sense that it can be
  parameterized to `None`), but is used for cases where the agent's processing
  requires some kind of transient, but stateful and potentially complex
  apparatus, such as a Python sandbox or simulator, etc., which can be expensive
  to set up, and which may require some kind of clean-up when done. The
  environment (E) is related to the other types in the following ways:
  * From a given agent instance, we can construct a fresh environment at any
    time (and as many times as we want) via the following context-manager
    construct:
    ```
      agent = ...
      async with agent.start_environment() as env:
         # In here, we can call other methods on `agent` using `env` as the
         # environment.
    ```
    The environment will automatically be cleaned up when exiting the `with`
    block.
  * The agent itself (once it has been set up) is expected to be stateless and
    thus safe to be shared across multiple threads. The environment, on the
    other hand, is allowed to be stateful and is not required to be thread-safe.
    In cases where an agent is used from multiple threads, a separate
    environment should be created for each thread. When performing an experiment
    run over a dataset, a typical usage pattern would be where we construct a
    single agent instance for the experiment run as a whole, but then create a
    separate environment for each of the top-level examples in the dataset.
    If the entire dataset will be processed in a single thread, however, then
    sharing a single environment across the entire experiment run is also an
    option.
  * While the environment is "stateful" in the sense that it is mutable and may
    mutate its internal state in the course of its processing, for maximal
    reproducibility, this internal state should ideally be maintained only for
    efficiency reasons, and the environment should ideally behave as if it were
    logically stateless. That is, when performing an agent operation like
    `agent.sample_next_step(state, env)` we should ideally get the same result
    (or at least, the result should be drawn from the same distribution)
    regardless of the value of `env`. One way of accomplishing this is to think
    of the "state" (S) as a compact and serializable memo of the state of the
    agent, whereas the actual full state of the agent may be a more complex /
    non-serializable object that can be reconstructed from the state memo (S) on
    demand, but which is expensive to reconstruct. In this model, the job of the
    environment (E) is to maintain a cache of these live agent states, so as to
    minimize the amount of wasted computational time required in reconstructing
    agent states. If the environment (E), for whatever reason, does not have
    a cached live state corresponding to a given state memo (S), however, it
    can still reconstruct a live agent state on demand, and so still return the
    same output as it would have if the agent state had been cached, just at the
    expense of more computational time.
  * The detailed API of the environment, and the details of how the agent
    interacts with its environment, however, are left up to the individual agent
    and environment implementations. The degree to which the environment
    maintains its ideal of reproducibility and "logical stateless" is similarly
    left up to the individual agent and environment implementations, and should
    be addressed in the agent's documentation.
  * The state (S) and state updates (U) are required to be serializable and
    copiable, and in general are to be thought of as pure, well-behaved data
    objects. The environment (E), on the other hand, is not expected to be
    serializable or copiable, and should be thought of as a "living" object, in
    the sense that it can contain non-serializable things like function pointers
    and messy things like thread pools, whose life cycle need to be carefully
    managed. The agent is also not required to be serializable or copiable, and
    so is allowed to contain non-serializable things like function pointers;
    for things like thread pools that require clean-up, however, it is better to
    maintain them strictly in the environment rather than in the agent, to take
    advantage of the environment's clearly-defined life cycle.

  An Agent can be called in one of three ways:
  * **Agent.__call__:** I.e., by using the agent as a callable. Runs the agent
    to completion and returns the final output. By default, this consists of
    simply repeatedly sampling a single next step until reaching a finished
    state. This is the simplest usage of the agent, in which the Agent can be
    thought of as simply a function that maps inputs (I) to outputs (O).
  * **Agent.stream_updates:** Yields a stream of state updates either until
    completion, or until the caller aborts the process. In this case, it is up
    to the caller to extract the output from the final state. This is how the
    agent would typically be called, for example, if it is expected that the
    agent may take a long time to run, and if we may want to pause the agent,
    save its state, and then resume execution at some later point in time.
    The stream of updates can be aggregated to form a stream of states.
  * **Agent.sample_next_step:** Samples a specified number of possible
    candidates for the next step of the agent strategy. This is the
    finest-grained way of driving the agent's execution. This is how
    the agent would be called, for example, if we want to wrap it with another
    outer agent, such that the outer agent is in charge of navigating the search
    space of possible trajectories (possibly in a non-linear way, for pursuing
    multiple branches of exploration), while the inner agent would be in charge
    of deciding the possible next steps based on the state of any given
    trajectory. The outer agent could potentially also implement a multi-agent
    strategy, in which it coordinates among multiple inner agents, who take
    turns in performing steps vs. a fully- or partially-shared state.
  """

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @abc.abstractmethod
  async def initialize_state(
      self, inputs: _I, environment: _E | None = None
  ) -> _S:
    """Returns a newly initialized state based on the input.

    Args:
      inputs: Input to the agent, representing the overall goal that the agent
        is trying to achieve.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).
    """

  # TODO: For more convenient usage in colab, provide an ordinary
  # `@contextlib.contextmanager` variant of this as well.
  @contextlib.asynccontextmanager
  async def start_environment(self) -> AsyncIterator[_E]:
    """Context manager to start the environment.

    Usage:
    ```
      agent = ...
      async with agent.start_environment() as env:
         # In here, we can call other methods on `agent` using `env` as the
         # environment.
    ```

    Yields:
      Environment object, which will be automatically cleaned up when exiting
      the `with` block.
    """
    del self
    # TODO: Ideally we would `yield None` here only if type `_E` is
    # parameterized to `None` and raise a `NotImplementedError` otherwise.
    # Is there any way to do this?
    yield None
    # if _E == type(None):
    #   yield None
    # else:
    #   raise NotImplementedError(
    #       f'Agent class {type(self)} needs an implementation of'
    #       f' `start_environment`, since environment type is non-None: {_E}'
    #   )

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace('SingleSampleAgent.sample_next_step', skip=['environment'])
  @abc.abstractmethod
  async def sample_next_step(
      self,
      *,
      state: _S,
      num_candidates: int,
      environment: _E | None = None,
  ) -> list[_U]:
    """Samples up to `num_candidates` possible next steps of the agent strategy.

    The candidates are sampled with replacement, which means that the same
    candidate next step may appear multiple times.

    Args:
      state: Current state of the agent.
      num_candidates: Number of possible next steps to generate. The idea is
        that the agent (or the caller of the agent) will choose one of these
        candidates to adopt as the actual next step. Depending on the agent
        implementation, it is possible that fewer than the requested number of
        candidates may be returned (or even no candidates, e.g., if no further
        state transitions are possible from the given state).
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).

    Returns:
      An incremental update to the agent state that would occur as a result of
      performing each of the possible next steps.
    """

  def is_finished(self, state: _S) -> bool:
    """Returns whether the agent strategy is in finished state.

    By default, this simply returns False, which means the Agent can be run
    indefinitely (until explicitly stopped by the caller, or via some other
    stopping condition). Sub-classes are encouraged to override this function.

    Args:
      state: Current state of the agent.
    """
    del state
    return False

  @abc.abstractmethod
  def extract_output(self, state: _S) -> _O | None:
    """Returns the output from the strategy if stopping at the given state.

    Can be called on any state, regardless of whether `is_finished() == True`.
    If the output is not defined for a given state (e.g., due to execution being
    unfinished), then should return `None`.

    Args:
      state: Current (presumably final) state of the agent.
    """

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @final
  async def stream_updates(
      self,
      *,
      initial_state: _S,
      environment: _E | None = None,
      max_steps: int | None = None,
      stop_condition: Callable[[_S], bool] | None = None,
  ) -> AsyncIterator[_U]:
    """Yields a stream of state updates resulting from executing the agent.

    Note that the stream of updates can be aggregated to form a stream of
    states. The caller is free to interrupt the stream of execution at any time
    and to attempt to extract an output from the final state, or from any other
    state along the way.

    Args:
      initial_state: Initial state of the agent.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).
      max_steps: If specified, then will stop after the given number of steps
        have been performed (if it doesn't already stop earlier). Does not
        count any steps that might have been performed in the construction of
        `initial_state`.
      stop_condition: If specified, then will stop once the state satisfies the
        given condition (if it doesn't already stop earlier).
    """
    state = copy.deepcopy(initial_state)
    step = 0
    while not self.is_finished(state):
      if max_steps is not None and step >= max_steps:
        break
      if stop_condition is not None and stop_condition(state):
        break
      candidate_updates = await self.sample_next_step(
          state=state, num_candidates=1, environment=environment
      )
      if not candidate_updates:
        break
      update = candidate_updates[0]
      state += update
      step += 1
      yield update

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @final
  async def stream_states(
      self,
      *,
      initial_state: _S,
      environment: _E | None = None,
      max_steps: int | None = None,
      stop_condition: Callable[[_S], bool] | None = None,
  ) -> AsyncIterator[_S]:
    """Yields the intermediate states resulting from executing the agent.

    Similar to `stream_updates`, except that it streams the full agent states.

    Args:
      initial_state: Initial state of the agent.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).
      max_steps: If specified, then will stop after the given number of steps
        have been performed (if it doesn't already stop earlier). Does not
        count any steps that might have been performed in the construction of
        `initial_state`.
      stop_condition: If specified, then will stop once the state satisfies the
        given condition (if it doesn't already stop earlier).
    """
    state = copy.deepcopy(initial_state)
    async for update in self.stream_updates(
        initial_state=state,
        environment=environment,
        max_steps=max_steps,
        stop_condition=stop_condition,
    ):
      # We do ordinary `+` here rather than `+=` to ensure we yield a stream of
      # distinct states, rather than updating a single state in place.
      state = state + update
      yield state

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      inputs: _I,
      *,
      initial_state: _S | None = None,
      max_steps: int | None = None,
      stop_condition: Callable[[_S], bool] | None = None,
      return_final_state: bool = False,
  ) -> _O | None | tuple[_O | None, _S]:
    """"Runs the agent until reaching finished state or other condition is met.

    This is the simplest way of invoking an agent. If an environment is
    needed, it will be automatically created and the cleaned up once execution
    is complete. In this usage, the agent can be thought of as simply a function
    that maps inputs (I) to outputs (O).

    Args:
      inputs: Input to the agent, representing the overall goal that the agent
        is trying to achieve.
      initial_state: Initial state of the agent. If specified, then `inputs` is
        ignored, as it is assumed that the state subsumes the original inputs;
        if not specified, then a new state is created based on `inputs`.
      max_steps: If specified, then will stop after the given number of steps
        have been performed (if it doesn't already stop earlier). Does not
        count any steps that might have been performed in the construction of
        `initial_state`.
      stop_condition: If specified, then will stop once the state satisfies the
        given condition (if it doesn't already stop earlier).
      return_final_state: If True, then returns both the agent's final output
        and the agent's final state. If False, then returns just the agent's
        final output.

    Returns:
      The final output of the agent, or optionally, a tuple of the final output
      and the agent's final state. If the agent failed to reach a state for
      for which an output is defined, then the output will be `None`.
    """
    async with self.start_environment() as env:
      if initial_state is not None:
        state = copy.deepcopy(initial_state)
      else:
        state = await self.initialize_state(inputs, environment=env)
      async for update in self.stream_updates(
          initial_state=state,
          environment=env,
          max_steps=max_steps,
          stop_condition=stop_condition,
      ):
        state += update
    output = self.extract_output(state)
    if return_final_state:
      return output, state
    else:
      return output

  @executing.make_executable(copy_self=False)
  @final
  async def start_environment_and_sample_next_step(
      self, *, state: _S, num_candidates: int
  ) -> list[_U]:
    """Starts a new environment and executes `sample_next_step`."""
    async with self.start_environment() as env:
      return await self.sample_next_step(
          state=state, num_candidates=num_candidates, environment=env
      )

  @executing.make_executable(copy_self=False)
  @final
  async def start_environment_and_stream_updates(
      self,
      *,
      initial_state: _S,
      max_steps: int | None = None,
      stop_condition: Callable[[_S], bool] | None = None,
  ) -> AsyncIterator[_U]:
    """Starts a new environment and executes `stream_updates`."""
    async with self.start_environment() as env:
      async for update in self.stream_updates(
          initial_state=initial_state,
          environment=env,
          max_steps=max_steps,
          stop_condition=stop_condition,
      ):
        yield update

  @executing.make_executable(copy_self=False)
  @final
  async def start_environment_and_stream_states(
      self,
      *,
      initial_state: _S,
      max_steps: int | None = None,
      stop_condition: Callable[[_S], bool] | None = None,
  ) -> AsyncIterator[_S]:
    """Starts a new environment and executes `stream_states`."""
    async with self.start_environment() as env:
      async for state in self.stream_states(
          initial_state=initial_state,
          environment=env,
          max_steps=max_steps,
          stop_condition=stop_condition,
      ):
        yield state


@dataclasses.dataclass
class SingleSampleAgent(Agent[_I, _O, _S, _U, _E]):
  """Agent whose implementation is based on generating one sample at a time.

  Specifically, the sub-class of `SingleSampleAgent` only needs to implement
  `_sample_single_next_step`, and then `sample_next_step` will be implemented
  automatically in terms of this.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  async def _sample_single_next_step(
      self, *, state: _S, environment: _E | None = None
  ) -> _U:
    """Samples one possible next step of the agent strategy.

    Args:
      state: Current state of the agent.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).

    Returns:
      An incremental update to the agent state that would occur as a result of
      performing the given step.
    """

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @tracing.trace('SingleSampleAgent.sample_next_step', skip=['environment'])
  @final
  async def sample_next_step(
      self, *, state: _S, num_candidates: int, environment: _E | None = None
  ) -> list[_U]:
    """Samples `num_candidates` possible next steps of the agent strategy.

    Overridden from base class (Agent).

    Args:
      state: Current state of the agent.
      num_candidates: Number of possible next steps to generate. The idea is
        that the agent (or the caller of the agent) will choose one of these
        candidates to adopt as the actual next step.
      environment: Environment in which to perform the operation. Can be omitted
        if the given agent does not require an environment (i.e., if the
        environment type `_E` is parameterized as `None`).

    Returns:
      An incremental update to the agent state that would occur as a result of
      performing each of the possible next steps.
    """
    if num_candidates == 1:
      # We can omit the `sampling.repeat` when num_candidates == 1, to avoid
      # unnecessary deep-copying of the executable.
      next_step = await self._sample_single_next_step(
          state=state, environment=environment
      )
      return [next_step]
    else:
      return list(
          await executing.par_iter(
              sampling.repeat(
                  self._sample_single_next_step(
                      state=state, environment=environment
                  ),
                  num_repeats=num_candidates,
              )
          )
      )


@dataclasses.dataclass
class AgentWrapper(
    Generic[_I, _O, _S, _U, _E],
    Agent[_I, _O, _S, _U, _E],
    metaclass=abc.ABCMeta,
):
  """Special case of an Agent that wraps an inner_agent.

  This class provides a way to not have to redefine the basic methods that are
  just delegated to the inner_agent.
  """

  inner_agent: Agent[_I, _O, _S, _U, _E]

  @executing.make_executable(copy_self=False, non_copied_args=['environment'])
  @final
  async def initialize_state(
      self, inputs: _I, environment: _E | None = None
  ) -> _S:
    """Overridden from base class (Agent)."""
    return self.inner_agent.initialize_state(inputs, environment)

  @contextlib.asynccontextmanager
  async def start_environment(self) -> AsyncIterator[_E]:
    """Overridden from base class (Agent)."""
    async with self.inner_agent.start_environment() as env:
      yield env

  @final
  def is_finished(self, state: _S) -> bool:
    """Overridden from base class (Agent)."""
    return self.inner_agent.is_finished(state)

  @final
  def extract_output(self, state: _S) -> _O | None:
    """Overridden from base class (Agent)."""
    return self.inner_agent.extract_output(state)


@dataclasses.dataclass
class ScoredUpdate(Generic[_U]):
  """A state update that also includes an associated score."""
  update: _U
  score: float

  def __radd__(self, state: _S) -> _S:
    """Required overload for supporting accumulation of updates.

    This covers the case where we add a `ScoredUpdate` to a state that does not
    support addition to a `ScoredUpdate`, in which case we try to simply add
    the update without the score.

    Args:
      state: State to add to.

    Returns:
      New state, with the update applied.
    """
    return state + self.update

  def __hash__(self) -> int:
    """Required in case we want to compare updates for deduplication."""
    return hash(self.update)


@dataclasses.dataclass
class UpdateListState(Generic[_I, _U]):
  """Simple agent state representation that bundles inputs with list of updates.

  Attributes:
    inputs: Input to the agent, representing the overall goal that the agent
      is trying to achieve.
    updates: Sequence of state updates that were generated by the agent so far.
  """
  inputs: _I = ''
  updates: list[_U] = dataclasses.field(default_factory=list)

  def __add__(self, update: _U) -> UpdateListState[_I, _U]:
    """Required overload for supporting accumulation of updates."""
    new_state = copy.deepcopy(self)
    new_state.updates.append(update)
    return new_state

  def __iadd__(self, update: _U) -> UpdateListState[_I, _U]:
    """Required overload for supporting accumulation of updates."""
    self.updates.append(update)
    return self

  def __hash__(self) -> int:
    """Required in case we want to compare states for deduplication."""
    return hash(self.inputs) + sum([hash(update) for update in self.updates])
