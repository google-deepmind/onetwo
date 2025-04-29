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

"""Functions and classes for sampling from a conditional distribution."""

import abc
from collections.abc import Awaitable, Callable, Sequence
import copy
import dataclasses
import itertools
from typing import Generic, ParamSpec, Protocol, TypeVar

from onetwo.core import caching
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils


Result = TypeVar('Result')
Executable = executing.Executable[Result]

# Type representing a strategy's arguments.
_Args = ParamSpec('_Args')

# Type representing a strategy's output.
_O = TypeVar('_O')

# Constants used in info fields.
MARGINALIZED_INPUTS_ID = 'marginalized_inputs_id'
MARGINALIZED_INPUTS_SIZE = 'marginalized_inputs_size'
SAMPLE_ID = 'sample_id'
SAMPLE_SIZE = 'sample_size'


@executing.make_executable('update_result_fn')
def _update_result_with_sample_id(
    result: Result,
    update_result_fn: Callable[[Result, int], Result],
    sample_id: int,
) -> Result:
  return update_result_fn(result, sample_id)


def repeat(
    executable: Executable[Result],
    num_repeats: int,
    update_result_fn: Callable[[Result, int], Result] | None = None,
    start_index: int = 0,
) -> list[Executable[Result]]:
  """Repeat n times an executable and update its requests.

  For caching purposes, all the requests are tagged with an extra cache key so
  that they can be distinguished and replayed appropriately.

  Args:
    executable: Executable to repeat.
    num_repeats: Number of times it should be repeated.
    update_result_fn: Function to apply to the results to add per instance
      information (takes the result and the sample_id as inputs and returns an
      updated result).
    start_index: If provided, the ids assigned to the sample start at this
      number. This can be used for example to get additional (distinct) samples
      in successive rounds:
      ```
      executable = ...
      executables = repeat(executable, 3)
      res = await par_iter(executables)
      if condition:
        executables = repeat(executable, 3, start_index=3)
        res = await par_iter(executables)
      ```

  Returns:
    A list of executables which each copy the source executable and update
    the requests on the fly to add an ID to the cache key and update the engine
    parameters, with the update_result_fn applied to the results.
  """
  wrappers = []

  for i in range(num_repeats):
    wrapper = caching.SamplingKeyUpdater(
        str(i + start_index), copy.deepcopy(executable)
    )
    if update_result_fn is not None:
      wrapper = _update_result_with_sample_id(  # pytype: disable=wrong-keyword-args
          wrapper, update_result_fn, sample_id=i
      )
    wrappers.append(wrapper)
  return wrappers


def repeat_and_execute(  # pytype: disable=invalid-annotation
    executable: Executable[Result],
    num_repeats: int,
    update_result_fn: Callable[[Result, int], Result] | None = None,
) -> executing.Executable[Sequence[Result]]:
  """Repeat n times an executable in parallel.

  For caching purposes, all the requests are tagged with an extra cache key so
  that they can be distinguished and replayed appropriately.

  Args:
    executable: Executable to repeat.
    num_repeats: Number of times it should be repeated.
    update_result_fn: Function to apply to the results to add per instance
      information (takes the result and the sample_id as inputs and returns an
      updated result).

  Returns:
    A Sequence of results, one for each repetition of the executable.
  """
  return executing.par_iter(
      repeat(executable, num_repeats, update_result_fn)
  )


class Sampler(Generic[_O], Protocol):
  """Interface for generating samples from a conditional distribution."""

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @abc.abstractmethod
  async def __call__(
      self,
      *args: _Args.args,
      num_samples: int = 1,
      start_index: int = 0,
      **kwargs: _Args.kwargs,
  ) -> Sequence[_O]:
    """Returns the specified number of samples of output for the given inputs.

    Args:
      *args: Positional arguments portion of the inputs to condition on.
      num_samples: The number of samples to return.
      start_index: If provided, the ids assigned to the sample start at this
        number. This can be used for example to get additional (distinct)
        samples in successive rounds.
      **kwargs: Keyword arguments portion of the inputs to condition on.
    """


@dataclasses.dataclass
class Repeated(Generic[_O], Sampler[_O]):
  """Sampler based on repeated calls to some underlying strategy.

  Serves as a callable with the same signature as the inner strategy, except
  that it takes additional keyword arguments `num_samples` and `start_index`,
  and instead of returning a single output, it returns a sequence of outputs.

  Attributes:
    inner: The inner strategy that is used for generating each sample. Could be,
      for example, a function that makes calls to an LLM with temperature > 0,
      or any other function that could potentially return a different answer
      each time it is called, via sampling from some kind of underlying
      distribution.
  """

  inner: Callable[_Args, _O] | Callable[_Args, Awaitable[_O]]

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      *args: _Args.args,
      num_samples: int = 1,
      start_index: int = 0,
      **kwargs: _Args.kwargs,
  ) -> Sequence[_O]:
    """Overridden from base class (Sampler)."""
    repeated_executable = repeat(
        executable=self.inner(*args, **kwargs),
        num_repeats=num_samples,
        start_index=start_index,
    )
    parallel_executable = executing.parallel(*repeated_executable)
    result_objects = await parallel_executable
    return result_objects


@dataclasses.dataclass
class RoundRobin(Generic[_O], Sampler[_O]):
  """Sampler that performs round-robin calls to multiple inner samplers.

  Serves as a callable with the same signature as the inner samplers, which are
  all expected to have the same signature as one another (or at minimum, to
  accept the same arguments).

  Attributes:
    inner: The inner samplers, over which round-robin calls are performed.
  """

  inner: Sequence[Sampler[_O]]

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self,
      *args: _Args.args,
      num_samples: int = 1,
      start_index: int = 0,
      **kwargs: _Args.kwargs,
  ) -> Sequence[_O]:
    """Overridden from base class (Sampler)."""
    num_samplers = len(self.inner)
    repeated_executable = []
    # We can do straightforward round-robin here, without any explicit batching,
    # since batching is handled automatically at the level of the backend, as
    # long as we execute the samplers in parallel.
    for sample_index in range(start_index, start_index + num_samples):
      sampler_id = sample_index % num_samplers
      sampler = self.inner[sampler_id]
      repeated_executable.append(
          sampler(*args, num_samples=1, start_index=sample_index, **kwargs),
      )

    # We execute the sampling as much as possible in parallel.
    parallel_executable = executing.parallel(*repeated_executable)
    result_object_batches = await parallel_executable
    result_objects = list(itertools.chain.from_iterable(result_object_batches))

    return result_objects
