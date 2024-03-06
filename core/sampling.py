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

"""Implementation of Self-Consistency as an Executable."""

from collections.abc import Callable
import copy
from typing import Sequence, TypeVar

from onetwo.core import caching
from onetwo.core import executing


Result = TypeVar('Result')
Executable = executing.Executable[Result]

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
      wrapper = _update_result_with_sample_id(
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
