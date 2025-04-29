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

"""Utilities for OneTwo unit tests involving Python execution."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
from typing import Any, Callable
import unittest

from onetwo.core import executing
from onetwo.stdlib.code_execution import python_execution

# Aliases for brevity.
_SandboxResult = python_execution.SandboxResult


class SandboxResultAssertions(unittest.TestCase):
  """Mixin class for SandboxResult assertions."""

  # pylint: disable=invalid-name
  def assertSandboxResultEqualIgnoringTiming(
      self,
      expected_result: _SandboxResult,
      actual_result: _SandboxResult,
  ) -> None:
    # Remove timing-related content.
    expected_without_timing = dataclasses.replace(expected_result, timing=None)
    actual_without_timing = dataclasses.replace(actual_result, timing=None)
    self.assertEqual(
        expected_without_timing,
        actual_without_timing,
        'SandboxResult differed by more than just'
        f' timing.\nExpected:\n{expected_result!r}\nActual:\n{actual_result!r}',
    )


@dataclasses.dataclass
class PythonSandboxForTest(python_execution.PythonSandbox):
  """Mock PythonSandbox.

  Attributes:
    hook_objects: Objects containing modifiable state of the hook functions.
      (Required as part of the PythonSandbox interface.)
    reply_by_request: Mapping from request to reply, or to a sequence of replies
      in case we want to return different results on the 1st call vs. the 2nd
      call, etc.
    default_reply: Default reply if not found in reply_by_request.
    requests: All `run` requests that were received, in the order received.
      (Same format as `unexpected_requests` below.)
    unexpected_requests: Requests that were not found in the corresponding
      mappings (i.e., requests for which we ended up falling back to returning
      the `default_reply`).
  """

  hook_objects: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # Attributes for controlling the replies to be returned.
  reply_by_request: Mapping[str, _SandboxResult | Sequence[_SandboxResult]] = (
      dataclasses.field(default_factory=dict)
  )
  default_reply: _SandboxResult = dataclasses.field(
      default_factory=_SandboxResult
  )

  # Attributes used for tracking the actual requests / replies (for assertions).
  requests: list[str] = dataclasses.field(init=False, default_factory=list)
  unexpected_requests: list[str] = dataclasses.field(
      init=False, default_factory=list
  )

  # Attributes used for tracking the actual requests / replies (internal).
  _num_run_calls_by_request: collections.Counter[str] = (
      dataclasses.field(init=False, default_factory=collections.Counter)
  )

  def is_stateful(self) -> bool:
    """See base class (PythonSandbox)."""
    return True

  @executing.make_executable  # pytype: disable=wrong-arg-types
  def run(self, code: str) -> _SandboxResult:
    """See base class (PythonSandbox)."""
    self.requests.append(code)

    # By request.
    if code in self.reply_by_request:
      reply = self.reply_by_request[code]
      if isinstance(reply, str):
        # Single reply specified. Always return it.
        return reply
      else:
        # Sequence of replies specified. Return the next (until we run out).
        reply_index = self._num_run_calls_by_request[code]
        self._num_run_calls_by_request[code] += 1
        if reply_index < len(reply):
          return reply[reply_index]

    # Default.
    self.unexpected_requests.append(code)
    return self.default_reply

  async def set_variables(self, **variables: Any) -> None:
    """See base class (PythonSandbox)."""
    raise NotImplementedError()

  async def get_variables(self, *names: str) -> Mapping[str, Any]:
    """See base class (PythonSandbox)."""
    raise NotImplementedError()

  def get_hook_object(self, key: str) -> Any:
    """See base class (PythonSandbox)."""
    return self.hook_objects.get(key)


@dataclasses.dataclass
class PythonSandboxForTestFactory(python_execution.PythonSandboxFactory):
  """Mock PythonSandboxFactory that returns a hard-coded PythonSandboxForTest.

  Attributes:
    default_sandbox: Sandbox to be returned by default on calls to
      `create_sandbox`. Note that in order to satisfy behavior expectations of
      `PythonPlanningAgent`, the factory will automatically overwrite the
      `hook_objects` member of this default sandbox each time before returning
      it from `create_sandbox`. (This should be fine in cases where we only need
      to return one sandbox; for tests in which we expect multiple sandboxes to
      be created, however, we may need more fine-grained configuration
      controlling the sandboxes to return.)
  """

  default_sandbox: PythonSandboxForTest = dataclasses.field(
      default_factory=PythonSandboxForTest
  )

  def create_sandbox(
      self,
      *,
      timeout: datetime.timedelta = datetime.timedelta(seconds=10),
      imports: Sequence[str] | str = tuple(),
      hooks: Mapping[str, Callable[..., Any]] | None = None,
      hook_objects: Mapping[str, Any] | None = None,
      allow_restarts: bool = False,
  ) -> PythonSandboxForTest:
    # If necessary, we can add here more detailed configurations controlling the
    # sandbox to return, e.g., depending on the parameters specified, or on the
    # number of times that `create_sandbox` has been called. For now we just
    # reuse the single `default_sandbox`.
    self.default_sandbox.hook_objects = hook_objects or {}
    return self.default_sandbox
