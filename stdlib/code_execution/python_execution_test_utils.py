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

import dataclasses
import unittest

from onetwo.stdlib.code_execution import python_execution


class SandboxResultAssertions(unittest.TestCase):
  """Mixin class for SandboxResult assertions."""

  # pylint: disable=invalid-name
  def assertSandboxResultEqualIgnoringTiming(
      self,
      expected_result: python_execution.SandboxResult,
      actual_result: python_execution.SandboxResult,
  ) -> None:
    # Remove timing-related content.
    expected_without_timing = dataclasses.replace(expected_result, timing=None)
    actual_without_timing = dataclasses.replace(actual_result, timing=None)
    return self.assertEqual(
        expected_without_timing,
        actual_without_timing,
        'SandboxResult differed by more than just'
        f' timing.\nExpected:\n{expected_result!r}\nActual:\n{actual_result!r}',
    )
