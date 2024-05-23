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

"""Utilities for OneTwo unit tests."""

import collections
from collections.abc import Mapping
import json
import pprint
from typing import Any
import unittest


class CounterAssertions(unittest.TestCase):
  """Mixin class for counter assertions."""

  # pylint: disable=invalid-name
  def assertCounterEqual(
      self,
      counter_first: collections.Counter[str],
      counter_second: collections.Counter[str],
  ) -> None:
    # Remove zero values.
    first = counter_first - collections.Counter()
    second = counter_second - collections.Counter()
    message = f'A - B contains: {pprint.pformat(first - second)}\n'
    message += f'B - A contains: {pprint.pformat(second - first)}'
    return self.assertEqual(dict(first), dict(second), message)


def maybe_read_file(filepath: str) -> str:
  """Returns the contents of the file if it exists, or else empty string."""
  try:
    with open(filepath) as f:
      file_contents = f.read()
    return file_contents
  except IOError:
    return ''


def maybe_read_json(filepath: str) -> Mapping[str, Any] | None:
  """Returns the file contents as JSON, or None if there is a problem."""
  file_contents = maybe_read_file(filepath)
  try:
    return json.loads(file_contents)
  except json.JSONDecodeError:
    return None

