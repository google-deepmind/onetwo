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

"""Defines the base Backend class."""

import dataclasses


@dataclasses.dataclass
class Backend:
  """Interface for a class that registers its method."""

  # Name under which the methods are registered by default.
  _default_name: str = dataclasses.field(
      init=False,
      default='default',
  )

  # Not an abstract method since by default it is ok not to register anything.
  def register(self, name: str | None = None):
    """Add the relevant methods to the registry.

    It is necessary for a child class to override this method in order to
    specify which methods it wants to expose/register.
    In particular this can be used to call `configure` or `get_variant` on
    builtins or to directly register methods of this object under particular
    names.
    For example the body of this method can look like:
    ```
    registry = routing.function_registry
    # Register self.my_fn with the provided or default name prepended.
    registry[f'{name or self._default_name}.my_fn'] = self.my_fn
    # Configure some builtin.
    llm.generate_text.configure(
        self.my_generate_text,
        temperature=default_temperature
    )
    # Configure a variant of the builtin with a different default parameter.
    variant = llm.generate_text.get_variant(
        self.my_generate_text
        temperature=0.0
    )
    # Register the variant under a special name.
    registry['llm.zero_temp'] = variant
    ```

    Args:
      name: An optional argument to use as prefix of the names in the registry,
        if None, the _default_name attribute may be used (it can be set in a
        child class declaration).
    """


def truncate_reply(reply: str, stop_sequences: list[str]) -> str:
  """Returns the truncated reply not including any stop sequence.

  Example:
    If reply is 'abcdefg' and stop_sequences is ['f', 'c'], then the truncated
    reply is 'ab'.

  Args:
    reply: The original reply to be truncated.
    stop_sequences: The substrings to truncate the reply at. The stop sequences
      are not included in the truncated reply.
  """
  if not stop_sequences:
    return reply

  # Some stop sequences may be overlapping, and we want to guarantee that we
  # have the shortest possible prefix after truncation.
  # So if we have a reply 'abcdefg' and use as stop sequences ['f', 'def']
  # we want to return 'abc'. This means we have to compute separately the
  # truncation for each stop sequence and pick the smallest obtained prefix.
  truncated_replies = set()
  for stop_sequence in stop_sequences:
    truncated_replies.add(reply.split(stop_sequence)[0])

  # min will sort alphabetically, which, since we only have prefixes of a common
  # string, will result in returning the shortest one.
  return min(truncated_replies)
