# Copyright 2026 DeepMind Technologies Limited.
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

"""Core data types for function calling.

These types describe *what the model asked for* and *what the tools returned*.
They are deliberately free of any rendering, parsing, execution or transport
concerns, and have no dependencies outside the standard library, so that they
can be shared by:

  * the text-parsing tool-use strategies in `onetwo.stdlib.tool_use`, which
    recover a `FunctionCall` from prose emitted by the model, and
  * the model-native function-calling builtins in `onetwo.builtins`, which
    receive a `FunctionCall` as structured data from the backend.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any


@dataclasses.dataclass(eq=False)
class FunctionCall:
  """A single function call (e.g., as may appear in one step of tool usage).

  Attributes:
    function_name: The name of the function to be called. Should match the name
      of a function that is registered in `onetwo.function_registry` (or when
      using a tool handler like `PythonToolUseEnvironment`, the name of a tool
      as per its `Tool.name`).
    args: The arguments of the function. Model-native function calling APIs
      express all arguments as keyword arguments, so this is typically empty for
      calls that originate from a backend, and is populated only by the
      text-parsing strategies, which can recover positional arguments.
    kwargs: The keyword arguments of the function.
  """

  function_name: str = ''
  args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # TODO: Delete this method, and the `eq=False` above, once the
  # deprecated `onetwo.stdlib.tool_use.llm_tool_use.FunctionCall` subclass has
  # been removed. Both exist only so that instances of that subclass compare
  # equal to instances of this class while the two coexist; afterwards the
  # dataclass-generated `__eq__` is sufficient and this class becomes a plain
  # `@dataclasses.dataclass`, as `FunctionResponse` below already is.
  def __eq__(self, other: Any) -> bool:
    """Returns True if `other` is a `FunctionCall` with equal field values."""
    # Compared field-wise rather than with the dataclass-generated `__eq__`,
    # which additionally requires `type(self) is type(other)`. Subclasses of
    # `FunctionCall` therefore compare equal to a plain `FunctionCall` carrying
    # the same values, which is what callers expect of a value type.
    if not isinstance(other, FunctionCall):
      return NotImplemented
    return (self.function_name, self.args, self.kwargs) == (
        other.function_name,
        other.args,
        other.kwargs,
    )


@dataclasses.dataclass
class FunctionResponse:
  """The outcome of executing a single `FunctionCall`.

  Attributes:
    function_name: The name of the function that was called, matching
      `FunctionCall.function_name`.
    response: The value returned by the function. `None` when `error` is set.
    error: A description of why the call failed, or `None` if it succeeded.
      Failures are represented rather than raised so that they can be reported
      back to the model, giving it the opportunity to correct the call.
  """

  function_name: str = ''
  response: Any = None
  error: str | None = None
