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

"""Definitions of built-in functions and utilities for tool use.

A tool can be an arbitrary function that we want to expose to the LLM so that
it can call it.
"""

import inspect
from typing import Any

from onetwo.builtins import builtins_base
from onetwo.core import constants
from onetwo.core import executing
from onetwo.core import routing


@builtins_base.Builtin
def run_tool(
    tool_name: str, tool_args: tuple[Any, ...], tool_kwargs: dict[str, Any]
) -> Any:
  """Interface of the run_tool built-in function.

  Runs a tool and returns the result.

  Args:
    tool_name: The name of the tool to run. Depending on the implementation,
      this could either be the name of a function in the function registry, or
      could be another tool name that is indirectly mapped to such a function.
    tool_args: Position args to pass to the tool function.
    tool_kwargs: Keyword args to pass to the tool function.

  Returns:
    The return value of the tool function.
  """
  del tool_name, tool_args, tool_kwargs
  raise NotImplementedError(
      'The implementation should be provided at runtime by calling `configure`'
      ' or `get_variant`. This function cannot be called directly.'
  )


@executing.make_executable
async def default_run_tool(
    tool_name: str, tool_args: tuple[Any, ...], tool_kwargs: dict[str, Any]
) -> Any:
  """Default implementation of run_tool which calls function_registry."""
  if tool_name == constants.ERROR_STRING:
    # ERROR_STRING as the tool_name is a special case, where we are expected
    # to simply echo the error message stored in the tool argument.
    if len(tool_args) != 1:
      raise ValueError(
          'When tool_name is ERROR_STRING, we expect there to be exactly one'
          ' argument containing the detailed error message (e.g., an error'
          ' that occurred when parsing the LLM response to determine the'
          f' tool call). Instead found {len(tool_args)} arguments:'
          f' {tool_name=},  {tool_args=},  {tool_kwargs=}'
      )
    return tool_args[0]

  try:
    tool_function = routing.function_registry.get(tool_name)
    if tool_function is None:
      raise ValueError(
          f'Function {tool_name} is not registered in the function_registry'
          f' ({routing.function_registry=}).'
      )
    if inspect.iscoroutinefunction(tool_function):
      value = await tool_function(*tool_args, **tool_kwargs)
    else:
      value = tool_function(*tool_args, **tool_kwargs)
    if isinstance(value, executing.Executable):
      return await value
    return value

  except ValueError as e:
    return f'{constants.ERROR_STRING}: {e}'


def reset_defaults():
  """Resets default implementations for all builtins in this file."""
  # Keep all module level `some_builtin.configure(...)` commands in this method.
  run_tool.configure(default_run_tool)

reset_defaults()
# TODO: Define an additional `run_tool_text` builtin that takes as
# input:
# (1) a string like `f('a', 'b')` or `f(x, 'b')` or `y = f(x, 'b')` that can be
#     parsed into a tool call (possibly with variable references and/or variable
#     assignment)
# (2) a context represented as a dictionary of variable values
#     (e.g., {'x': 'a'})
# and which returns as output a string containing the tool's return value or
# error message in a form appropriate for presenting to an LLM in a prompt,
# along with a dictionary of context updates in the case of a variable
# assignment (e.g., {'y': 'ab'})). This `run_tool_text` builtin can further be
# wrapped as a composable or as a callback in a Jinja prompt template.
