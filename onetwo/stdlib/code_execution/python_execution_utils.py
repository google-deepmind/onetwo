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

"""Helper methods, to be used by implementations."""

from __future__ import annotations

import ast
import datetime
import importlib
import json
import logging
import textwrap
from typing import Any, Callable, TypeAlias

from onetwo.stdlib.code_execution import python_execution

# Aliases
_ExecutionStatus: TypeAlias = python_execution.ExecutionStatus
_SandboxStatus: TypeAlias = python_execution.SandboxStatus
_SandboxResult: TypeAlias = python_execution.SandboxResult
_SandboxResultTiming: TypeAlias = python_execution.SandboxResultTiming

# Constants for hook communication
DATA_KEY: str = 'data'
EXCEPTION_KEY: str = 'exception'
EX_ARGS_KEY: str = 'args'
EX_CLASS_KEY: str = 'exception_class'
EX_HOOK_NAME_KEY: str = 'hook_name'


def current_timing(
    *, start: datetime.datetime, base: datetime.datetime
) -> _SandboxResultTiming:
  """Returns a timing object based on the current time and given start/base.

  Args:
    start: Time the execution request was sent to the sandbox.
    base: Time of last interaction with the sandbox (i.e., the later of the
      start time, and the time of the last callback to a hook function).
  """
  now = datetime.datetime.now()
  return _SandboxResultTiming(
      since_start=now - start, since_last_interaction=now - base
  )


def adjust_code_to_set_final_expression_value(
    code: str,
    variable_name: str = 'result',
    default_value: Any = None,
) -> str:
  r"""Returns adjusted code that store the final expression value in a variable.

  Ignores trailing comment lines when determining the "final expression value".

  If the provided code is not valid Python code, then will return the original
  code unchanged. (Ideally, the caller should perform validation of the code
  prior to calling this function.)

  Examples (assuming variable_name = 'result'):
  * 'x = 2\nx + 3' => 'x = 2\nresult = x + 3'.
  * 'x = 2\ndel x' => 'x = 2\ndel x\nresult = None'.
  * 'x = 2\n# Comment' => 'result = x = 2\n# Comment'.

  Args:
    code: String containing the original unadjusted Python code.
    variable_name: The variable name into which to store the final expression
      value.
    default_value: Default value to assign to `variable_name` in the case where
      the actual final expression value was undefined.
  """
  dedented_code = textwrap.dedent(code.rstrip())
  lines = dedented_code.split('\n')

  # Special case when `code` is empty or consists of only whitespace.
  if len(lines) == 1 and not lines[0].strip():
    lines = []

  try:
    parse_tree = ast.parse(dedented_code)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.warning(
        'Failed to parse Python code:\n```\n%s\n```',
        dedented_code,
        exc_info=True,
    )
    return code

  result_idx = None
  if parse_tree and parse_tree.body:
    # ast.AST.lineno is 1-indexed. We subtract 1 to make it 0-based.
    last_statement = parse_tree.body[-1]
    # The only Python statements that are compatible with variable assignment
    # (as far as we are aware...) are expressions and assignments, e.g.:
    # * Expression: `2 + 3` ==> `result = 2 + 3`
    # * Assignment: `y = x + 2` ==> `result = y = x + 2`
    # Other Python statements that are incompatible with variable assignment
    # include compound statements (`if`, `while`, `for`, etc.) and various
    # simple non-expression / non-assignment statements (e.g., `assert`, `del`,
    # `import`, `raise`, etc.).
    # For background, see: https://docs.python.org/3/reference/index.html
    if isinstance(last_statement, ast.Assign) or isinstance(
        last_statement, ast.Expr
    ):
      result_idx = last_statement.lineno - 1

  if result_idx is None:
    # In this case, since the last statement does not return a value that can be
    # assigned to the result variable, we just set the result equal to None.
    lines.append(f'{variable_name} = {default_value}')
  else:
    # If the last statement does return a value, then we can simply set the
    # result variable equal to that.
    lines[result_idx] = f'{variable_name} = ' + lines[result_idx]

  return '\n'.join(lines)


def parse_sandbox_result_json(
    result_str: str,
    start_time: datetime.datetime,
    base_time: datetime.datetime,
) -> _SandboxResult | None:
  """Tries to parse a JSON string into a SandboxResult object.

  Args:
    result_str: The string output from the sandbox, expected to be JSON.
    start_time: The time execution started.
    base_time: The time of the last interaction.

  Returns:
    A _SandboxResult if result_str is valid JSON and has the expected
    structure (contains at least 'stdout'), otherwise None.
  """
  timing = current_timing(start=start_time, base=base_time)

  try:
    # Note that this uses the JSON decoder with potential
    # implementation-specific customizations.
    result_dict = json.loads(result_str)
  except json.JSONDecodeError:
    logging.warning('Failed to parse result string:\n```\n%s\n```', result_str)
    return None

  if isinstance(result_dict, dict) and 'stdout' in result_dict:
    # Received a full SandboxResult in dict form, as expected.
    return _SandboxResult(
        final_expression_value=result_dict.get('final_expression_value'),
        stdout=result_dict.get('stdout', ''),
        sandbox_status=_SandboxStatus(
            result_dict.get('sandbox_status', 'AFTER_RUNNING_CODE')
        ),
        execution_status=_ExecutionStatus(
            result_dict.get('execution_status', 'SUCCESS')
        ),
        status_message=result_dict.get('status_message', ''),
        failure_details=python_execution.parse_failure_details(result_dict),
        timing=timing,
    )
  else:
    # JSON was valid, but not the expected SandboxResult structure.
    return None


def create_sandbox_hook_callable(
    hook_name: str, conn: Any
) -> Callable[..., Any]:
  """Creates a callable to be used inside a sandbox for a specific hook.

  This function is intended to be run *inside* the sandboxed process.
  Args:
    hook_name: The name of the hook.
    conn: The connection object to the main process.

  Returns:
    A wrapper function that communicates with the main process.
  """

  def hook_wrapper(*args, **kwargs):
    message = {'hook': hook_name, 'args': args, 'kwargs': kwargs}
    conn.send(json.dumps(message))
    result_str = conn.recv()
    result = json.loads(result_str)

    ex = result.get(EXCEPTION_KEY)
    if ex:
      error_class_name = ex.get(EX_CLASS_KEY, 'RuntimeError')
      error_args = ex.get(EX_ARGS_KEY, ())
      try:
        error_module = importlib.import_module('builtins')
        error_class = getattr(error_module, error_class_name)
      except AttributeError:
        error_class = RuntimeError
      error = error_class(*error_args)
      setattr(error, '_hook_name', ex.get(EX_HOOK_NAME_KEY))
      raise error
    return result.get(DATA_KEY)

  return hook_wrapper
