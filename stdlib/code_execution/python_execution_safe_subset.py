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

"""Library for execution of a safe subset of Python arithmetic expressions."""

import ast
from typing import Any


# TODO: Prepare a minimal implementation of the `PythonSandbox`
# interface that adopts the approach shown here of direct execution of a safe
# subset of Python, but extended to include a few additional safe features,
# such as multiple lines of code, reading/writing of variables, and a simulated
# `print` statement. This implementation would not support `imports`, `hooks`,
# or `hook_objects`, but could still satisfy the other aspects of the
# `PythonSandbox` interface, such as a `run` method that returns a
# `SandboxResult`, and an `is_stateful` method (that returns `False`).
def arithmetic_eval(expression: str) -> Any:
  """Returns the result of evaluating a Python-style arithmetic expression.

  Args:
    expression: A Python expression consisting purely of numbers and basic
      arithmetic operators. E.g., `(2 + 3) * 4`.
  """
  def _evaluate(node: ast.AST) -> Any:
    """Returns the result of recursively evaluating the given AST node."""
    if isinstance(node, ast.Constant):
      return node.value

    elif isinstance(node, ast.UnaryOp):
      if isinstance(node.op, ast.USub):
        return 0 - _evaluate(node.operand)
      elif isinstance(node.op, ast.UAdd):
        return _evaluate(node.operand)

    elif isinstance(node, ast.BinOp):
      if isinstance(node.op, ast.Add):
        return _evaluate(node.left) + _evaluate(node.right)
      elif isinstance(node.op, ast.Sub):
        return _evaluate(node.left) - _evaluate(node.right)
      elif isinstance(node.op, ast.Mult):
        return _evaluate(node.left) * _evaluate(node.right)
      elif isinstance(node.op, ast.Div):
        return _evaluate(node.left) / _evaluate(node.right)
      elif isinstance(node.op, ast.FloorDiv):
        return _evaluate(node.left) // _evaluate(node.right)
      elif isinstance(node.op, ast.Pow):
        return _evaluate(node.left) ** _evaluate(node.right)

    raise SyntaxError('Malformed arithmetic expression %r' % expression)

  node = ast.parse(expression.strip(), mode='eval').body
  return _evaluate(node)
