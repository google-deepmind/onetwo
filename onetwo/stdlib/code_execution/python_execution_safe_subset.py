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

"""Library for execution of a safe subset of Python arithmetic expressions."""

import ast
from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import io
import logging
import pprint
from typing import Any

from onetwo.core import executing
from onetwo.core import utils
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_multiprocess


def get_default_safe_callables() -> dict[str, Callable[..., Any]]:
  """Returns the default set of callables considered safe for safe_eval."""
  # Standard functions we always want to support.
  return {
      # Standalone functions.
      'bool': bool,
      'dict': dict,
      'int': int,
      'float': float,
      'len': len,
      'list': list,
      'print': print,
      'range': range,
      'set': set,
      'str': str,
      # Methods.
      'dict.clear': dict.clear,
      'dict.copy': dict.copy,
      'dict.fromkeys': dict.fromkeys,
      'dict.get': dict.get,
      'dict.items': dict.items,
      'dict.keys': dict.keys,
      'dict.pop': dict.pop,
      'dict.popitem': dict.popitem,
      'dict.setdefault': dict.setdefault,
      'dict.update': dict.update,
      'dict.values': dict.values,
      'list.append': list.append,
      'list.clear': list.clear,
      'list.copy': list.copy,
      'list.count': list.count,
      'list.extend': list.extend,
      'list.index': list.index,
      'list.insert': list.insert,
      'list.pop': list.pop,
      'list.remove': list.remove,
      'list.reverse': list.reverse,
      'list.sort': list.sort,
  }


def _has_star_args(node: ast.Call) -> bool:
  """Returns True if the callable node has *args or **kwargs."""
  return any(isinstance(arg, ast.Starred) for arg in node.args) or any(
      kw.arg is None for kw in node.keywords
  )


def _get_dotted_name(node: ast.AST) -> str | None:
  """Get the dotted name string from the given AST node."""
  if isinstance(node, ast.Name):
    return node.id
  elif isinstance(node, ast.Constant):
    # True/False/None.
    return str(node.value)
  elif isinstance(node, ast.Attribute):
    lhs = _get_dotted_name(node.value)
    return lhs + '.' + node.attr
  else:
    return None


def _primitive_compare(op: ast.AST, left_value: Any, right_value: Any) -> bool:
  """Returns True if the comparison holds."""
  if isinstance(op, ast.Eq):
    return left_value == right_value
  elif isinstance(op, ast.NotEq):
    return left_value != right_value
  elif isinstance(op, ast.Lt):
    return left_value < right_value
  elif isinstance(op, ast.LtE):
    return left_value <= right_value
  elif isinstance(op, ast.Gt):
    return left_value > right_value
  elif isinstance(op, ast.GtE):
    return left_value >= right_value

  raise SyntaxError('Unsupported comparison operation: %r' % op)


def _get_attribute_list(node: ast.Attribute) -> tuple[ast.Name, Sequence[str]]:
  """Get the list of flat attribute lookup names.

  Special attributes (starting with double underscore) are prohibited.
  Can't chain attribute evaluation and function call, e.g. `obj.func().attr`.

  Args:
    node: ast node to be parsed.

  Returns:
    Tuple(ast.Name node at the top of attribute chain, list of attribute names).

  Raises:
    SyntaxError: Occurs if the expression is incorrect or a prohibited attribute
      name is used.
  """
  assert isinstance(node, ast.Attribute)
  attrs = []
  while isinstance(node, ast.Attribute):
    if node.attr.startswith('__'):
      raise SyntaxError('Prohibited attribute %r' % node.attr)
    attrs.append(node.attr)
    node = node.value

  if not isinstance(node, ast.Name):
    # We would come here, for example, in a construction like `obj.func().attr`.
    raise SyntaxError('Restricted support for getattr: %r' % node)

  attrs.reverse()
  return node, attrs


def _get_attribute(node: ast.Attribute, context: dict[str, Any]) -> Any:
  """Returns the value of an attribute (e.g. `obj.field`).

  Attributes of attributes, e.g. `obj.field.attr` are allowed.
  Special attributes (starting with double underscore) are prohibited.
  Can't chain attribute evaluation and function call, e.g. `obj.func().attr`.

  Args:
    node: Node to be parsed.
    context: Mapping of variable name to value. Referenced objects will be
      looked up by name in this mapping.

  Returns:
    Value of the node (typing.Any)

  Raises:
    SyntaxError: Occurs if the expression is incorrect or a prohibited attribute
      name is used.
  """
  node, attrs = _get_attribute_list(node)
  result = context[node.id]
  for attr in attrs:
    result = getattr(result, attr)
  return result


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

    if isinstance(node, ast.UnaryOp):
      if isinstance(node.op, ast.USub):
        return 0 - _evaluate(node.operand)
      elif isinstance(node.op, ast.UAdd):
        return _evaluate(node.operand)

    if isinstance(node, ast.BinOp):
      if isinstance(node.op, ast.Add):
        return _evaluate(node.left) + _evaluate(node.right)
      if isinstance(node.op, ast.Sub):
        return _evaluate(node.left) - _evaluate(node.right)
      if isinstance(node.op, ast.Mult):
        return _evaluate(node.left) * _evaluate(node.right)
      if isinstance(node.op, ast.Div):
        return _evaluate(node.left) / _evaluate(node.right)
      if isinstance(node.op, ast.FloorDiv):
        return _evaluate(node.left) // _evaluate(node.right)
      if isinstance(node.op, ast.Mod):
        return _evaluate(node.left) % _evaluate(node.right)
      if isinstance(node.op, ast.Pow):
        return _evaluate(node.left) ** _evaluate(node.right)

    raise SyntaxError('Malformed arithmetic expression %r' % expression)

  # We use parse mode 'eval' to limit to single-line expressions.
  node = ast.parse(expression.strip(), mode='eval').body
  return _evaluate(node)


async def safe_eval(
    code: str,
    *,
    context: dict[str, Any] | None = None,
    allowed_callables: Mapping[str, Callable[..., Any]] | None = None,
    hooks: Mapping[str, Callable[..., Any]] | None = None,
) -> Any:
  """Returns the result of evaluating the given Python code.

  Unlike the native `eval` function, `safe_eval` cannot execute arbitrary Python
  code, but rather just a limited subset that can be evaluated safely.

  Subsumes the functionality of `arithmetic_eval`, and adds support for the
  following additional constructs:
  * Various Python literals (e.g., lists, tuples, dictionaries, etc., in
    addition to the numbers, strings, etc., that are supported by
    `arithmetic_eval`). Roughly the equivalent of `ast.literal_eval`.
  * Boolean expressions (e.g., `and`, `or`, `not`, `==`, `!=` '<`, '>', etc.).
  * References to values of variables from a specified context dictionary.
  * Variable assignment (including writing to the specified context dictionary).
  * Calls to functions (and other callables) from a specified allow list.
  * Sequential execution of multiple lines of code.
  * Conditionals (if statements).

  The following features are **not** supported due to safety concerns:
  * Imports.
  * Calls to functions other than those listed in `allowed_callables`.

  The following features are not supported currently, but may be added later:
  * Loops (for, while, etc.).
  * Writing to an object attribute (e.g., `x.id = 2`).

  Args:
    code: The code to execute.
    context: Optional mapping of variable name to value. If specified, then the
      sandbox will use these value for any such variables that may be referenced
      in `code`. If `code` contains any variable assignments, then the new
      values of these variables will also be stored in `context` for inspection
      by the caller or reuse in later calls to `safe_eval`.
    allowed_callables: Mapping of name to callable, representing the callables
      that are allowed to be called from within `call`. E.g., `{'print': print}`
      will allow the evaluator to call the native 'print'function.
    hooks: Mapping from string to functions for the hooks the sandbox can call.
      These are solely used to track the failures in the sandbox. Even though
      these hooks are already part of _allowed_callables parameter, we cannot
      differentiate between the hooks and the other non-hook callables.

  Raises:
    NameError: If the code references a variable that is not defined in
      `context`.
    SyntaxError: If the code is not valid Python or if it uses a language
      feature that we do not currently support.
  """
  if context is None:
    context = {}
  if allowed_callables is None:
    allowed_callables = {}

  async def _evaluate(node: ast.AST | list[ast.AST]) -> Any:
    """Returns the result of recursively evaluating the given AST node."""
    # Multi-line statements
    if isinstance(node, list):
      # As we execute each line, they can incrementally update `context`.
      result = None
      for sub_node in node:
        result = await _evaluate(sub_node)
      return result

    # Literals
    if isinstance(node, ast.Constant):
      return node.value
    if isinstance(node, ast.Tuple):
      return tuple([await _evaluate(elt) for elt in node.elts])
    if isinstance(node, ast.List):
      return [await _evaluate(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
      return {
          await _evaluate(k): await _evaluate(v)
          for k, v in zip(node.keys, node.values)
      }
    if isinstance(node, ast.Set):
      return {await _evaluate(elt) for elt in node.elts}

    # Unary operations
    if isinstance(node, ast.UnaryOp):
      if isinstance(node.op, ast.USub):
        return 0 - await _evaluate(node.operand)
      if isinstance(node.op, ast.UAdd):
        return await _evaluate(node.operand)
      if isinstance(node.op, ast.Not):
        return not await _evaluate(node.operand)

    # Binary operations
    if isinstance(node, ast.BinOp):
      if isinstance(node.op, ast.Add):
        return await _evaluate(node.left) + await _evaluate(node.right)
      if isinstance(node.op, ast.Sub):
        return await _evaluate(node.left) - await _evaluate(node.right)
      if isinstance(node.op, ast.Mult):
        return await _evaluate(node.left) * await _evaluate(node.right)
      if isinstance(node.op, ast.Div):
        return await _evaluate(node.left) / await _evaluate(node.right)
      if isinstance(node.op, ast.FloorDiv):
        return await _evaluate(node.left) // await _evaluate(node.right)
      if isinstance(node.op, ast.Mod):
        return await _evaluate(node.left) % await _evaluate(node.right)
      if isinstance(node.op, ast.Pow):
        return await _evaluate(node.left) ** await _evaluate(node.right)
      if isinstance(node.op, ast.BitAnd):
        return await _evaluate(node.left) & await _evaluate(node.right)
      if isinstance(node.op, ast.BitOr):
        return await _evaluate(node.left) | await _evaluate(node.right)
      if isinstance(node.op, ast.BitXor):
        return await _evaluate(node.left) ^ await _evaluate(node.right)

    # Boolean operations
    if isinstance(node, ast.BoolOp):
      if isinstance(node.op, ast.And):
        # Async equivalent of `return all(_evaluate(v) for v in node.values)`.
        for v in node.values:
          if not await _evaluate(v):
            return False
        return True
      if isinstance(node.op, ast.Or):
        # Async equivalent of `return any(_evaluate(v) for v in node.values)`.
        for v in node.values:
          if await _evaluate(v):
            return True
        return False

    # Comparison operations
    if isinstance(node, ast.Compare):
      left_value = await _evaluate(node.left)
      for op, comparator in zip(node.ops, node.comparators):
        right_value = await _evaluate(comparator)
        if not _primitive_compare(op, left_value, right_value):
          return False
        left_value = right_value
      return True

    # Variables
    if isinstance(node, ast.Name):
      if node.id not in context:
        raise NameError('name %r is not defined' % node.id)
      return context[node.id]

    # Object attributes
    if isinstance(node, ast.Attribute):
      return _get_attribute(node=node, context=context)

    # F-strings
    if isinstance(node, ast.FormattedValue):
      # See: https://docs.python.org/3/library/ast.html#ast.FormattedValue
      value = await _evaluate(node.value)
      if node.conversion == -1:
        return f'{value}'
      elif node.conversion == 115:
        return f'{value!s}'
      elif node.conversion == 114:
        return f'{value!r}'
      elif node.conversion == 97:
        return f'{value!a}'
      else:
        raise SyntaxError(
            f'Unsupported conversion for ast.FormattedValue: {node.conversion}'
        )
    if isinstance(node, ast.JoinedStr):
      return ''.join([await _evaluate(part) for part in node.values])

    # Conditionals
    if isinstance(node, ast.IfExp) or isinstance(node, ast.If):
      condition = await _evaluate(node.test)
      if condition:
        return await _evaluate(node.body)
      else:
        return await _evaluate(node.orelse)

    # Function calls
    if isinstance(node, ast.Call):
      # The following case supports calls to callable functions, supporting
      # positional and named arguments, but not *args or **kwargs.
      if _has_star_args(node):
        raise SyntaxError(
            'Star arguments not supported by this sandbox'
            f' (type={node.__class__.__name__})): {ast.unparse(node)} (parsed'
            f' as: {pprint.pformat(node)})'
        )
      # In the case where there is a dot in the function name, we first check
      # if allowed_callables contains the full dotted name. This could happen
      # if there is a library name to the left of the dot, e.g., if the code is
      # `math.exp(2)` and we have `allowed_callables = {`math.exp`: math.exp}`.
      func_node = node.func
      callable_name = _get_dotted_name(func_node)
      if callable_name is None:
        raise SyntaxError('Malformed code: %r' % code)
      if callable_name in allowed_callables:
        try:
          return await utils.call_and_maybe_await(
              allowed_callables[callable_name],
              *[await _evaluate(arg) for arg in node.args],
              **{kw.arg: await _evaluate(kw.value) for kw in node.keywords},
          )
        except Exception as e:
          if hooks and callable_name in hooks:
            setattr(e, '_hook_name', callable_name)
            raise e
          raise e
      # If allowed_callables doesn't contain the full dotted name, the other
      # possibility is that it may be referring to a method of an object,
      # e.g., if the code is `x = []\nx.append("a")`, and we have
      # `allowed_callables = {`list.append`: list.append}`.
      if isinstance(func_node, ast.Attribute):
        method_name = func_node.attr
        if not isinstance(method_name, str):
          raise SyntaxError(
              'Expected method name to be a string, but was'
              f' {type(method_name)}: {pprint.pformat(method_name)})'
          )
        object_of_method = await _evaluate(func_node.value)
        object_type_str = object_of_method.__class__.__name__
        dotted_method_name = f'{object_type_str}.{method_name}'
        if dotted_method_name in allowed_callables:
          args = [await _evaluate(arg) for arg in node.args]
          args = [object_of_method] + args
          return await utils.call_and_maybe_await(
              allowed_callables[dotted_method_name],
              *args,
              **{kw.arg: await _evaluate(kw.value) for kw in node.keywords},
          )

        # If we get this far, then we have exhausted all options for resolving
        # the callable.
        raise SyntaxError('Unknown callable: %r' % callable_name)

    # Variable assignments
    if isinstance(node, ast.Assign):
      value = await _evaluate(node.value)
      for target in node.targets:
        if not isinstance(target, ast.Name):
          raise SyntaxError(
              f'Unexpected assignment target type: {type(target)}'
          )
        context[target.id] = value
      return value

    # Augmented assignment (e.g., `+=`, `-=`)
    if isinstance(node, ast.AugAssign):
      if not isinstance(node.target, ast.Name):
        raise SyntaxError(
            f'Unexpected augmented assignment target type: {type(node.target)}'
        )
      target_value = await _evaluate(node.target)
      value = await _evaluate(node.value)
      if isinstance(node.op, ast.Add):
        context[node.target.id] = target_value + value
      elif isinstance(node.op, ast.BitAnd):
        context[node.target.id] = target_value & value
      elif isinstance(node.op, ast.BitOr):
        context[node.target.id] = target_value | value
      elif isinstance(node.op, ast.BitXor):
        context[node.target.id] = target_value ^ value
      elif isinstance(node.op, ast.Div):
        context[node.target.id] = target_value / value
      elif isinstance(node.op, ast.Mod):
        context[node.target.id] = target_value % value
      elif isinstance(node.op, ast.Mult):
        context[node.target.id] = target_value * value
      elif isinstance(node.op, ast.FloorDiv):
        context[node.target.id] = target_value // value
      elif isinstance(node.op, ast.Pow):
        context[node.target.id] = target_value**value
      elif isinstance(node.op, ast.Sub):
        context[node.target.id] = target_value - value
      else:
        raise SyntaxError(
            f'Unexpected augmented assignment operation: {node.op}'
        )
      return None

    # Standalone expressions
    if isinstance(node, ast.Expr):
      return await _evaluate(node.value)

    # raise SyntaxError(f'Code not supported by this sandbox: {code}')
    raise SyntaxError(
        f'Code not supported by this sandbox (type={node.__class__.__name__})):'
        f' {ast.unparse(node)} (parsed as: {pprint.pformat(node)})'
    )

  # We use parse mode 'exec' to allow code blocks with multiple statements.
  node = ast.parse(code.strip(), mode='exec').body
  return await _evaluate(node)


@dataclasses.dataclass
class PythonSandboxSafeSubset(python_execution.PythonSandbox):
  """A Python sandbox based on in-process execution of a safe subset of Python.

  Note that this is not strictly a sandbox in the sense that it does not isolate
  the Python code execution in a separate sandboxed process, but rather seeks to
  achieve a similar type of protection through a different means, specifically
  by limiting the subset of Python programs that can be evaluated, and by
  evaluating the code via a custom implementation, rather than via the native
  `eval` function.

  Limitations:
  * Does not support imports.
  * Does not support timeout.

  For production applications, we recommend using a full-featured PythonSandbox
  implementation based on actual sandboxing technology.

  Attributes:
    context: Mapping of variable name to value, representing the current state
      of the sandbox. The variables in the context can be referenced from code
      run in the sandbox and will be updated automatically in cases where the
      code contains variable assignments.
    hooks: Mapping from string to functions for the hooks the sandbox can call.
    hook_objects: Objects containing modifiable state of the hook functions.
      Takes the form of a mapping of variable name to value. By default, this is
      empty, but users of the sandbox are free to use this as a way to bundle
      together with the sandbox some instances of objects whose life cycle they
      want to have bound together with that of the sandbox, and whose contents
      can be modified by the hook functions. One typical usage pattern would be
      in the case where one of the hooks is a method of an object -- in that
      case, we can store the object itself here. (See usage example in the
      docstring of `PythonSandboxFactory`.)
  """

  context: dict[str, Any] = dataclasses.field(default_factory=dict)
  hooks: Mapping[str, Callable[..., Any]] = dataclasses.field(
      default_factory=dict
  )
  hook_objects: dict[str, Any] = dataclasses.field(default_factory=dict)

  def is_stateful(self) -> bool:
    """See base class (PythonSandbox)."""
    return True

  def _prepare_allowed_callables(self) -> Mapping[str, Callable[..., Any]]:
    """Prepares allowed callables for passing to `safe_eval`."""
    allowed_callables = get_default_safe_callables()
    # Custom hooks.
    allowed_callables.update(self.hooks)
    return allowed_callables

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def run(self, code: str) -> python_execution.SandboxResult:
    """See base class (PythonSandbox)."""
    # Precheck code syntax.
    try:
      compile(code, '<string>', 'exec')
    except SyntaxError as e:
      logging.info('Fast fail on bad syntax: %s \nCode: %s', e, code)
      return python_execution.SandboxResult(
          stdout='',
          sandbox_status=python_execution.SandboxStatus.BEFORE_RUNNING_CODE,
          execution_status=python_execution.ExecutionStatus.COMPILE_ERROR,
          status_message=f'{e.__class__.__name__} - {str(e)}',
      )

    # Run the code.
    with io.StringIO() as out, contextlib.redirect_stdout(out):
      try:
        value = await safe_eval(
            code=code,
            context=self.context,
            allowed_callables=self._prepare_allowed_callables(),
            hooks=self.hooks,
        )
        return python_execution.SandboxResult(
            final_expression_value=value, stdout=out.getvalue()
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        return python_execution.SandboxResult(
            stdout=out.getvalue(),
            execution_status=python_execution.ExecutionStatus.EXECUTION_ERROR,
            status_message=f'{e.__class__.__name__} - {str(e)}',
            failure_details=python_execution.FailureDetails(
                hook_name=getattr(e, '_hook_name', None),
                exception_class=e.__class__.__name__,
                exception_message=str(e),
            ),
        )

  def get_hook_object(self, key: str) -> Any:
    """See base class (PythonSandbox)."""
    return self.hook_objects.get(key)

  async def set_variables(self, **variables: Any) -> None:
    """See base class (PythonSandbox)."""
    self.context.update(variables)

  async def get_variables(self, *names: str) -> Mapping[str, Any]:
    """See base class (PythonSandbox)."""
    return {name: self.context.get(name) for name in names}


@dataclasses.dataclass
class PythonSandboxSafeSubsetFactory(python_execution.PythonSandboxFactory):
  """Factory for creating PythonSandboxSafeSubset instances."""

  def create_sandbox(
      self,
      *,
      timeout: datetime.timedelta = datetime.timedelta(seconds=10),
      imports: Sequence[str] | str = tuple(),
      hooks: Mapping[str, Callable[..., Any]] | None = None,
      hook_objects: Mapping[str, Any] | None = None,
      allow_restarts: bool = False,
  ) -> PythonSandboxSafeSubset:
    r"""Returns a newly-created Python sandbox.

    Overridden from base class (PythonSandboxFactory).

    Args:
      timeout: Amount of time after which `PythonSandbox.run` should time out.
        This feature is not supported in this sandbox implementation and will be
        ignored.
      imports: Additional import statements to run at the start of the sandbox.
        These are not supported in this sandbox implementation and should be
        left blank.
      hooks: Mapping from string to functions for the hooks the sandbox can
        call.
      hook_objects: Objects containing modifiable state of the hook functions.
        Takes the form of a mapping of variable name to value. By default, this
        is empty, but users of the sandbox are free to use this as a way to
        bundle together with the sandbox some instances of objects whose life
        cycle they want to have bound together with that of the sandbox, and
        whose contents can be modified by the hook functions. One typical usage
        pattern would be in the case where one of the hooks is a method of an
        object -- in that case, we can store the object itself here. (See usage
        example in the docstring of `PythonSandboxFactory`.)
      allow_restarts: Whether the sandbox should continue accepting requests
        after a restart. This feature is not supported in this sandbox
        implementation and should be left as False.
    """
    # Timeout is not supported in PythonSandboxSafeSubset. We just ignore it.
    del timeout
    if imports:
      raise ValueError('Imports are not supported in PythonSandboxSafeSubset.')
    if allow_restarts:
      raise ValueError('Restarts are not supported in PythonSandboxSafeSubset.')

    return PythonSandboxSafeSubset(
        hooks=hooks or {},
        hook_objects=hook_objects or {},
    )


@dataclasses.dataclass
class PythonSandboxSafeSubsetMultiProcess(
    python_execution_multiprocess.PythonSandboxMultiProcessWrapper
):
  """A multiprocessing sandbox using PythonSandboxSafeSubset as inner sandbox.

  This class specializes `PythonSandboxMultiProcessWrapper` to use
  `python_execution_safe_subset.PythonSandboxSafeSubset` for code execution
  within the child process. This ensures that only a restricted, safe subset
  of Python can be run, and that different instance of the sandbox will not
  interfere with one another.

  Attributes:
    timeout: Maximum duration for a single `run()` call.
    hooks: Mapping of hook names to functions.
    hook_objects: Objects accessible by hook functions.
    inner_sandbox_factory: Defaults to an instance of
      `PythonSandboxSafeSubsetFactory`.
  """

  inner_sandbox_factory: python_execution.PythonSandboxFactory = (
      dataclasses.field(default_factory=PythonSandboxSafeSubsetFactory)
  )


@dataclasses.dataclass
class PythonSandboxSafeSubsetMultiProcessFactory(
    python_execution.PythonSandboxFactory
):
  """Factory for creating PythonSandboxSafeSubsetMultiProcess instances."""

  def create_sandbox(
      self,
      *,
      timeout: datetime.timedelta = datetime.timedelta(seconds=10),
      imports: Sequence[str] | str = tuple(),
      hooks: Mapping[str, Callable[..., Any]] | None = None,
      hook_objects: Mapping[str, Any] | None = None,
      allow_restarts: bool = False,
  ) -> PythonSandboxSafeSubsetMultiProcess:
    """Returns a newly-created Python sandbox.

    Overridden from base class (PythonSandboxFactory).

    Args:
      timeout: Maximum duration for a single `run()` call before it times out.
      imports: Additional import statements to run at the start of the sandbox.
        These are not supported in this sandbox implementation and should be
        left blank.
      hooks: Mapping from string to functions for the hooks the sandbox can
        call.
      hook_objects: Objects containing modifiable state of the hook functions.
        Takes the form of a mapping of variable name to value. By default, this
        is empty, but users of the sandbox are free to use this as a way to
        bundle together with the sandbox some instances of objects whose life
        cycle they want to have bound together with that of the sandbox, and
        whose contents can be modified by the hook functions. One typical usage
        pattern would be in the case where one of the hooks is a method of an
        object -- in that case, we can store the object itself here. (See usage
        example in the docstring of `PythonSandboxFactory`.)
      allow_restarts: Whether the sandbox should continue accepting requests
        after a restart. This feature is not supported in this sandbox
        implementation and should be left as False.
    """
    if imports:
      logging.warning(
          'Imports are not supported in PythonSandboxSafeSubsetMultiProcess.'
      )
    if allow_restarts:
      logging.warning(
          'Restarts are not supported in PythonSandboxSafeSubsetMultiProcess.'
      )

    return PythonSandboxSafeSubsetMultiProcess(
        timeout=timeout,
        hooks=hooks or {},
        hook_objects=hook_objects or {},
    )
