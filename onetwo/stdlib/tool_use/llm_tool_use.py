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

"""Library for use of tools via LLM prompting."""

import ast
from collections.abc import Callable, Mapping
import dataclasses
import enum
import itertools
import json
import re
from typing import Any

from onetwo.core import routing
from onetwo.core import utils
import yaml


@enum.unique
class ArgumentFormat(enum.Enum):
  """Format for specifying the arguments of a tool invocation."""

  # Single line python format.
  # E.g.
  # `FlightSearch(origin="ZRH", destination="NRT")`
  # or
  # FlightSearch(origin="ZRH", destination="NRT")
  PYTHON = 'python_args_syntax'

  # YAML format.
  # E.g.
  # ```yaml
  # FlightSearch:
  #   origin: ZRH
  #   destination: NRT
  # ```
  YAML = 'yaml_args_syntax'

  # YAML format suitable for code values.
  # E.g.
  # ```yaml
  # Python:
  #   code: |
  #     def fn():
  #       return 0
  # ```
  YAML_CODE = 'yaml_code_args_syntax'

  # JSON format.
  # E.g.
  # ```json
  # {
  #   "FlightSearch": {
  #   "origin": "ZRH",
  #   "destination": "NRT"
  #   }
  # }
  # ```
  JSON = 'json_args_syntax'

  # Single-line JSON format.
  # E.g.
  # `{ "FlightSearch": { "origin": "ZRH", "destination": "NRT" } }`
  # or
  # { "FlightSearch": { "origin": "ZRH", "destination": "NRT" } }
  JSON_SINGLE = 'json_single_args_syntax'

  # Markdown-like format.
  # E.g.
  # ```tool_code
  # def fn():
  #   return 0
  # ```
  MARKDOWN = 'markdown_args_syntax'


# Special field name for arguments to a function that are not named but
# provided positionally.
VAR_POS = '_VAR_POS'


def _render_call_content(
    fmt: ArgumentFormat | None, function_name: str, *args, **kwargs
) -> str:
  """Renders the call of a function in the desired format."""
  if fmt == ArgumentFormat.YAML:
    if args:
      kwargs[VAR_POS] = list(args)
    return yaml.dump({function_name: kwargs})
  elif fmt == ArgumentFormat.YAML_CODE:
    if args:
      kwargs[VAR_POS] = list(args)

    # We override the default representer for strings so that strings
    # that span multiple lines are rendered over multiple lines.
    # This means the LLM sees the newlines as actual newline tokens instead
    # of \n or such.
    def default_representer(dumper, data):
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    def multiline_representer(dumper, data):
      if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
      else:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, multiline_representer)
    result = yaml.dump({function_name: kwargs})
    yaml.add_representer(str, default_representer)
    return result
  elif fmt == ArgumentFormat.MARKDOWN:
    # if args has multiple values, we just concatenate them across multiple
    # lines.
    if not args:
      raise ValueError('args should not be empty for markdown format')
    if kwargs:
      raise ValueError('kwargs are not supported for markdown format')
    if len(args) > 1:
      raise ValueError('multiple args are not supported for markdown format')
    return f'{function_name}\n{args[0]}\n'
  elif fmt in [ArgumentFormat.JSON, ArgumentFormat.JSON_SINGLE]:
    if args:
      kwargs[VAR_POS] = list(args)
    if fmt == ArgumentFormat.JSON_SINGLE:
      return json.dumps({function_name: kwargs})
    else:
      return json.dumps({function_name: kwargs}, indent=2)
  else:
    # By default we use the Python format.
    rendered_args = ', '.join(
        itertools.chain(
            (repr(v) for v in args),
            (f'{k}={repr(v)}' for k, v in kwargs.items()),
        )
    )
    return f'{function_name}({rendered_args})'


def _render_response_content(fmt: ArgumentFormat | None, value: Any) -> str:
  """Renders the call of a function in the desired format."""
  if fmt == ArgumentFormat.YAML:
    return yaml.dump(value)
  elif fmt == ArgumentFormat.YAML_CODE:
    return yaml.dump(value, default_style='|')
  elif fmt in [ArgumentFormat.JSON, ArgumentFormat.JSON_SINGLE]:
    if fmt == ArgumentFormat.JSON_SINGLE:
      return json.dumps(value)
    else:
      return json.dumps(value, indent=2)
  elif fmt == ArgumentFormat.MARKDOWN:
    return f'tool_outputs\n{value}'
  else:
    # By default we use the raw repr of the value.
    return repr(value)


def _parse_assign_targets(targets: list[ast.Expr], text: str) -> None:
  """Returns assign variables (targets) of python expression."""
  assign_targets = []
  if len(targets) != 1:
    raise ValueError(
        f'Assign targets in {text} should be a single object (Name or'
        f' Tuple). Is: {targets}'
    )
  if isinstance(targets[0], ast.Name):
    assign_targets.append(targets[0].id)
  elif isinstance(targets[0], ast.Tuple):
    for target in targets[0].elts:
      if not isinstance(target, ast.Name):
        raise ValueError(
            f'Assign target Tuple of {text} should consist of Name objects.'
            f' Does not hold for {ast.dump(target)}'
        )
      assign_targets.append(target.id)
  else:
    raise ValueError(
        f'Assign targets in {text} should be either a Name or Tuple.'
        f' Is: {targets[0]}'
    )
  return assign_targets


def _parse_call_content(
    fmt: ArgumentFormat, text: str, context_vars: Mapping[str, Any]
) -> tuple[list[str], str, tuple[Any, ...], dict[str, Any]]:
  """Parses a function call in the desired format."""
  if fmt == ArgumentFormat.PYTHON:
    try:
      parse_tree = ast.parse(text)
    except SyntaxError as e:
      raise ValueError(f'Invalid syntax for call: {text}') from e
    if len(parse_tree.body) != 1:
      raise ValueError(
          'Python function call should be parsed into a single object (expected'
          f' length 1, but was {len(parse_tree.body)}): {text}'
      )

    # Extract assign_targets.
    if isinstance(parse_tree.body[0], ast.Assign):
      assign_targets = _parse_assign_targets(parse_tree.body[0].targets, text)
    elif isinstance(parse_tree.body[0], ast.Expr):
      assign_targets = []
    else:
      raise ValueError(
          'Expected Python function call to be parsed as type `ast.Expr`, or '
          f'as `ast.Assign`, instead was {type(parse_tree.body[0])}: {text}'
      )

    # Process the call_node from here.
    call_node = parse_tree.body[0].value
    if not isinstance(call_node, ast.Call):
      raise ValueError(
          'Expected Python function call to be parsed as type `ast.Call`, but'
          f' instead was {type(call_node)}: {text}'
      )
    args = tuple()
    for arg in call_node.args:
      # If arg is a Name this represents a variable in
      # context.context_variables.
      if isinstance(arg, ast.Name):
        if arg.id not in context_vars:
          raise ValueError(
              f'Positional argument {arg.id} of {text} references a variable'
              ' which does not exist in context variables of the prompt'
              ' template.'
          )
        args += (context_vars[arg.id],)
        continue
      # Otherwise the arg should be a constant.
      try:
        args += (ast.literal_eval(arg),)
      except ValueError as e:
        raise ValueError(
            f'Invalid syntax for positional arguments of {text}, they should be'
            f' constants, not expressions. Argument: {ast.dump(arg)}'
        ) from e
    kwargs = {
        arg.arg: ast.literal_eval(arg.value) for arg in call_node.keywords
    }
    return assign_targets, call_node.func.id, args, kwargs
  elif fmt in [ArgumentFormat.YAML, ArgumentFormat.YAML_CODE]:
    # TODO: Figure out how to extend assign statements to YAML.
    contents = yaml.safe_load(text)
    fn = list(contents.keys())[0]
    args = contents[fn].get(VAR_POS, [])
    contents[fn].pop(VAR_POS, None)
    return ([], fn, tuple(args), contents[fn])
  elif fmt in [ArgumentFormat.JSON, ArgumentFormat.JSON_SINGLE]:
    # TODO: Figure out how to extend assign statements to JSON.
    contents = json.loads(text)
    fn = list(contents.keys())[0]
    args = contents[fn].get(VAR_POS, [])
    contents[fn].pop(VAR_POS, None)
    return ([], fn, tuple(args), contents[fn])
  elif fmt == ArgumentFormat.MARKDOWN:
    # This is a string of the form "tool_name\ncode".
    delimiter = text.find('\n')
    fn = text[:delimiter]
    contents = text[delimiter + 1 :].rstrip()
    return ([], fn, (contents,), dict())
  else:
    raise ValueError(f'Unknown format {fmt.value}')


def _parse_response_content(fmt: ArgumentFormat, text: str) -> Any:
  """Parses a response in the desired format."""
  if fmt == ArgumentFormat.PYTHON:
    try:
      parse_tree = ast.parse(text)
    except SyntaxError as e:
      raise ValueError(f'Invalid syntax for Python args: {text}') from e
    if len(parse_tree.body) != 1:
      raise ValueError(
          'Python function call should be parsed into a single object (expected'
          f' length 1, but was {len(parse_tree.body)}):'
          f' {text}'
      )
    expr_node = parse_tree.body[0]
    if not isinstance(expr_node, ast.Expr):
      raise ValueError(
          'Expected Python function call to be parsed as type `ast.Expr`, but'
          f' instead was {type(expr_node)}: {text}'
      )
    return ast.literal_eval(expr_node.value)
  elif fmt in [ArgumentFormat.YAML, ArgumentFormat.YAML_CODE]:
    return yaml.safe_load(text)
  elif fmt in [ArgumentFormat.JSON, ArgumentFormat.JSON_SINGLE]:
    return json.loads(text)
  else:
    raise ValueError(f'Unknown format {fmt.value}')


def _parse_and_consume(
    text: str,
    parsing_function: Callable[..., Any],
    context_vars: Mapping[str, Any],
) -> tuple[Any, ArgumentFormat, int]:
  """Determines the format and parses a function call or response.

  Args:
    text: The string to be parsed, which is assumed to start with a function
      call in one of the supported formats.
    parsing_function: Function that will do the parsing.
    context_vars: Mutable dictionary inside prompt template context.

  Returns:
    A tuple (parsed_value, format, index).
    The format is determined based on the presence of backticks.
    The index indicates where in the input string did the parsing stop.
  """
  # We assume first that we consumed the whole text, but will update the index
  # if we end up consuming less that than.
  index = len(text)
  stripped = text.lstrip()
  if stripped.startswith('```'):
    if stripped.startswith('```yaml'):
      stripped = stripped[7:].lstrip()
      fmt = ArgumentFormat.YAML
    elif stripped.startswith('```json'):
      stripped = stripped[7:].lstrip()
      fmt = ArgumentFormat.JSON
    else:
      # Match the string to ```${function_name}.
      pattern = r'```(\w+)\n'
      function_name = re.search(pattern, stripped)
      if function_name is not None:
        stripped = stripped[3:]
        fmt = ArgumentFormat.MARKDOWN
      else:
        raise ValueError(f'Could not detect the format of {stripped}')
    expected_end = '```'
  elif stripped.startswith('`'):
    # Move past the backtick.
    stripped = stripped[1:]
    if stripped.startswith('{'):
      fmt = ArgumentFormat.JSON_SINGLE
    else:
      fmt = ArgumentFormat.PYTHON
    expected_end = '`'
  else:
    # By default, we try and use the PYTHON format, assuming the end point is
    # a newline.
    fmt = ArgumentFormat.PYTHON
    expected_end = '\n'

  # We check whether there is an end triple backtick.
  end = stripped.find(expected_end)
  if end > 0:
    inside = stripped.split(expected_end)[0]
    # We update the index (shifting it by what has potentially been removed
    # from the original text).
    index = end + index - len(stripped)
    # Shifting the index past the expected end string.
    index += len(expected_end)
  else:
    inside = stripped

  return parsing_function(fmt, inside, context_vars), fmt, index


def parse_and_consume_call(
    text: str,
    context_vars: Mapping[str, Any],
) -> tuple[
    list[str], str, tuple[Any, ...], dict[str, Any], ArgumentFormat, int
]:
  """Determines the format and parses a function call.

  Args:
    text: Text to be parsed.
    context_vars: Mutable dictionary inside prompt template context.

  Returns:
    A tuple consisting of:
      - the assignment targets of the function call
      - the name of the function being called
      - the list of positional arguments
      - a dict of keyword arguments
      - the format that has been inferred
      - an index of where in the text the call ends
  """
  (targets, fn, args, kwargs), fmt, index = _parse_and_consume(
      text, _parse_call_content, context_vars
  )
  return targets, fn, args, kwargs, fmt, index


def _render(fmt: ArgumentFormat | None, contents: str) -> str:
  """Renders a string in the desired format with backticks delimiters.

  Args:
    fmt: Format to use.
    contents: The contents to be wrapped in appropriate delimiters.

  Returns:
    The wrapped text.
  """
  if fmt in [ArgumentFormat.PYTHON, ArgumentFormat.JSON_SINGLE]:
    return f'`{contents}`'
  elif fmt in [ArgumentFormat.YAML, ArgumentFormat.YAML_CODE]:
    return f'```yaml\n{contents}\n```'
  elif fmt == ArgumentFormat.JSON:
    return f'```json\n{contents}\n```'
  elif fmt == ArgumentFormat.MARKDOWN:
    return f'```{contents}```'
  else:
    # By default we use the Python format without backticks.
    return f'{contents}'


def render_call(
    fmt: ArgumentFormat | None, function_name: str, *args, **kwargs
) -> str:
  """Renders a function call in the desired format with backticks delimiters."""
  contents = _render_call_content(fmt, function_name, *args, **kwargs)
  return _render(fmt, contents)


def render_response(fmt: ArgumentFormat | None, value: Any) -> str:
  """Renders a response in the desired format with backticks delimiters."""
  contents = _render_response_content(fmt, value)
  return _render(fmt, contents)


def render_assignment_response(
    targets: list[str], value: Any, name: str
) -> str:
  """Renders response when output is assigned to variables in context."""
  del value
  return (
      f'I stored the output of {name} in the following variables:'
      f' {", ".join(targets)}.'
  )


@dataclasses.dataclass
class FunctionCall:
  """A single function call (e.g., as may appear in one step of tool usage).

  Attributes:
    function_name: The name of the function to be called. Should match the name
      of a function that is registered in `onetwo.function_registry` (or when
      using a tool handler like `PythonToolUseEnvironment`, the name of a tool
      as per its `Tool.name`).
    args: The arguments of the function.
    kwargs: The keyword arguments of the function.
  """

  function_name: str = ''
  args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def render(self, fmt: ArgumentFormat) -> str:
    return render_call(fmt, self.function_name, *self.args, **self.kwargs)


@dataclasses.dataclass(frozen=True)
class ToolExample:
  """Data structure to represent an example call to a tool.

  Attributes:
    function_call: The tool call.
    response: Expected response from the tool.
    rendering_format: Format to use to render the example.
  """

  function_call: FunctionCall = dataclasses.field(default_factory=FunctionCall)
  response: Any = None
  rendering_format: ArgumentFormat | None = None

  def render(self) -> str:
    rendered_call = self.function_call.render(self.rendering_format)
    rendered_response = render_response(self.rendering_format, self.response)
    return f'{rendered_call}\nwill return: {rendered_response}'


@dataclasses.dataclass
class Tool:
  """A callable that can be invoked as a tool by an LLM.

  Attributes:
    name: Name used by the LLM to call the tool.
    function: Function to be called upon executing this tool. If not specified,
      then we will fall back to look for a function of name `name_in_registry`
      in the function registry.
    description: String containing a human-readable description of the tool.
    example: String containing a human-readable example of the tool usage.
    color: Optional color name for printing the output from this tool.
    irrecoverable_error_types: Optional list of error types that are
      irrecoverable. This field is used to determine whether to terminate the
      agent when there are certain types of errors thrown by the tool execution.
      Some of the common error types can be KeyError, PermissionError,
      RuntimeError, etc. Configuring the error types per tool helps to determine
      whether to fail early and terminate the code execution or not. If this is
      empty, then the LLM will retry until reaching the maximum number of steps
      configured in the agent.
  """

  name: str
  function: Callable[..., Any] | None = None
  description: str | None = None
  example: str | ToolExample | None = None
  color: str | None = None
  irrecoverable_error_types: list[str] | None = None

  @property
  def example_str(self) -> str:
    """Returns the example formatted appropriately for insertion in a prompt."""
    if isinstance(self.example, str):
      return self.example
    elif isinstance(self.example, ToolExample):
      return self.example.render()
    else:
      return repr(self.example)

  async def __call__(self, *args, **kwargs):
    """Calls the tool on the given args."""
    tool_function = self.function or routing.function_registry.get(self.name)
    if tool_function is None:
      raise ValueError(
          f'Tool "{self.name}" neither provides a tool function directly, nor'
          f' is function "{self.name}" registered in the  function_registry '
          f'({routing.function_registry=}).'
      )
    return await utils.call_and_maybe_await(tool_function, *args, **kwargs)
