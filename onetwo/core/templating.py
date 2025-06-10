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

"""Library for executing a prompt template."""

import abc
import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
import copy
import dataclasses
import functools
import inspect
from typing import Any, Final, TypeVar

import dataclasses_json
import immutabledict
import jinja2
from jinja2 import ext as j2_ext
from jinja2 import parser as j2_parser
import jinja2.sandbox
from onetwo.core import constants
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import tracing
from onetwo.core import updating
from onetwo.core import utils



# Special fields for storing information in the Jinja context.
CONTEXT_VARS = constants.CONTEXT_VARS
OUTPUT_VARS: Final[str] = '__outputs__'
INTERNAL_VARS: Final[str] = '__internal__'

# Fields in the final output that will carry specific data.
# The filled-in prefix.
PROMPT_PREFIX = constants.PROMPT_PREFIX
# Rendered prefix with role tags.
PREFIX_WITH_ROLES: Final[str] = 'prefix_with_roles'
# Error message, if an error occurred during processing.
ERROR: Final[str] = 'error'
# Partial reply from the backend as we iterate through it.
ITERABLE_REPLY: Final[str] = '_iterable_reply'
# Specific context variables to handle dry run.
_DRY_RUN_PREFIX_VAR: Final[str] = '_dry_run_prefix'

_PREFIX_ROLE_TAGS_KEY: Final[str] = 'prefix_role_tags_config_registry_key'
_ADD_PREFIX_ROLE_TAGS_KEY: Final[str] = (
    'add_prefix_role_tags_config_registry_key'
)


def create_jinja2_environment(loader=jinja2.BaseLoader()) -> jinja2.Environment:
  """Creates a Jinja2 environment for use by OneTwo templates.

  Args:
    loader: A Jinja2 loader object that can load other templates from files if
      they are included in your prompt using the jinja include directive.

  Returns:
    A Jinja2 environment.
  """
  return jinja2.sandbox.SandboxedEnvironment(
      extensions=[
          'jinja2.ext.do',
          'jinja2.ext.loopcontrols',
          'onetwo.core.templating.SectionExtension',
          'onetwo.core.templating.RoleExtension',
      ],
      enable_async=True,
      undefined=jinja2.StrictUndefined,
      loader=loader,
  )


@dataclasses.dataclass
class _InnerContext:
  """Data used for the template execution (see PromptTemplateContext)."""
  section_stack: list[Any] = dataclasses.field(default_factory=list)
  role_stack: list[Any] = dataclasses.field(default_factory=list)
  role_indices: list[tuple[str, int, bool]] = dataclasses.field(
      default_factory=list
  )
  iterable_reply: Any | None = dataclasses.field(default=None)
  iterated_reply: Any | None = dataclasses.field(default=None)
  prefix_role_tags: dict[str, Sequence[str]] = dataclasses.field(
      default_factory=dict
  )
  add_prefix_role_tags: bool = dataclasses.field(default=False)


class PromptTemplateContext:
  """Represents the context of a PromptTemplate.

  Wraps the jinja2 context and provides methods to access the various
  parts.

  This context is composed of two parts:
  - A user-visible part intended to represent variables that the user has
    provided as inputs to the template (stored in the `input_variables`
    attribute), may
    create and use in the prompt template (stored in the `context_variables`
    attribute), as well as what the user will get as an output of the execution
    (stored in the `output_variables` attribute). There is also a special
    attribute
    called `prefix` which represents the filled-in prefix as execution proceeds
    (and it is also accessible as one of the context_variables for backwards
    compatibility).
  - An "internal" part, which is used by the template execution code to keep
    track of various data. These are the other attributes, which are stored in
    a dataclass (_InnerContext). In principle these should only be accessed by
    the
    JinjaTemplate class and not by any of the user-provided callbacks.
  TODO: To clarify this distinction, a natural next step would be to
  ensure that the user-defined callbacks receive an object that only has the
  visible part.

  Attributes:
    jinja_context: The underlying Jinja2 context object. This is an
      immutabledict with one entry `__vars__` which contains a dict of the
      variables that are set from within the template, e.g. using store or
      section. There is also a `__vars__.prefix` variable containing the prompt
      filled-in prefix so far.
    input_variables: Mapping of the variables that are provided as an input to
      the template.
    context_variables: dict of the variables that are stored in-context during
      the executing of the prompt template.
    output_variables: dict of the variables that will be returned as the output
      of the template executing.
    prefix: The rendered prefix so far.
    error: Error message, if an error occurred during processing.
    section_stack: The stack of sections that are currently open.
    role_stack: The stack of roles that are currently open.
    role_indices: The indices of the start and end of roles.
    iterable_reply: The reply that is being iterated through.
    iterated_reply: The final reply once the iteration is done.
    prefix_role_tags: The role tags to use when rendering the prefix.
    add_prefix_role_tags: Whether to add role tags to the prefix.
  """

  def __init__(self, context: Mapping[str, Any] | None = None):
    if context is None:
      self._context = {
          CONTEXT_VARS: {
              PROMPT_PREFIX: '',
          },
          OUTPUT_VARS: {},
          INTERNAL_VARS: _InnerContext()
      }
    else:
      self._context = dict(context)

  # We disable bad-return-type because we store the data with different types
  # in the jinja dict and the type checker tries to verify their type but
  # does not properly use the dict keys. Same for unsupported-operands.
  # pytype: disable=bad-return-type
  @property
  def input_variables(self) -> dict[str, Any]:
    return self._context

  @property
  def output_variables(self) -> dict[str, Any]:
    return self._context[OUTPUT_VARS]

  # pytype: disable=bad-return-type
  @property
  def context_variables(self) -> dict[str, Any]:
    return self._context[CONTEXT_VARS]

  @property
  def prefix(self) -> str:
    return self._context[CONTEXT_VARS][PROMPT_PREFIX]

  @prefix.setter
  def prefix(self, value: str) -> None:
    self._context[CONTEXT_VARS][PROMPT_PREFIX] = value  # pytype: disable=unsupported-operands

  @property
  def error(self) -> str | None:
    return self._context[CONTEXT_VARS].get(ERROR)  # pytype: disable=attribute-error

  @error.setter
  def error(self, value: str) -> None:
    self._context[CONTEXT_VARS][ERROR] = value  # pytype: disable=unsupported-operands

  @property
  def jinja_context(self) -> immutabledict.immutabledict[str, Any]:
    return immutabledict.immutabledict(self._context)

  @property
  def _internal(self) -> _InnerContext:
    """Accesses the "internal" part of the context.

    This is not supposed to be used directly, just to simplify the other
    accessors.
    """
    return self._context[INTERNAL_VARS]

  @property
  def section_stack(self) -> list[Any]:
    return self._internal.section_stack

  @property
  def role_stack(self) -> list[Any]:
    return self._internal.role_stack

  @property
  def role_indices(self) -> list[tuple[str, int, bool]]:
    return self._internal.role_indices

  @role_indices.setter
  def role_indices(self, value: list[tuple[str, int, bool]]) -> None:
    self._internal.role_indices = value

  @property
  def iterable_reply(self) -> Any:
    return self._internal.iterable_reply

  @iterable_reply.setter
  def iterable_reply(self, value: Any) -> None:
    self._internal.iterable_reply = value

  @property
  def iterated_reply(self) -> Any:
    return self._internal.iterated_reply

  @iterated_reply.setter
  def iterated_reply(self, value: Any) -> None:
    self._internal.iterated_reply = value

  @property
  def prefix_role_tags(self) -> dict[str, Sequence[str]]:
    return self._internal.prefix_role_tags

  @property
  def add_prefix_role_tags(self) -> bool:
    return self._internal.add_prefix_role_tags

  @add_prefix_role_tags.setter
  def add_prefix_role_tags(self, value: bool) -> None:
    self._internal.add_prefix_role_tags = value
  # pytype: enable=bad-return-type


# Generic type for decorators.
T = TypeVar('T')


class BeginEndExtension(j2_ext.Extension, metaclass=abc.ABCMeta):
  """Generic extension to define blocks with begin/end and parameters."""

  # Override this with the name of the start tag.
  tag_name = ''
  # This should contain tag_name and endtag_name.
  tags = {}

  def parse(self, parser: j2_parser.Parser) -> jinja2.nodes.Node:
    """Calls the _start and _end methods when parsing the corresponding tags."""
    context = jinja2.nodes.ContextReference()
    token = parser.stream.next_if(f'name:{self.tag_name}')
    if token is not None:
      lineno = token.lineno
      # We parse the comma-separated parameters of the section.
      args = [context]
      first_parameter = True
      while parser.stream.current.type != 'block_end':
        if not first_parameter:
          parser.stream.expect('comma')
        first_parameter = False
        token = parser.stream.expect('name')
        target = jinja2.nodes.Const(token.value, lineno=token.lineno)
        parser.stream.expect('assign')
        value = parser.parse_expression()
        args.append(target)
        args.append(value)
      result = self.call_method(
          'start', args, lineno=lineno
      )
    else:
      token = parser.stream.expect(f'name:end{self.tag_name}')
      lineno = token.lineno
      result = self.call_method(
          'end', [context], lineno=lineno
      )
    return jinja2.nodes.Output([result], lineno=lineno)

  def update_defaults(self, arguments: dict[str, Any], *args: Any):
    # These defaults are overridden by the user-provided arguments.
    for i in range(0, len(args), 2):
      name = args[i]
      value = args[i + 1]
      if name in arguments:
        arguments[name] = value
      else:
        raise ValueError(f'section got an unknown argument {name}')

  def start(self, context: Mapping[str, Any], *args: Any) -> str:
    """Wrapper of the processing of the start tag."""
    return self._start(PromptTemplateContext(context), *args)

  def end(self, context: Mapping[str, Any]) -> str:
    """Wrapper of the processing of the end tag."""
    return self._end(PromptTemplateContext(context))

  @abc.abstractmethod
  def _start(self, context: PromptTemplateContext, *args: Any) -> str:
    """Processing of the start tag."""

  @abc.abstractmethod
  def _end(self, context: PromptTemplateContext) -> str:
    """Processing of the end tag."""


class SectionExtension(BeginEndExtension):
  """Jinja2 extension to allow defining sections with special behaviour.

  Usage: {% section arguments %} ... {% endsection %}

  Note that for this to work properly, there should be an endsection tag on
  all the paths from an opening section tag. So for example, if you have some
  {% break %} statement in the middle of the section, you may need to add
  another {% endsection %} tag to ensure that the section is closed one way or
  another. The {% endsection %} tag will always close the most recently opened
  section so the binding of ends of sections to the corresponding starts is done
  at runtime.

  Arguments are provided as a comma-separated list of key=value pairs.
  - 'name': Name of the section (for storing its content in context or outputs).
  - 'hidden': If True, the section will be removed from the prefix once it is
    processed.
  - 'to_output': If True, the content of the section is added as an output
    variable (name should not be empty). The default is True.
  - 'to_context': If True, the content of the section is added as a context
    variable, available to the rest of the template as __vars__.name (name
    should not be empty). The default is False.
  """

  tag_name = 'section'
  tags = {'section', 'endsection'}

  # Constant representing a special argument to store the index of the beginning
  # of a section.
  START = '_start'

  def _start(self, context: PromptTemplateContext, *args: Any) -> str:
    """Adds section details to the stack."""
    # Here we specify the default values of the arguments.
    arguments = {
        'name': '',
        'hidden': False,
        'to_output': True,
        'to_context': False,
    }
    # These default are overridden by the user-provided arguments.
    self.update_defaults(arguments, *args)
    # We store the index at which the section start (which is the length
    # of the current prefix).
    arguments[self.START] = len(context.prefix)
    # We put all the data corresponding to this section (its start index
    # and its arguments) on a stack, so that we know about it when we reach
    # the end of the section (i.e. when the _end method is called).
    context.section_stack.append(arguments)
    return ''

  def _end(self, context: PromptTemplateContext) -> str:
    """Updates the context at the end of a section.

    This function is called during the rendering of the template when we reach
    an endsection statement. The prefix is possibly reset (if the section is
    hidden) and the contents of the section is possibly stored into the context
    or the outputs. This is all determined by the arguments of the section
    that were put on the stack when the section started.

    Args:
      context: The template executing context.

    Returns:
      A string to insert in the template, here we return an empty string
      since the endsection token doesn't produce anything by itself.
    """
    # We update the prefix (taking into account whether we passed the end
    # of a hidden section, in which case the prefix is reset to before the
    # beginning of this section.
    if not context.section_stack:
      raise ValueError(
          'Encountered {% endsection %} but no {% section %} block'
          f' started, current prefix: {context.prefix}'
      )
    section_args = context.section_stack.pop()  # pytype: disable=attribute-error
    name = section_args['name']
    prefix = context.prefix
    if name:
      section_content = prefix[section_args[self.START]:]
      if section_args['to_output']:
        context.output_variables[name] = section_content
      if section_args['to_context']:
        context.context_variables[name] = section_content
    if section_args['hidden']:
      # If the section is hidden, we reset the prefix to before the section
      # start and continue appending from there.
      context.prefix = prefix[: section_args[self.START]]
      # We remove all role starts/ends that have been removed due to resetting
      # the prefix.
      prefix_len = len(context.prefix)
      context.role_indices = [
          (name, index, is_start)
          for name, index, is_start in context.role_indices
          if index <= prefix_len
      ]
    return ''


class RoleExtension(BeginEndExtension):
  """Jinja2 extension to allow defining roles.

  Usage: {% role arguments %} ... {% endrole %}

  Note that for this to work properly, there should be an endrole tag on
  all the paths from an opening role tag. So for example, if you have some
  {% break %} statement in the middle of the role, you may need to add
  another {% endrole %} tag to ensure that the role is closed one way or
  another. The {% endrole %} tag will always close the most recently opened
  role so the binding of ends of roles to the corresponding starts is done
  at runtime.

  Arguments are provided as a comma-separated list of key=value pairs.
  - 'name': Name of the role.
  - 'hidden': If True, the whole will be removed from the prefix once it is
    processed. The default is False.
  - 'add_tags': If True, tags for the beginning and end of this role will be
    added to the prefix (and thus will be visible to the LLM). The default is
    False.
  """

  tag_name = 'role'
  tags = {'role', 'endrole'}

  # Constant representing a special argument to store the index of the beginning
  # of a role.
  START = '_start'

  def _start(self, context: PromptTemplateContext, *args: Any) -> str:
    """Adds role details to the stack."""
    # Here we specify the default values of the arguments.
    arguments = {
        'name': '',
        'hidden': False,
        'add_tags': False,
    }
    # These default are overridden by the user-provided arguments.
    self.update_defaults(arguments, *args)
    # We store the index at which the role start (which is the length
    # of the current prefix).
    arguments[self.START] = len(context.prefix)
    # We put all the data corresponding to this role (its start index
    # and its arguments) on a stack, so that we know about it when we reach
    # the end of the role (i.e. when the _end method is called).
    context.role_stack.append(arguments)
    if not arguments['hidden']:
      role_name = arguments['name']
      context.role_indices.append(
          (role_name, arguments[self.START], True)
      )
      if role_name and (
          context.add_prefix_role_tags or arguments['add_tags']
      ):
        tag = f'<{role_name}>'
        tags = context.prefix_role_tags.get(role_name, None)
        if tags:
          tag = tags[0] or ''
        return tag
    return ''

  def _end(self, context: PromptTemplateContext) -> str:
    """Updates the context at the end of a role.

    This function is called during the rendering of the template when we reach
    an endrole statement. The prefix is possibly reset (if the role is
    hidden) and the contents of the role is possibly stored into the context
    or the outputs. This is all determined by the arguments of the role
    that were put on the stack when the role started.

    Args:
      context: The template executing context.

    Returns:
      A string to insert in the template, here we return an empty string
      since the endrole token doesn't produce anything by itself.
    """
    # We update the prefix (taking into account whether we passed the end
    # of a hidden role, in which case the prefix is reset to before the
    # beginning of this role.
    if not context.role_stack:
      raise ValueError(
          'Encountered {% endrole %} but no {% role %} block'
          f' started, current prefix: {context.prefix}'
      )
    role_args = context.role_stack.pop()  # pytype: disable=attribute-error
    name = role_args['name']
    prefix = context.prefix
    prefix_len = len(prefix)
    if role_args['hidden']:
      # If the role is hidden, we reset the prefix to before the role
      # start and continue appending from there.
      context.prefix = prefix[:role_args[self.START]]
      prefix_len = len(context.prefix)
      # We remove all role starts/ends that have been removed due to resetting
      # the prefix.
      context.role_indices = [
          (name, index, is_start)
          for name, index, is_start in context.role_indices
          if index <= prefix_len
      ]
    if name and not role_args['hidden']:
      if context.add_prefix_role_tags or role_args['add_tags']:
        tags = context.prefix_role_tags.get(name, None)
        final_tag = f'</{name}>'
        if tags:
          final_tag = tags[1] or ''
        new_prefix_len = prefix_len + len(final_tag)
      else:
        final_tag = ''
        new_prefix_len = prefix_len
      context.role_indices.append(
          (name, new_prefix_len, False)
      )
    else:
      final_tag = ''
    return final_tag


# Decorator for callbacks.
F = TypeVar('F', bound=Callable[..., Any])


# There is a very subtle issue with registering callbacks that may use self.
# Indeed, if we register them in the initialize() method, they will store the
# reference to self, and when we copy the object (e.g. when using
# `self_consistency.repeat`) and execute a copy, the callback will use the wrong
# object!
# So we have two options:
# 1. We only register callbacks that don't make use of the self reference
#    and use the context for any communication between functions.
# 2. We register the callbacks later (inside _inner_execute or _inner_iterate).
# In order to avoid any issue, we go for the first option and thus make the
# callbacks functions that are defined outside the PromptTemplate class.


def _store_callback(
    context: PromptTemplateContext,
    varname: str,
    content: Any,
    append: bool = False,
    to_context: bool = True,
    to_output: bool = True,
) -> Any:
  """Callback to store content into output and possibly context variables.

  Args:
    context: The jinja context.
    varname: Variable name.
    content: Value to be stored.
    append: If True, the variable is treated as a list and we append the content
      to it.
    to_context: If True, we store the variable into the context.
    to_output: If True, we store the variable into the final output variables.

  Returns:
    A tuple (varname, content, updated_content), where updated_content is the
    list with the latest content appended (if append=True) or the content
    (if append=False).
  """
  if append:
    if varname in context.output_variables:
      updated_content = context.output_variables[varname] + [content]
    elif varname in context.context_variables:
      updated_content = context.context_variables[varname] + [content]
    else:
      updated_content = [content]
  else:
    updated_content = content

  if to_output:
    context.output_variables[varname] = updated_content
  if to_context:
    context.context_variables[varname] = updated_content

  return content


def add_mock_value(function: Callable[..., Any]) -> Callable[..., Any]:
  """Adds a mock_value to a callback. Call the callback when mock_value=None.

  Args:
    function: callback function with context.

  Returns:
    callback function with extra argument 'mock_value'. If mock_value is None,
    the callback function will be called. If mock_value is a string, the mock
    value will be returned directly. Note that using the mock_value will not
    result in a traced stage (in ExecutionResult).
  """

  @functools.wraps(function)
  async def add_mock_value_wrapper(
      context: Mapping[str, Any], *args, mock_value: str | None = None, **kwargs
  ):
    if mock_value is not None:
      return mock_value
    return await function(context, *args, **kwargs)

  return add_mock_value_wrapper


@tracing.trace(name='input', skip=['context'])
async def _input_callback(
    context: PromptTemplateContext,
    variable_name: str | None = None,
    prompt: str | None = None,
) -> str:
  """Callback to ask for user input.

  Args:
    context: Context information passed automatically to the function.
    variable_name: If a variable with this name has been provided as input, it
      will be returned, otherwise the executing will stop to wait for user
      input.
    prompt: The string to display to the user to sollicit input.

  Returns:
    The content provided by the user.
  """
  # Note that we read in the read-only part of the context where the
  # input variables are stored, not in the dynamic context.
  if variable_name in context.input_variables:
    return context.input_variables[variable_name]
  return input(prompt or '')


def html_tags(style: str) -> tuple[str, str]:
  """Convenience function to create HTML span tags."""
  return (f'<span style="{style}">', '</span>')


def _format_with_roles(
    text: str,
    role_indices: Sequence[tuple[str, int, bool]],
    role_tags: Mapping[str, tuple[str | None, str | None]],
    iterable_reply: str | None = None,
) -> str:
  """Inserts formatting tags for the roles into a text.

  For example:
  ```
  _format_with_roles(text='aaa bbb ccc ddd',
                     role_indices=[
                         ('x', 0, True), ('x', 3, False),
                         ('y', 4, True), ('y', 8, False)
                      ],
                      role_tags={
                          'x': ('<x>', '</x>'),
                          'y': (None, None),
                          ITERABLE_REPLY: ('<i>', '</i>'),
                      },
                      iterable_reply='eee')
  ```
  will return the following string:
  ```
  "<x>aaa</x> ccc ddd <i>eee</i>"
  ```

  Args:
    text: Raw text.
    role_indices: Sequence of (role_name, index, is_start) of where the roles
      start and end. The index specifies a character index in the text string,
      is_start is True for the start index and False for the end index.
    role_tags: Mapping from role_name to (start_tag, end_tag). If the tags are
      None, the corresponding section is not displayed. Note that if there is
      no entry for a particular tag name in role_tags, but this tag name appears
      in role_indices, then no specific tags will be added and the content will
      be displayed. Hence, in order to hide a part of the text, one has to
      explicitly add an entry for the role name to hide with (None, None) as a
      value in the role_tags Mapping.
    iterable_reply: If not None, this is a reply that is being streamed from the
      LLM and might be incomplete. We render it with the special role tag
      associated with the role name ITERABLE_REPLY. If there is an
      iterable_reply and the corresponding role tag is not equal to
      (None, None), then we display it with that role tag, and if there is no
      role tag with this name, we directly append it.

  Returns:
    A formatted text with the tags inserted.
  """
  formatted = ''
  last_index = 0
  display = True

  for role_name, index, is_start in role_indices:
    role_tag = role_tags.get(role_name, ('', ''))[0 if is_start else 1]
    if display:
      formatted += text[last_index:index]
    if role_tag is not None:
      formatted += role_tag
    last_index = index
    if role_tag is None:
      # When we have a None role_tag, we toggle the display mode.
      if is_start:
        display = False
      else:
        display = True
  if display:
    formatted += text[last_index:]
  if iterable_reply is not None:
    if ITERABLE_REPLY in role_tags:
      if role_tags[ITERABLE_REPLY][0] is None:
        return formatted  # We don't append the iterable reply.
      formatted += role_tags[ITERABLE_REPLY][0]
      formatted += iterable_reply
      formatted += role_tags[ITERABLE_REPLY][1]
    else:
      formatted += iterable_reply
  return formatted


@dataclasses.dataclass
class JinjaTemplate:
  """A Jinja2 template augmented with additional functionality.

  In particular, we manage the context as a PromptTemplateContext object.

  Attributes:
    name: Name of the prompt template.
    text: Text of the prompt template.
    filename: Path to a text file containing the prompt template.
    loader: A Jinja2 loader object that can load other templates from files if
      they are included in your prompt using the jinja include directive.
    skip_step_on_error: See parent class.
    role_tags: Tags to use to format the various roles. This is a Mapping from a
      role name to a pair of strings, one for the beginning and one for the end
      of the role. For example it could be something like {'user': ('<font
      color="blue">', '</font>')} if we want to add color in an HTML format. If
      the tags are None, the contents will be hidden from the output, but if
      there is no entry for a particular role name, the contents will be
      displayed without specific tags (similarly to having empty string tags
      {'user', ('', '')}).
    prefix_role_tags: Tags to use to format various roles when rendering the
      prefix to be sent to the llm (as opposed to when rendering
      prefix_withroles). Like `role_tags`, this is a mapping from role name to a
      tuple of (start_marker, end_marker). Unlike `role_tags`, these markers
      will be used when add_tags = True is specified for the role in the
      template or when `add_prefix_role_tags` is True for the template.
    add_prefix_role_tags: Sets the default behaviour of whether to add role tags
      to the prefix or not: If True - role tags will be added, if False - tags
      will only be added if add_tags=True is specified in the role block in the
      template.
    template_globals: Globals environment variables as used in jinja2 templates.
      By default adds the python function `zip`.
    move_inputs_to_context_vars: List of variables which are to be moved from
      the immutable `inputs` into the mutable CONTEXT_VARS. This enables passing
      around data objects through CONTEXT_VARS, which is necessary to enable
      multi-modal ReAct chains (see react_chain_j2.py). Defaults to an empty
      list. Only moves variables if they exist in `inputs` and does not throw an
      error if they do not.
  """
  name: str = 'JinjaTemplate'
  text: str | None = None
  filename: str | None = None
  loader: jinja2.BaseLoader | None = None

  role_tags: Mapping[str, tuple[str | None, str | None]] = dataclasses.field(
      default_factory=dict
  )

  prefix_role_tags: Mapping[str, tuple[str | None, str | None]] | None = None

  add_prefix_role_tags: bool | None = None

  template_globals: dict[str, Any] = dataclasses.field(
      default_factory=lambda: {'zip': zip}
  )

  move_inputs_to_context_vars: list[str] = dataclasses.field(
      default_factory=list)

  _callbacks: dict[str, Callable[..., Any]] = dataclasses.field(
      init=False,
      default_factory=dict,
      metadata=dataclasses_json.config(
          exclude=dataclasses_json.Exclude.ALWAYS,
          encoder=lambda _: None,
      ),
  )

  # We save the context variables in case we need to reformat the output
  # after the template has been executed.
  # Note that this is not safe when executing multiple times `render` or
  # `render_stream` in parallel.
  # TODO: Ideally this should be returned at the end of `render`
  # for cases when the user may want to reformat the result afterwards, but
  # not by default (as this would unnecessarily pollute the output). So the
  # task here is to add a flag to render which would return the context
  # variables as an extra output.
  _context: PromptTemplateContext = dataclasses.field(
      init=False,
      default_factory=PromptTemplateContext,
      metadata=dataclasses_json.config(
          exclude=dataclasses_json.Exclude.ALWAYS,
          encoder=lambda _: None,
      ),
  )

  # This is kept purely for backwards compatibility. Do not use.
  @property
  def _context_variables(self) -> Mapping[str, Any]:
    return self._context.context_variables

  def __post_init__(self):
    """See parent class."""
    if self.filename is not None:
      with open(self.filename, 'r') as f:
        self.text = f.read().strip()
    if self.text is None:
      raise ValueError('No prompt template or file provided')

    self._callbacks = {}
    self.register_callback('store', _store_callback, pass_context=True)
    self.register_callback('input', _input_callback, pass_context=True)

    if self.prefix_role_tags is None:
      self.prefix_role_tags = {}
      try:
        self.prefix_role_tags = routing.config_registry.get()[
            _PREFIX_ROLE_TAGS_KEY
        ]
      except KeyError:
        self.prefix_role_tags = {}

    if self.add_prefix_role_tags is None:
      try:
        self.add_prefix_role_tags = routing.config_registry.get()[
            _ADD_PREFIX_ROLE_TAGS_KEY
        ]
      except KeyError:
        self.add_prefix_role_tags = False

  @classmethod
  def set_default_prefix_role_tags(
      cls, prefix_role_tags: Mapping[str, tuple[str, str]]
  ):
    """Sets the default prefix role tags.

    Additionally, sets the default of add_prefix_role_tags to True.

    Args:
      prefix_role_tags: The default prefix role tags.
    """
    routing.config_registry.get()[_PREFIX_ROLE_TAGS_KEY] = prefix_role_tags
    routing.config_registry.get()[_ADD_PREFIX_ROLE_TAGS_KEY] = True

  def register_callback(
      self,
      name: str,
      function: Callable[..., Any],
      pass_context: bool = False,
  ):
    """Add callback to the template.

    A callback can be a sync or async function. It may or may not use the
    template context (see the pass_context argument).
    It may return an arbitrary value that will be passed back to the template.
    It may also have the effect of adding values to the context and to the
    template output (in case it takes in the context).

    Args:
      name: Name of the callback when called from a template.
      function: The function implementing the callback.
      pass_context: If True, the first argument of the callback should be the
        context, and it will be passed to the callback so that it can read/write
        into it. If False, the callback should not expect a context argument.
    """

    @jinja2.pass_context
    async def awrapper(context: Mapping[str, Any], *args, **kwargs) -> str:
      if pass_context:
        to_pass = PromptTemplateContext(context)
        result = await function(to_pass, *args, **kwargs)
      else:
        result = await function(*args, **kwargs)
      return result

    @jinja2.pass_context
    def wrapper(context: Mapping[str, Any], *args, **kwargs) -> str:
      if pass_context:
        to_pass = PromptTemplateContext(context)
        result = function(to_pass, *args, **kwargs)
      else:
        result = function(*args, **kwargs)
      return result

    if inspect.iscoroutinefunction(function):
      self._callbacks[name] = awrapper
    else:
      self._callbacks[name] = wrapper

  def _prepare_prompt(self) -> jinja2.Template:
    """Parses the prompt into a Jinja Template and adds the callbacks."""
    environment = create_jinja2_environment(self.loader)
    for k, v in self.template_globals.items():
      if k in environment.globals:
        raise ValueError(f'{k} is already in the jinja environment globals.')
      environment.globals[k] = v

    parsed_prompt = environment.from_string(self.text)
    parsed_prompt.globals.update(self._callbacks)
    return parsed_prompt

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def _iterate_through_prompt(
      self,
      context: PromptTemplateContext,
  ) -> AsyncIterator[str]:
    """Runs the template iteratively.

    This iterator doesn't yield anything as it writes its output directly
    into the context variables.

    Args:
      context: Initial value of the template context.

    Yields:
      None whenever a new stage is added.
    """
    template = self._prepare_prompt()
    ctx = template.new_context(context.jinja_context)
    result = tracing.execution_context.get()
    stages_length = -1
    try:
      async for s in template.generate_async(ctx):
        context.prefix += s
        # We don't yield the prefix at each prompt iteration but only when
        # the iteration results in adding a stage to the ExecutionResult.
        # Indeed, by default Jinja would have one iteration for every callback
        # call (before and after the call), but we only report intermediate
        # results when the callback is traced hence generates a new stage (and
        # this happens after the callback has returned to jinja).
        if result is not None and len(result.stages) > stages_length:
          stages_length = len(result.stages)
          yield ''  # This won't be used.
    except ValueError as e:
      # We append the error to the prefix so that it can be inspected.
      context.prefix += f'{constants.ERROR_STRING}: {e}'
      context.error = f'{e}'
      self._prompt_done = True
      yield ''  # This won't be used.

    self._prompt_done = True
    yield ''

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def _execute_reply(
      self, context: PromptTemplateContext
  ) -> AsyncIterator[str]:
    """Waits for an IterableReply and runs through it.

    If the replies are not iterable, this won't be triggered, so this is just
    a process we run asynchronously to watch for iterable replies and execute
    them.

    Args:
      context: Initial value of the template context.

    Yields:
      An empty string at the end of iterations just to yield back control and
      signal the processing is done. The actual reply is stored in
      context.iterated_reply.
    """
    while not self._prompt_done:
      await asyncio.sleep(0)
      if context.iterable_reply is not None:
        try:
          context.iterated_reply = await context.iterable_reply
        finally:
          if context.iterated_reply is None:
            context.iterated_reply = constants.ERROR_STRING
          context.iterable_reply = None
        yield ''  # This won't be used.

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def _iterate_through_reply(
      self, context: PromptTemplateContext
  ) -> AsyncIterator[str]:
    """Waits for an IterableReply and iterates through it.

    If the replies are not iterable, this won't be triggered, so this is just
    a process we run asynchronously to watch for iterable replies and execute
    them step-by-step so as to be able to report intermediate results as
    the processing happens (indeed we cannot do that directly in the Jinja
    callback since these callbacks cannot be AsyncIterators).

    Args:
      context: Initial value of the template context.

    Yields:
      The contents of the iterable reply.
    """
    while not self._prompt_done:
      await asyncio.sleep(0)
      if context.iterable_reply is not None:
        context.iterated_reply = None
        result = constants.ERROR_STRING
        try:
          async for r in context.iterable_reply:
            yield r
            if isinstance(r, updating.Update):
              result = r.to_result()
            else:
              result = r
        finally:
          context.iterated_reply = result
          context.iterable_reply = None

  def _postprocess_iterable_reply(self, iterable_reply: Any) -> str:
    return iterable_reply

  async def _render(
      self,
      iteration_depth: int,
      kwargs: Mapping[str, Any],
  ) -> AsyncIterator[Mapping[str, Any]]:
    """Actually processes and fills in the template iteratively."""
    context = PromptTemplateContext()
    context.prefix_role_tags.update(self.prefix_role_tags)
    context.add_prefix_role_tags = self.add_prefix_role_tags
    # Divide args between inputs and context_variables.
    inputs = {}
    for k, v in kwargs.items():
      if k in self.move_inputs_to_context_vars:
        context.context_variables[k] = copy.deepcopy(v)
      else:
        inputs[k] = v  # Deepcopy happens inside _iterate_through_prompt.
    context.input_variables.update(copy.deepcopy(inputs))

    # Saving the context so that it is accessible at the end of the execution
    # for re-rendering purposes.
    self._context = context
    self._prompt_done = False
    par = executing.parallel(
        self._iterate_through_prompt(context),
        self._iterate_through_reply(context)
        if iteration_depth
        else self._execute_reply(context),
    )
    updates = updating.Update()
    async for update in par.with_depth(iteration_depth=iteration_depth):
      updates += update
      # current_pair contains the output of the parallel executing, it may
      # contain one entry if we didn't get an iterable reply, otherwise it
      # will contain two entries, the first one containing None and the second
      # one containing the partial reply.
      current_pair = updates.to_result()
      context.output_variables[PROMPT_PREFIX] = context.prefix
      if context.error is not None:
        context.output_variables[ERROR] = context.error
      iterable_reply = None
      if (
          len(current_pair) > 1
          and context.iterable_reply is not None
      ):
        # Due to the parallel executing, we could get the yield from
        # _iterate_through_reply, after the iteration through the reply has
        # finished and has been added to the prefix. This is why we also
        # check that the ITERABLE_REPLY is not None (if it is None it means
        # we have added it to the prefix already).
        iterable_reply = self._postprocess_iterable_reply(current_pair[1])
      if self.role_tags:
        context.output_variables[PREFIX_WITH_ROLES] = _format_with_roles(
            context.prefix,
            context.role_indices,
            self.role_tags,
            iterable_reply=iterable_reply,
        )
      context.output_variables[ITERABLE_REPLY] = iterable_reply
      yield context.output_variables

  def reformat(
      self, role_tags: Mapping[str, tuple[str | None, str | None]]
  ) -> str:
    """Reformats a rendered template with different role tags.

    This is useful if after calling `render()` we want to render again the
    template with different role tags (e.g. if we want the rendered template
    both in HTML and in plain text).

    Args:
      role_tags: The new role_tags to use for formatting (see class docstring
        for details).

    Returns:
      The reformatted template as a string.

    Raises:
      ValueError if the template hasn't been processed yet.
    """
    if not self._context:
      raise ValueError(
          "Template hasn't been processed yet, run `render()` or"
          ' `render_stream()` first.'
      )
    iterable_reply = None
    if self._context.iterable_reply is not None:
      # This is the case where we are in the process of iterating through
      # an LLM reply.
      # We fetch the partial reply from the output variables.
      iterable_reply = self._context.output_variables.get(  # pytype: disable=attribute-error
          ITERABLE_REPLY, None
      )
    return _format_with_roles(
        self._context.prefix,
        self._context.role_indices,
        role_tags,
        iterable_reply=iterable_reply,
    )

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name=utils.FromInstance('name'))
  async def render_stream(
      self,
      **kwargs: Any,
  ) -> AsyncIterator[Mapping[str, Any]]:
    # We need to have iteration_depth=2 in order to step through the replies
    # and the prompt.
    async for outputs in self._render(2, kwargs):
      yield outputs

  @executing.make_executable  # pytype: disable=wrong-arg-types
  @tracing.trace(name=utils.FromInstance('name'))
  async def render(
      self,
      **kwargs: Any,
  ) -> Mapping[str, Any]:
    final_outputs = {}
    async for outputs in self._render(0, kwargs):
      final_outputs = outputs
    final_outputs.pop(ITERABLE_REPLY, None)
    return final_outputs
