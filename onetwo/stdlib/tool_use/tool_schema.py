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

"""Derives a tool declaration from the signature and docstring of a callable.

Model-native function calling requires every tool to be declared with a
machine-readable signature: a JSON Schema for its arguments, plus prose telling
the model when to use it. Writing that schema out by hand duplicates
information that a typed, documented Python function already carries, and the
two copies drift.

This module reads it off the function instead:

  ```python
  def search_entities(query: str, max_results: int = 5) -> str:
    '''Searches the knowledge graph for entities matching a query.

    Args:
      query: The text to search for.
      max_results: Maximum number of entities to return.

    '''
    ...

  tool = tool_schema.tool_from_callable(search_entities)
  tool.parameters == {
      'type': 'object',
      'properties': {
          'query': {'type': 'string', 'description': 'The text to search for.'},
          'max_results': {
              'type': 'integer',
              'description': 'Maximum number of entities to return.',
              'default': 5,
          },
      },
      'required': ['query'],
  }
  ```

Annotations are turned into schema by pydantic, so anything pydantic can
describe works here, including nested models, enums and `Literal`. Prose is
read by `docstring_parser`, so Google, NumPy and reST docstring styles are all
understood.

Inference is deliberately strict: a callable that cannot be described
faithfully raises rather than yielding a declaration that silently misleads the
model. See `schema_from_callable` for the exact rules.

The schema produced here is plain JSON Schema, which is the dialect MCP and the
OpenAI API speak. Backends whose schema language is narrower (Gemini and Beyond
accept a subset of OpenAPI 3.0, with no `$ref`) are responsible for lowering it
on the way out; this module does not pre-emptively restrict itself to the
smallest common denominator.

This module is deliberately separate from `llm_tool_use` because it pulls in
pydantic, which is heavier than the rest of the tool-use library. Only callers
that actually want schema inference pay for it.
"""

from collections.abc import Callable
import inspect
from typing import Any

import docstring_parser
from onetwo.stdlib.tool_use import llm_tool_use
import pydantic

# Parameter kinds that a function-calling API can address. `*args` and
# `**kwargs` are excluded: the wire format passes named arguments only, so a
# schema cannot express them.
_SUPPORTED_PARAMETER_KINDS = frozenset({
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
})

# Keys whose values are themselves schemas, and so need the same treatment as
# the node containing them.
_SUBSCHEMA_MAP_KEYS = (
    'properties',
    '$defs',
    'definitions',
    'patternProperties',
)
_SUBSCHEMA_KEYS = ('items', 'additionalProperties', 'not', 'contains')
_SUBSCHEMA_LIST_KEYS = ('anyOf', 'oneOf', 'allOf', 'prefixItems')


def _name_of(function: Callable[..., Any]) -> str:
  """Returns a name to identify `function` by, for use in error messages."""
  return getattr(function, '__name__', repr(function))


def _parse_docstring(
    function: Callable[..., Any],
) -> docstring_parser.Docstring:
  """Returns the parsed docstring of `function`.

  Args:
    function: The callable to read.

  Raises:
    ValueError: If `function` has no docstring, or one with no description. The
      description is the only place a model learns what the tool is for, so
      there is nothing useful to declare without it.
  """
  docstring = inspect.getdoc(function)
  parsed = docstring_parser.parse(docstring) if docstring else None
  if parsed is None or not (parsed.description or '').strip():
    raise ValueError(
        f'Cannot infer a tool declaration for {_name_of(function)!r}: it has'
        ' no docstring description. The description is what the model reads to'
        ' decide whether to call the tool, so it cannot be omitted. Add a'
        ' docstring summarising what the tool does and, ideally, an `Args:`'
        ' section describing each parameter.'
    )
  return parsed


def _strip_generated_titles(schema: Any) -> None:
  """Removes pydantic's auto-generated `title` keys from `schema`, in place.

  pydantic titles every field with a humanised version of its own name
  (`max_results` becomes `Max Results`), which tells the model nothing it
  cannot read off the property name and costs tokens in every request.

  Args:
    schema: A JSON Schema node. Non-dict values are ignored, so this can be
      called on any subschema position without checking it first.
  """
  if not isinstance(schema, dict):
    return
  schema.pop('title', None)
  for key in _SUBSCHEMA_MAP_KEYS:
    subschemas = schema.get(key)
    if isinstance(subschemas, dict):
      for subschema in subschemas.values():
        _strip_generated_titles(subschema)
  for key in _SUBSCHEMA_KEYS:
    _strip_generated_titles(schema.get(key))
  for key in _SUBSCHEMA_LIST_KEYS:
    subschemas = schema.get(key)
    if isinstance(subschemas, list):
      for subschema in subschemas:
        _strip_generated_titles(subschema)


def description_from_callable(function: Callable[..., Any]) -> str:
  """Returns the tool description to declare for `function`.

  This is the docstring's summary line plus any extended description, with the
  `Args:`, `Returns:` and `Raises:` sections dropped: per-parameter prose
  belongs in the schema, next to the parameter it describes.

  Args:
    function: The callable to describe.

  Raises:
    ValueError: If `function` has no docstring description.
  """
  # `_parse_docstring` has already rejected an absent description; the `or` is
  # only here because its static type admits None.
  return (_parse_docstring(function).description or '').strip()


def schema_from_callable(
    function: Callable[..., Any],
) -> dict[str, Any] | None:
  """Returns a JSON Schema for the arguments of `function`.

  Parameter types come from the annotations, defaults from the signature, and
  per-parameter prose from the `Args:` section of the docstring. A parameter
  without a default is listed in `required`.

  Inference fails loudly rather than guessing, because a declaration the model
  cannot act on is worse than no tool at all:

  - No docstring description: raises. The model has no way to tell what the
    tool is for. (An undocumented *parameter* is tolerated: its name and type
    often say enough, and demanding an `Args:` entry for every argument turns
    a useful constraint into busywork.)
  - An unannotated parameter: raises. Without a type there is nothing to put in
    the schema but "any JSON value", which invites the model to pass something
    the function will reject.
  - `*args` or `**kwargs`: raises. Function-calling APIs pass named arguments
    only, so these cannot be declared; accepting them silently would hide part
    of the signature from the model.

  Args:
    function: The callable to describe. Bound methods are supported; `self` is
      already absent from their signature.

  Returns:
    A JSON Schema object describing the arguments, or None if `function` takes
    none. Nullary tools are declared by omitting the schema rather than by an
    empty one, so that backends rejecting an empty property list can leave the
    field unset.

  Raises:
    ValueError: If `function` cannot be described faithfully, per the rules
      above.
  """
  parsed_docstring = _parse_docstring(function)
  param_descriptions = {
      param.arg_name: param.description
      for param in parsed_docstring.params
      if param.description
  }

  fields: dict[str, Any] = {}
  for name, parameter in inspect.signature(function).parameters.items():
    if parameter.kind not in _SUPPORTED_PARAMETER_KINDS:
      raise ValueError(
          f'Cannot infer a schema for {_name_of(function)!r}: parameter'
          f' {name!r} is variadic (*args or **kwargs). Function-calling APIs'
          ' pass named arguments only, so a variadic parameter cannot be'
          ' declared to the model. Replace it with explicit parameters.'
      )
    if parameter.annotation is inspect.Parameter.empty:
      raise ValueError(
          f'Cannot infer a schema for {_name_of(function)!r}: parameter'
          f' {name!r} has no type annotation, so the model would be told only'
          ' that it accepts any JSON value. Annotate it with the type the'
          ' function expects.'
      )
    description = param_descriptions.get(name)
    if parameter.default is inspect.Parameter.empty:
      # Omitting `default` entirely is how pydantic is told a field is
      # required, which is in turn how it lands in the schema's `required`.
      field = pydantic.Field(description=description)
    else:
      field = pydantic.Field(default=parameter.default, description=description)
    fields[name] = (parameter.annotation, field)

  if not fields:
    return None

  schema = pydantic.create_model(
      _name_of(function), **fields
  ).model_json_schema()
  _strip_generated_titles(schema)
  return schema


def tool_from_callable(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    example: str | llm_tool_use.ToolExample | None = None,
    color: str | None = None,
    irrecoverable_error_types: list[str] | None = None,
) -> llm_tool_use.Tool:
  """Returns a `Tool` wrapping `function`, with its declaration inferred.

  Args:
    function: The callable to expose as a tool. Its docstring and annotations
      become the tool's description and parameter schema, so it must satisfy the
      requirements documented on `schema_from_callable`.
    name: Name the model will call the tool by. Defaults to the name of
      `function`, and is worth setting when tools from several sources share a
      namespace and could collide.
    example: Passed through to `Tool`.
    color: Passed through to `Tool`.
    irrecoverable_error_types: Passed through to `Tool`.

  Raises:
    ValueError: If `function` cannot be described faithfully, per the rules
      documented on `schema_from_callable`.
  """
  return llm_tool_use.Tool(
      name=name or _name_of(function),
      function=function,
      description=description_from_callable(function),
      example=example,
      color=color,
      irrecoverable_error_types=irrecoverable_error_types,
      parameters=schema_from_callable(function),
  )
