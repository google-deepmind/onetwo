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

# The callables in this file are test fixtures standing in for the tools a user
# would write, and several are deliberately under-documented to exercise what
# inference does with them. gpylint's docstring-completeness checks would
# fight that.
# pylint: disable=g-doc-args,g-doc-return-or-yield


import enum
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import executing
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import tool_schema
import pydantic


def search_entities(query: str, max_results: int = 5) -> str:
  """Searches the knowledge graph for entities matching a query.

  Args:
    query: The text to search for.
    max_results: Maximum number of entities to return.

  Returns:
    The matching entities, as JSON.
  """
  return f'{query}:{max_results}'


def get_graph_statistics() -> str:
  """Returns entity and relation counts for the knowledge graph."""
  return '{}'


class SchemaFromCallableTest(parameterized.TestCase):

  def test_schema_matches_the_signature_and_docstring(self):
    self.assertEqual(
        {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The text to search for.',
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of entities to return.',
                    'default': 5,
                },
            },
            'required': ['query'],
        },
        tool_schema.schema_from_callable(search_entities),
    )

  def test_nullary_callable_has_no_schema(self):
    # A tool that takes no arguments is declared by omitting the schema, not by
    # an empty one.
    self.assertIsNone(tool_schema.schema_from_callable(get_graph_statistics))

  def test_undocumented_parameter_is_allowed(self):
    def tool(query: str, limit: int = 3) -> str:
      """Searches for something.

      Args:
        query: The text to search for.
      """
      return f'{query}:{limit}'

    schema = tool_schema.schema_from_callable(tool)

    assert schema is not None
    self.assertEqual(
        {'type': 'integer', 'default': 3}, schema['properties']['limit']
    )

  def test_literal_becomes_an_enum(self):
    def get_neighbors(
        entity_id: str, direction: Literal['in', 'out', 'both'] = 'both'
    ) -> str:
      """Returns the neighbors of an entity.

      Args:
        entity_id: The entity to start from.
        direction: Which edges to follow.
      """
      return f'{entity_id}:{direction}'

    schema = tool_schema.schema_from_callable(get_neighbors)

    assert schema is not None
    self.assertEqual(
        ['in', 'out', 'both'], schema['properties']['direction']['enum']
    )

  def test_enum_class_becomes_an_enum(self):
    class Unit(enum.Enum):
      CELSIUS = 'celsius'
      FAHRENHEIT = 'fahrenheit'

    def get_temperature(city: str, unit: Unit = Unit.CELSIUS) -> str:
      """Returns the temperature in a city.

      Args:
        city: The city to look up.
        unit: The unit to report the temperature in.
      """
      return f'{city}:{unit}'

    schema = tool_schema.schema_from_callable(get_temperature)

    assert schema is not None
    self.assertEqual(['celsius', 'fahrenheit'], schema['$defs']['Unit']['enum'])

  def test_nested_model_is_referenced_rather_than_inlined(self):
    # Backends whose schema language has no `$ref` are expected to inline it
    # while lowering; we emit standard JSON Schema here.
    class Location(pydantic.BaseModel):
      latitude: float
      longitude: float

    def get_forecast(location: Location) -> str:
      """Returns the forecast for a location.

      Args:
        location: Where to forecast for.
      """
      return str(location)

    schema = tool_schema.schema_from_callable(get_forecast)

    assert schema is not None
    self.assertIn('Location', schema['$defs'])
    self.assertEqual(
        ['latitude', 'longitude'],
        list(schema['$defs']['Location']['properties']),
    )

  def test_pydantic_generated_titles_are_stripped(self):
    class Location(pydantic.BaseModel):
      latitude: float

    def get_forecast(location: Location, max_days: int = 3) -> str:
      """Returns the forecast for a location.

      Args:
        location: Where to forecast for.
        max_days: How many days to forecast.
      """
      return f'{location}:{max_days}'

    schema = tool_schema.schema_from_callable(get_forecast)

    assert schema is not None
    self.assertNotIn('title', schema)
    self.assertNotIn('title', schema['properties']['max_days'])
    self.assertNotIn('title', schema['$defs']['Location'])
    self.assertNotIn(
        'title', schema['$defs']['Location']['properties']['latitude']
    )

  def test_parameter_named_title_survives(self):
    def publish(title: str) -> str:
      """Publishes a document.

      Args:
        title: The title of the document.
      """
      return title

    schema = tool_schema.schema_from_callable(publish)

    assert schema is not None
    self.assertEqual(['title'], list(schema['properties']))

  def test_missing_docstring_raises(self):
    def tool(query: str) -> str:
      return query

    with self.assertRaisesRegex(ValueError, 'no docstring description'):
      tool_schema.schema_from_callable(tool)

  def test_blank_docstring_raises(self):
    def tool(query: str) -> str:
      """ """  # pylint: disable=g-docstring-quotes
      return query

    with self.assertRaisesRegex(ValueError, 'no docstring description'):
      tool_schema.schema_from_callable(tool)

  def test_unannotated_parameter_raises(self):
    def tool(query) -> str:  # pylint: disable=missing-param-doc
      """Searches for something.

      Args:
        query: The text to search for.
      """
      return str(query)

    with self.assertRaisesRegex(ValueError, 'no type annotation'):
      tool_schema.schema_from_callable(tool)

  @parameterized.named_parameters(
      ('var_positional', lambda *args: None),
      ('var_keyword', lambda **kwargs: None),
  )
  def test_variadic_parameter_raises(self, function):
    function.__doc__ = 'Does something variadic.'

    with self.assertRaisesRegex(ValueError, 'variadic'):
      tool_schema.schema_from_callable(function)


class DescriptionFromCallableTest(parameterized.TestCase):

  def test_description_excludes_the_parameter_sections(self):
    self.assertEqual(
        'Searches the knowledge graph for entities matching a query.',
        tool_schema.description_from_callable(search_entities),
    )

  def test_description_includes_the_extended_summary(self):
    def tool() -> str:
      """Searches for something.

      It does so slowly, so prefer to call it once with a broad query.
      """
      return ''

    self.assertEqual(
        'Searches for something.\n\nIt does so slowly, so prefer to call it'
        ' once with a broad query.',
        tool_schema.description_from_callable(tool),
    )


class ToolFromCallableTest(parameterized.TestCase):

  def test_tool_is_fully_populated_from_the_callable(self):
    tool = tool_schema.tool_from_callable(search_entities)

    self.assertEqual('search_entities', tool.name)
    self.assertEqual(search_entities, tool.function)
    self.assertEqual(
        'Searches the knowledge graph for entities matching a query.',
        tool.description,
    )
    self.assertEqual(
        tool_schema.schema_from_callable(search_entities), tool.parameters
    )

  def test_the_resulting_tool_is_callable(self):
    tool = tool_schema.tool_from_callable(search_entities)

    self.assertEqual('cats:5', executing.run(tool(query='cats')))

  def test_name_can_be_overridden(self):
    # Tools drawn from several sources share one namespace, so the name the
    # model sees is not always the name of the function.
    tool = tool_schema.tool_from_callable(
        search_entities, name='graphrag__search_entities'
    )

    self.assertEqual('graphrag__search_entities', tool.name)
    self.assertEqual(search_entities, tool.function)

  def test_local_policy_fields_are_passed_through(self):
    tool = tool_schema.tool_from_callable(
        search_entities,
        example='search_entities(query="cats")',
        color='blue',
        irrecoverable_error_types=['PermissionError'],
    )

    self.assertEqual('search_entities(query="cats")', tool.example)
    self.assertEqual('blue', tool.color)
    self.assertEqual(['PermissionError'], tool.irrecoverable_error_types)

  def test_spec_of_an_inferred_tool_declares_the_schema(self):
    # The point of inference: `to_spec` now has something to project.
    spec = tool_schema.tool_from_callable(search_entities).to_spec()

    self.assertEqual('search_entities', spec.name)
    assert spec.parameters is not None
    self.assertEqual(['query'], spec.parameters['required'])

  def test_inferred_tool_is_an_ordinary_tool(self):
    tool = tool_schema.tool_from_callable(get_graph_statistics)

    self.assertIsInstance(tool, llm_tool_use.Tool)
    self.assertIsNone(tool.parameters)


if __name__ == '__main__':
  absltest.main()
