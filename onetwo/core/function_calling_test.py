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

"""Tests for the core function calling data types."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import function_calling


class _Subclass(function_calling.FunctionCall):
  """Stands in for the compatibility subclass in `llm_tool_use`."""


class FunctionCallTest(parameterized.TestCase):

  def test_defaults(self):
    call = function_calling.FunctionCall()

    self.assertEqual('', call.function_name)
    self.assertEqual((), call.args)
    self.assertEqual({}, call.kwargs)

  def test_positional_construction_matches_field_order(self):
    # Guards the field order, which positional callers rely on.
    call = function_calling.FunctionCall('Search', ('a',), {'b': 1})

    self.assertEqual(
        function_calling.FunctionCall(
            function_name='Search', args=('a',), kwargs={'b': 1}
        ),
        call,
    )

  def test_repr_is_unchanged_by_the_move(self):
    # Several callers serialise a `FunctionCall` by its repr and assert on the
    # resulting string, so the repr is effectively part of the public API.
    # Adding or reordering a field will break them; see b/561535278.
    self.assertEqual(
        "FunctionCall(function_name='Finish', args=('4',), kwargs={})",
        repr(function_calling.FunctionCall('Finish', ('4',), {})),
    )

  @parameterized.named_parameters(
      ('function_name', {'function_name': 'Other'}),
      ('args', {'args': ('z',)}),
      ('kwargs', {'kwargs': {'b': 2}}),
  )
  def test_not_equal_when_any_field_differs(self, overrides):
    call = function_calling.FunctionCall(
        function_name='Search', args=('a',), kwargs={'b': 1}
    )

    self.assertNotEqual(call, dataclasses.replace(call, **overrides))

  def test_equal_to_subclass_with_same_values(self):
    # The compatibility subclass in `llm_tool_use` must be interchangeable with
    # the core type while callers are being migrated, in both directions.
    base = function_calling.FunctionCall(function_name='Search', args=('a',))
    subclass = _Subclass(function_name='Search', args=('a',))

    self.assertEqual(base, subclass)
    self.assertEqual(subclass, base)

  def test_not_equal_to_unrelated_type(self):
    self.assertNotEqual(function_calling.FunctionCall(), 'Search()')

  def test_is_unhashable(self):
    # Matches the behaviour of the mutable dataclass this type replaces.
    with self.assertRaises(TypeError):
      hash(function_calling.FunctionCall())

  def test_is_mutable(self):
    call = function_calling.FunctionCall()

    call.function_name = 'Search'

    self.assertEqual('Search', call.function_name)


class FunctionResponseTest(parameterized.TestCase):

  def test_defaults(self):
    response = function_calling.FunctionResponse()

    self.assertEqual('', response.function_name)
    self.assertIsNone(response.response)
    self.assertIsNone(response.error)

  def test_equal_when_all_fields_match(self):
    self.assertEqual(
        function_calling.FunctionResponse(
            function_name='Search', response=['a']
        ),
        function_calling.FunctionResponse(
            function_name='Search', response=['a']
        ),
    )

  @parameterized.named_parameters(
      ('function_name', {'function_name': 'Other'}),
      ('response', {'response': ['z']}),
      ('error', {'error': 'boom'}),
  )
  def test_not_equal_when_any_field_differs(self, overrides):
    response = function_calling.FunctionResponse(
        function_name='Search', response=['a']
    )

    self.assertNotEqual(response, dataclasses.replace(response, **overrides))

  def test_error_is_represented_rather_than_raised(self):
    response = function_calling.FunctionResponse(
        function_name='Search', error='tool unavailable'
    )

    self.assertIsNone(response.response)
    self.assertEqual('tool unavailable', response.error)


if __name__ == '__main__':
  absltest.main()
