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

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.genai import types as genai_types
from onetwo.backends import google_genai_tool_utils


def _make_fc_part(name: str, args: dict[str, Any], fc_id: str | None = None):
  fc = mock.Mock()
  fc.name = name
  fc.args = args
  fc.id = fc_id
  return mock.Mock(function_call=fc, thought=None, text=None)


class GoogleGenAiToolUtilsTest(parameterized.TestCase):

  def test_dump_pydantic(self):
    class DummyPydanticModel:

      def model_dump(self, mode='json'):
        del mode
        return {'key': 'value'}

    self.assertEqual(
        google_genai_tool_utils.dump_pydantic(DummyPydanticModel()),
        {'key': 'value'},
    )
    self.assertEqual(
        google_genai_tool_utils.dump_pydantic({'nested': DummyPydanticModel()}),
        {'nested': {'key': 'value'}},
    )
    self.assertEqual(
        google_genai_tool_utils.dump_pydantic('plain_str'),
        'plain_str',
    )

  def test_tool_call_detail_dataclasses_to_dict(self):
    fc = google_genai_tool_utils.FunctionCallDetail(
        name='search', args={'q': 'test'}, id='c1', response={'hits': 5}
    )
    self.assertEqual(
        fc.to_dict(),
        {
            'type': 'function_call',
            'name': 'search',
            'args': {'q': 'test'},
            'id': 'c1',
            'response': {'hits': 5},
        },
    )

    fc_minimal = google_genai_tool_utils.FunctionCallDetail(name='search')
    self.assertEqual(
        fc_minimal.to_dict(),
        {'type': 'function_call', 'name': 'search'},
    )

    fr = google_genai_tool_utils.FunctionResponseDetail(
        name='search', response={'hits': 5}
    )
    self.assertEqual(
        fr.to_dict(),
        {
            'type': 'function_response',
            'name': 'search',
            'response': {'hits': 5},
        },
    )

    thought = google_genai_tool_utils.ThoughtDetail(text='Thinking...')
    self.assertEqual(
        thought.to_dict(),
        {'type': 'thought', 'text': 'Thinking...'},
    )

  @parameterized.named_parameters(
      ('none_parts', None, 0, []),
      ('empty_parts', [], 0, []),
      (
          'function_call_only',
          [_make_fc_part('foo', {'a': 1})],
          1,
          [
              google_genai_tool_utils.FunctionCallDetail(
                  name='foo', args={'a': 1}
              )
          ],
      ),
      (
          'function_call_with_id',
          [_make_fc_part('foo', {'a': 1}, fc_id='id_1')],
          1,
          [
              google_genai_tool_utils.FunctionCallDetail(
                  name='foo', args={'a': 1}, id='id_1'
              )
          ],
      ),
      (
          'thought_only',
          [mock.Mock(function_call=None, thought=True, text='Thinking...')],
          0,
          [google_genai_tool_utils.ThoughtDetail(text='Thinking...')],
      ),
      (
          'text_only',
          [mock.Mock(function_call=None, thought=False, text='Hello')],
          0,
          [],
      ),
      (
          'mixed_parts',
          [
              mock.Mock(function_call=None, thought=True, text='Thinking...'),
              _make_fc_part('bar', {'x': 2}),
              mock.Mock(function_call=None, thought=False, text='Final answer'),
          ],
          1,
          [
              google_genai_tool_utils.ThoughtDetail(text='Thinking...'),
              google_genai_tool_utils.FunctionCallDetail(
                  name='bar', args={'x': 2}
              ),
          ],
      ),
  )
  def test_extract_tool_calls_from_parts(
      self, parts, expected_count, expected_details
  ):
    count, details = google_genai_tool_utils.extract_tool_calls_from_parts(
        parts
    )
    self.assertEqual(count, expected_count)
    self.assertEqual(details, expected_details)

  def test_attach_function_response_to_tool_details(self):
    # 1. Matching by ID
    call_1 = google_genai_tool_utils.FunctionCallDetail(
        name='search', id='c1', args={}
    )
    call_2 = google_genai_tool_utils.FunctionCallDetail(
        name='search', id='c2', args={}
    )
    current_calls: list[google_genai_tool_utils.ToolCallDetail] = [
        call_1,
        call_2,
    ]
    resp_1 = genai_types.FunctionResponse(
        id='c2', name='search', response={'val': 2}
    )
    google_genai_tool_utils.attach_function_response_to_tool_details(
        resp_1, current_turn_details=current_calls
    )
    self.assertEqual(call_2.response, {'val': 2})
    self.assertIsNone(call_1.response)

    # 2. Sequential fallback when ID is absent
    resp_2 = genai_types.FunctionResponse(name='search', response={'val': 1})
    google_genai_tool_utils.attach_function_response_to_tool_details(
        resp_2, current_turn_details=current_calls
    )
    self.assertEqual(call_1.response, {'val': 1})

    # 3. Cross-turn matching from accumulated_details
    fetch_call = google_genai_tool_utils.FunctionCallDetail(
        name='fetch', id='f1', args={}
    )
    accumulated: list[google_genai_tool_utils.ToolCallDetail] = [fetch_call]
    curr_turn: list[google_genai_tool_utils.ToolCallDetail] = []
    resp_3 = genai_types.FunctionResponse(
        id='f1', name='fetch', response={'doc': 'abc'}
    )
    google_genai_tool_utils.attach_function_response_to_tool_details(
        resp_3, current_turn_details=curr_turn, accumulated_details=accumulated
    )
    self.assertEqual(fetch_call.response, {'doc': 'abc'})
    self.assertEmpty(curr_turn)

    # 4. Orphan response appends to current_turn
    resp_orphan = genai_types.FunctionResponse(
        name='unknown', response={'err': 'not found'}
    )
    google_genai_tool_utils.attach_function_response_to_tool_details(
        resp_orphan, current_turn_details=curr_turn
    )
    self.assertEqual(
        curr_turn,
        [
            google_genai_tool_utils.FunctionResponseDetail(
                name='unknown', response={'err': 'not found'}
            )
        ],
    )


if __name__ == '__main__':
  absltest.main()
