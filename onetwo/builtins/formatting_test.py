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

"""Unit tests for formatting."""

from collections.abc import Sequence
from typing import TypeAlias
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import formatting
from onetwo.core import content as content_lib


_Message: TypeAlias = content_lib.Message
_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


class FormatterForTest(formatting.Formatter):
  """Formatter for testing."""

  def is_role_supported(self, role: str| _PredefinedRole) -> bool:
    """Overridden from base class (Formatter)."""
    return role in {_PredefinedRole.USER, _PredefinedRole.MODEL}

  def is_already_formatted(self, content: Sequence[_Message]) -> bool:
    """Returns whether the content is already formatted."""
    return any([('</' in str(msg.content)) for msg in content])

  def _format(
      self,
      content: Sequence[_Message],
  ) -> _ChunkList:
    """Returns formatted ChunkList."""
    result = _ChunkList()
    for msg in content:
      role = (
          msg.role.value
          if isinstance(msg.role, content_lib.PredefinedRole)
          else msg.role
      )
      result += _Chunk(content=f'<{role}>{msg.content}</{role}>\n')
    return result


class FormattingTest(parameterized.TestCase):

  def test_format_content(self):
    content = [
        _Message(
            role=content_lib.PredefinedRole.USER,
            content='This is a user message.',
        ),
        _Message(
            role=content_lib.PredefinedRole.MODEL,
            content='This is an assistant message.',
        ),
    ]
    expected = _ChunkList([
        _Chunk(content='<user>This is a user message.</user>\n'),
        _Chunk(
            content='<model>This is an assistant message.</model>\n'
        ),
    ])
    self.assertEqual(expected, FormatterForTest().format(content))

  @parameterized.named_parameters(
      ('formatted', '<user>This is a user message.</user>', True),
      ('not_formatted', 'This is a user message.', False),
  )
  def test_format_content_already_formatted(self, content, expected):
    messages = [
        _Message(
            role=content_lib.PredefinedRole.USER,
            content=content,
        )
    ]
    formatter = FormatterForTest()

    with self.subTest('is_already_formatted'):
      self.assertEqual(
          formatter.is_already_formatted(messages),
          expected,
      )

    with self.subTest('raises_if_already_formatted'):
      if expected:
        with self.assertRaises(ValueError):
          formatter.format(messages, raise_error_if_already_formatted=True)

    with self.subTest('does_not_raise'):
      formatter.format(messages, raise_error_if_already_formatted=False)

  def test_supported_roles(self):
    messages = [
        _Message(
            role=content_lib.PredefinedRole.USER,
            content='user content',
        ),
        _Message(
            role=content_lib.PredefinedRole.SYSTEM,
            content='system content',
        ),
    ]
    formatter = FormatterForTest()

    with self.subTest('raises_if_unsupported_role'):
      with self.assertRaises(ValueError):
        formatter.format(messages, raise_error_if_unsupported_roles=True)

    with self.subTest('does_not_raise'):
      formatter.format(messages, raise_error_if_unsupported_roles=False)


class DefaultFormatterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'with_single_system_msg',
          (
              _Message(
                  role=_PredefinedRole.SYSTEM, content='Any instructions'
              ),
              _Message(role=_PredefinedRole.USER, content='Hello!'),
          ),
          False,
          (
              '**System**: Actor "Model" needs to obey the following'
              ' rules when generating the messages below:\nAny'
              ' instructions\n**User**: Hello!\n**Model**:'
          ),
      ),
      (
          'several_system_msgs',
          (
              _Message(role=_PredefinedRole.SYSTEM, content='Any instructions'),
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.SYSTEM, content='No instructions'),
          ),
          False,
          (
              '**System**: Actor "Model" needs to obey the following rules when'
              ' generating the messages below:\nAny instructions\n**User**:'
              ' Hello!\n**System**: Actor "Model" needs to obey the following'
              ' rules when generating the messages below:\nNo'
              ' instructions\n**Model**:'
          ),
      ),
      (
          'some_empty_system_msg',
          (
              _Message(role=_PredefinedRole.SYSTEM, content=''),
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.SYSTEM, content='Any instructions'),
              _Message(role=_PredefinedRole.SYSTEM, content='No instructions'),
          ),
          False,
          (
              '**User**: Hello!\n**System**: Actor "Model" needs to obey the'
              ' following rules when generating the messages below:\nAny'
              ' instructions\n**System**: Actor "Model" needs to obey the'
              ' following rules when generating the messages below:\nNo'
              ' instructions\n**Model**:'
          ),
      ),
      (
          'instruct_mode_single_msg_wo_fewshots',
          (_Message(role=_PredefinedRole.USER, content='Hello!'),),
          False,
          'Task: Hello!\nAnswer:',
      ),
      (
          'instruct_mode_single_msg_with_fewshots',
          (_Message(role=_PredefinedRole.USER, content='Hello!'),),
          True,
          'Task: Write me a palindrome.\nAnswer: Level\nTask: Hello!\nAnswer:',
      ),
      (
          'instruct_mode_with_empty_model_msg_wo_fewshots',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content=''),
          ),
          False,
          'Task: Hello!\nAnswer:',
      ),
      (
          'instruct_mode_with_empty_model_msg_with_fewshots',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content=''),
          ),
          True,
          'Task: Write me a palindrome.\nAnswer: Level\nTask: Hello!\nAnswer:',
      ),
      (
          'instruct_mode_with_model_msg_wo_fewshots',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content='Hey'),
          ),
          False,
          'Task: Hello!\nAnswer: Hey',
      ),
      (
          'instruct_mode_with_model_msg_with_fewshots',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content='Hey'),
          ),
          True,
          (
              'Task: Write me a palindrome.\nAnswer: Level\n'
              'Task: Hello!\nAnswer: Hey'
          ),
      ),
      (
          'chat_mode_user_last',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content='Hey.'),
              _Message(role=_PredefinedRole.USER, content='How are you?'),
          ),
          False,
          (
              '**User**: Hello!\n'
              '**Model**: Hey.\n'
              '**User**: How are you?\n'
              '**Model**:'
          ),
      ),
      (
          'chat_mode_assistant_last_dont_include_newline',
          (
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content='Hey.'),
              _Message(role=_PredefinedRole.USER, content='How are you?'),
              _Message(role=_PredefinedRole.MODEL, content='I am'),
          ),
          False,
          (
              '**User**: Hello!\n'
              '**Model**: Hey.\n'
              '**User**: How are you?\n'
              '**Model**: I am'
          ),
      ),
  )
  def test_format(self, messages, use_fewshots, expected_result):
    formatter = formatting.DefaultFormatter({'use_fewshots': use_fewshots})
    res = formatter.format(messages)
    self.assertEqual(expected_result, str(res))


class ConcatFormatterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'single_msg',
          (_Message(role=_PredefinedRole.USER, content='Hello!'),),
          'Hello!',
      ),
      (
          'several_msgs',
          (
              _Message(role=_PredefinedRole.SYSTEM, content='Any instructions'),
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content='Hey.'),
              _Message(role=_PredefinedRole.USER, content='How are you?'),
              _Message(role=_PredefinedRole.MODEL, content='I am'),
          ),
          (
              'Any instructionsHello!Hey.How are you?I am'
          ),
      ),
      (
          'several_msgs_including_empty_msgs',
          (
              _Message(role=_PredefinedRole.SYSTEM, content='Any instructions'),
              _Message(role=_PredefinedRole.SYSTEM, content=''),
              _Message(role=_PredefinedRole.USER, content='Hello!'),
              _Message(role=_PredefinedRole.MODEL, content=''),
              _Message(role=_PredefinedRole.USER, content=''),
              _Message(role=_PredefinedRole.MODEL, content='Hey.'),
              _Message(role=_PredefinedRole.USER, content='How are you?'),
              _Message(role=_PredefinedRole.MODEL, content='I am'),
          ),
          (
              'Any instructionsHello!Hey.How are you?I am'
          ),
      ),
      (
          'several_msgs_including_newlines',
          (
              _Message(role=_PredefinedRole.SYSTEM, content='Instructions\n'),
              _Message(role=_PredefinedRole.USER, content='\nHello!\n'),
              _Message(role=_PredefinedRole.MODEL, content='Hey.\n'),
              _Message(role=_PredefinedRole.USER, content='\nHow are you?\n'),
              _Message(role=_PredefinedRole.MODEL, content='I am\n'),
          ),
          (
              'Instructions\n\nHello!\nHey.\n\nHow are you?\nI am\n'
          ),
      ),
  )
  def test_format(self, messages, expected_result):
    formatter = formatting.ConcatFormatter()
    res = formatter.format(messages)
    self.assertEqual(expected_result, str(res))


if __name__ == '__main__':
  absltest.main()
