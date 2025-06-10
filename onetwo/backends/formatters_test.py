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

"""Unit tests for formatting."""
import asyncio
from collections.abc import Sequence
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
# Necessary for the FormatterName enum to be populated.
from onetwo.backends import formatters  # pylint: disable=unused-import
from onetwo.builtins import formatting
from onetwo.core import content as content_lib


_Message: TypeAlias = content_lib.Message
_PredefinedRole: TypeAlias = content_lib.PredefinedRole


class FormattersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'gemma_user_only',
          formatting.FormatterName.GEMMA,
          [_Message(role=_PredefinedRole.USER, content='Hello')],
          '<start_of_turn>user\nHello<end_of_turn>',
      ),
      (
          'gemma_user_and_model',
          formatting.FormatterName.GEMMA,
          [
              _Message(role=_PredefinedRole.USER, content='Hello'),
              _Message(role=_PredefinedRole.MODEL, content='What'),
          ],
          '<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nWhat',
      ),
      (
          'gemma_user_and_empty_model',
          formatting.FormatterName.GEMMA,
          [
              _Message(role=_PredefinedRole.USER, content='Hello'),
              _Message(role=_PredefinedRole.MODEL, content=''),
          ],
          '<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model',
      ),
  )
  def test_format(
      self,
      formatter: formatting.FormatterName,
      messages: Sequence[_Message],
      expected: str,
  ):
    async def wrapper():
      formatter_class = formatting.FORMATTER_CLASS_BY_NAME[formatter]
      formatter_instance = formatter_class()  # pytype: disable=not-instantiable
      result = formatter_instance.format(messages)
      return result

    result = asyncio.run(wrapper())
    self.assertEqual(str(result), expected)


if __name__ == '__main__':
  absltest.main()
