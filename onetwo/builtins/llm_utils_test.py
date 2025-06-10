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

"""Unit tests for llm_utils."""

from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized

from onetwo.builtins import llm_utils
from onetwo.core import content as content_lib

_ChunkList: TypeAlias = content_lib.ChunkList
_TokenHealingOption: TypeAlias = llm_utils.TokenHealingOption


class LlmUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('str_reply_noop', 'Bla.', 'Bla.'),
      ('str_reply_single_space', ' Bla.', 'Bla.'),
      ('str_reply_multiple_spaces', '   Bla.', 'Bla.'),
      (
          'str_with_details_noop',
          ('Bla.', {1: 2, 'a': 'b'}),
          ('Bla.', {1: 2, 'a': 'b'}),
      ),
      (
          'str_with_details_single_space',
          (' Bla.', {1: 2, 'a': 'b'}),
          ('Bla.', {1: 2, 'a': 'b'}),
      ),
      (
          'str_with_details_multiple_spaces',
          ('   Bla.', {1: 2, 'a': 'b'}),
          ('Bla.', {1: 2, 'a': 'b'}),
      ),
  )
  def test_space_heal_reply(self, reply, expected_output):
    self.assertEqual(llm_utils.space_heal_reply(reply), expected_output)

  @parameterized.named_parameters(
      (
          'no_healing',
          ' reply',
          'prompt',
          _TokenHealingOption.NONE,
          ' reply',
      ),
      (
          'no_healing_prompt_trailing_ws',
          ' reply',
          'prompt ',
          _TokenHealingOption.NONE,
          ' reply',
      ),
      (
          'space_healing_prompt_wo_trailing_ws',
          ' reply',
          'prompt',
          _TokenHealingOption.SPACE_HEALING,
          ' reply',
      ),
      (
          'space_healing_prompt_with_trailing_ws',
          '  reply',
          'prompt ',
          _TokenHealingOption.SPACE_HEALING,
          'reply',
      ),
      (
          'space_healing_chunklist_prompt_wo_trailing_ws',
          ' reply',
          _ChunkList(chunks=['12', b'13', '14']),
          _TokenHealingOption.SPACE_HEALING,
          ' reply',
      ),
      (
          'space_healing_chunklist_prompt_with_trailing_ws',
          '  reply',
          _ChunkList(chunks=['12', b'13', '14 ']),
          _TokenHealingOption.SPACE_HEALING,
          'reply',
      ),
      (
          'token_healing',
          '',
          '',
          _TokenHealingOption.TOKEN_HEALING,
          '',
      ),
  )
  def test_maybe_heal_reply(
      self, reply, prompt, healing_option, expected_output
  ):
    if healing_option == _TokenHealingOption.TOKEN_HEALING:
      # TODO: Change once token healing is implemented.
      with self.assertRaises(NotImplementedError):
        llm_utils.maybe_heal_reply(
            reply_text=reply,
            original_prompt=prompt,
            healing_option=healing_option,
        )
    else:
      result = llm_utils.maybe_heal_reply(
          reply_text=reply,
          original_prompt=prompt,
          healing_option=healing_option,
      )
      self.assertEqual(result, expected_output)

  @parameterized.named_parameters(
      (
          'no_healing',
          'prompt',
          _TokenHealingOption.NONE,
          _ChunkList(chunks=['prompt']),
      ),
      (
          'no_healing_chunklist',
          _ChunkList(chunks=['prompt']),
          _TokenHealingOption.NONE,
          _ChunkList(chunks=['prompt']),
      ),
      (
          'space_healing_prompt_wo_trailing_ws',
          'prompt',
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['prompt']),
      ),
      (
          'space_healing_prompt_with_trailing_ws',
          'prompt ',
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['prompt']),
      ),
      (
          'space_healing_chunklist_prompt_wo_trailing_ws',
          _ChunkList(chunks=['12', b'13', '14']),
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
      ),
      (
          'space_healing_chunklist_prompt_with_trailing_ws',
          _ChunkList(chunks=['12', b'13', '14 ']),
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
      ),
      (
          'space_healing_chunklist_prompt_with_trailing_ws_across_chunks',
          _ChunkList(chunks=['12', b'13', '14 ', '   ']),
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
      ),
      (
          'token_healing',
          '',
          _TokenHealingOption.TOKEN_HEALING,
          '',
      ),
  )
  def test_maybe_heal_prompt(
      self, prompt, healing_option, expected_output
  ):
    if healing_option == _TokenHealingOption.TOKEN_HEALING:
      # TODO: Change once token healing is implemented.
      with self.assertRaises(NotImplementedError):
        llm_utils.maybe_heal_prompt(
            original_prompt=prompt,
            healing_option=healing_option,
        )
    else:
      result = llm_utils.maybe_heal_prompt(
          original_prompt=prompt,
          healing_option=healing_option,
      )
      self.assertEqual(result, expected_output)

  @parameterized.named_parameters(
      (
          'no_healing',
          'prompt',
          ['a', ' b', '  c'],
          _TokenHealingOption.NONE,
          _ChunkList(chunks=['prompt']),
          ['a', ' b', '  c'],
      ),
      (
          'no_healing_chunklist',
          _ChunkList(chunks=['prompt']),
          ['a', ' b', '  c'],
          _TokenHealingOption.NONE,
          _ChunkList(chunks=['prompt']),
          ['a', ' b', '  c'],
      ),
      (
          'space_healing_prompt_wo_trailing_ws',
          'prompt',
          ['a', ' b', '  c'],
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['prompt']),
          ['a', ' b', '  c'],
      ),
      (
          'space_healing_prompt_with_trailing_ws',
          'prompt ',
          ['a', ' b', '  c'],
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['prompt']),
          [' a', ' b', '  c'],
      ),
      (
          'space_healing_chunklist_prompt_wo_trailing_ws',
          _ChunkList(chunks=['12', b'13', '14']),
          ['a', ' b', '  c'],
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
          ['a', ' b', '  c'],
      ),
      (
          'space_healing_chunklist_prompt_with_trailing_ws',
          _ChunkList(chunks=['12', b'13', '14 ']),
          ['a', ' b', '  c'],
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
          [' a', ' b', '  c'],
      ),
      (
          'space_healing_chunklist_prompt_with_trailing_ws_across_chunks',
          _ChunkList(chunks=['12', b'13', '14 ', '   ']),
          ['a', ' b', '  c'],
          _TokenHealingOption.SPACE_HEALING,
          _ChunkList(chunks=['12', b'13', '14']),
          [' a', ' b', '  c'],
      ),
      (
          'token_healing',
          '',
          [],
          _TokenHealingOption.TOKEN_HEALING,
          '',
          [],
      ),
  )
  def test_maybe_heal_prompt_and_targets(
      self, prompt, targets, healing_option, expected_prompt, expected_targets
  ):
    if healing_option == _TokenHealingOption.TOKEN_HEALING:
      # TODO: Change once token healing is implemented.
      with self.assertRaises(NotImplementedError):
        llm_utils.maybe_heal_prompt_and_targets(
            original_prompt=prompt,
            original_targets=targets,
            healing_option=healing_option,
        )
    else:
      healed_prompt, healed_targets = llm_utils.maybe_heal_prompt_and_targets(
          original_prompt=prompt,
          original_targets=targets,
          healing_option=healing_option,
      )
      self.assertEqual(healed_prompt, expected_prompt)
      self.assertSequenceEqual(healed_targets, expected_targets)

if __name__ == '__main__':
  absltest.main()
