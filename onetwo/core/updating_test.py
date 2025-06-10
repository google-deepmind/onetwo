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

from __future__ import annotations

import copy
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import updating


Update = updating.Update
ListUpdate = updating.ListUpdate

_S: TypeAlias = updating.AddableUpdate[str]


class UpdatingTest(parameterized.TestCase):

  def test_list_update_accumulate(self):
    u1 = Update('a')
    u2 = Update('b')
    l1 = ListUpdate([(u1, 0)])
    l2 = ListUpdate([(u2, 0)])
    l3 = copy.deepcopy(l2)

    with self.subTest('start_from_scratch'):
      self.assertEqual(Update() + l2, l3)

    with self.subTest('should_replace'):
      self.assertEqual(l1 + l2, l3)

  def test_plus_and_sum(self):
    list_updates1 = (ListUpdate([(i, i)]) for i in range(3))
    list_updates2 = (ListUpdate([(i, i)]) for i in range(3))
    update = Update()
    for u in list_updates1:
      update += u
    final = sum(list_updates2, start=Update())
    self.assertEqual(update, final)
    self.assertEqual(final.to_result(), [0, 1, 2])

  @parameterized.named_parameters(
      ('empty', ListUpdate([]), []),
      ('singleton', ListUpdate([(Update('a'), 0)]), ['a']),
      (
          'non_contiguous',
          ListUpdate([
              (Update('a'), 0),
              (Update('b'), 2),
              (Update('c'), 4),
          ]),
          ['a', None, 'b', None, 'c'],
      ),
      (
          'nested',
          ListUpdate([
              (ListUpdate([(Update('a'), 1)]), 1),
          ]),
          [None, [None, 'a']],
      ),
      (
          'nested_another_example',
          ListUpdate([
              (ListUpdate([(10, 1)]), 0),
              (ListUpdate([(20, 0)]), 2),
          ]),
          [[None, 10], None, [20,]],
      ),
      (
          'nested_deeper',
          ListUpdate([
              (
                  ListUpdate([
                      (ListUpdate([('ab', 1)]), 1)
                  ]),
                  0
              ),
              (
                  ListUpdate([
                      (ListUpdate([('cd', 0)]), 1)
                  ]),
                  2
              ),
          ]),
          [[None, [None, 'ab']], None, [None, ['cd']]],
      ),
  )
  def test_list_update_to_accumulate(self, list_update, result):
    self.assertListEqual(list_update.to_result(), result)

  def test_nested(self):
    u1 = Update('a')
    u2 = Update(u1)
    self.assertEqual(u1.to_result(), u2.to_result())

  def test_addable(self):
    l = _S('a')
    self.assertEqual(l.to_result(), 'a')
    l += 'b'
    self.assertEqual(l.to_result(), 'ab')
    l += _S('c') + 'd'
    self.assertEqual(l.to_result(), 'abcd')


if __name__ == '__main__':
  absltest.main()
