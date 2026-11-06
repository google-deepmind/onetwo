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

import dataclasses

from absl.testing import absltest
from onetwo.stdlib.retrieval import registry


@registry.register('test.Registered')
@dataclasses.dataclass
class _Registered:
  pass


class RegistryTest(absltest.TestCase):

  def test_register(self):
    self.assertIs(registry.get_class('test.Registered'), _Registered)
    self.assertEqual(registry.get_name(_Registered()), 'test.Registered')
    self.assertTrue(registry.is_registered(_Registered))
    self.assertFalse(registry.is_registered(object()))

  def test_registering_the_same_class_again_replaces_it(self):
    # Same module and qualified name, as when a module is reloaded.
    redefined = registry.register('test.Registered')(
        dataclasses.dataclass(
            type('_Registered', (), {'__module__': _Registered.__module__})
        )
    )
    self.addCleanup(registry.register('test.Registered'), _Registered)

    self.assertIs(registry.get_class('test.Registered'), redefined)

  def test_errors(self):
    with self.subTest('name_registered_to_another_class'):
      with self.assertRaisesRegex(ValueError, 'already registered'):

        @registry.register('test.Registered')
        @dataclasses.dataclass
        class _Other:  # pylint: disable=unused-variable
          pass

    with self.subTest('not_a_dataclass'):
      with self.assertRaisesRegex(TypeError, 'must be a dataclass'):
        registry.register('test.NotADataclass')(type('NotADataclass', (), {}))
    with self.subTest('unknown_name'):
      with self.assertRaisesRegex(KeyError, "Unknown class 'missing'"):
        registry.get_class('missing')
    with self.subTest('unregistered_class'):
      with self.assertRaisesRegex(KeyError, 'decorate it with'):
        registry.get_name(object())


if __name__ == '__main__':
  absltest.main()
