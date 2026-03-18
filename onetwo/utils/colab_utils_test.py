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

from absl.testing import absltest
from onetwo.utils import colab_utils


class ColabUtilsTest(absltest.TestCase):

  def test_cached_backends_enables_prints(self):
    own_cache_dir = self.create_tempdir()
    cb = colab_utils.CachedBackends(own_cache_directory=own_cache_dir.full_path)
    self.assertTrue(cb.enable_print_statements)


if __name__ == '__main__':
  absltest.main()
