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

from unittest import mock

from absl.testing import absltest
from etils import epath
from onetwo.core import results
from onetwo.stdlib.qa import evaluate_rag_utils


class SaveHtmlReportTest(absltest.TestCase):

  def test_save_html_report_writes_html_file(self):
    out_dir = self.create_tempdir().full_path
    path = evaluate_rag_utils.save_html_report(
        summary=results.EvaluationSummary(metrics={'accuracy': 1.0}),
        output_dir_path=out_dir,
    )
    self.assertTrue(path.exists())
    self.assertEqual(path.name, 'results.html')

  def test_create_run_directory_retries_on_file_exists_race(self):
    dataset_dir = epath.Path(self.create_tempdir().full_path) / 'ds'
    first = evaluate_rag_utils._create_run_directory(dataset_dir, 'p__hash_1')
    self.assertEqual(first.name, 'p__hash_1_run1')

    real_mkdir = type(first).mkdir
    raced = False

    def _racy_mkdir(path_self, *args, **kwargs):
      nonlocal raced
      if path_self.name.startswith('p__hash_1_run'):
        # Leaf mkdir must not pass parents=True so creation is atomic on CNS.
        self.assertFalse(kwargs.get('parents', False))
        self.assertFalse(kwargs.get('exist_ok', False))
      if path_self.name == 'p__hash_1_run2' and not raced:
        raced = True
        raise FileExistsError('simulated parallel run')
      return real_mkdir(path_self, *args, **kwargs)

    with mock.patch.object(type(first), 'mkdir', _racy_mkdir):
      created = evaluate_rag_utils._create_run_directory(
          dataset_dir, 'p__hash_1'
      )
    self.assertEqual(created.name, 'p__hash_1_run3')


if __name__ == '__main__':
  absltest.main()
