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
from etils import epath
from onetwo.stdlib.qa import evaluate_rag_lib


class GetResultsDirectoryTest(absltest.TestCase):

  def test_in_index_directory_by_default(self):
    self.assertEqual(
        evaluate_rag_lib.get_results_directory(
            output_directory=None,
            index_directory='/indexes/browsecomp_plus/my_index',
            dataset_name='browsecomp_plus',
            qa_strategy_name='rag',
            generate_model_name='gemini',
            max_results=20,
        ),
        epath.Path(
            '/indexes/browsecomp_plus/my_index/eval_results/rag/'
            'gemini__max_results_20'
        ),
    )

  def test_in_output_directory_if_set(self):
    self.assertEqual(
        evaluate_rag_lib.get_results_directory(
            output_directory='/results',
            index_directory='/indexes/browsecomp_plus/my_index',
            dataset_name='browsecomp_plus',
            qa_strategy_name='rag',
            generate_model_name='gemini',
            max_results=20,
        ),
        epath.Path('/results/browsecomp_plus/rag/gemini__max_results_20'),
    )

  def test_without_directories_raises(self):
    with self.assertRaisesRegex(ValueError, 'output_directory'):
      evaluate_rag_lib.get_results_directory(
          output_directory=None,
          index_directory='',
          dataset_name='browsecomp_plus',
          qa_strategy_name='rag',
          generate_model_name='gemini',
          max_results=20,
      )


if __name__ == '__main__':
  absltest.main()
