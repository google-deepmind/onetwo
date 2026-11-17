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
from onetwo.evaluation import datasets
from onetwo.stdlib.qa import evaluate_rag_lib
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.retrieval import rag_defaults


class GetResultsDirectoryTest(absltest.TestCase):

  def test_in_index_directory_by_default(self):
    self.assertEqual(
        evaluate_rag_lib.get_results_directory(
            output=None,
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
            output='/results',
            index_directory='/indexes/browsecomp_plus/my_index',
            dataset_name='browsecomp_plus',
            qa_strategy_name='rag',
            generate_model_name='gemini',
            max_results=20,
        ),
        epath.Path('/results/browsecomp_plus/rag/gemini__max_results_20'),
    )

  def test_without_directories_raises(self):
    with self.assertRaisesRegex(ValueError, 'output'):
      evaluate_rag_lib.get_results_directory(
          output=None,
          index_directory='',
          dataset_name='browsecomp_plus',
          qa_strategy_name='rag',
          generate_model_name='gemini',
          max_results=20,
      )

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  @mock.patch.object(
      datasets, 'load_examples', return_value=[{'question': 'q'}]
  )
  def test_evaluate_rag_cache(self, _, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    out_dir = self.create_tempdir().full_path
    cache_file = epath.Path(out_dir) / 'ds/rag/m__max_results_20/cache.json'

    evaluate_rag_lib.evaluate_rag(
        dataset_name='ds',
        output=out_dir,
        generate_model_name='m',
        metric_functions={},
        cache=False,
    )
    self.assertFalse(cache_file.exists())
    self.assertEqual(
        mock_eval_qa.call_args.kwargs['runtime_kwargs'],
        {'max_retries': 5, 'http_options': {'timeout': 300000}},
    )

    # First run with cache=True creates cache.json; second run loads it.
    for _ in range(2):
      evaluate_rag_lib.evaluate_rag(
          dataset_name='ds',
          output=out_dir,
          generate_model_name='m',
          metric_functions={},
          cache=True,
      )
      self.assertTrue(cache_file.exists())
      self.assertIn('cache', mock_eval_qa.call_args.kwargs['runtime_kwargs'])

  def test_get_default_runtime_kwargs(self):
    self.assertEqual(
        evaluate_rag_lib.get_default_runtime_kwargs(None),
        {'max_retries': 5, 'http_options': {'timeout': 300000}},
    )
    self.assertEqual(
        evaluate_rag_lib.get_default_runtime_kwargs(
            {'max_retries': 1, 'http_options': {'timeout': 10}},
            max_retries=3,
            http_timeout_ms=100,
        ),
        {'max_retries': 1, 'http_options': {'timeout': 10}},
    )
    self.assertEqual(
        evaluate_rag_lib.get_default_runtime_kwargs(
            {'max_retries': 1, 'http_options': None}
        ),
        {'max_retries': 1},
    )
    runtime_kwargs = {'http_options': {'base_url': 'url'}}
    self.assertEqual(
        evaluate_rag_lib.get_default_runtime_kwargs(
            runtime_kwargs, max_retries=3, http_timeout_ms=100
        ),
        {'max_retries': 3, 'http_options': {'base_url': 'url', 'timeout': 100}},
    )
    self.assertEqual(runtime_kwargs, {'http_options': {'base_url': 'url'}})

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  @mock.patch.object(
      datasets, 'load_examples', return_value=[{'question': 'q'}]
  )
  def test_evaluate_rag_default_metrics(self, _, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    evaluate_rag_lib.evaluate_rag(
        dataset_name='ds',
        output=self.create_tempdir().full_path,
        qa_strategy_name='retrieval_only',
        query_time_kwargs={'max_results': 50},
    )
    self.assertCountEqual(
        mock_eval_qa.call_args.kwargs['metric_functions'].keys(),
        [
            'recall@5',
            'recall@10',
            'recall@20',
            'recall@50',
            'mrr@5',
            'mrr@10',
            'mrr@20',
            'mrr@50',
        ],
    )

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  @mock.patch.object(
      datasets, 'load_examples', return_value=[{'question': 'q'}]
  )
  def test_evaluate_rag_generate_backend_config(self, _, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    # No metrics, so that no AI rater backend (needing an API key) is built.
    with self.subTest('default_model_uses_default_kwargs'):
      evaluate_rag_lib.evaluate_rag(
          dataset_name='ds',
          output=self.create_tempdir().full_path,
          metric_functions={},
      )
      config = mock_eval_qa.call_args.kwargs['rag_spec'].generate_backend_config
      self.assertEqual(
          config.model_name, rag_defaults.DEFAULT_QA_GENERATE_MODEL_NAME
      )
      self.assertEqual(
          config.kwargs, rag_defaults.DEFAULT_QA_GENERATE_BACKEND_KWARGS
      )
    with self.subTest('other_model_has_no_kwargs'):
      evaluate_rag_lib.evaluate_rag(
          dataset_name='ds',
          output=self.create_tempdir().full_path,
          metric_functions={},
          generate_model_name='some-other-model',
      )
      config = mock_eval_qa.call_args.kwargs['rag_spec'].generate_backend_config
      self.assertEqual(config.model_name, 'some-other-model')
      self.assertEqual(config.kwargs, {})
    with self.subTest('explicit_kwargs_replace_defaults'):
      high_thinking = {
          'chat_kwargs': {'thinking_config': {'thinking_level': 'HIGH'}}
      }
      evaluate_rag_lib.evaluate_rag(
          dataset_name='ds',
          output=self.create_tempdir().full_path,
          metric_functions={},
          generate_backend_kwargs=high_thinking,
      )
      config = mock_eval_qa.call_args.kwargs['rag_spec'].generate_backend_config
      self.assertEqual(
          config.model_name, rag_defaults.DEFAULT_QA_GENERATE_MODEL_NAME
      )
      self.assertEqual(config.kwargs, high_thinking)

if __name__ == '__main__':
  absltest.main()
