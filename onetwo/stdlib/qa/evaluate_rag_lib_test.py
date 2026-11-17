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
from onetwo.core import executing as ot_executing
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.stdlib.qa import evaluate_rag_lib
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults


class _CustomLoader(datasets.DatasetLoader):
  """An unregistered dataset loader."""

  @ot_executing.make_executable
  async def load_examples(self):
    return [{'question': 'q1', 'answer': 'a1', 'metadata': {}}]


class GetResultsDirectoryTest(absltest.TestCase):

  def test_in_index_directory_by_default(self):
    self.assertEqual(
        evaluate_rag_lib.get_results_directory(
            output=None,
            index='/indexes/browsecomp_plus/my_index',
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
            index='/indexes/browsecomp_plus/my_index',
            dataset_name='browsecomp_plus',
            qa_strategy_name='rag',
            generate_model_name='gemini',
            max_results=20,
        ),
        epath.Path('/results/browsecomp_plus/rag/gemini__max_results_20'),
    )

  def test_without_directories_returns_none_and_warns(self):
    for index in ('', indexing.EmptyIndex()):
      with self.subTest(index=index):
        with self.assertLogs(level='WARNING'):
          self.assertIsNone(
              evaluate_rag_lib.get_results_directory(
                  output=None,
                  index=index,
                  dataset_name='browsecomp_plus',
                  qa_strategy_name='rag',
                  generate_model_name='gemini',
                  max_results=20,
              )
          )

  def test_examples_name(self):
    custom_with_name = _CustomLoader()
    custom_with_name.dataset_name = 'my_examples'
    for examples, expected_name in (
        (
            datasets.get_dataset_loader('hotpotqa_validation'),
            'hotpotqa_validation',
        ),
        (_CustomLoader(), '_CustomLoader'),
        (custom_with_name, 'my_examples'),
        ('/some/path/my_examples.jsonl', 'my_examples'),
        ('relative/my_examples.json', 'my_examples'),
    ):
      with self.subTest(expected_name):
        self.assertEqual(
            evaluate_rag_lib._examples_name(examples), expected_name
        )

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  def test_evaluate_rag_loads_from_path(self, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    examples_file = self.create_tempdir().create_file(
        'custom_qa.jsonl',
        content='{"question": "q1", "answer": "a1"}\n',
    )
    out_dir = self.create_tempdir().full_path

    evaluate_rag_lib.evaluate_rag(
        examples=examples_file.full_path,
        output=out_dir,
        index='/idx',
        generate_model_name='m',
        metric_functions={},
    )

    self.assertEqual(
        mock_eval_qa.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )
    self.assertTrue(
        (
            epath.Path(out_dir) / 'custom_qa/rag/m__max_results_20/results.html'
        ).exists()
    )

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  def test_evaluate_rag_cache(self, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    out_dir = self.create_tempdir().full_path
    cache_file = (
        epath.Path(out_dir) / '_CustomLoader/rag/m__max_results_20/cache.json'
    )

    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=out_dir,
        index='/idx',
        generate_model_name='m',
        metric_functions={},
        cache=False,
    )
    self.assertEqual(
        mock_eval_qa.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )
    self.assertFalse(cache_file.exists())
    self.assertEqual(
        mock_eval_qa.call_args.kwargs['runtime_kwargs'],
        {'max_retries': 5, 'http_options': {'timeout': 300000}},
    )

    # First run with cache=True creates cache.json; second run loads it.
    for _ in range(2):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=out_dir,
          index='/idx',
          generate_model_name='m',
          metric_functions={},
          cache=True,
      )
      self.assertTrue(cache_file.exists())
      self.assertIn('cache', mock_eval_qa.call_args.kwargs['runtime_kwargs'])

  def test_get_default_extra_backend_kwargs(self):
    self.assertEqual(
        evaluate_rag_lib.get_default_extra_backend_kwargs(None),
        {'max_retries': 5, 'http_options': {'timeout': 300000}},
    )
    self.assertEqual(
        evaluate_rag_lib.get_default_extra_backend_kwargs(
            {'max_retries': 1, 'http_options': {'timeout': 10}},
            max_retries=3,
            http_timeout_ms=100,
        ),
        {'max_retries': 1, 'http_options': {'timeout': 10}},
    )
    self.assertEqual(
        evaluate_rag_lib.get_default_extra_backend_kwargs(
            {'max_retries': 1, 'http_options': None}
        ),
        {'max_retries': 1},
    )
    runtime_kwargs = {'http_options': {'base_url': 'url'}}
    self.assertEqual(
        evaluate_rag_lib.get_default_extra_backend_kwargs(
            runtime_kwargs, max_retries=3, http_timeout_ms=100
        ),
        {'max_retries': 3, 'http_options': {'base_url': 'url', 'timeout': 100}},
    )
    self.assertEqual(runtime_kwargs, {'http_options': {'base_url': 'url'}})

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  def test_evaluate_rag_default_metrics(self, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=self.create_tempdir().full_path,
        index='/idx',
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
  def test_evaluate_rag_generate_backend_config(self, mock_eval_qa):
    mock_eval_qa.return_value = results.EvaluationSummary()
    # No metrics, so that no AI rater backend (needing an API key) is built.
    with self.subTest('default_model_uses_default_kwargs'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index='/idx',
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
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index='/idx',
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
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index='/idx',
          metric_functions={},
          generate_backend_kwargs=high_thinking,
      )
      config = mock_eval_qa.call_args.kwargs['rag_spec'].generate_backend_config
      self.assertEqual(
          config.model_name, rag_defaults.DEFAULT_QA_GENERATE_MODEL_NAME
      )
      self.assertEqual(config.kwargs, high_thinking)

  @mock.patch.object(qa_evaluation, 'save_html_report')
  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  def test_evaluate_rag_with_index_object(self, mock_eval_qa, mock_save_html):
    mock_eval_qa.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()
    with self.assertLogs(level='WARNING'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=None,
          index=index,
          metric_functions={},
      )
    self.assertIs(mock_eval_qa.call_args.kwargs['index'], index)
    mock_save_html.assert_not_called()

  @mock.patch.object(qa_evaluation, 'evaluate_qa')
  @mock.patch.object(
      evaluate_rag_lib, '_load_examples', return_value=[{'question': 'q'}]
  )
  def test_evaluate_rag_empty_index_validation(
      self, mock_load_examples, mock_eval_qa
  ):
    mock_eval_qa.return_value = results.EvaluationSummary()
    out_dir = self.create_tempdir().full_path

    for strategy in ('rag', 'retrieval_only'):
      with self.subTest(strategy=strategy):
        with self.assertRaisesRegex(ValueError, 'index must be provided'):
          evaluate_rag_lib.evaluate_rag(
              examples=_CustomLoader(),
              output=out_dir,
              qa_strategy_name=strategy,
          )
    mock_load_examples.assert_not_called()

    for strategy in ('agentic_retrieval', 'closed_book'):
      with self.subTest(strategy=strategy):
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=out_dir,
            qa_strategy_name=strategy,
            metric_functions={},
        )
    self.assertEqual(mock_eval_qa.call_count, 2)
if __name__ == '__main__':
  absltest.main()
