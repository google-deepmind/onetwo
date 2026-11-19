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

import asyncio
from unittest import mock

from absl.testing import absltest
from etils import epath
from onetwo.core import caching
from onetwo.core import executing as ot_executing
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import evaluation as ot_evaluation
from onetwo.stdlib.qa import agentic_retrieval
from onetwo.stdlib.qa import evaluate_rag_lib
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import index_building
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

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_loads_from_path(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    examples_file = self.create_tempdir().create_file(
        'custom_qa.jsonl',
        content='{"question": "q1", "answer": "a1"}\n',
    )
    out_dir = self.create_tempdir().full_path

    evaluate_rag_lib.evaluate_rag(
        examples=examples_file.full_path,
        output=out_dir,
        index=indexing.EmptyIndex(),
        generate_model_name='m',
        evaluate_kwargs={'metric_functions': {}},
    )

    self.assertEqual(
        mock_eval.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )
    self.assertTrue(
        (
            epath.Path(out_dir) / 'custom_qa/rag/m__max_results_20/results.html'
        ).exists()
    )

  @mock.patch.object(
      index_building, 'load_index', return_value=indexing.EmptyIndex()
  )
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_cache(self, mock_eval, mock_load_index):
    mock_eval.return_value = results.EvaluationSummary()
    out_dir = self.create_tempdir().full_path
    cache_file = epath.Path(self.create_tempdir().full_path) / 'cache.json'

    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=out_dir,
        index='/idx',
        generate_model_name='m',
        evaluate_kwargs={'metric_functions': {}},
    )
    self.assertEqual(
        mock_eval.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )
    self.assertFalse(cache_file.exists())
    self.assertEqual(
        mock_load_index.call_args.kwargs['extra_backend_kwargs'],
        {'max_retries': 5, 'http_options': {'timeout': 300000}},
    )

    def _populate_cache(**kwargs):
      strategy = kwargs['strategy']
      fn_cache = strategy.runtime_kwargs['cache']
      asyncio.run(fn_cache.cache_value('k1', None, 'v1'))
      return results.EvaluationSummary()

    mock_eval.side_effect = _populate_cache
    extra_backend_kwargs = {
        'api_key': 'secret',
        'cache_filename': str(cache_file),
    }
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=out_dir,
        index='/idx',
        extra_backend_kwargs=extra_backend_kwargs,
        generate_model_name='m',
        evaluate_kwargs={'metric_functions': {}},
    )
    self.assertTrue(cache_file.exists())
    first_rt_kwargs = mock_eval.call_args.kwargs['strategy'].runtime_kwargs
    self.assertIsInstance(first_rt_kwargs['cache'], caching.SimpleFunctionCache)
    self.assertIs(
        mock_load_index.call_args.kwargs['extra_backend_kwargs']['cache'],
        first_rt_kwargs['cache'],
    )
    self.assertNotIn('cache_filename', first_rt_kwargs)
    self.assertEqual(
        extra_backend_kwargs,
        {'api_key': 'secret', 'cache_filename': str(cache_file)},
    )

    # Second run loads the saved cache from disk into a fresh cache object.
    mock_eval.side_effect = None
    mock_eval.return_value = results.EvaluationSummary()
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=out_dir,
        index='/idx',
        extra_backend_kwargs=extra_backend_kwargs,
        generate_model_name='m',
        evaluate_kwargs={'metric_functions': {}},
    )
    second_rt_kwargs = mock_eval.call_args.kwargs['strategy'].runtime_kwargs
    second_cache = second_rt_kwargs['cache']
    self.assertIsInstance(second_cache, caching.SimpleFunctionCache)
    self.assertIsNot(second_cache, first_rt_kwargs['cache'])
    self.assertEqual(second_cache.get_key_count(), 1)
    self.assertNotIn('cache_filename', second_rt_kwargs)

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

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_default_metrics(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=self.create_tempdir().full_path,
        index=indexing.EmptyIndex(),
        qa_strategy_name='retrieval_only',
        query_time_kwargs={'max_results': 50},
    )
    evaluate_kwargs = mock_eval.call_args.kwargs
    self.assertTrue(evaluate_kwargs['output_results'])
    self.assertNotIn('enable_tracing', evaluate_kwargs)
    self.assertCountEqual(
        evaluate_kwargs['metric_functions'].keys(),
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
    self.assertEqual(
        evaluate_kwargs['inputs_extractor']({'question': 'q?'}),
        (['q?'], {'max_results': 50}),
    )

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_generate_backend_config(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()
    # No metrics, so that no AI rater backend (needing an API key) is built.
    with self.subTest('default_model_uses_default_kwargs'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          evaluate_kwargs={'metric_functions': {}},
      )
      config = mock_eval.call_args.kwargs['strategy'].backend_config
      self.assertEqual(
          config, rag_defaults.default_qa_generate_backend_config()
      )
    with self.subTest('other_model_has_no_kwargs'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          evaluate_kwargs={'metric_functions': {}},
          generate_model_name='some-other-model',
      )
      config = mock_eval.call_args.kwargs['strategy'].backend_config
      self.assertEqual(config.model_name, 'some-other-model')
      self.assertEqual(config.kwargs, {})
    with self.subTest('explicit_kwargs_replace_defaults'):
      high_thinking = {
          'chat_kwargs': {'thinking_config': {'thinking_level': 'HIGH'}}
      }
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          evaluate_kwargs={'metric_functions': {}},
          generate_backend_kwargs=high_thinking,
      )
      config = mock_eval.call_args.kwargs['strategy'].backend_config
      self.assertEqual(
          config.model_name,
          rag_defaults.default_qa_generate_backend_config().model_name,
      )
      self.assertEqual(config.kwargs, high_thinking)

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_custom_evaluate_kwargs(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    custom_metrics = {'exact': lambda target, prediction: 1.0}
    evaluate_kwargs = {
        'metric_functions': custom_metrics,
        'enable_tracing': False,
        'chunk_size': 10,
    }

    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=self.create_tempdir().full_path,
        index=indexing.EmptyIndex(),
        evaluate_kwargs=evaluate_kwargs,
    )

    forwarded = dict(mock_eval.call_args.kwargs)
    forwarded.pop('strategy')
    forwarded.pop('examples')
    inputs_extractor = forwarded.pop('inputs_extractor')
    self.assertEqual(
        inputs_extractor({'question': 'q?'}), (['q?'], {'max_results': 20})
    )
    self.assertEqual(
        forwarded,
        {
            'output_results': True,
            'enable_tracing': False,
            'metric_functions': custom_metrics,
            'chunk_size': 10,
        },
    )
    self.assertNotIn('output_results', evaluate_kwargs)
    self.assertNotIn('inputs_extractor', evaluate_kwargs)

    custom_extractor = lambda x: ([x['q']], {})
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=self.create_tempdir().full_path,
        index=indexing.EmptyIndex(),
        evaluate_kwargs={
            'metric_functions': custom_metrics,
            'output_results': False,
            'inputs_extractor': custom_extractor,
        },
    )
    forwarded = dict(mock_eval.call_args.kwargs)
    forwarded.pop('strategy')
    forwarded.pop('examples')
    self.assertEqual(
        forwarded,
        {
            'metric_functions': custom_metrics,
            'output_results': False,
            'inputs_extractor': custom_extractor,
        },
    )

  @mock.patch.object(index_building, 'load_index')
  @mock.patch.object(qa_evaluation, 'save_html_report')
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_with_index_object(
      self, mock_eval, mock_save_html, mock_load_index
  ):
    mock_eval.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()
    with self.assertLogs(level='WARNING'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=None,
          index=index,
          evaluate_kwargs={'metric_functions': {}},
      )
    mock_load_index.assert_not_called()
    strategy = mock_eval.call_args.kwargs['strategy']
    self.assertIsInstance(strategy, strategies_lib.RAGStrategy)
    self.assertIs(strategy.retriever, index)
    mock_save_html.assert_not_called()

  @mock.patch.object(ot_evaluation, 'evaluate')
  @mock.patch.object(
      evaluate_rag_lib, '_load_examples', return_value=[{'question': 'q'}]
  )
  def test_evaluate_rag_empty_index_validation(
      self, mock_load_examples, mock_eval
  ):
    mock_eval.return_value = results.EvaluationSummary()
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

    for strategy, expected_cls in (
        ('agentic_retrieval', agentic_retrieval.AgenticRetrievalStrategy),
        ('closed_book', strategies_lib.ClosedBookStrategy),
    ):
      with self.subTest(strategy=strategy):
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=out_dir,
            qa_strategy_name=strategy,
            evaluate_kwargs={'metric_functions': {}},
        )
        self.assertIsInstance(
            mock_eval.call_args.kwargs['strategy'], expected_cls
        )
    self.assertEqual(mock_eval.call_count, 2)


if __name__ == '__main__':
  absltest.main()
