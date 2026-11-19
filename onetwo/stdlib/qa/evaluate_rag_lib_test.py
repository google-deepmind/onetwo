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
from absl.testing import parameterized
from etils import epath
from onetwo.backends import backend_registry
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
from onetwo.stdlib.tool_use import llm_tool_use


class _CustomLoader(datasets.DatasetLoader):
  """An unregistered dataset loader."""

  @ot_executing.make_executable
  async def load_examples(self):
    return [{'question': 'q1', 'answer': 'a1', 'metadata': {}}]


def _dummy_tool(query: str) -> str:
  """Dummy tool for agentic retrieval tests."""
  return query


class GetResultsDirectoryTest(parameterized.TestCase):

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
    for index in (None, '', indexing.EmptyIndex()):
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
        evaluate_kwargs={'metric_functions': {}},
    )

    self.assertEqual(
        mock_eval.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )
    default_model = rag_defaults.DEFAULT_GENERATE_MODEL
    self.assertTrue(
        (
            epath.Path(out_dir)
            / f'custom_qa/rag/{default_model}__max_results_20/results.html'
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
        evaluate_kwargs={'metric_functions': {}},
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
        strategy=strategies_lib.RetrievalOnlyStrategy(max_results=50),
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
            'mrr@5',
            'mrr@10',
            'mrr@20',
        ],
    )

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_preset_and_strategy_object_configs(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()
    default_config = rag_defaults.default_qa_generate_backend_config()

    with self.subTest('preset_gets_default_qa_generate_backend_config'):
      for preset in (evaluate_rag_lib.StrategyType.RAG, 'rag'):
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=self.create_tempdir().full_path,
            index=index,
            strategy=preset,  # pyrefly: ignore[bad-argument-type]
            extra_backend_kwargs={'api_key': 'eval-key'},
            evaluate_kwargs={'metric_functions': {}},
        )
        built = mock_eval.call_args.kwargs['strategy']
        self.assertIsInstance(built, strategies_lib.RAGStrategy)
        self.assertEqual(built.backend_config, default_config)
        self.assertEqual(built.max_results, 20)
        self.assertIs(built.retriever, index)
        self.assertIsNotNone(built.runtime_kwargs)
        self.assertEqual(built.runtime_kwargs['api_key'], 'eval-key')

    with self.subTest('unknown_strategy_name_raises_value_error'):
      with self.assertRaises(ValueError):
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=self.create_tempdir().full_path,
            index=index,
            strategy='long_context',  # pyrefly: ignore[bad-argument-type]
            evaluate_kwargs={'metric_functions': {}},
        )

    with self.subTest('retrieval_only_preset_defaults_max_results_to_20'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          strategy=evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
          evaluate_kwargs={'metric_functions': {}},
      )
      built = mock_eval.call_args.kwargs['strategy']
      self.assertIsInstance(built, strategies_lib.RetrievalOnlyStrategy)
      self.assertEqual(built.max_results, 20)

    with self.subTest('unbounded_strategy_object_preserves_none_max_results'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          strategy=strategies_lib.RAGStrategy(),
          evaluate_kwargs={'metric_functions': {}},
      )
      built = mock_eval.call_args.kwargs['strategy']
      self.assertIsInstance(built, strategies_lib.RAGStrategy)
      self.assertIsNone(built.max_results)

    with self.subTest('object_with_none_generator_gets_default_on_copy'):
      for caller_strategy, idx in (
          (strategies_lib.RAGStrategy(max_results=50), index),
          (
              agentic_retrieval.AgenticRetrievalStrategy(
                  max_steps=30, tools=[_dummy_tool]
              ),
              None,
          ),
      ):
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=self.create_tempdir().full_path,
            index=idx,
            strategy=caller_strategy,
            extra_backend_kwargs={'api_key': 'eval-key'},
            evaluate_kwargs={'metric_functions': {}},
        )
        built = mock_eval.call_args.kwargs['strategy']
        self.assertEqual(built.backend_config, default_config)
        self.assertEqual(built.runtime_kwargs['api_key'], 'eval-key')
        self.assertIsNone(caller_strategy.backend_config)
        self.assertIsNone(caller_strategy.runtime_kwargs)

    with self.subTest('object_with_explicit_config_keeps_it_without_mutating'):
      custom_config = backend_registry.GenerateBackendConfig(
          backend_class_name='GoogleGenAIAPI',
          model_name='custom-model',
          kwargs={'temperature': 0.2},
      )
      caller_strategy = strategies_lib.RAGStrategy(
          max_results=10,
          backend_config=custom_config,
      )
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          index=index,
          strategy=caller_strategy,
          extra_backend_kwargs={'api_key': 'eval-key'},
          evaluate_kwargs={'metric_functions': {}},
      )
      built = mock_eval.call_args.kwargs['strategy']
      self.assertIsInstance(built, strategies_lib.RAGStrategy)
      self.assertEqual(built.backend_config, custom_config)
      self.assertEqual(built.max_results, 10)
      self.assertIs(built.retriever, index)
      self.assertIsNotNone(built.runtime_kwargs)
      self.assertEqual(built.runtime_kwargs['api_key'], 'eval-key')
      self.assertIsNone(caller_strategy.retriever)
      self.assertIsNone(caller_strategy.runtime_kwargs)

    with self.subTest('non_empty_strategy_runtime_kwargs_preserved'):
      caller_strategy = strategies_lib.ClosedBookStrategy(
          runtime_kwargs={'api_key': 'strategy-key'},
      )
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=self.create_tempdir().full_path,
          strategy=caller_strategy,
          extra_backend_kwargs={'api_key': 'eval-key'},
          evaluate_kwargs={'metric_functions': {}},
      )
      built = mock_eval.call_args.kwargs['strategy']
      self.assertIsInstance(built, strategies_lib.ClosedBookStrategy)
      self.assertEqual(built.backend_config, default_config)
      self.assertEqual(built.runtime_kwargs, {'api_key': 'strategy-key'})
      self.assertIsNone(caller_strategy.backend_config)

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

    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=self.create_tempdir().full_path,
        index=indexing.EmptyIndex(),
        evaluate_kwargs={
            'metric_functions': custom_metrics,
            'output_results': False,
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
  def test_results_directory_derived_from_resolved_strategy(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()
    default_model = rag_defaults.DEFAULT_GENERATE_MODEL
    custom_cfg = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI', model_name='gemini-custom'
    )
    cases = [
        (
            'rag_preset',
            evaluate_rag_lib.StrategyType.RAG,
            index,
            f'rag/{default_model}__max_results_20',
        ),
        (
            'retrieval_only_preset',
            evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
            index,
            'retrieval_only/none__max_results_20',
        ),
        (
            'closed_book_preset',
            evaluate_rag_lib.StrategyType.CLOSED_BOOK,
            None,
            f'closed_book/{default_model}__max_results_none',
        ),
        (
            'rag_object_custom',
            strategies_lib.RAGStrategy(
                max_results=10, backend_config=custom_cfg
            ),
            index,
            'RAGStrategy/gemini-custom__max_results_10',
        ),
        (
            'rag_object_unbounded',
            strategies_lib.RAGStrategy(),
            index,
            f'RAGStrategy/{default_model}__max_results_none',
        ),
    ]
    for name, strategy, idx, expected_rel in cases:
      with self.subTest(name):
        out_dir = epath.Path(self.create_tempdir().full_path)
        evaluate_rag_lib.evaluate_rag(
            examples=_CustomLoader(),
            output=str(out_dir),
            index=idx,
            strategy=strategy,
            evaluate_kwargs={'metric_functions': {}},
        )
        self.assertTrue(
            (out_dir / '_CustomLoader' / expected_rel / 'results.html').exists()
        )

  def test_with_context_sets_retriever_without_mutating_caller(self):
    loaded_index = indexing.EmptyIndex()
    caller = strategies_lib.RAGStrategy()
    with mock.patch.object(
        index_building, 'load_index', return_value=loaded_index
    ) as mock_load:
      updated = evaluate_rag_lib._with_context(
          caller, '/saved/index', (), {'api_key': 'k'}
      )
    self.assertIsInstance(updated, strategies_lib.RAGStrategy)
    mock_load.assert_called_once_with(
        '/saved/index', extra_backend_kwargs={'api_key': 'k'}
    )
    self.assertIs(updated.retriever, loaded_index)
    self.assertIsNone(caller.retriever)

  def test_with_context_replaces_existing_retriever_with_warning(self):
    existing_index = indexing.EmptyIndex()
    new_index = indexing.EmptyIndex()
    caller = strategies_lib.RAGStrategy(
        retriever=existing_index,  # pyrefly: ignore[bad-argument-type]
    )
    with self.assertLogs(level='WARNING') as cm:
      replaced = evaluate_rag_lib._with_context(caller, new_index, (), {})
    self.assertIsInstance(replaced, strategies_lib.RAGStrategy)
    self.assertIs(replaced.retriever, new_index)
    self.assertIs(caller.retriever, existing_index)
    self.assertTrue(
        any('RAGStrategy has a retriever' in msg for msg in cm.output)
    )

  def test_with_context_missing_raises_or_keeps_own(self):
    own_index = indexing.EmptyIndex()
    rag_with_own = strategies_lib.RAGStrategy(
        retriever=own_index,  # pyrefly: ignore[bad-argument-type]
    )
    self.assertIs(
        evaluate_rag_lib._with_context(rag_with_own, None, (), {}), rag_with_own
    )

    agentic_with_own = agentic_retrieval.AgenticRetrievalStrategy(
        tools=[_dummy_tool]
    )
    self.assertIs(
        evaluate_rag_lib._with_context(agentic_with_own, None, (), {}),
        agentic_with_own,
    )

    closed_book = strategies_lib.ClosedBookStrategy()
    self.assertIs(
        evaluate_rag_lib._with_context(closed_book, None, (), {}), closed_book
    )

    with mock.patch.object(
        evaluate_rag_lib, '_load_examples'
    ) as mock_load_examples:
      for bad_strategy, expected_msg in (
          (evaluate_rag_lib.StrategyType.RAG, 'RAGStrategy needs an index'),
          (
              evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
              'RetrievalOnlyStrategy needs an index',
          ),
          (
              evaluate_rag_lib.StrategyType.AGENTIC_RETRIEVAL,
              'AgenticRetrievalStrategy needs mcp_urls',
          ),
      ):
        with self.subTest(strategy=bad_strategy):
          with self.assertRaisesRegex(ValueError, expected_msg):
            evaluate_rag_lib.evaluate_rag(
                examples=_CustomLoader(),
                index=None,
                strategy=bad_strategy,
                evaluate_kwargs={'metric_functions': {}},
            )
      mock_load_examples.assert_not_called()

  def test_with_context_mcp_urls_appends_tools_and_sets_default_doc_ids(self):
    urls = ['http://localhost:8775/mcp', 'https://example.com:8775/mcp']
    caller = agentic_retrieval.AgenticRetrievalStrategy(tools=[_dummy_tool])
    updated = evaluate_rag_lib._with_context(caller, None, urls, {})
    self.assertIsInstance(
        updated, agentic_retrieval.AgenticRetrievalStrategy
    )
    self.assertLen(updated.tools, 3)
    self.assertIs(updated.tools[0], _dummy_tool)
    self.assertIsInstance(updated.tools[1], llm_tool_use.ToolSource)
    self.assertIsInstance(updated.tools[2], llm_tool_use.ToolSource)
    self.assertIs(
        updated.doc_ids_from_answer, agentic_retrieval.extract_cited_doc_ids
    )
    self.assertEqual(list(caller.tools), [_dummy_tool])
    self.assertIsNone(caller.doc_ids_from_answer)

    custom_parser = lambda answer: ['custom']
    caller_custom_parser = agentic_retrieval.AgenticRetrievalStrategy(
        doc_ids_from_answer=custom_parser
    )
    updated_custom = evaluate_rag_lib._with_context(
        caller_custom_parser, None, [urls[0]], {}
    )
    self.assertIsInstance(
        updated_custom, agentic_retrieval.AgenticRetrievalStrategy
    )
    self.assertIs(updated_custom.doc_ids_from_answer, custom_parser)

  @parameterized.named_parameters(
      dict(
          testcase_name='rag_with_mcp_urls',
          strategy_factory=strategies_lib.RAGStrategy,
          index=None,
          mcp_urls=['http://localhost:8775/mcp'],
          expected='RAGStrategy does not use mcp_urls; pass an index.',
      ),
      dict(
          testcase_name='rag_with_both',
          strategy_factory=strategies_lib.RAGStrategy,
          index='/saved/index',
          mcp_urls=['http://localhost:8775/mcp'],
          expected='RAGStrategy does not use mcp_urls; pass an index.',
      ),
      dict(
          testcase_name='retrieval_only_with_both',
          strategy_factory=strategies_lib.RetrievalOnlyStrategy,
          index=indexing.EmptyIndex(),
          mcp_urls=['http://localhost:8775/mcp'],
          expected=(
              'RetrievalOnlyStrategy does not use mcp_urls; pass an index.'
          ),
      ),
      dict(
          testcase_name='agentic_with_index_path',
          strategy_factory=agentic_retrieval.AgenticRetrievalStrategy,
          index='/saved/index',
          mcp_urls=(),
          expected=(
              'AgenticRetrievalStrategy does not use index; pass mcp_urls.'
          ),
      ),
      dict(
          testcase_name='agentic_with_index_object',
          strategy_factory=agentic_retrieval.AgenticRetrievalStrategy,
          index=indexing.EmptyIndex(),
          mcp_urls=(),
          expected=(
              'AgenticRetrievalStrategy does not use index; pass mcp_urls.'
          ),
      ),
      dict(
          testcase_name='agentic_with_both',
          strategy_factory=agentic_retrieval.AgenticRetrievalStrategy,
          index='/saved/index',
          mcp_urls=['http://localhost:8775/mcp'],
          expected=(
              'AgenticRetrievalStrategy does not use index; pass mcp_urls.'
          ),
      ),
      dict(
          testcase_name='closed_book_with_index',
          strategy_factory=strategies_lib.ClosedBookStrategy,
          index='/saved/index',
          mcp_urls=(),
          expected='ClosedBookStrategy does not use index or mcp_urls.',
      ),
      dict(
          testcase_name='closed_book_with_mcp_urls',
          strategy_factory=strategies_lib.ClosedBookStrategy,
          index=None,
          mcp_urls=['http://localhost:8775/mcp'],
          expected='ClosedBookStrategy does not use index or mcp_urls.',
      ),
      dict(
          testcase_name='closed_book_with_both',
          strategy_factory=strategies_lib.ClosedBookStrategy,
          index='/saved/index',
          mcp_urls=['http://localhost:8775/mcp'],
          expected='ClosedBookStrategy does not use index or mcp_urls.',
      ),
  )
  def test_with_context_disallowed_arguments_raise_value_error(
      self, strategy_factory, index, mcp_urls, expected
  ):
    with mock.patch.object(index_building, 'load_index') as mock_load:
      with self.assertRaisesRegex(ValueError, expected):
        evaluate_rag_lib._with_context(strategy_factory(), index, mcp_urls, {})
    mock_load.assert_not_called()


if __name__ == '__main__':
  absltest.main()
