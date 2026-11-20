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
import dataclasses
import hashlib
import json
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
from onetwo.stdlib.qa import evaluate_rag_utils
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


@dataclasses.dataclass
class _StrategyWithBool(strategies_lib.ClosedBookStrategy):
  use_rerank: bool = True


class EvaluateRagLibTest(parameterized.TestCase):

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

    evaluate_rag_lib.evaluate_rag(
        examples=examples_file.full_path,
        output=None,
        index=indexing.EmptyIndex(),
        evaluate_kwargs={'metric_functions': {}},
    )

    self.assertEqual(
        mock_eval.call_args.kwargs['examples'],
        [{'question': 'q1', 'answer': 'a1', 'metadata': {}}],
    )

  @mock.patch.object(
      index_building, 'load_index', return_value=indexing.EmptyIndex()
  )
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_cache(self, mock_eval, mock_load_index):
    mock_eval.return_value = results.EvaluationSummary()
    cache_file = epath.Path(self.create_tempdir().full_path) / 'cache.json'

    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=None,
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
        output=None,
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
        output=None,
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
        output=None,
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
            output=None,
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
            output=None,
            index=index,
            strategy='long_context',  # pyrefly: ignore[bad-argument-type]
            evaluate_kwargs={'metric_functions': {}},
        )

    with self.subTest('retrieval_only_preset_defaults_max_results_to_20'):
      evaluate_rag_lib.evaluate_rag(
          examples=_CustomLoader(),
          output=None,
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
          output=None,
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
            output=None,
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
          output=None,
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
          output=None,
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
        output=None,
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
        output=None,
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

  @mock.patch.object(
      index_building, 'load_index', return_value=indexing.EmptyIndex()
  )
  @mock.patch.object(evaluate_rag_utils, 'save_html_report')
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_output_none_writes_nothing(
      self, mock_eval, mock_save_html, mock_load_index
  ):
    del mock_load_index
    mock_eval.return_value = results.EvaluationSummary(
        metrics={'recall@5': 1.0}
    )
    index_dir = epath.Path(self.create_tempdir().full_path)
    summary = evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=None,
        index=str(index_dir),
        evaluate_kwargs={'metric_functions': {}},
    )
    self.assertEqual(summary.metrics, {'recall@5': 1.0})
    mock_save_html.assert_not_called()
    self.assertEmpty(list(index_dir.iterdir()))

  @mock.patch.object(index_building, 'load_index')
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_with_index_object(self, mock_eval, mock_load_index):
    mock_eval.return_value = results.EvaluationSummary()
    out_dir = epath.Path(self.create_tempdir().full_path)
    index = indexing.EmptyIndex()
    evaluate_rag_lib.evaluate_rag(
        examples=_CustomLoader(),
        output=str(out_dir),
        index=index,
        evaluate_kwargs={'metric_functions': {}},
    )
    mock_load_index.assert_not_called()
    strategy = mock_eval.call_args.kwargs['strategy']
    self.assertIsInstance(strategy, strategies_lib.RAGStrategy)
    self.assertIs(strategy.retriever, index)
    run_dirs = list((out_dir / '_CustomLoader').iterdir())
    self.assertLen(run_dirs, 1)
    saved = json.loads((run_dirs[0] / 'eval_config.json').read_text())
    self.assertEqual(saved['index'], 'EmptyIndex')

  @mock.patch.object(
      index_building, 'load_index', return_value=indexing.EmptyIndex()
  )
  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_run_folder_names_and_eval_config(
      self, mock_eval, mock_load_index
  ):
    del mock_load_index
    mock_eval.return_value = results.EvaluationSummary()
    default_model = rag_defaults.DEFAULT_GENERATE_MODEL
    default_gen_dict = dataclasses.asdict(
        rag_defaults.default_qa_generate_backend_config()
    )
    custom_metrics = {'m1': lambda target, prediction: 1.0}
    loader = _CustomLoader()
    loader.dataset_name = 'browsecomp_plus'

    def _run_and_get_dir(
        out_dir: epath.Path,
        *,
        strategy,
        index=None,
        mcp_urls=(),
        extra_backend_kwargs=None,
    ) -> epath.Path:
      before = set(
          (out_dir / 'browsecomp_plus').iterdir()
          if (out_dir / 'browsecomp_plus').exists()
          else []
      )
      evaluate_rag_lib.evaluate_rag(
          examples=loader,
          output=str(out_dir),
          extra_backend_kwargs=extra_backend_kwargs,
          index=index,
          strategy=strategy,
          evaluate_kwargs={'metric_functions': custom_metrics},
          mcp_urls=mcp_urls,
      )
      after = set((out_dir / 'browsecomp_plus').iterdir())
      created = list(after - before)
      self.assertLen(created, 1)
      return created[0]

    def _config_hash(saved_config: dict[str, object]) -> str:
      return hashlib.sha256(
          json.dumps(saved_config, sort_keys=True).encode()
      ).hexdigest()[:8]

    with self.subTest('preset_rag_and_eval_config_omits_credentials'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      run_dir = _run_and_get_dir(
          out_dir,
          strategy=evaluate_rag_lib.StrategyType.RAG,
          index='/saved/index/dir',
          extra_backend_kwargs={
              'api_key': 'secret-api-key',
              'project': 'secret-project',
          },
      )
      self.assertCountEqual(
          [p.name for p in run_dir.iterdir()],
          ['results.html', 'eval_config.json'],
      )
      raw_config = (run_dir / 'eval_config.json').read_text()
      self.assertNotIn('secret-api-key', raw_config)
      self.assertNotIn('secret-project', raw_config)
      self.assertNotIn('extra_backend_kwargs', raw_config)
      self.assertNotIn('runtime_kwargs', raw_config)
      saved = json.loads(raw_config)
      self.assertEqual(
          saved,
          {
              'examples': 'browsecomp_plus',
              'index': '/saved/index/dir',
              'mcp_urls': [],
              'strategy': {
                  'preset': 'rag',
                  'class': 'RAGStrategy',
                  'config': {
                      'max_results': 20,
                      'min_score': None,
                      'backend_config': default_gen_dict,
                  },
              },
              'metric_functions': ['m1'],
          },
      )
      self.assertEqual(
          saved['strategy']['config']['backend_config']['model_name'],
          default_model,
      )
      for omitted in (
          'retriever',
          'tools',
          'prompt_builder',
          'doc_ids_from_answer',
      ):
        self.assertNotIn(omitted, saved['strategy']['config'])
      self.assertEqual(
          run_dir.name,
          f'rag_{default_model}_max_results=20'
          f'__hash_{_config_hash(saved)}_run1',
      )
      str_out_dir = epath.Path(self.create_tempdir().full_path)
      str_run_dir = _run_and_get_dir(
          str_out_dir,
          strategy='rag',
          index='/saved/index/dir',
      )
      self.assertEqual(str_run_dir.name, run_dir.name)

    with self.subTest('strategy_object_with_slash_in_model_and_secrets'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      strategy = strategies_lib.RAGStrategy(
          max_results=10,
          backend_config=backend_registry.GenerateBackendConfig(
              backend_class_name='GoogleGenAIAPI',
              model_name='models/gemini-x',
              kwargs={'temperature': 0.1},
          ),
          runtime_kwargs={'api_key': 'strategy-secret'},
      )
      run_dir = _run_and_get_dir(
          out_dir,
          strategy=strategy,
          index=indexing.EmptyIndex(),
          extra_backend_kwargs={'api_key': 'eval-secret'},
      )
      raw_config = (run_dir / 'eval_config.json').read_text()
      self.assertNotIn('strategy-secret', raw_config)
      self.assertNotIn('eval-secret', raw_config)
      self.assertNotIn('runtime_kwargs', raw_config)
      saved = json.loads(raw_config)
      self.assertIsNone(saved['strategy']['preset'])
      self.assertEqual(saved['strategy']['class'], 'RAGStrategy')
      self.assertEqual(
          run_dir.name,
          'RAGStrategy_models-gemini-x_max_results=10'
          f'__hash_{_config_hash(saved)}_run1',
      )

    with self.subTest('strategy_without_model_retrieval_only'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      preset_dir = _run_and_get_dir(
          out_dir,
          strategy=evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
          index=indexing.EmptyIndex(),
      )
      self.assertStartsWith(
          preset_dir.name, 'retrieval_only_max_results=20__hash_'
      )
      obj_dir = _run_and_get_dir(
          out_dir,
          strategy=strategies_lib.RetrievalOnlyStrategy(
              max_results=50, min_score=0.5
          ),
          index=indexing.EmptyIndex(),
      )
      self.assertStartsWith(
          obj_dir.name,
          'RetrievalOnlyStrategy_max_results=50_min_score=0.5__hash_',
      )

    with self.subTest('agentic_retrieval_omits_text_fields_and_hashes_them'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      url = 'http://localhost:8775/mcp'
      run1 = _run_and_get_dir(
          out_dir,
          strategy=agentic_retrieval.AgenticRetrievalStrategy(
              max_steps=30,
              additional_instructions='Use terse answers.',
              final_instruction='Cite sources.',
          ),
          mcp_urls=[url],
      )
      run2 = _run_and_get_dir(
          out_dir,
          strategy=agentic_retrieval.AgenticRetrievalStrategy(
              max_steps=30,
              additional_instructions='Give detailed answers.',
              final_instruction='Cite sources.',
          ),
          mcp_urls=[url],
      )
      run_other_mcp = _run_and_get_dir(
          out_dir,
          strategy=agentic_retrieval.AgenticRetrievalStrategy(
              max_steps=30,
              additional_instructions='Use terse answers.',
              final_instruction='Cite sources.',
          ),
          mcp_urls=['http://localhost:8776/mcp'],
      )
      prefix1, hash1 = run1.name.split('__hash_')
      prefix2, hash2 = run2.name.split('__hash_')
      prefix3, hash3 = run_other_mcp.name.split('__hash_')
      self.assertEqual(
          prefix1, f'AgenticRetrievalStrategy_{default_model}_max_steps=30'
      )
      self.assertEqual(prefix1, prefix2)
      self.assertEqual(prefix1, prefix3)
      self.assertNotEqual(hash1, hash2)
      self.assertNotEqual(hash1, hash3)
      self.assertNotIn('terse', run1.name)
      saved = json.loads((run1 / 'eval_config.json').read_text())
      self.assertIsNone(saved['index'])
      self.assertEqual(saved['mcp_urls'], [url])
      for omitted in (
          'tools',
          'retriever',
          'prompt_builder',
          'doc_ids_from_answer',
      ):
        self.assertNotIn(omitted, saved['strategy']['config'])

    with self.subTest('same_settings_twice_increments_run_and_different_index'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      run1 = _run_and_get_dir(
          out_dir, strategy=evaluate_rag_lib.StrategyType.RAG, index='/idx/v1'
      )
      run2 = _run_and_get_dir(
          out_dir, strategy=evaluate_rag_lib.StrategyType.RAG, index='/idx/v1'
      )
      run_other_idx = _run_and_get_dir(
          out_dir, strategy=evaluate_rag_lib.StrategyType.RAG, index='/idx/v2'
      )
      self.assertEndsWith(run1.name, '_run1')
      self.assertEndsWith(run2.name, '_run2')
      self.assertEqual(run1.name[:-1], run2.name[:-1])
      self.assertEqual(
          run1.name.split('__hash_')[0], run_other_idx.name.split('__hash_')[0]
      )
      self.assertNotEqual(
          run1.name.split('__hash_')[1], run_other_idx.name.split('__hash_')[1]
      )

    with self.subTest('every_preset_writes_eval_config_json'):
      preset_cases = (
          (evaluate_rag_lib.StrategyType.CLOSED_BOOK, None, (), True),
          (
              evaluate_rag_lib.StrategyType.RAG,
              indexing.EmptyIndex(),
              (),
              True,
          ),
          (
              evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
              indexing.EmptyIndex(),
              (),
              False,
          ),
          (
              evaluate_rag_lib.StrategyType.AGENTIC_RETRIEVAL,
              None,
              ['http://localhost:8775/mcp'],
              True,
          ),
      )
      for preset, idx, urls, has_backend in preset_cases:
        out_dir = epath.Path(self.create_tempdir().full_path)
        run_dir = _run_and_get_dir(
            out_dir, strategy=preset, index=idx, mcp_urls=urls
        )
        saved = json.loads((run_dir / 'eval_config.json').read_text())
        self.assertEqual(saved['strategy']['preset'], preset.value)
        config = saved['strategy']['config']
        self.assertNotIn('prompt_builder', config)
        self.assertNotIn('doc_ids_from_answer', config)
        if has_backend:
          self.assertIsInstance(config['backend_config'], dict)
          self.assertEqual(
              config['backend_config']['model_name'], default_model
          )

    with self.subTest('bool_included_and_backend_class_fallback'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      run_dir = _run_and_get_dir(
          out_dir,
          strategy=_StrategyWithBool(
              use_rerank=False,
              backend_config=backend_registry.GenerateBackendConfig(
                  backend_class_name='OllamaAPI', model_name=None
              ),
          ),
          index=None,
      )
      self.assertStartsWith(
          run_dir.name,
          '_StrategyWithBool_OllamaAPI_use_rerank=False__hash_',
      )
      saved = json.loads((run_dir / 'eval_config.json').read_text())
      self.assertIsNone(saved['index'])

    with self.subTest('metric_functions_none_records_empty_list'):
      out_dir = epath.Path(self.create_tempdir().full_path)
      evaluate_rag_lib.evaluate_rag(
          examples=loader,
          output=str(out_dir),
          index=indexing.EmptyIndex(),
          strategy=evaluate_rag_lib.StrategyType.RAG,
          evaluate_kwargs={'metric_functions': None},
      )
      run_dirs = list((out_dir / 'browsecomp_plus').iterdir())
      self.assertLen(run_dirs, 1)
      saved = json.loads((run_dirs[0] / 'eval_config.json').read_text())
      self.assertEqual(saved['metric_functions'], [])

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

  @mock.patch.object(ot_evaluation, 'evaluate')
  def test_evaluate_rag_signature_order_and_keyword_only_args(self, mock_eval):
    mock_eval.return_value = results.EvaluationSummary()
    index = indexing.EmptyIndex()

    # Positional arguments are (examples, index, strategy).
    evaluate_rag_lib.evaluate_rag(
        _CustomLoader(),
        index,
        evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
        output=None,
        evaluate_kwargs={'metric_functions': {}},
        extra_backend_kwargs={},
    )
    built = mock_eval.call_args.kwargs['strategy']
    self.assertIsInstance(built, strategies_lib.RetrievalOnlyStrategy)
    self.assertIs(built.retriever, index)

    # Trailing options (mcp_urls, output, evaluate_kwargs, extra_backend_kwargs)
    # are keyword-only, so a 4th positional argument fails with TypeError.
    with self.assertRaisesRegex(TypeError, 'positional'):
      evaluate_rag_lib.evaluate_rag(
          _CustomLoader(),
          index,
          evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
          '/tmp/out',  # pyrefly: ignore[bad-argument-count]
      )

    # Passing output as the 2nd positional argument alongside index= fails.
    with self.assertRaisesRegex(TypeError, 'index'):
      evaluate_rag_lib.evaluate_rag(
          _CustomLoader(),
          '/tmp/out',
          index=index,  # pyrefly: ignore[bad-keyword-argument]
      )


if __name__ == '__main__':
  absltest.main()
