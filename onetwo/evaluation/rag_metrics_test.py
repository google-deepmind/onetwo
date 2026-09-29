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

from typing import Any, TypeVar, cast
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import results
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa
from onetwo.stdlib.retrieval import retrieval_data_structures

StrategyResult = qa.StrategyResult


class MetricsLibTest(parameterized.TestCase):

  def test_default_golden_doc_ids_extractor_success_list(self):
    example = {'metadata': {'golden_doc_ids': ['doc1', 'doc2']}}
    gold_ids = rag_metrics._default_golden_doc_ids_extractor(example)
    self.assertEqual(gold_ids, ['doc1', 'doc2'])

  def test_default_golden_doc_ids_extractor_success_str(self):
    example = {'metadata': {'golden_doc_ids': 'doc1'}}
    gold_ids = rag_metrics._default_golden_doc_ids_extractor(example)
    self.assertEqual(gold_ids, ['doc1'])

  def test_default_golden_doc_ids_extractor_missing_key(self):
    example = {'metadata': {'some_other_key': 'doc1'}}
    with self.assertRaises(KeyError):
      rag_metrics._default_golden_doc_ids_extractor(example)

  def test_default_golden_doc_ids_extractor_missing_metadata(self):
    example = {'some_other_key': 'doc1'}
    with self.assertRaises(KeyError):
      rag_metrics._default_golden_doc_ids_extractor(example)

  def test_default_golden_doc_scores_extractor_success(self):
    example = {'metadata': {'golden_doc_scores': {'doc1': 2, 'doc2': 1}}}
    scores = rag_metrics._default_golden_doc_scores_extractor(example)
    self.assertEqual(scores, {'doc1': 2, 'doc2': 1})

  def test_default_golden_doc_scores_extractor_missing_key(self):
    example = {'metadata': {'some_other_key': 1}}
    with self.assertRaises(KeyError):
      rag_metrics._default_golden_doc_scores_extractor(example)

  def test_default_golden_doc_scores_extractor_missing_metadata(self):
    example = {'some_other_key': 1}
    with self.assertRaises(KeyError):
      rag_metrics._default_golden_doc_scores_extractor(example)

  @parameterized.named_parameters(
      ('recall_at_1_perfect', 1, ['doc1', 'doc3'], ['doc1'], 1.0),
      ('recall_at_1_zero', 1, ['doc1', 'doc3'], ['doc2'], 0.0),
      ('recall_at_2_partial', 2, ['doc1', 'doc3'], ['doc1', 'doc2'], 0.5),
      ('recall_all_partial', None, ['doc1', 'doc3'], ['doc1', 'doc2'], 0.5),
      ('recall_empty_prediction', 5, [], ['doc1'], 0.0),
      ('recall_empty_gold', 5, ['doc1'], [], 1.0),
      ('recall_empty_both', 5, [], [], 1.0),
      (
          'recall_at_2_perfect_different_order',
          2,
          ['doc2', 'doc1'],
          ['doc1', 'doc2'],
          1.0,
      ),
      (
          'recall_at_1_partial_different_pred',
          1,
          ['doc2', 'doc3'],
          ['doc1', 'doc2'],
          0.5,
      ),
      (
          'recall_chunked_perfect',
          2,
          ['doc1_chunk1', 'doc2_chunk1'],
          ['doc1', 'doc2'],
          1.0,
      ),
      (
          'recall_chunked_partial',
          1,
          ['doc1_chunk1', 'doc2_chunk1'],
          ['doc1', 'doc2'],
          0.5,
      ),
      (
          'recall_nested_chunked',
          2,
          ['doc1_chunk1_chunk2'],
          ['doc1'],
          1.0,
      ),
  )
  def test_recall_at_k_calculation(
      self, k, prediction_doc_ids, golden_doc_ids, expected_recall
  ):
    example = {'metadata': {'golden_doc_ids': golden_doc_ids}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in prediction_doc_ids
        ],
    )
    recall_fn = rag_metrics.make_recall_at_k(k=k)
    recall, extra_info = recall_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(recall, expected_recall)

    if golden_doc_ids:
      expected_retrieved_ids = (
          [doc.doc_id for doc in prediction.docs][:k]
          if k is not None
          else [doc.doc_id for doc in prediction.docs]
      )
      self.assertEqual(extra_info['golden_doc_ids'], golden_doc_ids)
      self.assertEqual(extra_info['retrieved_doc_ids'], expected_retrieved_ids)
    else:
      self.assertEqual(extra_info, {})

  def test_recall_at_k_custom_extractor(self):
    example = {'custom_answer_key': ['doc1']}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in ['doc1', 'doc3']
        ],
    )
    custom_extractor = lambda ex: ex['custom_answer_key']
    recall_fn = rag_metrics.make_recall_at_k(
        k=5, golden_doc_ids_extractor=custom_extractor
    )
    recall, extra_info = recall_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(recall, 1.0)
    self.assertEqual(extra_info['golden_doc_ids'], ['doc1'])
    self.assertEqual(extra_info['retrieved_doc_ids'], ['doc1', 'doc3'])

  def test_recall_at_k_no_strip(self):
    example = {'metadata': {'golden_doc_ids': ['doc1']}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in ['doc1_chunk1', 'doc2']
        ],
    )
    recall_fn = rag_metrics.make_recall_at_k(k=2, strip_chunk_suffix=False)
    recall, extra_info = recall_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(recall, 0.0)
    self.assertEqual(extra_info['golden_doc_ids'], ['doc1'])
    self.assertEqual(extra_info['retrieved_doc_ids'], ['doc1_chunk1', 'doc2'])

  def test_recall_at_k_none_prediction(self):
    recall_fn = rag_metrics.make_recall_at_k(k=5)
    score, info = recall_fn(
        target=None,
        prediction=None,
        example={'metadata': {'golden_doc_ids': ['doc1']}},
    )
    self.assertEqual(score, 0.0)
    self.assertEqual(info, {})

  def test_accuracy_critic_none_prediction(self):
    accuracy_critic_fn = rag_metrics.make_accuracy_critic()
    score, info = executing.run(
        accuracy_critic_fn(
            target='expected answer',
            prediction=None,
            example={'question': 'some question'},
        )
    )
    self.assertEqual(score, 0.0)
    self.assertEqual(info, {})

  @parameterized.named_parameters(
      ('ndcg_at_1_perfect', 1, ['doc1', 'doc3'], {'doc1': 2}, 1.0),
      ('ndcg_at_1_zero', 1, ['doc1', 'doc3'], {'doc2': 2}, 0.0),
      ('ndcg_at_2_perfect', 2, ['doc1', 'doc2'], {'doc1': 2, 'doc2': 1}, 1.0),
      ('ndcg_at_2_zero', 2, ['doc3', 'doc4'], {'doc1': 2, 'doc2': 1}, 0.0),
      ('ndcg_all_perfect', None, ['doc1', 'doc2'], {'doc1': 2, 'doc2': 1}, 1.0),
      ('ndcg_empty_prediction', 5, [], {'doc1': 2}, 0.0),
      ('ndcg_empty_gold', 5, ['doc1'], {}, 1.0),
      ('ndcg_empty_both', 5, [], {}, 1.0),
      (
          'ndcg_chunked_perfect',
          2,
          ['doc1_chunk1', 'doc2_chunk1'],
          {'doc1': 2, 'doc2': 1, 'doc3': 1},
          1.0,
      ),
      (
          'ndcg_nested_chunked',
          2,
          ['doc1_chunk1_chunk2'],
          {'doc1': 2},
          1.0,
      ),
      (
          'ndcg_all_zero_scores',
          2,
          ['doc1', 'doc2'],
          {'doc1': 0, 'doc2': 0},
          0.0,
      ),
      (
          'ndcg_duplicate_chunks',
          3,
          ['doc1_chunk1', 'doc1_chunk2', 'doc2_chunk1'],
          {'doc1': 2, 'doc2': 1},
          1.0,
      ),
  )
  def test_ndcg_at_k_calculation(
      self, k, prediction_doc_ids, golden_doc_scores, expected_ndcg
  ):
    example = {'metadata': {'golden_doc_scores': golden_doc_scores}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in prediction_doc_ids
        ],
    )
    ndcg_fn = rag_metrics.make_ndcg_at_k(k=k)
    ndcg, extra_info = ndcg_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(ndcg, expected_ndcg)

    if golden_doc_scores:
      expected_retrieved_ids = (
          [doc.doc_id for doc in prediction.docs][:k]
          if k is not None
          else [doc.doc_id for doc in prediction.docs]
      )
      self.assertEqual(extra_info['golden_doc_scores'], golden_doc_scores)
      self.assertEqual(extra_info['retrieved_doc_ids'], expected_retrieved_ids)
    else:
      self.assertEqual(extra_info, {})

  def test_ndcg_at_k_imperfect_order(self):
    # golden: {doc1: 2, doc2: 1}
    # retrieved: [doc2, doc1]
    # DCG (doc2 then doc1) = ~2.89
    # Ideal DCG = ~3.63
    # NDCG = DCG / IDCG = ~0.797
    example = {'metadata': {'golden_doc_scores': {'doc1': 2, 'doc2': 1}}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in ['doc2', 'doc1']
        ],
    )
    ndcg_fn = rag_metrics.make_ndcg_at_k(k=2)
    ndcg, extra_info = ndcg_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(ndcg, 0.797, places=2)
    self.assertAlmostEqual(extra_info['dcg'], 2.893, places=2)
    self.assertAlmostEqual(extra_info['idcg'], 3.631, places=2)

  def test_ndcg_at_k_custom_extractor(self):
    example = {'custom_scores_key': {'doc1': 2}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in ['doc1', 'doc3']
        ],
    )
    custom_extractor = lambda ex: ex['custom_scores_key']
    ndcg_fn = rag_metrics.make_ndcg_at_k(
        k=5, golden_doc_scores_extractor=custom_extractor
    )
    ndcg, extra_info = ndcg_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(ndcg, 1.0)
    self.assertEqual(extra_info['golden_doc_scores'], {'doc1': 2})
    self.assertEqual(extra_info['retrieved_doc_ids'], ['doc1', 'doc3'])

  def test_ndcg_at_k_no_strip(self):
    example = {'metadata': {'golden_doc_scores': {'doc1': 2}}}
    prediction = StrategyResult(
        answer='some answer',
        docs=[
            retrieval_data_structures.Document(doc_id=d)
            for d in ['doc1_chunk1', 'doc2']
        ],
    )
    ndcg_fn = rag_metrics.make_ndcg_at_k(k=2, strip_chunk_suffix=False)
    ndcg, extra_info = ndcg_fn(
        target=None, prediction=prediction, example=example
    )
    self.assertAlmostEqual(ndcg, 0.0)
    self.assertEqual(extra_info['golden_doc_scores'], {'doc1': 2})
    self.assertEqual(extra_info['retrieved_doc_ids'], ['doc1_chunk1', 'doc2'])

  def test_ndcg_at_k_missing_example(self):
    prediction = StrategyResult(
        answer='some answer',
        docs=[retrieval_data_structures.Document(doc_id='doc1')],
    )
    ndcg_fn = rag_metrics.make_ndcg_at_k(k=2)
    with self.assertRaises(ValueError):
      ndcg_fn(target=None, prediction=prediction, example=None)

  def test_ndcg_at_k_none_prediction(self):
    ndcg_fn = rag_metrics.make_ndcg_at_k(k=5)
    score, info = ndcg_fn(
        target=None,
        prediction=None,
        example={'metadata': {'golden_doc_scores': {'doc1': 2}}},
    )
    self.assertEqual(score, 0.0)
    self.assertEqual(info, {})

  def test_accuracy_critic_with_backend_config(self):
    _T = TypeVar('_T')

    class FakeBackend(backends_test_utils.LLMForTest):

      def __init__(self, **kwargs):
        super().__init__()
        self.called_generate_object = False

      @executing.make_executable(copy_self=False)
      async def generate_object(
          self,
          prompt: str | content_lib.ChunkList,
          cls: type[_T],
          **kwargs,
      ) -> _T:
        self.called_generate_object = True
        return cast(
            _T,
            rag_metrics.evaluation.NaiveCriticDecision(
                agree=True, reason='fake_reason'
            ),
        )

    backend_registry.BACKEND_REGISTRY['FakeBackend'] = FakeBackend
    self.addCleanup(backend_registry.BACKEND_REGISTRY.pop, 'FakeBackend', None)

    config = backend_registry.GenerateBackendConfig(
        backend_class_name='FakeBackend',
    )

    prediction = StrategyResult(answer='some answer', docs=[])

    accuracy_critic_fn = rag_metrics.make_accuracy_critic(backend_config=config)

    rating, extra_info = executing.run(
        accuracy_critic_fn(
            target='bla',
            prediction=prediction,
            example={'question': 'some question'},
        )
    )

    self.assertEqual(1.0, rating)
    self.assertEqual('fake_reason', extra_info['critic_reason'])

  def test_walk_llm_stages_empty_trace(self):
    trace = results.ExecutionResult(stage_name='root')
    self.assertEqual(rag_metrics._walk_llm_stages(trace), [])

  def test_walk_llm_stages_single_stage(self):
    stage = results.ExecutionResult(stage_name='GoogleGenAIAPI.chat')
    trace = results.ExecutionResult(stage_name='root', stages=[stage])
    self.assertEqual(rag_metrics._walk_llm_stages(trace), [stage])

  def test_walk_llm_stages_multi_stage(self):
    s1 = results.ExecutionResult(stage_name='GoogleGenAIAPI.chat')
    s2 = results.ExecutionResult(stage_name='GoogleGenAIAPI.generate_text')
    s3 = results.ExecutionResult(stage_name='GoogleGenAIAPI.instruct')
    s_other = results.ExecutionResult(stage_name='other_stage')
    trace = results.ExecutionResult(
        stage_name='root',
        stages=[
            s1,
            results.ExecutionResult(stage_name='sub', stages=[s2, s_other]),
            s3,
        ],
    )
    self.assertEqual(rag_metrics._walk_llm_stages(trace), [s1, s2, s3])

  def test_walk_llm_stages_skips_children_of_matched_stage(self):
    child = results.ExecutionResult(stage_name='inner.generate_text')
    parent = results.ExecutionResult(stage_name='outer.chat', stages=[child])
    trace = results.ExecutionResult(stage_name='root', stages=[parent])
    self.assertEqual(rag_metrics._walk_llm_stages(trace), [parent])

  def test_num_llm_calls(self):
    self.assertEqual(
        rag_metrics.num_llm_calls(
            target=None, prediction=None, execution_result=None
        ),
        0.0,
    )
    s1 = results.ExecutionResult(stage_name='GoogleGenAIAPI.chat')
    s2 = results.ExecutionResult(stage_name='GoogleGenAIAPI.generate_text')
    trace = results.ExecutionResult(stage_name='root', stages=[s1, s2])
    self.assertEqual(
        rag_metrics.num_llm_calls(
            target=None, prediction=None, execution_result=trace
        ),
        2.0,
    )

  def test_input_token_metrics(self):
    self.assertEqual(
        rag_metrics.total_input_tokens(None, None, execution_result=None), 0.0
    )
    self.assertEqual(
        rag_metrics.max_input_tokens(None, None, execution_result=None), 0.0
    )
    self.assertEqual(
        rag_metrics.average_input_tokens(None, None, execution_result=None), 0.0
    )

    s1 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.chat',
        info={'llm.tokens.prompt': 10, 'llm.tokens.tool_use_prompt': 3},
    )
    s2 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.generate_text',
        info={'llm.tokens.prompt': 20},
    )
    trace = results.ExecutionResult(stage_name='root', stages=[s1, s2])

    self.assertEqual(
        rag_metrics.total_input_tokens(None, None, execution_result=trace), 33.0
    )
    self.assertEqual(
        rag_metrics.max_input_tokens(None, None, execution_result=trace), 20.0
    )
    self.assertEqual(
        rag_metrics.average_input_tokens(None, None, execution_result=trace),
        16.5,
    )

  def test_output_token_metrics(self):
    self.assertEqual(
        rag_metrics.total_output_tokens(None, None, execution_result=None), 0.0
    )
    self.assertEqual(
        rag_metrics.max_output_tokens(None, None, execution_result=None), 0.0
    )
    self.assertEqual(
        rag_metrics.average_output_tokens(None, None, execution_result=None),
        0.0,
    )

    s1 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.chat',
        info={'llm.tokens.candidates': 5, 'llm.tokens.thoughts': 3},
    )
    s2 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.generate_text',
        info={'llm.tokens.candidates': 12},
    )
    trace = results.ExecutionResult(stage_name='root', stages=[s1, s2])

    self.assertEqual(
        rag_metrics.total_output_tokens(None, None, execution_result=trace),
        20.0,
    )
    self.assertEqual(
        rag_metrics.max_output_tokens(None, None, execution_result=trace), 12.0
    )
    self.assertEqual(
        rag_metrics.average_output_tokens(None, None, execution_result=trace),
        10.0,
    )

  def test_num_tool_calls(self):
    self.assertEqual(
        rag_metrics.num_tool_calls(None, None, execution_result=None), 0.0
    )
    s1 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.chat',
        info={'llm.tools.calls': 2},
    )
    s2 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.generate_text',
        info={'llm.tools.calls': 1},
    )
    trace = results.ExecutionResult(stage_name='root', stages=[s1, s2])
    self.assertEqual(
        rag_metrics.num_tool_calls(None, None, execution_result=trace), 3.0
    )

  def test_tool_details(self):
    self.assertEqual(
        rag_metrics.tool_details(None, None, execution_result=None), (0.0, {})
    )
    detail1 = {'name': 'search', 'args': {'q': 'test'}}
    detail2 = {'name': 'lookup', 'args': {'id': '123'}}
    s1 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.chat',
        info={'llm.tools.details': [detail1]},
    )
    s2 = results.ExecutionResult(
        stage_name='GoogleGenAIAPI.generate_text',
        info={'llm.tools.details': [detail2]},
    )
    trace = results.ExecutionResult(stage_name='root', stages=[s1, s2])
    score, info = rag_metrics.tool_details(None, None, execution_result=trace)
    self.assertEqual(score, 2.0)
    self.assertEqual(info, {'tool_details': [detail1, detail2]})

  def test_reads_a_client_side_function_calling_run(self):
    # The shape `Agent` produces: one stage per step, each wrapping the model
    # call it made. Counts and tokens are read from the nested model call. The
    # calls themselves are read from the agent's own stage, which is the only
    # one that saw what the tools returned.
    def step(calls: list[str], tokens: int) -> results.ExecutionResult:
      requested = [
          {'type': 'function_call', 'name': name, 'args': {}} for name in calls
      ]
      answered = [
          dict(detail, response=f'{detail["name"]} result')
          for detail in requested
      ]
      agent_info: dict[str, Any] = {'llm.tools.calls': 99}
      llm_info: dict[str, Any] = {
          'llm.tools.calls': len(calls),
          'llm.tokens.prompt': tokens,
      }
      if calls:
        agent_info['llm.tools.details'] = answered
        llm_info['llm.tools.details'] = requested
      return results.ExecutionResult(
          stage_name='Agent.step',
          # Disagrees with the stage below on purpose, to show which of the two
          # the metrics take.
          info=agent_info,
          stages=[
              results.ExecutionResult(
                  stage_name='GoogleGenAIAPI.generate_with_tools',
                  info=llm_info,
              )
          ],
      )

    trace = results.ExecutionResult(
        stage_name='root',
        stages=[
            step(['search', 'read'], 100),
            step(['read'], 200),
            step([], 250),
        ],
    )

    with self.subTest('counts_one_llm_call_per_step'):
      self.assertEqual(
          rag_metrics.num_llm_calls(None, None, execution_result=trace), 3.0
      )
    with self.subTest('counts_the_tools_the_model_call_reported'):
      self.assertEqual(
          rag_metrics.num_tool_calls(None, None, execution_result=trace), 3.0
      )
    with self.subTest('reads_tokens_from_the_model_call'):
      self.assertEqual(
          rag_metrics.total_input_tokens(None, None, execution_result=trace),
          550.0,
      )
    with self.subTest('reports_every_call_once_with_its_response'):
      _, info = rag_metrics.tool_details(None, None, execution_result=trace)
      self.assertEqual(
          [(d['name'], d.get('response')) for d in info['tool_details']],
          [
              ('search', 'search result'),
              ('read', 'read result'),
              ('read', 'read result'),
          ],
      )

  def test_reads_a_bare_generate_with_tools_call(self):
    trace = results.ExecutionResult(
        stage_name='root',
        stages=[
            results.ExecutionResult(
                stage_name='GoogleGenAIAPI.generate_with_tools',
                info={'llm.tokens.prompt': 10},
            )
        ],
    )
    self.assertEqual(
        rag_metrics.num_llm_calls(None, None, execution_result=trace), 1.0
    )

  def test_get_default_rag_metrics(self):
    metrics = rag_metrics.get_default_rag_metrics()
    expected_keys = {
        'accuracy',
        'runtime_seconds',
        'recall@5',
        'recall@10',
        'recall@20',
        'num_llm_calls',
        'total_input_tokens',
        'max_input_tokens',
        'average_input_tokens',
        'total_output_tokens',
        'max_output_tokens',
        'average_output_tokens',
        'num_tool_calls',
        'tool_details',
    }
    self.assertEqual(set(metrics.keys()), expected_keys)

  def test_get_default_rag_metrics_with_max_results(self):
    metrics = rag_metrics.get_default_rag_metrics(max_results=14)
    expected_keys = {
        'accuracy',
        'runtime_seconds',
        'recall@5',
        'recall@10',
        'recall@14',
        'recall@20',
        'num_llm_calls',
        'total_input_tokens',
        'max_input_tokens',
        'average_input_tokens',
        'total_output_tokens',
        'max_output_tokens',
        'average_output_tokens',
        'num_tool_calls',
        'tool_details',
    }
    self.assertEqual(set(metrics.keys()), expected_keys)

  def test_runtime_seconds_with_timing(self):
    trace = results.ExecutionResult(stage_name='root')
    trace.start_time = 100.0
    trace.end_time = 105.5
    elapsed = rag_metrics.runtime_seconds(
        target='a', prediction='b', execution_result=trace
    )
    self.assertAlmostEqual(elapsed, 5.5)

  def test_runtime_seconds_none_execution_result(self):
    elapsed = rag_metrics.runtime_seconds(
        target='a', prediction='b', execution_result=None
    )
    self.assertEqual(elapsed, 0.0)

  def test_runtime_seconds_no_timing(self):
    trace = results.ExecutionResult(stage_name='root')
    elapsed = rag_metrics.runtime_seconds(
        target='a', prediction='b', execution_result=trace
    )
    self.assertEqual(elapsed, 0.0)

if __name__ == '__main__':
  absltest.main()
