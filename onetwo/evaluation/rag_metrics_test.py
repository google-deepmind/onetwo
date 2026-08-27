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

from typing import TypeVar, cast
from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.core import executing
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


if __name__ == '__main__':
  absltest.main()
