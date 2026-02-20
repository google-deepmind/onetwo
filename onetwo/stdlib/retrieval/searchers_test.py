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
from absl.testing import parameterized
import numpy as np
from onetwo import ot
from onetwo.stdlib.retrieval import searchers


class SearchersTest(parameterized.TestCase):

  def test_brute_force_searcher(self):
    query_embeds = np.array([[0.7, 0.2, 0.1]])
    doc_embeds = np.array([
        [0.1, 0.1, 0.8],  # doc 0
        [0.8, 0.1, 0.1],  # doc 1
        [0.1, 0.8, 0.1],  # doc 2
    ])
    searcher = searchers.BruteForceSearcher()
    ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count

    with self.subTest('all_neighbors'):
      top_ids, top_scores = ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count
      np.testing.assert_array_equal(top_ids, [[1, 2, 0]])
      np.testing.assert_array_almost_equal(
          top_scores, [[0.59, 0.24, 0.17]], decimal=6
      )

    with self.subTest('with_k'):
      top_ids, top_scores = ot.run(
          searcher.search_batched(query_embeds, final_num_neighbors=2)  # pytype: disable=wrong-arg-count
      )
      np.testing.assert_array_equal(top_ids, [[1, 2]])
      np.testing.assert_array_almost_equal(
          top_scores, [[0.59, 0.24]], decimal=6
      )

  def test_brute_force_searcher_more_cases(self):
    doc_embeds = np.array([
        [0.1, 0.1, 0.8],  # doc 0
        [0.8, 0.1, 0.1],  # doc 1
        [0.1, 0.8, 0.1],  # doc 2
    ])

    with self.subTest('search_before_build'):
      searcher = searchers.BruteForceSearcher()
      query_embeds = np.array([[0.7, 0.2, 0.1]])
      with self.assertRaisesRegex(ValueError, 'Need to call `build'):
        ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count

    with self.subTest('multiple_queries'):
      searcher = searchers.BruteForceSearcher()
      ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count
      query_embeds = np.array([
          [0.7, 0.2, 0.1],  # query 0 -> 1, 2, 0
          [0.1, 0.9, 0.2],  # query 1 -> 2, 0, 1
      ])
      top_ids, top_scores = ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count
      np.testing.assert_array_equal(top_ids, [[1, 2, 0], [2, 0, 1]])
      np.testing.assert_array_almost_equal(
          top_scores, [[0.59, 0.24, 0.17], [0.75, 0.26, 0.19]], decimal=6
      )

    with self.subTest('k_greater_than_num_docs'):
      searcher = searchers.BruteForceSearcher()
      ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count
      query_embeds = np.array([[0.7, 0.2, 0.1]])
      top_ids, top_scores = ot.run(
          searcher.search_batched(query_embeds, final_num_neighbors=5)  # pytype: disable=wrong-arg-count
      )
      np.testing.assert_array_equal(top_ids, [[1, 2, 0]])
      np.testing.assert_array_almost_equal(
          top_scores, [[0.59, 0.24, 0.17]], decimal=6
      )

    with self.subTest('empty_docs'):
      searcher = searchers.BruteForceSearcher()
      empty_doc_embeds = np.empty((0, 3))
      ot.run(searcher.build(empty_doc_embeds))  # pytype: disable=wrong-arg-count
      query_embeds = np.array([[0.7, 0.2, 0.1]])

      with self.subTest('all_neighbors'):
        top_ids, top_scores = ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count
        self.assertEqual(top_ids.shape, (1, 0))
        self.assertEqual(top_scores.shape, (1, 0))

      with self.subTest('with_k'):
        top_ids_k, top_scores_k = ot.run(
            searcher.search_batched(query_embeds, final_num_neighbors=2)  # pytype: disable=wrong-arg-count
        )
        self.assertEqual(top_ids_k.shape, (1, 0))
        self.assertEqual(top_scores_k.shape, (1, 0))

  def test_brute_force_searcher_normalization(self):
    query_embeds = np.array([[1.0, 0.0, 0.0]])
    doc_embeds = np.array([
        [10.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
    ])

    with self.subTest('without_normalization'):
      searcher = searchers.BruteForceSearcher(normalize_embeds=False)
      ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count
      top_ids, top_scores = ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count
      np.testing.assert_array_equal(top_ids, [[0, 1]])
      np.testing.assert_array_almost_equal(top_scores, [[10.0, 0.9]])

    with self.subTest('with_normalization'):
      searcher = searchers.BruteForceSearcher(normalize_embeds=True)
      ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count
      top_ids, top_scores = ot.run(searcher.search_batched(query_embeds))  # pytype: disable=wrong-arg-count
      np.testing.assert_array_equal(top_ids, [[0, 1]])
      self.assertAlmostEqual(top_scores[0][0], 1.0, places=6)
      self.assertLess(top_scores[0][1], 1.0)

  def test_brute_force_searcher_equality(self):
    doc_embeds = np.array([[0.1, 0.2, 0.3]])

    searcher1 = searchers.BruteForceSearcher(normalize_embeds=True)
    searcher2 = searchers.BruteForceSearcher(normalize_embeds=False)

    ot.run(searcher1.build(doc_embeds))  # pytype: disable=wrong-arg-count
    ot.run(searcher2.build(doc_embeds))  # pytype: disable=wrong-arg-count
    self.assertNotEqual(searcher1, searcher2)

    searcher3 = searchers.BruteForceSearcher(normalize_embeds=True)
    ot.run(searcher3.build(doc_embeds))  # pytype: disable=wrong-arg-count
    self.assertEqual(searcher1, searcher3)


class ConstrainedBruteForceSearcherTest(parameterized.TestCase):

  def test_constrained_brute_force_searcher_with_k(self):
    query_embeds = np.array([[1.0, 0.0, 0.0]])
    doc_embeds = np.array([
        [0.1, 0.1, 0.8],  # doc 0
        [0.8, 0.1, 0.1],  # doc 1
        [0.15, 0.8, 0.05],  # doc 2
        [0.75, 0.15, 0.1],  # doc 3
    ])
    searcher = searchers.ConstrainedBruteForceSearcher()
    ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count

    with self.subTest('search_with_empty_indices'):
      top_ids, top_scores = ot.run(
          searcher.search_batched(
              query_embeds, doc_indices=[], final_num_neighbors=2  # pytype: disable=wrong-arg-count
          )
      )
      np.testing.assert_array_equal(top_ids, [[1, 3]])
      np.testing.assert_array_equal(top_scores, [[0.8, 0.75]])

    with self.subTest('search_within_indices'):
      # Search within indices [0, 2, 3], limiting to 2 results.
      doc_indices = [0, 2, 3]
      # Expected order: doc 3, doc 2.
      top_ids, top_scores = ot.run(
          searcher.search_batched(
              query_embeds, doc_indices=doc_indices, final_num_neighbors=2  # pytype: disable=wrong-arg-count
          )
      )
      np.testing.assert_array_equal(top_ids, [[3, 2]])
      np.testing.assert_array_equal(top_scores, [[0.75, 0.15]])

  def test_constrained_brute_force_searcher_index_out_of_bounds(self):
    query_embeds = np.array([[0.7, 0.2, 0.1]])
    doc_embeds = np.array([[0.1, 0.1, 0.8]])
    searcher = searchers.ConstrainedBruteForceSearcher()
    ot.run(searcher.build(doc_embeds))  # pytype: disable=wrong-arg-count

    with self.assertRaisesRegex(ValueError, '`doc_indices` are out of bounds'):
      ot.run(searcher.search_batched(query_embeds, doc_indices=[0, 1]))  # pytype: disable=wrong-arg-count

    with self.assertRaisesRegex(ValueError, '`doc_indices` are out of bounds'):
      ot.run(searcher.search_batched(query_embeds, doc_indices=[-1]))  # pytype: disable=wrong-arg-count


if __name__ == '__main__':
  absltest.main()
