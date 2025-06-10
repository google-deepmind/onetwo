# Copyright 2025 DeepMind Technologies Limited.
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
from onetwo.core import executing
from onetwo.stdlib.ensembling import distribution_metrics


# Simple custom base metric that gives full credit if the correct answer is
# included anywhere in the prediction (e.g., if target is `12` and the
# prediction is 'The answer is 12.')
def answer_included_sync(target: str, prediction: str):
  return 1.0 if target in prediction else 0.0


async def answer_included_async(target: str, prediction: str):
  return answer_included_sync(target, prediction)


@executing.make_executable  # pytype: disable=wrong-arg-types
def answer_included_executable(target: str, prediction: str):
  return answer_included_sync(target, prediction)


class DistributionMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('correct_at_1', 'a', [('a', 0.5), ('b', 0.3), ('c', 0.2)], 1, 1.0),
      ('wrong_at_1', 'b', [('a', 0.5), ('b', 0.3), ('c', 0.2)], 1, 0.0),
      ('correct_at_2', 'b', [('a', 0.5), ('b', 0.3), ('c', 0.2)], 2, 1.0),
      ('wrong_at_2', 'c', [('a', 0.5), ('b', 0.3), ('c', 0.2)], 2, 0.0),
      ('case_sensitive', 'A', [('a', 0.5), ('b', 0.3), ('c', 0.2)], 1, 0.0),
      ('empty_distribution', 'a', [], 1, 0.0),
      ('number_correct', 1.0, [(1, 0.5), (2, 0.3), (3, 0.2)], 1, 1.0),
      ('number_wrong', 2.0, [(1, 0.5), (2, 0.3), (3, 0.2)], 1, 0.0),
  )
  def test_accuracy_at_k_default_base_metric(
      self, target, predicted_distribution, k, expected
  ):
    metric = distribution_metrics.AccuracyAtK(k=k)
    actual = executing.run(
        metric(target=target, prediction=predicted_distribution)  # pytype: disable=wrong-keyword-args
    )
    self.assertEqual(expected, actual)

  @parameterized.named_parameters(
      ('sync', answer_included_sync),
      ('async', answer_included_async),
      ('executable', answer_included_executable),
  )
  def test_accuracy_at_k_custom_base_metric(self, base_metric):
    metric = distribution_metrics.AccuracyAtK(
        k=2, base_metric=base_metric
    )

    actual = executing.run(
        metric(target='b', prediction=[('=a', 0.5), ('=b', 0.3), ('=c', 0.2)])  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('correct_at_2'):
      self.assertEqual(1.0, actual, actual)

    actual = executing.run(
        metric(target='c', prediction=[('=a', 0.5), ('=b', 0.3), ('=c', 0.2)])  # pytype: disable=wrong-keyword-args
    )
    with self.subTest('wrong_at_2'):
      self.assertEqual(0.0, actual, actual)


if __name__ == '__main__':
  absltest.main()
