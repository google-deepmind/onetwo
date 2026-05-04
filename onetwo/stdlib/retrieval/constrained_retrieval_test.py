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
from onetwo.stdlib.retrieval import constrained_retrieval

RetrievalConstraint = constrained_retrieval.RetrievalConstraint
RetrievalConstraintType = constrained_retrieval.RetrievalConstraintType
RetrievalConstraints = constrained_retrieval.RetrievalConstraints


class ConstrainedRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock 'inverted index' where: Field -> Value -> List of IDs.
    self.field_to_val_id_map = {
        'color': {
            'red': [10, 20, 30],
            'blue': [11, 21, 31],
        },
        'status': {
            'active': [10, 11, 20],
            'archived': [21, 30, 31],
        },
        'tags': {
            'priority': [10, 31],
        },
    }

  def test_get_indices_for_constraint_equals(self):
    val_map = {'red': [10, 20], 'blue': [30]}
    constraint = RetrievalConstraint(
        field_name='color',
        constraint_type=RetrievalConstraintType.EQUALS,
        value='red',
    )
    result = constrained_retrieval.get_indices_for_constraint(
        constraint, val_map
    )
    self.assertEqual(result, {10, 20})

  def test_get_indices_for_constraint_list_contains_any(self):
    val_map = {'A': [1], 'B': [2], 'C': [3]}
    constraint = RetrievalConstraint(
        field_name='category',
        constraint_type=RetrievalConstraintType.LIST_CONTAINS_ANY,
        value=['A', 'C'],
    )
    result = constrained_retrieval.get_indices_for_constraint(
        constraint, val_map
    )
    self.assertEqual(result, {1, 3})

  def test_get_candidate_indices_and_intersection(self):
    # red ([10, 20, 30]) AND active ([10, 11, 20]) -> [10, 20]
    constraints = RetrievalConstraints(
        constraints=[
            RetrievalConstraint(
                field_name='color',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='red',
            ),
            RetrievalConstraint(
                field_name='status',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='active',
            ),
        ]
    )
    result = constrained_retrieval.get_candidate_indices(
        constraints, self.field_to_val_id_map
    )
    self.assertEqual(result, [10, 20])

  def test_get_candidate_indices_early_exit(self):
    # blue ([11, 21, 31]) AND priority ([10, 31]) -> [31].
    # Adding 'active' ([10, 11, 20]) to that results in [].
    constraints = RetrievalConstraints(
        constraints=[
            RetrievalConstraint(
                field_name='color',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='blue',
            ),
            RetrievalConstraint(
                field_name='tags',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='priority',
            ),
            RetrievalConstraint(
                field_name='status',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='active',
            ),
        ]
    )
    result = constrained_retrieval.get_candidate_indices(
        constraints, self.field_to_val_id_map
    )
    self.assertEqual(result, [])

  def test_get_candidate_indices_none_or_empty(self):
    self.assertIsNone(
        constrained_retrieval.get_candidate_indices(
            None, self.field_to_val_id_map
        )
    )

    empty_constraints = RetrievalConstraints(constraints=[])
    self.assertIsNone(
        constrained_retrieval.get_candidate_indices(
            empty_constraints, self.field_to_val_id_map
        )
    )

  def test_get_candidate_indices_unknown_field(self):
    constraints = RetrievalConstraints(
        constraints=[
            RetrievalConstraint(
                field_name='unknown',
                constraint_type=RetrievalConstraintType.EQUALS,
                value='val',
            )
        ]
    )
    result = constrained_retrieval.get_candidate_indices(
        constraints, self.field_to_val_id_map
    )
    self.assertEqual(result, [])

  def test_list_contains_any_type_error(self):
    constraint = RetrievalConstraint(
        field_name='tags',
        constraint_type=RetrievalConstraintType.LIST_CONTAINS_ANY,
        value='not_a_list',
    )
    with self.assertRaisesRegex(ValueError, 'must be a list'):
      constrained_retrieval.get_indices_for_constraint(constraint, {})


if __name__ == '__main__':
  absltest.main()
