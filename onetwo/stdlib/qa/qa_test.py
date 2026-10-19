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

import dataclasses

from absl.testing import absltest
from onetwo.stdlib.qa import qa
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document


@dataclasses.dataclass(frozen=True)
class _StructuredAnswer:
  label: str
  score: float


@dataclasses.dataclass(frozen=True)
class _ExtendedStrategyResult(qa.StrategyResult):
  extra: str = 'extra'


class StrategyResultTest(absltest.TestCase):

  def test_defaults(self):
    result = qa.StrategyResult()

    self.assertEqual('', result.answer)
    self.assertEqual([], result.docs)

  def test_docs_only(self):
    docs = [_Document(doc_id='d1')]

    result = qa.StrategyResult(docs=docs)

    self.assertEqual('', result.answer)
    self.assertEqual(docs, result.docs)

  def test_positional_construction_matches_field_order(self):
    docs = [_Document(doc_id='d1')]

    result = qa.StrategyResult('answer', docs)

    self.assertEqual('answer', result.answer)
    self.assertEqual(docs, result.docs)

  def test_is_base_strategy_result(self):
    self.assertIsInstance(qa.StrategyResult(), qa.BaseStrategyResult)

  def test_frozen_subclass_keeps_defaults(self):
    result = _ExtendedStrategyResult(docs=[_Document(doc_id='d1')])

    self.assertEqual('', result.answer)
    self.assertEqual('extra', result.extra)
    self.assertIsInstance(result, qa.StrategyResult)


class BaseStrategyResultTest(absltest.TestCase):

  def test_structured_answer(self):
    answer = _StructuredAnswer(label='yes', score=0.9)
    docs = [_Document(doc_id='d1')]

    result = qa.BaseStrategyResult[_StructuredAnswer](answer=answer, docs=docs)

    self.assertEqual(answer, result.answer)
    self.assertEqual(docs, result.docs)

  def test_docs_default_to_empty(self):
    result = qa.BaseStrategyResult(answer=_StructuredAnswer('no', 0.1))

    self.assertEqual([], result.docs)


if __name__ == '__main__':
  absltest.main()
