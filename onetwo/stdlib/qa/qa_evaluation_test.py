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

"""Tests for how `qa_evaluation` builds the strategy it evaluates."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.stdlib.qa import agentic_retrieval
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.tool_use import llm_tool_use


def _build(
    mcp_kwargs: dict[str, Any],
) -> agentic_retrieval.AgenticRetrievalStrategy:
  """Returns the strategy `evaluate_qa` builds from `mcp_kwargs`."""
  strategy = qa_evaluation._agentic_retrieval_strategy(mcp_kwargs)
  assert isinstance(strategy, agentic_retrieval.AgenticRetrievalStrategy)
  return strategy


class AgenticRetrievalStrategyTest(parameterized.TestCase):
  """Tests for building the `agentic_retrieval` strategy from `mcp_kwargs`."""

  @parameterized.named_parameters(
      dict(
          testcase_name='comma_separated_string',
          mcp_kwargs={'mcp_server_urls': 'http://a/mcp, http://b/mcp,'},
          expected_num_tools=2,
      ),
      dict(
          testcase_name='no_urls',
          mcp_kwargs={},
          expected_num_tools=0,
      ),
      dict(
          testcase_name='null_urls',
          mcp_kwargs={'mcp_server_urls': None},
          expected_num_tools=0,
      ),
  )
  def test_builds_one_tool_per_url(
      self, mcp_kwargs: dict[str, Any], expected_num_tools: int
  ):
    strategy = _build(mcp_kwargs)

    self.assertLen(strategy.tools, expected_num_tools)

  def test_defaults(self):
    strategy = _build({'mcp_server_urls': 'http://a/mcp', 'max_steps': 7})

    with self.subTest('urls_become_tool_sources'):
      self.assertEqual(
          [type(tool) for tool in strategy.tools], [llm_tool_use.ToolSource]
      )
    with self.subTest('reports_cited_documents'):
      self.assertIs(
          strategy.doc_ids_from_answer, agentic_retrieval.extract_cited_doc_ids
      )
    with self.subTest('forwards_other_kwargs'):
      self.assertEqual(strategy.max_steps, 7)

  def test_evaluate_qa_uses_index_object_without_loading(self):
    mock_load_index = self.enter_context(
        mock.patch.object(qa_evaluation.index_building, 'load_index')
    )
    index = indexing.EmptyIndex()
    summary = qa_evaluation.evaluate_qa(
        rag_spec=qa_spec.QAStrategySpec(qa_strategy_name='retrieval_only'),
        examples=[{'question': 'What is OneTwo?', 'answer': 'A library'}],
        metric_functions={},
        index=index,
    )
    mock_load_index.assert_not_called()
    self.assertEqual(summary.counters['total_count'], 1)


if __name__ == '__main__':
  absltest.main()
