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

# Copyright 2026 Google LLC
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

"""QA evaluation library."""

from collections.abc import Mapping, Sequence
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import results
from onetwo.core import routing
from onetwo.evaluation import evaluation as ot_evaluation
from onetwo.stdlib.qa import agentic_retrieval
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import qa_spec
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing


def _agentic_retrieval_strategy(
    mcp_kwargs: Mapping[str, Any],
) -> qa.QAStrategy:
  """Returns an `agentic_retrieval` strategy using the given MCP servers.

  Args:
    mcp_kwargs: `mcp_server_urls`, a comma-separated list of the URLs of the MCP
      servers whose tools the model may call, and any other JSON-serializable
      arguments of `AgenticRetrievalStrategy`, e.g. `max_steps`. The documents
      the answer cites are reported, as found by `extract_cited_doc_ids`.

  Returns:
    The strategy.
  """
  # Imported here so that the MCP client libraries are only needed by callers
  # that evaluate this strategy.
  from onetwo.stdlib.tool_use import mcp_tools  # pylint: disable=g-import-not-at-top

  strategy_kwargs = dict(mcp_kwargs)
  # `None`, e.g. from `{"mcp_server_urls": null}` in JSON, means no servers.
  urls = (strategy_kwargs.pop('mcp_server_urls', None) or '').split(',')
  return strategies_lib.get_strategy(
      'agentic_retrieval',
      tools=[mcp_tools.server(url.strip()) for url in urls if url.strip()],
      doc_ids_from_answer=agentic_retrieval.extract_cited_doc_ids,
      **strategy_kwargs,
  )


def evaluate_qa(
    rag_spec: qa_spec.QAStrategySpec,
    examples: Sequence[Mapping[str, Any]],
    metric_functions: Mapping[str, ot_evaluation.MetricFunction],
    index: str | indexing.DocumentIndex = '',
    runtime_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> results.EvaluationSummary:
  """Evaluates a QA strategy on a dataset.

  Args:
    rag_spec: Specification of the QA strategy.
    examples: Examples to evaluate on.
    metric_functions: Metrics to compute.
    index: Directory of a saved index, an index object, or '' when unused.
    runtime_kwargs: Runtime arguments for the backends (e.g. API key).
    **kwargs: Additional arguments passed to ot.evaluate.

  Returns:
    EvaluationSummary.
  """
  runtime_kwargs_dict = dict(runtime_kwargs) if runtime_kwargs else {}

  with routing.RegistryContext():
    if rag_spec.generate_backend_config is not None:
      gen_backend = backend_registry.get_backend(
          rag_spec.generate_backend_config,
          **runtime_kwargs_dict,
      )
      gen_backend.register()

    retriever = None
    if isinstance(index, str):
      if index:
        retriever = index_building.load_index(
            index, runtime_kwargs=runtime_kwargs_dict
        )
    else:
      retriever = index

    if rag_spec.qa_strategy_name == 'agentic_retrieval':
      strategy = _agentic_retrieval_strategy(rag_spec.mcp_kwargs)
    elif rag_spec.qa_strategy_name == 'closed_book':
      strategy = strategies_lib.get_strategy(
          rag_spec.qa_strategy_name,
      )
    else:
      strategy = strategies_lib.get_strategy(
          rag_spec.qa_strategy_name,
          retriever=retriever,
      )

    return ot_evaluation.evaluate(
        strategy=strategy,
        examples=examples,
        metric_functions=metric_functions,
        inputs_extractor=lambda x: (
            [x['question']],
            rag_spec.query_time_kwargs,
        ),
        **kwargs,
    )


def save_html_report(
    summary: results.EvaluationSummary,
    output_dir_path: str | epath.Path,
    filename: str = 'results.html',
    levels_to_expand: int = 2,
) -> epath.Path:
  """Saves an EvaluationSummary as an HTML report.

  Args:
    summary: The evaluation summary to save.
    output_dir_path: Directory where the report should be saved.
    filename: Filename for the report.
    levels_to_expand: Depth of results to expand in the HTML view.

  Returns:
    The path to the saved HTML file.
  """
  output_dir = epath.Path(output_dir_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  html_report_path = output_dir / filename
  html_report_path.write_text(
      results.HTMLRenderer(levels_to_expand=levels_to_expand).render(summary)
  )
  return html_report_path
