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
from onetwo.stdlib.qa import qa_spec
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import index_building


def evaluate_qa(
    rag_spec: qa_spec.QAStrategySpec,
    examples: Sequence[Mapping[str, Any]],
    metric_functions: Mapping[str, ot_evaluation.MetricFunction],
    runtime_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> results.EvaluationSummary:
  """Evaluates a QA strategy on a dataset.

  Args:
    rag_spec: Specification of the QA strategy and index.
    examples: Examples to evaluate on.
    metric_functions: Metrics to compute.
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

    if rag_spec.index_path:
      index = index_building.load_index(
          rag_spec.index_path, runtime_kwargs=runtime_kwargs_dict
      )
    else:
      index = None

    if rag_spec.qa_strategy_name == 'mcp':
      strategy = strategies_lib.get_strategy(
          rag_spec.qa_strategy_name,
          **rag_spec.mcp_kwargs,
      )
    elif rag_spec.qa_strategy_name == 'closed_book':
      strategy = strategies_lib.get_strategy(
          rag_spec.qa_strategy_name,
      )
    else:
      strategy = strategies_lib.get_strategy(
          rag_spec.qa_strategy_name,
          retriever=index,
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
