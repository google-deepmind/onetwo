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

r"""Binary to evaluate a RAG strategy on a QA dataset.
"""

import json
from typing import Any, Mapping, Sequence

from absl import app
from absl import flags
from onetwo.backends import backend_registry
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec


_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    'browsecomp_plus',
    'Name of the dataset to load from the registry.',
    allow_override=True,
)

_INDEX_DIRECTORY = flags.DEFINE_string(
    'index_directory',
    None,
    'Directory containing the pre-built index.',
    required=True,
)

_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    None,
    'Directory where the evaluation results will be saved.',
    required=True,
)

_GENERATE_MODEL_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'generate_model_backend_class_name',
    'GoogleGenAIAPI',
    'Name of the generation backend class.',
)

_GENERATE_MODEL_NAME = flags.DEFINE_string(
    'generate_model_name',
    'gemini-3.5-flash-lite',
    'Name of the generation model.',
)

_RUNTIME_KWARGS = flags.DEFINE_string(
    'runtime_kwargs',
    '{}',
    'JSON string containing runtime arguments for the backends (e.g. API key,'
    ' project).',
    allow_override=True,
)

_QA_STRATEGY_NAME = flags.DEFINE_string(
    'qa_strategy_name',
    'rag',
    'Name of the QA strategy to evaluate.',
)

# TODO: Consider combining with _RUNTIME_KWARGS.
_QUERY_TIME_KWARGS = flags.DEFINE_string(
    'query_time_kwargs',
    '{"max_results": 20}',
    'JSON string containing query-time arguments for the strategy.',
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    None,
    'Maximum number of examples to evaluate.',
)


def evaluate_rag(
    dataset_name: str,
    index_directory: str,
    output_directory: str,
    runtime_kwargs: Mapping[str, Any] | str,
    generate_model_backend_class_name: str = 'GoogleGenAIAPI',
    generate_model_name: str = 'gemini-3.5-flash-lite',
    qa_strategy_name: str = 'rag',
    query_time_kwargs: Mapping[str, Any] | str = '{"max_results": 20}',
    max_examples: int | None = 5,
    metric_functions: Mapping[str, Any] | None = None,
) -> results.EvaluationSummary:
  """Evaluates a RAG strategy on a QA dataset.

  Args:
    dataset_name: Name of the dataset to load from the registry.
    index_directory: Directory containing the pre-built index.
    output_directory: Directory where evaluation results will be saved.
    runtime_kwargs: Runtime arguments dict or JSON string (e.g. API key).
    generate_model_backend_class_name: Generation backend class name.
    generate_model_name: Generation model name.
    qa_strategy_name: QA strategy name.
    query_time_kwargs: Query-time arguments dict or JSON string.
    max_examples: Max examples to evaluate.
    metric_functions: Optional mapping of metric names to metric functions.
      Defaults to rag_metrics.get_default_rag_metrics().

  Returns:
    EvaluationSummary object containing metrics and detailed results.
  """
  if isinstance(query_time_kwargs, str):
    q_time_kwargs = json.loads(query_time_kwargs)
  else:
    q_time_kwargs = dict(query_time_kwargs)

  if isinstance(runtime_kwargs, str):
    rt_kwargs = json.loads(runtime_kwargs)
  else:
    rt_kwargs = dict(runtime_kwargs)

  gen_backend_config = backend_registry.GenerateBackendConfig(
      backend_class_name=generate_model_backend_class_name,
      model_name=generate_model_name,
  )
  spec = qa_spec.QAStrategySpec(
      qa_strategy_name=qa_strategy_name,
      index_path=index_directory,
      generate_backend_config=gen_backend_config,
      query_time_kwargs=q_time_kwargs,
  )

  # 2. Load Dataset Examples
  print(f'Loading dataset {dataset_name}...')
  examples = datasets.load_examples(
      dataset_name=dataset_name,
      max_number_of_examples=max_examples,
  )
  print(f'Loaded {len(examples)} examples.')

  if metric_functions is None:
    metric_functions = rag_metrics.get_default_rag_metrics()

  print('Running evaluation...')
  dict_examples = [dict(ex) for ex in examples]
  summary = qa_evaluation.evaluate_qa(
      rag_spec=spec,
      examples=dict_examples,
      metric_functions=metric_functions,
      runtime_kwargs=rt_kwargs,
      output_results=True,
      enable_tracing=True,
  )

  print(f'Evaluation Metrics: {summary.metrics}')

  # 5. Save Results as HTML report
  html_report_path = qa_evaluation.save_html_report(
      summary=summary,
      output_dir_path=output_directory,
  )
  print(f'HTML report saved to {html_report_path}')
  return summary


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  evaluate_rag(
      dataset_name=_DATASET_NAME.value,
      index_directory=_INDEX_DIRECTORY.value,
      output_directory=_OUTPUT_DIRECTORY.value,
      generate_model_backend_class_name=_GENERATE_MODEL_BACKEND_CLASS_NAME.value,
      generate_model_name=_GENERATE_MODEL_NAME.value,
      qa_strategy_name=_QA_STRATEGY_NAME.value,
      query_time_kwargs=_QUERY_TIME_KWARGS.value,
      runtime_kwargs=_RUNTIME_KWARGS.value,
      max_examples=_MAX_EXAMPLES.value,
  )


if __name__ == '__main__':
  app.run(main)
