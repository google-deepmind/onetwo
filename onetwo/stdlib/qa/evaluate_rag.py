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
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import executing as ot_executing
from onetwo.core import results as ot_results
from onetwo.evaluation import datasets
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    'browsecomp_plus',
    'Name of the dataset to load from the registry.',
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
    5,
    'Maximum number of examples to evaluate.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gen_backend_config = backend_registry.GenerateBackendConfig(
      backend_class_name=_GENERATE_MODEL_BACKEND_CLASS_NAME.value,
      model_name=_GENERATE_MODEL_NAME.value,
  )
  spec = qa_spec.QAStrategySpec(
      qa_strategy_name=_QA_STRATEGY_NAME.value,
      index_path=_INDEX_DIRECTORY.value,
      generate_backend_config=gen_backend_config,
      query_time_kwargs=json.loads(_QUERY_TIME_KWARGS.value),
  )

  # 2. Load Dataset Examples
  # TODO: Consider making this a helper function.
  print(f'Loading dataset {_DATASET_NAME.value}...')
  loader = datasets.get_dataset_loader(
      _DATASET_NAME.value,
      max_number_of_examples=_MAX_EXAMPLES.value,
  )
  examples = ot_executing.run(loader.load_examples())
  print(f'Loaded {len(list(examples))} examples.')

  metric_functions = {
      'accuracy': rag_metrics.make_accuracy_critic(),
      'recall@5': rag_metrics.make_recall_at_k(k=5),
      'recall@20': rag_metrics.make_recall_at_k(k=20),
  }

  print('Running evaluation...')
  dict_examples = [dict(ex) for ex in examples]
  summary = qa_evaluation.evaluate_qa(
      rag_spec=spec,
      examples=dict_examples,
      metric_functions=metric_functions,
      runtime_kwargs=json.loads(_RUNTIME_KWARGS.value),
      output_results=True,
      enable_tracing=True,
  )

  print(f'Evaluation Metrics: {summary.metrics}')

  # 5. Save Results as HTML report
  # TODO: Consider making this a helper function.
  output_dir = epath.Path(_OUTPUT_DIRECTORY.value)
  output_dir.mkdir(parents=True, exist_ok=True)
  html_report_path = output_dir / 'results.html'
  html_report_path.write_text(
      ot_results.HTMLRenderer(levels_to_expand=2).render(summary)
  )
  print(f'HTML report saved to {html_report_path}')


if __name__ == '__main__':
  app.run(main)
