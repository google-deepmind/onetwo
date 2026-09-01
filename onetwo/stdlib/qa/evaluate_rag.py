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
from onetwo.stdlib.qa import evaluate_rag_lib


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
    'gemini-3.7-flash',
    'Name of the generation model.',
)

_AI_RATER_GENERATE_MODEL_NAME = flags.DEFINE_string(
    'ai_rater_generate_model_name',
    'gemini-3.7-flash',
    'Name of the generation model used by the AI rater (accuracy critic).',
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

_MCP_KWARGS = flags.DEFINE_string(
    'mcp_kwargs',
    '{}',
    'JSON string containing strategy arguments passed at strategy'
    ' construction.',
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    None,
    'Maximum number of examples to evaluate.',
)

_MAX_RETRIES = flags.DEFINE_integer(
    'max_retries',
    5,
    'Maximum number of retries for backend requests.',
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    300000,
    'HTTP timeout in milliseconds for backend requests (default: 5 mins).',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  runtime_kwargs = json.loads(_RUNTIME_KWARGS.value)

  evaluate_rag_lib.evaluate_rag(
      dataset_name=_DATASET_NAME.value,
      index_directory=_INDEX_DIRECTORY.value or '',
      output_directory=_OUTPUT_DIRECTORY.value,
      generate_model_backend_class_name=_GENERATE_MODEL_BACKEND_CLASS_NAME.value,
      generate_model_name=_GENERATE_MODEL_NAME.value,
      ai_rater_generate_model_name=_AI_RATER_GENERATE_MODEL_NAME.value,
      qa_strategy_name=_QA_STRATEGY_NAME.value,
      query_time_kwargs=_QUERY_TIME_KWARGS.value,
      mcp_kwargs=_MCP_KWARGS.value,
      runtime_kwargs=runtime_kwargs,
      max_examples=_MAX_EXAMPLES.value,
      max_retries=_MAX_RETRIES.value,
      http_timeout=_HTTP_TIMEOUT.value,
  )


if __name__ == '__main__':
  app.run(main)
