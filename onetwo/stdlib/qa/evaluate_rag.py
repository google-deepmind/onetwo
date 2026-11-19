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
from onetwo.evaluation import datasets
from onetwo.stdlib.qa import evaluate_rag_lib
from onetwo.stdlib.retrieval import rag_defaults

_EXAMPLES = flags.DEFINE_string(
    'examples',
    'browsecomp_plus',
    'Registered dataset name or path to a .jsonl/.json examples file.',
    allow_override=True,
)

_INDEX = flags.DEFINE_string(
    'index',
    None,
    'Directory containing the pre-built index.',
)

_MCP_URLS = flags.DEFINE_list(
    'mcp_urls',
    [],
    (
        'Comma-separated URLs of MCP servers whose tools the agentic_retrieval'
        ' strategy can call.'
    ),
)

_OUTPUT = flags.DEFINE_string(
    'output',
    '',
    (
        'Base directory of the evaluation results, saved under'
        ' <output>/<dataset>/<strategy>/<model>__max_results_<N>/. If'
        ' empty, they are saved next to the index, under'
        ' <index>/eval_results/<strategy>/<model>__max_results_<N>/.'
    ),
    allow_override=True,
)

_GENERATE_MODEL_BACKEND_CLASS_NAME = flags.DEFINE_string(
    'generate_model_backend_class_name',
    rag_defaults.DEFAULT_BACKEND_CLASS_NAME,
    'Name of the generation backend class.',
)

_GENERATE_MODEL_NAME = flags.DEFINE_string(
    'generate_model_name',
    rag_defaults.DEFAULT_GENERATE_MODEL,
    'Name of the generation model.',
)

_GENERATE_BACKEND_KWARGS = flags.DEFINE_string(
    'generate_backend_kwargs',
    None,
    (
        'JSON string of constructor kwargs of the generation backend, e.g.'
        ' \'{"chat_kwargs": {"thinking_config": {"thinking_level": "HIGH"}}}\'.'
        ' If unset, defaults to medium thinking for the default model and to no'
        ' kwargs for other models (see'
        ' `rag_defaults.qa_generate_backend_config`).'
    ),
)

_EXTRA_BACKEND_KWARGS = flags.DEFINE_string(
    'extra_backend_kwargs',
    '{}',
    'JSON string of extra keyword arguments for the constructors of the'
    ' backends (e.g. an API key, or cache_filename for a shared cache).'
    ' Usually not needed: credentials are read from the environment (e.g.'
    ' GOOGLE_API_KEY).',
    allow_override=True,
)

_QA_STRATEGY_NAME = flags.DEFINE_string(
    'qa_strategy_name',
    'rag',
    'Name of the QA strategy to evaluate.',
)

# TODO: Consider combining with _EXTRA_BACKEND_KWARGS.
_QUERY_TIME_KWARGS = flags.DEFINE_string(
    'query_time_kwargs',
    '{"max_results": 20}',
    'JSON string containing query-time arguments for the strategy.',
)

_MCP_KWARGS = flags.DEFINE_string(
    'mcp_kwargs',
    '{}',
    'JSON string containing extra constructor arguments for the'
    ' agentic_retrieval strategy, e.g. \'{"max_steps": 30}\'. Pass MCP server'
    ' URLs via --mcp_urls.',
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    None,
    (
        'Maximum number of examples to evaluate (registered datasets only;'
        ' raises an error when --examples is a file path).'
    ),
)

_MAX_RETRIES = flags.DEFINE_integer(
    'max_retries',
    5,
    (
        'Maximum number of retries for backend requests. Ignored if set in'
        ' --extra_backend_kwargs.'
    ),
)

_HTTP_TIMEOUT = flags.DEFINE_integer(
    'http_timeout',
    300000,
    (
        'HTTP timeout in milliseconds for backend requests (default: 5 mins).'
        ' Ignored if set in --extra_backend_kwargs.'
    ),
)


def _examples_from_flag(
    value: str, max_examples: int | None
) -> str | datasets.DatasetLoader:
  """Resolves --examples to a registered DatasetLoader or file path."""
  registered = datasets.list_dataset_loaders()
  if value in registered:
    return datasets.get_dataset_loader(
        value, max_number_of_examples=max_examples
    )
  if max_examples is not None:
    raise app.UsageError(
        f'--max_examples only applies to registered datasets ({registered}).'
    )
  return value


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  evaluate_rag_lib.evaluate_rag(
      examples=_examples_from_flag(_EXAMPLES.value, _MAX_EXAMPLES.value),
      index=_INDEX.value or None,
      output=_OUTPUT.value or None,
      generate_model_backend_class_name=_GENERATE_MODEL_BACKEND_CLASS_NAME.value,
      generate_model_name=_GENERATE_MODEL_NAME.value,
      generate_backend_kwargs=(
          json.loads(_GENERATE_BACKEND_KWARGS.value)
          if _GENERATE_BACKEND_KWARGS.value is not None
          else None
      ),
      qa_strategy_name=_QA_STRATEGY_NAME.value,
      query_time_kwargs=json.loads(_QUERY_TIME_KWARGS.value),
      mcp_kwargs=json.loads(_MCP_KWARGS.value),
      mcp_urls=_MCP_URLS.value,
      extra_backend_kwargs=evaluate_rag_lib.get_default_extra_backend_kwargs(
          json.loads(_EXTRA_BACKEND_KWARGS.value),
          max_retries=_MAX_RETRIES.value,
          http_timeout_ms=_HTTP_TIMEOUT.value,
      ),
  )


if __name__ == '__main__':
  app.run(main)
