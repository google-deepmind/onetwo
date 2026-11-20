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

from collections.abc import Sequence
import dataclasses
import json
import os

from absl import app
from absl import flags
from absl import logging
from onetwo.backends import backend_registry
from onetwo.core import content as content_lib
from onetwo.evaluation import datasets
from onetwo.stdlib.qa import evaluate_rag_lib
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import rag_defaults

_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult

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
        'Parent directory of the evaluation runs. Default:'
        ' <index>/eval_results; without --index, results are not saved.'
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
    None,
    (
        'Name of the generation model. If unset, uses the default QA model of'
        ' --generate_model_backend_class_name.'
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

_QA_STRATEGY_NAME = flags.DEFINE_enum(
    'qa_strategy_name',
    'rag',
    ['closed_book', 'rag', 'retrieval_only', 'agentic_retrieval'],
    'Ready-made QA strategy to evaluate.',
)

_MAX_RESULTS = flags.DEFINE_integer(
    'max_results',
    20,
    (
        'Maximum number of documents to retrieve per question (for strategies'
        ' with a retriever).'
    ),
)

_MIN_SCORE = flags.DEFINE_float(
    'min_score',
    None,
    (
        'Minimum similarity score of retrieved documents (for strategies with a'
        ' retriever).'
    ),
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    None,
    (
        'Maximum number of examples to evaluate (registered datasets only;'
        ' raises an error when --examples is a file path).'
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


def _strategy_from_flags() -> qa.QAStrategy[_PromptContent, _StrategyResult]:
  """Returns the preset --qa_strategy_name with the settings of the flags."""
  name = _QA_STRATEGY_NAME.value
  if name in (
      evaluate_rag_lib.StrategyType.RAG,
      evaluate_rag_lib.StrategyType.RETRIEVAL_ONLY,
  ):
    strategy = strategies_lib.get_strategy(
        name,
        max_results=_MAX_RESULTS.value,
        min_score=_MIN_SCORE.value,
    )
  else:
    strategy = strategies_lib.get_strategy(name)
  if isinstance(strategy, backend_registry.BackendMixin):
    config = rag_defaults.qa_generate_backend_config(
        _GENERATE_MODEL_BACKEND_CLASS_NAME.value,
        _GENERATE_MODEL_NAME.value,
    )
    strategy = dataclasses.replace(strategy, backend_config=config)
  return strategy


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  output = _OUTPUT.value or None
  if output is None and _INDEX.value:
    # By default, the runs are saved next to the index.
    output = os.path.join(_INDEX.value, 'eval_results')
  if output is None:
    logging.warning('Results are not saved: pass --output to save them.')

  evaluate_rag_lib.evaluate_rag(
      examples=_examples_from_flag(_EXAMPLES.value, _MAX_EXAMPLES.value),
      index=_INDEX.value or None,
      strategy=_strategy_from_flags(),
      mcp_urls=_MCP_URLS.value,
      output=output,
      extra_backend_kwargs=json.loads(_EXTRA_BACKEND_KWARGS.value),
  )


if __name__ == '__main__':
  app.run(main)
