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

"""Library to evaluate a RAG strategy on a QA dataset."""

from collections.abc import Mapping
import json
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import caching
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec


# TODO: Move this function to a shared library and add tests.
def get_rag_cache_filename(
    output_directory: str | epath.Path,
    dataset_name: str,
    qa_strategy_name: str,
    generate_model_name: str,
    index_directory: str | epath.Path = '',
    max_results: int | str = 20,
) -> str:
  """Generates a cache filename following the hierarchical template.

  Template:
    {output_directory}/cache/{dataset}/{strategy}/{model_name}/{index_name}__max_results_{max_results}.json

  Args:
    output_directory: Base directory where results/caches are saved.
    dataset_name: Name of the dataset.
    qa_strategy_name: Name of the QA strategy.
    generate_model_name: Name of the generation model.
    index_directory: Path to pre-built index directory if used.
    max_results: Max results parameter from query_time_kwargs.

  Returns:
    Full path to the cache JSON file as a string.
  """
  if index_directory:
    index_name = epath.Path(index_directory).name
    filename = f'{index_name}__max_results_{max_results}.json'
  else:
    filename = f'max_results_{max_results}.json'

  cache_path = (
      epath.Path(output_directory)
      / 'cache'
      / dataset_name
      / qa_strategy_name
      / generate_model_name
      / filename
  )
  return str(cache_path)


def evaluate_rag(
    dataset_name: str,
    output_directory: str,
    runtime_kwargs: Mapping[str, Any] | str,
    index_directory: str = '',
    generate_model_backend_class_name: str = 'GoogleGenAIAPI',
    generate_model_name: str = 'gemini-3.7-flash',
    ai_rater_generate_model_name: str = 'gemini-3.7-flash',
    qa_strategy_name: str = 'rag',
    query_time_kwargs: Mapping[str, Any] | str = '{"max_results": 20}',
    mcp_kwargs: Mapping[str, Any] | str = '{}',
    max_examples: int | None = 5,
    metric_functions: Mapping[str, Any] | None = None,
    max_retries: int | None = 5,
    http_timeout: int | None = 300000,
    cache_results: bool = False,
    cache_filename: str | None = None,
) -> results.EvaluationSummary:
  """Evaluates a RAG strategy on a QA dataset.

  Args:
    dataset_name: Name of the dataset to load from the registry.
    output_directory: Directory where evaluation results will be saved.
    runtime_kwargs: Runtime arguments dict or JSON string (e.g. API key).
    index_directory: Directory containing the pre-built index.
    generate_model_backend_class_name: Generation backend class name.
    generate_model_name: Generation model name.
    ai_rater_generate_model_name: Generation model name used by the AI rater.
    qa_strategy_name: QA strategy name.
    query_time_kwargs: Query-time arguments dict or JSON string.
    mcp_kwargs: Arguments passed to get_strategy at construction (e.g.,
      mcp_server_urls).
    max_examples: Max examples to evaluate.
    metric_functions: Optional mapping of metric names to metric functions.
      Defaults to rag_metrics.get_default_rag_metrics().
    max_retries: Maximum number of retries for backend requests.
    http_timeout: HTTP timeout in milliseconds for backend requests.
    cache_results: Whether to enable caching for LLM generation.
    cache_filename: Explicit cache filename. If not provided and cache_results
      is True, it will be auto-generated.

  Returns:
    EvaluationSummary object containing metrics and detailed results.
  """
  if qa_strategy_name == 'mcp':
    try:
      # Keeping the import here to avoid importing in internal colab runtimes.
      from onetwo.stdlib.qa import mcp_strategies  # pyrefly: ignore[missing-module-attribute] # pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      print('mcp_strategies is not linked. Execution may fail.')

  if isinstance(query_time_kwargs, str):
    q_time_kwargs = json.loads(query_time_kwargs)
  else:
    q_time_kwargs = dict(query_time_kwargs)

  if isinstance(mcp_kwargs, str):
    mcp_kwargs = json.loads(mcp_kwargs)
  else:
    mcp_kwargs = dict(mcp_kwargs)

  if isinstance(runtime_kwargs, str):
    rt_kwargs = json.loads(runtime_kwargs)
  else:
    rt_kwargs = dict(runtime_kwargs) if runtime_kwargs else {}

  if 'max_retries' not in rt_kwargs and max_retries is not None:
    rt_kwargs['max_retries'] = max_retries
  if 'http_options' not in rt_kwargs and http_timeout is not None:
    rt_kwargs['http_options'] = {'timeout': http_timeout}
  elif 'http_options' in rt_kwargs and isinstance(
      rt_kwargs['http_options'], dict
  ):
    if 'timeout' not in rt_kwargs['http_options'] and http_timeout is not None:
      rt_kwargs['http_options']['timeout'] = http_timeout

  cache = None
  if cache_results or cache_filename:
    if not cache_filename:
      max_results = q_time_kwargs.get('max_results', 20)
      cache_filename = get_rag_cache_filename(
          output_directory=output_directory,
          dataset_name=dataset_name,
          qa_strategy_name=qa_strategy_name,
          generate_model_name=generate_model_name,
          index_directory=index_directory,
          max_results=max_results,
      )
    print(f'Using cache file: {cache_filename}')
    cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    if epath.Path(cache_filename).exists():
      print(f'Loading existing cache from {cache_filename}...')
      cache.load(overwrite=True)
    else:
      print('No existing cache found. A new cache will be created.')

  if cache is not None:
    rt_kwargs['cache'] = cache

  gen_backend_config = backend_registry.GenerateBackendConfig(
      backend_class_name=generate_model_backend_class_name,
      model_name=generate_model_name,
  )
  ai_rater_backend_config = backend_registry.GenerateBackendConfig(
      backend_class_name=generate_model_backend_class_name,
      model_name=ai_rater_generate_model_name,
  )
  spec = qa_spec.QAStrategySpec(
      qa_strategy_name=qa_strategy_name,
      index_path=index_directory,
      generate_backend_config=gen_backend_config,
      query_time_kwargs=q_time_kwargs,
      mcp_kwargs=mcp_kwargs,
  )

  # 2. Load Dataset Examples
  print(f'Loading dataset {dataset_name}...')
  examples = datasets.load_examples(
      dataset_name=dataset_name,
      max_number_of_examples=max_examples,
  )
  print(f'Loaded {len(examples)} examples.')

  if metric_functions is None:
    metric_functions = rag_metrics.get_default_rag_metrics(
        accuracy_critic_backend_config=ai_rater_backend_config,
        runtime_kwargs=rt_kwargs,
    )

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

  if cache is not None:
    print(f'Saving cache to {cache_filename}...')
    cache.save(overwrite=True)
    print(f'Cache saved successfully to {cache_filename}.')

  print(f'Evaluation Metrics: {summary.metrics}')

  # 5. Save Results as HTML report
  html_report_path = qa_evaluation.save_html_report(
      summary=summary,
      output_dir_path=output_directory,
  )
  print(f'HTML report saved to {html_report_path}')
  return summary
