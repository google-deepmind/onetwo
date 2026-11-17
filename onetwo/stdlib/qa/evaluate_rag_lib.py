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
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import caching
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec
from onetwo.stdlib.retrieval import rag_defaults


def get_results_directory(
    *,
    output: str | epath.Path | None,
    index_directory: str | epath.Path,
    dataset_name: str,
    qa_strategy_name: str,
    generate_model_name: str,
    max_results: int | str,
) -> epath.Path:
  """Returns the directory where the results of an evaluation are saved.

  Results of evaluations with different settings go to different directories:
  - `<output>/<dataset>/<strategy>/<model>__max_results_<N>/` if `output` is
    set;
  - else `<index_directory>/eval_results/<strategy>/<model>__max_results_<N>/`,
    next to the evaluated index (whose directory already names the dataset).

  Args:
    output: Base directory of the results, or None to save them in the index
      directory.
    index_directory: Directory of the evaluated index.
    dataset_name: Name of the dataset.
    qa_strategy_name: Name of the QA strategy.
    generate_model_name: Name of the generation model.
    max_results: Number of retrieved documents (`max_results` query-time arg).

  Raises:
    ValueError: If neither `output` nor `index_directory` is set.
  """
  run_name = f'{generate_model_name}__max_results_{max_results}'
  if output:
    return (
        epath.Path(output)
        / dataset_name
        / qa_strategy_name
        / run_name
    )
  if not index_directory:
    raise ValueError('Set `output` or `index_directory`.')
  return (
      epath.Path(index_directory) / 'eval_results' / qa_strategy_name / run_name
  )


def get_default_runtime_kwargs(
    runtime_kwargs: Mapping[str, Any] | None,
    *,
    max_retries: int = 5,
    http_timeout_ms: int = 300000,
) -> dict[str, Any]:
  """Returns `runtime_kwargs` with default backend request settings.

  Settings already in `runtime_kwargs` are kept. The input is not modified.

  Args:
    runtime_kwargs: Arguments for backends (e.g. API key).
    max_retries: Maximum number of retries of backend requests, set as
      `'max_retries'` if absent.
    http_timeout_ms: HTTP timeout of backend requests in milliseconds, set as
      `'http_options': {'timeout': http_timeout_ms}` if `'http_options'` is
      absent, or as its `'timeout'` if `'http_options'` is a dict without one.
      If `'http_options'` is explicitly `None`, `'http_options'` is omitted.

  Returns:
    A new dict of runtime kwargs with the defaults applied.
  """
  rt_kwargs = dict(runtime_kwargs) if runtime_kwargs else {}
  rt_kwargs.setdefault('max_retries', max_retries)
  http_options = rt_kwargs.get('http_options')
  if 'http_options' not in rt_kwargs:
    rt_kwargs['http_options'] = {'timeout': http_timeout_ms}
  elif http_options is None:
    del rt_kwargs['http_options']
  elif isinstance(http_options, dict) and 'timeout' not in http_options:
    rt_kwargs['http_options'] = {**http_options, 'timeout': http_timeout_ms}
  return rt_kwargs


def evaluate_rag(
    dataset_name: str,
    output: str | None = None,
    runtime_kwargs: Mapping[str, Any] | None = None,
    index_directory: str = '',
    generate_model_backend_class_name: str = (
        rag_defaults.DEFAULT_QA_GENERATE_BACKEND_CLASS_NAME
    ),
    generate_model_name: str = rag_defaults.DEFAULT_QA_GENERATE_MODEL_NAME,
    generate_backend_kwargs: Mapping[str, Any] | None = None,
    ai_rater_generate_model_name: str = (
        rag_defaults.DEFAULT_AI_RATER_GENERATE_MODEL_NAME
    ),
    qa_strategy_name: str = 'rag',
    query_time_kwargs: Mapping[str, Any] | None = None,
    mcp_kwargs: Mapping[str, Any] | None = None,
    max_examples: int | None = 5,
    metric_functions: Mapping[str, Any] | None = None,
    cache: bool = False,
) -> results.EvaluationSummary:
  """Evaluates a RAG strategy on a QA dataset.

  Args:
    dataset_name: Name of the dataset to load from the registry.
    output: Base directory of the results (HTML report and cache), or None to
      save them in `index_directory` (see `get_results_directory`).
    runtime_kwargs: Runtime arguments for the backends (e.g. API key). Backend
      requests default to 5 retries (`'max_retries'`) and a 300000 ms timeout
      (`'http_options': {'timeout': ...}`); see `get_default_runtime_kwargs`.
    index_directory: Directory containing the pre-built index.
    generate_model_backend_class_name: Generation backend class name.
    generate_model_name: Generation model name.
    generate_backend_kwargs: Constructor kwargs of the generation backend, e.g.
      `{'chat_kwargs': {'thinking_config': {'thinking_level': 'HIGH'}}}` for a
      `GoogleGenAIAPI` backend. If None, defaults to
      `rag_defaults.qa_generate_backend_kwargs(generate_model_name)`, i.e.
      medium thinking for the default model and no kwargs for other models.
    ai_rater_generate_model_name: Generation model name used by the AI rater.
    qa_strategy_name: QA strategy name.
    query_time_kwargs: Query-time arguments for the strategy. If None, defaults
      to `{'max_results': 20}`.
    mcp_kwargs: Arguments of the `agentic_retrieval` strategy, e.g.
      `mcp_server_urls`, a comma-separated list of MCP server URLs.
    max_examples: Max examples to evaluate.
    metric_functions: Optional mapping of metric names to metric functions. When
      None, defaults to `rag_metrics.get_default_rag_metrics()`.
    cache: Whether to cache LLM calls in `<results directory>/cache.json` and
      reuse the cache on reruns.

  Returns:
    EvaluationSummary object containing metrics and detailed results.
  """
  q_time_kwargs = (
      dict(query_time_kwargs)
      if query_time_kwargs is not None
      else {'max_results': 20}
  )
  mcp_kwargs = dict(mcp_kwargs) if mcp_kwargs else {}
  rt_kwargs = get_default_runtime_kwargs(runtime_kwargs)

  results_directory = get_results_directory(
      output=output,
      index_directory=index_directory,
      dataset_name=dataset_name,
      qa_strategy_name=qa_strategy_name,
      generate_model_name=generate_model_name,
      max_results=q_time_kwargs.get('max_results', 20),
  )

  fn_cache = None
  cache_filename = str(results_directory / 'cache.json')
  if cache:
    print(f'Using cache file: {cache_filename}')
    fn_cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    if epath.Path(cache_filename).exists():
      print(f'Loading existing cache from {cache_filename}...')
      fn_cache.load(overwrite=True)
    else:
      print('No existing cache found. A new cache will be created.')

  if fn_cache is not None:
    rt_kwargs['cache'] = fn_cache

  gen_backend_config = backend_registry.GenerateBackendConfig(
      backend_class_name=generate_model_backend_class_name,
      model_name=generate_model_name,
      kwargs=(
          dict(generate_backend_kwargs)
          if generate_backend_kwargs is not None
          else rag_defaults.qa_generate_backend_kwargs(generate_model_name)
      ),
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
        max_results=q_time_kwargs.get('max_results', None),
        retrieval_only=(qa_strategy_name == 'retrieval_only'),
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

  if fn_cache is not None:
    print(f'Saving cache to {cache_filename}...')
    fn_cache.save(overwrite=True)
    print(f'Cache saved successfully to {cache_filename}.')

  print(f'Evaluation Metrics: {summary.metrics}')

  # 5. Save Results as HTML report
  html_report_path = qa_evaluation.save_html_report(
      summary=summary,
      output_dir_path=results_directory,
  )
  print(f'HTML report saved to {html_report_path}')
  return summary
