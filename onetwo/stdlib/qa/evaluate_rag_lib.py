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
import dataclasses
import logging
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import caching
from onetwo.core import executing as ot_executing
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import qa_evaluation
from onetwo.stdlib.qa import qa_spec
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults


def _examples_name(examples: str | datasets.DatasetLoader) -> str:
  """Returns the name of `examples`, used in the results directory path."""
  if isinstance(examples, str):
    return epath.Path(examples).stem
  return examples.dataset_name or type(examples).__name__


def _load_examples(
    examples: str | datasets.DatasetLoader,
) -> list[dict[str, Any]]:
  """Loads examples from a file path or `DatasetLoader`."""
  print(f'Loading examples ({_examples_name(examples)})...')
  if isinstance(examples, str):
    examples = datasets.JsonlDatasetLoader(examples_path=examples)
  loaded = [dict(ex) for ex in ot_executing.run(examples.load_examples())]
  print(f'Loaded {len(loaded)} examples.')
  return loaded


def get_results_directory(
    *,
    output: str | epath.Path | None,
    index: str | epath.Path | indexing.DocumentIndex,
    dataset_name: str,
    qa_strategy_name: str,
    generate_model_name: str,
    max_results: int | str,
) -> epath.Path | None:
  """Returns the directory where the results of an evaluation are saved.

  Results of evaluations with different settings go to different directories:
  - `<output>/<dataset>/<strategy>/<model>__max_results_<N>/` if `output` is
    set;
  - `<index>/eval_results/<strategy>/<model>__max_results_<N>/`, next to the
    evaluated index (whose directory already names the dataset), if `index` is a
    directory;
  - else `None` (no directory on disk to save results).

  Args:
    output: Base directory of the results, or None to save them in the index
      directory.
    index: Directory of the evaluated index, or an index object.
    dataset_name: Name of the dataset.
    qa_strategy_name: Name of the QA strategy.
    generate_model_name: Name of the generation model.
    max_results: Number of retrieved documents (`max_results` query-time arg).
  """
  run_name = f'{generate_model_name}__max_results_{max_results}'
  if output:
    return (
        epath.Path(output)
        / dataset_name
        / qa_strategy_name
        / run_name
    )
  if not isinstance(index, (str, epath.Path)) or not index:
    logging.warning(
        'Neither output nor an index directory was provided; HTML report will'
        ' not be saved to disk.'
    )
    return None
  return epath.Path(index) / 'eval_results' / qa_strategy_name / run_name


def get_default_extra_backend_kwargs(
    extra_backend_kwargs: Mapping[str, Any] | None,
    *,
    max_retries: int = 5,
    http_timeout_ms: int = 300000,
) -> dict[str, Any]:
  """Returns `extra_backend_kwargs` with default backend request settings.

  Settings already in `extra_backend_kwargs` are kept. The input is not
  modified.

  Args:
    extra_backend_kwargs: Extra keyword arguments for the backends.
    max_retries: Maximum number of retries of backend requests, set as
      `'max_retries'` if absent.
    http_timeout_ms: HTTP timeout of backend requests in milliseconds, set as
      `'http_options': {'timeout': http_timeout_ms}` if `'http_options'` is
      absent, or as its `'timeout'` if `'http_options'` is a dict without one.
      If `'http_options'` is explicitly `None`, `'http_options'` is omitted.

  Returns:
    A new dict of extra backend kwargs with the defaults applied.
  """
  rt_kwargs = dict(extra_backend_kwargs) if extra_backend_kwargs else {}
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
    examples: str | datasets.DatasetLoader,
    output: str | None = None,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
    index: str | indexing.DocumentIndex = '',
    generate_model_backend_class_name: str = (
        rag_defaults.DEFAULT_BACKEND_CLASS_NAME
    ),
    generate_model_name: str | None = None,
    generate_backend_kwargs: Mapping[str, Any] | None = None,
    qa_strategy_name: str = 'rag',
    query_time_kwargs: Mapping[str, Any] | None = None,
    mcp_kwargs: Mapping[str, Any] | None = None,
    evaluate_kwargs: Mapping[str, Any] | None = None,
) -> results.EvaluationSummary:
  """Evaluates a RAG strategy on a QA dataset.

  Args:
    examples: Path to a `.jsonl` or `.json` examples file, or a
      `datasets.DatasetLoader` (e.g. `datasets.get_dataset_loader(name,
      max_number_of_examples=N)`).
    output: Base directory of the HTML report, or None to save it in `index`
      when `index` is a directory (see `get_results_directory`).
    extra_backend_kwargs: Extra keyword arguments for the constructors of the
      backends that this call creates (e.g. an API key): those of the index and
      the QA and AI rater generation backends. Usually not needed: credentials
      are read from the environment (e.g. `GOOGLE_API_KEY`). Not a stable part
      of the API. Backend requests default to 5 retries (`'max_retries'`) and a
      300000 ms timeout (`'http_options': {'timeout': ...}`); see
      `get_default_extra_backend_kwargs`. If `'cache_filename'` is set,
      `evaluate_rag` creates one shared `caching.SimpleFunctionCache` loaded
      from and saved to that path, passes it as `'cache'` to all backends, and
      does not forward `'cache_filename'` itself, so caching works with every
      backend accepting `'cache'`.
    index: Directory of a saved index or a `DocumentIndex` object. Can only be
      `''` when `qa_strategy_name` is `'agentic_retrieval'` or `'closed_book'`.
    generate_model_backend_class_name: Generation backend class name.
    generate_model_name: Generation model name. If None, the default QA model of
      `generate_model_backend_class_name` (see
      `rag_defaults.default_qa_generate_backend_config`).
    generate_backend_kwargs: Constructor kwargs of the generation backend, e.g.
      `{'chat_kwargs': {'thinking_config': {'thinking_level': 'HIGH'}}}` for a
      `GoogleGenAIAPI` backend. If None, the kwargs of the default QA config
      when `generate_model_name` is None (e.g. medium thinking for
      `GoogleGenAIAPI`), and no kwargs otherwise.
    qa_strategy_name: QA strategy name.
    query_time_kwargs: Query-time arguments for the strategy. If None, defaults
      to `{'max_results': 20}`.
    mcp_kwargs: Arguments of the `agentic_retrieval` strategy, e.g.
      `mcp_server_urls`, a comma-separated list of MCP server URLs.
    evaluate_kwargs: Keyword arguments forwarded to `ot.evaluate` (e.g.
      `metric_functions`, `chunk_size`). Defaults `'output_results'` to True and
      `'metric_functions'` to `rag_metrics.get_default_rag_metrics(...)` with
      the default AI rater when absent.

  Returns:
    EvaluationSummary object containing metrics and detailed results.

  Raises:
    ValueError: If `index` is `''` and `qa_strategy_name` is neither
      `'agentic_retrieval'` nor `'closed_book'`.
  """
  if isinstance(index, str) and not index and qa_strategy_name not in (
      'agentic_retrieval',
      'closed_book',
  ):
    raise ValueError(
        f'index must be provided for qa_strategy_name={qa_strategy_name!r}.'
    )
  q_time_kwargs = (
      dict(query_time_kwargs)
      if query_time_kwargs is not None
      else {'max_results': 20}
  )
  mcp_kwargs = dict(mcp_kwargs) if mcp_kwargs else {}
  rt_kwargs = get_default_extra_backend_kwargs(extra_backend_kwargs)
  cache_filename = rt_kwargs.pop('cache_filename', None)

  fn_cache = None
  if cache_filename:
    print(f'Using cache file: {cache_filename}')
    fn_cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
    if epath.Path(cache_filename).exists():
      print(f'Loading existing cache from {cache_filename}...')
      fn_cache.load(overwrite=True)
    else:
      print('No existing cache found. A new cache will be created.')
    rt_kwargs['cache'] = fn_cache

  gen_backend_config = rag_defaults.qa_generate_backend_config(
      generate_model_backend_class_name, generate_model_name
  )
  generate_model_name = gen_backend_config.model_name or ''
  if generate_backend_kwargs is not None:
    gen_backend_config = dataclasses.replace(
        gen_backend_config, kwargs=dict(generate_backend_kwargs)
    )

  results_directory = get_results_directory(
      output=output,
      index=index,
      dataset_name=_examples_name(examples),
      qa_strategy_name=qa_strategy_name,
      generate_model_name=generate_model_name,
      max_results=q_time_kwargs.get('max_results', 20),
  )

  spec = qa_spec.QAStrategySpec(
      qa_strategy_name=qa_strategy_name,
      generate_backend_config=gen_backend_config,
      query_time_kwargs=q_time_kwargs,
      mcp_kwargs=mcp_kwargs,
  )

  dict_examples = _load_examples(examples)

  evaluate_kwargs = dict(evaluate_kwargs) if evaluate_kwargs else {}
  evaluate_kwargs.setdefault('output_results', True)
  if 'metric_functions' not in evaluate_kwargs:
    ai_rater_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name=rag_defaults.DEFAULT_BACKEND_CLASS_NAME,
        model_name=rag_defaults.DEFAULT_AI_RATER_GENERATE_MODEL_NAME,
    )
    evaluate_kwargs['metric_functions'] = rag_metrics.get_default_rag_metrics(
        max_results=q_time_kwargs.get('max_results', None),
        retrieval_only=(qa_strategy_name == 'retrieval_only'),
        accuracy_critic_backend_config=ai_rater_backend_config,
        runtime_kwargs=rt_kwargs,
    )

  print('Running evaluation...')
  summary = qa_evaluation.evaluate_qa(
      rag_spec=spec,
      examples=dict_examples,
      index=index,
      runtime_kwargs=rt_kwargs,
      evaluate_kwargs=evaluate_kwargs,
  )

  if fn_cache is not None:
    print(f'Saving cache to {cache_filename}...')
    fn_cache.save(overwrite=True)
    print(f'Cache saved successfully to {cache_filename}.')

  print(f'Evaluation Metrics: {summary.metrics}')

  # 5. Save Results as HTML report
  if results_directory is not None:
    html_report_path = qa_evaluation.save_html_report(
        summary=summary,
        output_dir_path=results_directory,
    )
    print(f'HTML report saved to {html_report_path}')
  return summary
