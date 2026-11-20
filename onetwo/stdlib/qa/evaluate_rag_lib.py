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

"""Library to evaluate a RAG strategy on a QA dataset.

`evaluate_rag` resolves the strategy (a ready-made `StrategyType` or a strategy
object), attaches the index or MCP servers it searches, runs `ot.evaluate` on
the examples and, optionally, saves the results.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any

from absl import logging
from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing as ot_executing
from onetwo.core import results
from onetwo.evaluation import datasets
from onetwo.evaluation import evaluation as ot_evaluation
from onetwo.evaluation import rag_metrics
from onetwo.stdlib.qa import agentic_retrieval
from onetwo.stdlib.qa import evaluate_rag_utils
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import rag_defaults

_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult
_Strategy = qa.QAStrategy[_PromptContent, _StrategyResult]

# Default backend request settings of `evaluate_rag`. The timeout is longer than
# for index building, since QA strategies may generate long answers.
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_HTTP_TIMEOUT_MS = 300_000

# Number of documents retrieved by the ready-made retrieval strategies.
_DEFAULT_MAX_RESULTS = 20


class StrategyType(str, enum.Enum):
  """Ready-made QA strategies, with the defaults of rag.evaluate.

  CLOSED_BOOK: Answers without retrieving.
  RAG: Retrieves up to 20 documents from index once, then answers.
  RETRIEVAL_ONLY: Retrieves up to 20 documents from index, without answering.
  AGENTIC_RETRIEVAL: Searches the MCP servers in mcp_urls until it can answer.
  """

  CLOSED_BOOK = 'closed_book'
  RAG = 'rag'
  RETRIEVAL_ONLY = 'retrieval_only'
  AGENTIC_RETRIEVAL = 'agentic_retrieval'


def evaluate_rag(
    examples: str | datasets.DatasetLoader,
    index: str | indexing.DocumentIndex | None = None,
    strategy: StrategyType | _Strategy = StrategyType.RAG,
    *,
    mcp_urls: Sequence[str] = (),
    output: str | None = None,
    evaluate_kwargs: Mapping[str, Any] | None = None,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
) -> results.EvaluationSummary:
  """Evaluates a RAG strategy on a QA dataset.

  Args:
    examples: Path to a `.jsonl` or `.json` examples file, or a
      `datasets.DatasetLoader` (e.g. `datasets.get_dataset_loader(name,
      max_number_of_examples=N)`).
    index: Saved-index directory or index object, set as the `retriever` of
      retrieval strategies (e.g. `'rag'`, `'retrieval_only'`), overriding any
      existing one with a warning. Raises ValueError if passed to another
      strategy, or if missing when needed and no retriever is already set.
    strategy: Ready-made `StrategyType` (or its string name: `'rag'`,
      `'retrieval_only'`, `'closed_book'` or `'agentic_retrieval'`) or
      `QAStrategy` instance. The `'rag'` and `'retrieval_only'` presets retrieve
      up to 20 documents; on a strategy object, set `max_results`, otherwise all
      documents are returned. A missing `backend_config` or `runtime_kwargs` is
      filled on a copy, from `rag_defaults.default_qa_generate_backend_config()`
      and `extra_backend_kwargs`; the caller's strategy is never modified.
    mcp_urls: URLs of MCP servers to add to the `tools` of
      `'agentic_retrieval'`. Raises ValueError if passed to another strategy, or
      if missing when needed and no tools are already set. Passing both `index`
      and `mcp_urls` always raises ValueError.
    output: Parent folder of the results. Each run is saved in a new folder
      `<output>/<dataset>/<name>__hash_<h>_run<i>/` with `results.html` and
      `eval_config.json` (no credentials). If None, nothing is saved.
    evaluate_kwargs: Keyword arguments forwarded to `ot.evaluate` (e.g.
      `metric_functions`, `chunk_size`). When absent, `'output_results'`
      defaults to True and `'metric_functions'` to
      `rag_metrics.get_default_rag_metrics(...)` with `k = 5, 10, 20` and the
      default AI rater.
    extra_backend_kwargs: Extra constructor kwargs for the created backends
      (index, QA, AI rater), e.g. API keys (usually read from `GOOGLE_API_KEY`).
      Not a stable API. Defaults to 5 retries and a 300000 ms timeout (see
      `get_default_extra_backend_kwargs`). The special key `'cache_filename'`,
      if set, loads and saves one shared `caching.SimpleFunctionCache`, passed
      to the backends as `'cache'` (`'cache_filename'` itself is not forwarded).

  Returns:
    EvaluationSummary object containing metrics and detailed results.

  Raises:
    ValueError: If `strategy` is an unknown strategy name, `index` or `mcp_urls`
      is passed to a strategy that does not use it, or a needed `index` or
      `mcp_urls` is missing.
  """
  backend_kwargs = get_default_extra_backend_kwargs(extra_backend_kwargs)
  cache = _load_cache(backend_kwargs)
  strategy, preset_name = _resolve_strategy(strategy, backend_kwargs)
  strategy = _with_context(strategy, index, mcp_urls, backend_kwargs)
  loaded_examples = _load_examples(examples)
  evaluate_kwargs = _with_default_evaluate_kwargs(
      evaluate_kwargs, strategy, backend_kwargs
  )

  print('Running evaluation...')
  summary = ot_evaluation.evaluate(
      strategy=strategy,
      examples=loaded_examples,
      **evaluate_kwargs,
  )
  if cache is not None:
    _save_cache(cache)
  print(f'Evaluation Metrics: {summary.metrics}')

  if output is not None:
    # TODO: Pass fewer arguments to save_run_outputs, e.g. one
    # run-settings object.
    evaluate_rag_utils.save_run_outputs(
        summary,
        output=output,
        examples_name=_examples_name(examples),
        index=index,
        mcp_urls=mcp_urls,
        preset_name=preset_name,
        strategy=strategy,
        metric_functions=evaluate_kwargs['metric_functions'],
    )
  return summary


def get_default_extra_backend_kwargs(
    extra_backend_kwargs: Mapping[str, Any] | None,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    http_timeout_ms: int = _DEFAULT_HTTP_TIMEOUT_MS,
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
  kwargs = dict(extra_backend_kwargs) if extra_backend_kwargs else {}
  kwargs.setdefault('max_retries', max_retries)
  http_options = kwargs.get('http_options')
  if 'http_options' not in kwargs:
    kwargs['http_options'] = {'timeout': http_timeout_ms}
  elif http_options is None:
    del kwargs['http_options']
  elif isinstance(http_options, dict) and 'timeout' not in http_options:
    kwargs['http_options'] = {**http_options, 'timeout': http_timeout_ms}
  return kwargs


def _load_cache(
    backend_kwargs: dict[str, Any],
) -> caching.SimpleFunctionCache | None:
  """Returns the cache of `backend_kwargs['cache_filename']`, if any.

  Pops `'cache_filename'` from `backend_kwargs` and, if it was set, loads the
  cache from that file (if it exists) and sets it as `backend_kwargs['cache']`.

  Args:
    backend_kwargs: Extra backend kwargs, modified in place.
  """
  cache_filename = backend_kwargs.pop('cache_filename', None)
  if not cache_filename:
    return None
  print(f'Using cache file: {cache_filename}')
  cache = caching.SimpleFunctionCache(cache_filename=cache_filename)
  if epath.Path(cache_filename).exists():
    print(f'Loading existing cache from {cache_filename}...')
    cache.load(overwrite=True)
  else:
    print('No existing cache found. A new cache will be created.')
  backend_kwargs['cache'] = cache
  return cache


def _save_cache(cache: caching.SimpleFunctionCache) -> None:
  """Saves `cache` to its file."""
  print(f'Saving cache to {cache.cache_filename}...')
  cache.save(overwrite=True)
  print(f'Cache saved successfully to {cache.cache_filename}.')


def _resolve_strategy(
    strategy: StrategyType | _Strategy,
    backend_kwargs: Mapping[str, Any],
) -> tuple[_Strategy, str | None]:
  """Returns the strategy to evaluate and its preset name (None if not one).

  A strategy object without its own backend config or credentials gets the
  defaults, on a copy.

  Args:
    strategy: Ready-made `StrategyType` (or its name) or strategy object.
    backend_kwargs: Extra backend kwargs, used as the strategy's credentials.
  """
  if isinstance(strategy, str):
    strategy_type = StrategyType(strategy)
    return _preset_strategy(strategy_type, backend_kwargs), strategy_type.value
  if isinstance(strategy, backend_registry.BackendMixin):
    strategy = dataclasses.replace(
        strategy,
        backend_config=(
            strategy.backend_config
            or rag_defaults.default_qa_generate_backend_config()
        ),
        runtime_kwargs=strategy.runtime_kwargs or backend_kwargs,
    )
  return strategy, None


def _preset_strategy(
    strategy_type: StrategyType,
    runtime_kwargs: Mapping[str, Any],
) -> _Strategy:
  """Returns a new ready-made strategy of type `strategy_type`."""
  if strategy_type == StrategyType.RETRIEVAL_ONLY:
    # Without a limit, all documents of the index are retrieved.
    return strategies_lib.RetrievalOnlyStrategy(
        max_results=_DEFAULT_MAX_RESULTS
    )
  backend_config = rag_defaults.default_qa_generate_backend_config()
  if strategy_type == StrategyType.RAG:
    return strategies_lib.RAGStrategy(
        max_results=_DEFAULT_MAX_RESULTS,
        backend_config=backend_config,
        runtime_kwargs=runtime_kwargs,
    )
  if strategy_type == StrategyType.CLOSED_BOOK:
    return strategies_lib.ClosedBookStrategy(
        backend_config=backend_config,
        runtime_kwargs=runtime_kwargs,
    )
  if strategy_type == StrategyType.AGENTIC_RETRIEVAL:
    return agentic_retrieval.AgenticRetrievalStrategy(
        backend_config=backend_config,
        runtime_kwargs=runtime_kwargs,
    )
  raise ValueError(f'Unknown strategy type: {strategy_type}')


def _with_context(
    strategy: _Strategy,
    index: str | indexing.DocumentIndex | None,
    mcp_urls: Sequence[str],
    extra_backend_kwargs: Mapping[str, Any],
) -> _Strategy:
  """Returns a copy of `strategy` that searches `index` or `mcp_urls`."""
  name = type(strategy).__name__
  if isinstance(
      strategy,
      (strategies_lib.RAGStrategy, strategies_lib.RetrievalOnlyStrategy),
  ):
    if mcp_urls:
      raise ValueError(f'{name} does not use mcp_urls; pass an index.')
    return _with_retriever(strategy, index, extra_backend_kwargs)
  if isinstance(strategy, agentic_retrieval.AgenticRetrievalStrategy):
    if index is not None:
      raise ValueError(f'{name} does not use index; pass mcp_urls.')
    return _with_mcp_servers(strategy, mcp_urls)
  if index is not None or mcp_urls:
    raise ValueError(f'{name} does not use index or mcp_urls.')
  return strategy


def _with_retriever(
    strategy: strategies_lib.RAGStrategy | strategies_lib.RetrievalOnlyStrategy,
    index: str | indexing.DocumentIndex | None,
    extra_backend_kwargs: Mapping[str, Any],
) -> _Strategy:
  """Sets `index`, a saved-index directory or index object, as the retriever."""
  name = type(strategy).__name__
  if index is None:
    # Fail fast, before loading the examples, when no index is available.
    if strategy.retriever is None:
      raise ValueError(f'{name} needs an index.')
    return strategy
  # A passed index takes priority over a retriever already on the strategy.
  if strategy.retriever is not None:
    logging.warning(
        '%s has a retriever; using the index passed instead.', name
    )
  if isinstance(index, str):
    index = index_building.load_index(
        index, extra_backend_kwargs=extra_backend_kwargs
    )
  # Return a copy so the caller's strategy object is never mutated.
  # Indexes take str queries; evaluate_rag only asks str questions.
  return dataclasses.replace(
      strategy,
      retriever=index,  # pyrefly: ignore[bad-argument-type]
  )


def _with_mcp_servers(
    strategy: agentic_retrieval.AgenticRetrievalStrategy,
    mcp_urls: Sequence[str],
) -> agentic_retrieval.AgenticRetrievalStrategy:
  """Adds an MCP tool server per URL in `mcp_urls` to `strategy`."""
  name = type(strategy).__name__
  if not mcp_urls:
    # Fail fast when neither mcp_urls nor existing tools are provided.
    if not strategy.tools:
      raise ValueError(f'{name} needs mcp_urls.')
    return strategy
  # Imported here so that the MCP client libraries are only needed by callers
  # that evaluate this strategy.
  from onetwo.stdlib.tool_use import mcp_tools  # pylint: disable=g-import-not-at-top
  # Append MCP tools on a copy, keeping existing tools and citation parser.
  return dataclasses.replace(
      strategy,
      tools=[*strategy.tools, *(mcp_tools.server(url) for url in mcp_urls)],
      doc_ids_from_answer=(
          strategy.doc_ids_from_answer
          or agentic_retrieval.extract_cited_doc_ids
      ),
  )


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


def _with_default_evaluate_kwargs(
    evaluate_kwargs: Mapping[str, Any] | None,
    strategy: _Strategy,
    backend_kwargs: dict[str, Any],
) -> dict[str, Any]:
  """Returns `evaluate_kwargs` with the defaults of `evaluate_rag` (a copy)."""
  evaluate_kwargs = dict(evaluate_kwargs) if evaluate_kwargs else {}
  evaluate_kwargs.setdefault('output_results', True)
  if 'metric_functions' not in evaluate_kwargs:
    ai_rater_backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name=rag_defaults.DEFAULT_BACKEND_CLASS_NAME,
        model_name=rag_defaults.DEFAULT_AI_RATER_GENERATE_MODEL_NAME,
    )
    # A retrieval-only strategy gives no answer, so only retrieval is scored.
    retrieval_only = isinstance(strategy, strategies_lib.RetrievalOnlyStrategy)
    evaluate_kwargs['metric_functions'] = rag_metrics.get_default_rag_metrics(
        max_results=None,
        retrieval_only=retrieval_only,
        accuracy_critic_backend_config=ai_rater_backend_config,
        runtime_kwargs=backend_kwargs,
    )
  return evaluate_kwargs
