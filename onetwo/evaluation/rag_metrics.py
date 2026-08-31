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

"""Metrics library for RAG (Retrieval-Augmented Generation) Evaluation.

All metrics defined in this library conform to the `MetricFunction` protocol
defined in `evaluation.py` and expected by `evaluation.evaluate`.
"""

from collections.abc import Callable, Sequence
import math
import re
from typing import Any, Final

from onetwo.backends import backend_registry
from onetwo.core import executing
from onetwo.core import results
from onetwo.core import routing
from onetwo.evaluation import evaluation
from onetwo.stdlib.qa import qa

StrategyResult = qa.StrategyResult
_ExampleDict = evaluation.ExampleDict

_CHUNK_SUFFIX_RE = re.compile(r'(_chunk\d+)+$')


def _strip_chunk_suffix(doc_id: str) -> str:
  """Strips the '_chunkNNN' suffix from the document ID if present."""
  return _CHUNK_SUFFIX_RE.sub('', doc_id)


# TODO: Consider adding a metric that captures AUC-equivalent
# score for retrieval.


def _default_golden_doc_ids_extractor(example: _ExampleDict) -> list[str]:
  """Default extractor that retrieves golden doc IDs from the example.

  Expects the golden document IDs to be stored in the
  example['metadata']['golden_doc_ids'] field.

  Args:
    example: A mapping representing the dataset example.

  Returns:
    A list of golden document IDs as strings.

  Raises:
    KeyError: If the 'metadata' or 'golden_doc_ids' fields are not present in
      the example.
  """
  if 'metadata' not in example or 'golden_doc_ids' not in example['metadata']:
    raise KeyError(
        'The default golden_doc_ids_extractor expects '
        "'metadata': {'golden_doc_ids': ...} to be present in the example. "
        'If your dataset uses a different key or structure, please provide a '
        'custom golden_doc_ids_extractor to make_recall_at_k.'
    )
  gold_ids = example['metadata']['golden_doc_ids']
  if isinstance(gold_ids, str):
    return [gold_ids]
  return [str(gid) for gid in gold_ids]


def make_recall_at_k(
    k: int | None = None,
    golden_doc_ids_extractor: Callable[[_ExampleDict], list[str]] | None = None,
    strip_chunk_suffix: bool = True,
) -> Callable[..., tuple[float, dict[str, Any]]]:
  """Returns a recall@k metric function for RAG evaluation.

  Usage example:
    ```python
    recall_at_2 = make_recall_at_k(k=2)

    prediction = StrategyResult(
        answer='Generated answer',
        docs=['doc_2', 'doc_1', 'doc_3'],
    )
    example = {
        'metadata': {
            'golden_doc_ids': ['doc_1', 'doc_4'],
        }
    }

    # target is not used by recall, so we can pass an empty string.
    score, info = recall_at_2(
        target='',
        prediction=prediction,
        example=example,
    )
    # score: 0.5 (only 'doc_1' of the gold docs is in the top 2 retrieved docs)
    # info: {'golden_doc_ids': ['doc_1', 'doc_4'], 'retrieved_doc_ids':
    #         ['doc_2', 'doc_1']}
    ```

  Args:
    k: The number of retrieved documents to consider. If None, all retrieved
      documents are considered.
    golden_doc_ids_extractor: A function that takes the dataset example and
      returns a list of golden document IDs (as strings). If None, defaults to
      using the 'golden_doc_ids' field in the example's 'metadata' field.
    strip_chunk_suffix: If True, strips the '_chunkNNN' suffix from both
      retrieved and golden document IDs before computing recall.

  Returns:
    A metric function (conforming to `evaluation.MetricFunction`) that takes the
    target, prediction, and example, and returns a tuple of recall@k score and
    extra_info_dict (containing golden_doc_ids and retrieved_doc_ids).
  """
  golden_doc_ids_extractor = (
      golden_doc_ids_extractor or _default_golden_doc_ids_extractor
  )

  def recall_at_k(
      target: str,
      prediction: StrategyResult,
      example: _ExampleDict | None = None,
  ) -> tuple[float, dict[str, Any]]:
    """Computes recall@k of retrieved doc_ids against golden doc titles."""
    del target  # Unused.
    if prediction is None:
      return 0.0, {}
    if not example:
      raise ValueError('Example is required to retrieve golden_doc_ids.')

    gold_ids = golden_doc_ids_extractor(example)
    gold_ids = [str(gid) for gid in gold_ids]
    if not gold_ids:
      # If there are no gold docs defined, recall is trivially 1.0.
      return 1.0, {}

    retrieved_ids = [doc.doc_id for doc in prediction.docs]
    if k is not None:
      retrieved_ids = retrieved_ids[:k]
    retrieved_ids = [str(rid) for rid in retrieved_ids]

    original_retrieved_ids = list(retrieved_ids)
    if strip_chunk_suffix:
      retrieved_ids = [_strip_chunk_suffix(rid) for rid in retrieved_ids]

    intersection = set(gold_ids).intersection(set(retrieved_ids))
    extra_evaluation_info = {
        'golden_doc_ids': gold_ids,
        'retrieved_doc_ids': original_retrieved_ids,
    }
    return len(intersection) / len(gold_ids), extra_evaluation_info

  return recall_at_k


def make_accuracy_critic(
    backend_config: backend_registry.GenerateBackendConfig | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
) -> Callable[..., executing.Executable[tuple[float, dict[str, Any]]]]:
  """Returns an AI-Rater critic that evaluates the generated answer.

  Args:
    backend_config: Optional backend configuration to use for the evaluation. If
      None, the default backend is used.
    runtime_kwargs: Optional runtime arguments specifically needed to
      instantiate the backend (e.g. API key).
  """
  backend = None
  if backend_config is not None:
    backend = backend_registry.get_backend(
        backend_config, **(runtime_kwargs or {})
    )

  @executing.make_executable
  async def accuracy_critic_fn(
      target: str,
      prediction: StrategyResult,
      example: _ExampleDict | None = None,
  ) -> tuple[float, dict[str, Any]]:
    """AI-Rater critic that evaluates the generated answer against the target.

    Conforms to the `evaluation.MetricFunction` protocol.

    Uses OneTwo's structured_evaluation_critic.

    Args:
      target: The expected golden answer.
      prediction: The RAG strategy result containing the generated answer.
      example: The dataset example mapping.

    Returns:
      A tuple of (accuracy_score, extra_info_dict).
    """
    if prediction is None:
      return 0.0, {}
    with routing.RegistryContext():
      if backend is not None:
        backend.register()
      return await evaluation.structured_evaluation_critic(  # pyrefly: ignore[not-async]
          target=target,  # pyrefly: ignore[bad-argument-count, unexpected-keyword]
          prediction=prediction.answer,  # pyrefly: ignore[unexpected-keyword]
          example=example,  # pyrefly: ignore[unexpected-keyword]
      )

  return accuracy_critic_fn


accuracy_critic = make_accuracy_critic()


def _default_golden_doc_scores_extractor(
    example: _ExampleDict,
) -> dict[str, int | float]:
  """Default extractor that retrieves golden docs and their relevance scores.

  Expects the golden document scores to be stored in the
  example['metadata']['golden_doc_scores'] field.

  Args:
    example: A mapping representing the dataset example.

  Returns:
    A mapping of document ID to relevance score.

  Raises:
    KeyError: If the 'metadata' or 'golden_doc_scores' fields are not present in
      the example.
  """
  if (
      'metadata' not in example
      or 'golden_doc_scores' not in example['metadata']
  ):
    raise KeyError(
        'The default golden_doc_scores_extractor expects '
        "'metadata': {'golden_doc_scores': ...} to be present in the example. "
        'If your dataset uses a different key or structure, please provide a '
        'custom golden_doc_scores_extractor to make_ndcg_at_k.'
    )
  scores = example['metadata']['golden_doc_scores']
  return {str(k): v for k, v in scores.items()}


def make_ndcg_at_k(
    k: int | None = None,
    golden_doc_scores_extractor: (
        Callable[[_ExampleDict], dict[str, int | float]] | None
    ) = None,
    strip_chunk_suffix: bool = True,
) -> Callable[..., tuple[float, dict[str, Any]]]:
  """Returns an ndcg@k metric function for RAG evaluation.

  NDCG@k (Normalized Discounted Cumulative Gain at k) measures retrieval
  quality for datasets with graded relevance scores. When the retrieved items
  are chunks instead of documents, the retrieved chunks are mapped to parent doc
  IDs and deduplicated (preserving first-occurrence rank), ensuring each
  document is scored at most once.

  NOTE: This metric is useful when `golden_doc_scores` is available in the
  dataset metadata (e.g., NFCorpus). For datasets with `golden_doc_ids`
  (e.g., HotpotQA), use `make_recall_at_k` instead.

  Usage example:
    ```python
    ndcg_at_2 = make_ndcg_at_k(k=2)

    prediction = StrategyResult(
        answer='Generated answer',
        docs=['doc_2', 'doc_1', 'doc_3'],
    )
    example = {
        'metadata': {
            'golden_doc_scores': {'doc_1': 2, 'doc_4': 1},
        }
    }

    # target is not used by ndcg, so we can pass an empty string.
    score, info = ndcg_at_2(
        target='',
        prediction=prediction,
        example=example,
    )
    # score: ~0.52 (the most relevant doc was included, hence score > 0.5)
    # info: {'golden_doc_scores': {'doc_1': 2, 'doc_4': 1},
    #        'retrieved_doc_ids': ['doc_2', 'doc_1']}
    ```

  Args:
    k: The number of retrieved items (documents or chunks) to consider. If None,
      all retrieved items are considered.
    golden_doc_scores_extractor: A function that takes the dataset example and
      returns a dictionary mapping golden document IDs (as strings) to their
      graded relevance scores (int/float). If None, defaults to using the
      'golden_doc_scores' field in the example's 'metadata' field.
    strip_chunk_suffix: If True, strips the '_chunkNNN' suffix from retrieved
      document IDs and deduplicates identical document IDs before computing
      relevance scores.

  Returns:
    A metric function (conforming to `evaluation.MetricFunction`) that takes the
    target, prediction, and example, and returns a tuple of ndcg@k score and
    extra_info_dict.
  """
  golden_doc_scores_extractor = (
      golden_doc_scores_extractor or _default_golden_doc_scores_extractor
  )

  def ndcg_at_k(
      target: str,
      prediction: StrategyResult,
      example: _ExampleDict | None = None,
  ) -> tuple[float, dict[str, Any]]:
    """Computes NDCG@k of retrieved doc_ids against golden relevance scores."""
    del target  # Unused.
    if prediction is None:
      return 0.0, {}
    if not example:
      raise ValueError('Example is required to retrieve golden_doc_scores.')

    qrel_scores = golden_doc_scores_extractor(example)
    if not qrel_scores:
      # If there are no golden docs, ndcg is trivially 1.0.
      return 1.0, {}

    retrieved_ids = [doc.doc_id for doc in prediction.docs]
    if k is not None:
      retrieved_ids = retrieved_ids[:k]
    retrieved_ids = [str(rid) for rid in retrieved_ids]

    original_retrieved_ids = list(retrieved_ids)
    if strip_chunk_suffix:
      # Deduplicate document IDs while preserving the first occurrence order.
      # This ensures that each document contributes relevance at most once,
      # guaranteeing that NDCG <= 1.0 even when multiple chunks of the same
      # document are retrieved within the top-k results.
      seen_doc_ids = set()
      new_retrieved_ids = []
      for rid in retrieved_ids:
        stripped_id = _strip_chunk_suffix(rid)
        if stripped_id not in seen_doc_ids:
          new_retrieved_ids.append(stripped_id)
          seen_doc_ids.add(stripped_id)
      retrieved_ids = new_retrieved_ids

    # DCG calculation
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
      rel = qrel_scores.get(doc_id, 0)
      dcg += (2**rel - 1) / math.log2(i + 2)

    # IDCG calculation: sort all relevance scores descending, take top k
    ideal_rels = sorted(qrel_scores.values(), reverse=True)
    if k is not None:
      ideal_rels = ideal_rels[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
      idcg += (2**rel - 1) / math.log2(i + 2)

    extra_evaluation_info = {
        'golden_doc_scores': qrel_scores,
        'retrieved_doc_ids': original_retrieved_ids,
        'dcg': dcg,
        'idcg': idcg,
    }

    if idcg == 0:
      return 0.0, extra_evaluation_info

    return dcg / idcg, extra_evaluation_info

  return ndcg_at_k


# Stage name suffixes that identify LLM call stages in the execution trace.
_LLM_STAGE_SUFFIXES = ('.chat', '.generate_text', '.instruct')


def _walk_llm_stages(
    execution_result: results.ExecutionResult,
) -> list[results.ExecutionResult]:
  """Collects all LLM call stages from the execution trace tree.

  Recursively traverses the ``ExecutionResult`` tree and returns stages whose
  ``stage_name`` ends with a known LLM call suffix (``.chat``,
  ``.generate_text``, ``.instruct``). These stages contain the actual
  request/response in their inputs/outputs and token/tool counts in their info.

  Args:
    execution_result: Root of the execution trace tree.

  Returns:
    A flat list of ``ExecutionResult`` nodes corresponding to LLM calls.
  """
  llm_stages: list[results.ExecutionResult] = []

  def _traverse(node: results.ExecutionResult) -> None:
    if node.stage_name and node.stage_name.endswith(_LLM_STAGE_SUFFIXES):
      llm_stages.append(node)
      return
    for stage in node.stages:
      _traverse(stage)

  _traverse(execution_result)
  return llm_stages


def _find_info_value(stage: results.ExecutionResult, key: str) -> Any | None:
  """Recursively searches for a key in the stage's info or its sub-stages."""
  if key in stage.info:
    return stage.info[key]
  for substage in stage.stages:
    val = _find_info_value(substage, key)
    if val is not None:
      return val
  return None


def _get_counts_from_info(
    execution_result: results.ExecutionResult,
    *,
    info_keys: Sequence[str],
) -> list[int]:
  """Returns per-call counts from the execution trace info.

  Args:
    execution_result: Root of the execution trace tree.
    info_keys: The trace info keys to sum (e.g. ['llm.tokens.prompt']).
  """
  stages = _walk_llm_stages(execution_result)
  if not stages:
    return []

  counts = []
  for stage in stages:
    stage_sum = 0
    for key in info_keys:
      val = _find_info_value(stage, key)
      if val is not None:
        stage_sum += int(val)
    counts.append(stage_sum)

  return counts


def _get_input_token_counts(
    execution_result: results.ExecutionResult,
) -> list[int]:
  """Returns per-call input token counts from the execution trace."""
  # Note: 'llm.tokens.cached_content' is already included in 'llm.tokens.prompt'
  # according to the GenAI SDK specification.
  return _get_counts_from_info(
      execution_result,
      info_keys=['llm.tokens.prompt', 'llm.tokens.tool_use_prompt'],
  )


def _get_output_token_counts(
    execution_result: results.ExecutionResult,
) -> list[int]:
  """Returns per-call output token counts from the execution trace."""
  return _get_counts_from_info(
      execution_result,
      info_keys=['llm.tokens.candidates', 'llm.tokens.thoughts'],
  )


def _get_tool_call_counts(
    execution_result: results.ExecutionResult,
) -> list[int]:
  """Returns per-call tool call counts from the execution trace."""
  return _get_counts_from_info(
      execution_result,
      info_keys=['llm.tools.calls'],
  )


# --- LLM call metrics ---


def num_llm_calls(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Counts the number of LLM calls in the execution trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  return float(len(_walk_llm_stages(execution_result)))


# --- Input token metrics ---


def total_input_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Total input tokens across all LLM call stages in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_input_token_counts(execution_result)
  return float(sum(counts)) if counts else 0.0


def max_input_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Maximum input tokens across any single LLM call in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_input_token_counts(execution_result)
  return float(max(counts)) if counts else 0.0


def average_input_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Average input tokens per LLM call in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_input_token_counts(execution_result)
  return float(sum(counts)) / len(counts) if counts else 0.0


# --- Output token metrics ---


def total_output_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Total output tokens across all LLM call stages in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_output_token_counts(execution_result)
  return float(sum(counts)) if counts else 0.0


def max_output_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Maximum output tokens across any single LLM call in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_output_token_counts(execution_result)
  return float(max(counts)) if counts else 0.0


def average_output_tokens(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Average output tokens per LLM call in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_output_token_counts(execution_result)
  return float(sum(counts)) / len(counts) if counts else 0.0


# --- Tool call metrics ---


def num_tool_calls(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> float:
  """Total number of tool calls across all LLM stages in the trace."""
  del target, prediction, example
  if execution_result is None:
    return 0.0
  counts = _get_tool_call_counts(execution_result)
  return float(sum(counts)) if counts else 0.0


def tool_details(
    target: Any,
    prediction: Any,
    example: Any = None,
    execution_result: results.ExecutionResult | None = None,
) -> tuple[float, dict[str, Any]]:
  """Extracts llm.tools.details from the trace as extra info.

  Args:
    target: Golden output (unused).
    prediction: Strategy prediction (unused).
    example: Example dict (unused).
    execution_result: Root of the execution trace tree.

  Returns:
    A tuple of (number of tool details, dict containing the raw details list).
  """
  del target, prediction, example
  if execution_result is None:
    return 0.0, {}

  stages = _walk_llm_stages(execution_result)
  details = []
  for stage in stages:
    val = _find_info_value(stage, 'llm.tools.details')
    if val is not None:
      details.extend(val)

  return float(len(details)), {'tool_details': details}


def get_default_rag_metrics(
    accuracy_critic_backend_config: (
        backend_registry.GenerateBackendConfig | None
    ) = None,
    runtime_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Returns a dictionary of default metric functions for RAG evaluation.

  Includes retrieval metrics (recall@5, recall@20), generation quality
  (accuracy critic), and execution trace metrics (token counts, LLM calls,
  tool calls, and tool details).

  Args:
    accuracy_critic_backend_config: Optional backend config for the accuracy
      critic. If None, uses the default backend config.
    runtime_kwargs: Optional runtime arguments specifically needed to
      instantiate the backend (e.g. API key).

  Returns:
    A dictionary mapping metric names to metric functions.
  """
  return {
      'accuracy': make_accuracy_critic(
          backend_config=accuracy_critic_backend_config,
          runtime_kwargs=runtime_kwargs,
      ),
      'recall@5': make_recall_at_k(k=5),
      'recall@20': make_recall_at_k(k=20),
      'num_llm_calls': num_llm_calls,
      'total_input_tokens': total_input_tokens,
      'max_input_tokens': max_input_tokens,
      'average_input_tokens': average_input_tokens,
      'total_output_tokens': total_output_tokens,
      'max_output_tokens': max_output_tokens,
      'average_output_tokens': average_output_tokens,
      'num_tool_calls': num_tool_calls,
      'tool_details': tool_details,
  }


DEFAULT_RAG_METRICS: Final[dict[str, Any]] = get_default_rag_metrics()
