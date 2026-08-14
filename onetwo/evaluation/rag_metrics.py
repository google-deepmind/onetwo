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

from collections.abc import Callable
import re
from typing import Any

from onetwo.backends import backend_registry
from onetwo.core import executing
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
        '\'metadata\': {\'golden_doc_ids\': ...} to be present in the example. '
        'If your dataset uses a different key or structure, please provide a '
        'custom golden_doc_ids_extractor to make_recall_at_k.'
    )
  gold_ids = example['metadata']['golden_doc_ids']
  if isinstance(gold_ids, str):
    return [gold_ids]
  return [str(gid) for gid in gold_ids]


def make_recall_at_k(
    k: int | None = None,
    answer_doc_ids_extractor: Callable[[_ExampleDict], list[str]] | None = None,
    strip_chunk_suffix: bool = True,
) -> Callable[..., tuple[float, dict[str, Any]]]:
  """Returns a recall@k metric function for RAG evaluation.

  Usage example:
    ```python
    recall_at_2 = make_recall_at_k(k=2)

    prediction = StrategyResult(
        answer='Generated answer',
        doc_ids=['doc_2', 'doc_1', 'doc_3'],
    )
    example = {
        'metadata': {
            'answer_doc_ids': ['doc_1', 'doc_4'],
        }
    }

    # target is not used by recall, so we can pass an empty string.
    score, info = recall_at_2(
        target='',
        prediction=prediction,
        example=example,
    )
    # score: 0.5 (only 'doc_1' of the gold docs is in the top 2 retrieved docs)
    # info: {'answer_doc_ids': ['doc_1', 'doc_4'], 'retrieved_doc_ids':
    #         ['doc_2', 'doc_1']}
    ```

  Args:
    k: The number of retrieved documents to consider. If None, all retrieved
      documents are considered.
    answer_doc_ids_extractor: A function that takes the dataset example and
      returns a list of golden document IDs (as strings). If None, defaults to
      using the 'answer_doc_ids' field in the example's 'metadata' field.
    strip_chunk_suffix: If True, strips the '_chunkNNN' suffix from both
      retrieved and golden document IDs before computing recall.

  Returns:
    A metric function (conforming to `evaluation.MetricFunction`) that takes the
    target, prediction, and example, and returns a tuple of recall@k score and
    extra_info_dict (containing answer_doc_ids and retrieved_doc_ids).
  """
  answer_doc_ids_extractor = (
      answer_doc_ids_extractor or _default_golden_doc_ids_extractor
  )

  def recall_at_k(
      target: str,
      prediction: StrategyResult,
      example: _ExampleDict | None = None,
  ) -> tuple[float, dict[str, Any]]:
    """Computes recall@k of retrieved doc_ids against golden doc titles."""
    del target  # Unused.
    if not example:
      raise ValueError('Example is required to retrieve golden_doc_ids.')

    gold_ids = answer_doc_ids_extractor(example)
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
        'answer_doc_ids': gold_ids,
        'retrieved_doc_ids': original_retrieved_ids,
    }
    return len(intersection) / len(gold_ids), extra_evaluation_info

  return recall_at_k


def make_accuracy_critic(
    backend_config: backend_registry.GenerateBackendConfig | None = None,
) -> Callable[..., executing.Executable[tuple[float, dict[str, Any]]]]:
  """Returns an AI-Rater critic that evaluates the generated answer.

  Args:
    backend_config: Optional backend configuration to use for the evaluation.
      If None, the default backend is used.
  """
  backend = None
  if backend_config is not None:
    backend = backend_registry.get_backend(backend_config)

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
