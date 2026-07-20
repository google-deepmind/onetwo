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

"""Strategies registry with basic QA strategies.

This module provides a registry of QA strategies and concrete implementations
for common evaluation paradigms: closed-book, long-context, and RAG.

Design principles:
  1. **Decoupled index building from QA**: All strategies implement
     `QAStrategy`. Every dataset can be turned into a dataset with
     `QAExample`s, so index building is fully decoupled from QA evaluation.
  2. **Decoupled strategy definition from retrieval kwargs**: Parameters like
     `max_results` or `max_tokens` are optional kwargs of the strategy, which
     takes care of piping them to the retriever.
  3. **Strategies return answers and retrieved content**: Every strategy returns
     a `StrategyResult` containing the answer and the retrieved `doc_ids`,
     enabling easy downstream computation of recall@k and accuracy.
  4. **Multimodal**: All strategies are multimodal-compatible. Questions and
     documents can be `str` or `content_lib.ChunkList`.

Usage:
  ```python
  strategy = get_strategy('rag', retriever=my_retriever)
  result = ot.run(strategy(question='What is OneTwo?', max_results=2))
  print(result.answer, result.doc_ids)
  ```
"""

import dataclasses
import functools
from typing import Callable, Sequence

from onetwo.builtins import composables
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import qa_prompt_builder
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document
_Message = content_lib.Message
_PredefinedRole = content_lib.PredefinedRole

_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult

# =============================================================================
# Default QA templates
# =============================================================================
# These templates adopt the two most common prompting styles:
# 1. The model is allowed to answer freely based on the provided context and
#    its internal knowledge. This is the default strategy.
# 2. The model is only allowed to answer based on the provided context.
# The second prompting style might be useful for reducing the hallucinations of
# the model, or to perform ablation studies assessing how much information
# retrieved content helps the model answer the question. For example, these two
# prompts are analogous to the "Loose" and "Strict" prompts in Figure 6 of
# https://arxiv.org/pdf/2404.10198.

default_qa_template = lambda context, question: (
    composables.c(
        # pyrefly: ignore[bad-argument-type]
        'Your goal is to answer a query the best way you can. You will'
        ' be given a query and possibly some evidence that could help'
        ' you craft the answer.\n\nQUERY:\n'
    )
    + question
    + composables.c('\n\nCONTEXT:\n')  # pyrefly: ignore[bad-argument-type]
    + context
    + composables.c('\n\nPlease answer the query.')  # pyrefly: ignore[bad-argument-type]
)

default_qa_template_only_from_context = lambda context, question: (
    composables.c(
        # pyrefly: ignore[bad-argument-type]
        'Your goal is to answer a query the best way you can. You will'
        ' be given a query and possibly some evidence that could help'
        ' you craft the answer. If the context does not contain the'
        ' information needed to answer the query, reply "I don\'t have'
        ' enough context to answer the query." Do not make up an answer.\n'
        '\nQUERY:\n'
    )
    + question
    + composables.c('\n\nCONTEXT:\n')  # pyrefly: ignore[bad-argument-type]
    + context
    + composables.c('\n\nPlease answer the query.')  # pyrefly: ignore[bad-argument-type]
)

# =============================================================================
# Strategy implementations
# =============================================================================


@dataclasses.dataclass
class ClosedBookStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """A strategy that answers questions without any context (closed-book).

  The context field in the prompt is set to "None".

  Attributes:
    prompt_builder: The prompt builder used to construct the prompt.
  """

  prompt_builder: qa_prompt_builder.QADocumentPromptBuilder = dataclasses.field(
      default_factory=functools.partial(
          qa_prompt_builder.QADocumentPromptBuilder,
          template=default_qa_template,
      )
  )

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns an answer with no context."""
    dummy_doc = _Document(doc_id='none', content='None')
    context = await self.prompt_builder([dummy_doc], question)
    answer = await llm.chat(context.to_messages())
    return _StrategyResult(answer=str(answer), doc_ids=[])


@dataclasses.dataclass
class LongContextStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """A strategy that puts all documents in context (long-context).

  Attributes:
    documents: The full corpus of documents to use as context.
    prompt_builder: The prompt builder used to construct the prompt.
  """

  documents: Sequence[_Document] = ()
  prompt_builder: qa_prompt_builder.QADocumentPromptBuilder = dataclasses.field(
      default_factory=functools.partial(
          qa_prompt_builder.QADocumentPromptBuilder,
          template=default_qa_template,
      )
  )

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns an answer using all documents as context."""
    docs = (
        self.documents
        if self.documents
        else [_Document(doc_id='none', content='None')]
    )
    context = await self.prompt_builder(docs, question)
    answer = await llm.chat(context.to_messages())
    doc_ids = [doc.doc_id for doc in self.documents]
    return _StrategyResult(answer=str(answer), doc_ids=doc_ids)


@dataclasses.dataclass
class RAGStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """A strategy that retrieves relevant documents before answering (RAG).

  Attributes:
    retriever: The retriever to use for fetching relevant documents.
    prompt_builder: The prompt builder used to construct the prompt.
  """

  retriever: retrieval.Retriever[_PromptContent, _Document] | None = None
  prompt_builder: qa_prompt_builder.QADocumentPromptBuilder = dataclasses.field(
      default_factory=functools.partial(
          qa_prompt_builder.QADocumentPromptBuilder,
          template=default_qa_template,
      )
  )

  @executing.make_executable(copy_self=False, non_copied_args=['retriever'])
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns an answer using retrieved documents as context."""
    if self.retriever is None:
      raise ValueError(
          'RAGStrategy requires a retriever. Pass one at construction time.'
      )

    # Retrieve relevant documents.
    retrieved_docs = await self.retriever.retrieve(
        query=question,
        **query_time_kwargs,
    )
    retrieved_docs = list(retrieved_docs)

    docs = (
        retrieved_docs
        if retrieved_docs
        else [_Document(doc_id='none', content='None')]
    )
    context = await self.prompt_builder(docs, question)
    answer = await llm.chat(context.to_messages())
    doc_ids = [doc.doc_id for doc in retrieved_docs]
    return _StrategyResult(answer=str(answer), doc_ids=doc_ids)


@dataclasses.dataclass
class RetrievalOnlyStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """A strategy that only retrieves relevant documents without answering.

  Attributes:
    retriever: The retriever to use for fetching relevant documents.
  """

  retriever: retrieval.Retriever[_PromptContent, _Document] | None = None

  @executing.make_executable(copy_self=False, non_copied_args=['retriever'])
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns empty answer and retrieved document IDs."""
    if self.retriever is None:
      raise ValueError(
          'RetrievalOnlyStrategy requires a retriever. Pass one at construction'
          ' time.'
      )

    # Retrieve relevant documents.
    retrieved_docs = await self.retriever.retrieve(
        query=question,
        **query_time_kwargs,
    )
    retrieved_docs = list(retrieved_docs)
    doc_ids = [doc.doc_id for doc in retrieved_docs]
    return _StrategyResult(answer='', doc_ids=doc_ids)


# =============================================================================
# Strategy registry
# =============================================================================

# Type alias for strategy factory functions.
StrategyFactory = Callable[..., qa.QAStrategy]

_STRATEGY_REGISTRY: dict[str, StrategyFactory] = {}


def register_strategy(name: str, factory: StrategyFactory) -> None:
  """Registers a strategy factory under the given name.

  Args:
    name: The unique name for the strategy (e.g. 'closed_book', 'rag').
    factory: A callable that returns a QAStrategy instance.

  Raises:
    ValueError: If a strategy with the given name is already registered.
  """
  if name in _STRATEGY_REGISTRY:
    raise ValueError(
        f"Strategy '{name}' is already registered. Use a different name or"
        ' unregister the existing one first.'
    )
  _STRATEGY_REGISTRY[name] = factory


def unregister_strategy(name: str) -> None:
  """Removes a strategy from the registry.

  Args:
    name: The name of the strategy to remove.

  Raises:
    KeyError: If no strategy with the given name is registered.
  """
  if name not in _STRATEGY_REGISTRY:
    raise KeyError(f"Strategy '{name}' is not registered.")
  del _STRATEGY_REGISTRY[name]


def get_strategy(name: str, **kwargs) -> qa.QAStrategy:
  """Creates and returns a strategy instance by name.

  Args:
    name: The registered name of the strategy.
    **kwargs: Keyword arguments forwarded to the strategy factory.

  Returns:
    A QAStrategy instance.

  Raises:
    KeyError: If no strategy with the given name is registered.
  """
  if name not in _STRATEGY_REGISTRY:
    available = list(_STRATEGY_REGISTRY.keys())
    raise KeyError(
        f"Strategy '{name}' is not registered. Available strategies:"
        f' {available}'
    )
  return _STRATEGY_REGISTRY[name](**kwargs)


def list_strategies() -> list[str]:
  """Returns a sorted list of all registered strategy names."""
  return sorted(_STRATEGY_REGISTRY.keys())


# =============================================================================
# Register built-in strategies
# =============================================================================

register_strategy('closed_book', ClosedBookStrategy)
register_strategy('long_context', LongContextStrategy)
register_strategy('rag', RAGStrategy)
register_strategy('retrieval_only', RetrievalOnlyStrategy)
