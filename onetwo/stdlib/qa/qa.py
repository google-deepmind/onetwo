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

"""Question answering strategies and interfaces."""

import abc
import dataclasses
from typing import Any, Generic, Protocol, Sequence, TypeVar

import dataclasses_json
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document

# The type of the input question (e.g., str, or a complex prompt object).
QuestionT = TypeVar('QuestionT')
# The type of the generated answer (e.g., str, or a structured response).
AnswerT = TypeVar('AnswerT')
# The type of documents used as context (e.g., a structured document object).
DocT = TypeVar('DocT')
# The type of the answer carried by a `BaseStrategyResult`.
ResultAnswerT = TypeVar('ResultAnswerT')


class QAStrategy(Protocol[QuestionT, AnswerT]):
  """Generic interface for a strategy that answers a question.

  The primary way to use a QAStrategy instance is to call it directly
  (e.g., `strategy_instance(question)`) which invokes the `__call__` method.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: QuestionT,
  ) -> AnswerT:
    """Returns the answer to the question."""
    pass


class ContextualQAStrategy(
    Protocol[QuestionT, AnswerT, DocT],
):
  """Interface for strategy that answers questions using provided content.

  This interface defines a strategy where the knowledge source is explicitly
  provided at call time. It decouples the answering logic from the discovery of
  information, making it suitable for tasks where the context is determined by a
  prior retrieval process.
  """

  @abc.abstractmethod
  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: QuestionT,
      docs: Sequence[DocT],
  ) -> AnswerT:
    """Returns the answer to the question given documents as context.

    Args:
      question: The query or prompt to be answered.
      docs: A sequence of documents or context objects (DocT) to be used as the
        grounded source of truth for the answer.
    """
    pass


@dataclasses.dataclass
class SimpleQAStrategy(QAStrategy[str, str]):
  """A basic QAStrategy that maps a string question to a string answer.

  This is intended for simple prompting strategies where no external
  context is required or where context is already baked into the prompt.
  """

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: str,
  ) -> str:
    """Returns a plain string answer."""
    raise NotImplementedError('Subclasses must implement the generation logic.')


@dataclasses.dataclass
class SimpleContextualQAStrategy(ContextualQAStrategy[str, str, _Document]):
  """A ContextualQAStrategy that maps a string question to a string answer.

  This is intended for simple prompting strategies where the context is provided
  as a list of retrieval_data_structures.Document objects.
  """

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: str,
      docs: Sequence[_Document],
  ) -> str:
    """Returns a plain string answer."""
    raise NotImplementedError('Subclasses must implement the generation logic.')


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class QAExample:
  """An example (or few-shot exemplar) for use with a QAStrategy.

  Attributes:
    example_id: Optional unique identifier for the example within the dataset
      (e.g., query_id, example_id from source benchmarks).
    question: The question to ask.
    answer: The golden answer to the question.
    metadata: Optional metadata about the example (or other custom content).
      Note that if you plan on serializing/deserializing instances of this
      class, the values in the `metadata` dict must be types supported by the
      serialization library (like `dataclass_json`). This typically includes
      basic types such as str, int, float, bool, list, and dict. Other complex
      objects may not (de)serialize correctly.
  """

  example_id: str = ''
  question: str | content_lib.ChunkList = ''
  answer: Any = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ContextualQAExample(QAExample):
  """An example (or few-shot exemplar) for use with a ContextualQAStrategy.

  Attributes:
    docs: Documents (or passages) to use as context for the question.
  """

  docs: list[_Document] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class BaseStrategyResult(Generic[ResultAnswerT]):
  """The result returned by a QA strategy, generic over the answer type.

  Strategies with a free-text answer return `StrategyResult`. Strategies that
  produce a structured answer (e.g. via `llm.generate_object`) can return
  `BaseStrategyResult[MyAnswerClass]` directly, or a subclass of it.

  Attributes:
    answer: The generated answer, e.g. a string or a structured object.
    docs: The documents used as context. For example, this list will be empty
      for closed-book strategies, while for long-context strategies this will be
      the list of all documents in the dataset. For RAG strategies this will be
      the list of documents retrieved by the retriever.
  """

  answer: ResultAnswerT
  docs: list[_Document] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class StrategyResult(BaseStrategyResult[str]):
  """The result returned by QA strategies that produce a free-text answer.

  Attributes:
    answer: The generated answer string.
  """

  answer: str = ''
