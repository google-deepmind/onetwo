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
from typing import Protocol, Sequence, TypeVar

from onetwo.core import executing
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document

# The type of the input question (e.g., str, or a complex prompt object).
QuestionT = TypeVar('QuestionT')
# The type of the generated answer (e.g., str, or a structured response).
AnswerT = TypeVar('AnswerT')
# The type of documents used as context (e.g., a structured document object).
DocT = TypeVar('DocT')


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
