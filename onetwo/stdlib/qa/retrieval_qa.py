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

"""Question answering strategies that actively use a retriever."""

import abc
from typing import Protocol

from onetwo.core import executing
from onetwo.stdlib.qa import qa
from onetwo.stdlib.retrieval import retrieval
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document
QuestionT = qa.QuestionT
AnswerT = qa.AnswerT
RetrievalResultT = retrieval.RetrievalResultT


class RetrievalQAStrategy(Protocol[QuestionT, AnswerT, RetrievalResultT]):
  """Generic interface for QA strategies that actively use a retriever."""

  @abc.abstractmethod
  @executing.make_executable(copy_self=False, non_copied_args=['retriever'])
  async def __call__(
      self,
      question: QuestionT,
      retriever: retrieval.Retriever[QuestionT, RetrievalResultT],
  ) -> AnswerT:
    """Returns an answer by utilizing the provided retriever."""
    pass


class SimpleRetrievalQAStrategy(RetrievalQAStrategy[str, str, _Document]):
  """A basic RetrievalQAStrategy that maps a question to an answer."""

  @executing.make_executable(copy_self=False, non_copied_args=['retriever'])
  async def __call__(
      self,
      question: str,
      retriever: retrieval.Retriever[str, _Document],
  ) -> str:
    """Returns an answer by utilizing the provided retriever."""
    raise NotImplementedError('Subclasses must implement the generation logic.')
