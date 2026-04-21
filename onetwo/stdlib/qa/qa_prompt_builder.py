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

"""Utilities for building prompts for question answering (QA) tasks.

This module provides `QAPromptBuilder`, a configurable class for compiling a
list of documents and a query into a final prompt suitable for QA, typically
in a Retrieval-Augmented Generation (RAG) pipeline.

It leverages OneTwo's `composing` and `composables` libraries to natively
support multimodal content (such as text interleaved with images) and to produce
`Composable` objects that can be executed or chained with other operations.

How it works:
1.  **Document Transformation**: Each document is converted to a string or
    `ChunkList` using a configurable `doc_to_content` function. This function
    can be synchronous, asynchronous, or return an `Executable`.
2.  **Concatenation**: The formatted documents are joined together using a
    configurable `delimiter`.
3.  **Template Application**: The concatenated context and the query (also
    formatted via `query_to_content`) are injected into a `template` function
    that produces the final `Composable` prompt.

A specialized version, `QADocumentPromptBuilder`, is provided for working with
OneTwo's standard `Document` data structure and integrates with OneTwo's
`DocumentFormatter` protocol.
"""

import dataclasses
import inspect
from typing import Awaitable, Callable, Generic, Iterable, TypeAlias, TypeVar

from onetwo.builtins import composables
from onetwo.core import composing
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document
Composable = composing.Composable

DocT = TypeVar('DocT')
QueryT = TypeVar('QueryT')

PromptContent: TypeAlias = str | content_lib.ChunkList
PromptContentSource = PromptContent | Awaitable[PromptContent]


@dataclasses.dataclass
class QAPromptBuilder(Generic[DocT, QueryT]):
  """Compiles a list of documents and a query into a final prompt for QA."""

  doc_to_content: Callable[[DocT], PromptContentSource] = str
  delimiter: PromptContent = '\n\n'
  query_to_content: Callable[[QueryT], PromptContentSource] = str

  template: Callable[[Composable, Composable], Composable] = (
      lambda context, question: (
          composables.c(
              """Use the following pieces of context to answer the question at
    the end. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.

    Context:
"""
          )
          + context
          + composables.c('\n\nQuestion: ')
          + question
          + composables.c('\nAnswer:')
      )
  )

  @executing.make_executable(copy_self=False)
  async def _concatenate_documents(self, docs: Iterable[DocT]) -> Composable:
    """Formats each Document and concatenates them using the delimiter."""
    e = composables.c(content_lib.ChunkList())
    for i, d in enumerate(docs):
      if i > 0:
        e += composables.c(self.delimiter)
      content = self.doc_to_content(d)
      if inspect.isawaitable(content):
        content = await content
      e += composables.c(content)
    return e

  @executing.make_executable(copy_self=False)
  async def compile_prompt(
      self, docs: Iterable[DocT], question: QueryT
  ) -> Composable:
    """Renders prompt by injecting the context and query into the template."""
    context_block = await self._concatenate_documents(docs)  # pytype: disable=wrong-arg-count
    formatted_question = self.query_to_content(question)
    if inspect.isawaitable(formatted_question):
      formatted_question = await formatted_question
    formatted_question = composables.c(formatted_question)

    return self.template(context_block, formatted_question)

  @executing.make_executable(copy_self=False)
  async def __call__(
      self, docs: Iterable[DocT], question: QueryT
  ) -> Composable:
    """Alias for compile_prompt."""
    return await self.compile_prompt(docs, question)  # pytype: disable=wrong-arg-count


@dataclasses.dataclass
class QADocumentPromptBuilder(QAPromptBuilder[_Document, str]):
  """QAPromptBuilder specialized for OneTwo Document and string query."""

  document_formatter: document_formatting.DocumentFormatter = dataclasses.field(
      default_factory=document_formatting.NoFormatting
  )

  def __post_init__(self):
    self.doc_to_content = self._format_and_extract

  @executing.make_executable(copy_self=False)
  async def _format_and_extract(self, doc: _Document) -> PromptContent:
    formatted_doc: _Document = await self.document_formatter(doc)  # pytype: disable=wrong-arg-count
    return formatted_doc.content
