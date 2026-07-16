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

"""MCP-related QA strategies."""

import asyncio
import dataclasses
import logging
import re
from typing import Any

import fastmcp
from fastmcp.client import transports as fastmcp_transports
from google.genai import types as genai_types
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.stdlib.qa import qa

_PromptContent = str | content_lib.ChunkList
StrategyResult = qa.StrategyResult

# TODO: Think of a standard way of speficying a
# template and coupling it to how its output is parsed. This way, we would not
# need to worry about coupling the prompt with the output parsing.

# Minimally-modified version of the original prompt used in the BrowseComp-Plus
# paper (https://arxiv.org/abs/2508.06600). We modified the first part of the
# prompt to mention "an MCP server" instead of "a search engine" to make it
# more general purpose. Also added additional_instructions to allow passing
# MCP-server-specific instructions if needed.
default_qa_template_mcp = (
    'You are a deep research agent. You need to answer the given question by'
    ' interacting with an MCP server, using the tools provided. Please perform'
    ' reasoning and use the tools in a step by step, interleaved manner. You'
    ' may use the tools multiple times. {additional_instructions}'
    '\n\n'
    'Question: {question}'
    '\n\n'
    'Your response should be in the following format:\n'
    'Explanation: {{your explanation for your final answer. For this'
    ' explanation section only, you should cite your evidence documents inline'
    ' by enclosing their docids in square brackets [] at the end of sentences.'
    ' For example, [20].}}\n'
    'Exact Answer: {{your succinct, final answer}}\n'
    'Confidence: {{your confidence score between 0% and 100% for your answer}}'
)


def _extract_cited_doc_ids(answer: str) -> list[str]:
  """Extracts unique cited document IDs from the answer.

  Parses bracket-enclosed citations from the model's response text and returns
  them as a deduplicated list of strings in the order they first appear.
  Supports both single citations like ``[16935]`` and comma-separated
  citations like ``[73758, 81636]``.

  Args:
    answer: The raw answer string from the model.

  Returns:
    A list of unique document ID strings.
  """
  seen = set()
  doc_ids = []
  for match in re.finditer(r'\[([^\[\]]+)\]', answer):
    for doc_id in match.group(1).split(','):
      doc_id = doc_id.strip()
      if doc_id and doc_id not in seen:
        seen.add(doc_id)
        doc_ids.append(doc_id)
  return doc_ids


@dataclasses.dataclass
# TODO: b/535122834 - Rename to ClientSideMCPStrategy or something more generic
# once we standardize MCP tool passing and configuration across backends.
class GoogleGenAIMCPStrategy(qa.QAStrategy[_PromptContent, StrategyResult]):
  """A strategy that runs an MCP server client-side and passes it to the model.

  This strategy initiates a connection to the MCP server client-side and uses a
  manual async execution loop to call tools locally on the developer's machine.
  This allows testing with HTTP (localhost) MCP servers without needing HTTPS.

  Attributes:
    mcp_server_url: The URL of the MCP server to connect to.
    query_template: A format string used to wrap the question before sending it
      to the LLM. Must contain both ``{question}`` and
      ``{additional_instructions}`` placeholders.
  """

  mcp_server_url: str = ''
  query_template: str = default_qa_template_mcp
  _client: Any = dataclasses.field(default=None, init=False)
  _client_lock: asyncio.Lock = dataclasses.field(
      default_factory=asyncio.Lock, init=False
  )

  def __post_init__(self):
    if not self.mcp_server_url:
      raise ValueError('GoogleGenAIMCPStrategy requires mcp_server_url.')
    if '{question}' not in self.query_template:
      raise ValueError(
          'query_template must contain a {question} placeholder, got:'
          f' {self.query_template!r}'
      )
    if '{additional_instructions}' not in self.query_template:
      raise ValueError(
          'query_template must contain an {additional_instructions}'
          f' placeholder, got: {self.query_template!r}'
      )

  def _format_question(
      self,
      question: _PromptContent,
      additional_instructions: str,
  ) -> _PromptContent:
    """Formats the question by applying the query template.

    Args:
      question: The raw question, either a string or a ChunkList.
      additional_instructions: Extra instructions to inject into the template.

    Returns:
      The formatted question with the template applied.
    """
    instructions_block = (
        f'\n{additional_instructions}' if additional_instructions else ''
    )

    if isinstance(question, str):
      return self.query_template.format(
          question=question,
          additional_instructions=instructions_block,
      )
    elif isinstance(question, content_lib.ChunkList):
      template_with_instructions = self.query_template.replace(
          '{additional_instructions}', instructions_block
      )
      parts = template_with_instructions.split('{question}')
      if len(parts) == 2:
        head, tail = parts
        return content_lib.ChunkList(
            [content_lib.Chunk(head)]
            + list(question)
            + [content_lib.Chunk(tail)]
        )
    return question

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs: Any,
  ) -> StrategyResult:

    additional_instructions = query_time_kwargs.pop(
        'additional_instructions', ''
    )
    del query_time_kwargs

    question = self._format_question(question, additional_instructions)

    async with self._client_lock:
      if not self._client:
        logging.info(
            'Connecting to MCP server client-side: %s', self.mcp_server_url
        )
        transport = fastmcp_transports.StreamableHttpTransport(
            url=self.mcp_server_url
        )
        client = fastmcp.Client(transport)
        await client.__aenter__()

        # Monkey-patch __deepcopy__ to prevent copy.deepcopy from failing on
        # live session objects (like Futures) during GenAI SDK's internal
        # config copying. Returning self preserves the shared live connection.
        client.session.__deepcopy__ = lambda memo: client.session
        self._client = client

    kwargs = {
        'tools': [self._client.session],
        'automatic_function_calling': (
            genai_types.AutomaticFunctionCallingConfig(maximum_remote_calls=100)
        ),
    }

    messages = [
        content_lib.Message(
            role=content_lib.PredefinedRole.USER,
            content=question,
        )
    ]
    try:
      answer = await llm.chat(messages, **kwargs)  # pytype: disable=wrong-arg-count
      answer_str = str(answer)
      return StrategyResult(
          answer=answer_str,
          doc_ids=_extract_cited_doc_ids(answer_str),
      )
    except Exception as e:
      logging.exception('Error during GoogleGenAIMCPStrategy execution')
      raise RuntimeError(f'{type(e).__name__}: {str(e)}') from e
