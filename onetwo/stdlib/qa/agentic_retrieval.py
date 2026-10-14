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

"""QA in which the model drives its own tool use, retrieval included.

The RAG strategies in `strategies_lib` retrieve once, on the question as it was
asked, and answer from whatever comes back. That is the right shape for a
question whose answer sits in one passage, and the wrong one for a question
that has to be decomposed, or whose useful query terms only become apparent
after the first passage is read.

Here the model is given tools and decides for itself how to use them: how many
times to search, how to reword each query, which passages are worth reading in
full, and when it has enough to answer.

The tools are the caller's to choose, and this module deliberately supplies
none of them: what a corpus should look like to a model -- one search tool or
two, how much of a passage to show before it is asked for in full -- is a
question this module has no way to answer, and increasingly one the corpus
answers for itself, normally through an MCP server that declares its own tools
and is discovered at run time.

```python
  strategy = agentic_retrieval.AgenticRetrievalStrategy(
      tools=[mcp_tools.server('http://host:8777/mcp')],
  )
  result = ot.run(strategy('What is the population of the capital of France?'))
  print(result.answer)
```

Note: By default `StrategyResult.docs` is empty, so the retrieval metrics in
`rag_metrics` i.e. recall@k does not apply here, and only the answer-level ones,
such as an AI rater's accuracy, are meaningful. That is a gap rather than an
oversight: what a tool returned could be recovered by looking for document ids
in the finished conversation, but feeding that to recall@k would produce a
number that looks comparable to a single-shot strategy and is not.

A caller whose prompt asks the model to cite its evidence can instead pass
`doc_ids_from_answer` to report the documents the answer cites, e.g.
`extract_cited_doc_ids`. That measures what the model relied on, not what it
retrieved.

TODO: Work out what retrieval quality means for an agentic loop,
and report it. Three things need deciding before a number is worth having:

*   What counts as retrieved. The model may search several times and read only
    some of what comes back, so the candidate set is a union over calls, not
    one ranked list.
*   What `k` is. The model chooses how many calls to make, so the size of that
    union varies per question and per run, and comparing it against a baseline
    that retrieves a fixed `k` compares two different quantities.
*   What the ranking is. Documents arrive in the order the model happened to
    ask for them, which is chronology and not relevance.

The strategy is registered as `'agentic_retrieval'`, so it can also be built
through `strategies_lib.get_strategy('agentic_retrieval', tools=[...])`.
"""

from collections.abc import Callable, Sequence
import dataclasses
import re
from typing import Any, cast

from onetwo.agents import agent as agent_lib
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import strategies_lib
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document
_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult


# What the model is sent before it has made any tool call. A minimally-modified
# version of the prompt from the BrowseComp-Plus paper
# (https://arxiv.org/abs/2508.06600): it mentions "an MCP server" instead of "a
# search engine" to be more general purpose, and has an
# `{additional_instructions}` placeholder for MCP-server-specific instructions.
# Its wording refers to an MCP server, hence the name; `tools` need not be MCP
# tools, in which case a different `query_template` may be more appropriate.
# The answers it asks for cite their evidence in square brackets, which
# `extract_cited_doc_ids` parses.
DEFAULT_MCP_QUERY_TEMPLATE = (
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


def extract_cited_doc_ids(answer: str) -> list[str]:
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
class AgenticRetrievalStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """Answers a question by letting the model use the tools it is given.

  Attributes:
    tools: What the model may call. May be any of the types accepted by
      `agent_lib.AgentTool`.
    max_steps: Bound on the number of model calls, so that a model that keeps
      calling tools cannot loop forever. On reaching it the strategy returns no
      answer if the model's last turn still requests tool calls, unless
      `final_instruction` is set.
    final_instruction: If set, a model still calling tools when `max_steps` runs
      out gets one more call, without tools and with this instruction, to answer
      from what it has found so far (see `agent_lib.Agent`).
    query_template: What the model is sent before it has made any tool call. A
      `str.format` template with a `{question}` placeholder and, optionally, an
      `{additional_instructions}` one, the latter filled with whatever
      `additional_instructions` the question is asked with, on a line of its
      own, or with nothing. Literal braces must be doubled.
    generate_kwargs: Generation parameters passed through to the backend on
      every call, e.g. `{'temperature': 0.0}`.
    doc_ids_from_answer: If set, called on the final answer for the ids of the
      documents to report in `StrategyResult.docs`, e.g. `extract_cited_doc_ids`
      for the ones the answer cites. If not, no documents are reported; see the
      module docstring for why.
  """

  tools: Sequence[agent_lib.AgentTool] = ()
  max_steps: int = 100
  final_instruction: str = ''
  query_template: str = DEFAULT_MCP_QUERY_TEMPLATE
  generate_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  doc_ids_from_answer: Callable[[str], Sequence[str]] | None = None

  def _first_turn_content(
      self, question: _PromptContent, additional_instructions: str
  ) -> str:
    """Returns what the model is sent before it has made any tool call.

    Args:
      question: The question to answer.
      additional_instructions: Extra instructions for this question, or `''`.

    Returns:
      `query_template`, filled in.

    Raises:
      ValueError: If `query_template` has no `{question}` placeholder.
    """
    if '{question}' not in self.query_template:
      raise ValueError(
          'query_template must contain a {question} placeholder, got:'
          f' {self.query_template!r}'
      )
    instructions_block = (
        f'\n{additional_instructions}' if additional_instructions else ''
    )
    # TODO` keeps the text
    # of a `ChunkList` but replaces any non-text chunk with a placeholder.
    return self.query_template.format(
        question=str(question), additional_instructions=instructions_block
    )

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns the model's answer.

    Args:
      question: The question to answer.
      **query_time_kwargs: `additional_instructions`, if given, is a string
        added to what the model is told before the question, e.g. to describe
        the corpus the tools search. Anything else is accepted for consistency
        with the other strategies and ignored: which tools to use and how is the
        model's decision here, so there is nothing left for the caller to pass
        through.

    Returns:
      The model's answer, and the documents `doc_ids_from_answer` finds in it,
      or none if it is not set. See the module docstring for why `docs` is
      empty by default and what would have to be settled to fill it.
    """
    additional_instructions = query_time_kwargs.pop(
        'additional_instructions', ''
    )
    del query_time_kwargs
    # One question is one run of the agent.
    agent = agent_lib.Agent(
        tools=self.tools,
        max_steps=self.max_steps,
        final_instruction=self.final_instruction,
        generate_kwargs=self.generate_kwargs,
    )
    turn = fc_types.ConversationTurn(
        role=fc_types.ConversationRole.USER,
        content=self._first_turn_content(question, additional_instructions),
    )
    # Empty when the model was still calling tools as `max_steps` ran out.
    answer = cast(str, await agent([turn]))
    docs = []
    if self.doc_ids_from_answer is not None and answer:
      docs = [
          _Document(doc_id=doc_id)
          for doc_id in self.doc_ids_from_answer(answer)
      ]
    return _StrategyResult(answer=answer, docs=docs)


strategies_lib.register_strategy('agentic_retrieval', AgenticRetrievalStrategy)
