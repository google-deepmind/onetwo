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
      instructions=agentic_retrieval.RETRIEVAL_INSTRUCTIONS,
  )
  result = ot.run(strategy('What is the population of the capital of France?'))
  print(result.answer)
```

Note: `StrategyResult.docs` is always empty, so the retrieval metrics in
`rag_metrics` i.e. recall@k does not apply here, and only the answer-level ones,
such as an AI rater's accuracy, are meaningful. That is a gap rather than an
oversight: what a tool returned could be recovered by looking for document ids
in the finished conversation, but feeding that to recall@k would produce a
number that looks comparable to a single-shot strategy and is not.

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

from collections.abc import Sequence
import dataclasses
from typing import Any, cast

from onetwo.agents import agent as agent_lib
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import strategies_lib

_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult


# What the model is told before the question when it is given arbitrary tools.
DEFAULT_INSTRUCTIONS = (
    'Answer the question using the tools available to you. Use them as many'
    ' times as you need to. When you have enough to answer, reply with the'
    ' answer in prose rather than with another tool call.'
)

# The same, to pass as `instructions` when the tools search a corpus. It says
# nothing about how to search, since choosing that is the whole point of
# handing over the tools, and it names no tool, since which ones exist is the
# caller's business -- an MCP server is discovered at run time.
RETRIEVAL_INSTRUCTIONS = (
    'Answer the question using the tools available to you to search for'
    ' relevant passages. Search as many times as you need to, rewording the'
    ' query if the results are not useful, and read a passage in full when'
    ' its summary looks promising and a tool will give you the rest. When you'
    ' have enough to answer, reply with the answer in prose rather than with'
    ' another tool call.'
)


@dataclasses.dataclass
class AgenticRetrievalStrategy(qa.QAStrategy[_PromptContent, _StrategyResult]):
  """Answers a question by letting the model use the tools it is given.

  Attributes:
    tools: What the model may call. May be any of the types accepted by
      `agent_lib.AgentTool`.
    max_steps: Bound on the number of model calls, so that a model that keeps
      calling tools cannot loop forever. On reaching it the strategy returns
      whatever the model last said, which may be nothing.
    instructions: What the model is told before the question. See
      `RETRIEVAL_INSTRUCTIONS` for wording that suits a index.
    generate_kwargs: Generation parameters passed through to the backend on
      every call, e.g. `{'temperature': 0.0}`.
  """

  tools: Sequence[agent_lib.AgentTool] = ()
  max_steps: int = 100
  instructions: str = DEFAULT_INSTRUCTIONS
  generate_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs,
  ) -> _StrategyResult:
    """Returns the model's answer.

    Args:
      question: The question to answer.
      **query_time_kwargs: Accepted for consistency with the other strategies
        and ignored: which tools to use and how is the model's decision here, so
        there is nothing left for the caller to pass through.

    Returns:
      The model's answer, and no documents. See the module docstring for why
      `docs` is empty and what would have to be settled to fill it.
    """
    del query_time_kwargs
    # One question is one run of the agent.
    agent = agent_lib.Agent(
        tools=self.tools,
        max_steps=self.max_steps,
        generate_kwargs=self.generate_kwargs,
    )
    opening = f'{self.instructions}\n\nQUESTION:\n'
    turn = fc_types.ConversationTurn(
        role=fc_types.ConversationRole.USER, content=opening + question
    )
    final_turn = cast(fc_types.ConversationTurn | None, await agent([turn]))

    answer = ''
    if final_turn is not None and final_turn.content:
      answer = str(final_turn.content)
    return _StrategyResult(answer=answer, docs=[])


strategies_lib.register_strategy('agentic_retrieval', AgenticRetrievalStrategy)
