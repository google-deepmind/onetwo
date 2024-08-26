# Copyright 2024 DeepMind Technologies Limited.
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

"""Convenience functions and data structures for chain-of-thought strategies.

Generalized from the ideas of the chain-of-thought paper:
https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf

Note that chain-of-thought in its most basic form is simply a style of how to
write a prompt (where one prompts the LLM to output a reasoning chain followed
by a final answer, rather than just the final answer), and does not require any
special data structures. The data structures and helper functions here are
provided purely for convenience, for use in simple Q&A tasks and for didactic
purposes. Feel free to use them, or not, as you see fit.

Chain-of-thought variants illustrated here:
* Chain-of-thought implemented using a prompt template alone (w/2 calls).
* Chain-of-thought implemented using a prompt template (1 call) + answer parser.
* Few-shot chain-of-thought.
* Few-shot exemplars represented as data, so as to be reusable across different
  styles of prompt template.
* Few-shot chain-of-thought with different exemplars specified for each question
  (e.g., for dynamic exemplar selection).
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any
from onetwo.builtins import composables
from onetwo.core import executing
from onetwo.core import tracing
from onetwo.core import utils


@dataclasses.dataclass
class CoTReply:
  """Reply from a Chain-of-Thought strategy, bundling answer with reasoning.

  This data structure is provided for convenience for use in simple use cases
  and for compatibility with some of the other helper functions provided in this
  library. It is also fine, of course, for chain-of-thought strategies to return
  just the final answer, or to return some other arbitrary data structure.

  Attributes:
    reasoning: The reasoning that lead to the final answer.
    answer: The final answer.
  """

  reasoning: Any
  answer: Any


@dataclasses.dataclass
class QACoTExemplar:
  """QA Chain-of-Thought inputs+outputs, for use as a few-shot exemplar.

  (If your chain-of-thought strategy involves more complex inputs and outputs,
  feel free, of courrse, to define your own exemplar data structure.)

  Attributes:
    question: The question to answer.
    reasoning: The reasoning that lead to the final answer.
    answer: The final answer.
  """

  question: str
  answer: Any
  reasoning: Any


# Simple QA chain-of-thought prompt template that puts the reasoning and final
# answer on separate lines to avoid the need for a complicated parsing function.
# Note: This template is subject to change in the future. If you're interested
# in using it, we recommend cloning it into your own project and modifying as
# needed.
DEFAULT_QA_COT_PROMPT_TEXT = """\
{#- Preamble: Few-shots exemplars -#}
{%- for exemplar in exemplars -%}
Question: {{exemplar.question}}
Reasoning: {{exemplar.reasoning}}
Final answer: {{exemplar.answer}}
{{ '\n' }}
{%- endfor -%}

{#- Start of the processing of the actual inputs. -#}
Question: {{question}}
Reasoning: {{store("reasoning", generate_text(stop=["\\nFinal", "\\nAnswer", "\\nQuestion:"])) | trim }}
Final answer: {{store("answer", generate_text(stop=["\\nQuestion:"])) | trim }}
"""


@dataclasses.dataclass
class QACoTPromptJ2:
  """Basic chain-of-thought strategy based on a single Jinja2 prompt template.

  Attributes:
    prompt_text: Text of prompt template in Jinja2 format. Available input
      variables are `question` and `exemplars`. Should populate output variables
      `answer` and `reasoning`.
    exemplars: Default exemplars to use, if question-specific exemplars are not
      provided.
  """

  prompt_text: str = DEFAULT_QA_COT_PROMPT_TEXT
  exemplars: Sequence[QACoTExemplar] = tuple()

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self, question: str, exemplars: Sequence[QACoTExemplar] | None = None
  ) -> CoTReply:
    """Returns the final answer and reasoning for the given question.

    Args:
      question: The question to answer.
      exemplars: Optional few-shot exemplars to be used for this specific
        question (e.g., when performing dynamic exemplar selection).
    """
    if exemplars is None:
      exemplars = self.exemplars
    prompt_template = composables.j(self.prompt_text)
    _ = await prompt_template(question=question, exemplars=exemplars)
    return CoTReply(
        answer=prompt_template['answer'].strip(),
        reasoning=prompt_template['reasoning'].strip(),
    )


# Chain-of-thought prompt template of the form used for math word problem tasks
# like GSM8K in the original chain-of-thought paper. Rather than outputting
# separate `answer` and `reasoning` output variables, it outputs a single
# `reasoning_and_answer` variable, which is expected to be parsed into the
# `answer` and `reasoning` via a subsequent call to an answer parser.
DEFAULT_QA_COT_PROMPT_TEXT_WITH_ANSWER_PARSER = """\
{#- Preamble: Few-shots exemplars -#}
{%- for exemplar in exemplars -%}
Q: {{exemplar.question}}
A: {{exemplar.reasoning}} The answer is {{exemplar.answer}}.
{{ '\n' }}
{%- endfor -%}

{#- Start of the processing of the actual inputs. -#}
Q: {{question}}
A: {{store("reasoning_and_answer", generate_text(stop=["\\nQ:"])) | trim }}
"""


def default_cot_parse_answer(reasoning_and_answer: str) -> CoTReply:
  """Parses the LLM reply into the reasoning and final answer."""
  # List of possible delimiter strings, in order of precedence.
  reasoning_and_answer_lowercase = reasoning_and_answer.lower()
  delimiters = ['The answer is', 'Answer is']

  # We search for the delimiters in a case-insensitive way, and use the first
  # match that we find.
  for delimiter in delimiters:
    delimiter_lowercase = delimiter.lower()
    if delimiter_lowercase in reasoning_and_answer_lowercase:
      delimiter_start = reasoning_and_answer_lowercase.find(delimiter_lowercase)
      delimiter_end = delimiter_start + len(delimiter_lowercase)
      reasoning = reasoning_and_answer[0:delimiter_start].strip()
      answer = reasoning_and_answer[delimiter_end:].strip()
      # Strip trailing period.
      if answer.endswith('.'):
        answer = answer[:-1]
      return CoTReply(answer=answer, reasoning=reasoning)

  # If no delimiter string is found, then we fall back to treating the entire
  # string as the final answer.
  return CoTReply(answer=reasoning_and_answer, reasoning='')


@dataclasses.dataclass
class QACoTPromptWithAnswerParserJ2:
  """Chain-of-thought strategy using a Jinja2 prompt template + answer parser.

  Attributes:
    prompt_text: Text of prompt template in Jinja2 format. Available input
      variables are `question` and `exemplars`. Should populate output variable
      `reasoning_and_answer`.
    answer_parser: Function that parses the LLM reply into the reasoning and
      final answer.
    exemplars: Default exemplars to use, if question-specific exemplars are not
      provided.
  """

  prompt_text: str = DEFAULT_QA_COT_PROMPT_TEXT_WITH_ANSWER_PARSER
  answer_parser: Callable[[str], CoTReply] = default_cot_parse_answer
  exemplars: Sequence[QACoTExemplar] = tuple()

  @executing.make_executable(copy_self=False)
  @tracing.trace(name=utils.FROM_INSTANCE_CLASS_NAME)
  async def __call__(
      self, question: str, exemplars: Sequence[QACoTExemplar] | None = None
  ) -> CoTReply:
    """Returns the final answer and reasoning for the given question.

    Args:
      question: The question to answer.
      exemplars: Optional few-shot exemplars to be used for this specific
        question (e.g., when performing dynamic exemplar selection).
    """
    if exemplars is None:
      exemplars = self.exemplars
    prompt_template = composables.j(self.prompt_text)
    _ = await prompt_template(question=question, exemplars=exemplars)
    return self.answer_parser(prompt_template['reasoning_and_answer'])


# Exemplars used for math word problems in the original chain-of-thought paper.
# See Appendix G of https://arxiv.org/pdf/2201.11903.
# Compatible with both QACoTPromptJ2 and QACoTPromptWithAnswerParserJ2.
QA_COT_EXEMPLARS_ORIGINAL_MATH_WORD_PROBLEMS = [
    QACoTExemplar(
        question=(
            'There are 15 trees in the grove. Grove workers will plant trees in'
            ' the grove today. After they are done, there will be 21 trees. How'
            ' many trees did the grove workers plant today?'
        ),
        reasoning=(
            'There are 15 trees originally. Then there were 21 trees after some'
            ' more were planted. So there must have been 21 - 15 = 6.'
        ),
        answer='6',
    ),
    QACoTExemplar(
        question=(
            'If there are 3 cars in the parking lot and 2 more cars arrive, how'
            ' many cars are in the parking lot?'
        ),
        reasoning='There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.',
        answer='5',
    ),
    QACoTExemplar(
        question=(
            'Leah had 32 chocolates and her sister had 42. If they ate 35, how'
            ' many pieces do they have left in total?'
        ),
        reasoning=(
            'Originally, Leah had 32 chocolates. Her sister had 42. So in total'
            ' they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.'
        ),
        answer='39',
    ),
    QACoTExemplar(
        question=(
            'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason'
            ' has 12 lollipops. How many lollipops did Jason give to Denny?'
        ),
        reasoning=(
            'Jason started with 20 lollipops. Then he had 12 after giving some'
            ' to Denny. So he gave Denny 20 - 12 = 8.'
        ),
        answer='8',
    ),
    QACoTExemplar(
        question=(
            'Shawn has five toys. For Christmas, he got two toys each from his'
            ' mom and dad. How many toys does he have now?'
        ),
        reasoning=(
            'Shawn started with 5 toys. If he got 2 toys each from his mom and'
            ' dad, then that is 4 more toys. 5 + 4 = 9.'
        ),
        answer='9',
    ),
    QACoTExemplar(
        question=(
            'There were nine computers in the server room. Five more computers'
            ' were installed each day, from monday to thursday. How many'
            ' computers are now in the server room?'
        ),
        reasoning=(
            'There were originally 9 computers. For each of 4 days, 5 more'
            ' computers were added. So 5 * 4 = 20 computers were added. 9 + 20'
            ' is 29.'
        ),
        answer='29',
    ),
    QACoTExemplar(
        question=(
            'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On'
            ' wednesday, he lost 2 more. How many golf balls did he have at the'
            ' end of wednesday?'
        ),
        reasoning=(
            'Michael started with 58 golf balls. After losing 23 on tuesday, he'
            ' had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf'
            ' balls.'
        ),
        answer='33',
    ),
    QACoTExemplar(
        question=(
            'Olivia has $23. She bought five bagels for $3 each. How much money'
            ' does she have left?'
        ),
        reasoning=(
            'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 ='
            ' 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.'
        ),
        answer='8',
    ),
]
