# Copyright 2025 DeepMind Technologies Limited.
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

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.backends import backends_test_utils
from onetwo.core import executing
from onetwo.stdlib.reasoning import chain_of_thought

# Default reply for LLMForTest to return when it receives a prompt that it was
# not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'


class ChainOfThoughtTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('j2', chain_of_thought.QACoTPromptJ2),
      ('chat', chain_of_thought.QACoTPromptChat),
  )
  def test_qa_cot_prompt_zero_shot(self, prompt_class):
    question = 'What is the answer?'
    prompt = prompt_class()

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request_1 = """\
Question: What is the answer?
Reasoning: """
    expected_request_2 = """\
Question: What is the answer?
Reasoning: Let me think...
Final answer: """
    simulated_llm_reply_1 = 'Let me think... '
    simulated_llm_reply_2 = '\n42'
    expected_cot_reply = chain_of_thought.CoTReply(
        reasoning='Let me think...', answer='42'
    )

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request_1: simulated_llm_reply_1,
            expected_request_2: simulated_llm_reply_2,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    cot_reply = executing.run(prompt(question=question))

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_trimmed_llm_reply_as_answer'):
      self.assertEqual(expected_cot_reply, cot_reply)

  @parameterized.named_parameters(
      ('j2', chain_of_thought.QACoTPromptJ2),
      ('chat', chain_of_thought.QACoTPromptChat),
  )
  def test_qa_cot_prompt_few_shot(self, prompt_class):
    question = 'What is the answer?'
    exemplars = [
        chain_of_thought.QACoTExemplar(
            question='Q1', reasoning='R1', answer='A1'
        ),
        chain_of_thought.QACoTExemplar(
            question='Q2', reasoning='R2', answer='A2'
        ),
    ]
    prompt = prompt_class(exemplars=exemplars)

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request_1 = """\
Question: Q1
Reasoning: R1
Final answer: A1

Question: Q2
Reasoning: R2
Final answer: A2

Question: What is the answer?
Reasoning: """
    expected_request_2 = """\
Question: Q1
Reasoning: R1
Final answer: A1

Question: Q2
Reasoning: R2
Final answer: A2

Question: What is the answer?
Reasoning: Let me think...
Final answer: """
    simulated_llm_reply_1 = 'Let me think... '
    simulated_llm_reply_2 = '\n42'
    expected_cot_reply = chain_of_thought.CoTReply(
        reasoning='Let me think...', answer='42'
    )

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request_1: simulated_llm_reply_1,
            expected_request_2: simulated_llm_reply_2,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    cot_reply = executing.run(prompt(question=question))

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_trimmed_llm_reply_as_answer'):
      self.assertEqual(expected_cot_reply, cot_reply)

  @parameterized.named_parameters(
      ('j2', chain_of_thought.QACoTPromptWithAnswerParserJ2),
      ('chat', chain_of_thought.QACoTPromptWithAnswerParserChat),
  )
  def test_qa_cot_prompt_with_answer_parser_zero_shot(self, prompt_class):
    question = 'What is the answer?'
    prompt = prompt_class()

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request = """\
Q: What is the answer?
A: """
    simulated_llm_reply = 'Let me think... The answer is 42.'
    expected_cot_reply = chain_of_thought.CoTReply(
        reasoning='Let me think...', answer='42'
    )

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request: simulated_llm_reply,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    cot_reply = executing.run(prompt(question=question))

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_parsed_llm_reply'):
      self.assertEqual(expected_cot_reply, cot_reply)

  @parameterized.named_parameters(
      ('j2', chain_of_thought.QACoTPromptWithAnswerParserJ2),
      ('chat', chain_of_thought.QACoTPromptWithAnswerParserChat),
  )
  def test_qa_cot_prompt_with_answer_parser_few_shot(self, prompt_class):
    question = 'What is the answer?'
    exemplars = [
        chain_of_thought.QACoTExemplar(
            question='Q1', reasoning='R1', answer='A1'
        ),
        chain_of_thought.QACoTExemplar(
            question='Q2', reasoning='R2', answer='A2'
        ),
    ]
    prompt = prompt_class(exemplars=exemplars)

    # Expected requests (based on above inputs), along with simulated replies.
    expected_request = """\
Q: Q1
A: R1 The answer is A1.

Q: Q2
A: R2 The answer is A2.

Q: What is the answer?
A: """
    simulated_llm_reply = 'Let me think... The answer is 42.'
    expected_cot_reply = chain_of_thought.CoTReply(
        reasoning='Let me think...', answer='42'
    )

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request: simulated_llm_reply,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    cot_reply = executing.run(prompt(question=question))

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_parsed_llm_reply'):
      self.assertEqual(expected_cot_reply, cot_reply)

  @parameterized.named_parameters(
      ('j2', chain_of_thought.QACoTPromptWithAnswerParserJ2),
      ('chat', chain_of_thought.QACoTPromptWithAnswerParserChat),
  )
  def test_qa_cot_prompt_with_answer_parser_dynamic_exemplars(
      self, prompt_class):
    question = 'What is the answer?'
    default_exemplars = [
        chain_of_thought.QACoTExemplar(
            question='Q1', reasoning='R1', answer='A1'
        ),
    ]
    question_specific_exemplars = [
        chain_of_thought.QACoTExemplar(
            question='Q2', reasoning='R2', answer='A2'
        ),
    ]
    prompt = prompt_class(exemplars=default_exemplars)

    # Expected requests (based on above inputs), along with simulated replies.
    # We expect the prompt to be generated using the question-specific exemplars
    # rather than the default exemplars.
    expected_request = """\
Q: Q2
A: R2 The answer is A2.

Q: What is the answer?
A: """
    simulated_llm_reply = 'Let me think... The answer is 42.'
    expected_cot_reply = chain_of_thought.CoTReply(
        reasoning='Let me think...', answer='42'
    )

    # Now we can define our test LLM that sends deterministic answers to the
    # specified prompts.
    llm_backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request: simulated_llm_reply,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
    )
    llm_backend.register()

    # Now we execute the prompt and verify that the prompt was generated as
    # expected.
    cot_reply = executing.run(
        prompt(question=question, exemplars=question_specific_exemplars)
    )

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(llm_backend.unexpected_prompts)

    with self.subTest('should_return_the_parsed_llm_reply'):
      self.assertEqual(expected_cot_reply, cot_reply)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_string',
          reasoning_and_answer='',
          expected_cot_reply=chain_of_thought.CoTReply(
              reasoning='', answer=''
          ),
      ),
      dict(
          testcase_name='expected_form',
          reasoning_and_answer='Let me think... The answer is 42',
          expected_cot_reply=chain_of_thought.CoTReply(
              reasoning='Let me think...', answer='42'
          ),
      ),
      dict(
          testcase_name='ignores_extra_newline_between_reasoning_and_answer',
          reasoning_and_answer='Let me think...\nThe answer is 42',
          expected_cot_reply=chain_of_thought.CoTReply(
              reasoning='Let me think...', answer='42'
          ),
      ),
      dict(
          testcase_name='ignores_trailing_newline',
          reasoning_and_answer='Let me think... The answer is 42.\n',
          expected_cot_reply=chain_of_thought.CoTReply(
              reasoning='Let me think...', answer='42'
          ),
      ),
      dict(
          testcase_name='if_missing_delimiter_then_treats_reply_as_answer',
          reasoning_and_answer='Let me think... 42.',
          expected_cot_reply=chain_of_thought.CoTReply(
              reasoning='', answer='Let me think... 42.'
          ),
      ),
  )
  def test_default_cot_parse_answer(
      self,
      reasoning_and_answer: str,
      expected_cot_reply: chain_of_thought.CoTReply,
  ):
    self.assertEqual(
        expected_cot_reply,
        chain_of_thought.default_cot_parse_answer(reasoning_and_answer),
    )


if __name__ == '__main__':
  absltest.main()
