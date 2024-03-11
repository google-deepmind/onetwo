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

from collections.abc import Sequence
from typing import Final, TypeAlias, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing


_Message: TypeAlias = llm.Message
_T = TypeVar('_T')

Chunk = content_lib.Chunk
ChunkList = content_lib.ChunkList

_SEPARATOR: Final[str] = ' @@@ '


@executing.make_executable
def _generate_text_returns_prompt_and_stop(
    prompt: str | ChunkList,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
) -> str:
  del temperature, max_tokens, top_k, top_p, decoding_constraint
  # Concatenates prompt with last element in stop seq.
  assert stop is not None and stop  # Make sure nonempty stop was provided.
  return f'{prompt}{_SEPARATOR}{stop[-1]}'


class LlmTest(parameterized.TestCase):

  def test_generate_texts(self):
    @executing.make_executable
    def generate_test_function(
        prompt: str | ChunkList,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        decoding_constraint: str | None = None,
    ) -> str:
      del temperature, max_tokens, stop, top_k, top_p, decoding_constraint
      key = caching.context_sampling_key.get()
      return str(prompt) + ' done ' + key

    # We configure the generate_text function to use `gen`.
    # By default generate_texts is configured to call generate_text multiple
    # times, so we do not need to configure it to use it.
    llm.generate_text.configure(generate_test_function)
    result = executing.run(llm.generate_text(prompt='hello'))
    # As mentioned above calling generate_texts will automatically use
    # generate_text multiple times.
    results = executing.run(llm.generate_texts(prompt='hello', samples=2))
    self.assertEqual('hello done ', result)
    self.assertEqual(['hello done ', 'hello done 1'], results)

  def test_generate_object(self):
    @executing.make_executable
    def generate_test_function(
        prompt: str | ChunkList,
        cls: type[_T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> _T:
      del prompt, cls, temperature, max_tokens, top_k, top_p
      return 1

    llm.generate_object.configure(generate_test_function)
    result = executing.run(llm.generate_object(prompt='hello', cls=int))
    self.assertEqual(1, result)

  @parameterized.named_parameters(
      (
          'has_rules_msg',
          (
              _Message(role=llm.ROLE_INSTRUCTIONS, content='Any instructions'),
              _Message(role='user', content='Hello!'),
          ),
          (
              f'Actor "{llm.ROLE_MODEL}" needs to obey the following rules '
              'when generating the messages below:\n'
              'Any instructions\n\n'
              '**user**: Hello!\n'
              '**model**:'
          ),
      ),
      (
          'has_many_rules_msgs_first_is_used',
          (
              _Message(role=llm.ROLE_INSTRUCTIONS, content='Any instructions'),
              _Message(role='user', content='Hello!'),
              _Message(role=llm.ROLE_INSTRUCTIONS, content='No instructions'),
          ),
          (
              f'Actor "{llm.ROLE_MODEL}" needs to obey the following rules '
              'when generating the messages below:\n'
              'Any instructions\n\n'
              '**user**: Hello!\n'
              '**model**:'
          ),
      ),
      (
          'has_many_rules_msgs_first_nonempty_is_used',
          (
              _Message(role=llm.ROLE_INSTRUCTIONS, content=''),
              _Message(role='user', content='Hello!'),
              _Message(role=llm.ROLE_INSTRUCTIONS, content='Any instructions'),
              _Message(role=llm.ROLE_INSTRUCTIONS, content='No instructions'),
          ),
          (
              f'Actor "{llm.ROLE_MODEL}" needs to obey the following rules '
              'when generating the messages below:\n'
              'Any instructions\n\n'
              '**user**: Hello!\n'
              '**model**:'
          ),
      ),
      (
          'when_ends_with_user_switches_to_assistant',
          (_Message(role='user', content='Hello!'),),
          '**user**: Hello!\n**model**:',
      ),
      (
          'when_ends_with_empty_assiostant_does_not_change_role',
          (
              _Message(role='user', content='Hello!'),
              _Message(role=llm.ROLE_MODEL, content=''),
          ),
          '**user**: Hello!\n**model**:',
      ),
      (
          'when_ends_with_nonempty_assiostant_does_not_change_role',
          (
              _Message(role='user', content='Hello!'),
              _Message(role=llm.ROLE_MODEL, content='Hey'),
          ),
          '**user**: Hello!\n**model**: Hey',
      ),
  )
  def test_default_chat(self, messages, expected_result):

    # We configure only the generate_text function. By default chat is
    # configured with an implementation that is based on generate_text.
    llm.generate_text.configure(_generate_text_returns_prompt_and_stop)
    result = executing.run(llm.chat(messages))
    result, stop = result.split(_SEPARATOR)

    with self.subTest('prompt_formatted_as_expected'):
      self.assertEqual(result, expected_result)

    with self.subTest('stop_seq_added_as_expected'):
      self.assertEqual(stop, '\n**')

  def test_select(self):
    @executing.make_executable
    def score(
        prompt: str | ChunkList, targets: Sequence[str]
    ) -> Sequence[float]:
      del prompt
      return [float(i // 2) for i in range(len(targets))]

    llm.score_text.configure(score)
    targets = ['a', 'b', 'c', 'd', 'e', 'f']
    with self.subTest('score_returns_correct_results'):
      scores = executing.run(llm.score_text(prompt='hello', targets=targets))
      self.assertEqual([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], scores)

    with self.subTest('select_picks_e_or_f'):
      result = executing.run(
          llm.select(prompt='choose from a to f:', options=targets)
      )
      self.assertIn(result, ['e', 'f'])

    with self.subTest('select_returns_details'):
      result = executing.run(
          llm.select(
              prompt='choose from a to e:',
              options=targets[:-1],
              include_details=True,
          )
      )
      self.assertEqual(result, ('e', 4, [0.0, 0.0, 1.0, 1.0, 2.0]))

  def test_rank(self):
    @executing.make_executable
    def score(
        prompt: str | ChunkList, targets: Sequence[str]
    ) -> Sequence[float]:
      del prompt
      return [float(i // 2) for i in range(len(targets))]

    llm.score_text.configure(score)
    targets = ['a', 'b', 'c', 'd', 'e', 'f']

    with self.subTest('rank_picks_top3'):
      result = executing.run(
          llm.rank(prompt='choose from a to f:', options=targets, top_k=3)
      )
      self.assertListEqual(result, ['e', 'f', 'c'])

    with self.subTest('rank_returns_all'):
      result = executing.run(
          llm.rank(prompt='choose from a to f:', options=targets, top_k=0)
      )
      # Note that we rank by decreasing score but identical scores are kept
      # in the original order.
      self.assertListEqual(result, ['e', 'f', 'c', 'd', 'a', 'b'])

    with self.subTest('rank_returns_details'):
      result = executing.run(
          llm.rank(
              prompt='choose from a to e:',
              options=targets,
              top_k=3,
              include_details=True,
          )
      )
      self.assertEqual(
          result, (['e', 'f', 'c'], [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
      )

  def test_count_tokens(self):
    @executing.make_executable
    def tokenize(content: str | ChunkList) -> Sequence[int]:
      if isinstance(content, ChunkList):
        content = str(content)
      # We count the number of words in the string.
      return [len(chunk) for chunk in content.split()]

    llm.tokenize.configure(tokenize)
    result = executing.run(llm.count_tokens(content='hello world'))
    self.assertEqual(result, 2)

  @parameterized.named_parameters(
      (
          'prompt_no_prefix',
          'Write a code',
          None,
          'Task: Write a code\nAnswer:',
      ),
      (
          'empty_prompt',
          '',
          None,
          'Task:\nAnswer:',
      ),
      (
          'prompt_and_prefix',
          'Write a code',
          'Start',
          'Task: Write a code\nAnswer: Start',
      ),
  )
  def test_default_instruct(
      self,
      prompt,
      assistant_prefix,
      expected_result_wo_fewshot,
  ):
    """Verify that default implementation of instruct behaves as expected."""

    expected_result = llm.DEFAULT_INSTRUCT_FEWSHOT + expected_result_wo_fewshot
    # We configure only the generate_text function. By default instruct is
    # configured with an implementation that is based on generate_text.
    llm.generate_text.configure(_generate_text_returns_prompt_and_stop)
    result = executing.run(
        llm.instruct(
            prompt=prompt,
            assistant_prefix=assistant_prefix,
        )
    )
    result, stop = result.split(_SEPARATOR)

    with self.subTest('prompt_formatted_as_expected'):
      msg = f'\ngot:\n{result}\n***\nexpected:\n{expected_result}\n***'
      self.assertEqual(result, expected_result, msg)

    with self.subTest('stop_seq_added_as_expected'):
      self.assertEqual(stop, '\nTask:')

    result = executing.run(
        llm.instruct(
            prompt=prompt,
            assistant_prefix=assistant_prefix,
            use_fewshot=False,
        )
    )
    result, _ = result.split(_SEPARATOR)
    with self.subTest('prompt_without_fewshot_formatted_as_expected'):
      msg = f'\ngot:\n{result}\n***\nexpected:\n{expected_result}\n***'
      self.assertEqual(result, expected_result_wo_fewshot, msg)


if __name__ == '__main__':
  absltest.main()
