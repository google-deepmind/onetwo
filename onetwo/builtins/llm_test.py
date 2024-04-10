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
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing


_T = TypeVar('_T')

_Message: TypeAlias = content_lib.Message
_Chunk = content_lib.Chunk
_ChunkList = content_lib.ChunkList
_PredefinedRole = content_lib.PredefinedRole

_SEPARATOR: Final[str] = ' @@@ '


@executing.make_executable
def _generate_text_returns_prompt_and_stop(
    prompt: str | _ChunkList,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    decoding_constraint: str | None = None,
) -> str:
  del temperature, max_tokens, top_k, top_p, decoding_constraint
  # Concatenates prompt with the stop sequences.
  if stop:
    seqs = ','.join(sorted(stop))
    return f'{prompt}{_SEPARATOR}{seqs}'
  else:
    return f'{prompt}'


class FormatterForTest(formatting.Formatter):
  """Formatter for testing."""

  @property
  def role_map(self) -> dict[str | _PredefinedRole, str]:
    return {
        _PredefinedRole.USER: 'user',
        _PredefinedRole.MODEL: 'model',
    }

  def is_already_formatted(self, content: Sequence[_Message]) -> bool:
    """Returns whether the content is already formatted."""
    return any([('</' in str(msg.content)) for msg in content])

  def extra_stop_sequences(self) -> list[str]:
    return ['<user>']

  def _format(
      self,
      content: Sequence[_Message],
  ) -> _ChunkList:
    """Returns formatted ChunkList."""
    result = _ChunkList()
    for msg in content:
      role = (
          msg.role.value
          if isinstance(msg.role, _PredefinedRole)
          else msg.role
      )
      result += _Chunk(content=f'<{role}>{msg.content}</{role}>\n')
    return result


class LlmTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

  def test_generate_texts(self):
    @executing.make_executable
    def generate_test_function(
        prompt: str | _ChunkList,
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
        prompt: str | _ChunkList,
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

  def test_select(self):
    @executing.make_executable
    def score(
        prompt: str | _ChunkList, targets: Sequence[str]
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
        prompt: str | _ChunkList, targets: Sequence[str]
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
    def tokenize(content: str | _ChunkList) -> Sequence[int]:
      if isinstance(content, _ChunkList):
        content = str(content)
      # We count the number of words in the string.
      return [len(chunk) for chunk in content.split()]

    llm.tokenize.configure(tokenize)
    result = executing.run(llm.count_tokens(content='hello world'))
    self.assertEqual(result, 2)

  @parameterized.named_parameters(
      (
          'user_only_no_formatter',
          [_Message(role=_PredefinedRole.USER, content='Hello')],
          formatting.FormatterName.NONE,
          'Hello',
      ),
      (
          'user_and_model_no_formatter',
          [
              _Message(role=_PredefinedRole.USER, content='Hello'),
              _Message(role=_PredefinedRole.MODEL, content='What can I'),
          ],
          formatting.FormatterName.NONE,
          'HelloWhat can I',
      ),
      (
          'user_only_formatter',
          [_Message(role=_PredefinedRole.USER, content='Hello')],
          formatting.FormatterName.DEFAULT,
          'Task: Hello\nAnswer:',
      ),
      (
          'user_and_no_formatter',
          [
              _Message(role=_PredefinedRole.USER, content='Hello'),
              _Message(role=_PredefinedRole.MODEL, content='What can I'),
          ],
          formatting.FormatterName.DEFAULT,
          'Task: Hello\nAnswer: What can I',
      ),
  )
  def test_default_chat(
      self,
      messages: Sequence[_Message],
      formatter: formatting.FormatterName,
      expected_result: str,
  ):
    # We configure only the generate_text function. By default chat is
    # configured with an implementation that is based on generate_text.
    llm.generate_text.configure(_generate_text_returns_prompt_and_stop)
    result = executing.run(llm.chat(messages, formatter=formatter))
    if formatter == formatting.FormatterName.DEFAULT:
      result, stop = result.split(_SEPARATOR)
      with self.subTest('stop_seq_added_as_expected'):
        self.assertEqual(stop, '\n**User**:,\nTask:')

    with self.subTest('prompt_formatted_as_expected'):
      self.assertEqual(result, expected_result)

  def test_default_chat_raises(self):
    llm.generate_text.configure(_generate_text_returns_prompt_and_stop)
    with self.assertRaises(NotImplementedError):
      _ = executing.run(
          llm.chat(
              [_Message(role=_PredefinedRole.USER, content='Hello')],
              formatter=formatting.FormatterName.API,
          )
      )

  @parameterized.named_parameters(
      (
          'user_only_no_formatter',
          'Hello',
          None,
          formatting.FormatterName.NONE,
          'Hello',
      ),
      (
          'user_and_model_no_formatter',
          'Hello',
          'What can I',
          formatting.FormatterName.NONE,
          'HelloWhat can I',
      ),
      (
          'user_only_formatter',
          'Hello',
          None,
          formatting.FormatterName.DEFAULT,
          f'Task: Hello\nAnswer:{_SEPARATOR}\n**User**:,\nTask:',
      ),
      (
          'user_and_no_formatter',
          'Hello',
          'What can I',
          formatting.FormatterName.DEFAULT,
          f'Task: Hello\nAnswer: What can I{_SEPARATOR}\n**User**:,\nTask:',
      ),
  )
  def test_default_instruct(
      self,
      prompt: str,
      assistant_prefix: str,
      formatter: formatting.FormatterName,
      expected_result: str,
  ):
    # We configure only the generate_text function. By default instruct is
    # configured with an implementation that is based on generate_text.
    llm.generate_text.configure(_generate_text_returns_prompt_and_stop)
    result = executing.run(
        llm.instruct(
            prompt=prompt,
            assistant_prefix=assistant_prefix,
            formatter=formatter
        )
    )
    with self.subTest('prompt_formatted_as_expected'):
      self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
