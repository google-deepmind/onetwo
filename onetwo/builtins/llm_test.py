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

from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.builtins import llm_utils
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling

_T = TypeVar('_T')

_Message: TypeAlias = content_lib.Message
_TokenHealingOption: TypeAlias = llm.TokenHealingOption
_Chunk = content_lib.Chunk
_ChunkList = content_lib.ChunkList
_PredefinedRole = content_lib.PredefinedRole

_SEPARATOR: Final[str] = ' @@@ '


def score_factory(score_by_prefix_suffix_pair: Mapping[tuple[str, str], float]):
  """Utility method for constructing a scoring function."""

  @executing.make_executable
  def _score_text(
      prompt: str | _ChunkList,
      targets: Sequence[str],
      healing_option: _TokenHealingOption = _TokenHealingOption.NONE,
  ) -> Sequence[float]:
    healed_prompt, healed_targets = llm_utils.maybe_heal_prompt_and_targets(
        original_prompt=prompt,
        original_targets=targets,
        healing_option=healing_option,
    )
    if isinstance(healed_prompt, _ChunkList):
      healed_prompt = healed_prompt.to_simple_string()
    for healed_target in healed_targets:
      if (healed_prompt, healed_target) not in score_by_prefix_suffix_pair:
        raise ValueError(
            f'No score found for prompt={healed_prompt} '
            f'and target={healed_target}'
        )
    return [
        score_by_prefix_suffix_pair[(healed_prompt, healed_target)]
        for healed_target in healed_targets
    ]

  return _score_text


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
          msg.role.value if isinstance(msg.role, _PredefinedRole) else msg.role
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

    # We configure the generate_text function to use `generate_test_function`.
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
      return 1  # pyrefly: ignore[bad-return]

    llm.generate_object.configure(generate_test_function)
    result = executing.run(llm.generate_object(prompt='hello', cls=int))
    self.assertEqual(1, result)

  @parameterized.named_parameters(
      (
          'no_healing_no_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          False,
          _TokenHealingOption.NONE,
          'd',
      ),
      (
          'no_healing_with_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          True,
          _TokenHealingOption.NONE,
          ('d', 3, [0.1, 0.2, 0.3, 0.4]),
      ),
      (
          'space_healing_no_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          False,
          _TokenHealingOption.SPACE_HEALING,
          'a',
      ),
      (
          'space_healing_with_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          True,
          _TokenHealingOption.SPACE_HEALING,
          ('a', 0, [1.4, 1.3, 1.2, 1.1]),
      ),
  )
  def test_default_select(
      self, prompt, targets, include_details, healing_option, expected_result
  ):
    score_by_prefix_suffix_pair = {
        ('hello ', 'a'): 0.1,
        ('hello ', 'b'): 0.2,
        ('hello ', 'c'): 0.3,
        ('hello ', 'd'): 0.4,
        ('hello', ' a'): 1.4,
        ('hello', ' b'): 1.3,
        ('hello', ' c'): 1.2,
        ('hello', ' d'): 1.1,
    }
    llm.score_text.configure(score_factory(score_by_prefix_suffix_pair))

    exe = llm.select(
        prompt=prompt,
        options=targets,
        include_details=include_details,
        healing_option=healing_option,
    )
    result = executing.run(exe)
    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      (
          'no_healing_no_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          3,
          False,
          _TokenHealingOption.NONE,
          ['c', 'd', 'a'],
      ),
      (
          'no_healing_with_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          3,
          True,
          _TokenHealingOption.NONE,
          (['c', 'd', 'a'], [0.1, 0.1, 0.2, 0.2]),
      ),
      (
          'no_healing_no_details_topk_0',
          'hello ',
          ['a', 'b', 'c', 'd'],
          0,
          False,
          _TokenHealingOption.NONE,
          ['c', 'd', 'a', 'b'],
      ),
      (
          'space_healing_no_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          3,
          False,
          _TokenHealingOption.SPACE_HEALING,
          ['a', 'b', 'c'],
      ),
      (
          'space_healing_with_details',
          'hello ',
          ['a', 'b', 'c', 'd'],
          3,
          True,
          _TokenHealingOption.SPACE_HEALING,
          (['a', 'b', 'c'], [1.2, 1.2, 1.1, 1.1]),
      ),
  )
  def test_default_rank(
      self,
      prompt,
      targets,
      top_k,
      include_details,
      healing_option,
      expected_result,
  ):

    score_by_prefix_suffix_pair = {
        ('hello ', 'a'): 0.1,
        ('hello ', 'b'): 0.1,
        ('hello ', 'c'): 0.2,
        ('hello ', 'd'): 0.2,
        ('hello', ' a'): 1.2,
        ('hello', ' b'): 1.2,
        ('hello', ' c'): 1.1,
        ('hello', ' d'): 1.1,
    }
    llm.score_text.configure(score_factory(score_by_prefix_suffix_pair))

    exe = llm.rank(
        prompt=prompt,
        options=targets,
        include_details=include_details,
        healing_option=healing_option,
        top_k=top_k,
    )
    result = executing.run(exe)
    self.assertEqual(result, expected_result)

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

  def test_detokenize(self):

    @executing.make_executable
    def detokenize(tokens: Sequence[int]) -> str:
      # We simply convert the integers to strings and join them with spaces.
      return ' '.join([str(token) for token in tokens])

    llm.detokenize.configure(detokenize)
    result = executing.run(llm.detokenize(tokens=[1, 2, 3, 4]))
    self.assertEqual(result, '1 2 3 4')

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
            formatter=formatter,
        )
    )
    with self.subTest('prompt_formatted_as_expected'):
      self.assertEqual(result, expected_result)

  def test_generate_with_tools(self):
    llm.reset_defaults()
    with self.assertRaises(NotImplementedError):
      executing.run(llm.generate_with_tools(turns=[], tools=[]))

    expected = function_calling.ConversationTurn(
        role=function_calling.ConversationRole.MODEL, content='Done'
    )

    @executing.make_executable
    def fake_impl(
        turns: Sequence[function_calling.ConversationTurn],
        tools: Sequence[function_calling.ToolSpec],
        **kwargs,
    ) -> function_calling.ConversationTurn:
      del turns, tools, kwargs
      return expected

    llm.generate_with_tools.configure(fake_impl)
    self.assertEqual(
        expected, executing.run(llm.generate_with_tools(turns=[], tools=[]))
    )


class PrefixForTaskTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='retrieval_query',
          task_type=llm.EmbedTaskType.RETRIEVAL_QUERY,
          title=None,
          expected='task: search result | query: ',
      ),
      dict(
          testcase_name='task_type_as_string',
          task_type='CLASSIFICATION',
          title=None,
          expected='task: classification | query: ',
      ),
      dict(
          testcase_name='retrieval_document_with_title',
          task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='My title',
          expected='title: My title | text: ',
      ),
      dict(
          testcase_name='retrieval_document_without_title',
          task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title=None,
          expected='title: none | text: ',
      ),
      dict(
          testcase_name='title_without_task_type',
          task_type=None,
          title='My title',
          expected='title: My title | text: ',
      ),
      dict(
          testcase_name='no_task_type_and_no_title',
          task_type=None,
          title=None,
          expected=None,
      ),
      dict(
          testcase_name='no_task_type_and_empty_title',
          task_type=None,
          title='',
          expected=None,
      ),
      dict(
          testcase_name='retrieval_document_with_empty_title',
          task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
          title='',
          expected='title: none | text: ',
      ),
  )
  def test_prefix_for_task_type(self, task_type, title, expected):
    self.assertEqual(expected, llm.prefix_for_task_type(task_type, title))

  def test_prefix_defined_for_all_task_types(self):
    # `RETRIEVAL_DOCUMENT` is the only task type whose prefix is not static.
    self.assertEqual(
        set(llm.EmbedTaskType) - {llm.EmbedTaskType.RETRIEVAL_DOCUMENT},
        set(llm.PREFIX_FROM_TASK_TYPE),
    )


class EmbedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', {}, True),
      ('normalize', {'normalize': True}, True),
      ('do_not_normalize', {'normalize': False}, False),
  )
  def test_embed_passes_normalize_to_the_implementation(
      self, embed_kwargs, expected_normalize
  ):
    @executing.make_executable
    def fake_embed(
        content: str | _ChunkList, *, normalize: bool = True, **kwargs
    ) -> Sequence[float]:
      del content, kwargs
      return [1.0] if normalize else [0.0]

    llm.embed.configure(fake_embed)

    result = executing.run(llm.embed('hello', **embed_kwargs))

    self.assertEqual([1.0] if expected_normalize else [0.0], result)


if __name__ == '__main__':
  absltest.main()
