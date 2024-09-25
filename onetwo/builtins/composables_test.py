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

from collections.abc import Iterator, Sequence
from typing import TypeVar

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import composables as c
from onetwo.builtins import llm
from onetwo.core import composing
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import updating


_T = TypeVar('_T')


Chunk = content_lib.Chunk
ChunkList = content_lib.ChunkList
Message = content_lib.Message
PredefinedRole = content_lib.PredefinedRole


@executing.make_executable
def generate_test_temp_function(
    prompt: str | ChunkList,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
  del prompt, max_tokens, stop, top_k, top_p
  # Return the prompt with ' done' as a postfix, and specify the
  # temperature if it is not None.
  return str(temperature or '') + ' done'


@executing.make_executable
def generate_test_function(
    prompt: str | ChunkList,
) -> str:
  del prompt
  return ' done'


@executing.make_executable
async def chat_test_function(messages: Sequence[Message], **kwargs) -> str:
  del kwargs
  return ','.join(','.join(c.content for c in m.content) for m in messages)


class ComposablesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

  def test_composable(self):
    llm.generate_text.configure(generate_test_temp_function)
    result = executing.run('hello' + c.generate_text() + ' end')
    self.assertEqual('hello done end', result)

    result = executing.run('hello ' + c.generate_text(temperature=1.0))
    self.assertEqual('hello 1.0 done', result)

  def test_composable_stream(self):
    @executing.make_executable
    def iterator(text: str) -> Iterator[str]:
      chunks = text.split(' ')
      for i in range(1, 1 + len(chunks)):
        yield ' '.join(chunks[:i])

    @executing.make_executable
    def generate_test_stream_function(
        prompt: str | ChunkList,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
      del prompt, max_tokens, stop, top_k, top_p
      # Return the prompt with ' done' as a postfix, and specify the
      # temperature if it is not None.
      return iterator(str(temperature or '') + ' done')

    llm.generate_text.configure(generate_test_stream_function)
    result = executing.run('hello' + c.generate_text() + ' end')
    self.assertEqual('hello done end', result)

    result = executing.run('hello ' + c.generate_text(temperature=1.0))
    self.assertEqual('hello 1.0 done', result)

    with executing.safe_stream(
        'hello ' + c.generate_text(temperature=1.0), iteration_depth=-1
    ) as stream:
      result = []
      for update in stream:
        result.append(updating.Update(update).to_result())
      self.assertListEqual(['hello ', 'hello 1.0', 'hello 1.0 done'], result)

  def test_select(self):

    @executing.make_executable
    def score(
        prompt: str | ChunkList,
        targets: Sequence[str]
    ) -> str:
      del prompt
      # We score by the length of the target.
      return [float(len(target)) for target in targets]

    llm.generate_text.configure(generate_test_function)
    llm.score_text.configure(score)

    with self.subTest('gen_only'):
      result = executing.run('abc' + c.generate_text())
      self.assertEqual('abc done', result)

    with self.subTest('basic_select'):
      result = executing.run(c.select('a', 'ab', 'abc'))
      self.assertEqual('abc', result)

    with self.subTest('select_and_gen_composed'):
      result = executing.run(
          'hello' + c.generate_text() + c.select(' a', ' ab', ' abc') + ' end'
      )
      self.assertEqual('hello done abc end', result)

    with self.subTest('select_with_gen_options'):
      result = executing.run(c.select('a', 'ab', 'abc' + c.generate_text()))
      self.assertEqual('abc done', result)

    with self.subTest('select_of_selects'):
      chain1 = (
          'start '
          + c.select(
              c.select('a', 'ab') + c.generate_text(), c.select('c', 'd')
          )
          + ' end'
      )
      chain2 = (
          'start '
          + c.select(
              c.select('a', 'ab') + c.generate_text(),
              c.select('c', 'longer d'),
          )
          + ' end'
      )
      result1 = executing.run(chain1)
      result2 = executing.run(chain2)
      self.assertEqual('start ab done end', result1)
      self.assertEqual('start longer d end', result2)

  def test_generate_object(self):
    @executing.make_executable
    def generate_object_function(
        prompt: str | ChunkList,
        cls: type[_T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> _T:
      del prompt, max_tokens, stop, top_k, top_p, temperature, cls
      return [1, 2, 3]

    llm.generate_object.configure(generate_object_function)
    result = executing.run(
        'hello ' + c.generate_object(type(list[int])) + ' end'
    )
    self.assertEqual('hello [1, 2, 3] end', result)

    result = executing.run(
        'hello '
        + c.store('v', c.generate_object(type(list[int])))
        + c.f('{v[0]} end')
    )
    self.assertEqual('hello [1, 2, 3]1 end', result)

  def test_instruct(self):
    actual_prompts = []

    @executing.make_executable
    def generate_text_function(
        prompt: str | ChunkList,
        **kwargs,
    ) -> str:
      nonlocal actual_prompts
      del kwargs
      # Return the prompt with ' done' as a postfix.
      actual_prompts.append(prompt)
      return ' done'

    llm.generate_text.configure(generate_text_function)
    result = executing.run(
        'hello ' + c.instruct(assistant_prefix='world') + ' end'
    )
    self.assertEqual('hello world done end', result)
    self.assertStartsWith(actual_prompts[0], 'Task: ', actual_prompts[0])
    self.assertEndsWith(str(actual_prompts[0]), 'Answer: world')

  def test_format_strings(self):

    llm.generate_text.configure(generate_test_function)
    composable = c.f('var={var}') + c.generate_text()

    with self.subTest('raises_when_var_is_undefined'):
      with self.assertRaises(ValueError):
        _ = executing.run(composable)

    with self.subTest('string_gets_formatted'):
      res = executing.run(composable(var='value'))
      self.assertEqual('var=value done', res)

    composable = (
        c.store('v', c.f('var={var}')) + c.generate_text() + c.f(' v="{v}"')
    )
    with self.subTest('nested_store_gets_formatted'):
      res = executing.run(composable(var='value'))
      self.assertEqual('var=value done v="var=value"', res)

  @parameterized.named_parameters(
      ('bytes', [bytes(1)], 'hello <bytes>', ['hello ', bytes(1)]),
      ('string', ['test'], 'hello test', ['hello ', 'test']),
      (
          'string_generate',
          ['test', 'generate'],
          'hello test done',
          ['hello ', 'test', ' done'],
      ),
      ('chunk', [Chunk('test')], 'hello test', ['hello ', 'test']),
      (
          'chunk_list',
          [ChunkList(['test1 ', Chunk('test2')])],
          'hello test1 test2',
          ['hello ', 'test1 ', 'test2'],
      ),
  )
  def test_chunks(self, chunks, expected_result, expected_prefix):
    llm.generate_text.configure(generate_test_function)
    composable = 'hello '
    for chunk in chunks:
      if chunk == 'generate':
        composable += c.generate_text()
      else:
        composable += c.c(chunk)
    res = executing.run(composable)
    # Composable.get_result returns a string version of the ChunkList.
    with self.subTest('result'):
      self.assertEqual(expected_result, res)

    # We can also inspect the prefix from the context which contains a
    # ChunkList.
    context = composing.Context()
    composable += composing.get_context(context)
    _ = executing.run(composable)
    expected_prefix = content_lib.ChunkList(
        [content_lib.Chunk(c) for c in expected_prefix]
    )
    with self.subTest('prefix'):
      self.assertEqual(expected_prefix, context.prefix)

  def test_empty_composable(self):
    # This is how to create an empty Composable to which other content can then
    # be added.
    composable = c.c(ChunkList())
    res = executing.run(composable)
    with self.subTest('result'):
      self.assertEqual('', res)

    context = composing.Context()
    composable += composing.get_context(context)
    _ = executing.run(composable)
    with self.subTest('prefix'):
      self.assertEqual(content_lib.ChunkList(), context.prefix)

  def test_jinja(self):

    llm.generate_text.configure(generate_test_temp_function)
    with self.subTest('simple_generate'):
      composable = 'hello ' + c.j('{{var}}{{generate_text()}}') + ' end'
      res = executing.run(composable(var='value'))
      self.assertEqual('hello value done end', res)

    with self.subTest('propagate_context'):
      composable = (
          'hello ' + c.j('{{var}}{{store("res", generate_text())}}') + ' end'
      )
      res = executing.run(composable(var='value'))
      self.assertEqual('hello value done end', res)
      self.assertEqual(' done', composable['res'])

  def test_chat(self):
    llm.chat.configure(chat_test_function)
    composable = (
        c.c(
            's1',
            role=PredefinedRole.SYSTEM,
        )
        + c.c('u1')
        + c.c('u2', role=PredefinedRole.USER)
        + c.c('m1', role=PredefinedRole.MODEL)
        + c.c('u3', role=PredefinedRole.USER)
        + c.c('u4')
        + c.store('result', c.chat())
    )
    res = executing.run(composable)

    self.assertEqual('s1,u1,u2,m1,u3,u4', composable['result'])
    self.assertEqual('s1u1u2m1u3u4' + 's1,u1,u2,m1,u3,u4', res)


if __name__ == '__main__':
  absltest.main()
