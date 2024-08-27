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
import copy
import pprint

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import composing
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import updating

Chunk = content_lib.Chunk
ChunkList = content_lib.ChunkList
Message = content_lib.Message
PredefinedRole = content_lib.PredefinedRole


@composing.make_composable
def gen(context: composing.Context, continuation: str = '') -> str:
  del context
  return continuation + ' done'


@composing.make_composable
def gen_stream(
    context: composing.Context, continuation: str = ''
) -> Iterator[str]:
  del context
  words = (continuation + ' done').split(' ')
  for i in range(1, len(words) + 1):
    yield ' '.join(words[:i])


@composing.make_composable
def gen_return_stream(
    context: composing.Context, continuation: str = ''
) -> str:
  del context

  @executing.make_executable
  def stream(sentence: str) -> str:
    words = (sentence + ' done').split(' ')
    for i in range(1, len(words) + 1):
      yield ' '.join(words[:i])

  return stream(continuation)


@composing.make_composable
def gen_with_context(context: composing.Context, continuation: str = '') -> str:
  return continuation + context['context'] + ' done'


@composing.make_composable
def check_prefix(
    context: composing.Context,
) -> str:
  return f'[{context.prefix}] done'


@composing.make_join_composable
def join(
    context: composing.Context,
    options: Sequence[tuple[ChunkList, composing.Context]],
) -> tuple[str, composing.Context]:
  del context
  # Join the strings from all the branches.
  result = ','.join([str(p) for p, _ in options])
  # The context we select is the one from the first branch.
  final_context = options[0][1]
  return result, final_context


def _collect_stream(
    executable: executing.Executable, depth: int = 1
) -> list[str]:
  res = []
  with executing.safe_stream(executable, iteration_depth=depth) as stream:
    updates = updating.Update[str]()
    for update in stream:
      updates += update
      res.append(updates.to_result())
  return res


class ComposingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('str_object', 'test' + gen(), 'test done'),
      ('object_str', gen() + ' test', ' done test'),
      ('str_object_str', 'test' + gen() + ' end', 'test done end'),
      (
          'str_object_str_with_continuation',
          'test' + gen(' other') + ' end',
          'test other done end',
      ),
      ('object_object', gen() + gen(), ' done done'),
      ('prepend_str_to_chain', 'test' + (gen() + ' end'), 'test done end'),
  )
  def test_composing_run(self, executable, expected):
    res = executing.run(executable)
    self.assertEqual(res, expected)

  @parameterized.named_parameters(
      ('str_object', 'test' + gen(), ['test', 'test done']),
      ('object_str', gen() + ' test', [' done', ' done test']),
      (
          'str_object_str',
          'test' + gen() + ' end',
          ['test', 'test done', 'test done end'],
      ),
  )
  def test_composing_stream(self, executable, expected):
    self.assertEqual(_collect_stream(executable), expected)

  def test_make_composable_decoration(self):
    """Test make_composable decoration on top of make_executable."""
    # The following different ways of decorating should lead to the same final
    # result.
    # Indeed, 'context' will be automatically added by make_composable.
    # Also if the function is not decorated with make_executable, then
    # this will be automatically added.
    @composing.make_composable
    @executing.make_executable('continuation')
    def f(context: composing.Context, continuation: str = '') -> str:
      del context
      return continuation + ' done1'

    @composing.make_composable
    @executing.make_executable('context', 'continuation')
    def f2(context: composing.Context, continuation: str = '') -> str:
      del context
      return continuation + ' done2'

    @composing.make_composable('continuation')
    @executing.make_executable('context')
    def f3(context: composing.Context, continuation: str = '') -> str:
      del context
      return continuation + ' done3'

    @composing.make_composable('continuation')
    def f4(context: composing.Context, continuation: str = '') -> str:
      del context
      return continuation + ' done4'

    with self.subTest('non_exec_args_properly_set'):
      expected = {'context', 'continuation'}
      for fn in [f, f2, f3, f4]:
        self.assertSetEqual(
            expected, set(fn('a').nodes[0].executable.non_executed_args)
        )

    with self.subTest('correctly_executes'):
      chain = sum([fn(' a') for fn in [f2, f3, f4]], start=f(' a'))
      result = executing.run(chain)
      self.assertEqual(result, ' a done1 a done2 a done3 a done4')

  def test_composing_start_and_context(self):
    with self.subTest('chain_start_simple'):
      a = gen('a')
      b = gen(' b')
      c = gen(' c')
      chain = a + b + c
      self.assertEqual(executing.run(chain), 'a done b done c done')

    with self.subTest('chain_start_with_strings'):
      a = gen('a')
      b = gen('b')
      c = gen('c')
      chain = a + ', ' + b + ', ' + c
      self.assertEqual(executing.run(chain), 'a done, b done, c done')

    with self.subTest('chain_start_parenthesized'):
      # We test that addition is associative, i.e. no matter what the
      # parentheses are, we get a linear chain.
      a = gen('a')
      b = gen(' b')
      c = gen(' c')
      d = gen(' d')
      chain = a + ((b + c) + d)
      self.assertEqual(executing.run(chain), 'a done b done c done d done')

    with self.subTest('chain_context_simple'):
      a = gen('a')
      b = gen_with_context(' b ')
      c = gen(' c')
      context = composing.Context({'context': 'context'})
      init = composing.set_context(context)
      chain = init + a + b + c
      # The context should be propagated from init to b.
      self.assertEqual(executing.run(chain), 'a done b context done c done')

    with self.subTest('chain_context_with_strings'):
      a = gen('a')
      b = gen_with_context('b ')
      c = gen('c')
      context = composing.Context({'context': 'context'})
      chain = composing.set_context(context) + a + ', ' + b + ', ' + c
      # The context should be propagated from init to b even with strings in
      # the middle.
      self.assertEqual(executing.run(chain), 'a done, b context done, c done')

    with self.subTest('chain_context_parenthesized'):
      a = gen('a')
      b = gen_with_context(' b ')
      c = gen(' c')
      d = gen(' d')
      e = gen_with_context(' e ')
      f = gen_with_context(' f ')
      context = composing.Context({'context': 'context'})
      chain = composing.set_context(context) + a + ((b + c) + (d + e)) + f
      # The context should be correctly propagated despite the parentheses.
      self.assertEqual(
          executing.run(chain),
          'a done b context done c done d done e context done f context done',
      )

  @parameterized.named_parameters(
      (
          'gen_stream',
          gen_stream('a b c'),
          -1,
          ['a', 'a b', 'a b c', 'a b c done'],
      ),
      (
          'gen_return_stream',
          gen_return_stream('a b c'),
          -1,
          ['a', 'a b', 'a b c', 'a b c done'],
      ),
      (
          'gen_return_stream_shallow',
          gen_return_stream('a b c'),
          0,
          ['a b c done'],
      ),
      (
          'gen_return_stream_in_chain',
          'a ' + gen_return_stream('b c') + ' d ' + gen_return_stream('e f'),
          -1,
          [
              'a ',
              'a b',
              'a b c',
              'a b c done',
              'a b c done d ',
              'a b c done d e',
              'a b c done d e f',
              'a b c done d e f done',
          ],
      ),
  )
  def test_gen_stream(self, executable, depth, expected):
    with self.subTest('iterations_are_correct'):
      result = _collect_stream(executable, depth)
      self.assertListEqual(result, expected, pprint.pformat(result))

    with self.subTest('run_result_is_correct'):
      result = executing.run(executable)
      self.assertEqual(result, expected[-1], pprint.pformat(result))

  def test_re_execute(self):
    with self.subTest('run_twice'):
      executable = 'a' + gen() + ' b ' + gen()
      res = executing.run(executable)
      # The result is stored in the executable.
      self.assertEqual(
          executable._result, ChunkList(['a', ' done', ' b ', ' done'])
      )
      res2 = executing.run(executable)
      # We get the same result if we re-execute.
      self.assertEqual(res, res2)

    with self.subTest('stream_twice'):
      executable = 'a' + gen() + ' b ' + gen()
      res = _collect_stream(executable)
      # The stored result is the last element of the stream.
      self.assertEqual(
          executable._result, ChunkList(['a', ' done', ' b ', ' done'])
      )
      res2 = _collect_stream(executable)
      # Upon re-execution we directly get the same result.
      self.assertListEqual(res2, res)

  def test_re_execute_with_context(self):
    with self.subTest('run_twice'):
      executable = (
          'a' + composing.store('a', 'v') + '=' + composing.get_var('a')
      )
      res = executing.run(executable)
      # The result is stored in the executable.
      self.assertEqual(executable._result, ChunkList(['a', 'v', '=', 'v']))
      res2 = executing.run(executable)
      # We get the same result if we re-execute.
      self.assertEqual(res, res2)

    with self.subTest('stream_twice'):
      executable = (
          'a' + composing.store('a', 'v') + '=' + composing.get_var('a')
      )
      res = _collect_stream(executable)
      # The stored result is the last element of the stream.
      self.assertEqual(executable._result, ChunkList(['a', 'v', '=', 'v']))
      res2 = _collect_stream(executable)
      # Upon re-execution we directly get the same result.
      self.assertListEqual(res2, res)

  def test_stream_with_parentheses(self):
    with self.subTest('chain_context_parenthesized'):
      a = gen('a')
      b = ' b '
      c = gen(' c')
      d = gen(' d')
      e = ' e '
      f = gen(' f ')
      chain = a + ((b + c) + (d + e)) + f
      self.assertListEqual(
          _collect_stream(chain),
          [
              'a done',
              'a done b ',
              'a done b  c done',
              'a done b  c done d done',
              'a done b  c done d done e ',
              'a done b  c done d done e  f  done',
          ],
      )

    with self.subTest('chain_context_parenthesized'):
      a = gen('a')
      b = gen_with_context(' b ')
      c = gen(' c')
      d = gen(' d')
      e = gen_with_context(' e ')
      f = gen_with_context(' f ')
      context = composing.Context({'context': 'context'})
      chain = composing.set_context(context) + a + ((b + c) + (d + e)) + f
      # The context should be correctly propagated despite the parentheses.
      self.assertListEqual(
          _collect_stream(chain),
          [
              'a done',
              'a done b context done',
              'a done b context done c done',
              'a done b context done c done d done',
              'a done b context done c done d done e context done',
              (
                  'a done b context done c done d done e context done f context'
                  ' done'
              ),
          ],
      )

  def test_chaining_nested_functions_with_context(self):
    """Test chaining nested functions.

    In particular, if a Composable returns a Composable, the execution should
    further execute the returned Composable after having attached it to the
    correct prefix.
    """
    @composing.make_composable
    @executing.make_executable
    def fn(context: composing.Context) -> str:
      return 'read:' + context['content'] + ' '

    @composing.make_composable
    @executing.make_executable
    def fn2(
        context: composing.Context, b: bool, value: str
    ) -> str:
      if b:
        context['content'] = value
        return fn()
      else:
        context['content'] = value
        return composing.set_context(context) + fn()

    context = {'content': 'init'}
    a = composing.set_context(context) + fn()
    self.assertEqual(executing.run(a), 'read:init ')

    context = {'content': 'init'}
    a = composing.set_context(context) + fn2(True, 'test')
    self.assertEqual(executing.run(a), 'read:test ')

    context = {'content': 'init'}
    a = composing.set_context(context) + fn2(False, 'test')
    self.assertEqual(executing.run(a), 'read:test ')

    context = {'content': 'init'}
    a = (
        composing.set_context(context)
        + fn2(False, 'test1')
        + 'other '
        + fn()
        + fn2(False, 'test2')
        + fn()
    )
    self.assertEqual(
        executing.run(a), 'read:test1 other read:test1 read:test2 read:test2 '
    )

    context = {'content': 'init'}
    a = (
        composing.set_context(context)
        + fn2(True, 'test1')
        + 'other '
        + fn()
        + fn2(True, 'test2')
        + fn()
    )
    self.assertEqual(
        executing.run(a), 'read:test1 other read:test1 read:test2 read:test2 '
    )

  def test_chaining_nested_functions(self):
    """Test chaining nested functions.

    In particular, if a Composable returns a Composable, the execution should
    further execute the returned Composable after having attached it to the
    correct prefix.
    """
    @composing.make_composable
    @executing.make_executable
    def fn(context: composing.Context, b: bool) -> str:
      if b:
        return f'[{context.prefix}]'
      else:
        return gen() + ' end fn'

    @composing.make_composable
    @executing.make_executable
    def fn2(context: composing.Context, b: bool) -> str:
      del context
      if b:
        return fn(True)
      else:
        return gen() + fn(True) + ' end fn2'

    a = fn(True)  # Equivalent to: a = f'[{prefix}]'
    self.assertEqual(executing.run(a), '[]')
    a = 'hello' + fn(True)  # Equivalent to: a = 'hello' + f'[{prefix}]'
    self.assertEqual(executing.run(a), 'hello[hello]')
    a = fn(False)  # Equivalent to: a = gen() + ' end fn'
    self.assertEqual(executing.run(a), ' done end fn')
    a = fn(False) + ' end'  # Equivalent to: a = gen() + ' end fn' + ' end'
    self.assertEqual(executing.run(a), ' done end fn end')

    a = fn2(True)  # Equivalent to: a = f'[{prefix}]'
    self.assertEqual(executing.run(a), '[]')
    a = 'hello' + fn2(True)  # Equivalent to: a = 'hello' + f'[{prefix}]'
    self.assertEqual(executing.run(a), 'hello[hello]')
    a = fn2(False)  # Equivalent to: a = gen() + f'[{prefix}]' + ' end fn2'
    self.assertEqual(executing.run(a), ' done[ done] end fn2')

    a = fn(True) + fn2(False)
    # Equivalent to:
    # a = f'[{prefix}]' + gen() + f'[{prefix}]' + ' end fn2'
    self.assertEqual(executing.run(a), '[] done[[] done] end fn2')
    a = fn(False) + fn2(True)
    # Equivalent to:
    # a = gen() + ' end fn' + f'[{prefix}]'
    self.assertEqual(executing.run(a), ' done end fn[ done end fn]')

  def test_copying(self):
    executable = 'test' + gen() + ' end'
    other = copy.deepcopy(executable)
    res = executing.run(executable)
    res2 = executing.run(other)
    self.assertEqual(res, res2)

  def test_parallel_processing(self):
    executable = 'common' + join(
        ' test1' + gen(),
        gen(),
        gen() + ' end',
    )
    result = executing.run(executable)
    self.assertEqual(result, 'common test1 done, done, done end')

  def test_store(self):
    saved_context = composing.Context()
    result = executing.run(
        'test'
        + composing.section_start('var')
        + ' end'
        + composing.get_context(saved_context)
    )
    self.assertEqual(result, 'test end')
    self.assertEqual(
        saved_context._variables,
        {
            '_sections': [
                composing.SectionInfo('var', 1, ChunkList(['test']), False)
            ]
        },
    )
    saved_context = composing.Context()
    result = executing.run(
        'test'
        + composing.section_start('var')
        + gen()
        + composing.section_end()
        + composing.get_context(saved_context)
    )
    self.assertEqual(result, 'test done')
    self.assertEqual(
        saved_context._variables, {'_sections': [], 'var': ChunkList([' done'])}
    )
    saved_context = composing.Context()
    result = executing.run(
        'test'
        + (composing.section_start('var') + gen() + composing.section_end())
        + composing.get_context(saved_context)
    )
    self.assertEqual(result, 'test done')
    self.assertEqual(
        saved_context._variables, {'_sections': [], 'var': ChunkList([' done'])}
    )
    saved_context = composing.Context()
    executable = (
        'test'
        + composing.store('var', gen())
        + ' end'
        + composing.get_context(saved_context)
    )
    result = executing.run(executable)
    # We store the string value of the ChunkList!
    self.assertEqual(saved_context._variables, {'var': ' done'})
    self.assertEqual(result, 'test done end')
    executable = (
        'test'
        + composing.store('var', gen())
        + ' end'
        + composing.get_var('var')
    )
    result = executing.run(executable)
    with self.subTest('store_and_retrieve'):
      self.assertEqual(result, 'test done end done')

  @parameterized.named_parameters(
      ('hidden_section', True, 'test end'),
      ('not_hidden_section', False, 'test done end'),
  )
  def test_hidden_section(self, hidden: bool, expected: str):
    composable = (
        'test'
        + composing.section_start('var', hidden=hidden)
        + gen()
        + composing.section_end()
        + ' end'
    )
    result = executing.run(composable)
    self.assertEqual(result, expected)

  def test_section_context_manager(self):
    composable = composing.Composable()
    composable += 'test '
    with composable.section('v'):
      composable += 'content'

    with self.subTest('created_correct_section'):
      result = executing.run(composable)
      self.assertEqual(result, 'test content')
      self.assertEqual(composable['v'], ChunkList(['content']))

  def test_nested_store(self):
    saved_context = composing.Context()
    executable = (
        'first '
        + composing.store(
            'v3',
            composing.store('v1', check_prefix())
            + ' second '
            + composing.store('v2', 'test ' + check_prefix() + ' end'),
        )
        + ' third '
        + composing.store('v4', 'test')
        + composing.get_context(saved_context)
    )
    result = executing.run(executable)
    with self.subTest('prefix_gets_propagated_correctly'):
      self.assertEqual(
          result,
          'first [first ] done second test [first [first ] done second test ]'
          ' done end third test',
      )

    with self.subTest('context_gets_built_correctly'):
      self.assertDictEqual(
          saved_context._variables,
          {
              'v1': '[first ] done',
              'v2': 'test [first [first ] done second test ] done end',
              'v3': (
                  '[first ] done second test [first [first ] done second test ]'
                  ' done end'
              ),
              'v4': 'test',
          },
      )

  def test_decorate_with_iterate_argument(self):
    @composing.make_composable
    @executing.make_executable(iterate_argument='other')
    def fn(context: composing.Context) -> str:
      del context
      return ''

    with self.assertRaises(ValueError):
      executing.run(fn())

  def test_call_with_context(self):
    composable = 'var=' + composing.get_var('var')

    with self.subTest('raises_when_var_is_undefined'):
      with self.assertRaises(ValueError):
        _ = executing.run(composable)

    with self.subTest('context_gets_propagated'):
      res = executing.run(composable(var='value'))
      self.assertEqual(res, 'var=value')

  def test_getitem(self):
    extracted_context = composing.Context()
    c_get_context = (
        'v='
        + composing.store('v', 'value')
        + composing.get_context(extracted_context)
    )

    with self.subTest('context_gets_stored'):
      _ = executing.run(c_get_context)
      expected = composing.Context(
          {'v': 'value'}, ChunkList([Chunk('v='), Chunk('value')])
      )
      self.assertEqual(extracted_context, expected)
      self.assertEqual(c_get_context._final_context, expected)

    c = 'v=' + composing.store('v', 'value')

    with self.subTest('raises_when_not_executed'):
      with self.assertRaises(ValueError):
        _ = c['v']

    with self.subTest('get_item_returns_correct_value'):
      res = executing.run(c)
      self.assertEqual(c['v'], 'value')
      self.assertEqual(res, 'v=value')

    with self.subTest('raises_if_key_not_found'):
      with self.assertRaises(ValueError):
        _ = c['w']

  @parameterized.named_parameters(
      dict(
          testcase_name='no_explicit_roles',
          chunks=[
              Chunk('c1'),
              Chunk('c2'),
              Chunk('c3'),
          ],
          expected_messages=[
              Message(
                  PredefinedRole.USER,
                  ChunkList([
                      Chunk('c1', role=PredefinedRole.USER),
                      Chunk('c2', role=PredefinedRole.USER),
                      Chunk('c3', role=PredefinedRole.USER),
                  ]),
              )
          ],
      ),
      dict(
          testcase_name='with_explicit_roles',
          chunks=[
              Chunk('s1', role=PredefinedRole.SYSTEM),
              Chunk('u2', role=PredefinedRole.USER),
              Chunk('u3'),
              Chunk('m1', role=PredefinedRole.MODEL),
              Chunk('u4'),
              Chunk('u5', role=PredefinedRole.USER),
              Chunk('t1', role='thoughts'),
              Chunk('t2', role='thoughts'),
              Chunk('m2', role=PredefinedRole.MODEL),
          ],
          expected_messages=[
              Message(
                  PredefinedRole.SYSTEM,
                  ChunkList([
                      Chunk('s1', role=PredefinedRole.SYSTEM),
                  ]),
              ),
              Message(
                  PredefinedRole.USER,
                  ChunkList([
                      Chunk('u2', role=PredefinedRole.USER),
                      Chunk('u3', role=PredefinedRole.USER),
                  ]),
              ),
              Message(
                  PredefinedRole.MODEL,
                  ChunkList([
                      Chunk('m1', role=PredefinedRole.MODEL),
                  ]),
              ),
              Message(
                  PredefinedRole.USER,
                  ChunkList([
                      Chunk('u4', role=PredefinedRole.USER),
                      Chunk('u5', role=PredefinedRole.USER),
                  ]),
              ),
              Message(
                  'thoughts',
                  ChunkList([
                      Chunk('t1', role='thoughts'),
                      Chunk('t2', role='thoughts'),
                  ]),
              ),
              Message(
                  PredefinedRole.MODEL,
                  ChunkList([
                      Chunk('m2', role=PredefinedRole.MODEL),
                  ]),
              ),
          ],
      ),
  )
  def test_context_to_messages(self, chunks, expected_messages):
    ctx = composing.Context(prefix=ChunkList(chunks))
    self.assertListEqual(ctx.to_messages(), expected_messages)


if __name__ == '__main__':
  absltest.main()
