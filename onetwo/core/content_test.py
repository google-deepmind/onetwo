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

"""Unit tests for onetwo.core.content.

Note that Chunk and ChunkList are both decorated with `dataclass`. By default it
adds `__eq__` method which is based on comparing all of the attributes. For a
class `MyClass` decorated with `dataclass` assertEqual(MyClass(1), MyClass(1))
passes.
"""

from typing import Any, TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import content as content_lib
import PIL.Image

_Chunk: TypeAlias = content_lib.Chunk
_ChunkList: TypeAlias = content_lib.ChunkList


class ContentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('wrong_content_type_arg', 'test', 'bytes', True, None),
      ('unsupported_type', ['test'], None, True, None),
      ('supported_type_no_content_type', 'test', None, False, 'str'),
      ('supported_type_no_content_type_bytes', b'test', None, False, 'bytes'),
      ('supported_type_content_type', 'test', 'str', False, 'str'),
      ('wrong_prefix_str', b'test', 'str', True, None),
      ('wrong_prefix_ctrl', b'test', 'ctrl', True, None),
      ('correct_prefix_image', b'test', 'image/jpeg', False, 'image/jpeg'),
      ('correct_prefix_str', 'test', 'str', False, 'str'),
      ('correct_prefix_ctrl', 'test', 'ctrl', False, 'ctrl'),
      (
          'correct_prefix_pil',
          PIL.Image.Image(),
          'image/jpeg',
          False,
          'image/jpeg',
      ),
  )
  def test_chunk_creation_errors(
      self,
      content: Any,
      content_type: str,
      raises: bool,
      expected_content_type: str | None,
  ):
    if raises:
      with self.assertRaises(ValueError):
        if content_type is not None:
          _ = _Chunk(content, content_type)
        else:
          _ = _Chunk(content)
    else:
      if content_type is not None:
        c = _Chunk(content, content_type)
      else:
        c = _Chunk(content)
      self.assertEqual(c.content_type, expected_content_type)

  @parameterized.named_parameters(
      ('string_content', 'string test', 'str'),
      ('bytes_content', b'bytes test', 'bytes'),
      ('pil_image_content', PIL.Image.Image(), 'image/jpeg'),
  )
  def test_chunk_creation_with_unset_content_type(
      self,
      content: Any,
      expected_content_type: str | None,
  ):
    c = _Chunk(content)
    self.assertEqual(c.content_type, expected_content_type)

  def test_chunk_creation_with_unaccepted_content(
      self,
  ):
    with self.assertRaises(
        ValueError,
        msg=(
            "Creating a Chunk with content of type <class 'list'> which is not"
            ' one of the accepted types (str, bytes, PIL.Image.Image).'
        ),
    ):
      content: Any = ['list type content']
      _ = _Chunk(content)

  @parameterized.named_parameters(
      ('text_csv_content_type', 'test', 'text/csv'),
      ('text_css_content_type', b'test', 'text/css'),
      ('font_woff_content_type', PIL.Image.Image(), 'font/woff'),
      ('model_vrml_content_type', 'test', 'model/vrml'),
  )
  def test_chunk_creation_with_unknown_content_type(
      self,
      content: Any,
      content_type: str,
  ):
    with self.assertLogs() as logs:
      _ = _Chunk(content, content_type)
      self.assertEqual(
          logs.records[0].getMessage(),
          f'Creating a Chunk with unknown content_type: {content_type}. This'
          ' might cause errors if the type of the content is not compatible'
          ' with the provided content_type.',
      )

  @parameterized.named_parameters(
      ('string_content_bytes_content_type', 'test', 'bytes', 'str'),
      (
          'string_content_bytes_application_content_type',
          'test',
          'application/pdf',
          'str',
      ),
      ('string_content_image_content_type', 'test', 'image/jpeg', 'str'),
      ('bytes_content_str_content_type', b'test', 'ctrl', 'bytes'),
      (
          'image_content_image_content_type',
          PIL.Image.Image(),
          'str',
          'pil_image',
      ),
      (
          'image_content_bytes_video_content_type',
          PIL.Image.Image(),
          'video/mp4',
          'pil_image',
      ),
  )
  def test_chunk_creation_with_content_type_not_matching_content(
      self,
      content: Any,
      content_type: str,
      python_type: str,
  ):
    accepted_prefixes = ['str', 'ctrl']
    if python_type == 'bytes':
      accepted_prefixes = [
          'bytes',
          'image/',
          'video/',
          'audio/',
          'application/',
          'vision/',
      ]
    elif python_type == 'pil_image':
      accepted_prefixes = ['image/']

    with self.assertRaises(
        ValueError,
        msg=(
            f'Creating a Chunk with content of type {python_type} but'
            f' content_type is set to {content_type} which is not'
            f' compatible (accepted prefixes are {accepted_prefixes}).'
        ),
    ):
      _ = _Chunk(content, content_type)

  def test_chunk_list_add(self):
    c = _Chunk('test')
    l = _ChunkList()
    with self.subTest('add_chunk_to_chunk_list'):
      l += c
      self.assertEqual(l.chunks, [c])

    with self.subTest('add_chunk_list_to_chunk'):
      l = c + _ChunkList()
      self.assertEqual(l.chunks, [c])

    with self.subTest('add_chunk_list_to_chunk_list'):
      l += l
      self.assertEqual(l.chunks, [c, c])

    l = _ChunkList()
    l += 'hello '

    with self.subTest('add_string_to_chunk_list'):
      self.assertEqual(l.chunks, [_Chunk('hello ')])

    l = _ChunkList()
    l = 'hello ' + l

    with self.subTest('add_chunk_list_to_str'):
      self.assertEqual(l.chunks, [_Chunk('hello ')])

  def test_chunk_and_chunk_list_evaluates_to_true(self):
    with self.subTest('chunk_with_empty_content_evals_to_false'):
      self.assertFalse(_Chunk(''))
      self.assertFalse(_Chunk(b''))

    with self.subTest('chunk_with_non_empty_content_evals_to_true'):
      self.assertTrue(_Chunk('abc'))
      self.assertTrue(_Chunk(b'abc'))

    with self.subTest('chunk_list_with_empty_chunks_evals_to_false'):
      self.assertFalse(_ChunkList(chunks=[]))
      self.assertFalse(_ChunkList(chunks=[_Chunk(''), _Chunk(b''), _Chunk('')]))

    with self.subTest('chunk_list_with_non_empty_chunk_evals_to_true'):
      self.assertTrue(_ChunkList(chunks=[_Chunk(''), _Chunk(b'0'), _Chunk('')]))

  def test_chunk_and_chunk_list_str_functions(self):
    chunk = _Chunk('abccbbabbctest')

    with self.subTest('chunk_lstrip_works'):
      self.assertEqual(chunk.lstrip('abc'), _Chunk('test'))
      self.assertEqual(chunk.lstrip(' '), chunk)

    with self.subTest('chunk_rstrip_works'):
      self.assertEqual(_Chunk('testabbcbc').rstrip('abc'), _Chunk('test'))
      self.assertEqual(_Chunk('testabbcbc').rstrip(' '), _Chunk('testabbcbc'))

    with self.subTest('chunk_rstrip_does_not_touch_ctrl'):
      self.assertEqual(
          _Chunk('testabbcbc', content_type='ctrl').rstrip('abc'),
          _Chunk('testabbcbc', content_type='ctrl'),
      )

    with self.subTest('chunk_lstrip_does_not_touch_ctrl'):
      self.assertEqual(
          _Chunk('abbcbc', content_type='ctrl').lstrip('abc'),
          _Chunk('abbcbc', content_type='ctrl'),
      )

    with self.subTest('chunk_strip_works'):
      self.assertEqual(_Chunk('abbcbctestabbcbc').strip('abc'), _Chunk('test'))
      self.assertEqual(
          _Chunk('abbcbctestabbcbc').strip(' '), _Chunk('abbcbctestabbcbc')
      )

    with self.subTest('chunk_strip_does_not_touch_ctrl'):
      self.assertEqual(
          _Chunk('abbcbctestabbcbc', content_type='ctrl').strip('abc'),
          _Chunk('abbcbctestabbcbc', content_type='ctrl'),
      )

    chunk = _Chunk('abccbbabbctest')
    with self.subTest('chunk_startswith_works'):
      self.assertTrue(chunk.startswith('abc'))
      self.assertTrue(chunk.startswith('bc', 1))
      self.assertTrue(chunk.startswith('b', 1, 2))
      self.assertFalse(chunk.startswith('123'))

    with self.subTest('chunk_endswith_works'):
      self.assertTrue(chunk.endswith('test'))
      self.assertTrue(chunk.endswith('test', 10))
      self.assertFalse(chunk.endswith('test', 11))
      self.assertFalse(chunk.endswith('test', 10, 13))
      self.assertTrue(chunk.endswith('test', 10, 14))

    with self.subTest('chunk_list_startswith_works'):
      chunk_list = _ChunkList(chunks=[_Chunk('abc'), '12', b'13'])
      self.assertTrue(chunk_list.startswith('abc'))
      self.assertTrue(chunk_list.startswith('abc12'))
      self.assertTrue(chunk_list.startswith('c1', 2, 4))
      self.assertFalse(chunk_list.startswith('abc', 1))
      self.assertFalse(chunk_list.startswith('bc'))

    with self.subTest('chunk_list_endswith_works'):
      # 'abc<bytes>12'.
      chunk_list = _ChunkList(chunks=[_Chunk('abc'), b'13', '12'])
      self.assertTrue(chunk_list.endswith('12'))
      self.assertTrue(chunk_list.endswith('<bytes>12'))
      self.assertFalse(chunk_list.endswith('1312'))
      self.assertTrue(chunk_list.endswith('1', 10, 11))

  @parameterized.named_parameters(
      (
          'strip_empty',
          _ChunkList([]),
          'abc',
          _ChunkList([]),
      ),
      (
          'strip_wo_prop_no_empty',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
          'abc',
          _ChunkList(chunks=['12', b'13', _Chunk('test')]),
      ),
      (
          'no_strip_no_empty',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
          ' ',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
      ),
      (
          'no_strip_remove_empty',
          _ChunkList(chunks=['12', b'13', '', b'', PIL.Image.Image()]),
          'a',
          _ChunkList(chunks=['12', b'13']),
      ),
      (
          'no_strip_all_empty',
          _ChunkList(chunks=['', b'', '', b'', PIL.Image.Image()]),
          'a',
          _ChunkList(chunks=[]),
      ),
      (
          'strip_wo_prop_remove_empty',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc'), '', b'']),
          'abc',
          _ChunkList(chunks=['12', b'13', _Chunk('test')]),
      ),
      (
          'strip_all_wo_prop_remove_empty',
          _ChunkList(chunks=['abc', '', b'']),
          'abc',
          _ChunkList([]),
      ),
      (
          'strip_all_wo_prop_no_empty',
          _ChunkList(chunks=['abc']),
          'abc',
          _ChunkList([]),
      ),
      (
          'strip_all_with_prop_remove_empty',
          _ChunkList(chunks=['abc', 'bc', 'c', '', '', b'']),
          'abc',
          _ChunkList([]),
      ),
      (
          'strip_all_with_prop_no_empty',
          _ChunkList(chunks=['abc', 'bc', 'c']),
          'abc',
          _ChunkList([]),
      ),
      (
          'strip_with_prop_remove_empty',
          _ChunkList(chunks=['dcba', 'abc', 'a', '', '', b'']),
          'abc',
          _ChunkList(chunks=['d']),
      ),
      (
          'strip_with_prop_no_empty',
          _ChunkList(chunks=['dcba', 'abc', 'a']),
          'abc',
          _ChunkList(chunks=['d']),
      ),
      (
          'strip_with_prop_empty_interleaved',
          _ChunkList(chunks=['dcba', 'abc', 'a', '', 'b', b'', 'a']),
          'abc',
          _ChunkList(chunks=['d']),
      ),
  )
  def test_chunk_list_rstrip(self, chunk_list, rstrip_arg, expected):
    self.assertEqual(chunk_list.rstrip(rstrip_arg), expected)

  @parameterized.named_parameters(
      (
          'simple_strip_no_empty',
          _ChunkList(chunks=[_Chunk('abbcbctest'), '12', b'13']),
          'abc',
          _ChunkList(chunks=[_Chunk('test'), '12', b'13']),
      ),
      (
          'no_strip_no_empty',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
          'a',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
      ),
      (
          'no_strip_remove_empty',
          _ChunkList(chunks=['', b'', PIL.Image.Image(), '12', b'13']),
          'a',
          _ChunkList(chunks=['12', b'13']),
      ),
      (
          'no_strip_remove_all_empty',
          _ChunkList(chunks=['', b'', '', b'', PIL.Image.Image()]),
          'a',
          _ChunkList(chunks=[]),
      ),
      (
          'strip_remove_empty',
          _ChunkList(chunks=['', b'', _Chunk('abbcbctest'), '12', b'13']),
          'abc',
          _ChunkList(chunks=[_Chunk('test'), '12', b'13']),
      ),
      (
          'strip_remove_empty_dont_propagate_to_next',
          _ChunkList(chunks=['', b'', 'abc', 'abcd']),
          'abc',
          _ChunkList(chunks=['abcd']),
      ),
  )
  def test_chunk_list_lstrip(self, chunk_list, lstrip_arg, expected):
    self.assertEqual(chunk_list.lstrip(lstrip_arg), expected)

  @parameterized.named_parameters(
      (
          'simple_strip_no_empty',
          _ChunkList(
              chunks=[_Chunk('abbcbctest'), '12', b'13', _Chunk('testabbcbc')]
          ),
          'abc',
          _ChunkList(chunks=[_Chunk('test'), '12', b'13', _Chunk('test')]),
      ),
      (
          'no_strip_no_empty',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
          'a',
          _ChunkList(chunks=['12', b'13', _Chunk('testabbcbc')]),
      ),
      (
          'no_strip_remove_empty',
          _ChunkList(chunks=['', b'', PIL.Image.Image(), '12', b'13', '', b'']),
          'a',
          _ChunkList(chunks=['12', b'13']),
      ),
      (
          'no_strip_remove_all_empty',
          _ChunkList(chunks=['', b'', '', b'', PIL.Image.Image()]),
          'a',
          _ChunkList(chunks=[]),
      ),
      (
          'strip_remove_empty',
          _ChunkList(chunks=['', b'', _Chunk('abbcbctest'), '12', b'13']),
          'abc',
          _ChunkList(chunks=[_Chunk('test'), '12', b'13']),
      ),
      (
          'strip_remove_empty_dont_propagate_to_next',
          _ChunkList(chunks=['', b'', 'abc', 'abcd']),
          'abc',
          _ChunkList(chunks=['abcd']),
      ),
  )
  def test_chunk_list_strip(self, chunk_list, strip_arg, expected):
    self.assertEqual(chunk_list.strip(strip_arg), expected)

  def test_chunk_list_to_str(self):
    l = _ChunkList()
    l += 'hello '
    l += _Chunk('world')
    l += _Chunk(b'123')
    l += _Chunk('<ctrl>', content_type='ctrl')
    l += _Chunk(PIL.Image.Image())
    self.assertEqual(str(l), 'hello world<bytes><ctrl><image/jpeg>')

  def test_chunk_list_to_simple_string(self):
    l = _ChunkList()
    l += 'hello '
    l += _Chunk('world')
    l += _Chunk(b'123')
    l += _Chunk('<ctrl>', content_type='ctrl')
    l += _Chunk(PIL.Image.Image())
    l += _Chunk(' done')
    self.assertEqual(l.to_simple_string(), 'hello world<ctrl> done')

  @parameterized.named_parameters(
      ('empty_str', '', True),
      ('empty_bytes', b'', True),
      ('empty_pil', PIL.Image.Image(), True),
      ('str', 'abc', False),
      ('bytes', b'123', False),
      ('pil', PIL.Image.new(mode='RGB', size=(2, 2)), False),
  )
  def test_chunk_is_empty(self, chunk_content, expected):
    chunk = _Chunk(chunk_content)
    self.assertEqual(chunk.is_empty(), expected)


class MessageTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('string_should_stay_unchanged', 'a', 'a'),
      (
          'chunk_list_with_single_string_chunk_should_be_normalized',
          _ChunkList(['a']),
          'a',
      ),
      (
          'chunk_list_containing_metadata_should_not_be_normalized',
          _ChunkList([_Chunk('a', metadata=[{'b': 'c'}])]),
          _ChunkList([_Chunk('a', metadata=[{'b': 'c'}])]),
      ),
      (
          'chunk_list_with_multiple_chunks_should_not_be_normalized',
          _ChunkList(['a', 'b']),
          _ChunkList(['a', 'b']),
      ),
      (
          'chunk_list_with_non_string_chunk_should_not_be_normalized',
          _ChunkList([b'123']),
          _ChunkList([b'123']),
      ),
      (
          'list_of_chunks_with_single_string_chunk_should_be_normalized',
          [_Chunk('a')],
          'a',
      ),
      (
          'list_of_chunks_containing_metadata_should_not_be_normalized',
          [_Chunk('a', metadata=[{'b': 'c'}])],
          _ChunkList([_Chunk('a', metadata=[{'b': 'c'}])]),
      ),
      (
          'list_of_chunks_with_multiple_chunks_should_not_be_normalized',
          [_Chunk('a'), _Chunk('b')],
          _ChunkList(['a', 'b']),
      ),
      (
          'list_of_chunks_with_non_string_chunk_should_not_be_normalized',
          [_Chunk(b'123')],
          _ChunkList([b'123']),
      ),
  )
  def test_create_normalized(self, content, expected_message_content):
    role = content_lib.PredefinedRole.USER
    message = content_lib.Message.create_normalized(role, content)
    self.assertEqual(expected_message_content, message.content, message.content)

  @parameterized.named_parameters(
      (
          'string',
          'a',
          _ChunkList(['a']),
      ),
      (
          'chunk_list',
          _ChunkList(['a', 'b']),
          _ChunkList(['a', 'b']),
      ),
  )
  def test_get_chunk_list(self, message_content, expected_chunk_list):
    role = content_lib.PredefinedRole.USER
    message = content_lib.Message(role, message_content)
    chunk_list = message.get_chunk_list()
    self.assertEqual(expected_chunk_list, chunk_list, chunk_list)


if __name__ == '__main__':
  absltest.main()
