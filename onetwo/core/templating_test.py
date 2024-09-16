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

"""Tests for templating."""

from collections.abc import Callable, Mapping
import copy
import os
import pprint
import textwrap
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jinja2
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import templating
from onetwo.core import tracing


_DRY_RUN_PREFIX_VAR = templating._DRY_RUN_PREFIX_VAR

PROMPT_PREFIX = templating.PROMPT_PREFIX
Context = templating.PromptTemplateContext


def _llm(reply_by_prompt: Mapping[str, str]) -> Callable[[Context], str]:
  @tracing.trace('llm')
  async def llm(context: Context) -> str:
    return reply_by_prompt[context.prefix]
  return llm


class TemplatingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The `self.create_tempdir` method uses command line flag and such flags
    # are not marked as parsed by default when running with pytest. Marking as
    # parsed directly here to make the pytest run pass.
    flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      ('all_defined', None, {'var1': 'test', 'var2': ''}),
      (
          'some_undefined',
          ValueError,  # This will fail when tracing the call to the template.
          {'var1': 'test', 'var2': jinja2.Undefined()},
      ),
      (
          'some_absent',
          jinja2.exceptions.UndefinedError,
          {'var1': 'test'},  # This will fail when populating the template.
      ),
  )
  def test_j2_undefined(self, should_raise, vars_dict):
    prompt_text = '{{ var1 }}{{ var2 }}'
    prompt = templating.JinjaTemplate(text=prompt_text)

    if should_raise is not None:
      with self.assertRaises(should_raise):
        _ = executing.run(prompt.render(**vars_dict))
    else:
      with self.subTest('produces_correct_filled_template'):
        result = executing.run(prompt.render(**vars_dict))
        self.assertEqual(
            result[PROMPT_PREFIX],
            'test',
            pprint.pformat(result[PROMPT_PREFIX]),
        )

  def test_j2_simple(self):
    prompt_text = textwrap.dedent("""\
      Please give a rationale and then answer the question.
      Question: {{ question }}
      Answer: {{ llm() }}
    """)

    question = 'What is 2 * 3 * 4?'
    expected_request = textwrap.dedent(
        """\
      Please give a rationale and then answer the question.
      Question: What is 2 * 3 * 4?
      Answer: """)
    reply_text = '2 * 3 * 4 = 24.\n'

    expected_final_prefix = expected_request + reply_text

    prompt = templating.JinjaTemplate(text=prompt_text)
    prompt.register_callback(
        'llm',
        _llm(reply_by_prompt={expected_request: reply_text}),
        pass_context=True,
    )
    result = executing.run(prompt.render(question=question))

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  def test_j2_simple_store(self):
    prompt_text = textwrap.dedent("""\
      Please give a rationale and then answer the question.
      Question: {{ question }}
      Answer: {{ store('answer', llm() | trim) }}
    """)

    question = 'What is 2 * 3 * 4?'
    expected_request = textwrap.dedent(
        """\
      Please give a rationale and then answer the question.
      Question: What is 2 * 3 * 4?
      Answer: """)
    reply_text = '2 * 3 * 4 = 24.\n'
    expected_reply_text_stripped = reply_text.strip()
    expected_final_prefix = expected_request + expected_reply_text_stripped

    prompt = templating.JinjaTemplate(text=prompt_text)
    prompt.register_callback(
        'llm',
        _llm(reply_by_prompt={expected_request: reply_text}),
        pass_context=True,
    )
    result = executing.run(prompt.render(question=question))

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

    with self.subTest('produces_correct_outputs'):
      self.assertEqual(
          result['answer'],
          expected_reply_text_stripped,
          pprint.pformat(result['answer']),
      )

  @parameterized.named_parameters(
      (
          'to_context_false',
          """
            {{- store('answer', 'test', to_context=False) -}}
            {{- store('context_answer', __vars__.answer) -}}
          """,
          {
              'answer': 'test',
              'context_answer': jinja2.Undefined(),
              'prefix': 'test',
          },
      ),
      (
          'to_context_true',
          """
            {{- store('answer', 'test', to_context=True) -}}
            {{- store('context_answer', __vars__.answer) -}}
          """,
          {
              'answer': 'test',
              'context_answer': 'test',
              'prefix': 'testtest',
          },
      ),
      (
          'append_to_context',
          """
            {{- store('answer', 'test', append=True) -}}
            {{- store('context_answer', __vars__.answer) -}}
            {{- store('answer', 'test2', append=True) -}}
          """,
          {
              'answer': ['test', 'test2'],
              'context_answer': ['test'],
              'prefix': "test['test']test2",
          },
      ),
      (
          'append_to_output',
          """
            {{- store('answer', 'test', append=True, to_context=False) -}}
            {{- store('context_answer', __vars__.answer) -}}
            {{- store('answer', 'test2', append=True) -}}
          """,
          {
              'answer': ['test', 'test2'],
              'context_answer': jinja2.Undefined(),
              'prefix': 'testtest2',
          },
      ),
  )
  def test_j2_advanced_store(self, prompt_text, expected_result):
    prompt = templating.JinjaTemplate(
        text=prompt_text,
    )
    if expected_result['context_answer'] == jinja2.Undefined():
      with self.assertRaises(jinja2.exceptions.UndefinedError):
        _ = executing.run(prompt.render())
    else:
      with self.subTest('produces_correct_filled_template'):
        result = executing.run(prompt.render())
        self.assertEqual(result, expected_result, pprint.pformat(result))

  def test_j2_zip_in_control_block(self):
    prompt_text = textwrap.dedent("""\
        Please answer the following questions:
        {% for q, a in zip(example_questions, example_answers) -%}
        Q: {{ q }}
        A: {{ a }}
        {% endfor -%}
        Q: {{ question }}
        A: {{ llm() }}""")
    example_questions = ['What is 2**2?', 'What is 3**3?']
    example_answers = ['4', '27']
    question = 'What is 4**2?'

    expected_request = textwrap.dedent("""\
        Please answer the following questions:
        Q: What is 2**2?
        A: 4
        Q: What is 3**3?
        A: 27
        Q: What is 4**2?
        A: """)
    expected_final_prefix = expected_request + '16'
    prompt = templating.JinjaTemplate(text=prompt_text)
    prompt.register_callback(
        'llm',
        _llm(reply_by_prompt={expected_request: '16'}),
        pass_context=True,
    )
    result = executing.run(
        prompt.render(
            example_questions=example_questions,
            example_answers=example_answers,
            question=question,
        )
    )

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  def test_j2_template_globals_in_variable_block(self):
    prompt_text = textwrap.dedent("""\
        Please complete the following sequence:
        {{ join(zip(a, b)) }}, {{ llm() }}""")
    a = [1, 2, 3]
    b = [1, 4, 9]

    expected_request = textwrap.dedent("""\
        Please complete the following sequence:
        (1, 1), (2, 4), (3, 9), """)
    expected_final_prefix = expected_request + '(4, 16)'

    join = lambda x: ', '.join([str(e) for e in x])
    prompt = templating.JinjaTemplate(
        text=prompt_text, template_globals={'zip': zip, 'join': join}
    )
    prompt.register_callback(
        'llm',
        _llm(reply_by_prompt={expected_request: '(4, 16)'}),
        pass_context=True,
    )
    result = executing.run(prompt.render(a=a, b=b))

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  @parameterized.named_parameters(
      ('one_step', False),
      ('incremental', True),
  )
  def test_j2_sequence(self, incremental):
    prompt_text = textwrap.dedent("""\
      Please give a rationale and then answer the question.
      Question: {{ question }}
      Rationale: {{ store('rationale', llm()) }}
      Answer: {{ store('answer', llm()) }}
    """)

    question = 'What is 2 * 3 * 4?'
    expected_request_1 = textwrap.dedent(
        """\
      Please give a rationale and then answer the question.
      Question: What is 2 * 3 * 4?
      Rationale: """)
    reply_text_1 = '2 * 3 = 6. 6 * 4 = 24.'
    expected_request_2 = textwrap.dedent(
        """\
      Please give a rationale and then answer the question.
      Question: What is 2 * 3 * 4?
      Rationale: 2 * 3 = 6. 6 * 4 = 24.
      Answer: """)
    reply_text_2 = '24.'
    expected_final_prefix = expected_request_2 + reply_text_2
    expected_prefixes = [
        # We get this prefix before any request is made.
        'Please give a rationale and then answer the question.\nQuestion: ',
        # Then we get an update after the llm has returned the first reply.
        expected_request_1 + reply_text_1,
        # And again after the second reply.
        expected_final_prefix,
        # And once more with the final result.
        expected_final_prefix,
    ]

    prompt = templating.JinjaTemplate(text=prompt_text)
    prompt.register_callback(
        'llm',
        _llm(
            reply_by_prompt={
                expected_request_1: reply_text_1,
                expected_request_2: reply_text_2,
            }
        ),
        pass_context=True,
    )
    if not incremental:
      result = executing.run(prompt.render(question=question))[PROMPT_PREFIX]
    else:
      result = []
      with executing.safe_stream(
          prompt.render_stream(question=question)
      ) as iterator:
        for r in iterator:
          result.append(copy.deepcopy(r[PROMPT_PREFIX]))

    with self.subTest('produces_correct_filled_template'):
      if not incremental:
        self.assertEqual(result, expected_final_prefix)
      else:
        self.assertLen(result, len(expected_prefixes))
        for i, (r, er) in enumerate(zip(result, expected_prefixes)):
          self.assertEqual(r, er, f'result {i}: {r}, expected: {er}')

  def test_j2_callback_async(self):
    async def callback(context, answer, var):
      res = answer + '_parsed_' + var
      context.context_variables['result'] = res
      context.output_variables['result'] = res
      return res

    prompt_template = templating.JinjaTemplate(
        text='test{% set answer = llm() %}{% do func(answer, v) %}'
    )
    prompt_template.register_callback(
        'func',
        callback,
        pass_context=True,
    )
    prompt_template.register_callback(
        'llm', _llm(reply_by_prompt={'test': 'result'}), pass_context=True,
    )
    result = executing.run(prompt_template.render(v='first'))
    self.assertEqual(result['result'], 'result_parsed_first')

  def test_j2_sections(self):
    prompt_text = textwrap.dedent("""\
      part1
      {% section name=my_name -%}
      part2
      {{ llm() }}
      {% section hidden=true, name='inner' -%}
      part3
      {{ llm() }}
      {% endsection -%}
      {% section hidden=true -%}
      part4
      {{ llm() }}
      {% endsection -%}
      {% endsection -%}
      part5
      {{ llm() }}
      """)
    prompt = templating.JinjaTemplate(
        text=prompt_text,
    )
    requests = []

    def prefix(context):
      requests.append(context.prefix)
      return 'next'

    prompt.register_callback('llm', prefix, pass_context=True)
    result = executing.run(prompt.render(my_name='test'))

    expected_outputs = {'inner': 'part3\nnext\n', 'test': 'part2\nnext\n'}
    with self.subTest('should_store_section_values'):
      self.assertEqual(
          result['inner'], expected_outputs['inner'], pprint.pformat(result)
      )
      self.assertEqual(
          result['test'], expected_outputs['test'], pprint.pformat(result)
      )

    expected_requests = [
        'part1\npart2\n',
        'part1\npart2\nnext\npart3\n',
        'part1\npart2\nnext\npart4\n',
        'part1\npart2\nnext\npart5\n',
    ]
    with self.subTest('should_issue_correct_requests'):
      self.assertListEqual(
          requests, expected_requests, pprint.pformat(requests)
      )

    with self.subTest('should_hide_hidden_sections'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_requests[-1] + 'next',
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  @parameterized.named_parameters(
      (
          'all_tags',
          {
              'user': ('<user_tag>', '</user_tag>'),
              'other': ('<other_tag>', '</other_tag>'),
              'llm': ('<llm_tag>', '</llm_tag>'),
          },
          (
              'part1\n'
              '<user_tag><user>part2\n'
              '</user></user_tag><llm_tag>test\n'
              '</llm_tag>part4'
          ),
      ),
      (
          'None_tags',
          {
              'user': (None, None),
              'llm': ('<llm>', '</llm>'),
          },
          'part1\n<llm>test\n</llm>part4',
      ),
  )
  def test_j2_roles(self, role_tags, expected_prefix):
    prompt_text = textwrap.dedent("""\
      part1
      {% role name='user', add_tags=True -%}
      part2
      {% endrole -%}
      {% role name='other', hidden=True -%}
      part3
      {% endrole -%}
      {% role name='llm' -%}
      {{ llm() }}
      {% endrole -%}
      part4
      """)
    prompt = templating.JinjaTemplate(
        text=prompt_text,
        role_tags=role_tags,
    )
    prompt.register_callback('llm', lambda: 'test')
    result = executing.run(prompt.render())

    expected_outputs = {
        'prefix': 'part1\n<user>part2\n</user>test\npart4',
        'prefix_with_roles': expected_prefix,
    }
    with self.subTest('should_add_role_tags'):
      self.assertEqual(result, expected_outputs, pprint.pformat(result))

  @parameterized.named_parameters(
      (
          'all_tags',
          {
              'user': ('<user_tag>', '</user_tag>'),
              'other': ('<other_tag>', '</other_tag>'),
              'llm': ('<llm_tag>', '</llm_tag>'),
          },
          {
              'user': ('user_prefix_begin', 'user_prefix_end'),
              'other': ('', 'other_prefix_end'),
              'llm': ('', ''),
          },
          (
              'part1\nuser_prefix_begin\npart2\nuser_prefix_end\npart3\n'
              'other_prefix_end\ntest\npart4'
          ),
          (
              'part1\n'
              '<user_tag>user_prefix_begin\npart2\nuser_prefix_end</user_tag>'
              '<other_tag>\npart3\nother_prefix_end</other_tag>'
              '<llm_tag>\ntest\n</llm_tag>'
              'part4'
          ),
      ),
      (
          'None_tags',
          {
              'user': (None, None),
              'llm': ('<llm>', '</llm>'),
          },
          {
              'user': ('user_prefix_begin', 'user_prefix_end'),
              'other': ('', 'other_prefix_end'),
              'llm': ('', ''),
          },
          (
              'part1\nuser_prefix_begin\npart2\nuser_prefix_end\n'
              'part3\nother_prefix_end\ntest\npart4'
          ),
          'part1\n\npart3\nother_prefix_end<llm>\ntest\n</llm>part4',
      ),
  )
  def test_j2_prefix_roles(
      self,
      role_tags,
      prefix_role_tags,
      expected_prefix,
      expected_prefix_with_roles,
  ):
    prompt_text = textwrap.dedent("""\
      part1
      {% role name='user' %}
      part2
      {% endrole -%}
      {% role name='other' %}
      part3
      {% endrole -%}
      {% role name='llm' %}
      {{ llm() }}
      {% endrole -%}
      part4
      """)
    prompt = templating.JinjaTemplate(
        text=prompt_text,
        role_tags=role_tags,
        prefix_role_tags=prefix_role_tags,
        add_prefix_role_tags=True,
    )
    prompt.register_callback('llm', lambda: 'test')
    result = executing.run(prompt.render())

    expected_outputs = {
        'prefix': expected_prefix,
        'prefix_with_roles': expected_prefix_with_roles,
    }
    with self.subTest('should_add_role_tags'):
      self.assertEqual(result, expected_outputs, pprint.pformat(result))

  def test_j2_set_default_prefix_role_tags(self):
    prefix_role_tags = {
        'user': ('<user_tag>', '</user_tag>'),
        'model': ('<model_tag>', '</model_tag>'),
    }
    with routing.RegistryContext():
      templating.JinjaTemplate.set_default_prefix_role_tags(prefix_role_tags)
      with self.subTest('use default when not set explicitly'):
        template = templating.JinjaTemplate(text='Test template use defaults')
        self.assertDictEqual(prefix_role_tags, template.prefix_role_tags)
        self.assertTrue(template.add_prefix_role_tags)

      with self.subTest('prefer explicitly set values over defaults'):
        template = templating.JinjaTemplate(
            text='Test template user explicit values',
            prefix_role_tags={},
            add_prefix_role_tags=False,
        )
        self.assertDictEqual({}, template.prefix_role_tags)
        self.assertFalse(template.add_prefix_role_tags)

    with self.subTest('No defaults set'):
      explicit_tags = {'user': ('<explicit_user_tag>', '</explicit_user_tag>')}
      template = templating.JinjaTemplate(
          text='Test template no defaults set',
          prefix_role_tags=explicit_tags,
          add_prefix_role_tags=True,
      )
      self.assertDictEqual(explicit_tags, template.prefix_role_tags)
      self.assertTrue(template.add_prefix_role_tags)

  def test_j2_input(self):
    prompt_text = """
      {{- input('var') -}}
    """
    prompt = templating.JinjaTemplate(text=prompt_text)
    output = executing.run(prompt.render(var='test'))
    self.assertEqual(output['prefix'], 'test')

    with mock.patch(
        'onetwo.core.templating.input',
        create=True,
    ) as mock_input:
      mock_input.side_effect = ['a', 'b']
      output = executing.run(prompt.render())
      self.assertEqual(output['prefix'], 'a')

  def test_j2_input_with_stream(self):
    prompt_text = """before-
    {{- input('var') -}}
    -after-
    {{- input('var2') -}}
    -later"""
    prompt = templating.JinjaTemplate(text=prompt_text)
    res = []

    def cb(r):
      nonlocal res
      res.append(copy.deepcopy(r))

    # We use stream_with_callback to ensure that we get a call at every step
    # of the underlying iterator.
    executing.stream_with_callback(
        prompt.render_stream(var='test', var2='test2'), cb
    )
    expected = [
        {'_iterable_reply': None, 'prefix': 'before-'},
        {'_iterable_reply': None, 'prefix': 'before-test'},
        {'_iterable_reply': None, 'prefix': 'before-test-after-test2'},
        {'_iterable_reply': None, 'prefix': 'before-test-after-test2-later'},
    ]
    self.assertEqual(res, expected, pprint.pformat(res))

  def test_j2_move_inputs_to_context_vars(self):
    prompt_text = textwrap.dedent("""\
        input visible:{{ in_input }}
        {% if check_context -%}
        context invisible:{{ in_context_vars }}
        {% endif -%}
        {% if check_input -%}
        input in vars invisible:{{ __vars__.in_input }}
        {% endif -%}
        context in vars visible:{{ __vars__.in_context_vars }}
        """)

    # Indirectly test whether the inputs are on the correct place by using
    # the property that non-existing variables will render as empty strings.
    # TODO: Can we add context-variables to the tracing and check
    # this more directly?
    expected_result = textwrap.dedent("""\
        input visible:INPUT
        context in vars visible:CONTEXT""")

    prompt = templating.JinjaTemplate(
        text=prompt_text, move_inputs_to_context_vars=['in_context_vars']
    )
    with self.subTest('should_move_inputs_to_context_vars'):
      result = executing.run(
          prompt.render(in_input='INPUT', in_context_vars='CONTEXT',
                        check_context=False, check_input=False)
      )
      self.assertEqual(result[PROMPT_PREFIX], expected_result)

    with self.subTest('should_have_empty_in_context_vars'):
      with self.assertRaises(jinja2.UndefinedError):
        executing.run(
            prompt.render(in_input='INPUT', in_context_vars='CONTEXT',
                          check_context=True, check_input=False)
        )

    with self.subTest('should_have_empty_in_input'):
      with self.assertRaises(jinja2.UndefinedError):
        executing.run(
            prompt.render(in_input='INPUT', in_context_vars='CONTEXT',
                          check_context=False, check_input=True)
        )

  @parameterized.named_parameters(
      ('error', True, 'testing #ERROR#: something wrong', 'something wrong'),
      ('fine', False, 'testing ok', None),
  )
  def test_j2_error_in_callback(self, raises, expected_prefix, expected_error):
    def cb(raises: bool = True):
      if raises:
        raise ValueError('something wrong')
      else:
        return 'ok'

    prompt = templating.JinjaTemplate(
        text='testing {{ my_function(raises) }}'
    )
    prompt.register_callback('my_function', cb)
    result = executing.run(prompt.render(raises=raises))

    with self.subTest('prompt_prefix'):
      self.assertEqual(
          expected_prefix, result['prefix'], repr(result['prefix'])
      )

    with self.subTest('error_field'):
      self.assertEqual(
          expected_error, result.get('error'), repr(result.get('error'))
      )

  def test_j2_raises_at_include_if_no_loader_present(self):
    prompt = templating.JinjaTemplate(text="{% include 'test_file.jinja' %}")
    with self.assertRaises(TypeError):
      executing.run(prompt.render())

  def test_j2_includes_if_loader_is_present(self):
    tmp_dir = self.create_tempdir().full_path
    with open(os.path.join(tmp_dir, 'test_file.jinja'), 'wt+') as f:
      f.write('Test included prompt')
    loader = jinja2.FileSystemLoader(searchpath=tmp_dir)
    prompt = templating.JinjaTemplate(
        text="{% include 'test_file.jinja' %}", loader=loader
    )
    result = executing.run(prompt.render())
    self.assertEqual(
        result['prefix'], 'Test included prompt', repr(result['prefix'])
    )

  def test_render_template_with_code_execution_fails(self):
    prompt = templating.JinjaTemplate(
        text='var: {{ self.__init__.__globals__ }}'
    )

    with self.assertRaises(jinja2.exceptions.SecurityError):
      executing.run(prompt.render())


if __name__ == '__main__':
  absltest.main()
  absltest.main()
