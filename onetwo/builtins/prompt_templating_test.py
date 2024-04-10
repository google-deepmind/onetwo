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

import asyncio
import copy
import dataclasses
import pprint
import textwrap
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from onetwo.backends import backends_test_utils
from onetwo.builtins import prompt_templating
from onetwo.core import executing
from onetwo.core import sampling
from onetwo.core import templating


# Default reply for LanguageModelEngineForTest to return when it receives a
# prompt that it was not expecting.
DEFAULT_REPLY = 'UNKNOWN_PROMPT'

_DRY_RUN_PREFIX_VAR = templating._DRY_RUN_PREFIX_VAR

PROMPT_PREFIX = templating.PROMPT_PREFIX


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class _Person:
  name: str
  age: int


class PromptTemplatingTest(parameterized.TestCase):

  def test_j2_backend_params(self):
    prompt_text = textwrap.dedent("""\
      Please give a rationale and then answer the question.
      Question: {{ question }}
      Answer: {{ store('answer', llm(stop=['='], temperature=0.7, space_healing=False) | trim) }}
    """)

    question = 'What is 2 * 3 * 4?'
    expected_request = textwrap.dedent(
        """\
      Please give a rationale and then answer the question.
      Question: What is 2 * 3 * 4?
      Answer: """)
    reply_text = '2 * 3 * 4 = 24.\n'
    expected_reply_text_stripped = '2 * 3 * 4'  # Because of `trim`.
    expected_final_prefix = expected_request + expected_reply_text_stripped

    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={expected_request: reply_text},
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(prompt.render(question=question))

    with self.subTest('should_generate_only_the_expected_prompts'):
      self.assertEmpty(backend.unexpected_prompts)

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

  def test_j2_llm_mock_value_with_zipped_examples(self):
    prompt_text = textwrap.dedent("""\
        Please answer the following questions:
        {% for q, a in zip(example_questions, example_answers) -%}
        Q:{{ q }}
        A:{{ llm(mock_value=a, stop='\\n') }}
        {% endfor -%}
        """)

    example_questions = ['What is 2**2?', 'What is 3**3?']
    example_answers = ['4', None]

    expected_request = textwrap.dedent("""\
        Please answer the following questions:
        Q:What is 2**2?
        A:4
        Q:What is 3**3?
        A:""")
    expected_final_prefix = expected_request + DEFAULT_REPLY + '\n'
    backend = backends_test_utils.LLMForTest(default_reply=DEFAULT_REPLY)
    backend.register()

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(
        prompt.render(
            example_questions=example_questions, example_answers=example_answers
        )
    )

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  def test_j2_callback_iter_async(self):
    # We use a template that calls the llm, stores the result into 'answer',
    # and then takes the first character of answer to store it into
    # 'short_answer'. Only the call to `llm()` will generate substages in the
    # ExecutionResult, but the `store` calls will affect the output variables.
    prompt_template = prompt_templating.JinjaTemplateWithCallbacks(
        text=(
            'test{{ store("answer", llm()) }}'
            '{{ store("short_answer", __vars__.answer[0]) }}'
        )
    )

    # We enable iterable_replies on the test backend.
    # This will make the reply text come one character at a time.
    # On a real language model backend typically the reply would come one token
    # at a time, but we are just simulating some reply that gets built up
    # incrementally.
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={'test': 'ok'},
        reply_by_prompt_target={},
        default_reply='',
        default_score=0.0,
        iterable_replies=True,
    )
    backend.register()
    result = executing.run(prompt_template.render())
    with self.subTest('should_return_correct_results_on_run'):
      self.assertEqual(result[PROMPT_PREFIX], 'testoko', pprint.pformat(result))
      self.assertEqual(result['answer'], 'ok', pprint.pformat(result))
      self.assertEqual(result['short_answer'], 'o', pprint.pformat(result))

    result_stream = []
    with executing.safe_stream(prompt_template.render_stream()) as iterator:
      for update in iterator:
        result_stream.append(copy.deepcopy(update))

    expected_stream = [
        {PROMPT_PREFIX: 'test', templating.ITERABLE_REPLY: None},
        {PROMPT_PREFIX: 'test', templating.ITERABLE_REPLY: 'o'},
        {PROMPT_PREFIX: 'test', templating.ITERABLE_REPLY: 'ok'},
        {
            PROMPT_PREFIX: 'testok',
            templating.ITERABLE_REPLY: None,
            'answer': 'ok',
        },
        {
            PROMPT_PREFIX: 'testoko',
            templating.ITERABLE_REPLY: None,
            'answer': 'ok',
            'short_answer': 'o',
        },
    ]

    with self.subTest('should_return_correct_results_on_stream'):
      self.assertListEqual(
          result_stream, expected_stream, pprint.pformat(result_stream)
      )

  def test_j2_prompt_dynamic_exemplars(self):
    # Define a basic prompt with a repeated section for dynamic exemplars.
    prompt_text = textwrap.dedent("""\
      Please answer the following questions.
      {% for exemplar in exemplars %}
      Q: {{ exemplar.question }}
      A: {{ exemplar.answer }}
      {% endfor %}
      Q: {{ question }}
      A:{{ store('answer', llm() | trim) }}
    """)

    # Define a question and hypothetical decomposition reply from the backend,
    # along with other values that we expect the execution library to generate.
    question = 'What is 2 * 3 * 4?'
    expected_request = textwrap.dedent(
        """\
      Please answer the following questions.

      Q: What is 1 + 2?
      A: The answer is 3.

      Q: What is 1 + 2 + 3?
      A: The answer is 6.

      Q: What is 2 * 3 * 4?
      A:""")
    reply_text = ' The answer is 24.\n'
    expected_reply_text_stripped = 'The answer is 24.'

    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={expected_request: reply_text},
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()

    expected_final_prefix = expected_request + expected_reply_text_stripped
    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(
        prompt.render(
            exemplars=[
                {
                    'question': 'What is 1 + 2?',
                    'answer': 'The answer is 3.',
                },
                {
                    'question': 'What is 1 + 2 + 3?',
                    'answer': 'The answer is 6.',
                },
            ],
            question=question,
        )
    )

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_final_prefix,
          pprint.pformat(result[PROMPT_PREFIX]),
      )

  def test_execute_j2_prompts_repeated(self):
    # Define a basic prompt with a single LLM call. We will test the repetition
    # of this prompt to get several distinct samples in parallel.
    prompt_text = textwrap.dedent(
        """\
      Q: What is 1 + 2 + 3?
      A: Some explanation. The answer is 6.

      Q: What is 2 * 3?
      A: {{ store('answer', llm(space_healing=False)) }}
      """
    )

    expected_request = textwrap.dedent(
        """\
      Q: What is 1 + 2 + 3?
      A: Some explanation. The answer is 6.

      Q: What is 2 * 3?
      A: """
    )

    reply_text = []
    expected_prefixes = []
    for i in range(5):
      reply_text.append(f' Another explanation #{i}. The answer is 6.\n')
      expected_prefixes.append(expected_request + reply_text[-1])

    reply_by_prompt = {}
    reply_by_prompt[expected_request] = reply_text

    backend = backends_test_utils.LLMForTest(
        reply_by_prompt=reply_by_prompt,
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    executables = sampling.repeat_and_execute(prompt.render(), num_repeats=5)
    results = executing.run(executables)

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          [result[PROMPT_PREFIX] for result in results],
          expected_prefixes,
          pprint.pformat(results),
      )

  @parameterized.named_parameters(
      ('top1', 1, 'Apple is red'),
      ('top2', 2, 'Apple is red'),
      ('topk_larger_than_candidates_len', 4, 'Apple is red'),
  )
  def test_j2_choose(self, top_k: int, expected_prefix: str):
    prompt_text = (
        'Apple is {{ choose(["red", "green", "blue"], space_healing=False) }}'
    )
    if top_k > 1:
      prompt_text = (
          f'Apple is {{{{ choose(["red", "green", "blue"], {top_k}'
          ', space_healing=False) }}'
      )
    targets_and_scores = [
        ('red', -1.),
        ('green', -2.),
        ('blue', -3.),
    ]
    # Notice the double space after "is". This is because the test backend uses
    # f`{prefix} {suffix}` as a key, when looking up the score to return.
    reply_by_prompt_target = {
        f'Apple is  {target}': score for (target, score) in targets_and_scores
    }
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={},
        reply_by_prompt_target=reply_by_prompt_target,
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()

    if top_k > 1:
      pass

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(prompt.render())

    with self.subTest('should_generate_only_the_expected_prompts'):
      self.assertEmpty(backend.unexpected_prompts)

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result,
          {PROMPT_PREFIX: expected_prefix},
          pprint.pformat(result),
      )

  def test_j2_choose_callback_mock_value_with_zipped_examples(self):
    prompt_text = textwrap.dedent("""\
        Please answer the following questions:
        {% for q, a in zip(example_questions, example_answers) -%}
        Q:{{ q }}
        A:{{ choose(["yes", "no"], mock_value=a) }}
        {% endfor -%}
        """)

    example_questions = [
        'Are apples green?', 'Are cows red?', 'Are oranges blue?']
    example_answers = ['yes', 'no', None]

    expected_request = textwrap.dedent("""\
        Please answer the following questions:
        Q:Are apples green?
        A:yes
        Q:Are cows red?
        A:no
        Q:Are oranges blue?
        A:""")
    backend = backends_test_utils.LLMForTest(default_reply=DEFAULT_REPLY)
    backend.register()

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(
        prompt.render(
            example_questions=example_questions, example_answers=example_answers
        )
    )

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result[PROMPT_PREFIX],
          expected_request + 'yes\n',
          pprint.pformat(result),
      )

  @parameterized.named_parameters(
      (
          'make_list',
          'Make a list 1,2,3',
          list[int],
          """[1,2,3]""",
          """Make a list 1,2,3 [1, 2, 3]""",
      ),
      (
          'make_item',
          'Write 3.14',
          float,
          """3.14""",
          """Write 3.14 3.14""",
      ),
      (
          'make_class',
          'Generate person',
          _Person,
          """{'name': 'John Doe', 'age': 34}""",
          """Generate person _Person(name='John Doe', age=34)""",
      ),
  )
  def test_j2_generate_object(
      self, prefix: str, cls: type[Any], return_value: str, expected_prefix: str
  ):
    prompt_text = '{{ prefix }} {{ generate_object(cls) }}'
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={prefix: return_value},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()

    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    result = executing.run(prompt.render(prefix=prefix, cls=cls))

    with self.subTest('should_generate_only_the_expected_prompts'):
      self.assertEmpty(backend.unexpected_prompts)

    with self.subTest('produces_correct_filled_template'):
      self.assertEqual(
          result,
          {PROMPT_PREFIX: expected_prefix},
          pprint.pformat(result),
      )

  def test_j2_dry_run(self):
    expected_request = 'the first day of the week, i.e.'
    expected_reply = ', Sunday.\n\n'
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={expected_request: expected_reply},
        reply_by_prompt_target={},
        default_reply='some reply',
        default_score=0.0,
    )
    backend.register()
    prompt_text = textwrap.dedent("""\
      Today is{{ choose(['Tuesday', 'Monday', 'Wednesday']) }}.
      Tomorrow will be{{ llm(stop=['. '])}}.
      The number of days a week is {{ generate_object(int_cls) }}.
      Yesterday was{{ callback_with_llm() }}.""")
    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)

    async def my_callback() -> str:
      prompt = prompt_templating.JinjaTemplateWithCallbacks(
          text=(
              'the first day of the week, '
              'i.e.{{ llm(stop=[". "], max_tokens=5) }}'
          )
      )
      result = await prompt.render()
      return result[templating.PROMPT_PREFIX]

    def mock_my_callback(context):
      if _DRY_RUN_PREFIX_VAR not in context.output_variables:
        context.output_variables[_DRY_RUN_PREFIX_VAR] = {}
      if 'llm' not in context.output_variables[_DRY_RUN_PREFIX_VAR]:
        context.output_variables[_DRY_RUN_PREFIX_VAR]['llm'] = []
      context.output_variables[_DRY_RUN_PREFIX_VAR]['llm'].append(
          'the first day of the week, i.e.'
      )
      context.output_variables[_DRY_RUN_PREFIX_VAR]['abc'] = 'fgh'
      return 'Thursday'

    prompt.register_callback(name='callback_with_llm', function=my_callback)
    # Run without additional callbacks. In this case the `callback_with_llm`
    # callback will actually send the `llm` request.
    result = executing.run(prompt.dry_run({'int_cls': int},
                                          'AMAZING!', {int: 7}))
    with self.subTest('correct_result_without_additional_callbacks'):
      self.assertEqual(
          result,
          {
              'choose': ['Today is'],
              'llm': ['Today isTuesday.\nTomorrow will be'],
              'generate_object': ['Today isTuesday.\nTomorrow will beAMAZING!.'
                                  '\nThe number of days a week is '],
              'prefix': (
                  'Today isTuesday.\nTomorrow will beAMAZING!.\n'
                  'The number of days a week is 7.\n'
                  'Yesterday wasthe first day of the week, i.e., Sunday.\n\n.'
              )
          },
          pprint.pformat(result),
      )
    # Run with additional callback. In this case the no requests will get sent
    # to LLM.
    result = executing.run(prompt.dry_run(
        {'int_cls': int},
        'AMAZING!',
        {int: 7},
        [
            ('callback_with_llm', mock_my_callback, True),
        ]
    ))
    with self.subTest('correct_result_with_additional_callbacks'):
      self.assertEqual(
          result,
          {
              'choose': ['Today is'],
              'llm': [
                  'Today isTuesday.\nTomorrow will be',
                  'the first day of the week, i.e.'
              ],
              'generate_object': [
                  'Today isTuesday.\nTomorrow will beAMAZING!.\n'
                  'The number of days a week is '
              ],
              'abc': 'fgh',
              'prefix': (
                  'Today isTuesday.\nTomorrow will beAMAZING!.\n'
                  'The number of days a week is 7.\nYesterday wasThursday.'
              )
          },
          pprint.pformat(result),
      )
    with self.subTest('should_generate_only_the_expected_prompts'):
      self.assertEmpty(backend.unexpected_prompts)

  @parameterized.named_parameters(
      ('None', None, '1', 'test1'),
      ('one', 1, ['1'], "test['1']"),
      ('three', 3, ['1', '2', '3'], "test['1', '2', '3']"),
  )
  def test_samples(self, samples, expected_answer, expected_prefix):
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={'test': ['1', '2', '3', '4', '5']},
        reply_by_prompt_target={}, default_reply='', default_score=0.0,
    )
    backend.register()
    prompt = prompt_templating.JinjaTemplateWithCallbacks(
        text='test{{ store("answer", llm(samples=samples)) }}'
    )
    output = executing.run(prompt.render(samples=samples))
    with self.subTest('answer_should_contain_list'):
      self.assertEqual(output['answer'], expected_answer, output['answer'])

    with self.subTest('prefix_should_contain_list'):
      self.assertEqual(output['prefix'], expected_prefix, output['prefix'])

  @parameterized.named_parameters(
      ('None', None, 'long answer 1', 'testlong answer 1'),
      ('one', 1, ['long answer 1'], "test['long answer 1']"),
      (
          'three',
          3,
          ['long answer 1', 'long answer 2', 'long answer 3'],
          "test['long answer 1', 'long answer 2', 'long answer 3']",
      ),
  )
  def test_samples_streaming(self, samples, expected_answer, expected_prefix):
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            'test': [
                'long answer 1',
                'long answer 2',
                'long answer 3',
                'long answer 4',
                'long answer 5',
            ]
        },
        reply_by_prompt_target={},
        default_reply='',
        default_score=0.0,
        iterable_replies=True,
    )
    backend.register()
    prompt = prompt_templating.JinjaTemplateWithCallbacks(
        text='test{{ store("answer", llm(samples=samples)) }}'
    )
    output = executing.run(prompt.render(samples=samples))
    with self.subTest('answer_should_contain_list'):
      self.assertEqual(output['answer'], expected_answer, output['answer'])

    with self.subTest('prefix_should_contain_list'):
      self.assertEqual(output['prefix'], expected_prefix, output['prefix'])

  def test_j2_render_stream_iterable(self):
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={'test-': 'abc'},
        iterable_replies=True,
    )
    backend.register()
    prompt = prompt_templating.JinjaTemplateWithCallbacks(
        text='test-{{ store("answer", llm()) }}'
    )
    res = []

    def cb(r):
      nonlocal res
      res.append(copy.deepcopy(r))

    with self.subTest('iteration_depth=1'):
      # We use stream_with_callback to ensure that we get a call at every step
      # of the underlying iterator.
      executing.stream_with_callback(prompt.render_stream(), cb)
      expected = [
          {'_iterable_reply': None, 'prefix': 'test-'},
          {'_iterable_reply': 'a', 'prefix': 'test-'},
          {'_iterable_reply': 'ab', 'prefix': 'test-'},
          {'_iterable_reply': 'abc', 'prefix': 'test-'},
          {'_iterable_reply': None, 'prefix': 'test-abc', 'answer': 'abc'},
          {'_iterable_reply': None, 'prefix': 'test-abc', 'answer': 'abc'},
      ]
      self.assertEqual(res, expected, pprint.pformat(res))

    with self.subTest('iteration_depth=0'):
      executing.stream_with_callback(
          prompt.render_stream(), cb, iteration_depth=0
      )
      expected = [
          {'_iterable_reply': None, 'prefix': 'test-'},
          {'_iterable_reply': 'a', 'prefix': 'test-'},
          {'_iterable_reply': 'ab', 'prefix': 'test-'},
          {'_iterable_reply': 'abc', 'prefix': 'test-'},
          {'_iterable_reply': None, 'prefix': 'test-abc', 'answer': 'abc'},
          {'_iterable_reply': None, 'prefix': 'test-abc', 'answer': 'abc'},
          {'_iterable_reply': None, 'prefix': 'test-abc', 'answer': 'abc'},
      ]
      self.assertEqual(res, expected, pprint.pformat(res))

  def test_j2_dry_run_effects(self):
    """Making sure that dry run works well with multiple async calls.

    This test is meant to check one very specific and hard to debug issue that
    was observed in the past. Initial implementation of `dry_run` was based on
    storing `self._callbacks` dictionary in the beginning of the function and
    then restoring it before it returned. This way, all the callbacks that we
    mock in the `dry_run` should be properly restored back to normal. Bug was
    observed when executing multiple `dry_run` coroutines of the same prompt
    instance with `asyncio.gather(*coroutines)`. Because of the async nature of
    execution it is possible that when one of the coroutines starts its job
    `prompt._callbacks` contains not the original callbacks, but already mocked
    ones. In this case the coroutine will copy the mocked callbacks and later
    put them back to `prompt._callbacks`. After `asyncio.gather(*coroutines)`
    finishes the `prompt` instance contains modified (mocked) callbacks.
    """
    prompt_text = '{{ begin }}{{ llm() }}{{ choose(["1", "2"]) }}'
    prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
    llm_callback_fn = prompt._callbacks['llm']
    choose_callback_fn = prompt._callbacks['choose']
    generate_object_callback_fn = prompt._callbacks['generate_object']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    coroutines = [prompt.dry_run({'begin': letter}) for letter in letters]

    @executing.make_executable
    async def multiple_dry_runs():
      return await asyncio.gather(*coroutines)

    _ = executing.run(multiple_dry_runs())
    with self.subTest('llm_fn_did_not_change'):
      self.assertEqual(llm_callback_fn, prompt._callbacks['llm'])
    with self.subTest('choose_fn_did_not_change'):
      self.assertEqual(choose_callback_fn, prompt._callbacks['choose'])
    with self.subTest('generate_object_fn_did_not_change'):
      self.assertEqual(
          generate_object_callback_fn, prompt._callbacks['generate_object']
      )

  def test_j2_llm_space_healing(self):
    prompt_text_1 = 'Question: 1 + 1? Answer: {{ llm(space_healing=False) }}'
    prompt_text_2 = 'Question: 1 + 2? Answer: {{ llm() }}'

    expected_request_1 = 'Question: 1 + 1? Answer: '
    expected_request_2 = 'Question: 1 + 2? Answer:'

    reply_text_1 = ' 2'
    reply_text_2 = ' 3'

    backend_1 = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request_1: reply_text_1,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    # We create two separate backends to make sure the
    # `generates_only_the_expected_prompts` test behaves as expected.
    backend_2 = backends_test_utils.LLMForTest(
        reply_by_prompt={
            expected_request_2: reply_text_2,
        },
        reply_by_prompt_target={},
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )

    for backend, prompt_text, expected_result, test_name in [
        (
            backend_1,
            prompt_text_1,
            expected_request_1 + reply_text_1,
            '_without_healing',
        ),
        (
            backend_2,
            prompt_text_2,
            expected_request_2 + reply_text_2,
            '_with_healing',
        ),
    ]:
      backend.register()
      prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
      result = executing.run(prompt.render())

      with self.subTest(f'generates_only_the_expected_prompts{test_name}'):
        self.assertEmpty(backend.unexpected_prompts)

      with self.subTest(f'validate_llm{test_name}'):
        self.assertEqual(
            result[PROMPT_PREFIX], expected_result, result[PROMPT_PREFIX]
        )

  def test_j2_choose_space_healing(self):
    prompt_text_1 = (
        'Question: 1 + 1? Answer: {{ '
        'choose(["1", "2"], space_healing=False) }}'
    )
    prompt_text_2 = 'Question: 1 + 1? Answer: {{ choose(["1", "2"]) }}'

    expected_request_1 = 'Question: 1 + 1? Answer: '
    expected_request_2 = 'Question: 1 + 1? Answer:'

    backend_1 = backends_test_utils.LLMForTest(
        reply_by_prompt_target={
            f'{expected_request_1} 1': -2.,
            f'{expected_request_1} 2': -1.,
        },
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    # We create two separate backends to make sure the
    # `generates_only_the_expected_prompts` test behaves as expected.
    backend_2 = backends_test_utils.LLMForTest(
        reply_by_prompt_target={
            f'{expected_request_2} 1': -4.,
            f'{expected_request_2} 2': -3.,
        },
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )

    for backend, prompt_text, expected_result, test_name in [
        (
            backend_1,
            prompt_text_1,
            expected_request_1 + '2',
            '_without_healing',
        ),
        (
            backend_2,
            prompt_text_2,
            expected_request_2 + ' 2',
            '_with_healing',
        ),
    ]:
      backend.register()
      prompt = prompt_templating.JinjaTemplateWithCallbacks(text=prompt_text)
      result = executing.run(prompt.render())

      with self.subTest(f'generates_only_the_expected_prompts{test_name}'):
        self.assertEmpty(backend.unexpected_prompts)

      with self.subTest(f'validate_llm{test_name}'):
        self.assertEqual(
            result[PROMPT_PREFIX],
            expected_result,
            pprint.pformat(result[PROMPT_PREFIX]),
        )

  @parameterized.named_parameters(
      ('space', ' ', ' a', ' 2', ' 3'),
      ('no_space', '', ' a', '2', '3'),
  )
  def test_j2_space_healing_empty_prefix(
      self, prompt_prefix, expected_llm, expected_choose,
      expected_generate_object
  ):
    prompt1 = prompt_templating.JinjaTemplateWithCallbacks(
        text='{{ prompt_prefix }}{{ llm(space_healing=True) }}'
    )
    prompt2 = prompt_templating.JinjaTemplateWithCallbacks(
        text='{{ prompt_prefix }}{{ choose(["1", "2"], space_healing=True) }}'
    )
    prompt3 = prompt_templating.JinjaTemplateWithCallbacks(
        text=(
            '{{ prompt_prefix }}{{ generate_object(cls, space_healing=True) }}'
        ),
    )
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            '': ' a',  # We return a reply starting with a space.
        },
        reply_by_prompt_target={
            # LanguageModelEngineForTest looks up the key f'{prefix} {target}'.
            ' 1': -4.0,
            ' 2': -3.0,
        },
        default_reply=DEFAULT_REPLY,
        default_score=0.0,
    )
    backend.register()
    prefix1 = executing.run(prompt1.render(prompt_prefix=prompt_prefix))[
        'prefix'
    ]
    prefix2 = executing.run(prompt2.render(prompt_prefix=prompt_prefix))[
        'prefix'
    ]
    with self.subTest('generates_only_the_expected_prompts'):
      self.assertEmpty(backend.unexpected_prompts)

    with self.subTest('validate_llm'):
      self.assertEqual(prefix1, expected_llm, repr(prefix1))

    with self.subTest('validate_choose'):
      self.assertEqual(prefix2, expected_choose, repr(prefix2))
    backend = backends_test_utils.LLMForTest(
        reply_by_prompt={
            '': ' 3',  # We return a reply starting with a space.
        },
        default_reply=DEFAULT_REPLY,
    )
    backend.register()
    prefix3 = executing.run(prompt3.render(
        prompt_prefix=prompt_prefix, cls=int))['prefix']

    with self.subTest('generates_only_the_expected_prompts_generate_object'):
      self.assertEmpty(backend.unexpected_prompts)

    with self.subTest('validate_generate_object'):
      self.assertEqual(prefix3, expected_generate_object, repr(prefix3))


if __name__ == '__main__':
  absltest.main()
  absltest.main()
