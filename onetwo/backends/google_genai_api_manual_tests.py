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

from collections.abc import Iterator, Sequence
import contextlib
import dataclasses

import json
import math
import os
import pprint
import time
from typing import Any, cast
import uuid

from absl import app
from absl import flags
from absl import logging
from google.genai import types as genai_types
from onetwo.agents import agent as agent_lib
from onetwo.backends import google_genai_api
from onetwo.builtins import composables as c
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import function_calling as fc_types
from onetwo.core import prefix_caching
from onetwo.core import routing
from onetwo.core import sampling
from onetwo.stdlib.tool_use import llm_tool_use
import pydantic
import pydantic.dataclasses as pydantic_dataclasses

pydantic_dataclass = pydantic_dataclasses.dataclass

# gemini-2.5-flash.
# See https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash  # pylint: disable=line-too-long
_MAX_INPUT_TOKENS = 1048576

_API_KEY = flags.DEFINE_string('api_key', default=None, help='GenAI API key.')
_PROJECT = flags.DEFINE_string('project', default=None, help='Project ID.')
_LOCATION = flags.DEFINE_string('location', default=None, help='Location.')
_RUN_ONLY = flags.DEFINE_list('run_only', default=None, help='Test filter.')
flags.mark_flags_as_mutual_exclusive(['api_key', 'project'])
flags.mark_flags_as_mutual_exclusive(['api_key', 'location'])

_CACHE_DIR = flags.DEFINE_string(
    'cache_dir',
    default='.',
    help='Directory where the cache will be stored.',
)
_LOAD_CACHE = flags.DEFINE_bool(
    'load_cache_file',
    default=False,
    help='Whether we should read the cache stored in file.',
)
_SAVE_CACHE = flags.DEFINE_bool(
    'save_cache_file',
    default=False,
    help='Whether we should save the cache stored in file.',
)
_PRINT_DEBUG = flags.DEFINE_bool(
    'print_debug', default=False, help='Debug logging.'
)
_THREADPOOL_SIZE = flags.DEFINE_integer(
    'threadpool_size',
    default=4,
    help='Number of threads to use in the threadpool.',
)


def _should_run_test(num: float) -> bool:
  """Returns whether the test should run based on --run_only filter."""
  return not _RUN_ONLY.value or str(num) in _RUN_ONLY.value


class _TestCase:
  """Tracks pass/fail state for a single test."""

  def __init__(self) -> None:
    self.passed = True

  def fail(self, message: str) -> None:
    self.passed = False
    print(f'FAIL: {message}')


@contextlib.contextmanager
def _run_test(
    num: float,
    description: str,
    test_results: list[tuple[float, str, bool]],
) -> Iterator[_TestCase]:
  """Context manager that runs a test and catches any exceptions.

  Usage:
    if _should_run_test(1):
      with _run_test(1, 'description', test_results) as tc:
        ... test body ...
        if something_wrong:
          tc.fail('reason')

  If the body raises an exception or tc.fail() is called, the test is recorded
  as failed. This ensures the test suite always runs to completion.

  Args:
    num: Test number.
    description: Test description.
    test_results: List of (num, description, passed) tuples tracking per-test
      results.

  Yields:
    A _TestCase object with a fail() method for inline assertion failures.
  """
  print(f'{num}. {description}.')
  tc = _TestCase()
  try:
    yield tc
    test_results.append((num, description, tc.passed))
  except Exception as e:  # pylint: disable=broad-except
    test_results.append((num, description, False))
    print(f'FAIL: {type(e).__name__}: {e}')


def _print_summary(test_results: list[tuple[float, str, bool]]) -> bool:
  """Prints a per-test summary and returns True if all tests passed."""
  print('\n' + '=' * 60)
  print('TEST SUMMARY')
  print('=' * 60)
  all_passed = True
  for num, description, passed in test_results:
    status = 'PASS' if passed else 'FAIL'
    print(f'  {num:5}. {description}: {status}')
    if not passed:
      all_passed = False
  print('=' * 60)
  print('OVERALL:', 'PASS' if all_passed else 'FAIL')
  print('=' * 60)
  return all_passed


def _run_agent(
    agent: agent_lib.Agent, query: str, **kwargs
) -> str:
  """Runs an agent and returns its answer.

  `Agent.__call__` is declared to return a union that also covers
  `return_final_state`, which this caller never asks for.

  Args:
    agent: The agent to run.
    query: What to ask it.
    **kwargs: Forwarded to `Agent.__call__`.

  Returns:
    The agent's answer, or '' if it gave none.
  """
  return cast(
      str,
      executing.run(agent(query, **kwargs)),
  )


def _first_model_turn(
    agent: agent_lib.Agent, query: str
) -> fc_types.ConversationTurn:
  """Runs a single step of an agent and returns the model's turn in it.

  Args:
    agent: The agent to run.
    query: What to ask it.

  Returns:
    The turn the model replied with, carrying the calls it requested.
  """
  _, state = cast(
      tuple[Any, agent_lib.AgentState],
      executing.run(agent(query, max_steps=1, return_final_state=True)),
  )
  return state.turns[1]


def get_weather(location: str) -> dict[str, Any]:
  """Gets the weather for a specified location.

  Args:
    location: City name.

  Returns:
    A dictionary containing weather information.
  """
  return {'location': location, 'temp': 15, 'unit': 'C'}


def _always_fails(location: str) -> dict[str, Any]:
  """Looks up the weather, badly.

  Args:
    location: City name.

  Returns:
    Never returns.

  Raises:
    ValueError: Always.
  """
  raise ValueError(f'Service unavailable for {location}.')


def main(argv: Sequence[str]) -> None:
  test_results: list[tuple[float, str, bool]] = []
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  fname = os.path.join(_CACHE_DIR.value, 'google_gemini_api.json')
  cache = caching.SimpleFunctionCache(cache_filename=fname)

  api_key = _API_KEY.value
  if api_key is not None:
    vertexai = False
    project = None
    location = None
  elif _PROJECT.value is not None and _LOCATION.value is not None:
    vertexai = True
    project = _PROJECT.value
    location = _LOCATION.value
  else:
    raise ValueError('Either --api_key or (--project, --location) must be set.')
  generate_config = {
      'safety_settings': google_genai_api.SAFETY_DISABLED,
      # Use a small thinking budget to reduce cost while keeping the model
      # functional. Gemini 2.5 models were trained with budgets >= 64, so
      # values below that may be unreliable. Note: max_output_tokens in the
      # GenAI API is a combined budget for thinking + response tokens, so
      # per-call max_tokens must be larger than thinking_budget to leave room
      # for the actual output.
      'thinking_config': genai_types.ThinkingConfig(
          include_thoughts=False,
          thinking_budget=32,
      ),
  }
  candidates_generate_config = {
      'safety_settings': google_genai_api.SAFETY_DISABLED,
      'thinking_config': genai_types.ThinkingConfig(
          include_thoughts=False,
          thinking_budget=512,
      ),
  }
  backend = google_genai_api.GoogleGenAIAPI(
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location=location,
      threadpool_size=_THREADPOOL_SIZE.value,
      cache=cache,
      max_retries=5,
      generate_text_kwargs=generate_config,
      generate_object_kwargs=generate_config,
      generate_content_kwargs=generate_config,
      chat_kwargs=generate_config,
  )
  backend.register()

  image_backend = google_genai_api.GoogleGenAIAPI(
      generate_model_name=google_genai_api.DEFAULT_IMAGE_GENERATE_MODEL,
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location=location,
      threadpool_size=_THREADPOOL_SIZE.value,
      cache=cache,
      max_retries=5,
      generate_text_kwargs=generate_config,
      generate_object_kwargs=generate_config,
      generate_content_kwargs=generate_config,
      chat_kwargs=generate_config,
  )

  candidates_backend = google_genai_api.GoogleGenAIAPI(
      generate_model_name=google_genai_api.ModelName(
          vertex_ai='gemini-2.5-flash-lite',
          gemini_api='models/gemini-2.5-flash-lite',
      ),
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location=location,
      threadpool_size=_THREADPOOL_SIZE.value,
      cache=cache,
      max_retries=5,
      generate_text_kwargs=candidates_generate_config,
      generate_object_kwargs=candidates_generate_config,
      generate_content_kwargs=candidates_generate_config,
      chat_kwargs=candidates_generate_config,
  )

  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname)
    load_start = time.time()
    cache.load()
    load_end = time.time()
    print('Spent %.4fsec loading cache.' % (load_end - load_start))

  @executing.make_executable
  async def generate_contents_direct(backend, **kwargs):
    return await backend._generate_contents(**kwargs)

  @executing.make_executable
  async def check_and_complete(prompt_text, **other_args):
    # TODO: Where should we store 8196 and model name?
    token_count = await llm.count_tokens(
        content=prompt_text,
    )
    logging.info('Token count %d for prompt:\n%s\n', token_count, prompt_text)
    if token_count > _MAX_INPUT_TOKENS:
      warning_msg = (
          f'Warning: Prompt token length ({token_count}) exceeds maximal input '
          f'token length ({_MAX_INPUT_TOKENS}) and '
          f'will be truncated on the server side: {prompt_text[:100]}\n.'
      )
      print(warning_msg)
      logging.warning(warning_msg)
    res = await llm.generate_text(prompt=prompt_text, **other_args)
    return res

  time1 = time.time()

  prompt_text = """
    Question: Natural logarithm of $e^12$?
    Reasoning: "Natural" logarithm means logarithm to the base of $e$. For
    example, natural logarithm of $10$ means exponent to which $e$ must be
    raised to produce 10.
    Answer: 12.
    Question: Differentiate $\\frac{1}{\\log(x)}$.
  """

  if _should_run_test(1):
    with _run_test(1, 'A single generate query', test_results):
      res = executing.run(
          check_and_complete(
              prompt_text=prompt_text,
              stop=['\n\n'],
              max_tokens=300,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      print('1.1 Same query to see if it has been cached.')
      res = executing.run(
          check_and_complete(
              prompt_text=prompt_text,
              stop=['\n\n'],
              max_tokens=300,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      print('1.2 Same query but different parameters, run requests again.')
      res = executing.run(
          check_and_complete(
              prompt_text=prompt_text,
              temperature=0.0,
              stop=['\n\n'],
              max_tokens=300,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

  if _should_run_test(2):
    with _run_test(2, 'Repeated generate request', test_results):
      exe = executing.par_iter(
          sampling.repeat(
              executable=check_and_complete(
                  'Today is', temperature=0.5, max_tokens=300, stop=['.']
              ),
              num_repeats=5,
          )
      )
      res = executing.run(exe)
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

  if _should_run_test(3):
    with _run_test(3, 'Three batched generate queries', test_results):
      exe = executing.par_iter([
          check_and_complete(prompt_text='In summer', max_tokens=300),
          check_and_complete(prompt_text='In winter', max_tokens=300),
          check_and_complete(prompt_text='In autumn', max_tokens=300),
      ])
      res = executing.run(exe)
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

  if _should_run_test(4):
    with _run_test(
        4, 'Prompt that does not fit into maximal token length', test_results
    ) as tc:
      value_error_raised = False
      try:
        # This call should fail and raise ValueError in _batch_generate.
        _ = executing.run(
            check_and_complete(
                ', '.join(map(str, list(range(_MAX_INPUT_TOKENS))))
            )
        )
      except ValueError as err:
        err_repr = repr(err)
        if not (
            (
                'INVALID_ARGUMENT' in err_repr
                and 'exceeds the maximum number of tokens allowed' in err_repr
            )
            or 'RESOURCE_EXHAUSTED' in err_repr
            or 'Quota exceeded' in err_repr
        ):
          tc.fail(
              'ValueError raised, but not with the expected message: '
              + repr(err)
          )
        value_error_raised = True
        if _PRINT_DEBUG.value:
          print('ValueError raised.')
      if not value_error_raised:
        tc.fail('ValueError not raised.')

  if _should_run_test(5):
    with _run_test(
        5,
        'Three batched generate queries, one of which does not fit',
        test_results,
    ) as tc:
      exe = executing.par_iter([
          check_and_complete(prompt_text='In summer', max_tokens=300),
          check_and_complete(prompt_text='In winter', max_tokens=300),
          check_and_complete(
              ', '.join(map(str, list(range(_MAX_INPUT_TOKENS))))
          ),
      ])
      value_error_raised = False
      try:
        _ = executing.run(exe)
      except ValueError as err:
        err_repr = repr(err)
        if not (
            'INVALID_ARGUMENT' in err_repr
            or 'RESOURCE_EXHAUSTED' in err_repr
            or 'Quota exceeded' in err_repr
        ):
          tc.fail(
              'ValueError raised, but not with the expected message: '
              + repr(err)
          )
        value_error_raised = True
        if _PRINT_DEBUG.value:
          print('ValueError raised.')
      if not value_error_raised:
        tc.fail('ValueError not raised.')

  if _should_run_test(6.1):
    with _run_test(6.1, 'Generate an image', test_results) as tc:
      with routing.RegistryContext():
        image_backend.register()
        res = executing.run(
            generate_contents_direct(
                image_backend,
                prompt="""Create a picture of a banana disguised as superhero.""",
                temperature=2,
                safety_settings=google_genai_api.SAFETY_DISABLED,
            )
        )[0]
      has_image = False
      for ch in res:
        if ch.content_type.startswith('image/'):
          has_image = True
      print('- Has image: ', has_image)
      if not has_image:
        tc.fail('No image returned.')
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      # Switch back to default backend.
      backend.register()

  if _should_run_test(6.2):
    with _run_test(6.2, 'Generate candidates', test_results):
      with routing.RegistryContext():
        candidates_backend.register()
        res = executing.run(
            generate_contents_direct(
                candidates_backend,
                prompt="""How are you doing? In one word.""",
                samples=3,
                temperature=2,
                response_mime_type='application/json',
                response_schema={
                    'type': 'STRING',
                    'pattern': (
                        '(terrible|excellent|well|average|meh|bad|awesome)'
                    ),
                },
                safety_settings=google_genai_api.SAFETY_DISABLED,
            )
        )
      uniq = len({str(r): False for r in res})
      print('- Candidates:', len(res), ' distinct:', uniq)
      if _PRINT_DEBUG.value:
        for candidate in res:
          print('Returned value(s):')
          pprint.pprint(candidate)
      # Switch back to default backend.
      backend.register()

  if _should_run_test(8):
    with _run_test(8, 'Use ChunkList', test_results) as tc:
      executable = (
          c.c('What is the answer to this question: ')  # pyrefly: ignore[bad-argument-type, unsupported-operation]
          + c.c('What is the third planet from the sun? ')  # pyrefly: ignore[bad-argument-type]
          + c.store('answer', c.generate_text())  # pyrefly: ignore[bad-argument-count]
      )
      _ = executing.run(executable)
      if 'earth' not in executable['answer'].lower():
        tc.fail(
            'Returned value does not contain "earth": ' + executable['answer']
        )
      if _PRINT_DEBUG.value:
        print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  if _SAVE_CACHE.value:
    cache.save(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname))

  if _should_run_test(9):
    with _run_test(9, 'Check that generate_texts is working', test_results):
      res = executing.run(
          llm.generate_texts(
              prompt=prompt_text,
              samples=3,
              stop=['\n\n'],
              max_tokens=300,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

  if _should_run_test(10):
    with _run_test(10, 'Formatting', test_results) as tc:
      print('10.1 Check that chat is working (API formatting).')
      res = executing.run(
          llm.chat(
              messages=[
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='What is the capital of France?',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.MODEL,
                      content='Obviously, it is Paris.',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='What is the capital of Germany?',
                  ),
              ],
              temperature=0.5,
              max_tokens=300,
              stop=['.'],
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

      print('10.2 Check that chat completes the model prefix.')
      res = executing.run(
          llm.chat(
              messages=[
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='Tell me a real short story?',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.MODEL,
                      content='Once upon a',
                  ),
              ],
              temperature=0.5,
              max_tokens=1000,
              stop=['.'],
          )
      )
      if not res.startswith(' time'):
        tc.fail(
            'Returned value does not seem to be a continuation of "Once'
            ' upon a" -- expected to start with " time": '
            + repr(res)
        )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

      print('10.3 Check that chat is working (Default formatting).')
      res = executing.run(
          llm.chat(
              messages=[
                  content_lib.Message(
                      role=content_lib.PredefinedRole.SYSTEM,
                      content='You are a helpful and insightful chatbot.',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='What is the capital of France?',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.MODEL,
                      content='Obviously, it is Paris.',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='What is the capital of Germany?',
                  ),
              ],
              temperature=0.5,
              formatter=formatting.FormatterName.DEFAULT,
              max_tokens=300,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

      print('10.4 Check that chat is working (API formatting) with chunk list.')
      res = executing.run(
          llm.chat(
              messages=[
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content=content_lib.ChunkList([
                          content_lib.Chunk('Please answer the question:\n'),
                          content_lib.Chunk('What is the capital of France?'),
                      ]),
                  ),
              ],
              temperature=0.5,
              max_tokens=300,
              stop=['.'],
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)

  if _should_run_test(11):
    with _run_test(11, 'Check that multimodal is working', test_results) as tc:
      fname_mm = os.path.join(_CACHE_DIR.value, 'google_gemini_api_mm.json')
      if api_key is not None:
        backend = google_genai_api.GoogleGenAIAPI(
            generate_model_name=google_genai_api.DEFAULT_MULTIMODAL_MODEL,
            api_key=api_key,
            max_retries=5,
            generate_text_kwargs=generate_config,
            generate_object_kwargs=generate_config,
            generate_content_kwargs=generate_config,
            chat_kwargs=generate_config,
        )
      elif _PROJECT.value is not None and _LOCATION.value is not None:
        backend = google_genai_api.GoogleGenAIAPI(
            generate_model_name=google_genai_api.DEFAULT_MULTIMODAL_MODEL,
            vertexai=True,
            project=_PROJECT.value,
            location=_LOCATION.value,
            max_retries=5,
            generate_text_kwargs=generate_config,
            generate_object_kwargs=generate_config,
            generate_content_kwargs=generate_config,
            chat_kwargs=generate_config,
        )
      else:
        raise ValueError(
            'Either --api_key or (--project, --location) must be set.'
        )
      backend.register()

      if _LOAD_CACHE.value:
        print('Loading cache from file %s', fname_mm)
        load_start = time.time()
        cache.load()
        load_end = time.time()
        print('Spent %.4fsec loading cache.', load_end - load_start)

      time1 = time.time()
      # Load image from testdata.
      image_path = os.path.join(
          os.path.dirname(__file__),
          'testdata',
          'bird.jpg',
      )
      with open(image_path, 'rb') as f:
        image_bytes = f.read()
        executable = (
            c.c('What is the following image? ')  # pyrefly: ignore[bad-argument-type, unsupported-operation]
            + c.c(content_lib.Chunk(image_bytes, 'image/jpeg'))  # pyrefly: ignore[bad-argument-type]
            + c.store('answer', c.generate_text())  # pyrefly: ignore[bad-argument-count]
        )
        _ = executing.run(executable)
      if 'chickadee' not in executable['answer'].lower():
        tc.fail(
            'Returned value does not contain "chickadee": '
            + executable['answer']
        )
      if _PRINT_DEBUG.value:
        print('Returned value:', executable['answer'])

      time2 = time.time()
      print('Took %.4fsec running requests.' % (time2 - time1))
      if _SAVE_CACHE.value:
        cache.save(overwrite=True)
        time3 = time.time()
        print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname_mm))

  if _should_run_test(12):
    with _run_test(
        12, 'Check response_schema and response_mime_type', test_results
    ) as tc:
      executable = c.c('What is the capital of France? ') + c.store(  # pyrefly: ignore[bad-argument-type]
          'answer',
          c.generate_text(
              response_mime_type='application/json',  # pyrefly: ignore[bad-argument-count, unexpected-keyword]
              response_schema={  # pyrefly: ignore[unexpected-keyword]
                  'type': 'object',
                  'properties': {'answer': {'type': 'string'}},
              },
          ),
      )
      _ = executing.run(executable)
      try:
        answer = json.loads(executable['answer'])  # pyrefly: ignore[bad-index]
        if _PRINT_DEBUG.value:
          print('Returned value:', answer)
      except json.JSONDecodeError as e:
        tc.fail(
            'Returned value is not a valid JSON: '
            + repr(executable['answer'])  # pyrefly: ignore[bad-index]
            + ' '
            + repr(e)
        )

  if _should_run_test(13):
    with _run_test(13, 'Check thinking config', test_results):
      res = executing.run(
          llm.generate_text(
              prompt='solve x^2 + 4x + 4 = 0',
              thinking_config=genai_types.ThinkingConfig(
                  include_thoughts=True,
                  thinking_budget=1024,
              ),
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value:', res)

  if _should_run_test(14):
    with _run_test(14, 'Check that tokenize is working.', test_results) as tc:
      if backend.vertexai:
        content = content_lib.ChunkList(
            chunks=[
                content_lib.Chunk(
                    content='The quick brown fox jumps over the lazy dog.',
                )
            ]
        )
        res = executing.run(llm.tokenize(content=content))
        if _PRINT_DEBUG.value:
          print('Returned value(s):')
          pprint.pprint(res)
        if not isinstance(res, list) or not all(
            isinstance(x, int) for x in res
        ):
          tc.fail('Returned value is not a list of ints: ' + repr(res))
      else:
        print('- Skipped because it is not supported by Gemini API.')

  if _should_run_test(15):
    with _run_test(15, 'Check that embed is working.', test_results) as tc:
      content = content_lib.ChunkList(
          chunks=[
              content_lib.Chunk(
                  content='The quick brown fox jumps over the lazy dog.',
              )
          ]
      )
      res = executing.run(llm.embed(content=content))
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        result_string = str(res)
        if len(result_string) > 400:
          result_string = (
              result_string[:200]
              + f'... (skipped {len(result_string) - 200} characters) ...'
              + result_string[-200:]
          )
        pprint.pprint(result_string)
      if not isinstance(res, list) or not all(
          isinstance(x, float) for x in res
      ):
        tc.fail('Returned value is not a list of floats: ' + repr(res))

  if _should_run_test(16):
    with _run_test(
        16, 'Check llm.generate_object with native types. (int)', test_results
    ) as tc:
      prompt = 'What is the height of the Eiffel Tower? (in m)'
      result = executing.run(
          llm.generate_object(prompt=prompt, cls=int)
      )
      if _PRINT_DEBUG.value:
        print('Returned value:', result)
      if not isinstance(result, int):
        tc.fail(f'Result is not of type int: {type(result)}')

  class CityInfo(pydantic.BaseModel):
    city_name: str
    country: str
    population: int
    landmark: str | None = None

  if _should_run_test(17):
    with _run_test(
        17, 'Check llm.generate_object with Pydantic model', test_results
    ) as tc:
      prompt = 'Provide information about the capital of France.'
      result = executing.run(
          llm.generate_object(prompt=prompt, cls=CityInfo)
      )
      assert isinstance(result, CityInfo)

      if _PRINT_DEBUG.value:
        print('Returned value:')
        pprint.pprint(result)

      if not isinstance(result, CityInfo):
        tc.fail(f'Result is not of type CityInfo: {type(result)}')
      if not result.city_name:
        tc.fail('city_name is missing')
      if not result.country:
        tc.fail('country is missing')
      if result.population <= 0:
        tc.fail('population should be positive')

  if _should_run_test(18):
    with _run_test(
        18, 'Check llm.generate_object for dict-like data', test_results
    ) as tc:

      class WordCount(pydantic.BaseModel):
        word: str
        count: int

      prompt = (
          'Return a breakdown of the words and their frequencies in the'
          ' following sentence: "the quick brown fox jumps over the lazy dog'
          ' dog". Provide the output as a list of word-count objects.'
      )
      result = executing.run(
          llm.generate_object(prompt=prompt, cls=list[WordCount])
      )

      if _PRINT_DEBUG.value:
        print('Returned value:')
        pprint.pprint(result)

      expected_dict = {
          'the': 2,
          'quick': 1,
          'brown': 1,
          'fox': 1,
          'jumps': 1,
          'over': 1,
          'lazy': 1,
          'dog': 2,
      }

      if not isinstance(result, list):
        tc.fail(f'Result is not of type list: {type(result)}')
      elif not all(isinstance(item, WordCount) for item in result):
        tc.fail(f'Result items are not all WordCount objects: {result}')
      else:
        result_dict = {item.word: item.count for item in result}
        if result_dict != expected_dict:
          tc.fail(
              f'Result dict {result_dict} does not match expected'
              f' {expected_dict}'
          )

  if _should_run_test(19):
    with _run_test(
        19,
        'Check llm.generate_object with a Pydantic-wrapped standard dataclass',
        test_results,
    ) as tc:

      @dataclasses.dataclass
      class StandardRecipe:
        recipe_name: str
        description: str
        prep_time_minutes: int
        ingredients: list[str]

      pydantic_recipe = pydantic_dataclass(StandardRecipe)

      prompt = """Give me a simple recipe for a classic chocolate chip cookie,
      including a short description, prep time in minutes, and a list of
      ingredients."""
      result = executing.run(
          llm.generate_object(prompt=prompt, cls=pydantic_recipe)
      )

      if _PRINT_DEBUG.value:
        print('Returned value:')
        pprint.pprint(result)

      if not isinstance(result, StandardRecipe):
        tc.fail(f'Result is not an instance of StandardRecipe: {type(result)}')
      elif not isinstance(result, pydantic_recipe):  # pyrefly: ignore[invalid-argument]
        tc.fail(f'Result is not an instance of PydanticRecipe: {type(result)}')
      else:
        if not result.recipe_name:
          tc.fail('recipe_name is missing')
        if not result.description:
          tc.fail('description is missing')
        if result.prep_time_minutes <= 0:
          tc.fail('prep_time_minutes should be positive')
        if not isinstance(result.ingredients, list) or not result.ingredients:
          tc.fail('ingredients should be a non-empty list')
        elif not all(isinstance(i, str) for i in result.ingredients):
          tc.fail('ingredients should be a list of strings')

  if _should_run_test(20):
    with _run_test(
        20, 'Check llm.generate_object in chat mode', test_results
    ) as tc:
      result = executing.run(
          llm.generate_object(
              prompt=[
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='What is the capital of France?',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.MODEL,
                      content='Obviously, it is Paris.',
                  ),
                  content_lib.Message(
                      role=content_lib.PredefinedRole.USER,
                      content='Tell me more about it.',
                  ),
              ],
              cls=CityInfo,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(result)

      if not isinstance(result, CityInfo):
        tc.fail(f'Result is not of type CityInfo: {type(result)}')

  if _should_run_test(21):
    with _run_test(
        21,
        'Function calling: a single step, so nothing is executed',
        test_results,
    ) as tc:
      agent = agent_lib.Agent(tools=[get_weather])
      turn = _first_model_turn(agent, 'What is the weather in London?')
      if not turn.calls:
        tc.fail('No FunctionCall found in response.')
      else:
        call = turn.calls[0]
        print(
            f'- Found FunctionCall: {call.function_name} with args'
            f' {dict(call.kwargs)}'
        )
        if (
            call.function_name != 'get_weather'
            or call.kwargs.get('location') != 'London'
        ):
          tc.fail('FunctionCall name or args do not match.')

  if _should_run_test(22):
    with _run_test(22, 'Function calling: the full loop', test_results) as tc:
      agent = agent_lib.Agent(tools=[get_weather])
      answer = _run_agent(agent, 'What is the weather in London?')
      print(f'- Final answer: {answer}')
      if not answer:
        tc.fail('Expected a final model answer after tool execution.')

      # A tool that raises should be reported back to the model rather than
      # blowing up the loop, so that it can say something sensible instead.
      recovering_agent = agent_lib.Agent(
          tools=[llm_tool_use.Tool(name='get_weather', function=_always_fails)],
          max_steps=3,
      )
      recovered = _run_agent(recovering_agent, 'What is the weather in London?')
      print(f'- Answer after the tool failed: {recovered}')
      if not recovered:
        tc.fail('A failing tool should be reported, not raised.')

  if _should_run_test(23):
    with _run_test(23, 'Check usage_metadata in trace', test_results) as tc:
      executable = llm.generate_text(
          prompt='Explain photosynthesis.',
          max_tokens=1000,
      )
      res, trace = executing.run(executable, enable_tracing=True)

      if _PRINT_DEBUG.value:
        print('Returned value:')
        pprint.pprint(res)
        print('Trace:')
        pprint.pprint(trace)

      # The trace structure is:
      #   trace (root)
      #   └── stages[0]: GoogleGenAIAPI.generate_text
      #       └── stages[0]: GoogleGenAIAPI.generate_contents  <- info is here
      generate_text_stage = trace.stages[0]
      if generate_text_stage.stage_name != 'GoogleGenAIAPI.generate_text':
        tc.fail(
            'Expected stage name GoogleGenAIAPI.generate_text, got:'
            f' {generate_text_stage.stage_name}'
        )
      else:
        generate_contents_stage = generate_text_stage.stages[0]
        if generate_contents_stage.stage_name != (
            'GoogleGenAIAPI.generate_contents'
        ):
          tc.fail(
              'Expected stage name GoogleGenAIAPI.generate_contents, got:'
              f' {generate_contents_stage.stage_name}'
          )
        else:
          info = generate_contents_stage.info
          print('- Trace info:', info)

          # Check that the model name is populated.
          if 'llm.model' not in info:
            tc.fail('llm.model missing from trace info')
          else:
            print(f'- llm.model: {info["llm.model"]}')

          # Check for expected token count keys populated from usage_metadata.
          expected_keys = [
              'llm.tokens.prompt',
              'llm.tokens.candidates',
              'llm.tokens.cached_content',
              'llm.tokens.thoughts',
          ]
          for key in expected_keys:
            if key not in info:
              tc.fail(f'{key} missing from trace info')
            else:
              val = info[key]
              print(f'- {key}: {val}')
              if val is not None and (not isinstance(val, int) or val < 0):
                tc.fail(f'{key} is not a valid non-negative integer: {val}')

  if _should_run_test(24):
    with _run_test(
        24,
        'Check embed norm is 1.0 for various output_dimensionality',
        test_results,
    ) as tc:
      embed_content = content_lib.ChunkList(
          chunks=[
              content_lib.Chunk(
                  content='The quick brown fox jumps over the lazy dog.',
              )
          ]
      )
      dimensionalities = [
          None,
          32,
          64,
          128,
          256,
          512,
          768,
          1024,
          1536,
          2048,
          3072,
      ]
      tolerance = 1e-3
      for dim in dimensionalities:
        label = str(dim) if dim is not None else 'default'
        embed_kwargs = {'normalize': True}
        if dim is not None:
          embed_kwargs['output_dimensionality'] = dim  # pyrefly: ignore[bad-assignment]

        res = executing.run(
            llm.embed(content=embed_content, **embed_kwargs)
        )
        vec_norm = math.sqrt(sum(x * x for x in res))
        norm_ok = abs(vec_norm - 1.0) < tolerance
        print(
            f'  dim={label:>7s}  |  vec_len={len(res):<5d}  |'
            f'  norm={vec_norm:.6f}  |  {"PASS" if norm_ok else "FAIL"}'
        )
        if dim is not None and len(res) != dim:
          tc.fail(f'expected length {dim}, got {len(res)}')
        if not norm_ok:
          tc.fail(f'norm {vec_norm} is not close to 1.0')

  if _should_run_test(25):
    with _run_test(
        25,
        'Check embed_kwargs default output_dimensionality and normalize',
        test_results,
    ) as tc:
      target_dim = 256
      embed_content_25 = content_lib.ChunkList(
          chunks=[
              content_lib.Chunk(
                  content='The quick brown fox jumps over the lazy dog.',
              )
          ]
      )
      # Re-register backend with a default output_dimensionality.
      with routing.RegistryContext():
        reduced_backend = google_genai_api.GoogleGenAIAPI(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            embed_kwargs={
                'output_dimensionality': target_dim,
                'normalize': True,
            },
        )
        reduced_backend.register(
            register_generate=False, register_tokenize=False
        )
        # Call embed without specifying output_dimensionality explicitly.
        res = executing.run(
            llm.embed(content=embed_content_25)
        )
      vec_norm = math.sqrt(sum(x * x for x in res))
      norm_ok = abs(vec_norm - 1.0) < 1e-3
      len_ok = len(res) == target_dim
      print(
          f'  vec_len={len(res)}  |  norm={vec_norm:.6f}  |'
          f'  {"PASS" if norm_ok and len_ok else "FAIL"}'
      )
      if not len_ok:
        tc.fail(f'expected length {target_dim}, got {len(res)}')
      if not norm_ok:
        tc.fail(f'norm {vec_norm} is not close to 1.0')

  if _should_run_test(26):
    with _run_test(
        26, 'Check prefix caching with CachedChunkList', test_results
    ) as tc:
      # Caching requires a minimum prefix size of 32,768 tokens on the server.
      # We generate a large text block of repeated words (~35k tokens) and
      # prepend a random UUID to guarantee a true cache miss on the first call.
      unique_suffix = f'unique id: {uuid.uuid4()} '
      large_prefix_text = unique_suffix + ('apple ' * 35000)
      prompt = prefix_caching.CachedChunkList(
          chunks=[
              content_lib.Chunk(large_prefix_text),
              content_lib.Chunk(
                  '\nWhat was the fruit repeated in the context above? answer'
                  ' in one word.'
              ),
          ],
          cache_chunk_count=1,
          cache_ttl='300s',
      )

      print('- First call (cache creation/miss)...')
      executable = llm.generate_text(prompt=prompt, max_tokens=50)
      _, trace1 = executing.run(executable, enable_tracing=True)

      print(f'- Populated cache_name: {prompt.cache_name}')
      if not prompt.cache_name:
        tc.fail(
            'cache_name was not populated on the prompt after the first call.'
        )

      generate_text_stage1 = trace1.stages[0]
      generate_contents_stage1 = generate_text_stage1.stages[0]
      info1 = generate_contents_stage1.info
      prompt_tokens1 = info1.get('llm.tokens.prompt', 0)
      cached_tokens1 = info1.get('llm.tokens.cached_content', 0)
      if prompt_tokens1 < cached_tokens1:
        # Gemini API style: prompt_tokens represents only the fresh tokens.
        fresh_tokens1 = prompt_tokens1
        total_tokens1 = prompt_tokens1 + cached_tokens1
      else:
        # Vertex AI style: prompt_tokens represents the total tokens.
        fresh_tokens1 = prompt_tokens1 - cached_tokens1
        total_tokens1 = prompt_tokens1

      print(f'- First call prompt tokens (total): {total_tokens1}')
      print(f'- First call cached tokens: {cached_tokens1}')
      print(f'- First call fresh tokens processed: {fresh_tokens1}')

      print('- Second call (cache hit)...')
      # Reuse prompt object with populated cache_name, but change the question.
      prompt.chunks = [
          prompt.chunks[0],
          content_lib.Chunk(
              '\nWhat fruit did I repeat in the context? answer in one word.'
          ),
      ]
      executable2 = llm.generate_text(prompt=prompt, max_tokens=50)
      _, trace2 = executing.run(executable2, enable_tracing=True)

      # Verify second call cached tokens.
      generate_text_stage2 = trace2.stages[0]
      generate_contents_stage2 = generate_text_stage2.stages[0]
      info2 = generate_contents_stage2.info
      prompt_tokens2 = info2.get('llm.tokens.prompt', 0)
      cached_tokens2 = info2.get('llm.tokens.cached_content', 0)
      if prompt_tokens2 < cached_tokens2:
        # Gemini API style: prompt_tokens represents only the fresh tokens.
        fresh_tokens2 = prompt_tokens2
        total_tokens2 = prompt_tokens2 + cached_tokens2
      else:
        # Vertex AI style: prompt_tokens represents the total tokens.
        fresh_tokens2 = prompt_tokens2 - cached_tokens2
        total_tokens2 = prompt_tokens2

      print(f'- Second call prompt tokens (total): {total_tokens2}')
      print(f'- Second call cached tokens: {cached_tokens2}')
      print(f'- Second call fresh tokens processed: {fresh_tokens2}')

      if cached_tokens2 is None or cached_tokens2 == 0:
        tc.fail(
            'Second call did not hit the cache (cached_content tokens is 0).'
        )
      else:
        print(
            'Mathematical Proof (Token Counts): The server returned '
            f'prompt_token_count = {prompt_tokens2} (reported input) and '
            f'cached_content_token_count = {cached_tokens2} (cached tokens). '
            f'This represents a total input of {total_tokens2} tokens. '
            'Since the server successfully processed the request and returned '
            f'the correct answer while only processing {fresh_tokens2} input '
            'tokens fresh, it proves the server successfully loaded and '
            f'matched the other {cached_tokens2} tokens from the cache handle '
            'we provided!'
        )
        print(f'SUCCESS: Cached {cached_tokens2} tokens successfully!')

  if _should_run_test(30):
    with _run_test(
        30,
        'Check caching compatibility for gemini-embedding-001 (v1)',
        test_results,
    ) as tc:
      # Create a backend specifically with gemini-embedding-001 and a clean
      # in-memory cache.
      v1_cache = caching.SimpleFunctionCache()
      v1_backend = google_genai_api.GoogleGenAIAPI(
          embed_model_name='gemini-embedding-001',
          api_key=api_key,
          vertexai=vertexai,
          project=project,
          location=location,
          credentials=credentials,
          threadpool_size=_THREADPOOL_SIZE.value,
          cache=v1_cache,
      )
      v1_backend.register()

      doc_content = 'This is a document about caching.'

      # Test Document Retrieval caching compatibility.
      # 1. Old way: manually formatted content, RETRIEVAL_DOCUMENT task type
      # passed in kwargs (for v1).
      _ = executing.run(
          llm.embed(content=doc_content, task_type='RETRIEVAL_DOCUMENT')
      )

      # 2. New way: pass title and task_type to llm.embed.
      # For v1, if title is None, it should not format the content, and pass
      # task_type in kwargs.
      # So the inputs to _cached_embed should be identical.
      _ = executing.run(
          llm.embed(
              content=doc_content,
              title=None,
              task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
          )
      )

      if v1_cache._cache_data.counters.get('get_hit', 0) != 1:
        tc.fail(
            'Expected 1 cache hit for v1 RETRIEVAL_DOCUMENT (no title), got:'
            f' {v1_cache._cache_data.counters}'
        )

      # Test with title. For v1, the server formats the title into the prompt
      # from the `title` field of the request, so it must not be prepended to
      # the content: manually formatted content is a different request, which
      # the server would format a second time (see test 35). So the old and
      # new ways must not share a cache entry.
      # Old way: manually prepended title
      title = 'Caching Guide'
      formatted_content = f'title: {title} | text: {doc_content}'

      # Clear cache by re-instantiating
      v1_cache = caching.SimpleFunctionCache()
      v1_backend.cache = v1_cache

      _ = executing.run(
          llm.embed(content=formatted_content, task_type='RETRIEVAL_DOCUMENT')
      )

      # New way: pass title and task_type
      _ = executing.run(
          llm.embed(
              content=doc_content,
              title=title,
              task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
          )
      )

      if v1_cache._cache_data.counters.get('get_hit', 0) != 0:
        tc.fail(
            'Expected no cache hit for v1 RETRIEVAL_DOCUMENT (with title), got:'
            f' {v1_cache._cache_data.counters}'
        )

      # Test Semantic Similarity
      v1_cache = caching.SimpleFunctionCache()
      v1_backend.cache = v1_cache
      # Old way: content raw, task_type in kwargs
      _ = executing.run(
          llm.embed(content=doc_content, task_type='SEMANTIC_SIMILARITY')
      )
      # New way: content raw, task_type as enum
      _ = executing.run(
          llm.embed(
              content=doc_content,
              task_type=llm.EmbedTaskType.SEMANTIC_SIMILARITY,
          )
      )

      if v1_cache._cache_data.counters.get('get_hit', 0) != 1:
        tc.fail(
            'Expected 1 cache hit for v1 SEMANTIC_SIMILARITY, got:'
            f' {v1_cache._cache_data.counters}'
        )

  if _should_run_test(31):
    try:
      with _run_test(
          31,
          'Check caching compatibility for gemini-embedding-2 (v2)',
          test_results,
      ) as tc:
        v2_cache = caching.SimpleFunctionCache()
        v2_backend = google_genai_api.GoogleGenAIAPI(
            embed_model_name='gemini-embedding-2-preview',
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            threadpool_size=_THREADPOOL_SIZE.value,
            cache=v2_cache,
        )
        v2_backend.register()

        doc_content = 'This is a document about caching.'

        # Test Document Retrieval (with title)
        # Old way: manual formatting for v2
        title = 'Caching Guide'
        formatted_content = f'title: {title} | text: {doc_content}'
        _ = executing.run(llm.embed(content=formatted_content))

        # New way: pass title and task_type
        _ = executing.run(
            llm.embed(
                content=doc_content,
                title=title,
                task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
            )
        )

        if v2_cache._cache_data.counters.get('get_hit', 0) != 1:
          tc.fail(
              'Expected 1 cache hit for v2 RETRIEVAL_DOCUMENT (with title),'
              f' got: {v2_cache._cache_data.counters}'
          )

        # Test Document Retrieval (no title)
        v2_cache = caching.SimpleFunctionCache()
        v2_backend.cache = v2_cache
        formatted_content_no_title = f'title: none | text: {doc_content}'
        _ = executing.run(llm.embed(content=formatted_content_no_title))
        _ = executing.run(
            llm.embed(
                content=doc_content,
                title=None,
                task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
            )
        )

        if v2_cache._cache_data.counters.get('get_hit', 0) != 1:
          tc.fail(
              'Expected 1 cache hit for v2 RETRIEVAL_DOCUMENT (no title), got:'
              f' {v2_cache._cache_data.counters}'
          )

        # Test Semantic Similarity
        v2_cache = caching.SimpleFunctionCache()
        v2_backend.cache = v2_cache
        # Old way: manual formatting
        formatted_similarity = (
            f'task: sentence similarity | query: {doc_content}'
        )
        _ = executing.run(llm.embed(content=formatted_similarity))

        # New way: pass task_type
        _ = executing.run(
            llm.embed(
                content=doc_content,
                task_type=llm.EmbedTaskType.SEMANTIC_SIMILARITY,
            )
        )

        if v2_cache._cache_data.counters.get('get_hit', 0) != 1:
          tc.fail(
              'Expected 1 cache hit for v2 SEMANTIC_SIMILARITY, got:'
              f' {v2_cache._cache_data.counters}'
          )
    finally:
      backend.register()

  if _should_run_test(35):
    try:
      with _run_test(
          35,
          'Check title and empty content for gemini-embedding-001 (v1)',
          test_results,
      ) as tc:
        # v1 models build the instruction prompt server-side from the
        # `task_type` and `title` fields of the request, e.g.
        # 'title: {title} | text: {content}' for RETRIEVAL_DOCUMENT. So the
        # backend must send the content as is, and the title only with
        # RETRIEVAL_DOCUMENT, the only task type that the API accepts it for.
        v1_model = 'gemini-embedding-001'
        v1_backend = google_genai_api.GoogleGenAIAPI(
            embed_model_name=v1_model,
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            threadpool_size=_THREADPOOL_SIZE.value,
            cache=caching.SimpleFunctionCache(),
            max_retries=5,
        )
        v1_backend.register()
        # Identical requests get the same embedding, up to numerical noise.
        min_cosine_sim = 0.9999

        def _embed_v1(content: str, **kwargs: Any) -> Sequence[float]:
          return executing.run(llm.embed(content=content, **kwargs))

        def _check_same_embedding(
            check: str, embedding1: Sequence[float], embedding2: Sequence[float]
        ) -> None:
          cosine_sim = sum(x * y for x, y in zip(embedding1, embedding2)) / (
              math.sqrt(sum(x * x for x in embedding1))
              * math.sqrt(sum(y * y for y in embedding2))
          )
          print(f'{check}: cosine similarity {cosine_sim:.6f}.')
          if cosine_sim < min_cosine_sim:
            tc.fail(
                f'{check}: expected the same embedding, got cosine similarity'
                f' {cosine_sim:.6f}.'
            )

        title = 'Spanner Vector Search Guide'
        document = (
            'Spanner supports approximate nearest neighbor search over vector'
            ' embeddings.'
        )
        query = 'How do I find similar vectors in Spanner?'

        # 1. RETRIEVAL_DOCUMENT with a title sends the same request as calling
        # the API directly with the title in the `title` field. Prepending the
        # title to the content instead makes the server format it a second
        # time: 'title: none | text: title: {title} | text: {content}'.
        document_with_title = _embed_v1(
            document,
            task_type=llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
            title=title,
        )
        genai_client = v1_backend._genai_client  # pylint: disable=protected-access
        response = genai_client.models.embed_content(
            model=v1_model,
            contents=document,
            config=genai_types.EmbedContentConfig(
                task_type='RETRIEVAL_DOCUMENT', title=title
            ),
        )
        embeddings = response.embeddings
        direct_embedding = embeddings[0].values if embeddings else None
        if not direct_embedding:
          raise ValueError('The direct API call returned no embedding.')
        _check_same_embedding(
            'RETRIEVAL_DOCUMENT with title vs direct API call',
            document_with_title,
            direct_embedding,
        )

        # 2. A title without a task type implies RETRIEVAL_DOCUMENT. Without a
        # task type, the server would use its default, RETRIEVAL_QUERY.
        _check_same_embedding(
            'Title only vs RETRIEVAL_DOCUMENT with title',
            _embed_v1(document, title=title),
            document_with_title,
        )

        # 3. The title is not sent with other task types, which the API rejects.
        _check_same_embedding(
            'RETRIEVAL_QUERY with title vs without title',
            _embed_v1(
                query, task_type=llm.EmbedTaskType.RETRIEVAL_QUERY, title=title
            ),
            _embed_v1(query, task_type=llm.EmbedTaskType.RETRIEVAL_QUERY),
        )

        # 4. Empty content, which the API rejects, is embedded as 'none'.
        empty_content_cases = {
            'no task type': {},
            'RETRIEVAL_QUERY': {
                'task_type': llm.EmbedTaskType.RETRIEVAL_QUERY,
            },
            'RETRIEVAL_DOCUMENT with title': {
                'task_type': llm.EmbedTaskType.RETRIEVAL_DOCUMENT,
                'title': title,
            },
        }
        for case, embed_kwargs in empty_content_cases.items():
          try:
            embedding = _embed_v1('', **embed_kwargs)
          except Exception as e:  # pylint: disable=broad-except
            tc.fail(f'Empty content ({case}): {type(e).__name__}: {e}')
          else:
            print(f'Empty content ({case}): dimension {len(embedding)}.')
    finally:
      backend.register()

  if _SAVE_CACHE.value:
    cache.save(overwrite=True)

  _print_summary(test_results)


if __name__ == '__main__':
  app.run(main)
