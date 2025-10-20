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

from collections.abc import Sequence
import dataclasses
import json
import os
import pprint
import time

from absl import app
from absl import flags
from absl import logging
from google.genai import types as genai_types
from onetwo.backends import google_genai_api
from onetwo.builtins import composables as c
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling
from PIL import Image
import pydantic
import pydantic.dataclasses as pydantic_dataclasses


pydantic_dataclass = pydantic_dataclasses.dataclass



# gemini-2.5-flash.
# See https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash  # pylint: disable=line-too-long
_MAX_INPUT_TOKENS = 1048576

_API_KEY = flags.DEFINE_string('api_key', default=None, help='GenAI API key.')
_PROJECT = flags.DEFINE_string('project', default=None, help='Project ID.')
_LOCATION = flags.DEFINE_string('location', default=None, help='Location.')
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


def main(argv: Sequence[str]) -> None:
  success = True
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
  backend = google_genai_api.GoogleGenAIAPI(
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location=location,
      threadpool_size=_THREADPOOL_SIZE.value,
      cache=cache,
  )
  backend.register()

  # Disable (or limit) thinking by default to reduce cost and avoid subtle
  # interactions with the `max_tokens` parameter (which gets applied to the sum
  # of the thinking tokens and actual output tokens).
  thinking_config = genai_types.ThinkingConfig(
      include_thoughts=False,
      thinking_budget=0,
  )
  llm.generate_text.update(thinking_config=thinking_config)
  llm.generate_object.update(thinking_config=thinking_config)
  llm.chat.update(thinking_config=thinking_config)

  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname)
    load_start = time.time()
    cache.load()
    load_end = time.time()
    print('Spent %.4fsec loading cache.' % (load_end - load_start))

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def check_and_complete(prompt_text, **other_args):
    # TODO: Where should we store 8196 and model name?
    token_count = await llm.count_tokens(  # pytype: disable=wrong-keyword-args
        content=prompt_text,
    )
    logging.info('Token count %d for prompt:\n%s\n', token_count, prompt_text)
    if token_count > _MAX_INPUT_TOKENS:
      warning_msg = (
          f'Warning: Prompt token length ({token_count}) exceeds maximal input'
          f'token length ({_MAX_INPUT_TOKENS}) and '
          f'will be truncated on the server side: {prompt_text[:100]}\n.'
      )
      print(warning_msg)
      logging.warning(warning_msg)
    res = await llm.generate_text(prompt=prompt_text, **other_args)  # pytype: disable=wrong-keyword-args
    return res

  time1 = time.time()

  print('1. A single generate query.')
  prompt_text = """
    Question: Natural logarithm of $e^12$?
    Reasoning: "Natural" logarithm means logarithm to the base of $e$. For
     example, natural logarithm of $10$ means exponent to which $e$ must be
     raised to produce 10.
    Answer: 12.
    Question: Differentiate $\\frac{1}{\\log(x)}$.
  """
  res = executing.run(
      check_and_complete(
          prompt_text=prompt_text,
          stop=['\n\n'],
          max_tokens=5,
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
          max_tokens=5,
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
          max_tokens=5,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('2. Repeated generate request.')
  exe = executing.par_iter(
      sampling.repeat(
          executable=check_and_complete(
              'Today is', temperature=0.5, max_tokens=5, stop=['.']
          ),
          num_repeats=5,
      )
  )
  res = executing.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('3. Three batched generate queries.')
  exe = executing.par_iter([
      check_and_complete(prompt_text='In summer', max_tokens=5),
      check_and_complete(prompt_text='In winter', max_tokens=7),
      check_and_complete(prompt_text='In autumn', max_tokens=9),
  ])
  res = executing.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('4. Prompt that does not fit into maximal token length.')
  value_error_raised = False
  try:
    # This call should fail and raise ValueError in _batch_generate.
    _ = executing.run(
        check_and_complete(', '.join(map(str, list(range(_MAX_INPUT_TOKENS)))))
    )
  except ValueError as err:
    if 'GoogleGenAIAPI.generate_content raised' not in repr(err):
      success = False
      print('ValueError raised, but not with the expected message.', err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  if not value_error_raised:
    success = False
    print('ValueError not raised.')

  print('5. Three batched generate queries, one of which does not fit.')
  exe = executing.par_iter([
      check_and_complete(prompt_text='In summer', max_tokens=1),
      check_and_complete(prompt_text='In winter', max_tokens=32),
      check_and_complete(', '.join(map(str, list(range(_MAX_INPUT_TOKENS))))),
  ])
  value_error_raised = False
  try:
    _ = executing.run(exe)
  except ValueError as err:
    if 'GoogleGenAIAPI.generate_content raised' not in repr(err):
      success = False
      print('ValueError raised, but not with the expected message:', err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  if not value_error_raised:
    success = False
    print('ValueError not raised.')

  print('7.1 Generate text without healing.')
  # Expect something weird to happen.
  executable = llm.generate_text(  # pytype: disable=wrong-keyword-args
      prompt='When I sat on ',
      temperature=0.0,
      max_tokens=10,
      healing_option=llm.TokenHealingOption.NONE,
  )
  res = executing.run(executable)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('7.2 Generate text with space healing.')
  executable = llm.generate_text(  # pytype: disable=wrong-keyword-args
      prompt='When I sat on ',
      temperature=0.0,
      max_tokens=10,
      healing_option=llm.TokenHealingOption.SPACE_HEALING,
  )
  res = executing.run(executable)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('8. Use ChunkList.')
  executable = (
      c.c('What is the answer to this question: ')
      + c.c('What is the third planet from the sun? ')
      + c.store('answer', c.generate_text())
  )
  _ = executing.run(executable)
  if 'earth' not in executable['answer'].lower():
    success = False
    print('Returned value does not contain "earth":', executable['answer'])
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  if _SAVE_CACHE.value:
    cache.save(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname))

  print('9. Check that generate_texts is working.')
  res = executing.run(
      llm.generate_texts(  # pytype: disable=wrong-keyword-args
          prompt=prompt_text,
          samples=3,
          stop=['\n\n'],
          max_tokens=5,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('10.1 Check that chat is working (API formatting).')
  res = executing.run(
      llm.chat(  # pytype: disable=wrong-keyword-args
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
          max_tokens=15,
          stop=['.'],
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('10.2 Check that chat completes the model prefix.')
  res = executing.run(
      llm.chat(  # pytype: disable=wrong-keyword-args
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
  if not res.startswith('Once upon a'):
    success = False
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('10.3 Check that chat is working (Default formatting).')
  res = executing.run(
      llm.chat(  # pytype: disable=wrong-keyword-args
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
          max_tokens=15,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('10.4 Check that chat is working (API formatting) with chunk list.')
  res = executing.run(
      llm.chat(  # pytype: disable=wrong-keyword-args
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
          max_tokens=15,
          stop=['.'],
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('11. Check that multimodal is working.')
  fname_mm = os.path.join(_CACHE_DIR.value, 'google_gemini_api_mm.json')
  if api_key is not None:
    backend = google_genai_api.GoogleGenAIAPI(
        generate_model_name=google_genai_api.DEFAULT_MULTIMODAL_MODEL,
        api_key=api_key,
    )
  elif _PROJECT.value is not None and _LOCATION.value is not None:
    backend = google_genai_api.GoogleGenAIAPI(
        generate_model_name=google_genai_api.DEFAULT_MULTIMODAL_MODEL,
        vertexai=True,
        project=_PROJECT.value,
        location=_LOCATION.value,
    )
  else:
    raise ValueError('Either --api_key or (--project, --location) must be set.')
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
    executable = (
        c.c('What is the following image? ')
        + c.c(Image.open(f))
        + c.store('answer', c.generate_text())
    )
    _ = executing.run(executable)
  if 'chickadee' not in executable['answer'].lower():
    success = False
    print('Returned value does not contain "chickadee":', executable['answer'])
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  if _SAVE_CACHE.value:
    cache.save(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname_mm))

  print('12. Check response_schema and response_mime_type.')
  executable = c.c('What is the capital of France? ') + c.store(
      'answer',
      c.generate_text(
          response_mime_type='application/json',
          response_schema={
              'type': 'object',
              'properties': {'answer': {'type': 'string'}},
          },
      ),
  )
  _ = executing.run(executable)
  try:
    answer = json.loads(executable['answer'])
    if _PRINT_DEBUG.value:
      print('Returned value:', answer)
  except json.JSONDecodeError as e:
    success = False
    print('Returned value is not a valid JSON:', executable['answer'], e)

  print('13. Check thinking config.')
  res = executing.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='solve x^2 + 4x + 4 = 0',
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=1024,
          ),
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value:', res)

  if backend.vertexai:
    print('14. Check that tokenize is working.')
    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk(
                content='The quick brown fox jumps over the lazy dog.',
            )
        ]
    )
    res = executing.run(llm.tokenize(content=content))  # pytype: disable=wrong-keyword-args
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)
    if not isinstance(res, list) or not all(isinstance(x, int) for x in res):
      success = False
      print('Returned value is not a list of ints:', res)
  else:
    print('14. Skipping tokenize because it is not supported by Gemini API.')

  print('15. Check that embed is working.')
  content = content_lib.ChunkList(
      chunks=[
          content_lib.Chunk(
              content='The quick brown fox jumps over the lazy dog.',
          )
      ]
  )
  res = executing.run(llm.embed(content=content))  # pytype: disable=wrong-keyword-args
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  if not isinstance(res, list) or not all(isinstance(x, float) for x in res):
    success = False
    print('Returned value is not a list of floats:', res)

  print('16. Check llm.generate_object with native types. (int)')
  prompt = 'What is the height of the Eiffel Tower? (in m)'
  result = executing.run(
      llm.generate_object(prompt=prompt, cls=int)  # pytype: disable=wrong-keyword-args
  )
  if _PRINT_DEBUG.value:
    print('Returned value:', result)
  if not isinstance(result, int):
    success = False
    print(f'Result is not of type int: {type(result)}')

  class CityInfo(pydantic.BaseModel):
    city_name: str
    country: str
    population: int
    landmark: str | None = None

  print('17. Check llm.generate_object with Pydantic model.')
  prompt = 'Provide information about the capital of France.'
  result = executing.run(
      llm.generate_object(prompt=prompt, cls=CityInfo)  # pytype: disable=wrong-keyword-args
  )

  if _PRINT_DEBUG.value:
    print('Returned value:')
    pprint.pprint(result)

  if not isinstance(result, CityInfo):
    success = False
    print(f'Result is not of type CityInfo: {type(result)}')
  if not result.city_name:
    success = False
    print('city_name is missing')
  if not result.country:
    success = False
    print('country is missing')
  if result.population <= 0:
    success = False
    print('population should be positive')

  class WordCount(pydantic.BaseModel):
    word: str
    count: int

  print('18. Check llm.generate_object for dict-like data.')
  prompt = (
      'Return a breakdown of the words and their frequencies in the following'
      ' sentence: "the quick brown fox jumps over the lazy dog dog". '
      'Provide the output as a list of word-count objects.'
  )
  result = executing.run(
      llm.generate_object(prompt=prompt, cls=list[WordCount])  # pytype: disable=wrong-keyword-args
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
    success = False
    print(f'Result is not of type list: {type(result)}')
  elif not all(isinstance(item, WordCount) for item in result):
    success = False
    print(f'Result items are not all WordCount objects: {result}')
  else:
    result_dict = {item.word: item.count for item in result}
    if result_dict != expected_dict:
      success = False
      print(
          f'Result dict {result_dict} does not match expected {expected_dict}'
      )

  @dataclasses.dataclass
  class StandardRecipe:
    recipe_name: str
    description: str
    prep_time_minutes: int
    ingredients: list[str]

  pydantic_recipe = pydantic_dataclass(StandardRecipe)

  print(
      '19. Check llm.generate_object with a Pydantic-wrapped standard'
      ' dataclass.'
  )
  prompt = """Give me a simple recipe for a classic chocolate chip cookie,
  including a short description, prep time in minutes, and a list of
  ingredients."""
  result = executing.run(
      llm.generate_object(prompt=prompt, cls=pydantic_recipe)  # pytype: disable=wrong-keyword-args
  )

  if _PRINT_DEBUG.value:
    print('Returned value:')
    pprint.pprint(result)

  if not isinstance(result, StandardRecipe):
    success = False
    print(f'Result is not an instance of StandardRecipe: {type(result)}')
  elif not isinstance(result, pydantic_recipe):
    success = False
    print(f'Result is not an instance of PydanticRecipe: {type(result)}')
  else:
    if not result.recipe_name:
      success = False
      print('recipe_name is missing')
    if not result.description:
      success = False
      print('description is missing')
    if result.prep_time_minutes <= 0:
      success = False
      print('prep_time_minutes should be positive')
    if not isinstance(result.ingredients, list) or not result.ingredients:
      success = False
      print('ingredients should be a non-empty list')
    elif not all(isinstance(i, str) for i in result.ingredients):
      success = False
      print('ingredients should be a list of strings')

  if _SAVE_CACHE.value:
    cache.save(overwrite=True)

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
