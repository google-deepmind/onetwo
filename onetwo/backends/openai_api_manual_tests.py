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

from collections.abc import Sequence
import os
import pprint
import time

from absl import app
from absl import flags
from onetwo.backends import openai_api
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling
import pydantic




_API_KEY = flags.DEFINE_string(
    'api_key',
    default=None,
    help='OpenAI API key.',
)
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
_PRINT_DEBUG = flags.DEFINE_bool(
    'print_debug',
    default=False,
    help='Debug logging.',
)


def main(argv: Sequence[str]) -> None:
  success = True
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  api_key = _API_KEY.value

  # We test the latest GPT-5.x models using the Responses API.
  for model_name in ['gpt-5.4-mini', 'gpt-5.4', 'gpt-4o']:
    print('*** Running tests for %s. ***' % model_name)
    fname = os.path.join(
        _CACHE_DIR.value,
        f'openai_api_{model_name.replace("-", "_").replace(".", "_")}.json',
    )
    backend = openai_api.OpenAIAPI(
        model_name=model_name,
        cache=caching.SimpleFunctionCache(cache_filename=fname),
        api_key=api_key,
        batch_size=4,
    )
    backend.register()
    if _LOAD_CACHE.value:
      print('Loading cache from file %s', fname)
      load_start = time.time()
      assert isinstance(backend.cache, caching.SimpleFunctionCache)
      backend.cache.load()
      load_end = time.time()
      print('Spent %.4fsec loading cache.', load_end - load_start)

    print('1. A single generate query.')
    prompt = """
      Question: Natural logarithm of $e^12$?
      Reasoning: "Natural" logarithm means logarithm to the base of $e$. For
      example, natural logarithm of $10$ means exponent to which $e$ must be
      raised to produce 10.
      Answer: 12.
      Question: Differentiate $\\frac{1}{\\log(x)}$.
    """
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            max_tokens=25,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('1.1 Same query to see if it has been cached.')
    counters_before = backend._counters
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            max_tokens=25,
        )
    )
    counters_after = backend._counters
    if counters_before != counters_after:
      success = False
      print(
          'FAILURE: counters changed, indicating a cache miss.',
          counters_before,
          counters_after,
      )
    elif _PRINT_DEBUG.value:
      print('Counters did not change, indicating a cache hit.')
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('1.2 Same query but different parameters, run requests again.')
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            temperature=0.0,
            max_tokens=25,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('2. Repeated generate request.')
    exe = executing.par_iter(
        sampling.repeat(
            executable=llm.generate_text(  # pytype: disable=wrong-arg-count
                'Today is', temperature=0.5, max_tokens=25
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
        llm.generate_text(prompt='In summer', max_tokens=25),  # pytype: disable=wrong-keyword-args
        llm.generate_text(prompt='In winter', max_tokens=27),  # pytype: disable=wrong-keyword-args
        llm.generate_text(prompt='In autumn', max_tokens=29),  # pytype: disable=wrong-keyword-args
    ])
    res = executing.run(exe)
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('4. Check that include_details returns text and score.')
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt='In winter',
            max_tokens=25,
            include_details=True,
        )
    )
    if not isinstance(res, tuple) or 'text' not in res[1]:
      success = False
      print('FAILURE: include_details did not return expected format.')
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('5. Calling generate_texts to get multiple samples.')
    prompt = """How to say "I am hungry" in French?"""
    res = executing.run(
        llm.generate_texts(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            samples=3,
            temperature=0.5,
            max_tokens=25,
        )
    )
    if len(res) != 3:
      success = False
      print('FAILURE: returned %d samples, expected 3.' % len(res))
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6. Calling instruct: without fewshots.')
    res = executing.run(
        llm.instruct(  # pytype: disable=wrong-keyword-args
            prompt='Compose a sentence starting with H.',
            temperature=0.5,
            max_tokens=50,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.1 Calling instruct: with fewshots.')
    res = executing.run(
        llm.instruct(  # pytype: disable=wrong-keyword-args
            prompt='Compose a sentence starting with H.',
            temperature=0.5,
            max_tokens=50,
            formatter_kwargs={'use_fewshots': True},
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.2 Calling instruct and providing the prefix: without fewshots.')
    res = executing.run(
        llm.instruct(  # pytype: disable=wrong-keyword-args
            prompt='Compose a sentence starting with H.',
            assistant_prefix='His',
            temperature=0.5,
            max_tokens=50,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.2 Calling instruct and providing the prefix: with fewshots.')
    res = executing.run(
        llm.instruct(  # pytype: disable=wrong-keyword-args
            prompt='Compose a sentence starting with H.',
            assistant_prefix='His',
            temperature=0.5,
            max_tokens=50,
            formatter_kwargs={'use_fewshots': True},
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('7. Calling chat.')
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
            max_tokens=50,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    if not _LOAD_CACHE.value:
      start = time.time()
      assert isinstance(backend.cache, caching.SimpleFunctionCache)
      backend.cache.save(overwrite=True)
      print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

    # Test generate_object for all modern models.
    class CityInfo(pydantic.BaseModel):
      name: str
      population: int

    print('8. Generate object.')
    prompt = """What is the capital of France?"""
    res = executing.run(
        llm.generate_object(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            cls=CityInfo,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('9. Generate object with chat history.')
    prompt = [
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
            content='What is its population?',
        ),
    ]
    res = executing.run(
        llm.generate_object(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            cls=CityInfo,
        )
    )

    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    image_path = os.path.join(os.path.dirname(__file__), 'testdata/bird.jpg')
    try:
      with open(image_path, 'rb') as f:
        image_bytes = f.read()
      print('9.1 Generate text with image (instruct).')
      prompt = content_lib.ChunkList([
          content_lib.Chunk(content='Describe this image: '),
          content_lib.Chunk(content=image_bytes, content_type='image/jpeg'),
      ])
      res = executing.run(
          llm.instruct(  # pytype: disable=wrong-keyword-args
              prompt=prompt,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      print('9.2 Generate text with image (chat).')
      res = executing.run(
          llm.chat(  # pytype: disable=wrong-keyword-args
              messages=[
                  content_lib.Message(
                      content=prompt, role=content_lib.PredefinedRole.USER
                  )
              ],
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
    except Exception as e:  # pylint: disable=broad-except
      print(f'Failed to run image test: {e}')

    print('9.3 Generate object with primitive type (int).')
    prompt = 'What is the height of the Eiffel Tower? (in m)'
    try:
      res = executing.run(
          llm.generate_object(prompt=prompt, cls=int)  # pytype: disable=wrong-keyword-args
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      if not isinstance(res, int):
        print(f'FAILURE: expected int, got {type(res)}')
    except Exception as e:  # pylint: disable=broad-except
      print(f'Failed to run generate_object with int: {e}')

    print('9.4 Generate object with list of Pydantic models.')

    class WordCount(pydantic.BaseModel):
      word: str
      count: int

    prompt = (
        'Return a breakdown of the words and their frequencies in the following'
        ' sentence: "the quick brown fox jumps over the lazy dog dog". Provide'
        ' the output as a list of word-count objects.'
    )
    try:
      res = executing.run(
          llm.generate_object(prompt=prompt, cls=list[WordCount])  # pytype: disable=wrong-keyword-args
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      if not isinstance(res, list) or not all(
          isinstance(x, WordCount) for x in res
      ):
        print(f'FAILURE: expected list[WordCount], got {type(res)}')
    except Exception as e:  # pylint: disable=broad-except
      print(f'Failed to run generate_object with list[WordCount]: {e}')

    print('9.5 Generate object with list of integers.')
    prompt = 'Return the first 3 prime numbers as a list of integers.'
    try:
      res = executing.run(
          llm.generate_object(prompt=prompt, cls=list[int])  # pytype: disable=wrong-keyword-args
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
      if not isinstance(res, list) or not all(isinstance(x, int) for x in res):
        print(f'FAILURE: expected list[int], got {type(res)}')
    except Exception as e:  # pylint: disable=broad-except
      print(f'Failed to run generate_object with list[int]: {e}')

  # Reasoning tests (run outside the per-model loop, using a single model
  # that supports reasoning, e.g. o4-mini).
  print('\n*** Running reasoning and tool tests. ***')
  reasoning_model = 'o4-mini'
  fname = os.path.join(
      _CACHE_DIR.value,
      f'openai_api_{reasoning_model.replace("-", "_").replace(".", "_")}.json',
  )
  backend = openai_api.OpenAIAPI(
      model_name=reasoning_model,
      cache=caching.SimpleFunctionCache(cache_filename=fname),
      api_key=api_key,
      batch_size=4,
  )
  backend.register()

  for effort in ['low', 'medium', 'high']:
    print(f'10. Reasoning effort={effort}.')
    # Re-register with different reasoning to test per-instance config.
    backend_r = openai_api.OpenAIAPI(
        model_name=reasoning_model,
        cache=caching.SimpleFunctionCache(cache_filename=fname),
        api_key=api_key,
        batch_size=1,
        reasoning={'effort': effort},
    )
    backend_r.register()
    res = executing.run(
        llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt='How many r\'s are in the word "strawberry"?',
        )
    )
    if _PRINT_DEBUG.value:
      print(f'  Reasoning effort={effort} result:')
      pprint.pprint(res)

  # Tool use tests.
  print('\n*** Running tool use tests. ***')
  tool_model = 'gpt-5.4-mini'
  fname_tools = os.path.join(
      _CACHE_DIR.value,
      f'openai_api_{tool_model.replace("-", "_").replace(".", "_")}_tools.json',
  )

  web_search_tool = {'type': 'web_search'}

  print('11. Generate text with a single tool definition.')
  backend_t = openai_api.OpenAIAPI(
      model_name=tool_model,
      cache=caching.SimpleFunctionCache(cache_filename=fname_tools),
      api_key=api_key,
      batch_size=1,
      tools=[web_search_tool],
  )
  backend_t.register()
  res = executing.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='Who won the latest Formula 1 race?',
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('11.1 Chat with tools.')
  res = executing.run(
      llm.chat(  # pytype: disable=wrong-keyword-args
          messages=[
              content_lib.Message(
                  role=content_lib.PredefinedRole.USER,
                  content='What are the latest news about Gemini?',
              ),
          ],
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('11.2 Generate text with tools but prompt that does not need them.')
  res = executing.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='Write a haiku about the ocean.',
          max_tokens=100,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
