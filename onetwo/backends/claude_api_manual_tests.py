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
from onetwo.backends import claude_api
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling




_API_KEY = flags.DEFINE_string(
    'api_key',
    default=None,
    help='Anthropic API key.',
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

  model_name = claude_api.DEFAULT_MODEL
  print('*** Running tests for %s. ***' % model_name)
  fname = os.path.join(
      _CACHE_DIR.value,
      f'claude_api_{model_name.replace("-", "_").replace(".", "_")}.json',
  )
  backend = claude_api.ClaudeAPI(
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
  counters_before = dict(backend._counters)
  res = executing.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt=prompt,
          max_tokens=25,
      )
  )
  counters_after = dict(backend._counters)
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
          max_tokens=50,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('2. Repeated generate request.')
  exe = executing.par_iter(
      sampling.repeat(
          executable=llm.generate_text(  # pytype: disable=wrong-arg-count
              'Today is', max_tokens=25
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
          max_tokens=50,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== Multimodal tests ====================
  image_path = os.path.join(os.path.dirname(__file__), 'testdata/bird.jpg')
  try:
    with open(image_path, 'rb') as f:
      image_bytes = f.read()
    print('8. Generate text with image (instruct).')
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
    print('8.1 Generate text with image (chat).')
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

  # ==================== Count tokens tests ====================
  print('9. Count tokens with string input.')
  text = 'The quick brown fox jumps over the lazy dog.'
  res = executing.run(
      llm.count_tokens(content=text)  # pytype: disable=wrong-keyword-args
  )
  if not isinstance(res, int) or res <= 0:
    success = False
    print(f'FAILURE: expected positive int, got {res!r}')
  elif _PRINT_DEBUG.value:
    print(f'  Token count for string: {res}')

  print('9.1 Count tokens with ChunkList input.')
  chunks = content_lib.ChunkList([
      content_lib.Chunk('Hello, '),
      content_lib.Chunk('world!'),
  ])
  res_chunks = executing.run(
      llm.count_tokens(content=chunks)  # pytype: disable=wrong-keyword-args
  )
  if not isinstance(res_chunks, int) or res_chunks <= 0:
    success = False
    print(f'FAILURE: expected positive int, got {res_chunks!r}')
  elif _PRINT_DEBUG.value:
    print(f'  Token count for ChunkList: {res_chunks}')

  print('9.2 Count tokens with messages.')
  messages = [
      content_lib.Message(
          role=content_lib.PredefinedRole.USER,
          content='What is the capital of France?',
      ),
      content_lib.Message(
          role=content_lib.PredefinedRole.MODEL,
          content='Paris.',
      ),
      content_lib.Message(
          role=content_lib.PredefinedRole.USER,
          content='And of Germany?',
      ),
  ]
  res_msgs = executing.run(
      llm.count_tokens(content=messages)  # pytype: disable=wrong-keyword-args
  )
  if not isinstance(res_msgs, int) or res_msgs <= 0:
    success = False
    print(f'FAILURE: expected positive int, got {res_msgs!r}')
  elif _PRINT_DEBUG.value:
    print(f'  Token count for messages: {res_msgs}')

  print('9.3 Count tokens caching check.')
  counters_before = dict(backend._counters)
  res_cached = executing.run(
      llm.count_tokens(content=text)  # pytype: disable=wrong-keyword-args
  )
  counters_after = dict(backend._counters)
  if counters_before != counters_after:
    success = False
    print(
        'FAILURE: counters changed, indicating a cache miss.',
        counters_before,
        counters_after,
    )
  elif _PRINT_DEBUG.value:
    print('  Counters did not change, indicating a cache hit.')
  if res != res_cached:
    success = False
    print('FAILURE: cached result does not match original result.')

  print('9.4 Count tokens with image in ChunkList.')
  try:
    with open(image_path, 'rb') as f:
      image_bytes = f.read()
    image_chunks = content_lib.ChunkList([
        content_lib.Chunk(content='Describe this image:'),
        content_lib.Chunk(content=image_bytes, content_type='image/jpeg'),
    ])
    res_img = executing.run(
        llm.count_tokens(content=image_chunks)  # pytype: disable=wrong-keyword-args
    )
    if not isinstance(res_img, int) or res_img <= 0:
      success = False
      print(f'FAILURE: expected positive int, got {res_img!r}')
    elif _PRINT_DEBUG.value:
      print(f'  Token count for image ChunkList: {res_img}')
  except Exception as e:  # pylint: disable=broad-except
    print(f'Failed to count tokens with image: {e}')

  if not _LOAD_CACHE.value:
    start = time.time()
    assert isinstance(backend.cache, caching.SimpleFunctionCache)
    backend.cache.save(overwrite=True)
    print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

  print('PASS' if success else 'FAIL')

if __name__ == '__main__':
  app.run(main)
