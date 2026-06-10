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

"""Manual tests for the Ollama API backend.

Requires a running Ollama server (default: http://localhost:11434).
Pull a model first, e.g.: `ollama pull llama3.2`

Usage:
  python ollama_api_manual_tests.py --print_debug
  python ollama_api_manual_tests.py --model_name=gemma3:4b --print_debug
  python ollama_api_manual_tests.py --max_tokens=200 --print_debug
  python ollama_api_manual_tests.py --skip_multimodal --print_debug
  python ollama_api_manual_tests.py --load_cache_file --print_debug
"""

from collections.abc import Sequence
import os
import pprint
import time

from absl import app
from absl import flags
from onetwo import ot
from onetwo.backends import ollama_api
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import content as content_lib
from onetwo.core import sampling

_HOST = flags.DEFINE_string(
    'host',
    default=None,
    help=(
        'Ollama server host URL (e.g., http://localhost:11434). '
        'If not set, uses OLLAMA_HOST env var or defaults to localhost.'
    ),
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    default=ollama_api.DEFAULT_MODEL,
    help='Ollama model name (e.g., llama3.2, gemma3:4b, mistral).',
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
_SKIP_MULTIMODAL = flags.DEFINE_bool(
    'skip_multimodal',
    default=False,
    help='Skip the multimodal (image) test.',
)
_MAX_TOKENS = flags.DEFINE_integer(
    'max_tokens',
    default=50,
    help='Maximum number of tokens to generate per request.',
)


def main(argv: Sequence[str]) -> None:
  success = True
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_name = _MODEL_NAME.value
  print('*** Running tests for %s. ***' % model_name)
  fname = os.path.join(
      _CACHE_DIR.value,
      f'ollama_api_{model_name.replace("-", "_").replace(":", "_").replace(".", "_")}.json',
  )
  backend = ollama_api.OllamaAPI(
      model_name=model_name,
      host=_HOST.value,
      cache=caching.SimpleFunctionCache(cache_filename=fname),
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

  # ==================== generate_text tests ====================
  print('1. A single generate query.')
  prompt = """\
    Question: Natural logarithm of $e^12$?
    Reasoning: "Natural" logarithm means logarithm to the base of $e$. For
    example, natural logarithm of $10$ means exponent to which $e$ must be
    raised to produce 10.
    Answer: 12.
    Question: Differentiate $\\frac{1}{\\log(x)}$.
  """
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt=prompt,
          max_tokens=_MAX_TOKENS.value,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('1.1 Same query to see if it has been cached.')
  counters_before = dict(backend._counters)
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt=prompt,
          max_tokens=_MAX_TOKENS.value,
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
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt=prompt,
          max_tokens=_MAX_TOKENS.value,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== repeated / batched queries ====================
  print('2. Repeated generate request.')
  exe = ot.par_iter(
      sampling.repeat(
          executable=llm.generate_text(  # pytype: disable=wrong-arg-count
              'Today is', max_tokens=_MAX_TOKENS.value
          ),
          num_repeats=5,
      )
  )
  res = ot.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('3. Three batched generate queries.')
  exe = ot.par_iter([
      llm.generate_text(prompt='In summer', max_tokens=_MAX_TOKENS.value),  # pytype: disable=wrong-keyword-args
      llm.generate_text(prompt='In winter', max_tokens=_MAX_TOKENS.value),  # pytype: disable=wrong-keyword-args
      llm.generate_text(prompt='In autumn', max_tokens=_MAX_TOKENS.value),  # pytype: disable=wrong-keyword-args
  ])
  res = ot.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== include_details ====================
  print('4. Check that include_details returns text and score.')
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='In winter',
          max_tokens=_MAX_TOKENS.value,
          include_details=True,
      )
  )
  if not isinstance(res, tuple) or 'text' not in res[1]:
    success = False
    print('FAILURE: include_details did not return expected format.')
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== generate_texts ====================
  print('5. Calling generate_texts to get multiple samples.')
  prompt = """How to say "I am hungry" in French?"""
  res = ot.run(
      llm.generate_texts(  # pytype: disable=wrong-keyword-args
          prompt=prompt,
          samples=3,
          max_tokens=_MAX_TOKENS.value,
      )
  )
  if len(res) != 3:
    success = False
    print('FAILURE: returned %d samples, expected 3.' % len(res))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== chat ====================
  print('6. Calling chat.')
  res = ot.run(
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
          max_tokens=_MAX_TOKENS.value,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== Sampling parameters ====================
  print('7. Generate with temperature.')
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='Tell me a fun fact about penguins.',
          max_tokens=_MAX_TOKENS.value,
          temperature=0.9,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('7.1 Generate with top_k and top_p.')
  res = ot.run(
      llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='Tell me a fun fact about dolphins.',
          max_tokens=_MAX_TOKENS.value,
          top_k=40,
          top_p=0.95,
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  # ==================== Multimodal ====================
  if _SKIP_MULTIMODAL.value:
    print('8. Multimodal input. [SKIPPED]')
  else:
    print('8. Multimodal input.')
    image_path = os.path.join(
        os.path.dirname(__file__),
        'testdata',
        'bird.jpg',
    )
    if os.path.exists(image_path):
      with open(image_path, 'rb') as f:
        image_bytes = f.read()
      messages = [
          content_lib.Message(
              role=content_lib.PredefinedRole.USER,
              content=content_lib.ChunkList([
                  content_lib.Chunk('What is the following image? '),
                  content_lib.Chunk(image_bytes, 'image/jpeg'),
              ]),
          ),
      ]
      res = ot.run(
          llm.chat(  # pytype: disable=wrong-keyword-args
              messages=messages,
              max_tokens=_MAX_TOKENS.value,
          )
      )
      if _PRINT_DEBUG.value:
        print('Returned value(s):')
        pprint.pprint(res)
    else:
      print(f'Skipping multimodal test: {image_path} not found.')

  # ==================== Save cache ====================
  if not _LOAD_CACHE.value:
    start = time.time()
    assert isinstance(backend.cache, caching.SimpleFunctionCache)
    backend.cache.save(overwrite=True)
    print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
