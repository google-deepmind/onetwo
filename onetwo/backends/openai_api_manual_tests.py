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

from collections.abc import Sequence
import os
import pprint
import time

from absl import app
from absl import flags
from onetwo.backends import openai_api
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling





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

  # We test both new "chat api" model and "completions api" model.
  for model_name in ['gpt-3.5-turbo', 'gpt-3.5-turbo-instruct']:
    print('*** Running tests for %s. ***' % model_name)
    fname = os.path.join(
        _CACHE_DIR.value,
        f'openai_api_{model_name.replace("-", "_")}.json',
    )
    backend = openai_api.OpenAIAPI(
        model_name=model_name,
        cache_filename=fname,
        api_key=api_key,
        batch_size=4,
    )
    backend.register()
    if _LOAD_CACHE.value:
      print('Loading cache from file %s', fname)
      load_start = time.time()
      backend.load_cache()
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
        llm.generate_text(
            prompt=prompt,
            stop=['\n\n'],
            max_tokens=5,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('1.1 Same query to see if it has been cached.')
    counters_before = backend._counters
    res = executing.run(
        llm.generate_text(
            prompt=prompt,
            stop=['\n\n'],
            max_tokens=5,
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
        llm.generate_text(
            prompt=prompt,
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
            executable=llm.generate_text(
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
        llm.generate_text(prompt='In summer', max_tokens=5),
        llm.generate_text(prompt='In winter', max_tokens=7),
        llm.generate_text(prompt='In autumn', max_tokens=9),
    ])
    res = executing.run(exe)
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('4. Check that a non-zero score is returned.')
    res = executing.run(
        llm.generate_text(
            prompt='In winter',
            max_tokens=5,
            include_details=True,
        )
    )
    if (res[1]['score'] == 0.0):
      success = False
      print('FAILURE: score is 0.0.')
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('5. Calling generate_texts to get multiple samples.')
    prompt = """How to say "I am hungry" in French?"""
    # For `gpt-3.5-turbo-instruct` this seems to generate empty replies, because
    # most of them start with `\n\n`. Good test case!
    res = executing.run(
        llm.generate_texts(
            prompt=prompt,
            samples=3,
            temperature=0.5,
            stop=['\n\n'],
            max_tokens=5,
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
        llm.instruct(
            prompt='Compose a sentence starting with H.',
            temperature=0.5,
            max_tokens=15,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.1 Calling instruct: with fewshots.')
    res = executing.run(
        llm.instruct(
            prompt='Compose a sentence starting with H.',
            temperature=0.5,
            max_tokens=15,
            formatter_kwargs={'use_fewshots': True},
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.2 Calling instruct and providing the prefix: without fewshots.')
    res = executing.run(
        llm.instruct(
            prompt='Compose a sentence starting with H.',
            assistant_prefix='His',
            temperature=0.5,
            max_tokens=15,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('6.2 Calling instruct and providing the prefix: with fewshots.')
    res = executing.run(
        llm.instruct(
            prompt='Compose a sentence starting with H.',
            assistant_prefix='His',
            temperature=0.5,
            max_tokens=15,
            formatter_kwargs={'use_fewshots': True},
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    print('7. Calling chat.')
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
            max_tokens=15,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    if not _LOAD_CACHE.value:
      start = time.time()
      backend.save_cache(overwrite=True)
      print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
