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
import os
import pprint
import time

from absl import app
from absl import flags
from onetwo.backends import onetwo_api
from onetwo.builtins import llm
from onetwo.core import executing
from onetwo.core import sampling

_ENDPOINT = flags.DEFINE_string(
    'endpoint',
    default='http://localhost:9876',
    help='Endpoint to use.',
)
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir',
    default='/tmp/onetwo_api',
    help='Directory where the cache will be stored.',
)
_LOAD_CACHE = flags.DEFINE_bool(
    'load_cache_file',
    default=False,
    help='Whether we should read the cache stored in file.'
)
_PRINT_DEBUG = flags.DEFINE_bool(
    'print_debug',
    default=False,
    help='Debug logging.'
)


def main(argv: Sequence[str]) -> None:
  del argv
  fname = os.path.join(_CACHE_DIR.value, 'onetwo_api.json')
  backend = onetwo_api.OneTwoAPI(
      endpoint=_ENDPOINT.value,
      cache_filename=fname,
      batch_size=4,
  )
  backend.register()
  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname)
    load_start = time.time()
    backend.load_cache()
    load_end = time.time()
    print('Spent %.4fsec loading cache.' % (load_end - load_start))

  print('1. A single generate query.')
  prompt_text = """
    Question: Natural logarithm of $e^12$?
    Reasoning: "Natural" logarithm means logarithm to the base of $e$. For
     example, natural logarithm of $10$ means exponent to which $e$ must be
     raised to produce 10.
    Answer: 12.
    Question: Differentiate $\\frac{1}{\\log(x)}$.
  """
  res = executing.run(llm.generate_text(  # pytype: disable=wrong-keyword-args
      prompt=prompt_text,
      stop=['\n\n'],
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('1.1 Same query to see if it has been cached.')
  res = executing.run(llm.generate_text(  # pytype: disable=wrong-keyword-args
      prompt=prompt_text,
      stop=['\n\n'],
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('1.2 Same query but different parameters, run requests again.')
  res = executing.run(llm.generate_text(  # pytype: disable=wrong-keyword-args
      prompt=prompt_text,
      temperature=0.,
      stop=['\n\n'],
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('2. Repeated generate request.')
  exe = executing.par_iter(sampling.repeat(
      executable=llm.generate_text(  # pytype: disable=wrong-keyword-args
          prompt='Today is', temperature=0.5, stop=['.']),
      num_repeats=5,
  ))
  res = executing.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('3. Three batched generate queries.')
  exe = executing.par_iter([
      llm.generate_text(prompt='In summer', stop=['.']),  # pytype: disable=wrong-keyword-args
      llm.generate_text(prompt='In winter', max_tokens=32),  # pytype: disable=wrong-keyword-args
      llm.generate_text(prompt='In autumn'),  # pytype: disable=wrong-keyword-args
  ])
  res = executing.run(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  if not _LOAD_CACHE.value:
    start = time.time()
    backend.save_cache(overwrite=True)
    print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

  print('PASS')


if __name__ == '__main__':
  app.run(main)
