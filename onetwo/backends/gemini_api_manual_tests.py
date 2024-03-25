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
from absl import logging
from onetwo.backends import gemini_api
from onetwo.builtins import composables as c
from onetwo.builtins import llm
from onetwo.core import caching
from onetwo.core import executing
from onetwo.core import sampling





_API_KEY = flags.DEFINE_string('api_key', default=None, help='GenAI API key.')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir',
    default='.',
    help='Directory where the cache will be stored.'
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
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  api_key = _API_KEY.value
  fname = os.path.join(_CACHE_DIR.value, 'google_gemini_api.json')
  backend = gemini_api.GeminiAPI(
      cache_filename=fname, api_key=api_key, batch_size=4
  )
  backend.register()
  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname)
    load_start = time.time()
    backend.load_cache()
    load_end = time.time()
    print('Spent %.4fsec loading cache.', load_end - load_start)

  @executing.make_executable
  async def check_and_complete(prompt_text, **other_args):
    # TODO: Where should we store 8196 and model name?
    token_count = await llm.count_tokens(
        content=prompt_text,
    )
    logging.info('Token count %d for prompt:\n%s\n', token_count, prompt_text)
    max_token_count = 30720  # See https://ai.google.dev/models/gemini.
    if token_count > max_token_count:
      warning_msg = (
          f'Warning: Prompt token length ({token_count}) exceeds maximal input'
          f'token length ({max_token_count}) and '
          f'will be truncated on the server side: {prompt_text[:100]}\n.'
      )
      print(warning_msg)
      logging.warning(warning_msg)
    res = await llm.generate_text(prompt=prompt_text, **other_args)
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
  res = executing.run(check_and_complete(
      prompt_text=prompt_text,
      stop=['\n\n'],
      max_tokens=5,
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('1.1 Same query to see if it has been cached.')
  res = executing.run(check_and_complete(
      prompt_text=prompt_text,
      stop=['\n\n'],
      max_tokens=5,
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('1.2 Same query but different parameters, run requests again.')
  res = executing.run(check_and_complete(
      prompt_text=prompt_text,
      temperature=0.,
      stop=['\n\n'],
      max_tokens=5,
  ))
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('2. Repeated generate request.')
  exe = executing.par_iter(sampling.repeat(
      executable=check_and_complete(
          'Today is', temperature=0.5, max_tokens=5, stop=['.']
      ),
      num_repeats=5,
  ))
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
        check_and_complete(', '.join(map(str, list(range(10000)))))
    )
  except ValueError as err:
    assert 'GeminiAPI.generate_text raised' in repr(err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  assert value_error_raised

  print('5. Three batched generate queries, one of which does not fit.')
  exe = executing.par_iter([
      check_and_complete(prompt_text='In summer', max_tokens=1),
      check_and_complete(prompt_text='In winter', max_tokens=32),
      check_and_complete(', '.join(map(str, list(range(20000))))),
  ])
  value_error_raised = False
  try:
    _ = executing.run(exe)
  except ValueError as err:
    assert 'GeminiAPI.generate_text raised' in repr(err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  assert value_error_raised

  print('6.1 Safety settings: default raises warnings')
  execs = [llm.generate_text(f'Question: {d}+{d}?\nAnswer:') for d in range(16)]
  executable = executing.par_iter(execs)
  value_error_raised = False
  try:
    executing.run(executable)
  except ValueError as err:
    assert 'GeminiAPI.generate_text returned no answers.' in repr(err)
    assert 'This may be caused by safety filters:' in repr(err)
    assert 'finish_reason: SAFETY' in repr(err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  assert value_error_raised

  print('6.2 Safety settings: disable safety settings')
  llm.generate_text.update(safety_settings=gemini_api.SAFETY_DISABLED)
  execs = [llm.generate_text(f'Question: {d}+{d}?\nAnswer:') for d in range(16)]
  executable = executing.par_iter(execs)

  res = executing.run(executable)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('7. Use ChunkList.')
  executable = (
      c.c('What is the answer to this question: ')
      + c.c('What is the third planet from the sun? ')
      + c.store('answer', c.generate_text())
  )
  _ = executing.run(executable)
  assert 'earth' in executable['answer'].lower(), executable['answer']
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
  if not _LOAD_CACHE.value:
    handler.save(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname))

  print('8. Check that multimodal is working.')
  fname_mm = os.path.join(_CACHE_DIR.value, 'google_gemini_api_mm.json')
  backend = gemini_api.GeminiAPI(
      cache_filename=fname_mm,
      generate_model_name=gemini_api.DEFAULT_MULTIMODAL_MODEL,
      api_key=api_key,
  )
  backend.register()
  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname_mm)
    load_start = time.time()
    backend.load_cache()
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
        c.c('What is the following image? ')
        + c.c(image_bytes)
        + c.store('answer', c.generate_text())
    )
    _ = executing.run(executable)
  assert 'chickadee' in executable['answer'].lower(), executable['answer']
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  handler: caching.SimpleFunctionCache = getattr(backend, '_cache_handler')
  if not _LOAD_CACHE.value:
    handler.save(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname_mm))

  print('PASS')


if __name__ == '__main__':
  app.run(main)
