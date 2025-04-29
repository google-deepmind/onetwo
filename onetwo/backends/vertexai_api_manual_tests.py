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
from onetwo.backends import vertexai_api
from onetwo.builtins import composables as c
from onetwo.builtins import formatting
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import sampling





_VETEXT_AI_PARAMS_DOC = 'https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig'

_PROJECT = flags.DEFINE_string('project', default=None, help='Project ID.')
_LOCATION = flags.DEFINE_string('location', default=None, help='Location.')
_MODEL_NAME = flags.DEFINE_string(
    'generate_model_name',
    default='gemini-1.5-flash',
    help=(
        'The model to use for testing. Note that not all models support more'
        ' than 1 candidates (required by generate_texts). See'
        f' {_VETEXT_AI_PARAMS_DOC} for details.'
    ),
)
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
  success = True
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  project = _PROJECT.value
  location = _LOCATION.value
  fname = os.path.join(_CACHE_DIR.value, 'google_vertexai_api.json')
  backend = vertexai_api.VertexAIAPI(
      generate_model_name=(
          _MODEL_NAME.value or vertexai_api.DEFAULT_GENERATE_MODEL
      ),
      cache_filename=fname,
      project=project,
      location=location,
      batch_size=4,
  )
  backend.register()
  if _LOAD_CACHE.value:
    print('Loading cache from file %s', fname)
    load_start = time.time()
    backend.load_cache()
    load_end = time.time()
    print('Spent %.4fsec loading cache.' % (load_end - load_start))

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def check_and_complete(prompt_text, **other_args):
    """Check if the prompt is too long and complete it."""
    token_count = await llm.count_tokens(  # pytype: disable=wrong-keyword-args
        content=prompt_text,
    )
    logging.info('Token count %d for prompt:\n%s\n', token_count, prompt_text)
    # See https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini # pylint: disable=line-too-long
    max_token_count = 32760
    if token_count > max_token_count:
      warning_msg = (
          f'Warning: Prompt token length ({token_count}) exceeds maximal input'
          f'token length ({max_token_count}) and '
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
  print('1.3 Same query but include details.')
  res = executing.run(
      check_and_complete(
          prompt_text=prompt_text,
          temperature=0.0,
          stop=['\n\n'],
          max_tokens=5,
          include_details=True,
      )
  )
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
    if 'VertexAIAPI.generate_content raised' not in repr(err):
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
      check_and_complete(', '.join(map(str, list(range(20000))))),
  ])
  value_error_raised = False
  try:
    _ = executing.run(exe)
  except ValueError as err:
    if 'VertexAIAPI.generate_content raised' not in repr(err):
      success = False
      print('ValueError raised, but not with the expected message:', err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  if not value_error_raised:
    success = False
    print('ValueError not raised.')

  print('6.1 Safety settings: default may raise warnings')
  execs = [llm.generate_text(f'Question: {d}+{d}?\nAnswer:') for d in range(16)]  # pytype: disable=wrong-arg-count
  executable = executing.par_iter(execs)
  value_error_raised = False
  try:
    res = executing.run(executable)
  except ValueError as err:
    if 'VertexAIAPI.generate_text returned no answers.' not in repr(
        err
    ) or 'finish_reason: SAFETY' not in repr(err):
      success = False
      print('ValueError raised, but not with the expected message:', err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  if not value_error_raised:
    print('ValueError not raised.')
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

  print('6.2 Safety settings: disable safety settings')
  llm.generate_text.update(safety_settings=vertexai_api.SAFETY_DISABLED)
  execs = [llm.generate_text(f'Question: {d}+{d}?\nAnswer:') for d in range(16)]  # pytype: disable=wrong-arg-count
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
  if 'earth' not in executable['answer'].lower():
    success = False
    print('Returned value does not contain "earth":', executable['answer'])
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  if not _LOAD_CACHE.value:
    backend.save_cache(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname))

  def _try_multi_generation(exe):
    try:
      return executing.run(exe)
    except ValueError as err:
      if 'candidateCount must be 1' in repr(err):
        raise ValueError(
            f'Model under test {backend.generate_model_name} does not support'
            f' more than 1 candidates. Check {_VETEXT_AI_PARAMS_DOC} to find '
            'one that supports it.'
        ) from err
      raise err

  print('8. Check that generate_texts is working.')
  exe = llm.generate_texts(  # pytype: disable=wrong-keyword-args
      prompt=prompt_text,
      samples=3,
      stop=['\n\n'],
      max_tokens=5,
  )
  res = _try_multi_generation(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)
  print('8.1 Same query but include details.')
  exe = llm.generate_texts(  # pytype: disable=wrong-keyword-args
      prompt=prompt_text,
      samples=3,
      stop=['\n\n'],
      max_tokens=5,
      include_details=True,
  )
  res = _try_multi_generation(exe)
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

    print('9.1 Check that chat is working (API formatting).')
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
          max_tokens=15,
          stop=['.'],
      )
  )
  if _PRINT_DEBUG.value:
    print('Returned value(s):')
    pprint.pprint(res)

  print('9.2 Check that chat doesn\'t try to completes the model prefix.')
  value_error_raised = False
  try:
    _ = executing.run(
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
            max_tokens=10,
            stop=['.'],
        )
    )
  except ValueError as err:
    if 'last message must be a user message' not in repr(err):
      success = False
      print('ValueError raised, but not with the expected message.', err)
    value_error_raised = True
    if _PRINT_DEBUG.value:
      print('ValueError raised.')
  if not value_error_raised:
    success = False
    print('ValueError not raised.')

  print('9.3 Check that chat is working (Default formatting).')
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

  print('10. Check that multimodal is working.')
  fname_mm = os.path.join(_CACHE_DIR.value, 'google_vertexai_api_mm.json')
  backend = vertexai_api.VertexAIAPI(
      cache_filename=fname_mm,
      generate_model_name=vertexai_api.DEFAULT_MULTIMODAL_MODEL,
      project=project,
      location=location,
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
  if 'chickadee' not in executable['answer'].lower():
    success = False
    print('Returned value does not contain "chickadee":', executable['answer'])
  if _PRINT_DEBUG.value:
    print('Returned value:', executable['answer'])

  time2 = time.time()
  print('Took %.4fsec running requests.' % (time2 - time1))
  if not _LOAD_CACHE.value:
    backend.save_cache(overwrite=True)
    time3 = time.time()
    print('Took %.4fsec saving cache to %s.' % (time3 - time2, fname_mm))

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
