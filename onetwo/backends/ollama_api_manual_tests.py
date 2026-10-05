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
For embedding tests, also pull an embedding model, e.g.:
  `ollama pull embeddinggemma`

Usage:
  python ollama_api_manual_tests.py --print_debug
  python ollama_api_manual_tests.py --model_name=gemma3:4b --print_debug
  python ollama_api_manual_tests.py --max_tokens=200 --print_debug
  python ollama_api_manual_tests.py --skip_multimodal --print_debug
  python ollama_api_manual_tests.py --skip_embedding --print_debug
  python ollama_api_manual_tests.py --embed_model_name=all-minilm --print_debug
  python ollama_api_manual_tests.py --load_cache_file --print_debug
"""

from collections.abc import Sequence
import math
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
_EMBED_MODEL_NAME = flags.DEFINE_string(
    'embed_model_name',
    default='embeddinggemma',
    help=(
        'Ollama embedding model name (e.g., embeddinggemma, all-minilm,'
        ' nomic-embed-text).'
    ),
)
_SKIP_EMBEDDING = flags.DEFINE_bool(
    'skip_embedding',
    default=False,
    help='Skip the embedding tests.',
)
_SKIP_GENERATE = flags.DEFINE_bool(
    'skip_generate',
    default=False,
    help='Skip the generation, chat, and multimodal tests.',
)


def main(argv: Sequence[str]) -> None:
  success = True
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_name = _MODEL_NAME.value
  embed_model_name = _EMBED_MODEL_NAME.value
  print(
      '*** Running tests for %s and embedding model %s. ***'
      % (model_name, embed_model_name)
  )
  fname = os.path.join(
      _CACHE_DIR.value,
      f'ollama_api_{model_name.replace("-", "_").replace(":", "_").replace(".", "_")}.json',
  )
  backend = ollama_api.OllamaAPI(
      model_name=model_name,
      embed_model_name=embed_model_name,
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
  if _SKIP_GENERATE.value:
    print('1-8. Generation, chat, and multimodal tests. [SKIPPED]')
  else:
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
        llm.generate_text(
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
        llm.generate_text(
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
        llm.generate_text(
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
            executable=llm.generate_text(
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
        llm.generate_text(prompt='In summer', max_tokens=_MAX_TOKENS.value),
        llm.generate_text(prompt='In winter', max_tokens=_MAX_TOKENS.value),
        llm.generate_text(prompt='In autumn', max_tokens=_MAX_TOKENS.value),
    ])
    res = ot.run(exe)
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    # ==================== include_details ====================
    print('4. Check that include_details returns text and score.')
    res = ot.run(
        llm.generate_text(
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
        llm.generate_texts(
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
            max_tokens=_MAX_TOKENS.value,
        )
    )
    if _PRINT_DEBUG.value:
      print('Returned value(s):')
      pprint.pprint(res)

    # ==================== Sampling parameters ====================
    print('7. Generate with temperature.')
    res = ot.run(
        llm.generate_text(
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
        llm.generate_text(
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
            llm.chat(
                messages=messages,
                max_tokens=_MAX_TOKENS.value,
            )
        )
        if _PRINT_DEBUG.value:
          print('Returned value(s):')
          pprint.pprint(res)
      else:
        print(f'Skipping multimodal test: {image_path} not found.')

  # ==================== Embedding tests ====================
  if _SKIP_EMBEDDING.value:
    print('9. Embedding test. [SKIPPED]')
  else:
    print('9. Check that embed is working (plain string).')
    res = ot.run(
        llm.embed(content='The quick brown fox jumps over the lazy dog.')
    )
    if not isinstance(res, (list, tuple)) or not all(
        isinstance(x, float) for x in res
    ):
      success = False
      print('FAILURE: Returned value is not a list of floats:', res)
    else:
      print(f'  Returned {len(res)} floats. OK.')
    if _PRINT_DEBUG.value:
      result_string = str(res)
      if len(result_string) > 400:
        result_string = (
            result_string[:200]
            + f'... (skipped {len(result_string) - 200} characters) ...'
            + result_string[-200:]
        )
      print('Returned value(s):')
      pprint.pprint(result_string)

    print('9.1 Check that embed works with ChunkList.')
    content = content_lib.ChunkList(
        chunks=[
            content_lib.Chunk(
                content='The quick brown fox jumps over the lazy dog.',
            )
        ]
    )
    res_chunks = ot.run(llm.embed(content=content))
    if not isinstance(res_chunks, (list, tuple)) or not all(
        isinstance(x, float) for x in res_chunks
    ):
      success = False
      print('FAILURE: Returned value is not a list of floats:', res_chunks)
    else:
      print(f'  Returned {len(res_chunks)} floats. OK.')

    print('9.2 Same query to see if embed is cached.')
    counters_before = dict(backend._counters)
    res_cached = ot.run(
        llm.embed(content='The quick brown fox jumps over the lazy dog.')
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
    if res != res_cached:
      success = False
      print('FAILURE: cached result does not match original result.')

    print('10. Embedding norm check (default and normalized).')
    embed_content = 'The quick brown fox jumps over the lazy dog.'
    res = ot.run(llm.embed(content=embed_content))
    vec_norm = math.sqrt(sum(x * x for x in res))
    print(f'  Default vector length={len(res)} | norm={vec_norm:.6f}')

    res_norm = ot.run(
        llm.embed(content=embed_content, normalize=True)
    )
    vec_norm_2 = math.sqrt(sum(x * x for x in res_norm))
    print(f'  Normalized vector length={len(res_norm)} | norm={vec_norm_2:.6f}')
    if not math.isclose(vec_norm_2, 1.0, rel_tol=1e-3):
      success = False
      print(f'FAILURE: expected norm close to 1.0, got {vec_norm_2}')

    print('11. Three batched embed queries.')
    exe = ot.par_iter([
        llm.embed(content='In summer'),
        llm.embed(content='In winter'),
        llm.embed(content='In autumn'),
    ])
    res = ot.run(exe)
    if len(res) != 3:
      success = False
      print('FAILURE: returned %d results, expected 3.' % len(res))
    else:
      for i, emb in enumerate(res):
        if not isinstance(emb, (list, tuple)):
          success = False
          print(f'FAILURE: result {i} is not a list/tuple: {type(emb)}')
        else:
          print(f'  embed[{i}]: {len(emb)} floats.')

    print('12. Embedding similarity check.')
    emb_a = ot.run(llm.embed(content='The cat sat on the mat.'))
    emb_b = ot.run(llm.embed(content='A kitten was sitting on a rug.'))
    emb_c = ot.run(llm.embed(content='Quantum physics is fascinating.'))

    # Cosine similarity helper.
    def _cosine_sim(u, v):
      dot = sum(a * b for a, b in zip(u, v))
      nu = math.sqrt(sum(a * a for a in u))
      nv = math.sqrt(sum(b * b for b in v))
      return dot / (nu * nv) if nu > 0 and nv > 0 else 0.0

    sim_ab = _cosine_sim(emb_a, emb_b)
    sim_ac = _cosine_sim(emb_a, emb_c)
    print(f'  sim(cat/kitten)    = {sim_ab:.4f}')
    print(f'  sim(cat/quantum)   = {sim_ac:.4f}')
    if sim_ab <= sim_ac:
      success = False
      print(
          'FAILURE: expected similar sentences to have higher cosine'
          ' similarity.'
      )

    print('12.1. Embedding dimensionality truncation test with normalization.')
    default_len = len(emb_a)
    target_dim = default_len // 2
    if target_dim > 0:
      print(
          f'  Default dimension is {default_len}. Requesting {target_dim} with'
          ' normalization...'
      )
      res_truncated = ot.run(
          llm.embed(
              content='The cat sat on the mat.',
              output_dimensionality=target_dim,
              normalize=True,
          )
      )
      norm_trunc = math.sqrt(sum(x * x for x in res_truncated))
      if len(res_truncated) != target_dim:
        success = False
        print(
            f'FAILURE: expected truncated length {target_dim}, got'
            f' {len(res_truncated)}'
        )
      elif not math.isclose(norm_trunc, 1.0, rel_tol=1e-3):
        success = False
        print(f'FAILURE: expected norm close to 1.0, got {norm_trunc}')
      else:
        print(
            f'  Successfully received {len(res_truncated)} floats with norm'
            f' {norm_trunc:.6f}. OK.'
        )
    else:
      print('  Skipped: default dimension is too small to truncate.')

  # ==================== Save cache ====================
  if not _LOAD_CACHE.value:
    start = time.time()
    assert isinstance(backend.cache, caching.SimpleFunctionCache)
    backend.cache.save(overwrite=True)
    print('Took %.4fsec saving cache to %s.' % (time.time() - start, fname))

  print('PASS' if success else 'FAIL')


if __name__ == '__main__':
  app.run(main)
