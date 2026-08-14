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

"""Manual tests for strategies_lib using live Google GenAI backends.

These tests exercise the strategies (closed-book, long-context, RAG) end-to-end
against a real Gemini backend, including an embedding-based index for RAG.

"""

from collections.abc import Iterator, Sequence
import contextlib

_Document = retrieval_data_structures.Document
_Chunk = content_lib.Chunk
_ChunkList = content_lib.ChunkList

_API_KEY = flags.DEFINE_string('api_key', default=None, help='GenAI API key.')
_PROJECT = flags.DEFINE_string('project', default=None, help='Project ID.')
_LOCATION = flags.DEFINE_string('location', default=None, help='Location.')
_RUN_ONLY = flags.DEFINE_list('run_only', default=None, help='Test filter.')
_PRINT_DEBUG = flags.DEFINE_bool(
    'print_debug', default=False, help='Debug logging.'
)
flags.mark_flags_as_mutual_exclusive(['api_key', 'project'])
flags.mark_flags_as_mutual_exclusive(['api_key', 'location'])


# =============================================================================
# Test infrastructure (same pattern as google_genai_api_manual_tests.py)
# =============================================================================


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
  """Context manager that runs a test and catches any exceptions."""
  print(f'\n{num}. {description}.')
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


# =============================================================================
# Document corpora
# =============================================================================

# Text-only documents about world capitals.
_CAPITAL_DOCS = [
    _Document(
        doc_id='france',
        content=(
            'Paris is the capital city of France. It is known for the Eiffel'
            ' Tower, the Louvre Museum, and the Seine river.'
        ),
    ),
    _Document(
        doc_id='germany',
        content=(
            'Berlin is the capital city of Germany. It is known for the'
            ' Brandenburg Gate and the Berlin Wall.'
        ),
    ),
    _Document(
        doc_id='spain',
        content=(
            'Madrid is the capital city of Spain. It is known for the Royal'
            ' Palace and the Prado Museum.'
        ),
    ),
    _Document(
        doc_id='japan',
        content=(
            'Tokyo is the capital city of Japan. It is known for its'
            ' skyscrapers, electronics shops, and cherry blossoms.'
        ),
    ),
    _Document(
        doc_id='australia',
        content=(
            'Canberra is the capital city of Australia. It is home to many'
            ' national monuments and galleries.'
        ),
    ),
]


# Multimodal documents (text + image bytes from testdata).
def _load_multimodal_docs() -> list[_Document]:
  """Creates documents with real image bytes from testdata."""
  image_path = os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'backends',
      'testdata',
      'bird.jpg',
  )
  docs = []
  if os.path.exists(image_path):
    with open(image_path, 'rb') as f:
      image_bytes = f.read()
    docs.append(
        _Document(
            doc_id='bird_photo',
            content=_ChunkList([
                _Chunk('A photograph of a bird, likely a chickadee.'),
                _Chunk(image_bytes, content_type='image/jpeg'),
            ]),
        )
    )
  else:
    logging.warning(
        'Image file not found at %s, skipping image doc.', image_path
    )
    docs.append(
        _Document(
            doc_id='bird_photo',
            content='A photograph of a bird, likely a chickadee.',
        )
    )
  docs.append(
      _Document(
          doc_id='sunset_description',
          content=(
              'The sunset over the Pacific ocean is a breathtaking sight with'
              ' orange, pink, and purple hues painting the sky.'
          ),
      ),
  )
  return docs


# =============================================================================
# Main
# =============================================================================


def main(argv: Sequence[str]) -> None:
  test_results: list[tuple[float, str, bool]] = []
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # -----------------------------------------------------------------------
  # Backend setup (follows google_genai_api_manual_tests.py pattern).
  # -----------------------------------------------------------------------
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
      'thinking_config': genai_types.ThinkingConfig(
          include_thoughts=False,
          thinking_budget=32,
      ),
  }

  # LLM backend for chat / generation.
  llm_backend = google_genai_api.GoogleGenAIAPI(
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location=location,
      threadpool_size=4,
      max_retries=5,
      generate_text_kwargs=generate_config,
      generate_object_kwargs=generate_config,
      generate_content_kwargs=generate_config,
      chat_kwargs=generate_config,
  )

  # Embedding backend: gemini-embedding-2 with 768 dimensions.
  embed_backend = google_genai_api.GoogleGenAIAPI(
      embed_model_name='gemini-embedding-2',
      api_key=api_key,
      vertexai=vertexai,
      project=project,
      location='us' if vertexai else None,
      threadpool_size=4,
      max_retries=5,
      embed_kwargs={'output_dimensionality': 768},
  )

  # Register both backends: LLM for chat, embedding backend for embed.
  llm_backend.register(register_embed=False)
  embed_backend.register(register_generate=False, register_tokenize=False)

  time_start = time.time()

  # -----------------------------------------------------------------------
  # Test 1: Closed-book strategy
  # -----------------------------------------------------------------------
  if _should_run_test(1):
    with _run_test(
        1, 'ClosedBookStrategy: simple text question', test_results
    ) as tc:
      strategy = strategies_lib.ClosedBookStrategy()
      result = executing.run(
          strategy(question='What is the capital of France?')
      )
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if 'paris' not in result.answer.lower():
        tc.fail(f'Expected "paris" in answer, got: {result.answer}')
      if result.docs:
        tc.fail(f'Expected empty docs for closed-book, got: {result.docs}')

  # -----------------------------------------------------------------------
  # Test 2: Long-context strategy with text documents
  # -----------------------------------------------------------------------
  if _should_run_test(2):
    with _run_test(
        2, 'LongContextStrategy: text docs, capital question', test_results
    ) as tc:
      strategy = strategies_lib.LongContextStrategy(documents=_CAPITAL_DOCS)
      result = executing.run(strategy(question='What is the capital of Japan?'))
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if 'tokyo' not in result.answer.lower():
        tc.fail(f'Expected "tokyo" in answer, got: {result.answer}')
      if len(result.docs) != len(_CAPITAL_DOCS):
        tc.fail(f'Expected {len(_CAPITAL_DOCS)} docs, got {len(result.docs)}')

  # -----------------------------------------------------------------------
  # Test 3: RAG strategy with embedding-based index (text docs)
  # -----------------------------------------------------------------------
  if _should_run_test(3):
    with _run_test(
        3,
        'RAGStrategy: embedding index, text docs, capital question',
        test_results,
    ) as tc:
      # Build an embedding-based index.
      emb_index = indexing.EmbeddingBasedDocumentIndex()
      executing.run(
          emb_index.create_index(corpus_name='capitals', docs=_CAPITAL_DOCS)
      )
      print(f'  Index built with {emb_index.num_docs} docs.')

      strategy = strategies_lib.RAGStrategy(retriever=emb_index)  # pyrefly: ignore[bad-argument-type]
      result = executing.run(
          strategy(question='What is the capital of Spain?', max_results=2)
      )
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if 'madrid' not in result.answer.lower():
        tc.fail(f'Expected "madrid" in answer, got: {result.answer}')
      if len(result.docs) != 2:
        tc.fail(f'Expected 2 docs (max_results=2), got {len(result.docs)}')
      if 'spain' not in [d.doc_id for d in result.docs]:
        tc.fail(
            f'Expected "spain" in docs, got: {[d.doc_id for d in result.docs]}'
        )

  # -----------------------------------------------------------------------
  # Test 4: RAG strategy — verify retrieval relevance
  # -----------------------------------------------------------------------
  if _should_run_test(4):
    with _run_test(
        4,
        'RAGStrategy: embedding retrieval returns relevant docs first',
        test_results,
    ) as tc:
      emb_index = indexing.EmbeddingBasedDocumentIndex()
      executing.run(
          emb_index.create_index(corpus_name='capitals', docs=_CAPITAL_DOCS)
      )
      strategy = strategies_lib.RAGStrategy(retriever=emb_index)  # pyrefly: ignore[bad-argument-type]
      result = executing.run(
          strategy(
              question='Tell me about the Brandenburg Gate in Germany.',
              max_results=1,
          )
      )
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if result.docs[0].doc_id != 'germany':
        tc.fail(
            'Expected "germany" as first retrieved doc_id, got:'
            f' {result.docs[0].doc_id}'
        )

  # -----------------------------------------------------------------------
  # Test 5: Long-context with multimodal documents
  # -----------------------------------------------------------------------
  if _should_run_test(5):
    with _run_test(
        5,
        'LongContextStrategy: multimodal docs',
        test_results,
    ) as tc:
      mm_docs = _load_multimodal_docs()
      strategy = strategies_lib.LongContextStrategy(documents=mm_docs)
      result = executing.run(
          strategy(question='What kind of bird is in the photo?')
      )
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if not result.answer:
        tc.fail('Expected a non-empty answer.')
      if len(result.docs) != len(mm_docs):
        tc.fail(f'Expected {len(mm_docs)} docs, got {len(result.docs)}')

  # -----------------------------------------------------------------------
  # Test 6: RAG with multimodal documents
  # -----------------------------------------------------------------------
  if _should_run_test(6):
    with _run_test(
        6,
        'RAGStrategy: multimodal docs',
        test_results,
    ) as tc:
      mm_docs = _load_multimodal_docs()
      emb_index = indexing.EmbeddingBasedDocumentIndex()
      executing.run(
          emb_index.create_index(corpus_name='multimodal', docs=mm_docs)
      )
      print(f'  Index built with {emb_index.num_docs} docs.')
      strategy = strategies_lib.RAGStrategy(retriever=emb_index)  # pyrefly: ignore[bad-argument-type]
      result = executing.run(
          strategy(question='What kind of bird is in the photo?', max_results=1)
      )
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if 'bird' not in result.answer.lower():
        tc.fail(f'Expected "bird" in answer, got: {result.answer}')
      if 'bird_photo' not in [d.doc_id for d in result.docs]:
        tc.fail(
            'Expected "bird_photo" in docs, got:'
            f' {[d.doc_id for d in result.docs]}'
        )

  # -----------------------------------------------------------------------
  # Test 7: Strategy registry — get_strategy() works end-to-end
  # -----------------------------------------------------------------------
  if _should_run_test(7):
    with _run_test(
        7,
        'get_strategy("closed_book") end-to-end',
        test_results,
    ) as tc:
      strategy = strategies_lib.get_strategy('closed_book')
      result = executing.run(
          strategy(question='What color is the sky on a clear day?')
      )
      print(f'  Answer: {result.answer}')
      if 'blue' not in result.answer.lower():
        tc.fail(f'Expected "blue" in answer, got: {result.answer}')

  # -----------------------------------------------------------------------
  # Test 8: RAG with multimodal question (ChunkList query)
  # -----------------------------------------------------------------------
  if _should_run_test(8):
    with _run_test(
        8,
        'RAGStrategy: multimodal question (ChunkList)',
        test_results,
    ) as tc:
      emb_index = indexing.EmbeddingBasedDocumentIndex()
      executing.run(
          emb_index.create_index(corpus_name='capitals', docs=_CAPITAL_DOCS)
      )
      # Use a ChunkList question (text-only, but wrapped as multimodal).
      mm_question = _ChunkList([
          _Chunk('I want to visit a European capital known for the '),
          _Chunk('Eiffel Tower. Which country is it in?'),
      ])
      strategy = strategies_lib.RAGStrategy(retriever=emb_index)  # pyrefly: ignore[bad-argument-type]
      result = executing.run(strategy(question=mm_question, max_results=2))
      print(f'  Answer: {result.answer}')
      print(f'  Doc IDs: {[d.doc_id for d in result.docs]}')
      if 'france' not in result.answer.lower():
        tc.fail(f'Expected "france" in answer, got: {result.answer}')

  # -----------------------------------------------------------------------
  # Test 9: Closed-book with a harder question (sanity check)
  # -----------------------------------------------------------------------
  if _should_run_test(9):
    with _run_test(
        9,
        'ClosedBookStrategy: harder factual question',
        test_results,
    ) as tc:
      strategy = strategies_lib.ClosedBookStrategy()
      result = executing.run(
          strategy(question='What is the approximate speed of light in km/s?')
      )
      print(f'  Answer: {result.answer}')
      # Speed of light ≈ 300,000 km/s.
      if '300' not in result.answer:
        tc.fail(
            f'Expected "300" (as in 300,000 km/s) in answer: {result.answer}'
        )

  # -----------------------------------------------------------------------
  # Test 10: Long-context with 'only_from_context' template
  # -----------------------------------------------------------------------
  if _should_run_test(10):
    with _run_test(
        10,
        "LongContextStrategy: template 'only_from_context' refuses",
        test_results,
    ) as tc:
      prompt_builder = qa_prompt_builder.QADocumentPromptBuilder(
          template=strategies_lib.default_qa_template_only_from_context
      )
      strategy = strategies_lib.LongContextStrategy(
          documents=_CAPITAL_DOCS,
          prompt_builder=prompt_builder,
      )
      result = executing.run(
          strategy(question='What is the capital of Morocco?')
      )
      print(f'  Answer: {result.answer}')
      expected_phrase = "i don't have enough context"
      if expected_phrase not in result.answer.lower():
        tc.fail(f'Expected "{expected_phrase}" in answer, got: {result.answer}')

  # -----------------------------------------------------------------------
  # Summary
  # -----------------------------------------------------------------------
  time_end = time.time()
  print(f'\nTook {time_end - time_start:.2f}s running all tests.')
  _print_summary(test_results)


if __name__ == '__main__':
  app.run(main)
