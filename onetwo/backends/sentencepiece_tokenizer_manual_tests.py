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

"""Manual test for SentencePieceTokenizer."""

from absl import app
from onetwo.backends import sentencepiece_tokenizer
from onetwo.builtins import llm
from onetwo.core import executing

SentencePieceTokenizer = sentencepiece_tokenizer.SentencePieceTokenizer


def main(argv):
  del argv  # Unused

  async def run_test_for_model(model_name, expected_tokens):
    print(f'\n--- Testing model: {model_name} ---')
    tokenizer = SentencePieceTokenizer(model_name=model_name)
    tokenizer.register()

    print('Tokenizing "Hello world"...')
    tokens = await llm.tokenize('Hello world')  # pytype: disable=wrong-arg-count
    print(f'Token IDs: {tokens}')

    if list(tokens) != expected_tokens:
      print(
          f'Tokenization failed! Expected {expected_tokens}, got {list(tokens)}'
      )
      return False

    print('Detokenizing back to text...')
    text = await llm.detokenize(tokens)  # pytype: disable=wrong-arg-count
    print(f'Detokenized text: "{text}"')

    if text == 'Hello world':
      print('Detokenization test passed!')
      return True
    else:
      print(f'Detokenization test failed! Expected "Hello world", got "{text}"')
      return False

  async def run_all_tests():
    results = []

    t5_url = 'https://huggingface.co/t5-small/resolve/main/spiece.model'
    results.append(await run_test_for_model(t5_url, [8774, 296]))

    if all(results):
      print('\nAll tests passed!')
    else:
      print('\nSome tests failed!')

  executing.run(run_all_tests())


if __name__ == '__main__':
  app.run(main)
