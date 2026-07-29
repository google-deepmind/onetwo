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

r"""Manual tests for dataset loaders.

This script checks that given a dataset_name, it gets the dataset loader from
the registry and runs load_documents and load_examples normally.

Usage:
  python3 -m onetwo.evaluation.datasets.datasets_manual_tests \
    --dataset_name=hotpotqa_validation --max_number_of_examples=5
"""

from collections.abc import Sequence
import time

from absl import app
from absl import flags
from onetwo import ot
from onetwo.evaluation import datasets

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    default='hotpotqa_validation',
    help='Name of the dataset loader in the registry to test.',
)

_MAX_NUMBER_OF_EXAMPLES = flags.DEFINE_integer(
    'max_number_of_examples',
    default=5,
    help='Maximum number of examples to load.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('=' * 60)
  print('DATASET MANUAL TESTS')
  print('=' * 60)

  available = datasets.list_dataset_loaders()
  print(f'Available registered dataset loaders: {available}')
  print(
      f'Testing dataset loader: {_DATASET_NAME.value} (max_number_of_examples='
      f'{_MAX_NUMBER_OF_EXAMPLES.value})'
  )

  start_time = time.time()
  try:
    loader = datasets.get_dataset_loader(
        _DATASET_NAME.value,
        max_number_of_examples=_MAX_NUMBER_OF_EXAMPLES.value,
    )
  except TypeError:
    # Some loaders (e.g. jsonl) may take different argument names.
    loader = datasets.get_dataset_loader(_DATASET_NAME.value)

  print(f'\n1. Loading examples from {_DATASET_NAME.value}...')
  examples = list(ot.run(loader.load_examples()))
  print(f'   Loaded {len(examples)} examples.')
  for i, example in enumerate(examples[:3]):
    print(f'   Example [{i}] question: {example.get("question")}')
    print(f'   Example [{i}] answer: {example.get("answer")}')

  print(f'\n2. Loading documents from {_DATASET_NAME.value}...')
  documents = list(ot.run(loader.load_documents()))  # pyrefly: ignore[missing-attribute]
  print(f'   Loaded {len(documents)} documents.')
  for i, document in enumerate(documents[:3]):
    print(f'   Document [{i}] doc_id: {document.doc_id}')
    print(f'   Document [{i}] title: {document.title}')

  duration = time.time() - start_time
  print('\n' + '=' * 60)
  print(f'PASS: Loaded {_DATASET_NAME.value} in {duration:.2f}s.')
  print('=' * 60)


if __name__ == '__main__':
  app.run(main)
