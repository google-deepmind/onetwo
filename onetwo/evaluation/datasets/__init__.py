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

"""Loaders for QA datasets, returning OneTwo-compatible structures.

This package provides functions to load question answering (QA) datasets and
convert them into iterables of OneTwo evaluation examples (dictionaries with
'question' and 'answer' keys, compatible with `evaluation.evaluate`) and,
when applicable, iterables of `retrieval_data_structures.Document` objects.

To load datasets such as HotpotQA, this library uses TensorFlow Datasets (TFDS)
which provides a built-in bridge for HuggingFace datasets via the 'huggingface:'
prefix.

Usage example:

  ```python
  from onetwo.evaluation import datasets

  loader = datasets.HotpotQADatasetLoader(
      split='validation', max_number_of_examples=100
  )
  examples = ot.run(loader.load_examples())
  documents = ot.run(loader.load_documents())
  ```
"""

# pylint: disable=g-importing-member,g-bad-import-order
# Re-export tensorflow_datasets for compatibility with existing tests/mocks.

import tensorflow_datasets as tfds

# Re-export everything from datasets_lib and dataset modules.
from onetwo.evaluation.datasets.browsecomp_plus import BrowseCompPlusDatasetLoader
from onetwo.evaluation.datasets.datasets_lib import *
from onetwo.evaluation.datasets.hotpotqa import HotpotQADatasetLoader
from onetwo.evaluation.datasets.multihoprag import MultiHopRAGDatasetLoader
from onetwo.evaluation.datasets.narrativeqa import NarrativeQADatasetLoader
