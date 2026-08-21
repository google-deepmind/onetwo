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

"""Loader for the HotpotQA dataset."""

from collections.abc import Iterable
import dataclasses
import functools
from typing import Any

from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
import tensorflow_datasets as tfds

_Document = retrieval_data_structures.Document
_ExampleDict = datasets_lib.ExampleDict


@dataclasses.dataclass
class HotpotQADatasetLoader(datasets_lib.DatasetLoader[_ExampleDict]):
  """Loads the HotpotQA dataset via TFDS, returning examples and documents.

  This loader uses TFDS's HuggingFace bridge to load the `hotpot_qa` dataset
  and provides methods to return evaluation-compatible examples and `Document`
  objects. A `Document` is a dataclass with 'doc_id', 'title', and 'content'
  fields. `Document` objects are useful for storing information about documents
  that can be used for retrieval. For more information on `Documents` and how
  they can be retrieved, see
  `onetwo.stdlib.retrieval.retrieval_data_structures` and
  `onetwo.stdlib.retrieval.indexing`.

  Attributes:
    name: The HotpotQA configuration name. One of 'distractor' or 'fullwiki'.
    split: The dataset split to load (e.g. 'train', 'validation', 'test').
    max_number_of_examples: If set, only the first N examples are loaded. If
      None, all examples are loaded.
    max_number_of_documents: If set, only the first N documents are loaded. If
      None, all documents are loaded.
  """

  name: str = 'fullwiki'
  split: str = 'validation'
  max_number_of_examples: int | None = None
  max_number_of_documents: int | None = None

  def _build_documents_from_hotpotqa_context(
      self,
      context: dict[str, Any],
  ) -> list[_Document]:
    """Builds a list of Document objects from a HotpotQA context field.

    The TFDS HotpotQA context structure contains:
      - 'title': a list/array of paragraph titles.
      - 'sentences': a list/array of lists of sentences (one list per
        paragraph).

    Args:
      context: The 'context' field from a HotpotQA example.

    Returns:
      A list of Document objects, one per context paragraph.
    """
    documents = []
    titles = context.get('title', [])
    sentences_lists = context.get('sentences', [])
    for title, sentences in zip(titles, sentences_lists):
      title_str = datasets_lib.decode_value(title)
      sent_strs = datasets_lib.decode_list(sentences)
      content = ' '.join(sent_strs)
      documents.append(
          _Document(
              doc_id=title_str,
              title=title_str,
              content=content,
          )
      )
    return documents

  def _build_example_from_hotpotqa_row(
      self, row: dict[str, Any]
  ) -> _ExampleDict:
    """Converts a single HotpotQA row into an evaluation Example dict.

    The returned dict contains 'question' and 'answer' keys (compatible with
    the default extractors of `evaluation.evaluate`), plus a 'metadata' dict
    with additional HotpotQA-specific fields.

    Args:
      row: A single row from the HotpotQA dataset (with numpy/bytes values).

    Returns:
      A dictionary suitable for use as an evaluation example.
    """
    return {
        datasets_lib.EXAMPLE_FIELD_QUESTION: datasets_lib.decode_value(
            row['question']
        ),
        datasets_lib.EXAMPLE_FIELD_ANSWER: datasets_lib.decode_value(
            row['answer']
        ),
        datasets_lib.EXAMPLE_FIELD_METADATA: {
            'id': datasets_lib.decode_value(row.get('id', b'')),
            'type': datasets_lib.decode_value(row.get('type', b'')),
            'level': datasets_lib.decode_value(row.get('level', b'')),
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: row.get(
                'supporting_facts', {}
            ),
        },
    }

  def _load_hotpotqa_tfds_numpy_dataset(self):
    """Loads the HotpotQA dataset via TFDS and returns it as a numpy iterator.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = f'huggingface:hotpotqa__hotpot_qa/{self.name}'

    ds = tfds.load(tfds_name, split=self.split)
    return tfds.as_numpy(ds)

  @executing.make_executable()
  async def load_examples(self) -> Iterable[_ExampleDict]:
    """Loads the examples of the HotpotQA dataset.

    Returns:
      An iterable of evaluation example dictionaries.
    """
    ds_numpy = self._load_hotpotqa_tfds_numpy_dataset()
    examples: list[_ExampleDict] = []
    for row in ds_numpy:
      if (
          self.max_number_of_examples is not None
          and len(examples) == self.max_number_of_examples
      ):
        break
      examples.append(self._build_example_from_hotpotqa_row(row))
    return examples

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads the documents of the HotpotQA dataset.

    Returns:
      An iterable of Document objects, deduplicated by doc_id.
    """
    ds_numpy = self._load_hotpotqa_tfds_numpy_dataset()
    documents: list[_Document] = []
    count = 0
    for row in ds_numpy:
      if (
          self.max_number_of_documents is not None
          and len(documents) == self.max_number_of_documents
      ):
        break
      documents.extend(
          self._build_documents_from_hotpotqa_context(row['context'])
      )
      count += 1
    # Deduplicate documents by doc_id (the same paragraph can appear in the
    # context of multiple examples).
    seen_ids: set[str] = set()
    unique_documents: list[_Document] = []
    for doc in documents:
      if doc.doc_id not in seen_ids:
        seen_ids.add(doc.doc_id)
        unique_documents.append(doc)
    return unique_documents


# Register dataset loaders in the registry.
datasets_lib.register_dataset_loader(
    'hotpotqa_train',
    functools.partial(HotpotQADatasetLoader, split='train'),
)
datasets_lib.register_dataset_loader(
    'hotpotqa_validation',
    functools.partial(HotpotQADatasetLoader, split='validation'),
)
# NOTE: 'hotpotqa_test' does not contain answers to the questions.
datasets_lib.register_dataset_loader(
    'hotpotqa_test',
    functools.partial(HotpotQADatasetLoader, split='test'),
)
