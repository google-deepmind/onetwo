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

"""Loader for the NarrativeQA dataset."""

from collections.abc import Iterable
import dataclasses
import functools
import re
from typing import Any

import bs4
from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
import tensorflow_datasets as tfds

_Document = retrieval_data_structures.Document
_ExampleDict = datasets_lib.ExampleDict


@dataclasses.dataclass
class NarrativeQADatasetLoader(datasets_lib.DatasetLoader[_ExampleDict]):
  """Loads a modified NarrativeQA dataset via TFDS.

  This is a modified version of the original NarrativeQA dataset designed to
  make all questions share a single corpus. The book or script title is
  prepended to each question so that the model must first identify the
  relevant document in the corpus and then answer the question.

  For example, the original question:
    "What injury does the infant T. rex suffer from?"
  becomes:
    "In Jurassic Park: The Lost World, what injury does the infant T. rex
    suffer from?"

  This loader uses TFDS's HuggingFace bridge to load the `narrativeqa` dataset
  and provides methods to return evaluation-compatible examples and `Document`
  objects.

  The returned example dictionaries contain:
    - question: The question text, prepended with the document title.
    - answer: A list of answer texts.
    - metadata: Dictionary with 'golden_doc_ids' (list containing the ground
      truth document id).

  The returned `Document` objects contain:
    - doc_id: Unique identifier from the dataset.
    - title: The title of the book or script.
    - content: The full text of the document (optionally cleaned of HTML).
    - metadata: Dictionary with 'url', 'summary', and 'category'.

  Attributes:
    split: The dataset split to load (e.g. 'train', 'validation', 'test').
    max_number_of_examples: If set, only the first N examples (and their
      corresponding documents) are loaded. If None, all examples are loaded.
    clean_html: Whether to convert the html to raw text.
    doc_ids: If set, only examples and documents with these IDs are loaded.
  """

  split: str = 'train'
  max_number_of_examples: int | None = None
  clean_html: bool = True
  doc_ids: list[str] | None = None

  @classmethod
  def _clean_text(cls, text: str) -> str:
    """Removes html tags and trims unnecessary whitespace and new lines."""
    soup = bs4.BeautifulSoup(text, 'html.parser')
    pre = soup.find('pre')
    if pre:
      soup = pre
    text = soup.get_text()
    # Replace multiple spaces with just one.
    text = re.sub(' +', ' ', text)
    # Remove trailing spaces and new lines.
    text = text.strip()
    # Strip all lines.
    text = '\n'.join([line.strip() for line in text.splitlines()])
    # Ensure there are at most two successive new lines.
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

  def _build_document_from_row(self, row: dict[str, Any]) -> _Document:
    raw_doc = row['document']
    doc_id = datasets_lib.decode_value(raw_doc['id'])
    content = datasets_lib.decode_value(raw_doc['text'])
    if self.clean_html:
      content = self._clean_text(content)
    url = datasets_lib.decode_value(raw_doc['url'])
    title = datasets_lib.decode_value(raw_doc['summary']['title'])
    return _Document(
        doc_id=doc_id,
        title=title,
        content=content,
        metadata={
            'url': url,
            'summary': datasets_lib.decode_value(raw_doc['summary']['text']),
            'category': datasets_lib.decode_value(raw_doc['kind']),
        },
    )

  def _build_example_from_row(self, row: dict[str, Any]) -> _ExampleDict:
    title = datasets_lib.decode_value(row['document']['summary']['title'])
    question = datasets_lib.decode_value(row['question']['text'])
    return {
        datasets_lib.EXAMPLE_FIELD_QUESTION: (
            f'In {title}, {question[0].lower()}{question[1:]}'
        ),
        datasets_lib.EXAMPLE_FIELD_ANSWER: datasets_lib.decode_list(
            row['answers']['text']
        ),
        datasets_lib.EXAMPLE_FIELD_METADATA: {
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: [
                datasets_lib.decode_value(row['document']['id'])
            ],
        },
    }

  def _load_tfds_numpy_dataset(self):
    tfds_name = 'huggingface:deepmind__narrativeqa'

    ds = tfds.load(tfds_name, split=self.split)
    return tfds.as_numpy(ds)

  @executing.make_executable()
  async def load_examples(self) -> Iterable[_ExampleDict]:
    """Loads the examples of the NarrativeQA dataset."""
    ds_numpy = self._load_tfds_numpy_dataset()
    examples: list[_ExampleDict] = []
    for row in ds_numpy:
      doc_id = datasets_lib.decode_value(row['document']['id'])
      if self.doc_ids is not None and doc_id not in self.doc_ids:
        continue
      if (
          self.max_number_of_examples is not None
          and len(examples) == self.max_number_of_examples
      ):
        break
      example = self._build_example_from_row(row)
      examples.append(example)
    return examples

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads the documents of the NarrativeQA dataset."""
    ds_numpy = self._load_tfds_numpy_dataset()
    documents: list[_Document] = []
    seen_ids: set[str] = set()
    for row in ds_numpy:
      doc_id = datasets_lib.decode_value(row['document']['id'])

      if self.doc_ids is not None and doc_id not in self.doc_ids:
        continue
      if doc_id not in seen_ids:
        if (
            self.max_number_of_examples is not None
            and len(documents) == self.max_number_of_examples
        ):
          break
        seen_ids.add(doc_id)
        documents.append(self._build_document_from_row(row))
    return documents


# Register dataset loaders in the registry.
datasets_lib.register_dataset_loader(
    'narrativeqa_train',
    functools.partial(NarrativeQADatasetLoader, split='train'),
)
datasets_lib.register_dataset_loader(
    'narrativeqa_validation',
    functools.partial(NarrativeQADatasetLoader, split='validation'),
)
datasets_lib.register_dataset_loader(
    'narrativeqa_test',
    functools.partial(NarrativeQADatasetLoader, split='test'),
)
