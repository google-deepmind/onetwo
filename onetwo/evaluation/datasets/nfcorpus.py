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

"""Loader for the NFCorpus dataset.

NFCorpus (Nutrition Information Retrieval Corpus) is a full-text information
retrieval benchmark constructed for medical and nutrition-related queries.
Unlike standard question-answering (QA) datasets, NFCorpus does not contain
natural language answers. Instead, it consists of queries paired with relevant
documents from a corpus along with numerical relevance scores (qrels).
"""

import collections
from collections.abc import Iterable, Mapping
import dataclasses
import functools
from typing import Any, ClassVar

from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
import tensorflow_datasets as tfds

_Document = retrieval_data_structures.Document
_ExampleDict = datasets_lib.ExampleDict


@dataclasses.dataclass
class NFCorpusDatasetLoader(datasets_lib.DatasetLoader[_ExampleDict]):
  """Loads the NFCorpus dataset via TFDS, returning examples and documents.

  This loader uses TFDS's HuggingFace bridge to load the `beir/nfcorpus`
  queries, corpus, and qrels datasets, providing methods to return
  evaluation-compatible examples and `Document` objects.

  Note:
    NFCorpus is an information retrieval dataset and does not contain natural
    language answers. Each example contains a query and a dictionary of
    relevant golden document IDs mapped to their numerical relevance scores.

  The returned example dictionaries contain:
    - question (`EXAMPLE_FIELD_QUESTION`): The query text.
    - metadata (`EXAMPLE_FIELD_METADATA`): Dictionary containing:
        - 'id': The query ID.
        - 'split': The dataset split loaded (e.g. 'test', 'train',
        'validation').
        - 'golden_doc_scores' (`EXAMPLE_METADATA_FIELD_GOLDEN_DOC_SCORES`):
          A mapping from golden document ID to relevance score (e.g.,
          `{'doc1': 1, 'doc2': 2}`).
        - 'golden_doc_ids' (`EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS`): A list of
          golden document IDs (e.g., `['doc1', 'doc2']`).

  The returned `Document` objects contain:
    - doc_id: The document's unique identifier in the corpus.
    - title: Document title.
    - content: Full text body of the document.

  Attributes:
    split: The dataset split to load ('all', 'test', 'train', 'validation').
    max_number_of_examples: If set, only the first N examples are loaded. If
      None, all examples are loaded.
    max_number_of_documents: If set, only the first N documents are loaded. If
      None, all documents are loaded.
  """

  split: str = 'test'
  max_number_of_examples: int | None = None
  max_number_of_documents: int | None = None

  _corpus_split: ClassVar[str] = 'corpus'
  _queries_split: ClassVar[str] = 'queries'

  def _load_queries_tfds_numpy_dataset(self) -> Any:
    """Loads the NFCorpus queries dataset via TFDS.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = 'huggingface:beir__nfcorpus/queries'
    ds = tfds.load(tfds_name, split=self._queries_split)

    return tfds.as_numpy(ds)

  def _load_corpus_tfds_numpy_dataset(self) -> Any:
    """Loads the NFCorpus corpus dataset via TFDS.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = 'huggingface:beir__nfcorpus/corpus'
    ds = tfds.load(tfds_name, split=self._corpus_split)

    return tfds.as_numpy(ds)

  def _load_qrels_tfds_numpy_dataset(self) -> Any:
    """Loads the NFCorpus qrels dataset via TFDS.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = 'huggingface:beir__nfcorpus_qrels'
    ds = tfds.load(tfds_name, split=self.split)

    return tfds.as_numpy(ds)

  def _build_example_from_row(
      self,
      row: Mapping[str, Any],
      golden_doc_scores: Mapping[str, int],
  ) -> _ExampleDict:
    """Converts a single queries dataset row into an evaluation Example dict.

    Args:
      row: A single row from the queries dataset (with numpy/bytes values).
      golden_doc_scores: Mapping of golden document IDs to their scores.

    Returns:
      A dictionary suitable for use as an evaluation example.
    """
    query_id = datasets_lib.decode_value(
        row.get('_id', row.get('id', row.get('query_id', b'')))
    )
    query = datasets_lib.decode_value(row.get('text', row.get('query', b'')))
    golden_doc_scores = dict(golden_doc_scores)
    golden_doc_ids = list(golden_doc_scores.keys())

    return {
        datasets_lib.EXAMPLE_FIELD_QUESTION: query,
        # NFCorpus is an IR dataset without answers. Setting the answer to an
        # empty string for compatibility with default extractors in the
        # evaluation library.
        datasets_lib.EXAMPLE_FIELD_ANSWER: '',
        datasets_lib.EXAMPLE_FIELD_METADATA: {
            'id': query_id,
            'split': self.split,
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_SCORES: (
                golden_doc_scores
            ),
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: golden_doc_ids,
        },
    }

  def _build_document_from_corpus_row(
      self, row: Mapping[str, Any]
  ) -> _Document:
    """Converts a single corpus row into a Document object.

    Args:
      row: A single row from the corpus dataset (with numpy/bytes values).

    Returns:
      A Document object.
    """
    doc_id = datasets_lib.decode_value(
        row.get('_id', row.get('id', row.get('doc_id', b'')))
    )
    title = datasets_lib.decode_value(row.get('title', b''))
    content = datasets_lib.decode_value(
        row.get('text', row.get('content', b''))
    )

    return _Document(
        doc_id=doc_id,
        title=title,
        content=content,
    )

  @executing.make_executable()
  async def load_examples(self) -> Iterable[_ExampleDict]:
    """Loads the examples of the NFCorpus dataset.

    Returns:
      An iterable of evaluation example dictionaries.
    """
    qrels_ds = self._load_qrels_tfds_numpy_dataset()
    qrels_map: dict[str, dict[str, int]] = collections.defaultdict(dict)
    if qrels_ds is not None:
      for row in qrels_ds:
        query_id = datasets_lib.decode_value(
            row.get('query-id', row.get('query_id', b''))
        )
        corpus_id = datasets_lib.decode_value(
            row.get('corpus-id', row.get('corpus_id', b''))
        )
        score = int(row.get('score', 0))

        if query_id and corpus_id and score > 0:
          qrels_map[query_id][corpus_id] = max(
              qrels_map[query_id].get(corpus_id, 0), score
          )

    ds_numpy = self._load_queries_tfds_numpy_dataset()
    examples: list[_ExampleDict] = []
    for row in ds_numpy:
      if (
          self.max_number_of_examples is not None
          and len(examples) == self.max_number_of_examples
      ):
        break
      query_id = datasets_lib.decode_value(
          row.get('_id', row.get('id', row.get('query_id', b'')))
      )
      if not qrels_map.get(query_id):
        continue
      examples.append(self._build_example_from_row(row, qrels_map[query_id]))
    return examples

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads the documents of the NFCorpus dataset.

    Returns:
      An iterable of Document objects.
    """
    ds_numpy = self._load_corpus_tfds_numpy_dataset()
    documents: dict[str, _Document] = {}
    for row in ds_numpy:
      if (
          self.max_number_of_documents is not None
          and len(documents) == self.max_number_of_documents
      ):
        break
      doc = self._build_document_from_corpus_row(row)
      if doc.doc_id in documents:
        continue
      documents[doc.doc_id] = doc
    return list(documents.values())


# Register dataset loaders in the registry.
datasets_lib.register_dataset_loader(
    'nfcorpus_test',
    functools.partial(NFCorpusDatasetLoader, split='test'),
)
datasets_lib.register_dataset_loader(
    'nfcorpus_train',
    functools.partial(NFCorpusDatasetLoader, split='train'),
)
datasets_lib.register_dataset_loader(
    'nfcorpus_validation',
    functools.partial(NFCorpusDatasetLoader, split='validation'),
)
