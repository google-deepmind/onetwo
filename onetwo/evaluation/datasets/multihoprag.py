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

"""Loader for the MultiHopRAG dataset."""

from collections.abc import Iterable
import dataclasses
import hashlib
from typing import Any

from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
import tensorflow_datasets as tfds

_Document = retrieval_data_structures.Document
_ExampleDict = datasets_lib.ExampleDict


def _generate_doc_id(title: str) -> str:
  """Generates a short deterministic document ID from a title.

  Args:
    title: The document title.

  Returns:
    A 12-character hex string derived from the SHA-256 hash of the title.
  """
  return hashlib.sha256(title.encode('utf-8')).hexdigest()[:12]


@dataclasses.dataclass
class MultiHopRAGDatasetLoader(datasets_lib.DatasetLoader[_ExampleDict]):
  """Loads the MultiHopRAG dataset via TFDS, returning examples and documents.

  MultiHop-RAG is a QA dataset for evaluating retrieval and reasoning across
  documents with metadata in RAG pipelines. It contains 2556 queries, with
  evidence for each query distributed across 2 to 4 documents. The queries
  also involve document metadata, reflecting complex scenarios commonly found
  in real-world RAG applications.

  This loader uses TFDS's HuggingFace bridge to load the `MultiHopRAG` dataset
  (two configs: 'multihoprag' for questions and 'corpus' for documents) and
  provides methods to return evaluation-compatible examples and `Document`
  objects.

  The returned example dictionaries contain:
    - question: The query text.
    - answer: The answer text.
    - metadata: Dictionary with 'question_type', 'evidence_titles'
      (list of evidence document titles), and 'golden_doc_ids'
      (list of short hash IDs matching the corpus document IDs).

  The returned `Document` objects contain:
    - doc_id: A 12-character hex hash derived from the document title.
    - title: The title of the article.
    - content: The body text of the article.
    - metadata: Dictionary with 'author', 'category', 'published_at',
      'source', and 'url'.

  Attributes:
    max_number_of_examples: If set, only the first N examples are loaded. If
      None, all examples are loaded.
    max_number_of_documents: If set, only the first N documents are loaded. If
      None, all documents are loaded.
  """

  max_number_of_examples: int | None = None
  max_number_of_documents: int | None = None

  def _load_questions_tfds_numpy_dataset(self):
    """Loads the MultiHopRAG questions dataset via TFDS.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = 'huggingface:yixuantt__multihoprag/MultiHopRAG'
    ds = tfds.load(tfds_name, split='train')
    return tfds.as_numpy(ds)

  def _load_corpus_tfds_numpy_dataset(self):
    """Loads the MultiHopRAG corpus dataset via TFDS.

    Returns:
      A numpy-converted TFDS dataset iterator.
    """
    tfds_name = 'huggingface:yixuantt__multihoprag/corpus'
    ds = tfds.load(tfds_name, split='train')
    return tfds.as_numpy(ds)

  def _extract_evidence_titles(
      self, evidence_list: dict[str, Any]
  ) -> list[str]:
    """Extracts evidence document titles from TFDS flattened evidence_list.

    TFDS flattens nested sequence features. Depending on the TFDS backend
    version and mirror strategy (Merlion vs Public HuggingFace Bridge),
    dictionary keys may be fully-qualified (e.g.
    'MultiHopRAG/evidence_list/title')
    or bare (e.g. 'title'). Extracting the terminal component via split('/')[-1]
    guarantees seamless execution across both internal and external
    environments.

    Args:
      evidence_list: A dict from a TFDS row containing flattened evidence keys.

    Returns:
      A list of non-empty decoded evidence document title strings.
    """
    title_key = None
    for key in evidence_list:
      if key.split('/')[-1] == 'title':
        title_key = key
        break
    if title_key is None:
      return []

    raw_titles = datasets_lib.decode_list(evidence_list[title_key])
    # Filter out empty strings or non-string artifacts from decoding.
    return [t for t in raw_titles if t and isinstance(t, str)]

  def _build_example_from_row(self, row: dict[str, Any]) -> _ExampleDict:
    """Converts a single MultiHopRAG row into an evaluation Example dict.

    Args:
      row: A single row from the questions dataset (with numpy/bytes values).

    Returns:
      A dictionary suitable for use as an evaluation example.
    """
    query = datasets_lib.decode_value(row.get('query', b''))
    answer = datasets_lib.decode_value(row.get('answer', b''))
    question_type = datasets_lib.decode_value(row.get('question_type', b''))

    # Extract evidence document titles from the flattened TFDS structure.
    evidence_list = row.get('evidence_list', {})
    evidence_titles = self._extract_evidence_titles(evidence_list)

    # Generate golden doc IDs using the same hash function as documents.
    golden_doc_ids = [_generate_doc_id(t) for t in evidence_titles]

    return {
        datasets_lib.EXAMPLE_FIELD_QUESTION: query,
        datasets_lib.EXAMPLE_FIELD_ANSWER: answer,
        datasets_lib.EXAMPLE_FIELD_METADATA: {
            'question_type': question_type,
            'evidence_titles': evidence_titles,
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: golden_doc_ids,
        },
    }

  def _build_document_from_corpus_row(self, row: dict[str, Any]) -> _Document:
    """Converts a single corpus row into a Document object.

    Args:
      row: A single row from the corpus dataset (with numpy/bytes values).

    Returns:
      A Document object.
    """
    title = datasets_lib.decode_value(row.get('title', b''))
    body = datasets_lib.decode_value(row.get('body', b''))
    url = datasets_lib.decode_value(row.get('url', b''))
    author = datasets_lib.decode_value(row.get('author', b''))
    category = datasets_lib.decode_value(row.get('category', b''))
    published_at = datasets_lib.decode_value(row.get('published_at', b''))
    source = datasets_lib.decode_value(row.get('source', b''))

    # Use a short hash of the title as the doc_id for compactness.
    doc_id = _generate_doc_id(title)

    return _Document(
        doc_id=doc_id,
        title=title,
        content=body,
        metadata={
            'author': author,
            'category': category,
            'published_at': published_at,
            'source': source,
            'url': url,
        },
    )

  @executing.make_executable()
  async def load_examples(self) -> Iterable[_ExampleDict]:
    """Loads the examples of the MultiHopRAG dataset.

    Returns:
      An iterable of evaluation example dictionaries.
    """
    ds_numpy = self._load_questions_tfds_numpy_dataset()
    examples: list[_ExampleDict] = []
    for row in ds_numpy:
      if (
          self.max_number_of_examples is not None
          and len(examples) == self.max_number_of_examples
      ):
        break
      examples.append(self._build_example_from_row(row))
    return examples

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads the documents of the MultiHopRAG corpus.

    Returns:
      An iterable of Document objects, deduplicated by doc_id.
    """
    ds_numpy = self._load_corpus_tfds_numpy_dataset()
    documents: list[_Document] = []
    seen_ids: set[str] = set()
    for row in ds_numpy:
      doc = self._build_document_from_corpus_row(row)
      if doc.doc_id in seen_ids:
        continue
      if (
          self.max_number_of_documents is not None
          and len(documents) == self.max_number_of_documents
      ):
        break
      seen_ids.add(doc.doc_id)
      documents.append(doc)
    return documents


# Register dataset loader in the registry.
datasets_lib.register_dataset_loader('multihoprag', MultiHopRAGDatasetLoader)
