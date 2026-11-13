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

"""Loader for the SpanishGovernmentMultimodalQA dataset."""

import collections
from collections.abc import Iterable, Mapping
import dataclasses
import enum
import io
from typing import Any, Final

import numpy as np
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures
from PIL import Image as PILImage
import tensorflow_datasets as tfds

_Document = retrieval_data_structures.Document
_ExampleDict = datasets_lib.ExampleDict


@enum.unique
class QuestionLanguage(enum.StrEnum):
  """Language of the question."""

  EN = 'en'
  ES = 'es'


def _is_empty_or_placeholder_image(
    image: PILImage.Image, dominant_color_threshold: float = 0.95
) -> bool:
  """Returns True if the image is empty or dominated by a single pixel color."""
  if not image.getbbox():
    return True
  total_pixels = image.width * image.height
  histogram = image.convert('L').histogram()
  return max(histogram) / total_pixels >= dominant_color_threshold


def _maybe_build_image_chunk(
    image_array: np.ndarray,
) -> content_lib.Chunk | None:
  """Builds a PNG image Chunk, or returns None if empty or a placeholder."""
  pil_image = PILImage.fromarray(image_array)
  if _is_empty_or_placeholder_image(pil_image):
    return None
  buffer = io.BytesIO()
  pil_image.save(buffer, format='PNG')
  return content_lib.Chunk(content=buffer.getvalue(), content_type='image/png')


@dataclasses.dataclass
class SpanishGovernmentMultimodalQADatasetLoader(
    datasets_lib.DatasetLoader[_ExampleDict]
):
  """Loads SpanishGovernmentMultimodalQA dataset via TFDS.

  This loader uses TFDS's HuggingFace bridge to load the
  `megaelius/SpanishGovernmentMultimodalQA` dataset and provides methods to
  return evaluation-compatible examples and `Document` objects.
  The dataset has two configurations: 'questions' for examples, and 'corpus'
  for documents. Each row in 'corpus' generates one document containing an
  optional non-placeholder image chunk followed by a CV text chunk.

  Attributes:
    split: The dataset split to load (default: 'test').
    max_number_of_examples: If set, only the first N examples are loaded by
      `load_examples`. If None, all examples are loaded.
    max_number_of_documents: If set, only the first N documents are loaded by
      `load_documents`. If None, all documents from the corpus are loaded.
    question_language: The language of the question to load (default:
      QuestionLanguage.EN).
  """

  _FIELD_QUESTION_EN: Final[str] = 'question_en'
  _FIELD_QUESTION_ES: Final[str] = 'question_es'
  _FIELD_ANSWER: Final[str] = 'answer'
  _FIELD_FULL_NAME: Final[str] = 'full_name'
  _FIELD_ID_PERSON: Final[str] = 'id_person'
  _FIELD_CV_TEXT: Final[str] = 'cv_text'
  _FIELD_IMAGE: Final[str] = 'image'

  split: str = 'test'
  max_number_of_examples: int | None = None
  max_number_of_documents: int | None = None
  question_language: QuestionLanguage = QuestionLanguage.EN

  def _load_questions_tfds_numpy_dataset(self):
    """Loads questions dataset via TFDS and returns it as a numpy iterator."""
    tfds_name = (
        'huggingface:megaelius__spanishgovernmentmultimodalqa/questions'
    )
    ds = tfds.load(tfds_name, split=self.split)
    return tfds.as_numpy(ds)

  def _load_corpus_tfds_numpy_dataset(self):
    """Loads corpus dataset via TFDS and returns it as a numpy iterator."""
    tfds_name = 'huggingface:megaelius__spanishgovernmentmultimodalqa/corpus'
    ds = tfds.load(tfds_name, split=self.split)
    return tfds.as_numpy(ds)

  def _build_example_from_questions_row(
      self,
      row: dict[str, Any],
      doc_ids_by_name: Mapping[str, list[str]],
  ) -> _ExampleDict:
    """Converts a single questions row into an evaluation Example dict."""
    if self.question_language == QuestionLanguage.EN:
      question_field = self._FIELD_QUESTION_EN
    elif self.question_language == QuestionLanguage.ES:
      question_field = self._FIELD_QUESTION_ES
    else:
      raise ValueError(
          f'Unsupported question language: {self.question_language}'
      )
    question = datasets_lib.decode_value(row[question_field])
    answer = datasets_lib.decode_value(row[self._FIELD_ANSWER])
    return {
        datasets_lib.EXAMPLE_FIELD_QUESTION: question,
        datasets_lib.EXAMPLE_FIELD_ANSWER: answer,
        datasets_lib.EXAMPLE_FIELD_METADATA: {
            datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: (
                doc_ids_by_name.get(answer, [])
            ),
        },
    }

  @executing.make_executable()
  async def load_examples(self) -> Iterable[_ExampleDict]:
    """Loads the examples of the SpanishGovernmentMultimodalQA dataset.

    Returns:
      An iterable of evaluation example dictionaries.
    """
    ds_numpy = self._load_questions_tfds_numpy_dataset()
    doc_ids_by_name: dict[str, list[str]] = collections.defaultdict(list)
    for row in self._load_corpus_tfds_numpy_dataset():
      full_name = datasets_lib.decode_value(row[self._FIELD_FULL_NAME])
      id_person = datasets_lib.decode_value(row[self._FIELD_ID_PERSON])
      doc_ids_by_name[full_name].append(id_person)
    examples: list[_ExampleDict] = []
    for row in ds_numpy:
      if (
          self.max_number_of_examples is not None
          and len(examples) == self.max_number_of_examples
      ):
        break
      examples.append(
          self._build_example_from_questions_row(row, doc_ids_by_name)
      )
    return examples

  def _build_document_from_corpus_row(self, row: dict[str, Any]) -> _Document:
    """Builds a Document object (image and text) from a corpus row."""
    text_chunk = content_lib.Chunk(
        datasets_lib.decode_value(row[self._FIELD_CV_TEXT])
    )
    image_chunk = _maybe_build_image_chunk(row[self._FIELD_IMAGE])
    chunks = [image_chunk, text_chunk] if image_chunk else [text_chunk]
    return _Document(
        doc_id=datasets_lib.decode_value(row[self._FIELD_ID_PERSON]),
        title=datasets_lib.decode_value(row[self._FIELD_FULL_NAME]),
        content=content_lib.ChunkList(chunks),
    )

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads the documents of the SpanishGovernmentMultimodalQA dataset.

    Returns:
      An iterable of Document objects.
    """
    ds_numpy = self._load_corpus_tfds_numpy_dataset()
    documents: list[_Document] = []
    for row in ds_numpy:
      if (
          self.max_number_of_documents is not None
          and len(documents) == self.max_number_of_documents
      ):
        break
      documents.append(self._build_document_from_corpus_row(row))
    return documents


# Register dataset loader in the registry.
datasets_lib.register_dataset_loader(
    'spanish_gov_multimodal_qa', SpanishGovernmentMultimodalQADatasetLoader
)
