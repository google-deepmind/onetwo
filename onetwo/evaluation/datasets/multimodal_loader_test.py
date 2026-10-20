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

"""Tests for multimodal_loader module."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.core import content as content_lib
from onetwo.evaluation.datasets import datasets_lib
from onetwo.evaluation.datasets import multimodal_loader
from onetwo.stdlib.retrieval import retrieval_data_structures

# The loader does not parse binary files, so placeholder bytes are enough.
_FILES = {
    'a.txt': 'Plain text.',
    'b.pdf': b'%PDF-1.4 fake pdf bytes',
    'c.png': b'\x89PNG fake png bytes',
    'sub/d.mp3': b'ID3 fake mp3 bytes',
    'sub/e.mp4': b'fake mp4 bytes',
    'sub/deeper/f.csv': 'x,y\n1,2\n',
}
_UNSUPPORTED_FILES = {
    'unknown.xyz': b'unknown',
    'sub/data.json': '{"not": "loaded"}',
}


def _get_single_chunk(
    document: retrieval_data_structures.Document,
) -> content_lib.Chunk:
  """Returns the single chunk of a document's `ChunkList` content."""
  assert isinstance(document.content, content_lib.ChunkList)
  assert len(document.content) == 1
  return document.content.chunks[0]


class MultimodalDatasetLoaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.corpus_dir = self.create_tempdir().full_path
    for relative_path, content in (_FILES | _UNSUPPORTED_FILES).items():
      path = os.path.join(self.corpus_dir, relative_path)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      mode = 'wb' if isinstance(content, bytes) else 'w'
      with open(path, mode) as f:
        f.write(content)

  def test_load_documents(self):
    loader = multimodal_loader.MultimodalDatasetLoader(path=self.corpus_dir)

    documents = list(ot.run(loader.load_documents()))

    expected_content_types = {
        'a.txt': 'text/plain',
        'b.pdf': 'application/pdf',
        'c.png': 'image/png',
        'sub/d.mp3': 'audio/mpeg',
        'sub/e.mp4': 'video/mp4',
        'sub/deeper/f.csv': 'text/csv',
    }
    with self.subTest('loads_supported_files_sorted_by_path'):
      self.assertEqual(
          [doc.doc_id for doc in documents], sorted(expected_content_types)
      )
    for doc in documents:
      with self.subTest(f'{doc.doc_id}_has_correct_attributes'):
        expected_title = os.path.splitext(os.path.basename(doc.doc_id))[0]
        self.assertEqual(doc.title, expected_title)
        self.assertEqual(doc.url, os.path.join(self.corpus_dir, doc.doc_id))
        self.assertEqual(
            doc.content,
            content_lib.ChunkList([
                content_lib.Chunk(
                    _FILES[doc.doc_id],
                    content_type=expected_content_types[doc.doc_id],
                )
            ]),
        )

  def test_load_documents_text_is_str_and_binary_is_bytes(self):
    loader = multimodal_loader.MultimodalDatasetLoader(path=self.corpus_dir)

    documents = {doc.doc_id: doc for doc in ot.run(loader.load_documents())}

    self.assertIsInstance(_get_single_chunk(documents['a.txt']).content, str)
    self.assertEqual(documents['a.txt'].text, 'Plain text.')
    self.assertIsInstance(_get_single_chunk(documents['b.pdf']).content, bytes)
    self.assertEqual(documents['b.pdf'].text, '<application/pdf>')

  def test_load_documents_non_recursive(self):
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=self.corpus_dir, recursive=False
    )

    documents = ot.run(loader.load_documents())

    self.assertEqual(
        [doc.doc_id for doc in documents], ['a.txt', 'b.pdf', 'c.png']
    )

  @parameterized.named_parameters(
      ('zero', 0, []),
      ('two', 2, ['a.txt', 'b.pdf']),
      ('more_than_available', 100, sorted(_FILES)),
  )
  def test_load_documents_max_number_of_documents(
      self, max_number_of_documents, expected_doc_ids
  ):
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=self.corpus_dir,
        max_number_of_documents=max_number_of_documents,
    )

    documents = ot.run(loader.load_documents())

    self.assertEqual([doc.doc_id for doc in documents], expected_doc_ids)

  def test_load_documents_unsupported_file_raises_if_not_skipped(self):
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=self.corpus_dir, skip_unsupported_files=False
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported content type'):
      ot.run(loader.load_documents())

  def test_load_documents_missing_corpus_dir_raises(self):
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=os.path.join(self.corpus_dir, 'does_not_exist')
    )
    with self.assertRaises(FileNotFoundError):
      ot.run(loader.load_documents())

  @parameterized.named_parameters(
      (
          'jsonl',
          'examples.jsonl',
          (
              '{"question": "Q1", "answer": "A1", "golden_doc_ids":'
              ' ["b.pdf"]}\n{"question": "Q2", "answer": "A2"}\n'
          ),
      ),
      (
          'json',
          'examples.json',
          (
              '[{"question": "Q1", "answer": "A1", "golden_doc_ids":'
              ' ["b.pdf"]}, {"question": "Q2", "answer": "A2"}]'
          ),
      ),
  )
  def test_load_examples(self, file_name, content):
    examples_file = self.create_tempfile(file_path=file_name, content=content)
    loader = multimodal_loader.MultimodalDatasetLoader(
        path=self.corpus_dir, examples_path=examples_file.full_path
    )

    examples = ot.run(loader.load_examples())

    self.assertEqual(
        examples,
        [
            {
                datasets_lib.EXAMPLE_FIELD_QUESTION: 'Q1',
                datasets_lib.EXAMPLE_FIELD_ANSWER: 'A1',
                datasets_lib.EXAMPLE_FIELD_METADATA: {
                    datasets_lib.EXAMPLE_METADATA_FIELD_GOLDEN_DOC_IDS: [
                        'b.pdf'
                    ]
                },
            },
            {
                datasets_lib.EXAMPLE_FIELD_QUESTION: 'Q2',
                datasets_lib.EXAMPLE_FIELD_ANSWER: 'A2',
                datasets_lib.EXAMPLE_FIELD_METADATA: {},
            },
        ],
    )

  def test_load_examples_without_examples_path_raises(self):
    loader = multimodal_loader.MultimodalDatasetLoader(path=self.corpus_dir)
    with self.assertRaisesRegex(ValueError, 'examples_path is not configured'):
      ot.run(loader.load_examples())

  def test_registered_in_dataset_registry(self):
    loader = datasets_lib.get_dataset_loader(
        'multimodal_loader', path=self.corpus_dir, max_number_of_documents=1
    )

    self.assertIsInstance(loader, multimodal_loader.MultimodalDatasetLoader)
    self.assertEqual(
        [doc.doc_id for doc in ot.run(loader.load_documents())], ['a.txt']
    )


if __name__ == '__main__':
  absltest.main()
