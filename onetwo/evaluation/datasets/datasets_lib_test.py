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

"""Tests for datasets_lib module."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
from onetwo import ot
from onetwo.evaluation.datasets import datasets_lib


class JsonlDatasetLoaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def test_invalid_paths(self):
    with self.assertRaises(ValueError):
      datasets_lib.JsonlDatasetLoader(corpus_path='corpus.txt')
    with self.assertRaises(ValueError):
      datasets_lib.JsonlDatasetLoader(examples_path='examples.txt')

  def test_load_examples_json_success(self):
    # Create a fake examples.json
    examples_content = (
        '[{"question": "Q1", "answer": "A1", "extra": "E1"},'
        ' {"question": "Q2", "answer": "A2"}]'
    )
    examples_file = self.create_tempfile(
        file_path='examples.json', content=examples_content
    )

    loader = datasets_lib.JsonlDatasetLoader(
        examples_path=examples_file.full_path
    )
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)
    self.assertEqual(
        examples[0],  # pyrefly: ignore[bad-index]
        {
            datasets_lib.EXAMPLE_FIELD_QUESTION: 'Q1',
            datasets_lib.EXAMPLE_FIELD_ANSWER: 'A1',
            datasets_lib.EXAMPLE_FIELD_METADATA: {'extra': 'E1'},
        },
    )
    self.assertEqual(
        examples[1],  # pyrefly: ignore[bad-index]
        {
            datasets_lib.EXAMPLE_FIELD_QUESTION: 'Q2',
            datasets_lib.EXAMPLE_FIELD_ANSWER: 'A2',
            datasets_lib.EXAMPLE_FIELD_METADATA: {},
        },
    )

  def test_load_examples_jsonl_success(self):
    # Create a fake examples.jsonl
    examples_content = (
        '{"question": "Q1", "answer": "A1", "metadata": {"foo": "bar"}}\n'
        '{"question": "Q2", "answer": "A2"}\n'
    )
    examples_file = self.create_tempfile(
        file_path='examples.jsonl', content=examples_content
    )

    loader = datasets_lib.JsonlDatasetLoader(
        examples_path=examples_file.full_path
    )
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)
    self.assertEqual(
        examples[0],  # pyrefly: ignore[bad-index]
        {
            datasets_lib.EXAMPLE_FIELD_QUESTION: 'Q1',
            datasets_lib.EXAMPLE_FIELD_ANSWER: 'A1',
            datasets_lib.EXAMPLE_FIELD_METADATA: {'foo': 'bar'},
        },
    )

  def test_load_examples_max_number_of_examples(self):
    examples_content = (
        '{"question": "Q1", "answer": "A1"}\n'
        '{"question": "Q2", "answer": "A2"}\n'
        '{"question": "Q3", "answer": "A3"}\n'
    )
    examples_file = self.create_tempfile(
        file_path='examples.jsonl', content=examples_content
    )

    loader = datasets_lib.JsonlDatasetLoader(
        examples_path=examples_file.full_path, max_number_of_examples=2
    )
    examples = ot.run(loader.load_examples())

    self.assertLen(examples, 2)

  def test_load_examples_missing_file(self):
    loader = datasets_lib.JsonlDatasetLoader(examples_path='/nonexistent.jsonl')
    with self.assertRaises(FileNotFoundError):
      ot.run(loader.load_examples())

  def test_load_documents_success(self):
    corpus_content = (
        '{"doc_id": "D1", "title": "Title 1", "content": "Content 1"}\n'
        '{"doc_id": "D2", "content": "Content 2"}\n'
        '{"content": "Content 3"}\n'
    )
    corpus_file = self.create_tempfile(
        file_path='corpus.jsonl', content=corpus_content
    )

    loader = datasets_lib.JsonlDatasetLoader(corpus_path=corpus_file.full_path)
    documents = ot.run(loader.load_documents())

    self.assertLen(documents, 3)
    self.assertEqual(documents[0].doc_id, 'D1')  # pyrefly: ignore[bad-index]
    self.assertEqual(documents[0].title, 'Title 1')  # pyrefly: ignore[bad-index]
    self.assertEqual(documents[1].doc_id, 'D2')  # pyrefly: ignore[bad-index]
    self.assertEqual(documents[1].title, '')  # pyrefly: ignore[bad-index]
    # Check that a deterministic ID was generated for the 3rd doc
    self.assertNotEmpty(documents[2].doc_id)  # pyrefly: ignore[bad-index]
    self.assertEqual(documents[2].content, 'Content 3')  # pyrefly: ignore[bad-index]

  def test_load_documents_missing_file(self):
    loader = datasets_lib.JsonlDatasetLoader(corpus_path='/nonexistent.jsonl')
    with self.assertRaises(FileNotFoundError):
      ot.run(loader.load_documents())

  def test_from_directory(self):
    tempdir = self.create_tempdir()
    examples_file = tempdir.create_file(
        'examples.jsonl', content='{"question": "Q1", "answer": "A1"}'
    )
    corpus_file = tempdir.create_file(
        'corpus.jsonl',
        content='{"doc_id": "D1", "content": "Content 1"}',
    )

    loader = datasets_lib.JsonlDatasetLoader.from_directory(tempdir.full_path)
    self.assertEqual(str(loader.examples_path), examples_file.full_path)
    self.assertEqual(str(loader.corpus_path), corpus_file.full_path)


@dataclasses.dataclass
class _SplitLoader(datasets_lib.DatasetLoader):
  split: str = 'train'
  dataset_name: str | None = None

  @ot.make_executable
  async def load_examples(self):
    return []


class DatasetLoaderRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Save a copy of the registry to restore after each test.
    self._saved_registry = dict(datasets_lib._DATASET_LOADER_REGISTRY)

  def tearDown(self):
    super().tearDown()
    datasets_lib._DATASET_LOADER_REGISTRY.clear()
    datasets_lib._DATASET_LOADER_REGISTRY.update(self._saved_registry)

  def test_register_and_get_dataset_loader(self):
    class DummyLoader(datasets_lib.DatasetLoader):

      def __init__(self, foo: str = 'bar'):
        self.foo = foo

    datasets_lib.register_dataset_loader('dummy', DummyLoader)
    self.assertIn('dummy', datasets_lib.list_dataset_loaders())

    loader = datasets_lib.get_dataset_loader('dummy', foo='baz')
    self.assertIsInstance(loader, DummyLoader)
    self.assertEqual(loader.foo, 'baz')

  def test_register_duplicate_raises(self):
    class DummyLoader(datasets_lib.DatasetLoader):
      pass

    datasets_lib.register_dataset_loader('dummy_dup', DummyLoader)
    with self.assertRaises(ValueError):
      datasets_lib.register_dataset_loader('dummy_dup', DummyLoader)

  def test_unregister_dataset_loader(self):
    class DummyLoader(datasets_lib.DatasetLoader):
      pass

    datasets_lib.register_dataset_loader('dummy_unreg', DummyLoader)
    datasets_lib.unregister_dataset_loader('dummy_unreg')
    self.assertNotIn('dummy_unreg', datasets_lib.list_dataset_loaders())
    with self.assertRaises(KeyError):
      datasets_lib.unregister_dataset_loader('dummy_unreg')

  def test_get_unregistered_raises(self):
    with self.assertRaises(KeyError):
      datasets_lib.get_dataset_loader('nonexistent_loader')

  def test_get_dataset_loader_sets_dataset_name(self):
    datasets_lib.register_dataset_loader('dummy', _SplitLoader)
    for split in ('train', 'test'):
      datasets_lib.register_dataset_loader(
          f'dummy_{split}', functools.partial(_SplitLoader, split=split)
      )

    for name in ('dummy', 'dummy_train', 'dummy_test'):
      with self.subTest(name):
        self.assertEqual(
            datasets_lib.get_dataset_loader(name).dataset_name, name
        )

  def test_get_dataset_loader_keeps_dataset_name_set_by_loader(self):
    datasets_lib.register_dataset_loader(
        'dummy', functools.partial(_SplitLoader, dataset_name='custom_name')
    )

    self.assertEqual(
        datasets_lib.get_dataset_loader('dummy').dataset_name, 'custom_name'
    )

  def test_directly_constructed_loader_has_no_dataset_name(self):
    self.assertIsNone(_SplitLoader().dataset_name)


if __name__ == '__main__':
  absltest.main()
