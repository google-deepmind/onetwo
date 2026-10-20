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

"""Dataset loader for a multimodal corpus stored as a folder of files.

Each file in the folder becomes one `Document` whose content is a
`content_lib.ChunkList` with a single chunk, typed according to the file
extension (e.g. `text/plain` as `str`, `application/pdf` or `image/png` as
`bytes`). QA examples can optionally be loaded from a JSON or JSONL file.

Usage example:

  ```python
  from onetwo.evaluation import datasets

  loader = datasets.get_dataset_loader(
      'multimodal_loader',
      path='/path/to/corpus',
      examples_path='/path/to/examples.jsonl',
  )
  documents = ot.run(loader.load_documents())
  examples = ot.run(loader.load_examples())
  ```
"""

from collections.abc import Iterable
import dataclasses
import logging
import mimetypes

from etils import epath
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.evaluation.datasets import datasets_lib
from onetwo.stdlib.retrieval import retrieval_data_structures

_Document = retrieval_data_structures.Document

# Name under which `MultimodalDatasetLoader` is registered in the dataset
# loader registry.
DATASET_NAME = 'multimodal_loader'

# Content type prefixes of files that are loaded as `bytes`. Files with a
# `text/` content type are loaded as `str`; all other files are unsupported.
# These are a subset of the content types that `content_lib.Chunk` accepts for
# `bytes` content, see `_CONTENT_TYPE_PREFIXES_BY_PYTHON_TYPE` in
# onetwo/core/content.py.
_BYTES_CONTENT_TYPE_PREFIXES = ('application/pdf', 'image/', 'audio/', 'video/')

# Uses only Python's built-in extension -> content type mapping (not the
# system's mime.types files), so that results do not depend on the machine.
_MIME_TYPES = mimetypes.MimeTypes()


def _list_files(directory: epath.Path, recursive: bool) -> list[epath.Path]:
  """Returns the files in `directory`, sorted by path for determinism."""
  if recursive:
    files = []
    for root, _, names in directory.walk():
      files.extend(root / name for name in names)
  else:
    files = [path for path in directory.iterdir() if not path.is_dir()]
  return sorted(files)


def get_content_type(path: str | epath.Path) -> str | None:
  """Returns the content type of the file at `path`, based on its extension.

  Args:
    path: Path of the file.

  Returns:
    The content type (e.g. 'text/plain', 'application/pdf', 'image/png'), or
    None if the extension is unknown.
  """
  content_type, _ = _MIME_TYPES.guess_type(str(path), strict=False)
  return content_type


@dataclasses.dataclass(kw_only=True)
class MultimodalDatasetLoader(
    datasets_lib.DatasetLoader[datasets_lib.ExampleDict]
):
  """Loads a multimodal corpus of `Document`s from a folder of files.

  Each supported file becomes one `Document` with:
  * `doc_id`: the path of the file relative to `path` (e.g.
    'pdfs/report.pdf'), which is stable and can be referred to from the
    `golden_doc_ids` of the examples.
  * `title`: the file name without its extension.
  * `url`: the full path of the file.
  * `content`: a `ChunkList` with a single chunk. Files with a `text/*` content
    type are decoded as UTF-8 `str`; PDF, image, audio and video files are
    loaded as `bytes`. The chunk's `content_type` is the file's content type.

  Attributes:
    path: Directory containing the files of the corpus.
    examples_path: Optional path to a JSON or JSONL file of QA examples (see
      `datasets_lib.JsonlDatasetLoader` for the expected format).
    recursive: Whether to also load files in subdirectories of `path`.
    skip_unsupported_files: If True, files whose content type is unknown or
      unsupported are skipped with a warning. If False, they raise an error.
    max_number_of_documents: If set, only loads the first
      `max_number_of_documents` documents.
    max_number_of_examples: If set, only loads the first
      `max_number_of_examples` examples.
  """

  path: str | epath.Path
  examples_path: str | epath.Path | None = None
  recursive: bool = True
  skip_unsupported_files: bool = True
  max_number_of_documents: int | None = None
  max_number_of_examples: int | None = None

  def _load_document(
      self, root: epath.Path, file_path: epath.Path
  ) -> _Document | None:
    """Returns the `Document` for `file_path`, or None if it is skipped."""
    content_type = get_content_type(file_path)
    if content_type is not None and content_type.startswith('text/'):
      content = file_path.read_bytes().decode('utf-8', errors='replace')
    elif content_type is not None and content_type.startswith(
        _BYTES_CONTENT_TYPE_PREFIXES
    ):
      content = file_path.read_bytes()
    else:
      message = (
          f'Unsupported content type {content_type!r} for file {file_path}.'
      )
      if not self.skip_unsupported_files:
        raise ValueError(message)
      logging.warning('%s Skipping it.', message)
      return None
    return _Document(
        doc_id=str(file_path.relative_to(root)),
        title=file_path.stem,
        url=str(file_path),
        content=content_lib.ChunkList(
            [content_lib.Chunk(content, content_type=content_type)]
        ),
    )

  @executing.make_executable()
  async def load_documents(self) -> Iterable[_Document]:
    """Loads one document per supported file in `path`.

    Returns:
      A list of `Document`s, sorted by file path.

    Raises:
      ValueError: If a file is unsupported and `skip_unsupported_files` is
        False.
      FileNotFoundError: If `path` is not an existing directory.
    """
    root = epath.Path(self.path)
    if not root.is_dir():
      raise FileNotFoundError(f'Corpus directory does not exist: {root}')

    documents = []
    for file_path in _list_files(root, self.recursive):
      if (
          self.max_number_of_documents is not None
          and len(documents) >= self.max_number_of_documents
      ):
        break
      document = self._load_document(root, file_path)
      if document is not None:
        documents.append(document)
    return documents

  @executing.make_executable()
  async def load_examples(self) -> Iterable[datasets_lib.ExampleDict]:
    """Loads the examples from `examples_path` (JSON or JSONL).

    Returns:
      A list of evaluation example dictionaries.

    Raises:
      ValueError: If `examples_path` is not configured or has an unsupported
        suffix.
      FileNotFoundError: If `examples_path` does not exist.
    """
    if self.examples_path is None:
      raise ValueError('examples_path is not configured.')
    return datasets_lib.load_examples_from_file(
        self.examples_path, self.max_number_of_examples
    )


# Register dataset loader in the registry.
datasets_lib.register_dataset_loader(DATASET_NAME, MultimodalDatasetLoader)
