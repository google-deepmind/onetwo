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

"""A backend-free custom index and spec, for testing custom index support.

`KeywordDocumentIndex` ranks documents by the number of query words that occur
in the document content. It needs no model backends, which makes it suitable
for end-to-end tests of the custom index path (`index_spec_registry`,
`index_building`, `rag.build_index` and `rag.evaluate`).
"""

from collections.abc import Iterable, Iterator, Mapping
import contextlib
import dataclasses
import json
import re
from typing import Any
from unittest import mock

import dataclasses_json
from etils import epath
from onetwo.core import executing
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document

_DOCS_FILENAME = 'keyword_docs.jsonl'
_CONFIG_FILENAME = 'keyword_index.json'


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class KeywordDocumentIndexSpec(index_build_interface.IndexSpec):
  """Specification for building a `KeywordDocumentIndex`.

  Attributes:
    lowercase: Whether to lowercase the query and documents before matching.
  """

  lowercase: bool = True


@dataclasses.dataclass(kw_only=True)
class KeywordDocumentIndex(indexing.DocumentIndex):
  """Ranks documents by the number of query words they contain.

  Attributes:
    lowercase: Whether to lowercase the query and documents before matching.
  """

  lowercase: bool = True
  _docs: list[Document] = dataclasses.field(default_factory=list, repr=False)

  def _words(self, text: str) -> set[str]:
    if self.lowercase:
      text = text.lower()
    return set(re.findall(r'\w+', text))

  @property
  def num_docs(self) -> int:
    return len(self._docs)

  def get_docs(self) -> Iterable[Document]:
    return self._docs

  @executing.make_executable(copy_self=False)
  async def add_docs(self, docs: Iterable[Document]) -> None:
    self._docs.extend(docs)

  def destroy_index(self) -> None:
    self._docs.clear()

  @executing.make_executable(copy_self=False)
  async def retrieve(
      self,
      query: str,
      *,
      max_results: int | None = None,
      **kwargs,
  ) -> Iterable[Document]:
    del kwargs
    query_words = self._words(query)
    scored = [
        (len(query_words & self._words(str(doc.content))), i, doc)
        for i, doc in enumerate(self._docs)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [doc for score, _, doc in scored if score > 0]
    return ranked if max_results is None else ranked[:max_results]

  def save(self, base_path: str) -> None:
    path = epath.Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / _CONFIG_FILENAME).write_text(
        json.dumps({'lowercase': self.lowercase})
    )
    (path / _DOCS_FILENAME).write_text(
        ''.join(doc.to_json() + '\n' for doc in self._docs)
    )

  @executing.make_executable(copy_self=False)
  async def load(self, base_path: str) -> None:
    path = epath.Path(base_path)
    config = json.loads((path / _CONFIG_FILENAME).read_text())
    self.lowercase = config['lowercase']
    self._docs = [
        Document.from_json(line)
        for line in (path / _DOCS_FILENAME).read_text().splitlines()
        if line.strip()
    ]


def make_keyword_index(
    spec: KeywordDocumentIndexSpec, runtime_kwargs: Mapping[str, Any]
) -> KeywordDocumentIndex:
  """Index factory for `KeywordDocumentIndexSpec`."""
  del runtime_kwargs  # The keyword index has no backends.
  return KeywordDocumentIndex(lowercase=spec.lowercase)


@contextlib.contextmanager
def registered_keyword_index_spec(name: str | None = None) -> Iterator[None]:
  """Registers `KeywordDocumentIndexSpec` for the duration of the context.

  The registry is restored on exit, including any other registrations made
  inside the context, so that tests don't leak registrations.

  Args:
    name: Name to register the spec under. Defaults to the class name.

  Yields:
    None.
  """
  # pylint: disable=protected-access
  with mock.patch.dict(
      index_spec_registry._REGISTRATIONS_BY_NAME
  ), mock.patch.dict(index_spec_registry._REGISTRATIONS_BY_TYPE):
    # pylint: enable=protected-access
    index_spec_registry.register_index_spec(
        KeywordDocumentIndexSpec, make_keyword_index, name=name
    )
    yield
