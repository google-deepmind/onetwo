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

"""In-memory index runner implementation."""

from collections.abc import Iterable, Mapping
from typing import Any

from onetwo.core import executing
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization as ot_serialization

from onetwo.stdlib.retrieval import serialization


class LocalIndexRunner(
    index_build_interface.IndexRunner[
        indexing.DocumentIndex,
        Iterable[retrieval_data_structures.Document] | str,
    ]
):
  """In-memory index runner."""

  @executing.make_executable(copy_self=False)
  async def build_index(
      self,
      index: indexing.DocumentIndex,
      docs: Iterable[retrieval_data_structures.Document] | str,
      output_dir: str | None = None,
      runtime_kwargs: Mapping[str, Any] | None = None,
      *,
      storage_format: ot_serialization.StorageFormat = (
          ot_serialization.StorageFormat.JSONL_NPY
      ),
      **kwargs,
  ) -> indexing.DocumentIndex:
    """Builds `index` in memory, optionally saves it, and returns it.

    Overridden from base class (IndexRunner). With `runtime_kwargs`, a copy of
    `index` built from its config with them is built instead (`index` is not
    modified).
    """
    if isinstance(docs, str):
      docs = serialization.load_documents(
          docs, doc_class=retrieval_data_structures.Document
      )
    # Check before building, so that an index that can't be saved fails fast
    # instead of after a possibly expensive build.
    if output_dir and not indexing.supports_save(index):
      raise TypeError(
          f'{type(index).__name__} does not implement save(base_path) and'
          ' load(base_path), so it cannot be written to'
          f' output_dir={output_dir!r}.'
      )
    if runtime_kwargs is not None:
      index = config_serialization.from_config(
          config_serialization.get_config(index), runtime_kwargs
      )
    await index.add_docs(docs)

    if output_dir:
      index.save(output_dir, storage_format)

    return index
