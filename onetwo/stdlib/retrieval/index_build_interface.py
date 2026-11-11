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

"""Interface of index building runners."""

import abc
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import serialization


IndexT = TypeVar("IndexT", bound=indexing.DocumentIndex)
DocsT = TypeVar("DocsT")


class IndexRunner(abc.ABC, Generic[IndexT, DocsT]):
  """Base class for index building runners."""

  @executing.make_executable(copy_self=False)
  @abc.abstractmethod
  async def build_index(
      self,
      index: IndexT,
      docs: DocsT,
      output_dir: str | None = None,
      runtime_kwargs: Mapping[str, Any] | None = None,
      *,
      storage_format: serialization.StorageFormat = (
          serialization.StorageFormat.JSONL_NPY
      ),
      **kwargs,
  ) -> Any:
    """Builds the index.

    Args:
      index: The (empty) index to build.
      docs: Documents to index (e.g., Iterable[Document] or path).
      output_dir: Optional directory to save the built index.
      runtime_kwargs: Environment-specific settings and credentials for the
        index's backends (e.g. API key). Not saved with the index.
      storage_format: How the index is saved to `output_dir`.
      **kwargs: Runner-specific arguments.

    Returns:
      The result of the index building (depends on the runner).
    """
