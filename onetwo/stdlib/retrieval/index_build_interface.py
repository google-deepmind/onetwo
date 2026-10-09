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

"""Interface and specifications for index building."""

import abc
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any, Generic, TypeVar

import dataclasses_json
from onetwo.core import executing
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import searchers


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class SearcherConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for a Searcher.

  Attributes:
    class_name: Name of the searcher class (e.g. 'BruteForceSearcher').
    kwargs: Keyword arguments to pass to the searcher constructor.
  """

  class_name: str
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_searcher(config: SearcherConfig) -> searchers.Searcher:
  """Instantiates a Searcher from its configuration."""
  cls = getattr(searchers, config.class_name, None)
  if cls is None or not issubclass(cls, searchers.Searcher):
    raise ValueError(f"Unknown searcher: {config.class_name}")
  return cls(**config.kwargs)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class IndexSpec(dataclasses_json.DataClassJsonMixin, abc.ABC):
  """Base class for index specifications."""

  def make_index(
      self, runtime_kwargs: Mapping[str, Any]
  ) -> indexing.DocumentIndex:
    """Returns an empty index configured by this spec.

    Every buildable spec overrides this and is registered with
    `index_spec_registry.register_index_spec`.

    Args:
      runtime_kwargs: Environment-specific settings and credentials for the
        index's backends (e.g. API key). Not part of the spec.

    Raises:
      NotImplementedError: If the spec does not override this method.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement make_index."
    )

  def get_stable_hash(self) -> str:
    """Returns a stable 8-character hash of the specification."""
    json_dict = json.loads(self.to_json())
    stable_json = json.dumps(json_dict, sort_keys=True)
    return hashlib.md5(stable_json.encode("utf-8")).hexdigest()[:8]

  def __str__(self) -> str:
    """Returns a compact, path-friendly string (used in index output paths)."""
    return f"{type(self).__name__}__hash_{self.get_stable_hash()}"


SpecT = TypeVar("SpecT", bound=IndexSpec)
DocsT = TypeVar("DocsT")


class IndexRunner(abc.ABC, Generic[SpecT, DocsT]):
  """Base class for index building runners."""

  @executing.make_executable(copy_self=False)
  @abc.abstractmethod
  async def build_index(
      self,
      spec: SpecT,
      docs: DocsT,
      output_dir: str | None = None,
      **kwargs,
  ) -> Any:
    """Builds the index.

    Args:
      spec: Index specification.
      docs: Documents to index (e.g., Iterable[Document] or path).
      output_dir: Optional directory to save the built index.
      **kwargs: Runner-specific arguments.

    Returns:
      The result of the index building (depends on the runner).
    """
