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
import dataclasses
import hashlib
import json
from typing import Any, Generic, TypeVar

import dataclasses_json
from onetwo.backends import backend_registry
from onetwo.core import executing
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

  def get_stable_hash(self) -> str:
    """Returns a stable 8-character hash of the specification."""
    json_dict = json.loads(self.to_json())
    stable_json = json.dumps(json_dict, sort_keys=True)
    return hashlib.md5(stable_json.encode("utf-8")).hexdigest()[:8]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EmbeddingBasedDocumentIndexSpec(IndexSpec):
  """Specification for building an EmbeddingBasedDocumentIndex.

  Attributes:
    embed_backend_config: Configuration for the embedder backend.
    query_task_type: Task type to use when embedding queries.
    document_task_type: Task type to use when embedding documents.
    runtime_kwargs: Environment-specific deployment settings and credentials.
    searcher_config: Configuration for the searcher to be used by the index for
      retrieving documents (e.g., BruteForceSearcher).
  """

  embed_backend_config: backend_registry.EmbedBackendConfig
  query_task_type: str | None = "RETRIEVAL_QUERY"
  document_task_type: str | None = "RETRIEVAL_DOCUMENT"
  searcher_config: SearcherConfig = dataclasses.field(
      default_factory=lambda: SearcherConfig(class_name="BruteForceSearcher")
  )
  runtime_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata=dataclasses_json.config(exclude=lambda _: True),
  )

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    model_name = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return f"EmbeddingBasedDocumentIndex_{model_name}__hash_{spec_hash}"


# TODO: Standardize the spec-to-rewriter mapping and make it
# generic.
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SimpleRewritingEmbeddingBasedDocumentIndexSpec(
    EmbeddingBasedDocumentIndexSpec
):
  """Specification for building a simple RewritingEmbeddingBasedDocumentIndex.

  This spec is a convenience wrapper for users that want to use a combination of
  chunking and formatting for corpus rewriting.

  Attributes:
    max_tokens_per_chunk: Maximum tokens per chunk for chunking.
    format_str: Format string for document formatting.
    overlap_window: Number of tokens to overlap between chunks.
  """

  max_tokens_per_chunk: int | None = None
  format_str: str | None = "{text}"
  overlap_window: int = 0

  def __str__(self) -> str:
    """Returns a compact, path-friendly string representation."""
    model_name = (
        self.embed_backend_config.model_name
        or self.embed_backend_config.backend_class_name
    )
    spec_hash = self.get_stable_hash()
    return (
        "EmbeddingBasedDocumentIndex"
        f"_chunk_size_{self.max_tokens_per_chunk}"
        f"_overlap_window_{self.overlap_window}"
        f"_{model_name}"
        f"__hash_{spec_hash}"
    )


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
