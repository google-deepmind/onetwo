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

"""Index building utilities."""

from typing import Any, Callable, Iterable

from onetwo.core import executing

from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures

IndexSpec = index_build_interface.IndexSpec
SearcherConfig = index_build_interface.SearcherConfig
EmbeddingBasedDocumentIndexSpec = (
    index_build_interface.EmbeddingBasedDocumentIndexSpec
)
SimpleRewritingEmbeddingBasedDocumentIndexSpec = (
    index_build_interface.SimpleRewritingEmbeddingBasedDocumentIndexSpec
)
SimpleRewritingHippoRAGDocumentIndexSpec = (
    index_build_interface.SimpleRewritingHippoRAGDocumentIndexSpec
)

IndexRunner = index_build_interface.IndexRunner

_RUNNER_REGISTRY: dict[
    str, Callable[[], index_build_interface.IndexRunner[Any, Any]]
] = {}


def register_runner(
    name: str,
    runner_cls: Callable[[], index_build_interface.IndexRunner[Any, Any]],
) -> None:
  """Registers a new index builder runner."""
  _RUNNER_REGISTRY[name] = runner_cls


@executing.make_executable
async def build_index(
    spec: index_build_interface.IndexSpec,
    docs: Iterable[retrieval_data_structures.Document] | str,
    output_dir: str | None = None,
    runner: str = "local",
    **kwargs,
) -> Any:
  """Builds an index using the specified runner.

  Args:
    spec: Index specification.
    docs: Documents to index.
    output_dir: Optional directory to save the built index.
    runner: The runner to use (e.g., "local").
    **kwargs: Runner-specific arguments.

  Returns:
    The result of the index building (depends on the runner).
  """
  if runner not in _RUNNER_REGISTRY:
    raise ValueError(
        f"Unknown runner: {runner}. Registered runners: "
        f"{list(_RUNNER_REGISTRY.keys())}"
    )

  runner_impl = _RUNNER_REGISTRY[runner]()
  return await runner_impl.build_index(
      spec=spec,
      docs=docs,
      output_dir=output_dir,
      **kwargs,
  )


register_runner("local", local_index_runner.LocalIndexRunner)
# TODO: Register the "beam" runner once implemented.
