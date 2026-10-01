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

import getpass
from typing import Any, Callable, Iterable, Mapping

from etils import epath
from onetwo.core import executing

from onetwo.stdlib.retrieval import builtin_index_specs  # pylint: disable=unused-import
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures

# `builtin_index_specs` is imported above so that the built-in specs are
# registered in `index_spec_registry` for every user of this module.

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
    runtime_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> Any:
  """Builds an index using the specified runner.

  Args:
    spec: Index specification.
    docs: Documents to index.
    output_dir: Optional directory to save the built index.
    runner: The runner to use (e.g., "local").
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key). Not saved with the index.
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
      runtime_kwargs=runtime_kwargs,
      **kwargs,
  )


register_runner("local", local_index_runner.LocalIndexRunner)
# TODO: Register the "beam" runner once implemented.


# TODO: Determine output directory for OSS.
def resolve_index_output_directory(
    output_directory_pattern: str,
    dataset_name: str,
    spec: index_build_interface.IndexSpec,
) -> str:
  """Resolves the output directory for an index, handling internal paths.

  Args:
    output_directory_pattern: Pattern for the output directory (may contain
      {username}).
    dataset_name: Name of the dataset.
    spec: The index spec to use for sub-directory naming.

  Returns:
    The resolved path as a string.
  """
  path = output_directory_pattern
  return path
