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
import json
from typing import Any, Callable, Iterable, Mapping

from etils import epath
from onetwo.core import executing
from onetwo.stdlib.retrieval import config_serialization

from onetwo.stdlib.retrieval import embedding_index_specs
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import index_spec_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization

from onetwo.stdlib.retrieval.graphrag import hipporag_index_specs  # pylint: disable=unused-import

# `embedding_index_specs` (and `hipporag_index_specs`) are imported above so
# that the built-in specs are registered in `index_spec_registry` for every user
# of this module.

IndexSpec = index_build_interface.IndexSpec
SearcherConfig = index_build_interface.SearcherConfig
EmbeddingBasedDocumentIndexSpec = (
    embedding_index_specs.EmbeddingBasedDocumentIndexSpec
)
SimpleRewritingEmbeddingBasedDocumentIndexSpec = (
    embedding_index_specs.SimpleRewritingEmbeddingBasedDocumentIndexSpec
)

IndexRunner = index_build_interface.IndexRunner

# Name of the file, written next to a saved index, that describes its spec.
INDEX_SPEC_FILENAME = "index_spec.json"

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
    index: indexing.DocumentIndex,
    docs: Iterable[retrieval_data_structures.Document] | str,
    output_dir: str | None = None,
    runner: str = "local",
    runtime_kwargs: Mapping[str, Any] | None = None,
    storage_format: serialization.StorageFormat = (
        serialization.StorageFormat.JSONL_NPY
    ),
    **kwargs,
) -> Any:
  """Builds `index` using the specified runner.

  If `output_dir` is set, the index is saved there with its config
  (`index_config.json`, see `config_serialization`), so it can be loaded with
  `load_index`.

  Args:
    index: The (empty) index to build. The classes of the index and its
      components must be registered (see `registry`) to save it.
    docs: Documents to index.
    output_dir: Optional directory to save the built index.
    runner: The runner to use (e.g., "local").
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key). Not saved with the index.
    storage_format: How the index is saved to `output_dir`.
    **kwargs: Runner-specific arguments.

  Returns:
    The result of the index building (depends on the runner).
  """
  if runner not in _RUNNER_REGISTRY:
    raise ValueError(
        f"Unknown runner: {runner}. Registered runners: "
        f"{list(_RUNNER_REGISTRY.keys())}"
    )

  if output_dir:
    # Fail fast, before building, if the index cannot be saved.
    config_serialization.get_config(index)
  runner_impl = _RUNNER_REGISTRY[runner]()
  result = await runner_impl.build_index(
      index=index,
      docs=docs,
      output_dir=output_dir,
      runtime_kwargs=runtime_kwargs,
      storage_format=storage_format,
      **kwargs,
  )
  if output_dir:
    config_serialization.save_index_config(index, output_dir)
  return result


def save_index_spec(
    spec: index_build_interface.IndexSpec, index_directory: str
) -> None:
  """Writes `spec` to `index_directory/index_spec.json`.

  Args:
    spec: A spec whose exact type is registered in `index_spec_registry`.
    index_directory: Directory of the saved index.
  """
  path = epath.Path(index_directory)
  path.mkdir(parents=True, exist_ok=True)
  (path / INDEX_SPEC_FILENAME).write_text(
      json.dumps(index_spec_registry.spec_to_dict(spec), indent=2)
  )


def read_index_spec(
    index_directory: str,
) -> index_build_interface.IndexSpec | None:
  """Reads the spec saved by `save_index_spec`.

  Args:
    index_directory: Directory of the saved index.

  Returns:
    The saved spec, or None if `index_directory` has no `index_spec.json`
    (e.g. indexes built before specs were saved).

  Raises:
    KeyError: If the saved spec type is not registered (e.g. the module that
      registers a custom spec has not been imported).
  """
  spec_path = epath.Path(index_directory) / INDEX_SPEC_FILENAME
  if not spec_path.exists():
    return None
  return index_spec_registry.spec_from_dict(json.loads(spec_path.read_text()))


def load_index(
    index_directory: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> indexing.DocumentIndex:
  """Loads a saved index, whatever its type.

  The index is built from the `index_config.json` written next to it if any
  (see `config_serialization`), else from its `index_spec.json`. Directories
  without either file (built before specs were saved) are loaded as an
  `EmbeddingBasedDocumentIndex`.

  Args:
    index_directory: Directory of the saved index.
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key).

  Returns:
    The loaded index.

  Raises:
    KeyError: If the saved spec type or a class in the saved config is not
      registered (e.g. the module that registers it has not been imported).
    ValueError: If the saved config has an unsupported `format_version`.
    TypeError: If the saved config does not describe a `DocumentIndex`, or the
      index does not implement `load`.
  """
  runtime_kwargs = dict(runtime_kwargs or {})
  spec = None
  if (
      epath.Path(index_directory) / config_serialization.INDEX_CONFIG_FILENAME
  ).exists():
    index = config_serialization.make_index_from_config_file(
        index_directory, runtime_kwargs
    )
  elif (spec := read_index_spec(index_directory)) is None:
    index = indexing.EmbeddingBasedDocumentIndex(runtime_kwargs=runtime_kwargs)
  else:
    index = index_spec_registry.make_index(spec, runtime_kwargs)
  if not indexing.supports_save(index):
    raise TypeError(
        f"{type(index).__name__}"
        + (f" (built for {type(spec).__name__})" if spec else "")
        + f" does not implement load(base_path), so {index_directory!r}"
        " cannot be loaded."
    )
  executing.run(index.load(index_directory))
  return index


register_runner("local", local_index_runner.LocalIndexRunner)
# TODO: Register the "beam" runner once implemented.


# TODO: Determine output directory for OSS.
def resolve_index_output_directory(
    output_directory_pattern: str,
    dataset_name: str,
    subdirectory: str,
) -> str:
  """Resolves the output directory for an index, handling internal paths.

  Args:
    output_directory_pattern: Pattern for the output directory (may contain
      {username}).
    dataset_name: Name of the dataset.
    subdirectory: Name of the subdirectory of the index, e.g. `str(spec)`.

  Returns:
    The resolved path as a string.
  """
  path = output_directory_pattern
  return path
