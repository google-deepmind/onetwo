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
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from etils import epath
from onetwo.core import executing
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import local_index_runner
from onetwo.stdlib.retrieval import retrieval_data_structures
# Imported so that the classes of the built-in indexes and chunkers are
# registered (see `registry`) for every user of `load_index`.
from onetwo.stdlib.retrieval import routing_chunking  # pylint: disable=unused-import
from onetwo.stdlib.retrieval import serialization
from onetwo.stdlib.retrieval.graphrag import indexing as graphrag_indexing  # pylint: disable=unused-import

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


def load_index(
    index_directory: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> indexing.DocumentIndex:
  """Loads a saved index, whatever its type.

  The index is built from the `index_config.json` written next to it if any
  (see `config_serialization`). Directories without it (e.g. embedding indexes
  built by other tools) are loaded as an `EmbeddingBasedDocumentIndex`.

  Args:
    index_directory: Directory of the saved index.
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key).

  Returns:
    The loaded index.

  Raises:
    KeyError: If a class in the saved config is not registered (e.g. the module
    that registers it has not been imported).
    ValueError: If the saved config has an unsupported `format_version`.
    TypeError: If the saved config does not describe a `DocumentIndex`, or the
      index does not implement `load`.
  """
  runtime_kwargs = dict(runtime_kwargs or {})
  if (
      epath.Path(index_directory) / config_serialization.INDEX_CONFIG_FILENAME
  ).exists():
    index = config_serialization.make_index_from_config_file(
        index_directory, runtime_kwargs
    )
  else:
    index = indexing.EmbeddingBasedDocumentIndex(runtime_kwargs=runtime_kwargs)
  if not indexing.supports_save(index):
    raise TypeError(
        f"{type(index).__name__} does not implement load(base_path), so"
        f" {index_directory!r} cannot be loaded."
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
    subdirectory: Name of the subdirectory of the index.

  Returns:
    The resolved path as a string.
  """
  base_path = epath.Path(
      output_directory_pattern.format(username=getpass.getuser())
  )
  return str(base_path.expanduser() / dataset_name / subdirectory)


def _find(config: Any, key: str) -> Any:
  """Returns the first value of `key` in `config`, depth-first, else None."""
  if isinstance(config, Mapping):
    if key in config:
      return config[key]
    values = list(config.values())
  elif isinstance(config, list):
    values = config
  else:
    return None
  for value in values:
    if (found := _find(value, key)) is not None:
      return found
  return None


def index_dir_name(index: indexing.DocumentIndex) -> str:
  """Returns the name of the directory where `index` is saved when built.

  The name is made of the index class, its (first) chunker with its scalar
  (str, int, float, bool) settings (`none` if none), its (first) embedding model
  if any, and a hash of its config, so that each config gets its own directory,
  e.g.
  `EmbeddingBasedDocumentIndex_chunker_none_gemini-embedding-2__hash_1a2b3c4d`.

  Args:
    index: The index, whose classes must be registered (see `registry`).
  """
  config = config_serialization.get_config(index)
  name = config["class"]
  chunker = _find(config, "chunker")
  if chunker is None:
    name += "_chunker_none"
  else:
    settings = [
        f"{k}={v}"
        for k, v in sorted(chunker["config"].items())
        if isinstance(v, (str, int, float, bool))
    ]
    name += "_chunker_" + "-".join([chunker["class"], *settings])
  if (embedder := _find(config, "embed_backend_config")) is not None:
    embedder_config = embedder["config"]
    name += "_" + (
        embedder_config["model_name"] or embedder_config["backend_class_name"]
    )
  config_hash = hashlib.sha256(
      json.dumps(config, sort_keys=True).encode()
  ).hexdigest()[:8]
  # Model names may contain '/' (e.g. "models/...").
  return f"{name}__hash_{config_hash}".replace("/", "-")
