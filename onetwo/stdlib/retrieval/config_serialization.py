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

"""Saving and loading index configs (`index_config.json`).

The config of an object is the registered name of its class (see `registry`)
and its constructor (init) fields, e.g.:

  {'class': 'ChunkingCorpusRewriter',
   'config': {'chunker': {'class': 'ChunkByMaxWords',
                          'config': {'max_words_per_chunk': 100, ...}}}}

Field values are saved as follows:

* Objects of registered classes (inner indexes, rewriters, chunkers,
  formatters, searchers, backend configs) are saved as nested configs.
* `None`, `bool`, `int`, `float` and `str` values, enums (by value), lists (and
  tuples) and string-keyed dicts are saved as JSON values.
* `runtime_kwargs` (environment settings and credentials) is never saved.

Any other value (e.g. an object of an unregistered class) raises a `TypeError`.
Built state (`init=False` fields) is not part of the config: it is saved by the
index's own `save`.
"""

from collections.abc import Mapping
import dataclasses
import enum
import json
import typing
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import registry

# Backend configs are fields of indexes and chunkers. They are registered here,
# as `onetwo.backends` does not depend on the retrieval registry.
registry.register('EmbedBackendConfig')(backend_registry.EmbedBackendConfig)
registry.register('GenerateBackendConfig')(
    backend_registry.GenerateBackendConfig
)
registry.register('TokenizeBackendConfig')(
    backend_registry.TokenizeBackendConfig
)

# Name of the file, written next to a saved index, that holds its config.
INDEX_CONFIG_FILENAME = 'index_config.json'
FORMAT_VERSION = '1.0'

# TODO: Remove once runtime kwargs are passed to indexes by a
# `connect` method instead of being index fields.
_RUNTIME_KWARGS = 'runtime_kwargs'


def get_config(obj: Any) -> dict[str, Any]:
  """Returns the config of `obj`, an instance of a registered dataclass.

  Args:
    obj: The object, e.g. an index.

  Returns:
    A JSON-serializable dict `{'class': <registered name>, 'config': {...}}`.

  Raises:
    KeyError: If the class of `obj` is not registered.
    TypeError: If a field value cannot be saved (see the module docstring).
  """
  name = registry.get_name(obj)
  config = {}
  for field in dataclasses.fields(obj):
    if not field.init or field.name == _RUNTIME_KWARGS:
      continue
    config[field.name] = _to_config(
        getattr(obj, field.name), f'{name}.{field.name}'
    )
  return {'class': name, 'config': config}


def from_config(
    config: Mapping[str, Any],
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> Any:
  """Returns a new object built from `config` (as returned by `get_config`).

  Args:
    config: The config.
    runtime_kwargs: Environment settings and credentials (e.g. API key), passed
      to every (nested) object with a `runtime_kwargs` init field.

  Raises:
    KeyError: If a class in `config` is not registered (e.g. the module that
      registers it has not been imported).
  """
  cls = registry.get_class(config['class'])
  hints = typing.get_type_hints(cls)
  kwargs = {}
  for name, value in config['config'].items():
    value = _from_config(value, runtime_kwargs)
    hint = hints.get(name)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
      value = hint(value)
    kwargs[name] = value
  if runtime_kwargs is not None and any(
      f.name == _RUNTIME_KWARGS and f.init for f in dataclasses.fields(cls)
  ):
    kwargs[_RUNTIME_KWARGS] = dict(runtime_kwargs)
  return cls(**kwargs)


def save_index_config(index: indexing.Index, directory: str) -> None:
  """Writes the config of `index` to `directory/index_config.json`."""
  path = epath.Path(directory)
  path.mkdir(parents=True, exist_ok=True)
  (path / INDEX_CONFIG_FILENAME).write_text(
      json.dumps(
          {'format_version': FORMAT_VERSION, 'index': get_config(index)},
          indent=2,
      )
  )


def make_index_from_config_file(
    directory: str,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> indexing.DocumentIndex:
  """Returns a new, empty index built from `directory/index_config.json`.

  Args:
    directory: Directory the config was saved to by `save_index_config`.
    runtime_kwargs: Environment settings and credentials (see `from_config`).

  Raises:
    ValueError: If the config was saved in an unsupported `format_version`.
    TypeError: If the saved config does not describe a `DocumentIndex`.
  """
  path = epath.Path(directory) / INDEX_CONFIG_FILENAME
  saved = json.loads(path.read_text())
  if saved.get('format_version') != FORMAT_VERSION:
    raise ValueError(
        f'{path} has format_version {saved.get("format_version")!r}; only'
        f' {FORMAT_VERSION!r} is supported.'
    )
  index = from_config(saved['index'], runtime_kwargs)
  if not isinstance(index, indexing.DocumentIndex):
    raise TypeError(
        f'{path} describes a {type(index).__name__}, not a DocumentIndex.'
    )
  return index


def _to_config(value: Any, path: str) -> Any:
  """Returns the JSON-serializable config of a field value."""
  if registry.is_registered(value):
    return get_config(value)
  if isinstance(value, enum.Enum):
    return value.value
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, (list, tuple)):
    return [_to_config(v, f'{path}[{i}]') for i, v in enumerate(value)]
  if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
    return {k: _to_config(v, f'{path}[{k!r}]') for k, v in value.items()}
  raise TypeError(
      f'Cannot save {path} of type {type(value).__qualname__}: register its'
      ' class with `@registry.register(...)`.'
  )


def _from_config(value: Any, runtime_kwargs: Mapping[str, Any] | None) -> Any:
  """Returns the value for its config `value` (see `_to_config`)."""
  if isinstance(value, dict) and set(value) == {'class', 'config'}:
    return from_config(value, runtime_kwargs)
  if isinstance(value, list):
    return [_from_config(v, runtime_kwargs) for v in value]
  if isinstance(value, dict):
    return {k: _from_config(v, runtime_kwargs) for k, v in value.items()}
  return value
