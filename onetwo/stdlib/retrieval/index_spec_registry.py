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

"""Registry of the `IndexSpec` types that can be built, saved and loaded.

This is *not* a catalogue of built or stored indexes: each entry records an
`IndexSpec` type, which builds an empty `DocumentIndex` with its `make_index`
method, and the stable name under which the spec is saved. The local runner
(`LocalIndexRunner`, `runner='local'`, the default and the one used by
`rag.build_index`) uses this registry, so a custom index becomes buildable via
`index_building.build_index(spec=..., docs=...)` without subclassing any
runner. The Beam runner only supports the built-in embedding specs.

The built-in specs are registered by importing `embedding_index_specs`, which
`index_building` (and therefore `rag`) does. Import it yourself if you use this
module directly.

To make an index buildable, implement `make_index` on its spec and register the
spec with the decorator:

  @index_spec_registry.register_index_spec('my_index')
  @dataclasses_json.dataclass_json
  @dataclasses.dataclass(kw_only=True)
  class MyIndexSpec(index_build_interface.IndexSpec):
    my_param: int = 3

    def make_index(
        self, runtime_kwargs: Mapping[str, Any]
    ) -> indexing.DocumentIndex:
      # Empty index; docs are added later.
      return MyIndex(my_param=self.my_param, runtime_kwargs=runtime_kwargs)

The `name` (defaulting to the class name) is what gets written to
`index_spec.json` next to a built index, so that the index can be loaded again
from its directory. Keep it stable once indexes have been saved with it.

Lookup is by *exact* spec type: a subclass of a registered spec is not
buildable until it is registered itself. This prevents silently building the
parent's index for a subclass that adds new fields. To reuse the parent's
`make_index`, register the subclass explicitly.

Environment-specific settings and credentials (`runtime_kwargs`, e.g. API keys)
are not part of the spec: they are passed to `make_index` separately, so specs
only describe the index.
"""

from collections.abc import Callable, Mapping
import dataclasses
import json
from typing import Any, TypeVar

from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing

IndexSpec = index_build_interface.IndexSpec

_SpecT = TypeVar('_SpecT', bound=IndexSpec)


@dataclasses.dataclass(frozen=True)
class IndexSpecRegistration:
  """A registered index spec.

  Attributes:
    name: Stable name of the spec, used when serializing specs.
    spec_cls: The exact `IndexSpec` subclass.
  """

  name: str
  spec_cls: type[IndexSpec]


_REGISTRATIONS_BY_NAME: dict[str, IndexSpecRegistration] = {}
_REGISTRATIONS_BY_TYPE: dict[type[IndexSpec], IndexSpecRegistration] = {}


def _key_name(key: str | type[IndexSpec]) -> str:
  return key if isinstance(key, str) else key.__name__


def _get_registration(key: str | type[IndexSpec]) -> IndexSpecRegistration:
  """Returns the registration for a spec name or exact spec type."""
  if isinstance(key, str):
    registration = _REGISTRATIONS_BY_NAME.get(key)
  else:
    registration = _REGISTRATIONS_BY_TYPE.get(key)
  if registration is None:
    raise KeyError(
        f'Index spec {_key_name(key)} is not registered. Registered index'
        f' specs: {list_index_specs()}. Register one with'
        ' @index_spec_registry.register_index_spec(<name>).'
    )
  return registration


def register_index_spec(
    name: str | None = None,
) -> Callable[[type[_SpecT]], type[_SpecT]]:
  """Returns a class decorator that registers an `IndexSpec` subclass.

    @index_spec_registry.register_index_spec('my_index')
    @dataclasses_json.dataclass_json
    @dataclasses.dataclass(kw_only=True)
    class MyIndexSpec(index_build_interface.IndexSpec):
      ...

  Indexes are built with the spec's `make_index` method.

  Args:
    name: Stable name of the spec, which is written to `index_spec.json` when
      an index is saved. Defaults to the class name.

  Returns:
    A class decorator that registers the class and returns it unchanged. It
    raises `TypeError` if the class is not an `IndexSpec` subclass or does not
    implement `make_index`, and `ValueError` if the class or name is already
    registered.
  """

  def decorator(spec_cls: type[_SpecT]) -> type[_SpecT]:
    if not (isinstance(spec_cls, type) and issubclass(spec_cls, IndexSpec)):
      raise TypeError(f'{spec_cls!r} is not a subclass of IndexSpec.')
    if spec_cls.make_index is IndexSpec.make_index:
      raise TypeError(f'{spec_cls.__name__} must implement make_index.')
    spec_name = name or spec_cls.__name__
    if spec_cls in _REGISTRATIONS_BY_TYPE:
      raise ValueError(
          f'Index spec {spec_cls.__name__} is already registered as'
          f' {_REGISTRATIONS_BY_TYPE[spec_cls].name!r}.'
      )
    if spec_name in _REGISTRATIONS_BY_NAME:
      raise ValueError(
          f'Index spec name {spec_name!r} is already registered for'
          f' {_REGISTRATIONS_BY_NAME[spec_name].spec_cls.__name__}.'
      )
    registration = IndexSpecRegistration(name=spec_name, spec_cls=spec_cls)
    _REGISTRATIONS_BY_NAME[spec_name] = registration
    _REGISTRATIONS_BY_TYPE[spec_cls] = registration
    return spec_cls

  return decorator


def list_index_specs() -> list[str]:
  """Returns the sorted names of all registered index specs."""
  return sorted(_REGISTRATIONS_BY_NAME)


def is_registered(spec: IndexSpec) -> bool:
  """Returns whether the exact type of `spec` is registered."""
  return type(spec) in _REGISTRATIONS_BY_TYPE


def make_index(
    spec: IndexSpec,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> indexing.DocumentIndex:
  """Returns an empty index built by `spec.make_index`.

  Args:
    spec: The index specification. Its exact type must be registered.
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key).

  Raises:
    KeyError: If `type(spec)` is not registered.
  """
  _get_registration(type(spec))
  return spec.make_index(dict(runtime_kwargs or {}))


def spec_to_dict(spec: IndexSpec) -> dict[str, Any]:
  """Serializes `spec` together with its registered name.

  Fields excluded from the spec's JSON (e.g. `runtime_kwargs`, which may hold
  credentials) are not included.

  Args:
    spec: A spec whose exact type is registered.

  Returns:
    A JSON-serializable dict from which `spec_from_dict` restores the spec.

  Raises:
    KeyError: If `type(spec)` is not registered.
  """
  return {
      'spec_type': _get_registration(type(spec)).name,
      'spec': json.loads(spec.to_json()),
  }


def spec_from_dict(spec_dict: Mapping[str, Any]) -> IndexSpec:
  """Restores a spec from the output of `spec_to_dict`.

  Args:
    spec_dict: The dict produced by `spec_to_dict`.

  Returns:
    An instance of the registered spec class.

  Raises:
    KeyError: If the spec type is not registered (e.g. the module registering
      a custom spec has not been imported).
  """
  spec_cls = _get_registration(spec_dict['spec_type']).spec_cls
  return spec_cls.from_dict(spec_dict['spec'])
