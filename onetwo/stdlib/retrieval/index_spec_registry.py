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

"""Registry mapping `IndexSpec` types to the factories that build their indexes.

This is *not* a catalogue of built or stored indexes: each entry describes how
to construct an empty `DocumentIndex` from an `IndexSpec` of one exact type.
The local runner (`LocalIndexRunner`, `runner='local'`, the default and the one
used by `rag.build_index`) uses this registry, so a custom index becomes
buildable via `index_building.build_index(spec=..., docs=...)` without
subclassing any runner. The Beam runner only supports the built-in embedding
specs.

The built-in specs are registered by importing `embedding_index_specs`, which
`index_building` (and therefore `rag`) does. Import it yourself if you use this
module directly.

To make a custom index buildable:

  @dataclasses_json.dataclass_json
  @dataclasses.dataclass(kw_only=True)
  class MyIndexSpec(index_build_interface.IndexSpec):
    my_param: int = 3

  def make_my_index(
      spec: MyIndexSpec, runtime_kwargs: Mapping[str, Any]
  ) -> indexing.DocumentIndex:
    # Empty index; docs are added later.
    return MyIndex(my_param=spec.my_param, runtime_kwargs=runtime_kwargs)

  index_spec_registry.register_index_spec(
      MyIndexSpec, make_my_index, name='my_index'
  )

The `name` (defaulting to the class name) is what gets written to
`index_spec.json` next to a built index, so that the index can be loaded again
from its directory. Keep it stable once indexes have been saved with it.

Lookup is by *exact* spec type: a subclass of a registered spec is not
buildable until it is registered itself. This prevents silently building the
parent's index for a subclass that adds new fields. To reuse the parent's
factory, register it explicitly for the subclass.

Environment-specific settings and credentials (`runtime_kwargs`, e.g. API keys)
are not part of the spec: they are passed to the factory separately, so specs
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

# A function that builds an empty index for a given spec and runtime kwargs.
IndexFactory = Callable[[Any, Mapping[str, Any]], indexing.DocumentIndex]


@dataclasses.dataclass(frozen=True)
class IndexSpecRegistration:
  """A registered index spec.

  Attributes:
    name: Stable name of the spec, used when serializing specs.
    spec_cls: The exact `IndexSpec` subclass.
    factory: Builds an empty index from a `spec_cls` instance and runtime
      kwargs.
  """

  name: str
  spec_cls: type[IndexSpec]
  factory: IndexFactory


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
        f'No index factory registered for {_key_name(key)}. Registered index'
        f' specs: {list_index_specs()}. Register one with'
        ' index_spec_registry.register_index_spec(<spec class>, <factory>,'
        ' name=<name>).'
    )
  return registration


def register_index_spec(
    spec_cls: type[_SpecT],
    factory: Callable[[_SpecT, Mapping[str, Any]], indexing.DocumentIndex],
    *,
    name: str | None = None,
) -> None:
  """Registers the factory that builds indexes for `spec_cls`.

  Args:
    spec_cls: The exact `IndexSpec` subclass handled by `factory`.
    factory: A callable that takes a `spec_cls` instance and the runtime kwargs
      (environment-specific settings and credentials for the index's backends)
      and returns an empty `DocumentIndex` configured accordingly.
    name: Stable name for the spec, written to `index_spec.json` when an index
      is saved. Defaults to `spec_cls.__name__`.

  Raises:
    TypeError: If `spec_cls` is not a subclass of `IndexSpec`.
    ValueError: If `spec_cls` or `name` is already registered.
  """
  if not (isinstance(spec_cls, type) and issubclass(spec_cls, IndexSpec)):
    raise TypeError(f'{spec_cls!r} is not a subclass of IndexSpec.')
  name = name or spec_cls.__name__
  if spec_cls in _REGISTRATIONS_BY_TYPE:
    raise ValueError(
        f'Index spec {spec_cls.__name__} is already registered as'
        f' {_REGISTRATIONS_BY_TYPE[spec_cls].name!r}.'
    )
  if name in _REGISTRATIONS_BY_NAME:
    raise ValueError(
        f'Index spec name {name!r} is already registered for'
        f' {_REGISTRATIONS_BY_NAME[name].spec_cls.__name__}.'
    )
  registration = IndexSpecRegistration(
      name=name, spec_cls=spec_cls, factory=factory
  )
  _REGISTRATIONS_BY_NAME[name] = registration
  _REGISTRATIONS_BY_TYPE[spec_cls] = registration


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
  """Returns an empty index built by the factory registered for `type(spec)`.

  Args:
    spec: The index specification.
    runtime_kwargs: Environment-specific settings and credentials for the
      index's backends (e.g. API key).

  Raises:
    KeyError: If no factory is registered for `type(spec)`.
  """
  return _get_registration(type(spec)).factory(
      spec, dict(runtime_kwargs or {})
  )


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
