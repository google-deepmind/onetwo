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

The built-in specs are registered by importing `builtin_index_specs`, which
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

  index_spec_registry.register_index_spec(MyIndexSpec, make_my_index)

Lookup is by *exact* spec type: a subclass of a registered spec is not
buildable until it is registered itself. This prevents silently building the
parent's index for a subclass that adds new fields. To reuse the parent's
factory, register it explicitly for the subclass.

Environment-specific settings and credentials (`runtime_kwargs`, e.g. API keys)
are not part of the spec: they are passed to the factory separately, so specs
only describe the index.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from onetwo.stdlib.retrieval import index_build_interface
from onetwo.stdlib.retrieval import indexing

IndexSpec = index_build_interface.IndexSpec

_SpecT = TypeVar('_SpecT', bound=IndexSpec)

# A function that builds an empty index for a given spec and runtime kwargs.
IndexFactory = Callable[[Any, Mapping[str, Any]], indexing.DocumentIndex]

_INDEX_SPEC_REGISTRY: dict[type[IndexSpec], IndexFactory] = {}


def register_index_spec(
    spec_cls: type[_SpecT],
    factory: Callable[[_SpecT, Mapping[str, Any]], indexing.DocumentIndex],
) -> None:
  """Registers the factory that builds indexes for `spec_cls`.

  Args:
    spec_cls: The exact `IndexSpec` subclass handled by `factory`.
    factory: A callable that takes a `spec_cls` instance and the runtime kwargs
      (environment-specific settings and credentials for the index's backends)
      and returns an empty `DocumentIndex` configured accordingly.

  Raises:
    TypeError: If `spec_cls` is not a subclass of `IndexSpec`.
    ValueError: If `spec_cls` is already registered.
  """
  if not (isinstance(spec_cls, type) and issubclass(spec_cls, IndexSpec)):
    raise TypeError(f'{spec_cls!r} is not a subclass of IndexSpec.')
  if spec_cls in _INDEX_SPEC_REGISTRY:
    raise ValueError(f'Index spec {spec_cls.__name__} is already registered.')
  _INDEX_SPEC_REGISTRY[spec_cls] = factory


def list_index_specs() -> list[str]:
  """Returns the sorted names of all registered index specs."""
  return sorted(spec_cls.__name__ for spec_cls in _INDEX_SPEC_REGISTRY)


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
  spec_cls = type(spec)
  if spec_cls not in _INDEX_SPEC_REGISTRY:
    raise KeyError(
        f'No index factory registered for {spec_cls.__name__}. Registered index'
        f' specs: {list_index_specs()}. Register one with'
        f' index_spec_registry.register_index_spec({spec_cls.__name__},'
        ' <factory>).'
    )
  return _INDEX_SPEC_REGISTRY[spec_cls](spec, dict(runtime_kwargs or {}))
