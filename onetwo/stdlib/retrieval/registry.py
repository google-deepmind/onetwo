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

"""Registry of OneTwo RAG component classes, by stable name.

Indexes and their components (searchers, chunkers, formatters, rewriters,
inner indexes) are dataclasses whose constructor (init) fields are their
settings. Saved index configs will refer to the class of each such object by
the name it is registered under here, never by its Python path, so that classes
can be moved or renamed without breaking saved configs. The name is what saved
configs store, so it must stay stable.

Register a class with the `register` decorator:

  @registry.register('mylab.SentenceChunker')
  @dataclasses.dataclass
  class SentenceChunker(chunking.Chunker):
    ...

Built-in classes are registered under their class name. Custom classes should
use a prefix (e.g. `'mylab.SentenceChunker'`) to avoid clashes.

TODO: system12 defines another `EmbeddingBasedDocumentIndex`; it
needs its own explicit name if it is registered.
"""

from collections.abc import Callable
import dataclasses
from typing import Any, TypeVar

_T = TypeVar('_T', bound=type)

_CLASSES_BY_NAME: dict[str, type[Any]] = {}
_NAMES_BY_CLASS: dict[type[Any], str] = {}


def _same_class(a: type[Any], b: type[Any]) -> bool:
  """Returns whether `a` and `b` are the same class, possibly redefined."""
  return (a.__module__, a.__qualname__) == (b.__module__, b.__qualname__)


def register(name: str) -> Callable[[_T], _T]:
  """Returns a class decorator that registers the class under `name`.

  Registering the same class again (e.g. when a module is reloaded or a Colab
  cell is re-run) replaces the previous registration.

  Args:
    name: Stable name to register the class under.

  Raises:
    TypeError: If the decorated class is not a dataclass.
    ValueError: If `name` is already registered to another class.
  """

  def decorator(cls: _T) -> _T:
    if not dataclasses.is_dataclass(cls):
      raise TypeError(f'{cls.__name__} must be a dataclass to be registered.')
    registered_cls = _CLASSES_BY_NAME.get(name)
    if registered_cls is not None:
      if not _same_class(registered_cls, cls):
        raise ValueError(
            f'{name!r} is already registered to'
            f' {registered_cls.__module__}.{registered_cls.__qualname__}.'
        )
      _NAMES_BY_CLASS.pop(registered_cls, None)
    _CLASSES_BY_NAME[name] = cls
    _NAMES_BY_CLASS[cls] = name
    return cls

  return decorator


def get_class(name: str) -> type[Any]:
  """Returns the class registered under `name`.

  Args:
    name: The registered name.

  Raises:
    KeyError: If no class is registered under `name`.
  """
  if name not in _CLASSES_BY_NAME:
    raise KeyError(
        f'Unknown class {name!r}. Registered classes: {list_names()}. Import'
        ' the module that defines (and registers) it first.'
    )
  return _CLASSES_BY_NAME[name]


def get_name(obj_or_cls: Any) -> str:
  """Returns the registered name of the class of `obj_or_cls` (or the class).

  Args:
    obj_or_cls: An instance of a registered class, or a registered class.

  Raises:
    KeyError: If the class is not registered. (A subclass of a registered class
      is not registered unless it is decorated itself.)
  """
  cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
  if cls not in _NAMES_BY_CLASS:
    raise KeyError(
        f'{cls.__qualname__} is not registered; decorate it with'
        ' `@registry.register(name)`.'
    )
  return _NAMES_BY_CLASS[cls]


def list_names() -> list[str]:
  """Returns the sorted names of all registered classes."""
  return sorted(_CLASSES_BY_NAME)
