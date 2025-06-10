# Copyright 2025 DeepMind Technologies Limited.
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

"""Registry implementation.

The execution code emits various kinds of requests and they should be
processed by different types of backends.
A backend may process several kinds of requests (e.g. LLM backends may process
both `generate_text` and `score_text` requests).
"""

from __future__ import annotations
from collections.abc import Callable, MutableMapping, AsyncIterator
import contextvars
import copy
import dataclasses
from typing import Any, TypeAlias, TypeVar, final, Generic

from onetwo.core import executing
from onetwo.core import updating

# This context variable contains the global function registry that can be used
# anywhere in the code.
# It should be accessed via the function_registry wrapper.
_function_registry_var = contextvars.ContextVar[dict](
    'registry', default=dict()
)


_RegistryEntry: TypeAlias = Callable[..., Any]
_RegistryData: TypeAlias = tuple[dict[str, Any], dict[str, Any]]
_T = TypeVar('_T')


class RegistryReference:
  """Parent class to indicate that a Registry entry is a reference object.

  See routing._Registry.copy() for additional details.

  This is used to indicate when to actually copy the entry in the registry.
  Indeed, when creating a copy of the registry, we only do a shallow copy since
  the registered entries might be complex objects (e.g. a bound method that is
  bound to a complex object) that we don't want to copy.
  But we may also register some shallow objects that hold references to a
  method for example.
  So we use this parent class to indicate to _Registry.copy that it can
  copy it.
  """


class _Registry(MutableMapping[str, _RegistryEntry]):
  """Registry to store the mapping between names and functions."""

  @executing.make_executable  # pytype: disable=wrong-arg-types
  async def __call__(self, destination: str, *args, **kwargs) -> Any:
    # We call the registry function.
    result = _function_registry_var.get()[destination](*args, **kwargs)
    # If the result is an Executable, it will be executed, due to the
    # make_executable decorator which executes results by default.
    return result

  def __getitem__(self, key: str):
    if key not in _function_registry_var.get():
      raise KeyError(
          f'Key "{key}" not registered in registry:'
          f' {_function_registry_var.get().keys()}'
      )
    return _function_registry_var.get()[key]

  def __setitem__(self, key: str, item: _RegistryEntry):
    _function_registry_var.get()[key] = item

  def __delitem__(self, key: str):
    del _function_registry_var.get()[key]

  def __iter__(self):
    return iter(_function_registry_var.get())

  def __len__(self):
    return len(_function_registry_var.get())

  def __repr__(self):
    return repr(_function_registry_var.get())


def _copy_also_references(registry: dict[str, Any]) -> dict[str, Any]:
  """Returns a copy of the registry.

  This performs a shallow copy, but the entries that are references, i.e.
  of type executing.RegistryReference will be copied.
  This allows in particular to get the expected behaviour when copying
  a registry that contains builtin functions that have been configured,
  as the configuration is stored in a reference object and needs to be copied.

  Args:
    registry: The registry to copy.

  Returns:
    A copy of the registry.
  """
  registry_copy = dict()
  for key, entry in registry.items():
    registry_copy[key] = (
        copy.copy(entry)
        if isinstance(entry, RegistryReference)
        else entry
    )
  return registry_copy


function_registry = _Registry()
config_registry = contextvars.ContextVar[dict](
    'config_registry', default=dict()
)


def copy_registry() -> _RegistryData:
  return _copy_also_references(_function_registry_var.get()), copy.copy(
      config_registry.get()
  )


def set_registry(registry: _RegistryData) -> None:
  _function_registry_var.set(registry[0])
  config_registry.set(registry[1])


@dataclasses.dataclass
class _RegistryDataWrapper(
    Generic[_T], executing.Executable[_T]
):
  """Wraps an executable with the registry data.

  Attributes:
    wrapped: Executable to be wrapped.
    registry: Registry to use when executing this Executable.
  """

  wrapped: executing.Executable[_T]
  registry: _RegistryData

  @final
  async def _aiterate(
      self, iteration_depth: int = 1
  ) -> AsyncIterator[updating.Update[_T]]:
    """Yields the intermediate values and calls the final_value_callback."""
    with RegistryContext(self.registry):
      it = self.wrapped.with_depth(iteration_depth).__aiter__()
    while True:
      with RegistryContext(self.registry):
        try:
          update = await it.__anext__()
        except StopAsyncIteration:
          break
      yield update

  @final
  async def _aexec(self) -> _T:
    """Iterate this value until done (including calling final_value_callback).

    Returns:
      The final value given by the AsyncIterator _inner().
    """
    with RegistryContext(self.registry):
      result = await self.wrapped
    return result


def with_current_registry(
    executable: executing.Executable,
) -> _RegistryDataWrapper:
  """Wraps an executable and attaches the current registry to it."""
  registry = copy_registry()
  return _RegistryDataWrapper(executable, registry)


def with_registry(
    executable: executing.Executable, registry: _RegistryData
) -> _RegistryDataWrapper:
  """Wraps an executable and attaches the given registry to it."""
  return _RegistryDataWrapper(executable, registry)


@dataclasses.dataclass
class RegistryContext:
  """Context Manager to update the registry locally.

  Attributes:
    registry: An optional registry to use in this context manager.
      If None is provided, makes a "local" copy of the function_registry
      that can be modified within the context.
  """
  registry: _RegistryData | None = None

  def __enter__(self):
    if self.registry is None:
      # We create a copy of the current function_registry.
      updated = _copy_also_references(_function_registry_var.get())
      updated_config = copy.copy(config_registry.get())
    else:
      updated, updated_config = self.registry
    # We replace the current function_registry by the copy.
    self._token = _function_registry_var.set(updated)
    self._token_config = config_registry.set(updated_config)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # We reset the function_registry to its value prior to entering the
    # context manager.
    _function_registry_var.reset(self._token)
    config_registry.reset(self._token_config)
