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

"""Registry and configuration for OneTwo retrieval chunkers.

Only the common text chunkers defined in `chunking` are registered here.
Chunkers with heavier dependencies are registered by their own modules when
imported (e.g. importing `pdf_chunking` registers 'pdf_pages'), so that
depending on this registry does not pull in those dependencies.
"""

import dataclasses
from typing import Any

import dataclasses_json
from onetwo.stdlib.retrieval import chunking

CHUNKER_REGISTRY: dict[str, type[chunking.Chunker]] = {
    'none': chunking.NoChunking,
    'words': chunking.ChunkByMaxWords,
    'tokens': chunking.ChunkByMaxTokens,
}


@dataclasses.dataclass(frozen=True)
class ChunkerConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for instantiating a Chunker.

  Attributes:
    chunker_name: Registered chunking strategy name (e.g. 'none', 'words',
      'tokens', 'pdf_pages').
    kwargs: Keyword arguments to pass to the chunker constructor.
  """

  chunker_name: str = 'none'
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __str__(self) -> str:
    """Returns a compact, human-readable string representation."""
    parts = [self.chunker_name]
    for k, v in sorted(self.kwargs.items()):
      parts.append(f'{k}={v}')
    return '-'.join(parts)


def register_chunker(
    name: str,
    chunker_cls: type[chunking.Chunker],
) -> None:
  """Registers a chunker class under `name`.

  Args:
    name: Strategy name (e.g. `'words'`, `'tokens'`, `'none'`).
    chunker_cls: A `Chunker` subclass.

  Raises:
    TypeError: If `chunker_cls` is not a dataclass.
    ValueError: If `name` is already registered.
  """
  if not dataclasses.is_dataclass(chunker_cls):
    raise TypeError(
        f'{chunker_cls.__name__} must be a dataclass to be registered.'
    )
  if name in CHUNKER_REGISTRY:
    raise ValueError(f'Chunker {name!r} is already registered.')
  CHUNKER_REGISTRY[name] = chunker_cls


def list_chunkers() -> list[str]:
  """Returns the sorted names of all registered chunking strategies."""
  return sorted(CHUNKER_REGISTRY)


def get_chunker(config: ChunkerConfig) -> chunking.Chunker:
  """Instantiates a chunker from its configuration.

  Args:
    config: Configuration specifying the chunker strategy and keyword arguments.

  Returns:
    The instantiated `Chunker`.

  Raises:
    KeyError: If `config.chunker_name` is not registered.
  """
  if config.chunker_name not in CHUNKER_REGISTRY:
    raise KeyError(
        f'Unknown chunking strategy: {config.chunker_name!r}. Registered'
        f' chunkers: {list_chunkers()}. Make sure the module that registers'
        ' it is imported (e.g. `pdf_chunking` for \'pdf_pages\'), or register'
        ' one with chunker_registry.register_chunker(<name>, <chunker_cls>).'
    )
  return CHUNKER_REGISTRY[config.chunker_name].from_dict(config.kwargs)
