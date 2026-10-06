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

"""Specifications for QA strategies."""

from __future__ import annotations

import dataclasses
from typing import Any

import dataclasses_json
from onetwo.backends import backend_registry


@dataclasses.dataclass(frozen=True, kw_only=True)
class QAStrategySpec(dataclasses_json.DataClassJsonMixin):
  """Specification for QA strategies.

  Attributes:
    qa_strategy_name: The name of the QA strategy this spec configures.
    index_path: Path to the pre-built index (used for RAG).
    generate_backend_config: Config for the generation backend.
    query_time_kwargs: Extra keyword arguments forwarded to the strategy at call
      time. The `max_results` is set to 20 by default, so that if users forget
      to specify it, it won't end up returning all the documents in the index.
    mcp_kwargs: Extra keyword arguments for the `agentic_retrieval` strategy at
      construction time, e.g. `mcp_server_urls`, a comma-separated list of the
      MCP servers whose tools the model may call.
  """
  qa_strategy_name: str = 'rag'
  index_path: str = ''
  generate_backend_config: backend_registry.GenerateBackendConfig | None = None
  query_time_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=lambda: {'max_results': 20}
  )
  mcp_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __str__(self) -> str:
    """Returns a compact, human-readable string representation."""
    parts = []
    parts.append(self.qa_strategy_name)
    for k, v in sorted(self.query_time_kwargs.items()):
      parts.append(f'{k}={v}')
    if self.generate_backend_config:
      parts.append(str(self.generate_backend_config))
    return '-'.join(parts)
