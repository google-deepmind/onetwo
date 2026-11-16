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

"""RAG (Retrieval-Augmented Generation) API for OneTwo."""

from onetwo.stdlib.qa import evaluate_rag_lib as _evaluate_rag_module
from onetwo.stdlib.retrieval import build_index_lib as _build_index_module
from onetwo.stdlib.retrieval import serialization as _serialization
from onetwo.stdlib.retrieval.mcp import mcp_server as _mcp_server

build_index = _build_index_module.build_index
IndexType = _build_index_module.IndexType
evaluate = _evaluate_rag_module.evaluate_rag
StorageFormat = _serialization.StorageFormat
start_mcp = _mcp_server.start_mcp
