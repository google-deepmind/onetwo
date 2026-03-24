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

"""Utility functions that are shared across many of the OneTwo colabs.

Since these are intended primarily for use in colabs, they may contain `print`
statements, in cases where such output tends to be useful in a colab workflow.
(In colab, the output of `print` statements will be seen by default, whereas
logged content requires extra work in order to access.)
"""

import dataclasses
from onetwo.core import cached_backends


@dataclasses.dataclass(kw_only=True)
class CachedBackends(cached_backends.CachedBackends):
  """Manages a set of backends with their caches.

  Includes print statements for surfacing updates in Colab.
  See `cached_backends.CachedBackends` for more details.
  """

  enable_print_statements: bool = True
