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

"""Entry point to the OneTwo library.

```python
from onetwo import ot
```
"""

from onetwo import version
from onetwo.core import composing
from onetwo.core import executing
from onetwo.core import results
from onetwo.core import routing
from onetwo.core import sampling
from onetwo.evaluation import evaluation

__version__: str = version.__version__

compare_with_critic = evaluation.compare_with_critic
copy_registry = routing.copy_registry
evaluate = evaluation.evaluate
Executable = executing.Executable
function_call = routing.function_registry
function_registry = routing.function_registry
HTMLRenderer = results.HTMLRenderer
make_composable = composing.make_composable
make_executable = executing.make_executable
naive_comparison_critic = evaluation.naive_comparison_critic
naive_evaluation_critic = evaluation.naive_evaluation_critic
par_iter = executing.par_iter
parallel = executing.parallel
RegistryContext = routing.RegistryContext
repeat = sampling.repeat
run = executing.run
safe_stream = executing.safe_stream
set_registry = routing.set_registry
stream_updates = executing.stream_updates
stream_with_callback = executing.stream_with_callback
with_current_registry = routing.with_current_registry
with_registry = routing.with_registry

