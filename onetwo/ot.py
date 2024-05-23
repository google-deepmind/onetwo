# Copyright 2024 DeepMind Technologies Limited.
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

from onetwo.core import composing
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import sampling
from onetwo.evaluation import evaluation

evaluate = evaluation.evaluate
compare_with_critic = evaluation.compare_with_critic
naive_comparison_critic = evaluation.naive_comparison_critic
naive_evaluation_critic = evaluation.naive_evaluation_critic
Executable = executing.Executable
par_iter = executing.par_iter
parallel = executing.parallel
make_executable = executing.make_executable
repeat = sampling.repeat
function_registry = routing.function_registry
RegistryContext = routing.RegistryContext
function_call = routing.function_registry
with_current_registry = routing.with_current_registry
with_registry = routing.with_registry
copy_registry = routing.copy_registry
set_registry = routing.set_registry
make_composable = composing.make_composable
run = executing.run
safe_stream = executing.safe_stream
stream_updates = executing.stream_updates
stream_with_callback = executing.stream_with_callback

