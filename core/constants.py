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

"""Some constants used throughout the onetwo codebase."""

# Used when errors occur.
ERROR_STRING = '#ERROR#'

# Field in the caching key where the name of the cached function is stored.
CACHING_FUNCTION_NAME_KEY = '_destination'

# Prompt prefix.
PROMPT_PREFIX = 'prefix'

# The field of the Jinja context that contains variables.
CONTEXT_VARS = '__vars__'

# The context variable containing the execution result for a Jinja template.
RESULT_VAR = '_result'

# The fields in the Jinja context variables with results of `choose` command.
CHOICES_VAR = 'choices'
SCORES_VAR = 'scores'
