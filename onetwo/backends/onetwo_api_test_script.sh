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

#!/bin/bash

DIR="."
# Requires portpicker to be installed.
PORT=`python3 -m portpicker $$`

function clean_fail() {
  kill $!;
  exit 1
}

echo "Using port ${PORT}"

set -o xtrace

# Start model_server.
"${DIR}/onetwo/backends/run_model_server" --port="${PORT}" &
sleep 5

# Start client and run test.
"${DIR}/onetwo/backends/onetwo_api_manual_test" \
  --endpoint="http://localhost:${PORT}" || clean_fail

# Stop model_server.
kill $!
