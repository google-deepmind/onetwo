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

#!/bin/bash

# Run from the onetwo root dir that contains README.md and other files.
ONETWO_ROOT_DIR="."
START_SERVER_CMD="python3 ${ONETWO_ROOT_DIR}/onetwo/backends/run_model_server.py"
START_CLIENT_CMD="python3 ${ONETWO_ROOT_DIR}/onetwo/backends/onetwo_api_manual_test.py"
CACHE_DIR="${ONETWO_ROOT_DIR}/tmp"
# Requires portpicker to be installed.
PORT=`python3 -m portpicker $$`

function clean_fail() {
  kill $!;
  exit 1
}

echo "Using port ${PORT}"

set -o xtrace

# Start model_server.
${START_SERVER_CMD} --port="${PORT}" &
sleep 10

# Start client and run test: from scratch.
${START_CLIENT_CMD} \
  --endpoint="http://localhost:${PORT}" \
  --cache_dir=${CACHE_DIR} & sleep 3 || clean_fail

# Start client and run test: cached replies.
${START_CLIENT_CMD} \
  --endpoint="http://localhost:${PORT}" --cache_dir=${CACHE_DIR}\
  --load_cache_file=True || clean_fail

# Stop model_server.
kill $!
