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
