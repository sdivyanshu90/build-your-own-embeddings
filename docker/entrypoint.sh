#!/bin/sh
set -eu

set -- embedding-project serve --model-path "$MODEL_PATH" --host 0.0.0.0 --port 8000
if [ -f "${INDEX_PATH}/index_manifest.json" ]; then
  set -- "$@" --index-path "$INDEX_PATH"
fi
exec "$@"

