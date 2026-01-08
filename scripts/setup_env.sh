#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/env"
    exit 1
fi

ENV_PATH="$1"
echo "🔧 Creating: ${ENV_PATH}"
uv venv "${ENV_PATH}" --python 3.11
echo "✅ Done. Next: source scripts/activate_env.sh ${ENV_PATH}"

