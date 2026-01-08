#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: source $0 /path/to/env"
    return 1 2>/dev/null || exit 1
fi

ENV_PATH="$1"
if [ ! -f "${ENV_PATH}/bin/activate" ]; then
    echo "❌ Invalid env. Run: ./scripts/setup_env.sh ${ENV_PATH}"
    return 1 2>/dev/null || exit 1
fi

source "${ENV_PATH}/bin/activate"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo "✅ Activated: ${ENV_PATH}"

