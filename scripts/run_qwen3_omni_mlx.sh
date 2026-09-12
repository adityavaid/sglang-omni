#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_PYTHON="${MLX_PYTHON:-$REPO/../sglang-fork diff/.venv-mlx-dev/bin/python}"
SGLANG_SOURCE="${SGLANG_SOURCE:-$REPO/../sglang-core-ai/python}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3-Omni-30B-A3B-Instruct-4bit-93b3cbdd}"

if [[ ! -x "$MLX_PYTHON" ]]; then
    printf 'Python executable not found: %s\nSet MLX_PYTHON to your MLX environment.\n' "$MLX_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SGLANG_SOURCE/sglang/srt/arg_groups/model_override_base.py" ]]; then
    printf 'SGLANG_SOURCE must contain sglang/srt/arg_groups/model_override_base.py: %s\n' "$SGLANG_SOURCE" >&2
    exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    printf 'Checkpoint config not found: %s/config.json\nSet MODEL_DIR to the downloaded Qwen3-Omni MLX checkpoint.\n' "$MODEL_DIR" >&2
    exit 1
fi

# Do not repoint editable installs in the shared virtual environment.
export PYTHONPATH="$REPO:$SGLANG_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_USE_MLX=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SGLANG_OMNI_VIDEO_READER="${SGLANG_OMNI_VIDEO_READER:-pyav}"
cd "$REPO"
exec "$MLX_PYTHON" -m sglang_omni.cli serve \
    --model-path "$MODEL_DIR" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8008}" \
    "$@"
