#!/usr/bin/env bash
# Builds the native ASR runtime; opt-in Qwen3-Omni requires external FFmpeg 7.
# Note (Jiaxin Deng): MLX comes from the pinned mlx wheel, the same build the
# Python reference server uses.
set -euo pipefail

output="${1:?usage: build_runtime.sh <output-directory> [<work-directory>] (SGLANG_BUILD_QWEN3_OMNI=ON opts into FFmpeg-dependent Omni)}"
work="${2:-$output/.build}"
native_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_omni="${SGLANG_BUILD_QWEN3_OMNI:-OFF}"
case "$build_omni" in
  ON|OFF) ;;
  *) echo "SGLANG_BUILD_QWEN3_OMNI must be ON or OFF" >&2; exit 1 ;;
esac

test "$(uname -m)" = arm64 || { echo "Apple Silicon is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

mkdir -p "$work"
uv venv -q -p 3.12 --allow-existing "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q -r "$native_dir/scripts/build-tools.lock"
mlx_dir="$("$work/venv/bin/python" -c 'import mlx.core, os; print(os.path.join(os.path.dirname(mlx.core.__file__), "share/cmake/MLX"))')"
export PATH="$work/venv/bin:$PATH"
cmake -S "$native_dir" -B "$work/cmake" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$mlx_dir" -DCMAKE_INSTALL_PREFIX="$output" \
  -DSGLANG_BUILD_QWEN3_OMNI="$build_omni"
cmake --build "$work/cmake"
cmake --install "$work/cmake"
echo "VOXT_OMNI_RUNTIME=$output/bin/qwen3_asr_server"
