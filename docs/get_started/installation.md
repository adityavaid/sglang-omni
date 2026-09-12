# 🚀 Installation

Current stable release: **v0.1.5** on [PyPI](https://pypi.org/project/sglang-omni/).

Choose the path for your platform. Docker is recommended for NVIDIA CUDA —
UCX, flash-attn, SGLang, and CUDA are prebuilt. Apple Silicon has a dedicated
source installer below.

> **Intel GPU (XPU)?** For Intel Arc GPUs, see [Installation — Intel XPU](./installation_xpu.md), which uses [`pyproject_xpu.toml`](../../pyproject_xpu.toml) + the PyTorch XPU wheel index instead of the CUDA-only pins below.

> **Intel CPU?** Also not this page. See [Installation — Intel CPU](./installation_cpu.md), which uses [`pyproject_cpu.toml`](../../pyproject_cpu.toml) + the PyTorch CPU wheel index.

> **Ascend NPU?** See [Installation — Ascend NPU](./installation_npu.md) for the supported software stack, prerequisites, and installation helper.

## 🐳 Option A: Docker (recommended)

**1. Pull the image**

```bash
docker pull hongccc/sglang-omni:dev
```

Only the `dev` tag is published today. It moves with main — pin by digest for reproducible runs:

```bash
docker pull lmsysorg/sglang-omni@sha256:<digest>
```

**2. Run the container**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

**3. Install `sglang-omni` inside the container**

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

<a id="macos-apple-silicon"></a>

## 🍎 Option B: macOS Apple Silicon installer

From a checkout of this branch, run:

```bash
SGLANG_OMNI_EXTRAS=mlx ./install.sh
source .venv-apple/bin/activate
```

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.19` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.
`ffmpeg@7` is intentional: `torchcodec==0.15.0` ships loaders for FFmpeg 4 through 8
only, and the unversioned formula installs FFmpeg 9. At runtime, expose its libraries:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Homebrew must be installed before running the script. If `brew` is missing, the
script prints an error and exits; install it yourself from
[brew.sh](https://brew.sh), then rerun. The installer never invokes `sudo` or
Homebrew's bootstrapper. Use `--non-interactive` (or `NONINTERACTIVE=1`) to
disable Homebrew auto-update in CI, `SGLANG_OMNI_VENV=/path/to/venv` to choose a virtualenv, and
`SGLANG_OMNI_EXTRAS=audar-tts,fun-cosyvoice3` to enable optional extras.
The persistent SGLang source checkout defaults to
`~/.cache/sglang-omni/sglang-v0.5.19` and can be changed with
`SGLANG_SOURCE_DIR`. Slow or proxied networks can override the installer's uv
defaults with `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

This path currently supports macOS 14 or newer on `arm64` only (the pinned
`torch==2.13.0`, `torchvision==0.28.0` and `torchcodec==0.15.0` wheels are built
for `macosx_14_0_arm64`) and is intended for Qwen3-Omni's native MLX path and
Qwen3-ASR's existing MLX/Torch-MPS paths. Other platforms should use the
Docker, manual, or Intel XPU instructions below. Common failures are a missing
Homebrew/uv on `PATH`, an unavailable Python 3.12 toolchain, or forgetting the
`DYLD_LIBRARY_PATH` export when starting an audio server.

The `mlx` extra selects Qwen3-Omni's native MLX requirements
(`mlx>=0.32.2`, `mlx-lm>=0.31.2`); `mlx-vlm` is not required. The root
`pyproject.toml` remains the installation manifest.
[`pyproject_apple.toml`](../../pyproject_apple.toml) is a reference profile,
not an installer input: do not copy it over `pyproject.toml`. There is no
separate `scripts/apple/install_apple.sh` or checkpoint-preparation step.

### Reuse an existing MLX environment without modifying it

Skip the installer if a compatible environment already exists. From this
checkout, point Python at the checkout instead of changing shared editable
installs:

```bash
export PY="/absolute/path/to/existing-mlx-venv/bin/python"
export SGLANG_SOURCE="/absolute/path/to/compatible-sglang/python"
export PYTHONPATH="$PWD:$SGLANG_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m sglang_omni.cli serve --help
```

The environment must already provide the runtime dependencies. Prepending
compatible SGLang source also overrides an older editable install without
modifying it. Current main requires
`sglang/srt/arg_groups/model_override_base.py`; a checkout missing that module
is too old. This bypasses dependency installation, not compatibility checks.
The root manifest retains main's `sglang==0.5.19` pin, but that wheel is not
available on the package index used for this port. Use the supplied environment
and compatible local source override below rather than downgrading the root
pin or attempting to replace the shared environment's SGLang installation.

For Qwen3-Omni INT4, set `MODEL_DIR` to the complete downloaded checkpoint
directory containing `config.json`, processor assets, and **all** safetensor
shards. A Hub snapshot containing only configuration is not sufficient.
No copy, conversion, or dense code2wav sidecar is required:

```bash
SGLANG_USE_MLX=1 HF_HUB_OFFLINE=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" --host 127.0.0.1 --port 8008
```

Add `--text-only` to skip speech generation. See the
[Qwen3-Omni usage guide](../basic_usage/qwen3_omni.md#apple-silicon-mlx)
for the pinned checkpoint and runtime restrictions.

For the local sibling-checkout layout used by this port:

```bash
export PY="$PWD/../sglang-fork diff/.venv-mlx-dev/bin/python"
export SGLANG_SOURCE="$PWD/../sglang-core-ai/python"
export PYTHONPATH="$PWD:$SGLANG_SOURCE"
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct-4bit-93b3cbdd"
SGLANG_USE_MLX=1 HF_HUB_OFFLINE=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" --host 127.0.0.1 --port 8008
```

Run this from the repository root. The virtualenv's original editable SGLang
checkout is too old for this port; `sglang-core-ai/python` supplies the
compatible source without modifying either shared checkout or environment.

### Run from a hosted installer

The script also supports a downloaded or `curl | bash` invocation: when it is
not inside an sglang-omni checkout, it clones the repository specified by
`SGLANG_OMNI_REPO` and `SGLANG_OMNI_REF` into the cache and installs that
checkout. Prefer downloading, reviewing, and then running a pinned script:

```bash
curl -fsSLo /tmp/sglang-omni-install.sh \
  https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh
less /tmp/sglang-omni-install.sh
chmod +x /tmp/sglang-omni-install.sh
SGLANG_OMNI_REF=<commit> /tmp/sglang-omni-install.sh
```

Piping a remote script directly to Bash executes code without a review step;
use it only when that trade-off is acceptable:

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh \
  | SGLANG_OMNI_REF=<commit> bash
```

For a fork or an internal mirror, set `SGLANG_OMNI_REPO` and
`SGLANG_OMNI_REF` explicitly. The hosted mode stores the project checkout at
`~/.cache/sglang-omni/sglang-omni-<ref>` by default; override it with
`SGLANG_OMNI_PROJECT_DIR`.

## 🛠️ Option C: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.13.0` and SGLang 0.5.19's `nvidia-cutlass-dsl` 4.6.2 pin.

Then:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

Latest on the index without a pin: `uv pip install --prerelease=allow sglang-omni`.

### Install from source

For development or unreleased changes:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```
