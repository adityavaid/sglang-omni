# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is a multi-modal model
that accepts text, image, audio, and video input and can produce text-only or text + audio output.
This page covers every supported server configuration — use the generator to get the exact launch
command for your hardware, then check the tables to confirm your combination is supported.

## Prerequisites

```bash
docker pull hongccc/sglang-omni:dev
docker run -it --shm-size 32g --gpus all hongccc/sglang-omni:dev /bin/zsh
```

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

See [Installation](../get_started/installation.md) for Docker digests and source installs.

<a id="apple-silicon-mlx"></a>
### Apple Silicon (MLX)

Qwen3-Omni also runs on macOS Apple Silicon (`arm64`), reusing the Apple
platform integration and `SGLANG_USE_MLX` backend switch
introduced for [Qwen3-ASR](qwen3_asr.md#apple-silicon-mlx). Install with
[`install.sh`](../../install.sh) (see
[Installation](../get_started/installation.md#macos-apple-silicon)) or build
reuse an existing compatible MLX virtualenv as described there. For a new
environment, select `SGLANG_OMNI_EXTRAS=mlx`; the installer uses SGLang's
`all_mps` extra and the root project manifest. Do not replace `pyproject.toml`
with `pyproject_apple.toml`.

Select native MLX with `SGLANG_USE_MLX=1` and launch the downloaded pinned
`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` directory described below.
No extra artifact generation or copy step is required. This implementation
does not add a Qwen3-Omni Torch-MPS backend.

Launch commands and the full Apple runtime profile (one Metal device, greedy
generation, eager execution, SHM transport, no CUDA-only features) are in the
[Qwen3-Omni usage guide](../basic_usage/qwen3_omni.md#apple-silicon-mlx).

Before attempting a production-size checkpoint, check available unified memory
and measure peak usage with representative requests. Checkpoint size alone
does not account for activations, caches, and multimodal inputs. Serving smoke
tests do not establish production-size memory safety or semantic correctness.

#### Public 4-bit checkpoint

The public
[`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit)
checkpoint can be used for Apple MLX serving. Pin revision
`93b3cbddd65ed4babff8f22fba491cdba7a21778` so the commands and tensor layout
remain reproducible.

Other MLX-compatible 4-bit layouts may also load when they satisfy the Apple
checkpoint validator, but the pinned mlx-community checkpoint above is the
reference checkpoint for this implementation. Validator-accepted examples include
component-local thinker/talker shards and the root-namespaced MLX-VLM layout.

Current ownership for the Apple MLX path is:

| Runtime | Components |
|---|---|
| Native MLX | vision, audio, thinker, talker, code predictor, code2wav |
| CPU | preprocessing, token decoding |

Native MLX means those model components stay in MLX for the Apple launch, but
it does **not** imply radix cache, multi-request batching, or CUDA-oriented
optimizations.

The native implementation requires `mlx>=0.32.2` and `mlx-lm>=0.31.2`,
without an `mlx-vlm` dependency. It reuses MLX's fused SDPA (including vision
head dimension 72), normalization and standard RoPE kernels, plus MLX-LM's
KV caches and routed expert layers. Three-axis M-RoPE uses `mx.compile` to
reuse its graph and fuse elementwise operations while retaining external
multimodal positions. Vision attention still bounds query chunks to guard
against quadratic score buffers when a shape takes the unfused path.

Set the repository and environment paths:

```bash
export REPO="/path/to/sglang-omni"
export PY="${PY:-$REPO/.venv-apple/bin/python}"
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct-4bit-93b3cbdd"
export MODEL_REVISION="93b3cbddd65ed4babff8f22fba491cdba7a21778"
cd "$REPO"
export PYTHONPATH="$REPO${SGLANG_SOURCE:+:$SGLANG_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_OMNI_VIDEO_READER=pyav
```

Use the opt-in PyAV reader for this Apple environment: TorchCodec may be
installed but unable to load FFmpeg dylibs, and torchvision 0.28 no longer
provides `read_video`. PyAV decoding retains Qwen's frame sampling and resizing;
the GPU default is unchanged.

Set `PY` to an existing compatible virtualenv's Python before the block above
to reuse it without changing installed packages. If its editable SGLang source
is too old, set `SGLANG_SOURCE` to a compatible checkout's `python/` directory
before that block; see the
[existing-environment instructions](../get_started/installation.md#reuse-an-existing-mlx-environment-without-modifying-it).
If the pinned checkpoint is already downloaded, keep `MODEL_DIR` pointing at
its complete directory and skip the download:

```bash
export HF_HUB_OFFLINE=1
```

A Hub cache snapshot containing only configuration is not sufficient; all
safetensor shards and processor assets must be present. Otherwise download the
exact checkpoint (leave `HF_HUB_OFFLINE` unset):

```bash
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.path.expanduser(os.environ["MODEL_DIR"]),
)
print(path)
PY
```

#### Strict pinned-checkpoint recipe

For an opinionated community-checkpoint recipe, pin **both the repository and
the immutable revision**, verify the complete snapshot, and only then launch
offline. A YAML `model_path`, directory name, or `quantization.bits == 4` check
does not establish that the files match the intended repository.

With `PY`, `REPO`, and `PYTHONPATH` configured above, this direct command sequence
downloads only the selected repository into its revision-specific Hub snapshot.
It fails before serving if downloading, checksums, completeness, or the
extra-file check fails:

```bash
(
  set -eu
  HF="$(dirname "$PY")/hf"
  MODEL_ID="mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
  MODEL_REVISION="93b3cbddd65ed4babff8f22fba491cdba7a21778"
  MODEL_DIR="$(HF_HUB_OFFLINE=0 "$HF" download "$MODEL_ID" \
    --revision "$MODEL_REVISION" --quiet)"

  HF_HUB_OFFLINE=0 "$HF" cache verify "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR" \
    --fail-on-missing-files \
    --fail-on-extra-files

  HF_HUB_OFFLINE=1 SGLANG_USE_MLX=1 SGLANG_OMNI_VIDEO_READER=pyav \
    "$PY" -m sglang_omni.cli serve \
    --model-path "$MODEL_DIR" --host 127.0.0.1 --port 8008
)
```

These flags are available in `huggingface_hub==1.10.1`. Verification reads online
repository metadata and hashes the local files; it does not download model
weights. Downloading the full snapshot without `--include` filters prevents
accidentally omitting tokenizer, processor, talker, or vocoder assets. Once
verified, the server uses that local snapshot rather than resolving mutable
`main`. Keep the snapshot unmodified between verification and serving.

To reuse an existing complete directory without downloading another copy,
skip the `hf download` assignment and set `MODEL_DIR` to that directory before
the same strict verification and launch commands. An incomplete config-only
snapshot must fail the missing-file gate. A directory with leftover partial
downloads or unrelated files must fail the extra-file gate: use a clean,
dedicated snapshot rather than silently disabling the gate or deleting files
from a shared model directory.

This recipe establishes a **content match** to the pinned community revision,
not proof of the historical download source. It is an opt-in serving recipe,
not a global restriction in the MLX backend. The normal Apple validator still
checks architecture and required component metadata for compatible local or
custom-converted checkpoints; it is not a checksum or repository-provenance
validator. No new recipe loader or runtime repository allowlist is required.

#### Serve a previously downloaded directory

The native Apple MLX launch points `sgl-omni serve`
directly at that downloaded directory with `SGLANG_USE_MLX=1`. Keep the CLI in
the foreground for normal operation so logs and Ctrl-C remain attached to the
supervising terminal. There is no checkpoint-preparation step or dense code2wav
sidecar; code2wav also executes in MLX:

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8008
```

For MLX text-only serving, use the same CLI with `--text-only`:

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --text-only \
  --host 127.0.0.1 \
  --port 8008
```

Poll readiness from a second terminal:

```bash
until curl -fsS http://127.0.0.1:8008/v1/models >/dev/null; do
  sleep 2
done
```

Ctrl-C in the foreground terminal, or the external supervisor managing that
foreground process, terminates the canonical launcher.

For automated Bash smoke-test scripts only, start the same command in the
background and capture its exact process id. Use a ten-minute readiness deadline
(each HTTP probe is bounded to five seconds), fail early if the child exits, and
show the server log tail on readiness failure. The traps stop only that child,
including when readiness or a smoke request fails:

```bash
set -e

SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8008 >qwen3-omni-apple.log 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DEADLINE=$((SECONDS + 600))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Qwen3-Omni exited before readiness; server log:" >&2
    tail -n 100 qwen3-omni-apple.log >&2
    exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Timed out waiting for /v1/models after 600 seconds; server log:" >&2
    tail -n 100 qwen3-omni-apple.log >&2
    exit 1
  fi
  if curl -fsS --max-time 5 http://127.0.0.1:8008/v1/models >/dev/null; then
    break
  fi
  sleep 2
done

# Run smoke requests here.

kill -TERM "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT INT TERM
```

Apple scheduler restrictions remain explicit on the native MLX path:

- `tp_size=1`
- one resident request (`max_running_requests=1`)
- greedy generation only
- radix disabled
- overlap disabled
- mixed/chunked prefill disabled
- CUDA graphs disabled
- logprobs unsupported
- no partial talker start

Send a text request:

```bash
curl -fsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text"],
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1
  }' | tee text-response.json
```

Keep the foreground native MLX speech server running for the following text and
audio examples.

Send a non-streamed text-and-audio request:

```bash
curl -fsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"voice": "chelsie", "format": "wav"},
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1,
    "talker_temperature": 0,
    "talker_top_p": 1,
    "talker_top_k": -1,
    "talker_max_new_tokens": 128
  }' | tee speech-response.json
```

Decode and inspect the returned WAV:

```bash
"$PY" - <<'PY'
import base64
import json
import soundfile as sf

response = json.load(open("speech-response.json", encoding="utf-8"))
message = response["choices"][0]["message"]
open("speech-response.wav", "wb").write(base64.b64decode(message["audio"]["data"]))
audio, rate = sf.read("speech-response.wav")
print({"text": message["content"], "samples": len(audio), "sample_rate": rate})
PY
file speech-response.wav
```

Send the same request over SSE:

```bash
curl -NfsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"voice": "chelsie", "format": "wav"},
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1,
    "talker_temperature": 0,
    "talker_top_p": 1,
    "talker_top_k": -1,
    "talker_max_new_tokens": 128,
    "stream": true
  }' | tee speech-response.sse
grep -F 'data: [DONE]' speech-response.sse
```

#### Validation scope

Image and audio requests must use top-level `images` and `audios` arrays
(paths, URLs, or data URLs). Keep `messages[].content` as a string. Inline
OpenAI `image_url` and `input_audio` content blocks are not supported by this
Qwen preprocessor and can be stringified rather than decoded as media.
See the [media request format](../basic_usage/qwen3_omni.md#image-and-audio-requests).

The requests above are manual serving smoke checks, not a production
qualification suite. Save request/response JSON and generated audio when
evaluating a checkpoint. Check text against the input facts, listen to speech,
and measure memory across repeated text, image, audio, and video requests.
A valid WAV or completed SSE stream alone does not establish semantic quality.
This port does not include the previously documented Apple CI/semantic matrix;
no pass count or production-quality guarantee is claimed here.

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml`
(or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

| Mode | Topology | Thinker TP | Precision | Status |
|---|---|---|---|---|
| Thinker-only | — | — | BF16 | ✅ |
| Thinker-only | — | — | FP8 | ✅ |
| Thinker-only | — | — | AutoRound INT4 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Colocated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Colocated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |

## Input / Output Modalities

All input modality combinations work with both text-only and speech servers.
`modalities: ["text", "audio"]` requires a **speech-mode server** (omit `--text-only`).

| Input | Output | Speech server | Minimal request body | Notes |
|---|---|---|---|---|
| Text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]}` | — |
| Image + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]}` | — |
| Audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from image |
| Video + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]}` | — |
| Video + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Video | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from video |
| Text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]}` | — |
| Image + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from image |
| Video + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Video + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Video | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from video |

### Sampling Parameters

Standard sampling parameters apply to the thinker stage. When `modalities` includes `"audio"`, the additional talker-specific parameters below control the speech generation independently.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `1.0` | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker; OpenAI-compatible alias for `max_tokens` |
| `stop` | str \| list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | Both |
| `audio` | dict | `null` | Speech response format config, e.g. `{"format": "wav"}` |
| `talker_temperature` | float | `0.9` | Talker (audio output only) |
| `talker_top_p` | float | `1.0` | Talker (audio output only) |
| `talker_top_k` | int | `50` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `1.05` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `4096` | Talker (audio output only) |
| `stage_sampling` | dict | `null` | Per-stage sampling override |
| `stage_params` | dict | `null` | Per-stage non-sampling params |
| `video_fps` | float | `null` | Frame sampling rate for video input (uses server default if unset) |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all video frames |

### Known Limitations

- **`modalities: ["text", "audio"]` has no effect on a text-only server.** No error is raised — the response simply contains no audio. Use a speech-mode server (without `--text-only`) to get audio output.
- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Leaving a text query in `content` alongside audio causes the model to process both, which is usually not what you want.
- **Colocated topology does not support `--thinker.tp_size 2`.** The server raises a `ValueError` at startup ("Qwen Phase 1 colocation does not support thinker TP"). Use disaggregated topology for TP=2.
- **Requests that exceed the model's context length are rejected with an error.** The preprocessor raises a `ValueError` when the prompt token count alone meets or exceeds `max_seq_len`, or when `prompt tokens + max_new_tokens ≥ max_seq_len`. Reduce input length or lower `max_tokens` to stay within the limit.
- **Apple Silicon MLX keeps a restricted scheduler profile.** `tp_size=1`, `max_running_requests=1`, greedy generation only, radix disabled, overlap disabled, mixed/chunked prefill disabled, CUDA graphs disabled, logprobs unsupported, and no partial talker start. Native MLX does not imply radix cache, multi-request batching, or CUDA-oriented optimizations. MLX errors do not trigger a fallback to another backend. See [Apple Silicon (MLX)](#apple-silicon-mlx) above.
