# Native Qwen3-Omni

The C++20 MLX runtime loads the complete indexed Qwen3-Omni-30B-A3B-Instruct affine 4-bit checkpoint: thinker and talker MoE networks, the residual predictor with fifteen codebooks, vision and audio towers, and code2wav. The server and CLI share one inference engine. Python, Swift, shell commands, and external decoder processes are not used for inference.

## Build and dependencies

Use native/CMakeLists.txt with the existing MLX **0.32.3** wheel's C++ headers, CMake package, Metal library, and dylibs. Full Omni is explicitly opt-in: **SGLANG_BUILD_QWEN3_OMNI defaults to OFF**, preserving existing ASR targets and installation without FFmpeg or libcurl discovery. CMake, Ninja, the existing pinned nlohmann/json, PCRE2, and CivetWeb dependencies are unchanged. The tokenizer additionally accepts merge pairs from tokenizer.json when merges.txt is absent.

Omni also requires separately supplied **FFmpeg 7** development headers and dynamic libavcodec 61, libavformat 61, and libavutil 59 libraries, plus the macOS Foundation, AVFoundation, CoreGraphics, ImageIO, CoreMedia, CoreVideo, and system libcurl facilities. CMake accepts OMNI_FFMPEG_INCLUDE_DIR, OMNI_AVCODEC, OMNI_AVFORMAT, and OMNI_AVUTIL overrides; its Homebrew discovery hint is ffmpeg@7. No FFmpeg source or binaries are bundled.

```bash
cmake -S sglang_omni_mlx/native -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DMLX_DIR="$MLX_DIR" \
  -DSGLANG_BUILD_QWEN3_OMNI=ON
cmake --build build/native --target \
  qwen3_omni_generate qwen3_omni_server qwen3_omni_components
ctest --test-dir build/native -R '^qwen3_omni_components$' --output-on-failure
```

For the bundle script, use `SGLANG_BUILD_QWEN3_OMNI=ON sglang_omni_mlx/native/scripts/build_runtime.sh OUTPUT WORK`. Its default remains ASR-only. Mac CI explicitly enables Omni and requires a reviewed, already provisioned Homebrew ffmpeg@7 installation; missing headers or compatible dylibs fail the job rather than skipping Omni. `CMAKE_PREFIX_PATH` may point to a reviewed FFmpeg installation.

With the option enabled, installation adds both Omni executables to the existing model binaries and MLX runtime files. **It is not a self-contained Omni bundle:** FFmpeg and its transitive libraries must remain available at their linked locations. Do not copy local Homebrew libraries into a redistributable package without resolving the applicable licenses.

The locally qualified FFmpeg is Homebrew **7.1.5**, compiled with --enable-gpl and --enable-version3, reporting **GPL version 3 or later**. Its dylibs declare **macOS 26.0** minimum deployment; the validation executables were compiled for macOS 26.7. This does not qualify older macOS hosts. An LGPL-configured FFmpeg 7 build with an appropriate deployment target remains unqualified. See NOTICE.

## Interfaces

```bash
build/native/qwen3_omni_generate --version
build/native/qwen3_omni_generate --model-path "$MODEL" \
  --request request.json --allowed-local-media-path "$MEDIA_ROOT"
build/native/qwen3_omni_server --model-path "$MODEL" \
  --model-name qwen3-omni-mlx --host 127.0.0.1 --port 8000 \
  --allowed-local-media-path "$MEDIA_ROOT"
```

The CLI accepts repeated --request flags, loads the checkpoint once, and emits one final OpenAI completion JSON line per request, in command-line order. Single-request behavior is identical. Each request has fresh KV caches and request-local decoding state; an explicit seed resets thinker and talker RNG independently. CLI stream=true still produces one final JSON result. Logs go to stderr.

Both binaries accept --version without model loading. The JSON includes the **linked** mlx::core::version(), FFmpeg version, and FFmpeg codec license, not a caller-supplied reference version.

The server implements GET /health, GET /v1/models, and POST /v1/chat/completions. --allowed-local-media-path may be repeated. Local media must resolve to a regular file below an allowed directory; URLs and inline base64 data URLs are also accepted. HTTP downloads are bounded to 32 MiB and 30 seconds and restrict redirects to HTTP(S).

The existing supervisor protocol is supported:

```bash
build/native/qwen3_omni_server --supervised --model-kind qwen3_omni \
  --model-directory "$MODEL"
```

Without an explicit port, supervised mode chooses an ephemeral loopback port. The first stdout line is JSON containing event=ready, host, port, server_pid, and model_name. A stdin line containing {"command":"shutdown"} or stdin EOF requests shutdown. A different --model-kind is rejected before model loading. Standalone mode logs its listening endpoint to stderr and accepts SIGINT/SIGTERM.

## Requests and lifecycle

```json
{"model":"qwen3-omni-mlx","messages":[{"role":"user","content":"Say hello briefly."}],"modalities":["text","audio"],"audio":{"voice":"chelsie","format":"wav"},"temperature":0,"max_tokens":64,"talker_temperature":0,"talker_max_new_tokens":128,"stream":false}
```

Supported content includes text/input_text, image_url/input_image, audio_url/input_audio, and video_url, plus top-level images, videos, and audios. Ordered media parts, video with a separate audio source, and use_audio_in_video=true are supported. Images use native CoreGraphics preprocessing; video samples native presentation timestamps through AVFoundation; audio uses in-memory, protocol-disabled FFmpeg decoding followed by native mono resampling to 16 kHz.

Inline/downloaded video is materialized in the system temporary directory (overridable with `TMPDIR`), not the working directory. Files are created exclusively with mode 0600 and removed on success, error, and cancellation. The configured temporary directory must already exist and be writable.

Output modalities are text or text+audio. Voices are chelsie, ethan, and aiden. Speech is actual mono **24 kHz PCM16 WAV**, not a placeholder. SSE audio chunks are independently decodable WAV files; the final JSON WAV concatenates precisely their PCM payloads. Streaming code2wav retains at most 25 context frames and removes their overlapping samples before emission.

Requests are limited to 64 MiB JSON, 128 messages, 24 media sources, 8,192 expanded prompt tokens, 4,096 thinker tokens, and 4,096 speech frames. Audio is limited to 120 seconds. Images are limited to 16 megapixels before decoding. Video has duration, timestamp, frame-count, and pixel-budget bounds. Unknown fields, wrong models, malformed types, and invalid media return 400; a live concurrent generation returns 429; stopping returns 503.

There is one model execution at a time. Disconnects are observed during preprocessing and model layers using the current process's own TCP socket, without accessing CivetWeb private structures. A follow-up may wait up to two seconds for an already disconnected request's GPU cleanup; it does not queue behind a live request. CivetWeb workers use 8 MiB stacks because CoreGraphics image conversion exceeded the upstream 100 KiB default in a reproduced native crash.

Thinker decoding submits one next-step graph before observing the current sampled token. Residual predictors fence the previous codebook only after submitting the current one. Next talker logits overlap WAV serialization. No speculative next sampling draw is made at EOS, and synchronization occurs before releasing request caches or accepting the next model execution.

## Qualification and limitations

The task report records exact local commands and artifacts. Validation includes the real full checkpoint, all twelve input/output combinations, JSON/SSE equality of text and PCM, malformed inputs, live busy responses, disconnect recovery, and nongreedy speech.

This is a preservation-first port of the native implementation, **not a claim of Python-reference parity or identical outputs across MLX versions**. The preserved Swift server links MLX 0.32.0, whereas this runtime requires 0.32.3. Audio text and video speech length differ between these backends. An isolated diagnostic using identical C++ model code with the reference backend reproduced Swift's audio text and video sample count; audio stages from decoded samples through first-token logits matched Swift bit-for-bit on that same backend. Production remains pinned to 0.32.3. These measured C++/Swift differences are distinct from the already documented Swift/Python differences, not silently classified as pre-existing Python drift. Full cross-runtime or per-chip golden equivalence remains unclaimed.

Apple media APIs retain synchronous compatibility calls deprecated in newer SDKs. These are native calls, not subprocesses. Older-host deployment, alternative FFmpeg builds, production capacity, performance improvements, and the full ASR corpus remain outside this qualification.
