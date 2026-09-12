<!-- Thank you for your contribution! -->

## Motivation

Add opt-in native MLX inference for Qwen3-Omni on Apple Silicon while reusing
the existing SGLang Omni pipeline and OpenAI-compatible API.

Reference checkpoint:
[`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit)
at revision `93b3cbddd65ed4babff8f22fba491cdba7a21778`.

## Modifications

- Add native MLX implementations for Qwen3-Omni model components, including
  vision, audio, thinker, talker, code predictor, and code2wav.
- Integrate MLX workers with the existing seven-stage pipeline and shared-memory
  transport.
- Add Apple runtime validation, conservative MLX scheduling, PyAV video input,
  examples, and unit coverage.
- Enable the backend only with `SGLANG_USE_MLX=1`; no Torch-MPS fallback is used.

The current MLX profile uses one Metal device, one running request, and greedy
decoding. Unsupported sampling and logprob settings are rejected.

## Related Issues

N/A.

## Accuracy Test

Added focused unit coverage for MLX runtime selection, model loading, multimodal
feature handling, talker prefill, shared-memory transport, and video decoding.

## Benchmark & Profiling

Measured on an M4 Pro with 48 GiB unified memory using the pinned 4-bit checkpoint:

- 65/65 sequential API requests passed across text, image, audio, video, speech,
  streaming, and cancellation cases.
- 128-token text generation: 57.5 tokens/s end-to-end.
- Median TTFT: 110.9 ms for short text and 195.2 ms for the 128-token case.

## Checklist

- [x] Format code according to pre-commit.
- [x] Add unit tests.
- [x] Update documentation and examples.
- [x] Provide benchmark results.

## CI

CI requires a maintainer to add the `run-ci` label. Draft PRs are skipped.
