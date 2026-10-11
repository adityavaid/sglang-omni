# Router Mock Fleet Implementation Plan

**Goal:** Run model-free modality workers behind the real Rust router and report routing/protocol checks and synthetic performance.

**Architecture:** A Cargo example with configuration, worker, and runner modules. The existing router is an independently launched binary. TOML profiles are shared data, but request validation does not call the router classifier.

**Tech Stack:** Existing Tokio, Axum, reqwest, serde, TOML and tungstenite dependencies; example-only multipart/serialization/process features.

## Constraints

- No models, GPUs, external services, production router changes, or commits.
- Loopback-only listeners, bounded payloads/work, child cleanup, explicit uncovered features.
- Preserve existing worktree changes.

## Tasks

- [x] Configuration: first run `cargo test --example mock_fleet --locked` with failing configuration tests. Implement `Fleet::parse` and `Fleet::router_config`; rerun tests. Preserve profiles, remove mock tables, reject unsafe listener/invalid knobs.
- [x] Worker: add independent profile matching tests for wrong model/modality/format and correlation. Implement HTTP/SSE/PCM/WAV/multipart/voice/WebSocket responses and bounded observations. Rerun example tests.
- [x] Supervisor and runner: launch child workers on ephemeral ports, consume readiness announcements, validate and launch the router, wait for readiness, run profile probes and negative requests. Test cleanup and correct reporting with real sockets.
- [x] Benchmarks: bounded concurrency and request counts, direct and routed runs, byte counts, latency percentiles and first-body-byte times. Report errors and eligible-worker checks; retain failing outcomes.
- [x] Documentation and example: heterogeneous fleet, usage, limits and coverage. Run formatting, example tests, Clippy and an end-to-end example run. Inspect diagnostics and report remaining gaps.

## Verification (2026-09-20)

- 17 focused example tests pass, including four red/green review regressions.
- Example Clippy with `-D warnings` and Rust 1.90 compile check pass.
- Sample fleet: 73 HTTP workloads and 156 checks pass at concurrency 4.
- Serve startup and SIGTERM cleanup pass; no owned processes remain.
- Full locked Rust test command reaches the existing media reset failure:
	`oversized_heterogeneous_request_is_rejected_without_upstream_send` receives
	`Connection reset by peer`. Production router code was not modified.
- Per-core profiling, full protocol fidelity, active graceful drain, raw framing,
	and other untested capabilities are explicitly listed in the report/README.