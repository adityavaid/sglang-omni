# Router Mock Fleet

Approved direction: CPU-only Rust mock worker processes, the real Rust router,
and a configurable verification/load runner. No model imports or downloads.

## Configuration and isolation

Use the current router TOML schema with optional per-worker `mock` tables.
Keep correlated service profiles intact. Generate a router-only manifest with
actual worker endpoints; validate it using the router's `--check-config`.
Bind workers and the router to loopback. Spawn workers as separate processes;
bound startup, requests, and shutdown, and reap every owned child on exit.

## Protocol and verification

Workers independently validate model, modality, task, format and stream mode.
Responses carry mock identity headers and protocol-appropriate synthetic data.
Use deterministic JSON/SSE, real synthetic PCM/WAV, multipart ASR, worker-local
voice state, and bounded speech/realtime WebSocket sessions. Unsupported mock
formats must fail explicitly rather than return invalid encoded audio.

`serve` leaves the fleet available for manual clients. `run` executes supported
profile probes, independent rejection tests and bounded concurrent workloads,
then writes a JSON report including direct-worker measurements and router
metrics. Known router failures remain failures. Unexercised features are listed
as not tested; missing router capabilities are listed as unsupported.

Synthetic throughput is not real-model performance or the DP3/MPS exit proof.
Client first-body-byte latency is not router-internal TTFP or first audible audio.
No production router behavior is changed by this harness.