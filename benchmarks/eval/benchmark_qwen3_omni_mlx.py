# SPDX-License-Identifier: Apache-2.0
"""Small, sequential Qwen3-Omni API benchmark; not a quality/capacity evaluation."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import platform
import subprocess
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import av
import httpx
import numpy as np
from PIL import Image

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.runtime_metrics import collect_benchmark_provenance

MODEL = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
REVISION = "93b3cbddd65ed4babff8f22fba491cdba7a21778"


def payload(prompt: str, **kwargs) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
        **kwargs,
    }
    if "audio" in body.get("modalities", []):
        body.setdefault("talker_max_new_tokens", 256)
    return body


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getframerate() != 24000 or wav.getnchannels() != 1:
            raise ValueError("Expected mono 24 kHz speech output")
        frames = wav.getnframes()
        decoded = wav.readframes(frames)
        if not frames or len(decoded) != frames * wav.getsampwidth():
            raise ValueError("Empty or truncated WAV")
        if wav.getsampwidth() != 2:
            raise ValueError("Expected signed 16-bit PCM")
        if not np.any(np.frombuffer(decoded, dtype="<i2")):
            raise ValueError("Silent WAV")
        return frames / wav.getframerate()


def send_request(
    client: httpx.Client, body: dict, *, expected: str = "", cancel: bool = False
) -> tuple[RequestResult, list[bytes]]:
    result = RequestResult()
    audio: list[bytes] = []
    started = time.perf_counter()
    usage: dict = {}
    text = ""
    done = False
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        if body.get("stream"):
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"SSE error: {event['error']}")
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        if result.text_ttft_s is None:
                            result.text_ttft_s = time.perf_counter() - started
                        text += content
                    if delta.get("audio", {}).get("data"):
                        chunk = base64.b64decode(delta["audio"]["data"], validate=True)
                        duration = wav_duration(chunk)
                        arrived = time.perf_counter() - started
                        if result.audio_ttfp_s is None:
                            result.audio_ttfp_s = arrived
                            result.first_audio_payload_bytes = len(chunk)
                        result.chunk_audio_duration_s.append(duration)
                        audio.append(chunk)
                    if cancel and content:
                        if choice.get("finish_reason") is not None:
                            raise ValueError("Stream completed before cancellation")
                        result.text = text
                        result.latency_s = time.perf_counter() - started
                        result.is_success = True
                        return result, audio
            if not done:
                raise ValueError("Stream ended without [DONE]")
            if cancel:
                raise ValueError("Stream completed before cancellation")
        else:
            response.read()
            event = response.json()
            message = event["choices"][0]["message"]
            text = message.get("content") or ""
            usage = event.get("usage") or {}
            if (message.get("audio") or {}).get("data"):
                audio.append(base64.b64decode(message["audio"]["data"], validate=True))
    result.latency_s = time.perf_counter() - started
    result.engine_time_s = result.latency_s
    result.text = text
    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)
    if not text.strip() or expected.lower() not in text.lower():
        raise ValueError(f"Expected {expected!r}; received {text!r}")
    if body.get("max_tokens", 32) >= 128 and result.completion_tokens < 32:
        raise ValueError("Long-generation case produced fewer than 32 tokens")
    if "audio" in body.get("modalities", []):
        if not audio:
            raise ValueError("Requested speech, received no audio")
        result.audio_duration_s = sum(wav_duration(chunk) for chunk in audio)
        if result.audio_duration_s > 5:
            raise ValueError(
                f"Runaway speech for the short fixture: {result.audio_duration_s:.3f}s "
                f"of audio for {text!r}; expected at most 5s"
            )
        result.rtf = result.latency_s / result.audio_duration_s
        result.audio_chunk_count = len(audio)
    result.is_success = True
    return result, audio


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def make_fixtures(client: httpx.Client, directory: Path) -> dict:
    image = Image.new("RGB", (224, 224), "red")
    image.save(directory / "red.png")
    with av.open(str(directory / "red.mp4"), mode="w") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width = stream.height = 224
        stream.pix_fmt = "yuv420p"
        for _ in range(8):
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reference, waves = send_request(
        client,
        payload(
            "Say exactly: Hello.", modalities=["text", "audio"], audio={"format": "wav"}
        ),
        expected="hello",
    )
    (directory / "hello.wav").write_bytes(waves[0])
    return {
        "reference_generation": asdict(reference),
        "image": {"size": [224, 224], "color": "red"},
        "video": {"size": [224, 224], "frames": 8, "fps": 4, "duration_s": 2},
        "audio": {"sample_rate": 24000, "duration_s": wav_duration(waves[0])},
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("red.png", "red.mp4", "hello.wav")
        },
    }


def build_cases(directory: Path) -> list[tuple[str, dict, str]]:
    cases = [
        ("health", {}, "healthy"),
        ("text", payload("What is 2 + 2? Reply with only the number."), "4"),
        (
            "text_sse",
            payload(
                "What is the capital of France? Answer with one word.", stream=True
            ),
            "paris",
        ),
        (
            "text_long_sse",
            payload(
                "Explain how 4-bit model quantization saves memory compared with 16-bit "
                "weights. Discuss memory use, accuracy, and compute cost in detail.",
                stream=True,
                max_tokens=128,
            ),
            "",
        ),
        (
            "text_to_speech",
            payload(
                "Say exactly: Hello.",
                modalities=["text", "audio"],
                audio={"format": "wav"},
            ),
            "hello",
        ),
        (
            "text_to_speech_sse",
            payload(
                "Say exactly: Hello.",
                modalities=["text", "audio"],
                audio={"format": "wav"},
                stream=True,
            ),
            "hello",
        ),
    ]
    media = [
        (
            "image",
            "images",
            data_url(directory / "red.png", "image/png"),
            "What is the dominant color of this image? Reply with one word.",
            "red",
        ),
        (
            "audio",
            "audios",
            data_url(directory / "hello.wav", "audio/wav"),
            "Transcribe the speech exactly.",
            "hello",
        ),
        (
            "video",
            "videos",
            data_url(directory / "red.mp4", "video/mp4"),
            "What is the dominant color in this video? Reply with one word.",
            "red",
        ),
    ]
    for name, field, url, prompt, expected in media:
        for speech in (False, True):
            options = {field: [url]}
            if speech:
                options.update(modalities=["text", "audio"], audio={"format": "wav"})
            cases.append(
                (
                    f"{name}_to_{'speech' if speech else 'text'}",
                    payload(prompt, **options),
                    expected,
                )
            )
    cases.append(
        ("cancel_recovery", payload("What is 3 + 5? Reply with only the number."), "8")
    )
    return cases


def benchmark_case(
    client,
    name,
    body,
    expected,
    runs,
    on_sample: Callable[[dict], None] | None = None,
) -> dict:
    records = []
    for index in range(runs + 1):
        started = time.perf_counter()
        cancelled_after = None
        try:
            if name == "health":
                response = client.get("/health")
                response.raise_for_status()
                if response.json().get("status") != "healthy":
                    raise ValueError(f"Unhealthy server: {response.text}")
                result = RequestResult(
                    is_success=True, latency_s=time.perf_counter() - started
                )
            else:
                if name == "cancel_recovery":
                    cancelled, _ = send_request(
                        client,
                        payload(
                            "Write a detailed tutorial on programming, with many examples.",
                            max_tokens=512,
                            stream=True,
                        ),
                        cancel=True,
                    )
                    cancelled_after = cancelled.latency_s
                result, _ = send_request(client, body, expected=expected)
        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            IndexError,
            wave.Error,
            binascii.Error,
        ) as exc:
            result = RequestResult(
                is_success=False,
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.request_id = f"{name}-{index}"
        record = asdict(result)
        record["cancel_after_s"] = cancelled_after
        record["iteration_wall_s"] = time.perf_counter() - started
        records.append(record)
        if on_sample is not None:
            on_sample({"case": name, "warmup": index == 0, **record})
        print(
            f"{result.request_id}: {'OK' if result.is_success else result.error} "
            f"{result.latency_s:.3f}s {result.text[:80]!r}",
            flush=True,
        )
    fields = RequestResult.__dataclass_fields__
    measured = [
        RequestResult(**{k: v for k, v in row.items() if k in fields})
        for row in records[1:]
    ]
    return {
        "request": {
            k: ("<fixture data URL>" if k in ("images", "audios", "videos") else v)
            for k, v in body.items()
        },
        "expected_substring": expected,
        "warmup": records[0],
        "samples": records[1:],
        "summary": compute_speed_metrics(
            measured, wall_clock_s=sum(row["iteration_wall_s"] for row in records[1:])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cases", nargs="+", help="Optional subset of named cases")
    parser.add_argument(
        "--timeout", type=float, default=180, help="HTTP per-read timeout in seconds"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = collect_benchmark_provenance(
        model_id=MODEL,
        model_revision=REVISION,
        dataset_id="synthetic-api-smoke",
        dataset_revision=None,
        launch_command=None,
        server_config={
            "model_path": args.model_path,
            "SGLANG_USE_MLX": "1",
            "SGLANG_OMNI_VIDEO_READER": "pyav",
            "concurrency": 1,
        },
    )
    provenance["sglang_source"] = {
        "path": str(args.sglang_source),
        "commit": subprocess.check_output(
            ["git", "-C", str(args.sglang_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "-C", str(args.sglang_source), "status", "--porcelain"], text=True
        ),
    }
    if platform.system() == "Darwin":
        provenance["apple_host"] = {
            key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            for key in ("hw.memsize", "hw.model", "machdep.cpu.brand_string")
        }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "methodology": {
            "warmups_per_case": 1,
            "measured_runs_per_case": args.runs,
            "concurrency": 1,
            "base_url": args.base_url,
            "cache": "Repeated identical inputs: warm encoder caches; radix disabled.",
            "timing": "HTTP client end-to-end; stream TTFT at first nonempty text delta.",
            "speech_budget": "256 talker tokens; short-fixture audio over 5s is a failure.",
            "cancellation": "Close stream at first text delta; immediately time a recovery request.",
            "limitations": "Synthetic functional latency benchmark, not quality or capacity. "
            "Fixture generation warms speech before timing. No cold restart per case. "
            "Declared server model/source are supplied by the operator, not attested.",
        },
        "cases": {},
        "attempts": [],
    }
    output = args.output_dir / "results.json"

    def save_attempt(attempt: dict) -> None:
        report["attempts"].append(attempt)
        output.write_text(json.dumps(report, indent=2) + "\n")

    with httpx.Client(
        base_url=args.base_url, timeout=args.timeout, trust_env=False
    ) as client:
        client.get("/health").raise_for_status()
        report["fixtures"] = make_fixtures(client, args.output_dir)
        cases = build_cases(args.output_dir)
        if args.cases:
            unknown = set(args.cases) - {name for name, _, _ in cases}
            if unknown:
                parser.error(f"Unknown cases: {sorted(unknown)}")
            cases = [case for case in cases if case[0] in args.cases]
        report["methodology"]["selected_cases"] = [name for name, _, _ in cases]
        report["methodology"]["per_read_timeout_s"] = args.timeout
        for name, body, expected in cases:
            report["cases"][name] = benchmark_case(
                client, name, body, expected, args.runs, on_sample=save_attempt
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
        response = client.get("/health")
        response.raise_for_status()
        report["final_health"] = response.json()
        output.write_text(json.dumps(report, indent=2) + "\n")
    failures = sum(
        case["summary"]["failed_requests"] + (not case["warmup"]["is_success"])
        for case in report["cases"].values()
    )
    if report["final_health"].get("status") != "healthy":
        failures += 1
    if report["final_health"]["pending_completions"] != 0:
        failures += 1
    print(f"Saved {output}; failures={failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
