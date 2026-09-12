# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import wave

import httpx
import pytest

from benchmarks.eval.benchmark_qwen3_omni_mlx import (
    benchmark_case,
    payload,
    send_request,
    wav_duration,
)


def wav_bytes(*, silent=False, frames=2400):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes((b"\0\0" if silent else b"\x10\0") * frames)
    return buffer.getvalue()


def client_for(response):
    return httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(lambda request: response)
    )


def sse(content, *, done=True, audio=None):
    delta = {"content": content}
    if audio:
        delta["audio"] = {"data": base64.b64encode(audio).decode()}
    event = {"choices": [{"delta": delta, "finish_reason": None}]}
    text = f"data: {json.dumps(event)}\n\n"
    if done:
        text += 'data: {"choices": [], "usage": {"completion_tokens": 2}}\n\n'
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text)


def test_wav_duration():
    assert wav_duration(wav_bytes()) == pytest.approx(0.1)


def test_silent_and_truncated_audio_rejected():
    with pytest.raises(ValueError, match="Silent"):
        wav_duration(wav_bytes(silent=True))
    with pytest.raises(ValueError, match="truncated"):
        wav_duration(wav_bytes()[:-2])


def test_sse_text_usage_and_ttft():
    with client_for(sse("Paris")) as client:
        result, audio = send_request(
            client, payload("capital", stream=True), expected="paris"
        )
    assert result.is_success
    assert result.completion_tokens == 2
    assert 0 <= result.text_ttft_s <= result.latency_s
    assert not audio


def test_partial_sse_fails_without_done():
    with client_for(sse("Paris", done=False)) as client:
        with pytest.raises(ValueError, match=r"without \[DONE\]"):
            send_request(client, payload("capital", stream=True))


def test_streaming_speech_duration_and_first_audio():
    with client_for(sse("Hello", audio=wav_bytes())) as client:
        result, audio = send_request(
            client, payload("hello", stream=True, modalities=["text", "audio"])
        )
    assert result.audio_duration_s == pytest.approx(0.1)
    assert result.audio_ttfp_s is not None
    assert result.audio_chunk_count == len(audio) == 1
    assert result.rtf == pytest.approx(result.latency_s / 0.1)


def test_missing_requested_speech_is_failure():
    response = httpx.Response(
        200, json={"choices": [{"message": {"content": "Hello"}}]}
    )
    with client_for(response) as client:
        with pytest.raises(ValueError, match="no audio"):
            send_request(client, payload("hello", modalities=["text", "audio"]))


def test_semantic_failure_is_counted_and_warmup_excluded():
    response = httpx.Response(200, json={"choices": [{"message": {"content": "Blue"}}]})
    with client_for(response) as client:
        case = benchmark_case(client, "image", payload("color"), "red", 2)
    assert not case["warmup"]["is_success"]
    assert len(case["samples"]) == 2
    assert case["summary"]["total_requests"] == 2
    assert case["summary"]["failed_requests"] == 2
    assert "Blue" in case["samples"][0]["error"]


def test_cancel_closes_after_first_text_without_done():
    with client_for(sse("First", done=False)) as client:
        result, _ = send_request(client, payload("long", stream=True), cancel=True)
    assert result.is_success and result.text == "First"


def test_long_case_rejects_missing_token_usage():
    with client_for(sse("Long enough words", done=False)) as client:
        with pytest.raises(ValueError):
            send_request(client, payload("long", max_tokens=128, stream=True))
    response = httpx.Response(
        200, json={"choices": [{"message": {"content": "Short"}}]}
    )
    with client_for(response) as client:
        with pytest.raises(ValueError, match="fewer than 32 tokens"):
            send_request(client, payload("long", max_tokens=128))


def test_http_error_is_recorded():
    with client_for(httpx.Response(500, text="backend error")) as client:
        case = benchmark_case(client, "text", payload("hello"), "", 1)
    assert case["summary"]["failed_requests"] == 1
    assert "HTTPStatusError" in case["samples"][0]["error"]


def test_attempts_persist_before_next_iteration():
    attempts = []
    with client_for(httpx.Response(503, text="unavailable")) as client:
        case = benchmark_case(
            client, "text", payload("hello"), "", 1, on_sample=attempts.append
        )
    assert [attempt["warmup"] for attempt in attempts] == [True, False]
    assert all(not attempt["is_success"] for attempt in attempts)
    assert attempts[1]["request_id"] == case["samples"][0]["request_id"]


def test_speech_budget_is_separate_from_text_budget():
    body = payload("Hello", modalities=["text", "audio"])
    assert body["max_tokens"] == 32
    assert body["talker_max_new_tokens"] == 256
    assert "talker_max_new_tokens" not in payload("Text only")


def test_runaway_short_fixture_speech_is_not_a_success():
    with client_for(sse("Hello", audio=wav_bytes(frames=24000 * 6))) as client:
        with pytest.raises(ValueError, match="Runaway speech"):
            send_request(
                client, payload("Hello", stream=True, modalities=["text", "audio"])
            )
