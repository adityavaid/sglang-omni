# SPDX-License-Identifier: Apache-2.0
"""Check that changed speech, incomplete cases and stale provenance fail the gate."""

from __future__ import annotations

import base64
import copy
import io
import wave
from pathlib import Path

import check_omni_golden
import pytest


@pytest.fixture
def golden() -> check_omni_golden.Golden:
    return {
        "model": "model",
        "checkpoint_sha256": "0" * 64,
        "reference": {"revision": "revision", "mlx_version": "0.32.3"},
        "files": {},
        "requests": {"text": {"messages": [{"role": "user", "content": "Hello"}]}},
        "devices": {},
    }


@pytest.fixture
def outputs(golden: check_omni_golden.Golden) -> check_omni_golden.Outputs:
    return {
        "model": "model",
        "device": "Apple test",
        "spec_sha256": check_omni_golden.spec_digest(golden),
        "checkpoint_sha256": "0" * 64,
        "runtime_sha256": "1" * 64,
        "mlx_version": "0.32.3",
        "results": {
            "text": {
                "text": "Hello",
                "completion_tokens": 1,
                "finish_reason": "stop",
                "audio_sha256": None,
                "audio_samples": 0,
            }
        },
    }


def test_unknown_chip_requires_explicit_import(
    golden: check_omni_golden.Golden, outputs: check_omni_golden.Outputs
) -> None:
    assert check_omni_golden.check(golden, outputs)
    check_omni_golden.import_outputs(golden, outputs)
    assert check_omni_golden.check(golden, outputs) == []


@pytest.mark.parametrize("mutation", ["text", "audio", "missing", "extra"])
def test_changed_or_incomplete_results_fail(
    golden: check_omni_golden.Golden, outputs: check_omni_golden.Outputs, mutation: str
) -> None:
    check_omni_golden.import_outputs(golden, outputs)
    changed = copy.deepcopy(outputs)
    if mutation == "text":
        changed["results"]["text"]["text"] = "different"
    elif mutation == "audio":
        changed["results"]["text"]["audio_sha256"] = "changed"
    elif mutation == "missing":
        changed["results"].pop("text")
    else:
        changed["results"]["extra"] = changed["results"]["text"]
    assert check_omni_golden.check(golden, changed)


@pytest.mark.parametrize(
    "field", ["model", "spec_sha256", "checkpoint_sha256", "mlx_version"]
)
def test_import_rejects_wrong_provenance(
    golden: check_omni_golden.Golden, outputs: check_omni_golden.Outputs, field: str
) -> None:
    outputs[field] = "wrong"
    with pytest.raises(ValueError):
        check_omni_golden.import_outputs(golden, outputs)
    assert golden["devices"] == {}


def test_request_change_invalidates_recorded_outputs(
    golden: check_omni_golden.Golden, outputs: check_omni_golden.Outputs
) -> None:
    check_omni_golden.import_outputs(golden, outputs)
    golden["requests"]["text"]["max_tokens"] = 12
    assert check_omni_golden.check(golden, outputs)


def test_import_copies_results_and_rejects_empty(
    golden: check_omni_golden.Golden, outputs: check_omni_golden.Outputs
) -> None:
    check_omni_golden.import_outputs(golden, outputs)
    outputs["results"].clear()
    assert golden["devices"]["Apple test"]["results"]
    with pytest.raises(ValueError):
        check_omni_golden.import_outputs(golden, outputs)


def test_summary_checks_waveform_and_hashes_pcm() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x00\x02\x00")
    completion: check_omni_golden.Completion = {
        "choices": [
            {
                "message": {
                    "content": "Hello",
                    "audio": {"data": base64.b64encode(buffer.getvalue()).decode()},
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 1},
    }
    result = check_omni_golden.summarize(completion, True)
    assert result["audio_samples"] == 2
    assert (
        result["audio_sha256"]
        == "7b11c1133330cd161071bf23a0c9b6ce5320a8f3a0f83620035a72be46df4104"
    )
    with pytest.raises(ValueError, match="unexpected audio"):
        check_omni_golden.summarize(completion, False)
    completion["choices"][0]["message"]["audio"]["data"] = base64.b64encode(
        buffer.getvalue()[:-1]
    ).decode()
    with pytest.raises(ValueError, match="truncated"):
        check_omni_golden.summarize(completion, True)


def test_modified_or_additional_checkpoint_weights_fail(tmp_path: Path) -> None:
    pin = {
        "revision": "revision",
        "files": {
            "model.safetensors": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
    }
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"")
    check_omni_golden.verify_checkpoint(tmp_path, pin)
    weights.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="model.safetensors"):
        check_omni_golden.verify_checkpoint(tmp_path, pin)
    weights.write_bytes(b"")
    (tmp_path / "extra.safetensors").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="file set"):
        check_omni_golden.verify_checkpoint(tmp_path, pin)
