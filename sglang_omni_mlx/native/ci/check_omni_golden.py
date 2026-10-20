# SPDX-License-Identifier: Apache-2.0
"""Check multimodal chat goldens; import a chip's output only with explicit --import."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import NotRequired, TypedDict

from check_golden import chip, model_directory
from provision import sha256
from provision_models import FileSource


class Message(TypedDict):
    role: str
    content: str


class ChatRequest(TypedDict, total=False):
    model: str
    messages: list[Message]
    modalities: list[str]
    audio: dict[str, str]
    temperature: float
    max_tokens: int
    talker_temperature: float
    talker_max_new_tokens: int
    stream: bool
    images: list[str]
    audios: list[str]
    videos: list[str]
    video_fps: float
    video_max_frames: int
    video_max_pixels: int
    use_audio_in_video: bool


class Result(TypedDict):
    text: str
    completion_tokens: int
    finish_reason: str
    audio_sha256: str | None
    audio_samples: int


class Recording(TypedDict):
    spec_sha256: str
    checkpoint_sha256: str
    runtime_sha256: str
    mlx_version: str
    results: dict[str, Result]


class Outputs(Recording):
    model: str
    device: str


class Golden(TypedDict):
    model: str
    checkpoint_sha256: str
    reference: dict[str, str]
    files: dict[str, str]
    requests: dict[str, ChatRequest]
    devices: dict[str, Recording]


class OutputMessage(TypedDict):
    content: str
    audio: NotRequired[dict[str, str]]


class Choice(TypedDict):
    message: OutputMessage
    finish_reason: str


class Completion(TypedDict):
    choices: list[Choice]
    usage: dict[str, int]


def spec_digest(golden: Golden) -> str:
    specification = {name: value for name, value in golden.items() if name != "devices"}
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_outputs(golden: Golden, outputs: Outputs) -> None:
    if outputs["model"] != golden["model"]:
        raise ValueError("output model differs from golden")
    elif outputs["spec_sha256"] != spec_digest(golden):
        raise ValueError("request, media or reference provenance differs from golden")
    elif outputs["checkpoint_sha256"] != golden["checkpoint_sha256"]:
        raise ValueError("checkpoint provenance differs from golden")
    elif outputs["mlx_version"] != golden["reference"]["mlx_version"]:
        raise ValueError("executed MLX version differs from reference")
    elif len(outputs["runtime_sha256"]) != 64:
        raise ValueError("runtime binary hash is missing")
    elif not outputs["device"].strip():
        raise ValueError("output chip is missing")
    elif not golden["requests"] or set(outputs["results"]) != set(golden["requests"]):
        raise ValueError("output case set differs from golden")
    else:
        for name, result in outputs["results"].items():
            speech = "audio" in golden["requests"][name].get("modalities", [])
            if not result["text"].strip() or result["completion_tokens"] <= 0:
                raise ValueError(f"{name}: empty generation")
            elif result["finish_reason"] not in ("stop", "length"):
                raise ValueError(f"{name}: invalid finish reason")
            elif speech != (result["audio_samples"] > 0):
                raise ValueError(
                    f"{name}: output audio does not match requested modalities"
                )
            elif speech != bool(result["audio_sha256"]):
                raise ValueError(f"{name}: output audio hash is missing or unexpected")
            else:
                pass


def import_outputs(golden: Golden, outputs: Outputs) -> None:
    validate_outputs(golden, outputs)
    golden["devices"][outputs["device"]] = {
        "spec_sha256": outputs["spec_sha256"],
        "checkpoint_sha256": outputs["checkpoint_sha256"],
        "runtime_sha256": outputs["runtime_sha256"],
        "mlx_version": outputs["mlx_version"],
        "results": copy.deepcopy(outputs["results"]),
    }


def check(golden: Golden, outputs: Outputs) -> list[str]:
    try:
        validate_outputs(golden, outputs)
    except ValueError as error:
        return [str(error)]
    else:
        reference = golden["devices"].get(outputs["device"])
        if reference is None:
            return [
                f"No golden for {outputs['device']}; inspect and explicitly import its artifact"
            ]
        elif reference["spec_sha256"] != outputs["spec_sha256"]:
            return ["recorded golden has stale request, media or reference provenance"]
        elif set(reference["results"]) != set(outputs["results"]):
            return ["recorded golden has an incomplete case set"]
        else:
            return [
                f"{name}: output differs from its chip's golden"
                for name, result in outputs["results"].items()
                if result != reference["results"][name]
            ]


def summarize(completion: Completion, speech: bool) -> Result:
    if len(completion["choices"]) != 1:
        raise ValueError("expected exactly one completion choice")
    else:
        choice = completion["choices"][0]
        message = choice["message"]
        pcm = b""
        if speech:
            encoded = message["audio"]["data"]
            with wave.open(
                io.BytesIO(base64.b64decode(encoded, validate=True))
            ) as reader:
                if (
                    reader.getnchannels(),
                    reader.getsampwidth(),
                    reader.getframerate(),
                ) != (1, 2, 24000):
                    raise ValueError("expected mono PCM16 at 24 kHz")
                else:
                    pcm = reader.readframes(reader.getnframes())
                    if not pcm or len(pcm) != reader.getnframes() * 2 or not any(pcm):
                        raise ValueError("empty, truncated or silent generated audio")
                    else:
                        pass
        elif message.get("audio"):
            raise ValueError("unexpected audio in text-only completion")
        else:
            pass
        return {
            "text": message["content"],
            "completion_tokens": completion["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"],
            "audio_sha256": hashlib.sha256(pcm).hexdigest() if speech else None,
            "audio_samples": len(pcm) // 2,
        }


def verify_checkpoint(checkpoint: Path, pin: FileSource) -> str:
    weights = {path.name for path in checkpoint.glob("*.safetensors")}
    if weights != {name for name in pin["files"] if name.endswith(".safetensors")}:
        raise ValueError("checkpoint weight file set differs from pin")
    else:
        for filename, expected in pin["files"].items():
            if sha256(checkpoint / filename) != expected:
                raise ValueError(f"checkpoint hash differs: {filename}")
            else:
                pass
    encoded = json.dumps(pin, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_cases(
    runtime_bin: Path, checkpoint: Path, golden: Golden, media_root: Path
) -> Outputs:
    pins: dict[str, FileSource] = json.loads(
        (Path(__file__).parent / "model_pins.json").read_text()
    )
    pin = pins[golden["model"]]
    checkpoint_sha256 = verify_checkpoint(checkpoint, pin)
    if (
        pin["revision"] != golden["reference"]["revision"]
        or checkpoint_sha256 != golden["checkpoint_sha256"]
    ):
        raise ValueError("golden checkpoint identity differs from verified pin")
    else:
        pass
    executable = runtime_bin / "qwen3_omni_generate"
    version = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    runtime_version: dict[str, str] = json.loads(version.stdout)
    if runtime_version["mlx_version"] != golden["reference"]["mlx_version"]:
        raise ValueError("native MLX version differs from reference")
    else:
        pass
    media_root = media_root.resolve()
    for filename, expected in golden["files"].items():
        path = (media_root / filename).resolve()
        if not path.is_relative_to(media_root):
            raise ValueError(f"media escapes fixture root: {filename}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"fixture hash differs: {filename}")
        else:
            pass
    results: dict[str, Result] = {}
    with tempfile.TemporaryDirectory(prefix="omni-golden-") as temporary:
        request_path = Path(temporary) / "request.json"
        for name, specification in golden["requests"].items():
            request = copy.deepcopy(specification)
            for field in ("images", "audios", "videos"):
                filenames = request.get(field, [])
                for filename in filenames:
                    if filename not in golden["files"]:
                        raise ValueError(f"{name}: unpinned media {filename}")
                    else:
                        pass
                if filenames:
                    request[field] = [
                        str(media_root / filename) for filename in filenames
                    ]
                else:
                    pass
            request_path.write_text(json.dumps(request))
            completed = subprocess.run(
                [
                    str(executable),
                    "--model-path",
                    str(checkpoint),
                    "--request",
                    str(request_path),
                    "--allowed-local-media-path",
                    str(media_root),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            completion: Completion = json.loads(completed.stdout)
            results[name] = summarize(completion, "audio" in request["modalities"])
    return {
        "model": golden["model"],
        "device": chip(),
        "spec_sha256": spec_digest(golden),
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_sha256": sha256(executable),
        "mlx_version": runtime_version["mlx_version"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--runtime-bin", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "tests/data",
    )
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--import", dest="import_path", type=Path)
    arguments = parser.parse_args()
    golden: Golden = json.loads(arguments.golden.read_text())
    if arguments.import_path:
        outputs: Outputs = json.loads(arguments.import_path.read_text())
        import_outputs(golden, outputs)
        arguments.golden.write_text(
            json.dumps(golden, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Imported {outputs['device']} into {arguments.golden}")
    elif arguments.runtime_bin is None or (
        arguments.model_path is None and arguments.data_root is None
    ):
        parser.error(
            "--runtime-bin and either --model-path or --data-root are required"
        )
    else:
        checkpoint = arguments.model_path or model_directory(
            arguments.data_root, golden["model"]
        )
        outputs = run_cases(
            arguments.runtime_bin, checkpoint, golden, arguments.media_root
        )
        if arguments.outputs:
            arguments.outputs.mkdir(parents=True, exist_ok=True)
            destination = arguments.outputs / f"{arguments.golden.stem}.json"
            destination.write_text(
                json.dumps(outputs, indent=2, ensure_ascii=False) + "\n"
            )
        else:
            pass
        failures = check(golden, outputs)
        for failure in failures:
            print(failure)
        print(f"{len(outputs['results'])} cases, {len(failures)} golden failures")
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
