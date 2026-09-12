# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the native MLX Qwen3-Omni audio encoder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_omni.components.audio_encoder import (  # noqa: E402
    Qwen3OmniAudioEncoder,
)
from sglang_omni.models.qwen3_omni.mlx.audio import (  # noqa: E402
    Qwen3OmniMlxAudioStageEncoder,
    load_qwen3_omni_mlx_audio,
    qwen3_omni_audio_output_lengths,
    sanitize_audio_weights,
)
from tests.utils.build_tiny_qwen3_omni_checkpoint import (  # noqa: E402
    build_tiny_config,
    build_tiny_qwen3_omni_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny_qwen3_omni_mlx_audio")
    return build_tiny_qwen3_omni_checkpoint(root / "tiny")


def make_audio_stage_inputs(frame_lengths: list[int]) -> dict[str, torch.Tensor]:
    """Build padded processor-style rows from one concatenated mel stream."""

    mel_bins = build_tiny_config().thinker_config.audio_config.num_mel_bins
    total_frames = sum(frame_lengths)
    values = torch.arange(mel_bins * total_frames, dtype=torch.float32)
    packed = torch.sin(values.reshape(mel_bins, total_frames) / 97.0)
    max_frames = max(frame_lengths)
    features = torch.zeros(len(frame_lengths), mel_bins, max_frames)
    mask = torch.zeros(len(frame_lengths), max_frames, dtype=torch.long)

    cursor = 0
    for row, length in enumerate(frame_lengths):
        features[row, :, :length] = packed[:, cursor : cursor + length]
        mask[row, :length] = 1
        cursor += length

    return {
        "input_features": features,
        "feature_attention_mask": mask,
    }


def assert_cosine_close(
    actual: torch.Tensor, expected: torch.Tensor, threshold: float
) -> None:
    assert actual.shape == expected.shape
    similarity = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    ).item()
    assert similarity >= threshold, similarity


@pytest.mark.parametrize("frames", [37, 100, 437, 1200])
def test_mlx_audio_output_lengths_match_torch(frames: int) -> None:
    lengths = torch.tensor([frames], dtype=torch.long)
    expected = hf_modeling._get_feat_extract_output_lengths(lengths)
    actual = qwen3_omni_audio_output_lengths(mx.array([frames]))
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_mlx_audio_matches_torch_across_inference_window(tiny_checkpoint) -> None:
    inputs = make_audio_stage_inputs(frame_lengths=[320, 1200])
    expected = Qwen3OmniAudioEncoder(
        str(tiny_checkpoint), device="cpu", dtype="float32"
    )(**inputs)
    actual = Qwen3OmniMlxAudioStageEncoder(str(tiny_checkpoint))(**inputs)

    torch.testing.assert_close(
        actual["audio_output_lengths"],
        expected["audio_output_lengths"],
        rtol=0,
        atol=0,
    )
    assert_cosine_close(actual["audio_embeds"], expected["audio_embeds"], 0.9999)


def test_mlx_audio_mixed_batch_matches_independent_rows(tiny_checkpoint) -> None:
    encoder = Qwen3OmniMlxAudioStageEncoder(str(tiny_checkpoint))
    frame_lengths = [37, 1200]

    batched = encoder(**make_audio_stage_inputs(frame_lengths))
    separate = [
        encoder(**make_audio_stage_inputs([frame_length]))
        for frame_length in frame_lengths
    ]

    expected_lengths = torch.cat(
        [result["audio_output_lengths"] for result in separate]
    )
    expected_embeds = torch.cat([result["audio_embeds"] for result in separate])
    torch.testing.assert_close(
        batched["audio_output_lengths"], expected_lengths, rtol=0, atol=0
    )
    assert_cosine_close(batched["audio_embeds"], expected_embeds, 0.999999)


def test_mlx_audio_stage_accepts_prepacked_features(tiny_checkpoint) -> None:
    encoder = Qwen3OmniMlxAudioStageEncoder(str(tiny_checkpoint))
    frame_lengths = [37, 100]
    padded = make_audio_stage_inputs(frame_lengths)
    packed = torch.cat(
        [
            row[:, :length]
            for row, length in zip(padded["input_features"], frame_lengths, strict=True)
        ],
        dim=-1,
    )

    expected = encoder(**padded)
    actual = encoder(
        input_features=packed,
        audio_feature_lengths=torch.tensor(frame_lengths, dtype=torch.long),
    )

    assert_cosine_close(actual["audio_embeds"], expected["audio_embeds"], 0.999999)
    torch.testing.assert_close(
        actual["audio_output_lengths"],
        expected["audio_output_lengths"],
        rtol=0,
        atol=0,
    )


def test_mlx_audio_stage_returns_transport_safe_torch_cpu_tensors(
    tiny_checkpoint,
) -> None:
    outputs = Qwen3OmniMlxAudioStageEncoder(str(tiny_checkpoint))(
        **make_audio_stage_inputs([37, 100])
    )

    assert list(outputs) == [
        "audio_embeds",
        "audio_feature_lengths",
        "audio_output_lengths",
    ]
    assert all(isinstance(value, torch.Tensor) for value in outputs.values())
    assert all(value.device.type == "cpu" for value in outputs.values())
    assert outputs["audio_embeds"].dtype == torch.float32
    assert outputs["audio_feature_lengths"].dtype == torch.long
    assert outputs["audio_output_lengths"].dtype == torch.long


def test_audio_sanitizer_transposes_official_torch_kernels() -> None:
    kernel = mx.array(np.arange(2 * 1 * 3 * 3).reshape(2, 1, 3, 3))
    weights = {
        "thinker.audio_tower.conv2d1.weight": kernel,
        "thinker.audio_tower.layers.0.fc1.bias": mx.ones((4,)),
    }

    actual = sanitize_audio_weights(
        weights,
        expected_shapes={
            "conv2d1.weight": (2, 3, 3, 1),
            "layers.0.fc1.bias": (4,),
        },
    )

    assert set(actual) == {"conv2d1.weight", "layers.0.fc1.bias"}
    np.testing.assert_array_equal(
        np.asarray(actual["conv2d1.weight"]),
        np.asarray(kernel).transpose(0, 2, 3, 1),
    )


def test_audio_sanitizer_preserves_converted_channel_last_kernels() -> None:
    kernel = mx.array(np.arange(2 * 3 * 3 * 1).reshape(2, 3, 3, 1))

    actual = sanitize_audio_weights(
        {"conv2d1.weight": kernel},
        expected_shapes={"conv2d1.weight": (2, 3, 3, 1)},
    )

    np.testing.assert_array_equal(np.asarray(actual["conv2d1.weight"]), kernel)


def test_audio_sanitizer_rejects_malformed_convolution_layout() -> None:
    kernel = mx.zeros((2, 4, 3, 1))

    with pytest.raises(ValueError, match=r"conv2d1\.weight.*shape.*expected"):
        sanitize_audio_weights(
            {"thinker.audio_tower.conv2d1.weight": kernel},
            expected_shapes={"conv2d1.weight": (2, 3, 3, 1)},
        )


def _write_converted_audio_checkpoint(root: Path, source: Path) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(source / "config.json", root / "config.json")
    official: dict[str, mx.array] = {}
    for shard in sorted(source.glob("*.safetensors")):
        official.update(mx.load(str(shard)))

    converted: dict[str, mx.array] = {}
    for source_key, value in official.items():
        if not source_key.startswith("thinker.audio_tower."):
            continue
        key = source_key.removeprefix("thinker.audio_tower.")
        if key in {"conv2d1.weight", "conv2d2.weight", "conv2d3.weight"}:
            value = value.transpose(0, 2, 3, 1)
        converted[key] = value

    audio_dir = root / "audio"
    audio_dir.mkdir()
    mx.save_safetensors(str(audio_dir / "model.safetensors"), converted)
    return root


def test_mlx_audio_loads_converted_component_layout(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    converted = _write_converted_audio_checkpoint(
        tmp_path / "converted", tiny_checkpoint
    )
    inputs = make_audio_stage_inputs([100, 137])
    expected = Qwen3OmniMlxAudioStageEncoder(str(tiny_checkpoint))(**inputs)
    actual = Qwen3OmniMlxAudioStageEncoder(str(converted))(**inputs)

    torch.testing.assert_close(
        actual["audio_output_lengths"],
        expected["audio_output_lengths"],
        rtol=0,
        atol=0,
    )
    assert_cosine_close(actual["audio_embeds"], expected["audio_embeds"], 0.999999)


def test_mlx_audio_loader_uses_checkpoint_config(tiny_checkpoint: Path) -> None:
    raw = json.loads((tiny_checkpoint / "config.json").read_text(encoding="utf-8"))
    encoder = load_qwen3_omni_mlx_audio(str(tiny_checkpoint))

    assert encoder.config.d_model == raw["thinker_config"]["audio_config"]["d_model"]
    assert (
        encoder.config.n_window_infer
        == raw["thinker_config"]["audio_config"]["n_window_infer"]
    )
