# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the native MLX Qwen3-Omni vision encoder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

import sglang_omni.models.qwen3_omni.mlx.vision as mlx_vision  # noqa: E402
from sglang_omni.models.qwen3_omni.components.image_encoder import (  # noqa: E402
    Qwen3OmniImageEncoder,
)
from sglang_omni.models.qwen3_omni.mlx.vision import (  # noqa: E402
    Qwen3OmniMlxImageEncoder,
    load_qwen3_omni_mlx_vision,
    sanitize_vision_weights,
)
from tests.utils.build_tiny_qwen3_omni_checkpoint import (  # noqa: E402
    build_tiny_config,
    build_tiny_qwen3_omni_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny_qwen3_omni_mlx_vision")
    return build_tiny_qwen3_omni_checkpoint(root / "tiny")


def tiny_visual_inputs(
    grid: tuple[tuple[int, int, int], ...] = ((1, 4, 4),),
) -> tuple[torch.Tensor, torch.Tensor]:
    config = build_tiny_config().thinker_config.vision_config
    grid_thw = torch.tensor(grid, dtype=torch.long)
    patch_dim = (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )
    values = torch.arange(
        int(grid_thw.prod(dim=-1).sum()) * patch_dim,
        dtype=torch.float32,
    ).reshape(-1, patch_dim)
    return torch.sin(values / 97.0), grid_thw


def tiny_image_stage_inputs() -> dict[str, torch.Tensor]:
    pixel_values, image_grid_thw = tiny_visual_inputs()
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
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


def test_mlx_vision_returns_primary_and_deepstack_rows(tiny_checkpoint) -> None:
    encoder = load_qwen3_omni_mlx_vision(str(tiny_checkpoint))
    pixel_values, grid_thw = tiny_visual_inputs()

    primary, deepstack = encoder(
        mx.array(pixel_values.numpy()),
        mx.array(grid_thw.numpy()),
    )

    assert primary.shape[0] == int(grid_thw.prod()) // 4
    assert primary.shape[1] == encoder.config.out_hidden_size
    assert len(deepstack) == len(encoder.config.deepstack_visual_indexes)
    assert all(layer.shape == primary.shape for layer in deepstack)


def test_mlx_vision_attention_bounds_score_buffers(
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = load_qwen3_omni_mlx_vision(str(tiny_checkpoint))
    pixel_values, grid_thw = tiny_visual_inputs()
    max_score_elements = 128
    original_sdpa = mlx_vision.mx.fast.scaled_dot_product_attention
    observed_shapes = []

    def reject_explicit_softmax(*_args, **_kwargs):
        raise AssertionError("vision attention materialized its score matrix")

    def bounded_sdpa(query, key, value, *, scale, mask=None, sinks=None):
        score_elements = query.shape[1] * query.shape[2] * key.shape[2]
        if score_elements > max_score_elements:
            raise MemoryError(
                f"attention score buffer has {score_elements} elements, "
                f"limit is {max_score_elements}"
            )
        observed_shapes.append((query.shape, key.shape))
        return original_sdpa(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    monkeypatch.setattr(
        mlx_vision,
        "_VISION_ATTENTION_SCORE_BUDGET_BYTES",
        max_score_elements * 4,
        raising=False,
    )
    monkeypatch.setattr(mlx_vision.mx, "softmax", reject_explicit_softmax)
    monkeypatch.setattr(
        mlx_vision.mx.fast,
        "scaled_dot_product_attention",
        bounded_sdpa,
    )

    primary, deepstack = encoder(
        mx.array(pixel_values.numpy()),
        mx.array(grid_thw.numpy()),
    )
    mx.eval(primary, deepstack)

    assert primary.shape == (4, encoder.config.out_hidden_size)
    assert observed_shapes
    assert any(
        query_shape[2] < key_shape[2] for query_shape, key_shape in observed_shapes
    )


def test_mlx_vision_matches_torch_encoder(tiny_checkpoint) -> None:
    torch_encoder = Qwen3OmniImageEncoder(
        str(tiny_checkpoint), device="cpu", dtype="float32"
    )
    mlx_encoder = Qwen3OmniMlxImageEncoder(str(tiny_checkpoint))
    inputs = tiny_image_stage_inputs()

    expected = torch_encoder(**inputs)
    actual = mlx_encoder(**inputs)

    assert actual["image_grid_thw"].equal(expected["image_grid_thw"])
    assert actual["image_token_counts"].equal(expected["image_token_counts"])
    assert_cosine_close(actual["image_embeds"], expected["image_embeds"], 0.9999)
    for actual_layer, expected_layer in zip(
        actual["deepstack_visual_embeds_image"],
        expected["deepstack_visual_embeds_image"],
        strict=True,
    ):
        assert_cosine_close(actual_layer, expected_layer, 0.9999)


def test_mlx_vision_matches_torch_for_video_inputs(tiny_checkpoint: Path) -> None:
    torch_encoder = Qwen3OmniImageEncoder(
        str(tiny_checkpoint), device="cpu", dtype="float32"
    )
    mlx_encoder = Qwen3OmniMlxImageEncoder(str(tiny_checkpoint))
    pixel_values, video_grid_thw = tiny_visual_inputs(((2, 4, 4),))
    inputs = {
        "pixel_values_videos": pixel_values,
        "video_grid_thw": video_grid_thw,
    }

    expected = torch_encoder(**inputs)
    actual = mlx_encoder(**inputs)

    assert actual["video_grid_thw"].equal(expected["video_grid_thw"])
    assert actual["video_token_counts"].equal(expected["video_token_counts"])
    assert_cosine_close(actual["video_embeds"], expected["video_embeds"], 0.9999)
    for actual_layer, expected_layer in zip(
        actual["deepstack_visual_embeds_video"],
        expected["deepstack_visual_embeds_video"],
        strict=True,
    ):
        assert_cosine_close(actual_layer, expected_layer, 0.9999)


def test_mlx_vision_preserves_mixed_image_video_ordering(
    tiny_checkpoint: Path,
) -> None:
    torch_encoder = Qwen3OmniImageEncoder(
        str(tiny_checkpoint), device="cpu", dtype="float32"
    )
    mlx_encoder = Qwen3OmniMlxImageEncoder(str(tiny_checkpoint))
    image_pixels, image_grid = tiny_visual_inputs(((1, 4, 6), (1, 6, 4)))
    video_pixels, video_grid = tiny_visual_inputs(((2, 4, 4), (1, 4, 6)))
    inputs = {
        "pixel_values": image_pixels,
        "image_grid_thw": image_grid,
        "pixel_values_videos": video_pixels,
        "video_grid_thw": video_grid,
    }

    expected = torch_encoder(**inputs)
    actual = mlx_encoder(**inputs)

    assert list(actual) == list(expected)
    assert_cosine_close(actual["image_embeds"], expected["image_embeds"], 0.9999)
    assert_cosine_close(actual["video_embeds"], expected["video_embeds"], 0.9999)
    assert actual["image_token_counts"].tolist() == [6, 6]
    assert actual["video_token_counts"].tolist() == [8, 6]


def test_vision_sanitizer_maps_names_and_transposes_torch_patch_kernel() -> None:
    kernel = mx.array(np.arange(2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5))
    weights = {
        "thinker.visual.patch_embed.proj.weight": kernel,
        "thinker.vision_tower.merger.ln_q.weight": mx.ones((8,)),
        "thinker.visual.merger.mlp.0.weight": mx.ones((16, 16)),
        "thinker.visual.merger.mlp.2.bias": mx.ones((8,)),
        "thinker.visual.merger_list.0.ln_q.bias": mx.ones((16,)),
    }

    actual = sanitize_vision_weights(
        weights,
        expected_shapes={"patch_embed.proj.weight": (2, 2, 4, 5, 3)},
    )

    assert set(actual) == {
        "patch_embed.proj.weight",
        "merger.norm.weight",
        "merger.linear_fc1.weight",
        "merger.linear_fc2.bias",
        "deepstack_merger_list.0.norm.bias",
    }
    assert actual["patch_embed.proj.weight"].shape == (2, 2, 4, 5, 3)
    np.testing.assert_array_equal(
        np.asarray(actual["patch_embed.proj.weight"]),
        np.asarray(kernel).transpose(0, 2, 3, 4, 1),
    )


def test_vision_sanitizer_preserves_channel_last_patch_kernel() -> None:
    kernel = mx.array(np.arange(2 * 2 * 4 * 5 * 3).reshape(2, 2, 4, 5, 3))

    actual = sanitize_vision_weights(
        {"thinker.vision_tower.patch_embed.proj.weight": kernel},
        expected_shapes={"patch_embed.proj.weight": (2, 2, 4, 5, 3)},
    )

    assert actual["patch_embed.proj.weight"].shape == kernel.shape
    np.testing.assert_array_equal(
        np.asarray(actual["patch_embed.proj.weight"]),
        np.asarray(kernel),
    )


def test_vision_sanitizer_rejects_unrecognized_patch_kernel_shape() -> None:
    kernel = mx.zeros((2, 3, 2, 4, 6))

    with pytest.raises(
        ValueError,
        match=r"patch_embed\.proj\.weight.*shape.*expected",
    ):
        sanitize_vision_weights(
            {"thinker.vision_tower.patch_embed.proj.weight": kernel},
            expected_shapes={"patch_embed.proj.weight": (2, 2, 4, 5, 3)},
        )


def test_mlx_image_encoder_reads_bfloat16_itemsize_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visual = SimpleNamespace(
        config=SimpleNamespace(
            spatial_merge_size=2,
            out_hidden_size=8,
            deepstack_visual_indexes=(0,),
        ),
        patch_embed=SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16))
        ),
    )
    monkeypatch.setattr(
        mlx_vision,
        "load_qwen3_omni_mlx_vision",
        lambda _model_path: visual,
    )

    encoder = Qwen3OmniMlxImageEncoder("unused")

    assert encoder.visual_dtype_bytes == 2


def _write_converted_vision_checkpoint(root: Path, source: Path) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(source / "config.json", root / "config.json")
    official: dict[str, mx.array] = {}
    for shard in sorted(source.glob("*.safetensors")):
        official.update(mx.load(str(shard)))

    converted: dict[str, mx.array] = {}
    for source_key, value in official.items():
        if not source_key.startswith("thinker.visual."):
            continue
        key = source_key.removeprefix("thinker.visual.")
        key = key.replace("merger_list.", "deepstack_merger_list.")
        if key.startswith(("merger.", "deepstack_merger_list.")):
            key = key.replace(".ln_q.", ".norm.")
            key = key.replace(".mlp.0.", ".linear_fc1.")
            key = key.replace(".mlp.2.", ".linear_fc2.")
        if key == "patch_embed.proj.weight":
            value = value.transpose(0, 2, 3, 4, 1)
        converted[key] = value

    vision_dir = root / "vision"
    vision_dir.mkdir()
    mx.save_safetensors(str(vision_dir / "model.safetensors"), converted)
    return root


def _write_prefixed_bfloat16_vision_checkpoint(root: Path, source: Path) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(source / "config.json", root / "config.json")
    official: dict[str, mx.array] = {}
    for shard in sorted(source.glob("*.safetensors")):
        official.update(mx.load(str(shard)))

    converted: dict[str, mx.array] = {}
    for source_key, value in official.items():
        if not source_key.startswith("thinker.visual."):
            continue
        key = source_key.removeprefix("thinker.visual.")
        key = key.replace("merger_list.", "deepstack_merger_list.")
        if key.startswith(("merger.", "deepstack_merger_list.")):
            key = key.replace(".ln_q.", ".norm.")
            key = key.replace(".mlp.0.", ".linear_fc1.")
            key = key.replace(".mlp.2.", ".linear_fc2.")
        if key == "patch_embed.proj.weight":
            value = value.transpose(0, 2, 3, 4, 1)
        converted[f"thinker.vision_tower.{key}"] = value.astype(mx.bfloat16)

    vision_dir = root / "vision"
    vision_dir.mkdir()
    mx.save_safetensors(str(vision_dir / "model.safetensors"), converted)
    return root


def test_mlx_vision_loads_converted_component_layout(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    converted = _write_converted_vision_checkpoint(
        tmp_path / "converted", tiny_checkpoint
    )
    expected = Qwen3OmniMlxImageEncoder(str(tiny_checkpoint))(
        **tiny_image_stage_inputs()
    )
    actual = Qwen3OmniMlxImageEncoder(str(converted))(**tiny_image_stage_inputs())

    assert_cosine_close(actual["image_embeds"], expected["image_embeds"], 0.999999)
    for actual_layer, expected_layer in zip(
        actual["deepstack_visual_embeds_image"],
        expected["deepstack_visual_embeds_image"],
        strict=True,
    ):
        assert_cosine_close(actual_layer, expected_layer, 0.999999)


def test_mlx_vision_loads_prefixed_channel_last_bfloat16_checkpoint(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    converted = _write_prefixed_bfloat16_vision_checkpoint(
        tmp_path / "prefixed-converted",
        tiny_checkpoint,
    )

    encoder = Qwen3OmniMlxImageEncoder(str(converted))

    config = encoder.visual.config
    assert encoder.visual.patch_embed.proj.weight.shape == (
        config.hidden_size,
        config.temporal_patch_size,
        config.patch_size,
        config.patch_size,
        config.in_channels,
    )
    assert encoder.visual.patch_embed.proj.weight.dtype == mx.bfloat16
    assert encoder.visual_dtype_bytes == 2


def test_mlx_vision_rejects_malformed_grid(tiny_checkpoint: Path) -> None:
    encoder = load_qwen3_omni_mlx_vision(str(tiny_checkpoint))
    pixel_values, _ = tiny_visual_inputs()

    with pytest.raises(ValueError, match="divisible by spatial_merge_size"):
        encoder(
            pixel_values=mx.array(pixel_values.numpy()), grid_thw=mx.array([[1, 3, 4]])
        )


def test_mlx_vision_config_comes_from_checkpoint(tiny_checkpoint: Path) -> None:
    raw = json.loads((tiny_checkpoint / "config.json").read_text(encoding="utf-8"))
    encoder = load_qwen3_omni_mlx_vision(str(tiny_checkpoint))

    assert tuple(encoder.config.deepstack_visual_indexes) == tuple(
        raw["thinker_config"]["vision_config"]["deepstack_visual_indexes"]
    )
