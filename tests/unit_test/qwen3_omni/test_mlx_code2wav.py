# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni code2wav tests."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.models.qwen3_omni.mlx.code2wav import (  # noqa: E402
    CausalConv1d,
    CausalConvTranspose1d,
    Qwen3OmniMlxCode2Wav,
    SnakeBeta,
    load_qwen3_omni_mlx_code2wav,
    sanitize_code2wav_weights,
)
from sglang_omni.models.qwen3_omni.mlx.common import (  # noqa: E402
    load_qwen3_omni_mlx_component,
)
from sglang_omni.models.qwen3_omni.mlx.config import (  # noqa: E402
    Code2WavConfig,
    QuantizationConfig,
    Qwen3OmniMlxConfig,
)
from tests.utils.build_tiny_qwen3_omni_checkpoint import (  # noqa: E402
    build_tiny_config,
    build_tiny_qwen3_omni_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny_qwen3_omni_mlx_code2wav")
    return build_tiny_qwen3_omni_checkpoint(root / "tiny")


def _tiny_native_config() -> Code2WavConfig:
    raw = build_tiny_config().to_dict()
    return Qwen3OmniMlxConfig.from_dict(raw).code2wav


def _expected_shapes(model: Qwen3OmniMlxCode2Wav) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in tree_flatten(model.parameters())}


@pytest.mark.parametrize(
    ("length", "kernel_size", "dilation", "stride", "expected"),
    [
        (1, 7, 3, 1, 1),
        (11, 7, 3, 1, 11),
        (12, 5, 2, 3, 4),
        (13, 5, 2, 3, 5),
    ],
)
def test_causal_conv_preserves_ceil_stride_length(
    length: int,
    kernel_size: int,
    dilation: int,
    stride: int,
    expected: int,
) -> None:
    layer = CausalConv1d(
        4,
        6,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
    )
    output = layer(mx.zeros((1, 4, length)))
    assert output.shape == (1, 6, expected)


@pytest.mark.parametrize(("length", "stride"), [(1, 2), (7, 5), (11, 8)])
def test_causal_transposed_conv_multiplies_length(length: int, stride: int) -> None:
    layer = CausalConvTranspose1d(4, 2, kernel_size=2 * stride, stride=stride)
    output = layer(mx.zeros((1, 4, length)))
    assert output.shape == (1, 2, length * stride)


def test_snake_beta_uses_exponential_parameters() -> None:
    layer = SnakeBeta(2)
    layer.alpha = mx.array([np.log(2.0), np.log(3.0)], dtype=mx.float32)
    layer.beta = mx.array([np.log(4.0), np.log(5.0)], dtype=mx.float32)
    values = np.array([[[0.25, -0.5], [0.75, -1.0]]], dtype=np.float32)

    actual = layer(mx.array(values))
    expected = values + np.sin(values * np.array([2.0, 3.0])[None, :, None]) ** 2 / (
        np.array([4.0, 5.0])[None, :, None] + 1e-9
    )

    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_code_embedding_uses_exact_quantizer_offsets() -> None:
    config = replace(
        _tiny_native_config(),
        codebook_size=4,
        num_quantizers=2,
    )
    model = Qwen3OmniMlxCode2Wav(config)
    rows = np.repeat(
        np.arange(config.codebook_size * config.num_quantizers)[:, None],
        config.hidden_size,
        axis=1,
    ).astype(np.float32)
    model.code_embedding.weight = mx.array(rows)
    codes = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)

    actual = model.embed_codes(codes)

    mx.eval(actual)
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.array([[[3.0] * config.hidden_size, [4.0] * config.hidden_size]]),
    )


def test_code2wav_builds_configured_eight_layer_pre_transformer() -> None:
    config = Code2WavConfig.from_dict(
        Qwen3OmniMoeCode2WavConfig().to_dict(),
        path="code2wav_config",
    )
    model = Qwen3OmniMlxCode2Wav(config)
    assert len(model.pre_transformer.layers) == 8


def test_mlx_code2wav_rejects_wrong_quantizer_count() -> None:
    model = Qwen3OmniMlxCode2Wav(_tiny_native_config())
    codes = mx.zeros((1, model.config.num_quantizers - 1, 3), dtype=mx.int32)

    with pytest.raises(ValueError, match=r"Expected .* quantizers"):
        model(codes)


@pytest.mark.parametrize("bad_code", [-1, 2048])
def test_mlx_code2wav_rejects_codes_outside_codebook(bad_code: int) -> None:
    model = Qwen3OmniMlxCode2Wav(_tiny_native_config())
    codes = mx.zeros((1, model.config.num_quantizers, 2), dtype=mx.int32)
    codes[:, 0, 0] = bad_code

    with pytest.raises(ValueError, match=r"outside \[0, 2048\)"):
        model(codes)


def test_mlx_code2wav_module_keys_match_transformers() -> None:
    torch_config = build_tiny_config().code2wav_config
    torch_model = Qwen3OmniMoeCode2Wav._from_config(torch_config)
    mlx_model = Qwen3OmniMlxCode2Wav(_tiny_native_config())

    assert set(dict(tree_flatten(mlx_model.parameters()))) == set(
        torch_model.state_dict()
    )


@pytest.mark.parametrize(
    ("key", "source_shape", "target_shape", "permutation"),
    [
        ("decoder.0.conv.weight", (6, 4, 7), (6, 7, 4), (0, 2, 1)),
        (
            "upsample.0.1.dwconv.conv.weight",
            (4, 1, 7),
            (4, 7, 1),
            (0, 2, 1),
        ),
        ("upsample.0.0.conv.weight", (4, 2, 10), (2, 10, 4), (1, 2, 0)),
        (
            "decoder.1.block.1.conv.weight",
            (4, 2, 10),
            (2, 10, 4),
            (1, 2, 0),
        ),
    ],
)
def test_code2wav_sanitizer_transposes_only_supported_torch_convolutions(
    key: str,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    permutation: tuple[int, ...],
) -> None:
    source = mx.array(np.arange(np.prod(source_shape)).reshape(source_shape))

    actual = sanitize_code2wav_weights(
        {f"code2wav.{key}": source},
        expected_shapes={key: target_shape},
    )

    np.testing.assert_array_equal(
        np.asarray(actual[key]),
        np.asarray(source).transpose(permutation),
    )


@pytest.mark.parametrize(
    "key",
    [
        "decoder.0.conv.weight",
        "upsample.0.1.dwconv.conv.weight",
        "upsample.0.0.conv.weight",
        "decoder.1.block.1.conv.weight",
    ],
)
def test_code2wav_sanitizer_preserves_matching_mlx_convolution_layout(
    key: str,
) -> None:
    target = mx.array(np.arange(2 * 5 * 3).reshape(2, 5, 3))

    actual = sanitize_code2wav_weights(
        {key: target},
        expected_shapes={key: tuple(target.shape)},
    )

    np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(target))


def test_code2wav_sanitizer_rejects_malformed_convolution_layout() -> None:
    key = "decoder.1.block.1.conv.weight"
    malformed = mx.zeros((4, 3, 10))

    with pytest.raises(ValueError, match=r"decoder\.1\.block\.1\.conv\.weight.*shape"):
        sanitize_code2wav_weights(
            {f"code2wav.{key}": malformed},
            expected_shapes={key: (2, 10, 4)},
        )


def test_code2wav_loader_quantizes_complete_affine_groups_but_not_convolutions() -> (
    None
):
    config = _tiny_native_config()
    reference = Qwen3OmniMlxCode2Wav(config)
    quantized_paths = {
        "code_embedding",
        "pre_transformer.layers.0.self_attn.q_proj",
    }
    nn.quantize(
        reference,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: path in quantized_paths,
    )
    published = dict(tree_flatten(reference.parameters()))

    model = Qwen3OmniMlxCode2Wav(config)
    loaded = load_qwen3_omni_mlx_component(
        model,
        published,
        sanitizer=lambda weights: sanitize_code2wav_weights(
            weights,
            expected_shapes=_expected_shapes(model),
        ),
        quantization=QuantizationConfig(bits=4, group_size=32),
    )

    assert isinstance(loaded.code_embedding, nn.QuantizedEmbedding)
    assert isinstance(
        loaded.pre_transformer.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert type(loaded.decoder[0].conv) is nn.Conv1d
    codes = mx.zeros((1, config.num_quantizers, 1), dtype=mx.int32)
    expected = reference(codes)
    actual = loaded(codes)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("missing", ["scales", "biases"])
def test_code2wav_loader_rejects_incomplete_affine_group(missing: str) -> None:
    model = Qwen3OmniMlxCode2Wav(_tiny_native_config())
    weights = {
        "code_embedding.weight": mx.zeros((2048, 4), dtype=mx.uint32),
        "code_embedding.scales": mx.ones((2048, 1)),
        "code_embedding.biases": mx.zeros((2048, 1)),
    }
    del weights[f"code_embedding.{missing}"]
    expected_missing = "biases" if missing == "biases" else "scales"

    with pytest.raises(ValueError, match=rf"code_embedding.*{expected_missing}"):
        load_qwen3_omni_mlx_component(
            model,
            weights,
            sanitizer=lambda values: sanitize_code2wav_weights(
                values,
                expected_shapes=_expected_shapes(model),
            ),
            quantization=QuantizationConfig(bits=4, group_size=32),
        )


def _is_transposed_convolution(key: str) -> bool:
    return bool(
        re.fullmatch(r"upsample\.\d+\.0\.conv\.weight", key)
        or re.fullmatch(r"decoder\.\d+\.block\.1\.conv\.weight", key)
    )


def _is_convolution(key: str) -> bool:
    return key.endswith(".conv.weight")


def _write_converted_code2wav_checkpoint(root: Path, source: Path) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(source / "config.json", root / "config.json")
    official: dict[str, mx.array] = {}
    for shard in sorted(source.glob("*.safetensors")):
        official.update(mx.load(str(shard)))

    converted: dict[str, mx.array] = {}
    for source_key, value in official.items():
        if not source_key.startswith("code2wav."):
            continue
        key = source_key.removeprefix("code2wav.")
        if _is_transposed_convolution(key):
            value = value.transpose(1, 2, 0)
        elif _is_convolution(key):
            value = value.transpose(0, 2, 1)
        converted[key] = value

    code2wav_dir = root / "code2wav"
    code2wav_dir.mkdir()
    mx.save_safetensors(str(code2wav_dir / "model.safetensors"), converted)
    return root


def test_mlx_code2wav_loads_official_and_converted_layouts(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    converted = _write_converted_code2wav_checkpoint(
        tmp_path / "converted",
        tiny_checkpoint,
    )
    official_model = load_qwen3_omni_mlx_code2wav(str(tiny_checkpoint))
    converted_model = load_qwen3_omni_mlx_code2wav(str(converted))
    codes = mx.array(
        np.arange(official_model.config.num_quantizers * 3).reshape(
            1, official_model.config.num_quantizers, 3
        )
        % official_model.config.codebook_size,
        dtype=mx.int32,
    )

    official_wav = official_model(codes)
    converted_wav = converted_model(codes)
    mx.eval(official_wav, converted_wav)

    assert official_wav.shape == (1, 1, 3 * 1920)
    np.testing.assert_allclose(
        np.asarray(converted_wav),
        np.asarray(official_wav),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mlx_code2wav_prefers_root_native_weights_over_dense_sidecar(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    checkpoint = _write_converted_code2wav_checkpoint(
        tmp_path / "with-sidecar",
        tiny_checkpoint,
    )
    for source_file in tiny_checkpoint.glob("*.safetensors*"):
        shutil.copy2(source_file, checkpoint / source_file.name)

    expected = load_qwen3_omni_mlx_code2wav(str(tiny_checkpoint))
    actual = load_qwen3_omni_mlx_code2wav(str(checkpoint))
    expected_weights = dict(tree_flatten(expected.parameters()))
    actual_weights = dict(tree_flatten(actual.parameters()))

    assert set(actual_weights) == set(expected_weights)
    for key in expected_weights:
        np.testing.assert_array_equal(
            np.asarray(actual_weights[key]),
            np.asarray(expected_weights[key]),
        )


def test_mlx_code2wav_loads_mixed_quantized_and_dense_converted_checkpoint(
    tmp_path: Path,
) -> None:
    config = _tiny_native_config()
    reference = Qwen3OmniMlxCode2Wav(config)
    quantized_paths = {
        "code_embedding",
        "pre_transformer.layers.0.self_attn.q_proj",
        "upsample.0.1.pwconv1",
    }
    nn.quantize(
        reference,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: path in quantized_paths,
    )
    root = tmp_path / "quantized"
    root.mkdir()
    raw_config = build_tiny_config().to_dict()
    raw_config["quantization"] = {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
    }
    (root / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )
    weights = {
        f"code2wav.{key}": value for key, value in tree_flatten(reference.parameters())
    }
    mx.save_safetensors(str(root / "model.safetensors"), weights)

    loaded = load_qwen3_omni_mlx_code2wav(str(root))
    codes = mx.zeros((1, config.num_quantizers, 2), dtype=mx.int32)
    expected = reference(codes)
    actual = loaded(codes)
    mx.eval(expected, actual)

    assert isinstance(loaded.code_embedding, nn.QuantizedEmbedding)
    assert isinstance(
        loaded.pre_transformer.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(loaded.upsample[0][1].pwconv1, nn.QuantizedLinear)
    assert type(loaded.upsample[0][1].dwconv.conv) is nn.Conv1d
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mlx_code2wav_matches_transformers_with_exact_causal_length(
    tiny_checkpoint: Path,
) -> None:
    from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
        load_code2wav_model,
    )

    torch_model = load_code2wav_model(
        str(tiny_checkpoint),
        device="cpu",
        dtype="float32",
    )

    def _right_trim_only(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        if self.right_pad:
            hidden_states = hidden_states[..., : -self.right_pad]
        return hidden_states.contiguous()

    for module in torch_model.modules():
        if isinstance(module, Qwen3OmniMoeCausalTransConvNet):
            module.forward = MethodType(_right_trim_only, module)

    mlx_model = load_qwen3_omni_mlx_code2wav(str(tiny_checkpoint))
    codes = (
        torch.arange(mlx_model.config.num_quantizers * 3, dtype=torch.long)
        .reshape(1, mlx_model.config.num_quantizers, 3)
        .remainder(mlx_model.config.codebook_size)
    )

    expected = torch_model(codes).detach().numpy()
    actual = mlx_model(mx.array(codes.numpy(), dtype=mx.int32))
    mx.eval(actual)

    assert actual.shape == expected.shape == (1, 1, 3 * 1920)
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_mlx_code2wav_future_code_does_not_change_emitted_samples(
    tiny_checkpoint: Path,
) -> None:
    model = load_qwen3_omni_mlx_code2wav(str(tiny_checkpoint))
    first = np.zeros((1, model.config.num_quantizers, 3), dtype=np.int32)
    second = first.copy()
    second[:, :, -1] = 17

    first_wav = model(mx.array(first))
    second_wav = model(mx.array(second))
    mx.eval(first_wav, second_wav)

    emitted = 2 * model.total_upsample
    np.testing.assert_allclose(
        np.asarray(first_wav)[..., :emitted],
        np.asarray(second_wav)[..., :emitted],
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(
        np.asarray(first_wav)[..., emitted:],
        np.asarray(second_wav)[..., emitted:],
    )


def test_mlx_code2wav_sample_count_and_clipping(tiny_checkpoint: Path) -> None:
    model = load_qwen3_omni_mlx_code2wav(str(tiny_checkpoint))
    codes = mx.zeros(
        (1, model.config.num_quantizers, 3),
        dtype=mx.int32,
    )
    wav = model(codes)
    mx.eval(wav)

    assert wav.shape == (1, 1, 3 * model.total_upsample)
    assert model.total_upsample == 1920
    assert np.asarray(wav).min() >= -1.0
    assert np.asarray(wav).max() <= 1.0

    zero_weights = [
        (key, mx.zeros_like(value)) for key, value in tree_flatten(model.parameters())
    ]
    model.load_weights(zero_weights, strict=True)
    model.decoder[-1].conv.bias = mx.array([2.0])
    high = model(codes)
    model.decoder[-1].conv.bias = mx.array([-2.0])
    low = model(codes)
    mx.eval(high, low)
    np.testing.assert_array_equal(np.asarray(high), np.ones(high.shape))
    np.testing.assert_array_equal(np.asarray(low), -np.ones(low.shape))
