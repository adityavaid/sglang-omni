# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native MLX Qwen3-Omni primitives."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")

import mlx.nn as nn  # noqa: E402  (import after skip guard)
import torch  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.switch_layers import (  # noqa: E402
    QuantizedSwitchLinear,
    SwitchLinear,
)
from transformers.models.qwen3_omni_moe import (  # noqa: E402
    modeling_qwen3_omni_moe as ref,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
)

from sglang_omni.models.qwen3_omni.mlx import common as mlx_common  # noqa: E402
from sglang_omni.models.qwen3_omni.mlx.common import (  # noqa: E402
    SparseMoeBlock,
    _interleave_mrope,
    apply_multimodal_rope,
    quantize_converted_module,
    sanitize_qwen3_omni_weights,
    tie_lm_head_weights,
)
from sglang_omni.models.qwen3_omni.mlx.config import QuantizationConfig  # noqa: E402

_H = 32
_INTER = 64
_EXPERTS = 4
_TOPK = 2
_HEAD_DIM = 8
_SECTIONS = (2, 1, 1)
_THETA = 1000000.0


def deterministic_input(seed: int = 0) -> np.ndarray:
    """Fixed float32 (1, 3, hidden) tensor without touching global RNG state."""

    return np.random.default_rng(seed).standard_normal((1, 3, _H)).astype(np.float32)


def _thinker_text_config() -> Qwen3OmniMoeTextConfig:
    return Qwen3OmniMoeTextConfig(
        hidden_size=_H,
        intermediate_size=_INTER,
        moe_intermediate_size=_INTER,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=_HEAD_DIM,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOPK,
        norm_topk_prob=True,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": _THETA,
            "mrope_section": list(_SECTIONS),
        },
    )


def _talker_text_config(norm_topk_prob: bool = True) -> Qwen3OmniMoeTalkerTextConfig:
    return Qwen3OmniMoeTalkerTextConfig(
        hidden_size=_H,
        intermediate_size=_INTER,
        moe_intermediate_size=_INTER,
        shared_expert_intermediate_size=_INTER,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=_HEAD_DIM,
        num_local_experts=_EXPERTS,
        num_experts_per_tok=_TOPK,
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": _THETA,
            "mrope_section": list(_SECTIONS),
        },
    )


def _copy_router_and_experts(mlx_block: SparseMoeBlock, torch_block, rng) -> None:
    gate_up = rng.standard_normal((_EXPERTS, 2 * _INTER, _H)).astype(np.float32)
    down = rng.standard_normal((_EXPERTS, _H, _INTER)).astype(np.float32)
    gate_w = rng.standard_normal((_EXPERTS, _H)).astype(np.float32)

    mlx_block.experts.gate_up_proj.weight = mx.array(gate_up)
    mlx_block.experts.down_proj.weight = mx.array(down)
    mlx_block.gate.weight = mx.array(gate_w)
    with torch.no_grad():
        torch_block.experts.gate_up_proj.copy_(torch.tensor(gate_up))
        torch_block.experts.down_proj.copy_(torch.tensor(down))
        torch_block.gate.weight.copy_(torch.tensor(gate_w))


def equivalent_tiny_moe_blocks(
    *, shared: bool = False, norm_topk_prob: bool = True
) -> tuple[SparseMoeBlock, object]:
    """Build MLX + plain-PyTorch MoE blocks and copy identical weights in."""

    rng = np.random.default_rng(1234)
    if shared:
        cfg = _talker_text_config(norm_topk_prob=norm_topk_prob)
        torch_block = ref.Qwen3OmniMoeTalkerTextSparseMoeBlock(cfg)
        mlx_block = SparseMoeBlock(
            hidden_size=_H,
            moe_intermediate_size=_INTER,
            num_experts=_EXPERTS,
            num_experts_per_tok=_TOPK,
            norm_topk_prob=norm_topk_prob,
            shared_expert_intermediate_size=_INTER,
        )
    else:
        cfg = _thinker_text_config()
        cfg.norm_topk_prob = norm_topk_prob
        torch_block = ref.Qwen3OmniMoeThinkerTextSparseMoeBlock(cfg)
        mlx_block = SparseMoeBlock(
            hidden_size=_H,
            moe_intermediate_size=_INTER,
            num_experts=_EXPERTS,
            num_experts_per_tok=_TOPK,
            norm_topk_prob=norm_topk_prob,
        )

    _copy_router_and_experts(mlx_block, torch_block, rng)

    if shared:
        gate_p = rng.standard_normal((_INTER, _H)).astype(np.float32)
        up_p = rng.standard_normal((_INTER, _H)).astype(np.float32)
        down_p = rng.standard_normal((_H, _INTER)).astype(np.float32)
        shared_gate = rng.standard_normal((1, _H)).astype(np.float32)
        mlx_block.shared_expert.gate_proj.weight = mx.array(gate_p)
        mlx_block.shared_expert.up_proj.weight = mx.array(up_p)
        mlx_block.shared_expert.down_proj.weight = mx.array(down_p)
        mlx_block.shared_expert_gate.weight = mx.array(shared_gate)
        with torch.no_grad():
            torch_block.shared_expert.gate_proj.weight.copy_(torch.tensor(gate_p))
            torch_block.shared_expert.up_proj.weight.copy_(torch.tensor(up_p))
            torch_block.shared_expert.down_proj.weight.copy_(torch.tensor(down_p))
            torch_block.shared_expert_gate.weight.copy_(torch.tensor(shared_gate))

    return mlx_block, torch_block


def test_sparse_moe_matches_torch_reference() -> None:
    mlx_block, torch_block = equivalent_tiny_moe_blocks()
    x = deterministic_input()
    actual = mlx_block(mx.array(x))
    expected = torch_block(torch.tensor(x)).detach().numpy()
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-3, atol=2e-3)


def test_sparse_moe_talker_shared_expert_matches_torch() -> None:
    mlx_block, torch_block = equivalent_tiny_moe_blocks(shared=True)
    x = deterministic_input(seed=7)
    actual = mlx_block(mx.array(x))
    expected = torch_block(torch.tensor(x)).detach().numpy()
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-3, atol=2e-3)


def test_sparse_moe_shared_expert_branch_is_not_vacuous() -> None:
    # A talker block with a zeroed shared expert gate/branch must differ from
    # the full talker block, proving the shared branch actually contributes.
    mlx_block, _ = equivalent_tiny_moe_blocks(shared=True)
    x = mx.array(deterministic_input(seed=7))
    full = mlx_block(x)

    mlx_block.shared_expert.down_proj.weight = mx.zeros_like(
        mlx_block.shared_expert.down_proj.weight
    )
    routed_only = mlx_block(x)
    mx.eval(full, routed_only)
    assert not np.allclose(np.array(full), np.array(routed_only))


def test_sparse_moe_topk_normalization_matches_torch() -> None:
    normed_mlx, normed_torch = equivalent_tiny_moe_blocks(norm_topk_prob=True)
    raw_mlx, raw_torch = equivalent_tiny_moe_blocks(norm_topk_prob=False)
    x = deterministic_input(seed=3)

    normed = normed_mlx(mx.array(x))
    raw = raw_mlx(mx.array(x))
    mx.eval(normed, raw)

    normed_expected = normed_torch(torch.tensor(x)).detach().numpy()
    raw_expected = raw_torch(torch.tensor(x)).detach().numpy()

    np.testing.assert_allclose(np.array(normed), normed_expected, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.array(raw), raw_expected, rtol=2e-3, atol=2e-3)
    # Normalization must actually change the output, else the test is vacuous.
    assert not np.allclose(np.array(normed), np.array(raw))


def _reference_rope(
    positions_np: np.ndarray, q_np, k_np, *, sections=_SECTIONS, base=_THETA
):
    cfg = _thinker_text_config()
    cfg.rope_parameters["mrope_section"] = list(sections)
    cfg.rope_parameters["rope_theta"] = base
    rotary = ref.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    position_ids = torch.tensor(positions_np)[:, None, :]  # (3, bs=1, seq)
    dummy = torch.zeros(1, positions_np.shape[1], _H)
    cos, sin = rotary(dummy, position_ids)
    q_embed, k_embed = ref.apply_rotary_pos_emb(
        torch.tensor(q_np), torch.tensor(k_np), cos, sin
    )
    return q_embed.detach().numpy(), k_embed.detach().numpy()


def test_multimodal_rope_reuses_graph_with_live_positions(monkeypatch) -> None:
    rng = np.random.default_rng(81)
    q = rng.standard_normal((1, 3, 11, _HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((1, 1, 11, _HEAD_DIM)).astype(np.float32)
    interleave = mlx_common._interleave_mrope
    trace_count = 0

    def count_trace(*args):
        nonlocal trace_count
        trace_count += 1
        return interleave(*args)

    monkeypatch.setattr(mlx_common, "_interleave_mrope", count_trace)
    after_first = None
    for offset in (0, 31, 89):
        positions = np.array(
            [np.arange(11) + offset, np.arange(11) * 2, np.arange(11) * 3]
        )
        actual = apply_multimodal_rope(
            mx.array(q),
            mx.array(k),
            mx.array(positions),
            sections=_SECTIONS,
            base=_THETA,
        )
        mx.eval(actual)
        expected = _reference_rope(positions, q, k)
        for output, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                np.array(output), reference, atol=2e-5, rtol=2e-5
            )
        if after_first is None:
            after_first = trace_count
        # Repeated decode layers must not rebuild the Python selector/graph,
        # but their new external temporal/height/width positions must be used.
        assert trace_count == after_first


@pytest.mark.parametrize(
    "q_dtype,k_dtype", [(mx.float32, mx.float16), (mx.bfloat16, mx.float32)]
)
def test_multimodal_rope_specializes_config_and_input_dtypes(q_dtype, k_dtype) -> None:
    rng = np.random.default_rng(91)
    q = mx.array(rng.standard_normal((1, 4, 4, _HEAD_DIM)).astype(np.float32)).astype(
        q_dtype
    )
    k = mx.array(rng.standard_normal((1, 2, 4, _HEAD_DIM)).astype(np.float32)).astype(
        k_dtype
    )
    positions = np.array([[7, 13, 29, 61], [0, 1, 2, 3], [3, 2, 1, 0]])
    for sections, base in [
        (_SECTIONS, _THETA),
        ((0, 3, 1), 10000.0),
        (_SECTIONS, _THETA),
    ]:
        actual = apply_multimodal_rope(
            q, k, mx.array(positions), sections=sections, base=base
        )
        mx.eval(actual)
        expected = _reference_rope(
            positions,
            np.array(q.astype(mx.float32)),
            np.array(k.astype(mx.float32)),
            sections=sections,
            base=base,
        )
        for output, reference, dtype in zip(
            actual, expected, (q_dtype, k_dtype), strict=True
        ):
            assert output.dtype == dtype
            tolerance = 4e-2 if dtype == mx.bfloat16 else 5e-3
            np.testing.assert_allclose(
                np.array(output.astype(mx.float32)),
                reference,
                atol=tolerance,
                rtol=tolerance,
            )


def _reference_rope_dtype(positions_np: np.ndarray, q_np, k_np, torch_dtype):
    """Reference M-RoPE evaluated in a low-precision dtype."""

    cfg = _thinker_text_config()
    rotary = ref.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    position_ids = torch.tensor(positions_np)[:, None, :]  # (3, bs=1, seq)
    dummy = torch.zeros(1, positions_np.shape[1], _H, dtype=torch_dtype)
    cos, sin = rotary(dummy, position_ids)
    q_embed, k_embed = ref.apply_rotary_pos_emb(
        torch.tensor(q_np).to(torch_dtype),
        torch.tensor(k_np).to(torch_dtype),
        cos,
        sin,
    )
    assert q_embed.dtype == torch_dtype
    assert k_embed.dtype == torch_dtype
    return (
        q_embed.float().detach().numpy(),
        k_embed.float().detach().numpy(),
    )


@pytest.mark.parametrize(
    "mlx_dtype, torch_dtype, rtol, atol",
    [
        (mx.float16, torch.float16, 5e-3, 5e-3),
        (mx.bfloat16, torch.bfloat16, 4e-2, 4e-2),
    ],
)
def test_multimodal_rope_preserves_low_precision_dtype_and_parity(
    mlx_dtype, torch_dtype, rtol, atol
) -> None:
    # Low-precision Q/K must not be silently upcast to float32: the returned
    # tensors must keep the input dtype AND stay numerically faithful to the
    rng = np.random.default_rng(21)
    seq = 4
    positions = np.array([[0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 0, 1]], dtype=np.int64)
    q = rng.standard_normal((1, 4, seq, _HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((1, 2, seq, _HEAD_DIM)).astype(np.float32)

    q_actual, k_actual = apply_multimodal_rope(
        mx.array(q).astype(mlx_dtype),
        mx.array(k).astype(mlx_dtype),
        mx.array(positions),
        sections=_SECTIONS,
        base=_THETA,
    )
    mx.eval(q_actual, k_actual)

    assert q_actual.dtype == mlx_dtype
    assert k_actual.dtype == mlx_dtype

    q_expected, k_expected = _reference_rope_dtype(positions, q, k, torch_dtype)
    np.testing.assert_allclose(
        np.array(q_actual.astype(mx.float32)), q_expected, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        np.array(k_actual.astype(mx.float32)), k_expected, rtol=rtol, atol=atol
    )


def test_sparse_moe_expert_compute_rows_scale_with_routed_assignments() -> None:
    """Structural regression test (not numerical parity)."""

    from sglang_omni.models.qwen3_omni.mlx import common as common_module

    num_experts, top_k, num_tokens = 8, 2, 5
    mlx_block = SparseMoeBlock(
        hidden_size=_H,
        moe_intermediate_size=_INTER,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )

    processed_row_counts = []
    real_gather_mm = common_module.mx.gather_mm

    def _counting_gather_mm(a, b, *, rhs_indices, **kwargs):
        processed_row_counts.append(int(rhs_indices.size))
        return real_gather_mm(a, b, rhs_indices=rhs_indices, **kwargs)

    common_module.mx.gather_mm = _counting_gather_mm
    try:
        x = mx.array(
            np.random.default_rng(42)
            .standard_normal((1, num_tokens, _H))
            .astype(np.float32)
        )
        out = mlx_block(x)
        mx.eval(out)
    finally:
        common_module.mx.gather_mm = real_gather_mm

    assert (
        processed_row_counts
    ), "expert dispatch must route selected tokens through mx.gather_mm"

    routed_assignments = num_tokens * top_k
    all_token_all_expert_rows = num_tokens * num_experts
    for rows in processed_row_counts:
        assert rows == routed_assignments
        assert rows != all_token_all_expert_rows


def _reference_interleaved_freqs(
    positions_np: np.ndarray, sections: tuple[int, ...], head_dim: int
) -> np.ndarray:
    """``apply_interleaved_mrope`` from the Transformers reference, verbatim."""

    inv_freq = 1.0 / (
        _THETA ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    freqs = torch.tensor(
        positions_np.astype(np.float64)[:, :, None] * inv_freq[None, None, :]
    )
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = sections[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim][..., idx]
    return freqs_t.numpy()


def _mlx_interleaved_freqs(
    positions_np: np.ndarray, sections: tuple[int, ...], head_dim: int
) -> np.ndarray:
    """Drive ``_interleave_mrope`` exactly as ``apply_multimodal_rope`` does."""

    inv_freq = 1.0 / (
        _THETA ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    freqs = mx.array(
        positions_np.astype(np.float64)[:, :, None] * inv_freq[None, None, :]
    ).astype(mx.float32)
    out = _interleave_mrope(freqs, sections)
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize(
    ("head_dim", "sections"),
    [
        # Production Qwen3-Omni thinker width: head_dim 128, mrope_section
        # [24, 20, 20]; the widest interleaved column (59) stays inside 64.
        (128, (24, 20, 20)),
        # A section whose interleaved columns run past head_dim // 2. The
        # reference clamps the slice; column-index arithmetic overruns instead.
        (128, (16, 24, 24)),
        # Exactly on the boundary: the last width column is head_dim // 2 - 1.
        (128, (22, 21, 21)),
        # Height overruns while width does not.
        (64, (8, 16, 8)),
    ],
)
def test_interleaved_mrope_columns_match_the_clamped_reference(
    head_dim: int, sections: tuple[int, ...]
) -> None:
    positions = np.array([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=np.int64)

    actual = _mlx_interleaved_freqs(positions, sections, head_dim)
    expected = _reference_interleaved_freqs(positions, sections, head_dim)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_interleaved_mrope_still_selects_all_three_axes_at_production_width() -> None:
    """Pairing control for the clamping test: the production section widths must
    genuinely mix all three axis rows, so the parity above is not vacuous."""

    head_dim = 128
    positions = np.array([[0, 1], [10, 11], [20, 21]], dtype=np.int64)
    interleaved = _mlx_interleaved_freqs(positions, (24, 20, 20), head_dim)
    inv_freq = 1.0 / (
        _THETA ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    per_axis = positions.astype(np.float64)[:, :, None] * inv_freq[None, None, :]

    assert np.allclose(interleaved[:, 0], per_axis[0][:, 0])
    assert np.allclose(interleaved[:, 1], per_axis[1][:, 1])
    assert np.allclose(interleaved[:, 2], per_axis[2][:, 2])
    # Past 3 * 20 the reference has no height/width columns left to write.
    assert np.allclose(interleaved[:, 61], per_axis[0][:, 61])


def test_multimodal_rope_three_axis_selection_matches_torch() -> None:
    rng = np.random.default_rng(11)
    seq = 4
    # Distinct temporal / height / width rows so the interleaved section
    # selection is genuinely exercised.
    positions = np.array([[0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 0, 1]], dtype=np.int64)
    q = rng.standard_normal((1, 4, seq, _HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((1, 2, seq, _HEAD_DIM)).astype(np.float32)

    q_expected, k_expected = _reference_rope(positions, q, k)

    q_actual, k_actual = apply_multimodal_rope(
        mx.array(q),
        mx.array(k),
        mx.array(positions),
        sections=_SECTIONS,
        base=_THETA,
    )
    mx.eval(q_actual, k_actual)
    np.testing.assert_allclose(np.array(q_actual), q_expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(k_actual), k_expected, rtol=1e-4, atol=1e-4)


def test_multimodal_rope_axis_rows_are_not_interchangeable() -> None:
    # Swapping the height and width position rows must change the result,
    # proving the three axes map to distinct rotary sections.
    rng = np.random.default_rng(12)
    seq = 4
    q = rng.standard_normal((1, 4, seq, _HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((1, 2, seq, _HEAD_DIM)).astype(np.float32)
    positions = np.array([[0, 1, 2, 3], [0, 0, 1, 2], [0, 5, 6, 7]], dtype=np.int64)
    swapped = positions[[0, 2, 1], :]

    q1, _ = apply_multimodal_rope(
        mx.array(q), mx.array(k), mx.array(positions), sections=_SECTIONS, base=_THETA
    )
    q2, _ = apply_multimodal_rope(
        mx.array(q), mx.array(k), mx.array(swapped), sections=_SECTIONS, base=_THETA
    )
    mx.eval(q1, q2)
    assert not np.allclose(np.array(q1), np.array(q2))


def test_sanitize_strips_thinker_prefix_and_drops_other_components() -> None:
    weights = {
        "thinker.model.embed_tokens.weight": mx.zeros((8, 4)),
        "thinker.model.layers.0.mlp.experts.gate_up_proj": mx.zeros((4, 8, 4)),
        "talker.model.embed_tokens.weight": mx.zeros((8, 4)),
        "code2wav.code_embedding.weight": mx.zeros((8, 4)),
    }
    sanitized = sanitize_qwen3_omni_weights(weights, component="thinker")

    assert "model.embed_tokens.weight" in sanitized
    # The fused expert stack lands on its quantizable SwitchLinear weight name.
    assert "model.layers.0.mlp.experts.gate_up_proj.weight" in sanitized
    assert not any(key.startswith("talker.") for key in sanitized)
    assert not any(key.startswith("code2wav.") for key in sanitized)
    assert not any(key.startswith("thinker.") for key in sanitized)


def test_sanitize_normalizes_language_model_thinker_namespace() -> None:
    weights = {
        "thinker.language_model.model.embed_tokens.weight": mx.zeros((8, 4)),
        "thinker.language_model.lm_head.weight": mx.zeros((8, 4)),
        "thinker.audio_tower.layers.0.fc1.weight": mx.zeros((4, 4)),
    }

    sanitized = sanitize_qwen3_omni_weights(weights, component="thinker")

    assert set(sanitized) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "audio_tower.layers.0.fc1.weight",
    }


def test_sanitize_strips_talker_prefix() -> None:
    weights = {
        "talker.model.layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
        "thinker.model.embed_tokens.weight": mx.zeros((8, 4)),
    }
    sanitized = sanitize_qwen3_omni_weights(weights, component="talker")
    assert "model.layers.0.self_attn.q_proj.weight" in sanitized
    assert len(sanitized) == 1


def test_sanitize_transposes_conv2d_weight_only_for_hf_layout() -> None:
    hf_conv = np.arange(1 * 3 * 3 * 3, dtype=np.float32).reshape(1, 3, 3, 3)
    expert = np.arange(4 * 8 * 4, dtype=np.float32).reshape(4, 8, 4)
    weights = {
        "thinker.audio_tower.conv2d1.weight": mx.array(hf_conv),
        "thinker.model.layers.0.mlp.experts.gate_up_proj": mx.array(expert),
    }
    sanitized = sanitize_qwen3_omni_weights(weights, component="thinker")

    # HF Conv2d weight (out, in, kh, kw) -> MLX (out, kh, kw, in).
    assert sanitized["audio_tower.conv2d1.weight"].shape == (1, 3, 3, 3)
    np.testing.assert_array_equal(
        np.array(sanitized["audio_tower.conv2d1.weight"]),
        hf_conv.transpose(0, 2, 3, 1),
    )
    # Rank-3 expert stacks must NOT be transposed like convolution kernels.
    np.testing.assert_array_equal(
        np.array(sanitized["model.layers.0.mlp.experts.gate_up_proj.weight"]), expert
    )


def test_sanitize_does_not_transpose_already_converted_layout() -> None:
    conv = np.arange(1 * 3 * 3 * 3, dtype=np.float32).reshape(1, 3, 3, 3)
    weights = {"audio_tower.conv2d1.weight": mx.array(conv)}
    sanitized = sanitize_qwen3_omni_weights(weights, component="thinker")
    np.testing.assert_array_equal(
        np.array(sanitized["audio_tower.conv2d1.weight"]), conv
    )


def test_tie_lm_head_weights_shares_the_same_array_object() -> None:
    embed = nn.Embedding(16, 8)
    lm_head = nn.Linear(8, 16, bias=False)
    tie_lm_head_weights(lm_head, embed)

    # A genuine module-level tie: both modules reference the *same* array object,
    # not a one-time numeric copy.
    assert lm_head.weight is embed.weight

    # The single shared weight drives both the projection and the embedding's
    # ``as_linear`` readout identically.
    x = mx.ones((1, 8))
    logits = lm_head(x)
    reference = embed.as_linear(x)
    mx.eval(logits, reference)
    np.testing.assert_array_equal(np.array(logits), np.array(reference))

    flat = dict(
        tree_flatten({"embed": embed.parameters(), "head": lm_head.parameters()})
    )
    assert flat["embed.weight"] is flat["head.weight"]


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 32, bias=False)
        self.b = nn.Linear(64, 32, bias=False)


class TinyQuantizableModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


def _quantized_reference_weights() -> dict:
    ref_module = _TwoLinear()
    nn.quantize(
        ref_module, group_size=32, bits=4, class_predicate=lambda p, m: p == "a"
    )
    return dict(tree_flatten(ref_module.parameters()))


def _fixture_quantization_config() -> QuantizationConfig:
    """Parse quantization metadata exactly as a converted checkpoint carries it."""

    return QuantizationConfig.from_dict({"bits": 4, "group_size": 32})


def test_quantize_only_layers_with_scales_before_load() -> None:
    weights = _quantized_reference_weights()
    assert "a.scales" in weights and "a.biases" in weights
    assert "b.scales" not in weights

    model = _TwoLinear()
    quantize_converted_module(
        model, weights, quantization=_fixture_quantization_config()
    )

    assert isinstance(model.a, nn.QuantizedLinear)
    assert isinstance(model.b, nn.Linear)
    assert not isinstance(model.b, nn.QuantizedLinear)

    # Quantization must precede loading: the converted (packed) weights load
    # cleanly only because the module type already matches.
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())


def test_component_loader_rejects_incomplete_affine_group() -> None:
    loader = getattr(mlx_common, "load_qwen3_omni_mlx_component", None)
    assert loader is not None

    model = TinyQuantizableModule()
    weights = {
        "linear.weight": mx.zeros((2, 2), dtype=mx.uint32),
        "linear.scales": mx.ones((2, 1)),
    }

    with pytest.raises(ValueError, match="biases"):
        loader(
            model,
            weights,
            sanitizer=lambda values: dict(values),
            quantization=QuantizationConfig(bits=4, group_size=64),
        )


def test_quantize_incomplete_group_raises_with_full_layer_path() -> None:
    weights = _quantized_reference_weights()
    del weights["a.biases"]

    model = _TwoLinear()
    with pytest.raises(ValueError) as excinfo:
        quantize_converted_module(
            model, weights, quantization=_fixture_quantization_config()
        )
    assert "a" in str(excinfo.value)
    # No silent dense fallback: the error names the offending layer explicitly.
    assert "biases" in str(excinfo.value)


def test_quantize_biases_without_scales_raises_with_full_layer_path() -> None:
    # The opposite incomplete affine direction -- biases present but scales
    # missing -- must also be rejected loudly (no dense fallback), naming the
    weights = _quantized_reference_weights()
    del weights["a.scales"]
    assert "a.biases" in weights

    model = _TwoLinear()
    with pytest.raises(ValueError) as excinfo:
        quantize_converted_module(
            model, weights, quantization=_fixture_quantization_config()
        )
    assert "a" in str(excinfo.value)
    assert "scales" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Quantized MoE expert stacks: SwitchLinear -> QuantizedSwitchLinear


def _seeded_moe_block(seed: int = 99) -> SparseMoeBlock:
    rng = np.random.default_rng(seed)
    block = SparseMoeBlock(
        hidden_size=_H,
        moe_intermediate_size=_INTER,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOPK,
        norm_topk_prob=True,
    )
    block.experts.gate_up_proj.weight = mx.array(
        rng.standard_normal((_EXPERTS, 2 * _INTER, _H)).astype(np.float32)
    )
    block.experts.down_proj.weight = mx.array(
        rng.standard_normal((_EXPERTS, _H, _INTER)).astype(np.float32)
    )
    block.gate.weight = mx.array(rng.standard_normal((_EXPERTS, _H)).astype(np.float32))
    return block


def _converted_4bit_moe_export() -> tuple[SparseMoeBlock, dict[str, mx.array]]:
    """A quantized reference block plus the checkpoint a converter would ship."""

    reference = _seeded_moe_block()
    nn.quantize(reference, group_size=32, bits=4)
    published: dict[str, mx.array] = {}
    for key, value in tree_flatten(reference.parameters()):
        if key.endswith(".weight") and ".experts." in key:
            key = key[: -len(".weight")]
        published[key] = value
    return reference, published


def test_converted_expert_stacks_load_as_quantized_switch_linear() -> None:
    """The fused MoE stacks must become ``QuantizedSwitchLinear`` before load and"""

    reference, published = _converted_4bit_moe_export()
    sanitized = sanitize_qwen3_omni_weights(published, component="thinker")
    assert "experts.gate_up_proj.weight" in sanitized
    assert "experts.gate_up_proj.scales" in sanitized
    assert "experts.down_proj.biases" in sanitized

    model = _seeded_moe_block()
    assert isinstance(model.experts.gate_up_proj, SwitchLinear)
    assert not isinstance(model.experts.gate_up_proj, QuantizedSwitchLinear)

    quantize_converted_module(
        model, sanitized, quantization=_fixture_quantization_config()
    )

    assert isinstance(model.experts.gate_up_proj, QuantizedSwitchLinear)
    assert isinstance(model.experts.down_proj, QuantizedSwitchLinear)

    model.load_weights(list(sanitized.items()))
    mx.eval(model.parameters())

    x = mx.array(deterministic_input(5))
    actual = model(x)
    expected = reference(x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_mlx_vlm_split_expert_stacks_fuse_and_load_exactly() -> None:
    reference = _seeded_moe_block()
    nn.quantize(reference, group_size=32, bits=4)
    reference_weights = dict(tree_flatten(reference.parameters()))
    prefix = "talker.model.layers.0.mlp"
    published: dict[str, mx.array] = {}
    for suffix in ("weight", "scales", "biases"):
        gate_up = reference_weights[f"experts.gate_up_proj.{suffix}"]
        gate, up = mx.split(gate_up, 2, axis=1)
        published[f"{prefix}.switch_mlp.gate_proj.{suffix}"] = gate
        published[f"{prefix}.switch_mlp.up_proj.{suffix}"] = up
        published[f"{prefix}.switch_mlp.down_proj.{suffix}"] = reference_weights[
            f"experts.down_proj.{suffix}"
        ]
        published[f"{prefix}.gate.{suffix}"] = reference_weights[f"gate.{suffix}"]

    sanitized = sanitize_qwen3_omni_weights(published, component="talker")
    layer_prefix = "model.layers.0.mlp."
    block_weights = {
        key[len(layer_prefix) :]: value
        for key, value in sanitized.items()
        if key.startswith(layer_prefix)
    }

    assert set(block_weights) == {
        "experts.gate_up_proj.weight",
        "experts.gate_up_proj.scales",
        "experts.gate_up_proj.biases",
        "experts.down_proj.weight",
        "experts.down_proj.scales",
        "experts.down_proj.biases",
        "gate.weight",
        "gate.scales",
        "gate.biases",
    }

    model = _seeded_moe_block(seed=1)
    quantize_converted_module(
        model, block_weights, quantization=_fixture_quantization_config()
    )
    model.load_weights(list(block_weights.items()))
    x = mx.array(deterministic_input(8))
    actual = model(x)
    expected = reference(x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_converted_expert_scales_are_genuinely_consumed() -> None:
    """Pairing control: perturbing only the expert ``scales`` must change the
    output, so the loaded quantized stack is not silently ignoring them."""

    _, published = _converted_4bit_moe_export()
    sanitized = sanitize_qwen3_omni_weights(published, component="thinker")

    baseline = _seeded_moe_block()
    quantize_converted_module(
        baseline, sanitized, quantization=_fixture_quantization_config()
    )
    baseline.load_weights(list(sanitized.items()))

    perturbed_weights = dict(sanitized)
    perturbed_weights["experts.gate_up_proj.scales"] = (
        perturbed_weights["experts.gate_up_proj.scales"] * 2.0
    )
    perturbed = _seeded_moe_block()
    quantize_converted_module(
        perturbed, perturbed_weights, quantization=_fixture_quantization_config()
    )
    perturbed.load_weights(list(perturbed_weights.items()))

    x = mx.array(deterministic_input(6))
    base_out = baseline(x)
    perturbed_out = perturbed(x)
    mx.eval(base_out, perturbed_out)
    assert not np.allclose(np.array(base_out), np.array(perturbed_out))


def test_dense_expert_stacks_stay_switch_linear() -> None:
    """Pairing control: a dense export carries no expert ``scales``, so the
    stacks must stay dense ``SwitchLinear`` and still load."""

    dense = _seeded_moe_block()
    published = {
        (
            key[: -len(".weight")]
            if key.endswith(".weight") and ".experts." in key
            else key
        ): value
        for key, value in tree_flatten(dense.parameters())
    }
    sanitized = sanitize_qwen3_omni_weights(published, component="thinker")

    model = _seeded_moe_block(seed=1)
    quantize_converted_module(
        model, sanitized, quantization=_fixture_quantization_config()
    )

    assert not isinstance(model.experts.gate_up_proj, QuantizedSwitchLinear)
    model.load_weights(list(sanitized.items()))

    x = mx.array(deterministic_input(7))
    actual = model(x)
    expected = dense(x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_incomplete_quantized_expert_group_names_the_expert_stack() -> None:
    _, published = _converted_4bit_moe_export()
    sanitized = sanitize_qwen3_omni_weights(published, component="thinker")
    del sanitized["experts.gate_up_proj.biases"]

    model = _seeded_moe_block()
    with pytest.raises(ValueError) as excinfo:
        quantize_converted_module(
            model, sanitized, quantization=_fixture_quantization_config()
        )
    assert "experts.gate_up_proj" in str(excinfo.value)
    assert "biases" in str(excinfo.value)
