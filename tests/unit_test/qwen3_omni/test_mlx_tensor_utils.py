# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_omni.mlx.tensor_utils import (
    mlx_to_torch,
    torch_to_mlx,
)


def test_torch_to_mlx_handles_bfloat16_at_numpy_boundary() -> None:
    tensor = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)

    converted = torch_to_mlx(tensor)

    assert converted.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(converted), tensor.float().numpy())


def test_mlx_to_torch_returns_contiguous_float32_cpu_tensor() -> None:
    converted = mlx_to_torch(mx.array([[1.0, 2.0]], dtype=mx.float16))

    assert converted.device == torch.device("cpu")
    assert converted.dtype == torch.float32
    assert converted.is_contiguous()
