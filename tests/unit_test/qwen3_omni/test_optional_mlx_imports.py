# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "export"),
    [
        (
            "sglang_omni.models.qwen3_omni.components.audio_encoder",
            "Qwen3OmniAudioEncoder",
        ),
        (
            "sglang_omni.models.qwen3_omni.components.image_encoder",
            "Qwen3OmniImageEncoder",
        ),
        (
            "sglang_omni.models.qwen3_omni.stages",
            "create_image_encoder_executor",
        ),
        (
            "sglang_omni.models.qwen3_omni.stages",
            "create_audio_encoder_executor",
        ),
        ("sglang_omni.models.qwen3_omni.mlx", "QuantizationConfig"),
    ],
)
def test_shared_modules_import_without_mlx(module_name: str, export: str) -> None:
    script = """
import importlib
import importlib.abc
import importlib.util
import sys

original_find_spec = importlib.util.find_spec

def find_spec_without_mlx(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx."):
        return None
    return original_find_spec(name, *args, **kwargs)

class NoMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ModuleNotFoundError(
                "MLX is unavailable in this non-Apple installation", name=fullname
            )

importlib.util.find_spec = find_spec_without_mlx
sys.meta_path.insert(0, NoMlx())
module = importlib.import_module(sys.argv[1])
assert getattr(module, sys.argv[2]) is not None
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
    env = dict(os.environ)
    env.pop("SGLANG_USE_MLX", None)
    completed = subprocess.run(
        [sys.executable, "-c", script, module_name, export],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_mlx_test_modules_collect_cleanly_without_mlx() -> None:
    script = """
import importlib.abc
import importlib.util
import sys

original_find_spec = importlib.util.find_spec

def find_spec_without_mlx(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx."):
        return None
    return original_find_spec(name, *args, **kwargs)

class NoMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ModuleNotFoundError(
                "MLX is unavailable in this non-Apple installation", name=fullname
            )

importlib.util.find_spec = find_spec_without_mlx
sys.meta_path.insert(0, NoMlx())

import pytest

exit_code = pytest.main(["--collect-only", "-q", *sys.argv[1:]])
assert exit_code in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED), exit_code
"""
    test_dir = Path(__file__).resolve().parent
    test_modules = [str(path) for path in sorted(test_dir.glob("test_mlx_*.py"))]
    env = dict(os.environ)
    env.pop("SGLANG_USE_MLX", None)

    completed = subprocess.run(
        [sys.executable, "-c", script, *test_modules],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
