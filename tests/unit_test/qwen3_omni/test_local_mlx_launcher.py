# SPDX-License-Identifier: Apache-2.0
"""The local launcher must reject missing assets before starting workers."""

import os
import subprocess
import sys
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[3] / "scripts/run_qwen3_omni_mlx.sh"


def _launch(**overrides):
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        env={**os.environ, **overrides},
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_missing_python_reports_environment_path(tmp_path):
    missing = str(tmp_path / "missing-python")
    result = _launch(MLX_PYTHON=missing)
    assert result.returncode != 0
    assert f"Python executable not found: {missing}" in result.stderr


def test_missing_sglang_source_reports_required_api(tmp_path):
    result = _launch(MLX_PYTHON=sys.executable, SGLANG_SOURCE=str(tmp_path))
    assert result.returncode != 0
    assert "SGLANG_SOURCE must contain" in result.stderr
    assert "sglang/srt/arg_groups/model_override_base.py" in result.stderr


def test_missing_model_reports_checkpoint_path(tmp_path):
    source = tmp_path / "source"
    api = source / "sglang/srt/arg_groups/model_override_base.py"
    api.parent.mkdir(parents=True)
    api.touch()
    model = str(tmp_path / "missing-model")
    result = _launch(
        MLX_PYTHON=sys.executable,
        SGLANG_SOURCE=str(source),
        MODEL_DIR=model,
    )
    assert result.returncode != 0
    assert f"Checkpoint config not found: {model}/config.json" in result.stderr
