# SPDX-License-Identifier: Apache-2.0
"""Claim-to-execution checks for the mps_dp weight-share support registry.

Every model advertised as weight-share supported must resolve through the real
launcher preflight, every unsupported topology must fail before any resource
is created, and the docs table must agree with the code registries.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from sglang_omni.utils import ipc_weights

REPO_ROOT = Path(__file__).resolve().parents[3]
MPS_DP_DIR = REPO_ROOT / "examples" / "mps_dp"
DOCS_PAGE = REPO_ROOT / "docs" / "basic_usage" / "mps_dp.md"

if "transformers" not in sys.modules:
    transformers = ModuleType("transformers")
    transformers.__path__ = []

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("stub AutoConfig is unavailable in unit tests")

    def _no_init_weights(*args, **kwargs):
        raise NotImplementedError

    hub = ModuleType("transformers.utils.hub")
    hub.cached_file = lambda *args, **kwargs: None
    utils = ModuleType("transformers.utils")
    utils.hub = hub
    initialization = ModuleType("transformers.initialization")
    initialization.no_init_weights = _no_init_weights
    modeling_utils = ModuleType("transformers.modeling_utils")
    modeling_utils.no_init_weights = _no_init_weights
    transformers.AutoConfig = _AutoConfig
    transformers.utils = utils
    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = utils
    sys.modules["transformers.utils.hub"] = hub
    sys.modules["transformers.initialization"] = initialization
    sys.modules["transformers.modeling_utils"] = modeling_utils

_spec = importlib.util.spec_from_file_location(
    "mps_dp_config", MPS_DP_DIR / "config.py"
)
mps_dp_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mps_dp_config)

CONFIG_SPECS = {
    "HiggsTtsPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "MossTTSLocalPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "MossTTSPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "MossTranscribeDiarizePipelineConfig": {
        "stage_name": "asr",
        "mem_fraction_roles": {"asr": "asr"},
        "talker_roles": {},
        "generation_roles": {"generation": "asr"},
    },
    "Qwen3ASRPipelineConfig": {
        "stage_name": "asr",
        "mem_fraction_roles": {"asr": "asr"},
        "talker_roles": {},
        "generation_roles": {"generation": "asr"},
    },
    "WhisperASRPipelineConfig": {
        "stage_name": "asr",
        "mem_fraction_roles": {"asr": "asr"},
        "talker_roles": {},
        "generation_roles": {"generation": "asr"},
    },
    "FunASRPipelineConfig": {
        "stage_name": "asr",
        "mem_fraction_roles": {"asr": "asr"},
        "talker_roles": {},
        "generation_roles": {"generation": "asr"},
    },
    "VoxtralTTSPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "Qwen3TTSPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "MingTTSPipelineConfig": {
        "stage_name": "tts_engine",
        "mem_fraction_roles": {"talker": "tts_engine"},
        "talker_roles": {"talker": "tts_engine"},
        "generation_roles": {"generation": "tts_engine"},
    },
    "Qwen3OmniSpeechPipelineConfig": {
        "stage_name": "thinker",
        "mem_fraction_roles": {"thinker": "thinker"},
        "talker_roles": {"talker": "talker"},
        "generation_roles": {"generation": "thinker"},
        "extra_stage_names": ("talker",),
    },
    "LLaDA2UniPipelineConfig": {
        "stage_name": None,
        "mem_fraction_roles": {},
        "talker_roles": {},
        "generation_roles": {},
    },
    "Qwen3OmniPipelineConfig": {
        "stage_name": None,
        "mem_fraction_roles": {},
        "talker_roles": {},
        "generation_roles": {},
    },
}
VALIDATED_CONFIG_CLASS_NAMES = tuple(mps_dp_config.WEIGHT_SHARE_VALIDATED_CONFIGS)


def _parse_bytes(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    multipliers = {
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            return int(value[: -len(suffix)]) * multiplier
    raise ValueError(f"unsupported byte literal {value!r}")


def _build_fake_pipeline_config(
    config_cls: str,
    *,
    max_total_tokens: int | None = None,
    kv_cache_bytes: int | None = None,
    total_reserve_bytes: int | None = None,
):
    spec = CONFIG_SPECS[config_cls]
    stage_name = spec["stage_name"]
    stages = []
    if stage_name is not None:
        stage_names = (stage_name, *spec.get("extra_stage_names", ()))
        for current_name in stage_names:
            stages.append(
                SimpleNamespace(
                    name=current_name,
                    runtime=SimpleNamespace(
                        memory=SimpleNamespace(
                            kv_cache_bytes=(
                                kv_cache_bytes if current_name == stage_name else None
                            ),
                            total_reserve_bytes=(
                                total_reserve_bytes
                                if current_name == stage_name
                                else None
                            ),
                        )
                    ),
                    max_total_tokens=(
                        max_total_tokens if current_name == stage_name else None
                    ),
                )
            )

    config_type = type(
        config_cls,
        (),
        {
            "mem_fraction_role_to_stage": classmethod(
                lambda cls: dict(spec["mem_fraction_roles"])
            ),
            "talker_sglang_role_to_stage": classmethod(
                lambda cls: dict(spec["talker_roles"])
            ),
            "generation_sglang_role_to_stage": classmethod(
                lambda cls: dict(spec["generation_roles"])
            ),
        },
    )
    pipeline_config = config_type()
    pipeline_config.stages = stages
    return pipeline_config


def _fake_config_manager_from_file(file_path: str):
    data = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
    config_cls = data["config_cls"]
    spec = CONFIG_SPECS[config_cls]
    stage_name = spec["stage_name"]
    max_total_tokens = None
    if stage_name is not None:
        max_total_tokens = (
            ((data.get("runtime_overrides") or {}).get(stage_name) or {})
            .get("server_args_overrides", {})
            .get("max_total_tokens")
        )
    memory = {}
    if stage_name is not None:
        memory = (
            (((data.get("stage_overrides") or {}).get(stage_name) or {}).get("runtime"))
            or {}
        ).get("memory", {})
    return SimpleNamespace(
        config=_build_fake_pipeline_config(
            config_cls,
            max_total_tokens=max_total_tokens,
            kv_cache_bytes=_parse_bytes(memory.get("kv_cache_bytes")),
            total_reserve_bytes=_parse_bytes(memory.get("total_reserve_bytes")),
        )
    )


def _fake_resolve_stage_static_factory_args(stage, _pipeline_config):
    if getattr(stage, "max_total_tokens", None) is None:
        return {}
    return {"server_args_overrides": {"max_total_tokens": stage.max_total_tokens}}


@pytest.fixture(autouse=True)
def _fake_launcher_config_resolution(monkeypatch):
    monkeypatch.setattr(
        mps_dp_config,
        "ConfigManager",
        SimpleNamespace(from_file=_fake_config_manager_from_file),
    )
    monkeypatch.setattr(
        mps_dp_config,
        "resolve_stage_static_factory_args",
        _fake_resolve_stage_static_factory_args,
    )


def _write_yaml(tmp_path: Path, config_cls: str, name: str = "probe") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"config_cls: {config_cls}\nname: {name}\nmodel_path: dummy/none\n",
        encoding="utf-8",
    )
    return path


def _write_budget_yaml(
    tmp_path: Path,
    *,
    kv_cache_bytes: str,
    total_reserve_bytes: str,
    name: str = "budget-probe",
) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        "\n".join(
            [
                "config_cls: Qwen3ASRPipelineConfig",
                f"name: {name}",
                "model_path: dummy/none",
                "stage_overrides:",
                "  asr:",
                "    runtime:",
                "      memory:",
                f"        kv_cache_bytes: {kv_cache_bytes}",
                f"        total_reserve_bytes: {total_reserve_bytes}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_gpu_info_wrapper(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "python_with_mps_stubs.py"
    config_specs = repr(CONFIG_SPECS)
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "",
                "import json",
                "import os",
                "import runpy",
                "import sys",
                "from pathlib import Path",
                "from types import ModuleType, SimpleNamespace",
                "",
                "import yaml",
                "",
                f"sys.path.insert(0, {str(REPO_ROOT)!r})",
                "",
                "script = sys.argv[1] if len(sys.argv) > 1 else ''",
                "if script.endswith(os.path.join('examples', 'mps_dp', 'config.py')):",
                f"    CONFIG_SPECS = {config_specs}",
                "    def _parse_bytes(value):",
                "        if value is None or isinstance(value, int):",
                "            return value",
                "        multipliers = {",
                "            'KiB': 1024,",
                "            'MiB': 1024**2,",
                "            'GiB': 1024**3,",
                "            'TiB': 1024**4,",
                "        }",
                "        for suffix, multiplier in multipliers.items():",
                "            if value.endswith(suffix):",
                "                return int(value[:-len(suffix)]) * multiplier",
                "        raise ValueError(f'unsupported byte literal {value!r}')",
                "",
                "    def _build_fake_pipeline_config(",
                "        config_cls, max_total_tokens=None, kv_cache_bytes=None,",
                "        total_reserve_bytes=None,",
                "    ):",
                "        spec = CONFIG_SPECS[config_cls]",
                "        stage_name = spec['stage_name']",
                "        stages = []",
                "        if stage_name is not None:",
                "            stage_names = (stage_name, *spec.get('extra_stage_names', ()))",
                "            for current_name in stage_names:",
                "                stages.append(",
                "                    SimpleNamespace(",
                "                        name=current_name,",
                "                        runtime=SimpleNamespace(",
                "                            memory=SimpleNamespace(",
                "                                kv_cache_bytes=(",
                "                                    kv_cache_bytes if current_name == stage_name else None",
                "                                ),",
                "                                total_reserve_bytes=(",
                "                                    total_reserve_bytes if current_name == stage_name else None",
                "                                ),",
                "                            )",
                "                        ),",
                "                        max_total_tokens=(",
                "                            max_total_tokens if current_name == stage_name else None",
                "                        ),",
                "                    )",
                "                )",
                "        config_type = type(",
                "            config_cls,",
                "            (),",
                "            {",
                "                'mem_fraction_role_to_stage': classmethod(",
                "                    lambda cls: dict(spec['mem_fraction_roles'])",
                "                ),",
                "                'talker_sglang_role_to_stage': classmethod(",
                "                    lambda cls: dict(spec['talker_roles'])",
                "                ),",
                "                'generation_sglang_role_to_stage': classmethod(",
                "                    lambda cls: dict(spec['generation_roles'])",
                "                ),",
                "            },",
                "        )",
                "        pipeline_config = config_type()",
                "        pipeline_config.stages = stages",
                "        return pipeline_config",
                "",
                "    class _FakeConfigManager:",
                "        @staticmethod",
                "        def from_file(file_path):",
                "            data = yaml.safe_load(Path(file_path).read_text(encoding='utf-8'))",
                "            config_cls = data['config_cls']",
                "            spec = CONFIG_SPECS[config_cls]",
                "            stage_name = spec['stage_name']",
                "            max_total_tokens = None",
                "            if stage_name is not None:",
                "                max_total_tokens = (",
                "                    ((data.get('runtime_overrides') or {}).get(stage_name) or {})",
                "                    .get('server_args_overrides', {})",
                "                    .get('max_total_tokens')",
                "                )",
                "            memory = {}",
                "            if stage_name is not None:",
                "                memory = (",
                "                    (((data.get('stage_overrides') or {}).get(stage_name) or {}).get('runtime'))",
                "                    or {}",
                "                ).get('memory', {})",
                "            return SimpleNamespace(",
                "                config=_build_fake_pipeline_config(",
                "                    config_cls,",
                "                    max_total_tokens=max_total_tokens,",
                "                    kv_cache_bytes=_parse_bytes(memory.get('kv_cache_bytes')),",
                "                    total_reserve_bytes=_parse_bytes(memory.get('total_reserve_bytes')),",
                "                )",
                "            )",
                "",
                "    def _fake_resolve_stage_static_factory_args(stage, _pipeline_config):",
                "        if getattr(stage, 'max_total_tokens', None) is None:",
                "            return {}",
                "        return {'server_args_overrides': {'max_total_tokens': stage.max_total_tokens}}",
                "",
                "    config_manager = ModuleType('sglang_omni.config.manager')",
                "    config_manager.ConfigManager = _FakeConfigManager",
                "    config_runtime = ModuleType('sglang_omni.config.runtime')",
                "    config_runtime.resolve_stage_static_factory_args = _fake_resolve_stage_static_factory_args",
                "    sys.modules['sglang_omni.config.manager'] = config_manager",
                "    sys.modules['sglang_omni.config.runtime'] = config_runtime",
                "",
                "    record_path = os.environ.get('MPS_DP_RECORD_CONFIG_INVOCATIONS')",
                "    if record_path:",
                "        with Path(record_path).open('a', encoding='utf-8') as handle:",
                "            handle.write(json.dumps({",
                "                'argv': sys.argv[1:],",
                "                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),",
                "            }) + '\\n')",
                "",
                "    import sglang_omni.utils.gpu_memory as gpu_memory",
                "    from sglang_omni.utils.gpu_memory import GpuDeviceInfo",
                "",
                "    total_memory_bytes = os.environ.get('MPS_DP_FAKE_TOTAL_MEMORY_BYTES')",
                "    gpu_name = os.environ.get('MPS_DP_FAKE_GPU_NAME', 'RTX 4070 Ti')",
                "    if total_memory_bytes is not None:",
                "        gpu_memory.get_gpu_device_info = lambda gpu_id: GpuDeviceInfo(",
                "            logical_gpu_id=gpu_id,",
                "            device_id=gpu_id,",
                "            name=gpu_name,",
                "            total_memory_bytes=int(total_memory_bytes),",
                "        )",
                "    sys.argv = sys.argv[1:]",
                "    runpy.run_path(script, run_name='__main__')",
                "else:",
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_nvidia_smi(tmp_path: Path, *, uuid: str = "GPU-physical-0") -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path = bin_dir / "nvidia-smi"
    path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                'if [ "$1" = "--query-gpu=uuid" ] && [ "$2" = "--format=csv,noheader" ] && [ "$3" = "-i" ] && [ "$4" = "0" ]; then',
                f"  printf '%s\\n' '{uuid}'",
                "  exit 0",
                "fi",
                'echo "unexpected nvidia-smi args: $*" >&2',
                "exit 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return bin_dir


def test_registry_matches_weight_share_policies():
    # Note (Jiaxin Deng): implications, not an identity: every validated
    # config's architecture must be gate-enabled and never audit-only, and
    # (under today's binary support rule) every gate-enabled architecture must
    # have at least one validated config. One architecture may back several
    # configs.
    registry = mps_dp_config.WEIGHT_SHARE_VALIDATED_CONFIGS
    assert set(registry) == set(VALIDATED_CONFIG_CLASS_NAMES)
    assert set(registry.values()) <= set(ipc_weights.WEIGHT_SHARE_POLICIES)
    assert set(ipc_weights.WEIGHT_SHARE_POLICIES) <= set(registry.values())
    assert not set(registry.values()) & set(
        ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES
    )


def test_every_validated_config_class_has_a_drivable_topology():
    for name in VALIDATED_CONFIG_CLASS_NAMES:
        cls = type(_build_fake_pipeline_config(name))
        stage = cls.generation_sglang_role_to_stage().get("generation")
        assert stage, f"{name} declares no generation stage"
        union = {
            *cls.mem_fraction_role_to_stage().values(),
            *cls.talker_sglang_role_to_stage().values(),
            *cls.generation_sglang_role_to_stage().values(),
        }
        assert union == {stage}, f"{name} is not a single-SGLang-engine pipeline"


def test_every_shipped_example_config_resolves_with_sharing(tmp_path):
    yamls = sorted((MPS_DP_DIR / "configs").glob("*.yaml"))
    assert yamls, "no example configs shipped"
    for yaml_path in yamls:
        config_cls = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["config_cls"]
        if config_cls not in VALIDATED_CONFIG_CLASS_NAMES:
            continue
        value = mps_dp_config.resolve_max_total_tokens(
            yaml_path, require_single_sglang_engine=True, weight_share=True
        )
        assert (
            isinstance(value, int) and value > 0
        ), f"{yaml_path.name} does not pin a positive max_total_tokens"


@pytest.mark.parametrize(
    "config_cls", ["LLaDA2UniPipelineConfig", "Qwen3OmniPipelineConfig"]
)
def test_pipelines_without_generation_stage_fail_at_any_n(tmp_path, config_cls):
    yaml_path = _write_yaml(tmp_path, config_cls)
    with pytest.raises(ValueError, match="does not declare a generation stage"):
        mps_dp_config.resolve_max_total_tokens(yaml_path)


def test_multi_engine_pipeline_fails_the_singleton_check(tmp_path):
    yaml_path = _write_yaml(tmp_path, "Qwen3OmniSpeechPipelineConfig")
    with pytest.raises(ValueError, match="one SGLang engine stage"):
        mps_dp_config.resolve_max_total_tokens(
            yaml_path, require_single_sglang_engine=True
        )


@pytest.mark.parametrize(
    "config_cls",
    ["VoxtralTTSPipelineConfig", "Qwen3TTSPipelineConfig", "MingTTSPipelineConfig"],
)
def test_weight_share_rejected_for_unvalidated_configs(tmp_path, config_cls):
    yaml_path = _write_yaml(tmp_path, config_cls)
    with pytest.raises(ValueError, match="not passed end-to-end validation"):
        mps_dp_config.resolve_max_total_tokens(yaml_path, weight_share=True)


def test_docs_table_matches_the_code_registries():
    text = DOCS_PAGE.read_text(encoding="utf-8")
    supported_rows = [
        line
        for line in text.splitlines()
        if line.startswith("|") and "| Supported" in line
    ]
    for arch in ipc_weights.WEIGHT_SHARE_POLICIES:
        assert any(
            arch in row for row in supported_rows
        ), f"{arch} is gate-supported but has no Supported row in {DOCS_PAGE.name}"
    for arch in ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES:
        assert not any(
            arch in row for row in supported_rows
        ), f"{arch} is audit-only but {DOCS_PAGE.name} lists it as Supported"


@pytest.mark.skipif(os.name != "posix", reason="launch.sh needs a POSIX shell")
class TestLaunchFailsClosedBeforeResources:
    def _run(self, tmp_path, yaml_path, **env_extra):
        state_root = tmp_path / "state"
        env = os.environ.copy()
        env.update(
            {
                "STATE_ROOT": str(state_root),
                "CONFIG": str(yaml_path),
                "N": "2",
                "CORE_BLOCKS": "0 1",
                "PYTHON_BIN": str(_write_gpu_info_wrapper(tmp_path)),
            }
        )
        env.update(env_extra)
        proc = subprocess.run(
            ["bash", str(MPS_DP_DIR / "launch.sh"), "up"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc, state_root

    def test_unsupported_topology_leaves_no_state(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, "LLaDA2UniPipelineConfig")
        proc, state_root = self._run(tmp_path, yaml_path)
        assert proc.returncode != 0
        assert "does not declare a generation stage" in proc.stdout + proc.stderr
        assert not state_root.exists()

    def test_unvalidated_weight_share_leaves_no_state(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, "VoxtralTTSPipelineConfig")
        proc, state_root = self._run(tmp_path, yaml_path, WEIGHT_SHARE="1")
        assert proc.returncode != 0
        assert "not passed end-to-end validation" in proc.stdout + proc.stderr
        assert not state_root.exists()

    def test_budget_preflight_failure_leaves_no_state(self, tmp_path):
        yaml_path = _write_budget_yaml(
            tmp_path,
            kv_cache_bytes="6GiB",
            total_reserve_bytes="9GiB",
        )
        fake_nvidia_smi_dir = _write_fake_nvidia_smi(tmp_path)
        proc, state_root = self._run(
            tmp_path,
            yaml_path,
            MPS_DP_FAKE_TOTAL_MEMORY_BYTES=str(16 * 1024**3),
            PATH=f"{fake_nvidia_smi_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "requested=18.00GiB" in combined
        assert "available=16.00GiB" in combined
        assert not state_root.exists()

    def test_budget_preflight_uses_resolved_physical_gpu_uuid(self, tmp_path):
        yaml_path = _write_budget_yaml(
            tmp_path,
            kv_cache_bytes="6GiB",
            total_reserve_bytes="9GiB",
        )
        record_path = tmp_path / "config_invocations.jsonl"
        fake_nvidia_smi_dir = _write_fake_nvidia_smi(tmp_path)
        proc, state_root = self._run(
            tmp_path,
            yaml_path,
            CUDA_VISIBLE_DEVICES="GPU-inherited-1",
            MPS_DP_FAKE_TOTAL_MEMORY_BYTES=str(16 * 1024**3),
            MPS_DP_RECORD_CONFIG_INVOCATIONS=str(record_path),
            PATH=f"{fake_nvidia_smi_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )

        assert proc.returncode != 0
        assert not state_root.exists()

        invocations = [
            json.loads(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
        ]
        budget_call = next(
            invocation
            for invocation in invocations
            if "--print-mps-memory-budget" in invocation["argv"]
        )
        assert budget_call["cuda_visible_devices"] == "GPU-physical-0"
        gpu_id_index = budget_call["argv"].index("--gpu-id")
        assert budget_call["argv"][gpu_id_index + 1] == "0"

    def test_run_id_traversal_is_rejected(self, tmp_path):
        yaml_path = (
            REPO_ROOT / "examples" / "mps_dp" / "configs" / "higgs_h100_dp3.yaml"
        )
        proc, state_root = self._run(
            tmp_path,
            yaml_path,
            RUN_ID="../gpu-1/run-x",
            MAX_TOTAL_TOKENS="1000",
            BASE_PORT="29411",
        )
        assert proc.returncode != 0
        assert not state_root.exists()
        # Note (Jiaxin Deng): the RUN_ID check sits after the GPU probes, so
        # only assert its message where nvidia-smi exists; without a GPU the
        # launch still fails closed before any state is created.
        if shutil.which("nvidia-smi"):
            assert "RUN_ID must be a single" in proc.stdout + proc.stderr
