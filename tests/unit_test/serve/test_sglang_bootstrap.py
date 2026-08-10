# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

if "sglang.srt.server_args" not in sys.modules:
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = []
    srt_pkg = types.ModuleType("sglang.srt")
    srt_pkg.__path__ = []
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    server_args_mod.get_global_server_args = lambda: None
    sys.modules.setdefault("sglang", sglang_pkg)
    sys.modules.setdefault("sglang.srt", srt_pkg)
    sys.modules.setdefault("sglang.srt.server_args", server_args_mod)

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageMemoryConfig,
    StageRuntimeConfig,
    resolve_stage_factory_args,
)
from sglang_omni.model_runner import model_worker as model_worker_module
from sglang_omni.scheduling import bootstrap
from tests.unit_test.fakes import FakeServerArgs


def _install_fake_sglang_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefill_manager: object,
    decode_manager: object,
    create_tree_cache: object,
) -> None:
    fake_backend = types.ModuleType("sglang_omni.scheduling.sglang_backend")
    fake_backend.PrefillManager = prefill_manager
    fake_backend.DecodeManager = decode_manager
    fake_backend.create_tree_cache = create_tree_cache
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        fake_backend,
    )


def runtime_bootstrap_factory(
    model_path: str,
    *,
    gpu_id: int,
    kv_cache_bytes: int | None = None,
):
    del model_path
    server_args = FakeServerArgs(
        page_size=1,
        disable_overlap_schedule=False,
        chunked_prefill_size=8,
        max_prefill_tokens=16,
        random_seed=7,
    )
    return bootstrap.create_sglang_infrastructure(
        server_args,
        gpu_id,
        kv_cache_bytes=kv_cache_bytes,
    )


def test_runtime_configuration_reports_global_backend_for_each_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 89,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=0,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=0, sm=89, architecture=ada, "
        "attention_backend=flashinfer, decode_attention_backend=flashinfer, "
        "prefill_attention_backend=flashinfer, sampling_backend=pytorch"
    )


def test_runtime_configuration_reports_explicit_phase_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 90,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend="triton",
        prefill_attention_backend="fa3",
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("fa3", "triton"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=1,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=1, sm=90, architecture=hopper, "
        "attention_backend=flashinfer, decode_attention_backend=triton, "
        "prefill_attention_backend=fa3, sampling_backend=pytorch"
    )


def test_create_sglang_infrastructure_runs_0515_initialization_phases(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: events.append("runtime_configuration")
        or "runtime configuration",
    )

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("model_worker")
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    _install_fake_sglang_backend(
        monkeypatch,
        prefill_manager=FakePrefillManager,
        decode_manager=FakeDecodeManager,
        create_tree_cache=lambda *args: ("tree_cache", args),
    )

    server_args = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_overlap_schedule=False,
        chunked_prefill_size=8,
        max_prefill_tokens=16,
    )
    infrastructure = bootstrap.create_sglang_infrastructure(server_args, 0)

    assert events == [
        "runtime_configuration",
        "model_worker",
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]
    assert infrastructure[0].model_runner.model is FakeRunner.model


def test_resolved_factory_kv_budget_reaches_runner_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_init_calls: list[dict[str, object]] = []

    class FakeSGLModelRunner:
        def __init__(self, **kwargs) -> None:
            runner_init_calls.append(dict(kwargs))
            self.device = "cuda:0"
            self.tp_group = SimpleNamespace(cpu_group=object())
            self.model = object()
            self.req_to_token_pool = SimpleNamespace(size=1, max_context_len=1)
            self.token_to_kv_pool_allocator = SimpleNamespace(size=1)

        def alloc_memory_pool(self) -> None:
            return None

        def init_attention_backends(self) -> None:
            return None

        def init_cuda_graphs(self) -> None:
            return None

    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = []
    srt_pkg = types.ModuleType("sglang.srt")
    srt_pkg.__path__ = []
    utils_mod = types.ModuleType("sglang.srt.utils")
    utils_mod.broadcast_pyobj = lambda values, *_args, **_kwargs: values
    utils_mod.set_random_seed = lambda _seed: None
    fake_runner_mod = types.ModuleType("sglang_omni.model_runner.sglang_model_runner")
    fake_runner_mod.SGLModelRunner = FakeSGLModelRunner

    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.model_runner.sglang_model_runner",
        fake_runner_mod,
    )
    monkeypatch.setattr(
        model_worker_module.ModelWorker,
        "_init_model_config",
        lambda self: setattr(self, "model_config", SimpleNamespace()),
    )
    monkeypatch.setattr(
        model_worker_module.ModelWorker,
        "_configure_backend_policy",
        lambda self: None,
    )
    monkeypatch.setattr(
        model_worker_module.ModelWorker,
        "_init_dllm_algorithm",
        lambda self: setattr(self, "dllm_algorithm", None),
    )
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args, **_kwargs: "runtime configuration",
    )
    _install_fake_sglang_backend(
        monkeypatch,
        prefill_manager=lambda **kwargs: kwargs,
        decode_manager=lambda **kwargs: kwargs,
        create_tree_cache=lambda *args: ("tree_cache", args),
    )

    stage = StageConfig(
        name="thinker",
        process="pipeline",
        factory="tests.unit_test.serve.test_sglang_bootstrap.runtime_bootstrap_factory",
        terminal=True,
        gpu=0,
        runtime=StageRuntimeConfig(memory=StageMemoryConfig(kv_cache_bytes="2GiB")),
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)
    runtime_bootstrap_factory(**args)

    assert args["kv_cache_bytes"] == 2 * 1024**3
    assert runner_init_calls[0]["kv_cache_bytes"] == 2 * 1024**3


def test_cuda_graph_init_scopes_prefill_embedding_capture_flag() -> None:
    model_config = SimpleNamespace(is_multimodal=False)
    capture_values: list[bool] = []
    model_worker = SimpleNamespace(
        model_config=model_config,
        enable_prefill_input_embeds=True,
        model_runner=SimpleNamespace(
            init_cuda_graphs=lambda: capture_values.append(model_config.is_multimodal)
        ),
    )

    bootstrap.init_sglang_cuda_graphs(model_worker)

    assert capture_values == [True]
    assert model_config.is_multimodal is False


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        (
            SimpleNamespace(
                chunked_prefill_size=8192,
                max_prefill_tokens=16384,
                context_length=32768,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
                ),
            ),
            8192,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=-1,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=None,
            ),
            16384,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=512,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=1024, bs=[4, 1024]),
                ),
            ),
            1024,
        ),
    ],
)
def test_hidden_capture_max_tokens_covers_eager_and_graph_forwards(
    server_args: SimpleNamespace,
    expected: int,
) -> None:
    assert bootstrap._hidden_capture_max_tokens(server_args) == expected


def test_hidden_capture_max_tokens_covers_non_chunked_context_length() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=8192,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=None,
    )

    assert bootstrap._hidden_capture_max_tokens(server_args) == 32768


def test_hidden_capture_max_tokens_rejects_missing_capacity_sources() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=None,
        max_running_requests=0,
        cuda_graph_config=None,
    )

    with pytest.raises(ValueError, match="hidden capture capacity"):
        bootstrap._hidden_capture_max_tokens(server_args)


def test_hidden_capture_is_installed_before_graph_initialization(monkeypatch) -> None:
    events: list[object] = []

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.model_runner = FakeRunner()
            events.append("model_worker")

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    def fake_install(model, layers, *, max_tokens) -> None:
        events.append(("install_hidden_capture", model, layers, max_tokens))

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args: "runtime configuration",
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(
        hidden_capture_module,
        "install_hidden_capture_hooks",
        fake_install,
    )
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree_cache", args),
    )
    server_args = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_overlap_schedule=False,
        chunked_prefill_size=8192,
        max_prefill_tokens=16384,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
            prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
        ),
    )

    bootstrap.create_sglang_infrastructure(
        server_args,
        0,
        capture_hidden_layers=[0, 24],
    )

    assert events == [
        "model_worker",
        ("install_hidden_capture", FakeRunner.model, [0, 24], 8192),
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]


def test_defer_cuda_graph_restores_requested_graph_capture(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=False)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        seen.append(bool(server_args.disable_cuda_graph))
        return ("infra", gpu_id, kwargs)

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, infrastructure = (
        bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            3,
            model_arch_override="TestModel",
        )
    )

    assert want_cuda_graph is True
    assert seen == [True]
    assert server_args.disable_cuda_graph is False
    assert infrastructure == (
        "infra",
        3,
        {
            "defer_cuda_graph_capture": True,
            "model_arch_override": "TestModel",
        },
    )


def test_defer_cuda_graph_leaves_disabled_graph_capture_disabled(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=True)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        seen.append(bool(server_args.disable_cuda_graph))
        return object()

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, _ = bootstrap.create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        0,
    )

    assert want_cuda_graph is False
    assert seen == [True]
    assert server_args.disable_cuda_graph is True
