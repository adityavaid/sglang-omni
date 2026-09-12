# SPDX-License-Identifier: Apache-2.0
"""Scheduler-compatible zero-weight worker for externally executed backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner import external_model_worker

# Importing the SGLang backend package applies Omni's Apple Torch-inductor
# preparation, which must run before anything imports SGLang itself.
from sglang_omni.scheduling import bootstrap, sglang_backend


class _FakeGroup:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cpu_group = f"{name}-cpu"
        self.ranks = [0]


def _fake_model_config(context_len: int = 4096) -> SimpleNamespace:
    """A ModelConfig stand-in carrying only what MlxModelRunnerStub reads."""
    text_config = SimpleNamespace(vocab_size=8, num_hidden_layers=2)
    hf_config = SimpleNamespace(
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        text_config=None,
        get_text_config=lambda: text_config,
    )
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
        is_draft_model=False,
        linear_attn_registry_result=None,
        is_hybrid_swa=False,
        is_hybrid_swa_compress=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        dtype=torch.float16,
        context_len=context_len,
        num_hidden_layers=2,
        num_attention_layers=2,
        vocab_size=8,
        is_generation=True,
        is_multimodal=False,
        is_multimodal_chunked_prefill_supported=False,
        use_ngram_embedding=False,
    )


def _fake_server_args(**overrides):
    """Publish a real ``ServerArgs`` carrying every field the constructors read."""
    from sglang.srt.runtime_context import get_context

    fields = {
        "device": "cpu",
        "skip_tokenizer_init": True,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "max_running_requests": 1,
        "mem_fraction_static": 0.5,
        "tp_size": 1,
        "dp_size": 1,
        "random_seed": 0,
    }
    fields.update(overrides)
    return get_context().override_server_args(**fields)


@pytest.fixture(name="lightweight_sglang_worker")
def _lightweight_sglang_worker(monkeypatch: pytest.MonkeyPatch):
    """Neutralize the device, distributed, and checkpoint setup of the bases."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers import tp_worker as tp_worker_module
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_device

    tp_group = _FakeGroup("tp")
    attention_tp_group = _FakeGroup("attn-tp")
    model_config = _fake_model_config()

    monkeypatch.setattr(
        ModelConfig,
        "from_server_args",
        staticmethod(lambda *args, **kwargs: model_config),
    )

    def _light_init(
        self,
        *,
        model_config,
        mem_fraction_static,
        gpu_id,
        ps,
        nccl_port,
        server_args,
        is_draft_worker=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        memory_pool_config=None,
        **_ignored,
    ) -> None:
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.gpu_id = gpu_id
        self.ps = ps
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.memory_pool_config = memory_pool_config
        self.device = get_device().device
        self.tp_group = tp_group
        self.attention_tp_group = attention_tp_group
        self.initialize()

    monkeypatch.setattr(ModelRunner, "__init__", _light_init)
    monkeypatch.setattr(tp_worker_module, "get_pp_group", lambda: _FakeGroup("pp"))
    monkeypatch.setattr(
        tp_worker_module, "get_world_group", lambda: _FakeGroup("world")
    )
    monkeypatch.setattr(
        tp_worker_module,
        "broadcast_pyobj",
        lambda values, rank, group, src=0: values,
    )
    return SimpleNamespace(
        model_config=model_config,
        tp_group=tp_group,
        attention_tp_group=attention_tp_group,
    )


def _build_worker(
    published_server_args,
    *,
    backend_name: str = "native_mlx",
    model_arch_override: str = "Qwen3OmniTalker",
):
    return external_model_worker.create_external_model_worker(
        config=SimpleNamespace(
            model_arch_override=model_arch_override,
            nccl_port=9000,
        ),
        server_args=published_server_args,
        gpu_id=0,
        tp_rank=0,
        backend_name=backend_name,
    )


def test_external_worker_builds_zero_weight_scheduler_pools(
    lightweight_sglang_worker,
) -> None:
    override = _fake_server_args(max_total_tokens=128)
    server_args = override.install()
    try:
        worker = _build_worker(server_args)

        req_pool, kv_pool = worker.get_memory_pool()
        assert torch.device(req_pool.device).type == "cpu"
        assert torch.device(kv_pool.device).type == "cpu"
        assert worker.model_runner.model.__class__.__name__ == "_DummyModel"
        assert worker.model_runner.max_total_num_tokens == 128
        assert worker.uses_external_forward is True
    finally:
        override.restore()


def test_external_worker_pool_size_defaults_to_the_model_context_limit(
    lightweight_sglang_worker,
) -> None:
    override = _fake_server_args(max_total_tokens=None)
    server_args = override.install()
    try:
        worker = _build_worker(server_args)

        context_len = lightweight_sglang_worker.model_config.context_len
        assert worker.model_runner._mlx_pool_size == context_len
        assert worker.model_runner.max_total_num_tokens == context_len
    finally:
        override.restore()


def test_external_worker_refuses_to_own_the_model_forward(
    lightweight_sglang_worker,
) -> None:
    override = _fake_server_args(max_total_tokens=128)
    server_args = override.install()
    try:
        worker = _build_worker(server_args, backend_name="native_mlx")

        with pytest.raises(RuntimeError, match="native_mlx"):
            worker.forward_batch_generation(batch=None)
    finally:
        override.restore()


def test_external_worker_kv_cache_release_hook_needs_no_mlx_runner(
    lightweight_sglang_worker,
) -> None:
    """The scheduler's release hook must not reach for an MLX runner."""
    override = _fake_server_args(max_total_tokens=128)
    server_args = override.install()
    try:
        worker = _build_worker(server_args, backend_name="native_mlx")

        assert not hasattr(worker, "_mlx_runner")
        req = SimpleNamespace(rid="req-0", mamba_last_track_seqlen=7)
        worker.prepare_for_kv_cache_release(req)
        assert req.mamba_last_track_seqlen == 7
    finally:
        override.restore()


def test_external_worker_exposes_the_scheduler_process_groups(
    lightweight_sglang_worker,
) -> None:
    override = _fake_server_args(max_total_tokens=128)
    server_args = override.install()
    try:
        worker = _build_worker(server_args)

        assert worker.get_tp_group() is lightweight_sglang_worker.tp_group
        attention_tp_group = lightweight_sglang_worker.attention_tp_group
        assert worker.get_attention_tp_group() is attention_tp_group
        assert worker.get_attention_tp_cpu_group() == attention_tp_group.cpu_group
        assert worker.tp_rank == 0
    finally:
        override.restore()


@pytest.mark.parametrize("backend", ["external", "asr", "thinker"])
@pytest.mark.parametrize("max_total_tokens", [None, 96])
def test_worker_startup_uses_published_context_not_raw_server_args(
    lightweight_sglang_worker,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    max_total_tokens: int | None,
) -> None:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    from sglang_omni.model_runner import mlx_model_worker

    captured = {}
    validations = []
    original_validate = MlxModelRunnerStub.validate_startup_weight_load_mode

    def validate_startup() -> None:
        validations.append(True)
        original_validate()

    monkeypatch.setattr(
        MlxModelRunnerStub,
        "validate_startup_weight_load_mode",
        staticmethod(validate_startup),
    )

    def init_worker(self, *, gpu_id, ps, nccl_port, **kwargs):
        # Deliberately expose no legacy fields: runtime getters own all config.
        self.server_args = object()
        self.model_config = lightweight_sglang_worker.model_config
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.is_draft_worker = False
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.memory_pool_config = None
        self._init_model_runner()

    monkeypatch.setattr(MlxTpModelWorker, "__init__", init_worker)

    class NativeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.pool_size = kwargs.get("pool_size", 73)

    architecture = "Qwen3OmniTalker"
    if backend == "asr":
        from sglang_omni.models.qwen3_asr.mlx import runner

        architecture = "Qwen3ASRForConditionalGeneration"
        monkeypatch.setattr(
            runner, "make_qwen3_asr_mlx_runner_class", lambda: NativeRunner
        )
    elif backend == "thinker":
        pytest.importorskip("mlx.core")
        from sglang_omni.models.qwen3_omni.mlx import runner

        architecture = "Qwen3OmniThinkerForCausalLM"
        monkeypatch.setattr(
            runner, "make_qwen3_omni_thinker_mlx_runner_class", lambda: NativeRunner
        )

    override = _fake_server_args(
        model_path="published-checkpoint",
        trust_remote_code=True,
        revision="published-revision",
        quantization=None,
        max_total_tokens=max_total_tokens,
        random_seed=123,
        mlx_enable_sampling=False,
        enable_deterministic_inference=True,
    )
    server_args = override.install()
    try:
        config = SimpleNamespace(
            model_arch_override=architecture,
            nccl_port=9000,
            capture_hidden_layers=(0, 24),
        )
        if backend == "external":
            worker = _build_worker(server_args)
            expected_pool = max_total_tokens or worker.model_config.context_len
        else:
            worker = mlx_model_worker.create_mlx_model_worker(
                config=config, server_args=server_args, gpu_id=0
            )
            expected = {
                "model_path": "published-checkpoint",
                "trust_remote_code": True,
                "revision": "published-revision",
                "disable_radix_cache": True,
                "mem_fraction_static": 0.5,
                "quantization": None,
                "enable_sampling": False,
                "sampling_rng_seed": 123,
                "deterministic_seeding": True,
            }
            if max_total_tokens is not None:
                expected["pool_size"] = max_total_tokens
            if backend == "thinker":
                expected["capture_hidden_layers"] = (0, 24)
            assert captured == expected
            expected_pool = max_total_tokens or 73

        assert validations
        assert worker.model_runner._mlx_pool_size == expected_pool
        assert worker.model_runner.mem_fraction_static == 0.5
    finally:
        override.restore()


def test_external_parallel_state_reads_resolved_topology() -> None:
    override = _fake_server_args(tp_size=4, dcp_size=2, attn_cp_size=1)
    server_args = override.install()
    try:
        state = external_model_worker._build_parallel_state(
            server_args, gpu_id=2, tp_rank=3
        )
        assert (state.tp_rank, state.tp_size) == (3, 4)
        assert (state.attn_tp_rank, state.attn_tp_size) == (3, 4)
        assert (state.attn_dcp_rank, state.attn_dcp_size) == (1, 2)
        assert state.gpu_id == 2
    finally:
        override.restore()


class _FakeExternalRunner:
    def __init__(self) -> None:
        self.model = object()
        self.model_config = SimpleNamespace(vocab_size=1)
        self.events: list[str] = []

    def alloc_memory_pool(self) -> None:
        self.events.append("alloc_memory_pool")

    def init_attention_backends(self) -> None:
        self.events.append("init_attention_backends")

    def init_cuda_graphs(self) -> None:
        raise AssertionError("an external-forward worker owns no Torch graphs")


class _FakeExternalWorker:
    uses_external_forward = True

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.model_runner = _FakeExternalRunner()
        self.model_config = SimpleNamespace(is_multimodal=False)

    def get_memory_pool(self):
        return "req_pool", "kv_pool"


class _FakeStandardWorker(_FakeExternalWorker):
    uses_external_forward = False
    enable_prefill_input_embeds = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_runner.init_cuda_graphs = lambda: self.model_runner.events.append(
            "init_cuda_graphs"
        )


def _speech_server_args() -> SimpleNamespace:
    return SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        max_prefill_tokens=8192,
        context_length=32768,
        max_running_requests=1,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=None),
            prefill=SimpleNamespace(max_bs=None),
        ),
    )


@pytest.fixture(name="infrastructure_env")
def _infrastructure_env(monkeypatch: pytest.MonkeyPatch):
    from sglang.srt.hardware_backend.mlx import runtime as tensor_bridge

    from sglang_omni.model_runner import _hidden_capture as hidden_capture_module
    from sglang_omni.model_runner import external_model_worker as external_module
    from sglang_omni.model_runner import model_worker as model_worker_module
    from sglang_omni.models.qwen3_omni import apple_runtime

    installs: list[tuple] = []
    built: list[tuple[str, dict]] = []
    runtime_overrides = []

    def publish_runtime() -> None:
        override = _fake_server_args(
            page_size=1,
            chunked_prefill_size=-1,
            max_prefill_tokens=8192,
            context_length=32768,
            cuda_graph_config=_speech_server_args().cuda_graph_config,
        )
        override.install()
        runtime_overrides.append(override)

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args: "runtime configuration",
    )
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree_cache", args),
    )
    monkeypatch.setattr(
        hidden_capture_module,
        "install_hidden_capture_hooks",
        lambda model, layers, *, max_tokens: installs.append(
            (model, layers, max_tokens)
        ),
    )
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_apple_backend", lambda: True)

    def fake_create_external_model_worker(**kwargs):
        publish_runtime()
        built.append(("external", kwargs))
        return _FakeExternalWorker(**kwargs)

    def fake_model_worker(**kwargs):
        publish_runtime()
        built.append(("standard", kwargs))
        return _FakeStandardWorker(**kwargs)

    monkeypatch.setattr(
        external_module,
        "create_external_model_worker",
        fake_create_external_model_worker,
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", fake_model_worker)
    try:
        yield SimpleNamespace(
            bootstrap=bootstrap,
            installs=installs,
            built=built,
            publish_runtime=publish_runtime,
        )
    finally:
        for override in reversed(runtime_overrides):
            override.restore()


@pytest.mark.parametrize(
    "model_arch_override",
    ["Qwen3OmniThinkerForCausalLM", "Qwen3OmniTalker"],
)
def test_apple_qwen_infrastructure_requires_mlx(
    infrastructure_env,
    model_arch_override: str,
) -> None:
    with pytest.raises(ValueError, match="SGLANG_USE_MLX=1"):
        infrastructure_env.bootstrap.create_sglang_infrastructure(
            _speech_server_args(),
            0,
            model_arch_override=model_arch_override,
            capture_hidden_layers=[0, 24],
        )
    assert infrastructure_env.built == []
    assert infrastructure_env.installs == []


def test_speech_infrastructure_keeps_hidden_capture_for_standard_workers(
    infrastructure_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import apple_runtime

    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_apple_backend", lambda: False)

    worker, *_ = infrastructure_env.bootstrap.create_sglang_infrastructure(
        _speech_server_args(),
        0,
        model_arch_override="Qwen3OmniThinkerForCausalLM",
        capture_hidden_layers=[0, 24],
    )

    assert isinstance(worker, _FakeStandardWorker)
    assert infrastructure_env.built[0][0] == "standard"
    assert infrastructure_env.installs == [(worker.model_runner.model, [0, 24], 32768)]
    assert "init_cuda_graphs" in worker.model_runner.events


def test_non_qwen3_omni_apple_stages_keep_the_standard_worker(
    infrastructure_env,
) -> None:
    worker, *_ = infrastructure_env.bootstrap.create_sglang_infrastructure(
        _speech_server_args(),
        0,
        model_arch_override="Qwen3ASRForConditionalGeneration",
    )

    assert isinstance(worker, _FakeStandardWorker)
    assert infrastructure_env.built[0][0] == "standard"


def test_mlx_backend_still_owns_qwen3_omni_apple_stage_selection(
    infrastructure_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime as tensor_bridge

    from sglang_omni.model_runner import mlx_model_worker

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    built: list[dict] = []

    def create_worker(**kwargs):
        infrastructure_env.publish_runtime()
        built.append(kwargs)
        return _FakeExternalWorker(**kwargs)

    monkeypatch.setattr(
        mlx_model_worker,
        "create_mlx_model_worker",
        create_worker,
    )

    infrastructure_env.bootstrap.create_sglang_infrastructure(
        _speech_server_args(),
        0,
        model_arch_override="Qwen3OmniTalker",
    )

    assert len(built) == 1
    assert infrastructure_env.built == []


def test_cuda_graph_init_is_skipped_for_external_forward_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime as tensor_bridge

    # The MLX branch returns first, so it must be off or this test would pass
    # without ever reaching the external-forward marker branch under test.
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    worker = _FakeExternalWorker()

    bootstrap.init_sglang_cuda_graphs(worker)

    assert worker.model_runner.events == []


def test_cuda_graph_init_still_runs_for_standard_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime as tensor_bridge

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    worker = _FakeStandardWorker()

    bootstrap.init_sglang_cuda_graphs(worker)

    assert worker.model_runner.events == ["init_cuda_graphs"]


def _install_talker_scheduler_fakes(monkeypatch: pytest.MonkeyPatch, model_worker):
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.models.qwen3_omni import request_builders, talker_model_runner
    from sglang_omni.models.qwen3_omni import talker_scheduler as talker_scheduler_mod
    from sglang_omni.models.qwen3_omni.components import talker_prefill
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import sglang_backend

    talker_config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=2048),
        accept_hidden_layer=0,
        codec_bos_id=0,
        codec_eos_token_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        speaker_id={},
    )
    thinker_config = SimpleNamespace(
        audio_token_id=0,
        image_token_id=1,
        video_token_id=2,
    )
    hf_config = SimpleNamespace(
        thinker_config=thinker_config,
        talker_config=talker_config,
        tts_bos_token_id=0,
        tts_eos_token_id=1,
        tts_pad_token_id=2,
        im_start_token_id=3,
        im_end_token_id=4,
        system_token_id=5,
        user_token_id=6,
        assistant_token_id=7,
    )
    model_config = SimpleNamespace(
        model_path="model",
        vocab_size=10,
        hf_config=hf_config,
    )
    model_worker.model_runner.model_config = model_config

    monkeypatch.setattr(
        talker_scheduler_mod,
        "configure_talker_server_args",
        lambda server_args, feedback_enabled: False,
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            model_worker,
            object(),
            object(),
            object(),
            model_config,
        ),
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "init_sglang_cuda_graphs",
        lambda worker: None,
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        request_builders,
        "make_talker_scheduler_adapters",
        lambda **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        talker_prefill, "TalkerPrefillBuilder", lambda **kwargs: object()
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", lambda **kwargs: None)
    monkeypatch.setattr(
        talker_scheduler_mod,
        "QwenTalkerScheduler",
        lambda **kwargs: SimpleNamespace(
            outbox=object(),
            bind_model_runner=lambda runner: None,
        ),
    )
    monkeypatch.setattr(
        talker_model_runner,
        "QwenTalkerModelRunner",
        lambda *args, **kwargs: object(),
    )
    return model_config


def test_talker_scheduler_skips_sampler_wiring_for_external_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap

    model = SimpleNamespace()
    model_worker = _FakeExternalWorker()
    model_worker.model_runner.model = model
    _install_talker_scheduler_fakes(monkeypatch, model_worker)

    qwen_bootstrap.create_talker_scheduler(SimpleNamespace())

    assert not hasattr(model, "_sampler")


def test_talker_scheduler_still_wires_the_sampler_for_standard_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap

    model = SimpleNamespace()
    sampler = object()
    model_worker = _FakeStandardWorker()
    model_worker.model_runner.model = model
    model_worker.model_runner.sampler = sampler
    _install_talker_scheduler_fakes(monkeypatch, model_worker)

    qwen_bootstrap.create_talker_scheduler(SimpleNamespace())

    assert model._sampler is sampler


def _install_thinker_scheduler_fakes(monkeypatch: pytest.MonkeyPatch, model_worker):
    """Fake every thinker dependency except the output-processor wiring."""
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.model_runner import thinker_model_runner
    from sglang_omni.models.qwen3_omni import request_builders
    from sglang_omni.models.qwen3_omni import (
        thinker_model_runner as qwen_thinker_model_runner,
    )
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import omni_scheduler, sglang_backend

    model_config = SimpleNamespace(
        model_path="model",
        vocab_size=10,
        hf_config=SimpleNamespace(
            thinker_config=SimpleNamespace(),
            talker_config=SimpleNamespace(accept_hidden_layer=24),
        ),
    )
    output_processors: list[dict] = []

    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            model_worker,
            object(),
            object(),
            object(),
            model_config,
        ),
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "init_sglang_cuda_graphs",
        lambda worker: None,
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        request_builders,
        "make_thinker_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        request_builders,
        "make_thinker_stream_output_builder",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: output_processors.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        thinker_model_runner, "ThinkerModelRunner", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        qwen_thinker_model_runner,
        "Qwen3OmniThinkerModelRunner",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        omni_scheduler, "OmniScheduler", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    return output_processors


def _thinker_server_args() -> SimpleNamespace:
    from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

    return SimpleNamespace(
        disable_cuda_graph=True,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=CudaGraphBackend.DISABLED)
        ),
    )


def test_thinker_scheduler_withholds_the_stub_model_from_the_output_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap

    model_worker = _FakeExternalWorker()
    output_processors = _install_thinker_scheduler_fakes(monkeypatch, model_worker)
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.apple_runtime.qwen3_omni_uses_mlx_backend",
        lambda: True,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.mlx.runner.Qwen3OmniMlxSchedulerModelRunner",
        lambda *args: SimpleNamespace(),
    )

    qwen_bootstrap.create_thinker_scheduler(_thinker_server_args(), speech_enabled=True)

    assert len(output_processors) == 1
    kwargs = output_processors[0]
    assert kwargs["capture_hidden"] is True
    assert kwargs["capture_hidden_layers"] == [0, 24]
    # No hidden capture hooks are installed on an external worker's stub model,
    # so the processor must read the runner-supplied hidden states instead.
    assert kwargs["model"] is None


def test_thinker_scheduler_still_passes_the_model_for_standard_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap

    model_worker = _FakeStandardWorker()
    output_processors = _install_thinker_scheduler_fakes(monkeypatch, model_worker)

    qwen_bootstrap.create_thinker_scheduler(_thinker_server_args(), speech_enabled=True)

    assert output_processors[0]["model"] is model_worker.model_runner.model


def test_thinker_scheduler_passes_no_model_without_hidden_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap

    model_worker = _FakeStandardWorker()
    output_processors = _install_thinker_scheduler_fakes(monkeypatch, model_worker)

    qwen_bootstrap.create_thinker_scheduler(
        _thinker_server_args(), speech_enabled=False
    )

    assert output_processors[0]["capture_hidden"] is False
    assert output_processors[0]["model"] is None
