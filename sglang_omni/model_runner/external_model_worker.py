# SPDX-License-Identifier: Apache-2.0
"""Zero-weight SGLang worker for stages whose forward runs outside SGLang."""

from __future__ import annotations

from typing import NoReturn

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorkerConfig

# Architectures whose Apple-Silicon stages hand model execution to an external
# runner instead of SGLang's own model runner.
EXTERNAL_FORWARD_ARCHITECTURES = frozenset(
    {
        "Qwen3OmniThinkerForCausalLM",
        "Qwen3OmniTalker",
    }
)


def initialize_mlx_worker_runtime(
    server_args: ServerArgs, *, gpu_id: int, tp_rank: int, nccl_port: int | None
) -> int:
    """Publish worker ranks before constructing the MLX bookkeeping model."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.runtime_context import (
        SpawnRanks,
        get_device,
        publish,
        spawn_world_rank,
    )
    from sglang.srt.server_args import PortArgs

    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
            gpu_id=gpu_id,
        ),
    )
    if nccl_port is None:
        nccl_port = PortArgs.init_new(server_args).nccl_port
    else:
        pass
    bootstrap.init_parallel_runtime(
        server_args=server_args, device=get_device().device, dist_port=nccl_port
    )
    bootstrap.init_layer_runtime(model_config=ModelConfig.from_server_args(server_args))
    return nccl_port


def make_external_worker_class():
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.runtime_context import get_parallel, get_schedule

    class OmniExternalModelWorker(MlxTpModelWorker):
        """Scheduler bookkeeping for a stage whose runner owns the forward."""

        # Read by create_sglang_infrastructure() and the Qwen3-Omni bootstrap:
        # this worker has no real Torch model, so hidden-state capture hooks,
        uses_external_forward = True

        def __init__(
            self,
            *,
            external_backend_name: str,
            server_args: ServerArgs,
            gpu_id: int,
            nccl_port: int | None = None,
        ) -> None:
            self.external_backend_name = external_backend_name
            super().__init__(
                server_args=server_args, gpu_id=gpu_id, nccl_port=nccl_port
            )

        @property
        def tp_rank(self) -> int:
            return get_parallel().tp_rank

        def _init_model_runner(self) -> None:
            MlxModelRunnerStub.validate_startup_weight_load_mode()
            self._model_runner = MlxModelRunnerStub(  # noqa: leading-underscore - external runtime contract
                model_config=self.model_config,
                mem_fraction_static=get_schedule().mem_fraction_static,
                gpu_id=self.gpu_id,
                nccl_port=self.nccl_port,
                server_args=self.server_args,
                is_draft_worker=self.is_draft_worker,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=self.memory_pool_config,
                # The external runner owns the real cache, so the scheduler's
                # token budget is the serving cap or the model context bound.
                mlx_pool_size=(
                    get_schedule().max_total_tokens or self.model_config.context_len
                ),
            )
            self._mlx_active_rids: set[str] = (
                set()
            )  # noqa: leading-underscore - external runtime contract
            self._mlx_pool_initialized = (
                False  # noqa: leading-underscore - external runtime contract
            )

        def forward_batch_generation(self, *args: object, **kwargs: object) -> NoReturn:
            raise RuntimeError(
                f"{self.external_backend_name} runner must own model forward"
            )

        def prepare_for_kv_cache_release(self, req: Req) -> None:
            """No MLX auxiliary state exists to snapshot on this worker."""
            del req

        def get_tp_group(self):
            return self.model_runner.tp_group

        def get_attention_tp_group(self):
            return self.model_runner.attention_tp_group

        def get_attention_tp_cpu_group(self):
            return self.model_runner.attention_tp_group.cpu_group

    return OmniExternalModelWorker


def uses_external_forward(model_worker: object) -> bool:
    """Whether this worker's model forward runs outside SGLang."""
    return bool(getattr(model_worker, "uses_external_forward", False))


def create_external_model_worker(
    *,
    config: ModelWorkerConfig,
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int = 0,
    backend_name: str,
):
    """Construct a zero-weight worker with Omni's scheduler-facing contract."""
    if config.model_arch_override not in EXTERNAL_FORWARD_ARCHITECTURES:
        raise NotImplementedError(
            "Omni's external model worker supports only "
            f"{sorted(EXTERNAL_FORWARD_ARCHITECTURES)}; got "
            f"{config.model_arch_override!r}"
        )
    else:
        pass

    nccl_port = initialize_mlx_worker_runtime(
        server_args, gpu_id=gpu_id, tp_rank=tp_rank, nccl_port=config.nccl_port
    )
    worker_class = make_external_worker_class()
    return worker_class(
        external_backend_name=backend_name,
        server_args=server_args,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
    )
