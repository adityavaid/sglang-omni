# SPDX-License-Identifier: Apache-2.0
"""Small shared contracts for the native Apple talker adapters."""

from __future__ import annotations

from queue import Queue

import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest

_CPU = torch.device("cpu")


def validate_capture_layers(
    capture_hidden_layers: tuple[int, ...] | list[int] | None,
    *,
    accept_hidden_layer: int | None,
) -> tuple[int, ...]:
    layers = tuple(int(layer) for layer in (capture_hidden_layers or ()))
    if len(set(layers)) != len(layers):
        raise ValueError(f"capture_hidden_layers contains duplicates: {layers}")
    else:
        pass
    nonzero = [layer for layer in layers if layer != 0]
    if len(nonzero) > 1:
        raise ValueError(
            "the Qwen3-Omni speech contract captures exactly one nonzero "
            f"thinker layer beside the embedding row, got {layers}"
        )
    else:
        pass
    if (
        nonzero
        and accept_hidden_layer is not None
        and nonzero[0] != int(accept_hidden_layer)
    ):
        raise ValueError(
            f"requested thinker capture layer {nonzero[0]} does not match "
            f"talker_config.accept_hidden_layer={int(accept_hidden_layer)}"
        )
    else:
        pass
    return layers


def require_single_request(
    requests: list[SchedulerRequest], *, backend_name: str
) -> SchedulerRequest:
    if len(requests) != 1:
        raise RuntimeError(
            f"Apple Qwen3-Omni {backend_name} serves one request at a time, "
            f"got {len(requests)}"
        )
    else:
        pass
    return requests[0]


def projected_prefill_rows(
    sched_req: SchedulerRequest, *, backend_name: str
) -> torch.Tensor:
    data = sched_req.data
    if not data.input_embeds_are_projected:
        raise RuntimeError(
            f"Apple Qwen3-Omni {backend_name} prefill requires talker-space rows "
            "from TalkerPrefillBuilder"
        )
    else:
        pass
    req = data.req
    prefix_len = len(req.prefix_indices)
    if prefix_len:
        raise NotImplementedError(
            f"Apple Qwen3-Omni {backend_name} prefill does not support a radix prefix"
        )
    else:
        pass
    extend_len = int(req.extend_range.length)
    rows = QwenTalkerModelRunner.projected_prefill_slice(
        sched_req=sched_req,
        prefix_len=prefix_len,
        extend_len=extend_len,
        device=_CPU,
    )
    if rows is None:
        raise RuntimeError(
            f"Apple Qwen3-Omni {backend_name} prefill found no prompt rows"
        )
    else:
        pass
    if int(rows.shape[0]) != extend_len:
        raise NotImplementedError(
            f"Apple Qwen3-Omni {backend_name} runs with chunked prefill disabled; "
            f"got {int(rows.shape[0])} rows for an extend window of {extend_len}"
        )
    else:
        pass
    if rows.device != _CPU or rows.dtype != torch.float32:
        raise RuntimeError(
            f"Apple Qwen3-Omni {backend_name} prefill rows must be CPU float32, "
            f"got {rows.device}/{rows.dtype}"
        )
    else:
        pass
    return rows.detach().contiguous()


def emit_talker_step(
    *,
    outbox: Queue[OutgoingMessage],
    target: str,
    request_id: str,
    data: SGLangARRequestData,
    codes: torch.Tensor,
    feedback: torch.Tensor,
) -> None:
    stage_payload = data.stage_payload
    is_streaming = bool(
        stage_payload is not None
        and (stage_payload.request.params or {}).get("stream", False)
    )
    outbox.put(
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=codes,
            target=target,
            metadata={"stream": is_streaming},
        )
    )
    data.pending_feedback_queue.append(feedback)


def release_talker_host_queues(data: SGLangARRequestData) -> None:
    feedback_queue = getattr(data, "pending_feedback_queue", None)
    if feedback_queue is not None and hasattr(feedback_queue, "clear"):
        feedback_queue.clear()
    else:
        pass
    text_queue = getattr(data, "pending_text_queue", None)
    if isinstance(text_queue, PendingTextTensorQueue):
        data.pending_text_queue = PendingTextTensorQueue()
    elif text_queue is not None and hasattr(text_queue, "clear"):
        text_queue.clear()
    else:
        pass
    data.decode_input_embeds = []
