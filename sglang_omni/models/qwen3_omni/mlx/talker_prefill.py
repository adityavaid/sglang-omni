# SPDX-License-Identifier: Apache-2.0
"""Native MLX construction of Qwen3-Omni talker prefill rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    load_thinker_embedding_rows,
    packed_embedding_rows_mlx,
    resolve_embed_source,
    resolve_speaker_id,
)
from sglang_omni.models.qwen3_omni.mlx.talker import Qwen3OmniMlxTalker
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import mlx_to_torch as mlx_to_torch
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import torch_to_mlx as torch_to_mlx
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

__all__ = ["Qwen3OmniMlxTalkerPrefillBuilder", "build_mlx_prefill_input"]


def load_mlx_embedding_rows(
    model_path: str,
    row_ids: list[int],
) -> mx.array:
    source = resolve_embed_source(model_path)
    if source.quantization is not None:
        return packed_embedding_rows_mlx(model_path, source, row_ids)
    else:
        pass
    rows = load_thinker_embedding_rows(model_path, row_ids)
    converted = torch_to_mlx(rows)
    return converted.astype(mx.bfloat16) if rows.dtype == torch.bfloat16 else converted


def coerce_feature_array(value: object) -> mx.array | None:
    if value is None:
        return None
    else:
        pass
    if isinstance(value, mx.array):
        array = value
    elif isinstance(value, torch.Tensor):
        array = torch_to_mlx(value)
    elif isinstance(value, (list, tuple)):
        arrays = [coerce_feature_array(item) for item in value]
        arrays = [array for array in arrays if array is not None]
        if not arrays:
            return None
        else:
            pass
        array = mx.concatenate(arrays, axis=0)
    else:
        return None

    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    else:
        pass
    if array.ndim > 2:
        return array.reshape(-1, array.shape[-1])
    else:
        pass
    return array


def segment_chat_template(
    input_ids: list[int],
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
) -> list[dict[str, Any]]:
    role_map = {
        system_token_id: "system",
        user_token_id: "user",
        assistant_token_id: "assistant",
    }
    starts = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == im_start_token_id
    ]
    segments = []
    for index, start in enumerate(starts):
        role_token = input_ids[start + 1] if start + 1 < len(input_ids) else None
        segments.append(
            {
                "role": role_map.get(role_token, "unknown"),
                "start": start,
                "end": starts[index + 1] if index + 1 < len(starts) else len(input_ids),
            }
        )
    return segments


def build_mlx_user_part(
    *,
    thinker_embed: mx.array,
    thinker_hidden: mx.array,
    multimodal_mask: mx.array,
    text_projection: Callable[[mx.array], mx.array],
    hidden_projection: Callable[[mx.array], mx.array],
) -> mx.array:
    return mx.where(
        multimodal_mask[:, None],
        hidden_projection(thinker_hidden),
        text_projection(thinker_embed),
    )


def build_mlx_assistant_part(
    *,
    assistant_embed: mx.array,
    text_projection: Callable[[mx.array], mx.array],
    codec_embedding: Callable[[mx.array], mx.array],
    tts_bos_embed: mx.array,
    tts_eos_embed: mx.array,
    tts_pad_embed: mx.array,
    speaker_id: int,
    special_token_ids: Mapping[str, int],
) -> dict[str, mx.array]:
    projected = text_projection(assistant_embed)
    fourth_token = (
        projected[3:4]
        if projected.shape[0] > 3
        else mx.zeros((1, projected.shape[-1]), dtype=projected.dtype)
    )
    text_hidden = mx.concatenate(
        [
            projected[:3],
            mx.broadcast_to(tts_pad_embed, (4, tts_pad_embed.shape[-1])),
            tts_bos_embed,
            fourth_token,
        ],
        axis=0,
    )
    codec_ids = mx.array(
        [
            special_token_ids["codec_nothink_id"],
            special_token_ids["codec_think_bos_id"],
            special_token_ids["codec_think_eos_id"],
            speaker_id,
            special_token_ids["codec_pad_id"],
            special_token_ids["codec_bos_id"],
        ],
        dtype=mx.int32,
    )
    codec_hidden = mx.concatenate(
        [
            mx.zeros((3, text_hidden.shape[-1]), dtype=text_hidden.dtype),
            codec_embedding(codec_ids),
        ],
        axis=0,
    )
    future_text_rows = (
        mx.concatenate([projected[4:], tts_eos_embed], axis=0)
        if projected.shape[0] > 4
        else mx.array(tts_eos_embed)
    )
    return {
        "input_embeds": text_hidden + codec_hidden,
        "input_ids": mx.full(
            (text_hidden.shape[0],),
            special_token_ids["tts_pad_token_id"],
            dtype=mx.int32,
        ),
        "future_text_rows": future_text_rows,
    }


def build_mlx_prefill_input(
    *,
    thinker_embed: mx.array,
    thinker_hidden: mx.array,
    thinker_input_ids: list[int],
    multimodal_mask: mx.array,
    text_projection: Callable[[mx.array], mx.array],
    hidden_projection: Callable[[mx.array], mx.array],
    codec_embedding: Callable[[mx.array], mx.array],
    tts_bos_embed: mx.array,
    tts_eos_embed: mx.array,
    tts_pad_embed: mx.array,
    special_token_ids: Mapping[str, int],
    speaker_id: int,
    include_assistant_eos: bool = True,
) -> dict[str, mx.array | None]:
    segments = segment_chat_template(
        thinker_input_ids,
        im_start_token_id=special_token_ids["im_start_token_id"],
        system_token_id=special_token_ids["system_token_id"],
        user_token_id=special_token_ids["user_token_id"],
        assistant_token_id=special_token_ids["assistant_token_id"],
    )
    assistant_indices = [
        index
        for index, segment in enumerate(segments)
        if segment["role"] == "assistant"
    ]
    last_assistant = assistant_indices[-1] if assistant_indices else None
    all_embeds: list[mx.array] = []
    all_ids: list[mx.array] = []
    future_text_rows = None

    for segment_index, segment in enumerate(segments):
        if segment["role"] == "system":
            continue
        else:
            pass
        start, end = int(segment["start"]), int(segment["end"])
        if segment["role"] == "user":
            all_embeds.append(
                build_mlx_user_part(
                    thinker_embed=thinker_embed[start:end],
                    thinker_hidden=thinker_hidden[start:end],
                    multimodal_mask=multimodal_mask[start:end],
                    text_projection=text_projection,
                    hidden_projection=hidden_projection,
                )
            )
            all_ids.append(mx.array(thinker_input_ids[start:end], dtype=mx.int32))
        elif segment["role"] == "assistant":
            if last_assistant is not None and segment_index != last_assistant:
                continue
            else:
                pass
            assistant_embed = thinker_embed[start:end]
            if (
                assistant_embed.shape[0] > 0
                and thinker_input_ids[end - 1] == special_token_ids["im_end_token_id"]
            ):
                assistant_embed = assistant_embed[:-1]
            else:
                pass
            assistant = build_mlx_assistant_part(
                assistant_embed=assistant_embed,
                text_projection=text_projection,
                codec_embedding=codec_embedding,
                tts_bos_embed=tts_bos_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                speaker_id=speaker_id,
                special_token_ids=special_token_ids,
            )
            all_embeds.append(assistant["input_embeds"])
            all_ids.append(assistant["input_ids"])
            future_text_rows = assistant["future_text_rows"]
            if (
                not include_assistant_eos
                and future_text_rows is not None
                and future_text_rows.shape[0] > 0
            ):
                future_text_rows = future_text_rows[:-1]
            else:
                pass
        else:
            pass

    return {
        "input_embeds": mx.concatenate(all_embeds, axis=0),
        "input_ids": mx.concatenate(all_ids, axis=0),
        "future_text_rows": future_text_rows,
    }


class Qwen3OmniMlxTalkerPrefillBuilder:
    def __init__(
        self,
        *,
        talker: Qwen3OmniMlxTalker,
        model_path: str,
        special_token_ids: Mapping[str, int],
        speaker_map: Mapping[str, int],
        accept_hidden_layer: int = 24,
    ) -> None:
        model_dir = Path(model_path)
        self.model_path = str(
            model_dir
            if model_dir.exists()
            else resolve_model_path(model_path, local_files_only=False)
        )
        self.talker = talker
        self.accept_hidden_layer = int(accept_hidden_layer)
        self.special_token_ids = {
            name: int(token_id) for name, token_id in special_token_ids.items()
        }
        self.speaker_map = {
            str(name).lower(): int(speaker_id)
            for name, speaker_id in speaker_map.items()
        }
        self.thinker_embed_cache: dict[int, mx.array] = {}
        self.tts_special_cache: tuple[mx.array, mx.array, mx.array] | None = None

    @classmethod
    def from_talker(
        cls,
        talker: Qwen3OmniMlxTalker,
        *,
        model_path: str,
        special_token_ids: Mapping[str, int],
        speaker_map: Mapping[str, int],
        accept_hidden_layer: int = 24,
    ) -> "Qwen3OmniMlxTalkerPrefillBuilder":
        return cls(
            talker=talker,
            model_path=model_path,
            special_token_ids=special_token_ids,
            speaker_map=speaker_map,
            accept_hidden_layer=accept_hidden_layer,
        )

    def build_prompt_prefill(
        self,
        payload: StagePayload,
        thinker_chunks: list[StreamItem],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        if not thinker_chunks:
            raise ValueError("prompt prefill requires thinker chunks")
        else:
            pass

        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs = (
            self.reconstruct_prompt_states(state)
        )
        prompt_mask = self.build_multimodal_mask(prompt_ids)
        media_ids = {
            self.special_token_ids.get(name)
            for name in ("audio_token_id", "image_token_id", "video_token_id")
        }
        if any(token_id in media_ids for token_id in prompt_ids):
            first_chunk = thinker_chunks[0]
            layer = (first_chunk.metadata or {}).get("prompt_hidden_layer")
            if layer != self.accept_hidden_layer:
                raise ValueError(
                    "MLX media prompt requires thinker prompt hidden states "
                    f"from layer {self.accept_hidden_layer}; got {layer!r}"
                )
            else:
                pass
            captured = first_chunk.data
            if isinstance(captured, torch.Tensor):
                captured = torch_to_mlx(captured)
            else:
                pass
            if (
                not isinstance(captured, mx.array)
                or captured.shape != prompt_embed.shape
            ):
                raise ValueError(
                    "MLX thinker prompt hidden states must match prompt shape "
                    f"{prompt_embed.shape}; got {getattr(captured, 'shape', None)}"
                )
            else:
                pass
            prompt_hidden = mx.where(
                prompt_mask[:, None],
                captured.astype(prompt_hidden.dtype),
                prompt_hidden,
            )
        else:
            pass
        assistant_ids = self.extract_chunk_token_ids(thinker_chunks)
        assistant_embed = self.load_prompt_token_embeddings(assistant_ids)
        thinker_ids = [*prompt_ids, *assistant_ids]
        thinker_embed = mx.concatenate([prompt_embed, assistant_embed], axis=0)
        thinker_hidden = mx.concatenate([prompt_hidden, assistant_embed], axis=0)
        multimodal_mask = self.build_multimodal_mask(thinker_ids)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.get_tts_special_embeds()
        prefill = build_mlx_prefill_input(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            thinker_input_ids=thinker_ids,
            multimodal_mask=multimodal_mask,
            text_projection=self.talker.text_projection,
            hidden_projection=self.talker.hidden_projection,
            codec_embedding=self.talker.model.codec_embedding,
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            special_token_ids=self.special_token_ids,
            speaker_id=resolve_speaker_id(payload.request.params, self.speaker_map),
            include_assistant_eos=thinker_done,
        )
        input_embeds = prefill["input_embeds"]
        input_ids = prefill["input_ids"]
        future_text_rows = prefill["future_text_rows"]
        assert input_embeds is not None
        assert input_ids is not None
        mx.eval(input_embeds, input_ids, tts_pad_embed, tts_eos_embed)
        if future_text_rows is not None:
            mx.eval(future_text_rows)
            pending_text_queue = PendingTextTensorQueue.from_tensor(
                mlx_to_torch(future_text_rows)
            )
        else:
            pending_text_queue = PendingTextTensorQueue()
        return {
            "input_embeds": mlx_to_torch(input_embeds),
            "input_ids": mlx_to_torch(input_ids).to(torch.long),
            "pending_text_queue": pending_text_queue,
            "tts_pad_embed": mlx_to_torch(tts_pad_embed[0]),
            "tts_eos_embed": mlx_to_torch(tts_eos_embed[0]),
            "prompt_model_inputs": prompt_model_inputs,
        }

    def append_text_chunk(
        self, req_data: SGLangARRequestData, chunk: StreamItem
    ) -> None:
        if req_data.thinker_chunks_done:
            return
        else:
            pass
        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if (
            token_id is not None
            and int(token_id) == self.special_token_ids["im_end_token_id"]
        ):
            return
        else:
            pass
        pending = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending, PendingTextTensorQueue):
            pending = coerce_pending_text_queue(pending)
            req_data.pending_text_queue = pending
        else:
            pass
        pending.append(self.project_assistant_chunk(chunk))

    def mark_thinker_done(self, req_data: SGLangARRequestData) -> None:
        if req_data.thinker_chunks_done:
            return
        else:
            pass
        req_data.thinker_chunks_done = True
        pending = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending, PendingTextTensorQueue):
            pending = coerce_pending_text_queue(pending)
            req_data.pending_text_queue = pending
        else:
            pass
        if isinstance(req_data.tts_eos_embed, torch.Tensor):
            pending.append(req_data.tts_eos_embed)
        else:
            pass

    def extract_chunk_token_ids(self, thinker_chunks: list[StreamItem]) -> list[int]:
        return [int((chunk.metadata or {})["token_id"]) for chunk in thinker_chunks]

    def project_assistant_chunk(self, chunk: StreamItem) -> torch.Tensor:
        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if token_id is not None:
            row = self.load_prompt_token_embeddings([int(token_id)])
        else:
            row = torch_to_mlx(chunk.data).reshape(1, -1)
        projected = self.talker.text_projection(row)
        mx.eval(projected)
        return mlx_to_torch(projected[0])

    def build_multimodal_mask(self, token_ids: list[int]) -> mx.array:
        ids = mx.array(token_ids, dtype=mx.int32)
        mask = mx.zeros((len(token_ids),), dtype=mx.bool_)
        for name in ("audio_token_id", "image_token_id", "video_token_id"):
            token_id = self.special_token_ids.get(name)
            if token_id is not None:
                mask = mx.logical_or(mask, ids == int(token_id))
            else:
                pass
        return mask

    def get_tts_special_embeds(
        self,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if self.tts_special_cache is None:
            rows = self.load_prompt_token_embeddings(
                [
                    self.special_token_ids["tts_bos_token_id"],
                    self.special_token_ids["tts_eos_token_id"],
                    self.special_token_ids["tts_pad_token_id"],
                ]
            )
            projected = self.talker.text_projection(rows)
            mx.eval(projected)
            self.tts_special_cache = (
                projected[0:1],
                projected[1:2],
                projected[2:3],
            )
        else:
            pass
        return self.tts_special_cache

    def reconstruct_prompt_states(
        self,
        state: Qwen3OmniPipelineState,
    ) -> tuple[list[int], mx.array, mx.array, dict[str, Any]]:
        prompt = state.prompt or {}
        prompt_input_ids = prompt["input_ids"]
        if prompt_input_ids.dim() == 2:
            prompt_input_ids = prompt_input_ids[0]
        else:
            pass
        prompt_ids = [int(token_id) for token_id in prompt_input_ids.tolist()]
        prompt_embed = self.load_prompt_token_embeddings(prompt_ids)
        prompt_hidden = mx.array(prompt_embed)
        prompt_model_inputs = self.prompt_model_inputs(state)

        for name in ("audio", "image", "video"):
            token_id = self.special_token_ids.get(f"{name}_token_id")
            features = coerce_feature_array(prompt_model_inputs.get(f"{name}_embeds"))
            if token_id is None or features is None:
                continue
            else:
                pass
            positions = [
                index
                for index, prompt_id in enumerate(prompt_ids)
                if prompt_id == int(token_id)
            ]
            count = len(positions)
            if count == 0:
                continue
            else:
                pass
            if int(features.shape[0]) != count:
                raise ValueError(
                    f"{name} feature rows ({features.shape[0]}) do not match "
                    f"prompt placeholders ({count})"
                )
            else:
                pass
            indices = mx.array(positions, dtype=mx.int32)
            prompt_embed[indices] = features.astype(prompt_embed.dtype)
            prompt_hidden[indices] = mx.zeros(
                (count, prompt_hidden.shape[-1]),
                dtype=prompt_hidden.dtype,
            )
        return prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs

    def load_prompt_token_embeddings(self, token_ids: list[int]) -> mx.array:
        missing = [
            token_id
            for token_id in dict.fromkeys(token_ids)
            if token_id not in self.thinker_embed_cache
        ]
        if missing:
            rows = load_mlx_embedding_rows(
                self.model_path,
                missing,
            )
            mx.eval(rows)
            for token_id, row in zip(missing, rows):
                self.thinker_embed_cache[token_id] = mx.array(row)
        else:
            pass
        return mx.stack([self.thinker_embed_cache[token_id] for token_id in token_ids])

    @staticmethod
    def prompt_model_inputs(state: Qwen3OmniPipelineState) -> dict[str, Any]:
        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs")
        if isinstance(model_inputs, dict):
            return dict(model_inputs)
        else:
            pass
        prompt_model_inputs = dict(thinker_inputs)
        prompt_model_inputs.pop("capture_model_output_keys", None)
        return prompt_model_inputs
