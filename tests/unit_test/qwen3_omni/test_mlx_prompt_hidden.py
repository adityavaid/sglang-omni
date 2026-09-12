# SPDX-License-Identifier: Apache-2.0
"""Preserve MLX thinker prompt captures through speech-stage transport."""

from types import SimpleNamespace

import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_omni.mlx.runner import Qwen3OmniThinkerMlxRunner
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
    Qwen3OmniMlxTalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor


def _fixture(tmp_path, media_ids):
    special = dict(
        im_start_token_id=1,
        im_end_token_id=2,
        system_token_id=3,
        user_token_id=4,
        assistant_token_id=5,
        audio_token_id=6,
        image_token_id=7,
        video_token_id=8,
        tts_bos_token_id=9,
        tts_eos_token_id=10,
        tts_pad_token_id=11,
        codec_nothink_id=12,
        codec_think_bos_id=13,
        codec_think_eos_id=14,
        codec_pad_id=15,
        codec_bos_id=16,
    )
    prompt_ids = [1, 4, 20, *media_ids, 2, 1, 5, 20]
    features = {
        f"{name}_embeds": torch.full((media_ids.count(token), 4), -99.0)
        for name, token in (("audio", 6), ("image", 7), ("video", 8))
        if token in media_ids
    }
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor(prompt_ids)},
        thinker_inputs={"model_inputs": features},
    )
    payload = StagePayload(
        request_id="media",
        request=OmniRequest(inputs=[], params={}),
        data=state.to_dict(),
    )
    talker = SimpleNamespace(
        text_projection=lambda x: x * 2,
        hidden_projection=lambda x: x * 3 + 1,
        model=SimpleNamespace(codec_embedding=lambda ids: mx.zeros((len(ids), 4))),
    )
    builder = Qwen3OmniMlxTalkerPrefillBuilder(
        talker=talker,
        model_path=str(tmp_path),
        special_token_ids=special,
        speaker_map={"ethan": 17},
    )
    builder._thinker_embed_cache = {
        token: mx.full((4,), float(token)) for token in range(30)
    }
    return builder, payload, prompt_ids


@pytest.mark.parametrize("media_ids", [[7], [6], [8], [7, 6, 8, 7]])
def test_mlx_prompt_layer24_survives_capture_stream_and_prefill(tmp_path, media_ids):
    builder, payload, prompt_ids = _fixture(tmp_path, media_ids)
    n = len(prompt_ids)
    layer24 = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 100
    runner = object.__new__(Qwen3OmniThinkerMlxRunner)
    runner.capture_layers = (0, 24)
    runner.accept_hidden_layer = 24
    pending = SimpleNamespace()
    runner._attach_hidden_states(
        pending,
        SimpleNamespace(
            hidden_states={
                "embed": mx.zeros((n, 4)),
                24: mx.array(layer24.numpy()),
            }
        ),
        capture_prompt=True,
    )
    captures = runner.pop_hidden_states(pending)
    data = SimpleNamespace(
        stage_payload=payload, req=SimpleNamespace(inflight_middle_chunks=0)
    )
    scheduled = SimpleNamespace(request_id="media", data=data)
    output = SGLangOutputProcessor(capture_hidden=True).process(
        SimpleNamespace(
            next_token_ids=torch.tensor([21]),
            logits_output=SimpleNamespace(hidden_states=captures),
        ),
        SimpleNamespace(
            requests=[scheduled], batch_data=SimpleNamespace(reqs=[data.req])
        ),
    )["media"]
    chunks = make_thinker_stream_output_builder(mlx_prompt_hidden_layer=24)(
        "media", data, output
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["token_id"] == 21
    assert chunks[0].metadata["prompt_hidden_layer"] == 24
    torch.testing.assert_close(chunks[0].data, layer24)
    chunks.append(SimpleNamespace(data=torch.zeros(4), metadata={"token_id": 2}))
    result = builder.build_prompt_prefill(payload, chunks, thinker_done=True)
    for pos in range(3, 3 + len(media_ids)):
        torch.testing.assert_close(result["input_embeds"][pos], layer24[pos] * 3 + 1)
    torch.testing.assert_close(result["input_embeds"][2], torch.full((4,), 40.0))
    assert len(result["pending_text_queue"]) == 1
    assert runner.pop_hidden_states(pending) is None


@pytest.mark.parametrize("case", ["missing", "wrong_layer", "short"])
def test_mlx_media_prefill_rejects_missing_or_misaligned_capture(tmp_path, case):
    builder, payload, ids = _fixture(tmp_path, [7])
    metadata = {"token_id": 21}
    rows = torch.ones((len(ids), 4))
    if case != "missing":
        metadata["prompt_hidden_layer"] = 23 if case == "wrong_layer" else 24
    if case == "short":
        rows = rows[:-1]
    with pytest.raises(ValueError, match="prompt.*hidden|hidden.*prompt"):
        builder.build_prompt_prefill(
            payload, [SimpleNamespace(data=rows, metadata=metadata)], thinker_done=True
        )


def test_mlx_text_prefill_does_not_require_prompt_capture(tmp_path):
    builder, payload, _ = _fixture(tmp_path, [])
    result = builder.build_prompt_prefill(
        payload,
        [SimpleNamespace(data=torch.zeros(4), metadata={"token_id": 21})],
        thinker_done=True,
    )
    torch.testing.assert_close(result["input_embeds"][2], torch.full((4,), 40.0))
