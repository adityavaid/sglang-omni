# SPDX-License-Identifier: Apache-2.0
"""Video input on Apple must not require TorchCodec's external FFmpeg libraries."""

import av
import numpy as np
import pytest

from sglang_omni.preprocessing.video import VideoDecodeError, load_video_path


@pytest.fixture
def red_video(tmp_path):
    path = tmp_path / "red.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 112
        stream.pix_fmt = "yuv420p"
        for _ in range(8):
            pixels = np.zeros((112, 112, 3), dtype=np.uint8)
            pixels[:, :, 0] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_pyav_reader_samples_resizes_and_preserves_rgb(monkeypatch, red_video):
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_READER", "pyav")
    video, fps = load_video_path(
        red_video, fps=2, max_frames=4, min_pixels=3136, max_pixels=12544
    )
    assert video.shape == (4, 3, 112, 112)
    assert fps == pytest.approx(2.0)
    assert video[:, 0].mean().item() > 240
    assert video[:, 1:].abs().max().item() < 10


def test_pyav_reader_surfaces_corrupt_input(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_READER", "pyav")
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(VideoDecodeError, match="corrupt.mp4"):
        load_video_path(path)


def test_video_reader_rejects_unknown_backend(monkeypatch, red_video):
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_READER", "typo")
    with pytest.raises(VideoDecodeError, match="Unsupported.*typo"):
        load_video_path(red_video)
