"""Tests for TorchCodec clip geometry and source resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rknn_super_resolution.data.decode import (
    TorchCodecFrameDecoder,
    apply_geometry,
    require_video_file,
)


def test_apply_geometry_crops_flips_and_resizes():
    frames = torch.zeros(2, 3, 8, 12, dtype=torch.uint8)
    frames[:, 0, :, 4:8] = 40
    frames[:, 0, :, 8:] = 200
    frames[:, 1] = 10
    lr, hr = apply_geometry(
        frames,
        (4, 0, 12, 8),
        True,
        lr_size=(4, 4),
        hr_size=(12, 12),
    )
    assert lr.shape == (2, 3, 4, 4)
    assert hr.shape == (2, 3, 12, 12)
    assert lr.dtype == torch.float32
    assert hr.min() >= 0 and hr.max() <= 255
    # Horizontal flip moves the red block from the right half to the left.
    assert lr[0, 0, 0, 0] > lr[0, 0, 0, -1]


def test_require_video_file_accepts_mp4(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    assert require_video_file(video) == video.resolve()


def test_require_video_file_rejects_directory_and_images(tmp_path: Path):
    sequence = tmp_path / "clip"
    sequence.mkdir()
    (sequence / "im00000.webp").write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported OpenVidHD media"):
        require_video_file(sequence)
    with pytest.raises(ValueError, match="unsupported OpenVidHD media"):
        require_video_file(sequence / "im00000.webp")


def test_torchcodec_decoder_uses_video_backend(monkeypatch):
    class _Batch:
        def __init__(self, data: torch.Tensor) -> None:
            self.data = data

    class _FakeVideoDecoder:
        cpu_fallback = False

        def __init__(self, source, device=None, num_ffmpeg_threads=1) -> None:
            self.source = source
            self.device = device
            self.num_ffmpeg_threads = num_ffmpeg_threads

        def get_frames_at(self, indices):
            frames = torch.zeros(len(indices), 3, 8, 12, dtype=torch.uint8)
            frames[:, 0] = 180
            return _Batch(frames)

    def _video_decoder(self, path: str):
        return _FakeVideoDecoder(path, device=self.device)

    monkeypatch.setattr(TorchCodecFrameDecoder, "_video_decoder", _video_decoder)
    decoder = TorchCodecFrameDecoder(
        torch.device("cpu"),
        lr_size=(4, 6),
        hr_size=(12, 18),
    )
    lr, hr = decoder.decode_batch(
        {
            "source": ["/tmp/clip.mp4"],
            "frame_indices": torch.tensor([[2, 3, 4]]),
            "crop": torch.tensor([[0, 0, 12, 8]]),
            "hflip": torch.tensor([False]),
        }
    )
    assert lr.shape == (1, 3, 3, 4, 6)
    assert hr.shape == (1, 3, 3, 12, 18)
    decoder.close()
