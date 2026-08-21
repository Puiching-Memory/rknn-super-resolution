"""Tests for TorchCodec clip geometry and source resolution."""

from __future__ import annotations

from pathlib import Path

import torch

from rk3588_mobile_sr.data.decode import (
    TorchCodecFrameDecoder,
    apply_geometry,
    resolve_sequence_source,
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
    assert hr.shape == (3, 12, 12)
    assert lr.dtype == torch.float32
    assert hr.min() >= 0 and hr.max() <= 255
    # Horizontal flip moves the red block from the right half to the left.
    assert lr[0, 0, 0, 0] > lr[0, 0, 0, -1]


def test_resolve_sequence_source_uses_sidecar(tmp_path: Path):
    sequence = tmp_path / "clip"
    sequence.mkdir()
    (sequence / "im00000.webp").write_bytes(b"")
    sidecar = sequence / "sequence.mp4"
    sidecar.write_bytes(b"video")
    kind, source = resolve_sequence_source(tmp_path, "clip")
    assert kind == "video"
    assert source == sidecar.resolve()


def test_resolve_sequence_source_images_directory(tmp_path: Path):
    sequence = tmp_path / "clip"
    sequence.mkdir()
    (sequence / "im00000.png").write_bytes(b"")
    kind, source = resolve_sequence_source(tmp_path, "clip")
    assert kind == "images"
    assert source == sequence.resolve()


def test_resolve_sequence_source_video_file(tmp_path: Path):
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"video")
    kind, source = resolve_sequence_source(tmp_path, "clip.mkv")
    assert kind == "video"
    assert source == video.resolve()


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
            "kind": ["video"],
            "source": ["/tmp/clip.mp4"],
            "paths": [[]],
            "frame_indices": torch.tensor([[2, 3, 4]]),
            "crop": torch.tensor([[0, 0, 12, 8]]),
            "hflip": torch.tensor([False]),
        }
    )
    assert lr.shape == (1, 3, 3, 4, 6)
    assert hr.shape == (1, 3, 12, 18)
    decoder.close()
