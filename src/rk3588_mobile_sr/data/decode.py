"""TorchCodec GPU decoding and canvas geometry for OpenVidHD clips."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from rk3588_mobile_sr.utils.run_logger import logger

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_VIDEO_SIDECARS = ("sequence.mp4", "sequence.mkv", "sequence.webm", "sequence.mov")
_JPEG_SUFFIXES = {".jpg", ".jpeg"}
_DECODER_CACHE_SIZE = 8


class FrameDecoder(Protocol):
    def decode_batch(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]: ...

    def close(self) -> None: ...


def resolve_sequence_source(root: Path, relative_path: str) -> tuple[str, Path]:
    """Return ``("video" | "images", path)`` for an OpenVidHD sequence entry."""
    candidate = (root / relative_path).resolve()
    if candidate.is_file():
        if candidate.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported OpenVidHD media file: {candidate}")
        return "video", candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"OpenVidHD sequence path does not exist: {candidate}")
    for name in _VIDEO_SIDECARS:
        sidecar = candidate / name
        if sidecar.is_file():
            return "video", sidecar
    return "images", candidate


def apply_geometry(
    frames: torch.Tensor,
    crop: torch.Tensor | Sequence[int],
    hflip: bool,
    *,
    lr_size: tuple[int, int],
    hr_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop/flip a ``TCHW`` RGB clip and resize to the LR sequence plus HR target."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("decoded frames must have shape Tx3xHxW")
    left, top, right, bottom = (int(value) for value in crop)
    height, width = frames.shape[-2:]
    left = min(max(left, 0), width)
    right = min(max(right, left + 1), width)
    top = min(max(top, 0), height)
    bottom = min(max(bottom, top + 1), height)
    cropped = frames[:, :, top:bottom, left:right]
    if hflip:
        cropped = cropped.flip(-1)
    rgb = cropped.float()
    lr = F.interpolate(
        rgb,
        size=lr_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    hr = F.interpolate(
        rgb[-1:],
        size=hr_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return lr.round_().clamp_(0, 255), hr[0].round_().clamp_(0, 255)


def _as_chw(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim == 3:
        return frame
    if frame.ndim == 4 and frame.shape[0] == 1:
        return frame[0]
    raise ValueError(f"expected a still image CHW tensor, got shape {tuple(frame.shape)}")


class TorchCodecFrameDecoder:
    """Decode OpenVidHD clips with TorchCodec and build LR/HR canvases on ``device``.

    Video sources use ``VideoDecoder(device=cuda)`` (NVDEC by default). Still-image
    sequences use TorchCodec image decoders; JPEG can decode on GPU via nvJPEG.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        lr_size: tuple[int, int],
        hr_size: tuple[int, int],
        cache_size: int = _DECODER_CACHE_SIZE,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.device = torch.device(device)
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.cache_size = cache_size
        self._lock = Lock()
        self._decoders: OrderedDict[str, Any] = OrderedDict()
        self._fallback_logged: set[str] = set()

    def close(self) -> None:
        with self._lock:
            self._decoders.clear()

    def decode_batch(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        kinds = batch["kind"]
        lr_clips: list[torch.Tensor] = []
        hr_targets: list[torch.Tensor] = []
        with self._lock:
            for index, kind in enumerate(kinds):
                frames = self._decode_sample(kind, index, batch)
                lr, hr = apply_geometry(
                    frames,
                    batch["crop"][index],
                    bool(batch["hflip"][index].item()),
                    lr_size=self.lr_size,
                    hr_size=self.hr_size,
                )
                lr_clips.append(lr)
                hr_targets.append(hr)
        return torch.stack(lr_clips), torch.stack(hr_targets)

    def _decode_sample(self, kind: str, index: int, batch: Mapping[str, Any]) -> torch.Tensor:
        if kind == "video":
            return self._decode_video(
                str(batch["source"][index]),
                batch["frame_indices"][index],
            )
        if kind == "images":
            return self._decode_images(list(batch["paths"][index]))
        raise ValueError(f"unsupported OpenVidHD source kind {kind!r}")

    def _decode_video(self, path: str, frame_indices: torch.Tensor) -> torch.Tensor:
        decoder = self._video_decoder(path)
        indices = [int(value) for value in frame_indices.tolist()]
        frames = decoder.get_frames_at(indices).data
        self._log_cpu_fallback(path, decoder)
        if frames.ndim != 4:
            raise ValueError(f"TorchCodec video decode must return TCHW, got {tuple(frames.shape)}")
        if frames.device != self.device:
            frames = frames.to(self.device, non_blocking=True)
        return frames.contiguous()

    def _decode_images(self, paths: Sequence[str]) -> torch.Tensor:
        if not paths:
            raise ValueError("image sequence has no frame paths")
        suffixes = {Path(path).suffix.lower() for path in paths}
        if suffixes <= _JPEG_SUFFIXES:
            from torchcodec.decoders import decode_jpeg

            decoded = decode_jpeg(list(paths), device=self.device)
            if isinstance(decoded, torch.Tensor):
                stacked = decoded if decoded.ndim == 4 else decoded.unsqueeze(0)
            else:
                stacked = torch.stack([_as_chw(frame) for frame in decoded], dim=0)
            if stacked.device != self.device:
                stacked = stacked.to(self.device, non_blocking=True)
            return stacked.contiguous()

        from torchcodec.decoders import decode_image

        frames = [_as_chw(decode_image(path)) for path in paths]
        return torch.stack(frames, dim=0).to(self.device, non_blocking=True).contiguous()

    def _video_decoder(self, path: str) -> Any:
        decoder = self._decoders.get(path)
        if decoder is not None:
            self._decoders.move_to_end(path)
            return decoder
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(path, device=self.device, num_ffmpeg_threads=1)
        self._decoders[path] = decoder
        self._decoders.move_to_end(path)
        while len(self._decoders) > self.cache_size:
            self._decoders.popitem(last=False)
        return decoder

    def _log_cpu_fallback(self, path: str, decoder: Any) -> None:
        fallback = getattr(decoder, "cpu_fallback", None)
        if not fallback or path in self._fallback_logged:
            return
        self._fallback_logged.add(path)
        logger.warning(
            "TorchCodec NVDEC unavailable; CPU decode for {}: {}",
            path,
            fallback,
        )
