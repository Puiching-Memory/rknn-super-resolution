"""Raw YUV420p video source with frame and clip seeking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rk3588_mobile_sr.data.types import SourceRecord


def yuv420_frame_bytes(width: int, height: int) -> int:
    return width * height * 3 // 2


def read_yuv420_frame(path: Path, *, width: int, height: int, frame_index: int) -> torch.Tensor:
    """Read one YUV420p frame and return RGB float tensor CHW in [0, 255]."""
    frame_bytes = yuv420_frame_bytes(width, height)
    offset = frame_index * frame_bytes
    file_size = path.stat().st_size
    if offset + frame_bytes > file_size:
        raise IndexError(
            f"frame {frame_index} out of range for {path} "
            f"(frames={file_size // frame_bytes}, need {frame_bytes} bytes)"
        )

    y_size = width * height
    uv_size = y_size // 4
    with path.open("rb") as handle:
        handle.seek(offset)
        plane_y = np.frombuffer(handle.read(y_size), dtype=np.uint8).reshape(height, width)
        plane_u = np.frombuffer(handle.read(uv_size), dtype=np.uint8).reshape(height // 2, width // 2)
        plane_v = np.frombuffer(handle.read(uv_size), dtype=np.uint8).reshape(height // 2, width // 2)

    rgb = _yuv420_to_rgb_numpy(plane_y, plane_u, plane_v)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


def read_yuv420_clip(
    path: Path,
    *,
    width: int,
    height: int,
    start: int,
    count: int,
) -> torch.Tensor:
    """Return RGB clip tensor NCHW float [0, 255]."""
    frames = [
        read_yuv420_frame(path, width=width, height=height, frame_index=start + i) for i in range(count)
    ]
    return torch.stack(frames, dim=0)


def read_yuv420_patch(
    path: Path,
    *,
    width: int,
    height: int,
    frame_index: int,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    mmap: np.memmap | None = None,
) -> torch.Tensor:
    """Read a YUV420p crop and return RGB float CHW in [0, 255].

    Coordinates are forced even for YUV420 chroma alignment. Prefer passing a
    cached ``mmap`` from the caller to avoid re-opening the file every crop.
    """
    top = top & ~1
    left = left & ~1
    crop_h = crop_h & ~1
    crop_w = crop_w & ~1
    if top < 0 or left < 0 or top + crop_h > height or left + crop_w > width:
        raise ValueError(
            f"crop ({top},{left},{crop_h}x{crop_w}) outside {width}x{height} for {path}"
        )

    frame_bytes = yuv420_frame_bytes(width, height)
    offset = frame_index * frame_bytes
    y_size = width * height
    uv_w = width // 2
    uv_top = top // 2
    uv_left = left // 2
    uv_crop_h = crop_h // 2
    uv_crop_w = crop_w // 2

    own_mmap = mmap is None
    mm = np.memmap(path, mode="r", dtype=np.uint8) if own_mmap else mmap
    assert mm is not None
    try:
        if offset + frame_bytes > mm.size:
            raise IndexError(
                f"frame {frame_index} out of range for {path} "
                f"(frames={mm.size // frame_bytes})"
            )

        y_rows = [
            np.array(
                mm[offset + row * width + left : offset + row * width + left + crop_w],
                copy=True,
            )
            for row in range(top, top + crop_h)
        ]
        plane_y = np.stack(y_rows, axis=0)

        u_base = offset + y_size
        u_rows = [
            np.array(
                mm[
                    u_base
                    + row * uv_w
                    + uv_left : u_base
                    + row * uv_w
                    + uv_left
                    + uv_crop_w
                ],
                copy=True,
            )
            for row in range(uv_top, uv_top + uv_crop_h)
        ]
        plane_u = np.stack(u_rows, axis=0)

        v_base = offset + y_size + (y_size // 4)
        v_rows = [
            np.array(
                mm[
                    v_base
                    + row * uv_w
                    + uv_left : v_base
                    + row * uv_w
                    + uv_left
                    + uv_crop_w
                ],
                copy=True,
            )
            for row in range(uv_top, uv_top + uv_crop_h)
        ]
        plane_v = np.stack(v_rows, axis=0)
    finally:
        if own_mmap:
            del mm

    rgb = _yuv420_to_rgb_numpy(plane_y, plane_u, plane_v)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


def _yuv420_to_rgb_numpy(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[: y.shape[0], : y.shape[1]]
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[: y.shape[0], : y.shape[1]]
    y_f = y.astype(np.float32)
    u_f = u_up.astype(np.float32) - 128.0
    v_f = v_up.astype(np.float32) - 128.0
    r = y_f + 1.402 * v_f
    g = y_f - 0.344 * u_f - 0.714 * v_f
    b = y_f + 1.772 * u_f
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


class YuvVideoSource:
    """UVG-style raw YUV420p file source."""

    def __init__(self, record: SourceRecord, project_root: Path) -> None:
        self.record = record
        self.path = (project_root / record.path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    @property
    def frames(self) -> int:
        if self.record.frames > 0:
            return self.record.frames
        frame_bytes = yuv420_frame_bytes(self.record.width, self.record.height)
        return self.path.stat().st_size // frame_bytes

    def read_frame(self, index: int) -> torch.Tensor:
        return read_yuv420_frame(
            self.path,
            width=self.record.width,
            height=self.record.height,
            frame_index=index,
        )

    def read_patch(
        self,
        frame_index: int,
        top: int,
        left: int,
        crop_h: int,
        crop_w: int,
    ) -> torch.Tensor:
        return read_yuv420_patch(
            self.path,
            width=self.record.width,
            height=self.record.height,
            frame_index=frame_index,
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
        )

    def read_clip(self, start: int, count: int) -> torch.Tensor:
        return read_yuv420_clip(
            self.path,
            width=self.record.width,
            height=self.record.height,
            start=start,
            count=count,
        )

