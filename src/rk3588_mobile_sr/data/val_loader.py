"""Fixed validation loader for canvas codec training."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rk3588_mobile_sr.data.manifest import load_manifest, load_val_manifest
from rk3588_mobile_sr.data.train_loader import TrainDataSettings
from rk3588_mobile_sr.data.types import SourceRecord, ValSampleMeta, ValSampleSpec
from rk3588_mobile_sr.data.yuv_utils import rgb_to_model_colorspace
from rk3588_mobile_sr.data.yuv_video import YuvVideoSource, read_yuv420_frame


def val_sequence_name(spec: ValSampleSpec) -> str:
    """Extract UVG sequence name from a val manifest row."""
    return spec.source.id.split("@")[0].split("/")[-1]


def val_spec_slug(spec: ValSampleSpec) -> str:
    """Filesystem-safe key for SwanLab image logging."""
    seq = val_sequence_name(spec)
    rate = f"{spec.bitrate_kbps}k" if spec.bitrate_kbps is not None else "unknown"
    return f"{seq}_{spec.codec}_{rate}".replace(".", "p")


def val_sample_meta(
    spec: ValSampleSpec,
    *,
    lr_size: tuple[int, int],
    hr_size: tuple[int, int],
) -> ValSampleMeta:
    return ValSampleMeta(
        slug=val_spec_slug(spec),
        sequence=val_sequence_name(spec),
        codec=spec.codec,
        bitrate_kbps=spec.bitrate_kbps,
        frame_index=spec.frame_index,
        lr_size=lr_size,
        hr_size=hr_size,
    )


def select_val_vis_indices(specs: list[ValSampleSpec], num: int) -> list[int]:
    """Pick diverse canvas samples: one sequence each, prefer libx264 @ 800kbps."""
    if num <= 0 or not specs:
        return []
    by_seq: dict[str, list[int]] = {}
    for index, spec in enumerate(specs):
        by_seq.setdefault(val_sequence_name(spec), []).append(index)

    def rank_index(index: int) -> tuple[int, int, int]:
        spec = specs[index]
        codec_rank = {"libx264": 0, "libx265": 1, "libsvtav1": 2}.get(spec.codec, 9)
        bitrate = spec.bitrate_kbps or 0
        target_delta = abs(bitrate - 800)
        return (target_delta, codec_rank, -bitrate)

    chosen = [min(indices, key=rank_index) for indices in by_seq.values()]
    chosen.sort(key=lambda i: val_sequence_name(specs[i]))
    if len(chosen) >= num:
        return chosen[:num]
    extras = [i for i in range(len(specs)) if i not in chosen]
    extras.sort(key=rank_index)
    return chosen + extras[: num - len(chosen)]


def resolve_codec_clip_record(
    spec: ValSampleSpec,
    codec_records: dict[str, SourceRecord],
) -> SourceRecord | None:
    """Map a UVG val row to the closest offline codec_clip in the codec cache."""
    target_seq = val_sequence_name(spec)
    candidates: list[tuple[int, int, SourceRecord]] = []
    for record in codec_records.values():
        source_id = str(record.extra.get("source_id", record.id))
        seq = source_id.split("/")[-1]
        if seq != target_seq:
            continue
        if record.extra.get("codec") != spec.codec:
            continue
        if record.extra.get("bitrate_kbps") != spec.bitrate_kbps:
            continue
        clip_start = int(record.extra.get("clip_start", 0))
        clip_end = clip_start + int(record.frames)
        if clip_start <= spec.frame_index < clip_end:
            contains = 0
        else:
            contains = min(abs(spec.frame_index - clip_start), abs(spec.frame_index - clip_end))
        clip_delta = abs(clip_start - spec.clip_start)
        candidates.append((contains, clip_delta, record))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2].id))
    return candidates[0][2]


def resolve_codec_clip_for_spec(
    spec: ValSampleSpec,
    codec_records: dict[str, SourceRecord],
) -> SourceRecord | None:
    """Return the offline codec_clip record backing a val sample, if any."""
    record = spec.source
    if record.type == "codec_clip":
        return record
    if record.type == "yuv_video":
        return resolve_codec_clip_record(spec, codec_records)
    return None


def codec_clip_paths_for_spec(
    spec: ValSampleSpec,
    codec_records: dict[str, SourceRecord],
    project_root: Path,
) -> tuple[Path, Path] | None:
    """Resolve on-disk LR codec and HR lossless MP4 paths for a val sample."""
    clip = resolve_codec_clip_for_spec(spec, codec_records)
    if clip is None:
        return None
    hr_mp4 = clip.extra.get("hr_mp4_path")
    if not hr_mp4:
        return None
    lr_path = (project_root / clip.path).resolve()
    hr_path = (project_root / hr_mp4).resolve()
    if not lr_path.is_file() or not hr_path.is_file():
        return None
    return lr_path, hr_path


def _resize_canvas_rgb(
    rgb: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(rgb.unsqueeze(0), size=size, mode="area").squeeze(0)


def _decode_codec_clip_rgb_pair(
    record: SourceRecord,
    *,
    project_root: Path,
    frame_index: int,
    clip_start: int,
    lr_size: tuple[int, int],
    hr_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode LR from the codec mp4 (ffmpeg) and HR from the *raw source YUV*.

    HR supervision is read directly from the original YUV420p file so the
    validation target is bit-exact (no mezzanine re-encode). LR is decoded from
    the offline codec clip via an ffmpeg subprocess (torchcodec removed).
    """
    lr_path = (project_root / record.path).resolve()
    source_path = record.extra.get("source_path")
    if not source_path:
        raise ValueError(f"codec_clip {record.id} missing source_path for HR YUV")
    hr_yuv_path = (project_root / source_path).resolve()
    clip_origin = int(record.extra.get("clip_start", clip_start))
    rel = max(0, min(frame_index - clip_origin, record.frames - 1))
    lr_w = int(record.extra.get("lr_width", lr_size[1]))
    lr_h = int(record.extra.get("lr_height", lr_size[0]))
    lr_rgb = _decode_mp4_frame_ffmpeg(lr_path, rel, width=lr_w, height=lr_h)
    hr_abs_frame = clip_origin + rel
    hr_rgb = read_yuv420_frame(
        hr_yuv_path,
        width=record.width,
        height=record.height,
        frame_index=hr_abs_frame,
    )
    return (
        _resize_canvas_rgb(lr_rgb, size=lr_size),
        _resize_canvas_rgb(hr_rgb, size=hr_size),
    )


def iter_val_batches(
    loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield LR/HR tensor batches from a validation loader."""
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            yield batch[0], batch[1]
        else:
            raise TypeError("validation loader batch must contain LR and HR tensors")


def _collate_pairs(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    lr = torch.stack([item[0] for item in batch], dim=0)
    hr = torch.stack([item[1] for item in batch], dim=0)
    return lr, hr


def _decode_mp4_frame_ffmpeg(
    path: Path,
    frame_index: int,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Decode a single RGB frame from an mp4 via an ffmpeg subprocess."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed for {path}@{frame_index}: {err}")
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = width * height * 3
    if arr.size != expected:
        raise RuntimeError(
            f"decoded frame size {arr.size} != {expected} ({width}x{height}x3) "
            f"for {path}@{frame_index}"
        )
    rgb = arr.reshape(height, width, 3)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


class FixedValDataset(Dataset):
    """Deterministic val samples."""

    def __init__(
        self,
        specs: list[ValSampleSpec],
        *,
        settings: TrainDataSettings,
        project_root: Path,
        codec_records: dict[str, SourceRecord],
    ) -> None:
        self.specs = specs
        self.settings = settings
        self.project_root = project_root
        self.codec_records = codec_records
        self._yuv_sources: dict[str, YuvVideoSource] = {}

    def _yuv(self, record: SourceRecord) -> YuvVideoSource:
        if record.id not in self._yuv_sources:
            self._yuv_sources[record.id] = YuvVideoSource(record, self.project_root)
        return self._yuv_sources[record.id]

    def __len__(self) -> int:
        return len(self.specs)

    def decode_rgb_pair(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode aligned LR/HR canvas frames as RGB before colorspace conversion."""
        spec = self.specs[index]
        record = spec.source
        lr_size = self.settings.lr_size
        hr_size = self.settings.hr_size

        if record.type == "codec_clip":
            return _decode_codec_clip_rgb_pair(
                record,
                project_root=self.project_root,
                frame_index=spec.frame_index,
                clip_start=spec.clip_start,
                lr_size=lr_size,
                hr_size=hr_size,
            )

        if record.type == "yuv_video":
            clip = resolve_codec_clip_record(spec, self.codec_records)
            if clip is not None:
                return _decode_codec_clip_rgb_pair(
                    clip,
                    project_root=self.project_root,
                    frame_index=spec.frame_index,
                    clip_start=spec.clip_start,
                    lr_size=lr_size,
                    hr_size=hr_size,
                )

            yuv = self._yuv(record)
            hr_rgb = _resize_canvas_rgb(
                yuv.read_frame(spec.frame_index),
                size=hr_size,
            )
            lr_rgb = _resize_canvas_rgb(hr_rgb, size=lr_size)
            return lr_rgb, hr_rgb

        raise ValueError(f"Unsupported val source type {record.type!r}")

    def codec_clip_paths_for_index(self, index: int) -> tuple[Path, Path] | None:
        return codec_clip_paths_for_spec(
            self.specs[index],
            self.codec_records,
            self.project_root,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_rgb, hr_rgb = self.decode_rgb_pair(index)
        lr = rgb_to_model_colorspace(
            lr_rgb,
            colorspace=self.settings.colorspace,
            nv12_simulate=self.settings.nv12_simulate,
        )
        hr = rgb_to_model_colorspace(
            hr_rgb,
            colorspace=self.settings.colorspace,
            nv12_simulate=self.settings.nv12_simulate,
        )
        return lr, hr


def build_val_loader(
    settings: TrainDataSettings,
    *,
    val_manifest: str,
    batch_size: int = 1,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
) -> tuple[DataLoader, DistributedSampler | None]:
    root = Path(settings.project_root or Path(__file__).resolve().parents[3])
    specs = load_val_manifest(val_manifest, project_root=root)
    codec_records = {
        r.id: r
        for r in load_manifest(settings.codec_manifest, project_root=root)
        if r.type == "codec_clip"
    }
    dataset = FixedValDataset(
        specs,
        settings=settings,
        project_root=root,
        codec_records=codec_records,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        pin_memory=True,
    )
    return loader, sampler
