"""Fixed validation loader for canvas codec training."""

from __future__ import annotations

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
from rk3588_mobile_sr.data.yuv_video import YuvVideoSource


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
    """Pick diverse canvas samples across sequence, codec, and bitrate.

    Greedy over ``(sequence, codec, bitrate)`` novelty / balance — not biased
    toward 800k — so data preview and val panels cover the compression ladder.
    """
    if num <= 0 or not specs:
        return []
    num = min(num, len(specs))
    codec_rank = {"libx264": 0, "libx265": 1, "libsvtav1": 2}

    chosen: list[int] = []
    seq_count: dict[str, int] = {}
    codec_count: dict[str, int] = {}
    bitrate_count: dict[int | None, int] = {}
    remaining = set(range(len(specs)))

    while len(chosen) < num and remaining:

        def score(index: int) -> tuple[int, int, int, int, int, int, str, int]:
            spec = specs[index]
            seq = val_sequence_name(spec)
            bitrate = spec.bitrate_kbps
            novelty = (
                int(seq_count.get(seq, 0) == 0)
                + int(codec_count.get(spec.codec, 0) == 0)
                + int(bitrate_count.get(bitrate, 0) == 0)
            )
            br_value = bitrate if bitrate is not None else 10**9
            # 1) new dimensions first  2) balance counts  3) prefer harder bitrates
            return (
                -novelty,
                bitrate_count.get(bitrate, 0),
                codec_count.get(spec.codec, 0),
                seq_count.get(seq, 0),
                br_value,
                codec_rank.get(spec.codec, 9),
                seq,
                index,
            )

        best = min(remaining, key=score)
        remaining.remove(best)
        chosen.append(best)
        picked = specs[best]
        seq = val_sequence_name(picked)
        seq_count[seq] = seq_count.get(seq, 0) + 1
        codec_count[picked.codec] = codec_count.get(picked.codec, 0) + 1
        bitrate_count[picked.bitrate_kbps] = bitrate_count.get(picked.bitrate_kbps, 0) + 1

    chosen.sort(
        key=lambda i: (
            val_sequence_name(specs[i]),
            specs[i].bitrate_kbps or 0,
            codec_rank.get(specs[i].codec, 9),
            i,
        )
    )
    return chosen


def resolve_codec_clip_record(
    spec: ValSampleSpec,
    codec_records: dict[str, SourceRecord],
) -> SourceRecord | None:
    """Map a UVG val row to an offline codec_clip that covers ``frame_index``.

    Only clips with matching ``(sequence, codec, bitrate)`` whose window
    contains ``spec.frame_index`` are accepted. Prefer the clip whose
    ``clip_start`` matches the val row; never fall back to a temporally
    unrelated train clip (that used to clamp to frame 0 of the wrong window).
    """
    target_seq = val_sequence_name(spec)
    candidates: list[tuple[int, str, SourceRecord]] = []
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
        if not (clip_start <= spec.frame_index < clip_end):
            continue
        clip_delta = abs(clip_start - spec.clip_start)
        candidates.append((clip_delta, record.id, record))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]))
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


def _resize_canvas_rgb(
    rgb: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(rgb.unsqueeze(0), size=size, mode="area").squeeze(0)


def _decode_npy_frame(
    path: Path,
    frame_index: int,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Read one RGB frame from an offline-baked .npy clip as CHW float."""
    arr = np.load(path, mmap_mode="r")
    if frame_index < 0 or frame_index >= arr.shape[0]:
        raise IndexError(f"frame {frame_index} out of range for {path} (n={arr.shape[0]})")
    frame = np.array(arr[frame_index], copy=True)
    if frame.shape != (height, width, 3):
        raise RuntimeError(
            f"frame shape {frame.shape} != ({height}, {width}, 3) for {path}@{frame_index}"
        )
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()


def _decode_codec_clip_rgb_pair(
    record: SourceRecord,
    *,
    project_root: Path,
    frame_index: int,
    clip_start: int,
    lr_size: tuple[int, int],
    hr_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode LR/HR from offline-baked RGB .npy clips (same format as training)."""
    lr_path = (project_root / record.path).resolve()
    hr_rel = record.extra.get("hr_path")
    if not hr_rel:
        raise ValueError(f"codec_clip {record.id} missing hr_path for HR .npy")
    hr_path = (project_root / hr_rel).resolve()
    clip_origin = int(record.extra.get("clip_start", clip_start))
    rel = max(0, min(frame_index - clip_origin, record.frames - 1))
    lr_w = int(record.extra.get("lr_width", lr_size[1]))
    lr_h = int(record.extra.get("lr_height", lr_size[0]))
    lr_rgb = _decode_npy_frame(lr_path, rel, width=lr_w, height=lr_h)
    hr_rgb = _decode_npy_frame(
        hr_path,
        rel,
        width=record.width,
        height=record.height,
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
            if clip is None:
                raise FileNotFoundError(
                    f"No offline codec clip covers val sample {record.id} "
                    f"(seq={val_sequence_name(spec)}, codec={spec.codec}, "
                    f"bitrate={spec.bitrate_kbps}k, frame={spec.frame_index}). "
                    "Rebuild the codec cache so fixed-val bake jobs are included "
                    "(Snakemake write_codec_manifest / iter_val_encode_jobs)."
                )
            return _decode_codec_clip_rgb_pair(
                clip,
                project_root=self.project_root,
                frame_index=spec.frame_index,
                clip_start=spec.clip_start,
                lr_size=lr_size,
                hr_size=hr_size,
            )

        raise ValueError(f"Unsupported val source type {record.type!r}")

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
