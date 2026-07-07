"""Core types for manifest-driven data pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CodecEncodeMode = Literal["intra_only", "temporal_gop", "bitrate_sweep"]


@dataclass(frozen=True)
class SourceRecord:
    """One manifest row describing a training/val source."""

    id: str
    type: Literal["yuv_video", "codec_clip", "image"]
    path: str
    weight: float = 1.0
    width: int = 0
    height: int = 0
    fps: int = 0
    frames: int = 0
    pix_fmt: str = "yuv420p"
    bit_depth: int = 8
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SourceRecord:
        source_id = row.get("id") or row.get("name") or row["path"]
        return cls(
            id=str(source_id),
            type=row["type"],  # type: ignore[arg-type]
            path=row["path"],
            weight=float(row.get("weight", 1.0)),
            width=int(row.get("width", 0)),
            height=int(row.get("height", 0)),
            fps=int(row.get("fps", 0)),
            frames=int(row.get("frames", 0)),
            pix_fmt=str(row.get("pix_fmt", "yuv420p")),
            bit_depth=int(row.get("bit_depth", 8)),
            tags=tuple(row.get("tags", ())),
            extra={k: v for k, v in row.items() if k not in cls.__dataclass_fields__},
        )


@dataclass(frozen=True)
class ValSampleMeta:
    """Metadata for a fixed validation canvas sample."""

    slug: str
    sequence: str
    codec: str
    bitrate_kbps: int | None
    frame_index: int
    lr_size: tuple[int, int]
    hr_size: tuple[int, int]

    def caption(self) -> str:
        rate = f"{self.bitrate_kbps}kbps" if self.bitrate_kbps is not None else "?"
        lr_h, lr_w = self.lr_size
        hr_h, hr_w = self.hr_size
        return (
            f"{self.sequence} | {self.codec} @ {rate} | frame {self.frame_index}\n"
            f"codec LR {lr_w}x{lr_h} (NN x3) | SR {hr_w}x{hr_h} | HR {hr_w}x{hr_h}"
        )

    def data_preview_header(self, *, colorspace: str) -> str:
        rate = f"{self.bitrate_kbps}kbps" if self.bitrate_kbps is not None else "?"
        lr_h, lr_w = self.lr_size
        hr_h, hr_w = self.hr_size
        roundtrip = "YUV↔RGB" if colorspace == "yuv" else "RGB"
        return (
            f"{self.sequence} | {self.codec} @ {rate} | frame {self.frame_index}\n"
            f"codec LR {lr_w}x{lr_h} (NN x3) | HR {hr_w}x{hr_h} | HR {roundtrip} roundtrip"
        )


@dataclass
class ValSampleSpec:
    """Fixed validation row (extends SourceRecord fields via extra)."""

    source: SourceRecord
    frame_index: int = 0
    clip_start: int = 0
    codec: str = "libx264"
    bitrate_kbps: int | None = None
    crf: int | None = None
    encode_mode: CodecEncodeMode = "bitrate_sweep"
