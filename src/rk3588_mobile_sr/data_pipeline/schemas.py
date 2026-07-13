"""Pydantic schemas for the Snakemake data pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SourceRow(BaseModel):
    """One train manifest row (raw YUV video only; still-image sources removed)."""

    id: str
    type: Literal["yuv_video"] = "yuv_video"
    path: str
    weight: float = 1.0
    width: int
    height: int
    fps: int = 30
    frames: int = 0
    pix_fmt: str = "yuv420p"
    bit_depth: int = 8
    tags: list[str] = Field(default_factory=list)

    @field_validator("width", "height")
    @classmethod
    def positive_dims(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("width and height must be positive")
        return value

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> SourceRow:
        return cls.model_validate(row)

    def safe_id(self) -> str:
        return self.id.replace("/", "__")


class ValRow(BaseModel):
    """Fixed validation manifest row."""

    id: str
    type: Literal["yuv_video"] = "yuv_video"
    path: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    frames: int = 0
    weight: float = 1.0
    clip_start: int = 0
    frame_index: int = 0
    encode_mode: Literal["temporal_gop", "intra_only", "bitrate_sweep"] = "temporal_gop"
    codec: str
    bitrate_kbps: int


class CodecClipRow(BaseModel):
    """One codec_cache manifest row consumed by train_loader."""

    id: str
    type: Literal["codec_clip"] = "codec_clip"
    path: str
    weight: float
    width: int
    height: int
    fps: int
    frames: int
    source_id: str
    source_path: str
    clip_start: int
    clip_frames: int
    codec: str
    bitrate_kbps: int
    gop: int
    lr_height: int
    lr_width: int
    encode_mode: Literal["intra_only", "temporal_gop"]
    tags: list[str] = Field(default_factory=lambda: ["codec_offline"])
    hr_mp4_path: str

    def to_json(self) -> dict[str, Any]:
        return self.model_dump()
