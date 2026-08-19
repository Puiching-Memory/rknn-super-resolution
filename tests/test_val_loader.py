"""Tests for UVG fixed validation loader."""

from pathlib import Path

from rk3588_mobile_sr.data.types import SourceRecord, ValSampleSpec
from rk3588_mobile_sr.data.val_loader import (
    resolve_codec_clip_for_spec,
    resolve_codec_clip_record,
    val_sequence_name,
)


def _yuv_spec(seq: str, codec: str, bitrate: int, *, frame_index: int = 72) -> ValSampleSpec:
    return ValSampleSpec(
        source=SourceRecord(
            id=f"uvg/{seq}@{codec}@{bitrate}k",
            type="yuv_video",
            path=f"data/UVG_raw/yuv_1080p/{seq}_1920x1080_50fps.yuv",
            width=1920,
            height=1080,
            fps=50,
            frames=600,
        ),
        frame_index=frame_index,
        clip_start=60,
        codec=codec,
        bitrate_kbps=bitrate,
    )


def _codec_clip(
    seq: str,
    codec: str,
    bitrate: int,
    *,
    clip_start: int,
) -> SourceRecord:
    return SourceRecord(
        id=f"uvg/{seq}@s{clip_start}@{codec}@{bitrate}k",
        type="codec_clip",
        path=f"data/raw_cache/uvg__{seq}_s{clip_start}_g16_{codec}_{bitrate}k_lr.npy",
        width=1920,
        height=1080,
        fps=50,
        frames=24,
        extra={
            "source_id": f"uvg/{seq}",
            "source_path": f"data/UVG_raw/yuv_1080p/{seq}_1920x1080_50fps.yuv",
            "hr_path": f"data/raw_cache/uvg__{seq}_s{clip_start}_hr.npy",
            "clip_start": clip_start,
            "codec": codec,
            "bitrate_kbps": bitrate,
            "lr_height": 360,
            "lr_width": 640,
        },
    )


def test_resolve_codec_clip_prefers_clip_containing_frame_index():
    spec = _yuv_spec("FlowerFocus", "libx264", 800, frame_index=72)
    records = {
        "a": _codec_clip("FlowerFocus", "libx264", 800, clip_start=25),
        "b": _codec_clip("FlowerFocus", "libx264", 800, clip_start=60),
        "c": _codec_clip("FlowerFocus", "libx264", 800, clip_start=104),
    }
    resolved = resolve_codec_clip_record(spec, records)
    assert resolved is not None
    assert resolved.extra["clip_start"] == 60


def test_resolve_codec_clip_rejects_temporally_unrelated_clip():
    spec = _yuv_spec("Beauty", "libx264", 150, frame_index=72)
    records = {"a": _codec_clip("Beauty", "libx264", 150, clip_start=341)}
    assert resolve_codec_clip_record(spec, records) is None


def test_resolve_codec_clip_matches_sequence_codec_and_bitrate():
    spec = _yuv_spec("Beauty", "libx265", 500)
    records = {
        "a": _codec_clip("Beauty", "libx264", 500, clip_start=60),
        "b": _codec_clip("Beauty", "libx265", 800, clip_start=60),
        "c": _codec_clip("Beauty", "libx265", 500, clip_start=60),
    }
    resolved = resolve_codec_clip_record(spec, records)
    assert resolved is not None
    assert val_sequence_name(spec) == "Beauty"
    assert resolved.id.endswith("@libx265@500k")
    assert resolved.extra["clip_start"] == 60


def test_resolve_codec_clip_for_spec_direct_codec_clip():
    clip = _codec_clip("Beauty", "libx264", 800, clip_start=25)
    spec = ValSampleSpec(source=clip, frame_index=30, codec="libx264", bitrate_kbps=800)
    assert resolve_codec_clip_for_spec(spec, {}) is clip

