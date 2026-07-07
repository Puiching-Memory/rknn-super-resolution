"""SwanLab validation image helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rk3588_mobile_sr.config import default_config_path
from rk3588_mobile_sr.data.types import ValSampleSpec, SourceRecord
from rk3588_mobile_sr.data.val_loader import select_val_vis_indices, val_spec_slug
from rk3588_mobile_sr.data.yuv_utils import rgb_to_yuv444
from rk3588_mobile_sr.utils.swanlab_logging import (
    _append_codec_clip_videos,
    _looks_like_local_run_suffix,
    collect_sr_validation_panels,
    find_swanlab_run_id,
    load_swanlab_run_record,
    lookup_swanlab_cloud_run_id,
    make_codec_canvas_panel,
    make_data_preview_panel,
    resolve_swanlab_run_id,
    rgb_diff_stats,
    save_swanlab_run_record,
)
from rk3588_mobile_sr.utils.train_framework import resolve_colorspace


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _spec(seq: str, codec: str, bitrate: int) -> ValSampleSpec:
    return ValSampleSpec(
        source=SourceRecord(
            id=f"uvg/{seq}@{codec}@{bitrate}k",
            type="yuv_video",
            path=f"data/{seq}.yuv",
            width=1920,
            height=1080,
            fps=50,
            frames=600,
        ),
        frame_index=72,
        clip_start=60,
        codec=codec,
        bitrate_kbps=bitrate,
    )


def test_append_codec_clip_videos_adds_lr_and_hr_keys(tmp_path: Path, monkeypatch):
    lr = tmp_path / "lr.mp4"
    hr = tmp_path / "hr.mp4"
    lr.write_bytes(b"lr")
    hr.write_bytes(b"hr")

    class _FakeVideo:
        def __init__(self, path: str, *, caption: str = "") -> None:
            self.path = path
            self.caption = caption

    monkeypatch.setattr(
        "rk3588_mobile_sr.utils.swanlab_logging.swanlab.Video",
        _FakeVideo,
    )
    payload: dict = {}
    _append_codec_clip_videos(
        payload,
        key_prefix="val",
        slug="Beauty_libx264_800k",
        lr_video=lr,
        hr_video=hr,
        caption="Beauty | libx264 @ 800kbps",
    )
    assert set(payload) == {
        "val/Beauty_libx264_800k/lr_clip",
        "val/Beauty_libx264_800k/hr_clip",
    }
    assert payload["val/Beauty_libx264_800k/lr_clip"].path == str(lr)


def test_resolve_colorspace_from_yaml():
    args = argparse.Namespace(colorspace=None, config=str(default_config_path()))
    assert resolve_colorspace(args) == "yuv"


def test_select_val_vis_indices_prefers_diverse_sequences():
    specs = [
        _spec("Beauty", "libx264", 300),
        _spec("Beauty", "libx264", 800),
        _spec("Bosphorus", "libx265", 800),
        _spec("CityAlley", "libsvtav1", 500),
    ]
    indices = select_val_vis_indices(specs, 3)
    assert len(indices) == 3
    assert indices[0] == 1  # Beauty @ 800k
    sequences = {specs[i].source.id.split("@")[0].split("/")[-1] for i in indices}
    assert sequences == {"Beauty", "Bosphorus", "CityAlley"}


def test_val_spec_slug():
    assert val_spec_slug(_spec("FlowerFocus", "libx264", 800)) == "FlowerFocus_libx264_800k"


def test_make_codec_canvas_panel_uses_nearest_lr_upscale():
    lr = torch.zeros(3, 12, 12)
    lr[:, 0::3, 0::3] = 255.0
    sr = torch.zeros(3, 36, 36)
    hr = torch.zeros(3, 36, 36)
    panel = make_codec_canvas_panel(lr, sr, hr, include_detail=False)
    assert panel.shape == (36, 36 * 3, 3)
    assert panel[:, :36, 0].max() == 255


def test_make_data_preview_panel_includes_roundtrip_column():
    lr = torch.zeros(3, 12, 12)
    lr[:, 0::3, 0::3] = 255.0
    hr = torch.zeros(3, 36, 36)
    hr[0] = 180.0
    hr[1] = 90.0
    hr[2] = 45.0
    panel = make_data_preview_panel(lr, hr, colorspace="yuv")
    assert panel.shape == (72, 36 * 3, 3)


def test_rgb_diff_stats_zero_for_identical_images():
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    stats = rgb_diff_stats(img, img)
    assert stats["mean_abs"] == 0.0
    assert stats["psnr"] == 100.0


def test_collect_sr_validation_panels_converts_yuv_to_rgb():
    rgb = torch.zeros(3, 32, 32)
    rgb[0] = 200.0
    yuv = rgb_to_yuv444(rgb.unsqueeze(0))[0]
    loader = iter([(yuv.unsqueeze(0), yuv.unsqueeze(0))])

    panels = collect_sr_validation_panels(
        _Identity(),
        loader,
        torch.device("cpu"),
        num_samples=1,
        colorspace="yuv",
        include_detail=False,
    )
    hr_col = panels[0][:, 64:96, :]
    assert hr_col[..., 0].mean() > hr_col[..., 1].mean()


def test_find_swanlab_run_id_prefers_earliest_matching_experiment(tmp_path: Path):
    swanlog = tmp_path / "swanlog"
    for stamp, run_id, experiment in [
        ("20260703_160900", "sx4e2ajn", "stage1-8gpu-20260703-1608"),
        ("20260703_170951", "732xjcw5", "stage1-8gpu-20260703-1608"),
    ]:
        run_dir = swanlog / f"run-{stamp}-{run_id}"
        files = run_dir / "files"
        files.mkdir(parents=True)
        meta = {
            "runtime": {
                "command": f"... --swanlab_experiment {experiment}",
            }
        }
        (files / "swanlab-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    assert (
        find_swanlab_run_id(tmp_path, "stage1-8gpu-20260703-1608") == "sx4e2ajn"
    )


def test_save_and_load_swanlab_run_record(tmp_path: Path):
    save_swanlab_run_record(
        tmp_path,
        run_id="gjid8vrb1jigurxquwabt",
        project="rk3588-mobile-sr",
        experiment_name="stage1-8gpu-20260703-1608",
    )
    record = load_swanlab_run_record(tmp_path)
    assert record is not None
    assert record["run_id"] == "gjid8vrb1jigurxquwabt"
    assert find_swanlab_run_id(tmp_path, None) == "gjid8vrb1jigurxquwabt"


def test_local_run_suffix_detection():
    assert _looks_like_local_run_suffix("sx4e2ajn")
    assert not _looks_like_local_run_suffix("gjid8vrb1jigurxquwabt")


def test_resolve_swanlab_run_id_uses_cloud_lookup_for_local_suffix(tmp_path: Path, monkeypatch):
    save_swanlab_run_record(
        tmp_path,
        run_id="sx4e2ajn",
        project="rk3588-mobile-sr",
        experiment_name="stage1-8gpu-20260703-1608",
    )
    monkeypatch.setattr(
        "rk3588_mobile_sr.utils.swanlab_logging.lookup_swanlab_cloud_run_id",
        lambda project, experiment_name: "gjid8vrb1jigurxquwabt",
    )
    run_id = resolve_swanlab_run_id(
        save_dir=tmp_path,
        project="rk3588-mobile-sr",
        experiment_name="stage1-8gpu-20260703-1608",
        resume_checkpoint="dummy.pth",
    )
    assert run_id == "gjid8vrb1jigurxquwabt"


def test_lookup_swanlab_cloud_run_id_live():
    run_id = lookup_swanlab_cloud_run_id(
        "rk3588-mobile-sr",
        "stage1-8gpu-20260703-1608",
    )
    assert run_id == "gjid8vrb1jigurxquwabt"
