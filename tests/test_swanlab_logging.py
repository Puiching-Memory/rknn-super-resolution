"""SwanLab validation image helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rknn_super_resolution.config import default_config_path
from rknn_super_resolution.data.yuv_utils import rgb_to_yuv444
from rknn_super_resolution.utils.swanlab_logging import (
    _looks_like_local_run_suffix,
    build_swanlab_run_config,
    collect_data_preview_samples,
    collect_sr_validation_panels,
    find_swanlab_run_id,
    load_swanlab_run_record,
    make_data_preview_panel,
    make_sr_panel,
    resolve_swanlab_run_id,
    rgb_diff_stats,
    save_sr_panels,
    save_swanlab_run_record,
)
from rknn_super_resolution.utils.train_framework import resolve_colorspace


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_resolve_colorspace_from_yaml():
    args = argparse.Namespace(colorspace=None, config=str(default_config_path()))
    assert resolve_colorspace(args) == "yuv"


def test_make_sr_panel_uses_nearest_lr_upscale():
    lr = torch.zeros(3, 12, 12)
    lr[:, 0::3, 0::3] = 255.0
    sr = torch.zeros(3, 36, 36)
    hr = torch.zeros(3, 36, 36)
    panel = make_sr_panel(lr, sr, hr, include_detail=False)
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


def test_collect_data_preview_forwards_train_colorspace(monkeypatch):
    seen: dict[str, str] = {}

    def fake_panel(lr, hr, *, colorspace="yuv", **kwargs):
        seen["colorspace"] = colorspace
        return np.zeros((8, 24, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "rknn_super_resolution.utils.swanlab_logging.make_data_preview_panel",
        fake_panel,
    )
    rgb = torch.zeros(1, 3, 8, 8)
    collect_data_preview_samples(iter([(rgb, rgb)]), num_samples=1, colorspace="yuv")
    assert seen["colorspace"] == "yuv"


def test_build_swanlab_run_config_omits_nones_and_records_vmaf_v1():
    args = argparse.Namespace(
        vmaf_model="1080p",
        colorspace="yuv",
        q_indices=[0, 21, 42, 63],
        resume=None,
        num_workers=4,
    )
    config = build_swanlab_run_config(args, world_size=2)
    assert "resume" not in config
    assert config["colorspace"] == "yuv"
    assert config["q_indices"] == [0, 21, 42, 63]
    assert config["vmaf_family"] == "v1"
    assert config["vmaf_model_id"] == "vmaf_v1.0.16_3d0h"
    assert config["vmaf_enc_size"] == [360, 640]
    assert config["world_size"] == 2
    assert not any(value == {} for value in config.values())


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


def test_save_sr_panels_writes_png(tmp_path: Path):
    panel = np.zeros((36, 108, 3), dtype=np.uint8)
    out = save_sr_panels([panel], tmp_path)
    path = out / "sample_000.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_find_swanlab_run_id_prefers_earliest_matching_experiment(tmp_path: Path):
    swanlog = tmp_path / "swanlog"
    for stamp, run_id, experiment in [
        ("20260821_160900", "sx4e2ajn", "train-8gpu-20260821-1608"),
        ("20260821_170951", "732xjcw5", "train-8gpu-20260821-1608"),
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

    assert find_swanlab_run_id(tmp_path, "train-8gpu-20260821-1608") == "sx4e2ajn"


def test_save_and_load_swanlab_run_record(tmp_path: Path):
    save_swanlab_run_record(
        tmp_path,
        run_id="gjid8vrb1jigurxquwabt",
        project="rknn-super-resolution",
        experiment_name="train-8gpu-20260821-1608",
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
        project="rknn-super-resolution",
        experiment_name="train-8gpu-20260821-1608",
    )
    monkeypatch.setattr(
        "rknn_super_resolution.utils.swanlab_logging.lookup_swanlab_cloud_run_id",
        lambda project, experiment_name: "gjid8vrb1jigurxquwabt",
    )
    run_id = resolve_swanlab_run_id(
        save_dir=tmp_path,
        project="rknn-super-resolution",
        experiment_name="train-8gpu-20260821-1608",
        resume_checkpoint="dummy.pth",
    )
    assert run_id == "gjid8vrb1jigurxquwabt"
