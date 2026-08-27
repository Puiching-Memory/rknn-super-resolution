"""Regenerate SR preview panels and the labeled report grid for a training run.

Reconstructs the QAT-stable model from ``{run_dir}/best.pth``, runs it on
distinct validation scenes (one q_index per scene, cycling the configured
q_indices), and rewrites:

- ``{run_dir}/sr_preview/sample_*.png`` — attribution panels from
  ``collect_sr_validation_panels`` (bicubic baseline vs SR vs HR with per-tile
  PSNR/SSIM burned in),
- ``{run_dir}/report_charts/sr_preview_grid.png`` — all panels re-tiled with
  centered Chinese titles and per-group colored borders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from rknn_super_resolution.config import load_config
from rknn_super_resolution.data.decode import TorchCodecFrameDecoder
from rknn_super_resolution.data.mlvc_loader import MLVCBatchProcessor, MLVCValidationLoader
from rknn_super_resolution.data.mlvc_runtime import FrozenMLVCRuntime
from rknn_super_resolution.data.openvid import (
    OpenVidSequenceDataset,
    collate_openvid_batch,
    load_openvid_index,
    select_unique_source_indices,
    split_sequence_indices,
)
from rknn_super_resolution.models.graph_format import PT2E_QAT_FORMAT
from rknn_super_resolution.models.qat_utils import (
    disable_qat_observers,
    prepare_model_for_qat,
)
from rknn_super_resolution.utils.sr_metrics import iter_val_batches
from rknn_super_resolution.utils.swanlab_logging import (
    collect_sr_validation_panels,
    save_sr_panels,
)
from rknn_super_resolution.utils.train_framework import (
    build_model,
    load_training_module_state_dict,
    setup_device,
)

ROOT = Path(__file__).resolve().parents[3]

GRID_TITLES = [
    ["传统上采样: Bicubic", "SR 输出 (Phase-RLFN)", "HR 参考"],
    ["误差: Bicubic vs HR", "误差: SR vs HR", "SR 效果: 绿=改善 红=变差"],
    ["放大: Bicubic | HR", "放大: SR | HR", "放大: Bicubic | SR"],
]
GRID_TILE_W = 640
GRID_TILE_H = 360
GRID_TITLE_H = 44
GRID_CAPTION_H = 40
GRID_BORDER = 10
GRID_PAD = 10
GRID_GAP = 24
GRID_COLS = 2
GROUP_COLORS = [
    "#e11d48",
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
]
GRID_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


@dataclass(frozen=True)
class ReportArgs:
    run_dir: Path
    checkpoint: str
    num_samples: int
    config: str | None
    device: str
    grid_only: bool


def parse_args() -> ReportArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--grid_only",
        action="store_true",
        help="skip model inference; re-tile existing sr_preview panels into the grid",
    )
    ns = parser.parse_args()
    return ReportArgs(
        run_dir=ns.run_dir,
        checkpoint=ns.checkpoint,
        num_samples=ns.num_samples,
        config=ns.config,
        device=ns.device,
        grid_only=ns.grid_only,
    )


def _namespace(config: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        scale=None,
        num_channels=None,
        num_blocks=None,
        phase_factor=None,
        codec_feature_channels=None,
        codec_project_channels=None,
        codec_upsample_factor=None,
        negative_slope=None,
        batch_size=1,
        q_indices=None,
        dataset_description=None,
        video_root=None,
        mlvc_repo=None,
        mlvc_checkpoint=None,
        mlvc_variant=None,
        sequence_frames=None,
        num_workers=None,
        colorspace=None,
        codec_context=None,
        codec_dropout=None,
        device="cuda",
    )


def _load_qat_stable_model(args: argparse.Namespace, checkpoint: Path, device: torch.device):
    """Rebuild the PT2E QAT inference graph (fake quant on, observers off)."""
    cfg = load_config(getattr(args, "config", None))
    model = build_model(args, device)
    lr_h, lr_w = (int(v) for v in cfg.data.lr_size)
    example_inputs = (
        torch.randn(
            1,
            model.core_in_channels,
            lr_h // model.phase_factor,
            lr_w // model.phase_factor,
            device=device,
        ),
    )
    if cfg.data.codec_context:
        example_inputs += (
            torch.randn(
                1,
                model.codec_feature_channels,
                ((lr_h + 15) // 16) * 2,
                ((lr_w + 15) // 16) * 2,
                device=device,
            ),
        )
    model = prepare_model_for_qat(
        model,
        example_inputs=example_inputs,
    )
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or raw.get("graph_format") != PT2E_QAT_FORMAT:
        raise TypeError(f"expected a {PT2E_QAT_FORMAT} checkpoint: {checkpoint}")
    load_training_module_state_dict(model, raw["state_dict"])
    model.eval()
    disable_qat_observers(model)
    return model


def _close(loader) -> None:
    close = getattr(loader, "close", None)
    if close is not None:
        close()


def _build_preview_loader(
    device: torch.device,
    config: str | None,
    num_samples: int,
) -> tuple[MLVCValidationLoader, list[str]]:
    """Build a validation loader over distinct scenes for report previews.

    The training val split interleaves ``q_indices`` per scene, so the first N
    samples repeat the same clips. Here each preview sample comes from its own
    sequence, cycling q_indices across scenes.
    """
    cfg = load_config(config)
    data = cfg.data
    description = (ROOT / data.dataset_description).resolve()
    video_root = (ROOT / data.video_root).resolve()
    sequences = load_openvid_index(description, video_root=video_root)
    manifest = (ROOT / data.split_manifest).resolve() if data.split_manifest else None
    _train, val_indices, _test = split_sequence_indices(
        sequences,
        val_fraction=data.val_fraction,
        test_fraction=data.test_fraction,
        seed=data.split_seed,
        manifest_path=manifest,
    )
    val_indices = select_unique_source_indices(sequences, val_indices)
    dataset = OpenVidSequenceDataset(
        sequences,
        indices=val_indices,
        sequence_frames=data.sequence_frames,
        lr_size=data.lr_size,
        hr_size=data.hr_size,
        training_split=False,
        augment=False,
        q_indices=tuple(data.q_indices),
        max_samples=num_samples,
    )
    n_q = len(data.q_indices)
    n_scenes = min(num_samples, len(dataset.sequences))
    picks = [i * n_q + (i % n_q) for i in range(n_scenes)]
    cpu_loader = DataLoader(
        Subset(dataset, picks),
        batch_size=1,
        shuffle=False,
        num_workers=data.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_openvid_batch,
    )
    runtime = FrozenMLVCRuntime(
        repo=(ROOT / data.mlvc_repo).resolve(),
        checkpoint=(ROOT / data.mlvc_checkpoint).resolve(),
        variant=data.mlvc_variant,
        device=device,
        amp=data.mlvc_amp,
    )
    decoder = TorchCodecFrameDecoder(device, lr_size=data.lr_size, hr_size=data.hr_size)
    processor = MLVCBatchProcessor(
        runtime,
        decoder=decoder,
        device=device,
        q_indices=tuple(data.q_indices),
        colorspace=data.colorspace,
        scale=cfg.model.scale,
        codec_context=data.codec_context,
        codec_dropout=data.codec_dropout,
    )
    captions = [
        f"{Path(dataset.sequences[i].path).stem} · q={data.q_indices[i % n_q]}"
        for i in range(n_scenes)
    ]
    return MLVCValidationLoader(cpu_loader, processor), captions


def regenerate_panels(report: ReportArgs) -> tuple[list[np.ndarray], list[str]]:
    args = _namespace(report.config)
    device = setup_device(args)
    colorspace = load_config(report.config).data.colorspace
    model = _load_qat_stable_model(args, report.run_dir / report.checkpoint, device)
    val_loader, captions = _build_preview_loader(device, report.config, report.num_samples)
    try:
        panels = collect_sr_validation_panels(
            model,
            iter_val_batches(val_loader),
            device,
            num_samples=report.num_samples,
            colorspace=colorspace,
        )
    finally:
        _close(val_loader)
    if not panels:
        raise RuntimeError("validation loader produced no SR panels")
    for stale in (report.run_dir / "sr_preview").glob("sample_*.png"):
        stale.unlink()
    save_sr_panels(panels, report.run_dir, subdir="sr_preview")
    return panels, captions


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in GRID_FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("no CJK font found; install fonts-noto-cjk or fonts-wqy-zenhei")


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str = "white",
) -> None:
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (x0 + (x1 - x0 - (right - left)) / 2 - left, y0 + (y1 - y0 - (bottom - top)) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def assemble_grid(
    panels: list[np.ndarray],
    out_path: Path,
    *,
    tile_hw: tuple[int, int],
    captions: list[str] | None = None,
) -> Path:
    """Re-tile panels into the labeled report grid (2 groups per row)."""
    tile_h, tile_w = tile_hw
    rows = len(GRID_TITLES)
    cols = len(GRID_TITLES[0])
    for panel in panels:
        ph, pw = panel.shape[:2]
        if ph != rows * tile_h or pw != cols * tile_w:
            raise ValueError(
                f"panel {pw}x{ph} does not match expected {cols}x{rows} tiles of {tile_w}x{tile_h}"
            )

    title_font = _load_font(26)
    caption_font = _load_font(28)
    group_w = cols * GRID_TILE_W + 2 * GRID_BORDER + 2 * GRID_PAD
    group_h = GRID_CAPTION_H + rows * (GRID_TITLE_H + GRID_TILE_H) + 2 * GRID_BORDER + 2 * GRID_PAD
    grid_rows = (len(panels) + GRID_COLS - 1) // GRID_COLS
    canvas_w = GRID_COLS * group_w + (GRID_COLS - 1) * GRID_GAP
    canvas_h = grid_rows * group_h + (grid_rows - 1) * GRID_GAP
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)

    for idx, panel in enumerate(panels):
        image = Image.fromarray(panel)
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]
        gx = (idx % GRID_COLS) * (group_w + GRID_GAP)
        gy = (idx // GRID_COLS) * (group_h + GRID_GAP)
        draw.rectangle(
            (gx, gy, gx + group_w - 1, gy + group_h - 1), outline=color, width=GRID_BORDER
        )
        caption = f"样本 {idx + 1}"
        if captions is not None:
            caption = f"{caption} · {captions[idx]}"
        _centered_text(
            draw,
            (
                gx + GRID_BORDER,
                gy + GRID_BORDER,
                gx + group_w - GRID_BORDER,
                gy + GRID_BORDER + GRID_CAPTION_H,
            ),
            caption,
            caption_font,
            fill=color,
        )
        for r in range(rows):
            for c in range(cols):
                tile = image.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h))
                tile = tile.resize((GRID_TILE_W, GRID_TILE_H), Image.LANCZOS)
                tx = gx + GRID_BORDER + GRID_PAD + c * GRID_TILE_W
                ty = gy + GRID_BORDER + GRID_PAD + GRID_CAPTION_H + r * (GRID_TITLE_H + GRID_TILE_H)
                draw.rectangle((tx, ty, tx + GRID_TILE_W, ty + GRID_TITLE_H), fill="#111827")
                _centered_text(
                    draw,
                    (tx, ty, tx + GRID_TILE_W, ty + GRID_TITLE_H),
                    GRID_TITLES[r][c],
                    title_font,
                )
                canvas.paste(tile, (tx, ty + GRID_TITLE_H))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main() -> None:
    report = parse_args()
    captions = None
    if report.grid_only:
        panels = [
            np.asarray(Image.open(p))
            for p in sorted((report.run_dir / "sr_preview").glob("sample_*.png"))
        ][: report.num_samples]
        if not panels:
            raise RuntimeError(f"no panels under {report.run_dir}/sr_preview")
    else:
        panels, captions = regenerate_panels(report)
    tile_hw = tuple(int(v) for v in load_config(report.config).data.hr_size)
    out = assemble_grid(
        panels,
        report.run_dir / "report_charts" / "sr_preview_grid.png",
        tile_hw=tile_hw,
        captions=captions,
    )
    print(f"{len(panels)} panels -> {out}")


if __name__ == "__main__":
    main()
