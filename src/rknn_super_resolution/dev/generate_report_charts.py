"""Generate report chart PNGs under docs/report_assets/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
METRICS_JSON = ROOT / "stage1_metrics.json"
OUT_DIR = ROOT / "docs" / "report_assets"

FONT_PATHS = [
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WenQuanYi Zen Hei"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
]

FONT_CANDIDATES = [name for _, name in FONT_PATHS] + [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
]

DPI = 200


def configure_cjk_font() -> str:
    cache_dir = Path.home() / ".cache" / "matplotlib"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("fontlist-*.json"):
            cache_file.unlink(missing_ok=True)

    for path, preferred_name in FONT_PATHS:
        font_path = Path(path)
        if not font_path.exists():
            continue
        font_manager.fontManager.addfont(font_path)
        name = font_manager.FontProperties(fname=path).get_name()
        plt.rcParams.update(
            {
                "font.family": name,
                "font.sans-serif": [name, preferred_name, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "text.antialiased": True,
                "figure.dpi": DPI,
                "savefig.dpi": DPI,
            }
        )
        return name

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError("No CJK font found. Install fonts-noto-cjk or fonts-wqy-zenhei.")


def load_series() -> dict[str, list[dict]]:
    with METRICS_JSON.open(encoding="utf-8") as f:
        data = json.load(f)["data"]["list"]
    return {item["key"]: item["metrics"] for item in data}


def save_psnr(series: dict[str, list[dict]]) -> None:
    psnr = series["val/psnr"]
    best = series["val/best_psnr"]
    steps = [m["index"] for m in psnr]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        steps,
        [m["data"] for m in psnr],
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=5,
        label="验证 PSNR",
    )
    ax.plot(
        steps,
        [m["data"] for m in best],
        "--",
        color="#16a34a",
        linewidth=2,
        label="累计最优 PSNR",
    )
    ax.axhline(29.8, color="#d97706", linestyle=":", linewidth=2, label="目标 29.8 dB")
    ax.set(
        xlabel="训练步数 (step)", ylabel="PSNR (dB)", title="Stage 1 验证 PSNR 曲线", ylim=(27, 32)
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage1_psnr.png", dpi=DPI)
    plt.close(fig)


def save_loss(series: dict[str, list[dict]]) -> None:
    loss = series["train/loss"]
    steps = [m["index"] for m in loss]
    values = [m["data"] for m in loss]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, values, color="#7c3aed", linewidth=2)
    ax.fill_between(steps, values, alpha=0.15, color="#7c3aed")
    ax.set(xlabel="训练步数 (step)", ylabel="L1 损失", title="Stage 1 训练 Loss 收敛")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage1_loss.png", dpi=DPI)
    plt.close(fig)


def save_ssim(series: dict[str, list[dict]]) -> None:
    ssim = series["val/ssim"]
    steps = [m["index"] for m in ssim]
    values = [m["data"] for m in ssim]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, values, "o-", color="#059669", linewidth=2, markersize=5)
    ax.set(xlabel="训练步数 (step)", ylabel="SSIM", title="Stage 1 验证 SSIM", ylim=(0.80, 0.90))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage1_ssim.png", dpi=DPI)
    plt.close(fig)


def save_psnr_dist() -> None:
    labels = ["最小值", "P10", "中位数", "均值", "P90"]
    values = [19.26, 23.79, 30.80, 30.77, 37.90]
    colors = ["#ef4444", "#f97316", "#2563eb", "#2563eb", "#16a34a"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.1f}",
            ha="center",
            fontsize=10,
        )
    ax.set(ylabel="PSNR (dB)", title="Stage 1 最终验证集 PSNR 分布 (step 17k)", ylim=(0, 42))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage1_psnr_dist.png", dpi=DPI)
    plt.close(fig)


def save_pipeline() -> None:
    stages = [
        ("Stage 1\nFP32 基线", "#16a34a"),
        ("Stage 2\n蒸馏微调", "#2563eb"),
        ("Stage 3\nQAT 量化", "#9ca3af"),
        ("ONNX\n导出", "#9ca3af"),
        ("RKNN\nRK3588", "#9ca3af"),
    ]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    for i, (name, color) in enumerate(stages):
        ax.barh(0, 1, left=i, height=0.62, color=color, edgecolor="white", linewidth=2)
        ax.text(
            i + 0.5,
            0,
            name,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
        )
    ax.set_xlim(0, len(stages))
    ax.set_ylim(-0.5, 0.85)
    ax.axis("off")
    ax.set_title("训练与部署流水线进度", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pipeline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = configure_cjk_font()
    series = load_series()
    save_psnr(series)
    save_loss(series)
    save_ssim(series)
    save_psnr_dist()
    save_pipeline()
    print(f"Charts saved to {OUT_DIR} (font: {font})")


if __name__ == "__main__":
    main()
