"""Unified CLI for training, evaluation, export, and deployment."""

from __future__ import annotations

import sys
from typing import NoReturn

import typer

from rk3588_mobile_sr.deploy.rknn import main as rknn_main
from rk3588_mobile_sr.eval.psnr import evaluate
from rk3588_mobile_sr.eval.psnr import parse_args as eval_parse_args
from rk3588_mobile_sr.export.onnx import main as export_main
from rk3588_mobile_sr.train.stage1 import main as train_stage1_main
from rk3588_mobile_sr.train.stage2 import main as train_stage2_main
from rk3588_mobile_sr.train.stage3_qat import main as train_stage3_main

app = typer.Typer(
    name="rk3588-mobile-sr",
    help="RK3588 MobileOne 3x super-resolution: train, evaluate, export, deploy.",
    no_args_is_help=True,
)

train_app = typer.Typer(help="Training stages (use with torchrun for DDP).")
app.add_typer(train_app, name="train")


def _run_module_main(main_fn) -> NoReturn:
    """Invoke a script-style main() and exit with its status code."""
    try:
        main_fn()
    except SystemExit as exc:
        raise typer.Exit(exc.code if exc.code is not None else 0) from exc
    raise typer.Exit(0)


@train_app.command("stage1")
def train_stage1() -> None:
    """Stage 1: FP32 L1 baseline training."""
    _run_module_main(train_stage1_main)


@train_app.command("stage2")
def train_stage2() -> None:
    """Stage 2: knowledge distillation + DCT perceptual finetuning."""
    _run_module_main(train_stage2_main)


@train_app.command("stage3-qat")
def train_stage3_qat() -> None:
    """Stage 3: quantization-aware training."""
    _run_module_main(train_stage3_main)


@app.command("eval")
def eval_psnr() -> None:
    """Evaluate PSNR/SSIM on a validation set."""
    args = eval_parse_args()
    evaluate(args)


@app.command("export-onnx")
def export_onnx() -> None:
    """Export fused MobileOneSR to ONNX."""
    _run_module_main(export_main)


@app.command("convert-rknn")
def convert_rknn() -> None:
    """Convert ONNX to RKNN (requires rknn-toolkit2)."""
    _run_module_main(rknn_main)


def main() -> None:
    """Entry point for the console script."""
    # When invoked as `rk3588-mobile-sr train stage1 -- ...`, forward remaining
    # argv to the underlying argparse-based scripts unchanged.
    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "train"
        and sys.argv[2]
        in {
            "stage1",
            "stage2",
            "stage3-qat",
        }
    ):
        stage = sys.argv[2]
        sys.argv = [f"train_{stage.replace('-', '_')}"] + sys.argv[3:]
        if stage == "stage1":
            train_stage1_main()
        elif stage == "stage2":
            train_stage2_main()
        else:
            train_stage3_main()
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        sys.argv = ["eval_psnr"] + sys.argv[2:]
        args = eval_parse_args()
        evaluate(args)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "export-onnx":
        sys.argv = ["export_onnx"] + sys.argv[2:]
        export_main()
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "convert-rknn":
        sys.argv = ["rknn_convert"] + sys.argv[2:]
        rknn_main()
        return

    app()


if __name__ == "__main__":
    main()
