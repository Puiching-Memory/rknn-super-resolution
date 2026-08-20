"""Unified CLI for training, evaluation, export, and deployment."""

from __future__ import annotations

import sys
from typing import NoReturn

import typer

app = typer.Typer(
    name="rk3588-mobile-sr",
    help="RK3576 Phase-MobileOne 3x super-resolution: train, evaluate, export, deploy.",
    no_args_is_help=True,
)

def _run_module_main(main_fn) -> NoReturn:
    """Invoke a script-style main() and exit with its status code."""
    try:
        main_fn()
    except SystemExit as exc:
        raise typer.Exit(exc.code if exc.code is not None else 0) from exc
    raise typer.Exit(0)


@app.command("train")
def train() -> None:
    """Train FP32 through QAT in one plateau-driven run."""
    from rk3588_mobile_sr.train.unified import main as train_main

    _run_module_main(train_main)


@app.command("eval")
def eval_psnr() -> None:
    """Evaluate PSNR/SSIM on a validation set."""
    from rk3588_mobile_sr.eval.psnr import evaluate, parse_args

    args = parse_args()
    evaluate(args)


@app.command("export-onnx")
def export_onnx() -> None:
    """Export fused MobileOneSR to ONNX."""
    from rk3588_mobile_sr.deploy.onnx import main as export_main

    _run_module_main(export_main)


@app.command("convert-rknn")
def convert_rknn() -> None:
    """Convert ONNX to RKNN (requires rknn-toolkit2)."""
    from rk3588_mobile_sr.deploy.rknn import main as rknn_main

    _run_module_main(rknn_main)


def main() -> None:
    """Entry point for the console script."""
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        from rk3588_mobile_sr.train.unified import main as train_main

        sys.argv = ["train"] + sys.argv[2:]
        train_main()
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        from rk3588_mobile_sr.eval.psnr import evaluate, parse_args

        sys.argv = ["eval_psnr"] + sys.argv[2:]
        args = parse_args()
        evaluate(args)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "export-onnx":
        from rk3588_mobile_sr.deploy.onnx import main as export_main

        sys.argv = ["export_onnx"] + sys.argv[2:]
        export_main()
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "convert-rknn":
        from rk3588_mobile_sr.deploy.rknn import main as rknn_main

        sys.argv = ["rknn_convert"] + sys.argv[2:]
        rknn_main()
        return

    app()


if __name__ == "__main__":
    main()
