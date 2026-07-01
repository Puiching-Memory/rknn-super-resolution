"""Convert ONNX model to RKNN INT8 for RK3588."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.rknn_eval import add_eval_args, run_post_build_eval
from rk3588_mobile_sr.deploy.rknn_env import needs_rknn_reexec, reexec_in_rknn_python, resolve_rknn_python


def import_rknn():
    try:
        from rknn.api import RKNN

        return RKNN
    except ImportError as e:
        python = resolve_rknn_python(None)
        raise ImportError(
            f"rknn-toolkit2 is required in the RKNN Python env ({python}). "
            f"Re-run via `rk3588-mobile-sr convert-rknn` to auto re-exec, "
            f"or install the wheel into that interpreter."
        ) from e


def parse_args():
    deploy = load_config().deploy
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output", type=str, default=deploy.rknn_output)
    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--calib_dir", type=str, default=deploy.calib_dir)
    parser.add_argument("--input_size", type=str, default="3,360,640")
    parser.add_argument(
        "--quantize",
        type=str,
        default=deploy.rknn_quantize,
        choices=["normal", "mmse", "kl_divergence", "dynamic"],
        help="PTQ scale search algorithm passed to rknn.config(quantized_algorithm=...).",
    )
    parser.add_argument(
        "--quantized_method",
        type=str,
        default=deploy.rknn_quantized_method,
        choices=["channel", "layer"],
        help="Weight quantization granularity (per-channel is usually more accurate).",
    )
    parser.add_argument("--do_quantization", action="store_true", default=True)
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument(
        "--hybrid",
        type=str,
        default=None,
        choices=["proposal", "auto"],
        help=(
            "INT8+FP16 hybrid quantization: "
            "'proposal' runs hybrid_quantization_step1/2 with RKNN layer suggestions; "
            "'auto' passes auto_hybrid=True to build()."
        ),
    )
    parser.add_argument(
        "--hybrid_cfg",
        type=str,
        default=None,
        help="Use an existing *.quantization.cfg from a prior hybrid step1 (skips step1).",
    )
    parser.add_argument(
        "--hybrid_workdir",
        type=str,
        default=None,
        help="Directory for hybrid step1 artifacts (*.model, *.data, *.quantization.cfg).",
    )
    parser.add_argument(
        "--hybrid_proposal_images",
        type=int,
        default=20,
        help="Calibration images for hybrid proposal analysis (step1 only).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="RKNN-dedicated Python interpreter (default: deploy.rknn_python or RK3588_RKNN_PYTHON).",
    )
    add_eval_args(parser)
    return parser.parse_args()


def _stem_from_onnx(onnx_path: Path) -> str:
    return onnx_path.stem


def _hybrid_artifacts(workdir: Path, onnx_path: Path) -> tuple[Path, Path, Path]:
    stem = _stem_from_onnx(onnx_path)
    return (
        workdir / f"{stem}.model",
        workdir / f"{stem}.data",
        workdir / f"{stem}.quantization.cfg",
    )


def _config_kwargs(args: argparse.Namespace, *, do_quantization: bool) -> dict:
    kwargs = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[1, 1, 1]],
        "target_platform": args.target,
        "quantized_algorithm": args.quantize,
    }
    if do_quantization:
        kwargs["quantized_method"] = args.quantized_method
    return kwargs


def _run_hybrid_build(
    rknn,
    args: argparse.Namespace,
    *,
    do_quantization: bool,
    onnx_path: Path,
) -> str:
    """Return quant_mode label for eval."""
    workdir = Path(args.hybrid_workdir) if args.hybrid_workdir else onnx_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    model_path, data_path, cfg_path = _hybrid_artifacts(workdir, onnx_path)

    if args.hybrid_cfg:
        cfg_path = Path(args.hybrid_cfg)
        if not cfg_path.is_file():
            print(f"Hybrid cfg not found: {cfg_path}")
            sys.exit(1)
        stem = cfg_path.name.removesuffix(".quantization.cfg")
        model_path = workdir / f"{stem}.model"
        data_path = workdir / f"{stem}.data"
    elif args.hybrid == "proposal":
        print("--> Hybrid step 1 (proposal)")
        prev_cwd = os.getcwd()
        try:
            os.chdir(workdir)
            ret = rknn.hybrid_quantization_step1(
                dataset=args.calib_dir,
                proposal=True,
                proposal_dataset_size=args.hybrid_proposal_images,
            )
        finally:
            os.chdir(prev_cwd)
        if ret != 0:
            print("hybrid_quantization_step1 failed!")
            sys.exit(ret)
        if not cfg_path.is_file():
            print(f"Hybrid cfg not generated: {cfg_path}")
            sys.exit(1)
        print(f"--> Hybrid cfg: {cfg_path}")

    if not model_path.is_file() or not data_path.is_file():
        print(f"Hybrid artifacts missing: {model_path}, {data_path}")
        sys.exit(1)

    print("--> Hybrid step 2")
    ret = rknn.hybrid_quantization_step2(
        model_input=str(model_path),
        data_input=str(data_path),
        model_quantization_cfg=str(cfg_path),
    )
    if ret != 0:
        print("hybrid_quantization_step2 failed!")
        sys.exit(ret)
    return "INT8+FP16 hybrid"


def _parse_input_size(spec: str) -> list[int]:
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"input_size must be C,H,W, got {spec!r}")
    return parts


def main():
    args = parse_args()
    rknn_python = resolve_rknn_python(args.python)
    if needs_rknn_reexec(rknn_python):
        reexec_in_rknn_python(rknn_python, sys.argv[1:])

    RKNN = import_rknn()
    do_quantization = args.do_quantization and not args.no_quantization
    _parse_input_size(args.input_size)
    onnx_path = Path(args.onnx)
    use_hybrid = args.hybrid is not None or args.hybrid_cfg is not None
    if use_hybrid and not do_quantization:
        print("Hybrid quantization requires INT8 PTQ (--no_quantization conflicts).")
        sys.exit(1)
    quant_mode = "INT8+FP16 hybrid" if use_hybrid else ("INT8" if do_quantization else "FP16")

    rknn = RKNN(verbose=True)
    try:
        print("--> Config model")
        ret = rknn.config(**_config_kwargs(args, do_quantization=do_quantization))
        if ret != 0:
            print("Config model failed!")
            sys.exit(ret)

        if use_hybrid and args.hybrid != "auto" and not args.hybrid_cfg:
            print("--> Loading ONNX (hybrid step 1)")
            ret = rknn.load_onnx(model=args.onnx)
            if ret != 0:
                print("Load ONNX failed!")
                sys.exit(ret)
            quant_mode = _run_hybrid_build(rknn, args, do_quantization=do_quantization, onnx_path=onnx_path)
        elif use_hybrid and args.hybrid_cfg:
            quant_mode = _run_hybrid_build(rknn, args, do_quantization=do_quantization, onnx_path=onnx_path)
        else:
            print("--> Loading ONNX")
            ret = rknn.load_onnx(model=args.onnx)
            if ret != 0:
                print("Load ONNX failed!")
                sys.exit(ret)

            print("--> Building RKNN")
            ret = rknn.build(
                do_quantization=do_quantization,
                dataset=args.calib_dir if do_quantization else None,
                auto_hybrid=args.hybrid == "auto",
            )
            if ret != 0:
                print("Build model failed!")
                sys.exit(ret)

        if args.eval:
            print("--> Evaluating accuracy (RKNN simulator)")
            ret = rknn.init_runtime()
            if ret != 0:
                print("Init runtime for eval failed!")
                sys.exit(ret)
            run_post_build_eval(args, rknn, quant_mode=quant_mode)

        print("--> Export RKNN")
        ret = rknn.export_rknn(args.output)
        if ret != 0:
            print("Export RKNN failed!")
            sys.exit(ret)
    finally:
        rknn.release()

    print(f"RKNN exported to {args.output}")


if __name__ == "__main__":
    main()
