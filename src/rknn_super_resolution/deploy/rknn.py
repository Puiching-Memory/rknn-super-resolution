"""Convert ONNX model to RKNN INT8 for RK3588."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rknn_super_resolution.config import load_config
from rknn_super_resolution.deploy.rknn_env import (
    needs_rknn_reexec,
    reexec_in_rknn_python,
    resolve_rknn_python,
)
from rknn_super_resolution.deploy.rknn_eval import add_eval_args, run_post_build_eval


def import_rknn():
    try:
        from rknn.api import RKNN

        return RKNN
    except ImportError as e:
        python = resolve_rknn_python(None)
        raise ImportError(
            f"rknn-toolkit2 is required in the RKNN Python env ({python}). "
            f"Re-run via `rknn-super-resolution convert-rknn` to auto re-exec, "
            f"or install the wheel into that interpreter."
        ) from e


def parse_args():
    cfg = load_config()
    deploy = cfg.deploy
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output", type=str, default=deploy.rknn_output)
    parser.add_argument("--target", type=str, default=deploy.target)
    parser.add_argument("--calib_dir", type=str, default=deploy.calib_dir)
    parser.add_argument("--input_size", type=str, default=_default_input_size(cfg))
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
    parser.add_argument(
        "--encrypt",
        action=argparse.BooleanOptionalAction,
        default=deploy.rknn_encrypt,
        help=(
            "After export, encrypt the RKNN via export_encrypted_rknn_model "
            "(decrypted automatically by the NPU driver at load time)."
        ),
    )
    parser.add_argument(
        "--crypt-level",
        type=int,
        default=deploy.rknn_crypt_level,
        choices=[1, 2, 3],
        help="Encryption level 1–3: higher is more secure but slower to decrypt on-device.",
    )
    parser.add_argument(
        "--encrypted-output",
        type=str,
        default=None,
        help="Encrypted model path; default: {output_stem}.crypt.rknn (RKNN toolkit convention).",
    )
    add_eval_args(parser, cfg.model)
    return parser.parse_args()


def _warn_nhwc_onnx_output(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(onnx_path))
    if not model.graph.output:
        return
    dims = [
        d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim if d.dim_value > 0
    ]
    if len(dims) == 4 and dims[-1] in (1, 3) and dims[1] not in (1, 3):
        print(
            "WARNING: ONNX output looks NHWC "
            f"{dims}. RKNN will compile an extra Transpose (memory/latency cost). "
            "Prefer default NCHW ONNX unless the consumer requires NHWC from RKNN."
        )


def _stem_from_onnx(onnx_path: Path) -> str:
    return onnx_path.stem


def _hybrid_artifacts(workdir: Path, onnx_path: Path) -> tuple[Path, Path, Path]:
    stem = _stem_from_onnx(onnx_path)
    return (
        workdir / f"{stem}.model",
        workdir / f"{stem}.data",
        workdir / f"{stem}.quantization.cfg",
    )


def _resolve_calib_dir(path: str) -> str:
    """Resolve calibration list to an absolute path (hybrid step1 may chdir)."""
    calib = Path(path).expanduser()
    if not calib.is_absolute():
        calib = (Path.cwd() / calib).resolve()
    return str(calib)


def _config_kwargs(args: argparse.Namespace, *, do_quantization: bool) -> dict:
    input_sizes = _parse_input_size(args.input_size)
    kwargs = {
        "mean_values": [[0] * channels for channels, _, _ in input_sizes],
        "std_values": [[1] * channels for channels, _, _ in input_sizes],
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
        calib_dir = _resolve_calib_dir(args.calib_dir)
        prev_cwd = os.getcwd()
        try:
            os.chdir(workdir)
            ret = rknn.hybrid_quantization_step1(
                dataset=calib_dir,
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


def _default_encrypted_output(output: Path) -> Path:
    """Match rknn-toolkit2 default: ``foo.rknn`` -> ``foo.crypt.rknn``."""
    return output.with_suffix(".crypt.rknn")


def _resolve_encrypted_output(output: Path, explicit: str | None) -> Path:
    if explicit:
        enc = Path(explicit).expanduser()
        if not enc.is_absolute():
            enc = (Path.cwd() / enc).resolve()
        return enc
    return _default_encrypted_output(output)


def _export_rknn(
    rknn,
    output: str,
    *,
    encrypt: bool,
    crypt_level: int,
    encrypted_output: str | None,
) -> None:
    out_path = Path(output)
    print("--> Export RKNN")
    ret = rknn.export_rknn(output)
    if ret != 0:
        print("Export RKNN failed!")
        sys.exit(ret)
    print(f"RKNN exported to {output}")

    if not encrypt:
        return

    enc_path = _resolve_encrypted_output(out_path, encrypted_output)
    print(f"--> Encrypt RKNN (crypt_level={crypt_level})")
    ret = rknn.export_encrypted_rknn_model(
        str(out_path),
        output_model=str(enc_path),
        crypt_level=crypt_level,
    )
    if ret != 0:
        print("Encrypt RKNN failed!")
        sys.exit(ret)
    print(f"Encrypted RKNN exported to {enc_path}")


def _parse_input_size(spec: str) -> list[list[int]]:
    sizes = [[int(x.strip()) for x in item.split(",")] for item in spec.split(";")]
    if any(len(parts) != 3 for parts in sizes):
        raise ValueError(f"input_size must be C,H,W[;C,H,W], got {spec!r}")
    return sizes


def _default_input_size(cfg) -> str:
    model = cfg.model
    deploy = cfg.deploy
    factor = model.phase_factor
    if deploy.input_h % factor or deploy.input_w % factor:
        raise ValueError("deploy input size must be divisible by model.phase_factor")
    phases = (
        f"{model.in_channels * factor * factor},"
        f"{deploy.input_h // factor},{deploy.input_w // factor}"
    )
    if not deploy.codec_context:
        return phases
    codec_h = ((deploy.input_h + 15) // 16) * 2
    codec_w = ((deploy.input_w + 15) // 16) * 2
    return f"{phases};{model.codec_feature_channels},{codec_h},{codec_w}"


def main():
    args = parse_args()
    rknn_python = resolve_rknn_python(args.python)
    if needs_rknn_reexec(rknn_python):
        reexec_in_rknn_python(rknn_python, sys.argv[1:])

    rknn_class = import_rknn()
    do_quantization = args.do_quantization and not args.no_quantization
    args.calib_dir = _resolve_calib_dir(args.calib_dir)
    _parse_input_size(args.input_size)
    onnx_path = Path(args.onnx)
    use_hybrid = args.hybrid is not None or args.hybrid_cfg is not None
    if use_hybrid and not do_quantization:
        print("Hybrid quantization requires INT8 PTQ (--no_quantization conflicts).")
        sys.exit(1)
    quant_mode = "INT8+FP16 hybrid" if use_hybrid else ("INT8" if do_quantization else "FP16")

    rknn = rknn_class(verbose=True)
    try:
        print("--> Config model")
        print("--> Input format: MLVC BT.709 YCbCr444")
        _warn_nhwc_onnx_output(onnx_path)
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
            quant_mode = _run_hybrid_build(
                rknn, args, do_quantization=do_quantization, onnx_path=onnx_path
            )
        elif use_hybrid and args.hybrid_cfg:
            quant_mode = _run_hybrid_build(
                rknn, args, do_quantization=do_quantization, onnx_path=onnx_path
            )
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

        _export_rknn(
            rknn,
            args.output,
            encrypt=args.encrypt,
            crypt_level=args.crypt_level,
            encrypted_output=args.encrypted_output,
        )
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
