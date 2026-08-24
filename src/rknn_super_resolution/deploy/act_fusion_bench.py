"""Benchmark RKNN Conv+activation fusion and INT8 output fidelity.

Exports minimal Conv+Act blocks to ONNX (main training env), then builds INT8
RKNN models in the dedicated rknn-toolkit2 Python env and reports:
  - ONNX operator breakdown
  - Compiled NPU/CPU operators (fusion detection)
  - FP32 (ONNX Runtime) vs INT8 (RKNN simulator) output match PSNR
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rknn_super_resolution.config import load_config
from rknn_super_resolution.deploy.rknn_env import resolve_rknn_python
from rknn_super_resolution.deploy.rknn_eval import _rknn_output_to_hwc, psnr_numpy
from rknn_super_resolution.deploy.targets import TESTED_RKNN_TARGETS, normalize_rknn_target

# ONNX export runs in the project env; RKNN build runs in .venv-rknn.
DEFAULT_ACTIVATIONS = (
    "relu",
    "leaky_relu",
    "prelu",
    "silu",
    "gelu",
    "mish",
    "hardswish",
    "elu",
    "relu6",
    "hardtanh",
    "tanh",
    "sigmoid",
)

_RKNN_OP_RE = re.compile(
    r"RKNN:.*?\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\(1,",
)


@dataclass
class ActBenchRow:
    name: str
    onnx_ops: list[str]
    build_ok: bool
    error: str
    npu_ops: list[str]
    cpu_ops: list[str]
    fused_npu_op: str | None
    conv_act_fused: bool
    fuse_notes: list[str]
    match_psnr: float | None
    match_psnr_min: float | None
    num_images: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_activation(name: str):
    import torch.nn as nn

    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
        "prelu": lambda: nn.PReLU(32),
        "silu": lambda: nn.SiLU(inplace=True),
        "gelu": lambda: nn.GELU(),
        "mish": lambda: nn.Mish(inplace=True),
        "hardswish": lambda: nn.Hardswish(inplace=True),
        "elu": lambda: nn.ELU(inplace=True),
        "relu6": lambda: nn.ReLU6(inplace=True),
        "hardtanh": lambda: nn.Hardtanh(0.0, 255.0, inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    if name not in factories:
        raise KeyError(f"Unknown activation {name!r}; supported: {sorted(factories)}")
    return factories[name]()


def _conv_act_model(name: str, channels: int = 32):
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=True)
            self.act = _make_activation(name)

        def forward(self, x):
            return self.act(self.conv(x))

    return _Block()


def export_onnx_models(
    workdir: Path,
    activations: tuple[str, ...],
    *,
    input_h: int,
    input_w: int,
    seed: int,
) -> dict[str, Path]:
    import torch

    workdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dummy = torch.randn(1, 3, input_h, input_w)
    paths: dict[str, Path] = {}
    for name in activations:
        model = _conv_act_model(name).eval()
        out = workdir / f"conv_{name}.onnx"
        torch.onnx.export(
            model,
            (dummy,),
            str(out),
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )
        paths[name] = out
    return paths


def _onnx_ops(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path))
    return [node.op_type for node in model.graph.node]


_IGNORE_OPS = frozenset({"InputOperator", "OutputOperator"})


def _extract_last_op_table(log: str) -> str:
    marker = "Network Layer Information Table"
    start = log.rfind(marker)
    if start < 0:
        return log
    end = log.find("<<<<<< end: rknn::RKNNModelRegCmdbuildPass", start)
    if end < 0:
        return log[start:]
    return log[start:end]


def _parse_rknn_ops(log: str) -> list[tuple[int, str, str, str]]:
    table = _extract_last_op_table(log)
    rows: list[tuple[int, str, str, str]] = []
    for line in table.splitlines():
        match = _RKNN_OP_RE.search(line)
        if not match:
            continue
        op_id, op_type, dtype, target = match.groups()
        if op_type in {"OpType", "ID"}:
            continue
        rows.append((int(op_id), op_type, dtype, target))
    return rows


def _parse_fuse_notes(log: str) -> list[str]:
    notes: list[str] = []
    capture = False
    for line in log.splitlines():
        if "fuse_ops results:" in line:
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if stripped.startswith("D fuse_ops done."):
                break
            if stripped.startswith("D ") and ":" in stripped:
                notes.append(stripped.removeprefix("D ").strip())
    return notes


def _load_calib_rgb(
    calib_list: Path,
    *,
    max_images: int,
    input_h: int,
    input_w: int,
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for calibration image loading") from exc

    images: list[np.ndarray] = []
    for line in calib_list.read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_file():
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_AREA)
        images.append(rgb.astype(np.uint8))
        if len(images) >= max_images:
            break
    if not images:
        raise FileNotFoundError(f"No readable calibration images in {calib_list}")
    return images


def _onnx_fp32_predictor(onnx_path: Path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def predict(rgb: np.ndarray) -> np.ndarray:
        nchw = np.transpose(rgb.astype(np.float32), (2, 0, 1))[None, ...]
        out = sess.run(None, {"input": nchw})[0]
        if out.ndim == 4 and out.shape[1] in (1, 3, 32):
            return np.transpose(out[0], (1, 2, 0))
        return _rknn_output_to_hwc(out)

    return predict


def _capture_build_log(log_path: Path, build_fn: Callable[[], None]) -> str:
    """Capture native RKNN toolkit stdout/stderr (bypasses Python redirection)."""
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        build_fn()
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(log_fd)
        os.close(stdout_fd)
        os.close(stderr_fd)
    return log_path.read_text(encoding="utf-8", errors="replace")


def _benchmark_one_rknn(
    name: str,
    onnx_path: Path,
    *,
    calib_list: Path,
    max_images: int,
    input_h: int,
    input_w: int,
    target: str,
    quantize: str,
    quantized_method: str,
    log_dir: Path | None = None,
) -> ActBenchRow:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    build_ok = False
    error = ""
    npu_ops: list[str] = []
    cpu_ops: list[str] = []
    fuse_notes: list[str] = []
    match_psnr = None
    match_psnr_min = None
    num_images = 0

    try:
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            target_platform=target,
            quantized_algorithm=quantize,
            quantized_method=quantized_method,
        )
        if rknn.load_onnx(str(onnx_path)) != 0:
            error = "load_onnx failed"
        else:
            build_log_path = (log_dir or Path("/tmp")) / f"{name}_build.log"
            build_log_path.parent.mkdir(parents=True, exist_ok=True)

            def _build() -> None:
                nonlocal error, build_ok
                if rknn.build(do_quantization=True, dataset=str(calib_list)) != 0:
                    error = "build failed"
                elif rknn.init_runtime() != 0:
                    error = "init_runtime failed"
                else:
                    build_ok = True

            log = _capture_build_log(build_log_path, _build)
            fuse_notes = _parse_fuse_notes(log)
        for _op_id, op_type, _dtype, target_dev in _parse_rknn_ops(log):
            if op_type in _IGNORE_OPS:
                continue
            if target_dev == "NPU":
                npu_ops.append(op_type)
            elif target_dev == "CPU":
                cpu_ops.append(op_type)

        if build_ok:
            images = _load_calib_rgb(
                calib_list,
                max_images=max_images,
                input_h=input_h,
                input_w=input_w,
            )
            fp32_predict = _onnx_fp32_predictor(onnx_path)
            match_psnrs: list[float] = []
            for rgb in images:
                fp32_out = fp32_predict(rgb)
                batch = np.expand_dims(rgb.astype(np.uint8), axis=0)
                rknn_out = _rknn_output_to_hwc(
                    rknn.inference(inputs=[batch], data_format="nhwc")[0]
                )
                h = min(fp32_out.shape[0], rknn_out.shape[0])
                w = min(fp32_out.shape[1], rknn_out.shape[1])
                c = min(fp32_out.shape[2], rknn_out.shape[2])
                match_psnrs.append(psnr_numpy(rknn_out[:h, :w, :c], fp32_out[:h, :w, :c]))
            num_images = len(match_psnrs)
            arr = np.asarray(match_psnrs, dtype=np.float64)
            match_psnr = float(arr.mean())
            match_psnr_min = float(arr.min())
    except Exception as exc:  # pragma: no cover - surfaced in CLI table
        error = str(exc)
    finally:
        rknn.release()

    fused_candidates = [
        op for op in npu_ops if op.startswith("Conv") and op not in {"Conv", "ConvTranspose"}
    ]
    fused_npu_op = fused_candidates[0] if fused_candidates else None
    activation_cpu_ops = {
        "Relu",
        "LeakyRelu",
        "PRelu",
        "Mish",
        "Sigmoid",
        "Tanh",
        "Gelu",
        "Elu",
        "HardSwish",
        "Clip",
        "Swish",
    }
    conv_act_fused = fused_npu_op is not None and not any(
        op in cpu_ops for op in activation_cpu_ops
    )

    return ActBenchRow(
        name=name,
        onnx_ops=_onnx_ops(onnx_path),
        build_ok=build_ok,
        error=error,
        npu_ops=npu_ops,
        cpu_ops=cpu_ops,
        fused_npu_op=fused_npu_op,
        conv_act_fused=conv_act_fused,
        fuse_notes=fuse_notes,
        match_psnr=match_psnr,
        match_psnr_min=match_psnr_min,
        num_images=num_images,
    )


def format_table(rows: list[ActBenchRow]) -> str:
    headers = (
        "Activation",
        "Build",
        "ONNX ops",
        "Fused NPU op",
        "CPU ops",
        "FP32↔INT8 PSNR",
        "Notes",
    )
    table_rows: list[list[str]] = []
    for row in rows:
        if row.build_ok:
            build = "OK"
            psnr = (
                f"{row.match_psnr:.2f} dB (min {row.match_psnr_min:.2f}, n={row.num_images})"
                if row.match_psnr is not None
                else "-"
            )
        else:
            build = f"FAIL: {row.error or 'unknown'}"
            psnr = "-"
        fused = row.fused_npu_op or ("no" if row.build_ok else "-")
        if row.build_ok and not row.conv_act_fused:
            if row.cpu_ops:
                fused = f"Conv + {','.join(row.cpu_ops)} (split)"
            elif fused != "no":
                fused = f"{fused} (split)"
            else:
                fused = "no (split)"
        notes = "; ".join(row.fuse_notes[:2]) if row.fuse_notes else ""
        table_rows.append(
            [
                row.name,
                build,
                "+".join(row.onnx_ops),
                fused,
                ",".join(row.cpu_ops) or "-",
                psnr,
                notes,
            ]
        )

    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cell in enumerate(tr):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [
        _fmt(headers),
        _fmt(["-" * w for w in widths]),
        *(_fmt(tr) for tr in table_rows),
    ]
    return "\n".join(lines)


def _row_to_json(row: ActBenchRow) -> str:
    return json.dumps(dataclasses.asdict(row))


def _row_from_json(payload: str) -> ActBenchRow:
    return ActBenchRow(**json.loads(payload))


def _benchmark_one_subprocess(
    name: str,
    onnx_path: Path,
    args: argparse.Namespace,
    *,
    calib: Path,
    log_dir: Path,
) -> ActBenchRow:
    src = _src_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH")]))
    cmd = [
        sys.executable,
        "-m",
        "rknn_super_resolution.deploy.act_fusion_bench",
        "--phase",
        "single",
        "--activations",
        name,
        "--workdir",
        str(Path(args.workdir).resolve()),
        "--calib_dir",
        args.calib_dir,
        "--max-images",
        str(args.max_images),
        "--input_h",
        str(args.input_h),
        "--input_w",
        str(args.input_w),
        "--target",
        args.target,
        "--quantize",
        args.quantize,
        "--quantized_method",
        args.quantized_method,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.getcwd())
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise SystemExit(
            f"RKNN subprocess failed for {name} (rc={proc.returncode}): {' | '.join(tail)}"
        )
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return _row_from_json(line)
    raise SystemExit(f"RKNN subprocess for {name} returned no JSON payload")


def _resolve_calib(path: str) -> Path:
    calib = Path(path).expanduser()
    if not calib.is_absolute():
        calib = (Path.cwd() / calib).resolve()
    return calib


def _run_rknn_phase(args: argparse.Namespace) -> list[ActBenchRow]:
    workdir = Path(args.workdir).resolve()
    calib = _resolve_calib(args.calib_dir)
    onnx_paths = {name: workdir / f"conv_{name}.onnx" for name in args.activations}
    missing = [str(p) for p in onnx_paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing ONNX files (run export phase first): {missing[:3]}")

    rows: list[ActBenchRow] = []
    log_dir = workdir / "rknn_logs"
    use_subprocess = len(onnx_paths) > 1
    for name, onnx_path in onnx_paths.items():
        print(f"--> RKNN build: {name}")
        if use_subprocess:
            rows.append(
                _benchmark_one_subprocess(name, onnx_path, args, calib=calib, log_dir=log_dir)
            )
        else:
            rows.append(
                _benchmark_one_rknn(
                    name,
                    onnx_path,
                    calib_list=calib,
                    max_images=args.max_images,
                    input_h=args.input_h,
                    input_w=args.input_w,
                    target=args.target,
                    quantize=args.quantize,
                    quantized_method=args.quantized_method,
                    log_dir=log_dir,
                )
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    deploy = load_config().deploy
    parser = argparse.ArgumentParser(
        description="Benchmark RKNN Conv+activation fusion and INT8 output fidelity.",
    )
    parser.add_argument(
        "--activations",
        type=str,
        default=",".join(DEFAULT_ACTIVATIONS),
        help=f"Comma-separated list (default: {','.join(DEFAULT_ACTIVATIONS)})",
    )
    parser.add_argument("--workdir", type=str, default="checkpoints/act_fusion_bench")
    parser.add_argument("--calib_dir", type=str, default=deploy.calib_dir)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--input_h", type=int, default=64)
    parser.add_argument("--input_w", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target",
        type=normalize_rknn_target,
        default=deploy.target,
        help=f"RKNN Toolkit target_platform (tested: {', '.join(TESTED_RKNN_TARGETS)}).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=deploy.rknn_quantize,
        choices=["normal", "mmse", "kl_divergence", "dynamic"],
    )
    parser.add_argument(
        "--quantized_method",
        type=str,
        default=deploy.rknn_quantized_method,
        choices=["channel", "layer"],
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="RKNN-dedicated Python interpreter.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "export", "rknn", "single"],
        help="export: ONNX only; rknn: build/eval existing ONNX; single: one activation JSON row; all: both.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the summary table.",
    )
    return parser.parse_args(argv)


def _spawn_rknn_phase(args: argparse.Namespace, rknn_python: Path) -> int:
    if not rknn_python.is_file():
        raise SystemExit(
            f"RKNN Python not found: {rknn_python}\n"
            "Create .venv-rknn and install rknn-toolkit2, or pass --python."
        )
    src = _src_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH")]))
    cmd = [
        str(rknn_python),
        "-m",
        "rknn_super_resolution.deploy.act_fusion_bench",
        "--phase",
        args.phase if args.phase != "all" else "rknn",
        "--workdir",
        str(Path(args.workdir).resolve()),
        "--calib_dir",
        args.calib_dir,
        "--max-images",
        str(args.max_images),
        "--input_h",
        str(args.input_h),
        "--input_w",
        str(args.input_w),
        "--target",
        args.target,
        "--quantize",
        args.quantize,
        "--quantized_method",
        args.quantized_method,
        "--activations",
        ",".join(args.activations),
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    print(f"--> Re-exec RKNN phase with: {rknn_python}")
    return subprocess.call(cmd, env=env, cwd=os.getcwd())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.activations = tuple(a.strip() for a in args.activations.split(",") if a.strip())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "export"}:
        print(f"--> Exporting ONNX to {workdir.resolve()}")
        export_onnx_models(
            workdir,
            args.activations,
            input_h=args.input_h,
            input_w=args.input_w,
            seed=args.seed,
        )

    rows: list[ActBenchRow] = []
    if args.phase in {"all", "rknn", "single"}:
        rknn_python = resolve_rknn_python(args.python)
        in_rknn_env = Path(sys.prefix).absolute() == rknn_python.parent.parent.absolute()
        if not in_rknn_env:
            raise SystemExit(_spawn_rknn_phase(args, rknn_python))
        if args.phase == "single":
            if len(args.activations) != 1:
                raise SystemExit("--phase single requires exactly one activation in --activations")
            name = args.activations[0]
            onnx_path = Path(args.workdir).resolve() / f"conv_{name}.onnx"
            row = _benchmark_one_rknn(
                name,
                onnx_path,
                calib_list=_resolve_calib(args.calib_dir),
                max_images=args.max_images,
                input_h=args.input_h,
                input_w=args.input_w,
                target=args.target,
                quantize=args.quantize,
                quantized_method=args.quantized_method,
                log_dir=Path(args.workdir).resolve() / "rknn_logs",
            )
            print(_row_to_json(row))
            return
        rows = _run_rknn_phase(args)

    if rows:
        table = format_table(rows)
        print()
        print(table)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(table + "\n")
            print(f"\nWrote summary to {out.resolve()}")


if __name__ == "__main__":
    main()
