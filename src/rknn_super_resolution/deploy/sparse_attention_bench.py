"""Export and RKNN-compile sparse local-attention micrographs.

The benchmark isolates the deployment cost of coordinate-preserving value
transport.  It intentionally varies token resolution and value width while
keeping a five-candidate cross attention pattern fixed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rknn_super_resolution.deploy.act_fusion_bench import (
    _IGNORE_OPS,
    _capture_build_log,
    _parse_rknn_ops,
)
from rknn_super_resolution.deploy.targets import normalize_rknn_target

DEFAULT_VARIANTS = (
    "coarse_phase:45:80:16:64",
    "mid_compact:90:160:16:16",
    "mid_phase:90:160:16:32",
    "full_compact:180:320:16:16",
)
OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_CYCLES_RE = re.compile(r"^(\d+)/(\d+)/(\d+)$")


@dataclass(frozen=True)
class Variant:
    name: str
    height: int
    width: int
    key_channels: int
    value_channels: int


@dataclass
class BenchResult:
    variant: str
    height: int
    width: int
    key_channels: int
    value_channels: int
    state_kib: float
    build_ok: bool
    error: str
    npu_ops: int
    cpu_ops: list[str]
    total_cycles: int
    rw_kib: int
    simulator_finite: bool


def parse_variant(raw: str) -> Variant:
    parts = raw.split(":")
    if len(parts) != 5:
        raise ValueError("variant must be NAME:H:W:KEY_CHANNELS:VALUE_CHANNELS")
    name = parts[0]
    height, width, key_channels, value_channels = map(int, parts[1:])
    if not name or min(height, width, key_channels, value_channels) < 1:
        raise ValueError(f"invalid sparse-attention variant: {raw!r}")
    return Variant(name, height, width, key_channels, value_channels)


def parse_layer_totals(log: str) -> tuple[int, int]:
    """Sum compiler total cycles and RW KiB from the final layer table."""
    marker = "Network Layer Information Table"
    start = log.rfind(marker)
    if start < 0:
        return 0, 0
    end = log.find("<<<<<< end: rknn::RKNNModelRegCmdbuildPass", start)
    table = log[start:] if end < 0 else log[start:end]
    total_cycles = 0
    rw_kib = 0
    for line in table.splitlines():
        tokens = line.split()
        for index, token in enumerate(tokens):
            match = _CYCLES_RE.match(token)
            if match is None:
                continue
            total_cycles += int(match.group(3))
            for candidate in tokens[index + 1 :]:
                if candidate.isdigit():
                    rw_kib += int(candidate)
                    break
            break
    return total_cycles, rw_kib


def _shift(tensor, dy: int, dx: int):
    import torch.nn.functional as F

    padded = F.pad(tensor, (1, 1, 1, 1))
    y = 1 + dy
    x = 1 + dx
    return padded[..., y : y + tensor.shape[-2], x : x + tensor.shape[-1]]


def _model():
    import torch
    import torch.nn as nn

    class SparseCrossAttention(nn.Module):
        def forward(self, query, key, value):
            keys = [_shift(key, dy, dx) for dy, dx in OFFSETS]
            values = [_shift(value, dy, dx) for dy, dx in OFFSETS]
            scores = torch.cat(
                [(query * candidate).mean(dim=1, keepdim=True) for candidate in keys],
                dim=1,
            )
            weights = torch.softmax(scores, dim=1)
            parts = weights.split(1, dim=1)
            aligned = parts[0] * values[0]
            for weight, candidate in zip(parts[1:], values[1:], strict=True):
                aligned = aligned + weight * candidate
            return aligned

    return SparseCrossAttention()


def export_variant(workdir: Path, variant: Variant, *, seed: int) -> None:
    import torch

    variant_dir = workdir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        1, variant.key_channels, variant.height, variant.width, generator=generator
    )
    key = torch.randn(
        1, variant.key_channels, variant.height, variant.width, generator=generator
    )
    value = torch.randn(
        1, variant.value_channels, variant.height, variant.width, generator=generator
    )
    torch.onnx.export(
        _model().eval(),
        (query, key, value),
        str(variant_dir / "model.onnx"),
        input_names=["query", "key", "value"],
        output_names=["aligned"],
        opset_version=18,
        dynamo=True,
    )
    rows = []
    rng = np.random.default_rng(seed)
    for index in range(4):
        paths = []
        for name, channels in (
            ("query", variant.key_channels),
            ("key", variant.key_channels),
            ("value", variant.value_channels),
        ):
            path = (variant_dir / f"{name}_{index}.npy").resolve()
            array = rng.uniform(
                -1.0,
                1.0,
                (1, channels, variant.height, variant.width),
            ).astype(np.float32)
            np.save(path, array)
            paths.append(str(path))
        rows.append(" ".join(paths))
    (variant_dir / "dataset.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_variant(workdir: Path, variant: Variant, *, target: str) -> BenchResult:
    from rknn.api import RKNN

    variant_dir = workdir / variant.name
    log_path = variant_dir / f"{target}_build.log"
    rknn = RKNN(verbose=True)
    build_ok = False
    error = ""
    simulator_finite = False
    log = ""
    try:
        rknn.config(
            mean_values=[
                [0] * variant.key_channels,
                [0] * variant.key_channels,
                [0] * variant.value_channels,
            ],
            std_values=[
                [1] * variant.key_channels,
                [1] * variant.key_channels,
                [1] * variant.value_channels,
            ],
            target_platform=target,
            quantized_algorithm="normal",
            quantized_method="channel",
        )
        if rknn.load_onnx(str(variant_dir / "model.onnx")) != 0:
            error = "load_onnx failed"
        else:
            def _build() -> None:
                nonlocal build_ok, error
                if rknn.build(
                    do_quantization=True,
                    dataset=str(variant_dir / "dataset.txt"),
                ) != 0:
                    error = "build failed"
                elif rknn.init_runtime() != 0:
                    error = "init_runtime failed"
                else:
                    build_ok = True

            log = _capture_build_log(log_path, _build)
        if build_ok:
            arrays = [
                np.load(variant_dir / f"{name}_0.npy")
                for name in ("query", "key", "value")
            ]
            outputs = rknn.inference(inputs=arrays, data_format="nchw")
            simulator_finite = bool(outputs and np.isfinite(outputs[0]).all())
            rknn.export_rknn(str(variant_dir / f"{target}_int8.rknn"))
    except Exception as exc:  # pragma: no cover - toolkit diagnostics
        error = str(exc)
    finally:
        rknn.release()

    ops = _parse_rknn_ops(log)
    compute_ops = [row for row in ops if row[1] not in _IGNORE_OPS]
    cpu_ops = [row[1] for row in compute_ops if row[3] == "CPU"]
    total_cycles, rw_kib = parse_layer_totals(log)
    return BenchResult(
        variant=variant.name,
        height=variant.height,
        width=variant.width,
        key_channels=variant.key_channels,
        value_channels=variant.value_channels,
        state_kib=(
            variant.height
            * variant.width
            * (variant.key_channels + variant.value_channels)
            / 1024
        ),
        build_ok=build_ok,
        error=error,
        npu_ops=sum(row[3] == "NPU" for row in compute_ops),
        cpu_ops=cpu_ops,
        total_cycles=total_cycles,
        rw_kib=rw_kib,
        simulator_finite=simulator_finite,
    )


def _write_summary(workdir: Path, rows: list[BenchResult]) -> None:
    payload = {"schema_version": 1, "results": [asdict(row) for row in rows]}
    (workdir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "| Variant | Shape | Key/Value C | State KiB | NPU/CPU | Cycles | RW MiB | Simulator |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.variant} | {row.height}×{row.width} | "
            f"{row.key_channels}/{row.value_channels} | {row.state_kib:.1f} | "
            f"{row.npu_ops}/{len(row.cpu_ops)} | {row.total_cycles:,} | "
            f"{row.rw_kib / 1024:.3f} | "
            f"{'PASS' if row.simulator_finite else 'FAIL'} |"
        )
    markdown = "\n".join(lines) + "\n"
    (workdir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("export", "rknn"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--target", type=normalize_rknn_target, default="rk3576")
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(parse_variant(raw) for raw in (args.variants or DEFAULT_VARIANTS))
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.phase == "export":
        for variant in variants:
            export_variant(args.workdir, variant, seed=args.seed)
        return
    rows = [build_variant(args.workdir, variant, target=args.target) for variant in variants]
    _write_summary(args.workdir, rows)


if __name__ == "__main__":
    main()
