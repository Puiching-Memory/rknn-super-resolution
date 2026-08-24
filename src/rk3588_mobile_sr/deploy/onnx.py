"""Export the Phase-RLFN residual core to ONNX for RKNN conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.export_prep import prepare_float_for_export
from rk3588_mobile_sr.models import PhaseRLFNSR
from rk3588_mobile_sr.models.qat_utils import load_qat_checkpoint_for_export


class _SRCore(nn.Module):
    def __init__(self, model: PhaseRLFNSR) -> None:
        super().__init__()
        self.model = model

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        return self.model.forward_core(phases)


class _CodecCore(nn.Module):
    def __init__(self, model: PhaseRLFNSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, phases: torch.Tensor, codec_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.model.forward_core(phases, codec_feature)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output", default=cfg.deploy.onnx_output)
    parser.add_argument("--scale", type=int, default=cfg.model.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.model.num_channels)
    parser.add_argument("--num_blocks", type=int, default=cfg.model.num_blocks)
    parser.add_argument("--phase_factor", type=int, default=cfg.model.phase_factor)
    parser.add_argument(
        "--codec-context",
        action=argparse.BooleanOptionalAction,
        default=cfg.deploy.codec_context,
    )
    parser.add_argument(
        "--from-qat", action="store_true", help="load a prepared QAT checkpoint"
    )
    parser.add_argument("--backend", default=cfg.training.backend)
    parser.add_argument("--input_h", type=int, default=cfg.deploy.input_h)
    parser.add_argument("--input_w", type=int, default=cfg.deploy.input_w)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--static", action="store_true")
    parser.add_argument(
        "--weight-clip", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--clip-min", type=float, default=cfg.training.clip_min)
    parser.add_argument("--clip-max", type=float, default=cfg.training.clip_max)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    cfg = load_config()
    model = PhaseRLFNSR(
        scale=args.scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        phase_factor=args.phase_factor,
        codec_feature_channels=cfg.model.codec_feature_channels,
        codec_project_channels=cfg.model.codec_project_channels,
        codec_upsample_factor=cfg.model.codec_upsample_factor,
    ).to(device)
    raw = torch.load(args.weight, map_location=device, weights_only=False)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else None
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {args.weight}")
    if args.input_h % args.phase_factor or args.input_w % args.phase_factor:
        raise ValueError("input_h and input_w must be divisible by phase_factor")
    core_h = args.input_h // args.phase_factor
    core_w = args.input_w // args.phase_factor
    codec_h = ((args.input_h + 15) // 16) * 2
    codec_w = ((args.input_w + 15) // 16) * 2
    phases = torch.randn(1, model.core_in_channels, core_h, core_w, device=device)
    codec = torch.randn(
        1, model.codec_feature_channels, codec_h, codec_w, device=device
    )
    example_inputs = (phases, codec) if args.codec_context else (phases,)

    if args.from_qat:
        model = load_qat_checkpoint_for_export(
            model, state_dict, example_inputs, backend=args.backend
        )
    else:
        model.load_state_dict(state_dict, strict=True)
        prepare_float_for_export(
            model,
            clip_min=args.clip_min if args.weight_clip else None,
            clip_max=args.clip_max if args.weight_clip else None,
        )

    wrapper: nn.Module = _CodecCore(model) if args.codec_context else _SRCore(model)
    input_names = ["phases", "codec_feature"] if args.codec_context else ["phases"]
    dynamic_shapes = [
        {0: Dim("batch"), 2: Dim("phase_height"), 3: Dim("phase_width")}
    ]
    if args.codec_context:
        dynamic_shapes.append(
            {0: Dim("batch"), 2: Dim("codec_height"), 3: Dim("codec_width")}
        )
    export_kwargs: dict = {
        "input_names": input_names,
        "output_names": ["phase_residual"],
        "opset_version": 18,
        "dynamo": True,
        "external_data": False,
    }
    if not args.static:
        export_kwargs["dynamic_shapes"] = tuple(dynamic_shapes)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, example_inputs, str(output), **export_kwargs)
    print(
        f"ONNX exported to {output}: phases={tuple(phases.shape)}, "
        f"codec={tuple(codec.shape) if args.codec_context else None}, "
        f"output_channels={model.core_out_channels}"
    )


if __name__ == "__main__":
    main()
