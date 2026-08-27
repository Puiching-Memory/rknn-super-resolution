"""Export the Phase-RLFN residual core to ONNX for RKNN conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from rknn_super_resolution.config import load_config
from rknn_super_resolution.deploy.export_prep import prepare_float_for_export
from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.models.graph_format import FLOAT_GRAPH_FORMAT, PT2E_QAT_FORMAT
from rknn_super_resolution.models.qat_utils import load_qat_weights_for_rknn_export


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

    def forward(self, phases: torch.Tensor, codec_feature: torch.Tensor) -> torch.Tensor:
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
    parser.add_argument("--input_h", type=int, default=cfg.deploy.input_h)
    parser.add_argument("--input_w", type=int, default=cfg.deploy.input_w)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--weight-clip", action=argparse.BooleanOptionalAction, default=False)
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
    if not isinstance(raw, dict) or not {"graph_format", "state_dict"}.issubset(raw):
        raise TypeError(f"Expected a versioned model checkpoint in {args.weight}")
    state_dict = raw["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid checkpoint state_dict in {args.weight}")
    if args.input_h % args.phase_factor or args.input_w % args.phase_factor:
        raise ValueError("input_h and input_w must be divisible by phase_factor")
    core_h = args.input_h // args.phase_factor
    core_w = args.input_w // args.phase_factor
    codec_h = ((args.input_h + 15) // 16) * 2
    codec_w = ((args.input_w + 15) // 16) * 2
    phases = torch.randn(1, model.core_in_channels, core_h, core_w, device=device)
    codec = torch.randn(1, model.codec_feature_channels, codec_h, codec_w, device=device)
    example_inputs = (phases, codec) if args.codec_context else (phases,)

    graph_format = raw["graph_format"]
    if graph_format == PT2E_QAT_FORMAT:
        load_qat_weights_for_rknn_export(model, state_dict)
        prepare_float_for_export(
            model,
            clip_min=args.clip_min if args.weight_clip else None,
            clip_max=args.clip_max if args.weight_clip else None,
        )
    elif graph_format == FLOAT_GRAPH_FORMAT:
        model.load_state_dict(state_dict, strict=True)
        prepare_float_for_export(
            model,
            clip_min=args.clip_min if args.weight_clip else None,
            clip_max=args.clip_max if args.weight_clip else None,
        )
    else:
        raise ValueError(f"Unsupported checkpoint graph format: {graph_format}")

    wrapper: nn.Module = _CodecCore(model) if args.codec_context else _SRCore(model)
    wrapper.eval()
    input_names = ["phases", "codec_feature"] if args.codec_context else ["phases"]
    export_kwargs: dict = {
        "input_names": input_names,
        "output_names": ["phase_residual"],
        "opset_version": 18,
        "dynamo": True,
        "external_data": False,
    }

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
