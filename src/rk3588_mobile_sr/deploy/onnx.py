"""Export fused MobileOneSR to ONNX for RKNN conversion."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.export_prep import clip_deploy_weights, fused_weight_report, prepare_float_for_export
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.models.qat_utils import load_deploy_float_from_qat_checkpoint
from rk3588_mobile_sr.utils.train_framework import _normalize_state_dict, require_cuda


class _NHWCOutputWrapper(nn.Module):
    """Append NCHW -> NHWC permute for RKNN / RGA-friendly output layout."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 3, 1)


def parse_args():
    cfg = load_config()
    deploy = cfg.deploy
    stage3 = cfg.stage3_qat
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--output", type=str, default=deploy.onnx_output)
    parser.add_argument("--scale", type=int, default=cfg.model.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.model.num_channels)
    parser.add_argument("--num_blocks", type=int, default=cfg.model.num_blocks)
    parser.add_argument("--num_conv_branches", type=int, default=cfg.model.num_conv_branches)
    parser.add_argument("--qat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--from-qat",
        action="store_true",
        help="Load a Stage-3 QAT checkpoint (fused deploy graph, fake-quant disabled).",
    )
    parser.add_argument("--backend", type=str, default=stage3.backend)
    parser.add_argument("--input_h", type=int, default=deploy.input_h)
    parser.add_argument("--input_w", type=int, default=deploy.input_w)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Export fixed input shape (recommended for RKNN conversion).",
    )
    parser.add_argument(
        "--bn-recalibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refresh BN running stats before deploy fuse (float export; usually keep off).",
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=deploy.calib_dir,
        help="Text file listing LR images for BN recalibration.",
    )
    parser.add_argument("--bn_batches", type=int, default=stage3.bn_batches)
    parser.add_argument(
        "--identity-var-floor",
        type=float,
        default=1e-2,
        help="Lower bound on identity BN running_var during deploy fuse (0=disable).",
    )
    parser.add_argument("--clip-min", type=float, default=stage3.clip_min)
    parser.add_argument("--clip-max", type=float, default=stage3.clip_max)
    parser.add_argument(
        "--weight-clip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clip fused conv weights after deploy fuse. Default: on for --from-qat, off for float.",
    )
    parser.add_argument(
        "--output-nhwc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Permute ONNX output to NHWC (1,H,W,3). Default off: adds RKNN Transpose, "
            "~3x internal memory vs NCHW; prefer NCHW export + on-board RGA convert."
        ),
    )
    return parser.parse_args()


def _warn_output_nhwc_enabled() -> None:
    print(
        "WARNING: --output-nhwc appends NCHW→NHWC Transpose in the graph. "
        "RKNN may insert an extra layout op (~32 MB/frame RW at 1080p) and internal "
        "tensor memory can grow ~3x vs default NCHW. Accuracy is unchanged; latency/"
        "memory usually favor NCHW export + RGA format convert on the board."
    )


def main():
    args = parse_args()
    require_cuda()
    device = torch.device("cuda")

    model = MobileOneSR(
        scale=args.scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        num_conv_branches=args.num_conv_branches,
    ).to(device)
    raw = torch.load(args.weight, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = _normalize_state_dict(raw["state_dict"])
    elif isinstance(raw, dict):
        state_dict = _normalize_state_dict(raw)
    else:
        raise TypeError(f"Unsupported checkpoint format in {args.weight}")

    clip_min = None
    clip_max = None
    use_clip = args.weight_clip
    if use_clip is None:
        use_clip = args.from_qat or args.qat
    if use_clip:
        clip_min, clip_max = args.clip_min, args.clip_max

    if args.from_qat or args.qat:
        model = load_deploy_float_from_qat_checkpoint(
            model,
            state_dict,
            identity_var_floor=args.identity_var_floor,
        )
        if clip_min is not None and clip_max is not None:
            clip_deploy_weights(model, clip_min, clip_max)
            print(f"--> Fused weights clipped to [{clip_min}, {clip_max}]")
        peaks = fused_weight_report(model)
        print(f"--> Fused weight peaks: {peaks} (max={max(peaks.values()):.4f})")
        model.eval()
    else:
        model.load_state_dict(state_dict)
        prepare_float_for_export(
            model,
            device=device,
            calib_list=args.calib_dir if args.bn_recalibrate else None,
            input_h=args.input_h,
            input_w=args.input_w,
            bn_batches=args.bn_batches,
            identity_var_floor=args.identity_var_floor,
            clip_min=clip_min,
            clip_max=clip_max,
            do_bn_recalibrate=args.bn_recalibrate,
        )

    if args.output_nhwc:
        _warn_output_nhwc_enabled()
        model = _NHWCOutputWrapper(model)
        model.eval()
        print("--> Output layout: NHWC (1, H, W, 3)")

    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    export_kwargs: dict = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": 18,
        "dynamo": True,
        "external_data": False,
    }
    if not args.static:
        export_kwargs["dynamic_shapes"] = (
            {
                0: Dim("batch"),
                2: Dim("height"),
                3: Dim("width"),
            },
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy_input,), str(output), **export_kwargs)
    print(f"ONNX exported to {output}")


if __name__ == "__main__":
    main()
