"""Export fused MobileOneSR to ONNX for RKNN conversion."""

import argparse

import torch

from models.mobileone_sr import MobileOneSR
from models.qat_utils import prepare_model_for_qat
from utils.train_framework import require_cuda


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--output", type=str, default="mobileone_sr_x3.onnx")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_conv_branches", type=int, default=4)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--backend", type=str, default="qnnpack")
    parser.add_argument("--input_h", type=int, default=360)
    parser.add_argument("--input_w", type=int, default=640)
    return parser.parse_args()


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
    model.load_state_dict(torch.load(args.weight, map_location=device))

    if args.qat:
        dummy = (torch.randn(1, 3, args.input_h, args.input_w).to(device),)
        model = prepare_model_for_qat(model, backend=args.backend, example_inputs=dummy)
        model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
        model.eval()
    else:
        model.switch_to_deploy()
        model.eval()

    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"ONNX exported to {args.output}")


if __name__ == "__main__":
    main()
