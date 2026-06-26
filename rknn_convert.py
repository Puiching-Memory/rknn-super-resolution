"""Convert ONNX model to RKNN INT8 for RK3588."""

import argparse

# def import_rknn():
#     try:
#         from rknn.api import RKNN
#         return RKNN
#     except ImportError as e:
#         raise ImportError(
#             "rknn-toolkit2 is required. Install it in your RKNN conversion environment."
#         ) from e


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output", type=str, default="mobileone_sr_x3.rknn")
    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--calib_dir", type=str, required=True)
    parser.add_argument("--input_size", type=str, default="3,360,640")
    parser.add_argument("--quantize", type=str, default="normal", choices=["normal", "dynamic"])
    parser.add_argument("--do_quantization", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # RKNN = import_rknn()
    raise NotImplementedError(
        "RKNN conversion requires rknn-toolkit2 and a calibration dataset. "
        "Use this script as a template and fill in RKNN API calls."
    )
    # rknn = RKNN(verbose=True)
    # rknn.config(
    #     target_platform=args.target,
    #     quantize=args.quantize,
    #     # ...
    # )
    # rknn.load_onnx(model=args.onnx, inputs=[...], input_size_list=[...])
    # rknn.build(do_quantization=args.do_quantization, dataset=args.calib_dir)
    # rknn.export_rknn(args.output)


if __name__ == "__main__":
    main()
