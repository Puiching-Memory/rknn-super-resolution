"""Phase-domain CPU packing and RKNN configuration tests."""

from argparse import Namespace

import numpy as np
import torch

from rknn_super_resolution.config import load_config
from rknn_super_resolution.deploy.onnx import _CodecCore, _SRCore
from rknn_super_resolution.deploy.rknn import (
    _config_kwargs,
    _default_input_size,
    _load_smoke_inputs,
    _onnx_has_qdq,
    parse_args,
)
from rknn_super_resolution.deploy.rknn_eval import (
    pixel_shuffle_nchw_to_hwc,
    pixel_unshuffle_hwc_to_nchw,
)
from rknn_super_resolution.models import PhaseRLFNSR


def test_numpy_phase_packing_uses_pixel_unshuffle_order() -> None:
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    packed = pixel_unshuffle_hwc_to_nchw(image, 2)
    expected = np.stack(
        [
            image[row::2, col::2, channel]
            for channel in range(3)
            for row in range(2)
            for col in range(2)
        ]
    )[np.newaxis]
    assert np.array_equal(packed, expected)
    assert np.array_equal(pixel_shuffle_nchw_to_hwc(packed, 2), image)


def test_phase_defaults() -> None:
    cfg = load_config()
    assert _default_input_size(cfg) == "12,180,320;96,46,80"
    args = Namespace(
        input_size="12,180,320;96,46,80",
        target="rk3576",
        quantize="kl_divergence",
        quantized_method="channel",
    )
    kwargs = _config_kwargs(args, do_quantization=True)
    assert kwargs["mean_values"] == [[0] * 12, [0] * 96]
    assert kwargs["std_values"] == [[1] * 12, [1] * 96]


def test_rknn_cli_accepts_tested_and_unlisted_boards() -> None:
    default = parse_args(["--onnx", "model.onnx", "--target", "rk3576"])
    assert default.target == "rk3576"
    assert default.quantization is True
    assert parse_args(["--onnx", "model.onnx", "--target", "RK3588"]).target == "rk3588"
    assert parse_args(["--onnx", "model.onnx", "--target", "RV1126B"]).target == "rv1126b"
    assert parse_args(["--onnx", "model.onnx", "--target", "RK9999"]).target == "rk9999"
    assert parse_args(["--onnx", "model.onnx", "--no-quantization"]).quantization is False


def test_onnx_wrappers_match_core_contracts() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1).eval()
    phases = torch.randn(1, 12, 8, 10)
    codec = torch.randn(1, 96, 2, 3)
    with torch.no_grad():
        assert torch.equal(_SRCore(model)(phases), model.forward_core(phases))
        assert torch.equal(_CodecCore(model)(phases, codec), model.forward_core(phases, codec))


def test_rknn_detects_prequantized_qdq_onnx(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
        ],
        "qdq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1]),
            helper.make_tensor("zero", TensorProto.UINT8, [], [0]),
        ],
    )
    path = tmp_path / "qdq.onnx"
    onnx.save(helper.make_model(graph), path)

    assert _onnx_has_qdq(path) is True


def test_load_rknn_smoke_inputs_validates_two_input_contract(tmp_path) -> None:
    phases = np.zeros((1, 12, 8, 10), dtype=np.uint8)
    codec = np.zeros((1, 96, 2, 3), dtype=np.float32)
    phase_path = tmp_path / "phases.npy"
    codec_path = tmp_path / "codec.npy"
    np.save(phase_path, phases)
    np.save(codec_path, codec)
    dataset = tmp_path / "calibration.txt"
    dataset.write_text(f"{phase_path} {codec_path}\n", encoding="utf-8")

    loaded = _load_smoke_inputs(dataset, [(12, 8, 10), (96, 2, 3)])

    assert np.array_equal(loaded[0], phases)
    assert np.array_equal(loaded[1], codec)
