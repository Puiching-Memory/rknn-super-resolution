"""Phase-domain CPU packing and RKNN configuration tests."""

from argparse import Namespace

import numpy as np
import torch

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.onnx import _CoreWrapper
from rk3588_mobile_sr.deploy.rknn import _config_kwargs, _default_input_size
from rk3588_mobile_sr.deploy.rknn_eval import (
    pixel_shuffle_nchw_to_hwc,
    pixel_unshuffle_hwc_to_nchw,
)
from rk3588_mobile_sr.models import MobileOneSR


def test_numpy_phase_packing_uses_pixel_unshuffle_order():
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    packed = pixel_unshuffle_hwc_to_nchw(image, 2)
    expected = np.stack(
        [image[row::2, col::2, channel] for channel in range(3) for row in range(2) for col in range(2)]
    )[np.newaxis]

    assert np.array_equal(packed, expected)
    assert np.array_equal(pixel_shuffle_nchw_to_hwc(packed, 2), image)


def test_rk3576_phase_defaults():
    cfg = load_config()
    assert _default_input_size(cfg) == "12,180,320"

    args = Namespace(
        input_size="12,180,320",
        target="rk3576",
        quantize="kl_divergence",
        quantized_method="channel",
    )
    kwargs = _config_kwargs(args, do_quantization=True)
    assert kwargs["target_platform"] == "rk3576"
    assert kwargs["mean_values"] == [[0] * 12]
    assert kwargs["std_values"] == [[1] * 12]


def test_onnx_core_wrapper_matches_rk3576_contract():
    cfg = load_config()
    model = MobileOneSR(
        num_channels=cfg.model.num_channels,
        num_blocks=cfg.model.num_blocks,
        scale=cfg.model.scale,
        phase_factor=cfg.model.phase_factor,
        output_kernel_size=cfg.model.output_kernel_size,
    )
    model.eval()
    core_h = cfg.deploy.input_h // cfg.model.phase_factor
    core_w = cfg.deploy.input_w // cfg.model.phase_factor
    packed = torch.randn(1, model.core_in_channels, core_h, core_w)
    wrapped = _CoreWrapper(model)
    with torch.no_grad():
        core = wrapped(packed)
        via_model = model.forward_core(packed)

    assert model.core_in_channels == 12
    assert model.core_out_channels == 108
    assert packed.shape == (1, 12, 180, 320)
    assert core.shape == (1, 108, 180, 320)
    assert torch.equal(core, via_model)
