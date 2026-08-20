"""Phase-domain CPU packing and RKNN configuration tests."""

from argparse import Namespace

import numpy as np

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.rknn import _config_kwargs, _default_input_size
from rk3588_mobile_sr.deploy.rknn_eval import (
    pixel_shuffle_nchw_to_hwc,
    pixel_unshuffle_hwc_to_nchw,
)


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
