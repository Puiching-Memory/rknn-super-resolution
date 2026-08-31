"""Tests for sparse-attention RKNN benchmark helpers."""

import pytest

from rknn_super_resolution.deploy.sparse_attention_bench import (
    parse_layer_totals,
    parse_variant,
)


def test_parse_variant() -> None:
    variant = parse_variant("mid:90:160:8:32")
    assert variant.name == "mid"
    assert variant.height == 90
    assert variant.value_channels == 32
    with pytest.raises(ValueError):
        parse_variant("broken:90:160")


def test_parse_layer_totals_uses_final_table() -> None:
    log = """
Network Layer Information Table
D RKNN: [00:00:00.000] 0 Conv INT8 NPU (...) (...) 10/20/30 0% 40 Conv:a
<<<<<< end: rknn::RKNNModelRegCmdbuildPass
Network Layer Information Table
D RKNN: [00:00:00.001] 0 Conv INT8 NPU (...) (...) 100/200/300 0% 400 Conv:b
D RKNN: [00:00:00.002] 1 Add INT8 NPU (...) (...) 0/0/0 50 Add:c
<<<<<< end: rknn::RKNNModelRegCmdbuildPass
"""
    assert parse_layer_totals(log) == (300, 450)
