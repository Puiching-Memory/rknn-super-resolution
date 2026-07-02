"""Unit tests for RKNN activation fusion benchmark helpers."""

from rk3588_mobile_sr.deploy.act_fusion_bench import (
    _parse_fuse_notes,
    _parse_rknn_ops,
    format_table,
    ActBenchRow,
)


def test_parse_rknn_ops_extracts_npu_conv_relu():
    log = (
        "D RKNN: [00:00:00.000] ID   OpType             DataType Target\n"
        "D RKNN: [00:00:00.001] 0    InputOperator      INT8     CPU\n"
        "D RKNN: [00:00:00.002] 1    ConvRelu           INT8     NPU    (1,3,64,64)\n"
        "D RKNN: [00:00:00.003] 2    OutputOperator     INT8     CPU\n"
    )
    ops = _parse_rknn_ops(log)
    assert (1, "ConvRelu", "INT8", "NPU") in ops


def test_parse_fuse_notes_captures_gelu_rewrite():
    log = (
        "D fuse_ops results:\n"
        "D     replace_torch_gelu4: remove node = ['a'], add node = ['b']\n"
        "D fuse_ops done.\n"
    )
    notes = _parse_fuse_notes(log)
    assert len(notes) == 1
    assert "replace_torch_gelu4" in notes[0]


def test_format_table_includes_fusion_and_psnr():
    rows = [
        ActBenchRow(
            name="relu",
            onnx_ops=["Conv", "Relu"],
            build_ok=True,
            error="",
            npu_ops=["ConvRelu"],
            cpu_ops=[],
            fused_npu_op="ConvRelu",
            conv_act_fused=True,
            fuse_notes=[],
            match_psnr=42.5,
            match_psnr_min=40.0,
            num_images=5,
        ),
        ActBenchRow(
            name="mish",
            onnx_ops=["Conv", "Mish"],
            build_ok=True,
            error="",
            npu_ops=["Conv"],
            cpu_ops=["Mish"],
            fused_npu_op=None,
            conv_act_fused=False,
            fuse_notes=[],
            match_psnr=30.1,
            match_psnr_min=28.0,
            num_images=5,
        ),
    ]
    text = format_table(rows)
    assert "ConvRelu" in text
    assert "42.50 dB" in text
    assert "Mish" in text
