"""Tests for core-only QAT preparation."""

import onnx
import pytest
import torch

from rknn_super_resolution.deploy.onnx import _CodecCore
from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.models.qat_utils import (
    convert_qat_model,
    load_qat_weights_for_rknn_export,
    prepare_model_for_qat,
)


def test_qat_prepares_only_residual_core() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1, codec_project_channels=4)
    phases = torch.randn(1, 12, 8, 8)
    codec = torch.randn(1, 96, 2, 2)
    prepared = prepare_model_for_qat(model, example_inputs=(phases, codec))
    assert isinstance(prepared.core, torch.fx.GraphModule)
    assert isinstance(prepared.input_unshuffle, torch.nn.PixelUnshuffle)
    assert isinstance(prepared.output_shuffle, torch.nn.PixelShuffle)
    output = prepared(
        torch.rand(2, 3, 16, 16) * 255.0,
        torch.randn(2, 96, 2, 2),
    )
    assert output.shape == (2, 3, 48, 48)
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in prepared.parameters())

    rknn_export = PhaseRLFNSR(num_channels=8, num_blocks=1, codec_project_channels=4)
    load_qat_weights_for_rknn_export(rknn_export, prepared.state_dict())
    for key, value in rknn_export.state_dict().items():
        assert torch.equal(value, prepared.state_dict()[key])


@pytest.mark.filterwarnings(
    "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning"
)
def test_qat_conversion_exports_standard_onnx_qdq(tmp_path) -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1, codec_project_channels=4)
    prepared = prepare_model_for_qat(
        model,
        example_inputs=(
            torch.randn(1, 12, 8, 8),
            torch.randn(1, 96, 2, 2),
        ),
    )
    with torch.no_grad():
        prepared(torch.rand(1, 3, 16, 16) * 255.0, torch.randn(1, 96, 2, 2))

    converted = convert_qat_model(prepared)

    targets = {str(node.target) for node in converted.core.graph.nodes}
    assert any("quantize_per_tensor" in target for target in targets)
    assert any("dequantize_per_tensor" in target for target in targets)

    path = tmp_path / "qat_qdq.onnx"
    torch.onnx.export(
        _CodecCore(converted).eval(),
        (torch.randn(1, 12, 8, 8), torch.randn(1, 96, 2, 2)),
        path,
        input_names=["phases", "codec_feature"],
        output_names=["phase_residual"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    op_types = {node.op_type for node in onnx.load(path).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"}.issubset(op_types)


def test_qat_supports_sr_only_core() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1, codec_project_channels=4)
    prepared = prepare_model_for_qat(
        model,
        example_inputs=(torch.randn(1, 12, 8, 8),),
    )

    with torch.no_grad():
        output = prepared(torch.rand(2, 3, 16, 16) * 255.0)

    assert output.shape == (2, 3, 48, 48)
