"""PT2E quantization-aware training helpers for the residual core."""

from __future__ import annotations

import torch
from torch.export import Dim
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    disable_observer,
    move_exported_model_to_eval,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import (
    ArmInductorQuantizer,
    get_default_arm_inductor_quantization_config,
)

from .phase_rlfn_sr import PhaseRLFNSR


def _dynamic_batch_shapes(example_inputs: tuple[torch.Tensor, ...]) -> tuple[dict, ...]:
    batch = Dim("batch", min=1)
    return tuple({0: batch} for _ in example_inputs)


def _capture_inputs(example_inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    batch_sizes = {tensor.shape[0] for tensor in example_inputs}
    if len(batch_sizes) != 1:
        raise ValueError("PT2E example inputs must share the same batch size")
    if batch_sizes != {1}:
        return example_inputs
    return tuple(tensor.expand(2, *tensor.shape[1:]).contiguous() for tensor in example_inputs)


def disable_qat_observers(model: torch.nn.Module) -> None:
    """Freeze PT2E observer statistics while keeping fake quantization enabled."""
    model.apply(disable_observer)


def prepare_model_for_qat(
    model: PhaseRLFNSR,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
) -> PhaseRLFNSR:
    """Export and prepare only the NPU residual core for PT2E QAT."""
    model.switch_to_deploy()
    model.train()
    if example_inputs is None:
        example_inputs = (
            torch.randn(1, model.core_in_channels, 180, 320),
            torch.randn(1, model.codec_feature_channels, 46, 80),
        )
    capture_inputs = _capture_inputs(example_inputs)
    exported_core = torch.export.export(
        model.core,
        capture_inputs,
        dynamic_shapes=_dynamic_batch_shapes(capture_inputs),
        strict=True,
    ).module()
    quantizer = ArmInductorQuantizer().set_global(
        get_default_arm_inductor_quantization_config(is_qat=True)
    )
    model.core = prepare_qat_pt2e(exported_core, quantizer)
    allow_exported_model_train_eval(model.core)
    model.train()
    return model


def convert_qat_model(prepared_model: PhaseRLFNSR) -> PhaseRLFNSR:
    """Convert a prepared PT2E core in place to a portable Q/DQ graph."""
    prepared_model.core = convert_pt2e(prepared_model.core)
    allow_exported_model_train_eval(prepared_model.core)
    move_exported_model_to_eval(prepared_model.core)
    prepared_model.eval()
    return prepared_model


def load_qat_weights_for_rknn_export(
    model: PhaseRLFNSR,
    state_dict: dict[str, torch.Tensor],
) -> PhaseRLFNSR:
    """Load native tensors from a PT2E checkpoint into an eager RKNN export graph.

    RKNN Toolkit2 2.3.2 lowers generic PT2E Q/DQ residual graphs mostly to
    FP16. Exporting the QAT-trained native weights as float ONNX lets RKNN PTQ
    produce the intended all-INT8 graph from representative MLVC inputs.
    """
    native_state = model.state_dict()
    missing = sorted(native_state.keys() - state_dict.keys())
    if missing:
        raise RuntimeError(f"PT2E checkpoint is missing native model tensors: {missing}")
    model.load_state_dict({key: state_dict[key] for key in native_state}, strict=True)
    return model
