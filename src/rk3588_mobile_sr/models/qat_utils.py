"""QAT helpers for the integer Phase-RLFN residual core."""

from __future__ import annotations

import copy

import torch
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from .phase_rlfn_sr import PhaseRLFNSR


def get_qconfig(backend: str = "qnnpack"):
    """Return the backend QAT configuration used by the exported core."""
    return get_default_qat_qconfig(backend)


def prepare_model_for_qat(
    model: PhaseRLFNSR,
    backend: str = "qnnpack",
    example_inputs: tuple[torch.Tensor, ...] | None = None,
) -> PhaseRLFNSR:
    """Prepare only the NPU residual core, leaving RGA bicubic in float."""
    model.switch_to_deploy()
    model.train()
    if example_inputs is None:
        example_inputs = (
            torch.randn(1, model.core_in_channels, 180, 320),
            torch.randn(1, model.codec_feature_channels, 46, 80),
        )
    mapping = QConfigMapping().set_global(get_qconfig(backend))
    model.core = prepare_qat_fx(model.core, mapping, example_inputs)
    return model


def convert_qat_model(prepared_model: PhaseRLFNSR) -> PhaseRLFNSR:
    """Convert a prepared model without quantizing bicubic/add/clip operations."""
    quantized = copy.deepcopy(prepared_model).eval()
    quantized.core = convert_fx(quantized.core)
    return quantized


def load_qat_checkpoint_for_export(
    model: PhaseRLFNSR,
    state_dict: dict[str, torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    backend: str = "qnnpack",
) -> PhaseRLFNSR:
    """Reconstruct the QAT core, load its state and disable fake quantization."""
    prepared = prepare_model_for_qat(model, backend=backend, example_inputs=example_inputs)
    prepared.load_state_dict(state_dict, strict=True)
    prepared.eval()
    prepared.apply(torch.ao.quantization.disable_observer)
    prepared.apply(torch.ao.quantization.disable_fake_quant)
    return prepared
