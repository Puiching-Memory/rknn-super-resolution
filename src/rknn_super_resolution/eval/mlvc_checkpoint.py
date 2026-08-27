"""Evaluate a training checkpoint once on the complete held-out OpenVidHD test split."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from rknn_super_resolution.config import AppConfig, load_config
from rknn_super_resolution.data.mlvc_loader import build_mlvc_evaluation_loader
from rknn_super_resolution.distributed.context import DistributedContext, distributed_session
from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.models.graph_format import FLOAT_GRAPH_FORMAT, PT2E_QAT_FORMAT
from rknn_super_resolution.models.qat_utils import (
    disable_qat_observers,
    prepare_model_for_qat,
)
from rknn_super_resolution.utils.sr_metrics import validate_ddp_extended
from rknn_super_resolution.utils.train_framework import load_training_module_state_dict
from rknn_super_resolution.utils.vmaf_metric import resolve_vmaf_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="archived dataset_split.json from the training run",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--metric", choices=("vmaf", "psnr"), default="vmaf")
    parser.add_argument("--vmaf_model", type=str, default="1080p")
    return parser.parse_args()


def load_checkpoint_model(
    config: AppConfig,
    checkpoint: Path,
    device: torch.device,
) -> PhaseRLFNSR:
    """Reconstruct the float or QAT graph encoded by unified or bare weights."""
    model_cfg = config.model
    model = PhaseRLFNSR(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        num_channels=model_cfg.num_channels,
        num_blocks=model_cfg.num_blocks,
        scale=model_cfg.scale,
        phase_factor=model_cfg.phase_factor,
        codec_feature_channels=model_cfg.codec_feature_channels,
        codec_project_channels=model_cfg.codec_project_channels,
        codec_upsample_factor=model_cfg.codec_upsample_factor,
        negative_slope=model_cfg.negative_slope,
    ).to(device)
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or not {"graph_format", "phase", "state_dict"}.issubset(raw):
        raise TypeError(f"expected a versioned model checkpoint: {checkpoint}")
    state = raw["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
    ):
        raise TypeError(f"invalid checkpoint state_dict: {checkpoint}")
    phase = str(raw["phase"])
    graph_format = raw["graph_format"]
    if graph_format == PT2E_QAT_FORMAT:
        if not phase.startswith("qat"):
            raise ValueError(f"PT2E QAT checkpoint has non-QAT phase: {phase}")
        lr_h, lr_w = config.data.lr_size
        example_inputs = (
            torch.randn(
                1,
                model.core_in_channels,
                lr_h // model.phase_factor,
                lr_w // model.phase_factor,
                device=device,
            ),
        )
        if config.data.codec_context:
            example_inputs += (
                torch.randn(
                    1,
                    model.codec_feature_channels,
                    ((lr_h + 15) // 16) * 2,
                    ((lr_w + 15) // 16) * 2,
                    device=device,
                ),
            )
        model = prepare_model_for_qat(
            model,
            example_inputs=example_inputs,
        )
    elif graph_format == FLOAT_GRAPH_FORMAT:
        if phase.startswith("qat"):
            raise ValueError(f"float checkpoint has QAT phase: {phase}")
        model.switch_to_deploy()
    else:
        raise ValueError(f"unsupported checkpoint graph format: {graph_format}")
    load_training_module_state_dict(model, state)
    model.eval()
    if graph_format == PT2E_QAT_FORMAT:
        disable_qat_observers(model)
    return model


@contextmanager
def evaluation_session() -> Iterator[DistributedContext]:
    """Use torchrun when configured, otherwise evaluate on the first CUDA device."""
    distributed_env = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if distributed_env & os.environ.keys():
        with distributed_session() as ctx:
            yield ctx
        return
    if not torch.cuda.is_available():
        raise RuntimeError("MLVC checkpoint evaluation requires a CUDA device")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    yield DistributedContext(rank=0, world_size=1, device=device)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.metric == "vmaf":
        resolve_vmaf_binary()

    config = load_config(args.config)
    if args.split_manifest is not None:
        config.data = replace(config.data, split_manifest=str(args.split_manifest.resolve()))
    project_root = Path(__file__).resolve().parents[3]
    with evaluation_session() as ctx:
        model = load_checkpoint_model(config, args.checkpoint, ctx.device)
        test_loader = build_mlvc_evaluation_loader(
            config.data,
            split="test",
            device=ctx.device,
            scale=config.model.scale,
            colorspace=config.data.colorspace,
            batch_size=args.batch_size,
            rank=ctx.rank,
            world_size=ctx.world_size,
            project_root=project_root,
        )
        try:
            _score, metrics = validate_ddp_extended(
                model,
                test_loader,
                ctx.rank,
                ctx.world_size,
                scale=config.model.scale,
                compute_vmaf=args.metric == "vmaf",
                vmaf_model=args.vmaf_model,
                vmaf_enc_size=config.data.lr_size,
                colorspace=config.data.colorspace,
            )
        finally:
            test_loader.close()

        if ctx.is_main:
            if metrics is None:
                raise RuntimeError("test evaluation produced no metrics")
            result = {
                key.replace("val/", "test/"): value for key, value in metrics.to_log_dict().items()
            }
            result["test/source_videos"] = len(test_loader.dataset.sequences)
            result["test/samples"] = len(test_loader.dataset)
            result["test/q_indices"] = list(config.data.q_indices)
            print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
