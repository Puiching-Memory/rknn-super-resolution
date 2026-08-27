"""Tests for training-session artifact handling."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from rknn_super_resolution.distributed.context import DistributedContext
from rknn_super_resolution.train.session import TrainSession


def test_build_loaders_archives_fixed_dataset_split(tmp_path: Path):
    manifest = tmp_path / "split.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  split_manifest: {manifest}\n", encoding="utf-8")
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    session = TrainSession(
        DistributedContext(rank=0, world_size=1, device=torch.device("cpu")),
        Namespace(config=str(config)),
        save_dir=save_dir,
        experiment_name="test",
    )

    with patch(
        "rknn_super_resolution.train.session.build_loaders",
        return_value=("train", "validation"),
    ):
        loaders = session.build_loaders()

    assert loaders.train == "train"
    assert loaders.val == "validation"
    assert (save_dir / "dataset_split.json").read_text(encoding="utf-8") == '{"version": 1}\n'
