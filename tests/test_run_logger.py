"""Tests for run_logger."""

from pathlib import Path

from rknn_super_resolution.utils.run_logger import logger, setup_run_logger


def test_setup_run_logger_rank0_writes_file(tmp_path: Path) -> None:
    setup_run_logger(tmp_path, rank=0)
    logger.info("hello")
    logger.complete()
    log_path = tmp_path / "train.log"
    assert log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "training run started" in text
    assert "hello" in text


def test_setup_run_logger_nonzero_rank_is_silent(tmp_path: Path) -> None:
    setup_run_logger(tmp_path, rank=1)
    logger.info("should not appear")
    assert not (tmp_path / "train.log").exists()
