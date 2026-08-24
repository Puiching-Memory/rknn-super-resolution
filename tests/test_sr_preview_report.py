"""Tests for the SR preview report grid assembly."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rknn_super_resolution.dev.sr_preview_report import (
    GRID_BORDER,
    GRID_CAPTION_H,
    GRID_COLS,
    GRID_GAP,
    GRID_PAD,
    GRID_TILE_H,
    GRID_TILE_W,
    GRID_TITLE_H,
    GRID_TITLES,
    assemble_grid,
)


def test_assemble_grid_layout(tmp_path):
    tile_hw = (16, 24)
    rows, cols = len(GRID_TITLES), len(GRID_TITLES[0])
    panels = [
        np.full((rows * tile_hw[0], cols * tile_hw[1], 3), 128, dtype=np.uint8)
        for _ in range(3)
    ]
    out = assemble_grid(
        panels, tmp_path / "grid.png", tile_hw=tile_hw, captions=["a", "b", "c"]
    )
    canvas = Image.open(out)
    group_w = cols * GRID_TILE_W + 2 * GRID_BORDER + 2 * GRID_PAD
    group_h = (
        GRID_CAPTION_H + rows * (GRID_TITLE_H + GRID_TILE_H) + 2 * GRID_BORDER + 2 * GRID_PAD
    )
    grid_rows = 2
    assert canvas.size == (
        GRID_COLS * group_w + (GRID_COLS - 1) * GRID_GAP,
        grid_rows * group_h + (grid_rows - 1) * GRID_GAP,
    )


def test_assemble_grid_rejects_mismatched_panel(tmp_path):
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        assemble_grid([bad], tmp_path / "grid.png", tile_hw=(16, 24))
