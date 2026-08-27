"""Tests for background batch prefetch lifecycle."""

from __future__ import annotations

import time

from rknn_super_resolution.data.prefetch import BatchPrefetcher


def test_close_unblocks_full_queue_and_stops_worker() -> None:
    def infinite_loader():
        while True:
            yield object()

    prefetcher = BatchPrefetcher(infinite_loader(), buffer_size=1)
    time.sleep(0.05)

    started = time.monotonic()
    prefetcher.close()

    assert time.monotonic() - started < 0.5
    assert not prefetcher._thread.is_alive()
