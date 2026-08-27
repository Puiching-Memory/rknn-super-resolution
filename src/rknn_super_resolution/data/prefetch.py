"""Background batch prefetching to overlap data loading with GPU compute."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


class BatchPrefetcher(Iterator[T]):
    """Prefetch batches from a loader on a daemon thread."""

    def __init__(self, loader, *, buffer_size: int = 2) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self._loader = iter(loader)
        self._queue: queue.Queue[T | BaseException | None] = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="batch-prefetch", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for batch in self._loader:
                if not self._put(batch):
                    break
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(None)

    def _put(self, item: T | BaseException | None) -> bool:
        """Put an item unless shutdown was requested while the queue was full."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
            except queue.Full:
                continue
            return True
        return False

    def __iter__(self) -> BatchPrefetcher[T]:
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
