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
                if self._stop.is_set():
                    break
                self._queue.put(batch)
        except BaseException as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(None)

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
            self._queue.get_nowait()
        self._queue.put(None)
