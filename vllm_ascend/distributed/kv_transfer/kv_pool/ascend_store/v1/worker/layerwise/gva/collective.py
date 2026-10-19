"""The GVA Worker's isolated CPU channel for serial background Store tasks.

Never reuse the model's process group: background Store and foreground model
collectives can otherwise enter the same group in different rank orders.
The resource owner closes this channel only after its executors have drained.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch.distributed as dist

STORE_COLLECTIVE_TIMEOUT_SECONDS = 60


class StoreCollective:
    def __init__(self, ranks: tuple[int, ...]) -> None:
        self._size = len(ranks)
        self._group = dist.new_group(
            ranks=list(ranks),
            backend="gloo",
            timeout=timedelta(seconds=STORE_COLLECTIVE_TIMEOUT_SECONDS),
            use_local_synchronization=True,
        )
        self._closed = False

    def all_gather(self, value: Any) -> tuple[Any, ...]:
        if self._closed:
            raise RuntimeError("Store collective is closed")
        results: list[Any] = [None] * self._size
        dist.all_gather_object(results, value, group=self._group)
        return tuple(results)

    def close(self) -> None:
        if not self._closed:
            dist.destroy_process_group(self._group)
            self._closed = True
