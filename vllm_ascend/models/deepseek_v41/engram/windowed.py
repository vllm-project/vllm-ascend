# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run original WKV projections in router/dispatch windows."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch

from .embedding import AscendParallelEngramEmbedding
from .layer import AscendEngram


def shard_engram_rows(rows: torch.Tensor, *, rank: int, size: int) -> torch.Tensor:
    """Freeze the model's TP rank outside a MoE parallel context."""
    padding = (-rows.shape[0]) % size
    if padding:
        rows = torch.cat((rows, rows.new_zeros((padding, rows.shape[1]))))
    count = rows.shape[0] // size
    result = rows[rank * count : (rank + 1) * count]
    return result if padding else result.clone()


@contextmanager
def router_projection_window(experts, callback):
    previous = getattr(experts, "_engram_wkv_window", None)
    experts._engram_wkv_window = callback
    try:
        yield
    finally:
        experts._engram_wkv_window = previous


class EngramProjectionSchedule:
    """Per-forward state; the model owns persistent buffers, events and stream.

    Router0 can prepare Engram1; the previous Engram layer's router prepares
    the next one. Each job calls the original projection without weight
    slicing. Replay uses captured dependencies and fixed addresses.
    """

    def __init__(
        self,
        layers: dict[int, AscendEngram],
        rows: dict[int, torch.Tensor],
        outputs: dict[int, torch.Tensor],
        stream,
        events: dict,
        *,
        tokens: int,
        shard: Callable[[torch.Tensor], torch.Tensor],
        lookup_ready: dict | None = None,
        external: bool = False,
    ):
        self.layers, self.rows, self.outputs = layers, rows, outputs
        self.stream, self.events = stream, events
        self.tokens, self.shard = tokens, shard
        self.lookup_ready, self.external = lookup_ready or {}, external
        consumers = sorted(layers)
        self.triggers = {
            (target - 1 if i == 0 else consumers[i - 1]): target for i, target in enumerate(consumers) if target > 0
        }
        self.started: set[int] = set()
        self.pending = False

    def submit_router(self, layer_id: int) -> None:
        target = self.triggers.get(layer_id)
        if target is None or target in self.started:
            return
        self.pending = True  # Join even if the first submission raises.
        event = self.events[target]
        event.record(torch.npu.current_stream())
        self.stream.wait_event(event)
        with torch.npu.stream(self.stream):
            ready = self.lookup_ready.get(target)
            if ready is not None:
                AscendParallelEngramEmbedding.wait_lookup(ready, external=self.external)
            local_rows = self.shard(self.rows[target])[: self.tokens]
            local_rows.record_stream(self.stream)
            output = self.outputs[target][: self.tokens]
            output.record_stream(self.stream)
            output.copy_(self.layers[target].project(local_rows))
        self.started.add(target)

    def ready_for(self, layer_id: int) -> bool:
        return layer_id in self.started

    def finish(self, layer_id: int) -> torch.Tensor:
        if not self.ready_for(layer_id):
            raise RuntimeError("Engram projection was not submitted")
        self.close()
        return self.outputs[layer_id][: self.tokens]

    def close(self) -> None:
        if self.pending:
            torch.npu.current_stream().wait_stream(self.stream)
            self.pending = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
