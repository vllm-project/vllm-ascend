# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-address resident/selection workspaces for DSA Decode offload."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class DsaLayerWorkspace:
    """Fixed tensors for one non-MTP sparse layer."""

    layer_id: int
    resident_kv_cache: torch.Tensor
    resident_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_k_rope: torch.Tensor
    selection_block_table: torch.Tensor
    selection_query_lens: torch.Tensor | None = None
    selection_default_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.selection_block_table.dtype != torch.int32 or self.selection_block_table.ndim != 2:
            raise ValueError("selection_block_table must be a 2-D int32 tensor")
        rows = self.selection_block_table.shape[0]
        if self.selection_query_lens is not None:
            if self.selection_query_lens.dtype != torch.int32 or self.selection_query_lens.shape != (rows,):
                raise ValueError("selection_query_lens must be a row-sized int32 tensor")
        if self.selection_default_indices is not None:
            if self.selection_default_indices.dtype != torch.int32 or self.selection_default_indices.shape[0] != rows:
                raise ValueError("selection_default_indices must have one int32 row per query")
        if self.resident_kv_cache.shape[0] != self.resident_k_rope.shape[0]:
            raise ValueError("resident KV and RoPE must have the same row capacity")


class DsaResidentState:
    """Own fixed per-layer workspaces and the optional cross-step fence."""

    def __init__(self, workspaces: Sequence[DsaLayerWorkspace]):
        self._workspaces = {workspace.layer_id: workspace for workspace in workspaces}
        if len(self._workspaces) != len(workspaces):
            raise ValueError("Duplicate DSA layer workspace")
        self._last_install_events: list[object] = []
        self._lock = threading.RLock()

    @property
    def workspaces(self) -> Mapping[int, DsaLayerWorkspace]:
        return self._workspaces

    def workspace(self, layer_id: int) -> DsaLayerWorkspace:
        try:
            return self._workspaces[layer_id]
        except KeyError as error:
            raise KeyError(f"Unknown non-MTP DSA layer {layer_id}") from error

    def wait_previous_install(self) -> None:
        """Wait for step-complete events before resident metadata reuse."""

        events = tuple(self._last_install_events)
        for event in events:
            event.wait()
        # A successful wait consumes the events. prepare_step can be called
        # again by the eager Plan path without emitting a duplicate wait.
        with self._lock:
            del self._last_install_events[: len(events)]

    def record_final_install_event(self, event: object | None) -> None:
        """Record a model-step completion event after all DSA Install ops."""

        if event is not None:
            self._last_install_events.append(event)

    def clear(self) -> None:
        with self._lock:
            self._last_install_events.clear()
