# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree graph observations from host control flow, without device timing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class _GraphPhaseStats:
    captures: int = 0
    replays: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    workspaces: dict[Hashable, int] = field(default_factory=dict)


class TreeGraphStats:
    """One runner-owned counter set, updated only outside graph execution.

    Callers use a fixed phase name and one workspace key per retained gear.
    No per-step samples or tensor values are retained. Workspace bytes describe
    explicitly owned tensor storage, not allocator peaks or graph-pool usage.
    Capture/replay counts record successful submissions, not synchronized
    device completions. Performance timing belongs to an external profiler.
    """

    def __init__(self) -> None:
        self._phases: dict[str, _GraphPhaseStats] = defaultdict(_GraphPhaseStats)

    def record_lookup(self, phase: str, hit: bool) -> None:
        stats = self._phases[phase]
        if hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1

    def record_capture(self, phase: str) -> None:
        self._phases[phase].captures += 1

    def record_replay(self, phase: str) -> None:
        self._phases[phase].replays += 1

    def set_workspace_bytes(self, phase: str, key: Hashable, nbytes: int) -> None:
        """Replace one gear's retained-storage accounting using host metadata."""
        if nbytes < 0:
            raise ValueError("Workspace size cannot be negative")
        self._phases[phase].workspaces[key] = nbytes

    @staticmethod
    def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
        """Count allocated storage once for explicitly provided tensor aliases."""
        storage_bytes = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            storage_bytes[key] = storage.nbytes()
        return sum(storage_bytes.values())

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return host counters and aggregate retained bytes, without graph keys."""
        result = {}
        for phase, stats in self._phases.items():
            result[phase] = {
                "captures": stats.captures,
                "replays": stats.replays,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "workspace_gears": len(stats.workspaces),
                "workspace_bytes": sum(stats.workspaces.values()),
            }
        return result
