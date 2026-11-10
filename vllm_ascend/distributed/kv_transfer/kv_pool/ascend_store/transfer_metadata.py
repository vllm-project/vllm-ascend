# SPDX-License-Identifier: Apache-2.0
"""Per-request arrays and worker-prepared cache transfer state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.request_state import RequestTracker


@dataclass
class RequestTransferState:
    block_ids_np: np.ndarray | None = None
    block_ids_by_group_np: list[np.ndarray] | None = None
    block_gvas_np: np.ndarray | None = None
    block_gvas_by_group_np: list[np.ndarray] | None = None
    gva_block_offset: int = 0
    load_block_gvas_np: np.ndarray | None = None
    load_block_gvas_by_group_np: list[np.ndarray] | None = None
    load_gva_block_offset: int = 0
    partial_save_gva_per_group: list[int] = field(default_factory=list)
    partial_load_gva_per_group: list[int] = field(default_factory=list)
    save_keys: list[str] | None = None
    load_keys: list[str] = field(default_factory=list)
    save_block_keys: list[str | None] = field(default_factory=list)
    save_key_block_offset: int = 0
    save_last_block_key: str | None = None
    load_block_keys: list[str | None] = field(default_factory=list)
    load_key_block_offset: int = 0
    load_last_block_key: str | None = None
    store_masks: tuple[Sequence[bool] | None, ...] | None = None
    load_masks: tuple[Sequence[bool] | None, ...] | None = None

    @classmethod
    def from_legacy_fields(cls, **fields: Any) -> RequestTransferState:
        # Normalize only supplied legacy arguments; the normal tracker path
        # needs neither extra empty-list allocations nor key-list copies.
        for name in ("save_block_keys", "load_block_keys", "load_keys"):
            if name in fields:
                fields[name] = [] if fields[name] is None else list(fields[name])
        for name in ("partial_save_gva_per_group", "partial_load_gva_per_group"):
            if name in fields:
                fields[name] = fields[name] or []
        return cls(**fields)

    @classmethod
    def from_tracker(cls, tracker: RequestTracker) -> RequestTransferState:
        grouped_ids = [np.asarray(ids, dtype=np.int64) for ids in tracker.allocated_block_ids_by_group]
        return cls(
            block_ids_np=grouped_ids[0] if grouped_ids else np.empty(0, dtype=np.int64),
            block_ids_by_group_np=grouped_ids or None,
            block_gvas_np=np.asarray(tracker.block_gvas, dtype=np.int64),
            block_gvas_by_group_np=[np.asarray(gvas, dtype=np.int64) for gvas in tracker.block_gvas_by_group]
            if tracker.block_gvas_by_group
            else None,
            gva_block_offset=tracker.gva_block_offset,
        )

    def get_block_ids_array(self, group_id: int) -> np.ndarray | None:
        if self.block_ids_by_group_np is not None and group_id < len(self.block_ids_by_group_np):
            return self.block_ids_by_group_np[group_id]
        return self.block_ids_np

    def get_transfer_arrays(self, group_id: int, is_save: bool = True) -> tuple[np.ndarray | None, np.ndarray | None]:
        grouped_ids = self.block_ids_by_group_np
        grouped_gvas = self.block_gvas_by_group_np if is_save else self.load_block_gvas_by_group_np
        if (
            grouped_ids is not None
            and grouped_gvas is not None
            and group_id < len(grouped_ids)
            and group_id < len(grouped_gvas)
        ):
            return grouped_ids[group_id], grouped_gvas[group_id]
        return self.block_ids_np, self.block_gvas_np if is_save else self.load_block_gvas_np
