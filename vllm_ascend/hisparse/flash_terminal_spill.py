# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Terminal spill compatibility for the pinned Flash HiSparse coordinator."""

from dataclasses import fields
from typing import cast

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.hisparse.coordinator import (
    HiSparseCoordinator,
    _HiSparseRequestState,
    _PendingPublication,
)


class _FlashHiSparseCoordinator(HiSparseCoordinator):
    _terminal_spill_copies: dict[int, dict[int, tuple[KVCacheBlock, ...]]]

    def has_pending_reclamation(self) -> bool:
        return bool(self._terminal_spill_copies) or super().has_pending_reclamation()

    def _apply_enqueued_spills(self) -> None:
        # Preserve the original transfer reference for selected detached pages.
        # Other transfers release their references in the stock pin's order.
        unpinned: list[KVCacheBlock] = []
        for pending in self.pending_spills.values():
            if (
                pending.enqueue_applied
                or not pending.expected_worker_completions
                or pending.worker_completions < pending.expected_worker_completions
            ):
                continue
            publication = pending.request_state.publication
            page_idx = pending.page[1]
            if (
                not pending.restore
                and publication is not None
                and publication.detached_blocks is not None
                and publication.num_cached_blocks <= page_idx < len(publication.detached_blocks)
            ):
                copies = self._terminal_spill_copies.setdefault(id(publication), {})
                copies[page_idx] = pending.resident_blocks
            else:
                unpinned.extend(pending.resident_blocks)
            pending.enqueue_applied = True
        if unpinned:
            self.gpu_pool.free_blocks(reversed(unpinned))

    def _publish_detached_blocks(self, state: _HiSparseRequestState) -> None:
        publication = state.publication
        super()._publish_detached_blocks(state)
        if publication is None or state.publication is not None:
            return
        copies = self._terminal_spill_copies.pop(id(publication), None)
        if not copies:
            return
        blocks = publication.detached_blocks
        assert blocks is not None
        num_full_blocks = min(state.ready_prefix_pages, publication.num_pages)
        # Host publication and device bookkeeping run within one scheduler call.
        for page_idx in sorted(copies, reverse=True):
            resident_blocks = copies[page_idx]
            if page_idx < num_full_blocks:
                self._store_copy(blocks[page_idx], resident_blocks)
                self.gpu_pool.unpin_blocks(resident_blocks, self._on_block_reused)
            else:
                self.gpu_pool.free_blocks(resident_blocks)


def install_flash_terminal_spill_compat(coordinator: HiSparseCoordinator) -> None:
    """Extend one Flash coordinator before its first admission."""
    if isinstance(coordinator, _FlashHiSparseCoordinator) or any(
        field.name == "detached_copies" for field in fields(_PendingPublication)
    ):
        return
    if type(coordinator) is not HiSparseCoordinator:
        raise RuntimeError("Unexpected Core coordinator for Flash terminal spill compatibility.")
    if coordinator.request_states or coordinator.pending_spills:
        raise RuntimeError("Install Flash terminal spill compatibility before admission.")
    # All cache managers retain the same coordinator instance.
    coordinator.__class__ = _FlashHiSparseCoordinator
    cast(_FlashHiSparseCoordinator, coordinator)._terminal_spill_copies = {}
