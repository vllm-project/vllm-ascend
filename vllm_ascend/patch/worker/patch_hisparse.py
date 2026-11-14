# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bridge Core's local-query cache handles to PCP's gathered KV write layout.

Installed only for a direct HiSparse + PCP MRV2 connector. The wrappers live
for the worker process; non-PCP metadata retains Core's existing behavior.
Keep scheduling, DMA submission and completion fences owned by Core.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle


def write_target(self, num_input_rows: int, num_slot_rows: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    assert self.view is not None and self.slot_mapping is not None
    num_rows = min(num_input_rows, num_slot_rows, self.num_write_tokens)
    # Gathered decode includes rank padding masked by negative slots.
    if self.decode_batch and not self.write_has_prefill and self.num_write_tokens == self.num_actual_tokens:
        num_rows = min(num_rows, self.runtime.max_num_reqs)
    return self.view.cache, self.slot_mapping[:num_rows], num_rows


def finish_kv_update(self) -> None:
    # HiSparse is optional on supported Core releases; load it only on use.
    from vllm.v1.hisparse import runtime as runtime_module

    if self.dummy_batch:
        return
    if (
        self.submit_layer_mirror is not None
        and self.write_has_prefill
        and runtime_module.get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
    ):
        self.submit_layer_mirror()


def _enqueue_host_mirror(
    self,
    ready_event: torch.Event | None = None,
) -> None:
    active = [(index, handle) for index, handle in enumerate(self.cache_handles) if handle.num_write_tokens != 0]
    if not active:
        return

    def mirror_key(handle: HiSparseCacheHandle) -> tuple:
        slots = handle.mirror_slot_mapping
        return (
            handle.num_write_tokens,
            handle.write_has_prefill,
            handle.host_mirror_required,
            handle.runtime.eager_host_mirror,
            None if slots is None else slots.data_ptr(),
        )

    keys = {mirror_key(handle) for _, handle in active}
    if len(keys) > 1:
        raise RuntimeError(f"HiSparse cache layers disagree on mirror metadata: {list(keys)}.")
    cache = active[0][1]
    if not cache.host_mirror_required:
        return
    dst_slots = cache.mirror_slot_mapping
    if dst_slots is None:
        raise RuntimeError("HiSparse host mirror has no source slot mapping.")
    num_rows = min(cache.num_write_tokens, dst_slots.numel())
    if num_rows == 0:
        return
    if self.is_host_writer:
        # The drafter has not written its layers' rows yet; the next
        # start_step mirrors them after it runs.
        self._draft_mirror_pending = bool(self._draft_layers)
        expected_layers = {index for index, _ in active}.difference(self._draft_layers)
        if self._per_layer_mirrored and self._per_layer_mirrored != expected_layers:
            raise RuntimeError(
                "HiSparse per-layer DMA did not mirror every active layer: "
                f"expected {sorted(expected_layers)}, got "
                f"{sorted(self._per_layer_mirrored)}."
            )
        pending_layers = tuple(sorted(expected_layers - self._submitted_mirror_layers))
        if pending_layers:
            self._enqueue_row_dma(pending_layers, ready_event=ready_event)
            self._submitted_mirror_layers.update(pending_layers)
    assert cache.write_req_id_per_token is not None
    self._invalidate_written_slots(
        [handle.runtime for _, handle in active if handle.runtime.is_group_leader],
        dst_slots[:num_rows],
        cache.write_req_id_per_token[:num_rows],
    )


def install_pcp_cache_layout() -> None:
    """Install once, before preparing or capturing the first PCP forward."""
    # Worker registration must also import on Core releases without HiSparse.
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle

    if getattr(HiSparseCacheHandle, "_ascend_pcp_layout_installed", False):
        return
    original_prepare = HiSparseCacheHandle._prepare_for_batch
    original_clear = HiSparseConnectorWorker._clear_forward_mirror_state

    @wraps(original_prepare)
    def prepare(self, attn_metadata: Any | None) -> None:
        # Preserve upstream decode/group-prefetch policy across Core revisions.
        original_prepare(self, attn_metadata)
        self.num_write_tokens = getattr(attn_metadata, "hisparse_num_write_tokens", self.num_actual_tokens)
        self.write_req_id_per_token = getattr(attn_metadata, "hisparse_write_req_id_per_token", self.req_id_per_token)
        self.write_has_prefill = getattr(attn_metadata, "hisparse_write_has_prefill", not self.decode_batch)
        query_states = getattr(attn_metadata, "hisparse_request_state_indices", None)
        if query_states is not None:
            self.runtime.request_state_indices = query_states
        self.host_mirror_required = attn_metadata is not None and (
            self.write_has_prefill or self.runtime.eager_host_mirror
        )

    @wraps(original_clear)
    def clear(self) -> None:
        original_clear(self)
        for cache in self.cache_handles:
            cache.num_write_tokens = 0
            cache.write_req_id_per_token = None
            cache.write_has_prefill = False

    HiSparseCacheHandle._prepare_for_batch = prepare
    HiSparseCacheHandle.write_target = write_target
    HiSparseCacheHandle.finish_kv_update = finish_kv_update
    HiSparseConnectorWorker._clear_forward_mirror_state = clear
    HiSparseConnectorWorker._enqueue_host_mirror = _enqueue_host_mirror
    HiSparseCacheHandle._ascend_pcp_layout_installed = True


def prepare_pcp_mirror_staging(worker: HiSparseConnectorWorker) -> None:
    """Size pinned Host staging for gathered rows before graph capture."""
    config = worker.vllm_config
    capacity = (
        config.scheduler_config.max_num_batched_tokens * config.parallel_config.prefill_context_parallel_size
        + config.scheduler_config.max_num_seqs * (config.num_lookahead_tokens + 1)
    )
    staging = worker._slot_mapping_staging
    if staging is not None and staging.slots.numel() < capacity:
        staging.slots = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
    for cache in worker.cache_handles:
        cache.num_write_tokens = 0
        cache.write_req_id_per_token = None
        cache.write_has_prefill = False
