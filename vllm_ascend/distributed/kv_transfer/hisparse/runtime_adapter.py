# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend-owned HiSparse runtime operations for the frozen PR62592 API.

Bind resolver methods to individual NPU runtimes. The two remaining common
call sites have version-checked, NPU-only wrappers; CUDA calls use originals.
Remove the wrappers when equivalent upstream backend hooks are available.
No scheduler, allocation policy, model-runner timing or global Torch changes.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import in_piecewise_cudagraph
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_ascend.patch.worker.patch_hisparse_streams import create_event, native_current_stream

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime

COMMON_RUNTIME_SHA256 = "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926"
COMMON_WORKER_SHA256 = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"


def _ensure_gpu_sources(
    self,
    resident_state_rows: torch.Tensor,
    state_indices: torch.Tensor,
    resident_block_size: int,
) -> None:
    """Resolve which staged rows can be served from the resident cache.

    ``resident_state_rows`` is a resident group's persistent table by
    request state row and ``state_indices`` holds each staged request's
    state row. Computed once per plan and resident group (every layer in
    the group shares the table); non-null resident pages become
    miss_mask=0 rows gathered device-to-device by ``gather_prefill_cache``.
    """
    source_key = (resident_state_rows.data_ptr(), resident_block_size)
    if self.gpu_source_key == source_key:
        return
    block_size = self.block_size
    if resident_block_size <= 0 or block_size % resident_block_size != 0:
        return
    device = self.row_ids.device
    num_unique = self.row_ids.shape[1] // block_size
    host_ids = self.row_ids[0].view(num_unique, block_size)[:, 0] // block_size
    new_bt = self.block_table.to(torch.int64)
    num_rows, num_cols = new_bt.shape
    if num_rows == 0 or state_indices.shape[0] < num_rows:
        return
    resident_block_table = resident_state_rows.index_select(0, state_indices[:num_rows].clamp(min=0))
    # One representative (row, col) per unique host block: any request
    # referencing the block holds an equivalent (refcounted) resident view.
    flat_pos = torch.arange(num_rows * num_cols, device=device)
    rep = torch.full((num_unique,), num_rows * num_cols, device=device, dtype=torch.int64)
    rep.scatter_reduce_(0, new_bt.reshape(-1), flat_pos, reduce="amin", include_self=True)
    rep_row = (rep // num_cols).clamp(max=resident_block_table.shape[0] - 1)
    rep_col = rep % num_cols
    pages_per_block = block_size // resident_block_size
    res_pos = rep_col[:, None] * pages_per_block + torch.arange(pages_per_block, device=device)
    res_cols = resident_block_table.shape[1]
    res_blocks = torch.where(
        res_pos < res_cols,
        torch.gather(
            resident_block_table.to(torch.int64)[rep_row],
            1,
            res_pos.clamp(max=max(res_cols - 1, 0)),
        ),
        torch.zeros_like(res_pos),
    )
    offsets = torch.arange(block_size, device=device)
    per_off_block = res_blocks.index_select(1, offsets // resident_block_size)
    gpu_rows = per_off_block * resident_block_size + (offsets % resident_block_size)[None, :]
    # Block id 0 is the null block in both id spaces.
    valid_rows = self.row_ids.view(num_unique, block_size) >= 0
    hit = (per_off_block > 0) & (host_ids[:, None] > 0) & valid_rows
    self.gpu_row_ids = torch.where(hit, gpu_rows, -1).reshape(1, -1).to(torch.int32)
    self.miss_mask = ((self.gpu_row_ids < 0) & valid_rows.view(1, -1)).int()
    self.gpu_source_key = source_key


def _resolve_residency(
    self,
    *,
    resident: HiSparseCacheHandle | None = None,
    req_id_per_token: torch.Tensor,
    num_valid_rows: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    block_size: int,
    return_valid_counts: bool = False,
    shared_rows: slice,
    attention_indices_out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
) -> None:
    """Resolve logical top-k positions and compact rows requiring swaps."""
    num_tokens = topk_indices.shape[0]
    hot = self.hot
    host_cache = self.host_cache
    assert hot.block_size == block_size
    request_state_indices = self.request_state_indices
    assert request_state_indices is not None
    group = self.index_group
    shared = group.shared_topk

    device_topk_rows = shared.device_topk_rows[shared_rows]
    valid_topk_counts = None
    if return_valid_counts:
        valid_topk_counts = shared.valid_topk_counts[shared_rows] if valid_counts_out is None else valid_counts_out
    physical_topk_indices = (
        shared.physical_topk_indices[shared_rows] if attention_indices_out is None else attention_indices_out
    )
    swap_host_physical_rows = shared.swap_host_physical_rows[shared_rows]
    swap_device_physical_rows = shared.swap_device_physical_rows[shared_rows]
    swap_counts = shared.swap_counts[shared_rows]

    # Rows past num_valid_rows and rows without request state come out as
    # -1 so the attention kernel masks them.
    self.cache_ops.hisparse_resolve_residency(
        host_cache,
        hot.cache,
        self.hot_block_table,
        topk_indices,
        device_topk_rows,
        group.device_global_indices,
        group.lru_slots,
        request_state_indices,
        self.region_stride,
        self.max_union_rows,
        None,
        group.swap_stats,
        physical_topk_indices,
        hot.attention_block_stride,
        req_id_per_token[:num_tokens].contiguous(),
        block_table,
        block_size,
        None,
        valid_topk_counts,
        swap_host_physical_rows,
        swap_device_physical_rows,
        swap_counts,
        resident.block_table if resident is not None else None,
        resident.view.block_size if resident is not None and resident.view is not None else 0,
        0,
        num_valid_rows,
    )


def _resolve_and_stage_group(
    self,
    *,
    resident: HiSparseCacheHandle,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    logical_topk_indices: torch.Tensor,
    block_size: int,
    return_valid_counts: bool,
    shared_rows: slice,
    attention_indices_out: torch.Tensor | None,
    valid_counts_out: torch.Tensor | None,
    num_valid_rows: torch.Tensor,
) -> None:
    group = self.index_group
    compute_stream = native_current_stream()
    if group.logical_topk_ready is not None and not in_piecewise_cudagraph():
        group.copy_stream.wait_event(group.logical_topk_ready)
    else:
        group.copy_stream.wait_stream(compute_stream)
    with group.copy_stream:
        self._resolve_residency(
            resident=resident,
            req_id_per_token=req_id_per_token,
            num_valid_rows=num_valid_rows,
            block_table=block_table,
            topk_indices=logical_topk_indices,
            block_size=block_size,
            return_valid_counts=return_valid_counts,
            shared_rows=shared_rows,
            attention_indices_out=attention_indices_out,
            valid_counts_out=valid_counts_out,
        )
        runtimes = [self]
        # Prefetch historical missing rows for pure MTP verification too.
        # Keep Resident writes, Host policy and each layer's ready event intact.
        if resident.decode_batch or (
            0 < resident.num_actual_tokens <= self.max_swap_rows
            and resident.num_decode_tokens == resident.num_actual_tokens
        ):
            runtimes.extend(group.followers)
        for runtime in runtimes:
            runtime._swap_rows(shared_rows)
            if runtime._layer_ready_event is None:
                runtime._layer_ready_event = create_event()
            runtime._layer_ready_event.record(group.copy_stream)
            runtime._swap_staged = True


def _enqueue_host_mirror(
    self,
    ready_event: torch.Event | None = None,
) -> None:
    active = [(index, handle) for index, handle in enumerate(self.cache_handles) if handle.num_actual_tokens != 0]
    if not active:
        return

    def mirror_key(handle: HiSparseCacheHandle) -> tuple:
        slots = handle.mirror_slot_mapping
        return (
            handle.num_actual_tokens,
            handle.num_decode_tokens,
            handle.decode_batch,
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
    num_rows = min(cache.num_actual_tokens, dst_slots.numel())
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
    assert cache.req_id_per_token is not None
    self._ascend_slot_invalidator(
        [handle.runtime for _, handle in active if handle.runtime.is_group_leader],
        dst_slots[:num_rows],
        cache.req_id_per_token[:num_rows],
    )


class AscendHiSparseSlotInvalidator:
    """Drop hot copies of rewritten host slots for many index groups at once."""

    def __init__(self, request_state_indices: torch.Tensor) -> None:
        self.request_state_indices = request_state_indices
        self._group_tables: dict[tuple[int, ...], torch.Tensor] = {}

    def __call__(
        self,
        leaders: list[HiSparseRuntime],
        written_slots: torch.Tensor,
        req_id_per_token: torch.Tensor,
    ) -> None:
        num_tokens = min(written_slots.numel(), req_id_per_token.numel())
        if num_tokens == 0 or not leaders:
            return
        tables = [leader.index_group.device_global_indices for leader in leaders]
        key = tuple(table.data_ptr() for table in tables)
        group_table = self._group_tables.get(key)
        if group_table is None:
            if any(table.shape != tables[0].shape for table in tables):
                raise ValueError("HiSparse index groups must share one hot layout.")
            group_table = async_tensor_h2d(list(key), self.request_state_indices.device, torch.int64)
            self._group_tables[key] = group_table
        num_state_rows, region_stride = tables[0].shape
        leaders[0].cache_ops.hisparse_invalidate_written_slots(
            group_table,
            self.request_state_indices,
            req_id_per_token[:num_tokens],
            written_slots[:num_tokens],
            num_state_rows,
            region_stride,
        )


def _verify_source(obj, digest):
    path = Path(inspect.getsourcefile(obj))
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
        raise RuntimeError("Unreviewed HiSparse call site; Ascend adapter not installed")


def _invalidate_written_slots(self, written_slots, req_id_per_token):
    owner = getattr(self, "_ascend_single_invalidator", None)
    if owner is None:
        owner = AscendHiSparseSlotInvalidator(self.request_state_indices)
        self._ascend_single_invalidator = owner
    owner([self], written_slots, req_id_per_token)


def install_call_sites():
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
    from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan

    original = HiSparsePrefillStagingPlan.ensure_gpu_sources
    if not getattr(original, "_ascend_backend_adapter", False):
        _verify_source(original, COMMON_RUNTIME_SHA256)

        # Freeze the captured function; subsequent wrappers use another local.
        ensure_original = original

        @wraps(ensure_original)
        def ensure(plan, resident_state_rows, state_indices, resident_block_size):
            if resident_state_rows.device.type != "npu":
                return ensure_original(plan, resident_state_rows, state_indices, resident_block_size)
            return _ensure_gpu_sources(plan, resident_state_rows, state_indices, resident_block_size)

        ensure._ascend_backend_adapter = True
        ensure._ascend_original = ensure_original
        HiSparsePrefillStagingPlan.ensure_gpu_sources = ensure

    original_mirror = HiSparseConnectorWorker._enqueue_host_mirror
    if not getattr(original_mirror, "_ascend_backend_adapter", False):
        _verify_source(original_mirror, COMMON_WORKER_SHA256)

        @wraps(original_mirror)
        def mirror(worker, ready_event=None):
            handles = worker.cache_handles
            if not handles or handles[0].runtime.device.type != "npu":
                return original_mirror(worker, ready_event=ready_event)
            invalidator = getattr(worker, "_ascend_slot_invalidator", None)
            if invalidator is None:
                if any(not getattr(h.runtime, "_ascend_runtime_backend_bound", False) for h in handles):
                    raise RuntimeError("All HiSparse NPU runtimes must use the same backend adapter")
                invalidator = AscendHiSparseSlotInvalidator(worker.request_state_indices)
                worker._ascend_slot_invalidator = invalidator
            elif invalidator.request_state_indices is not worker.request_state_indices:
                raise RuntimeError("Recreate the worker after changing request-state storage")
            return _enqueue_host_mirror(worker, ready_event=ready_event)

        mirror._ascend_backend_adapter = True
        mirror._ascend_original = original_mirror
        HiSparseConnectorWorker._enqueue_host_mirror = mirror


def bind_runtime(runtime):
    if runtime.device.type != "npu":
        raise ValueError("Ascend HiSparse adapter requires an NPU runtime")
    if getattr(runtime, "_ascend_runtime_backend_bound", False):
        return
    _verify_source(type(runtime)._resolve_residency, COMMON_RUNTIME_SHA256)
    install_call_sites()
    runtime._resolve_residency = MethodType(_resolve_residency, runtime)
    runtime._resolve_and_stage_group = MethodType(_resolve_and_stage_group, runtime)
    runtime.invalidate_written_slots = MethodType(_invalidate_written_slots, runtime)
    runtime._ascend_runtime_backend_bound = True
