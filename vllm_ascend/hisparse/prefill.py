# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse Core's prefill plan with NPU-safe column selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.overrides import TorchFunctionMode
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan
from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan as build_core_prefill_plan

from vllm_ascend.hisparse.cache_ops import gather_valid_resident_rows

_PREFILL_LOOKAHEAD = "ascend_hisparse_prefill_lookahead"


class _PrefillColumnSelection(TorchFunctionMode):
    """Keep a local workaround for the NPU singleton-column indexing defect.

    torch-npu 2.10.0.post7 / CANN 9.2.1 can interleave rows for ``x[:, ids]``
    when x has one column. index_select has the same semantics and returns
    correct rows. Remove this scope when the supported device stack is fixed.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.__getitem__:
            tensor, index = args
            if (
                tensor.device.type == "npu"
                and tensor.ndim == 2
                and isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], slice)
                and index[0] == slice(None)
                and isinstance(index[1], torch.Tensor)
                and index[1].ndim == 1
            ):
                return torch.index_select(tensor, 1, index[1])
        return func(*args, **(kwargs or {}))


class AscendHiSparsePrefillStagingPlan(HiSparsePrefillStagingPlan):
    def ensure_gpu_sources(self, resident_state_rows, state_indices, resident_block_size) -> None:
        # Core retains source-key caching, representative-page selection and
        # current Resident/Host masks. The override affects only this call.
        with _PrefillColumnSelection():
            super().ensure_gpu_sources(resident_state_rows, state_indices, resident_block_size)


@dataclass(eq=False)
class _StagingSlot:
    tensor: torch.Tensor
    available: torch.npu.Event
    in_use: bool = False

    def finish(self) -> None:
        # This event follows the last Attention read on the consumer stream.
        self.available = torch.npu.Event()
        self.available.record()
        self.in_use = False


@dataclass
class _PrefillLookahead:
    slots: list[_StagingSlot] = field(default_factory=list)
    pending: tuple | None = None


def _host_view(runtime, block_size):
    cache = runtime.host_cache
    return cache.view(-1, block_size, cache.shape[-1])


def _stage_next(state, cache, plan) -> None:
    runtime = cache.runtime
    host = _host_view(runtime, plan.block_size)
    rows = plan.row_ids.size(1)
    if rows == 0:
        return
    shape = (rows // plan.block_size, plan.block_size, host.size(2))
    slot = next(
        (
            slot
            for slot in state.slots
            if not slot.in_use
            and slot.tensor.shape == shape
            and slot.tensor.dtype == host.dtype
            and slot.tensor.device == plan.block_table.device
        ),
        None,
    )
    if slot is None and len(state.slots) == 2:
        return
    stream = runtime.index_group.copy_stream
    side = torch.npu.Stream(
        stream_id=stream.stream_id, device_index=stream.device_index, device_type=stream.device_type
    )
    # The scheduler's prior Host publication and this plan's device writes
    # precede this boundary on the compute stream.
    side.wait_stream(torch.npu.current_stream())
    if slot is not None:
        side.wait_event(slot.available)
        # A failed enqueue must not make the possibly-written slot reusable.
        slot.in_use = True
    ready = torch.npu.Event()
    with torch.npu.stream(side):
        for tensor in (plan.row_ids, plan.dst_rows, plan.miss_mask):
            tensor.record_stream(side)
        if slot is None:
            staged = runtime.gather_prefill_cache(host, plan, resident_cache=None)
        else:
            slot.tensor.record_stream(side)
            runtime.cache_ops.hisparse_gather_plan(
                host.view(-1, host.size(2)),
                slot.tensor,
                plan.row_ids,
                plan.dst_rows,
                plan.miss_mask,
                None,
                None,
                0,
            )
        ready.record(side)
    if slot is None:
        slot = _StagingSlot(staged, ready)
        state.slots.append(slot)
    else:
        slot.available = ready
        slot.in_use = False
    state.pending = (cache, plan, plan.gpu_row_ids, plan.miss_mask, slot)


def stage_prefill_rows(group, layer_index, metadata):
    """Pipeline historical Host misses for one same-indexer follower."""
    cache = group.cache(layer_index)
    plan = metadata.prefill.host_staging_plan
    host = _host_view(cache.runtime, plan.block_size)
    if cache.all_context_pages_resident or metadata.hisparse_for_capture:
        return (*group.stage_prefill_rows(layer_index, host, metadata), None)
    context = get_forward_context()
    if (
        context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or not isinstance(context.attn_metadata, dict)
        or torch.npu.is_current_stream_capturing()
    ):
        return (*group.stage_prefill_rows(layer_index, host, metadata), None)

    state = context.additional_kwargs.get(_PREFILL_LOOKAHEAD)
    pending = None
    if state is not None:
        pending, state.pending = state.pending, None
    slot = None
    if pending is not None:
        owner, owner_plan, gpu_rows, miss_mask, prefetched = pending
        if owner is cache and owner_plan is plan and gpu_rows is plan.gpu_row_ids and miss_mask is plan.miss_mask:
            slot = prefetched
    if slot is not None:
        slot.in_use = True
        main = torch.npu.current_stream()
        main.wait_event(slot.available)
        slot.tensor.record_stream(main)
        # The follower has now written its current KV. Patch its Resident
        # rows without overwriting the prefetched Host bytes.
        gather_valid_resident_rows(cache.view.cache, slot.tensor, plan.gpu_row_ids[0])
        staged, table = slot.tensor, plan.block_table
        req_ids = metadata.req_id_per_token[metadata.num_decode_tokens :]
        if metadata.num_decodes:
            req_ids = req_ids - metadata.num_decodes
    else:
        staged, table, req_ids = group.stage_prefill_rows(layer_index, host, metadata)

    next_index = layer_index + 1
    if next_index < len(group.caches) and plan.gpu_row_ids is not None and cache.view is not None:
        follower = group.cache(next_index)
        if (
            context.attn_metadata.get(getattr(follower, "_ascend_prefill_layer_name", None)) is metadata
            and follower.view is not None
            and follower.block_table is not None
            and cache.block_table is not None
            and follower.block_table.data_ptr() == cache.block_table.data_ptr()
            and follower.view.block_size == cache.view.block_size
            and plan.gpu_source_key == (cache.block_table.data_ptr(), cache.view.block_size)
            and cache.source_block_table is not None
            and follower.source_block_table is cache.source_block_table
            and follower.runtime.request_state_indices is cache.runtime.request_state_indices
            and follower.runtime.index_group is cache.runtime.index_group
        ):
            if state is None:
                state = _PrefillLookahead()
                context.additional_kwargs[_PREFILL_LOOKAHEAD] = state
            _stage_next(state, follower, plan)
    return staged, table, req_ids, slot


def build_hisparse_prefill_staging_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    staging_block_capacity: int,
) -> AscendHiSparsePrefillStagingPlan:
    plan = build_core_prefill_plan(block_table, seq_lens, block_size, staging_block_capacity)
    return AscendHiSparsePrefillStagingPlan(
        block_table=plan.block_table,
        row_ids=plan.row_ids,
        dst_rows=plan.dst_rows,
        miss_mask=plan.miss_mask,
        block_size=plan.block_size,
    )
