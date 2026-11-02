# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager all-resident sparse remapping; host/LRU paths keep their normal owner."""

from __future__ import annotations

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.hisparse_resident_state import ResidentForwardState, eligible_resident_forward
from vllm_ascend.ops.hisparse_layout import packed_cache_storage


@triton.jit
def resident_index_kernel(
    logical,
    requests,
    states,
    table,
    num_valid_rows,
    physical,
    counts,
    TOP_K: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    NUM_STATE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ATTENTION_STRIDE: tl.constexpr,
    PHYSICAL_CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK_K)
    request = tl.load(requests + row.to(tl.int64) * REQUEST_STRIDE)
    active = (row < tl.load(num_valid_rows)) & (request >= 0) & (request < NUM_REQUESTS)
    state = tl.load(states + request.to(tl.int64) * STATE_STRIDE, mask=active, other=-1)
    active &= (state >= 0) & (state < NUM_STATE_ROWS)
    token = tl.load(logical + row.to(tl.int64) * INPUT_STRIDE + i, mask=(i < TOP_K) & active, other=-1)
    column = token // BLOCK_SIZE
    valid = active & (i < TOP_K) & (token >= 0) & (column >= 0) & (column < TABLE_COLS)
    block = tl.load(
        table + request.to(tl.int64) * TABLE_STRIDE + column.to(tl.int64) * TABLE_COL_STRIDE,
        mask=valid,
        other=0,
    )
    encoded = block.to(tl.int64) * ATTENTION_STRIDE + token % BLOCK_SIZE
    valid &= (block > 0) & (block < NUM_BLOCKS) & (encoded >= 0) & (encoded < PHYSICAL_CAPACITY)
    tl.store(physical + row.to(tl.int64) * OUTPUT_STRIDE + i, tl.where(valid, encoded, -1).to(tl.int32), mask=i < TOP_K)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), 0))


def metadata_signature(metadata):
    if metadata is None:
        return None
    return (
        metadata.num_actual_tokens,
        metadata.num_decode_tokens,
        metadata.num_input_tokens,
        metadata.num_reqs,
        metadata.block_size,
        metadata.req_id_per_token.data_ptr(),
        metadata.req_id_per_token.numel(),
        metadata.query_start_loc.data_ptr(),
        metadata.query_start_loc.numel(),
    )


def begin_resident_forward(group, layer_index, metadata, *, enforce_eager):
    if not enforce_eager:
        return
    state = getattr(group, "_ascend_resident_forward", None)
    if state is None:
        if layer_index != 0:
            raise RuntimeError("Resident follower has no group leader state")
        state = ResidentForwardState()
        group._ascend_resident_forward = state
    state.begin(layer_index, metadata_signature(metadata))


def group_mapping_contract(group):
    leader = group.cache(0)
    view = leader.view
    if view is None or leader.block_table is None:
        raise RuntimeError("Resident leader is not bound")
    source_index = leader.runtime.resident_source_index
    capacity = None
    for cache in group.caches:
        if cache.view is None or cache.block_table is None:
            raise RuntimeError("Resident follower is not bound")
        if (
            cache.view.cache.data_ptr() != cache.runtime.hot.cache.data_ptr()
            or cache.view.cache.stride() != cache.runtime.hot.cache.stride()
        ):
            raise RuntimeError("Resident and hot aliases differ within a layer")
        if (
            cache.runtime.resident_source_index != source_index
            or cache.block_table.data_ptr() != leader.block_table.data_ptr()
            or cache.block_table.shape != leader.block_table.shape
            or cache.block_table.stride() != leader.block_table.stride()
            or cache.view.block_size != view.block_size
            or cache.view.attention_block_stride != view.attention_block_stride
            or cache.view.cache.shape != view.cache.shape
        ):
            raise RuntimeError("Index-group resident tables or physical strides differ")
        _, _, _, layer_capacity = packed_cache_storage(cache.runtime.hot.attention_cache)
        capacity = layer_capacity if capacity is None else min(capacity, layer_capacity)
    last_row = (view.cache.shape[0] - 1) * view.attention_block_stride + view.block_size - 1
    if capacity is None or last_row >= capacity or last_row >= torch.iinfo(torch.int32).max:
        raise RuntimeError("Resident physical mapping exceeds the actual attention cache")
    return leader, capacity


def resident_topk_or_none(group, layer_index, metadata, logical, *, enforce_eager):
    if not enforce_eager:
        return None
    state = group._ascend_resident_forward
    backend = group.cache(0).runtime.backend
    eligible = eligible_resident_forward(
        enforce_eager,
        get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
        and not backend.is_current_stream_capturing(),
        metadata is not None,
        group.cache(0).dummy_batch,
        [cache.all_context_pages_resident for cache in group.caches],
    )
    if not state.choose(layer_index, metadata_signature(metadata), eligible):
        return None
    rows, topk = logical.shape
    if logical.dtype != torch.int32 or logical.stride(1) != 1:
        raise ValueError("Resident logical indices require contiguous int32 columns")
    if rows > group.physical_topk_indices.shape[0] or topk != group.physical_topk_indices.shape[1]:
        raise ValueError("Resident mapping exceeds the group's profiled buffer")
    compute = backend.current_stream()
    if layer_index == 0:
        leader, capacity = group_mapping_contract(group)
        # The table index_select is enqueued after this wait on the same stream
        # as the remap. An event recorded before index_select would not order it.
        group.side_stream.wait_stream(compute)
        with backend.stream_context(group.side_stream, generic=True):
            table = leader.batch_block_table()
            physical = group.physical_topk_indices[:rows]
            counts = group.valid_topk_counts[:rows]
            requests = metadata.req_id_per_token[:rows]
            states = leader.runtime.request_state_indices
            if states is None:
                raise RuntimeError("Missing current request-state indirection")
            if requests.numel() != rows or states.numel() < metadata.num_reqs or table.shape[0] < metadata.num_reqs:
                raise ValueError("Resident metadata does not cover every query/request")
            if any(t.device != logical.device for t in (requests, states, table, physical, counts)):
                raise ValueError("Resident remapping tensors must share the query device")
            if physical.dtype != torch.int32 or counts.dtype != torch.int32 or table.dtype != torch.int32:
                raise ValueError("Resident table/output/count buffers must be int32")
            resident_index_kernel[(rows,)](
                logical,
                requests,
                states,
                table,
                metadata.query_start_loc[-1:],
                physical,
                counts,
                TOP_K=topk,
                INPUT_STRIDE=logical.stride(0),
                REQUEST_STRIDE=requests.stride(0),
                STATE_STRIDE=states.stride(0),
                TABLE_STRIDE=table.stride(0),
                TABLE_COL_STRIDE=table.stride(1),
                OUTPUT_STRIDE=physical.stride(0),
                NUM_REQUESTS=metadata.num_reqs,
                NUM_STATE_ROWS=leader.block_table.shape[0],
                TABLE_COLS=table.shape[1],
                NUM_BLOCKS=leader.view.cache.shape[0],
                BLOCK_SIZE=leader.view.block_size,
                ATTENTION_STRIDE=leader.view.attention_block_stride,
                PHYSICAL_CAPACITY=capacity,
                BLOCK_K=triton.next_power_of_2(topk),
                multibuffer=False,
            )
            group.physical_topk_ready.record(group.side_stream)
        state.publish(layer_index, physical)
    compute.wait_event(group.physical_topk_ready)
    physical = state.consume()
    if physical.shape != logical.shape:
        raise RuntimeError("Resident follower query shape differs from its leader")
    return physical
