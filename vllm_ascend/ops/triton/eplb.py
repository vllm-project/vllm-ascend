# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.distributed.eplb.tensor_ops import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

_SMALL_ASSIGNMENT_LIMIT = 2048
_MEDIUM_ASSIGNMENT_LIMIT = 8192
_SMALL_TILE = 64
_MEDIUM_TILE = 128
_LARGE_TILE = 1024
_MAX_RECORD_BLOCK_SIZE = 256
# Source-route recording compares tiles of routes against chunks of experts.
# Eight unrolled tiles per program amortize the launch, and one atomic per
# expert per program keeps the number of atomic lanes low: an atomic per route
# saturates at a few ten million updates per second.
_SOURCE_ROUTE_TILE_SIZE = 16
_SOURCE_ROUTE_TILES_PER_PROGRAM = 8
_SOURCE_ROUTE_MAX_EXPERT_CHUNK = 512


def _mapping_block_size(numel: int) -> int:
    # Small tiles distribute decode and small prefills across more cores.
    # Large tiles amortize launch and memory-access setup in large prefills.
    if numel <= _SMALL_ASSIGNMENT_LIMIT:
        return _SMALL_TILE
    return _MEDIUM_TILE if numel <= _MEDIUM_ASSIGNMENT_LIMIT else _LARGE_TILE


@triton.jit(do_not_specialize=["numel"])
def _map_to_physical_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    num_logical_experts,
    numel,
    routing_table_rows,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1)
    valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
    safe_logical_id = tl.where(valid_logical_id, logical_id, 0)
    routing_row = (offsets // TOPK) % routing_table_rows
    routing_index = routing_row * num_logical_experts + safe_logical_id
    physical_id = tl.load(
        routing_table_ptr + routing_index,
        mask=mask & valid_logical_id,
        other=-1,
    )
    tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)


class EplbMapKernel(AscendTritonJitKernel["EplbMapKernel.CompileKey"]):
    kernel = _map_to_physical_kernel
    # Explicit binding avoids Python signature binding on each routing call.
    bind_launch_inputs = False

    @dataclass(frozen=True)
    class CompileKey:
        num_logical_experts: int
        topk: int
        block_size: int
        ids_dtype: torch.dtype

    def dispatch(self, *, num_logical_experts: int, topk: int, block_size: int, ids_dtype: torch.dtype) -> CompileKey:
        return self.CompileKey(
            num_logical_experts=num_logical_experts,
            topk=topk,
            block_size=block_size,
            ids_dtype=ids_dtype,
        )

    def get_warmup_keys(self, num_logical_experts: int, topk: int) -> list[CompileKey]:
        # Cover eager prefills beyond the graph buckets and both router ID types.
        return self._trace_dispatch(self.dispatch)(
            num_logical_experts=num_logical_experts,
            topk=topk,
            block_size=(_SMALL_TILE, _MEDIUM_TILE, _LARGE_TILE),
            ids_dtype=(torch.int32, torch.int64),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            topk_ids=TritonWarmupTensor(compile_key.ids_dtype),
            routing_table=TritonWarmupTensor(torch.int32),
            physical_ids=TritonWarmupTensor(compile_key.ids_dtype),
            num_logical_experts=compile_key.num_logical_experts,
            numel=compile_key.block_size,
            routing_table_rows=EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
            topk=compile_key.topk,
            block_size=compile_key.block_size,
        )

    @kernel_launcher
    def __call__(
        self,
        topk_ids,
        routing_table,
        physical_ids,
        num_logical_experts: int,
        numel: int,
        routing_table_rows: int,
        topk: int,
        block_size: int,
    ) -> LaunchSpec:
        return (triton.cdiv(numel, block_size),), dict(
            topk_ids_ptr=topk_ids,
            routing_table_ptr=routing_table,
            physical_ids_ptr=physical_ids,
            num_logical_experts=num_logical_experts,
            numel=numel,
            routing_table_rows=routing_table_rows,
            TOPK=topk,
            BLOCK_SIZE=block_size,
        )


_EPLB_MAP_KERNEL = EplbMapKernel()


def map_to_physical_triton(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map contiguous [T, K] logical expert IDs through the periodic table."""
    numel = topk_ids.numel()
    if numel == 0:
        return topk_ids

    physical_ids = torch.empty_like(topk_ids)
    _EPLB_MAP_KERNEL(
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_replica_routing_table.shape[1],
        numel,
        expert_replica_routing_table.shape[0],
        topk=topk_ids.shape[1],
        block_size=_mapping_block_size(numel),
    )
    return physical_ids


@triton.jit
def _record_expert_tokens_kernel(
    expert_tokens_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_local_experts,
    local_expert_start,
    group_list_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if tl.load(record_enabled_ptr) != 0:
        mask = offsets < num_local_experts
        current = tl.load(expert_tokens_ptr + offsets, mask=mask, other=0)
        # Narrow before arithmetic when the destination is narrower: integer
        # add/subtract has the same result modulo the destination width.
        if current.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
            current = current.to(expert_load_ptr.dtype.element_ty)
        if group_list_type == 1:
            local_load = current
        else:
            previous_offset = tl.maximum(offsets - 1, 0)
            previous = tl.load(expert_tokens_ptr + previous_offset, mask=mask & (offsets > 0), other=0)
            if previous.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
                previous = previous.to(expert_load_ptr.dtype.element_ty)
            local_load = current - previous
        load_offsets = local_expert_start + offsets
        previous_load = tl.load(expert_load_ptr + load_offsets, mask=mask, other=0)
        tl.store(expert_load_ptr + load_offsets, previous_load + local_load, mask=mask)


def record_expert_tokens_triton(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    """Accumulate operator-provided local counts when collection is enabled."""
    num_local_experts = expert_tokens.numel()
    if num_local_experts == 0:
        return
    block_size = min(_MAX_RECORD_BLOCK_SIZE, triton.next_power_of_2(num_local_experts))
    _record_expert_tokens_kernel[(triton.cdiv(num_local_experts, block_size),)](
        expert_tokens,
        expert_load_view,
        record_enabled,
        num_local_experts,
        local_expert_start,
        group_list_type=group_list_type,
        BLOCK_SIZE=block_size,
    )


@triton.jit(
    do_not_specialize=["topk", "num_experts", "numel"],
    # Comm-method masks are chunks of a shared buffer, so their address
    # alignment changes with the batch shape.
    do_not_specialize_on_alignment=["source_mask_ptr"],
)
def _record_source_routes_kernel(
    topk_ids_ptr,
    source_mask_ptr,
    record_enabled_ptr,
    expert_load_ptr,
    topk,
    num_experts,
    numel,
    ROUTES_PER_TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    EXPERT_CHUNK: tl.constexpr,
):
    if tl.load(record_enabled_ptr) != 0:
        experts = tl.program_id(1) * EXPERT_CHUNK + tl.arange(0, EXPERT_CHUNK)
        counts = tl.zeros([EXPERT_CHUNK], dtype=tl.int32)
        program_start = tl.program_id(0) * (ROUTES_PER_TILE * TILES_PER_PROGRAM)
        for tile in tl.static_range(TILES_PER_PROGRAM):
            offsets = program_start + tile * ROUTES_PER_TILE + tl.arange(0, ROUTES_PER_TILE)
            route_valid = offsets < numel
            expert_id = tl.load(topk_ids_ptr + offsets, mask=route_valid, other=-1)
            # Only this tile-wide check runs in the ID dtype. The comparison
            # against every expert of the chunk runs on 32-bit IDs.
            in_range = (expert_id >= 0) & (expert_id < num_experts)
            expert_id = tl.where(in_range, expert_id, -1).to(tl.int32)
            source_valid = tl.load(source_mask_ptr + offsets // topk, mask=route_valid, other=0) != 0
            valid = route_valid & source_valid & in_range
            hit = (expert_id[:, None] == experts[None, :]) & valid[:, None]
            counts += tl.sum(hit.to(tl.int32), axis=0)
        # A lane past the last expert still forms an address; leave it out.
        tl.atomic_add(expert_load_ptr + experts, counts, mask=experts < num_experts)


def record_source_routes_triton(
    topk_ids: torch.Tensor,
    source_mask: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
) -> None:
    """Accumulate the eligible routes of each physical expert when collection is enabled."""
    numel = topk_ids.numel()
    if numel == 0:
        return
    num_experts = expert_load_view.numel()
    # Small expert counts compare against a chunk no wider than themselves.
    expert_chunk = min(triton.next_power_of_2(num_experts), _SOURCE_ROUTE_MAX_EXPERT_CHUNK)
    routes_per_program = _SOURCE_ROUTE_TILE_SIZE * _SOURCE_ROUTE_TILES_PER_PROGRAM
    _record_source_routes_kernel[(triton.cdiv(numel, routes_per_program), triton.cdiv(num_experts, expert_chunk))](
        topk_ids,
        source_mask,
        record_enabled,
        expert_load_view,
        topk_ids.shape[1],
        num_experts,
        numel,
        ROUTES_PER_TILE=_SOURCE_ROUTE_TILE_SIZE,
        TILES_PER_PROGRAM=_SOURCE_ROUTE_TILES_PER_PROGRAM,
        EXPERT_CHUNK=expert_chunk,
    )
